"""
generate_final_answer_v2.py
===========================
Финальная сборка answer.csv для v2:
  benchmark_queries x benchmark_items
    -> пул из каналов (pool_v2.build_pool_features, ТА ЖЕ функция, что в обучении)
    -> N моделей + ансамбль (./work_v2/models/ensemble.json)
    -> эвристики top-50 (./work_v2/postprocess_params.json, подобраны на valid)
    -> answer.csv

История (переходы локаций, микрокатегории) здесь строится по ВСЕМУ train -
на обучении она строилась по 97% train; все признаки истории - доли/вероятности,
а не счётчики, поэтому сопоставимы по масштабу.

Запуск:  uv run python generate_final_answer_v2.py
"""
import os
import json
import numpy as np
import pandas as pd
import config_v2 as C
from data_v2 import load_train, load_benchmark, load_emb
from history_v2 import HistoryStats
from pool_v2 import CorpusIndex, build_pool_features
from postprocess_v2 import select_topk, load_params, add_helper_cols
from models_v2 import load_ensemble, model_ranks, ensemble_score

OUT_PATH = "./answer.csv"
BENCH_POOL_PATH = f"{C.WORK_DIR}/pool_benchmark.parquet"
TRAIN_POOL_PATH = f"{C.WORK_DIR}/pool_train_F.parquet"   # только для сравнения распределений признаков


def main():
    os.makedirs(C.WORK_DIR, exist_ok=True)
    bq, bi = load_benchmark()
    bq = bq.reset_index(drop=True); bi = bi.reset_index(drop=True)
    print(f"benchmark: queries={len(bq)} items={len(bi)}")

    bi_emb = load_emb(C.BENCH_EMB_DIR, "item_emb.npy")
    bq_emb = load_emb(C.BENCH_EMB_DIR, "query_emb.npy")
    if bi_emb is not None:
        assert len(bi_emb) == len(bi) and len(bq_emb) == len(bq), "кэш эмбеддингов benchmark не соответствует parquet"
        bi_emb = np.asarray(bi_emb, dtype=np.float32); bq_emb = np.asarray(bq_emb, dtype=np.float32)

    # история = ВЕСЬ train (при обучении была только H = 97% train; все признаки - доли/вероятности)
    cols = ["item_id", "item_microcat_id", "item_category_id", "item_location_id"]
    tq, ti, tp = load_train(item_columns=cols)
    meta = pd.concat([ti[cols], bi[cols]])
    history = HistoryStats().fit(tq, tp, meta, item_level=C.USE_ITEM_HISTORY)
    del tq, ti, tp, meta

    corpus = CorpusIndex(bi, item_emb=bi_emb, history=history, emb2_prefer="bench")
    build_pool_features(corpus, bq, bq_emb, BENCH_POOL_PATH, C.CHANNELS, C.BATCH_QUERIES, geo_min_p=C.GEO_MIN_P)
    del corpus
    bi = bi[["item_id", "item_location_id", "item_rating_reviews_count"]]
    pool = pd.read_parquet(BENCH_POOL_PATH)
    add_helper_cols(pool)

    rankers, weights, FEATS = load_ensemble(C.MODELS_DIR)
    missing = [c for c in FEATS if c not in pool.columns]
    assert not missing, f"в пуле бенчмарка нет признаков {missing} - конфиг обучения и сборки разошёлся"
    ranks = model_ranks(pool, rankers, FEATS)
    score = ensemble_score(ranks, weights)
    pool["score"] = score
    print(f"ансамбль: {[r.name for r in rankers]}, веса {weights}")

    # ---- контроль сдвига распределений: train-пул (F) vs benchmark-пул
    if os.path.exists(TRAIN_POOL_PATH):
        trp = pd.read_parquet(TRAIN_POOL_PATH, columns=FEATS)
        cmp = pd.DataFrame({"train_mean": trp.mean(), "bench_mean": pool[FEATS].mean()})
        cmp["ratio"] = (cmp["bench_mean"] / cmp["train_mean"].replace(0, np.nan)).round(2)
        print("\nсредние признаков: train-пул vs benchmark-пул (сильные расхождения = повод задуматься)")
        print(cmp.round(4).to_string())
        del trp

    pp = load_params(C.POSTPROC_PARAMS_PATH)
    print(f"эвристики постобработки: {pp}")
    m = select_topk(pool, score, pp, C.K)
    top = pool.loc[m, ["qrow", "item_pos"]].assign(prio=score[m]).sort_values(["qrow", "prio"], ascending=[True, False])
    answers = top.groupby("qrow")["item_pos"].apply(list).reindex(range(len(bq)))

    # ---- добивка до 50 (если пул запроса оказался меньше 50): популярные в локации, затем по отзывам
    item_ids = bi["item_id"].values
    fill_score = np.log1p(pd.to_numeric(bi["item_rating_reviews_count"], errors="coerce").fillna(0).values)
    loc_items = {loc: g.index.values[np.argsort(-fill_score[g.index.values])][:200]
                 for loc, g in bi.groupby("item_location_id")}
    global_fill = np.argsort(-fill_score)[:200]
    final, n_filled = [], 0
    for r in range(len(bq)):
        lst = answers.iloc[r] if isinstance(answers.iloc[r], list) else []
        if len(lst) < C.K:
            have = set(lst)
            for p in list(loc_items.get(bq.at[r, "search_location_id"], [])) + list(global_fill):
                if len(lst) >= C.K:
                    break
                if p not in have:
                    lst.append(p); have.add(p); n_filled += 1
        final.append([item_ids[p] for p in lst[:C.K]])
    print(f"добито до {C.K}: {n_filled} позиций")

    # ---- проверки формата
    valid = set(item_ids)
    assert len(final) == len(bq)
    for a in final:
        assert len(a) <= C.K and len(set(a)) == len(a)
        assert all(x in valid for x in a)
    out = pd.DataFrame({"query_id": bq["query_id"].astype(str).values,
                        "answer": [" ".join(a) for a in final]})
    assert out["query_id"].is_unique and set(out["query_id"]) == set(bq["query_id"].astype(str))
    if not out["query_id"].str.len().eq(16).all():
        print("[warn] не все query_id длиной 16 символов - проверьте, что они прочитаны как строки")
    out.to_csv(OUT_PATH, index=False)
    print(f"сохранено {OUT_PATH}: {out.shape}; среднее число id в строке: "
          f"{out['answer'].str.split().str.len().mean():.1f}")

    # для опционального 06_cross_encoder_rerank.py
    keep = ["qrow", "item_pos", "score", "item_microcat", "rev_bucket", "filter_ok", "in_geo", "rating_ok", "mc_plausible"]
    pool[[c for c in keep if c in pool.columns] + [c for c in pool.columns if c.startswith("rank_")]].to_parquet(
        f"{C.WORK_DIR}/bench_scores.parquet", index=False)


if __name__ == "__main__":
    main()
