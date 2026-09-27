"""05. answer.csv для benchmark.

Пул и признаки строятся той же функцией, что в 03. История (переходы локаций,
микрокатегории) здесь по всему train. Ранжирование - ансамбль из 04 (FINAL_MODELS_DIR).

Нужно: data/*.parquet, эмбеддинги benchmark обоих энкодеров (02), модели 04.
Около 5 минут. Эмбеддинги train не нужны.
"""
import os
import numpy as np
import pandas as pd

import config as C
from data import load_train, load_benchmark, load_emb
from history import HistoryStats
from pool import CorpusIndex, build_pool_features
from postprocess import select_topk, load_params, add_helper_cols
from rankers import load_ensemble, model_ranks, ensemble_score

BENCH_POOL_PATH = f"{C.WORK_DIR}/pool_benchmark.parquet"
TRAIN_POOL_PATH = f"{C.WORK_DIR}/pool_train_F.parquet"   # только для сравнения распределений


def main():
    os.makedirs(C.WORK_DIR, exist_ok=True)
    bq, bi = load_benchmark()
    bq, bi = bq.reset_index(drop=True), bi.reset_index(drop=True)
    print(f"benchmark: queries={len(bq)} items={len(bi)}")

    bi_emb = np.asarray(load_emb(C.BENCH_EMB_DIR, "item_emb.npy"), dtype=np.float32)
    bq_emb = np.asarray(load_emb(C.BENCH_EMB_DIR, "query_emb.npy"), dtype=np.float32)
    assert len(bi_emb) == len(bi) and len(bq_emb) == len(bq), "эмбеддинги benchmark не соответствуют parquet"

    cols = ["item_id", "item_microcat_id", "item_category_id", "item_location_id"]
    tq, ti, tp = load_train(item_columns=cols)
    history = HistoryStats().fit(tq, tp, pd.concat([ti[cols], bi[cols]]))
    del tq, ti, tp

    corpus = CorpusIndex(bi, bi_emb, history, emb2_prefer="bench")
    build_pool_features(corpus, bq, bq_emb, BENCH_POOL_PATH, C.CHANNELS, C.BATCH_QUERIES, C.GEO_MIN_P)
    del corpus
    bi = bi[["item_id", "item_location_id", "item_rating_reviews_count"]]
    pool = pd.read_parquet(BENCH_POOL_PATH)
    add_helper_cols(pool)

    rankers, weights, FEATS = load_ensemble(C.FINAL_MODELS_DIR)
    missing = [c for c in FEATS if c not in pool.columns]
    assert not missing, f"в пуле нет признаков {missing}: конфиг обучения и предсказания разошёлся"
    score = ensemble_score(model_ranks(pool, rankers, FEATS), weights)
    print(f"ансамбль: {[r.name for r in rankers]}, веса {weights}")

    # сдвиг распределений признаков train-пул -> benchmark-пул
    if os.path.exists(TRAIN_POOL_PATH):
        trp = pd.read_parquet(TRAIN_POOL_PATH, columns=FEATS)
        cmp = pd.DataFrame({"train_mean": trp.mean(), "bench_mean": pool[FEATS].mean()})
        cmp["ratio"] = (cmp["bench_mean"] / cmp["train_mean"].replace(0, np.nan)).round(2)
        print("\nсредние признаков, train-пул vs benchmark-пул:")
        print(cmp.round(4).to_string())
        del trp

    pp = load_params(C.POSTPROC_PARAMS_PATH)
    print(f"эвристики: {pp}")
    m = select_topk(pool, score, pp, C.K)
    top = pool.loc[m, ["qrow", "item_pos"]].assign(prio=score[m]).sort_values(["qrow", "prio"], ascending=[True, False])
    answers = top.groupby("qrow")["item_pos"].apply(list).reindex(range(len(bq)))

    # если в пуле меньше 50 кандидатов: добивка объявлениями локации, затем всей страны, по числу отзывов
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
                    lst.append(p)
                    have.add(p)
                    n_filled += 1
        final.append([item_ids[p] for p in lst[:C.K]])
    print(f"добито до {C.K}: {n_filled} позиций")

    # формат
    valid = set(item_ids)
    assert len(final) == len(bq)
    for a in final:
        assert len(a) <= C.K and len(set(a)) == len(a) and all(x in valid for x in a)
    out = pd.DataFrame({"query_id": bq["query_id"].astype(str).values,
                        "answer": [" ".join(a) for a in final]})
    assert out["query_id"].is_unique and set(out["query_id"]) == set(bq["query_id"].astype(str))
    out.to_csv(C.ANSWER_PATH, index=False)
    print(f"сохранено {C.ANSWER_PATH}: {out.shape}, id в строке: {out['answer'].str.split().str.len().mean():.1f}")


if __name__ == "__main__":
    main()
