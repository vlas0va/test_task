"""
generate_quick_geo_answer.py
============================
Быстрый ответ БЕЗ обученных моделей (ни эмбеддингов, ни LightGBM не нужно,
хотя если кэши эмбеддингов benchmark есть - dense-каналы тоже подключатся):
  пул из гео-каналов BM25 (pool_v2) -> простой взвешенный скор -> top-50.

score = bm25_rel_geo + 1.0*geo_p + 2*in_geo + 0.3*cov_title + 0.3*filter_ok
  bm25_rel_geo - BM25 объявления / лучший BM25 в гео-зоне запроса
  geo_p        - P(локация объявления | локация поиска) по истории train
  in_geo       - объявление в гео-зоне запроса
  cov_title    - доля слов запроса в заголовке
  filter_ok    - проходит фильтр "Вид/Тип услуги" из запроса (nan -> 1)
Офлайн (бенчмарк-подобные отложенные запросы train): recall@50 ~0.73-0.77.

Зачем: безопасная попытка, которая проверяет главную гипотезу v2 (гео-пул)
до обучения чего-либо. Время: ~5-10 минут на ноутбуке.
Запуск: uv run python generate_quick_geo_answer.py  -> answer_geo_simple.csv
"""
import numpy as np
import pandas as pd

import config_v2 as C
from data_v2 import load_train, load_benchmark, load_emb
from history_v2 import HistoryStats
from pool_v2 import CorpusIndex, build_pool_features

OUT_PATH = "./answer_geo_simple.csv"
POOL_PATH = f"{C.WORK_DIR}/pool_benchmark_simple.parquet"


def simple_score(d: pd.DataFrame, w=C.SIMPLE_SCORE_WEIGHTS):
    s = d["bm25_rel_geo"].values + w["geo_p"] * d["geo_p"].values + 2.0 * d["in_geo"].values
    s = s + w["cov_title"] * d["cov_title"].values
    s = s + w["filter_ok"] * np.nan_to_num(d["filter_ok"].values, nan=1.0)
    if "cos" in d.columns:   # если есть эмбеддинги - добавляем косинус (вес не подбирался, 1.0)
        s = s + d["cos"].values
    return s


def main():
    bq, bi = load_benchmark()
    bq = bq.reset_index(drop=True); bi = bi.reset_index(drop=True)
    cols = ["item_id", "item_microcat_id", "item_category_id", "item_location_id"]
    tq, ti, tp = load_train(item_columns=cols)
    history = HistoryStats().fit(tq, tp, pd.concat([ti[cols], bi[cols]]), item_level=False)
    del tq, ti, tp
    bi_emb = load_emb(C.BENCH_EMB_DIR, "item_emb.npy")
    bq_emb = load_emb(C.BENCH_EMB_DIR, "query_emb.npy")
    if bi_emb is not None:
        bi_emb, bq_emb = np.asarray(bi_emb, np.float32), np.asarray(bq_emb, np.float32)
    corpus = CorpusIndex(bi, item_emb=bi_emb, history=history)
    build_pool_features(corpus, bq, bq_emb, POOL_PATH, C.CHANNELS, C.BATCH_QUERIES, geo_min_p=C.GEO_MIN_P)
    del corpus
    pool = pd.read_parquet(POOL_PATH)
    pool["score"] = simple_score(pool)
    pool = pool.sort_values(["qrow", "score"], ascending=[True, False])
    top = pool.groupby("qrow").head(C.K).groupby("qrow")["item_pos"].apply(list).reindex(range(len(bq)))

    item_ids = bi["item_id"].values
    reviews = pd.to_numeric(bi["item_rating_reviews_count"], errors="coerce").fillna(0).values
    by_loc = {loc: g.index.values[np.argsort(-reviews[g.index.values])][:100] for loc, g in bi.groupby("item_location_id")}
    glob = np.argsort(-reviews)[:100]
    out, n_fill = [], 0
    for r in range(len(bq)):
        lst = list(top.iloc[r]) if isinstance(top.iloc[r], list) else []
        have = set(lst)
        for p in list(by_loc.get(bq.at[r, "search_location_id"], [])) + list(glob):
            if len(lst) >= C.K:
                break
            if p not in have:
                lst.append(p); have.add(p); n_fill += 1
        out.append(" ".join(item_ids[lst[:C.K]]))
    ans = pd.DataFrame({"query_id": bq["query_id"].astype(str), "answer": out})
    assert ans["query_id"].is_unique and len(ans) == len(bq)
    assert ans["answer"].str.split().map(lambda a: len(a) == len(set(a)) and len(a) <= C.K).all()
    ans.to_csv(OUT_PATH, index=False)
    print(f"сохранено {OUT_PATH}: {ans.shape}, добито {n_fill} позиций; "
          f"в среднем id в строке {ans['answer'].str.split().str.len().mean():.1f}")


if __name__ == "__main__":
    main()
