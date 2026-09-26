"""
generate_final_answer.py
==========================
Финальная сборка: применяет модель 1 (BM25, широкий пул кандидатов) +
опционально модель 2 (эмбеддинги, embed_candidates.parquet) + модель 3
(LightGBM-ре-ранкер, reranker_lgb.txt) к benchmark_queries/benchmark_items и
сохраняет answer.csv в требуемом формате.

Если reranker_lgb.txt ещё не обучен (train_reranker.py не запускался) - скрипт
автоматически откатывается на проверенный baseline "BM25 + гео-буст"
(тот же метод, что в generate_submission.py) для этого запроса. Так что
можно на любом этапе прогнать этот скрипт и получить рабочий answer.csv,
постепенно докручивая модели 2/3.

Вся сборка признаков и ранжирование - ВЕКТОРИЗОВАНЫ (см. features.py,
flatten_candidates/build_candidate_features_batch): работаем сразу со всеми
кандидатами всех запросов как с одной большой таблицей, а не в Python-цикле
по 2452 запросам - иначе (проверено на практике) сборка признаков растягивается
на много минут даже на не очень большом пуле кандидатов.

Порядок использования:
  1. (обязательно) ничего дополнительно готовить не нужно - BM25 строится тут.
  2. (опционально) embed_and_retrieve.py в РЕЖИМЕ benchmark (QUERIES_FILE=
     benchmark_queries.parquet, ITEMS_FILE=benchmark_items.parquet,
     QUERY_ID_COL="query_id") -> embed_candidates.parquet
  3. (опционально, но тогда обязательно (2) на train, см. train_reranker.py)
     train_reranker.py -> reranker_lgb.txt
  4. python generate_final_answer.py -> answer.csv
"""
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

from text_prep import build_query_text, iter_item_texts, stemming_generator
from bm25 import BM25Index
from features import (
    flatten_candidates, build_candidate_features_batch,
    precompute_item_arrays, precompute_query_arrays,
)

BASE = "."  # папка с *.parquet - по умолчанию рядом со скриптом (запускать из папки с данными)
OUT_PATH = "./answer.csv"

CANDIDATE_POOL = 300       # широкий пул для ре-ранкера; без ре-ранкера участвует напрямую в BM25+гео-скоре
K = 50
LOCATION_BOOST = 40.0      # fallback-скор, если LightGBM-модели ещё нет

BENCH_EMBED_CANDIDATES_PATH = "./embed_candidates.parquet"  # см. embed_and_retrieve.py, режим benchmark
RERANKER_PATH = "./reranker_lgb.txt"


def load_embed_lookup(path):
    if not os.path.exists(path):
        print(f"[инфо] {path} не найден - работаем без эмбеддинг-канала")
        return None
    df = pd.read_parquet(path)
    return df.set_index(["query_id", "item_id"])["embed_score"]


def main():
    bq = pd.read_parquet(f"{BASE}/benchmark_queries.parquet")
    bi = pd.read_parquet(f"{BASE}/benchmark_items.parquet")
    print(f"benchmark_queries={len(bq)}  benchmark_items={len(bi)}")

    item_ids = bi["item_id"].values
    embed_lookup = load_embed_lookup(BENCH_EMBED_CANDIDATES_PATH)

    reranker = None
    if os.path.exists(RERANKER_PATH):
        reranker = lgb.Booster(model_file=RERANKER_PATH)
        print(f"загружен ре-ранкер {RERANKER_PATH}")
    else:
        print(f"[инфо] {RERANKER_PATH} не найден - откатываемся на BM25 + гео-буст (LOCATION_BOOST={LOCATION_BOOST})")

    # --- модель 1: BM25 на benchmark_items ---
    t0 = time.time()
    bm25 = BM25Index(analyzer="word", ngram_range=(1, 1), min_df=2, k1=1.2, b=0.4)
    bm25.fit(stemming_generator(iter_item_texts(
        bi, title_repeat=5, params_maxlen=600, desc_maxlen=400
    )))
    print(f"BM25 index built in {time.time()-t0:.1f}s")

    query_text = build_query_text(bq)
    query_text_stem = list(stemming_generator(query_text))

    t0 = time.time()
    top_idx, top_scores = bm25.topk(query_text_stem, k=CANDIDATE_POOL, return_scores=True)
    print(f"BM25 top-{CANDIDATE_POOL} for all queries in {time.time()-t0:.1f}s")

    t0 = time.time()
    flat = flatten_candidates(bq, top_idx, top_scores, item_ids)
    print(f"flattened: {len(flat['query_pos'])} (query, item) строк")

    if reranker is not None:
        # ВАЖНО (см. features.py докстринг): считаем числовые признаки объявлений
        # и запросов ОДИН РАЗ на уровне корпуса (компактные numpy-массивы), а не
        # тащим сырые текстовые колонки в раздутую "плоскую" таблицу кандидатов
        # (735 600 строк на benchmark) - именно это раньше валило процесс по OOM.
        item_arrays = precompute_item_arrays(bi)
        query_arrays = precompute_query_arrays(bq)
        feats = build_candidate_features_batch(
            flat, item_arrays, query_arrays,
            query_ids_for_lookup=bq["query_id"].values, embed_lookup=embed_lookup,
        )
        pred = reranker.predict(feats)
        final_score = pred
    else:
        loc_match = (bi["item_location_id"].values[flat["item_pos"]] ==
                     bq["search_location_id"].values[flat["query_pos"]]).astype(np.float32)
        final_score = flat["bm25_score"] + LOCATION_BOOST * loc_match
    print(f"признаки и скоринг за {time.time()-t0:.1f}s")

    # --- top-K на запрос: сортируем весь плоский массив один раз, режем по группам ---
    t0 = time.time()
    result = pd.DataFrame({
        "query_pos": flat["query_pos"],
        "item_id": flat["item_id"],
        "score": final_score,
    })
    result.sort_values(["query_pos", "score"], ascending=[True, False], inplace=True)
    top = result.groupby("query_pos", sort=False).head(K)
    answers_by_pos = top.groupby("query_pos", sort=False)["item_id"].apply(list)
    answers = [answers_by_pos.get(i, []) for i in range(len(bq))]
    print(f"ранжирование top-{K} за {time.time()-t0:.1f}s")

    # --- sanity-проверки формата ---
    assert len(answers) == len(bq)
    for a in answers:
        assert len(a) <= K
        assert len(set(a)) == len(a)
    valid_items = set(item_ids)
    bad = sum(1 for a in answers for x in a if x not in valid_items)
    assert bad == 0, f"{bad} item_id не найдены в корпусе"

    out = pd.DataFrame({
        "query_id": bq["query_id"].values,
        "answer": [" ".join(a) for a in answers],
    })
    assert out["query_id"].is_unique
    assert set(out["query_id"]) == set(bq["query_id"])

    out.to_csv(OUT_PATH, index=False)
    print(f"сохранено {OUT_PATH}: {out.shape}")


if __name__ == "__main__":
    main()
