"""
generate_submission.py
=======================
Финальный скрипт: строит кандидатов (до 50 объявлений на запрос) для
benchmark_queries.parquet из корпуса benchmark_items.parquet и сохраняет
answer.csv в требуемом формате.

Метод (подобран и провалидирован офлайн на train.parquet, см. evaluate_offline.py
и text_prep.py/bm25.py с комментариями по каждому шагу):

1. Лексический канал: BM25 (Okapi BM25, классический метод информационного
   поиска, реализация - bm25.py, без внешних библиотек типа rank_bm25 и без
   каких-либо предобученных моделей/интернета) по словам, с русским
   стеммингом (nltk SnowballStemmer, чисто алгоритмический, офлайн).
   Текст объявления = заголовок x5 + параметры (первые 600 симв.) +
   описание (первые 400 симв.). Такие веса и параметры (k1=1.2, b=0.4)
   подобраны перебором на офлайн-валидации по recall@50 (см. evaluate_offline.py).

2. Геосигнал: если объявление находится в той же локации (item_location_id),
   что и место, где выполнялся поиск (search_location_id), это сильный
   позитивный сигнал - на train.parquet 82% выбранных пользователем
   объявлений находятся ровно в той же локации, что и запрос. Мы прибавляем
   к BM25-скору большой аддитивный бонус за совпадение локации (это НЕ
   жёсткий фильтр: если подходящих объявлений в этой локации меньше 50,
   оставшиеся слоты всё равно заполнятся лучшими по тексту объявлениями из
   других локаций - открытая деградация, а не потеря кандидатов).
   Офлайн-валидация показала, что этот сигнал даёт кратный прирост recall@50
   (с ~0.22 до ~0.76 на отложенной части train) - ожидаемо для маркетплейса
   услуг, где локальность критична (мастер "рядом" гораздо релевантнее).

Оба шага объединены в одну аддитивную скор-функцию:
    score(q, item) = BM25(q, item) + LOCATION_BOOST * [item_location_id == search_location_id]
и для каждого запроса берутся 50 объявлений с максимальным score.
"""
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp

from text_prep import build_query_text, iter_item_texts, stemming_generator
from bm25 import BM25Index

BASE = "."  # папка с *.parquet - по умолчанию рядом со скриптом (запускать из папки с данными)
OUT_PATH = "/mnt/user-data/outputs/answer.csv"

# --- гиперпараметры, подобранные на офлайн-валидации (evaluate_offline.py) ---
BM25_K1 = 1.2
BM25_B = 0.4
TITLE_REPEAT = 5
PARAMS_MAXLEN = 600
DESC_MAXLEN = 400
LOCATION_BOOST = 40.0
K = 50
QUERY_BATCH = 150


def main():
    t_start = time.time()

    bq = pd.read_parquet(f"{BASE}/benchmark_queries.parquet")
    bi = pd.read_parquet(f"{BASE}/benchmark_items.parquet")
    print(f"benchmark_queries={len(bq)}  benchmark_items={len(bi)}")

    item_ids = bi["item_id"].values

    # --- BM25-индекс по корпусу объявлений (со стеммингом слов) ---
    t0 = time.time()
    bm25 = BM25Index(analyzer="word", ngram_range=(1, 1), min_df=2, k1=BM25_K1, b=BM25_B)
    bm25.fit(stemming_generator(iter_item_texts(
        bi, title_repeat=TITLE_REPEAT, params_maxlen=PARAMS_MAXLEN, desc_maxlen=DESC_MAXLEN
    )))
    print(f"BM25 index built in {time.time()-t0:.1f}s, vocab={len(bm25.vectorizer.vocabulary_)}")

    query_text = build_query_text(bq)
    query_text_stem = list(stemming_generator(query_text))
    Q = bm25.transform_query(query_text_stem)  # (n_queries, vocab) счётчики термов запроса

    # --- one-hot по локациям для аддитивного гео-буста ---
    all_locs = pd.unique(pd.concat([bq["search_location_id"], bi["item_location_id"]], ignore_index=True))
    loc_to_idx = {loc: i for i, loc in enumerate(all_locs)}
    n_loc = len(all_locs)

    item_loc_idx = bi["item_location_id"].map(loc_to_idx).values
    item_loc_oh = sp.csr_matrix(
        (np.ones(len(item_loc_idx), dtype=np.float32), (np.arange(len(item_loc_idx)), item_loc_idx)),
        shape=(len(item_loc_idx), n_loc),
    )
    q_loc_idx = bq["search_location_id"].map(loc_to_idx).values
    q_loc_oh = sp.csr_matrix(
        (np.ones(len(q_loc_idx), dtype=np.float32), (np.arange(len(q_loc_idx)), q_loc_idx)),
        shape=(len(q_loc_idx), n_loc),
    )

    # --- батчами считаем итоговый score = BM25 + LOCATION_BOOST * location_match,
    # берём top-50 индексов, батчами - чтобы не материализовать всю (2452 x 189212)
    # плотную матрицу разом (это ~1.9 ГБ на батч из 150 запросов, полностью разом
    # вышло бы дороже и не нужно) ---
    t0 = time.time()
    n_q = Q.shape[0]
    answers = []
    for start in range(0, n_q, QUERY_BATCH):
        end = min(start + QUERY_BATCH, n_q)
        bm25_batch = (Q[start:end] @ bm25.doc_matrix_.T).toarray()
        loc_batch = (q_loc_oh[start:end] @ item_loc_oh.T).toarray()
        total = bm25_batch + LOCATION_BOOST * loc_batch
        for row in total:
            if K < len(row):
                top = np.argpartition(-row, K)[:K]
            else:
                top = np.arange(len(row))
            top = top[np.argsort(-row[top])]
            answers.append(item_ids[top])
    print(f"scored & ranked all queries in {time.time()-t0:.1f}s")

    # --- sanity-проверки формата перед сохранением ---
    assert len(answers) == len(bq)
    for a in answers:
        assert len(a) <= K
        assert len(set(a)) == len(a)  # без повторов внутри строки

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
    print(f"saved {OUT_PATH}, shape={out.shape}, total time {time.time()-t_start:.1f}s")
    print(out.head(3).to_string())


if __name__ == "__main__":
    main()
