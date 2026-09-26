"""
features.py
============
Построение признаков пары (запрос, объявление) для обучаемого ре-ранкера
(модель 3 в общей схеме: BM25+гео (модель 1) + текстовые эмбеддинги, которые
считает пользователь локально (модель 2) + этот LightGBM-ре-ранкер (модель 3),
который учится комбинировать сигналы вместо ручных весов).

Все признаки здесь - дёшево вычислимые, без обращения к каким-либо внешним
сервисам/моделям.

ВАЖНО про производительность и память (это не теоретическое соображение -
проверено на практике и один раз уже уронило скрипт по OOM):
  - Кандидатов на запрос - до CANDIDATE_POOL (обычно 300), запросов - тысячи,
    итоговая "плоская" таблица - сотни тысяч строк.
  - Если для каждой такой строки тащить с собой ПОЛНЫЕ текстовые колонки
    объявления (item_description_raw, item_infm_params_text - каждая может
    весить килобайты) через items_df.iloc[ipos], один и тот же текст одного
    популярного объявления дублируется столько раз, сколько у него кандидатских
    вхождений по разным запросам - на benchmark (189к объявлений, 2452 запроса,
    пул 300) это раздувается до нескольких ГБ и валит процесс по памяти.
  - Решение: НИКОГДА не тащим сырые текстовые колонки во "плоскую" стадию.
    Вместо этого считаем нужные из них ЧИСЛОВЫЕ производные (длины строк и
    т.п.) ОДИН РАЗ на уровне корпуса (precompute_item_arrays), и уже эти
    компактные numpy-массивы индексируем по позициям кандидатов - это дёшево
    (просто int32/float32 массивы, без дублирования строк).
"""
import numpy as np
import pandas as pd


def price_to_float(s):
    """item_price лежит как decimal128, после pandas читается как object/Decimal.
    В данных есть явный мусор (-1, 999999999999 и т.п.) - клипуем в разумный
    диапазон, чтобы выброс не взрывал масштаб признака."""
    v = pd.to_numeric(s, errors="coerce").astype(float)
    v = v.clip(lower=0, upper=1_000_000)  # выше миллиона рублей за разовую услугу - почти наверняка мусор/заглушка
    return v


def precompute_item_arrays(items_df: pd.DataFrame) -> dict:
    """
    Считает один раз (на уровне корпуса, не на уровне раздутой "плоской"
    таблицы кандидатов) все числовые производные от полей объявления, которые
    нужны для признаков. Дальше эти массивы дёшево индексируются по позициям
    кандидатов (см. build_candidate_features_batch).
    """
    return dict(
        location_id=items_df["item_location_id"].values,
        category_id=items_df["item_category_id"].values,
        rating=items_df["item_rating"].fillna(0.0).values.astype(np.float32),
        rating_is_missing=items_df["item_rating"].isna().values.astype(np.int8),
        reviews_count=np.log1p(items_df["item_rating_reviews_count"].fillna(0.0).values).astype(np.float32),
        price=np.log1p(price_to_float(items_df["item_price"]).fillna(0.0).values).astype(np.float32),
        phone_hidden=items_df["item_is_phone_hidden"].values.astype(np.int8),
        message_forbidden=items_df["item_is_message_forbidden"].values.astype(np.int8),
        title_len=items_df["item_title_raw"].fillna("").str.len().values.astype(np.int32),
        desc_len=items_df["item_description_raw"].fillna("").str.len().values.astype(np.int32),
    )


def precompute_query_arrays(query_df: pd.DataFrame) -> dict:
    """То же самое для запросов (дёшево и без этой оптимизации - запросов
    всегда немного, но для единообразия и на случай больших N_TRAIN_QUERIES)."""
    return dict(
        location_id=query_df["search_location_id"].values,
        category=query_df["search_category"].values,
        has_rating_filter=query_df["search_infm_params_text"].fillna("")
            .str.contains("Рейтинг пользователя").values.astype(np.int8),
    )


def flatten_candidates(query_df, top_idx, top_scores, item_ids):
    """
    Превращает "top-N кандидатов на запрос" (список списков от BM25Index.topk)
    в один плоский набор параллельных numpy-массивов - основа для векторизованного
    построения признаков и для финального ранжирования.

    Возвращает dict: query_pos (номер запроса 0..n-1 на каждую плоскую строку),
    item_pos (позиция объявления в корпусе), item_id, bm25_score, group_sizes.
    """
    group_sizes = np.array([len(idxs) for idxs in top_idx])
    query_pos = np.repeat(np.arange(len(top_idx)), group_sizes)
    item_pos = np.concatenate(top_idx) if len(top_idx) else np.array([], dtype=int)
    bm25_score = np.concatenate(top_scores) if len(top_scores) else np.array([], dtype=np.float32)
    item_id = item_ids[item_pos]
    return dict(query_pos=query_pos, item_pos=item_pos, item_id=item_id,
                bm25_score=bm25_score, group_sizes=group_sizes)


def build_candidate_features_batch(
    flat: dict,
    item_arrays: dict,
    query_arrays: dict,
    query_ids_for_lookup: np.ndarray = None,
    embed_lookup: pd.Series = None,
) -> pd.DataFrame:
    """
    Векторизованно строит признаки сразу для ВСЕХ "плоских" кандидатов.

    flat - результат flatten_candidates(...).
    item_arrays - результат precompute_item_arrays(items_df) (тот же items_df,
    что использовался для BM25/flatten_candidates).
    query_arrays - результат precompute_query_arrays(query_df) (тот же
    query_df, что использовался для flatten_candidates).
    query_ids_for_lookup / embed_lookup - опционально, для джойна с эмбеддинг-
    скорами (модель 2): query_ids_for_lookup - array тех же query_id/query_uid,
    что в query_df, тем же порядком; embed_lookup - pd.Series с MultiIndex
    (query_id, item_id) -> embed_score (см. embed_and_retrieve.py).
    """
    qpos = flat["query_pos"]
    ipos = flat["item_pos"]
    n = len(qpos)

    feats = pd.DataFrame(index=np.arange(n))
    feats["bm25_word"] = flat["bm25_score"]

    if embed_lookup is not None and query_ids_for_lookup is not None:
        lookup_keys = pd.MultiIndex.from_arrays(
            [query_ids_for_lookup[qpos], flat["item_id"]]
        )
        embed_scores = embed_lookup.reindex(lookup_keys).to_numpy(dtype=np.float32)
        embed_scores = np.nan_to_num(embed_scores, nan=0.0)
        feats["embed_score"] = embed_scores
        feats["has_embed_score"] = (embed_scores != 0.0).astype(np.int8)

    item_rating = item_arrays["rating"][ipos]

    feats["location_match"] = (item_arrays["location_id"][ipos] == query_arrays["location_id"][qpos]).astype(np.int8)
    feats["category_match"] = (item_arrays["category_id"][ipos] == query_arrays["category"][qpos]).astype(np.int8)
    feats["item_rating"] = item_rating
    feats["item_rating_is_missing"] = item_arrays["rating_is_missing"][ipos]
    feats["item_reviews_count"] = item_arrays["reviews_count"][ipos]
    feats["item_price"] = item_arrays["price"][ipos]
    feats["item_phone_hidden"] = item_arrays["phone_hidden"][ipos]
    feats["item_message_forbidden"] = item_arrays["message_forbidden"][ipos]
    feats["title_len"] = item_arrays["title_len"][ipos]
    feats["desc_len"] = item_arrays["desc_len"][ipos]

    has_rating_filter = query_arrays["has_rating_filter"][qpos]
    feats["query_has_rating_filter"] = has_rating_filter
    feats["rating_filter_satisfied"] = np.where(has_rating_filter, (item_rating >= 4.0).astype(np.int8), 0)

    return feats
