"""Числовые признаки объявления. Считаются один раз на корпус и дальше
индексируются по позициям кандидатов (тексты в таблицу пар не попадают)."""
import numpy as np
import pandas as pd


def price_to_float(s):
    """item_price - decimal. Есть мусор вроде -1 и 999999999999, режем до [0, 1e6]."""
    return pd.to_numeric(s, errors="coerce").astype(float).clip(lower=0, upper=1_000_000)


def precompute_item_arrays(items: pd.DataFrame) -> dict:
    return dict(
        rating=items["item_rating"].fillna(0.0).values.astype(np.float32),
        rating_is_missing=items["item_rating"].isna().values.astype(np.int8),
        reviews_count=np.log1p(items["item_rating_reviews_count"].fillna(0.0).values).astype(np.float32),
        price=np.log1p(price_to_float(items["item_price"]).fillna(0.0).values).astype(np.float32),
        phone_hidden=items["item_is_phone_hidden"].values.astype(np.int8),
        message_forbidden=items["item_is_message_forbidden"].values.astype(np.int8),
        title_len=items["item_title_raw"].fillna("").str.len().values.astype(np.int32),
        desc_len=items["item_description_raw"].fillna("").str.len().values.astype(np.int32),
    )
