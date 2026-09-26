"""
data_v2.py
==========
Загрузка данных и разбиения, общие для скриптов v2.
"""
import os
import numpy as np
import pandas as pd

import config_v2 as C


def load_train(base=C.BASE, item_columns=None):
    """item_columns: читать только эти колонки train_items (тексты весят гигабайты)."""
    tq = pd.read_parquet(f"{base}/train_queries.parquet")
    ti = pd.read_parquet(f"{base}/train_items.parquet", columns=item_columns)
    tp = pd.read_parquet(f"{base}/train_pairs.parquet")
    return tq, ti, tp


def load_benchmark(base=C.BASE):
    bq = pd.read_parquet(f"{base}/benchmark_queries.parquet")
    bi = pd.read_parquet(f"{base}/benchmark_items.parquet")
    return bq, bi


def biencoder_val_uids(tq, seed=C.SPLIT_SEED, frac=C.BIENCODER_VAL_FRACTION):
    """Точное воспроизведение val-split из 01_train_biencoder.ipynb (ячейка
    'Train/val split'): эти запросы голова эмбеддингов НЕ видела при обучении."""
    uniq = tq["query_uid"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * frac))
    return np.asarray(uniq[:n_val])


def load_emb(dir_, name):
    path = os.path.join(dir_, name)
    if not os.path.exists(path):
        print(f"[инфо] {path} нет - dense-каналы будут выключены")
        return None
    return np.load(path, mmap_mode="r")


def build_train_corpus(ti, bi=None, ti_emb=None, bi_emb=None):
    """Корпус для обучения ре-ранкера: train_items (+ опционально объявления
    бенчмарка, которых нет в train, как немаркированные дистракторы).
    Эмбеддинги склеиваются в том же порядке."""
    if bi is None:
        return ti.reset_index(drop=True), (None if ti_emb is None else np.asarray(ti_emb, dtype=np.float32))
    extra_mask = ~bi["item_id"].isin(set(ti["item_id"]))
    extra = bi[extra_mask]
    items = pd.concat([ti, extra[ti.columns.intersection(extra.columns)]], ignore_index=True)
    emb = None
    if ti_emb is not None and bi_emb is not None:
        assert ti_emb.shape[1] == bi_emb.shape[1], "размерности эмбеддингов train/benchmark не совпадают"
        emb = np.vstack([np.asarray(ti_emb, dtype=np.float32),
                         np.asarray(bi_emb, dtype=np.float32)[np.nonzero(extra_mask.values)[0]]])
    print(f"[corpus] train_items={len(ti)} + доп. из benchmark={int(extra_mask.sum())} -> {len(items)}")
    return items, emb


def benchmark_seen_share(bq, tq):
    """Доля запросов бенчмарка, чей текст (стемы) встречался в train."""
    from text_prep import query_key_series
    return float(query_key_series(bq).isin(set(query_key_series(tq))).mean())


def select_rerank_splits(fq: pd.DataFrame, h_keys: set, n_pos: pd.Series, seen_share: float,
                         n_valid: int, n_test: int, seed=C.SPLIT_SEED,
                         max_pos=C.MAX_POS_PER_QUERY, min_chars=C.MIN_QUERY_CHARS, dedup=C.DEDUP_TRAIN_TEXTS):
    """
    Делит F-запросы (их не видела голова эмбеддингов) на train/valid/test ре-ранкера.

    valid/test собираются "как бенчмарк":
      * по одному запросу на текст (бенчмарк - хвост распределения, train - голова);
      * доля запросов, чей текст встречался в истории H, = seen_share (для бенчмарка ~0.41).
    train ре-ранкера = остальные F-запросы после чистки:
      * без текстов, попавших в valid/test (иначе утечка по тексту);
      * без запросов с > max_pos выборами ("листатели"/боты: до 278 выборов на запрос);
      * без пустых запросов;
      * не больше dedup запросов на один текст ("маникюр" не должен задавить хвост).
    Возвращает три массива query_uid.
    """
    from text_prep import query_key_series
    d = fq[["query_uid", "search_query"]].copy()
    d["qkey"] = query_key_series(fq).values
    d["seen"] = d["qkey"].isin(h_keys)
    d["npos"] = n_pos.reindex(d["query_uid"]).fillna(0).values
    one_per_text = d.sample(frac=1.0, random_state=seed).drop_duplicates("qkey")
    one_per_text = one_per_text[(one_per_text["npos"] > 0) & (one_per_text["npos"] <= max_pos)]
    seen, unseen = one_per_text[one_per_text["seen"]], one_per_text[~one_per_text["seen"]]
    n_total = n_valid + n_test
    target_unseen = int(round(n_total * (1 - seen_share)))
    n_unseen = min(len(unseen), target_unseen)
    if n_unseen == 0:
        print("[split][warn] в F нет запросов с новыми текстами - eval только из знакомых текстов")
        n_seen = min(len(seen), n_total)
    elif n_unseen < target_unseen:
        # новых текстов в F не хватает: уменьшаем eval, но сохраняем пропорцию как в бенчмарке
        n_seen = min(len(seen), int(round(n_unseen * seen_share / max(1 - seen_share, 1e-6))))
    else:
        n_seen = min(len(seen), n_total - n_unseen)
    ev = pd.concat([unseen.sample(n_unseen, random_state=seed), seen.sample(n_seen, random_state=seed)])
    ev = ev.sample(frac=1.0, random_state=seed + 1)
    n_va = int(len(ev) * n_valid / n_total)
    va, te = ev.iloc[:n_va], ev.iloc[n_va:]
    tr = d[~d["qkey"].isin(set(ev["qkey"]))]
    before = len(tr)
    tr = tr[(tr["npos"] > 0) & (tr["npos"] <= max_pos) & (tr["search_query"].fillna("").str.strip().str.len() >= min_chars)]
    tr = tr.sample(frac=1.0, random_state=seed).groupby("qkey").head(dedup)
    print(f"[split] eval: {len(ev)} (новых текстов {n_unseen}, виденных {n_seen}; доля виденных {n_seen/max(len(ev),1):.2f}"
          f" vs бенчмарк {seen_share:.2f}) -> valid {len(va)} / test {len(te)};  train: {before} -> {len(tr)} после чистки")
    return tr["query_uid"].values, va["query_uid"].values, te["query_uid"].values
