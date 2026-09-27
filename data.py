"""Загрузка данных, корпус для обучения ре-ранкера, разбиение запросов train."""
import os
import numpy as np
import pandas as pd

import config as C
from text_prep import query_key_series


def load_train(item_columns=None):
    """item_columns - читать только эти колонки train_items (тексты весят гигабайты)."""
    tq = pd.read_parquet(f"{C.DATA_DIR}/train_queries.parquet")
    ti = pd.read_parquet(f"{C.DATA_DIR}/train_items.parquet", columns=item_columns)
    tp = pd.read_parquet(f"{C.DATA_DIR}/train_pairs.parquet")
    return tq, ti, tp


def load_benchmark():
    bq = pd.read_parquet(f"{C.DATA_DIR}/benchmark_queries.parquet")
    bi = pd.read_parquet(f"{C.DATA_DIR}/benchmark_items.parquet")
    return bq, bi


def load_emb(dir_, name):
    """Эмбеддинги из 02 (mmap, в память не читаются целиком)."""
    path = os.path.join(dir_, name)
    assert os.path.exists(path), f"нет {path}: сначала 02_encode.py"
    return np.load(path, mmap_mode="r")


def biencoder_val_uids(tq, seed=C.SPLIT_SEED, frac=C.BIENCODER_VAL_FRACTION):
    """Тот же val-split, что в 01: эти запросы голова эмбеддингов не видела."""
    uniq = tq["query_uid"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * frac))
    return np.asarray(uniq[:n_val])


def build_train_corpus(ti, bi, ti_emb, bi_emb):
    """Корпус для обучения ре-ранкера: train_items + объявления бенчмарка, которых нет
    в train (без разметки, как дистракторы). В train_items только объявления, которые
    кто-то выбрал, а в бенчмарке 90% корпуса - "чужие" объявления."""
    extra_mask = ~bi["item_id"].isin(set(ti["item_id"]))
    extra = bi[extra_mask]
    items = pd.concat([ti, extra[ti.columns.intersection(extra.columns)]], ignore_index=True)
    assert ti_emb.shape[1] == bi_emb.shape[1]
    # заполняем из mmap без промежуточных копий (vstack требовал вдвое больше памяти)
    extra_idx = np.nonzero(extra_mask.values)[0]
    emb = np.empty((len(ti_emb) + len(extra_idx), ti_emb.shape[1]), dtype=np.float32)
    emb[:len(ti_emb)] = ti_emb
    emb[len(ti_emb):] = bi_emb[extra_idx]
    print(f"[corpus] train_items={len(ti)} + доп. из benchmark={int(extra_mask.sum())} -> {len(items)}")
    return items, emb


def benchmark_seen_share(bq, tq):
    """Доля запросов бенчмарка, чей текст (стемы) встречался в train."""
    return float(query_key_series(bq).isin(set(query_key_series(tq))).mean())


def _select_eval(fq, h_keys, n_pos, seen_share, n_valid, n_test, seed=C.SPLIT_SEED):
    """valid/test "как бенчмарк": один запрос на текст, доля знакомых текстов = seen_share."""
    d = fq[["query_uid", "search_query"]].copy()
    d["qkey"] = query_key_series(fq).values
    d["seen"] = d["qkey"].isin(h_keys)
    d["npos"] = n_pos.reindex(d["query_uid"]).fillna(0).values
    one = d.sample(frac=1.0, random_state=seed).drop_duplicates("qkey")
    one = one[(one["npos"] > 0) & (one["npos"] <= C.MAX_POS_PER_QUERY)]
    seen, unseen = one[one["seen"]], one[~one["seen"]]
    n_total = n_valid + n_test
    target_unseen = int(round(n_total * (1 - seen_share)))
    n_unseen = min(len(unseen), target_unseen)
    if n_unseen == 0:
        n_seen = min(len(seen), n_total)
    elif n_unseen < target_unseen:
        # новых текстов не хватает: уменьшаем eval, сохраняя пропорцию
        n_seen = min(len(seen), int(round(n_unseen * seen_share / max(1 - seen_share, 1e-6))))
    else:
        n_seen = min(len(seen), n_total - n_unseen)
    ev = pd.concat([unseen.sample(n_unseen, random_state=seed), seen.sample(n_seen, random_state=seed)])
    ev = ev.sample(frac=1.0, random_state=seed + 1)
    n_va = int(len(ev) * n_valid / n_total)
    print(f"[split] eval: {len(ev)} (новых текстов {n_unseen}, знакомых {n_seen}; "
          f"доля знакомых {n_seen / max(len(ev), 1):.2f}, у бенчмарка {seen_share:.2f}) "
          f"-> valid {n_va} / test {len(ev) - n_va}")
    return ev.iloc[:n_va]["query_uid"].values, ev.iloc[n_va:]["query_uid"].values


def _clean_train(fq, exclude_uids, n_pos, max_queries, seed=C.SPLIT_SEED):
    """train ре-ранкера: без текстов из valid/test, без выбросов, <= DEDUP_TRAIN_TEXTS на текст."""
    d = fq[["query_uid", "search_query"]].copy()
    d["qkey"] = query_key_series(fq).values
    d["npos"] = n_pos.reindex(d["query_uid"]).fillna(0).values
    ev_keys = set(d.loc[d["query_uid"].isin(exclude_uids), "qkey"])
    tr = d[~d["qkey"].isin(ev_keys)]
    before = len(tr)
    tr = tr[(tr["npos"] > 0) & (tr["npos"] <= C.MAX_POS_PER_QUERY)
            & (tr["search_query"].fillna("").str.strip().str.len() >= C.MIN_QUERY_CHARS)]
    tr = tr.sample(frac=1.0, random_state=seed).groupby("qkey").head(C.DEDUP_TRAIN_TEXTS)
    if max_queries and len(tr) > max_queries:
        tr = tr.sample(max_queries, random_state=seed)
    print(f"[split] train ре-ранкера: {before} -> {len(tr)} после чистки")
    return tr["query_uid"].values


def rerank_splits(tq, tp, bq):
    """Разбиение запросов train для ре-ранкера.

    H - запросы, на которых учились головы эмбеддингов (90%): только история
        (переходы локаций, P(микрокатегория | слова)).
    F - отложенные в 01 (10%): train / valid / test ре-ранкера.
    Возвращает (hq, tr_uids, va_uids, te_uids).
    """
    f_set = set(biencoder_val_uids(tq))
    hq = tq[~tq["query_uid"].isin(f_set)]
    fq = tq[tq["query_uid"].isin(f_set)]
    n_pos = tp.groupby("query_uid").size()
    seen_share = C.EVAL_SEEN_SHARE if C.EVAL_SEEN_SHARE is not None else benchmark_seen_share(bq, tq)
    small = set(biencoder_val_uids(tq, frac=C.EVAL_FROM_FRACTION))
    va, te = _select_eval(tq[tq["query_uid"].isin(small)],
                          set(query_key_series(tq[~tq["query_uid"].isin(small)])),
                          n_pos, seen_share, C.RERANK_VALID_N, C.RERANK_TEST_N)
    tr = _clean_train(fq, set(va) | set(te), n_pos, C.MAX_RERANK_TRAIN_QUERIES)
    return hq, tr, va, te
