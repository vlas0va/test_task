# %% [markdown]
# # 03. Пул кандидатов, ранжировщики, ансамбль, оценка на test
#
# Вход: parquet в data/, эмбеддинги обоих энкодеров из 02.
# Выход: artifacts/work/pool_train_F.parquet, models/, postprocess_params.json, eval_*.parquet.
#
# Запуск: `python 03_train_reranker.py` или по ячейкам `# %%` в VS Code.
# Около часа на RTX 3060 Laptop / 16 ГБ RAM (пул ~27 мин, модели ~8 мин).

# %%
import os
import gc
import numpy as np
import pandas as pd

import config as C
from data import load_train, load_benchmark, load_emb, build_train_corpus, rerank_splits
from history import HistoryStats
from pool import CorpusIndex, build_pool_features, feature_columns, CHANNEL_NAMES, NO_RANK
from postprocess import (recall_at_k, recall_from_mask, pool_recall, tune_postprocess, select_topk,
                         add_helper_cols, save_params)
from rankers import train_one, model_ranks, ensemble_score, tune_weights, save_ensemble

os.makedirs(C.WORK_DIR, exist_ok=True)

# %% разбиение запросов train
tq, ti, tp = load_train()
bq, bi = load_benchmark()
hq, tr_uids, va_uids, te_uids = rerank_splits(tq, tp, bq)
use_uids = np.concatenate([tr_uids, va_uids, te_uids])

# %% корпус: train_items + объявления бенчмарка как дистракторы
ti_emb = load_emb(C.TRAIN_EMB_DIR, "item_emb.npy")
tq_emb = load_emb(C.TRAIN_EMB_DIR, "query_emb.npy")
bi_emb = load_emb(C.BENCH_EMB_DIR, "item_emb.npy")
assert len(ti_emb) == len(ti) and len(tq_emb) == len(tq), "эмбеддинги train не соответствуют parquet"
items, item_emb = build_train_corpus(ti, bi, ti_emb, bi_emb)
del bi_emb, ti_emb
gc.collect()

# %% история только по H, чтобы позитивы ре-ранкера не попадали в статистики
hp = tp[tp["query_uid"].isin(set(hq["query_uid"]))]
history = HistoryStats().fit(hq, hp, items)
corpus = CorpusIndex(items, item_emb, history)
items = items[["item_id"]]
del ti, bi
gc.collect()

# %% пул и признаки
fq = tq.set_index("query_uid").loc[use_uids].reset_index()
pos_in_tq = pd.Series(np.arange(len(tq)), index=tq["query_uid"].values).loc[use_uids].values
# mmap читается быстрее по возрастанию индексов, потом возвращаем исходный порядок
fq_emb = np.asarray(tq_emb[np.sort(pos_in_tq)], dtype=np.float32)[np.argsort(np.argsort(pos_in_tq))]

pos_of = pd.Series(np.arange(len(items)), index=items["item_id"].values)
pos_of = pos_of[~pos_of.index.duplicated()]
fp = tp[tp["query_uid"].isin(set(use_uids))]
uid_to_row = pd.Series(np.arange(len(fq)), index=fq["query_uid"].values)
positives = {}
for r, p in zip(uid_to_row.reindex(fp["query_uid"]).values, pos_of.reindex(fp["item_id"]).values):
    if not np.isnan(p):
        positives.setdefault(int(r), set()).add(int(p))
# число выборов по запросу, включая не попавшие в пул (знаменатель recall)
n_rel = pd.Series({r: len(s) for r, s in positives.items()}).reindex(range(len(fq))).fillna(0).astype(int)

POOL_PATH = f"{C.WORK_DIR}/pool_train_F.parquet"
if os.path.exists(POOL_PATH):
    print(f"[pool] {POOL_PATH} уже есть, использую его (после изменения каналов/признаков удалить)")
else:
    build_pool_features(corpus, fq, fq_emb, POOL_PATH, C.CHANNELS, C.BATCH_QUERIES, C.GEO_MIN_P,
                        positives=positives)
del corpus, item_emb
gc.collect()

# %% разметка train/valid/test, прореживание негативов train
pool = pd.read_parquet(POOL_PATH)
row_split = np.where(fq["query_uid"].isin(set(va_uids)), "valid",
                     np.where(fq["query_uid"].isin(set(te_uids)), "test", "train"))
pool["split"] = row_split[pool["qrow"].values]
rng = np.random.default_rng(0)
drop = (pool["split"].values == "train") & (pool["label"].values == 0) & (rng.random(len(pool)) >= C.NEG_KEEP)
pool = pool[~drop].reset_index(drop=True)
del drop
gc.collect()
add_helper_cols(pool)
FEATS = [c for c in feature_columns(pool) if c not in ("rev_bucket", "mc_plausible")]
print(f"pool: {pool.shape}, признаков: {len(FEATS)}")
print(pool.groupby("split")["label"].agg(["size", "sum"]))
rows_of = {s: np.nonzero(row_split == s)[0] for s in ("train", "valid", "test")}

# %% потолок пула и вклад каналов
for part in ("valid", "test"):
    d = pool[pool["split"] == part]
    nr = n_rel.loc[rows_of[part]]
    q, y = d["qrow"].values, d["label"].values
    print(f"\n[{part}] потолок пула = {pool_recall(q, y, nr):.4f}  (в среднем {len(d) / len(nr):.0f} кандидатов)")
    for ch in CHANNEL_NAMES:
        col = f"rank_{ch}"
        if col in d:
            print(f"    {ch:15s} top-50: {pool_recall(q, y, nr, d[col].values < 50):.4f}"
                  f"   весь канал: {pool_recall(q, y, nr, d[col].values < NO_RANK):.4f}")
    simple = np.nan_to_num(d["bm25_rel_geo"].values, nan=0.0) + 2 * d["in_geo"].values + d["cos"].values
    for k in (50, 100, 200, 500):
        print(f"    простой скор bm25_rel_geo + cos + 2*in_geo, recall@{k}: {recall_at_k(q, simple, y, nr, k):.4f}")

# %% обучение моделей
tr = pool[pool["split"] == "train"]
tr = tr[tr.groupby("qrow")["label"].transform("max") > 0]     # запросы без позитивов в пуле ничему не учат
va = pool[pool["split"] == "valid"]
va_fit = va[va.groupby("qrow")["label"].transform("max") > 0]
rankers = [train_one(name, tr, va_fit, FEATS, task_type=C.CATBOOST_TASK_TYPE) for name in C.MODELS]
del tr
gc.collect()

imp = pd.Series(rankers[0].model.feature_importance("gain"), index=FEATS).sort_values(ascending=False)
print(imp.round(0).to_string())

# %% ансамбль: веса на valid, оценка на test
va = pool[pool["split"] == "valid"].reset_index(drop=True)
te = pool[pool["split"] == "test"].reset_index(drop=True)
nr_va, nr_te = n_rel.loc[rows_of["valid"]], n_rel.loc[rows_of["test"]]
ranks_va = model_ranks(va, rankers, FEATS)
ranks_te = model_ranks(te, rankers, FEATS)
weights = tune_weights(ranks_va, va["qrow"].values, va["label"].values, nr_va, recall_at_k)

q, y = te["qrow"].values, te["label"].values
print("\nTEST recall@50:")
print(f"  {'BM25 + 40*location_match (v1)':32s} "
      f"{recall_at_k(q, te['bm25'].values + 40 * te['location_match'].values, y, nr_te):.4f}")
for name, rk in ranks_te.items():
    print(f"  {name:32s} {recall_at_k(q, -rk.astype(float), y, nr_te):.4f}")
ens_te = ensemble_score(ranks_te, weights)
print(f"  {'ансамбль':32s} {recall_at_k(q, ens_te, y, nr_te):.4f}")
print(f"  {'потолок пула':32s} {pool_recall(q, y, nr_te):.4f}")

# %% эвристики top-50: подбор на valid, проверка на test
ens_va = ensemble_score(ranks_va, weights)
pp = tune_postprocess(va, ens_va, nr_va)
r0 = recall_at_k(q, ens_te, y, nr_te)
r1 = recall_from_mask(q, select_topk(te, ens_te, pp), y, nr_te)
print(f"TEST: ансамбль {r0:.4f} -> с эвристиками {r1:.4f}")
if r1 < r0:
    pp = {"quotas": {}, "hard": [], "diversity": {}}

# %% recall по сегментам test
_, per_q = recall_from_mask(q, select_topk(te, ens_te, pp), y, nr_te, return_per_query=True)
seg = fq.loc[rows_of["test"], ["search_category"]].copy()
seg["recall"] = per_q.reindex(rows_of["test"]).values
seg["has_filter"] = fq.loc[rows_of["test"], "search_infm_params_text"].fillna("").str.len().gt(0).values
seg["q_words"] = fq.loc[rows_of["test"], "search_query"].fillna("").str.split().str.len().clip(upper=5).values
seg["loc_has_items"] = te.groupby("qrow")["query_loc_has_items"].first().reindex(rows_of["test"]).values
for col in ["has_filter", "q_words", "loc_has_items"]:
    print(seg.groupby(col)["recall"].agg(["mean", "size"]).round(4), "\n")

# %% сохранение (скоры valid/test нужны для 06_analyze_errors.py)
save_ensemble(C.MODELS_DIR, rankers, weights, FEATS)
save_params(pp, C.POSTPROC_PARAMS_PATH)
keep = ["qrow", "item_pos", "label", "split"] + [c for c in va.columns if c.startswith("rank_")]
out = pd.concat([va, te])[keep].copy()
out["score"] = np.concatenate([ensemble_score(ranks_va, weights), ens_te])
out["query_uid"] = fq["query_uid"].values[out["qrow"].values]
out["item_id"] = items["item_id"].values[out["item_pos"].values]
out.to_parquet(f"{C.WORK_DIR}/eval_scores.parquet", index=False)
pd.DataFrame({"qrow": np.arange(len(fq)), "query_uid": fq["query_uid"].values, "split": row_split,
              "n_rel": n_rel.values}).to_parquet(f"{C.WORK_DIR}/eval_queries.parquet", index=False)
print("сохранено:", C.MODELS_DIR)
