# %% [markdown]
# # 04 — ре-ранкеры: гео-пул из каналов → N моделей → ансамбль → эвристики top-50
#
# Запуск: в VS Code по ячейкам `# %%` (Shift+Enter) или целиком
# `uv run python 04_train_reranker_v2.py`.
#
# Нужно заранее: train_* / benchmark_* parquet, ./retrieve_cache_train и
# ./retrieve_cache_benchmark из 02 (без них пайплайн работает, но без dense-каналов).
#
# Схема запросов train (против утечек):
# * H (97%) - запросы, на которых училась голова эмбеддингов → история
#   (переходы локаций, P(микрокатегория | слова));
# * F (3%, val-split из 01) - голова их не видела → честный embed-скор.
#   F делится на train ре-ранкера / valid / test; valid и test собраны
#   "как бенчмарк" (один запрос на текст, 41% знакомых текстов).

# %%
import os, json, time, gc
import numpy as np
import pandas as pd

import config_v2 as C
from data_v2 import (load_train, load_benchmark, biencoder_val_uids, select_rerank_splits,
                     load_emb, build_train_corpus, benchmark_seen_share, clean_train_uids)
from history_v2 import HistoryStats
from pool_v2 import CorpusIndex, build_pool_features, feature_columns, CHANNEL_NAMES, NO_RANK
from postprocess_v2 import (recall_at_k, recall_from_mask, pool_recall, tune_postprocess, select_topk,
                            add_helper_cols, save_params)
from models_v2 import train_one, model_ranks, ensemble_score, tune_weights, save_ensemble
from text_prep import query_key_series


os.makedirs(C.WORK_DIR, exist_ok=True)
NEG_KEEP = 0.5   # доля негативов в ОБУЧЕНИИ (0.5 = в 2 раза меньше RAM); valid/test не трогаем

# %% данные и разбиение
tq, ti, tp = load_train()
bq, bi = load_benchmark()
f_uids = biencoder_val_uids(tq)
f_set = set(f_uids)
hq = tq[~tq["query_uid"].isin(f_set)]
fq_all = tq[tq["query_uid"].isin(f_set)]
n_pos = tp.groupby("query_uid").size()
seen_share = C.EVAL_SEEN_SHARE if C.EVAL_SEEN_SHARE is not None else benchmark_seen_share(bq, tq)
# valid/test - из тех же первых 3% запросов, что в прошлых экспериментах (сравнение честное);
# train ре-ранкера - из ВСЕХ F-запросов (10%), кроме eval
eval_frac = getattr(C, "EVAL_FROM_FRACTION", C.BIENCODER_VAL_FRACTION)
f_small = set(biencoder_val_uids(tq, frac=eval_frac))
_, va_uids, te_uids = select_rerank_splits(
    tq[tq["query_uid"].isin(f_small)], set(query_key_series(tq[~tq["query_uid"].isin(f_small)])),
    n_pos, seen_share, C.RERANK_VALID_N, C.RERANK_TEST_N)
tr_uids = clean_train_uids(fq_all, set(va_uids) | set(te_uids), n_pos,
                           max_queries=getattr(C, "MAX_RERANK_TRAIN_QUERIES", None))
use_uids = np.concatenate([tr_uids, va_uids, te_uids])

# %% корпус (+ дистракторы из benchmark) и эмбеддинги
ti_emb = load_emb(C.TRAIN_EMB_DIR, "item_emb.npy")
tq_emb = load_emb(C.TRAIN_EMB_DIR, "query_emb.npy")
if ti_emb is not None:
    assert len(ti_emb) == len(ti), "retrieve_cache_train/item_emb.npy не соответствует train_items.parquet"
    assert len(tq_emb) == len(tq), "retrieve_cache_train/query_emb.npy не соответствует train_queries.parquet"
bi_aug = bi_emb = None
if C.AUGMENT_TRAIN_CORPUS_WITH_BENCH:
    bi_aug = bi
    bi_emb = load_emb(C.BENCH_EMB_DIR, "item_emb.npy")
    if ti_emb is not None and bi_emb is None:
        print("[warn] нет эмбеддингов benchmark - дистракторы из benchmark не добавляем")
        bi_aug = None
items, item_emb = build_train_corpus(ti, bi_aug, ti_emb, bi_emb)
del bi_emb, ti_emb
gc.collect()

# %% история по запросам H: переходы локаций, микрокатегории (+ item-level, если включено)
hp = tp[tp["query_uid"].isin(set(hq["query_uid"]))]
history = HistoryStats().fit(hq, hp, items, item_level=C.USE_ITEM_HISTORY)
corpus = CorpusIndex(items, item_emb=item_emb, history=history)
# тексты больше не нужны: оставляем только id (иначе 3 копии текстов 500к объявлений = OOM)
items = items[["item_id"]]
del ti, bi, bi_aug
gc.collect()

# %% пул + признаки для выбранных F-запросов
fq = tq.set_index("query_uid").loc[use_uids].reset_index()
fq_emb = None
if tq_emb is not None:
    pos_in_tq = pd.Series(np.arange(len(tq)), index=tq["query_uid"].values).loc[use_uids].values
    fq_emb = np.asarray(tq_emb[np.sort(pos_in_tq)], dtype=np.float32)[np.argsort(np.argsort(pos_in_tq))]
pos_of = pd.Series(np.arange(len(items)), index=items["item_id"].values)
pos_of = pos_of[~pos_of.index.duplicated()]
fp = tp[tp["query_uid"].isin(set(use_uids))]
uid_to_row = pd.Series(np.arange(len(fq)), index=fq["query_uid"].values)
positives = {}
for r, p in zip(uid_to_row.reindex(fp["query_uid"]).values, pos_of.reindex(fp["item_id"]).values):
    if not np.isnan(p):
        positives.setdefault(int(r), set()).add(int(p))
n_rel = pd.Series({r: len(s) for r, s in positives.items()}).reindex(range(len(fq))).fillna(0).astype(int)

POOL_PATH = f"{C.WORK_DIR}/pool_train_F.parquet"
if os.path.exists(POOL_PATH):
    # пул уже посчитан в этом WORK_DIR - не пересчитываем
    # (если меняли каналы/признаки/GEO_MIN_P - удалите файл или смените WORK_DIR)
    print(f"[pool] {POOL_PATH} уже есть - переиспользую")
else:
    build_pool_features(corpus, fq, fq_emb, POOL_PATH, C.CHANNELS, C.BATCH_QUERIES,
                        positives=positives, geo_min_p=C.GEO_MIN_P)
del corpus, item_emb
gc.collect()

# %% загрузка пула, разметка train/valid/test
pool = pd.read_parquet(POOL_PATH)
row_split = np.where(fq["query_uid"].isin(set(va_uids)), "valid",
                     np.where(fq["query_uid"].isin(set(te_uids)), "test", "train"))
pool["split"] = row_split[pool["qrow"].values]
# прореживаем негативы TRAIN сразу после загрузки, до всех копий (экономия памяти);
# valid/test не трогаем - оценка остаётся честной
if NEG_KEEP < 1.0:
    rng = np.random.default_rng(0)
    drop = (pool["split"].values == "train") & (pool["label"].values == 0) & (rng.random(len(pool)) >= NEG_KEEP)
    pool = pool[~drop].reset_index(drop=True)
    del drop
    gc.collect()
add_helper_cols(pool)
FEATS = [c for c in feature_columns(pool) if c not in C.DROP_FEATURES and c not in ("rev_bucket", "mc_plausible")]
print(f"pool: {pool.shape}, признаков: {len(FEATS)}")
print(pool.groupby("split")["label"].agg(["size", "sum"]))
rows_of = {s: np.nonzero(row_split == s)[0] for s in ("train", "valid", "test")}

# %% ПОТОЛОК: какая доля позитивов вообще попала в пул; по каналам; recall@200/500 простых скоров
# Это ответ на вопрос "если брать top-200/1000, там уже все релевантные?"
for part in ("valid", "test"):
    d = pool[pool["split"] == part]
    nr = n_rel.loc[rows_of[part]]
    q, y = d["qrow"].values, d["label"].values
    print(f"\n[{part}] pool recall (потолок) = {pool_recall(q, y, nr):.4f}  (средний пул {len(d)/len(nr):.0f})")
    for ch in CHANNEL_NAMES:
        col = f"rank_{ch}"
        if col in d:
            print(f"    {ch:15s} top-50: {pool_recall(q, y, nr, d[col].values < 50):.4f}"
                  f"   весь канал: {pool_recall(q, y, nr, d[col].values < NO_RANK):.4f}")
    simple = np.nan_to_num(d["bm25_rel_geo"].values, nan=0.0) + 2 * d["in_geo"].values + (d["cos"].values if "cos" in d else 0)
    for k in (50, 100, 200, 500):
        print(f"    простой скор (bm25_rel_geo + cos + гео) recall@{k}: {recall_at_k(q, simple, y, nr, k):.4f}")

# %% обучение N моделей
tr = pool[pool["split"] == "train"]
tr = tr[tr.groupby("qrow")["label"].transform("max") > 0]      # lambdarank нечему учиться без позитивов

va = pool[pool["split"] == "valid"]
va_fit = va[va.groupby("qrow")["label"].transform("max") > 0]
rankers = []
for name in C.MODELS:
    r = train_one(name, tr, va_fit, FEATS, task_type=C.CATBOOST_TASK_TYPE)
    if r is not None:
        rankers.append(r)
del tr
gc.collect()

if rankers[0].kind == "lgb":
    imp = pd.Series(rankers[0].model.feature_importance("gain"), index=FEATS).sort_values(ascending=False)
    print(imp.round(0).to_string())

# %% ансамбль: веса на valid, оценка каждой модели и ансамбля на test
va = pool[pool["split"] == "valid"].reset_index(drop=True)
te = pool[pool["split"] == "test"].reset_index(drop=True)
nr_va, nr_te = n_rel.loc[rows_of["valid"]], n_rel.loc[rows_of["test"]]
ranks_va = model_ranks(va, rankers, FEATS)
ranks_te = model_ranks(te, rankers, FEATS)
weights = tune_weights(ranks_va, va["qrow"].values, va["label"].values, nr_va, recall_at_k)

print("\nTEST recall@50:")
q, y = te["qrow"].values, te["label"].values
print(f"  {'bm25+40*loc (v1 baseline)':32s} {recall_at_k(q, te['bm25'].values + 40 * te['location_match'].values, y, nr_te):.4f}")
for name, rk in ranks_te.items():
    print(f"  {name:32s} {recall_at_k(q, -rk.astype(float), y, nr_te):.4f}")
ens_te = ensemble_score(ranks_te, weights)
print(f"  {'АНСАМБЛЬ':32s} {recall_at_k(q, ens_te, y, nr_te):.4f}")
print(f"  {'потолок пула':32s} {pool_recall(q, y, nr_te):.4f}")

# %% эвристики top-50 (квоты каналов/моделей, жёсткие фильтры, разнообразие): подбор на valid
ens_va = ensemble_score(ranks_va, weights)
pp = tune_postprocess(va, ens_va, nr_va)
r0 = recall_at_k(q, ens_te, y, nr_te)
r1 = recall_from_mask(q, select_topk(te, ens_te, pp), y, nr_te)
print(f"TEST: ансамбль {r0:.4f} -> с эвристиками {r1:.4f}")
if r1 < r0:
    print("эвристики на test не помогли - сохраняем без эвристик")
    pp = {"quotas": {}, "hard": [], "diversity": {}}

# %% разбивка по сегментам (test) - где проседаем
_, per_q = recall_from_mask(q, select_topk(te, ens_te, pp), y, nr_te, return_per_query=True)
seg = fq.loc[rows_of["test"], ["search_category", "search_location_id"]].copy()
seg["recall"] = per_q.reindex(rows_of["test"]).values
seg["has_filter"] = fq.loc[rows_of["test"], "search_infm_params_text"].fillna("").str.len().gt(0).values
seg["q_words"] = fq.loc[rows_of["test"], "search_query"].fillna("").str.split().str.len().clip(upper=5).values
seg["loc_has_items"] = te.groupby("qrow")["query_loc_has_items"].first().reindex(rows_of["test"]).values
seg["cat0"] = seg["search_category"] == 0
for col in ["has_filter", "q_words", "loc_has_items", "cat0"]:
    print(seg.groupby(col)["recall"].agg(["mean", "size"]).round(4), "\n")

# %% сохранение
save_ensemble(C.MODELS_DIR, rankers, weights, FEATS)
save_params(pp, C.POSTPROC_PARAMS_PATH)
keep = ["qrow", "item_pos", "label", "split", "item_microcat", "rev_bucket", "in_geo", "filter_ok", "rating_ok", "mc_plausible"]
out = pd.concat([va, te])[[c for c in keep if c in va.columns] +
                          [c for c in va.columns if c.startswith("rank_")]].copy()
out["score"] = np.concatenate([ens_va, ens_te])
out["query_uid"] = fq["query_uid"].values[out["qrow"].values]
out["item_id"] = items["item_id"].values[out["item_pos"].values]
out.to_parquet(f"{C.WORK_DIR}/eval_scores.parquet", index=False)
pd.DataFrame({"qrow": np.arange(len(fq)), "query_uid": fq["query_uid"].values, "split": row_split,
              "n_rel": n_rel.values}).to_parquet(f"{C.WORK_DIR}/eval_queries.parquet", index=False)
print("сохранено:", C.MODELS_DIR, C.POSTPROC_PARAMS_PATH, f"{C.WORK_DIR}/eval_scores.parquet")
