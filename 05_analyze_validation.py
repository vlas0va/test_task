# %% [markdown]
# # 05 — как модель ранжирует на отложенных запросах (valid/test из 04)
#
# Требует результатов 04_train_reranker_v2.py (./work_v2/eval_scores.parquet,
# eval_queries.parquet). Ничего не обучает.
#
# Что показывает:
# * где находятся позитивы: в топ-10 / 50 / 100 / 300 / в пуле ниже / вообще не в пуле;
# * какие каналы нашли пропущенные позитивы (понять, какой канал расширять);
# * примеры запросов "глазами": запрос → что выбрал пользователь → что мы поставили в топ;
# * CSV с промахами для ручного просмотра (./work_v2/misses.csv).

# %%
import numpy as np
import pandas as pd
import config_v2 as C
from data_v2 import load_train, load_benchmark

pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
SPLIT = "test"   # или "valid"

sc = pd.read_parquet(f"{C.WORK_DIR}/eval_scores.parquet")
eq = pd.read_parquet(f"{C.WORK_DIR}/eval_queries.parquet")
tq, ti, tp = load_train()
_, bi = load_benchmark()
items = pd.concat([ti, bi[~bi["item_id"].isin(set(ti["item_id"]))]], ignore_index=True).set_index("item_id")

sc = sc[sc["split"] == SPLIT].copy()
eq = eq[(eq["split"] == SPLIT) & (eq["n_rel"] > 0)]
sc["rank"] = sc.groupby("qrow")["score"].rank(ascending=False, method="first").astype(int) - 1
qinfo = tq.set_index("query_uid")

# %% 1. где оказываются позитивы
pos_pairs = tp[tp["query_uid"].isin(set(eq["query_uid"]))]
found = sc[sc["label"] == 1][["query_uid", "item_id", "rank"] + [c for c in sc.columns if c.startswith("rank_")]]
allpos = pos_pairs.merge(found, on=["query_uid", "item_id"], how="left")
bins = [-1, 9, 49, 99, 299, 10**6]
labels = ["top-10", "11-50", "51-100", "101-300", ">300 (в пуле)"]
allpos["bucket"] = pd.cut(allpos["rank"], bins=bins, labels=labels).astype(str)
allpos.loc[allpos["rank"].isna(), "bucket"] = "НЕ в пуле"
print(f"[{SPLIT}] позитивов: {len(allpos)}")
print(allpos["bucket"].value_counts(normalize=True).reindex(labels + ["НЕ в пуле"]).round(4))

# %% 2. пропуски: какие каналы их видели
miss = allpos[~allpos["bucket"].isin(["top-10", "11-50"])].copy()
chan_cols = [c for c in allpos.columns if c.startswith("rank_") and c != "rank"]
print("\nпропущенные позитивы (в пуле, но ниже 50): в каком канале и на каком месте они были (медиана ранга, доля найденных каналом):")
inpool = miss[miss["bucket"] != "НЕ в пуле"]
for c in chan_cols:
    v = inpool[c]
    hit = v < 9999
    print(f"  {c:18s} нашёл {hit.mean():.2f}, медианный ранг в канале {v[hit].median() if hit.any() else float('nan'):.0f}")

# свойства пропусков vs попаданий
allpos = allpos.merge(qinfo[["search_query", "search_location_id", "search_infm_params_text"]],
                      left_on="query_uid", right_index=True)
allpos["item_loc"] = items["item_location_id"].reindex(allpos["item_id"]).values
allpos["loc_match"] = allpos["item_loc"] == allpos["search_location_id"]
allpos["hit50"] = allpos["rank"] < 50
print("\nпопадание в топ-50 в зависимости от совпадения локации позитива:")
print(allpos.groupby("loc_match")["hit50"].agg(["mean", "size"]).round(3))

# %% 3. примеры "глазами"
def show_query(uid, n_top=10):
    qi = qinfo.loc[uid]
    print("=" * 100)
    print(f"ЗАПРОС: «{qi['search_query']}»  loc={qi['search_location_id']}  фильтр: {qi['search_infm_params_text']}")
    for _, r in allpos[allpos["query_uid"] == uid].iterrows():
        it = items.loc[r["item_id"]]
        rk = "нет в пуле" if pd.isna(r["rank"]) else int(r["rank"])
        print(f"   ВЫБРАЛ [{rk}] loc={it['item_location_id']} ★{it['item_rating']} ({it['item_rating_reviews_count']}): "
              f"{str(it['item_title_raw'])[:90]}")
    top = sc[sc["query_uid"] == uid].nsmallest(n_top, "rank")
    for _, r in top.iterrows():
        it = items.loc[r["item_id"]]
        mark = "✔" if r["label"] == 1 else " "
        print(f"   {mark} #{r['rank']:<3d} loc={it['item_location_id']} ★{it['item_rating']} "
              f"({it['item_rating_reviews_count']}): {str(it['item_title_raw'])[:90]}")


rng = np.random.default_rng(0)
miss_uids = allpos.loc[~allpos["hit50"], "query_uid"].unique()
for uid in rng.choice(miss_uids, size=min(8, len(miss_uids)), replace=False):
    show_query(uid)

# %% 4. CSV с промахами для ручного разбора
rows = []
for uid in miss_uids:
    qi = qinfo.loc[uid]
    top5 = sc[sc["query_uid"] == uid].nsmallest(5, "rank")["item_id"]
    for _, r in allpos[(allpos["query_uid"] == uid) & (~allpos["hit50"])].iterrows():
        rows.append({
            "query": qi["search_query"], "filter": qi["search_infm_params_text"], "loc": qi["search_location_id"],
            "missed_title": items.at[r["item_id"], "item_title_raw"], "missed_rank": r["rank"],
            "missed_loc_match": r["loc_match"],
            **{f"top{i+1}": items.at[t, "item_title_raw"] for i, t in enumerate(top5)},
        })
pd.DataFrame(rows).to_csv(f"{C.WORK_DIR}/misses.csv", index=False, encoding="utf-8-sig")
print(f"сохранено {C.WORK_DIR}/misses.csv ({len(rows)} строк) - откройте, отсортируйте по query")
