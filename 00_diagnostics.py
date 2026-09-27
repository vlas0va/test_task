# %% [markdown]
# # 00. Статистики данных, на которых основаны решения в пайплайне
#
# Ничего не обучает, несколько минут. Числа в комментариях - результат на данных задачи.

# %%
import numpy as np
import pandas as pd

from data import load_train, load_benchmark, biencoder_val_uids
from text_prep import query_key_series, parse_rating_threshold, parse_filter_pairs

tq, ti, tp = load_train()
bq, bi = load_benchmark()
print(f"train: queries={len(tq)} items={len(ti)} pairs={len(tp)} | benchmark: queries={len(bq)} items={len(bi)}")

# %% 1. Пересечение train и benchmark
# Корпус бенчмарка есть в train_items на 9.6%, тексты запросов - на 41%.
# В схеме обучения ре-ранкера (F против H) - 42% и 88%. Признаки уровня конкретного
# объявления (популярность и т.п.) на бенчмарке работали бы хуже, чем на обучении -> не используются.
print("объявления бенчмарка, которые есть в train_items:", round(bi["item_id"].isin(set(ti["item_id"])).mean(), 3))
tq["qkey"] = query_key_series(tq).values
bq["qkey"] = query_key_series(bq).values
print("запросы бенчмарка, чей текст (стемы) есть в train:", round(bq["qkey"].isin(set(tq["qkey"])).mean(), 3))

f_uids = set(biencoder_val_uids(tq))
fq, hq = tq[tq["query_uid"].isin(f_uids)], tq[~tq["query_uid"].isin(f_uids)]
fp = tp[tp["query_uid"].isin(f_uids)]
h_items = set(tp.loc[~tp["query_uid"].isin(f_uids), "item_id"])
print("[F против H] тексты F, которые есть в H:", round(fq["qkey"].isin(set(hq["qkey"])).mean(), 3))
print("[F против H] позитивы F, которые выбирали в H:", round(fp["item_id"].isin(h_items).mean(), 3))

# %% 2. Локация
# 82% выборов из локации поиска. У 17% запросов бенчмарка в их локации нет ни одного объявления:
# это регионы (107620, 107621, 621540), выборы оттуда уходят в конкретные города.
tpx = tp.merge(tq[["query_uid", "search_location_id", "search_category", "search_infm_params_text"]],
               on="query_uid").merge(
    ti[["item_id", "item_location_id", "item_category_id", "item_rating"]], on="item_id")
tpx["loc_match"] = tpx["item_location_id"] == tpx["search_location_id"]
print("позитивы из той же локации:", round(tpx["loc_match"].mean(), 3))
loc_sizes = bi.groupby("item_location_id").size()
print("запросы бенчмарка без объявлений своей локации:",
      round((bq["search_location_id"].map(loc_sizes).fillna(0) == 0).mean(), 3))
print("частые переходы локаций среди выборов не из своей локации:")
print(tpx[~tpx["loc_match"]].groupby(["search_location_id", "item_location_id"]).size()
      .sort_values(ascending=False).head(10))

# %% 3. Категория
# search_category почти всегда 114 (в бенчмарке 9% нулей) - как фильтр бесполезна.
print("search_category, train:\n", tq["search_category"].value_counts(normalize=True).head(5).round(4))
print("search_category, benchmark:\n", bq["search_category"].value_counts(normalize=True).head(5).round(4))

# %% 4. Фильтр рейтинга
# 0.2% запросов train и 0 запросов бенчмарка - жёсткий фильтр по рейтингу ничего не даст.
thr_tr = tq["search_infm_params_text"].map(parse_rating_threshold)
thr_b = bq["search_infm_params_text"].map(parse_rating_threshold)
print("доля запросов с фильтром рейтинга: train", round(thr_tr.notna().mean(), 4),
      " benchmark", round(thr_b.notna().mean(), 4))

# %% 5. Фильтры "Вид услуги" / "Тип услуги"
# У выбранных объявлений пара есть в параметрах: "Вид услуги" 98%, "Тип услуги" 95.5%.
# "Кто оказывает услуги", "Онлайн-запись" в параметрах не отражаются.
x = tpx.merge(ti[["item_id", "item_infm_params_text"]], on="item_id")
rows = []
for sp_, ip in zip(x["search_infm_params_text"], x["item_infm_params_text"]):
    ip = str(ip).lower()
    for k, v in parse_filter_pairs(sp_):
        if v:
            rows.append((k, f"{k} {v}".lower() in ip))
r = pd.DataFrame(rows, columns=["key", "match"])
print(r.groupby("key")["match"].agg(["mean", "size"]).sort_values("size", ascending=False).round(3))

# %% 6. Выбросы и сдвиг распределения
# До 278 выборов на запрос (99.9% <= 18). В train 354к запросов и 74к уникальных текстов,
# 2.45 слова в среднем; в бенчмарке 3.2 слова и 59% новых текстов.
n = tp.groupby("query_uid").size()
print("выборов на запрос, квантили:", n.quantile([.5, .9, .99, .999]).to_dict(), " запросов с >20:", int((n > 20).sum()))
print("слов в запросе: train", round(tq["search_query"].str.split().str.len().mean(), 2),
      " benchmark", round(bq["search_query"].str.split().str.len().mean(), 2))
print("уникальных текстов в train:", tq["search_query"].str.lower().nunique(), "из", len(tq))
