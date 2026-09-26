# %% [markdown]
# # 00 — диагностика данных: что из идей имеет смысл, а что нет
#
# Ничего не обучает, только считает статистики (минуты). Отвечает на вопросы:
# 1. Пересекаются ли объявления/тексты запросов бенчмарка с train → стоит ли
#    включать признаки истории (config_v2.USE_ITEM_HISTORY)?
# 2. Насколько важна локация (доля выбранных объявлений из той же локации)?
# 3. Можно ли делать ЖЁСТКИЙ фильтр по рейтингу (все ли позитивы проходят фильтр)?
# 4. Что вообще лежит в search_infm_params_text и search_category?

# %%
import numpy as np
import pandas as pd
import config_v2 as C
from data_v2 import load_train, load_benchmark, biencoder_val_uids
from text_prep import query_key_series, parse_rating_threshold

tq, ti, tp = load_train()
bq, bi = load_benchmark()
print(f"train: queries={len(tq)} items={len(ti)} pairs={len(tp)}  |  benchmark: queries={len(bq)} items={len(bi)}")
print("позитивов на запрос (train):", tp.groupby("query_uid").size().describe().round(2).to_dict())

# %% 1. Пересечения train <-> benchmark (главное для USE_ITEM_HISTORY)
train_items = set(ti["item_id"])
share_bi_in_train = bi["item_id"].isin(train_items).mean()
print(f"доля объявлений КОРПУСА бенчмарка, которые есть в train_items: {share_bi_in_train:.3f}")

tq["qkey"] = query_key_series(tq).values
bq["qkey"] = query_key_series(bq).values
train_keys = set(tq["qkey"])
print(f"доля запросов бенчмарка, чей текст (стемы) встречался в train: {bq['qkey'].isin(train_keys).mean():.3f}")
tq_loc = set(zip(tq["qkey"], tq["search_location_id"]))
print(f"   ... текст+локация встречались в train: "
      f"{np.mean([k in tq_loc for k in zip(bq['qkey'], bq['search_location_id'])]):.3f}")

# то же самое для схемы обучения ре-ранкера: F (3% val) против H (история)
f_uids = set(biencoder_val_uids(tq))
fq, hq = tq[tq["query_uid"].isin(f_uids)], tq[~tq["query_uid"].isin(f_uids)]
h_keys = set(hq["qkey"])
fp = tp[tp["query_uid"].isin(f_uids)]
h_items = set(tp.loc[~tp["query_uid"].isin(f_uids), "item_id"])
print(f"[схема обучения] доля F-запросов, чей текст есть в H: {fq['qkey'].isin(h_keys).mean():.3f}")
print(f"[схема обучения] доля позитивов F, которые выбирались и в H: {fp['item_id'].isin(h_items).mean():.3f}")
print("""
КАК ЧИТАТЬ: если доли для бенчмарка и для схемы обучения похожи - признаки
истории переносятся честно, USE_ITEM_HISTORY=True. Если у бенчмарка они сильно
НИЖЕ (например, бенчмарк собран из новых объявлений/новых запросов) - модель
на обучении переоценит историю -> ставьте USE_ITEM_HISTORY=False.""")

# %% 2. Локация
tpx = tp.merge(tq[["query_uid", "search_location_id", "search_is_delivery_search", "search_category",
                   "search_infm_params_text"]], on="query_uid").merge(
    ti[["item_id", "item_location_id", "item_category_id", "item_microcat_id", "item_rating"]], on="item_id")
tpx["loc_match"] = tpx["item_location_id"] == tpx["search_location_id"]
print("доля позитивов из той же локации:", round(tpx["loc_match"].mean(), 3))
print(tpx.groupby("search_is_delivery_search")["loc_match"].agg(["mean", "size"]).round(3))
loc_sizes = bi.groupby("item_location_id").size()
bq_sup = bq["search_location_id"].map(loc_sizes).fillna(0)
print("объявлений корпуса бенчмарка в локации запроса:", bq_sup.describe().round(0).to_dict())
print(f"доля запросов бенчмарка, у которых в корпусе НЕТ объявлений их локации: {(bq_sup == 0).mean():.3f}")

# %% 3. Категория запроса vs категория объявления
print("доля позитивов с item_category_id == search_category:",
      round((tpx["item_category_id"] == tpx["search_category"]).mean(), 3))
print("search_category train (топ):\n", tq["search_category"].value_counts(normalize=True).head(8).round(4))
print("search_category benchmark (топ):\n", bq["search_category"].value_counts(normalize=True).head(8).round(4))
print("какие item_category у позитивов при search_category == 0 (train):")
print(tpx.loc[tpx["search_category"] == 0, "item_category_id"].value_counts().head(5))

# %% 4. Фильтр рейтинга: можно ли делать жёстким?
tpx["thr"] = tpx["search_infm_params_text"].map(parse_rating_threshold)
w = tpx[tpx["thr"].notna()]
if len(w):
    ok = w["item_rating"].fillna(-1) >= w["thr"]
    print(f"позитивов у запросов с фильтром рейтинга: {len(w)}")
    print(f"   проходят фильтр: {ok.mean():.4f}   без рейтинга вообще: {w['item_rating'].isna().mean():.4f}")
    print("   пороги:", w["thr"].value_counts().to_dict())
    print("""
КАК ЧИТАТЬ: если 'проходят фильтр' ~0.99+ - жёсткий фильтр безопасен и
освобождает места в топ-50 (в postprocess_v2 это rating_hard=True, он всё
равно подбирается по valid автоматически).""")
print("доля запросов с фильтром рейтинга: train",
      round(tq["search_infm_params_text"].map(parse_rating_threshold).notna().mean(), 3),
      " benchmark", round(bq["search_infm_params_text"].map(parse_rating_threshold).notna().mean(), 3))

# %% 5. Какие вообще бывают фильтры и параметры
def head_words(s, n=2):
    return " ".join(str(s).split()[:n]) if isinstance(s, str) and s else "<пусто>"

print("search_infm_params_text (первые 2 слова), benchmark:")
print(bq["search_infm_params_text"].map(head_words).value_counts().head(15))
print("\nпримеры item_infm_params_text:")
for s in ti["item_infm_params_text"].dropna().sample(5, random_state=0):
    print("  ", s[:300].replace("\n", " | "))
print("\nпустых search_query: train", round((tq["search_query"].fillna("").str.strip() == "").mean(), 4),
      " benchmark", round((bq["search_query"].fillna("").str.strip() == "").mean(), 4))

# %% 6. Фильтры "Вид/Тип услуги": доля позитивов, у которых пара "ключ значение" есть в параметрах
from text_prep import parse_filter_pairs
x = tpx.merge(ti[["item_id", "item_infm_params_text"]], on="item_id")
rows = []
for sp, ip in zip(x["search_infm_params_text"], x["item_infm_params_text"]):
    ip = str(ip).lower()
    for k, v in parse_filter_pairs(sp):
        if v:
            rows.append((k, f"{k} {v}".lower() in ip))
r = pd.DataFrame(rows, columns=["key", "match"])
print(r.groupby("key")["match"].agg(["mean", "size"]).sort_values("size", ascending=False).round(3))
print("-> ключи с долей ~0.95+ годятся для жёсткого фильтра / канала *_geo_filt (text_prep.CHECKABLE_FILTER_KEYS)")

# %% 7. Локации поиска без объявлений: куда уходят выборы (переходы локаций)
mm = tpx[~tpx["loc_match"]]
print("топ переходов search_location -> item_location среди НЕ-местных выборов:")
print(mm.groupby(["search_location_id", "item_location_id"]).size().sort_values(ascending=False).head(10))
no_local = bq[~bq["search_location_id"].isin(set(bi["item_location_id"]))]
print(f"запросов бенчмарка без объявлений своей локации: {len(no_local)}; их локации в train как локации поиска: "
      f"{no_local['search_location_id'].isin(set(tq['search_location_id'])).mean():.3f}")

# %% 8. Выбросы и сдвиг распределения train -> benchmark
n = tp.groupby("query_uid").size()
print("выборов на запрос: квантили", n.quantile([.5, .9, .99, .999]).to_dict(), " запросов с >20:", int((n > 20).sum()))
print("слов в запросе: train", round(tq["search_query"].str.split().str.len().mean(), 2),
      " benchmark", round(bq["search_query"].str.split().str.len().mean(), 2))
print("уникальных текстов в train:", tq["search_query"].str.lower().nunique(), "из", len(tq))
