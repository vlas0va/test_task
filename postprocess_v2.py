"""
postprocess_v2.py
=================
Recall@K по плоской таблице пула + эвристики финального отбора топ-50 поверх
скора ре-ранкера/ансамбля. ВСЕ эвристики подбираются на valid (tune_postprocess)
и принимаются, только если дают прирост > min_gain, - а не "на глаз".

Эвристики:
  1. Квоты (quotas): гарантировать попадание top-q любого столбца rank_* -
     канала кандидатов (rank_bm25_geo, ...) или отдельной модели ансамбля
     (rank_model_lgb_lambdarank, ...). Это и есть "обучить N моделей и брать
     разные товары из каждой".
  2. Жёсткие фильтры (hard): объявления, у которых признак = 0, уходят в КОНЕЦ
     списка (не выкидываются: если нормальных меньше 50, место всё равно занимается):
        filter_ok  - не проходит фильтр "Вид/Тип услуги" (по train 98%/95% позитивов проходят)
        in_geo     - вне гео-зоны запроса (осторожно: 18% позитивов не из своей локации)
        rating_ok  - ниже порога рейтинга из фильтра (в бенчмарке таких запросов 0)
        mc_plausible - микрокатегория объявления почти не встречается для слов запроса
                     (P(микрокат | слова) < 1%) - "фильтрация по категории": search_category
                     в данных бесполезна (114 почти всегда), реальная категория - микрокатегория
  3. Разнообразие (diversity): последние D мест из 50 отдаются лучшим по скору
     объявлениям, у которых значение столбца НЕ встречается среди первых 50-D:
        item_microcat - другая микрокатегория (запрос неоднозначен: "шар" - воздушные / бильярд)
        in_geo        - "не местное" объявление (18% выборов - из другой локации)
        rev_bucket    - другая "отзывная группа" (0 отзывов / 1-10 / 10+)
"""
import json
import numpy as np
import pandas as pd

BIG = 1e6
DIVERSITY_COLS = ("item_microcat", "in_geo", "rev_bucket")
HARD_COLS = ("filter_ok", "mc_plausible", "in_geo", "rating_ok")


def add_helper_cols(df: pd.DataFrame):
    if "rev_bucket" not in df.columns and "item_reviews_count" in df.columns:
        # item_reviews_count = log1p(n): 0 | 1..10 | >10
        r = df["item_reviews_count"].values
        df["rev_bucket"] = np.where(r <= 0, 0, np.where(r <= np.log1p(10), 1, 2)).astype(np.int8)
    if "mc_plausible" not in df.columns and "tok_mc_share" in df.columns:
        v = df["tok_mc_share"].values
        df["mc_plausible"] = ((v >= 0.01) | np.isnan(v)).astype(np.int8)
    return df


def topk_mask(qrow: np.ndarray, prio: np.ndarray, k: int = 50) -> np.ndarray:
    """Булева маска строк, попавших в top-k своего запроса по prio."""
    order = np.lexsort((-prio, qrow))
    q_sorted = qrow[order]
    grp_start = np.r_[0, np.nonzero(np.diff(q_sorted))[0] + 1]
    grp_len = np.diff(np.r_[grp_start, len(q_sorted)])
    within = np.arange(len(q_sorted)) - np.repeat(grp_start, grp_len)
    mask = np.zeros(len(qrow), bool)
    mask[order[within < k]] = True
    return mask


def adjusted_priority(df: pd.DataFrame, score: np.ndarray, params: dict) -> np.ndarray:
    prio = score.astype(np.float64).copy()
    for col, q in params.get("quotas", {}).items():
        c = col if col.startswith("rank_") else f"rank_{col}"
        if q and c in df.columns:
            prio += BIG * (df[c].values < q)
    for col in params.get("hard", []):
        if col in df.columns:
            prio -= 10 * BIG * (df[col].values == 0)
    return prio


def select_topk(df: pd.DataFrame, score: np.ndarray, params: dict, k: int = 50) -> np.ndarray:
    """Маска выбранных строк (<= k на запрос) с учётом всех эвристик."""
    prio = adjusted_priority(df, score, params)
    div = {c: d for c, d in params.get("diversity", {}).items() if d and c in df.columns}
    if not div:
        return topk_mask(df["qrow"].values, prio, k)
    n_div = sum(div.values())
    main = topk_mask(df["qrow"].values, prio, k - n_div)
    qrow = df["qrow"].values
    order = np.lexsort((-prio, qrow))
    q_sorted = qrow[order]
    bounds = np.r_[0, np.nonzero(np.diff(q_sorted))[0] + 1, len(q_sorted)]
    vals = {c: df[c].values for c in div}
    mask = main.copy()
    for s, e in zip(bounds[:-1], bounds[1:]):
        idx = order[s:e]                    # строки запроса по убыванию приоритета
        in_main = main[idx]
        taken = int(in_main.sum())
        rest = idx[~in_main]
        used = np.zeros(len(rest), bool)
        for c, d in div.items():
            seen = set(vals[c][idx[in_main]].tolist())
            cand = np.nonzero((~used) & ~np.isin(vals[c][rest], list(seen)))[0][:d]
            used[cand] = True
        # оставшиеся свободные места - просто следующие по приоритету
        free = k - taken - int(used.sum())
        if free > 0:
            extra = np.nonzero(~used)[0][:free]
            used[extra] = True
        mask[rest[used]] = True
    return mask


def recall_from_mask(qrow, mask, label, n_rel: pd.Series, return_per_query=False):
    """n_rel: qrow -> число релевантных ВСЕГО (включая не попавшие в пул)."""
    hits = pd.Series(label[mask].astype(np.int32)).groupby(qrow[mask]).sum()
    hits = hits.reindex(n_rel.index).fillna(0)
    per_q = (hits / n_rel.clip(lower=1))[n_rel > 0]
    r = float(per_q.mean())
    return (r, per_q) if return_per_query else r


def recall_at_k(qrow, prio, label, n_rel: pd.Series, k: int = 50, return_per_query=False):
    return recall_from_mask(qrow, topk_mask(qrow, prio, k), label, n_rel, return_per_query)


def pool_recall(qrow, label, n_rel: pd.Series, mask=None):
    """Потолок: доля релевантных, попавших в пул вообще (или в подмножество mask)."""
    m = np.ones(len(qrow), bool) if mask is None else mask
    return recall_from_mask(qrow, m, label, n_rel)


def eval_params(df, score, n_rel, params, k=50):
    return recall_from_mask(df["qrow"].values, select_topk(df, score, params, k), df["label"].values, n_rel)


def tune_postprocess(df_valid: pd.DataFrame, score_valid: np.ndarray, n_rel_valid: pd.Series,
                     k=50, verbose=True, min_gain=0.002, quota_values=(0, 3, 5, 10),
                     div_values=(0, 1, 2, 3)):
    """Координатный спуск по: жёстким фильтрам, квотам (все rank_*), разнообразию."""
    add_helper_cols(df_valid)
    rank_cols = [c for c in df_valid.columns if c.startswith("rank_")]
    cur = {"quotas": {}, "hard": [], "diversity": {}}
    best_r = eval_params(df_valid, score_valid, n_rel_valid, cur, k)
    if verbose:
        print(f"[postprocess] без эвристик: recall@{k} = {best_r:.4f}")

    def try_(p, what):
        nonlocal best_r, cur
        r = eval_params(df_valid, score_valid, n_rel_valid, p, k)
        if r > best_r + min_gain:
            best_r, cur = r, p
            if verbose:
                print(f"   + {what}: {r:.4f}")

    for col in HARD_COLS:
        if col in df_valid.columns:
            try_({**cur, "hard": cur["hard"] + [col]}, f"hard {col}")
    for _ in range(2):
        for c in rank_cols:
            for q in quota_values:
                try_({**cur, "quotas": {**cur["quotas"], c: q}}, f"quota {c}={q}")
        for c in DIVERSITY_COLS:
            if c in df_valid.columns:
                for d in div_values:
                    try_({**cur, "diversity": {**cur["diversity"], c: d}}, f"diversity {c}={d}")
    if verbose:
        print(f"[postprocess] итог на valid: {best_r:.4f}  params={cur}")
    return cur


def save_params(params, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


def load_params(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"quotas": {}, "hard": [], "diversity": {}}


def apply_postprocess(df, score, params):
    """Совместимость: приоритет без разнообразия (для 06)."""
    return adjusted_priority(df, score, params)
