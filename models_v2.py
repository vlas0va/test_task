"""
models_v2.py
============
Несколько моделей ранжирования поверх одного и того же пула/признаков +
ансамбль. Идея "обучить N моделей и брать разные товары из них": модели с
разными лоссами ошибаются по-разному, поэтому
  1) скор ансамбля = взвешенный RRF: sum_m w_m / (60 + rank_m) (ранги внутри
     запроса, так что шкалы скоров моделей не важны);
  2) плюс квоты "top-q от каждой модели" (столбцы rank_model_<имя>) -
     подбираются в postprocess_v2.tune_postprocess вместе с остальными эвристиками.
Веса подбираются на valid; модель с весом 0 фактически выключена.

Модели:
  lgb_lambdarank    - LightGBM, lambdarank (оптимизирует NDCG@50 внутри запроса)
  lgb_binary        - LightGBM, бинарная классификация "выбран / нет" (pointwise)
  catboost_yetirank - CatBoostRanker, YetiRank (часто сильнее lambdarank на ранжировании;
                      нужен `uv add catboost`, на GPU: config_v2.CATBOOST_TASK_TYPE="GPU")
"""
import os
import json
import time
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb

RRF_K = 60


def _groups(d):
    return d.groupby("qrow", sort=False).size().values  # строки отсортированы по qrow


class Ranker:
    def __init__(self, name, model, kind):
        self.name, self.model, self.kind = name, model, kind

    def predict(self, X):
        if self.kind == "lgb":
            return self.model.predict(X, num_iteration=self.model.best_iteration or None)
        return self.model.predict(X)

    def save(self, dir_):
        os.makedirs(dir_, exist_ok=True)
        path = os.path.join(dir_, f"{self.name}.{'txt' if self.kind == 'lgb' else 'cbm'}")
        if self.kind == "lgb":
            self.model.save_model(path, num_iteration=self.model.best_iteration or None)
        else:
            self.model.save_model(path)
        return path

    @staticmethod
    def load(name, path):
        if path.endswith(".txt"):
            return Ranker(name, lgb.Booster(model_file=path), "lgb")
        from catboost import CatBoost
        m = CatBoost()
        m.load_model(path)
        return Ranker(name, m, "cat")

def _cap_groups(d, max_rows):
    """CatBoost на GPU не принимает запросы длиннее 1023 строк. Оставляем в каждом
    запросе все позитивы + лучших кандидатов по min-рангу среди каналов (худшие
    негативы отбрасываются только для обучения CatBoost; порядок строк сохраняется)."""
    rank_cols = [c for c in d.columns if c.startswith("rank_") and not c.startswith("rank_model_")]
    key = d[rank_cols].min(axis=1) - 1e6 * d["label"]
    pos_in_group = key.groupby(d["qrow"].values).rank(method="first").values
    return d[pos_in_group <= max_rows]

def train_one(name, tr, va, feats, task_type="CPU"):
    t0 = time.time()
    if name == "lgb_lambdarank":
        params = dict(objective="lambdarank", metric="ndcg", eval_at=[50], lambdarank_truncation_level=60,
                      learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0)
        dtr = lgb.Dataset(tr[feats], label=tr["label"], group=_groups(tr))
        dva = lgb.Dataset(va[feats], label=va["label"], group=_groups(va), reference=dtr)
        m = lgb.train(params, dtr, 3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
        r = Ranker(name, m, "lgb")
    elif name == "lgb_binary":
        params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
                      min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                      lambda_l2=1.0, verbose=-1, seed=1)
        dtr = lgb.Dataset(tr[feats], label=tr["label"])
        dva = lgb.Dataset(va[feats], label=va["label"], reference=dtr)
        m = lgb.train(params, dtr, 3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
        r = Ranker(name, m, "lgb")
    elif name == "catboost_yetirank":
        try:
            from catboost import CatBoostRanker, Pool
        except ImportError:
            print("[models] catboost не установлен (uv add catboost) - пропускаю catboost_yetirank")
            return None
        if task_type == "GPU":
            tr, va = _cap_groups(tr, 1000), _cap_groups(va, 1000)
        ptr = Pool(tr[feats], label=tr["label"], group_id=tr["qrow"].values)        
        pva = Pool(va[feats], label=va["label"], group_id=va["qrow"].values)
        # на CPU YetiRank на ~5 млн строк - десятки минут; на GPU (task_type="GPU") в разы быстрее
        m = CatBoostRanker(loss_function="YetiRank", eval_metric="NDCG:top=50", iterations=1500,
                           learning_rate=0.1, depth=6, l2_leaf_reg=3, random_seed=2,
                           od_type="Iter", od_wait=150, use_best_model=True,
                           task_type=task_type, verbose=200)
        m.fit(ptr, eval_set=pva)
        r = Ranker(name, m, "cat")
    else:
        raise ValueError(name)
    print(f"[models] {name}: {time.time()-t0:.0f}s")
    return r


def rank_within_query(qrow, score):
    """0 = лучший внутри своего запроса."""
    order = np.lexsort((-score, qrow))
    q_sorted = qrow[order]
    grp_start = np.r_[0, np.nonzero(np.diff(q_sorted))[0] + 1]
    grp_len = np.diff(np.r_[grp_start, len(q_sorted)])
    within = np.arange(len(q_sorted)) - np.repeat(grp_start, grp_len)
    out = np.empty(len(score), np.int32)
    out[order] = within
    return out


def model_ranks(df, rankers, feats):
    """Добавляет в df столбцы rank_model_<имя>; возвращает dict имя -> ранги."""
    ranks = {}
    for r in rankers:
        s = r.predict(df[feats])
        ranks[r.name] = rank_within_query(df["qrow"].values, s)
        df[f"rank_model_{r.name}"] = ranks[r.name].astype(np.int16).clip(max=9999)
    return ranks


def ensemble_score(ranks: dict, weights: dict):
    s = None
    for name, rk in ranks.items():
        w = weights.get(name, 0.0)
        if w:
            v = w / (RRF_K + rk.astype(np.float64))
            s = v if s is None else s + v
    return s


def tune_weights(ranks: dict, qrow, label, n_rel, recall_fn, grid=(0.0, 0.5, 1.0, 2.0), verbose=True):
    names = list(ranks)
    best = (-1, None)
    for ws in itertools.product(grid, repeat=len(names)):
        if not any(ws):
            continue
        w = dict(zip(names, ws))
        r = recall_fn(qrow, ensemble_score(ranks, w), label, n_rel)
        if r > best[0] + 1e-6:
            best = (r, w)
    if verbose:
        print(f"[ensemble] лучшие веса на valid: {best[1]} -> recall@50 {best[0]:.4f}")
    return best[1]


def save_ensemble(dir_, rankers, weights, feats):
    os.makedirs(dir_, exist_ok=True)
    meta = {"models": [{"name": r.name, "path": r.save(dir_)} for r in rankers],
            "weights": weights, "features": feats, "rrf_k": RRF_K}
    with open(os.path.join(dir_, "ensemble.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_ensemble(dir_):
    with open(os.path.join(dir_, "ensemble.json")) as f:
        meta = json.load(f)
    rankers = [Ranker.load(m["name"], m["path"]) for m in meta["models"]]
    return rankers, meta["weights"], meta["features"]
