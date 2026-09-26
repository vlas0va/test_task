"""
04b_refit_all.py
================
Финальное дообучение: те же модели, те же признаки, то же число деревьев
(x1.1 на больший объём данных) и те же веса ансамбля, что в проверенном
прогоне 04 - но обучение на ВСЕХ запросах пула (train + valid + test).
Проверить офлайн уже нечем, поэтому запускать только на конфигурации,
которая уже прошла честный test.
Результат: {WORK_DIR}/models_refit -> в config_v2 поставить MODELS_DIR на неё
и запустить generate_final_answer_v2.py.
"""
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb

import config_v2 as C
from models_v2 import Ranker, _groups, _cap_groups, save_ensemble, load_ensemble

NEG_KEEP = 0.5          # как в проверенном прогоне
BOOST = 1.1             # данных на ~10% больше -> чуть больше деревьев
OUT_DIR = f"{C.WORK_DIR}/models_refit"

# 1. модели проверенного прогона: число итераций, веса ансамбля, список признаков
old, weights, FEATS = load_ensemble(C.MODELS_DIR)
iters = {r.name: (r.model.current_iteration() if r.kind == "lgb" else r.model.tree_count_) for r in old}
print("итерации проверенного прогона:", iters, " веса:", weights)

# 2. весь пул (train + valid + test), негативы прорежены так же, как при обучении
pool = pd.read_parquet(f"{C.WORK_DIR}/pool_train_F.parquet", columns=FEATS + ["qrow", "label"])
if NEG_KEEP < 1.0:
    rng = np.random.default_rng(0)
    pool = pool[(pool["label"].values == 1) | (rng.random(len(pool)) < NEG_KEEP)]
pool = pool[pool.groupby("qrow")["label"].transform("max") > 0].reset_index(drop=True)
gc.collect()
print(f"обучение на {pool['qrow'].nunique()} запросах, {len(pool)} строк")

# 3. переобучение с фиксированным числом итераций (без early stopping)
new = []
for name in [r.name for r in old]:
    n = max(50, int(iters[name] * BOOST))
    print(f"[refit] {name}: {n} итераций")
    if name == "lgb_lambdarank":
        params = dict(objective="lambdarank", metric="ndcg", eval_at=[50], lambdarank_truncation_level=60,
                      learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0)
        m = lgb.train(params, lgb.Dataset(pool[FEATS], label=pool["label"], group=_groups(pool)), n)
        new.append(Ranker(name, m, "lgb"))
    elif name == "lgb_binary":
        params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
                      min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                      lambda_l2=1.0, verbose=-1, seed=1)
        m = lgb.train(params, lgb.Dataset(pool[FEATS], label=pool["label"]), n)
        new.append(Ranker(name, m, "lgb"))
    elif name == "catboost_yetirank":
        from catboost import CatBoostRanker, Pool
        d = _cap_groups(pool, 1000) if C.CATBOOST_TASK_TYPE == "GPU" else pool
        m = CatBoostRanker(loss_function="YetiRank", iterations=n, learning_rate=0.1, depth=6,
                           l2_leaf_reg=3, random_seed=2, task_type=C.CATBOOST_TASK_TYPE, verbose=200)
        m.fit(Pool(d[FEATS], label=d["label"], group_id=d["qrow"].values))
        new.append(Ranker(name, m, "cat"))
    gc.collect()

save_ensemble(OUT_DIR, new, weights, FEATS)
print("сохранено:", OUT_DIR, "-> в config_v2: MODELS_DIR = f\"{WORK_DIR}/models_refit\"")