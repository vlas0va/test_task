"""04. Переобучение ансамбля на всех запросах пула (train + valid + test).

Модели, признаки и веса ансамбля - из 03. Число итераций = найденное early stopping
в 03, умноженное на 1.1 (данных примерно на 10% больше). Проверить результат офлайн
уже не на чем, поэтому запускать только после 03 с проверенной конфигурацией.

Выход: artifacts/work/models_refit - этими моделями 05 считает ответ.
"""
import gc
import numpy as np
import pandas as pd

import config as C
from rankers import load_ensemble, save_ensemble, train_one

BOOST = 1.1

old, weights, FEATS = load_ensemble(C.MODELS_DIR)
iters = {r.name: r.n_iter() for r in old}
print("итерации в 03:", iters, " веса:", weights)

pool = pd.read_parquet(f"{C.WORK_DIR}/pool_train_F.parquet", columns=FEATS + ["qrow", "label"])
rng = np.random.default_rng(0)
pool = pool[(pool["label"].values == 1) | (rng.random(len(pool)) < C.NEG_KEEP)]
pool = pool[pool.groupby("qrow")["label"].transform("max") > 0].reset_index(drop=True)
gc.collect()
print(f"обучение на {pool['qrow'].nunique()} запросах, {len(pool)} строк")

new = []
for r in old:
    n = max(50, int(iters[r.name] * BOOST))
    print(f"[refit] {r.name}: {n} итераций")
    new.append(train_one(r.name, pool, None, FEATS, task_type=C.CATBOOST_TASK_TYPE, n_iter=n))
    gc.collect()

save_ensemble(C.FINAL_MODELS_DIR, new, weights, FEATS)
print("сохранено:", C.FINAL_MODELS_DIR)
