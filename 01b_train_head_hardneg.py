# %% [markdown]
# # 01b (опционально) — голова эмбеддингов с HARD-негативами
#
# Про идею "LLM для майнинга негативов": LLM тут не нужна. Хард-негативы
# достаются бесплатно из самой эмбеддинг-модели: для запроса берём объявления,
# которые модель считает похожими (ранги 30..200 по косинусу), но пользователь
# их не выбирал. LLM могла бы только ФИЛЬТРОВАТЬ ложные негативы ("это тоже
# подходит, просто не выбрали"), но на ~350к запросов это дни генерации на 6 ГБ.
# Вместо неё - дешёвый фильтр: не берём в негативы объявления, которые
# выбирались по ТАКОМУ ЖЕ тексту запроса в других запросах train.
#
# Почему ранги 30..200, а не 1..30: в услугах куча равнозначно подходящих
# объявлений (100 "автоподборов" в городе), самые верхние "негативы" чаще всего
# ложные - обучение на них учит модель отталкивать правильные объявления.
#
# Использует кэш ./embed_cache из 01 (сырые e5, e5 не гоняется!). Обучение
# головы - минуты. Затем: собирает ./biencoder_hardneg -> в 02 поставьте
# MODEL_PATH="./biencoder_hardneg" и НОВЫЕ CACHE_DIR, прогоните оба режима,
# затем 04 и generate_final_answer_v2 (в config_v2 поправьте TRAIN_EMB_DIR/BENCH_EMB_DIR).
#
# Сравнивайте val recall@50 с тем, что печатал 01 (та же процедура оценки).

# %%
import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

import config_v2 as C
from data_v2 import load_train, biencoder_val_uids
from custom_head import DoubleGeGLUHead
from text_prep import query_key_series

BASE_MODEL = "intfloat/multilingual-e5-base"   # тот же, что в 01
EMBED_CACHE_DIR = "./embed_cache"               # из 01
CHECKPOINT_DIR = "./head_checkpoints_hn"
OUTPUT_DIR = "./biencoder_hardneg"
INIT_HEAD = None      # например "./head_checkpoints/epoch_6" - стартовать с уже обученной головы
EPOCHS = 6
BATCH_SIZE = 256
HEAD_LR = 1e-4 if INIT_HEAD else 2e-4
WARMUP_RATIO = 0.1
SCALE = 20.0
HN_FROM, HN_TO = 30, 200   # из какого диапазона рангов брать хард-негативы
HN_REFRESH_EVERY = 2       # перемайнивать негативы текущей головой каждые N эпох (0 = только сырой e5)
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# %% данные, split как в 01
tq, ti, tp = load_train()
val_uids = set(biencoder_val_uids(tq))
uid_to_pos = pd.Series(np.arange(len(tq)), index=tq["query_uid"].values)
item_to_pos = pd.Series(np.arange(len(ti)), index=ti["item_id"].values)
tp = tp.assign(qp=uid_to_pos.reindex(tp["query_uid"]).values, ip=item_to_pos.reindex(tp["item_id"]).values).dropna()
tp["qp"] = tp["qp"].astype(np.int64); tp["ip"] = tp["ip"].astype(np.int64)
is_val = tp["query_uid"].isin(val_uids).values
train_q_pos, train_i_pos = tp["qp"].values[~is_val], tp["ip"].values[~is_val]
val_pairs = tp[is_val]
print(f"train pairs {len(train_q_pos)}, val pairs {len(val_pairs)}")

# ложные негативы: все объявления, выбиравшиеся по такому же тексту запроса (только train-часть)
qkey = query_key_series(tq).values
tr_df = pd.DataFrame({"k": qkey[train_q_pos], "ip": train_i_pos})
key_items = tr_df.groupby("k")["ip"].apply(lambda s: set(s.values)).to_dict()

# %% кэш e5 -> GPU
q_cache = np.load(f"{EMBED_CACHE_DIR}/query_emb.npy", mmap_mode="r")
i_cache = np.load(f"{EMBED_CACHE_DIR}/item_emb.npy", mmap_mode="r")
assert len(q_cache) == len(tq) and len(i_cache) == len(ti), "embed_cache не соответствует train_*.parquet"
dim = q_cache.shape[1]
I_all = torch.from_numpy(np.array(i_cache, dtype=np.float32)).to(DEVICE)   # ~1 ГБ fp32 на GPU; при OOM -> .half()
Q_all = torch.from_numpy(np.array(q_cache, dtype=np.float32)).to(DEVICE)

head = DoubleGeGLUHead(dim=dim)
if INIT_HEAD:
    head = DoubleGeGLUHead.load(INIT_HEAD)
head = head.to(DEVICE)


@torch.no_grad()
def encode_all(x, use_head, bs=8192):
    outs = []
    for s in range(0, len(x), bs):
        v = x[s:s + bs]
        if use_head:
            v = head({"sentence_embedding": v})["sentence_embedding"]
        outs.append(F.normalize(v, dim=-1).half())
    return torch.cat(outs)


@torch.no_grad()
def mine_hard_negatives(use_head):
    """Для каждого уникального train-запроса - кандидаты рангов HN_FROM..HN_TO по косинусу."""
    head.eval()
    I_n = encode_all(I_all, use_head)
    uq = np.unique(train_q_pos)
    Q_n = encode_all(Q_all[torch.from_numpy(uq).to(DEVICE)], use_head)
    cand = np.zeros((len(uq), HN_TO - HN_FROM), np.int32)
    for s in tqdm(range(0, len(uq), 1024), desc="mining"):
        sims = Q_n[s:s + 1024] @ I_n.T
        top = sims.topk(HN_TO, dim=1).indices[:, HN_FROM:]
        cand[s:s + 1024] = top.cpu().numpy()
    del I_n, Q_n
    head.train()
    return pd.Series(np.arange(len(uq)), index=uq), cand


def sample_negatives(q_pos, i_pos, row_of, cand, rng):
    """1 хард-негатив на пару; пропускаем ложные негативы (выбирались по тому же тексту)."""
    rows = row_of.reindex(q_pos).values
    out = np.empty(len(q_pos), np.int64)
    for j, (r, qp, ip) in enumerate(zip(rows, q_pos, i_pos)):
        bad = key_items.get(qkey[qp], ())
        c = cand[r]
        for _ in range(8):
            x = int(c[rng.integers(len(c))])
            if x != ip and x not in bad:
                break
        out[j] = x
    return out


# %% оценка - та же процедура, что в 01 (val-запросы + 20000 случайных дистракторов)
rel_items_pos = np.unique(val_pairs["ip"].values)
rng_eval = np.random.default_rng(SEED)
non_rel = np.setdiff1d(np.arange(len(ti)), rel_items_pos)
eval_corpus = np.unique(np.concatenate([rel_items_pos, rng_eval.choice(non_rel, size=min(20000, len(non_rel)), replace=False)]))
val_q = val_pairs["qp"].unique()
rel = val_pairs.groupby("qp")["ip"].apply(set).to_dict()


@torch.no_grad()
def eval_recall(k=50):
    head.eval()
    cv = encode_all(I_all[torch.from_numpy(eval_corpus).to(DEVICE)], True)
    qv = encode_all(Q_all[torch.from_numpy(val_q).to(DEVICE)], True)
    rec = []
    for s in range(0, len(val_q), 1024):
        top = (qv[s:s + 1024] @ cv.T).topk(k, dim=1).indices.cpu().numpy()
        for j, qp in enumerate(val_q[s:s + 1024]):
            hit = set(eval_corpus[top[j]].tolist())
            rec.append(len(hit & rel[qp]) / len(rel[qp]))
    head.train()
    return float(np.mean(rec))


print("recall@50 до обучения:", round(eval_recall(), 4))

# %% обучение: InfoNCE с in-batch негативами + 1 хард-негатив на пару
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
n = len(train_q_pos)
steps = (n + BATCH_SIZE - 1) // BATCH_SIZE
opt = AdamW(head.parameters(), lr=HEAD_LR)
sch = get_linear_schedule_with_warmup(opt, int(steps * EPOCHS * WARMUP_RATIO), steps * EPOCHS)
row_of, cand = mine_hard_negatives(use_head=INIT_HEAD is not None)
best = (-1, -1.0)
for ep in range(EPOCHS):
    if ep > 0 and HN_REFRESH_EVERY and ep % HN_REFRESH_EVERY == 0:
        row_of, cand = mine_hard_negatives(use_head=True)
    rng = np.random.default_rng(SEED + ep)
    neg_all = sample_negatives(train_q_pos, train_i_pos, row_of, cand, rng)
    perm = rng.permutation(n)
    tot = 0.0
    for st in tqdm(range(steps), desc=f"epoch {ep+1}/{EPOCHS}"):
        b = perm[st * BATCH_SIZE:(st + 1) * BATCH_SIZE]
        qx = Q_all[torch.from_numpy(train_q_pos[b]).to(DEVICE)]
        px = I_all[torch.from_numpy(train_i_pos[b]).to(DEVICE)]
        nx = I_all[torch.from_numpy(neg_all[b]).to(DEVICE)]
        q = F.normalize(head({"sentence_embedding": qx})["sentence_embedding"], dim=-1)
        d = F.normalize(head({"sentence_embedding": torch.cat([px, nx])})["sentence_embedding"], dim=-1)
        logits = (q @ d.T) * SCALE                   # (B, 2B): свои позитивы на диагонали
        loss = F.cross_entropy(logits, torch.arange(len(b), device=DEVICE))
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        tot += loss.item()
    r = eval_recall()
    head.save(f"{CHECKPOINT_DIR}/epoch_{ep+1}")
    print(f"epoch {ep+1}: loss={tot/steps:.4f}  val recall@50={r:.4f}")
    if r > best[1]:
        best = (ep + 1, r)
print("лучшая эпоха:", best)

# %% сборка полной модели (e5 + лучшая голова) - как в 01
from sentence_transformers import SentenceTransformer
base = SentenceTransformer(BASE_MODEL, device=DEVICE)
best_head = DoubleGeGLUHead.load(f"{CHECKPOINT_DIR}/epoch_{best[0]}").to(DEVICE)
SentenceTransformer(modules=[base._first_module(), base[1], best_head], device=DEVICE).save(OUTPUT_DIR)
print("сохранено", OUTPUT_DIR)
