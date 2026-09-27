"""
02b_apply_head.py
=================
Быстрая замена 02 для замороженного бэкбона: итоговые эмбеддинги = head(сырой эмбеддинг бэкбона).

* train: сырые эмбеддинги УЖЕ посчитаны в 01 (EMBED_CACHE_DIR) - прогоняем через голову,
  e5/bge второй раз не запускаем (экономит ~1 час на 345к объявлений);
* benchmark: сырые эмбеддинги считаем бэкбоном (тот же текст и префиксы, что в 01),
  затем та же голова.

На выходе - то же, что давал 02: {CACHE}/query_emb.npy и item_emb.npy (нормированные, float32),
которые читает 04 / generate_final_answer_v2 (config_v2.TRAIN_EMB_DIR / BENCH_EMB_DIR).

Запуск: uv run python -u 02b_apply_head.py
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from text_prep import build_embedding_query_text, build_item_text_for_embedding

# ------------------------------------------------------------------ конфиг (синхронно с 01!)
MODEL_DIR = "./biencoder_bgem3"               # OUTPUT_DIR из 01 (бэкбон + лучшая голова)
RAW_TRAIN_CACHE = "./embed_cache_bgem3"       # EMBED_CACHE_DIR из 01 (сырые эмбеддинги train)
BASE_MODEL_NAME = "deepvk/USER-bge-m3"        # BASE_MODEL из 01 - только для выбора префиксов
MAX_SEQ_LEN = 256                             # как в 01
TRAIN_OUT = "./retrieve_cache_train_bgem3"
BENCH_OUT = "./retrieve_cache_benchmark_bgem3"
BATCH_ITEM = 32                               # при нехватке VRAM - 16
BATCH_QUERY = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

e5 = "e5" in BASE_MODEL_NAME.lower()
Q_PREFIX = "query: " if e5 else ""
P_PREFIX = "passage: " if e5 else ""

full = SentenceTransformer(MODEL_DIR, device=DEVICE, trust_remote_code=True)
head = full[2].float().eval()                                     # DoubleGeGLUHead
backbone = SentenceTransformer(modules=[full[0], full[1]], device=DEVICE)
backbone.max_seq_length = MAX_SEQ_LEN
backbone.half()
print(f"устройство {DEVICE}, dim={full.get_sentence_embedding_dimension()}, префиксы: {Q_PREFIX!r}/{P_PREFIX!r}")


@torch.no_grad()
def apply_head(raw, bs=8192):
    """raw: (n, dim) numpy/mmap -> head -> L2-норма -> float32 numpy."""
    out = np.empty((len(raw), raw.shape[1]), dtype=np.float32)
    for s in range(0, len(raw), bs):
        x = torch.from_numpy(np.array(raw[s:s + bs], dtype=np.float32)).to(DEVICE)
        y = head({"sentence_embedding": x})["sentence_embedding"]
        out[s:s + bs] = F.normalize(y, dim=-1).cpu().numpy()
    return out


def encode_raw(texts, bs):
    """Сырой эмбеддинг бэкбона - ровно как в 01 (fp16, без нормировки)."""
    return backbone.encode(texts, batch_size=bs, show_progress_bar=True,
                           convert_to_numpy=True).astype(np.float32)


# ------------------------------------------------------------------ train: голова поверх кэша 01
os.makedirs(TRAIN_OUT, exist_ok=True)
for name in ("query_emb.npy", "item_emb.npy"):
    dst = os.path.join(TRAIN_OUT, name)
    if os.path.exists(dst):
        print(f"[train] {dst} уже есть - пропускаю")
        continue
    raw = np.load(os.path.join(RAW_TRAIN_CACHE, name), mmap_mode="r")
    t0 = time.time()
    np.save(dst, apply_head(raw))
    print(f"[train] {dst}: {raw.shape} за {time.time()-t0:.0f}s")

# ------------------------------------------------------------------ benchmark: бэкбон + голова
os.makedirs(BENCH_OUT, exist_ok=True)
bq = pd.read_parquet("benchmark_queries.parquet")
q_dst = os.path.join(BENCH_OUT, "query_emb.npy")
if not os.path.exists(q_dst):
    q_text = (Q_PREFIX + build_embedding_query_text(bq)).tolist()
    np.save(q_dst, apply_head(encode_raw(q_text, BATCH_QUERY)))
    print(f"[bench] {q_dst}: {len(bq)}")

i_dst = os.path.join(BENCH_OUT, "item_emb.npy")
if not os.path.exists(i_dst):
    bi = pd.read_parquet("benchmark_items.parquet",
                         columns=["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw"])
    it_text = (P_PREFIX + build_item_text_for_embedding(bi)).tolist()
    del bi
    t0 = time.time()
    raw = encode_raw(it_text, BATCH_ITEM)
    np.save(i_dst, apply_head(raw))
    print(f"[bench] {i_dst}: {raw.shape} за {time.time()-t0:.0f}s")

print("готово:", TRAIN_OUT, BENCH_OUT)
