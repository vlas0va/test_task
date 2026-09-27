"""02. Итоговые эмбеддинги запросов и объявлений (train и benchmark) для одного энкодера.

    python 02_encode.py e5
    python 02_encode.py bge
    python 02_encode.py e5 bench     # только benchmark (для 05 достаточно)

Модель = замороженный бэкбон с HuggingFace + голова из 01 (artifacts/<enc>/head).
Режим из config.ENCODERS[enc]["encode_mode"]:
  full      - текст -> бэкбон + голова в fp16, нормировка (e5);
  head_only - train: голова поверх сырых эмбеддингов из 01, benchmark: бэкбон fp16 -> голова fp32 (bge).
Выход: artifacts/<enc>/emb_train/{query,item}_emb.npy, artifacts/<enc>/emb_bench/... (float32, L2 = 1).
Уже посчитанные файлы пропускаются.

Время на RTX 3060 Laptop: e5 ~1.5 ч (train 345к объявлений + benchmark), bge ~30 мин (только benchmark).
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

import config as C
from custom_head import DoubleGeGLUHead
from text_prep import build_embedding_query_text, build_item_text_for_embedding

ENC = sys.argv[1] if len(sys.argv) > 1 else "e5"
ONLY_BENCH = len(sys.argv) > 2 and sys.argv[2] == "bench"
cfg = C.ENCODERS[ENC]
paths = C.enc_paths(ENC)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Q_BATCH = 128

base = SentenceTransformer(cfg["base_model"], device=DEVICE)
if cfg["max_seq_length"]:
    base.max_seq_length = cfg["max_seq_length"]
head = DoubleGeGLUHead.load(paths["head"]).to(DEVICE).eval()
print(f"{ENC}: {cfg['base_model']}, режим {cfg['encode_mode']}, устройство {DEVICE}")

if cfg["encode_mode"] == "full":
    model = SentenceTransformer(modules=[base[0], base[1], head], device=DEVICE).half()
else:
    backbone = SentenceTransformer(modules=[base[0], base[1]], device=DEVICE).half()


@torch.no_grad()
def apply_head(raw, bs=8192):
    """(n, dim) сырые эмбеддинги -> голова fp32 -> L2-нормировка."""
    out = np.empty((len(raw), raw.shape[1]), dtype=np.float32)
    for s in range(0, len(raw), bs):
        x = torch.from_numpy(np.array(raw[s:s + bs], dtype=np.float32)).to(DEVICE)
        out[s:s + bs] = F.normalize(head({"sentence_embedding": x})["sentence_embedding"], dim=-1).cpu().numpy()
    return out


def encode(texts, bs):
    if cfg["encode_mode"] == "full":
        return model.encode(texts, batch_size=bs, show_progress_bar=True,
                            normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    raw = backbone.encode(texts, batch_size=bs, show_progress_bar=True, convert_to_numpy=True)
    return apply_head(raw.astype(np.float32))


def run(split, queries_file, items_file):
    out_dir = paths[split]
    os.makedirs(out_dir, exist_ok=True)
    q_dst, i_dst = os.path.join(out_dir, "query_emb.npy"), os.path.join(out_dir, "item_emb.npy")
    raw_ok = split == "train" and cfg["encode_mode"] == "head_only"

    if not os.path.exists(q_dst):
        t0 = time.time()
        if raw_ok:
            emb = apply_head(np.load(os.path.join(paths["raw"], "query_emb.npy"), mmap_mode="r"))
        else:
            q = pd.read_parquet(f"{C.DATA_DIR}/{queries_file}")
            emb = encode((cfg["q_prefix"] + build_embedding_query_text(q)).tolist(), Q_BATCH)
        np.save(q_dst, emb)
        print(f"[{split}] {q_dst}: {emb.shape}, {time.time() - t0:.0f}s")

    if not os.path.exists(i_dst):
        t0 = time.time()
        if raw_ok:
            emb = apply_head(np.load(os.path.join(paths["raw"], "item_emb.npy"), mmap_mode="r"))
        else:
            it = pd.read_parquet(f"{C.DATA_DIR}/{items_file}",
                                 columns=["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw"])
            texts = (cfg["p_prefix"] + build_item_text_for_embedding(it)).tolist()
            del it
            emb = encode(texts, cfg["enc_batch"])
        np.save(i_dst, emb)
        print(f"[{split}] {i_dst}: {emb.shape}, {time.time() - t0:.0f}s")


if not ONLY_BENCH:
    run("train", "train_queries.parquet", "train_items.parquet")
run("bench", "benchmark_queries.parquet", "benchmark_items.parquet")
print("готово:", paths["train"], paths["bench"])
