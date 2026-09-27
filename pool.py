"""Пул кандидатов и признаки пар (запрос, объявление).

Для батча запросов за один проход:
  1. скоры против всего корпуса плотными матрицами (батч x N): BM25 по полному тексту,
     BM25 по заголовку, косинусы двух эмбеддеров;
  2. кандидаты из каналов config.CHANNELS (top-k по скору с масками гео-зоны и фильтров);
  3. объединение каналов без дублей;
  4. признаки для каждой пары пула, включая ранг в каждом канале.

Одна и та же функция строит пул для обучения (03) и для бенчмарка (05).
"""
import os
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pyarrow as pa
import pyarrow.parquet as pq

from bm25 import BM25Index
from features import precompute_item_arrays
from text_prep import (iter_item_texts, iter_title_texts, stemming_generator, build_query_text,
                       query_only_text, filter_only_text, parse_rating_threshold, checkable_filters)

CHANNEL_NAMES = ["bm25_global", "bm25_geo", "bm25_geo_filt", "dense_global", "dense_geo",
                 "dense_geo_filt", "dense2_global", "dense2_geo", "dense2_geo_filt"]
NO_RANK = 9999   # канал не нашёл объявление
NON_FEATURE_COLS = {"qrow", "item_pos", "label", "item_microcat", "split"}


# ---------------------------------------------------------------- утилиты

def _binary(X):
    X = X.tocsr(copy=True).astype(np.float32)
    X.data[:] = 1.0
    return X


def _binary_shared(X):
    """Та же структура CSR, значения = 1 (indices/indptr общие с X, без копий)."""
    return sp.csr_matrix((np.ones(X.nnz, np.float32), X.indices, X.indptr), shape=X.shape, copy=False)


_CHECK_KEYS_LOWER = tuple(k.lower() for k in ("Вид услуги", "Тип услуги", "Предмет или специальность"))


def _compact_params(p) -> str:
    """Из параметров объявления оставляем куски '<ключ> + 120 символов' для проверяемых
    ключей. Обрезать начало нельзя: 'Тип услуги автосервиса' в медиане на 836-м символе."""
    if not isinstance(p, str) or not p:
        return ""
    p = p.lower()
    parts = []
    for k in _CHECK_KEYS_LOWER:
        i = p.find(k)
        while i >= 0:
            parts.append(p[i:i + len(k) + 120])
            i = p.find(k, i + len(k))
    return " | ".join(parts)


def topk_rows(M, k, valid=None):
    """Построчный top-k плотной матрицы; valid - маска допустимых элементов."""
    k = min(k, M.shape[1])
    if valid is not None:
        M = np.where(valid, M, -np.inf)
    part = np.argpartition(-M, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(M, part, axis=1)
    order = np.argsort(-vals, axis=1)
    part = np.take_along_axis(part, order, axis=1)
    vals = np.take_along_axis(vals, order, axis=1)
    idx_out, sc_out = [], []
    for r in range(M.shape[0]):
        ok = np.isfinite(vals[r])
        idx_out.append(part[r][ok])
        sc_out.append(vals[r][ok].astype(np.float32))
    return idx_out, sc_out


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ---------------------------------------------------------------- второй эмбеддер
# Подгружается по item_id / query_uid / query_id из config.EMB2, поэтому 03-05
# передают в пул только основной эмбеддер. Хранится в float16 (516к x 1024 = 1 ГБ).

def _emb2_cfg():
    import config as C
    return getattr(C, "EMB2", None), C.DATA_DIR


def _gather_fp16(src, rows, chunk=50000):
    """Строки rows из mmap-массива в float16, без полной копии в float32."""
    out = np.empty((len(rows), src.shape[1]), np.float16)
    order = np.argsort(rows)
    for s in range(0, len(rows), chunk):
        o = order[s:s + chunk]
        out[o] = np.asarray(src[rows[o]], dtype=np.float16)
    return out


def load_emb2_items(item_ids, prefer="train"):
    """Эмбеддинги 2-го энкодера для корпуса по item_id. prefer - откуда брать объявление,
    которое есть и в train, и в benchmark: "train" для корпуса 03, "bench" для 05."""
    cfg, data_dir = _emb2_cfg()
    if cfg is None:
        return None
    srcs = {}
    for key, name in (("train", "train_items.parquet"), ("bench", "benchmark_items.parquet")):
        path = os.path.join(cfg[key], "item_emb.npy")
        if not os.path.exists(path):
            continue
        ids = pd.read_parquet(os.path.join(data_dir, name), columns=["item_id"])["item_id"].values
        emb = np.load(path, mmap_mode="r")
        assert len(emb) == len(ids), f"{path}: {emb.shape} != {name} ({len(ids)})"
        pos = pd.Series(np.arange(len(ids)), index=ids)
        srcs[key] = (emb, pos[~pos.index.duplicated()])
    order = [k for k in [prefer] + [k for k in ("train", "bench") if k != prefer] if k in srcs]
    if not order:
        return None
    out = np.zeros((len(item_ids), srcs[order[0]][0].shape[1]), np.float16)
    done = np.zeros(len(item_ids), bool)
    for k in order:
        emb, pos = srcs[k]
        rows = pos.reindex(item_ids).values
        m = ~done & ~np.isnan(rows)
        if m.any():
            out[m] = _gather_fp16(emb, rows[m].astype(np.int64))
            done |= m
        print(f"[emb2] {k}: {int(m.sum())} объявлений")
    assert done.all(), f"[emb2] нет эмбеддинга для {int((~done).sum())} объявлений"
    return out


def load_emb2_queries(queries):
    """Эмбеддинги 2-го энкодера для запросов: train по query_uid, бенчмарк по query_id."""
    cfg, data_dir = _emb2_cfg()
    if cfg is None:
        return None
    if "query_uid" in queries.columns:
        key, col, name = "train", "query_uid", "train_queries.parquet"
    else:
        key, col, name = "bench", "query_id", "benchmark_queries.parquet"
    path = os.path.join(cfg[key], "query_emb.npy")
    assert os.path.exists(path), f"нет {path}"
    ids = pd.read_parquet(os.path.join(data_dir, name), columns=[col])[col].values
    emb = np.load(path, mmap_mode="r")
    assert len(emb) == len(ids), f"{path}: {emb.shape} != {name} ({len(ids)})"
    pos = pd.Series(np.arange(len(ids)), index=ids)
    rows = pos[~pos.index.duplicated()].reindex(queries[col].values).values
    assert not np.isnan(rows).any()
    return _gather_fp16(emb, rows.astype(np.int64)).astype(np.float32)


class _Emb2Scorer:
    """Косинусы запросов со всем корпусом по 2-му эмбеддеру (fp16).
    На GPU - матрица в VRAM. На CPU - то же вычисление в fp32 с округлением результата
    до fp16, как на GPU: так ответ совпадает независимо от устройства."""

    def __init__(self, item_emb2_fp16):
        self.N = len(item_emb2_fp16)
        self.gpu = None
        try:
            import torch
            if torch.cuda.is_available():
                self.torch = torch
                self.gpu = torch.from_numpy(item_emb2_fp16).cuda()
                print(f"[emb2] {tuple(item_emb2_fp16.shape)} fp16 на GPU")
        except Exception as e:
            print(f"[emb2] GPU недоступен ({e}), считаю на CPU")
            self.gpu = None
        self.cpu = None if self.gpu is not None else item_emb2_fp16

    def __del__(self):
        # освободить VRAM сразу: дальше в 03 CatBoost на GPU
        if getattr(self, "gpu", None) is not None:
            self.gpu = None
            try:
                self.torch.cuda.empty_cache()
            except Exception:
                pass

    def scores(self, qe, chunk=65536):
        q16 = np.ascontiguousarray(qe, dtype=np.float16)
        if self.gpu is not None:
            with self.torch.no_grad():
                q = self.torch.from_numpy(q16).cuda()
                return (q @ self.gpu.T).float().cpu().numpy()
        q32 = q16.astype(np.float32)
        out = np.empty((len(qe), self.N), np.float32)
        for s in range(0, self.N, chunk):
            out[:, s:s + chunk] = (q32 @ self.cpu[s:s + chunk].astype(np.float32).T).astype(np.float16)
        return out


# ---------------------------------------------------------------- корпус

class CorpusIndex:
    """Всё, что считается один раз на корпус."""

    def __init__(self, items: pd.DataFrame, item_emb: np.ndarray, history, emb2_prefer="train"):
        from config import BM25_KW, BM25_ITEM_TEXT_KW, BM25_TITLE_KW
        self.items = items.reset_index(drop=True)
        self.item_ids = self.items["item_id"].values
        self.N = len(self.items)

        t0 = time.time()
        self.bm25 = BM25Index(analyzer="word", ngram_range=(1, 1), **BM25_KW)
        self.bm25.fit(stemming_generator(iter_item_texts(self.items, **BM25_ITEM_TEXT_KW)))
        self.bm25_title = BM25Index(analyzer="word", ngram_range=(1, 1), **BM25_TITLE_KW)
        self.bm25_title.fit(stemming_generator(iter_title_texts(self.items)))
        # "слово есть в документе" - для признаков покрытия слов запроса
        self.D_bin = _binary_shared(self.bm25.doc_matrix_)
        self.T_bin = _binary_shared(self.bm25_title.doc_matrix_)
        print(f"[corpus] N={self.N}, BM25 (full + title) built in {time.time() - t0:.1f}s")

        self.arr = precompute_item_arrays(self.items)
        loc_codes, loc_uniques = pd.factorize(self.items["item_location_id"].fillna(-999))
        self.loc_uniques = pd.Index(loc_uniques)
        self.item_loc_code = loc_codes.astype(np.int32)
        self.loc_size = np.bincount(loc_codes, minlength=len(loc_uniques)).astype(np.float32)

        # центр локации = медиана координат её объявлений (у запроса координат нет)
        lat = pd.to_numeric(self.items["item_latitude"], errors="coerce").astype(float)
        lon = pd.to_numeric(self.items["item_longitude"], errors="coerce").astype(float)
        cent = pd.DataFrame({"c": loc_codes, "lat": lat, "lon": lon}).groupby("c")[["lat", "lon"]].median()
        self.loc_lat = cent["lat"].reindex(range(len(loc_uniques))).values.astype(np.float64)
        self.loc_lon = cent["lon"].reindex(range(len(loc_uniques))).values.astype(np.float64)
        self.item_lat, self.item_lon = lat.values, lon.values
        self.item_microcat = self.items["item_microcat_id"].values

        self.params_lower = pd.Series([_compact_params(p) for p in self.items["item_infm_params_text"].values])
        self._filter_mask_cache = {}
        # тексты дальше не нужны
        keep = [c for c in self.items.columns
                if c not in ("item_title_raw", "item_description_raw", "item_infm_params_text")]
        self.items = self.items[keep]

        assert len(item_emb) == self.N, f"item_emb {item_emb.shape} != corpus {self.N}"
        self.item_emb = np.ascontiguousarray(item_emb, dtype=np.float32)
        self.emb2 = None
        e2 = load_emb2_items(self.item_ids, prefer=emb2_prefer)
        if e2 is not None:
            self.emb2 = _Emb2Scorer(e2)
            del e2
        self.history = history
        history.set_corpus(self.items)

    def loc_code_of(self, loc_values):
        return self.loc_uniques.get_indexer(loc_values)   # -1: объявлений этой локации нет

    def filter_mask(self, kv: str) -> np.ndarray:
        """(N,) bool: есть ли в параметрах объявления пара 'вид услуги красота, здоровье'."""
        m = self._filter_mask_cache.get(kv)
        if m is None:
            m = self._filter_mask_cache[kv] = self.params_lower.str.contains(kv, regex=False).values
        return m


# ---------------------------------------------------------------- запросы

def prepare_queries(corpus: CorpusIndex, queries: pd.DataFrame):
    q = queries.reset_index(drop=True)
    qp = {}
    qp["text_stem"] = list(stemming_generator(build_query_text(q)))
    qp["qonly_stem"] = list(stemming_generator(query_only_text(q)))
    qp["filt_stem"] = list(stemming_generator(filter_only_text(q)))
    qp["loc_code"] = corpus.loc_code_of(q["search_location_id"].values)
    qp["loc_raw"] = q["search_location_id"].values
    qp["category"] = q["search_category"].values
    qp["rating_thr"] = q["search_infm_params_text"].map(parse_rating_threshold).values.astype(np.float32)
    qp["has_filter"] = (q["search_infm_params_text"].fillna("").str.len() > 0).values.astype(np.int8)
    qp["filters"] = q["search_infm_params_text"].map(checkable_filters).tolist()
    qp["q_ntok"] = np.array([len(s.split()) for s in qp["qonly_stem"]], np.int16)
    qp["qkey"] = corpus.history.query_keys(q)
    return q, qp


# ---------------------------------------------------------------- пул и признаки

def build_pool_features(corpus: CorpusIndex, queries: pd.DataFrame, query_emb: np.ndarray,
                        out_path: str, channels: dict, batch_size: int, geo_min_p: float,
                        positives: dict = None):
    """Пул и признаки для всех запросов -> parquet (пишется по батчам).
    positives: {номер запроса: set(позиций объявлений)} - для обучения (колонка label)."""
    q, qp = prepare_queries(corpus, queries)
    assert len(query_emb) == len(q), f"query_emb {query_emb.shape} != queries {len(q)}"
    query_emb2 = load_emb2_queries(q) if corpus.emb2 is not None else None
    use_dense2 = query_emb2 is not None
    hist = corpus.history
    ch_active = [c for c in CHANNEL_NAMES if channels.get(c, 0) > 0
                 and (use_dense2 or not c.startswith("dense2_"))]
    print(f"[pool] queries={len(q)} каналы: {ch_active}")

    Q_full = corpus.bm25.transform_query(qp["text_stem"]).astype(np.float32)
    Q_title = corpus.bm25_title.transform_query(qp["qonly_stem"]).astype(np.float32)
    Q_only_bin = _binary(corpus.bm25.transform_query(qp["qonly_stem"]))
    Q_title_bin = _binary(Q_title)
    F_bin = _binary(corpus.bm25.transform_query(qp["filt_stem"]))
    nq_tok = np.maximum(np.asarray(Q_only_bin.sum(axis=1)).ravel(), 1)
    nq_tok_t = np.maximum(np.asarray(Q_title_bin.sum(axis=1)).ravel(), 1)
    nf_tok = np.asarray(F_bin.sum(axis=1)).ravel()
    mc_dist = hist.query_mc_dist(qp["qkey"])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = None
    t0 = time.time()
    n_rows = 0
    for start in range(0, len(q), batch_size):
        end = min(start + batch_size, len(q))
        sl = slice(start, end)
        b = end - start

        # гео-зона: P(локация объявления | локация поиска) >= geo_min_p или та же локация
        locm = corpus.item_loc_code[None, :] == qp["loc_code"][sl][:, None]
        Wl = hist.geo_weights(qp["loc_raw"][sl], corpus.loc_uniques)
        geo_w = Wl[:, corpus.item_loc_code]
        in_geo = (geo_w >= geo_min_p) | locm
        top_loc = Wl.argmax(axis=1)          # центр зоны, если в локации поиска нет объявлений
        has_w = Wl.max(axis=1) > 0

        # фильтры "Вид/Тип услуги"
        filt_ok = np.ones((b, corpus.N), bool)
        has_filt = np.zeros(b, bool)
        for r, kvs in enumerate(qp["filters"][sl]):
            for kv in kvs:
                filt_ok[r] &= corpus.filter_mask(kv)
                has_filt[r] = True

        # каналы
        S = (Q_full[sl] @ corpus.bm25.doc_matrix_.T).toarray()
        has_bm = S > 0
        C = query_emb[sl].astype(np.float32) @ corpus.item_emb.T
        cands = {}
        if "bm25_global" in ch_active:
            cands["bm25_global"] = topk_rows(S, channels["bm25_global"], valid=has_bm)
        if "bm25_geo" in ch_active:
            cands["bm25_geo"] = topk_rows(S, channels["bm25_geo"], valid=has_bm & in_geo)
        if "bm25_geo_filt" in ch_active and has_filt.any():
            cands["bm25_geo_filt"] = topk_rows(S, channels["bm25_geo_filt"],
                                               valid=has_bm & in_geo & filt_ok & has_filt[:, None])
        if "dense_global" in ch_active:
            cands["dense_global"] = topk_rows(C, channels["dense_global"])
        if "dense_geo" in ch_active:
            cands["dense_geo"] = topk_rows(C, channels["dense_geo"], valid=in_geo)
        if "dense_geo_filt" in ch_active and has_filt.any():
            cands["dense_geo_filt"] = topk_rows(C, channels["dense_geo_filt"],
                                                valid=in_geo & filt_ok & has_filt[:, None])
        if use_dense2:
            C2 = corpus.emb2.scores(query_emb2[sl])
            if "dense2_global" in ch_active:
                cands["dense2_global"] = topk_rows(C2, channels["dense2_global"])
            if "dense2_geo" in ch_active:
                cands["dense2_geo"] = topk_rows(C2, channels["dense2_geo"], valid=in_geo)
            if "dense2_geo_filt" in ch_active and has_filt.any():
                cands["dense2_geo_filt"] = topk_rows(C2, channels["dense2_geo_filt"],
                                                     valid=in_geo & filt_ok & has_filt[:, None])

        # объединение каналов + ранг в каждом канале
        parts_q, parts_i, parts_c, parts_r = [], [], [], []
        for ci, ch in enumerate(ch_active):
            if ch not in cands:
                continue
            for r, idx in enumerate(cands[ch][0]):
                if len(idx) == 0:
                    continue
                parts_q.append(np.full(len(idx), r, np.int64))
                parts_i.append(idx.astype(np.int64))
                parts_c.append(np.full(len(idx), ci, np.int8))
                parts_r.append(np.arange(len(idx), dtype=np.int16))
        if not parts_q:
            continue
        lq, li = np.concatenate(parts_q), np.concatenate(parts_i)
        lc, lr = np.concatenate(parts_c), np.concatenate(parts_r)
        uniq, inv = np.unique(lq * corpus.N + li, return_inverse=True)   # сортировка по (запрос, объявление)
        ranks = np.full((len(uniq), len(ch_active)), NO_RANK, np.int16)
        np.minimum.at(ranks, (inv, lc), lr)
        qloc = (uniq // corpus.N).astype(np.int64)    # номер запроса в батче
        ipos = (uniq % corpus.N).astype(np.int64)     # позиция в корпусе
        qglob = qloc + start

        # признаки
        f = {}
        f["qrow"] = qglob.astype(np.int32)
        f["item_pos"] = ipos.astype(np.int32)
        f["item_microcat"] = corpus.item_microcat[ipos].astype(np.int64)
        for ci, ch in enumerate(ch_active):
            f[f"rank_{ch}"] = ranks[:, ci]
        f["n_channels"] = (ranks < NO_RANK).sum(axis=1).astype(np.int8)

        bm = S[qloc, ipos]
        row_max = S.max(axis=1)
        geo_row_max = np.where(in_geo, S, 0).max(axis=1)
        f["bm25"] = bm
        f["bm25_rel"] = bm / np.maximum(row_max[qloc], 1e-6)
        # если в гео-зоне нет ни одного совпадения по BM25 - nan
        f["bm25_rel_geo"] = np.where(geo_row_max[qloc] > 0, bm / np.maximum(geo_row_max[qloc], 1e-6), np.nan)
        f["geo_supply_bm25"] = np.log1p((has_bm & in_geo).sum(axis=1)[qloc]).astype(np.float32)
        f["geo_supply_filt"] = np.log1p((has_bm & in_geo & filt_ok).sum(axis=1)[qloc]).astype(np.float32)
        f["n_bm25_hits_log"] = np.log1p(has_bm.sum(axis=1)[qloc]).astype(np.float32)
        del S, has_bm

        f["cov_full"] = (Q_only_bin[sl] @ corpus.D_bin.T).toarray()[qloc, ipos] / nq_tok[qglob]
        St = (Q_title[sl] @ corpus.bm25_title.doc_matrix_.T).toarray()
        f["bm25_title"] = St[qloc, ipos]
        del St
        f["cov_title"] = (Q_title_bin[sl] @ corpus.T_bin.T).toarray()[qloc, ipos] / nq_tok_t[qglob]
        fcov = (F_bin[sl] @ corpus.D_bin.T).toarray()[qloc, ipos]
        f["filter_cov"] = np.where(nf_tok[qglob] > 0, fcov / np.maximum(nf_tok[qglob], 1), np.nan).astype(np.float32)
        # 1 - все проверяемые фильтры совпали, 0 - нет, nan - фильтров нет
        f["filter_ok"] = np.where(has_filt[qloc], filt_ok[qloc, ipos].astype(np.float32), np.nan)

        cos = C[qloc, ipos]
        f["cos"] = cos
        f["cos_minus_max"] = cos - C.max(axis=1)[qloc]
        f["cos_minus_max_geo"] = cos - np.where(in_geo, C, -1.0).max(axis=1)[qloc]
        del C
        if use_dense2:
            cos2 = C2[qloc, ipos]
            f["cos2"] = cos2
            f["cos2_minus_max"] = cos2 - C2.max(axis=1)[qloc]
            f["cos2_minus_max_geo"] = cos2 - np.where(in_geo, C2, -1.0).max(axis=1)[qloc]
            f["cos_mean12"] = 0.5 * (f["cos"] + cos2)
            del C2

        a = corpus.arr
        f["location_match"] = locm[qloc, ipos].astype(np.int8)
        f["in_geo"] = in_geo[qloc, ipos].astype(np.int8)
        f["geo_p"] = geo_w[qloc, ipos].astype(np.float32)
        f["query_loc_has_items"] = (qp["loc_code"][qglob] >= 0).astype(np.int8)
        del locm, in_geo, geo_w, filt_ok
        # расстояние до центра локации поиска (или самой вероятной локации гео-зоны)
        cl = np.where(qp["loc_code"][sl] >= 0, qp["loc_code"][sl], top_loc)[qloc]
        okc = has_w[qloc] | (qp["loc_code"][qglob] >= 0)
        dist = np.full(len(ipos), np.nan, np.float32)
        dist[okc] = haversine_km(corpus.loc_lat[cl[okc]], corpus.loc_lon[cl[okc]],
                                 corpus.item_lat[ipos[okc]], corpus.item_lon[ipos[okc]])
        f["dist_km_log"] = np.log1p(dist)
        qlc = qp["loc_code"][qglob]
        f["query_loc_size"] = np.where(qlc >= 0, np.log1p(corpus.loc_size[np.maximum(qlc, 0)]), 0).astype(np.float32)
        f["item_loc_size"] = np.log1p(corpus.loc_size[corpus.item_loc_code[ipos]]).astype(np.float32)
        f["search_category_is0"] = (qp["category"][qglob] == 0).astype(np.int8)
        f["has_filter"] = qp["has_filter"][qglob]
        f["q_ntok"] = qp["q_ntok"][qglob]
        f["item_rating"] = np.where(a["rating_is_missing"][ipos] == 1, np.nan, a["rating"][ipos]).astype(np.float32)
        f["item_reviews_count"] = a["reviews_count"][ipos]
        f["item_price"] = a["price"][ipos]
        f["item_phone_hidden"] = a["phone_hidden"][ipos]
        f["item_message_forbidden"] = a["message_forbidden"][ipos]
        f["title_len"] = a["title_len"][ipos]
        f["desc_len"] = a["desc_len"][ipos]
        thr = qp["rating_thr"][qglob]
        f["rating_ok"] = np.where(~np.isnan(thr),
                                  (np.nan_to_num(f["item_rating"], nan=-1) >= thr).astype(np.float32), np.nan)
        f.update(hist.pair_features(qp["qkey"][sl], qloc, ipos, mc_dist[sl]))

        if positives is not None:
            lab = np.zeros(len(ipos), np.int8)
            for r in np.unique(qglob):
                pos_set = positives.get(int(r))
                if pos_set:
                    m = qglob == r
                    lab[m] = np.isin(ipos[m], np.fromiter(pos_set, np.int64))
            f["label"] = lab

        df = pd.DataFrame({k: (v.astype(np.float32) if v.dtype == np.float64 else v) for k, v in f.items()})
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        n_rows += len(df)
        if (start // batch_size) % 10 == 0:
            print(f"  [{end}/{len(q)}] rows={n_rows}  пул {n_rows / end:.0f}/запрос  {time.time() - t0:.0f}s")
    if writer is not None:
        writer.close()
    print(f"[pool] {out_path}: {n_rows} строк ({n_rows / max(len(q), 1):.0f} на запрос), {time.time() - t0:.0f}s")
    return out_path


def feature_columns(df_or_cols):
    cols = df_or_cols.columns if hasattr(df_or_cols, "columns") else df_or_cols
    return [c for c in cols if c not in NON_FEATURE_COLS]
