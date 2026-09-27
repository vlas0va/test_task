"""Статистики по истории выборов (train_pairs), которые переносятся на новые запросы:

* переходы локаций P(локация объявления | локация поиска) - гео-зона запроса и признак geo_p.
  82% выборов из локации поиска, остальные уходят в соседние города; у 17% запросов
  бенчмарка в самой локации поиска объявлений нет вообще (это регионы).
* P(микрокатегория | слова запроса) - признаки tok_mc_*.
* доля выборов в микрокатегории объявления по такому же тексту запроса - hist_text_*.

Признаки уровня конкретного объявления (популярность, "это объявление выбирали по
такому же запросу") не используются: корпус бенчмарка пересекается с train_items
на 9.6%, а в схеме обучения на 42%, модель переоценила бы их.

При обучении история строится только по запросам H (их не видит ре-ранкер),
на бенчмарке - по всему train. Все признаки - доли, масштаб не зависит от объёма.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from text_prep import query_key_series


class HistoryStats:
    def fit(self, hist_queries, hist_pairs, items_meta, loc_smooth=5.0):
        """hist_queries: query_uid, search_query, search_location_id
        hist_pairs: query_uid, item_id
        items_meta: item_id, item_microcat_id, item_location_id (все известные объявления)"""
        q = hist_queries[["query_uid", "search_query", "search_location_id"]].copy()
        q["qkey"] = query_key_series(q).values
        p = hist_pairs.merge(q[["query_uid", "qkey", "search_location_id"]], on="query_uid", how="inner")
        meta = items_meta[["item_id", "item_microcat_id", "item_location_id"]].drop_duplicates("item_id").set_index("item_id")
        p["mc"] = meta["item_microcat_id"].reindex(p["item_id"]).values
        p["item_loc"] = meta["item_location_id"].reindex(p["item_id"]).values
        self.n_pairs = max(len(p), 1)

        # переходы локаций; сглаживание: +loc_smooth выборов в своей локации
        lt = p.dropna(subset=["item_loc"]).groupby(["search_location_id", "item_loc"]).size().rename("cnt").reset_index()
        self.loc_trans = {}
        for sl, g in lt.groupby("search_location_id", sort=False):
            locs = g["item_loc"].values.astype(np.int64)
            cnt = g["cnt"].values.astype(np.float64)
            if sl in set(locs):
                cnt = cnt + loc_smooth * (locs == sl)
            else:
                locs = np.r_[locs, sl]
                cnt = np.r_[cnt, loc_smooth]
            self.loc_trans[sl] = (locs, (cnt / cnt.sum()).astype(np.float32))

        self.text_cnt = p.groupby("qkey").size()
        self.text_mc_cnt = p.groupby(["qkey", "mc"]).size()

        # матрица слово -> распределение микрокатегорий
        self.tok_vec = CountVectorizer(binary=True, token_pattern=r"(?u)\b\w+\b", min_df=2)
        Xq = self.tok_vec.fit_transform(p["qkey"].values)
        mc_codes, mc_uniques = pd.factorize(p["mc"])
        self.mc_index = pd.Index(mc_uniques)
        ok = mc_codes >= 0
        M = sp.csr_matrix((np.ones(ok.sum(), np.float32), (np.nonzero(ok)[0], mc_codes[ok])),
                          shape=(len(p), len(mc_uniques)))
        TM = (Xq.T @ M).tocsr().astype(np.float32)
        row_sum = np.asarray(TM.sum(axis=1)).ravel()
        row_sum[row_sum == 0] = 1
        self.tok_mc = sp.diags(1.0 / row_sum) @ TM
        print(f"[history] pairs={len(p)} texts={len(self.text_cnt)} microcats={len(mc_uniques)} "
              f"search_locs={len(self.loc_trans)}")
        return self

    def set_corpus(self, items):
        """Привязка к корпусу (на обучении и на бенчмарке корпуса разные)."""
        self.corpus_mc = items["item_microcat_id"].values
        self.corpus_mc_code = self.mc_index.get_indexer(self.corpus_mc)   # -1: микрокатегории нет в истории
        return self

    def geo_weights(self, q_locs, loc_index: pd.Index) -> np.ndarray:
        """(запросы, локации корпуса): P(локация объявления | локация поиска).
        Для локации без истории - 1 для неё самой."""
        W = np.zeros((len(q_locs), len(loc_index)), np.float32)
        for r, sl in enumerate(q_locs):
            got = self.loc_trans.get(sl)
            if got is None:
                c = loc_index.get_indexer([sl])[0]
                if c >= 0:
                    W[r, c] = 1.0
                continue
            locs, pr = got
            c = loc_index.get_indexer(locs)
            ok = c >= 0
            W[r, c[ok]] = pr[ok]
        return W

    def query_keys(self, queries) -> np.ndarray:
        return query_key_series(queries).values

    def query_mc_dist(self, qkeys):
        """(запросы, микрокатегории): распределение по словам запроса."""
        D = (self.tok_vec.transform(qkeys).astype(np.float32) @ self.tok_mc).tocsr()
        s = np.asarray(D.sum(axis=1)).ravel()
        s[s == 0] = 1
        return (sp.diags(1.0 / s) @ D).tocsr()

    def pair_features(self, qkeys_b, qpos, ipos, mc_dist_b):
        """Признаки пар батча: qpos - номер запроса в батче, ipos - позиция в корпусе."""
        keys = qkeys_b[qpos]
        mcs = self.corpus_mc[ipos]
        f = {}
        tc = self.text_cnt.reindex(keys).to_numpy(np.float32)
        tm = self.text_mc_cnt.reindex(pd.MultiIndex.from_arrays([keys, mcs])).to_numpy(np.float32)
        f["hist_text_cnt"] = np.nan_to_num(np.log1p(tc / self.n_pairs * 1e5), nan=0.0)
        f["hist_text_mc_share"] = np.where(np.isnan(tc), np.nan, np.nan_to_num(tm / tc, nan=0.0)).astype(np.float32)
        code = self.corpus_mc_code[ipos]
        vals = np.zeros(len(ipos), np.float32)
        ok = code >= 0
        if ok.any():
            vals[ok] = np.asarray(mc_dist_b[qpos[ok], code[ok]]).ravel()
        f["tok_mc_share"] = vals
        top_mc = np.asarray(mc_dist_b.argmax(axis=1)).ravel()
        has_d = np.asarray(mc_dist_b.sum(axis=1)).ravel() > 0
        f["tok_mc_is_top"] = np.where(has_d[qpos], (code == top_mc[qpos]).astype(np.float32), np.nan)
        return f
