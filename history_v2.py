"""
history_v2.py
=============
"История" = пары (запрос, выбранное объявление) из train.parquet, которые
используются как лог поведения пользователей (коллаборативный сигнал), а не
только как обучающие пары для моделей.

Что отсюда берётся:
  * item_pop        - сколько раз объявление выбирали (нормировано на размер истории)
  * loc_item_share  - доля выборов в этой локации, пришедшаяся на объявление
  * text_item_share - доля выборов по ТАКОМУ ЖЕ тексту запроса (стемы, без
                      учёта порядка), пришедшаяся на это объявление
  * text_mc_share   - доля выборов по такому же тексту в микрокатегории объявления
  * tok_mc_share    - то же, но через отдельные слова запроса (работает и для
                      запросов, которых в истории не было): P(microcat | слова)
  * канал кандидатов hist_text: объявления, которые выбирали по такому же тексту

Все признаки - ДОЛИ (share), а не сырые счётчики, чтобы они были сопоставимы
при обучении (история = 97% train) и на бенчмарке (история = весь train).

ПРО УТЕЧКИ: при обучении ре-ранкера история строится ТОЛЬКО по запросам, не
входящим в обучающие/валидационные запросы ре-ранкера (см. 04_train_reranker_v2.py),
иначе позитивы "видят сами себя" в счётчиках.

Это не хардкод ответов: статистика считается только по train, для любого
нового запроса одинаково, без знания разметки бенчмарка.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from text_prep import query_key_series


class HistoryStats:
    def fit(self, hist_queries: pd.DataFrame, hist_pairs: pd.DataFrame, items_meta: pd.DataFrame,
            item_level: bool = False, loc_smooth: float = 5.0):
        """
        item_level: считать ли признаки уровня КОНКРЕТНОГО объявления (популярность,
        "по такому же тексту выбирали это объявление"). На реальных данных корпус
        бенчмарка пересекается с train всего на ~10% (00_diagnostics), а в схеме
        обучения - на ~42%, поэтому по умолчанию ВЫКЛЮЧЕНО: модель переоценила бы
        эти признаки. Уровень микрокатегорий и переходы локаций переносятся честно.
        hist_queries: query_uid, search_query, search_location_id (только запросы истории)
        hist_pairs:   query_uid, item_id (только для запросов истории)
        items_meta:   item_id, item_microcat_id, item_category_id (все известные объявления)
        """
        q = hist_queries[["query_uid", "search_query", "search_location_id"]].copy()
        q["qkey"] = query_key_series(q).values
        p = hist_pairs.merge(q[["query_uid", "qkey", "search_location_id"]], on="query_uid", how="inner")
        meta = items_meta[["item_id", "item_microcat_id", "item_location_id"]].drop_duplicates("item_id").set_index("item_id")
        p["mc"] = meta["item_microcat_id"].reindex(p["item_id"]).values
        self.n_pairs = max(len(p), 1)
        self.item_level = item_level
        p["item_loc"] = meta["item_location_id"].reindex(p["item_id"]).values

        # ---- переходы локаций: P(локация объявления | локация поиска).
        # На данных: 18% выборов - НЕ из локации поиска, а для части search_location_id
        # (107620, 107621, 621540, ... - похоже на регионы/агломерации) объявлений
        # в самой локации нет ВООБЩЕ (17% запросов бенчмарка!). Для них "гео" = те
        # города, куда реально уходили выборы из этой локации в train.
        # Сглаживание: + loc_smooth псевдо-выборов "в своей же локации".
        lt = p.dropna(subset=["item_loc"]).groupby(["search_location_id", "item_loc"]).size().rename("cnt").reset_index()
        self.loc_smooth = loc_smooth
        self.loc_trans = {}
        for sl, g in lt.groupby("search_location_id", sort=False):
            locs = g["item_loc"].values.astype(np.int64)
            cnt = g["cnt"].values.astype(np.float64)
            if sl in set(locs):
                cnt = cnt + loc_smooth * (locs == sl)
            else:
                locs = np.r_[locs, sl]; cnt = np.r_[cnt, loc_smooth]
            self.loc_trans[sl] = (locs, (cnt / cnt.sum()).astype(np.float32))

        self.item_cnt = p.groupby("item_id").size()
        self.loc_item_cnt = p.groupby(["search_location_id", "item_id"]).size()
        self.loc_cnt = p.groupby("search_location_id").size()
        self.text_item_cnt = p.groupby(["qkey", "item_id"]).size()
        self.text_cnt = p.groupby("qkey").size()
        self.text_mc_cnt = p.groupby(["qkey", "mc"]).size()

        # P(microcat | слово запроса): матрица слово x микрокатегория
        self.tok_vec = CountVectorizer(binary=True, token_pattern=r"(?u)\b\w+\b", min_df=2)
        Xq = self.tok_vec.fit_transform(p["qkey"].values)             # (n_pairs, V)
        mc_codes, mc_uniques = pd.factorize(p["mc"])
        self.mc_index = pd.Index(mc_uniques)
        valid = mc_codes >= 0
        M = sp.csr_matrix((np.ones(valid.sum(), np.float32), (np.nonzero(valid)[0], mc_codes[valid])),
                          shape=(len(p), len(mc_uniques)))
        TM = (Xq.T @ M).tocsr().astype(np.float32)                      # (V, C)
        row_sum = np.asarray(TM.sum(axis=1)).ravel()
        row_sum[row_sum == 0] = 1
        self.tok_mc = sp.diags(1.0 / row_sum) @ TM                      # строки = распределения
        print(f"[history] pairs={len(p)} items={len(self.item_cnt)} texts={len(self.text_cnt)} "
              f"microcats={len(mc_uniques)} search_locs={len(self.loc_trans)} item_level={item_level}")
        return self

    # ------------------------------------------------------------------ привязка к корпусу
    def set_corpus(self, items: pd.DataFrame):
        """Готовит массивы по позициям корпуса (корпус на обучении и на бенчмарке разный)."""
        ids = items["item_id"].values
        self.corpus_item_ids = ids
        self.corpus_loc = items["item_location_id"].values
        self.corpus_mc = items["item_microcat_id"].values
        self.item_pop = (self.item_cnt.reindex(ids).fillna(0).values / self.n_pairs * 1e5).astype(np.float32)
        pos_of = pd.Series(np.arange(len(ids)), index=ids)
        pos_of = pos_of[~pos_of.index.duplicated()]
        # для канала hist_text: по каждому тексту - позиции объявлений корпуса и счётчики
        t = self.text_item_cnt.reset_index(name="cnt")
        t["pos"] = pos_of.reindex(t["item_id"]).values
        t = t.dropna(subset=["pos"])
        t["pos"] = t["pos"].astype(np.int64)
        self.text_to_items = {k: (g["pos"].values, g["cnt"].values.astype(np.float32))
                              for k, g in t.groupby("qkey", sort=False)}
        mc_code = self.mc_index.get_indexer(self.corpus_mc)   # -1 если микрокатегории нет в истории
        self.corpus_mc_code = mc_code
        return self

    def geo_weights(self, q_locs, loc_index: pd.Index) -> np.ndarray:
        """(n_q, n_corpus_locs): P(локация объявления | локация поиска) по истории.
        loc_index - pd.Index локаций корпуса (порядок кодов CorpusIndex).
        Для локации поиска без истории: 1.0 для своей локации."""
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

    def query_keys(self, queries: pd.DataFrame) -> np.ndarray:
        return query_key_series(queries).values

    # ------------------------------------------------------------------ канал кандидатов
    def hist_text_topk(self, qkeys, q_locs, n):
        """Для каждого запроса - до n объявлений, выбиравшихся по такому же тексту;
        сначала из той же локации, внутри - по числу выборов."""
        out_idx, out_score = [], []
        for key, loc in zip(qkeys, q_locs):
            got = self.text_to_items.get(key)
            if got is None:
                out_idx.append(np.empty(0, np.int64)); out_score.append(np.empty(0, np.float32))
                continue
            pos, cnt = got
            score = np.log1p(cnt) + 3.0 * (self.corpus_loc[pos] == loc)
            order = np.argsort(-score)[:n]
            out_idx.append(pos[order]); out_score.append(score[order].astype(np.float32))
        return out_idx, out_score

    # ------------------------------------------------------------------ признаки
    def query_mc_dist(self, qkeys):
        """(n_q, C) распределение микрокатегорий по словам запроса."""
        Xq = self.tok_vec.transform(qkeys).astype(np.float32)
        D = (Xq @ self.tok_mc).tocsr()
        s = np.asarray(D.sum(axis=1)).ravel()
        s[s == 0] = 1
        return (sp.diags(1.0 / s) @ D).tocsr()

    def pair_features(self, qkeys_b, q_locs_b, qpos, ipos, mc_dist_b):
        """Признаки для плоских пар батча (qpos - локальные номера запросов батча,
        ipos - позиции в корпусе)."""
        item_ids = self.corpus_item_ids[ipos]
        keys = qkeys_b[qpos]
        locs = q_locs_b[qpos]
        f = {}
        mcs = self.corpus_mc[ipos]
        tc = self.text_cnt.reindex(keys).to_numpy(np.float32)
        tm = self.text_mc_cnt.reindex(pd.MultiIndex.from_arrays([keys, mcs])).to_numpy(np.float32)
        f["hist_text_cnt"] = np.nan_to_num(np.log1p(tc / self.n_pairs * 1e5), nan=0.0)  # насколько частый запрос
        f["hist_text_mc_share"] = np.where(np.isnan(tc), np.nan, np.nan_to_num(tm / tc, nan=0.0)).astype(np.float32)
        code = self.corpus_mc_code[ipos]
        vals = np.zeros(len(ipos), np.float32)
        ok = code >= 0
        if ok.any():
            vals[ok] = np.asarray(mc_dist_b[qpos[ok], code[ok]]).ravel()
        f["tok_mc_share"] = vals
        # микрокатегория объявления = самая вероятная для слов запроса?
        top_mc = np.asarray(mc_dist_b.argmax(axis=1)).ravel()
        has_d = np.asarray(mc_dist_b.sum(axis=1)).ravel() > 0
        f["tok_mc_is_top"] = np.where(has_d[qpos], (code == top_mc[qpos]).astype(np.float32), np.nan)
        if not self.item_level:
            return f

        f["hist_item_pop"] = self.item_pop[ipos]
        f["hist_item_pop_is0"] = (self.item_pop[ipos] == 0).astype(np.int8)

        li = self.loc_item_cnt.reindex(pd.MultiIndex.from_arrays([locs, item_ids])).to_numpy(np.float32)
        lc = self.loc_cnt.reindex(locs).to_numpy(np.float32)
        f["hist_loc_item_share"] = np.nan_to_num(li / lc, nan=0.0)

        ti = self.text_item_cnt.reindex(pd.MultiIndex.from_arrays([keys, item_ids])).to_numpy(np.float32)
        f["hist_text_item_share"] = np.nan_to_num(ti / tc, nan=0.0)
        return f
