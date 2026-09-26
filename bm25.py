"""
bm25.py
=======
Векторизованная реализация BM25 (Okapi BM25) поверх sklearn.CountVectorizer и
разреженных матриц scipy. Это классический метод информационного поиска
(лексический sparse-retrieval), без каких-либо внешних моделей и без обращений
в интернет - воспроизводится где угодно с scikit-learn/scipy/numpy.

Почему не готовая библиотека rank_bm25: она скорит запрос против корпуса
поэлементно в чистом Python и на корпусе в ~190к документов будет
неприемлемо медленной. Здесь скоринг всех 2452 запросов против всех 189212
документов сведён к одному разреженному матричному умножению
(query_term_matrix @ bm25_weighted_doc_term_matrix.T), что выполняется за
секунды.

Формула BM25 для пары (запрос Q, документ D):
    score(Q, D) = sum_{t in Q} IDF(t) * tf(t,D) * (k1+1) /
                  (tf(t,D) + k1 * (1 - b + b * |D| / avgdl))

IDF берём в стандартном для BM25 виде (Robertson-Sparck Jones с сглаживанием):
    IDF(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
Такая формула всегда >= 0, в отличие от "классического" idf, который для очень
частых терминов может уйти в минус.
"""
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    analyzer: str = "word"          # "word" или "char_wb"
    ngram_range: tuple = (1, 1)
    min_df: int = 2
    max_features: int = None
    token_pattern: str = r"(?u)\b\w+\b"
    vectorizer_type: str = "count"  # "count" (точный словарь) или "hashing" (без словаря, для экономии памяти)
    n_hash_features: int = 2 ** 20

    def _make_vectorizer(self):
        if self.vectorizer_type == "hashing":
            # HashingVectorizer не строит словарь в памяти (термин -> индекс
            # получается хэш-функцией "на лету"), поэтому fit_transform не
            # требует держать в памяти промежуточный словарь из десятков
            # миллионов уникальных символьных n-грамм - только это и позволяет
            # посчитать char-n-граммы на корпусе в ~345к объявлений в
            # контейнере с ограниченной памятью. Плата за это - небольшая
            # (в данном масштабе пренебрежимая) вероятность коллизий хэшей.
            kwargs = dict(
                analyzer=self.analyzer,
                ngram_range=self.ngram_range,
                n_features=self.n_hash_features,
                alternate_sign=False,  # нужны неотрицательные "псевдо-счётчики" для BM25
                norm=None,
                lowercase=False,
            )
            if self.analyzer == "word":
                kwargs["token_pattern"] = self.token_pattern
            return HashingVectorizer(**kwargs)
        else:
            kwargs = dict(
                analyzer=self.analyzer,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_features=self.max_features,
                lowercase=False,  # текст уже нормализован заранее
            )
            if self.analyzer == "word":
                kwargs["token_pattern"] = self.token_pattern
            return CountVectorizer(**kwargs)

    def fit(self, corpus_texts):
        """Строит словарь (или хэш-схему) и BM25-взвешенную матрицу термин-документ по корпусу объявлений."""
        self.vectorizer = self._make_vectorizer()

        if self.vectorizer_type == "hashing":
            X = self.vectorizer.transform(corpus_texts)
        else:
            X = self.vectorizer.fit_transform(corpus_texts)  # (n_docs, n_terms), сырые tf
        X = X.tocsr()
        n_docs, n_terms = X.shape

        doc_len = np.asarray(X.sum(axis=1)).ravel()
        avgdl = doc_len.mean() if n_docs > 0 else 1.0
        self.avgdl_ = avgdl

        # document frequency термина = число документов, где термин встретился
        df = np.diff(X.tocsc().indptr)  # длина = n_terms, число ненулевых в каждом столбце
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        self.idf_ = idf

        # BM25-насыщение TF: tf' = tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl))
        denom_per_doc = self.k1 * (1.0 - self.b + self.b * doc_len / avgdl)  # длина n_docs
        data = X.data.astype(np.float32)
        # denom для каждого ненулевого элемента = denom_per_doc его строки (документа)
        row_nnz = np.diff(X.indptr)
        denom_expanded = np.repeat(denom_per_doc, row_nnz).astype(np.float32)
        new_data = data * np.float32(self.k1 + 1.0) / (data + denom_expanded)

        X_bm25 = sp.csr_matrix((new_data, X.indices, X.indptr), shape=X.shape)
        # умножаем каждый столбец j на idf[j]: делаем это через умножение на
        # разреженную диагональную матрицу (idf на диагонали) - корректно и
        # быстро (сложность O(nnz)), в отличие от прямой манипуляции индексами
        # CSC-формата (там indices - это индексы СТРОК внутри столбца, а не
        # номера столбцов, поэтому naive "data *= idf[indices]" даёт неверный
        # результат / IndexError).
        X_bm25 = X_bm25 @ sp.diags(idf.astype(np.float32))
        X_bm25 = X_bm25.tocsr()
        X_bm25.data = X_bm25.data.astype(np.float32)
        self.doc_matrix_ = X_bm25  # (n_docs, n_terms), уже полностью взвешенная BM25-матрица (float32 для экономии памяти)
        self.n_docs_ = n_docs
        return self

    def transform_query(self, query_texts):
        """Query term matrix (n_queries, n_terms) - сырые счётчики термов запроса."""
        return self.vectorizer.transform(query_texts).tocsr()

    def score_all(self, query_texts):
        """
        Возвращает разреженную матрицу (n_queries, n_docs) BM25-скоров.
        score = Q_counts @ doc_matrix.T
        (термин запроса, которого нет в словаре корпуса, просто не участвует в сумме -
        это ожидаемое поведение: он не мог совпасть ни с одним документом).
        """
        Q = self.transform_query(query_texts)
        scores = Q @ self.doc_matrix_.T
        return scores.tocsr()

    def topk(self, query_texts, k=50, batch_size=200, return_scores=False):
        """
        Возвращает список списков индексов документов (в порядке убывания score)
        для каждого запроса. Работает батчами, чтобы не материализовать сразу
        огромную плотную матрицу для всех запросов.
        Если return_scores=True, дополнительно возвращает список массивов
        BM25-скоров той же длины (нужно для скорингового, а не рангового,
        объединения нескольких каналов - см. combine.py).
        """
        results = []
        scores_out = []
        query_list = list(query_texts)
        for start in range(0, len(query_list), batch_size):
            batch = query_list[start:start + batch_size]
            S = self.score_all(batch)  # sparse (b, n_docs)
            S = S.toarray()
            # argpartition для top-k (быстрее полной сортировки), потом сортируем сами k
            for row in S:
                if k < len(row):
                    idx = np.argpartition(-row, k)[:k]
                else:
                    idx = np.arange(len(row))
                idx = idx[np.argsort(-row[idx])]
                results.append(idx)
                if return_scores:
                    scores_out.append(row[idx])
        if return_scores:
            return results, scores_out
        return results
