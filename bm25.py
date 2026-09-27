"""BM25 на разреженных матрицах: скоринг батча запросов против всего корпуса
одним умножением матриц (rank_bm25 считает поэлементно и на 500к документов медленный).

score(Q, D) = sum_t IDF(t) * tf(t,D) * (k1+1) / (tf(t,D) + k1 * (1 - b + b*|D|/avgdl))
IDF(t) = log(1 + (N - df + 0.5) / (df + 0.5))
"""
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    analyzer: str = "word"
    ngram_range: tuple = (1, 1)
    min_df: int = 2
    token_pattern: str = r"(?u)\b\w+\b"

    def fit(self, corpus_texts):
        # текст уже нормализован и застеммлен, lowercase не нужен
        self.vectorizer = CountVectorizer(analyzer=self.analyzer, ngram_range=self.ngram_range,
                                          min_df=self.min_df, token_pattern=self.token_pattern,
                                          lowercase=False)
        X = self.vectorizer.fit_transform(corpus_texts).tocsr()     # (docs, terms), сырые tf
        n_docs = X.shape[0]
        doc_len = np.asarray(X.sum(axis=1)).ravel()
        avgdl = doc_len.mean() if n_docs > 0 else 1.0

        df = np.diff(X.tocsc().indptr)
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        denom = self.k1 * (1.0 - self.b + self.b * doc_len / avgdl)
        data = X.data.astype(np.float32)
        denom = np.repeat(denom, np.diff(X.indptr)).astype(np.float32)
        tf = data * np.float32(self.k1 + 1.0) / (data + denom)

        W = sp.csr_matrix((tf, X.indices, X.indptr), shape=X.shape)
        W = (W @ sp.diags(idf.astype(np.float32))).tocsr()
        W.data = W.data.astype(np.float32)
        self.doc_matrix_ = W
        return self

    def transform_query(self, query_texts):
        """Счётчики слов запросов (queries, terms). Скор = Q @ doc_matrix_.T"""
        return self.vectorizer.transform(query_texts).tocsr()
