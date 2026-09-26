"""
evaluate_offline.py
====================
Офлайн-валидация методов кандидатогенерации на train.parquet (у нас нет
доступа к реальной разметке benchmark_queries, а попыток отправки всего 7,
поэтому нужно уметь мерить Recall@50 самостоятельно).

Методология:
  - train_queries.parquet / train_items.parquet / train_pairs.parquet - это
    train.parquet, разложенный на три таблицы без дублирования текста
    (уникальные запросы, уникальные объявления, и связи запрос<->объявление).
    "Запрос" в train определён как уникальная комбинация всех search_* полей -
    ровно так же, как в benchmark_queries один query_id соответствует одной
    комбинации признаков.
  - Берём случайную выборку query_uid как "отложенный бенчмарк", корпусом
    служат ВСЕ объявления train_items (344 825 шт - даже больше, чем в
    реальном бенчмарке (189 212), то есть это скорее консервативная,
    чем оптимистичная оценка recall).
  - Считаем Recall@50 = mean(|top50 ∩ релевантные| / |релевантные|) - ровно
    формула из условия задачи.

Здесь сравниваются несколько конфигураций (word-BM25, char-BM25, гибрид),
чтобы выбрать лучшую перед прогоном на реальном benchmark_queries/benchmark_items.
"""
import gc
import time
import numpy as np
import pandas as pd

from text_prep import build_query_text, iter_item_texts, iter_item_texts_char, stemming_generator
from bm25 import BM25Index

BASE = "."  # папка с *.parquet - по умолчанию рядом со скриптом (запускать из папки с данными)
RNG = np.random.default_rng(42)


def load_train():
    tq = pd.read_parquet(f"{BASE}/train_queries.parquet")
    ti = pd.read_parquet(f"{BASE}/train_items.parquet")
    tp = pd.read_parquet(f"{BASE}/train_pairs.parquet")
    return tq, ti, tp


def recall_at_k(topk_item_ids, relevant_sets):
    """topk_item_ids: список списков item_id (по одному на запрос, той же длины,
    что relevant_sets). relevant_sets: список set(item_id)."""
    vals = []
    for pred, rel in zip(topk_item_ids, relevant_sets):
        if not rel:
            continue
        hit = len(set(pred) & rel)
        vals.append(hit / len(rel))
    return float(np.mean(vals)), len(vals)


def rrf_combine(rank_lists, k=60):
    """
    Reciprocal Rank Fusion: score(doc) = sum_i 1/(k + rank_i(doc)).
    rank_lists: список списков индексов документов (в порядке убывания релевантности)
    для ОДНОГО запроса, по одному списку на канал (word-BM25, char-BM25, ...).
    Документы, отсутствующие в списке какого-то канала, просто не получают от
    него вклад (это и есть смысл "объединения кандидатов из разных каналов").
    Возвращает индексы документов, отсортированные по убыванию комбинированного score.
    """
    scores = {}
    for lst in rank_lists:
        for rank, doc_idx in enumerate(lst):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    return [d for d, _ in ordered]


def run_eval(n_eval_queries=1500, k=50, seed=42):
    tq, ti, tp = load_train()
    print(f"train_queries={len(tq)} train_items={len(ti)} train_pairs={len(tp)}")

    # ground truth: query_uid -> set(item_id)
    gt = tp.groupby("query_uid")["item_id"].apply(set)

    rng = np.random.default_rng(seed)
    eval_uids = rng.choice(tq["query_uid"].values, size=min(n_eval_queries, len(tq)), replace=False)
    eval_q = tq[tq["query_uid"].isin(eval_uids)].reset_index(drop=True)
    relevant_sets = [gt.get(u, set()) for u in eval_q["query_uid"]]
    print(f"eval queries: {len(eval_q)}, avg relevant per query: {np.mean([len(s) for s in relevant_sets]):.3f}")

    item_ids = ti["item_id"].values
    query_text = build_query_text(eval_q)

    results = {}

    # --- канал 1: word-level BM25 ---
    # iter_item_texts(ti) - генератор; CountVectorizer читает его по одному
    # документу за раз, не материализуя весь корпус текста в памяти разом.
    t0 = time.time()
    word_idx = BM25Index(analyzer="word", ngram_range=(1, 1), min_df=2)
    word_idx.fit(iter_item_texts(ti))
    print(f"word BM25 fit in {time.time()-t0:.1f}s, vocab={len(word_idx.vectorizer.vocabulary_)}")
    t0 = time.time()
    word_top = word_idx.topk(query_text, k=k)
    print(f"word BM25 topk in {time.time()-t0:.1f}s")
    word_pred_ids = [item_ids[idx] for idx in word_top]
    r, n = recall_at_k(word_pred_ids, relevant_sets)
    results["word_bm25"] = r
    print(f"[word_bm25]        recall@{k} = {r:.4f}  (n={n})")

    # --- канал 1b: то же самое, но со стеммингом (word_bm25 без стемминга
    # показывает низкий recall, несмотря на то что лексическое пересечение
    # почти всегда есть - подозрение на несовпадение словоформ) ---
    t0 = time.time()
    word_stem_idx = BM25Index(analyzer="word", ngram_range=(1, 1), min_df=2)
    word_stem_idx.fit(stemming_generator(iter_item_texts(ti)))
    print(f"word BM25 (stemmed) fit in {time.time()-t0:.1f}s, vocab={len(word_stem_idx.vectorizer.vocabulary_)}")
    query_text_stem = list(stemming_generator(query_text))
    t0 = time.time()
    word_stem_top = word_stem_idx.topk(query_text_stem, k=k)
    print(f"word BM25 (stemmed) topk in {time.time()-t0:.1f}s")
    word_stem_pred_ids = [item_ids[idx] for idx in word_stem_top]
    r, n = recall_at_k(word_stem_pred_ids, relevant_sets)
    results["word_bm25_stemmed"] = r
    print(f"[word_bm25_stemmed] recall@{k} = {r:.4f}  (n={n})")
    word_stem_top_big = word_stem_idx.topk(query_text_stem, k=k * 3)
    del word_stem_idx
    gc.collect()

    # запасаем "широкие" top-k списки word-канала перед тем, как освободить его
    # память под char-канал (иначе на корпусе в 344к объявлений с двумя
    # sparse-матрицами разом легко упереться в память контейнера)
    word_top_big = word_idx.topk(query_text, k=k * 3)
    del word_idx
    gc.collect()

    # --- канал 2: char n-gram BM25 (устойчив к словоформам и "слипшимся"
    # словам вроде "автопроверка" vs "проверка авто" - как показал разбор
    # примеров, это реальная и частая проблема в этих данных) ---
    t0 = time.time()
    char_idx = BM25Index(analyzer="char_wb", ngram_range=(3, 4), min_df=3, max_features=200_000)
    char_idx.fit(iter_item_texts_char(ti))
    print(f"char BM25 fit in {time.time()-t0:.1f}s, vocab={len(char_idx.vectorizer.vocabulary_)}")
    t0 = time.time()
    char_top = char_idx.topk(query_text, k=k)
    print(f"char BM25 topk in {time.time()-t0:.1f}s")
    char_pred_ids = [item_ids[idx] for idx in char_top]
    r, n = recall_at_k(char_pred_ids, relevant_sets)
    results["char_bm25"] = r
    print(f"[char_bm25]        recall@{k} = {r:.4f}  (n={n})")

    # --- гибриды: RRF по разным сочетаниям каналов, берём топ-k достаточно
    # большие списки с запасом (k*3), чтобы после объединения точно набралось
    # k хороших кандидатов ---
    t0 = time.time()
    char_top_big = char_idx.topk(query_text, k=k * 3)
    print(f"char wide topk in {time.time()-t0:.1f}s")

    def hybrid_recall(name, lists_per_query):
        pred_ids = []
        for lists in lists_per_query:
            combined = rrf_combine(lists)[:k]
            pred_ids.append(item_ids[combined])
        r, n = recall_at_k(pred_ids, relevant_sets)
        results[name] = r
        print(f"[{name}] recall@{k} = {r:.4f}  (n={n})")

    hybrid_recall("hybrid_word_char", zip(word_top_big, char_top_big))
    hybrid_recall("hybrid_wordstem_char", zip(word_stem_top_big, char_top_big))
    hybrid_recall("hybrid_all3", zip(word_top_big, word_stem_top_big, char_top_big))

    print("\n=== summary ===")
    for name, val in results.items():
        print(f"  {name:25s} {val:.4f}")

    return results


if __name__ == "__main__":
    run_eval()
