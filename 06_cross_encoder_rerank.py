# %% [markdown]
# # 06 (опционально) — дорогой ре-ранкинг верхушки пула cross-encoder'ом / LLM-ре-ранкером
#
# Идея "LLM в самом конце": берём top-M (по умолчанию 100) кандидатов после
# LightGBM и пересчитываем их моделью, которая видит запрос и объявление
# ВМЕСТЕ (cross-encoder). Генеративную LLM ("вот 100 объявлений, верни 50 id")
# сознательно не используем:
#   * 2452 запроса x ~100 кандидатов x ~40 токенов = ~10 млн входных токенов +
#     генерация 50 id по 16 hex-символов (~500 токенов на запрос) - на 7-8B
#     модели в 4 битах на RTX 3060 Laptop это 10+ часов;
#   * LLM путает/выдумывает 16-символьные id и порядок; позиционный bias на
#     длинных списках;
#   * pointwise-ре-ранкер (Qwen3-Reranker - это и есть LLM, дообученная отвечать
#     yes/no) даёт калиброванный скор на пару, батчится, 0.6B влезает в 6 ГБ с запасом.
#
# ЧЕСТНОЕ ОЖИДАНИЕ: "релевантность" здесь = "что пользователь выбрал". Среди
# 100 объявлений "автоподбор в Москве" cross-encoder не отличит выбранное от
# соседнего - это решают гео/рейтинг/популярность, т.е. LightGBM. CE помогает
# в основном выкинуть тематически неверные (запрос "баня на дровах" -> продажа
# дров). Поэтому это СМЕШИВАНИЕ с LightGBM с весом w, подобранным на valid;
# если w=0 оказывается лучшим - этап не даёт пользы, и это тоже ответ.
#
# Режимы: MODE="valid" - посчитать CE на valid/test из 04, подобрать w;
#         MODE="bench" - применить к бенчмарку (нужен запуск generate_final_answer_v2.py до этого).
#
# Модели (скачиваются один раз с HuggingFace, дальше работают офлайн):
#   * "tomaarsen/Qwen3-Reranker-0.6B-seq-cls" - Qwen3-Reranker-0.6B в формате CrossEncoder, мультиязычный
#   * "BAAI/bge-reranker-v2-m3"               - классический cross-encoder на XLM-R (568M), хороший русский
# Время: ~200-400 пар/с на RTX 3060 Laptop в fp16 при max_length=256 -> 245k пар бенчмарка ~ 15-25 мин.

# %%
import os, json, time
import numpy as np
import pandas as pd

import config_v2 as C
from data_v2 import load_train, load_benchmark
from text_prep import build_embedding_item_text_structured, clean_filter_text
from postprocess_v2 import recall_from_mask, select_topk, load_params

MODE = "valid"                 # "valid" -> подбор веса; "bench" -> answer_ce.csv
MODEL_NAME = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"   # или "BAAI/bge-reranker-v2-m3"
TOP_M = 100                    # сколько верхних кандидатов LightGBM пересчитывать
MAX_LEN = 256
BATCH = 32
RRF_K = 60
W_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
CE_PARAMS_PATH = f"{C.WORK_DIR}/ce_params.json"

QWEN_PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
               'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN_TASK = ("Given a user's search query on a Russian classifieds site, judge whether the service "
             "advertisement is what the user is looking for")


def load_scorer(model_name=MODEL_NAME):
    """Возвращает функцию score(list_of_(query, doc)) -> np.array."""
    import torch
    from sentence_transformers import CrossEncoder
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {"torch_dtype": torch.float16} if dev == "cuda" else {}
    ce = CrossEncoder(model_name, device=dev, max_length=MAX_LEN, model_kwargs=kw)
    is_qwen = "qwen3-reranker" in model_name.lower()

    def score(pairs):
        if is_qwen:
            pairs = [(f"{QWEN_PREFIX}<Instruct>: {QWEN_TASK}\n<Query>: {q}\n", f"<Document>: {d}{QWEN_SUFFIX}")
                     for q, d in pairs]
        return np.asarray(ce.predict(pairs, batch_size=BATCH, show_progress_bar=True), dtype=np.float32)
    return score


def query_texts(qdf):
    return [f"{q}" + (f" ({f})" if f else "") for q, f in
            zip(qdf["search_query"].fillna(""), qdf["search_infm_params_text"].map(clean_filter_text))]


def ce_scores_cached(top, q_text_by_row, item_text_by_pos, cache_path, scorer):
    """Скорит пары (qrow, item_pos) с кэшем на диск: при падении/перезапуске
    уже посчитанное не пересчитывается."""
    done = pd.read_parquet(cache_path) if os.path.exists(cache_path) else pd.DataFrame({"qrow": pd.Series(dtype=np.int32), "item_pos": pd.Series(dtype=np.int32),
                                                                                    "ce": pd.Series(dtype=np.float32)})
    key = top[["qrow", "item_pos"]].merge(done, on=["qrow", "item_pos"], how="left")
    todo = key[key["ce"].isna()][["qrow", "item_pos"]]
    print(f"CE: всего пар {len(key)}, из кэша {len(key)-len(todo)}, считать {len(todo)}")
    CHUNK = 20000
    for s in range(0, len(todo), CHUNK):
        part = todo.iloc[s:s + CHUNK].copy()
        t0 = time.time()
        part["ce"] = scorer([(q_text_by_row[r], item_text_by_pos[p]) for r, p in zip(part["qrow"], part["item_pos"])])
        done = part if len(done) == 0 else pd.concat([done, part], ignore_index=True)
        done.to_parquet(cache_path, index=False)
        print(f"  {s+len(part)}/{len(todo)}  ({len(part)/(time.time()-t0):.0f} пар/с)")
    return top[["qrow", "item_pos"]].merge(done, on=["qrow", "item_pos"], how="left")["ce"].to_numpy(dtype=np.float32)


def blend(df, w):
    """RRF-смешивание: 1/(k+rank_lgb) + w/(k+rank_ce) внутри top-M; остальным - только lgb."""
    r_l = df.groupby("qrow")["score"].rank(ascending=False, method="first")
    s = 1.0 / (RRF_K + r_l.values)
    ce = df["ce"].values
    has = ~np.isnan(ce)
    r_c = pd.Series(np.where(has, ce, -np.inf)).groupby(df["qrow"].values).rank(ascending=False, method="first").values
    return s + np.where(has, w / (RRF_K + r_c), 0.0)


def take_top_m(df, m):
    df = df.copy()
    df["_r"] = df.groupby("qrow")["score"].rank(ascending=False, method="first")
    return df[df["_r"] <= m].drop(columns="_r")


# %%
def main(scorer=None):
    scorer = scorer or load_scorer()

    if MODE == "valid":
        tq, ti, tp = load_train()
        _, bi = load_benchmark()
        items = pd.concat([ti, bi[~bi["item_id"].isin(set(ti["item_id"]))]], ignore_index=True)
        sc = pd.read_parquet(f"{C.WORK_DIR}/eval_scores.parquet")
        eq = pd.read_parquet(f"{C.WORK_DIR}/eval_queries.parquet")
        fq = tq.set_index("query_uid").loc[eq["query_uid"].values].reset_index()
        qtext = dict(zip(eq["qrow"].values, query_texts(fq)))
        top = take_top_m(sc, TOP_M)
        pos_needed = np.unique(top["item_pos"].values)
        # item_pos в eval_scores - позиции в корпусе обучения (train_items + доп. из benchmark)
        pos_to_id = dict(zip(sc["item_pos"].values, sc["item_id"].values))
        sub = items.set_index("item_id").loc[[pos_to_id[p] for p in pos_needed]].reset_index()
        itext = dict(zip(pos_needed, build_embedding_item_text_structured(sub).values))
        sc["ce"] = np.nan
        sc.loc[top.index, "ce"] = ce_scores_cached(top, qtext, itext, f"{C.WORK_DIR}/ce_cache_eval.parquet", scorer)

        n_rel = eq.set_index("qrow")["n_rel"]
        pp = load_params(C.POSTPROC_PARAMS_PATH)
        res = {}
        for split in ("valid", "test"):
            d = sc[sc["split"] == split].reset_index(drop=True)
            nr = n_rel.loc[eq.loc[eq["split"] == split, "qrow"].values]
            nr = nr[nr > 0]
            for w in W_GRID:
                m = select_topk(d, blend(d, w) * 1e3, pp)  # *1e3: масштаб RRF-скоров ~0.01-0.03
                res[(split, w)] = recall_from_mask(d["qrow"].values, m, d["label"].values, nr)
        tab = pd.Series(res).unstack(0)
        print(tab.round(4))
        best_w = float(tab["valid"].idxmax())
        print(f"лучший w на valid = {best_w}; test: {tab.loc[best_w, 'test']:.4f} vs без CE {tab.loc[0.0, 'test']:.4f}")
        json.dump({"w": best_w, "model": MODEL_NAME, "top_m": TOP_M}, open(CE_PARAMS_PATH, "w"))

    else:  # MODE == "bench"
        bq, bi = load_benchmark()
        prm = json.load(open(CE_PARAMS_PATH))
        assert prm["model"] == MODEL_NAME
        sc = pd.read_parquet(f"{C.WORK_DIR}/bench_scores.parquet")
        qtext = dict(enumerate(query_texts(bq.reset_index(drop=True))))
        top = take_top_m(sc, prm["top_m"])
        pos_needed = np.unique(top["item_pos"].values)
        itext = dict(zip(pos_needed, build_embedding_item_text_structured(bi.iloc[pos_needed]).values))
        sc["ce"] = np.nan
        sc.loc[top.index, "ce"] = ce_scores_cached(top, qtext, itext, f"{C.WORK_DIR}/ce_cache_bench.parquet", scorer)
        bl = blend(sc, prm["w"]) * 1e3
        m = select_topk(sc, bl, load_params(C.POSTPROC_PARAMS_PATH), C.K)
        ans = sc.loc[m].assign(p=bl[m]).sort_values(["qrow", "p"], ascending=[True, False]) \
                .groupby("qrow")["item_pos"].apply(list).reindex(range(len(bq)))
        ids = bi["item_id"].values
        base = pd.read_csv("./answer.csv", dtype=str).set_index("query_id")["answer"]
        out = []
        for r, qid in enumerate(bq["query_id"].astype(str)):
            lst = [ids[p] for p in (ans.iloc[r] if isinstance(ans.iloc[r], list) else [])]
            # если в пуле < 50 - добиваем из answer.csv (там уже есть добивка)
            for x in base.get(qid, "").split():
                if len(lst) >= C.K:
                    break
                if x not in lst:
                    lst.append(x)
            out.append(" ".join(lst[:C.K]))
        pd.DataFrame({"query_id": bq["query_id"].astype(str), "answer": out}).to_csv("./answer_ce.csv", index=False)
        print("сохранено ./answer_ce.csv")


if __name__ == "__main__":
    main()
