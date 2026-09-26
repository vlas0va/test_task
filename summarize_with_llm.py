"""
summarize_with_llm.py
=======================
ОПЦИОНАЛЬНЫЙ шаг препроцессинга: сжимает шумные текстовые поля объявления
(item_infm_params_text - часто прайс-лист/график работы на 1000+ символов,
item_description_raw) в короткую чистую суммаризацию через ЛОКАЛЬНУЮ LLM
(Ollama - у вас уже стоит судя по установщику в Downloads).

Зачем это нужно: BM25 просто обрезает длинные поля по символам (см.
text_prep.py) - грубо, но дёшево. Для эмбеддинг-модели (модель 2) вместо
грубой обрезки можно скормить куда более информативное, компактное summary -
"что за услуга, кто оказывает, какие условия" в 1-2 предложениях, без мусора
про конкретные цены на 20 позиций прайс-листа. Это НЕ обязательный шаг -
train_biencoder.py/embed_and_retrieve.py прекрасно работают и без него
(просто на сырых обрезанных текстах), но обычно даёт эмбеддингам чище вход.

Работает через Ollama HTTP API (http://localhost:11434) - т.е. ничего никуда
в интернет не уходит, всё локально на вашей машине. Перед запуском:
    ollama pull qwen2.5:7b-instruct     # или любая другая модель с хорошим русским
    ollama serve                         # если ещё не запущен как сервис

ВАЖНО про время выполнения: суммаризация ~189-345 тыс. объявлений через LLM -
это медленно (даже на GPU - часы). Практичные варианты:
  1) Ограничить LIMIT (например, суммаризировать только объявления, которые
     реально попадают в кандидаты BM25/эмбеддингов, а не весь корпус разом) -
     см. ITEM_IDS_FILTER ниже.
  2) Пропустить этот шаг вообще - train_biencoder.py и embed_and_retrieve.py
     одинаково хорошо работают без суммаризации, просто с сырым текстом.

Прогресс сохраняется инкрементально (append в parquet-совместимый jsonl) -
можно прерывать и перезапускать, уже обработанные item_id не пересчитываются.
"""
import json
import os
import time
import requests
import pandas as pd

# ---------------------------------------------------------------------------
BASE = "."  # папка с *.parquet - по умолчанию рядом со скриптом (запускать из папки с данными)
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"   # замените на свою модель (ollama list, чтобы посмотреть что есть)
OUT_JSONL = "./item_summaries.jsonl"   # промежуточный файл (инкрементальный, resumable)
OUT_PARQUET = "./item_summaries.parquet"  # финальный файл для text_prep.py / train_biencoder.py

ITEM_IDS_FILTER = None  # None = все item_id из ITEMS_FILE; либо list/set конкретных item_id
ITEMS_FILE = "benchmark_items.parquet"  # или "train_items.parquet"
LIMIT = None  # для быстрого теста поставьте, например, 200

PROMPT_TEMPLATE = """Ниже - заголовок, параметры и описание объявления об услуге с сервиса объявлений.
Сожми это в ОДНО-ДВА коротких предложения на русском: какая именно услуга оказывается,
кто оказывает (частник/компания), ключевые условия (если есть что-то важное:
выезд на дом, онлайн-запись и т.п.). Никаких цен, списков и форматирования - только
плотный содержательный текст, без вступлений вроде "Это объявление о...".

Заголовок: {title}
Параметры: {params}
Описание: {desc}

Суммаризация:"""


def call_ollama(prompt: str, retries=3, timeout=60) -> str:
    for attempt in range(retries):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": MODEL, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.1, "num_predict": 120}},
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()["response"].strip()
        except Exception as e:
            if attempt == retries - 1:
                print(f"[warn] ollama call failed after {retries} attempts: {e}")
                return ""
            time.sleep(2 * (attempt + 1))
    return ""


def load_done_ids(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["item_id"])
            except Exception:
                continue
    return done


def main():
    items = pd.read_parquet(f"{BASE}/{ITEMS_FILE}")
    if ITEM_IDS_FILTER is not None:
        items = items[items["item_id"].isin(ITEM_IDS_FILTER)]
    if LIMIT:
        items = items.head(LIMIT)

    done_ids = load_done_ids(OUT_JSONL)
    items_todo = items[~items["item_id"].isin(done_ids)]
    print(f"всего объявлений: {len(items)}, уже сделано ранее: {len(done_ids)}, осталось: {len(items_todo)}")

    with open(OUT_JSONL, "a", encoding="utf-8") as f:
        for i, (_, row) in enumerate(items_todo.iterrows()):
            prompt = PROMPT_TEMPLATE.format(
                title=(row["item_title_raw"] or "")[:150],
                params=(row["item_infm_params_text"] or "")[:600],
                desc=(row["item_description_raw"] or "")[:600],
            )
            summary = call_ollama(prompt)
            f.write(json.dumps({"item_id": row["item_id"], "summary": summary}, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(items_todo)} обработано")

    # финальная сборка в parquet
    all_rows = []
    with open(OUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            all_rows.append(json.loads(line))
    out = pd.DataFrame(all_rows).drop_duplicates("item_id", keep="last")
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"сохранено {OUT_PARQUET}: {out.shape}")


if __name__ == "__main__":
    main()
