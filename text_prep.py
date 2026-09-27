"""Тексты запросов и объявлений: для BM25 (нормализация, стемминг) и для эмбеддеров."""
import re
import pandas as pd

_CLEAN_RE = re.compile(r"[^0-9a-zа-яё\s]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def _safe_str(x):
    """None и NaN -> пустая строка."""
    if x is None:
        return ""
    if isinstance(x, float) and x != x:
        return ""
    return x


def normalize_text(s: str) -> str:
    """Нижний регистр, ё -> е, без пунктуации, одинарные пробелы."""
    if s is None:
        return ""
    s = str(s).lower().replace("ё", "е")
    s = _CLEAN_RE.sub(" ", s)
    return _SPACE_RE.sub(" ", s).strip()


# ---------------------------------------------------------------- BM25

def _row_item_text(title, params, desc, title_repeat, params_repeat, desc_repeat,
                   params_maxlen, desc_maxlen):
    title = normalize_text(_safe_str(title))
    params = normalize_text(_safe_str(params)[:params_maxlen])
    desc = normalize_text(_safe_str(desc)[:desc_maxlen])
    parts = [title] * title_repeat + [params] * params_repeat + [desc] * desc_repeat
    return _SPACE_RE.sub(" ", " ".join(parts)).strip()


def iter_item_texts(df: pd.DataFrame, title_repeat=3, params_repeat=1, desc_repeat=1,
                    params_maxlen=300, desc_maxlen=250):
    """Текст объявления для BM25: заголовок (повтор = больший вес) + начало параметров
    + начало описания. Дальше в параметрах и описании обычно прайс и график работы.
    Генератор, чтобы не держать 500к текстов в памяти."""
    cols = df[["item_title_raw", "item_infm_params_text", "item_description_raw"]]
    for title, params, desc in cols.itertuples(index=False, name=None):
        yield _row_item_text(title, params, desc, title_repeat, params_repeat, desc_repeat,
                             params_maxlen, desc_maxlen)


def iter_title_texts(df: pd.DataFrame):
    for t in df["item_title_raw"]:
        yield normalize_text(_safe_str(t))


def stemming_generator(text_iter):
    """Snowball-стеммер для русского, с кэшем по словам."""
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("russian")
    cache = {}

    def stem_word(w):
        s = cache.get(w)
        if s is None:
            s = cache[w] = stemmer.stem(w)
        return s

    for text in text_iter:
        yield " ".join(stem_word(w) for w in text.split())


_RATING_RE = re.compile(r"рейтинг[^0-9]{0,40}?(\d+(?:[.,]\d+)?)", re.IGNORECASE)
_RATING_PHRASE_RE = re.compile(
    r"рейтинг\s+пользователя\s*\d+(?:[.,]\d+)?\s*звезд\w*(?:\s+и\s+выше)?", re.IGNORECASE)
# служебные слова фильтров: из "Вид услуги X" в BM25 идёт только X
_FILTER_STOP = {"вид", "услуги", "услуга", "тип", "и", "выше", "звезды", "звезд", "рейтинг",
                "пользователя", "не", "важно", "любой", "любая", "любое"}


def parse_rating_threshold(s) -> float:
    """'Рейтинг пользователя 4 звезды и выше' -> 4.0, иначе nan."""
    s = _safe_str(s)
    m = _RATING_RE.search(s) if s else None
    if not m:
        return float("nan")
    try:
        return float(m.group(1).replace(",", "."))
    except ValueError:
        return float("nan")


def clean_filter_text(s) -> str:
    """Текст фильтров без фразы про рейтинг и служебных слов."""
    s = _safe_str(s)
    if not s:
        return ""
    s = _RATING_PHRASE_RE.sub(" ", s)
    toks = [t for t in normalize_text(s).split() if t not in _FILTER_STOP and not t.isdigit()]
    return " ".join(toks)


def build_query_text(df: pd.DataFrame, query_repeat: int = 2) -> pd.Series:
    """Запрос для BM25: сам запрос дважды + значения фильтров."""
    q = df["search_query"].fillna("").map(normalize_text)
    filt = df["search_infm_params_text"].map(clean_filter_text)
    text = (q + " ") * query_repeat + filt
    return text.map(lambda s: _SPACE_RE.sub(" ", s).strip())


def query_only_text(df: pd.DataFrame) -> pd.Series:
    return df["search_query"].fillna("").map(normalize_text)


def filter_only_text(df: pd.DataFrame) -> pd.Series:
    return df["search_infm_params_text"].map(clean_filter_text)


def query_key_series(df: pd.DataFrame) -> pd.Series:
    """Ключ "тот же запрос": отсортированное множество стемов.
    'скупка телевизоров' == 'телевизор скупка'."""
    stemmed = list(stemming_generator(df["search_query"].fillna("").map(normalize_text)))
    return pd.Series([" ".join(sorted(set(s.split()))) for s in stemmed], index=df.index)


# ---------------------------------------------------------------- фильтры поиска
# search_infm_params_text - пары "ключ значение" подряд без разделителей.
# По train у выбранных объявлений пара "Вид услуги X" есть в параметрах в 98% случаев,
# "Тип услуги Y" - в 95.5%. Остальные ключи в параметрах объявления не отражаются.
FILTER_KEYS = ["Тип услуги автосервиса", "Вид услуги", "Тип услуги", "Кто оказывает услуги",
               "Онлайн-запись", "Срочная услуга", "Предмет или направление", "Предмет или специальность",
               "Аренда авто", "Поиск по слотам", "Рейтинг пользователя"]
CHECKABLE_FILTER_KEYS = ("Вид услуги", "Тип услуги", "Тип услуги автосервиса", "Предмет или специальность")
_FKEY_RE = re.compile("(" + "|".join(sorted(map(re.escape, FILTER_KEYS), key=len, reverse=True)) + ")")


def parse_filter_pairs(s) -> list:
    """'Тип услуги Услуги парикмахера Вид услуги Красота, здоровье' ->
    [('Тип услуги', 'Услуги парикмахера'), ('Вид услуги', 'Красота, здоровье')]"""
    s = _safe_str(s)
    if not s:
        return []
    parts = _FKEY_RE.split(s)
    out = []
    for i in range(1, len(parts), 2):
        v = parts[i + 1].strip() if i + 1 < len(parts) else ""
        out.append((parts[i], v))
    return out


def checkable_filters(s) -> list:
    """Пары, которые можно проверить по параметрам объявления: 'ключ значение' в нижнем регистре."""
    return [f"{k} {v}".lower() for k, v in parse_filter_pairs(s) if k in CHECKABLE_FILTER_KEYS and v]


# ---------------------------------------------------------------- эмбеддеры
# Без повторов полей и без чистки: энкодеры обучены на естественном тексте.

def _clip(s, maxlen):
    s = _safe_str(s)
    return s[:maxlen] if s else ""


def build_embedding_query_text(df: pd.DataFrame) -> pd.Series:
    """Запрос + фильтры."""
    def row(q, filt):
        return ". ".join(p for p in [_clip(q, 200), _clip(filt, 300)] if p)
    return pd.Series([row(q, f) for q, f in zip(df["search_query"], df["search_infm_params_text"])],
                     index=df.index)


def build_item_text_for_embedding(df: pd.DataFrame) -> pd.Series:
    """Заголовок. Параметры. Описание - с обрезкой из config.EMB_ITEM_TEXT_KW."""
    from config import EMB_ITEM_TEXT_KW as kw
    titles = df["item_title_raw"].map(lambda x: _clip(x, kw["title_maxlen"]))
    params = df["item_infm_params_text"].map(lambda x: _clip(x, kw["params_maxlen"]))
    desc = df["item_description_raw"].map(lambda x: _clip(x, kw["desc_maxlen"]))
    body = params.str.cat(desc, sep=". ", na_rep="")
    text = titles.str.cat(body, sep=". ", na_rep="")
    return text.map(lambda s: _SPACE_RE.sub(" ", s).strip())
