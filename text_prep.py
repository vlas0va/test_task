"""
text_prep.py
============
Подготовка текстовых полей запроса и объявления для лексического поиска
(BM25 по словам и по символьным n-граммам).

Идея простая: у нас нет отдельного размеченного набора "что важнее — заголовок
или описание", поэтому вместо честного multi-field BM25 (BM25F) используем
прагматичный приём — конкатенацию полей с повторением более важных полей
несколько раз. Повторение слова N раз увеличивает его term frequency в BM25/TF-IDF
ровно так же, как если бы это поле имело больший вес — это дешёвый, но рабочий
способ эмулировать полевые веса без переписывания формулы BM25 под multi-field.

Все функции работают с pandas.Series строк и возвращают pandas.Series строк.
"""
import re
import pandas as pd

# Паттерн для "чистки" текста: оставляем кириллицу, латиницу, цифры и пробелы.
_CLEAN_RE = re.compile(r"[^0-9a-zа-яё\s]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Базовая нормализация: нижний регистр, ё->е, убрать пунктуацию, схлопнуть пробелы."""
    if s is None:
        return ""
    s = str(s).lower()
    s = s.replace("ё", "е")
    s = _CLEAN_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    return s


def build_query_text(df: pd.DataFrame, query_repeat: int = 2) -> pd.Series:
    """
    Текст запроса = сам поисковый запрос (продублированный query_repeat раз,
    чтобы он доминировал над текстом фильтров) + текстовые фильтры поиска.
    search_infm_params_text часто пуст - это нормально, NaN/None обрабатываются.
    Запросов мало (тысячи), поэтому здесь материализация в pandas.Series безопасна.
    """
    q = df["search_query"].fillna("").map(normalize_text)
    filt = df["search_infm_params_text"].fillna("").map(normalize_text)
    text = (q + " ") * query_repeat + filt
    return text.map(lambda s: _SPACE_RE.sub(" ", s).strip())


def _safe_str(x):
    """None и NaN (в т.ч. float('nan') из пропусков в числовых/object-колонках
    pandas) - на пустую строку. Обычный `x or ""` тут не работает: NaN - "истинное"
    значение с точки зрения Python (падает только на 0.0), поэтому пропуски
    в исходных данных проходили бы как float и ломали строковые операции."""
    if x is None:
        return ""
    if isinstance(x, float) and x != x:  # NaN != NaN - самый дешёвый способ проверки без pandas
        return ""
    return x


def _row_item_text(title, params, desc, title_repeat, params_repeat, desc_repeat,
                    params_maxlen, desc_maxlen):
    title = normalize_text(_safe_str(title))
    params = normalize_text(_safe_str(params)[:params_maxlen])
    desc = normalize_text(_safe_str(desc)[:desc_maxlen])
    parts = [title] * title_repeat + [params] * params_repeat + [desc] * desc_repeat
    return _SPACE_RE.sub(" ", " ".join(parts)).strip()


def iter_item_texts(
    df: pd.DataFrame,
    title_repeat: int = 3,
    params_repeat: int = 1,
    desc_repeat: int = 1,
    params_maxlen: int = 300,
    desc_maxlen: int = 250,
):
    """
    Генератор текста объявлений: заголовок (с повтором title_repeat раз -
    самый сильный сигнал соответствия запросу) + начало параметров объявления
    (содержат "Вид услуги X"/"Тип услуги Y" - то же самое поле по смыслу, что
    и search_infm_params_text у запроса) + начало описания.

    Оба длинных поля (infm_params_text и description) обрезаются: судя по
    разведочному анализу, содержательная часть (тип/вид услуги) обычно
    находится в начале, а дальше идёт прайс-лист/график работы - шум для
    лексического сопоставления с коротким запросом, который к тому же
    раздувает длину документа и портит нормировку длины в BM25.

    Это ГЕНЕРАТОР, а не pandas.Series: на корпусе в ~200-350 тыс. объявлений
    материализация полного текста в памяти (несколько копий подряд из-за
    цепочки pandas-операций) требует многих гигабайт ОЗУ. CountVectorizer
    прекрасно принимает итератор строк и обрабатывает документы по одному,
    не держа в памяти всё разом - это и даёт основной выигрыш по памяти.
    """
    cols = df[["item_title_raw", "item_infm_params_text", "item_description_raw"]]
    for title, params, desc in cols.itertuples(index=False, name=None):
        yield _row_item_text(title, params, desc, title_repeat, params_repeat, desc_repeat,
                              params_maxlen, desc_maxlen)


def iter_item_texts_char(df: pd.DataFrame, title_repeat: int = 2, params_maxlen: int = 120):
    """
    Облегчённая версия текста объявления для СИМВОЛЬНОГО n-граммного канала:
    только заголовок (продублированный) + короткий кусок параметров, БЕЗ
    описания. Символьные n-граммы (3-4 символа) на полном тексте объявления
    дают огромный объём токенов (сотни миллионов на корпус из ~200-350 тыс.
    объявлений) - это упирается в память даже с HashingVectorizer, при этом
    решаемая этим каналом проблема ("автопроверка" в запросе не совпадает
    словно с "проверка авто" в объявлении - "слипшиеся" русские составные
    слова) в первую очередь решается именно на уровне заголовка, самом
    коротком и информативном поле. Поэтому здесь сознательно жертвуем частью
    описания/параметров ради того, чтобы канал вообще был вычислимым.
    """
    yield from iter_item_texts(
        df,
        title_repeat=title_repeat,
        params_repeat=1,
        desc_repeat=0,
        params_maxlen=params_maxlen,
        desc_maxlen=0,
    )


def _get_stemmer():
    from nltk.stem.snowball import SnowballStemmer
    return SnowballStemmer("russian")


def stemming_generator(text_iter):
    """
    Оборачивает любой генератор текстов и приводит каждое слово к основе
    (stem) с помощью SnowballStemmer('russian') из nltk - это чисто
    алгоритмический, правило-ориентированный стеммер, без скачивания каких-
    либо моделей/словарей из интернета, работает полностью офлайн.

    Зачем это нужно: русский язык морфологически богат ("шары", "шариками",
    "шара" - разные словоформы одного слова), а BM25/TF-IDF по словам сравнивает
    токены буквально. Без нормализации словоформ запрос "шары" не совпадёт
    по слову с объявлением, где то же самое слово стоит в другом падеже.
    Замер на этих данных показал, что у ~93% отложенных запросов есть хотя бы
    одно дословное пересечение с текстом релевантного объявления, но реальный
    recall BM25 по словам заметно ниже - подозрение в первую очередь падает на
    несовпадение словоформ, стемминг должен это частично исправить.

    Кэшируем результат стемминга по каждому уникальному слову (обычных слов
    в языке кладезь, но не десятки миллионов) - иначе стемминг всего корпуса
    объявлений (десятки миллионов словоупотреблений) может занять больше
    10 минут; с кэшем по уникальным словам - единицы секунд после прогрева.
    """
    stemmer = _get_stemmer()
    cache = {}

    def stem_word(w):
        s = cache.get(w)
        if s is None:
            s = stemmer.stem(w)
            cache[w] = s
        return s

    for text in text_iter:
        yield " ".join(stem_word(w) for w in text.split())


def build_item_text(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Совместимая обёртка поверх iter_item_texts, возвращающая pandas.Series.
    Использовать только на небольших df (holdout-корпусах), для полного
    корпуса лучше напрямую передавать iter_item_texts(...) в BM25Index.fit."""
    return pd.Series(list(iter_item_texts(df, **kwargs)), index=df.index)


# ============================================================================
# Текст для НЕЙРОСЕТЕВЫХ эмбеддингов (bi-encoder / InfoNCE, train_biencoder.py,
# embed_and_retrieve.py). В отличие от BM25-текста выше, тут НЕ нужно:
#   - повторять поля (title x3/x5) - это трюк исключительно для term-frequency
#     в BM25, эмбеддинг-модели не считают "частоту слова", повтор только
#     тратит место в контекстном окне и слегка "спамит" смысл;
#   - агрессивно чистить пунктуацию/регистр (normalize_text) - современные
#     трансформерные энкодеры (BGE-M3, multilingual-e5 и т.п.) обучались на
#     естественном тексте и сами внутри всё токенизируют, лишняя чистка
#     скорее вредит (убирает часть сигнала, например заглавные буквы брендов).
# Поэтому здесь текст собирается "по-человечески": поля через перевод строки,
# только обрезка длины (у энкодеров есть предел по токенам, обычно 512).
# ============================================================================

def _clip(s, maxlen):
    s = _safe_str(s)
    return s[:maxlen] if s else ""


def build_embedding_query_text(df: pd.DataFrame) -> pd.Series:
    """Текст запроса для эмбеддинга: сам запрос + текстовые фильтры (без повторов)."""
    def row(q, filt):
        parts = [p for p in [_clip(q, 200), _clip(filt, 300)] if p]
        return ". ".join(parts)
    return pd.Series(
        [row(q, f) for q, f in zip(df["search_query"], df["search_infm_params_text"])],
        index=df.index,
    )


def build_embedding_item_text(df: pd.DataFrame, use_summary: pd.Series = None,
                               title_maxlen=150, params_maxlen=500, desc_maxlen=500) -> pd.Series:
    """
    Текст объявления для эмбеддинга: заголовок + параметры + описание, без
    повторов, естественным образом через перевод строки.

    use_summary: опционально - pd.Series с LLM-суммаризацией объявления
    (см. summarize_with_llm.py), индексированная так же, как df. Если задана,
    ПОДМЕНЯЕТ params+description на короткую суммаризацию (сжатый, чистый от
    прайс-листов/графиков текст) - обычно даёт эмбеддингу меньше шума на
    входе и позволяет уложиться в лимит токенов модели без грубой обрезки.
    """
    titles = df["item_title_raw"].map(lambda x: _clip(x, title_maxlen))
    if use_summary is not None:
        body = use_summary.reindex(df.index).fillna("")
    else:
        params = df["item_infm_params_text"].map(lambda x: _clip(x, params_maxlen))
        desc = df["item_description_raw"].map(lambda x: _clip(x, desc_maxlen))
        body = params.str.cat(desc, sep=". ", na_rep="")
    text = titles.str.cat(body, sep=". ", na_rep="")
    return text.map(lambda s: _SPACE_RE.sub(" ", s).strip())
