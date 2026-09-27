"""Параметры пайплайна. Общие для обучения (03, 04) и предсказания (05)."""

DATA_DIR = "./data"          # train_*.parquet, benchmark_*.parquet
ART = "./artifacts"          # всё, что считается: эмбеддинги, пулы, модели

# ---------------------------------------------------------------- эмбеддеры
# Бэкбон заморожен, обучается только голова DoubleGeGLUHead (01).
# encode_mode (02):
#   "full"      - текст -> бэкбон + голова в fp16 целиком (так считался e5);
#   "head_only" - сырые эмбеддинги бэкбона (кэш из 01) -> голова в fp32
#                 (так считался bge-m3: не нужно второй раз кодировать 345к объявлений).
# Числа отличаются только в пределах fp16.
ENCODERS = {
    "e5": dict(
        base_model="intfloat/multilingual-e5-base",
        max_seq_length=None,                  # по умолчанию модели, 512
        q_prefix="query: ", p_prefix="passage: ",
        encode_mode="full",
        raw_batch=16, enc_batch=32,           # батч объявлений в 01 и 02 (6 ГБ VRAM)
    ),
    "bge": dict(
        base_model="deepvk/USER-bge-m3",
        max_seq_length=256,                   # по умолчанию 8192, на 6 ГБ VRAM не помещается
        q_prefix="", p_prefix="",
        encode_mode="head_only",
        raw_batch=16, enc_batch=32,
    ),
}


def enc_paths(name):
    """Пути артефактов одного эмбеддера."""
    root = f"{ART}/{name}"
    return dict(
        raw=f"{root}/raw",                    # сырые эмбеддинги бэкбона train (01)
        checkpoints=f"{root}/head_checkpoints",
        head=f"{root}/head",                  # лучшая голова (01)
        train=f"{root}/emb_train",            # итоговые эмбеддинги train (02)
        bench=f"{root}/emb_bench",            # итоговые эмбеддинги benchmark (02)
    )


# текст объявления для эмбеддера (одинаковый в 01 и 02)
EMB_ITEM_TEXT_KW = dict(title_maxlen=150, params_maxlen=500, desc_maxlen=500)

# основной эмбеддер -> каналы dense_*, признаки cos*
TRAIN_EMB_DIR = enc_paths("e5")["train"]
BENCH_EMB_DIR = enc_paths("e5")["bench"]
# второй эмбеддер -> каналы dense2_*, признаки cos2*, cos_mean12. None - выключен.
EMB2 = dict(train=enc_paths("bge")["train"], bench=enc_paths("bge")["bench"])

# ---------------------------------------------------------------- BM25
BM25_KW = dict(k1=1.2, b=0.4, min_df=2)
BM25_ITEM_TEXT_KW = dict(title_repeat=5, params_maxlen=600, desc_maxlen=400)
BM25_TITLE_KW = dict(k1=1.2, b=0.3, min_df=2)   # отдельный индекс по заголовкам

# ---------------------------------------------------------------- каналы кандидатов
# Сколько кандидатов даёт каждый канал. Пул = объединение без дублей (~750 на запрос).
# geo  - только объявления гео-зоны запроса (см. GEO_MIN_P);
# filt - плюс совпадение фильтров "Вид услуги"/"Тип услуги" с параметрами объявления.
CHANNELS = dict(
    bm25_global=100,
    bm25_geo=300,
    bm25_geo_filt=150,
    dense_global=100,
    dense_geo=300,
    dense_geo_filt=150,
    dense2_global=100,
    dense2_geo=300,
    dense2_geo_filt=150,
)
# Локация объявления входит в гео-зону, если по истории train
# P(локация объявления | локация поиска) >= GEO_MIN_P (плюс своя локация).
GEO_MIN_P = 0.005
BATCH_QUERIES = 64          # запросов за проход; 32, если не хватает RAM

# ---------------------------------------------------------------- разбиение и чистка train
SPLIT_SEED = 42
# Доля запросов, отложенная в 01 при обучении голов. Ре-ранкер учится только
# на них: для остальных запросов косинус завышен (голова их видела).
BIENCODER_VAL_FRACTION = 0.10
# valid/test ре-ранкера берутся из первых 3% этого разбиения (одинаковы во всех экспериментах)
EVAL_FROM_FRACTION = 0.03
EVAL_SEEN_SHARE = None      # доля "знакомых" текстов в valid/test; None - как у бенчмарка (0.41)
RERANK_VALID_N = 1500
RERANK_TEST_N = 1500
MAX_RERANK_TRAIN_QUERIES = 30000
MAX_POS_PER_QUERY = 20      # запросы с большим числом выборов - выбросы (до 278)
MIN_QUERY_CHARS = 2
DEDUP_TRAIN_TEXTS = 3       # не больше 3 запросов с одинаковым текстом
NEG_KEEP = 0.5              # доля негативов train, которая остаётся при обучении ранжировщиков

# ---------------------------------------------------------------- ранжирование
MODELS = ["lgb_lambdarank", "lgb_binary", "catboost_yetirank"]
CATBOOST_TASK_TYPE = "GPU"  # "CPU" без видеокарты (медленнее, результат может отличаться)
K = 50

WORK_DIR = f"{ART}/work"                       # пулы, модели, оценки
MODELS_DIR = f"{WORK_DIR}/models"              # модели 03 (обучены на train, проверены на test)
FINAL_MODELS_DIR = f"{WORK_DIR}/models_refit"  # модели 04 (train+valid+test), ими считается ответ
POSTPROC_PARAMS_PATH = f"{WORK_DIR}/postprocess_params.json"
ANSWER_PATH = "./answer.csv"
