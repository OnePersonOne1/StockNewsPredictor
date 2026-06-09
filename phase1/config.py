"""
config.py — 전 단계 공용 하이퍼파라미터 / 경로 정의

Phase 1(데이터셋·라벨 구축)과 Phase 2(모델 학습·평가) 설정을 한 곳에 모은다.
모든 모듈은 여기서 값을 import 해 사용한다 (매직 넘버 금지).
"""
from __future__ import annotations
import os
from pathlib import Path

# ===========================================================================
# 공통 경로
# ===========================================================================
ROOT = Path(__file__).resolve().parents[1]
PHASE1_DIR = ROOT / "phase1"
PHASE2_DIR = ROOT / "phase2"
_PROC = PHASE1_DIR / "data" / "processed"

PRICES_PARQUET = _PROC / "prices.parquet"

NEWS_GLOB = "data/NewsResult_*.xlsx"   # data/ 의 BIGKinds 원본 export (2021~2024)

SEED = 42

# ===========================================================================
# 실험 프로필 (EXP_PROFILE 환경변수로 선택; 기본 = 기존 2024-only)
# ---------------------------------------------------------------------------
#   y2024     : 학습 2024-01~10 / val 11 / test 12  (현 주 실험, 불변)
#   multiyear : 학습 2021~2023 / val 2023H2 말 / test 2024 전체
#               (2021~2023 BIGKinds 헤드라인 xlsx + 2021~ 가격 필요)
# 프로필마다 split·데이터셋 경로·포함 연도·가격 백필 시작이 달라진다.
# 기존 2024 산출물은 그대로 보존(다른 parquet 경로) → 두 실험 공존.
# ===========================================================================
EXP_PROFILE = os.environ.get("EXP_PROFILE", "y2024")

_PROFILES = {
    "y2024": {
        "split_bounds": {
            "train": ("2024-01-01", "2024-10-31"),
            "val":   ("2024-11-01", "2024-11-30"),
            "test":  ("2024-12-01", "2024-12-31"),
        },
        "dataset_file": "dataset_final.parquet",
        "data_years": (2024,),
        "price_back_start": None,        # 가격 백필 불필요(2024 CSV 보유)
    },
    "multiyear": {
        "split_bounds": {
            "train": ("2021-01-01", "2023-09-30"),
            "val":   ("2023-10-01", "2023-12-31"),
            "test":  ("2024-01-01", "2024-12-31"),
        },
        "dataset_file": "dataset_2124.parquet",
        "data_years": (2021, 2022, 2023, 2024),
        "price_back_start": "20210101",  # pykrx 로 2021~2023 백필 (KRX 자격증명 필요)
    },
}

# 단일연도 소표본 복제 프로필(EXP-F): 2024 와 동일 구조(Jan~Oct/Nov/Dec)를 정상기
# 2021·2022·2023 에 재현 → 'EXP-A 붕괴가 데이터 크기 탓인지 2025 특수성 탓인지' 분리.
# 가격은 prices.parquet(2021~2026, EXP-D 에서 구축)에 이미 포함 → 백필 불필요.
for _yr in (2021, 2022, 2023):
    _PROFILES[f"y{_yr}"] = {
        "split_bounds": {
            "train": (f"{_yr}-01-01", f"{_yr}-10-31"),
            "val":   (f"{_yr}-11-01", f"{_yr}-11-30"),
            "test":  (f"{_yr}-12-01", f"{_yr}-12-31"),
        },
        "dataset_file": f"dataset_{_yr}.parquet",
        "data_years": (_yr,),
        "price_back_start": None,
    }
if EXP_PROFILE not in _PROFILES:
    raise ValueError(f"알 수 없는 EXP_PROFILE={EXP_PROFILE!r} "
                     f"(가능: {list(_PROFILES)})")
_P = _PROFILES[EXP_PROFILE]

# ===========================================================================
# 헤드라인 관련성 필터 (HEADLINE_FILTER 환경변수; 기본 all)
# ---------------------------------------------------------------------------
# BIGKinds '통합 분류1' 카테고리로 잡음 기사를 걸러 신호/잡음비를 높이는 실험.
#   all    : 전체(현재) — 필터 없음
#   market : 경제>증권_증시 만 (일평균 ~28건, 시장 직결)
#   macro  : 증권+금융+국제경제+외환+경제일반 (일평균 ~66건, 시장·거시)
# 칼럼/오피니언은 BIGKinds 별도 카테고리가 없고 제목태그로 일 ~6건뿐이라 제외.
# (HEADLINE_CATEGORY_REGEX 가 None 이면 build_dataset 가 필터를 건너뜀)
# ===========================================================================
HEADLINE_FILTER = os.environ.get("HEADLINE_FILTER", "all")
_FILTERS = {
    "all":       None,
    "market":    r"증권_증시",
    "macro":     r"증권_증시|금융_재테크|국제경제|외환|경제일반",
    # EXP-G: 한국장의 높은 IT/반도체 비중 반영. 반도체는 경제>반도체로 IT_과학과
    # 별도 분류되어 둘 다 포함. it=섹터 단독, market_it=시장+섹터(볼륨↑로 RoBERTa 보완).
    "it":        r"IT_과학|반도체",
    "market_it": r"증권_증시|IT_과학|반도체",
}
if HEADLINE_FILTER not in _FILTERS:
    raise ValueError(f"알 수 없는 HEADLINE_FILTER={HEADLINE_FILTER!r} "
                     f"(가능: {list(_FILTERS)})")
HEADLINE_CATEGORY_REGEX = _FILTERS[HEADLINE_FILTER]

# 프로필·필터 조합별 접미사 (y2024+all → 빈 문자열 → 기존 산출물 보존)
_TAGS = ([] if EXP_PROFILE == "y2024" else [EXP_PROFILE]) \
        + ([] if HEADLINE_FILTER == "all" else [HEADLINE_FILTER])
_SUF = ("_" + "_".join(_TAGS)) if _TAGS else ""

_DS_BASE = _P["dataset_file"][:-len(".parquet")]
DATASET_FINAL = _PROC / (_DS_BASE
                         + ("" if HEADLINE_FILTER == "all" else f"_{HEADLINE_FILTER}")
                         + ".parquet")

# ===========================================================================
# Phase 1 — 데이터셋 / 라벨 구축
# ===========================================================================
INDEX_NAMES = ("KOSPI", "KOSDAQ")
HORIZONS = (1, 5, 21, 252)        # 예측 horizon (거래일)

# 3-class ternary 라벨 임계: eps_h = EPS_SIGMA_MULT * sigma_h
# (sigma_h 는 train split 의 ret_h 표준편차, 아래 §사전확정 사항)
EPS_SIGMA_MULT = 0.3

# Temporal split 경계 (행의 거래일 기준, 양끝 포함) — 프로필에서 주입
SPLIT_BOUNDS = _P["split_bounds"]
DATA_YEARS = _P["data_years"]            # dataset 행으로 포함할 연도
PRICE_BACK_START = _P["price_back_start"]  # 가격 백필 시작(YYYYMMDD) 또는 None

# ===========================================================================
# Phase 2 — 모델 / 학습 / 평가
# ===========================================================================
# --- 인코더 / 토크나이즈 ---
ENCODER_NAME = "klue/roberta-base"
MAX_HEADLINES = 30      # 일별 헤드라인 중 시간순 최신 N개 (GPU 메모리 제약)
MAX_LENGTH = 64         # 헤드라인 토큰 최대 길이
N_CLASSES = 3           # {-1, 0, +1}

# --- 학습 ---
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EPOCHS = 4              # 3~5 권장
BATCH_SIZE = 8         # T4 16GB 가정; OOM 시 4로 fallback
FREEZE_ENCODER = False
GRAD_CLIP = 1.0

# --- 경로 (프로필·필터 조합별 분리: y2024+all 은 접미사 없음 → 기존 산출물 보존) ---
CHECKPOINT_DIR = PHASE2_DIR / "data" / f"checkpoints{_SUF}"
BEST_CKPT = CHECKPOINT_DIR / "best.pt"
RESULTS_DIR = PHASE2_DIR / f"results{_SUF}"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_JSON = RESULTS_DIR / "metrics.json"
TOP_ATTN_CSV = RESULTS_DIR / "top_attention_headlines.csv"

# 라벨 정수 인코딩: {-1,0,+1} <-> {0,1,2} (CrossEntropy 용)
LABEL2IDX = {-1: 0, 0: 1, 1: 2}
IDX2LABEL = {0: -1, 1: 0, 2: 1}
CLASS_NAMES = ("down(-1)", "flat(0)", "up(+1)")
