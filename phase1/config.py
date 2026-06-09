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

NEWS_GLOB = "NewsResult_*.xlsx"   # 프로젝트 루트의 BIGKinds 원본 export

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
if EXP_PROFILE not in _PROFILES:
    raise ValueError(f"알 수 없는 EXP_PROFILE={EXP_PROFILE!r} "
                     f"(가능: {list(_PROFILES)})")
_P = _PROFILES[EXP_PROFILE]

DATASET_FINAL = _PROC / _P["dataset_file"]

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

# --- 경로 (프로필별 분리: y2024 는 접미사 없음 → 기존 산출물 보존) ---
_SUF = "" if EXP_PROFILE == "y2024" else f"_{EXP_PROFILE}"
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
