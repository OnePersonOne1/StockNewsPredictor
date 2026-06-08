"""
config.py — 전 단계 공용 하이퍼파라미터 / 경로 정의

Phase 1(데이터셋·라벨 구축)과 Phase 2(모델 학습·평가) 설정을 한 곳에 모은다.
모든 모듈은 여기서 값을 import 해 사용한다 (매직 넘버 금지).
"""
from __future__ import annotations
from pathlib import Path

# ===========================================================================
# 공통 경로
# ===========================================================================
ROOT = Path(__file__).resolve().parents[1]
PHASE1_DIR = ROOT / "phase1"
PHASE2_DIR = ROOT / "phase2"

PRICES_PARQUET = PHASE1_DIR / "data" / "processed" / "prices.parquet"
DATASET_FINAL = PHASE1_DIR / "data" / "processed" / "dataset_final.parquet"

NEWS_GLOB = "NewsResult_*.xlsx"   # 프로젝트 루트의 BIGKinds 원본 export

SEED = 42

# ===========================================================================
# Phase 1 — 데이터셋 / 라벨 구축
# ===========================================================================
INDEX_NAMES = ("KOSPI", "KOSDAQ")
HORIZONS = (1, 5, 21, 252)        # 예측 horizon (거래일)

# 3-class ternary 라벨 임계: eps_h = EPS_SIGMA_MULT * sigma_h
# (sigma_h 는 train split 의 ret_h 표준편차, 아래 §사전확정 사항)
EPS_SIGMA_MULT = 0.3

# Temporal split 경계 (행의 거래일 기준, 양끝 포함)
SPLIT_BOUNDS = {
    "train": ("2024-01-01", "2024-10-31"),
    "val":   ("2024-11-01", "2024-11-30"),
    "test":  ("2024-12-01", "2024-12-31"),
}

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

# --- 경로 ---
CHECKPOINT_DIR = PHASE2_DIR / "data" / "checkpoints"
BEST_CKPT = CHECKPOINT_DIR / "best.pt"
RESULTS_DIR = PHASE2_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_JSON = RESULTS_DIR / "metrics.json"
TOP_ATTN_CSV = RESULTS_DIR / "top_attention_headlines.csv"

# 라벨 정수 인코딩: {-1,0,+1} <-> {0,1,2} (CrossEntropy 용)
LABEL2IDX = {-1: 0, 0: 1, 1: 2}
IDX2LABEL = {0: -1, 1: 0, 2: 1}
CLASS_NAMES = ("down(-1)", "flat(0)", "up(+1)")
