"""
build_labels.py — 3-class ternary 라벨 부여 (Phase 1)

확정 설계:
  - 라벨: y_h = +1 if ret_h >  eps_h
              -1 if ret_h < -eps_h
               0 otherwise
  - eps_h = 0.3 * sigma_h
  - sigma_h = TRAIN split 의 ret_h 표준편차  ← 반드시 train 에서만 계산
              (KOSPI·KOSDAQ 를 jointly 학습하므로, horizon 당 하나의 sigma_h 를
               두 지수 train 수익률을 합쳐서 추정. notation σ_h 와 일치)
  - 같은 sigma_h 를 val/test 라벨에도 그대로 적용 (데이터 누설 방지)

dataset_final.parquet 에 label_h{1,5,21,252} 컬럼을 추가해 덮어쓰고,
sigma_h 를 sigma.json 으로도 저장한다(Phase 2 검증/재현용).

이 모듈의 compute_sigma / apply_labels 는 Phase 2 의 dataset.py 에서도
재사용되어 라벨 로직의 single source of truth 가 된다.
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

try:  # 직접 실행(python phase1/build_labels.py) 시
    from config import (
        DATASET_FINAL, PHASE1_DIR, HORIZONS, EPS_SIGMA_MULT, SPLIT_BOUNDS,
    )
except ImportError:  # phase2 등에서 `from phase1.build_labels import ...` 시
    from phase1.config import (
        DATASET_FINAL, PHASE1_DIR, HORIZONS, EPS_SIGMA_MULT, SPLIT_BOUNDS,
    )

SIGMA_JSON = PHASE1_DIR / "data" / "processed" / "sigma.json"


def assign_split(df: pd.DataFrame) -> pd.Series:
    """행의 date(거래일) 기준 temporal split 라벨 부여."""
    date = pd.to_datetime(df["date"])
    split = pd.Series(index=df.index, dtype="object")
    for name, (lo, hi) in SPLIT_BOUNDS.items():
        mask = (date >= pd.Timestamp(lo)) & (date <= pd.Timestamp(hi))
        split[mask] = name
    return split


def compute_sigma(df_train: pd.DataFrame) -> dict[int, float]:
    """TRAIN 행들의 horizon별 ret_h 표준편차 (두 지수 합쳐서)."""
    sigma = {}
    for h in HORIZONS:
        s = float(df_train[f"ret_h{h}"].std(ddof=1))
        sigma[h] = s
    return sigma


def apply_labels(df: pd.DataFrame, sigma: dict[int, float]) -> pd.DataFrame:
    """sigma_h 로 ternary 라벨 컬럼(label_h*) 부여. 원본 복사본 반환."""
    df = df.copy()
    for h in HORIZONS:
        eps = EPS_SIGMA_MULT * sigma[h]
        ret = df[f"ret_h{h}"]
        lab = pd.Series(0, index=df.index, dtype="int64")
        lab[ret > eps] = 1
        lab[ret < -eps] = -1
        lab[ret.isna()] = pd.NA  # 수익률 결측 시 라벨도 결측
        df[f"label_h{h}"] = lab
    return df


def build() -> pd.DataFrame:
    if not DATASET_FINAL.exists():
        raise FileNotFoundError(
            f"{DATASET_FINAL} 없음. 먼저 build_dataset.py 실행."
        )
    df = pd.read_parquet(DATASET_FINAL)
    df["split"] = assign_split(df)

    train = df[df["split"] == "train"]
    if len(train) == 0:
        raise RuntimeError("train split 이 비어 있음 — SPLIT_BOUNDS 확인")
    sigma = compute_sigma(train)

    df = apply_labels(df, sigma)

    # 저장
    df.to_parquet(DATASET_FINAL, index=False)
    SIGMA_JSON.write_text(json.dumps({str(k): v for k, v in sigma.items()},
                                     ensure_ascii=False, indent=2),
                          encoding="utf-8")

    # 리포트
    print("sigma_h (train 기준):")
    for h in HORIZONS:
        print(f"  h={h:>3}: sigma={sigma[h]:.4f}  eps={EPS_SIGMA_MULT*sigma[h]:.4f}")
    print("\nsplit별 행수:", df["split"].value_counts().to_dict())
    print("\n라벨 분포 (전체):")
    for h in HORIZONS:
        vc = df[f"label_h{h}"].value_counts(dropna=False).to_dict()
        print(f"  label_h{h}: {vc}")
    print(f"\n저장: {DATASET_FINAL}")
    print(f"저장: {SIGMA_JSON}")
    return df


if __name__ == "__main__":
    build()
