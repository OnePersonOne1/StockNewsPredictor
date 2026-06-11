"""
shuffle_validation.py — h=252 점수가 '헤드라인 예측력'인지 '시점 단서(라벨 자기상관)'인지 검증

세 가지를 한 번에:
  (A) REAL      : 학습된 모델을 실제 test 에 평가 → horizon별 macro-F1 재현
  (B) SHUFFLE   : test 안에서 headlines 를 행 간 무작위 셔플(날짜·라벨 고정).
                  → 점수가 유지되면 '헤드라인 내용'이 아니라 위치/상수에 의존(시점 단서).
                  → 0.50 부근으로 떨어지면 점수가 헤드라인 내용에 의존함은 맞음.
  (C) SHIFT-NULL: 실제 예측을 고정하고 라벨 시퀀스를 원형 이동(circular shift)시켜
                  macro-F1 분포를 만든 permutation 검정. 라벨이 거의 단일 전환(블록 2개)인
                  h=252 는 자기상관이 커서, '제대로 맞힌' 것처럼 보여도 우연 정렬과
                  구분되지 않음을 p-value 로 보임.

사용:  EXP_PROFILE=samsung_cv HEADLINE_ORDER=time BINARY=1 python3 phase2/shuffle_validation.py
"""
from __future__ import annotations
import sys
import pathlib

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase1.config import (  # noqa: E402
    DATASET_FINAL, ENCODER_NAME, HORIZONS, MAX_LENGTH, MAX_HEADLINES,
    BEST_CKPT, INDEX_NAMES, CLASS_IDX, RESULTS_DIR,
)
from phase2.dataset import make_splits  # noqa: E402
from phase2.model import HeadlineAttentionModel  # noqa: E402

N_SHUFFLE = 10        # 셔플 시드 수
N_SHIFT = None        # None=가능한 모든 원형 이동(=n) 사용


def _mf1(t, p):
    m = t != -100
    return f1_score(t[m], p[m], average="macro", labels=CLASS_IDX, zero_division=0)


@torch.no_grad()
def eval_ordered(model, ds, device, bs=8):
    """ds 순서대로 horizon별 (preds, trues) np 배열 반환."""
    loader = DataLoader(ds, batch_size=bs, shuffle=False)
    preds = {h: [] for h in HORIZONS}
    trues = {h: [] for h in HORIZONS}
    for tok, labels, _meta in loader:
        tk = {k: v.to(device) for k, v in tok.items()}
        out = model(**tk)
        for h in HORIZONS:
            preds[h].append(out["logits"][f"h{h}"].argmax(-1).cpu().numpy())
            trues[h].append(labels[f"h{h}"].numpy())
    return {h: (np.concatenate(preds[h]), np.concatenate(trues[h])) for h in HORIZONS}


def shift_null(p, t, n_shift=None):
    """예측 p 고정, 라벨 t 를 원형 이동시킨 macro-F1 분포로 p-value.
    p_value = P(shifted_mf1 >= real_mf1).  자기상관 보존 permutation 검정."""
    m = t != -100
    p, t = p[m], t[m]
    real = f1_score(t, p, average="macro", labels=CLASS_IDX, zero_division=0)
    n = len(t)
    shifts = range(1, n) if n_shift is None else \
        np.linspace(1, n - 1, n_shift, dtype=int)
    null = [f1_score(np.roll(t, k), p, average="macro", labels=CLASS_IDX,
                     zero_division=0) for k in shifts]
    null = np.array(null)
    p_value = float((null >= real).mean())
    return real, null, p_value


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(BEST_CKPT, map_location=device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ENCODER_NAME)
    max_h = ckpt.get("config", {}).get("max_headlines", MAX_HEADLINES)
    _, _, test_ds, _ = make_splits(DATASET_FINAL, tok, max_h, MAX_LENGTH)

    # 날짜 순 정렬(원형 이동 검정이 실제 시간순을 쓰도록)
    test_ds.df = test_ds.df.sort_values("date").reset_index(drop=True)
    headlines0 = test_ds.df["headlines"].to_numpy().copy()
    n = len(test_ds.df)

    model = HeadlineAttentionModel(ENCODER_NAME, HORIZONS).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    print("=" * 72)
    print(f"shuffle_validation — {RESULTS_DIR.name}  (test n={n})")
    print("=" * 72)

    # ---- (A) REAL ----
    real_eval = eval_ordered(model, test_ds, device)
    real_mf1 = {h: _mf1(t, p) for h, (p, t) in real_eval.items()}
    print("\n[A] REAL  horizon별 macro-F1:")
    for h in HORIZONS:
        print(f"    h={h:<4} macro-F1 = {real_mf1[h]:.3f}")

    # ---- (B) SHUFFLE: test 안에서 headlines 행간 셔플 ----
    print(f"\n[B] SHUFFLE headlines ({N_SHUFFLE} seeds) — 날짜·라벨 고정, 헤드라인만 뒤섞음:")
    shuf = {h: [] for h in HORIZONS}
    for seed in range(N_SHUFFLE):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        test_ds.df["headlines"] = headlines0[perm]
        ev = eval_ordered(model, test_ds, device)
        for h in HORIZONS:
            p_s, t_s = ev[h]
            shuf[h].append(_mf1(t_s, p_s))
    test_ds.df["headlines"] = headlines0  # 원복
    print(f"    {'h':>5} {'REAL':>7} {'SHUFFLE mean±std':>20} {'Δ(real-shuf)':>13}")
    for h in HORIZONS:
        arr = np.array(shuf[h])
        print(f"    {h:>5} {real_mf1[h]:>7.3f} {arr.mean():>10.3f} ± {arr.std():<6.3f}"
              f" {real_mf1[h]-arr.mean():>13.3f}")

    # ---- (C) SHIFT-NULL: 자기상관 고려 유의성 ----
    print("\n[C] SHIFT-NULL (라벨 원형 이동 permutation, 예측 고정) — p=P(null≥real):")
    print(f"    {'h':>5} {'real':>7} {'null mean':>10} {'null max':>9} {'p-value':>9}")
    for h in HORIZONS:
        p, t = real_eval[h]
        real, null, pv = shift_null(p, t)
        print(f"    {h:>5} {real:>7.3f} {null.mean():>10.3f} {null.max():>9.3f} {pv:>9.3f}")

    print("\n해석:")
    print("  - (B)에서 h=252 SHUFFLE 이 REAL 과 비슷 → 헤드라인 내용 아닌 위치/상수 의존(시점).")
    print("  - (C)에서 h=252 p-value 가 크면(예 >0.05) 자기상관만으로도 그 점수가 흔함 → 무의미.")
    print("  - 대조: h=1 은 라벨이 매일 뒤집혀 (B)·(C) 모두 신호 없음이 자명히 드러남.")


if __name__ == "__main__":
    main()
