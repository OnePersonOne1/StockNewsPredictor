"""
signal_summary.py — EXP-H 신호 검정 종합 그림 (다년)

좌: 단일 시드(42) 모델별 IC (TF-IDF≈0, wordcount 음(역방향), RoBERTa 양(+)).
우: RoBERTa IC 시드 스윕(6 시드) — 셀별 분포·평균(24/24 양수, KOSPI h5 최강).
입력: results_multiyear/{signal_metrics.csv, signal_seed_sweep.csv}
산출: results_multiyear/figures/signal_summary.png
"""
from __future__ import annotations
import sys
import pathlib

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from phase1.config import PHASE2_DIR  # noqa: E402

RD = PHASE2_DIR / "results_multiyear"


def _font():
    import matplotlib
    for fam in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic"):
        if any(f.name == fam for f in matplotlib.font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = fam; break
    matplotlib.rcParams["axes.unicode_minus"] = False


def run():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _font()
    met = pd.read_csv(RD / "signal_metrics.csv")
    sw = pd.read_csv(RD / "signal_seed_sweep.csv")
    cells = [("KOSPI", 1), ("KOSPI", 5), ("KOSDAQ", 1), ("KOSDAQ", 5)]
    labels = [f"{i}\nh{h}" for i, h in cells]
    x = np.arange(len(cells))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    # 좌: 모델별 IC (seed 42)
    w = 0.26
    for k, model in enumerate(["TF-IDF", "wordcount", "RoBERTa"]):
        vals = [met[(met.model == model) & (met["index"] == i) & (met.horizon == h)]
                ["IC"].iloc[0] for i, h in cells]
        axL.bar(x + (k - 1) * w, vals, w, label=model)
    axL.axhline(0, color="black", lw=0.8)
    axL.set_xticks(x); axL.set_xticklabels(labels)
    axL.set_ylabel("IC (Spearman, score vs 실제수익)")
    axL.set_title("모델별 IC (단일 시드) — TF-IDF≈0, wordcount 음, RoBERTa 양")
    axL.legend(fontsize=8); axL.grid(axis="y", alpha=0.3)

    # 우: RoBERTa IC 시드 스윕
    for xi, (i, h) in enumerate(cells):
        g = sw[(sw["index"] == i) & (sw.horizon == h)]["IC"].to_numpy()
        axR.scatter([xi] * len(g), g, color="tab:blue", alpha=0.7, zorder=3)
        axR.plot([xi - 0.2, xi + 0.2], [g.mean()] * 2, color="red", lw=2)
    axR.axhline(0, color="black", lw=0.8, label="신호 없음(IC=0)")
    axR.set_xticks(x); axR.set_xticklabels(labels)
    axR.set_ylabel("RoBERTa IC")
    axR.set_title("RoBERTa IC 시드 스윕(6) — 24/24 양수, 빨강=평균")
    axR.legend(fontsize=8); axR.grid(axis="y", alpha=0.3)

    fig.suptitle("EXP-H 신호 검정 — macro-F1 이 가린 약신호가 IC 로 검출됨 (다년 test=2024)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = RD / "figures" / "signal_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig)
    print("저장:", out)


if __name__ == "__main__":
    run()
