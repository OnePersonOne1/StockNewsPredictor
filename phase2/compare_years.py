"""
compare_years.py — EXP-F: 데이터 크기 vs 시기(2025 특수성) 분리 (Phase 2)

단일연도 소표본 복제(2021/2022/2023, 정상기) + 2024(EXP-A, 2025 접촉) + 다년(EXP-D)
의 RoBERTa test 결과를 한 자리에서 비교. 핵심 지표는 h1/h5 macro-F1 과 '예측에 등장한
클래스 수'(1=단일클래스 붕괴).

산출: phase2/results/exp_f_datasize.{csv,md}, figures/exp_f_datasize.png
"""
from __future__ import annotations
import sys
import json
import pathlib

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from phase1.config import INDEX_NAMES, PHASE2_DIR  # noqa: E402

# (라벨, 결과디렉터리, 데이터규모, 시기)
CASES = [
    ("2024 (EXP-A)", "results", "소(406)", "2025 접촉"),
    ("2021", "results_y2021", "소(408)", "정상"),
    ("2022", "results_y2022", "소(406)", "정상"),
    ("2023", "results_y2023", "소(408)", "정상"),
    ("multiyear (EXP-D)", "results_multiyear", "대(1358)", "정상"),
]


def _setup_korean_font():
    import matplotlib
    for fam in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic"):
        if any(f.name == fam for f in matplotlib.font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = fam
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def _load():
    rows = []
    for label, d, size, period in CASES:
        p = PHASE2_DIR / d / "metrics.json"
        if not p.exists():
            print(f"  (없음) {p}"); continue
        m = json.load(open(p, encoding="utf-8"))
        cells = {(r["index"], r["horizon"]): r["macro_f1"] for r in m["cells"]}
        for idx in INDEX_NAMES:
            for h in (1, 5):
                cm = np.array(m["confusion_matrices"][f"{idx}_h{h}"])
                rows.append({"case": label, "size": size, "period": period,
                             "index": idx, "horizon": h,
                             "macro_f1": round(cells[(idx, h)], 3),
                             "pred_classes": int((cm.sum(0) > 0).sum())})
    return pd.DataFrame(rows)


def _plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_korean_font()
    (PHASE2_DIR / "results" / "figures").mkdir(parents=True, exist_ok=True)
    order = [c[0] for c in CASES]
    x = np.arange(len(order)); w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, h in zip(axes, (1, 5)):
        for k, idx in enumerate(INDEX_NAMES):
            vals, pcs = [], []
            for c in order:
                r = df[(df.case == c) & (df["index"] == idx) & (df.horizon == h)]
                vals.append(r["macro_f1"].iloc[0] if len(r) else 0)
                pcs.append(r["pred_classes"].iloc[0] if len(r) else 0)
            bars = ax.bar(x + (k - 0.5) * w, vals, w, label=idx)
            for b, pc in zip(bars, pcs):  # 막대 위에 예측클래스 수
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                        str(pc), ha="center", va="bottom", fontsize=8)
        ax.axhline(1/3, color="gray", ls=":", label="random≈0.333")
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
        ax.set_title(f"RoBERTa test macro-F1 — h={h}")
        if h == 1:
            ax.set_ylabel("macro-F1"); ax.legend(fontsize=8)
    fig.suptitle("데이터 크기 vs 시기(2025) — 막대 위 숫자=예측에 등장한 클래스 수(1=붕괴)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = PHASE2_DIR / "results" / "figures" / "exp_f_datasize.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def run():
    df = _load()
    df.to_csv(PHASE2_DIR / "results" / "exp_f_datasize.csv", index=False, encoding="utf-8-sig")
    md = ["# EXP-F: 데이터 크기 vs 2025 특수성 — RoBERTa h1/h5", "",
          "| case | 규모 | 시기 | index | h1 F1 | h1 클래스 | h5 F1 | h5 클래스 |",
          "|---|---|---|---|---|---|---|---|"]
    for (case, size, period, idx), g in df.groupby(["case", "size", "period", "index"], sort=False):
        r1 = g[g.horizon == 1].iloc[0]; r5 = g[g.horizon == 5].iloc[0]
        md.append(f"| {case} | {size} | {period} | {idx} | {r1.macro_f1} | "
                  f"{r1.pred_classes} | {r5.macro_f1} | {r5.pred_classes} |")
    (PHASE2_DIR / "results" / "exp_f_datasize.md").write_text("\n".join(md), encoding="utf-8")
    fig = _plot(df)
    print(df.to_string(index=False))
    print("\n저장:", PHASE2_DIR / "results" / "exp_f_datasize.csv")
    print("저장:", PHASE2_DIR / "results" / "exp_f_datasize.md")
    print("저장:", fig)


if __name__ == "__main__":
    run()
