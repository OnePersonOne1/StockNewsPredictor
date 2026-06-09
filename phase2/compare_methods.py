"""
compare_methods.py — 방법론 사다리 비교 (Phase 2 종합)

네 단계 예측기를 같은 8셀(horizon×index) 틀에서 macro-F1 로 비교한다:
  1. random        — 무작위 3-class (이론값 1/3)
  2. wordcount     — 비-ML 렉시콘 단어 개수 (wordcount_baseline.py)
  3. TF-IDF+LogReg — 고전 ML 기저 (baseline.py)
  4. RoBERTa attn  — 주 모델 (evaluate.py)

입력 CSV(이미 생성돼 있어야 함):
  results/wordcount_metrics.csv, results/baseline_metrics.csv, results/test_metrics.csv
산출:
  results/method_comparison.csv / .md / .tex
  results/figures/method_comparison.png  (KOSPI/KOSDAQ 2분할, horizon별 곡선)
"""
from __future__ import annotations
import sys
import pathlib

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase1.config import HORIZONS, INDEX_NAMES, RESULTS_DIR, FIGURES_DIR  # noqa: E402

RANDOM_F1 = 1.0 / 3.0
METHODS = [  # (라벨, csv 파일명)
    ("wordcount", "wordcount_metrics.csv"),
    ("TF-IDF", "baseline_metrics.csv"),
    ("RoBERTa", "test_metrics.csv"),
]


def _setup_korean_font():
    import matplotlib
    for fam in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic"):
        if any(f.name == fam for f in matplotlib.font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = fam
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def _load() -> pd.DataFrame:
    frames = []
    for label, fname in METHODS:
        path = RESULTS_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"{path} 없음 — 해당 스크립트를 먼저 실행하세요.")
        d = pd.read_csv(path)[["index", "horizon", "macro_f1"]].copy()
        d["method"] = label
        frames.append(d)
    long = pd.concat(frames, ignore_index=True)
    # random 행 추가
    rnd = [{"index": idx, "horizon": h, "macro_f1": RANDOM_F1, "method": "random"}
           for idx in INDEX_NAMES for h in HORIZONS]
    return pd.concat([pd.DataFrame(rnd), long], ignore_index=True)


def _plot(long: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _setup_korean_font()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    styles = {"random": dict(color="gray", ls=":", marker=None),
              "wordcount": dict(color="tab:red", ls="--", marker="s"),
              "TF-IDF": dict(color="tab:green", ls="--", marker="x"),
              "RoBERTa": dict(color="tab:blue", ls="-", marker="o")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, idx in zip(axes, INDEX_NAMES):
        for method in ["random", "wordcount", "TF-IDF", "RoBERTa"]:
            sub = long[(long["index"] == idx) & (long["method"] == method)] \
                .sort_values("horizon")
            ax.plot(sub["horizon"], sub["macro_f1"], label=method, **styles[method])
        ax.set_xscale("log"); ax.set_xticks(HORIZONS)
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel("horizon (trading days)")
        ax.set_title(idx)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("macro-F1")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("방법론별 horizon 예측력 (macro-F1; 높을수록 좋음)")
    fig.tight_layout()
    out = FIGURES_DIR / "method_comparison.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def _tables(long: pd.DataFrame):
    order = ["random", "wordcount", "TF-IDF", "RoBERTa"]
    wide = (long.pivot_table(index=["index", "method"], columns="horizon",
                             values="macro_f1")
            .reindex(order, level="method"))
    wide.to_csv(RESULTS_DIR / "method_comparison.csv", encoding="utf-8-sig")

    # Markdown
    md = ["# 방법론 비교 (test macro-F1)", "",
          "| index | method | " + " | ".join(f"h={h}" for h in HORIZONS) + " |",
          "|---|---|" + "---|" * len(HORIZONS)]
    for idx in INDEX_NAMES:
        for m in order:
            vals = " | ".join(f"{wide.loc[(idx, m), h]:.3f}" for h in HORIZONS)
            md.append(f"| {idx} | {m} | {vals} |")
    md += ["",
           "주: random=1/3(이론). h=21/h=252 는 test 표본이 작고 거의 단일 클래스라",
           "신뢰도 낮음 — 주 비교는 h=1, h=5."]
    (RESULTS_DIR / "method_comparison.md").write_text("\n".join(md), encoding="utf-8")

    # LaTeX (h별 컬럼, index×method 행)
    lines = [r"\begin{table}[t]\centering",
             r"\caption{Test macro-F1 by method, horizon, and index.}",
             r"\begin{tabular}{ll" + "r" * len(HORIZONS) + "}", r"\toprule",
             "index & method & " + " & ".join(f"$h{{=}}{h}$" for h in HORIZONS)
             + r" \\", r"\midrule"]
    for idx in INDEX_NAMES:
        for m in order:
            vals = " & ".join(f"{wide.loc[(idx, m), h]:.3f}" for h in HORIZONS)
            lines.append(f"{idx} & {m} & {vals} " + r"\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    (RESULTS_DIR / "method_comparison.tex").write_text("\n".join(lines), encoding="utf-8")
    return wide


def run():
    long = _load()
    wide = _tables(long)
    fig = _plot(long)

    print("=" * 70)
    print("방법론 비교 — test macro-F1 (random=0.333)")
    print("=" * 70)
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(wide.to_string())
    print("\n저장:", RESULTS_DIR / "method_comparison.csv")
    print("저장:", RESULTS_DIR / "method_comparison.md")
    print("저장:", RESULTS_DIR / "method_comparison.tex")
    print("저장:", fig)
    print("\n[해석] 주 비교는 h=1,5. h=21/252 는 단일클래스 붕괴로 신뢰도 낮음.")
    return wide


if __name__ == "__main__":
    run()
