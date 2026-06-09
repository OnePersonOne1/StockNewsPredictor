"""
compare_filters.py — 헤드라인 관련성 필터 ablation 종합 (Phase 2)

전체(all) vs 증권_증시(market) vs 금융·거시(macro) 세 입력에서의 test macro-F1 을
TF-IDF / RoBERTa 각각에 대해 한 표/그림으로 비교한다.

입력: results{,_market,_macro}/{baseline_metrics,test_metrics}.csv
산출(주 results/ 디렉터리):
  results/filter_ablation.csv / .md
  results/figures/filter_ablation.png
"""
from __future__ import annotations
import sys
import pathlib

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase1.config import HORIZONS, INDEX_NAMES, PHASE2_DIR  # noqa: E402

RESULTS_MAIN = PHASE2_DIR / "results"
FILTERS = [("all", PHASE2_DIR / "results"),
           ("market", PHASE2_DIR / "results_market"),
           ("macro", PHASE2_DIR / "results_macro")]
MODELS = [("TF-IDF", "baseline_metrics.csv"), ("RoBERTa", "test_metrics.csv")]


def _setup_korean_font():
    import matplotlib
    for fam in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic"):
        if any(f.name == fam for f in matplotlib.font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = fam
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def _load() -> pd.DataFrame:
    rows = []
    for model, fname in MODELS:
        for filt, rdir in FILTERS:
            p = rdir / fname
            if not p.exists():
                print(f"  (없음, 건너뜀) {p}")
                continue
            d = pd.read_csv(p)
            for _, r in d.iterrows():
                rows.append({"model": model, "filter": filt,
                             "index": r["index"], "horizon": int(r["horizon"]),
                             "macro_f1": float(r["macro_f1"])})
    return pd.DataFrame.from_records(rows)


def _plot(long: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_korean_font()
    (RESULTS_MAIN / "figures").mkdir(parents=True, exist_ok=True)

    order = ["all", "market", "macro"]
    x = np.arange(len(order))
    # 주 결과 h=1,5 만, 2x2 (model × horizon), 막대 그룹=index
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for col, h in enumerate((1, 5)):
        for row, model in enumerate(["TF-IDF", "RoBERTa"]):
            ax = axes[row, col]
            w = 0.38
            for k, idx in enumerate(INDEX_NAMES):
                vals = [long[(long.model == model) & (long["filter"] == f)
                             & (long["index"] == idx) & (long.horizon == h)]
                        ["macro_f1"].mean() for f in order]
                ax.bar(x + (k - 0.5) * w, vals, w, label=idx)
            ax.axhline(1/3, color="gray", ls=":", lw=1, label="random≈0.333")
            ax.set_xticks(x); ax.set_xticklabels(order)
            ax.set_title(f"{model} — h={h}")
            if col == 0:
                ax.set_ylabel("test macro-F1")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle("헤드라인 관련성 필터 ablation (주 결과 h=1,5; 높을수록 좋음)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = RESULTS_MAIN / "figures" / "filter_ablation.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def run():
    long = _load()
    if long.empty:
        raise RuntimeError("입력 결과 CSV 없음 — 각 필터로 baseline/evaluate 먼저 실행.")
    long.to_csv(RESULTS_MAIN / "filter_ablation.csv", index=False, encoding="utf-8-sig")

    md = ["# 헤드라인 관련성 필터 ablation — test macro-F1", "",
          "all=전체, market=증권_증시, macro=금융·증시·거시. 주 결과는 h=1,5.", ""]
    for model, _ in MODELS:
        md += [f"## {model}", "",
               "| index | filter | " + " | ".join(f"h={h}" for h in HORIZONS) + " |",
               "|---|---|" + "---|" * len(HORIZONS)]
        for idx in INDEX_NAMES:
            for filt in ["all", "market", "macro"]:
                vals = []
                for h in HORIZONS:
                    v = long[(long.model == model) & (long["filter"] == filt)
                             & (long["index"] == idx) & (long.horizon == h)]["macro_f1"]
                    vals.append(f"{v.mean():.3f}" if len(v) else "—")
                md.append(f"| {idx} | {filt} | " + " | ".join(vals) + " |")
        md.append("")
    (RESULTS_MAIN / "filter_ablation.md").write_text("\n".join(md), encoding="utf-8")

    fig = _plot(long)

    print("=" * 70)
    print("헤드라인 필터 ablation — test macro-F1 (h=1,5 주 결과)")
    print("=" * 70)
    piv = (long[long.horizon.isin([1, 5])]
           .pivot_table(index=["model", "index"], columns=["filter", "horizon"],
                        values="macro_f1"))
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}",
                           "display.width", 200):
        print(piv.to_string())
    print("\n저장:", RESULTS_MAIN / "filter_ablation.csv")
    print("저장:", RESULTS_MAIN / "filter_ablation.md")
    print("저장:", fig)


if __name__ == "__main__":
    run()
