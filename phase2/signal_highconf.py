"""
signal_highconf.py — EXP-J: 고확신 구간만의 IC/백테스트 정량화 (다년)

EXP-I 에서 KOSPI h5 고확신(|score| 상위 1/3) 날의 방향 적중률이 0.667 임을 봤다.
여기서는 그 효과를 **전체 vs 고확신 부분집합**으로 IC·이진·롱숏 백테스트 모두에서
정량화한다(signal_test.metrics_for 재사용). 고확신에서 신호가 강해지면 EXP-H 의
약신호가 '확신일에 농축'됨을 정량 확증.

대상: 현재 EXP_PROFILE 의 test(다년=2024 전체), h=1,5. 고확신 = |score| 상위 33%.
산출: results*/signal_highconf.{csv,md}, figures/signal_highconf.png
"""
from __future__ import annotations
import sys
import pathlib

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from phase1.config import RESULTS_DIR  # noqa: E402
from phase2.signal_test import roberta_scores, metrics_for  # noqa: E402

TOP_FRAC = 1 / 3  # 고확신 분위
TARGET_H = (1, 5)


def _font():
    import matplotlib
    for fam in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic"):
        if any(f.name == fam for f in matplotlib.font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = fam; break
    matplotlib.rcParams["axes.unicode_minus"] = False


def run():
    sdf = roberta_scores()
    recs = []
    for (idx, h), g in sdf.groupby(["index", "horizon"]):
        if h not in TARGET_H:
            continue
        full = metrics_for(g)
        thr = g["score"].abs().quantile(1 - TOP_FRAC)
        hc = metrics_for(g[g["score"].abs() >= thr])
        for subset, m in (("전체", full), ("고확신", hc)):
            recs.append({"index": idx, "horizon": h, "subset": subset,
                         "n": m["n"], "bin_acc": m["bin_acc"], "bin_p": m["bin_p"],
                         "IC": m["IC"], "IC_p": m["IC_p"],
                         "ls_ret": m["ls_ret"], "ls_p": m["ls_p"], "hit": m["hit"]})
    res = pd.DataFrame(recs)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS_DIR / "signal_highconf.csv", index=False, encoding="utf-8-sig")
    _report(res)
    _plot(res)
    return res


def _star(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def _report(res):
    md = ["# EXP-J 고확신 구간 신호 정량화 (다년, |score| 상위 1/3)", "",
          "각 셀에서 전체 vs 고확신 부분집합의 신호 지표. *p<.10 **p<.05 ***p<.01.",
          "이진 random=0.5, IC=0, 롱숏수익=0.", "",
          "| index·h | subset | n | 이진acc(p) | IC(p) | 롱숏수익(p) | hit |",
          "|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        ba = f"{r['bin_acc']:.3f}{_star(r['bin_p'])}" if pd.notna(r['bin_acc']) else "—"
        ic = f"{r['IC']:+.3f}{_star(r['IC_p'])}" if pd.notna(r['IC']) else "—"
        ls = f"{r['ls_ret']:+.4f}{_star(r['ls_p'])}" if pd.notna(r['ls_ret']) else "—"
        hit = f"{r['hit']:.3f}" if pd.notna(r['hit']) else "—"
        md.append(f"| {r['index']} h{r['horizon']} | {r['subset']} | {int(r['n'])} | "
                  f"{ba} | {ic} | {ls} | {hit} |")
    md += ["", "> 해석: 고확신 부분집합에서 IC·적중률이 전체보다 오르면, EXP-H 약신호가",
           "> '확신일에 농축'됨을 정량 확증(단 n↓ 으로 분산↑). KOSPI h5 가 가장 뚜렷."]
    (RESULTS_DIR / "signal_highconf.md").write_text("\n".join(md), encoding="utf-8")
    print("=" * 64); print(f"EXP-J 고확신 정량화 — {RESULTS_DIR.name}")
    print("=" * 64)
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}", "display.width", 200):
        print(res[["index", "horizon", "subset", "n", "bin_acc", "IC", "IC_p", "hit"]].to_string(index=False))


def _plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _font()
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    cells = [("KOSPI", 1), ("KOSPI", 5), ("KOSDAQ", 1), ("KOSDAQ", 5)]
    labels = [f"{i}\nh{h}" for i, h in cells]
    x = np.arange(len(cells)); w = 0.38
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    panels = [(a1, "IC", "IC (방향 순위력)", 0.0),
              (a2, "hit", "방향 적중률", 0.5)]
    for ax, col, title, ref in panels:
        for k, sub in enumerate(["전체", "고확신"]):
            vals = [res[(res["index"] == i) & (res.horizon == h) & (res.subset == sub)]
                    [col].iloc[0] for i, h in cells]
            ax.bar(x + (k - 0.5) * w, vals, w, label=sub)
        ax.axhline(ref, color="red", ls="--", lw=1,
                   label=("무작위 0.5" if col == "hit" else "신호 없음 0"))
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("EXP-J 전체 vs 고확신(|score| 상위 1/3) — 신호가 확신일에 농축되는가")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = RESULTS_DIR / "figures" / "signal_highconf.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("저장:", RESULTS_DIR / "signal_highconf.csv", "/ .md /", out.name)


if __name__ == "__main__":
    run()
