"""
compare_attention.py — 케이스 간 attention 집중도 요약 (Phase 2)

각 실험 케이스 폴더의 attention_stats.json 을 모아, attention 이 균일(평균 pooling=
학습 실패)에 가까운지 차별화(학습 성공)됐는지를 한 표/그림으로 비교한다.
attention 은 단일 query 라 horizon 공통(=케이스당 하나의 분포)이다.

산출: phase2/results/attention_summary.{csv,md}, figures/attention_summary.png
"""
from __future__ import annotations
import sys
import json
import pathlib

import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from phase1.config import PHASE2_DIR  # noqa: E402

# (케이스 라벨, 결과 폴더, 데이터 규모)
CASES = [
    ("2024 all (EXP-A)", "results", "소"),
    ("2024 market (EXP-C)", "results_market", "소"),
    ("2024 macro (EXP-C)", "results_macro", "소"),
    ("2021 (EXP-F)", "results_y2021", "소"),
    ("2022 (EXP-F)", "results_y2022", "소"),
    ("2023 (EXP-F)", "results_y2023", "소"),
    ("multiyear all (EXP-D)", "results_multiyear", "대"),
    ("multiyear market (EXP-E)", "results_multiyear_market", "대"),
    ("multiyear macro (EXP-E)", "results_multiyear_macro", "대"),
    ("multiyear it (EXP-G)", "results_multiyear_it", "대"),
    ("multiyear market_it (EXP-G)", "results_multiyear_market_it", "대"),
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
    for label, d, size in CASES:
        p = PHASE2_DIR / d / "attention_stats.json"
        if not p.exists():
            print(f"  (없음) {p}"); continue
        s = json.load(open(p, encoding="utf-8"))
        rows.append({"case": label, "size": size,
                     "norm_entropy": round(s["mean_norm_entropy"], 3),
                     "top1_lift": round(s["mean_top1_lift"], 2),
                     "top5_mass": round(s["mean_top5_mass"], 3)})
    return pd.DataFrame(rows)


def _plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _setup_korean_font()
    (PHASE2_DIR / "results" / "figures").mkdir(parents=True, exist_ok=True)
    y = np.arange(len(df))[::-1]
    colors = ["tab:orange" if s == "소" else "tab:blue" for s in df["size"]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(y, df["top1_lift"], color=colors)
    ax.axvline(1.0, color="red", ls="--", lw=1, label="균일(평균 pooling)=1.0")
    ax.set_yticks(y); ax.set_yticklabels(df["case"], fontsize=8)
    ax.set_xlabel("top-1 attention lift (1=균일/평균, ↑=차별화)")
    ax.set_title("케이스별 attention 집중도 — 소표본(주황)=붕괴, 대표본(파랑)=차별화")
    # 막대 끝에 엔트로피 표기
    for yi, (lift, ent) in zip(y, zip(df["top1_lift"], df["norm_entropy"])):
        ax.text(lift + 0.1, yi, f"Hn={ent}", va="center", fontsize=7)
    ax.legend(fontsize=8); fig.tight_layout()
    out = PHASE2_DIR / "results" / "figures" / "attention_summary.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def run():
    df = _load()
    df.to_csv(PHASE2_DIR / "results" / "attention_summary.csv", index=False, encoding="utf-8-sig")
    md = ["# 케이스별 attention 집중도 (단일 query → horizon 공통)", "",
          "norm_entropy 1=균일(평균 pooling=학습 실패), top1_lift 1=균일·↑=차별화.", "",
          "| 케이스 | 규모 | norm_entropy | top1_lift | top5_mass |",
          "|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r['case']} | {r['size']} | {r['norm_entropy']} | "
                  f"{r['top1_lift']} | {r['top5_mass']} |")
    md += ["", "관찰: 소표본 케이스는 모두 attention 붕괴(lift~1=평균 pooling)이고, "
           "대표본은 차별화(lift 6~8). 예외 multiyear market_it 의 붕괴는 seed 42 학습 "
           "실패와 일치(시드 변동성, EXP-G §4)."]
    (PHASE2_DIR / "results" / "attention_summary.md").write_text("\n".join(md), encoding="utf-8")
    fig = _plot(df)
    print(df.to_string(index=False))
    print("\n저장:", PHASE2_DIR / "results" / "attention_summary.csv")
    print("저장:", PHASE2_DIR / "results" / "attention_summary.md")
    print("저장:", fig)


if __name__ == "__main__":
    run()
