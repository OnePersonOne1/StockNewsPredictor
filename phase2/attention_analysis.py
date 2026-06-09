"""
attention_analysis.py — Attention 가중치 심층 분석 (Phase 2 해석)

analyze.py 가 (날짜,지수)별 top-5 헤드라인만 뽑는 데 비해, 이 스크립트는
attention pooling 이 '실제로 헤드라인을 차별화하는가'를 정량적으로 진단한다.

산출:
  results/attention_stats.json            — 집계 통계
  results/attention_analysis.md           — 사람이 읽는 요약 + 키워드 표
  results/figures/attention_entropy.png   — 정규화 엔트로피 분포(균일=1)
  results/figures/attention_heatmap.png   — test 행 × 순위별 정렬 가중치 히트맵

핵심 진단 지표:
  - normalized entropy  H̃ = H(α)/log(n_active) ∈ [0,1]; 1=완전 균일(=평균 pooling)
  - top-1 lift          = max α / (1/n_active);  1=균일, >1=특정 헤드라인 집중
  - top-5 mass          = 상위 5개 가중치 합
키워드 분석: 고-가중치 vs 저-가중치 헤드라인의 빈출 한국어 토큰 비교
  (형태소 분석 없이 2자 이상 한글 어절 빈도 — 한계는 보고서에 명시).
"""
from __future__ import annotations
import sys
import pathlib
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase1.config import (  # noqa: E402
    DATASET_FINAL, ENCODER_NAME, HORIZONS, MAX_LENGTH,
    BEST_CKPT, RESULTS_DIR, FIGURES_DIR,
)
from phase2.dataset import make_splits  # noqa: E402
from phase2.model import HeadlineAttentionModel  # noqa: E402

# 브래킷 태그/형식어 등 분석 노이즈 (빈출하나 의미 약함)
_STOP = {"사설", "종합", "속보", "단독", "포토", "영상", "그래픽", "오늘",
         "특징주", "전망", "이슈", "분석", "기자", "년", "월", "일"}
_HANGUL = re.compile(r"[가-힣]{2,}")


def _setup_korean_font():
    """그림 한글 깨짐 방지: Noto Sans CJK 적용 (없으면 무시)."""
    import matplotlib
    for fam in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic"):
        if any(f.name == fam for f in matplotlib.font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = fam
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


def _tokens(text: str) -> list[str]:
    return [t for t in _HANGUL.findall(str(text)) if t not in _STOP]


@torch.no_grad()
def collect_attention(model, test_ds, device):
    """test 각 행의 전체 α 벡터(실제 헤드라인부)와 헤드라인 텍스트 수집."""
    model.eval()
    rows = []
    for i in range(len(test_ds)):
        tokenized, _, meta = test_ds[i]
        n_act = int(tokenized["headline_mask"].sum())
        tk = {k: v.unsqueeze(0).to(device) for k, v in tokenized.items()}
        alpha = model(**tk)["attention"][0, :n_act].cpu().numpy()
        headlines = [str(h) for h in
                     list(test_ds.df.iloc[i]["headlines"])[:n_act]]
        rows.append({"date": meta["date"], "index": meta["index_name"],
                     "alpha": alpha, "headlines": headlines, "n_active": n_act})
    return rows


def _entropy_stats(rows):
    recs = []
    for r in rows:
        a = r["alpha"]; n = r["n_active"]
        a = np.clip(a, 1e-12, None)
        H = float(-(a * np.log(a)).sum())
        H_norm = H / np.log(n) if n > 1 else 0.0
        recs.append({
            "date": r["date"], "index": r["index"], "n_active": n,
            "norm_entropy": H_norm,
            "top1_lift": float(a.max() / (1.0 / n)),
            "top5_mass": float(np.sort(a)[::-1][:5].sum()),
        })
    return pd.DataFrame.from_records(recs)


def _keyword_compare(rows, k_each: int = 3, top_terms: int = 15):
    """행마다 고-α 상위 k개 vs 저-α 하위 k개 헤드라인의 토큰 빈도 비교."""
    hi, lo = Counter(), Counter()
    for r in rows:
        a = r["alpha"]; hs = r["headlines"]
        order = np.argsort(-a)
        for j in order[:k_each]:
            hi.update(_tokens(hs[j]))
        for j in order[-k_each:]:
            lo.update(_tokens(hs[j]))
    return hi.most_common(top_terms), lo.most_common(top_terms)


def _plot_entropy(stats: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_korean_font()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(stats["norm_entropy"], bins=20, range=(0, 1),
            color="steelblue", edgecolor="white")
    ax.axvline(1.0, color="red", ls="--", label="균일(평균 pooling)=1.0")
    ax.axvline(stats["norm_entropy"].mean(), color="black", ls=":",
               label=f"평균={stats['norm_entropy'].mean():.3f}")
    ax.set_xlabel("normalized attention entropy  H/log n  (1=uniform)")
    ax.set_ylabel("test 행 수")
    ax.set_title("Attention concentration (1 에 붙을수록 평균과 무차별)")
    ax.legend(fontsize=8); fig.tight_layout()
    out = FIGURES_DIR / "attention_entropy.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def _plot_heatmap(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_korean_font()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    M = max(r["n_active"] for r in rows)
    mat = np.full((len(rows), M), np.nan)
    for i, r in enumerate(rows):
        sa = np.sort(r["alpha"])[::-1]
        mat[i, :len(sa)] = sa
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(mat, aspect="auto", cmap="viridis",
                   vmin=0, vmax=np.nanpercentile(mat, 99))
    ax.set_xlabel("헤드라인 순위 (α 내림차순)")
    ax.set_ylabel("test 행 (날짜×지수)")
    ax.set_title("정렬된 attention 가중치 (균일=1/n≈%.3f)" % (1.0 / M))
    fig.colorbar(im, ax=ax, label="α")
    fig.tight_layout()
    out = FIGURES_DIR / "attention_heatmap.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def _plot_attention_maps(rows, stats, n_examples: int = 4):
    """가장 차별화가 큰(top-1 lift 최대) 날들의 헤드라인별 α map (가로 막대)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_korean_font()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # top1_lift 큰 순으로 대표 행 선택 (가장 '집중된' 사례)
    pick = stats.sort_values("top1_lift", ascending=False).head(n_examples)
    keyed = {(r["date"], r["index"]): r for r in rows}

    ncol = 2
    nrow = int(np.ceil(len(pick) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 4.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, prow) in zip(axes, pick.iterrows()):
        r = keyed[(prow["date"], prow["index"])]
        a = r["alpha"]; hs = r["headlines"]
        order = np.argsort(a)                       # 오름차순: 큰 값이 위로
        a_s = a[order]
        labels = [(hs[j][:28] + "…") if len(hs[j]) > 28 else hs[j] for j in order]
        y = np.arange(len(a_s))
        colors = plt.cm.viridis(a_s / a_s.max())
        ax.barh(y, a_s, color=colors)
        ax.axvline(1.0 / len(a_s), color="red", ls="--", lw=1,
                   label="균일 1/n=%.4f" % (1.0 / len(a_s)))
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("attention α")
        ax.set_title(f"{r['date']} {r['index']}  "
                     f"(top-1 lift {prow['top1_lift']:.2f}×, "
                     f"Hnorm={prow['norm_entropy']:.3f})", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
    for ax in axes[len(pick):]:
        ax.axis("off")
    fig.suptitle("Attention map — 헤드라인별 가중치 (가장 집중된 test 사례)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = FIGURES_DIR / "attention_map.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def run(ckpt_path=BEST_CKPT):
    ckpt_path = pathlib.Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"체크포인트 없음: {ckpt_path} (train.py 먼저 실행)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    max_headlines = ckpt.get("config", {}).get("max_headlines")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    _, _, test_ds, _ = make_splits(DATASET_FINAL, tokenizer, max_headlines, MAX_LENGTH)

    model = HeadlineAttentionModel(ENCODER_NAME, HORIZONS).to(device)
    model.load_state_dict(ckpt["model_state"])

    rows = collect_attention(model, test_ds, device)
    stats = _entropy_stats(rows)
    hi, lo = _keyword_compare(rows)

    uniform = float(np.mean(1.0 / stats["n_active"]))
    summary = {
        "n_test_rows": len(rows),
        "mean_n_active": float(stats["n_active"].mean()),
        "mean_norm_entropy": float(stats["norm_entropy"].mean()),
        "min_norm_entropy": float(stats["norm_entropy"].min()),
        "mean_top1_lift": float(stats["top1_lift"].mean()),
        "max_top1_lift": float(stats["top1_lift"].max()),
        "mean_top5_mass": float(stats["top5_mass"].mean()),
        "uniform_weight_1_over_n": uniform,
        "interpretation": (
            "norm_entropy≈1 이고 top1_lift≈1 이면 attention 이 균일에 가까워 "
            "pooling 이 사실상 평균과 무차별(헤드라인 차별화 실패)."),
    }

    fig1 = _plot_entropy(stats)
    fig2 = _plot_heatmap(rows)
    fig3 = _plot_attention_maps(rows, stats)
    _report(summary, hi, lo, stats, fig1, fig2, fig3)
    return summary


def _report(summary, hi, lo, stats, fig1, fig2, fig3):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "attention_stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    stats.to_csv(RESULTS_DIR / "attention_per_row.csv",
                 index=False, encoding="utf-8-sig")

    md = ["# Attention 심층 분석 요약", "",
          f"- test 행 수: {summary['n_test_rows']} "
          f"(평균 실제 헤드라인 {summary['mean_n_active']:.0f}개, "
          f"균일 가중치 1/n≈{summary['uniform_weight_1_over_n']:.4f})",
          f"- **정규화 엔트로피 평균 {summary['mean_norm_entropy']:.3f}** "
          f"(1=완전 균일=평균 pooling; 최소 {summary['min_norm_entropy']:.3f})",
          f"- **top-1 lift 평균 {summary['mean_top1_lift']:.2f}×** "
          f"(1=균일; 최대 {summary['max_top1_lift']:.2f}×)",
          f"- top-5 가중치 합 평균 {summary['mean_top5_mass']:.3f}",
          "",
          f"> 해석: {summary['interpretation']}", "",
          "## 고-가중치 vs 저-가중치 헤드라인 빈출 토큰 (행별 상·하위 3개)", "",
          "| 순위 | 고-α 토큰(빈도) | 저-α 토큰(빈도) |", "|---|---|---|"]
    for i in range(max(len(hi), len(lo))):
        h = f"{hi[i][0]} ({hi[i][1]})" if i < len(hi) else ""
        l = f"{lo[i][0]} ({lo[i][1]})" if i < len(lo) else ""
        md.append(f"| {i+1} | {h} | {l} |")
    md += ["", f"그림: `{fig1.name}`, `{fig2.name}`, `{fig3.name}`"]
    (RESULTS_DIR / "attention_analysis.md").write_text("\n".join(md), encoding="utf-8")

    print("=" * 64)
    print("Attention 심층 분석")
    print("=" * 64)
    print(f"정규화 엔트로피 평균 = {summary['mean_norm_entropy']:.3f} (1=균일/평균 pooling)")
    print(f"top-1 lift 평균      = {summary['mean_top1_lift']:.2f}× (1=균일)")
    print(f"top-5 mass 평균      = {summary['mean_top5_mass']:.3f}")
    print("\n저장:", RESULTS_DIR / "attention_stats.json")
    print("저장:", RESULTS_DIR / "attention_analysis.md")
    print("저장:", RESULTS_DIR / "attention_per_row.csv")
    print("저장:", fig1)
    print("저장:", fig2)
    print("저장:", fig3)


if __name__ == "__main__":
    run()
