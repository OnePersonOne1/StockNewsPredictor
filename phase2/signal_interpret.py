"""
signal_interpret.py — EXP-I: 신호의 출처 해석 (다년 RoBERTa, test=2024)

EXP-H 에서 검출된 약신호(특히 KOSPI h5, IC≈0.15)가 *무엇에서* 오는지 해석한다:

  1) 확신도 보정(calibration): |score| 분위로 test 일을 나눠 방향 적중률(hit rate).
     확신이 높을수록 적중률이 오르면 신호가 실재하고 확신일에 집중됨을 뜻한다.
  2) attention 상위 헤드라인: 모델이 (정확히) 상승/하락을 맞힌 고확신 날의 top-α 헤드라인.
  3) 키워드 비교: 정확-상승 vs 정확-하락 날, 고-α 헤드라인의 빈출 한국어 토큰.

score = P(up) − P(down). 대상 index 는 KOSPI(신호 강), KOSDAQ(대조). h=1,5.
산출: results_multiyear/signal_interpret.md, top_attention_signal.csv,
      figures/signal_calibration.png
"""
from __future__ import annotations
import sys
import re
import pathlib
from collections import Counter

import numpy as np
import pandas as pd
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from phase1.config import (  # noqa: E402
    DATASET_FINAL, ENCODER_NAME, HORIZONS, MAX_LENGTH, BEST_CKPT, RESULTS_DIR,
)
from phase2.dataset import make_splits  # noqa: E402
from phase2.model import HeadlineAttentionModel  # noqa: E402

_STOP = {"사설", "종합", "속보", "단독", "포토", "영상", "그래픽", "오늘", "특징주",
         "전망", "이슈", "분석", "기자", "년", "월", "일", "칼럼", "에세이"}
_HANGUL = re.compile(r"[가-힣]{2,}")
TARGET_H = (1, 5)


def _tokens(t):
    return [w for w in _HANGUL.findall(str(t)) if w not in _STOP]


def _font():
    import matplotlib
    for fam in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic"):
        if any(f.name == fam for f in matplotlib.font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = fam; break
    matplotlib.rcParams["axes.unicode_minus"] = False


@torch.no_grad()
def collect(ckpt_path=BEST_CKPT):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    mh = ckpt.get("config", {}).get("max_headlines")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ENCODER_NAME)
    _, _, test_ds, _ = make_splits(DATASET_FINAL, tok, mh, MAX_LENGTH)
    model = HeadlineAttentionModel(ENCODER_NAME, HORIZONS).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    rows = []
    for i in range(len(test_ds)):
        tokenized, _, meta = test_ds[i]
        n = int(tokenized["headline_mask"].sum())
        tk = {k: v.unsqueeze(0).to(device) for k, v in tokenized.items()}
        out = model(**tk)
        alpha = out["attention"][0, :n].cpu().numpy()
        hl = [str(h) for h in list(test_ds.df.iloc[i]["headlines"])[:n]]
        top_idx = np.argsort(-alpha)[:3]
        rec = {"date": meta["date"], "index": meta["index_name"],
               "top_headlines": [hl[j] for j in top_idx],
               "top_alpha": [float(alpha[j]) for j in top_idx]}
        r = test_ds.df.iloc[i]
        for h in HORIZONS:
            p = torch.softmax(out["logits"][f"h{h}"], -1)[0].cpu().numpy()
            rec[f"score_h{h}"] = float(p[2] - p[0])
            rec[f"ret_h{h}"] = float(r[f"ret_h{h}"])
        rows.append(rec)
    return pd.DataFrame(rows)


def calibration(df, index, h, nbins=3):
    g = df[df["index"] == index].copy()
    g = g[~g[f"ret_h{h}"].isna()]
    s = g[f"score_h{h}"].abs()
    # |score| 분위 구간
    try:
        g["bin"] = pd.qcut(s, nbins, labels=[f"Q{i+1}" for i in range(nbins)], duplicates="drop")
    except ValueError:
        g["bin"] = "Q1"
    out = []
    for b, gb in g.groupby("bin", observed=True):
        hit = np.sign(gb[f"score_h{h}"]) == np.sign(gb[f"ret_h{h}"])
        nz = gb[f"ret_h{h}"] != 0
        out.append({"bin": str(b), "n": int(nz.sum()),
                    "hit_rate": float(hit[nz].mean()) if nz.sum() else np.nan,
                    "mean_absscore": float(gb[f"score_h{h}"].abs().mean())})
    return pd.DataFrame(out)


def keyword_compare(df, index, h, top_n=15):
    g = df[df["index"] == index]
    up, dn = Counter(), Counter()
    for _, r in g.iterrows():
        if np.isnan(r[f"ret_h{h}"]):
            continue
        correct = np.sign(r[f"score_h{h}"]) == np.sign(r[f"ret_h{h}"])
        if not correct or r[f"ret_h{h}"] == 0:
            continue
        toks = []
        for hl in r["top_headlines"]:
            toks += _tokens(hl)
        (up if r[f"ret_h{h}"] > 0 else dn).update(toks)
    return up.most_common(top_n), dn.most_common(top_n)


def examples(df, index, h, k=5):
    g = df[df["index"] == index].copy()
    g = g[~g[f"ret_h{h}"].isna()]
    g["correct"] = np.sign(g[f"score_h{h}"]) == np.sign(g[f"ret_h{h}"])
    g = g[g["correct"] & (g[f"ret_h{h}"] != 0)]
    g["abss"] = g[f"score_h{h}"].abs()
    return g.sort_values("abss", ascending=False).head(k)


def _plot_calib(calibs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _font()
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(calibs), figsize=(5 * len(calibs), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, c) in zip(axes, calibs):
        ax.bar(c["bin"], c["hit_rate"], color="tab:blue")
        ax.axhline(0.5, color="red", ls="--", label="무작위 0.5")
        for x, (hr, n) in enumerate(zip(c["hit_rate"], c["n"])):
            ax.text(x, hr + 0.01, f"n={n}", ha="center", fontsize=8)
        ax.set_title(title); ax.set_xlabel("|score| 분위 (낮음→높음)")
        ax.set_ylim(0, 0.75)
    axes[0].set_ylabel("방향 적중률"); axes[0].legend(fontsize=8)
    fig.suptitle("EXP-I 확신도 보정 — 확신(|score|)이 높을수록 적중률↑ 이면 신호 실재")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = RESULTS_DIR / "figures" / "signal_calibration.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def run():
    df = collect()
    # top attention 헤드라인 저장(전 test 일)
    recs = []
    for _, r in df.iterrows():
        for hl, a in zip(r["top_headlines"], r["top_alpha"]):
            recs.append({"date": r["date"], "index": r["index"],
                         "attention": round(a, 4), "headline": hl,
                         "score_h5": round(r["score_h5"], 3), "ret_h5": round(r["ret_h5"], 4)})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_csv(RESULTS_DIR / "top_attention_signal.csv",
                              index=False, encoding="utf-8-sig")

    md = ["# EXP-I 신호의 출처 — KOSPI 방향 신호 해석 (다년, test=2024)", ""]
    calibs = []
    for index in ("KOSPI", "KOSDAQ"):
        for h in TARGET_H:
            c = calibration(df, index, h)
            calibs.append((f"{index} h{h}", c))
            md += [f"## {index} · h={h}", "", "**확신도 보정(|score| 분위별 방향 적중률):**",
                   "", "| 분위 | n | 적중률 | 평균|score| |", "|---|---|---|---|"]
            for _, r in c.iterrows():
                md.append(f"| {r['bin']} | {r['n']} | {r['hit_rate']:.3f} | {r['mean_absscore']:.3f} |")
            up, dn = keyword_compare(df, index, h)
            md += ["", "**정확히 맞힌 날의 top-α 헤드라인 빈출 토큰:**", "",
                   "| 순위 | 상승 맞힘 | 하락 맞힘 |", "|---|---|---|"]
            for i in range(max(len(up), len(dn))):
                u = f"{up[i][0]}({up[i][1]})" if i < len(up) else ""
                d = f"{dn[i][0]}({dn[i][1]})" if i < len(dn) else ""
                md.append(f"| {i+1} | {u} | {d} |")
            ex = examples(df, index, h)
            md += ["", "**고확신 정답 사례(top-α 헤드라인):**"]
            for _, r in ex.iterrows():
                md.append(f"- {r['date']} score={r[f'score_h{h}']:+.2f} "
                          f"ret={r[f'ret_h{h}']:+.3f} | {r['top_headlines'][0]}")
            md.append("")
    fig = _plot_calib([calibs[i] for i in (1, 3)])  # KOSPI h5, KOSDAQ h5
    md += [f"그림: `{fig.name}` (KOSPI/KOSDAQ h5 확신도 보정)", "",
           "> 해석: KOSPI 에서 확신(|score|)이 높은 분위일수록 방향 적중률이 무작위(0.5)를",
           "> 웃돌면, EXP-H 의 양(+) IC 가 '고확신일에 집중된 실신호'임을 뒷받침한다."]
    (RESULTS_DIR / "signal_interpret.md").write_text("\n".join(md), encoding="utf-8")

    print("=" * 60); print("EXP-I 신호 해석 — 확신도 보정(적중률)")
    print("=" * 60)
    for title, c in calibs:
        print(f"[{title}] " + " | ".join(f"{r['bin']}:{r['hit_rate']:.2f}(n{r['n']})"
                                         for _, r in c.iterrows()))
    print("\n저장:", RESULTS_DIR / "signal_interpret.md",
          "/ top_attention_signal.csv /", fig.name)


if __name__ == "__main__":
    run()
