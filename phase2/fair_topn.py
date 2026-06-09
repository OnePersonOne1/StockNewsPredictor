"""
fair_topn.py — EXP-R: 공정 비교 (TF-IDF·wordcount 도 top-30 으로) × {3-class, 이진}

지금까지 TF-IDF·wordcount 는 하루 **전체** 헤드라인을, RoBERTa 는 **최신 30개**만 썼다
(같은 조건 아님). 여기서는 TF-IDF·wordcount 도 최신 N(=30)개로 제한해 **같은 입력**으로
맞추고, **3-class macro-F1** 과 **이진 up/down(+IC)** 두 지표 모두 비교한다.

대상: 다년 test=2024. RoBERTa(top-30) 는 저장된 EXP-D(3-class)·EXP-H(이진)에서 참조.
산출: results_multiyear/fair_topn.{csv,md}, figures/fair_topn.png
"""
from __future__ import annotations
import sys
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from phase1.config import (DATASET_FINAL, HORIZONS, INDEX_NAMES, SEED,  # noqa: E402
                           PHASE2_DIR)
from phase1.build_labels import compute_sigma, apply_labels, assign_split  # noqa: E402
from phase2.signal_test import metrics_for  # noqa: E402
from phase2.wordcount_baseline import _net_score, POS_SEEDS, NEG_SEEDS  # noqa: E402

RD = PHASE2_DIR / "results_multiyear"
TOPNS = [30, 0]   # 0 = 전체


def _load_df():
    df = pd.read_parquet(DATASET_FINAL)
    if "split" not in df.columns:
        df["split"] = assign_split(df)
    return apply_labels(df, compute_sigma(df[df.split == "train"]))


def _doc(headlines, n):
    hs = list(headlines)[:n] if n > 0 else list(headlines)
    return " ".join(str(h) for h in hs)


def tfidf_eval(df, topn):
    """(index,h) → 3-class macro-F1 + 이진/IC (top-N TF-IDF+LogReg)."""
    rows = []
    for idx in INDEX_NAMES:
        sub = df[df.index_name == idx]
        tr, te = sub[sub.split == "train"], sub[sub.split == "test"]
        vec = TfidfVectorizer(min_df=2, max_features=20000)
        Xtr = vec.fit_transform([_doc(h, topn) for h in tr["headlines"]])
        Xte = vec.transform([_doc(h, topn) for h in te["headlines"]])
        for h in HORIZONS:
            ytr = tr[f"label_h{h}"].astype(int).to_numpy()
            clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=SEED).fit(Xtr, ytr)
            yte = te[f"label_h{h}"].astype(int).to_numpy()
            pred = clf.predict(Xte)
            mf1 = f1_score(yte, pred, average="macro", labels=[-1, 0, 1], zero_division=0)
            cls = list(clf.classes_)
            proba = clf.predict_proba(Xte)
            p_up = proba[:, cls.index(1)] if 1 in cls else np.zeros(len(te))
            p_dn = proba[:, cls.index(-1)] if -1 in cls else np.zeros(len(te))
            g = pd.DataFrame({"score": p_up - p_dn,
                              "ret": te[f"ret_h{h}"].to_numpy(), "true": yte})
            m = metrics_for(g)
            rows.append({"model": "TF-IDF", "index": idx, "horizon": h,
                         "macro_f1": mf1, "bin_acc": m["bin_acc"], "IC": m["IC"], "IC_p": m["IC_p"]})
    return rows


def wordcount_eval(df, topn):
    """(index,h) → 3-class macro-F1 + 이진/IC (top-N 렉시콘 net score)."""
    rows = []
    for idx in INDEX_NAMES:
        sub = df[df.index_name == idx].copy()
        sub["score"] = sub["headlines"].map(lambda hs: float(_net_score(_doc(hs, topn))))
        tr, te = sub[sub.split == "train"], sub[sub.split == "test"]
        for h in HORIZONS:
            ytr = tr[f"label_h{h}"].astype(int).to_numpy()
            s_tr, s_te = tr["score"].to_numpy(), te["score"].to_numpy()
            yte = te[f"label_h{h}"].astype(int).to_numpy()
            # 3-class: train 분위수 매칭(wordcount_baseline 와 동일 규칙)
            p_dn = float(np.mean(ytr == -1)); p_fl = float(np.mean(ytr == 0))
            tau_lo = np.quantile(s_tr, p_dn) if p_dn > 0 else s_tr.min() - 1
            tau_hi = np.quantile(s_tr, p_dn + p_fl) if (p_dn + p_fl) < 1 else s_tr.max() + 1
            pred = np.zeros_like(s_te, dtype=int)
            pred[s_te > tau_hi] = 1; pred[s_te < tau_lo] = -1
            mf1 = f1_score(yte, pred, average="macro", labels=[-1, 0, 1], zero_division=0)
            g = pd.DataFrame({"score": s_te, "ret": te[f"ret_h{h}"].to_numpy(), "true": yte})
            m = metrics_for(g)
            rows.append({"model": "wordcount", "index": idx, "horizon": h,
                         "macro_f1": mf1, "bin_acc": m["bin_acc"], "IC": m["IC"], "IC_p": m["IC_p"]})
    return rows


def roberta_ref():
    """RoBERTa(top-30) 참조: 3-class=EXP-D metrics.json, 이진/IC=EXP-H signal_metrics.csv."""
    rows = []
    m3 = json.load(open(RD / "metrics.json", encoding="utf-8"))
    f1 = {(r["index"], r["horizon"]): r["macro_f1"] for r in m3["cells"]}
    sig = pd.read_csv(RD / "signal_metrics.csv")
    sig = sig[sig.model == "RoBERTa"]   # model 필터(필수)
    for idx in INDEX_NAMES:
        for h in HORIZONS:
            sr = sig[(sig["index"] == idx) & (sig.horizon == h)]
            rows.append({"model": "RoBERTa", "index": idx, "horizon": h,
                         "macro_f1": f1[(idx, h)],
                         "bin_acc": float(sr["bin_acc"].iloc[0]) if len(sr) else np.nan,
                         "IC": float(sr["IC"].iloc[0]) if len(sr) else np.nan,
                         "IC_p": float(sr["IC_p"].iloc[0]) if len(sr) else np.nan})
    return rows


def run():
    df = _load_df()
    recs = []
    for topn in TOPNS:
        tag = f"top{topn}" if topn else "all"
        for r in tfidf_eval(df, topn) + wordcount_eval(df, topn):
            recs.append({**r, "input": tag})
    # RoBERTa 는 top-30 만(구조상 항상 top-30)
    for r in roberta_ref():
        recs.append({**r, "input": "top30"})
    res = pd.DataFrame(recs)
    RD.mkdir(parents=True, exist_ok=True)
    res.to_csv(RD / "fair_topn.csv", index=False, encoding="utf-8-sig")
    _report(res); _plot(res)
    return res


def _report(res):
    md = ["# EXP-R 공정 비교 — 동일 top-30 입력 × {3-class, 이진}", "",
          "TF-IDF·wordcount 를 RoBERTa 와 같은 최신 30개로 제한. 비교: all(전체) vs top30.",
          "이진 random=0.5, IC=0; 3-class random macro-F1≈0.333.", ""]
    for h in (1, 5):
        md += [f"## h = {h}", "",
               "| model | input | KOSPI 3cls/bin/IC | KOSDAQ 3cls/bin/IC |", "|---|---|---|---|"]
        for model in ["TF-IDF", "wordcount", "RoBERTa"]:
            for inp in (["top30", "all"] if model != "RoBERTa" else ["top30"]):
                def cell(idx):
                    r = res[(res.model == model) & (res.input == inp) &
                            (res["index"] == idx) & (res.horizon == h)]
                    if not len(r):
                        return "—"
                    r = r.iloc[0]
                    return f"{r.macro_f1:.3f}/{r.bin_acc:.3f}/{r.IC:+.3f}"
                md.append(f"| {model} | {inp} | {cell('KOSPI')} | {cell('KOSDAQ')} |")
        md.append("")
    (RD / "fair_topn.md").write_text("\n".join(md), encoding="utf-8")
    print("=" * 70); print("EXP-R 공정 비교 (top-30 동일 입력) — h=1,5")
    print("=" * 70)
    show = res[res.horizon.isin([1, 5])].pivot_table(
        index=["model", "input"], columns=["index", "horizon"],
        values=["macro_f1", "IC"])
    with pd.option_context("display.float_format", lambda v: f"{v:+.3f}", "display.width", 220):
        print(show.to_string())


def _plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    for fam in ("Noto Sans CJK KR", "Noto Sans CJK JP"):
        if any(f.name == fam for f in fm.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = fam; break
    matplotlib.rcParams["axes.unicode_minus"] = False
    (RD / "figures").mkdir(parents=True, exist_ok=True)
    bars = [("TF-IDF", "all"), ("TF-IDF", "top30"), ("wordcount", "all"),
            ("wordcount", "top30"), ("RoBERTa", "top30")]
    labels = [f"{m}\n{i}" for m, i in bars]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for col, metric in enumerate(["macro_f1", "IC"]):
        for row, h in enumerate((1, 5)):
            ax = axes[row, col]; x = np.arange(len(bars)); w = 0.38
            for k, idx in enumerate(INDEX_NAMES):
                vals = []
                for m, i in bars:
                    r = res[(res.model == m) & (res.input == i) &
                            (res["index"] == idx) & (res.horizon == h)]
                    vals.append(r[metric].iloc[0] if len(r) else 0)
                ax.bar(x + (k - 0.5) * w, vals, w, label=idx)
            ref = 1/3 if metric == "macro_f1" else 0.0
            ax.axhline(ref, color="red", ls="--", lw=1)
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
            ax.set_title(f"{'3-class macro-F1' if metric=='macro_f1' else 'IC'} — h={h}")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle("EXP-R 공정 비교 — 동일 top-30 입력 (점선: macro-F1 0.333 / IC 0)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = RD / "figures" / "fair_topn.png"
    fig.savefig(out, dpi=150); plt.close(fig); print("저장:", out)


if __name__ == "__main__":
    run()
