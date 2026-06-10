"""
signal_test.py — EXP-H: '정말 신호가 없는가' 엄밀 검증 (Phase 2)

3-class macro-F1 은 예측 불가한 flat 클래스 때문에 약신호를 가릴 수 있다. 여기서는
연속 점수(signed confidence)를 만들어 세 각도로 신호 유무를 검정한다:

  1) 이진 up/down  : true≠0 행만, 상승/하락 정확도 + binomial p(vs 0.5) + AUC
  2) IC            : Spearman corr( score, 실제 forward return ret_h ) + p값
  3) 롱숏 백테스트  : position=sign(score), 평균 수익 = mean(sign(score)·ret_h),
                     순열검정 p(부호 셔플), 적중률(hit rate)

모델별 연속 점수:
  - RoBERTa / TF-IDF : score = P(up) − P(down)
  - wordcount        : net = 긍정−부정 단어수 (horizon 공통)

대상: 현재 EXP_PROFILE 의 test split(다년이면 2024 전체). h=1,5 가 주 결과.
산출: results*/signal_metrics.csv, signal_report.md
사용: EXP_PROFILE=multiyear python phase2/signal_test.py
"""
from __future__ import annotations
import sys
import pathlib

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase1.config import (  # noqa: E402
    DATASET_FINAL, ENCODER_NAME, HORIZONS, INDEX_NAMES, MAX_LENGTH, SEED,
    BEST_CKPT, RESULTS_DIR,
)
from phase1.build_labels import compute_sigma, apply_labels  # noqa: E402
from phase2.dataset import make_splits  # noqa: E402
from phase2.model import HeadlineAttentionModel  # noqa: E402
from phase2.wordcount_baseline import _net_score, _day_doc  # noqa: E402

IDX = {-1: 0, 0: 1, 1: 2}


# ----- 모델별 (index,h) → DataFrame[score, ret, true_lbl] -------------------
@torch.no_grad()
def roberta_scores(ckpt_path=BEST_CKPT):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    mh = ckpt.get("config", {}).get("max_headlines")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ENCODER_NAME)
    _, _, test_ds, _ = make_splits(DATASET_FINAL, tok, mh, MAX_LENGTH)
    model = HeadlineAttentionModel(ENCODER_NAME, HORIZONS).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    rows = []
    loader = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)
    df = test_ds.df.reset_index(drop=True)
    pos = 0
    for tokenized, _, meta in loader:
        tk = {k: v.to(device) for k, v in tokenized.items()}
        out = model(**tk)
        B = tk["index_id"].size(0)
        for h in HORIZONS:
            p = torch.softmax(out["logits"][f"h{h}"], -1).cpu().numpy()  # [B,3]
            for b in range(B):
                r = df.iloc[pos + b]
                rows.append({"index": r["index_name"], "horizon": h,
                             "score": float(p[b, -1] - p[b, 0]),  # up−down (2/3-class 공통)
                             "ret": float(r[f"ret_h{h}"]),
                             "true": int(r[f"label_h{h}"])})
        pos += B
    return pd.DataFrame(rows)


def tfidf_scores():
    df = pd.read_parquet(DATASET_FINAL)
    if "split" not in df.columns:
        from phase1.build_labels import assign_split
        df["split"] = assign_split(df)
    df = apply_labels(df, compute_sigma(df[df.split == "train"]))
    from phase1.config import BASELINE_TOPN
    _n = BASELINE_TOPN
    docs = lambda s: [" ".join(str(x) for x in (list(row)[:_n] if _n > 0 else row))
                      for row in s["headlines"]]
    rows = []
    for idx in INDEX_NAMES:
        sub = df[df.index_name == idx]
        tr, te = sub[sub.split == "train"], sub[sub.split == "test"]
        vec = TfidfVectorizer(min_df=2, max_features=20000)
        Xtr, Xte = vec.fit_transform(docs(tr)), vec.transform(docs(te))
        for h in HORIZONS:
            ytr = tr[f"label_h{h}"].astype(int).to_numpy()
            clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=SEED).fit(Xtr, ytr)
            classes = list(clf.classes_)
            proba = clf.predict_proba(Xte)
            p_up = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(te))
            p_dn = proba[:, classes.index(-1)] if -1 in classes else np.zeros(len(te))
            for i, (_, r) in enumerate(te.iterrows()):
                rows.append({"index": idx, "horizon": h,
                             "score": float(p_up[i] - p_dn[i]),
                             "ret": float(r[f"ret_h{h}"]),
                             "true": int(r[f"label_h{h}"])})
    return pd.DataFrame(rows)


def wordcount_scores():
    df = pd.read_parquet(DATASET_FINAL)
    if "split" not in df.columns:
        from phase1.build_labels import assign_split
        df["split"] = assign_split(df)
    df = apply_labels(df, compute_sigma(df[df.split == "train"]))
    te = df[df.split == "test"].copy()
    te["score"] = te["headlines"].map(lambda hs: float(_net_score(_day_doc(hs))))
    rows = []
    for _, r in te.iterrows():
        for h in HORIZONS:
            rows.append({"index": r["index_name"], "horizon": h,
                         "score": r["score"], "ret": float(r[f"ret_h{h}"]),
                         "true": int(r[f"label_h{h}"])})
    return pd.DataFrame(rows)


# ----- 검정 ---------------------------------------------------------------
def _perm_p(pos_ret, n=5000):
    """롱숏 평균수익의 순열검정(부호 무작위)."""
    obs = float(np.mean(pos_ret))
    rng = np.random.default_rng(SEED)
    signs = np.sign(pos_ret); mag = np.abs(pos_ret)
    cnt = 0
    for _ in range(n):
        s = rng.choice([-1, 1], size=len(mag))
        if abs(np.mean(s * mag)) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (n + 1)


def metrics_for(g):
    """(index,h) 한 셀의 신호 지표."""
    score = g["score"].to_numpy(); ret = g["ret"].to_numpy(); true = g["true"].to_numpy()
    ok = ~np.isnan(ret)
    score, ret, true = score[ok], ret[ok], true[ok]
    out = {"n": len(true)}
    # 1) 이진 up/down (true≠0)
    m = true != 0
    if m.sum() >= 5 and len(np.unique(true[m])) == 2:
        yb = (true[m] == 1).astype(int)
        pred = (score[m] > 0).astype(int)
        acc = float(np.mean(pred == yb))
        k = int(np.sum(pred == yb))
        out["bin_acc"] = acc
        out["bin_p"] = float(stats.binomtest(k, m.sum(), 0.5).pvalue)
        try:
            out["auc"] = float(roc_auc_score(yb, score[m]))
        except ValueError:
            out["auc"] = np.nan
    else:
        out["bin_acc"] = out["bin_p"] = out["auc"] = np.nan
    # 2) IC (Spearman score vs ret)
    if np.std(score) > 0:
        ic, icp = stats.spearmanr(score, ret)
        out["IC"] = float(ic); out["IC_p"] = float(icp)
    else:
        out["IC"] = out["IC_p"] = np.nan
    # 3) 롱숏 백테스트
    pos_ret = np.sign(score) * ret
    nz = np.sign(score) != 0
    if nz.sum() >= 5:
        mean_ret, pp = _perm_p(pos_ret[nz])
        out["ls_ret"] = mean_ret
        out["ls_p"] = pp
        hit = ret[nz] != 0
        out["hit"] = float(np.mean(np.sign(score[nz][hit]) == np.sign(ret[nz][hit]))) if hit.sum() else np.nan
    else:
        out["ls_ret"] = out["ls_p"] = out["hit"] = np.nan
    return out


def run():
    models = {"TF-IDF": tfidf_scores(), "wordcount": wordcount_scores(),
              "RoBERTa": roberta_scores()}
    recs = []
    for name, sdf in models.items():
        for (idx, h), g in sdf.groupby(["index", "horizon"]):
            recs.append({"model": name, "index": idx, "horizon": h, **metrics_for(g)})
    res = pd.DataFrame(recs)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS_DIR / "signal_metrics.csv", index=False, encoding="utf-8-sig")
    _report(res)
    return res


def _star(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def _report(res):
    md = ["# EXP-H 신호 검정 — 이진 up/down · IC · 롱숏 백테스트", "",
          "주 결과 h=1,5. 이진 random=0.5, IC=0, 롱숏 평균수익=0. *p<.10 **p<.05 ***p<.01.",
          "(test = 현재 프로필 test split.)", ""]
    for h in HORIZONS:
        md += [f"## h = {h}", "",
               "| model | index | 이진acc(p) | AUC | IC(p) | 롱숏수익(p) | hit |",
               "|---|---|---|---|---|---|---|"]
        sub = res[res.horizon == h]
        for _, r in sub.iterrows():
            ba = f"{r['bin_acc']:.3f}{_star(r['bin_p'])}" if pd.notna(r['bin_acc']) else "—"
            ic = f"{r['IC']:+.3f}{_star(r['IC_p'])}" if pd.notna(r['IC']) else "—"
            ls = f"{r['ls_ret']:+.4f}{_star(r['ls_p'])}" if pd.notna(r['ls_ret']) else "—"
            auc = f"{r['auc']:.3f}" if pd.notna(r['auc']) else "—"
            hit = f"{r['hit']:.3f}" if pd.notna(r['hit']) else "—"
            md.append(f"| {r['model']} | {r['index']} | {ba} | {auc} | {ic} | {ls} | {hit} |")
        md.append("")
    n_sig = int(((res["bin_p"] < 0.05) | (res["IC_p"] < 0.05) | (res["ls_p"] < 0.05)).sum())
    md.append(f"**유의(p<.05) 셀: {n_sig} / {len(res)}** "
              f"(무작위면 우연히 ~{0.05*len(res):.0f}개 기대 — 다중비교 주의).")
    (RESULTS_DIR / "signal_report.md").write_text("\n".join(md), encoding="utf-8")

    print("=" * 70)
    print(f"EXP-H 신호 검정 — {RESULTS_DIR.name}")
    print("=" * 70)
    show = res[res.horizon.isin([1, 5])][
        ["model", "index", "horizon", "bin_acc", "bin_p", "IC", "IC_p", "ls_ret", "ls_p"]]
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}", "display.width", 200):
        print(show.to_string(index=False))
    print(f"\n유의(p<.05) 셀: {n_sig}/{len(res)} (무작위 기대 ~{0.05*len(res):.0f})")
    print("저장:", RESULTS_DIR / "signal_metrics.csv", "/ signal_report.md")


def roberta_only():
    """시드 스윕용: RoBERTa 만 빠르게 평가해 파싱 가능한 라인 출력."""
    sdf = roberta_scores()
    for (idx, h), g in sdf.groupby(["index", "horizon"]):
        if h not in (1, 5):
            continue
        m = metrics_for(g)
        print(f"SWEEP {idx} h{h} bin_acc={m['bin_acc']:.3f} IC={m['IC']:+.3f} "
              f"IC_p={m['IC_p']:.3f}")


if __name__ == "__main__":
    import os
    if os.environ.get("SIGNAL_ROBERTA_ONLY"):
        roberta_only()
    else:
        run()
