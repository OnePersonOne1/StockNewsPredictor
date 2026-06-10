"""
evaluate.py — test set 최종 평가 (Phase 2)

산출:
  - horizon × index 8셀 accuracy / macro-F1 표  (콘솔 + CSV + Markdown + LaTeX)
  - per-class precision/recall (어느 class 가 약한가)
  - confusion matrix (텍스트, 8개)
  - horizon decay 곡선 plot (figures/horizon_decay.png; baseline 오버레이)
  - 결과를 results/metrics.json 으로 저장

주의: h=252(및 사실상 h=21)는 test 표본이 매우 작고 거의 단일 클래스라
      통계적 신뢰도가 낮음 — 표/그림에 caveat 명시.
"""
from __future__ import annotations
import sys
import pathlib
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase1.config import (  # noqa: E402
    DATASET_FINAL, ENCODER_NAME, HORIZONS, INDEX_NAMES, MAX_LENGTH,
    BEST_CKPT, RESULTS_DIR, FIGURES_DIR, METRICS_JSON, CLASS_NAMES, IDX2LABEL,
    CLASS_IDX,
)
from phase2.dataset import make_splits  # noqa: E402
from phase2.model import HeadlineAttentionModel  # noqa: E402

# CLASS_IDX 는 config 에서(이진=[0,1], 3-class=[0,1,2])


@torch.no_grad()
def collect_predictions(model, loader, device):
    """test 전체에 대해 (index, horizon)별 pred/true 수집."""
    model.eval()
    rows = []  # (index_name, {h: pred}, {h: true})
    out_pred = {(idx, h): [] for idx in INDEX_NAMES for h in HORIZONS}
    out_true = {(idx, h): [] for idx in INDEX_NAMES for h in HORIZONS}
    for tokenized, labels, meta in loader:
        tk = {k: v.to(device) for k, v in tokenized.items()}
        out = model(**tk)
        idx_names = meta["index_name"]
        for h in HORIZONS:
            pred = out["logits"][f"h{h}"].argmax(-1).cpu().numpy()
            true = labels[f"h{h}"].numpy()
            for b, name in enumerate(idx_names):
                out_pred[(name, h)].append(int(pred[b]))
                out_true[(name, h)].append(int(true[b]))
    return out_pred, out_true


def evaluate_cells(out_pred, out_true):
    records, per_class, confusions = [], {}, {}
    for idx in INDEX_NAMES:
        for h in HORIZONS:
            t = np.array(out_true[(idx, h)]); p = np.array(out_pred[(idx, h)])
            m = t != -100                   # NaN(ignore) 라벨 제외(예: 2025 test 의 h252)
            t, p = t[m], p[m]
            acc = accuracy_score(t, p) if len(t) else float("nan")
            mf1 = f1_score(t, p, average="macro", labels=CLASS_IDX, zero_division=0)
            prec = precision_score(t, p, average=None, labels=CLASS_IDX, zero_division=0)
            rec = recall_score(t, p, average=None, labels=CLASS_IDX, zero_division=0)
            cm = confusion_matrix(t, p, labels=CLASS_IDX)
            records.append({"index": idx, "horizon": h, "n_test": len(t),
                            "accuracy": acc, "macro_f1": mf1})
            per_class[f"{idx}_h{h}"] = {
                CLASS_NAMES[c]: {"precision": float(prec[c]),
                                 "recall": float(rec[c])} for c in CLASS_IDX}
            confusions[f"{idx}_h{h}"] = cm.tolist()
    return pd.DataFrame.from_records(records), per_class, confusions


def _latex_table(res: pd.DataFrame, value: str, caption: str) -> str:
    piv = res.pivot(index="index", columns="horizon", values=value)
    cols = " & ".join([f"h={h}" for h in HORIZONS])
    lines = [r"\begin{table}[t]\centering",
             r"\caption{%s}" % caption,
             r"\begin{tabular}{l" + "r" * len(HORIZONS) + "}",
             r"\toprule",
             f"index & {cols} \\\\", r"\midrule"]
    for idx in piv.index:
        vals = " & ".join(f"{piv.loc[idx, h]:.3f}" for h in HORIZONS)
        lines.append(f"{idx} & {vals} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _decay_plot(res: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for idx in INDEX_NAMES:
        sub = res[res["index"] == idx].sort_values("horizon")
        ax.plot(sub["horizon"], sub["macro_f1"], marker="o", label=f"{idx} (model)")
    # baseline 오버레이 (있으면)
    base_csv = RESULTS_DIR / "baseline_metrics.csv"
    if base_csv.exists():
        base = pd.read_csv(base_csv)
        for idx in INDEX_NAMES:
            sub = base[base["index"] == idx].sort_values("horizon")
            ax.plot(sub["horizon"], sub["macro_f1"], marker="x", linestyle="--",
                    alpha=0.6, label=f"{idx} (TF-IDF)")
    ax.axhline(1/3, color="gray", ls=":", label="random (macro-F1≈1/3)")
    ax.set_xscale("log"); ax.set_xticks(HORIZONS)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("horizon (trading days)"); ax.set_ylabel("macro-F1")
    ax.set_title("Headline predictive power vs horizon")
    ax.legend(fontsize=8); fig.tight_layout()
    out = FIGURES_DIR / "horizon_decay.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def run(ckpt_path=BEST_CKPT):
    ckpt_path = pathlib.Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"체크포인트 없음: {ckpt_path} (train.py 먼저 실행)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    max_headlines = ckpt.get("config", {}).get("max_headlines")
    _, _, test_ds, _ = make_splits(DATASET_FINAL, tokenizer, max_headlines, MAX_LENGTH)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=8, shuffle=False)

    model = HeadlineAttentionModel(ENCODER_NAME, HORIZONS).to(device)
    model.load_state_dict(ckpt["model_state"])

    out_pred, out_true = collect_predictions(model, loader, device)
    res, per_class, confusions = evaluate_cells(out_pred, out_true)

    _report(res, per_class, confusions)
    return res


def _report(res, per_class, confusions):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS_DIR / "test_metrics.csv", index=False, encoding="utf-8-sig")

    print("=" * 66)
    print("최종 test 평가 — horizon × index (RoBERTa attention model)")
    print("=" * 66)
    print("\n[accuracy]")
    print(res.pivot(index="index", columns="horizon", values="accuracy").round(3).to_string())
    print("\n[macro-F1]")
    print(res.pivot(index="index", columns="horizon", values="macro_f1").round(3).to_string())

    # confusion matrix 텍스트
    print("\n[confusion matrix]  행=true, 열=pred, 순서=", CLASS_NAMES)
    for key, cm in confusions.items():
        print(f"  {key}: {cm}")

    # LaTeX
    tex = (_latex_table(res, "accuracy",
                        "Test accuracy by horizon and index (RoBERTa).") + "\n\n" +
           _latex_table(res, "macro_f1",
                        "Test macro-F1 by horizon and index (RoBERTa)."))
    (RESULTS_DIR / "test_metrics.tex").write_text(tex, encoding="utf-8")

    fig_path = _decay_plot(res)

    metrics = {
        "cells": res.to_dict(orient="records"),
        "per_class_precision_recall": per_class,
        "confusion_matrices": confusions,
        "note_h252": "h=252(및 h=21) test 표본 매우 작고 거의 단일 클래스 → 신뢰도 낮음",
    }
    METRICS_JSON.write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print("\n저장:", RESULTS_DIR / "test_metrics.csv")
    print("저장:", RESULTS_DIR / "test_metrics.tex")
    print("저장:", fig_path)
    print("저장:", METRICS_JSON)
    print("\n[caveat] h=252 및 h=21 은 test n≈20, 거의 단일 클래스 → 통계적 신뢰도 낮음.")


if __name__ == "__main__":
    run()
