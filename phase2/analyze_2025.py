"""
analyze_2025.py — EXP-W: 2025 시장 예외성 분석 (삼성전자)

train=2021-2023 으로 학습한 모델(EXP-V)이 2025 에서 약한 이유를 검증:
'2025 가 유례없는 상승장(분포 이동)'인가? 연도별로
  (a) 삼성전자 연수익·상승일 비율·라벨 분포
  (b) 모델 방향 적중률 vs '항상 up' 기준(상승장 베이스라인)
  (c) 모델 up-예측 비율 vs 실제 up 비율 (드리프트 추종 실패)
를 비교. 모델은 binary(results_samsung_ordtime_bin) 사용.
산출: results_samsung_ordtime/exp_w_2025.{csv,md}, figures/exp_w_2025.png
"""
from __future__ import annotations
import sys
import pathlib

import numpy as np
import pandas as pd
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from phase1.config import (DATASET_FINAL, ENCODER_NAME, HORIZONS, MAX_LENGTH,  # noqa
                           BEST_CKPT, RESULTS_DIR, PHASE2_DIR)
from phase2.dataset import make_splits  # noqa: E402
from phase2.model import HeadlineAttentionModel  # noqa: E402

# 현재 프로필의 binary 체크포인트(BINARY=1 환경에서 실행 → samsung 또는 samsung_cv).
BIN_CKPT = BEST_CKPT


def _font():
    import matplotlib
    for fam in ("Noto Sans CJK KR", "Noto Sans CJK JP"):
        if any(f.name == fam for f in matplotlib.font_manager.fontManager.ttflist):
            matplotlib.rcParams["font.family"] = fam; break
    matplotlib.rcParams["axes.unicode_minus"] = False


def yearly_stats():
    px = pd.read_parquet(PHASE2_DIR.parent / "phase1/data/processed/prices_samsung.parquet")
    px["year"] = px["date"].dt.year
    rows = []
    d = pd.read_parquet(DATASET_FINAL); d["year"] = pd.to_datetime(d["date"]).dt.year
    for y in range(2021, 2026):
        g = px[px.year == y]; gd = d[d.year == y]
        if not len(g):
            continue
        rows.append({"year": y,
                     "annual_return": g["close"].iloc[-1] / g["close"].iloc[0] - 1,
                     "up_h1": float((gd["ret_h1"] > 0).mean()),
                     "up_h5": float((gd["ret_h5"] > 0).mean()),
                     "uplabel_h5": float((gd["label_h5"] == 1).mean())})
    return pd.DataFrame(rows)


@torch.no_grad()
def model_by_year(mh=64):
    """binary 모델(train2021-23)로 2024·2025 연도별 방향 적중률 + 항상up 기준.
    반드시 BINARY=1 환경에서 실행(모델 N_CLASSES=2)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(BIN_CKPT, map_location=device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ENCODER_NAME)
    model = HeadlineAttentionModel(ENCODER_NAME, HORIZONS).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    tr_ds, val_ds, test_ds, _ = make_splits(DATASET_FINAL, tok, mh, MAX_LENGTH)

    recs = []
    for split, ds in (("val", val_ds), ("test", test_ds)):
        loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)
        df = ds.df.reset_index(drop=True)
        df["year"] = pd.to_datetime(df["date"]).dt.year
        pr = {h: [] for h in HORIZONS}
        for tokenized, _, _ in loader:
            tk = {k: v.to(device) for k, v in tokenized.items()}
            out = model(**tk)
            for h in HORIZONS:
                pr[h].extend(out["logits"][f"h{h}"].argmax(-1).cpu().numpy().tolist())
        for h in (1, 5):
            preds = np.array(pr[h]); ret = df[f"ret_h{h}"].to_numpy()
            for year in sorted(df["year"].unique()):
                ym = (df["year"].to_numpy() == year) & (~np.isnan(ret))
                if ym.sum() < 5:
                    continue
                pred_up = preds[ym]; actual_up = (ret[ym] > 0).astype(int)
                recs.append({"split": split, "year": int(year), "horizon": h,
                             "n": int(ym.sum()),
                             "model_acc": float((pred_up == actual_up).mean()),
                             "always_up_acc": float(actual_up.mean()),
                             "model_up_rate": float(pred_up.mean()),
                             "actual_up_rate": float(actual_up.mean())})
    return pd.DataFrame(recs)


def run():
    ys = yearly_stats()
    mb = model_by_year()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ys.to_csv(RESULTS_DIR / "exp_w_yearly.csv", index=False, encoding="utf-8-sig")
    mb.to_csv(RESULTS_DIR / "exp_w_model_by_year.csv", index=False, encoding="utf-8-sig")

    md = ["# EXP-W: 2025 시장 예외성 (삼성전자, train=2021–2023)", "",
          "## 연도별 삼성전자", "", "| 연도 | 연수익 | ret_h1>0 | ret_h5>0 | up라벨(h5) |",
          "|---|---|---|---|---|"]
    for _, r in ys.iterrows():
        md.append(f"| {int(r.year)} | {r.annual_return:+.1%} | {r.up_h1:.0%} | "
                  f"{r.up_h5:.0%} | {r.uplabel_h5:.0%} |")
    md += ["", "## 모델(binary, train2021–23) 연도별 방향 적중률 vs '항상 up'", "",
           "| 연도 | h | n | 모델적중 | 항상up적중 | 모델up예측율 | 실제up율 |",
           "|---|---|---|---|---|---|---|"]
    for _, r in mb.iterrows():
        md.append(f"| {int(r.year)} | {int(r.horizon)} | {int(r.n)} | {r.model_acc:.3f} | "
                  f"{r.always_up_acc:.3f} | {r.model_up_rate:.0%} | {r.actual_up_rate:.0%} |")
    (RESULTS_DIR / "exp_w_2025.md").write_text("\n".join(md), encoding="utf-8")

    _plot(ys, mb)
    print(ys.to_string(index=False)); print(); print(mb.to_string(index=False))
    print("\n저장:", RESULTS_DIR / "exp_w_2025.md")


def _plot(ys, mb):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _font()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    c = ["tab:red" if v < 0 else "tab:blue" for v in ys["annual_return"]]
    a1.bar(ys["year"].astype(str), ys["annual_return"] * 100, color=c)
    a1.axhline(0, color="black", lw=0.8)
    a1.set_title("삼성전자 연수익률(%) — 2025 +124% 이례적 상승")
    a1.set_ylabel("연수익률(%)")
    for x, v in zip(ys["year"].astype(str), ys["annual_return"] * 100):
        a1.text(x, v + (3 if v >= 0 else -8), f"{v:+.0f}%", ha="center", fontsize=8)
    # 모델 up예측 vs 실제 up (h5)
    m5 = mb[mb.horizon == 5]
    x = np.arange(len(m5)); w = 0.38
    a2.bar(x - w/2, m5["model_up_rate"], w, label="모델 up예측율")
    a2.bar(x + w/2, m5["actual_up_rate"], w, label="실제 up율")
    a2.axhline(0.5, color="gray", ls=":", lw=1)
    a2.set_xticks(x); a2.set_xticklabels([f"{int(y)}\nh5" for y in m5["year"]])
    a2.set_title("모델은 균형 예측, 2025 실제는 up 쏠림 → 드리프트 추종 실패")
    a2.legend(fontsize=8); a2.set_ylim(0, 0.75)
    fig.tight_layout()
    out = RESULTS_DIR / "figures" / "exp_w_2025.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150); plt.close(fig); print("저장:", out)


if __name__ == "__main__":
    run()
