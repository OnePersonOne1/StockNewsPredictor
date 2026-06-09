"""
wordcount_baseline.py — 비-ML 단어 개수(렉시콘) 기저 모델 (Phase 2 비교용)

ML 이전의 가장 단순한 규칙 기반 예측:
  - 하루(거래일)의 모든 헤드라인을 합쳐, 직접 만든 '금융 방향성 시드 사전'으로
    긍정 단어 출현수 − 부정 단어 출현수 = net score 를 센다.
  - net score 가 높을수록 '상승(+1)', 낮을수록 '하락(-1)' 이라는 가설을 그대로 규칙화.
  - 3-class 경계는 **train 라벨 분포에 맞춘 분위수(quantile matching)** 로 정한다:
        τ_lo = quantile(train_score, p_down),  τ_hi = quantile(train_score, p_down+p_flat)
    → test 는 동일 경계 적용 (학습 파라미터 없음, base rate 만 train 에서 차용).

위치: TF-IDF+LogReg(고전 ML) 보다 아래의 lower-bound. 주 모델/라벨/split 불변.

한계(보고서에 명시): 시드 사전은 소규모 수작업이며 형태소 분석 없이 부분문자열
매칭이라 '상승세/상승했다'는 잡지만 부정어 결합('상승하지 못')은 구분 못 한다.
"""
from __future__ import annotations
import sys
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase1.config import DATASET_FINAL, HORIZONS, INDEX_NAMES, RESULTS_DIR  # noqa: E402

EVAL_SPLIT = "test"

# --- 금융 방향성 시드 사전 (수작업, 부분문자열 매칭) -------------------------
POS_SEEDS = [
    "상승", "급등", "오름", "강세", "호조", "개선", "회복", "반등", "흑자",
    "최고", "신고가", "돌파", "호황", "증가", "성장", "확대", "기대", "호재",
    "순항", "선전", "활기", "낙관", "상향", "플러스", "수혜", "훈풍", "청신호",
    "랠리", "급증", "사상최대",
]
NEG_SEEDS = [
    "하락", "급락", "내림", "약세", "부진", "악화", "위기", "침체", "적자",
    "최저", "신저가", "추락", "폭락", "둔화", "감소", "위축", "우려", "악재",
    "불안", "공포", "경고", "비관", "하향", "마이너스", "충격", "리스크",
    "한파", "적신호", "패닉", "쇼크",
]


def _day_doc(headlines) -> str:
    """행의 헤드라인 배열 → 공백 join 단일 문자열."""
    return " ".join(str(h) for h in headlines)


def _net_score(doc: str) -> int:
    """긍정 시드 출현수 − 부정 시드 출현수."""
    pos = sum(doc.count(w) for w in POS_SEEDS)
    neg = sum(doc.count(w) for w in NEG_SEEDS)
    return pos - neg


def _fit_thresholds(scores: np.ndarray, labels: np.ndarray):
    """train 라벨 분포에 맞춘 분위수 경계 (τ_lo, τ_hi)."""
    n = len(labels)
    p_down = float(np.mean(labels == -1))
    p_flat = float(np.mean(labels == 0))
    tau_lo = float(np.quantile(scores, p_down)) if p_down > 0 else float(scores.min() - 1)
    tau_hi = (float(np.quantile(scores, p_down + p_flat))
              if (p_down + p_flat) < 1 else float(scores.max() + 1))
    return tau_lo, tau_hi


def _predict(scores: np.ndarray, tau_lo: float, tau_hi: float) -> np.ndarray:
    pred = np.zeros_like(scores, dtype=int)        # 기본 보합(0)
    pred[scores > tau_hi] = 1
    pred[scores < tau_lo] = -1
    return pred


def run() -> pd.DataFrame:
    if not DATASET_FINAL.exists():
        raise FileNotFoundError(f"{DATASET_FINAL} 없음. Phase 1 먼저 실행.")
    df = pd.read_parquet(DATASET_FINAL).copy()
    df["score"] = df["headlines"].map(lambda hs: _net_score(_day_doc(hs)))

    # 사전 커버리지 (해석용): net score 가 0 이 아닌 거래일 비율
    cover = float(np.mean(df["score"] != 0))

    records = []
    for index_name in INDEX_NAMES:
        sub = df[df["index_name"] == index_name]
        tr = sub[sub["split"] == "train"]
        te = sub[sub["split"] == EVAL_SPLIT]
        s_tr = tr["score"].to_numpy()
        s_te = te["score"].to_numpy()

        for h in HORIZONS:
            ytr = tr[f"label_h{h}"].astype(int).to_numpy()
            yte = te[f"label_h{h}"].astype(int).to_numpy()

            tau_lo, tau_hi = _fit_thresholds(s_tr, ytr)
            pred = _predict(s_te, tau_lo, tau_hi)

            acc = accuracy_score(yte, pred)
            mf1 = f1_score(yte, pred, average="macro", labels=[-1, 0, 1],
                           zero_division=0)
            records.append({
                "index": index_name, "horizon": h, "n_test": len(yte),
                "accuracy": acc, "macro_f1": mf1,
                "tau_lo": tau_lo, "tau_hi": tau_hi,
            })

    res = pd.DataFrame.from_records(records)
    _report(res, cover)
    return res


def _report(res: pd.DataFrame, cover: float) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "wordcount_metrics.csv"
    res.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("=" * 64)
    print("단어 개수(렉시콘) 비-ML 기저 — eval split = %s" % EVAL_SPLIT)
    print(f"시드: 긍정 {len(POS_SEEDS)}개 / 부정 {len(NEG_SEEDS)}개 | "
          f"net≠0 거래일 비율 = {cover:.1%} | 무작위 macro-F1 = 0.333")
    print("=" * 64)
    print("\n[accuracy]  (열=horizon)")
    print(res.pivot(index="index", columns="horizon", values="accuracy").round(3).to_string())
    print("\n[macro-F1]")
    print(res.pivot(index="index", columns="horizon", values="macro_f1").round(3).to_string())

    md = ["| index | horizon | n_test | accuracy | macro-F1 | τ_lo | τ_hi |",
          "|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        md.append("| %s | %d | %d | %.3f | %.3f | %.1f | %.1f |" % (
            r["index"], r["horizon"], r["n_test"], r["accuracy"], r["macro_f1"],
            r["tau_lo"], r["tau_hi"]))
    (RESULTS_DIR / "wordcount_metrics.md").write_text("\n".join(md), encoding="utf-8")

    print("\n저장:", csv_path)
    print("저장:", RESULTS_DIR / "wordcount_metrics.md")


if __name__ == "__main__":
    run()
