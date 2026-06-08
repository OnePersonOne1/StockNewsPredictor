"""
baseline.py — TF-IDF + Logistic Regression 기저 모델 (Phase 2)

주 모델(RoBERTa)의 lower bound 를 잡기 위한 단순 기저:
  - 거래일별 헤드라인을 공백으로 join → 하나의 document
  - 4 horizon × 2 index = 8개 독립 모델
  - TF-IDF(train fit) → LogisticRegression
  - 출력: 8셀 accuracy / macro-F1 표 (콘솔 + CSV + Markdown + LaTeX)

라벨/σ_h 는 dataset_final.parquet 에 이미 부여돼 있고(build_labels), train 에서
계산된 sigma 로 만들어졌으므로 baseline 도 동일 라벨을 사용한다.
무작위 3-class 기저율은 33.3%, 다수 클래스 기저율도 함께 보고한다.
"""
from __future__ import annotations
import sys
import pathlib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase1.config import DATASET_FINAL, HORIZONS, INDEX_NAMES, SEED, RESULTS_DIR  # noqa: E402

EVAL_SPLIT = "test"   # 기저 성능은 test 에서 보고 (train fit)


def _docs(df: pd.DataFrame) -> list[str]:
    """행별 헤드라인 리스트 → 공백 join 단일 document."""
    return [" ".join(str(h) for h in row) for row in df["headlines"]]


def run() -> pd.DataFrame:
    if not DATASET_FINAL.exists():
        raise FileNotFoundError(f"{DATASET_FINAL} 없음. Phase 1 먼저 실행.")
    df = pd.read_parquet(DATASET_FINAL)

    records = []
    for index_name in INDEX_NAMES:
        sub = df[df["index_name"] == index_name]
        tr = sub[sub["split"] == "train"]
        te = sub[sub["split"] == EVAL_SPLIT]

        # TF-IDF: train 에서만 vocabulary 학습
        vec = TfidfVectorizer(min_df=2, max_features=20000, ngram_range=(1, 1))
        Xtr = vec.fit_transform(_docs(tr))
        Xte = vec.transform(_docs(te))

        for h in HORIZONS:
            ytr = tr[f"label_h{h}"].astype(int).to_numpy()
            yte = te[f"label_h{h}"].astype(int).to_numpy()

            clf = LogisticRegression(max_iter=2000, C=1.0,
                                     class_weight="balanced",
                                     random_state=SEED)
            clf.fit(Xtr, ytr)
            pred = clf.predict(Xte)

            acc = accuracy_score(yte, pred)
            mf1 = f1_score(yte, pred, average="macro", labels=[-1, 0, 1],
                           zero_division=0)
            # 다수 클래스 기저율 (train 의 최빈 클래스로 test 예측)
            majority = tr[f"label_h{h}"].astype(int).mode().iloc[0]
            maj_acc = accuracy_score(yte, np.full_like(yte, majority))

            records.append({
                "index": index_name, "horizon": h, "n_test": len(yte),
                "accuracy": acc, "macro_f1": mf1, "majority_acc": maj_acc,
            })

    res = pd.DataFrame.from_records(records)
    _report(res)
    return res


def _report(res: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "baseline_metrics.csv"
    res.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 콘솔 표
    print("=" * 64)
    print("TF-IDF + LogReg 기저 성능 (eval split = %s)" % EVAL_SPLIT)
    print("무작위 3-class 기저율 = 0.333")
    print("=" * 64)
    pivot_acc = res.pivot(index="index", columns="horizon", values="accuracy")
    pivot_f1 = res.pivot(index="index", columns="horizon", values="macro_f1")
    print("\n[accuracy]  (열=horizon)")
    print(pivot_acc.round(3).to_string())
    print("\n[macro-F1]")
    print(pivot_f1.round(3).to_string())

    # Markdown 표 (보고서 붙여넣기용)
    md = ["| index | horizon | n_test | accuracy | macro-F1 | majority_acc |",
          "|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        md.append("| %s | %d | %d | %.3f | %.3f | %.3f |" % (
            r["index"], r["horizon"], r["n_test"],
            r["accuracy"], r["macro_f1"], r["majority_acc"]))
    md_path = RESULTS_DIR / "baseline_metrics.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print("\n저장:", csv_path)
    print("저장:", md_path)
    print("\n주의: h=252 는 test n=%d (지수당 ~20) 로 표본이 매우 작아 "
          "신뢰도 낮음." % int(res[res.horizon == 252]["n_test"].iloc[0]))


if __name__ == "__main__":
    np.random.seed(SEED)
    run()
