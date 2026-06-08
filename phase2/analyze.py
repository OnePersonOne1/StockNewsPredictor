"""
analyze.py — Attention 가중치 해석 (Phase 2)

학습된 모델로 test set 각 (날짜, 지수)에 대해 attention 가중치 α_i 를 추출하고,
가중치 상위 5개 헤드라인을 뽑아 "어떤 종류의 헤드라인이 높은 가중치를 받는가"
분석에 사용한다.

출력: results/top_attention_headlines.csv
  컬럼: date, index_name, rank, attention_weight, headline

attention 은 horizon 공통(단일 pooling)이므로 (날짜,지수)당 하나의 α 벡터.
"""
from __future__ import annotations
import sys
import pathlib
import argparse

import numpy as np
import pandas as pd
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase1.config import (  # noqa: E402
    DATASET_FINAL, ENCODER_NAME, HORIZONS, MAX_LENGTH,
    BEST_CKPT, RESULTS_DIR, TOP_ATTN_CSV,
)
from phase2.dataset import NewsHeadlineDataset, make_splits  # noqa: E402
from phase2.model import HeadlineAttentionModel  # noqa: E402


@torch.no_grad()
def run(ckpt_path=BEST_CKPT, top_k: int = 5, date_filter: str | None = None):
    ckpt_path = pathlib.Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"체크포인트 없음: {ckpt_path} (train.py 먼저 실행)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    max_headlines = ckpt.get("config", {}).get("max_headlines")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)

    # test dataset (sigma 는 train 에서 — make_splits 가 일관 처리)
    _, _, test_ds, _ = make_splits(DATASET_FINAL, tokenizer, max_headlines, MAX_LENGTH)

    model = HeadlineAttentionModel(ENCODER_NAME, HORIZONS).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    records = []
    for i in range(len(test_ds)):
        tokenized, _, meta = test_ds[i]
        if date_filter and meta["date"] != date_filter:
            continue
        # batch=1 forward
        tk = {k: v.unsqueeze(0).to(device) for k, v in tokenized.items()}
        out = model(**tk)
        alpha = out["attention"][0].cpu().numpy()  # [M]

        # 해당 행의 실제 헤드라인(최신순 상위 max_headlines)
        headlines = [str(h) for h in
                     list(test_ds.df.iloc[i]["headlines"])[:len(alpha)]]
        order = np.argsort(-alpha)[:top_k]
        for rank, j in enumerate(order, 1):
            if j >= len(headlines):
                continue
            records.append({
                "date": meta["date"], "index_name": meta["index_name"],
                "rank": rank, "attention_weight": float(alpha[j]),
                "headline": headlines[j],
            })

    df = pd.DataFrame.from_records(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TOP_ATTN_CSV, index=False, encoding="utf-8-sig")

    print(f"top-{top_k} attention 헤드라인 추출: {len(df)} 행 → {TOP_ATTN_CSV}")
    # 미리보기: 처음 몇 (날짜,지수)
    for (d, idx), g in df.groupby(["date", "index_name"]):
        print(f"\n[{d} {idx}]")
        for _, r in g.iterrows():
            print(f"  #{r['rank']} ({r['attention_weight']:.3f}) {r['headline']}")
        if d != df["date"].iloc[-1]:
            break  # 미리보기는 한 묶음만
    return df


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--date", type=str, default=None,
                   help="특정 날짜만 (YYYY-MM-DD). 미지정 시 test 전체")
    return p


if __name__ == "__main__":
    a = build_argparser().parse_args()
    run(top_k=a.top_k, date_filter=a.date)
