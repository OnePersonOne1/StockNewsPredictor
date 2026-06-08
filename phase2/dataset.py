"""
dataset.py — PyTorch Dataset (Phase 2)

dataset_final.parquet 을 읽어 split별로 (토큰화 헤드라인, 라벨, 메타) 를 제공.

핵심:
  - split ∈ {train, val, test} : 거래일 기준 temporal 분할 (parquet 의 split 컬럼)
  - sigma_h 는 train split 의 ret_h 에서만 계산 → val/test 는 동일 sigma 적용
    (sigma_dict 미전달 + split='train' 이면 계산해 self.sigma_dict 로 노출,
     val/test 는 train 에서 얻은 sigma_dict 를 반드시 전달받아야 함)
  - 라벨은 ret_h + sigma 로 그 자리에서 부여(build_labels 의 함수 재사용)하여
    "train-only sigma" 데이터흐름을 Phase 2 안에서도 명시적으로 보장
  - max_headlines: 최신순 상위 N개만 사용(원본은 일평균 ~430건 → GPU 초과)
  - 실제 헤드라인 수 < max_headlines 인 경우 headline_mask 로 패딩 슬롯 표시
    (모델의 attention pooling 이 패딩을 softmax 에서 제외)
"""
from __future__ import annotations
import sys
import pathlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phase1.config import (  # noqa: E402
    DATASET_FINAL, HORIZONS, MAX_HEADLINES, MAX_LENGTH,
    LABEL2IDX, INDEX_NAMES,
)
from phase1.build_labels import compute_sigma, apply_labels, assign_split  # noqa: E402

INDEX2ID = {name: i for i, name in enumerate(INDEX_NAMES)}


class NewsHeadlineDataset(Dataset):
    def __init__(self, parquet_path, split, tokenizer,
                 max_headlines: int = MAX_HEADLINES,
                 max_length: int = MAX_LENGTH,
                 sigma_dict: dict[int, float] | None = None):
        parquet_path = pathlib.Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"데이터셋 없음: {parquet_path}")
        if split not in ("train", "val", "test"):
            raise ValueError(f"split 은 train/val/test 중 하나: {split}")

        df = pd.read_parquet(parquet_path)
        if "split" not in df.columns:
            df["split"] = assign_split(df)

        # --- sigma_h: train 에서만 계산 ---
        if sigma_dict is None:
            if split != "train":
                raise ValueError(
                    "val/test 는 train 에서 계산한 sigma_dict 를 전달해야 합니다."
                )
            sigma_dict = compute_sigma(df[df["split"] == "train"])
        self.sigma_dict = sigma_dict

        # --- 라벨 부여 (train-only sigma) 후 split 추출 ---
        df = apply_labels(df, sigma_dict)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"split={split} 이 비어 있습니다.")

        self.tokenizer = tokenizer
        self.max_headlines = max_headlines
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # 최신순 상위 max_headlines (parquet 에 이미 최신순 저장됨)
        headlines = [str(h) for h in list(row["headlines"])[:self.max_headlines]]
        n_actual = len(headlines)

        enc = self.tokenizer(
            headlines,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]            # [n_actual, max_length]
        attention_mask = enc["attention_mask"]  # [n_actual, max_length]

        # 헤드라인 차원을 max_headlines 로 패딩
        pad_n = self.max_headlines - n_actual
        if pad_n > 0:
            pad_ids = torch.zeros((pad_n, self.max_length), dtype=input_ids.dtype)
            pad_att = torch.zeros((pad_n, self.max_length), dtype=attention_mask.dtype)
            input_ids = torch.cat([input_ids, pad_ids], dim=0)
            attention_mask = torch.cat([attention_mask, pad_att], dim=0)

        headline_mask = torch.zeros(self.max_headlines, dtype=torch.bool)
        headline_mask[:n_actual] = True  # True=실제 헤드라인, False=패딩

        tokenized = {
            "input_ids": input_ids,             # [max_headlines, max_length]
            "attention_mask": attention_mask,   # [max_headlines, max_length]
            "headline_mask": headline_mask,     # [max_headlines]
            "index_id": torch.tensor(INDEX2ID[row["index_name"]], dtype=torch.long),
        }

        labels = {}
        for h in HORIZONS:
            v = row[f"label_h{h}"]
            labels[f"h{h}"] = torch.tensor(LABEL2IDX[int(v)], dtype=torch.long)

        meta = {
            "date": str(pd.Timestamp(row["date"]).date()),
            "index_name": str(row["index_name"]),
            "n_headlines_actual": int(n_actual),
        }
        return tokenized, labels, meta


def make_splits(parquet_path=DATASET_FINAL, tokenizer=None,
                max_headlines: int = MAX_HEADLINES, max_length: int = MAX_LENGTH):
    """train/val/test 세 Dataset 을 일관된 sigma_dict 로 생성하는 헬퍼."""
    train = NewsHeadlineDataset(parquet_path, "train", tokenizer,
                                max_headlines, max_length, sigma_dict=None)
    sigma = train.sigma_dict
    val = NewsHeadlineDataset(parquet_path, "val", tokenizer,
                              max_headlines, max_length, sigma_dict=sigma)
    test = NewsHeadlineDataset(parquet_path, "test", tokenizer,
                               max_headlines, max_length, sigma_dict=sigma)
    return train, val, test, sigma


if __name__ == "__main__":
    # smoke test: 작은 batch 1개 정상 출력 확인
    from transformers import AutoTokenizer
    from phase1.config import ENCODER_NAME

    tok = AutoTokenizer.from_pretrained(ENCODER_NAME)
    ds = NewsHeadlineDataset(DATASET_FINAL, "train", tok,
                             max_headlines=5, max_length=32)
    print(f"train 크기: {len(ds)}, sigma_dict: "
          f"{ {k: round(v,4) for k,v in ds.sigma_dict.items()} }")
    tk, lb, mt = ds[0]
    print("input_ids:", tuple(tk["input_ids"].shape),
          "attention_mask:", tuple(tk["attention_mask"].shape),
          "headline_mask sum:", int(tk["headline_mask"].sum()),
          "index_id:", int(tk["index_id"]))
    print("labels:", {k: int(v) for k, v in lb.items()})
    print("meta:", mt)

    # default collate 로 batch 구성 확인
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=2)
    btk, blb, bmt = next(iter(dl))
    print("batched input_ids:", tuple(btk["input_ids"].shape),
          "batched h1 labels:", blb["h1"].tolist(),
          "batched index_name:", bmt["index_name"])
