"""
build_dataset.py — 헤드라인 + 가격 결합 데이터셋 구축 (Phase 1, 라벨 제외)

흐름:
  1) BIGKinds 원본 xlsx 6개에서 (일자, 제목) 추출 → 일별 헤드라인 묶음
  2) 각 뉴스 날짜 D 를 "D 이후 첫 거래일 T" 로 매핑 (T > D, 엄격 부등호)
     → look-ahead 봉쇄: 거래일 T 의 종가로 측정을 시작하되, 입력 헤드라인은
       전부 T 의 장 시작 이전(전 거래일·주말·휴일)에 발행된 것만 사용
  3) 각 거래일 T(2024) × 지수에 대해 close 와 forward return ret_h 계산
       ret_h = close[T_{k+h}] / close[T_k] - 1   (h 거래일 후)

라벨(label_h*)은 build_labels.py 에서 σ_h(train) 기반으로 부여한다.

출력 스키마 (라벨 제외):
  date, index_name, close, headlines(list[str]), n_headlines, ret_h{1,5,21,252}

주: 한 거래일의 헤드라인은 지수와 무관(동일 뉴스풀)하므로 KOSPI/KOSDAQ 행이
    같은 headlines 를 공유한다. 헤드라인은 '최신순'(일자 내림차순)으로 저장되어
    Phase 2 의 max_headlines=30 이 가장 최근 30개를 취하도록 한다.
"""
from __future__ import annotations
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from config import (  # noqa: E402  (직접 실행/패키지 실행 양쪽 지원은 아래 처리)
    ROOT,
    NEWS_GLOB,
    PRICES_PARQUET,
    DATASET_FINAL,
    INDEX_NAMES,
    HORIZONS,
)

# BIGKinds export 컬럼명 (한글 고정)
COL_DATE = "일자"
COL_TITLE = "제목"


def load_headlines() -> pd.DataFrame:
    """원본 xlsx 들을 읽어 (date, title) 롱포맷 DataFrame 반환."""
    files = sorted(glob.glob(str(ROOT / NEWS_GLOB)))
    if not files:
        raise FileNotFoundError(f"뉴스 원본 없음: {ROOT / NEWS_GLOB}")
    frames = []
    for f in files:
        d = pd.read_excel(f, usecols=[COL_DATE, COL_TITLE])
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={COL_DATE: "news_date", COL_TITLE: "title"})
    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"].str.len() > 0]
    # 일자: 20240101(int) → datetime
    df["news_date"] = pd.to_datetime(df["news_date"].astype(int).astype(str),
                                     format="%Y%m%d")
    print(f"헤드라인 로드: {len(df)} 건 "
          f"({df['news_date'].min().date()} ~ {df['news_date'].max().date()}), "
          f"파일 {len(files)}개")
    return df


def load_prices() -> pd.DataFrame:
    """가격 parquet 로드 (build_prices.py 산출물)."""
    if not PRICES_PARQUET.exists():
        raise FileNotFoundError(
            f"가격 parquet 없음: {PRICES_PARQUET}\n"
            f"먼저 `python phase1/build_prices.py` 를 실행하세요."
        )
    prices = pd.read_parquet(PRICES_PARQUET)
    prices["date"] = pd.to_datetime(prices["date"])
    return prices.sort_values(["index_name", "date"]).reset_index(drop=True)


def map_news_to_trading_day(news: pd.DataFrame,
                            trading_days: np.ndarray) -> pd.DataFrame:
    """
    각 뉴스 날짜 D → 첫 거래일 T (T > D, 엄격 부등호).
    trading_days 는 정렬된 거래일 배열(datetime64). searchsorted side='right'
    로 D 보다 '엄격히 큰' 첫 거래일의 인덱스를 찾는다.
    """
    td = np.sort(trading_days.astype("datetime64[ns]"))
    nd = news["news_date"].values.astype("datetime64[ns]")
    idx = np.searchsorted(td, nd, side="right")   # T > D 보장
    valid = idx < len(td)                          # 마지막 거래일 이후 뉴스는 버림
    out = news.loc[valid].copy()
    out["trading_day"] = td[idx[valid]]
    return out


def aggregate_headlines(mapped: pd.DataFrame) -> pd.DataFrame:
    """거래일별 헤드라인 리스트(최신순) + 개수."""
    # 최신순: 뉴스일자 내림차순(버킷 내에서 T 에 가까운 것이 먼저)
    mapped = mapped.sort_values(["trading_day", "news_date"],
                                ascending=[True, False])
    grp = mapped.groupby("trading_day")
    agg = pd.DataFrame({
        "headlines": grp["title"].apply(list),
        "n_headlines": grp["title"].size(),
    }).reset_index().rename(columns={"trading_day": "date"})
    return agg


def add_forward_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """지수별로 close 와 ret_h{HORIZONS} 부여 (h 거래일 후 수익률)."""
    out = []
    for name, g in prices.groupby("index_name"):
        g = g.sort_values("date").reset_index(drop=True).copy()
        for h in HORIZONS:
            g[f"ret_h{h}"] = g["close"].shift(-h) / g["close"] - 1.0
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build() -> pd.DataFrame:
    news = load_headlines()
    prices = load_prices()

    trading_days = prices["date"].unique()
    mapped = map_news_to_trading_day(news, trading_days)
    daily = aggregate_headlines(mapped)

    prices_ret = add_forward_returns(prices)

    # 2024 거래일만 행으로 사용 (헤드라인은 2024년만 존재)
    is_2024 = prices_ret["date"].dt.year == 2024
    rows = prices_ret.loc[is_2024, ["date", "index_name", "close",
                                    *[f"ret_h{h}" for h in HORIZONS]]]

    # 헤드라인 결합 (지수 무관 → date 로 join)
    df = rows.merge(daily, on="date", how="inner")

    # 정렬/검증
    df = df.sort_values(["date", "index_name"]).reset_index(drop=True)
    _sanity_checks(df)

    DATASET_FINAL.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_FINAL, index=False)
    print(f"저장: {DATASET_FINAL}  ({len(df)} 행, "
          f"지수별 {df.groupby('index_name').size().to_dict()})")
    return df


def _sanity_checks(df: pd.DataFrame) -> None:
    assert df["n_headlines"].min() > 0, "헤드라인 0개 거래일 존재"
    for name in INDEX_NAMES:
        assert (df["index_name"] == name).any(), f"{name} 행 없음"
    # forward return 결측 확인 (가격 연장 덕에 2024 전부 계산되어야 함)
    for h in HORIZONS:
        n_nan = df[f"ret_h{h}"].isna().sum()
        if n_nan:
            print(f"  주의: ret_h{h} 결측 {n_nan}행 (가격 시계열 말단 부족)")


if __name__ == "__main__":
    out = build()
    # smoke 출력: 첫 행 요약
    r = out.iloc[0]
    print("\n[sample] date=%s index=%s close=%.2f n_headlines=%d" %
          (r["date"].date(), r["index_name"], r["close"], r["n_headlines"]))
    print("  ret:", {f"h{h}": round(float(r[f'ret_h{h}']), 4) for h in HORIZONS})
    print("  headline[0]:", out.iloc[0]["headlines"][0])
