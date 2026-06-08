"""
build_prices.py — KOSPI/KOSDAQ 일별 종가 시계열 구축 (Phase 1)

업로드된 2024년 CSV(KRX_download.py 산출물)에 더해, forward return 계산에
필요한 미래 구간(2025~2026 상반기) 종가를 pykrx로 연장 다운로드한다.

왜 연장이 필요한가:
  test split = 2024-12. 이 날짜들의 ret_h5/h21/h252 는 각각 2025-01, 2025-02,
  2025-12 무렵의 종가가 있어야 계산된다. 2024년 종가만으로는 테스트셋의 다중
  horizon 라벨이 전부 결측이 되어 평가가 불가능하다.

출력: phase1/data/processed/prices.parquet
  컬럼: date(datetime64), index_name('KOSPI'|'KOSDAQ'), close(float)

주의: KRX 지수 API는 로그인 자격증명(KRX_ID/KRX_PW 환경변수)을 요구한다.
      KRX_download.py 와 동일한 자격증명을 사용한다.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import pandas as pd

# --- 경로 / 상수 -----------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]          # 프로젝트 루트
RAW_2024 = {
    "KOSPI": ROOT / "kospi_2024.csv",
    "KOSDAQ": ROOT / "kosdaq_2024.csv",
}
TICKER = {"KOSPI": "1001", "KOSDAQ": "2001"}
EXTEND_START = "20250101"   # 2024년 다음 거래일부터
EXTEND_END = "20260606"     # h=252 (≈1거래년) 커버에 충분한 버퍼
OUT_PATH = ROOT / "phase1" / "data" / "processed" / "prices.parquet"


def _load_2024(index_name: str) -> pd.DataFrame:
    """업로드된 2024 CSV에서 (date, close) 추출."""
    path = RAW_2024[index_name]
    if not path.exists():
        raise FileNotFoundError(f"2024 가격 CSV 없음: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={"날짜": "date", "종가": "close"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "close"]].copy()


def _download_extension(index_name: str) -> pd.DataFrame:
    """pykrx로 2025~2026 상반기 종가 다운로드. 실패 시 명확히 에러."""
    if not (os.environ.get("KRX_ID") and os.environ.get("KRX_PW")):
        raise RuntimeError(
            "KRX_ID / KRX_PW 환경변수가 필요합니다 (KRX_download.py 참조). "
            "예: KRX_ID=... KRX_PW=... python phase1/build_prices.py"
        )
    from pykrx import stock
    import pykrx.stock.stock_api as api

    # pykrx 1.2.7 의 지수명 부가조회가 간헐적으로 실패 → 우회(데이터엔 무관)
    api.get_index_ticker_name = lambda t: index_name

    df = stock.get_index_ohlcv_by_date(EXTEND_START, EXTEND_END, TICKER[index_name])
    if df is None or len(df) == 0:
        raise RuntimeError(f"{index_name} 연장 다운로드 결과가 비어있음")
    out = df.reset_index()[["날짜", "종가"]].rename(
        columns={"날짜": "date", "종가": "close"}
    )
    out["date"] = pd.to_datetime(out["date"])
    return out


def build() -> pd.DataFrame:
    frames = []
    for index_name in ("KOSPI", "KOSDAQ"):
        base = _load_2024(index_name)
        ext = _download_extension(index_name)
        merged = (
            pd.concat([base, ext], ignore_index=True)
            .drop_duplicates(subset="date")
            .sort_values("date")
            .reset_index(drop=True)
        )
        merged["index_name"] = index_name
        frames.append(merged[["date", "index_name", "close"]])
        print(
            f"[{index_name}] 2024={len(base)} + 연장={len(ext)} "
            f"=> 합계 {len(merged)} 거래일 "
            f"({merged['date'].min().date()} ~ {merged['date'].max().date()})"
        )
    prices = pd.concat(frames, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(OUT_PATH, index=False)
    print(f"저장: {OUT_PATH}  (총 {len(prices)} 행)")
    return prices


if __name__ == "__main__":
    build()
