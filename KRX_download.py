"""
KRX_download.py — KOSPI/KOSDAQ 2024 지수 종가 다운로드 (원본 수집 스크립트)

pykrx 지수 API 는 KRX 로그인 자격증명을 요구한다. 보안을 위해 자격증명은
소스에 하드코딩하지 않고 환경변수(KRX_ID, KRX_PW)에서 읽는다.

사용:
  # PowerShell
  $env:KRX_ID="your_id"; $env:KRX_PW="your_pw"; python KRX_download.py
  # bash
  KRX_ID=your_id KRX_PW=your_pw python KRX_download.py
"""
import os
import sys

if not (os.environ.get("KRX_ID") and os.environ.get("KRX_PW")):
    sys.exit("KRX_ID / KRX_PW 환경변수를 설정하세요 (자격증명 하드코딩 금지).")

from pykrx import stock

# KOSPI 종합주가지수 (지수 코드 1001)
kospi = stock.get_index_ohlcv_by_date("20240101", "20241231", "1001")
kospi.to_csv("kospi_2024.csv", encoding="utf-8-sig")

# KOSDAQ 지수 (지수 코드 2001)
kosdaq = stock.get_index_ohlcv_by_date("20240101", "20241231", "2001")
kosdaq.to_csv("kosdaq_2024.csv", encoding="utf-8-sig")

print("저장 완료: kospi_2024.csv, kosdaq_2024.csv")
