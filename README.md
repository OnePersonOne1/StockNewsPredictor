# StockNewsPredictor — 뉴스 헤드라인의 horizon별 주가 예측력

> **연구 질문:** 뉴스 헤드라인의 주가 예측력은 얼마나 먼 미래까지 유효한가?

2024년 한 해 한국 경제·국제 뉴스 헤드라인(BIGKinds, 한국경제·조선일보·한겨레,
약 104,920건)으로 KOSPI·KOSDAQ 지수의 방향(3-class {-1, 0, +1})을
horizon h ∈ {1, 5, 21, 252} 거래일에 대해 예측한다.

한국 대학 학부 컴퓨팅 기초 과목 기말 보고서용 프로젝트.

## 구성

```
.
├── KRX_download.py            # KOSPI/KOSDAQ 2024 종가 원본 수집 (자격증명은 환경변수)
├── NewsResult_*.xlsx          # BIGKinds 원본 헤드라인 export (6개, 2024 전체)
├── kospi_2024.csv, kosdaq_2024.csv
├── phase1/                    # 데이터셋·라벨 구축
│   ├── config.py              # 전 단계 공용 하이퍼파라미터 (Phase 1/2)
│   ├── build_prices.py        # 가격 시계열을 2025~2026 상반기까지 연장 다운로드
│   ├── build_dataset.py       # 헤드라인↔거래일 매핑 + forward return
│   ├── build_labels.py        # σ_h(train) 기반 ternary 라벨
│   └── data/processed/dataset_final.parquet
└── phase2/                    # 모델 학습·평가 (상세는 phase2/README.md)
    ├── dataset.py  baseline.py  model.py  train.py  evaluate.py  analyze.py
    └── results/
```

## 확정된 설계 결정

- **3-class ternary**, 임계 ε_h = 0.3·σ_h
- **σ_h 는 train split 에서만 계산** → val/test 에 동일 적용 (데이터 누설 방지)
- **Temporal split**: train(2024-01~10) / val(2024-11) / test(2024-12)
- **Look-ahead 봉쇄**: 뉴스 날짜 D → 첫 거래일 T (T > D 엄격 부등호).
  거래일 T 의 입력 헤드라인은 전부 T 의 장 시작 이전에 발행된 것만 사용.
- Encoder `klue/roberta-base`, learnable query attention pooling, multi-task 4 horizon.

## 재현 순서 (GPU 머신, 예: RTX 4090)

```bash
git clone <this-repo>
cd StockNewsPredictor
pip install -r phase2/requirements.txt          # torch 는 CUDA 빌드로!

# --- Phase 1: 데이터셋 구축 ---
KRX_ID=... KRX_PW=... python phase1/build_prices.py   # 가격 연장(자격증명 필요)
python phase1/build_dataset.py
python phase1/build_labels.py

# --- Phase 2: 모델 ---
python phase2/baseline.py        # 기저 (GPU 불필요)
python phase2/train.py --epochs 1   # loss 감소 확인
python phase2/train.py              # full 학습
python phase2/evaluate.py           # 8셀 표 + decay plot + metrics.json
python phase2/analyze.py            # attention 상위 헤드라인
```

> `dataset_final.parquet`, `prices.parquet` 가 이미 저장소에 포함되어 있으면
> Phase 1 을 건너뛰고 바로 Phase 2 를 실행할 수 있다.

## 핵심 결과 요약 (정직한 보고)

- **h=1, h=5** 에서만 test 라벨이 3-class 로 실제 섞여 있어 의미 있는 평가가 가능.
  기저(TF-IDF) 기준 KOSPI h=5 가 가장 높음(acc 0.55, macro-F1 0.356, 무작위 0.333).
- **h=21, h=252** 는 test 창(1개월)이 짧아 forward 수익률이 **시장 드리프트에 지배**
  되어 거의 **단일 클래스(+1)** 로 붕괴 → 해당 셀 metric 은 신뢰도 낮음(보조 결과).
- 결론 방향: 헤드라인 신호는 **단기(h≤5)에 약하게 존재**, 장기로 갈수록 시장
  전체 추세에 묻힌다.

## 데이터 출처 / 라이선스 주의

- 헤드라인: **BIGKinds**(한국언론진흥재단) export. 뉴스 제목은 저작권이 있는
  콘텐츠이므로, 본 저장소의 `NewsResult_*.xlsx` 는 **학술적 비영리 재현 목적**으로만
  사용한다. 상업적 재배포 금지.
- 가격: KRX(pykrx). `KRX_download.py`/`build_prices.py` 는 KRX 로그인 자격증명을
  **환경변수**로만 받는다 (소스에 하드코딩 금지).
