# StockNewsPredictor — 뉴스 헤드라인의 horizon별 주가 예측력

> **연구 질문:** 뉴스 헤드라인의 주가 방향 예측력은 얼마나 먼 미래(horizon)까지 유효한가?

한국 경제·국제 뉴스 헤드라인(BIGKinds, 한국경제·조선일보·한겨레)으로 KOSPI·KOSDAQ
지수의 방향(3-class {-1, 0, +1})을 horizon h ∈ {1, 5, 21, 252} 거래일에 대해 예측한다.

학부 컴퓨팅사고와 데이터분석 과목 기말고사 대체 과제 프로젝트

- 헤드라인: **2021~2024년** 24개 파일(`data/NewsResult_*.xlsx`), 연 ~10만 건.
- 가격: KOSPI/KOSDAQ 지수 종가(2021~2026, **FinanceDataReader** — 자격증명 불필요).
- **실험별 정식 보고서는 [`docs/`](docs/README.md) 에 있다(EXP-A ~ EXP-G).**

---

## 1. 디렉터리 구성

```
.
├── data/NewsResult_*.xlsx     # BIGKinds 원본 헤드라인 (2021~2024, 24개; Git LFS)
├── kospi_2024.csv, kosdaq_2024.csv   # KRX 2024 종가(원본 시드)
├── KRX_download.py            # (참고) KRX 종가 수집 — 현재는 FDR 로 대체
├── HANDOFF.md, HANDOFF_MULTIYEAR.md  # 세션 인수인계 / 다년 실험 런북
├── docs/                      # ★ 실험 보고서 (README=인덱스, experiments/EXP-*.md, discussion.md)
├── phase1/                    # 데이터셋·라벨 구축
│   ├── config.py              # 공용 설정 + 실험 스위치(EXP_PROFILE, HEADLINE_FILTER)
│   ├── build_prices.py        # 가격 시계열(FDR, 프로필별 범위)
│   ├── build_dataset.py       # 헤드라인↔거래일 매핑 + forward return + 카테고리 필터
│   ├── build_labels.py        # σ_h(train) 기반 ternary 라벨
│   └── data/processed/        # dataset_final.parquet(2024), prices.parquet, 파생 parquet
└── phase2/                    # 모델 학습·평가 + 분석 (상세 phase2/README.md)
    ├── model.py dataset.py train.py evaluate.py baseline.py analyze.py
    ├── wordcount_baseline.py  # 비-ML 렉시콘 단어개수 기저
    ├── experiment_freeze.py   # encoder freeze 실험(EXP-B)
    ├── compare_methods.py compare_filters.py compare_years.py compare_attention.py
    └── results*/              # ★ 실험 케이스별 산출물 (아래 §3 지도)
```

## 2. 실험 스위치 — 같은 코드로 여러 조건

모든 빌드/학습/평가는 **두 환경변수**로 조건을 바꾼다. 기본값은 원래의 2024 실험이며,
산출물은 조건별 폴더로 자동 분리되어 서로 덮어쓰지 않는다.

**`EXP_PROFILE`** (기간·split·데이터 규모):

| 프로필 | train / val / test | 규모 | 용도 |
|---|---|---|---|
| `y2024` (기본) | 2024-01~10 / 11 / 12 | 소(406) | 주 실험(EXP-A) |
| `y2021`/`y2022`/`y2023` | Y-01~10 / Y-11 / Y-12 | 소(~400) | 데이터크기 통제(EXP-F) |
| `multiyear` | 2021-01~2023-09 / 2023-10~12 / **2024 전체** | 대(1358) | 데이터 확대(EXP-D) |

**`HEADLINE_FILTER`** (BIGKinds 통합분류1 카테고리로 잡음 제거):

| 필터 | 포함 카테고리 | 일평균(2024) |
|---|---|---|
| `all` (기본) | 전체 | ~287 |
| `market` | 경제>증권_증시 | ~28 |
| `macro` | 증권+금융_재테크+국제경제+외환+경제일반 | ~66 |
| `it` | IT_과학 + 경제>반도체 | ~27 |
| `market_it` | 증권 + IT_과학 + 반도체 | ~55 |

예) `EXP_PROFILE=multiyear HEADLINE_FILTER=market python phase2/train.py`

## 3. 결과 폴더 지도 — 무엇을 어떤 조건으로 실험했나

| 폴더 | 실험 | EXP_PROFILE | HEADLINE_FILTER | test 구간 |
|---|---|---|---|---|
| `phase2/results/` | EXP-A 주 모델 + 모든 비교 허브 | y2024 | all | 2024-12 (40행) |
| `phase2/results_market/` | EXP-C 관련성 필터 | y2024 | market | 2024-12 |
| `phase2/results_macro/` | EXP-C 관련성 필터 | y2024 | macro | 2024-12 |
| `phase2/results_y2021/` | EXP-F 소표본 복제 | y2021 | all | 2021-12 |
| `phase2/results_y2022/` | EXP-F 소표본 복제 | y2022 | all | 2022-12 |
| `phase2/results_y2023/` | EXP-F 소표본 복제 | y2023 | all | 2023-12 |
| `phase2/results_multiyear/` | EXP-D 데이터 확대 | multiyear | all | **2024 전체(488행)** |
| `phase2/results_multiyear_market/` | EXP-E 다년×필터 | multiyear | market | 2024 전체 |
| `phase2/results_multiyear_macro/` | EXP-E 다년×필터 | multiyear | macro | 2024 전체 |
| `phase2/results_multiyear_it/` | EXP-G IT 섹터 | multiyear | it | 2024 전체 |
| `phase2/results_multiyear_market_it/` | EXP-G IT 섹터 | multiyear | market_it | 2024 전체 |

> 규칙: 폴더명 = `results` + (프로필≠y2024 시 `_<프로필>`) + (필터≠all 시 `_<필터>`).
> 파생 데이터셋(`dataset_*.parquet`)·변형 체크포인트(`checkpoints_*/`)는 재현 가능하므로
> Git 에 올리지 않는다(gitignore).

## 4. 결과 폴더 안의 파일 — 무엇인가

**공통(모든 `results*/`):**

| 파일 | 내용 |
|---|---|
| `baseline_metrics.csv/md` | TF-IDF+LogReg 기저, 8셀(horizon×index) acc·macro-F1 |
| `test_metrics.csv/tex` | RoBERTa 주 모델 8셀 acc·macro-F1 (LaTeX 표 포함) |
| `metrics.json` | per-class precision/recall + confusion matrix(8개) |
| `figures/horizon_decay.png` | horizon별 macro-F1 곡선(baseline 오버레이) |
| `figures/attention_map.png` | 헤드라인별 attention 가중치 시각화(가장 집중된 사례) |
| `figures/attention_heatmap.png` | test 행 × 순위 정렬 가중치 히트맵 |
| `figures/attention_entropy.png` | 정규화 엔트로피 분포(1=균일=평균 pooling) |
| `attention_analysis.md`, `attention_stats.json`, `attention_per_row.csv` | attention 집중도 통계·키워드 |

**비교 허브(`results/`, `results_multiyear/`)에만 추가:**

| 파일 | 내용 |
|---|---|
| `wordcount_metrics.csv/md` | 비-ML 렉시콘 단어개수 기저 |
| `method_comparison.{csv,md,tex}`, `figures/method_comparison.png` | random/wordcount/TF-IDF/RoBERTa 사다리 |
| `filter_ablation.{csv,md}`, `figures/filter_ablation.png` | 필터(all/market/macro/it/market_it) 비교 |
| `roberta_freeze_metrics.csv`, `roberta_freeze_compare.md` | encoder freeze 실험(EXP-B, results/ 만) |
| `signal_metrics.csv`, `signal_report.md` | EXP-H 신호 검정(이진 up/down·IC·롱숏 백테스트+유의성) |
| `signal_seed_sweep.csv/md`, `figures/signal_summary.png` | EXP-H RoBERTa IC 6-시드 견고성(results_multiyear/) |
| `top_attention_headlines.csv`, `train_log.txt` | attention 상위 헤드라인 / 학습 로그(results/ 만) |

**`results/` 의 케이스-간 요약(연구 전체):**

| 파일 | 내용 | 출처 스크립트 |
|---|---|---|
| `exp_f_datasize.{csv,md}`, `figures/exp_f_datasize.png` | 데이터 크기 vs 2025 특수성(EXP-F) | `compare_years.py` |
| `attention_summary.{csv,md}`, `figures/attention_summary.png` | 11개 케이스 attention 집중도 | `compare_attention.py` |
| `EXPERIMENT_LOG.md` | 누적 실험 로그(정식 보고서는 `docs/`) | — |

## 5. 확정된 설계 (전 실험 불변)

- **3-class ternary**, 임계 ε_h = 0.3·σ_h. **σ_h 는 train 에서만** 계산(누설 방지).
- **Temporal split**(랜덤 분할 금지). **Look-ahead 봉쇄**: 뉴스일 D → 첫 거래일 T(T>D 엄격),
  입력 헤드라인은 T 장 시작 이전 발행분만.
- Encoder **`klue/roberta-base`**, **단일 learnable query** attention pooling(→ attention 은
  horizon 공통), multi-task 4 horizon head. 모델 확대(large) 미사용.

## 6. 재현

```bash
pip install -r phase2/requirements.txt          # torch 는 driver 에 맞는 CUDA 빌드로
                                                 # (cu130 은 너무 신버전일 수 있음; cu128 권장)

# Phase 1 — 데이터셋 (가격은 FDR, 자격증명 불필요)
[EXP_PROFILE=multiyear] python phase1/build_prices.py
[EXP_PROFILE=… HEADLINE_FILTER=…] python phase1/build_dataset.py
[EXP_PROFILE=… HEADLINE_FILTER=…] python phase1/build_labels.py

# Phase 2 — 모델/평가/분석 (같은 EXP_PROFILE/HEADLINE_FILTER 접두)
python phase2/baseline.py
python phase2/train.py --batch-size 16     # --seed N 으로 변동성 점검 가능
python phase2/evaluate.py
python phase2/attention_analysis.py
# 비교: compare_methods.py / compare_filters.py / compare_years.py / compare_attention.py
```

> 기본(`y2024`+`all`)의 `dataset_final.parquet`·`prices.parquet` 는 저장소에 포함(LFS).
> 다른 조건은 위 환경변수로 빌드하면 별도 parquet/폴더로 생성된다.

## 7. 핵심 결과 (요약 — 자세히는 [`docs/`](docs/README.md))

- **헤드라인의 주가 방향 예측력은 본질적으로 약하다**: 충분한 데이터에서 건강하게 학습한
  RoBERTa 조차 무작위(0.333) 부근(0.27~0.35). 주 결과는 h1, h5(장기는 평가창 한계).
- **저자원에서 딥러닝 붕괴는 데이터 규모의 산물**(EXP-A~F): 406행에선 예측·attention 모두
  붕괴, 3년 데이터(EXP-D)에서 해소(attention lift 1.07×→7.85×). 정상기 소표본(EXP-F)도
  붕괴 → 2025 특수성 아닌 데이터 크기 문제.
- **입력 정제는 TF-IDF 를 키우고(EXP-C·E), IT 섹터에 신호가 농축**(EXP-G: 9%로 전체에 준함).
  단 RoBERTa 미세 비교는 단일 시드 변동(±0.04+)에 유의.

## 8. 데이터 출처 / 라이선스

- 헤드라인: **BIGKinds**(한국언론진흥재단) export. 제목은 저작권 콘텐츠이므로
  `data/NewsResult_*.xlsx` 는 **학술 비영리 재현 목적**으로만 사용(상업적 재배포 금지).
- 가격: KOSPI/KOSDAQ 지수 종가(FinanceDataReader; KRX 와 종가 일치 확인).
  KRX 직접 수집(`KRX_download.py`)은 자격증명을 **환경변수**로만 받는다.
