# StockNewsPredictor — 뉴스 헤드라인의 horizon별 주가 예측력

> **연구 질문:** 뉴스 헤드라인의 주가 방향 예측력은 얼마나 먼 미래(horizon)까지 유효한가?

한국 경제·국제 뉴스 헤드라인(BIGKinds, 한국경제·조선일보·한겨레)으로 KOSPI·KOSDAQ
지수의 방향을 horizon h ∈ {1, 5, 21, 252} 거래일에 대해 예측한다. 기본은 3-class
{-1,0,+1}, 이진(up/down) 모드도 지원. 학부 **컴퓨팅사고와 데이터분석** 과목 기말고사 대체
과제 프로젝트 — **성능보다 과정·해석·정직한 보고가 핵심.**

- 헤드라인: **2021~2024 본체**(`data/NewsResult_*.xlsx`) + **IT 섹션**(`data/IT_section/`) +
  **삼성전자 단독 2021~2025**(`data/Samsung_Electronics/`). 모두 BIGKinds export(Git LFS).
- 가격: 지수·개별종목 종가(2021~2026, **FinanceDataReader** — 자격증명 불필요).
- **실험별 정식 보고서는 [`docs/`](docs/README.md) 에 있다(EXP-A ~ EXP-X, 고찰 `discussion.md`).**

### 핵심 결론 한눈에
1. **헤드라인의 방향 예측력은 약하지만 "없지는 않다".** 3-class macro-F1 로는 모두 무작위
   부근이나, 이는 평가 지표가 신호를 가린 탓 — **IC(이진 방향) 로 보면 RoBERTa 가 약한 양(+)
   신호**(KOSPI h5 IC ≈0.15, 다중시드 안정). 단 거래 가능 수준은 아님(EXP-H).
2. **저자원에서 딥러닝 붕괴는 데이터 크기 문제**(EXP-A~F, M): 406행 붕괴 → 3년(1358) 해소.
3. **입력은 적고·최신·관련 있는 게 최적**: 정제 필터가 TF-IDF↑(EXP-C·E·G), 헤드라인을 더 넣거나
   (EXP-P) 랜덤 셔플(EXP-Q)하면 RoBERTa↓, **실제 시각 정렬이 최선**(EXP-S).
4. **4 epoch 은 대표본에 부족**(EXP-T) — 다년/단일종목은 더 길게.
5. **단일종목(삼성)도 OOS 약신호**(EXP-V), 그 원인은 **2025 유례없는 +124% 폭등장**(EXP-W).
   2024 를 완전 held-out 으로 재검(EXP-X)하면 2024 우위는 일부 낙관편향(0.59→0.52)이나
   **2025 레짐-이동 실패 결론은 설계와 무관하게 견고**.

---

## 1. 디렉터리 구성

```
.
├── data/NewsResult_*.xlsx     # BIGKinds 본체 헤드라인 (2021~2024; Git LFS)
│   ├── IT_section/            # IT·과학 섹션 추가 export (INCLUDE_IT/IT_ONLY 용)
│   └── Samsung_Electronics/   # 삼성전자 단독 검색식 export 2021~2025 (samsung 프로필)
├── kospi_2024.csv, kosdaq_2024.csv   # KRX 2024 종가(원본 시드)
├── KRX_download.py            # (참고) KRX 종가 수집 — 현재는 FDR 로 대체
├── docs/                      # ★ 실험 보고서 (README=인덱스, experiments/EXP-*.md, discussion.md,
│                              #   HANDOFF_for_report.md = 보고서 작성 인수인계,
│                              #   bigkinds_company_keywords.md = 종목별 뉴스 수집 검색식)
├── phase1/                    # 데이터셋·라벨 구축
│   ├── config.py              # 공용 설정 + 모든 실험 스위치(§2)
│   ├── build_prices.py        # 가격 시계열(FDR, 프로필별 범위; 지수 또는 STOCK_TICKER)
│   ├── build_dataset.py       # 헤드라인↔거래일 매핑 + forward return + 필터/시각정렬/본문
│   ├── build_labels.py        # σ_h(train) 기반 ternary 라벨 (BINARY 시 미사용)
│   └── data/processed/        # dataset_*.parquet, prices*.parquet (조건별 파생)
└── phase2/                    # 모델 학습·평가 + 분석 (상세 phase2/README.md)
    ├── model.py dataset.py train.py evaluate.py baseline.py analyze.py
    ├── wordcount_baseline.py  # 비-ML 렉시콘 단어개수 기저
    ├── experiment_freeze.py   # encoder freeze 실험(EXP-B)
    ├── signal_test.py signal_summary.py signal_highconf.py signal_interpret.py
    │                          # EXP-H/I/J 방향신호 검정·고확신·해석(이진 up/down·IC·백테스트)
    ├── fair_topn.py           # EXP-R 동일입력(top-N) 공정비교 기저
    ├── analyze_2025.py        # EXP-W/X 연도별 레짐 이동 분석(삼성)
    ├── compare_methods.py compare_filters.py compare_years.py compare_attention.py
    └── results*/              # ★ 실험 케이스별 산출물 (아래 §3 지도)
```

## 2. 실험 스위치 — 같은 코드로 여러 조건

모든 빌드/학습/평가는 **환경변수**로 조건을 바꾼다. 기본값은 원래의 2024 지수 3-class
실험이며, 산출물은 조건별 폴더(`_SUF` 접미)로 자동 분리되어 서로 덮어쓰지 않는다.

**`EXP_PROFILE`** (대상·기간·split·데이터 규모):

| 프로필 | 대상 | train / val / test | 규모 | 용도 |
|---|---|---|---|---|
| `y2024` (기본) | 지수 | 2024-01~10 / 11 / 12 | 소(406) | 주 실험(EXP-A) |
| `y2021`/`y2022`/`y2023` | 지수 | Y-01~10 / Y-11 / Y-12 | 소(~400) | 데이터크기 통제(EXP-F) |
| `multiyear` | 지수 | 2021~2023-09 / 2023-10~12 / **2024 전체** | 대(1358) | 데이터 확대(EXP-D) |
| `samsung` | 삼성전자(005930) | 2021~2023 / 2024 / **2025** | 중 | 단일종목(EXP-V/W) |
| `samsung_cv` | 삼성전자 | 2021~2022 / 2023 / **2024+2025** | 중 | 2024 완전 held-out(EXP-X) |

**`HEADLINE_FILTER`** (BIGKinds 통합분류1 카테고리로 잡음 제거):

| 필터 | 포함 카테고리 | 일평균(2024) |
|---|---|---|
| `all` (기본) | 전체 | ~287 |
| `market` | 경제>증권_증시 | ~28 |
| `macro` | 증권+금융_재테크+국제경제+외환+경제일반 | ~66 |
| `it` | IT_과학 + 경제>반도체 | ~27 |
| `market_it` | 증권 + IT_과학 + 반도체 | ~55 |

**그 외 토글**(모두 산출물 경로에 반영, 기본=왼쪽):

| 스위치 | 값 | 효과 / 관련 실험 |
|---|---|---|
| `BINARY` | `0`/`1` | 3-class → 이진 up/down(N_CLASSES=2). IC·방향검정용(EXP-H, M, V~X) |
| `HEADLINE_ORDER` | `date`/`time` | 일자정렬 → 뉴스식별자 HHMMSS 실시각 정렬. time 이 최선(EXP-S) |
| `HEADLINE_SAMPLE` | `recent`/`random` | top-N 선택을 최신순 vs 랜덤. recent 우세(EXP-Q) |
| `MAX_HEADLINES` | 정수(기본 30) | 하루에 읽는 헤드라인 수. 많을수록 RoBERTa↓(EXP-P), 삼성=64(전량) |
| `BASELINE_TOPN` | `0`/정수 | TF-IDF·wordcount 도 동일 top-N 으로 제한 → 공정비교(EXP-R) |
| `GRAD_CKPT` | `0`/`1` | gradient checkpointing — 큰 MAX_HEADLINES 학습 메모리 절감 |
| `INCLUDE_IT` | `0`/`1` | `data/IT_section/` 을 본체에 병합(EXP-L) |
| `IT_ONLY` | `0`/`1` | IT 섹션만 단독 사용(EXP-N) |
| `USE_BODY` | `0`/`1` | 제목+본문 입력(MAX_LENGTH↑). 개선 없음 — 신호는 제목(EXP-K) |
| `STOCK_TICKER` | 코드 | 개별종목 FDR 티커(프로필이 지정; 예 005930) |

예) `EXP_PROFILE=samsung HEADLINE_ORDER=time BINARY=1 python phase2/train.py --max-headlines 64 --batch-size 8 --epochs 30`

## 3. 결과 폴더 지도 — 무엇을 어떤 조건으로 실험했나

폴더명 = `results` + (프로필≠y2024 시 `_<프로필>`) + (필터≠all 시 `_<필터>`) +
(토글 켜짐 시 `_body`/`_itonly`/`_itaug`/`_ordtime`/`_random`/`_bin` 등). 22개 폴더 중 대표:

| 폴더 | 실험 | 조건 | test 구간 |
|---|---|---|---|
| `results/` | EXP-A 주 모델 + **모든 비교 허브** | y2024 / all | 2024-12 (40행) |
| `results_market/`, `results_macro/` | EXP-C 관련성 필터 | y2024 | 2024-12 |
| `results_y2021/`·`_y2022/`·`_y2023/` | EXP-F 소표본 복제 | y20XX / all | 20XX-12 |
| `results_multiyear/` | EXP-D/H/T **다년 허브**(신호·시드·로그) | multiyear / all | **2024 전체(488)** |
| `results_multiyear_market/`·`_macro/` | EXP-E 다년×필터 | multiyear | 2024 전체 |
| `results_multiyear_it/`·`_market_it/` | EXP-G IT 섹터 | multiyear | 2024 전체 |
| `results_multiyear_itaug/`·`_itonly/`·`_it_itaug/` | EXP-L/N IT 병합·단독 | INCLUDE_IT/IT_ONLY | 2024 전체 |
| `results_multiyear_body/` | EXP-K 제목+본문 | USE_BODY | 2024 전체 |
| `results_multiyear_ordtime/`·`_random/` | EXP-S/Q 시각·랜덤 정렬 | HEADLINE_ORDER/SAMPLE | 2024 전체 |
| `results_samsung_ordtime/`·`_bin/` | EXP-V/W 삼성 3-class·이진 | samsung / time | **2025** |
| `results_samsung_top64_ordtime/` | EXP-V 삼성 mh64(전량) | samsung | 2025 |
| `results_samsung_cv_ordtime_bin/` | EXP-X 2024 완전 held-out | samsung_cv / bin | **2024+2025** |

> 파생 데이터셋(`dataset_*.parquet`)·변형 체크포인트(`checkpoints_*/`)는 재현 가능하므로
> Git 에 올리지 않는다(gitignore). 전체 실험↔폴더 대응은 [`docs/README.md`](docs/README.md).

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
| `signal_highconf.*`, `signal_interpret.*` | EXP-I/J 고확신 셀 적중률·attend 키워드 |
| `fair_topn_*.{csv,md}` | EXP-R 동일 top-N 입력 공정비교 |
| `trainlog_mh*_bs*_ep*_seed*.txt` | **학습 로그 자동저장**(train.py, EXP-T 이후 전 학습) |
| `exp_w_*.{csv,md}`, `figures/exp_w_2025.png` | EXP-W/X 연도별 레짐 이동(삼성 results 폴더) |
| `top_attention_headlines.csv` | attention 상위 헤드라인(results/ 만) |

**`results/` 의 케이스-간 요약(연구 전체):**

| 파일 | 내용 | 출처 스크립트 |
|---|---|---|
| `exp_f_datasize.{csv,md}`, `figures/exp_f_datasize.png` | 데이터 크기 vs 2025 특수성(EXP-F) | `compare_years.py` |
| `attention_summary.{csv,md}`, `figures/attention_summary.png` | 케이스별 attention 집중도 | `compare_attention.py` |
| `EXPERIMENT_LOG.md` | 누적 실험 로그(정식 보고서는 `docs/`) | — |

## 5. 확정된 설계 (전 실험 불변)

- **3-class ternary**(기본), 임계 ε_h = 0.3·σ_h. **σ_h 는 train 에서만** 계산(누설 방지).
  방향 신호 검정 시 **이진 up/down**(`BINARY=1`)으로 전환, IC(Spearman)로 평가.
- **Temporal split**(랜덤 분할 금지). **Look-ahead 봉쇄**: 뉴스일 D → 첫 거래일 T(T>D 엄격),
  입력 헤드라인은 T 장 시작 이전 발행분만. forward return 미존재 시 `-100` ignore 라벨.
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

# Phase 2 — 모델/평가/분석 (같은 EXP_PROFILE/HEADLINE_FILTER… 접두)
python phase2/baseline.py
python phase2/train.py --batch-size 16     # --seed N, --max-headlines, --epochs;
                                            # 학습 로그는 trainlog_*.txt 로 자동 저장
python phase2/evaluate.py
python phase2/attention_analysis.py
# 방향 신호: BINARY=1 python phase2/signal_test.py (→ signal_summary/highconf/interpret)
# 공정비교: BASELINE_TOPN=30 python phase2/fair_topn.py
# 삼성 레짐분석: EXP_PROFILE=samsung HEADLINE_ORDER=time BINARY=1 python phase2/analyze_2025.py
# 비교: compare_methods.py / compare_filters.py / compare_years.py / compare_attention.py
```

> 기본(`y2024`+`all`)의 `dataset_final.parquet`·`prices.parquet` 는 저장소에 포함(LFS).
> 다른 조건은 위 환경변수로 빌드하면 별도 parquet/폴더로 생성된다.

## 7. 핵심 결과 (요약 — 자세히는 [`docs/`](docs/README.md))

- **평가 지표가 결론을 가른다**(EXP-H, R): 3-class **macro-F1** 로는 RoBERTa·TF-IDF 모두
  무작위(0.333) 부근(0.27~0.37)이라 "신호 없음"처럼 보이나, **이진 방향 IC** 로 보면
  **RoBERTa 가 약한 양(+)신호**(KOSPI h5 IC≈0.15, 다중시드 안정), TF-IDF 는 IC≈0, wordcount 는
  음(-)신호. 단 롱숏 백테스트는 유의하지 않음 → **탐지는 되나 거래 가능한 알파는 아님**(약형 효율).
- **저자원 딥러닝 붕괴는 데이터 규모의 산물**(EXP-A~F, M): 406행에선 예측·attention 모두 붕괴,
  3년(1358)에서 해소(attention lift 1.07×→7.85×). 정상기 소표본(EXP-F)도 붕괴 → 데이터 크기 문제.
- **입력은 적고·최신·실시각순이 최적**: 정제 필터가 TF-IDF↑(EXP-C·E), IT 섹터에 신호 농축
  (EXP-G, 단 신호는 'IT 주제'가 아닌 '경제적 프레이밍'에 있음 — EXP-N). 헤드라인을 더 넣으면
  RoBERTa↓(EXP-P), 랜덤 셔플도 손해(EXP-Q), **뉴스식별자 실시각 정렬이 최선**(EXP-S). 본문 추가
  무효 — 신호는 제목(EXP-K).
- **4 epoch 은 대표본에 부족**(EXP-T): 다년 val 이 10 epoch 까지 0.258→0.343 상승 → 기존 다년
  결과는 보수적. 소표본은 2 epoch 에 과적합.
- **단일종목(삼성)으로 확장**(EXP-V~X): 2024(정상기) OOS 에선 방향 적중이 '항상 up'을 상회해
  **약신호 존재**(EXP-W 0.59 → 2024 완전 held-out EXP-X 0.52, 일부는 val-선택 낙관편향),
  그러나 **2025 +124.5% 유례없는 폭등장**에선 실패(up 33% 예측 vs 실제 62%). 원인은 신호 부재가
  아니라 **레짐 이동**(2021~23 하락우세로 학습→상승장 체계적 오답)이며, 이는 **설계와 무관하게 견고**.
- 방법론 주의: RoBERTa 단일시드 변동 ±0.04+ → TF-IDF(결정적)·추세·다중시드로 판단.

## 8. 데이터 출처 / 라이선스

- 헤드라인: **BIGKinds**(한국언론진흥재단) export. 제목은 저작권 콘텐츠이므로
  `data/NewsResult_*.xlsx` 는 **학술 비영리 재현 목적**으로만 사용(상업적 재배포 금지).
- 가격: KOSPI/KOSDAQ 지수 종가(FinanceDataReader; KRX 와 종가 일치 확인).
  KRX 직접 수집(`KRX_download.py`)은 자격증명을 **환경변수**로만 받는다.
