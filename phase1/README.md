# Phase 1 — 데이터셋 / 라벨 구축

BIGKinds 원본 헤드라인 + KRX 지수 종가를 결합해 Phase 2 학습용
`dataset_final.parquet` 를 생성한다.

## 파이프라인

```
build_prices.py   원본 2024 CSV + pykrx 연장(2025~2026 상반기) → prices.parquet
       │            (test=2024-12 의 h5/h21/h252 forward return 계산에 필수)
build_dataset.py  xlsx 6개 → (일자,제목) → 거래일 매핑(T>D) → forward return
       │            → dataset_final.parquet (라벨 제외)
build_labels.py   train 의 ret_h 로 σ_h 계산 → ε_h=0.3σ_h → ternary 라벨 부여
```

실행:
```bash
KRX_ID=... KRX_PW=... python phase1/build_prices.py
python phase1/build_dataset.py
python phase1/build_labels.py
```

## 최종 스키마 (`data/processed/dataset_final.parquet`)

| 컬럼 | 설명 |
|---|---|
| `date` | 거래일 (datetime) |
| `index_name` | 'KOSPI' \| 'KOSDAQ' |
| `close` | 종가 |
| `headlines` | list[str], **최신순** (Phase 2 가 상위 30개 사용) |
| `n_headlines` | 그날 매핑된 헤드라인 수 |
| `ret_h{1,5,21,252}` | h 거래일 후 수익률 |
| `label_h{1,5,21,252}` | ternary {-1,0,+1} |
| `split` | 'train' \| 'val' \| 'test' |

규모: 488 행 (KOSPI 244 + KOSDAQ 244), split train 406 / val 42 / test 40.

## 핵심 설계

- **Look-ahead 봉쇄**: 뉴스 날짜 D 를 `searchsorted(side='right')` 로 **T > D 인
  첫 거래일** 에 매핑. 거래일 T 가 모으는 헤드라인 = 직전 거래일·주말·휴일에
  발행된, T 의 장 시작 이전 뉴스뿐. 당일(T) 발행 뉴스는 다음 거래일로 넘어간다.
- **σ_h 는 train 에서만**: `compute_sigma(train)` 로 horizon별 표준편차 산출,
  두 지수를 합쳐 추정(σ_h notation 과 일치). 같은 σ_h 를 val/test 라벨에 적용.
- **가격 연장 이유**: 2024 거래일은 244개뿐이라, 2024 종가만으로는 12월 test 의
  중·장기 horizon forward return 이 전부 결측이 된다. 따라서 2025~2026 상반기
  종가를 연장 다운로드(동일 KRX 출처)하여 모든 2024 날짜의 h=252 까지 계산한다.

## 알려진 한계

- 장기 horizon(특히 h=21, h=252)의 test 라벨은 짧은 test 창(2024-12) 탓에 거의
  단일 클래스로 쏠린다 (시장 드리프트 지배). Phase 2 결과 해석 시 반드시 고려.
