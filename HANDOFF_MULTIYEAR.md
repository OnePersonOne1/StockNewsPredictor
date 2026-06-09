# HANDOFF — 다중연도 실험 (2021–2023 학습 → 2024 예측)

> 주 실험(2024-only)이 **406행으로 RoBERTa fine-tune → test 단일클래스 붕괴**,
> TF-IDF 미달이라는 한계를 보였다(`phase2/results/EXPERIMENT_LOG.md`). 근본 원인은
> 데이터 부족 + 짧은 test 창(2024-12, 40행) + val→test 분포 이동이다. 이를 정면
> 해결하기 위해 **학습 데이터를 3년으로 늘리고 test 를 2024 전체로** 바꾼다.
> 한국 시장은 2024 와 2025/2026(유래없는 상승장)의 괴리가 커, 학습 라벨을
> 2021–2024 "박스피" 국면에 두고 2025 상승장 오염을 피하는 효과도 있다.

## 0. 설계 (config 프로필 `multiyear`)

| split | 기간 | 목적 |
|---|---|---|
| train | 2021-01-01 ~ 2023-09-30 | ~1,300+행 (현 406행 대비 3배+) |
| val   | 2023-10-01 ~ 2023-12-31 | temporal hold-out |
| test  | 2024-01-01 ~ 2024-12-31 | **2024 전체**(~490행), 3-class 실제 분포 |

- encoder=klue/roberta-base, ε=0.3σ, σ는 train(2021–2023)에서만 계산 — **불변**.
- 모델은 base 유지(large 미사용 — 병목은 용량이 아니라 데이터).
- caveat: 2024 말 행의 h=252 forward 는 여전히 2025 상승장으로 넘어감 → h21/h252
  는 보조 결과로만(주 결과 h1,h5).

## 1. 사용자가 제공해야 할 두 입력 ⚠️

1. **2021·2022·2023 BIGKinds 헤드라인 export(xlsx)** — 2024 와 **동일 매체
   (한국경제/조선일보/한겨레), 동일 카테고리(경제·국제)**. 파일명은 글롭
   `NewsResult_*.xlsx` 에 걸리게 저장소 루트에 두면 됨 (예:
   `NewsResult_20210101-20210228.xlsx` …). 컬럼은 2024 와 동일(`일자`,`제목`)이어야 함.
2. **KRX 자격증명**(`KRX_ID`/`KRX_PW`) — 2021–2023 지수 종가 백필용. pykrx 지수
   API 는 로그인 필수이며 현재 세션엔 자격증명이 없음. (소스 하드코딩 금지)

> 둘 중 하나라도 없으면 이 실험은 실행 불가. 가격은 자격증명만 있으면 자동.

## 2. 실행 순서 (모든 단계에 `EXP_PROFILE=multiyear`)

```bash
# (1) 헤드라인: 2021~2023 xlsx 를 저장소 루트에 배치 (NewsResult_2021*.xlsx ...)

# (2) 가격 백필 2021~2023 (+기존 2024 CSV +2025~2026 미래) — 자격증명 필요
EXP_PROFILE=multiyear KRX_ID=... KRX_PW=... python phase1/build_prices.py
#   주의: prices.parquet 를 2021~2026 superset 으로 덮어씀(2024-only 빌드와 호환).

# (3) 데이터셋 + 라벨 (→ dataset_2124.parquet, 프로필 경로)
EXP_PROFILE=multiyear python phase1/build_dataset.py
EXP_PROFILE=multiyear python phase1/build_labels.py

# (4) Phase 2 — 산출물은 results_multiyear/ 로 자동 분리(2024 결과 보존)
EXP_PROFILE=multiyear python phase2/baseline.py
EXP_PROFILE=multiyear python phase2/train.py --batch-size 16
EXP_PROFILE=multiyear python phase2/evaluate.py
EXP_PROFILE=multiyear python phase2/experiment_freeze.py
EXP_PROFILE=multiyear python phase2/attention_analysis.py
EXP_PROFILE=multiyear python phase2/compare_methods.py
```

## 3. 파이프라인이 이미 준비된 것 (이번 커밋)

- `phase1/config.py`: `EXP_PROFILE` 스위치. 기본 `y2024` 는 **완전 불변**(검증함:
  train 406, 동일 sigma). `multiyear` 는 split/연도/데이터셋경로/가격백필/결과·
  체크포인트 경로를 자동 분리.
- `phase1/build_dataset.py`: 하드코딩 `year==2024` → `DATA_YEARS` (프로필).
- `phase1/build_prices.py`: `_download_range()` 일반화 + `PRICE_BACK_START` 백필.
- Phase 2 코드는 **수정 불필요**(모두 config 경로/스플릿을 import).

## 4. 기대/주의

- 가설: 학습 데이터 3배↑ + test 2024 전체 → 단일클래스 붕괴 완화, TF-IDF 와의
  비교가 공정해짐. 다만 **헤드라인 신호 자체가 약하다는 결론이 유지될 수도 있음** —
  그 경우도 정직하게 보고(EXPERIMENT_LOG.md 에 추가).
- 2021–2023 BIGKinds export 의 일평균 헤드라인 수가 2024(~427)와 다르면
  MAX_HEADLINES 영향 점검.
- build_dataset 의 sanity check(헤드라인 0개 거래일)가 연휴 구간에서 걸릴 수 있음 —
  걸리면 해당 거래일 처리(drop) 확인.
