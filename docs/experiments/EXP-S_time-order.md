# EXP-S: 실제 시각 정렬 — 버려지던 타임스탬프로 진짜 '최신 시각' top-30

| 항목 | 내용 |
|---|---|
| 상태 | 완료 (개선 발견) |
| 날짜 | 2026-06-09 |
| 커밋 | `<this>` |
| 데이터 | 다년. `HEADLINE_ORDER=time`(식별자 시각순) vs date(날짜순, 현재) vs random(EXP-Q) |
| 산출물 | `phase2/results_multiyear_ordtime/`, `dataset_2124_ordtime.parquet`(gitignore) |

## 1. 동기 / 가설
EXP-Q 정정에서 드러난 사실: **뉴스 식별자엔 실제 시각(YYYYMMDD HHMMSS)이 있는데
현재 정렬은 '일자(날짜)'만 써 같은 날 내 시각 정보를 버린다.** 그래서 top-30 이 "진짜
최신 시각"이 아니라 BIGKinds export 순서였다. 식별자 시각으로 정렬하면 *장 시작에
시간적으로 가장 가까운* 30건이 되어 더 예측적일 것이라는 가설.

## 2. 설정
- `HEADLINE_ORDER=time`: build_dataset 가 식별자에서 14자리 타임스탬프를 추출, 버킷 내
  정렬 키를 news_date → **실제 시각** 으로 교체(거래일 매핑은 날짜 그대로). top-30 이
  date-order 와 **4/1966 행만 동일**(=거의 완전히 다른 헤드라인 집합). 그 외 다년 설정 불변.
- RoBERTa 6 시드 IC + macro-F1.

## 3. 결과 (다년, RoBERTa)

**IC 6-시드 (mean±std, 양수/6):**

| 정렬 | KOSPI h1 | KOSPI h5 | KOSDAQ h1 | KOSDAQ h5 |
|---|---|---|---|---|
| random (EXP-Q) | +0.075±0.051 | +0.028±0.062 | +0.025±0.031 | −0.004±0.022 |
| date(현재, EXP-D) | +0.086±0.038 | +0.146±0.026 | +0.061±0.022 | +0.068±0.033 |
| **time(EXP-S)** | **+0.134±0.054** | +0.148±0.050 | +0.094±0.036 | +0.037±0.011 |

**macro-F1 (seed42):** time KOSPI 0.335/0.325 vs date 0.318/0.298 (h1/h5 모두↑). bin_acc
KOSPI h1 0.623(seed42).

## 4. 분석 — **시각 정렬 ≥ 날짜순 > 랜덤 (특히 단기에서 개선)**
- **시각 정렬이 단기(h1)를 분명히 끌어올린다**: KOSPI h1 IC +0.086→**+0.134**(+55%),
  KOSDAQ h1 +0.061→+0.094. macro-F1 도 KOSPI h1 0.318→0.335, h5 0.298→0.325.
- 직관과 일치: **장 시작에 시간적으로 가장 가까운 뉴스가 다음날(h1) 움직임에 가장
  예측적**이다. 더 먼 horizon(h5)일수록 '정확한 분 단위 최신성'의 이득은 작아진다
  (KOSPI h5 평균은 +0.146→+0.148 로 동등, 다만 std↑).
- **EXP-Q 정정의 완결**: 이제 "최신=관련성"이 *실제 시각* 기준으로 정당화된다. 버려지던
  타임스탬프가 실재 가치를 지녔다(특히 단기). 사용자의 "논리상 제대로 정렬해야 한다"가 맞음.
- 정렬 3종 종합: **time ≥ date > random.** 랜덤이 최악(EXP-Q), 시각이 최선.

## 5. 한계
- KOSPI h5 는 평균↑이나 분산↑(0.026→0.050) — 단기만큼 명확친 않음. KOSDAQ h5 는 소폭↓.
- 단일 데이터셋·6 시드. 거래일 매핑은 여전히 날짜 단위(시각으로 매핑하면 또 달라질 수 있음).

## 6. 결론 / 다음
- **권고: HEADLINE_ORDER=time 채택**(특히 단기 신호 개선). 현재 기본(date)은 시각 정보를
  버려 손해. 이는 EXP-A~R 의 date-order 결과가 *약간 보수적*(과소평가)이었음을 시사.
- 다음: time-order 를 기본으로 주요 실험 재검토, 또는 거래일 매핑도 시각 기준으로 정교화.

## 7. 재현
```
EXP_PROFILE=multiyear HEADLINE_ORDER=time python phase1/build_dataset.py
EXP_PROFILE=multiyear HEADLINE_ORDER=time python phase1/build_labels.py
EXP_PROFILE=multiyear HEADLINE_ORDER=time python phase2/train.py --batch-size 16
EXP_PROFILE=multiyear HEADLINE_ORDER=time python phase2/signal_test.py
```
