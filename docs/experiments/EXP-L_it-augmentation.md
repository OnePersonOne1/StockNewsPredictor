# EXP-L: IT 자료 보강 (data/IT_section 병합)

| 항목 | 내용 |
|---|---|
| 상태 | 완료 |
| 날짜 | 2026-06-09 |
| 커밋 | `<this>` |
| 데이터 | 다년 + `INCLUDE_IT=1` (본체 + data/IT_section, 식별자 dedup), 필터 it·market_it |
| 산출물 | `phase2/results_multiyear_{it,market_it}_itaug/`, `dataset_2124_*_itaug.parquet`(gitignore) |

## 1. 동기 / 가설
EXP-G 는 사용자가 BIGKinds export 시 IT 섹션을 일부 제외해 **IT 과소표집** 상태였다.
사용자가 `data/IT_section/`(2021–2024 IT 범주 115,232건)을 추가 제공 → 본체와 병합(식별자
dedup)해 **충분한 IT 커버리지**에서 IT 섹터 신호가 강해지는지 재검(EXP-G 의 정식판).

## 2. 설정 & 데이터 처리
- `INCLUDE_IT=1` 시 build_dataset 가 `data/IT_section/*.xlsx`(8개)를 본체 24개에 더하고
  **뉴스 식별자로 중복 제거**. 그 외 다년 설정·모델·필터 정의 불변.
- **발견·수정한 버그**: `뉴스 식별자`(긴 문자열 ID)를 pandas 가 **float 로 읽어 정밀도가
  깨져** 중복이 붕괴(538k→12.8k)했다. `dtype=str` 강제로 수정 → 정상 dedup(538,305→
  **477,443**, 중복 ~11% 제거).
- **효과**: IT 필터 헤드라인 볼륨이 **2배** (it 일 39.6→**81.8건**; market_it 일 ~55→
  더 많음). EXP-G 가 우려한 'market/it 필터의 낮은 볼륨'(RoBERTa attention 손해)을 해소.

## 3. 결과 (다년, seed 42)

**TF-IDF macro-F1 (h=1 / h=5):**

| 필터 | KOSPI | KOSDAQ |
|---|---|---|
| it (보강) | 0.333 / 0.271 | 0.294 / 0.260 |
| **market_it (보강)** | **0.387** / 0.305 | 0.322 / **0.350** |

**RoBERTa IC (방향 신호, h=5):**

| 필터 | KOSPI h5 | KOSDAQ h5 |
|---|---|---|
| it (보강) | **−0.135*** | −0.088 |
| market_it (보강) | **+0.108** | **+0.137*** |

(*p<.05. RoBERTa macro-F1: it KOSPI 0.288/0.239, market_it 0.269/0.298.)

## 4. 분석
- **TF-IDF 는 IT 보강에서도 강함**: market_it KOSPI h1 **0.387** 은 전 실험 통틀어 최상위.
  IT 결합이 시장 카테고리와 함께 bag-of-words 에 유용한 어휘를 더한다(EXP-C·E·G 와 일관).
- **RoBERTa IC 는 단일 시드에서 불안정·부호 엇갈림**: 같은 IT 보강인데 it 는 KOSPI h5
  IC −0.135(유의), market_it 는 +0.108. **부호가 뒤집힌다** → 이는 IT 보강의 효과라기보다
  **RoBERTa 단일 시드 변동**(EXP-G·H 에서 ±0.04, 부호 flip 가능)의 발현. 볼륨을 2배로
  늘려도 단일 시드로는 IC 의 방향조차 확정할 수 없다.
- → **IT 보강이 RoBERTa 방향 신호를 강화한다고 단정 불가**(단일 시드). 다중 시드 평균이
  필요. 이 불안정성은 사용자의 다음 지시(전 실험 이분법/IC 재평가, 원인 탐색)의 동기와 직결.

## 5. 한계
- **단일 시드** — RoBERTa IC 결론 불가(부호 flip). 다중 시드 재평가 필요(→ EXP-M).
- IT_section 도 ~200자 스니펫·동일 3개 매체. 식별자 dedup 후 볼륨 2배지만 test 는 동일 488행.

## 6. 결론 / 다음
- 데이터 측면: IT 보강으로 IT 커버리지 정상화(볼륨 2배), dedup 버그 수정. TF-IDF 는 견조.
- 모델 측면: **단일 시드 RoBERTa IC 로는 IT 보강 효과 판정 불가** → **EXP-M: 전 실험을
  이분법/IC 로 다중 시드 재평가**해 어떤 조건이 방향 신호를 내는지 원인 규명.

## 7. 재현
```
for F in it market_it; do
  EXP_PROFILE=multiyear INCLUDE_IT=1 HEADLINE_FILTER=$F python phase1/build_dataset.py
  EXP_PROFILE=multiyear INCLUDE_IT=1 HEADLINE_FILTER=$F python phase1/build_labels.py
  EXP_PROFILE=multiyear INCLUDE_IT=1 HEADLINE_FILTER=$F python phase2/train.py --batch-size 16
  EXP_PROFILE=multiyear INCLUDE_IT=1 HEADLINE_FILTER=$F python phase2/evaluate.py
  EXP_PROFILE=multiyear INCLUDE_IT=1 HEADLINE_FILTER=$F python phase2/signal_test.py
done
```
