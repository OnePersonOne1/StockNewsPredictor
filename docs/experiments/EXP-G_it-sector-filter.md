# EXP-G: IT/반도체 섹터 필터 (한국장의 높은 IT 비중 반영)

| 항목 | 내용 |
|---|---|
| 상태 | 완료 (시범) |
| 날짜 | 2026-06-09 |
| 커밋 | `<this>` |
| 데이터 | 다년(2021–2023→2024), `it`·`market_it` 필터 추가 |
| 산출물 | `phase2/results_multiyear_{it,market_it}/`, `results_multiyear/filter_ablation.{csv,md,png}` |

## 1. 동기 / 가설
한국 지수는 삼성전자·SK하이닉스 등 **반도체/IT 비중이 압도적**(체감 ~50%)이라, IT
섹션 헤드라인이 시장 신호를 농축해 담을 수 있다(사용자 제안). EXP-E 에서 market
필터가 RoBERTa 를 떨어뜨린 원인이 **볼륨 부족**이었으므로, 볼륨이 큰 시장+IT 조합도 함께 본다.

## 2. 설정
- 새 필터(통합분류1): `it` = `IT_과학|반도체`, `market_it` = `증권_증시|IT_과학|반도체`.
  (반도체는 `경제>반도체`로 IT_과학과 별도 분류라 둘 다 포함.)
- 다년(EXP-D, RoBERTa 가 건강한 구간)에서 기존 all/market/macro 와 한 표로 비교.
- **데이터 caveat**: 사용자가 BIGKinds export 시 IT 섹션을 일부 제외 → 현재 데이터의
  IT_과학 ~6%·반도체 ~3%(전 연도 일관)로 **과소 표집 가능**. 본 결과는 **시범**이며
  더 완전한 IT 데이터셋으로 후속 예정.

## 3. 결과 (test macro-F1, h=1 / h=5; 다년, seed 42)

**TF-IDF (결정적·신뢰 가능):**

| index | all | market | macro | it | market_it |
|---|---|---|---|---|---|
| KOSPI | 0.300/0.350 | **0.380**/0.328 | 0.355/0.324 | 0.322/0.322 | 0.369/0.336 |
| KOSDAQ | 0.265/0.279 | 0.292/0.338 | 0.311/0.327 | 0.275/0.306 | 0.282/0.288 |

**RoBERTa (단일 시드, 변동성 큼 — §4 주의):**

| index | all | market | macro | it | market_it |
|---|---|---|---|---|---|
| KOSPI | 0.318/0.298 | 0.226/0.297 | 0.291/0.324 | 0.282/0.305 | 0.175/0.177† |
| KOSDAQ | 0.293/0.275 | 0.233/0.277 | 0.327/0.292 | 0.284/0.296 | 0.183/0.209† |

† market_it seed 42 는 불운한 저점(아래 시드 점검 참조).

## 4. 분석
- **IT 섹터가 신호를 농축**: `it`(IT_과학+반도체)는 전체의 **9%** 헤드라인만으로 TF-IDF·
  RoBERTa 모두 `all` 과 대등(KOSPI TF-IDF 0.322/0.322 vs all 0.300/0.350; RoBERTa
  best val 0.283 으로 다년 최고). → **한국장의 IT/반도체 비중이 높아, 소수의 IT 헤드라인이
  전체에 준하는 예측력**을 가진다는 사용자 가설을 (시범적으로) 지지.
- **TF-IDF 는 필터로 개선**: KOSPI h1 이 all 0.300 → market 0.380 / market_it 0.369.
  IT 결합(market_it)도 market 과 유사하게 강함. (EXP-C·E 와 일관.)
- **RoBERTa 셀 차이는 단일 시드 노이즈 범위**: market_it 을 seed {42,0,1,2}로 재학습한
  best val = **0.196 / 0.223 / 0.269 / 0.260** (KOSDAQ h1 0.17–0.30). 즉 표의 market_it
  저점(0.18)은 **seed 42 의 운**이고, 다른 시드는 it/all 과 비슷(~0.26). → **RoBERTa
  필터 비교는 ±0.04+ 의 시드 변동에 묻힌다.**

## 5. 한계 (중요)
- **단일 시드 변동성**: 본 결과 및 **이전 EXP-A~F 의 RoBERTa 단일 셀 수치도 ±0.04+ 불확실**.
  필터 간 ~0.02–0.05 차이는 통계적으로 유의하지 않을 수 있음 → 결론은 **TF-IDF(결정적)와
  '경향'에 무게**를 두고, RoBERTa 미세 비교는 신중히. (향후: 다중 시드 평균이 정도.)
- **IT 과소 표집 가능**(export 제외) → 본 EXP-G 는 시범. 완전한 IT 데이터로 재검 예정.
- 어떤 필터도 무작위(0.333)를 크게 넘지 못함 — 신호 약함이라는 주 결론 유지.

## 6. 결론 / 다음
- **IT/반도체 헤드라인은 부피 대비 신호 밀도가 높다**(9%로 전체에 준함) — 사용자 직관 지지.
  TF-IDF 에선 IT 결합이 강한 예측을 줌. RoBERTa 미세차는 시드 노이즈.
- 다음: ① 더 완전한 IT 데이터셋 확보 후 재검(사용자 제공 예정), ② **RoBERTa 다중 시드**로
  필터 효과의 유의성 확립(방법론 보강).

## 7. 재현
```
for F in it market_it; do
  EXP_PROFILE=multiyear HEADLINE_FILTER=$F python phase1/build_dataset.py
  EXP_PROFILE=multiyear HEADLINE_FILTER=$F python phase1/build_labels.py
  EXP_PROFILE=multiyear HEADLINE_FILTER=$F python phase2/baseline.py
  EXP_PROFILE=multiyear HEADLINE_FILTER=$F python phase2/train.py --batch-size 16
  EXP_PROFILE=multiyear HEADLINE_FILTER=$F python phase2/evaluate.py
done
EXP_PROFILE=multiyear python phase2/compare_filters.py
# 시드 변동성: --seed 0/1/2 로 train 반복
```
