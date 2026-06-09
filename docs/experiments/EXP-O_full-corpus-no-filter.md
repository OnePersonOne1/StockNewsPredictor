# EXP-O: data/ 폴더 전체 결합 (본체 ∪ IT_section, 필터 없음)

| 항목 | 내용 |
|---|---|
| 상태 | 완료 (예상 밖 발견: RoBERTa 입력이 본체와 동일) |
| 날짜 | 2026-06-09 |
| 커밋 | `<this>` |
| 데이터 | 다년 + `INCLUDE_IT=1` + `all`(필터 없음) = data/ 안 **모든** 파일(32개, dedup 477,443) |
| 산출물 | `phase2/results_multiyear_itaug/`(RoBERTa 는 EXP-D 와 동일), `dataset_2124_itaug.parquet` |

## 1. 동기 / 가설
지금까지 **본체 data/(EXP-D, all)** 와 **본체∪IT_section 의 IT 필터 부분집합(EXP-L,
it/market_it)** 은 했지만, **data/ 폴더의 모든 파일을 필터 없이(all) 통째로** 쓴 적은
없었다(사용자 지적). 이를 채워, IT_section 을 전체에 섞으면 신호가 희석/개선되는지 본다.

## 2. 설정
- `EXP_PROFILE=multiyear INCLUDE_IT=1 HEADLINE_FILTER=all`. 코퍼스 477,443건(본체+IT_section,
  식별자 dedup), **일평균 484.9건**(본체 431 + IT). 모델·라벨 불변. RoBERTa·TF-IDF 평가.

## 3. 결과 — **RoBERTa 는 본체(EXP-D)와 *완전히 동일*, TF-IDF 만 미세 희석**

**핵심 검증**: 결합 코퍼스와 본체 코퍼스의 **일별 top-30 헤드라인이 1966/1966 행 모두
동일**. IT_section 첫 기사는 정렬상 **176번째**에 위치(top-30 밖).

| 모델 | 입력 소비 | 결합-all 결과 |
|---|---|---|
| **RoBERTa** | 일별 **최신 30개만**(MAX_HEADLINES=30) | **EXP-D 와 소수점까지 동일** (macro-F1 KOSPI 0.318/0.298, IC KOSPI h5 +0.182, 다중시드 +0.146±0.026) |
| **TF-IDF** | 하루 **전체** 헤드라인 join | 미세 **희석**: KOSPI h5 0.350→**0.308**, KOSDAQ h5 0.279→0.274 |

## 4. 분석 — **MAX_HEADLINES=30 때문에 비여과 IT_section 은 RoBERTa 에 '투명'하다**
- RoBERTa 는 일별 **최신 30개**만 본다. 본체가 일 431건으로 빽빽해 그 30칸을 가득 채우고,
  IT_section 기사는 정렬상 176번째 이후에 위치 → **단 한 건도 top-30 에 못 든다.** 따라서
  입력 텐서가 본체와 **바이트 단위로 동일**해 학습·예측·IC 가 전부 동일하다(우연 아님).
- 반면 **TF-IDF 는 하루 전체를 쓰므로** IT_section 어휘가 섞여 **약하게 희석**된다
  (KOSPI h5 0.350→0.308) — 일반 IT 뉴스가 잡음으로 작용(EXP-N 과 일관).
- **함의 1 (방법론)**: '데이터를 다 넣었다'와 '모델이 다 봤다'는 다르다. top-30 절단 때문에
  비여과 추가 자료는 RoBERTa 에 무효다. IT 를 모델에 보이게 하려면 **카테고리 필터(EXP-G)**
  로 IT 기사를 top-30 에 띄우거나, MAX_HEADLINES 를 키워야 한다.
- **함의 2 (해석 통합)**: EXP-G 의 `it` 필터가 효과를 낸 메커니즘이 분명해진다 — 필터가
  비-IT 기사를 제거해 IT 기사가 top-30 에 *진입*했기 때문이다. 그냥 더하면(EXP-O) 묻힌다.

## 5. 한계
- news_date 가 **날짜 단위**(시각 없음)라 같은 날 내 정렬은 concat 순서(본체 먼저)에 의존 —
  top-30 이 본체로 채워지는 한 원인. 분 단위 타임스탬프가 있으면 결과가 달라질 수 있음.
- 결합-all 의 RoBERTa 다중시드는 EXP-H 와 동일하므로 재계산 생략(중복).

## 6. 결론 / 다음
- **data/ 폴더 전체(비여과)를 써도 RoBERTa 결과는 본체와 동일**(top-30 절단), TF-IDF 는
  미세 희석. 즉 **비여과 IT_section 추가는 이득 없음**(EXP-N·G 와 정합).
- 의미 있게 '전체'를 쓰려면: (a) MAX_HEADLINES↑, (b) 카테고리/시간 균형 선택, (c) 필터
  (EXP-G). → EXP-O 2단계 후보: **MAX_HEADLINES 확대 시 결합-all 이 달라지는가**.

## 7. 재현
```
EXP_PROFILE=multiyear INCLUDE_IT=1 python phase1/build_dataset.py   # 일 484.9건
EXP_PROFILE=multiyear INCLUDE_IT=1 python phase1/build_labels.py
EXP_PROFILE=multiyear INCLUDE_IT=1 python phase2/baseline.py        # TF-IDF 미세 희석
EXP_PROFILE=multiyear INCLUDE_IT=1 python phase2/train.py --batch-size 16
# top-30 동일성 검증: dataset_2124.parquet vs dataset_2124_itaug.parquet 의 headlines[:30] 비교
```
