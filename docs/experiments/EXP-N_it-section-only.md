# EXP-N: IT_section 단독 실험 (본체 data/ 미사용)

| 항목 | 내용 |
|---|---|
| 상태 | 완료 |
| 날짜 | 2026-06-09 |
| 커밋 | `<this>` |
| 데이터 | 다년 + `IT_ONLY=1` (코퍼스 = data/IT_section 만, 115,232건) |
| 산출물 | `phase2/results_multiyear_itonly/`, `dataset_2124_itonly.parquet`(gitignore) |

## 1. 동기 / 가설
지금까지 IT 관련 입력은 **(a) 본체 econ/intl 뉴스의 IT 카테고리 부분집합**(EXP-G `it`),
**(b) 본체 ∪ IT_section**(EXP-L) 두 가지였다. 사용자 요청으로 **IT_section 단독**(순수
IT 섹션 코퍼스)을 분리해, "IT 기사만으로 지수 방향이 얼마나 잡히는가"를 본체 기반
실험과 대조한다.

## 2. 설정
- `IT_ONLY=1` 시 build_dataset 가 본체 대신 `data/IT_section/*.xlsx`(8개)만 사용.
  코퍼스 115,232건(IT_과학 59%, 경제 25%, 사회·문화·국제 등), **일평균 117건**, 다년
  split(train 1358/val 120/test 488), 모델·라벨·ε 불변. RoBERTa 6 시드(IC).

## 3. 결과 (다년, test=2024 전체)

**RoBERTa macro-F1 (seed 42):**

| index | h1 | h5 | h21 | h252 |
|---|---|---|---|---|
| KOSPI | 0.165 | 0.181 | 0.177 | 0.064 |
| KOSDAQ | 0.169 | 0.168 | 0.217 | 0.225 |

**RoBERTa IC (KOSPI h5, 6 시드) vs 본체 비교:**

| 입력 구성 | macro-F1 KOSPI h5 | KOSPI h5 IC |
|---|---|---|
| 본체 data 전체 (EXP-D) | 0.298 | **+0.146 ± 0.026** (6/6 양수) |
| IT_section **단독** (EXP-N) | **0.181** | **+0.035 ± 0.033** (5/6 양수) |

(TF-IDF macro-F1 도 IT단독 KOSPI ~0.17 로 본체(~0.35) 대비 크게 낮음.)

## 4. 분석 — **IT_section 단독은 약한 예측기 (경제 프레이밍이 핵심)**
- **macro-F1 이 무작위(0.333)에도 못 미치고**(0.17~0.18), **방향 IC 도 데이터-전체의
  ~1/4**(+0.035 vs +0.146)로 미약하다. IC 는 5/6 시드 양수로 *방향은 맞지만* 크기가 작다.
- **해석**: BIGKinds IT_section 은 모바일·과학·콘텐츠·보안·인터넷 등 **일반 IT/과학
  뉴스**가 다수라 **시장과 직접 관련이 낮다.** 예측 신호는 'IT 라는 주제'가 아니라
  **경제·금융 프레이밍**(증권·실적·금리·매크로; EXP-I 의 고-attention 키워드와 일치)에서
  나온다.
- **EXP-G 의 중요한 정교화**: EXP-G 에서 `it` 필터(전체의 9%로 전체에 준함)가 신호를
  냈던 것은 **경제 뉴스 *안의* IT/반도체 기사**(증권·산업 맥락)였기 때문이다. **순수
  IT_section 코퍼스(EXP-N)는 그 맥락이 없어 훨씬 약하다** → "IT 비중이 신호"라는 직관은
  *경제 프레이밍을 동반한 IT* 에 한정된다.

## 5. 한계
- IT_section 도 ~200자 스니펫·동일 3개 매체. 단일 데이터셋. test 488행 공통.
- KOSPI h5 IC 가 약하지만 5/6 양수 → 미약한 신호의 존재는 부정 못 함(단 크기 무의미 수준).

## 6. 결론 / 다음
- **IT_section 단독 ≪ 본체 econ/intl 코퍼스.** 시장 방향 신호는 *일반 IT 뉴스*가 아니라
  *경제·금융 맥락의 기사*에 있다. (입력 구성 비교: 본체전체 ≳ 본체내 IT부분집합 ≫ IT단독.)
- 다음(후보): 필터/본문/IT 축의 다중 시드 IC(EXP-M 2단계), 또는 경제 카테고리 정제
  강화.

## 7. 재현
```
EXP_PROFILE=multiyear IT_ONLY=1 python phase1/build_dataset.py
EXP_PROFILE=multiyear IT_ONLY=1 python phase1/build_labels.py
EXP_PROFILE=multiyear IT_ONLY=1 python phase2/train.py --batch-size 16
EXP_PROFILE=multiyear IT_ONLY=1 python phase2/evaluate.py
EXP_PROFILE=multiyear IT_ONLY=1 python phase2/signal_test.py
```
