# EXP-X: 2024 완전 held-out 재검증 (삼성전자, train 2021–22 / val 2023 / test 2024+2025)

| 항목 | 내용 |
|---|---|
| 상태 | 완료 |
| 날짜 | 2026-06-10 |
| 커밋 | `<this>` |
| 데이터 | `samsung_cv` 프로필: 삼성전자(005930), 뉴스 data/Samsung_Electronics(2021–2025) |
| split | **train 2021–2022 / val 2023 / test 2024+2025** (2024 도 학습·val 모두 미사용) |
| 산출물 | `phase2/results_samsung_cv_ordtime{,_bin}/`, `…_bin/exp_w_2025.{md,csv}`, `figures/exp_w_2025.png` |

## 1. 동기 / 가설
EXP-W 는 **val=2024** 로 best-val epoch 을 골랐다. 사용자가 정확히 지적했듯, 그러면 "30 epoch
중 2024 에 가장 잘 맞은 모델"이 선택되어 **2024 적중률(h5 0.590)이 다소 낙관적**일 수 있다
(2024 가 backprop 에 들어간 것은 아니지만 체크포인트 선택에는 쓰임). 본 실험은 이를 분리한다:

> **2024 를 val 에서도 빼고**(val=2023), 2024 를 **완전 held-out test** 로 둘 때도
> ① 2024 의 방향 적중 우위가 유지되는가? ② 2025 의 레짐-이동 실패 결론은 그대로인가?

즉 EXP-W 의 두 결론(2024 실력 / 2025 예외성) 중 **무엇이 val 선택의 산물이고 무엇이 견고한지**
가려낸다. (사용자 지시: "train 2021-22 / val 2023 / test 2024+2025 로 재학습.")

## 2. 설정
- `samsung_cv` 프로필 신설(`config.py`): split_bounds train(21–22)/val(23)/test(24–25),
  dataset_samsung_cv.parquet. 그 외는 EXP-V/W 와 동일 — mh64(전체 헤드라인), 시각정렬,
  batch 8, **30 epoch**, 3-class·binary 둘 다, TF-IDF(top64) 동일입력 비교.
- 분석 스크립트 버그 수정: `analyze_2025.py` 의 `BIN_CKPT` 가 EXP-W 체크포인트로 **하드코딩**
  돼 있던 것을 config 의 `BEST_CKPT`(현재 프로필 체크포인트)로 교체 — 최초 실행분은 옛 모델을
  로드했어 폐기하고, **samsung_cv 모델로 재산출**한 수치를 아래 보고한다.

## 3. 결과

**(a) 학습** — binary val best **0.546 @ epoch 5**, 3-class val best **0.338**.
3-class test(2024+2025 혼합) macro-F1: RoBERTa h1 0.331 / h5 0.360 / h21 0.358 / h252 0.448,
TF-IDF(top64) h1 0.368 / h5 0.314 / h21 0.275 / h252 0.361 — **모두 무작위(0.333) 부근**
(지수·EXP-V 와 동일한 그림).

**(b) binary 모델(train 2021–22) 연도별 방향 적중률 vs '항상 up'** (핵심):

| split | 연도 | h | n | 모델적중 | '항상 up'적중 | 모델 up예측율 | 실제 up율 |
|---|---|---|---|---|---|---|---|
| val | 2023 | 5 | 245 | 0.543 | 0.535 | 58% | 53% |
| **test** | **2024** | **5** | 244 | **0.520** | 0.426 | 38% | 43% |
| **test** | **2025** | **5** | 242 | **0.500** | **0.620** | **33%** | **62%** |
| test | 2024 | 1 | 244 | 0.500 | 0.430 | 45% | 43% |
| test | 2025 | 1 | 242 | 0.574 | 0.541 | 58% | 54% |

**(c) EXP-W 대비 — 2024 h5 적중률의 변화:**

| 실험 | train | val | 2024 지위 | **2024 h5 적중** | 2025 h5 적중 |
|---|---|---|---|---|---|
| EXP-W | 2021–2023 | **2024** | val(선택에 사용) | **0.590** | 0.442 |
| **EXP-X** | 2021–2022 | 2023 | **완전 held-out** | **0.520** | 0.500 |
| (기준) '항상 up' | | | | 0.426 | 0.620 |

## 4. 분석

### 4.1 2024 의 "실력"은 일부 과대평가였다 — 그러나 사라지진 않는다
- EXP-W 의 강한 **0.590** 은 (ⓐ val=2024 체크포인트 선택의 낙관 + ⓑ 학습기에 2023 한 해가
  더 있었음)이 **합쳐진 값**이며, 두 효과는 본 설계로 완전히 분리되진 않는다(교란).
- 2024 를 완전 held-out 으로 두고 학습기도 2021–22 로 줄이면 **0.520 으로 하락**. 여전히
  '항상 up'(0.426)보다 위, 무작위(0.5) 수준 → **약한 above-baseline 신호는 남지만**, "무작위를
  뚜렷이 능가"라던 EXP-W 표현은 **"무작위 부근, 항상-up 보다 소폭 우위"로 약화**해야 정직하다.
- 교훈: **val 선택은 backprop 이 아니어도 평가를 낙관 편향**시킬 수 있다(사용자 지적이 옳았다).
  EXP-W 의 2024 수치는 보조 근거로만 쓰고, 핵심 결론은 2024 에 의존하지 않아야 한다.

### 4.2 2025 레짐-이동 실패 결론은 **견고하다** (설계 무관)
- 두 설계 모두 2025 에서 모델이 **up 을 24–33% 만 예측**(실제 62%) → '항상 up'(0.62)에
  체계적으로 진다. 학습기(하락 우세)에 맞춰진 모델이 **유례없는 +124.5% 폭등장**에서 잘못된
  쪽에 선다 — 분포 이동(distribution shift)의 직접 증거이며, **val 을 2023 으로 바꿔도 동일**.
- 즉 EXP-W 의 핵심(=2025 예외성)은 체크포인트 선택과 무관하게 성립한다. EXP-W §4 한계에서
  "핵심 결론은 2024 수치에 의존하지 않는다"고 적은 것을 **EXP-X 가 실증**했다.

## 5. 한계
- **ⓐ val 선택 효과와 ⓑ 학습데이터 1년 감소가 교란** — 0.59→0.52 하락분을 둘로 못 가른다.
  순수 val 효과만 보려면 train 을 2021–23 로 고정하고 val 만 2024↔2023 으로 바꿔야 하나,
  그러면 2023 이 train·val 양쪽이라 또 다른 누설. 완전 분리는 별도 K-fold 가 필요(과제 범위 밖).
- 단일 시드. test=2024+2025 라도 두 해 각각은 단년(레짐 의존). h21·h252 는 2025 forward 가
  2026+ 가격을 요구해 NaN 多(신뢰 낮음, ignore 처리).
- **binary `horizon_decay.png` 의 h=252=0.646 은 예측력 아님 — 라벨이 시간순 단일 전환(블록 2개,
  유효 독립표본 ≈2)이라 생긴 자기상관 신기루.** 셔플(0.646→0.50)·shift-null(p=0.12, null
  최댓값>real)로 검증, 모든 horizon 비유의 → **[[EXP-Y]] 참조.** 메인 메시지는 h=1/5/21≈0.50.
- 산출 파일명은 스크립트 공유로 `exp_w_2025.*` 이지만 **폴더가 `results_samsung_cv_ordtime_bin`**
  이라 EXP-W(`results_samsung_ordtime_bin`)와 구분된다(내용은 EXP-X).

## 6. 결론 / 다음
- **2024 의 방향 우위는 일부 낙관 편향**(0.59→0.52)이었다 — 사용자의 방법론 지적이 타당.
  그래도 '항상 up' 은 넘으므로 약신호는 잔존.
- **2025 의 예외적 폭등장이 OOS 실패의 원인이라는 결론은 설계와 무관하게 견고**(up 33% 예측 vs
  실제 62%). EXP-V→W→X 로 이 결론은 누적 검증됨.
- 다음(있다면): 롤링 재학습(매년 갱신)으로 2025 포함 재검, 또는 드리프트(클래스 사전확률) 보정.
  본 단일종목 라인은 EXP-X 로 마무리하고, 남은 것은 지수 전체-데이터 30 epoch 최종 실험(EXP-U).

## 7. 재현
```
EXP_PROFILE=samsung_cv python phase1/build_prices.py
EXP_PROFILE=samsung_cv HEADLINE_ORDER=time python phase1/build_dataset.py
EXP_PROFILE=samsung_cv HEADLINE_ORDER=time python phase1/build_labels.py
# binary
EXP_PROFILE=samsung_cv HEADLINE_ORDER=time BINARY=1 python phase2/train.py --batch-size 8 --max-headlines 64 --epochs 30
EXP_PROFILE=samsung_cv HEADLINE_ORDER=time BINARY=1 python phase2/evaluate.py
EXP_PROFILE=samsung_cv HEADLINE_ORDER=time BINARY=1 python phase2/analyze_2025.py   # 연도별(수정된 BEST_CKPT)
# 3-class + TF-IDF
EXP_PROFILE=samsung_cv HEADLINE_ORDER=time python phase2/train.py --batch-size 8 --max-headlines 64 --epochs 30
EXP_PROFILE=samsung_cv HEADLINE_ORDER=time python phase2/evaluate.py
EXP_PROFILE=samsung_cv HEADLINE_ORDER=time BASELINE_TOPN=64 python phase2/baseline.py
```
