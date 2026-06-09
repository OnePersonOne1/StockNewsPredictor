# EXP-Q: top-30 선택 방식 — 최신순(recency) vs 랜덤(random)

| 항목 | 내용 |
|---|---|
| 상태 | 완료 |
| 날짜 | 2026-06-09 |
| 커밋 | `<this>` |
| 데이터 | 다년. `HEADLINE_SAMPLE=random`(행별 시드 랜덤 30개) vs recent(최신 30, 기본) |
| 산출물 | `phase2/results_multiyear_random/`(체크포인트 gitignore), `signal_seed_sweep.csv` |

## 1. 동기 / 가설
지적: *"top-30 을 뽑을 때 랜덤으로 섞어야 하지 않나?"* 현재는 **최신순 상위 30개**를
결정적으로 사용한다(`dataset.py`). 최신 선택이 **편향**일 수 있고, 랜덤 샘플이 하루를
더 대표할 수 있다. EXP-P 는 *수량*만 봤지 *선택 방식*은 미검 → 직접 비교.

## 2. 설정
- `HEADLINE_SAMPLE=random`: `dataset.py.__getitem__` 에서 하루 헤드라인 중 **행별
  시드(np.random.default_rng(idx)) 랜덤 30개**를 비복원 추출(재현 가능, 정렬 유지).
  recent 는 최신 30개(기본). 그 외 다년 설정 불변. RoBERTa 6 시드 IC.

## 3. 결과 (RoBERTa IC, 6 시드, KOSPI/KOSDAQ h1·h5)

| 선택 방식 | KOSPI h5 IC mean±std | 양수/6 | KOSPI h1 | KOSDAQ h1 | KOSDAQ h5 |
|---|---|---|---|---|---|
| **recent(최신, 현재)** | **+0.146 ± 0.026** | **6/6** | +0.086±0.038 | +0.061±0.022 | +0.068±0.033 |
| random(랜덤) | +0.028 ± **0.062** | 3/6 | +0.075±0.051 | +0.025±0.031 | −0.004±0.022 |

(macro-F1 seed42: random KOSPI 0.265/0.317 vs recent 0.318/0.298 — 혼재·노이즈.)

## 4. 분석 — **최신순이 랜덤보다 낫다(선택은 편향이 아니라 옳은 설계)**
- 신호가 가장 강한 **KOSPI h5 에서 IC 가 +0.146(안정, 6/6 양수) → +0.028(노이즈, 3/6)**
  로 **약화되고 분산이 ~2.4배(0.026→0.062)** 커진다. KOSDAQ h5 도 +0.068 → −0.004.
- 즉 **거래일 직전 최신 30개가 랜덤 30개보다 더 정보적**이다. 최신=시의성=관련성이라는
  가설을 직접 확증.
- **EXP-P 와 수렴**: 더 넣어도(EXP-P) 나쁘고, 랜덤으로 바꿔도(EXP-Q) 나쁘다 →
  **'적고·최신·관련 있는 30개'가 sweet spot.** top-30 최신 선택은 메모리 절충이자 *최적 선택*.

## 5. 한계
- 행별 고정 랜덤(에폭 간 재샘플링=증강 아님). 진짜 per-epoch 랜덤 증강은 미검(다를 수 있음).
- 단일 데이터셋. IC 는 다중 시드로 측정했으나 effect 는 약함(약신호 영역).

## 6. 결론 / 다음
- **랜덤 셔플은 도움이 안 된다 — 최신순이 더 강하고 안정적.** 현재 설계 정당화.
- 다음 후보: per-epoch 랜덤 재샘플링(증강)이 과적합을 줄여 *대표본에서* 도움 되는지.

## 7. 재현
```
EXP_PROFILE=multiyear HEADLINE_SAMPLE=random python phase2/train.py --batch-size 16
EXP_PROFILE=multiyear HEADLINE_SAMPLE=random python phase2/signal_test.py
# recent(기본)와 비교: EXP-D/H (results_multiyear)
```
