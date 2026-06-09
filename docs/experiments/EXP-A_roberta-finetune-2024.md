# EXP-A: RoBERTa fine-tune (2024, MAX_HEADLINES=30) — 주 모델

| 항목 | 내용 |
|---|---|
| 상태 | 완료 |
| 날짜 | 2026-06-09 |
| 커밋 | `62163f2` (학습·결과), `d738a67` (원인 진단) |
| 데이터 | 2024 전체 헤드라인, split train406/val42/test40 |
| 산출물 | `phase2/results/{test_metrics.*,metrics.json,figures/horizon_decay.png}`, 체크포인트 `best.pt`(LFS) |

## 1. 동기 / 가설
헤드라인 [CLS] 임베딩을 learnable-query attention 으로 일별 묶어 horizon별 방향을
예측. 사전학습 RoBERTa(klue/roberta-base) fine-tune 이 TF-IDF 기저를 능가하는지 확인.

## 2. 설정
- 모델: `HeadlineAttentionModel` — 각 헤드라인 [CLS] → query attention pooling →
  index 임베딩 가산 → horizon별 선형 head.
- encoder=klue/roberta-base **전체 fine-tune**, batch 16, 4 epoch, lr 2e-5,
  warmup 0.1, MAX_HEADLINES=30, MAX_LENGTH=64. best val macro-F1(평균) 기준 체크포인트.
- 라벨 ε=0.3σ_h(train), temporal split. (모두 고정 설계)

## 3. 결과
- 학습: train_loss 1.154→1.084(미미한 감소), **best val macro-F1=0.2514**(epoch 2),
  이후 val 악화(학습 정체).
- **test macro-F1**

  | index | h=1 | h=5 | h=21 | h=252 |
  |---|---|---|---|---|
  | KOSPI | 0.111 | 0.207 | 0.000 | 0.333 |
  | KOSDAQ | 0.032 | 0.173 | 0.000 | 0.333 |

- TF-IDF+LogReg 기저(동일 split): KOSPI 0.279/0.356, KOSDAQ 0.246/0.250 (h1/h5)
  → **주 모델이 기저보다 낮음.**
- confusion matrix (열=예측): 모든 셀에서 **단일 클래스만 예측**
  (h1→전부 flat, h5→전부 down). 예: `KOSPI_h1 [[0,9,0],[0,4,0],[0,7,0]]`.

## 4. 분석 (왜 기저보다 낮은가)
단일클래스 붕괴가 직접 원인이며(단일 클래스 예측의 macro-F1 상한 ~0.1–0.2), 불균형
때문이 아니다(train 라벨 137/136/133 거의 균형). 근본 원인:
1. **데이터 부족**: 406행으로 1.1억 파라미터 fine-tune → 거의 학습 안 됨. 소표본에선
   볼록·저분산 TF-IDF+LogReg 가 구조적으로 유리.
2. **attention 붕괴**: 정규화 엔트로피 평균 **1.000**, top-1 lift **1.07×**
   → pooling 이 30개 [CLS]의 사실상 평균. 일별 표현이 거의 일정 → argmax 고정.
   (`phase2/results/attention_analysis.md`, `attention_*` 그림)
3. **정보 비대칭**: TF-IDF 는 하루 전체(~427) 헤드라인, RoBERTa 는 30개만.
4. **val→test 분포 이동**: h5 train down-heavy(156/110/140)·val down-heavy(24/6/12)
   ·test up-heavy(16/5/19) → "down 으로 찍는" 모델이 val 엔 맞고 test 엔 오답.

## 5. 한계
- test 창이 2024-12 한 달(40행)로 짧고, h21/h252 는 거의 단일 클래스 → 신뢰도 낮음.
- 헤드라인에 시장 무관 기사 다수(→ EXP-C 에서 다룸).

## 6. 결론 / 다음
- **음성 결과**: 저자원에서 딥러닝이 단순 기저에 진다. 정직하게 보고.
- 다음: 병목이 (a) 과대 파라미터면 **freeze**(EXP-B), (b) 입력 잡음이면 **필터**
  (EXP-C), (c) 데이터 규모면 **다년 학습**(EXP-D)로 분해해 검증.

## 7. 재현
```
python phase2/baseline.py
python phase2/train.py --batch-size 16
python phase2/evaluate.py
python phase2/attention_analysis.py
```
