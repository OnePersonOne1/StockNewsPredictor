# EXP-D: 2021–2023 학습 → 2024 예측 (다년 데이터)

| 항목 | 내용 |
|---|---|
| 상태 | **완료** |
| 날짜 | 2026-06-09 |
| 커밋 | 파이프라인 `99b9d12`, 실행 `<this>` |
| 데이터 | 2021–2023 헤드라인 학습 / 2024 전체 test (FDR 가격, 자격증명 불필요) |
| 산출물 | `phase2/results_multiyear/`, `dataset_2124.parquet`(gitignore) |

## 1. 동기 / 가설
EXP-A·B·C 가 일관되게 **데이터 규모가 RoBERTa 붕괴의 근본 병목**임을 가리켰다.
학습창을 3년(2021–2023)으로 늘리고 **test 를 2024 전체**로 바꾸면 (a) 단일클래스
붕괴 완화, (b) 짧은 test 창·val→test 이동 문제 해소가 기대된다.

## 2. 설정
- split: **train 1358**(2021-01~2023-09) / val 120(2023-10~12) / **test 488**(2024 전체).
  (기존 2024-only: 406/42/40)
- encoder=klue/roberta-base, ε=0.3σ(train), σ-train-only, base 모델 **불변**.
  batch 16, 4 epoch, lr 2e-5, MAX_HEADLINES=30.
- 헤드라인 423,073건(2021–2024, `data/` 24개 xlsx). 가격은 **FinanceDataReader**
  로 2021–2026 수집(KRX CSV 2024-01-02 와 종가 일치 확인 → pykrx/자격증명 불필요).

## 3. 결과
- 학습이 **실제로 진행됨**: train_loss 1.081→**0.971**(2024-only 은 1.154→1.084 정체),
  val macro-F1 0.135→0.192→0.221→**0.258** 단조 상승.
- **단일클래스 붕괴 해소**: h1 은 3개 클래스 모두 예측, h5/h21/h252 는 up/down
  양방향 예측(flat 만 회피). 2024-only 의 "전부 1개 클래스"와 대조.
- **test macro-F1 (test=2024 전체):**

  | index | 방법 | h=1 | h=5 | h=21 | h=252 |
  |---|---|---|---|---|---|
  | KOSPI | wordcount | 0.306 | 0.324 | 0.354 | 0.232 |
  | | TF-IDF | 0.300 | 0.350 | 0.304 | 0.224 |
  | | **RoBERTa** | **0.318** | 0.298 | 0.325 | 0.277 |
  | KOSDAQ | wordcount | 0.318 | 0.279 | 0.294 | 0.227 |
  | | TF-IDF | 0.265 | 0.279 | 0.291 | 0.294 |
  | | **RoBERTa** | 0.293 | 0.275 | 0.317 | 0.288 |

- **attention 부활**: 정규화 엔트로피 1.000→**0.768**, top-1 lift 1.07×→**7.85×**,
  top-5 mass 0.18→**0.64**. learnable query 가 헤드라인을 실제로 차별화하기 시작.

## 4. 분석
- **가설 확정**: 데이터 부족이 EXP-A 의 예측 붕괴와 attention 붕괴를 **둘 다** 유발했고,
  3년 데이터가 **둘 다** 해소. RoBERTa 가 TF-IDF·wordcount 와 대등해지고 일부 셀에서
  최고(KOSPI h1 0.318, KOSDAQ h21 0.317).
- **그러나 헤드라인 신호 자체는 약함**: 세 방법 모두 0.27–0.35 로 무작위(0.333)
  **근처에 밀집**, 어느 것도 분명히 상회하지 못함. 이제 이 결론은 *모델 결함의
  산물이 아니라* 건강한 모델 + 제대로 된 test(488행, 3-class 실분포) 위에서 얻은
  **정직하고 견고한 결과**다.
- h21/h252 도 2024-only 의 퇴화(0.000/0.333)에서 벗어나 의미 있는 값(~0.28–0.35)을
  가짐 → test 창 확대의 효과.

## 5. 한계
- 2024 말 행의 h252 forward 는 여전히 2025 상승장으로 넘어감(드리프트) → h252 는
  보조 결과. 가격은 FDR(=KRX) 단일 소스.
- 신호가 약해 절대 성능은 무작위 부근 — "예측 가능성이 낮다"는 결론은 유지.

## 6. 결론 / 다음
- **핵심 교훈**: "딥러닝이 진다"는 EXP-A 결론은 데이터 규모의 산물이었다. 데이터를
  늘리면 RoBERTa 는 정상 학습하나, **헤드라인의 주가 방향 예측력은 본질적으로 약하다**
  (모든 방법이 무작위 부근). 이것이 본 연구의 주된 정직한 결론.
- 다음(EXP-E 후보): 다년 + 관련성 필터 결합(`EXP_PROFILE=multiyear
  HEADLINE_FILTER=market`) — 잡음 제거가 다년에서도 신호를 끌어올리는지.

## 7. 재현
```
EXP_PROFILE=multiyear python phase1/build_prices.py     # FDR, 자격증명 불필요
EXP_PROFILE=multiyear python phase1/build_dataset.py
EXP_PROFILE=multiyear python phase1/build_labels.py
EXP_PROFILE=multiyear python phase2/baseline.py
EXP_PROFILE=multiyear python phase2/train.py --batch-size 16
EXP_PROFILE=multiyear python phase2/evaluate.py
EXP_PROFILE=multiyear python phase2/attention_analysis.py
EXP_PROFILE=multiyear python phase2/wordcount_baseline.py
EXP_PROFILE=multiyear python phase2/compare_methods.py
```
