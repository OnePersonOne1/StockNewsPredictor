# Phase 2 — 뉴스 헤드라인 기반 주가 방향 예측 모델

뉴스 헤드라인의 주가 예측력이 **얼마나 먼 미래(horizon)까지 유효한가**를
KOSPI·KOSDAQ, horizon h ∈ {1, 5, 21, 252} 거래일, 3-class ternary
{-1, 0, +1} 예측으로 검증한다.

## 모델 개요

```
e_i   = klue/roberta-base(headline_i)[CLS]         # 헤드라인 임베딩
α_i   = softmax_i( q · e_i )                        # learnable query attention
d_t   = Σ_i α_i e_i  (+ index_emb)                  # 일별 표현 (지수 정보 주입)
ŷ^(h) = softmax( W_h d_t + b_h )                    # horizon별 3-class
L     = mean_h CE( ŷ^(h), y^(h) )                   # 4 horizon multi-task 평균
```

- 일별 헤드라인 중 **최신 30개**(`MAX_HEADLINES`)만 사용 (일평균 ~430건은 메모리 초과)
- 패딩 헤드라인은 attention softmax 에서 **마스킹 제외**
- KOSPI·KOSDAQ **jointly 학습**(헤드라인 표현 공유, 지수는 embedding 으로 구분)

## 의존성 설치

```bash
pip install -r phase2/requirements.txt
# torch 는 반드시 CUDA 빌드로 (requirements.txt 상단 주석 참고)
python -c "import torch; print(torch.cuda.is_available())"   # True 확인
```

## 실행 순서

> 전제: `phase1/data/processed/dataset_final.parquet` 가 존재해야 함.
> 없으면 먼저 Phase 1 실행 (루트 README 참고).

```bash
# 0) (선택) 기저 성능 — GPU 불필요, 가장 먼저 lower bound 확인
python phase2/baseline.py

# 1) 모듈 smoke test (코드 정상 동작 검증)
python phase2/dataset.py            # batch 1개 정상 출력
python phase2/model.py              # forward shape 확인 (인코더 다운로드)
python phase2/train.py --smoke      # 초소형 1-step

# 2) 1 epoch 시험 → train loss 감소 확인
python phase2/train.py --epochs 1

# 3) full 학습 (best val macro-F1 체크포인트 저장)
python phase2/train.py              # 기본 EPOCHS=4, BATCH_SIZE=8

# 4) 최종 평가 → 8셀 표 / confusion / decay plot / metrics.json
python phase2/evaluate.py

# 5) attention 해석 → 일별 상위 5개 헤드라인
python phase2/analyze.py
```

## 산출물

| 파일 | 내용 |
|---|---|
| `results/baseline_metrics.{csv,md}` | TF-IDF+LogReg 기저 8셀 표 |
| `results/test_metrics.{csv,tex}` | RoBERTa 8셀 accuracy/macro-F1 (LaTeX 포함) |
| `results/metrics.json` | per-class precision/recall, confusion matrix 등 |
| `results/figures/horizon_decay.png` | horizon별 예측력 곡선 (그림 2용) |
| `results/top_attention_headlines.csv` | 일별 attention 상위 5개 헤드라인 (표 4용) |
| `data/checkpoints/best.pt` | best val macro-F1 체크포인트 |

## GPU 메모리 가이드

- `MAX_HEADLINES=30`, `MAX_LENGTH=64`, `BATCH_SIZE=8` 기준 약 **8–10GB VRAM**.
  - forward 당 인코딩 시퀀스 = batch×headlines = 8×30 = 240개.
- **OOM 시**: `--batch-size 4` → 그래도 부족하면 `--max-headlines 20`.
- RTX 4090(24GB) 에서는 `--batch-size 16` 까지 여유.

## 결과 해석 주의 (중요)

- **h=1, h=5** 만 test 라벨이 3-class 로 실제 섞여 있어 의미 있는 비교가 가능하다.
- **h=21, h=252** 는 test 창(2024-12, 지수당 n≈20)이 짧아 forward 수익률이
  **시장 전체 드리프트에 지배**되어 거의 **단일 클래스(+1)** 로 붕괴한다.
  → 이 셀의 높은 accuracy 는 "잘 맞춤"이 아니라 단일 클래스 쏠림의 산물이며
    macro-F1 이 이를 드러낸다. 보고서에서는 **보조 결과**로만 다룰 것.
- 따라서 본 연구의 핵심 메시지는 "헤드라인 신호는 단기(h≤5)에 약하게 존재하나
  장기로 갈수록 시장 드리프트에 묻힌다" 이며, 이는 그대로 보고할 부정적/중립적
  결과를 포함한다.
