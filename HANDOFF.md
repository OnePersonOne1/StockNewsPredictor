# HANDOFF — 작업 인수인계 (RTX 4090 / Linux 세션용)

> 이 문서는 **새 Claude Code 세션**(또는 본인)이 4090 Linux 머신에서 이어서
> 작업할 때 맥락을 잃지 않도록 정리한 것이다. CLI 세션은 머신 간 동기화되지
> 않으므로, 이 저장소가 사실상의 컨텍스트다.

## 1. 이 프로젝트가 뭐냐

- **연구 질문:** 뉴스 헤드라인의 주가 예측력은 얼마나 먼 미래(horizon)까지 유효한가?
- 2024년 BIGKinds 경제·국제 헤드라인 104,920건(한국경제/조선일보/한겨레) +
  KOSPI·KOSDAQ 지수로, 방향 3-class {-1,0,+1} 을 h ∈ {1,5,21,252} 거래일 예측.
- 한국 대학 학부 기초 과목 기말 보고서용. **성능보다 보고서 퀄리티·해석이 중요.**

## 2. 지금까지 한 일 (Windows 머신, GPU 없음)

- **Phase 1 전체를 새로 구축함.** (원래 지시서는 "Phase 1 완성됨"을 전제했으나
  실제로는 코드도 산출물도 없었고, BIGKinds 원본 xlsx + 2024 가격 CSV 뿐이었다.)
  - `phase1/build_prices.py`: pykrx로 가격을 **2025~2026 상반기까지 연장**
    (안 하면 test=2024-12 의 h5/h21/h252 forward return 이 전부 결측).
  - `phase1/build_dataset.py`: 헤드라인 → 거래일 매핑(**T>D 엄격, look-ahead 봉쇄**)
    → forward return.
  - `phase1/build_labels.py`: **σ_h 를 train 에서만** 계산, ε=0.3σ_h ternary 라벨.
  - 산출: `phase1/data/processed/dataset_final.parquet` (**488행**, KOSPI/KOSDAQ 각 244,
    split train406/val42/test40). **이미 저장소에 LFS로 포함됨 → Phase 1 재실행 불필요.**
- **Phase 2 코드 전체 작성** + 일부 검증:
  - `baseline.py`: TF-IDF+LogReg 8셀 표 — **실측 완료**(아래 결과).
  - `model.py`: 마스킹 attention pooling **수학 검증 완료**(패딩부 가중치 0, 합 1).
  - `dataset.py`/`train.py`/`evaluate.py`/`analyze.py`: 작성 완료.

### ⚠️ 아직 검증 못 한 구간 (여기가 4090에서 첫 확인 지점)
Windows 머신에 GPU·transformers 가 없어 **transformer 경유 경로를 실행하지 못함**:
`dataset.py`/`model.py`/`train.py` 의 인코더 연동. → 4090에서 `--smoke` 가 첫 검증.
import/shape 오류가 나면 거기부터 잡으면 된다.

## 3. 4090에서 할 일 (순서대로)

```bash
# (1) 클론 + LFS
sudo apt-get install -y git-lfs   # 없으면
git lfs install
git clone https://github.com/OnePersonOne1/StockNewsPredictor.git
cd StockNewsPredictor
git lfs pull
ls -lh phase1/data/processed/dataset_final.parquet   # ~8.6MB면 정상(134B면 lfs pull)

# (2) 환경 (torch는 CUDA 빌드 먼저!)
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r phase2/requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # True 확인

# (3) 코드 검증 — ★ transformer 경유 첫 실행
python phase2/baseline.py        # (선택) 기저 재확인, GPU 불필요
python phase2/dataset.py         # batch 1개 출력
python phase2/model.py           # forward shape (klue/roberta-base ~440MB)
python phase2/train.py --smoke   # ★ 통합 1-step (여기 통과하면 연동 OK)

# (4) 학습
python phase2/train.py --epochs 1   # ★ train_loss 감소 확인(안 하면 lr 문제)
python phase2/train.py              # full (EPOCHS=4, BATCH 8); 4090이면 --batch-size 16 가능

# (5) 평가 + 해석
python phase2/evaluate.py        # 8셀 표/decay plot/metrics.json
python phase2/analyze.py         # attention 상위 헤드라인

# (6) 결과 회수(선택)
git add phase2/results && git commit -m "add training results" && git push
```

## 4. 보고서 산출물 매핑

| 파일 | 보고서 위치 |
|---|---|
| `phase2/results/test_metrics.tex` | 표 2/3 — 8셀 accuracy·macro-F1 (LaTeX) |
| `phase2/results/figures/horizon_decay.png` | 그림 2 — horizon별 예측력 곡선(baseline 오버레이) |
| `phase2/results/top_attention_headlines.csv` | 표 4/부록 — attention 상위 헤드라인 |
| `phase2/results/metrics.json` | per-class precision/recall, confusion matrix |

## 5. ★ 결과 해석의 핵심 (반드시 반영)

- **h=1, h=5 = 주 결과.** test 라벨이 3-class 로 실제 섞여 의미 있는 평가 가능.
- **h=21, h=252 = 보조 결과.** test 창(2024-12, 지수당 n≈20)이 짧아 forward 수익률이
  **시장 전체 드리프트(2025 회복장)에 지배 → 거의 단일 클래스(+1)로 붕괴**.
  - 그래서 높은 accuracy 는 "잘 맞춤"이 아니라 단일클래스 쏠림의 산물.
    **macro-F1 로 판단**할 것(단일클래스면 ~0.333).
- 메시지 방향: "헤드라인 신호는 **단기(h≤5)에 약하게 존재**, 장기로 갈수록 시장
  추세에 묻힌다." 부정적/중립 결과도 그대로 보고.

### baseline 실측 (참고 lower bound, eval=test)
| index | h=1 | h=5 | h=21 | h=252 |
|---|---|---|---|---|
| KOSPI acc / F1 | 0.40 / 0.279 | **0.55 / 0.356** | 0.00 / 0.000 | 1.00 / 0.333 |
| KOSDAQ acc / F1 | 0.40 / 0.246 | 0.40 / 0.250 | 0.00 / 0.000 | 0.05 / 0.032 |
(무작위 3-class = 0.333. h21/h252 의 0.00·1.00 은 위 단일클래스 붕괴 때문.)

## 6. 절대 하지 말 것 (사전 확정)

- 시계열 random split 금지(temporal 고정), σ_h 를 전체 데이터로 계산 금지(train만).
- 추가 feature engineering(sentiment/FinBERT 등)·외부 API·모델 확대(large) 금지.
- 결과 과장·cherry-pick 금지(부정 결과도 그대로).
- 확정 설계(ε=0.3σ_h, split 경계, encoder=klue/roberta-base) 변경 금지.

## 7. 환경/자격증명 메모

- KRX 가격 재다운로드가 필요할 때만 `KRX_ID`/`KRX_PW` 환경변수 설정(소스 하드코딩 금지).
  단, parquet 가 저장소에 있으므로 보통 불필요.
- 대용량(xlsx/parquet)은 Git LFS 로 관리됨.
