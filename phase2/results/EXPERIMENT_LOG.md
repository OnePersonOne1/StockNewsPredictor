# Phase 2 실험·실패 분석 로그

> 보고서의 "방법" 및 "한계·고찰" 절을 위한 과정 기록. **부정/부분 실패 결과를
> 숨기지 않고 원인까지 분석**한다. 고정 설계(encoder=klue/roberta-base, ε=0.3σ,
> temporal split)는 전 실험에서 불변.

## 0. 데이터 규모와 라벨 분포 (모든 분석의 출발점)

| split | n | 기간 | h1 (down/flat/up) | h5 (down/flat/up) |
|---|---|---|---|---|
| train | 406 | 2024-01~10 | 137/136/133 | 156/110/140 |
| val   | 42  | 2024-11    | 19/13/10    | 24/6/12 |
| test  | 40  | 2024-12    | 18/5/17     | 16/5/19 |

- train 라벨은 ε=0.3σ 설계로 **거의 균형** → 이후의 단일클래스 붕괴는 불균형이 아님.
- **val 과 test 의 분포가 다르다.** 특히 h5: val 은 down 우세(24/42), test 는 up
  우세(19/40). 연속한 두 달(11월 vs 12월) 사이 시장 국면이 바뀐 결과.
  → "val 에서 좋은 모델"이 "test 에서 좋은 모델"과 어긋나는 구조적 원인.

## 1. 실험 A — 주 모델: RoBERTa fine-tune, MAX_HEADLINES=30

설정: 전체 fine-tune, batch 16, 4 epoch, lr 2e-5. best val macro-F1=**0.2514**(epoch 2).

### 결과: test 에서 단일 클래스로 붕괴 (실패)
confusion matrix (열=예측, 순서 down/flat/up):

| cell | matrix | 예측 클래스 수 |
|---|---|---|
| KOSPI_h1 | `[[0,9,0],[0,4,0],[0,7,0]]` | **1 (전부 flat)** |
| KOSDAQ_h1 | `[[0,9,0],[0,1,0],[0,10,0]]` | **1 (전부 flat)** |
| KOSPI_h5 | `[[9,0,0],[3,0,0],[8,0,0]]` | **1 (전부 down)** |
| KOSDAQ_h5 | `[[7,0,0],[2,0,0],[11,0,0]]` | **1 (전부 down)** |

test macro-F1: KOSPI h1/h5 = 0.111/0.207, KOSDAQ = 0.032/0.173 → **TF-IDF(0.279/0.356,
0.246/0.250) 보다 낮음.**

### 원인 분석
1. **데이터 부족**: 406행으로 1.1억 파라미터 fine-tune → train_loss 1.154→1.084 로
   거의 안 떨어지고 epoch 2 이후 val 악화(학습 정체). 소표본에서는 볼록·저분산인
   TF-IDF+LogReg 가 구조적으로 유리.
2. **attention 붕괴**: 정규화 엔트로피 평균=1.000, top-1 lift=1.07× → pooling 이
   사실상 30개 [CLS] 의 **평균**. 일별 표현이 날마다 거의 일정 → 선형 head 출력이
   상수에 가까움 → argmax 가 한 클래스에 고정. (`attention_analysis.py` 참조)
3. **정보 비대칭**: TF-IDF 는 하루 전체(~427개) 헤드라인을 보는데 RoBERTa 는 30개만.
4. **val→test 이동(§0)**: ft 가 h5 에서 'down' 단일 예측 → val(down 우세)엔 맞지만
   test(up 우세)엔 대량 오답.

## 2. 실험 B — 가설 검증: encoder freeze + MAX_HEADLINES=100

가설: 붕괴의 1차 원인이 '소표본 대비 과대 파라미터'라면, **encoder 를 동결**해
사전학습 특징만 쓰고 작은 head(~1.2만 파라미터)만 학습하면 저분산으로 붕괴가
완화되어야 한다. 동시에 헤드라인 30→100 으로 TF-IDF 와의 정보 비대칭을 축소.
(encoder 동결 시 [CLS] 임베딩을 1회 캐시 → head 학습 수초. lr 1e-3, 30 epoch.)

### 결과: val 은 개선, test 는 혼재 — **부분 성공/부분 실패**

| index | method | h=1 | h=5 | h=21 | h=252 |
|---|---|---|---|---|---|
| KOSPI | TF-IDF | **0.279** | **0.356** | 0.000 | 0.333 |
| | RoBERTa(ft, mh30) | 0.111 | 0.207 | 0.000 | 0.333 |
| | RoBERTa(freeze, mh100) | 0.087 | 0.190 | 0.263 | 0.333 |
| KOSDAQ | TF-IDF | 0.246 | 0.250 | 0.000 | 0.032 |
| | RoBERTa(ft, mh30) | 0.032 | 0.173 | 0.000 | 0.333 |
| | RoBERTa(freeze, mh100) | **0.207** | 0.173 | 0.000 | 0.000 |

- **best val macro-F1 0.2514 → 0.3054 로 상승** (가설대로 학습 안정화).
- **KOSDAQ h1 0.032 → 0.207** 로 ft 대비 크게 개선(TF-IDF 0.246 에 근접). KOSPI_h1
  은 예측 클래스가 1→2 로 늘어 붕괴가 일부 풀림.
- 그러나 **대부분 셀은 여전히 단일클래스**(아래)이고, **종합적으로 TF-IDF 를 넘지
  못함**. KOSPI h1 은 오히려 소폭 하락(0.111→0.087).

예측 클래스 수(1=붕괴): KOSPI h1=2, 그 외 7개 셀 모두 1.

### 왜 완전히 해결되지 않았나 (실패의 잔여 원인)
1. **데이터 규모가 근본 병목**: freeze 는 분산을 줄였을 뿐 신호를 만들지 못한다.
   406행으로는 헤드라인→방향의 약한 신호를 안정적으로 학습하기 어렵다.
2. **동결 [CLS] 의 한계**: klue/roberta-base 의 [CLS] 는 fine-tune 없이는 좋은
   문장 임베딩이 아니다(RoBERTa 계열 공통). 평균 pooling 한 일별 특징의 변별력이 낮음.
3. **val→test 분포 이동(§0)**: val 기준 best 체크포인트가 test 국면(12월 up 우세)과
   어긋남 → val 0.305 의 개선이 test 로 이어지지 못함. 데이터가 한 해(2024)뿐이라
   test 창이 짧고 시장 드리프트에 지배되는 한계가 그대로 노출.

## 3. 종합 결론 (보고서 메시지)

- **헤드라인 신호는 단기(h≤5)에 약하게만 존재**하고, 어떤 모델도 안정적으로
  무작위(0.333)를 넘지 못한다.
- **저자원(=406행) 환경에서는 단순 TF-IDF+LogReg 가 RoBERTa 보다 우수**하다.
  freeze+헤드라인 확대로 RoBERTa 의 붕괴를 *부분* 완화했으나 TF-IDF 를 넘진 못했다 —
  이는 모델 결함이라기보다 **데이터 규모·기간(2024 단년)·약한 신호**의 한계다.
- h=21/h=252 는 test 가 거의 단일 클래스라 통계적 의미가 없음(앞서 명시).

### 재현
```
python phase2/baseline.py            # TF-IDF
python phase2/train.py --batch-size 16   # 주 모델(ft, mh30)
python phase2/evaluate.py
python phase2/experiment_freeze.py   # 실험 B (freeze, mh100)
python phase2/attention_analysis.py  # attention 붕괴 진단
python phase2/compare_methods.py     # 방법론 비교표/그림
```
관련 산출: `roberta_freeze_metrics.csv`, `roberta_freeze_compare.md`,
`method_comparison.*`, `attention_*`.
