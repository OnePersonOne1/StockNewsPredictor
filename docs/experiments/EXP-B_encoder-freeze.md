# EXP-B: Encoder freeze + 헤드라인 확대 (MAX_HEADLINES=100)

| 항목 | 내용 |
|---|---|
| 상태 | 완료 (부분 성공) |
| 날짜 | 2026-06-09 |
| 커밋 | `d738a67` |
| 데이터 | 2024 전체, split train406/val42/test40 |
| 산출물 | `phase2/results/{roberta_freeze_metrics.csv,roberta_freeze_compare.md}`, 체크포인트 `freeze_mh100.pt`(gitignore) |

## 1. 동기 / 가설
EXP-A 의 단일클래스 붕괴가 '소표본 대비 과대 파라미터' 때문이라는 가설을 검증.
**encoder 를 동결**해 사전학습 특징만 쓰고 작은 head(~1.2만 파라미터)만 학습하면
저분산으로 붕괴가 완화되어야 한다. 동시에 헤드라인 30→100 으로 TF-IDF 와의 정보
비대칭을 축소.

## 2. 설정
- encoder=klue/roberta-base **freeze**(requires_grad=False), MAX_HEADLINES=100.
- encoder 출력이 상수이므로 [CLS] 임베딩을 **1회 캐시** 후 query/index_emb/head 만
  학습(lr 1e-3, 30 epoch, batch 32). → 학습 수초.
- 구현: `model.pool_and_classify()` 분리 + frozen 시 encoder `no_grad`(메모리 절약,
  수학 동일). unfrozen 동작 불변.

## 3. 결과
- **best val macro-F1 0.2514 → 0.3054** (가설대로 학습 안정화).
- **test macro-F1 비교 (h=1 / h=5):**

  | index | TF-IDF | RoBERTa(ft,mh30) | RoBERTa(freeze,mh100) |
  |---|---|---|---|
  | KOSPI | 0.279 / 0.356 | 0.111 / 0.207 | 0.087 / 0.190 |
  | KOSDAQ | 0.246 / 0.250 | 0.032 / 0.173 | **0.207** / 0.173 |

- 예측 클래스 수(1=붕괴): KOSPI_h1 만 2, 나머지 7개 셀 1.

## 4. 분석
- **부분 성공**: val 개선 + **KOSDAQ h1 0.032→0.207**(큰 완화), KOSPI_h1 붕괴 일부 해소.
- **그러나 TF-IDF 미달**, 대부분 셀 여전히 단일클래스. 잔여 원인:
  1. **데이터 규모가 근본 병목**: freeze 는 분산만 줄일 뿐 신호를 만들지 못함.
  2. **동결 [CLS] 한계**: klue/roberta-base 의 [CLS] 는 fine-tune 없이는 좋은 문장
     임베딩이 아님 → 평균 pooling 한 일별 특징의 변별력 낮음.
  3. **val→test 이동**: val 기준 best 가 test 국면(12월)과 어긋나 0.305 개선이
     test 로 전이되지 못함.

## 5. 한계
- 동결이라도 [CLS] 품질·소표본·짧은 test 창 한계는 그대로.

## 6. 결론 / 다음
- 병목은 '과대 파라미터'만이 아니라 **데이터 규모**임이 확인됨(부분 완화에 그침).
- 다음: 입력 잡음 제거(EXP-C), 데이터 확대(EXP-D).

## 7. 재현
```
python phase2/experiment_freeze.py            # mh100, freeze
python phase2/experiment_freeze.py --max-headlines 30   # 대조
```
