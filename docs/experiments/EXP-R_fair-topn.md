# EXP-R: 공정 비교 — TF-IDF·wordcount 도 top-30, 3-class & 이진 모두

| 항목 | 내용 |
|---|---|
| 상태 | 완료 |
| 날짜 | 2026-06-09 |
| 커밋 | `<this>` |
| 데이터 | 다년 test=2024. TF-IDF·wordcount: top-30 vs 전체. RoBERTa: top-30(EXP-D/H 참조) |
| 산출물 | `phase2/results_multiyear/fair_topn.{csv,md}`, `figures/fair_topn.png`; 스크립트 `fair_topn.py` |

## 1. 동기 / 가설
지적: *"TF-IDF·wordcount 는 하루 전체(~430건)를, RoBERTa 는 최신 30건만 → 같은 조건이
아니다."* 정당한 비판. TF-IDF·wordcount 도 **최신 30개로 제한**해 RoBERTa 와 동일 입력으로
맞추고, **3-class macro-F1** 과 **이진 up/down(IC)** 두 지표 모두에서 재비교한다.

## 2. 설정
- `BASELINE_TOPN=30` 으로 baseline/wordcount/`fair_topn.py` 가 일별 최신 30개만 사용.
- `fair_topn.py`: TF-IDF(+LogReg)·wordcount 를 top-30/전체 각각에 대해 3-class(argmax)와
  이진/IC(`metrics_for`) 동시 산출. RoBERTa(top-30)는 저장된 EXP-D(3-class)·EXP-H(이진).

## 3. 결과 — **동일 top-30 입력에서도 '지표별 반전' 유지**

**KOSPI h5 (대표 셀):**

| model (top-30) | 3-class macro-F1 | 이진 IC |
|---|---|---|
| **TF-IDF** | **0.368** | +0.024 |
| RoBERTa | 0.298 | **+0.182** |
| wordcount | 0.275 | +0.015 |

**top-30 제한이 baseline 에 주는 영향(전체→top30):**

| model | macro-F1 KOSPI h5 | IC KOSPI h5 |
|---|---|---|
| TF-IDF | 0.350 → **0.368** | −0.035 → +0.024 |
| wordcount | 0.324 → 0.275 | −0.063 → +0.015 |

(그림 `fair_topn.png`: 좌=macro-F1 TF-IDF 우위, 우=IC RoBERTa 우위.)

## 4. 분석 — **공정하게 맞춰도 결론 불변**
1. **3-class macro-F1: TF-IDF ≥ RoBERTa > wordcount** (KOSPI h5 0.368/0.298/0.275).
   같은 입력에서도 TF-IDF 의 분류 정확도 우위 유지.
2. **이진 IC: RoBERTa ≫ TF-IDF ≈ wordcount** (KOSPI h5 +0.182 / +0.024 / +0.015).
   같은 입력에서도 RoBERTa 의 방향 순위력 우위 유지.
3. **즉 EXP-H 의 '지표별 반전'은 입력 비대칭의 산물이 아니라 실재**한다 — TF-IDF 는
   *분류/클래스 구조*에, RoBERTa 는 *방향 순위*에 강하다(서로 다른 능력).
4. **부수**: TF-IDF 를 top-30 으로 줄여도 **나빠지지 않는다**(KOSPI h5 0.350→0.368, IC
   −0.035→+0.024). 즉 전체 헤드라인의 정보 우위가 TF-IDF 에 큰 이득은 아니었다 →
   비대칭이 TF-IDF 를 *부당하게* 유리하게 한 건 아니다. wordcount 는 텍스트가 줄어
   3-class 가 소폭↓(렉시콘 카운트 기반).

## 5. 한계
- RoBERTa IC 는 단일 시드(seed42, +0.182)로 참조. 다중 시드 평균은 +0.146±0.026(EXP-H).
- top-30 은 최신순(EXP-Q 에서 랜덤보다 우수 확인). 단일 데이터셋.

## 6. 결론 / 다음
- **공정 비교(동일 top-30)에서도 결론 유지**: 3-class=TF-IDF, 방향 IC=RoBERTa. 입력 비대칭은
  결론을 바꾸지 않으며, 두 모델은 *서로 다른 지표에서* 강하다.
- 이로써 "같은 조건이 아니다"라는 우려가 해소된다 — 같은 조건으로 맞춰도 그림은 동일.

## 7. 재현
```
EXP_PROFILE=multiyear python phase2/fair_topn.py     # TF-IDF·wordcount top30/all × 3cls·이진
# 개별: EXP_PROFILE=multiyear BASELINE_TOPN=30 python phase2/baseline.py (3-class top30)
```
