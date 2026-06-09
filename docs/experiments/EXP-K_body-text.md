# EXP-K: 본문(제목+본문) 사용

| 항목 | 내용 |
|---|---|
| 상태 | 완료 |
| 날짜 | 2026-06-09 |
| 커밋 | `<this>` |
| 데이터 | 다년, `USE_BODY=1`(제목+본문), MAX_LENGTH 64→128, batch 4 |
| 산출물 | `phase2/results_multiyear_body/`, `dataset_2124_body.parquet`(gitignore) |

## 1. 동기 / 가설
지금까지 **제목만** 사용했다. BIGKinds 본문(~200자 스니펫, 100% 채움)을 더하면 문맥이
풍부해져 신호가 강해질까? 각 헤드라인을 `제목 + 본문`으로 결합해 검증.

## 2. 설정
- build_dataset 가 `USE_BODY=1` 시 본문 컬럼을 읽어 `제목 + 본문`(공백 정리)으로 결합.
- 본문 스니펫이 ~200자(~110토큰)라 **MAX_LENGTH 64→128**, attention 메모리(len²)로
  batch 16→**4** 로 축소(메모리 동등). 그 외 다년 설정·모델 불변. seed 42.

## 3. 결과 (제목만=EXP-D vs 제목+본문)

| 지표 | 제목만 (EXP-D) | 제목+본문 (EXP-K) |
|---|---|---|
| best val macro-F1 | 0.258 | **0.296** |
| RoBERTa macro-F1 KOSDAQ h1/h5 | 0.293 / 0.275 | **0.315 / 0.330** |
| RoBERTa macro-F1 KOSPI h1/h5 | 0.318 / 0.298 | 0.312 / 0.303 |
| **RoBERTa IC KOSPI h5** | **+0.182** | **+0.092** |
| RoBERTa IC KOSDAQ h5 | +0.112 | +0.091 |
| TF-IDF macro-F1 KOSPI h5 | 0.350 | 0.314 |

## 4. 분석 — **본문은 예측을 개선하지 못했다(핵심 신호는 오히려 약화)**
- **macro-F1 은 소폭 혼재**(KOSDAQ↑, KOSPI~) 이지만, **EXP-H 의 핵심 방향신호인 KOSPI h5
  IC 가 0.182→0.092 로 절반으로 떨어졌다.** TF-IDF KOSPI h5 도 0.350→0.314 하락.
- 해석: **제목이 가장 정보 밀도 높은 신호**이고, 본문 스니펫(일반 서술 ~200자)은 일별
  [CLS] 표현을 **희석**해 방향 순위력을 깎는다. 더 긴 입력은 학습은 시키지만(val↑)
  *예측에 유용한* 신호를 늘리지 못함.
- **단일 시드 caveat**: KOSPI h5 IC 는 시드 변동 ±0.04(EXP-G/H)지만, 0.092 는 제목
  시드 범위(0.115~0.182)보다 낮아 *희석 효과는 실재*로 보인다(단 본문도 1 시드).

## 5. 한계
- 본문이 **전문이 아니라 ~200자 스니펫**(BIGKinds export 제한) — 전문이면 다를 수 있음.
- 단일 시드, MAX_LENGTH 128 truncation. batch 4 로 학습 동역학 약간 다름.

## 6. 결론 / 다음
- **본문 추가는 비용(메모리·시간) 대비 이득 없음 — 제목만으로 충분(오히려 우수).**
  헤드라인(제목)이 신호의 핵심이라는 본 연구 가정을 역으로 뒷받침.
- 다음: IT 자료 보강(EXP-L).

## 7. 재현
```
EXP_PROFILE=multiyear USE_BODY=1 python phase1/build_dataset.py
EXP_PROFILE=multiyear USE_BODY=1 python phase1/build_labels.py
EXP_PROFILE=multiyear USE_BODY=1 python phase2/baseline.py
EXP_PROFILE=multiyear USE_BODY=1 python phase2/train.py --batch-size 4
EXP_PROFILE=multiyear USE_BODY=1 python phase2/evaluate.py
EXP_PROFILE=multiyear USE_BODY=1 python phase2/signal_test.py
```
