# EXP-D: 2021–2023 학습 → 2024 예측 (다년 데이터)

| 항목 | 내용 |
|---|---|
| 상태 | **대기** (사용자 입력 2종 필요) |
| 날짜 | 2026-06-09 (파이프라인 준비) |
| 커밋 | `99b9d12` (파이프라인), 런북 `HANDOFF_MULTIYEAR.md` |
| 데이터 | (예정) 2021–2023 헤드라인 학습 / 2024 전체 test |
| 산출물 | (예정) `phase2/results_multiyear/`, `dataset_2124.parquet` |

## 1. 동기 / 가설
EXP-A·B·C 가 일관되게 **데이터 규모가 RoBERTa 붕괴의 근본 병목**임을 가리킨다.
학습창을 3년(2021–2023, ~1,480행 추정)으로 늘리고 **test 를 2024 전체(~490행)** 로
바꾸면, (a) 단일클래스 붕괴 완화, (b) 짧은 test 창·val→test 이동 문제 해소가 기대된다.
한국 시장은 2024 vs 2025/2026(유래없는 상승장) 괴리가 커, 학습 라벨을 2021–2024
"박스피" 국면에 두어 2025 상승장 오염도 피한다.

## 2. 설정 (예정)
- split: train 2021-01~2023-09 / val 2023-10~12 / test 2024 전체.
- encoder=klue/roberta-base, ε=0.3σ, σ-train-only **불변**. 모델 크기도 base 유지
  (병목은 용량 아닌 데이터 → large 미사용, HANDOFF §6 준수).
- `EXP_PROFILE=multiyear` 프로필이 split·데이터셋경로·가격백필·결과경로를 자동 분리.
  EXP-C 필터와 결합 가능: `EXP_PROFILE=multiyear HEADLINE_FILTER=market`.

## 3. 결과
미실행.

## 4. 분석
미실행. (가설: data↑ → 붕괴 완화. 단, 헤드라인 신호가 본질적으로 약하다는 결론이
유지될 수도 있음 — 그 경우도 정직하게 기록.)

## 5. 차단 요인 (실행 전 필요)
1. **2021·2022·2023 BIGKinds 헤드라인 xlsx** — 동일 매체(한국경제/조선일보/한겨레),
   경제·국제. `NewsResult_*.xlsx` 로 저장소 루트에. (현재 2024년치만 보유)
2. **KRX 자격증명**(`KRX_ID`/`KRX_PW`) — pykrx 지수 API 가 로그인 필수(확인됨).
   2021–2023 가격 백필용.

## 6. 결론 / 다음
입력 확보 시 `HANDOFF_MULTIYEAR.md` 런북대로 실행하고 본 보고서의 §3·§4 를 채운다.

## 7. 재현 (입력 확보 후)
```
EXP_PROFILE=multiyear KRX_ID=... KRX_PW=... python phase1/build_prices.py
EXP_PROFILE=multiyear python phase1/build_dataset.py
EXP_PROFILE=multiyear python phase1/build_labels.py
EXP_PROFILE=multiyear python phase2/baseline.py
EXP_PROFILE=multiyear python phase2/train.py --batch-size 16
EXP_PROFILE=multiyear python phase2/evaluate.py
```
