# EXP-C: 헤드라인 관련성 필터 (전체 vs 증권_증시 vs 금융·거시)

| 항목 | 내용 |
|---|---|
| 상태 | 완료 |
| 날짜 | 2026-06-09 |
| 커밋 | `acd4e6a` |
| 데이터 | 2024 전체, BIGKinds '통합 분류1' 카테고리 필터 |
| 산출물 | `phase2/results_{market,macro}/`, `phase2/results/filter_ablation.{csv,md}`, `figures/filter_ablation.png` |

## 1. 동기 / 가설
원본은 하루 ~287건(매핑 후 행당 ~427건) 중 유통·자동차·부동산·국제일반 등 **주가와
무관한 기사가 다수**라 신호가 희석된다(EXP-A 의 attention 균일 붕괴와 같은 맥락,
사용자 관찰). 시장 관련 카테고리만 남겨 신호/잡음비를 높이면 성능이 오르는가?

## 2. 설정
- BIGKinds '통합 분류1' 로 필터(라벨·split·σ·encoder **불변**):
  - `all` : 전체(현재)
  - `market` : 경제>증권_증시 (일평균 ~28.5건, 0건인 날 없음)
  - `macro` : 증권+금융_재테크+국제경제+외환+경제일반 (일평균 ~66건)
- 칼럼/오피니언은 BIGKinds 별도 카테고리가 없고 제목태그로 일 ~6.6건(8일 0건) +
  EXP-A attention 분석상 저신호 → 제외.
- `HEADLINE_FILTER` 환경변수로 선택, 산출물은 프로필·필터별 경로로 분리.

## 3. 결과
**TF-IDF test macro-F1 (h=1 / h=5):**

| index | all | market | macro |
|---|---|---|---|
| KOSDAQ | 0.246 / 0.250 | 0.325 / **0.448** | **0.399** / 0.365 |
| KOSPI | 0.279 / 0.356 | 0.259 / 0.378 | 0.251 / **0.411** |

**RoBERTa(ft) test macro-F1**: 세 필터 모두 **동일** — KOSPI 0.111/0.207,
KOSDAQ 0.032/0.173, best val 0.2514 불변(단일클래스 붕괴 지속).

## 4. 분석
- **잡음 제거가 신호를 살림(가설 실증)**: TF-IDF 에서 KOSDAQ h5 0.250→0.448(+0.198),
  KOSPI h5 0.356→0.411. 필터 후 여러 셀이 무작위(0.333)를 분명히 상회.
  → 현재까지 **최강 신호 = 시장 카테고리 정제 헤드라인 위의 TF-IDF**.
- **KOSPI h1 은 소폭 하락**(0.279→0.251): 대형주(KOSPI)는 거시·정책 등 넓은 맥락도
  단기 신호에 기여 → 과한 필터가 일부 정보를 깎음. 중소형주 중심 KOSDAQ 는 시장
  직결 정제 효과가 큼.
- **RoBERTa 는 입력 정제와 무관하게 붕괴 지속** → 딥러닝 병목은 입력 잡음이 아니라
  **소표본(406행)** 임을 재확인(EXP-A·B 와 일치). 정제로 market 은 일 ~28건이라
  MAX_HEADLINES=30 이 거의 전부를 담아 절단 손실도 사라지지만, 그래도 RoBERTa 는
  학습 불가 — 신호 존재 ≠ 딥러닝이 살릴 수 있음.

## 5. 한계
- 카테고리 태깅 자체가 BIGKinds 기준(완벽치 않음). 'market' 표본이 작아 일부 날은
  헤드라인 수가 적음.
- KOSPI h1 처럼 필터가 역효과인 셀 존재 → 필터는 만능이 아님.

## 6. 결론 / 다음
- **사용자 가설 확인**: 관련없는 기사가 신호를 희석했고, 정제가 TF-IDF 를 크게 끌어올림.
- 다음: **데이터 확대(EXP-D)** 와 **필터 결합**(`EXP_PROFILE=multiyear
  HEADLINE_FILTER=market`)으로 'data↑ + noise↓' 가 RoBERTa 붕괴를 푸는지 검증.

## 7. 재현
```
for F in market macro; do
  HEADLINE_FILTER=$F python phase1/build_dataset.py
  HEADLINE_FILTER=$F python phase1/build_labels.py
  HEADLINE_FILTER=$F python phase2/baseline.py
  HEADLINE_FILTER=$F python phase2/train.py --batch-size 16
  HEADLINE_FILTER=$F python phase2/evaluate.py
done
python phase2/compare_filters.py
```
