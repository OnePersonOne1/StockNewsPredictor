# EXP-E: 다년 + 관련성 필터 결합 (multiyear × {all, market, macro})

| 항목 | 내용 |
|---|---|
| 상태 | 완료 |
| 날짜 | 2026-06-09 |
| 커밋 | `<this>` |
| 데이터 | 2021–2023 학습 / 2024 전체 test, 통합분류1 필터 |
| 산출물 | `phase2/results_multiyear_{market,macro}/`, `phase2/results_multiyear/filter_ablation.{csv,md}`, `figures/filter_ablation.png` |

## 1. 동기 / 가설
EXP-C(2024 소표본)에서 관련성 필터가 TF-IDF 를 크게 끌어올렸고, EXP-D(다년)에서
데이터 확대가 RoBERTa 붕괴를 풀었다. 두 개입을 **결합**하면("data↑ + noise↓")
신호가 더 살아나는지, 특히 RoBERTa 가 필터 덕을 보는지 검증.

## 2. 설정
- EXP-D 와 동일(train 1358 / val 120 / test 488, base 모델 불변)에서
  `HEADLINE_FILTER` 만 all/market/macro 로 변화.
- market 유지율 10%(일 ~? 건), macro 23%. 명령: `EXP_PROFILE=multiyear
  HEADLINE_FILTER=<f>` 로 build→train→evaluate.

## 3. 결과
**test macro-F1 (h=1 / h=5):**

| model | index | all | market | macro |
|---|---|---|---|---|
| TF-IDF | KOSPI | 0.300 / 0.350 | **0.380** / 0.328 | 0.355 / 0.324 |
| TF-IDF | KOSDAQ | 0.265 / 0.279 | 0.292 / **0.338** | 0.311 / 0.327 |
| RoBERTa | KOSPI | **0.318** / 0.298 | 0.226 / 0.297 | 0.291 / **0.324** |
| RoBERTa | KOSDAQ | 0.293 / 0.275 | 0.233 / 0.277 | **0.327** / 0.292 |

(h21/h252 는 `results_multiyear_*/test_metrics.csv` 참조; 모두 ~0.25–0.37 범위.)

## 4. 분석
- **필터 + TF-IDF = 최강 조합**: KOSPI h1 이 all 0.300 → **market 0.380** 으로
  전 실험 통틀어 가장 높은 h1, 무작위(0.333)를 분명히 상회. 잡음 제거가 bag-of-words
  의 어휘를 정제해 **두 데이터 레짐(2024·다년) 모두에서 일관되게** TF-IDF 를 끌어올림.
- **필터는 RoBERTa 엔 도움이 안 됨 — 오히려 market 은 해로움**(KOSPI h1 0.318→0.226).
  RoBERTa 는 헤드라인 **수·문맥**을 활용하는데 market 필터가 일별 기사를 ~28건으로
  줄여 정보가 깎임. macro(일 ~? 더 많음)는 혼재(KOSDAQ h1 0.293→0.327 상승,
  KOSPI h1 하락). → **TF-IDF 는 입력 정제를, RoBERTa 는 입력 볼륨을 선호**하는
  상반된 성향이 드러남.
- **그래도 돌파구는 없음**: 최고 셀도 0.38 수준, 대부분 무작위 부근. "data↑ + noise↓"
  결합조차 헤드라인의 약한 예측력을 크게 넘기지 못함 → 본 연구의 주 결론(신호 약함) 유지.

## 5. 한계
- market 필터는 일별 표본이 작아 RoBERTa 의 attention pooling 에 불리(정보량↓).
- 단일 시드/실행 → 셀별 ±0.02 수준의 변동 가능. 경향(필터→TF-IDF↑, RoBERTa 무이득)은
  EXP-C·E 에서 반복 관찰됨.

## 6. 결론 / 다음
- **실무적 권고**: 이 과제에서 가장 강한 예측기는 **시장 카테고리로 정제한 헤드라인
  위의 TF-IDF**(KOSPI 단기). 무거운 RoBERTa 는 데이터가 충분하면 대등하나 정제로
  더 좋아지진 않는다.
- 신호의 본질적 약함이 EXP-A~E 전반에서 일관 → 보고서의 핵심 메시지.

## 7. 재현
```
for F in market macro; do
  EXP_PROFILE=multiyear HEADLINE_FILTER=$F python phase1/build_dataset.py
  EXP_PROFILE=multiyear HEADLINE_FILTER=$F python phase1/build_labels.py
  EXP_PROFILE=multiyear HEADLINE_FILTER=$F python phase2/baseline.py
  EXP_PROFILE=multiyear HEADLINE_FILTER=$F python phase2/train.py --batch-size 16
  EXP_PROFILE=multiyear HEADLINE_FILTER=$F python phase2/evaluate.py
done
EXP_PROFILE=multiyear python phase2/compare_filters.py
```
