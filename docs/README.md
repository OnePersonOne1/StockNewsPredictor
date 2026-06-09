# 실험 보고서 모음 (StockNewsPredictor)

> **연구 질문:** 뉴스 헤드라인의 주가 방향 예측력은 얼마나 먼 horizon 까지 유효한가?
> 2024년 BIGKinds 경제·국제 헤드라인 + KOSPI/KOSDAQ 지수로 방향 3-class
> {-1,0,+1} 을 h ∈ {1,5,21,252} 거래일 예측. **성능보다 과정·해석·정직한 보고가
> 핵심.** 이 폴더는 **실험별 정식 보고서**의 모음이며, 새 실험마다 새 보고서를 추가한다.

## 작성 규칙 (앞으로 매 실험)

1. 새 실험은 `docs/experiments/EXP-<글자>_<슬러그>.md` 로 **새 파일** 작성
   (`_TEMPLATE.md` 형식 준수: 메타 → 가설 → 설정 → 결과 → 분석 → 한계 → 결론 → 재현).
2. 본 `README.md` 의 **실험 인덱스**와 **종합 결론**을 갱신.
3. 부정·실패·부분성공 결과도 **원인까지 분석해 그대로** 기록(과정이 평가 대상).
4. 수치는 산출 아티팩트(`phase2/results*/…`)에 근거하고, 추측과 구분.

## 실험 인덱스

| ID | 제목 | 상태 | 한줄 결과 |
|---|---|---|---|
| [EXP-A](experiments/EXP-A_roberta-finetune-2024.md) | RoBERTa fine-tune (2024, mh30) — 주 모델 | 완료 | test 단일클래스 붕괴, TF-IDF 미달 |
| [EXP-B](experiments/EXP-B_encoder-freeze.md) | Encoder freeze + 헤드라인 확대(mh100) | 완료 | val↑(0.25→0.31) 부분완화, test 여전히 TF-IDF 미달 |
| [EXP-C](experiments/EXP-C_headline-relevance-filter.md) | 헤드라인 관련성 필터 (증권/거시) | 완료 | **TF-IDF 대폭↑**(KOSDAQ h5 0.25→0.45), RoBERTa 불변 |
| [EXP-D](experiments/EXP-D_multiyear-plan.md) | 2021–2023 학습 → 2024 예측 | **완료** | **데이터 확대로 RoBERTa 붕괴 해소**(attention·예측 부활), 단 모든 방법 ~무작위 |
| [EXP-E](experiments/EXP-E_multiyear-filter.md) | 다년 × 관련성 필터 결합 | 완료 | **필터+TF-IDF 최강**(KOSPI h1 0.380), 필터는 RoBERTa 엔 무이득(market 은 해로움) |
| [EXP-F](experiments/EXP-F_datasize-vs-period.md) | 데이터 크기 vs 2025 특수성 | 완료 | **붕괴 원인=데이터 크기** (정상기 2021/22/23 소표본도 붕괴), 2025 특수성 아님 |
| [EXP-G](experiments/EXP-G_it-sector-filter.md) | IT/반도체 섹터 필터 | 완료(시범) | **IT 9%가 전체에 준하는 신호**(섹터 비중↑ 지지), RoBERTa 셀차는 시드 노이즈 |
| [EXP-H](experiments/EXP-H_signal-test.md) | 신호 검정(이진·IC·백테스트) | 완료 | **macro-F1 이 약신호를 가렸음**: RoBERTa IC 24/24 시드·셀 양수, KOSPI h5 IC 0.15(p<.05) |
| [EXP-I](experiments/EXP-I_signal-interpret.md) | 신호의 출처 해석 | 완료 | **KOSPI h5 고확신날 적중률 67%**, 어닝·미국매크로·금리 헤드라인에 주목 |
| [EXP-J](experiments/EXP-J_high-confidence.md) | 고확신 구간 정량화 | 완료 | KOSPI h5 고확신서 이진acc 0.54→0.66, IC 0.18→0.21 (단 n↓·백테스트 비유의) |
| [EXP-K](experiments/EXP-K_body-text.md) | 본문(제목+본문) 사용 | 완료 | **개선 없음** — 핵심 IC 오히려 약화(KOSPI h5 0.18→0.09), 제목이 신호 핵심 |
| [EXP-L](experiments/EXP-L_it-augmentation.md) | IT 자료 보강(IT_section 병합) | 완료 | IT 볼륨 2배·dedup 버그 수정. TF-IDF 견조(market_it KOSPI h1 0.387), RoBERTa IC 단일시드 부호 flip→판정불가 |
| [EXP-M](experiments/EXP-M_binary-reeval-cause.md) | 전 실험 이분법/IC 재평가(원인) | 완료(1단계) | **원인=데이터 크기**: 소표본 IC std±0.1~0.3(부호 flip), 대표본만 +0.146±0.026 안정 |
| [EXP-N](experiments/EXP-N_it-section-only.md) | IT_section 단독 | 완료 | **IT 단독은 약함**(macro-F1 0.18, IC +0.035 vs 본체 +0.146) — 신호는 경제 프레이밍에 |
| [EXP-O](experiments/EXP-O_full-corpus-no-filter.md) | data/ 전체 결합(비여과) | 완료 | **RoBERTa 는 본체와 완전 동일**(top-30 절단 → IT_section 176번째라 투명), TF-IDF만 미세 희석 |

**고찰**: [모델 유형 × 입력 정제의 상호작용](discussion.md) — BoW 는 정제(차원↓)가 큰 레버,
LLM 은 데이터·볼륨이 큰 레버이며 정제는 noise↓ vs volume↓ trade-off.

## 데이터 개요 (2024)

- 헤드라인 104,920건(한국경제/조선일보/한겨레, 경제·국제), 일평균 ~287건.
- 거래일×지수 행: KOSPI/KOSDAQ 각 244 → split **train 406 / val 42 / test 40**
  (temporal: train 2024-01~10 / val 11 / test 12).
- 라벨: ε=0.3σ_h ternary, σ_h 는 **train 에서만** 계산. train 라벨은 거의 균형
  (h1 137/136/133) → 이후 단일클래스 붕괴는 불균형이 아님.

## 종합 결론 (현재까지)

1. **헤드라인의 주가 방향 예측력은 약하지만, "없다"고 단정하면 틀린다(EXP-H).**
   3-class macro-F1 로는 모두 무작위(0.333) 부근이지만, 이는 예측 불가한 flat 클래스와
   분류 정확도 지표가 *방향* 신호를 가린 탓이다. **IC(Spearman) 로 보면 RoBERTa IC 가
   6시드 × 4셀 모두 양(+)**이고 **KOSPI h5 IC≈0.15 가 6시드 중 4번 p<.05**(셀의 Spearman
   p값). *(주의: 이전의 "24/24 ⇒ p≈6e-8"은 셀·시드가 독립이 아니라 무효 — EXP-H §3 정정.)*
   단 IC 가 약해(분산 ~2%) **롱숏 백테스트는 비유의 → 거래 가능 alpha 는 아님**(약형
   효율성과 정합). 즉 신호는 **실재하나 약하다**. 흥미로운 반전: macro-F1 1등 TF-IDF 는
   방향 IC≈0, wordcount 는 유의한 음(역발상) 신호. → **지표 선택이 결론을 좌우.**
2. **EXP-A 의 "딥러닝이 진다"는 데이터 규모의 산물이었다.** 406행에선 RoBERTa 가
   예측·attention 모두 붕괴(EXP-A·B·C)했으나, **3년 데이터(EXP-D)에서 붕괴가 해소**
   (attention top-1 lift 1.07×→7.85×, 예측이 3클래스로 분산)되고 TF-IDF 와 대등해짐.
   → 딥러닝의 병목은 모델 용량이 아니라 **데이터 규모**. 큰 모델은 해법이 아님.
   **EXP-F 통제 실험으로 확정**: 정상기 2021·2022·2023 소표본도 모두 붕괴 →
   원인은 데이터 크기이고 **2025 특수성이 아니다**. (h21/h252 단일클래스 퇴화는
   1개월 test 창 × 장기 horizon 의 구조적 산물로 연도 불문 발생, 2025 고유 아님.)
3. **입력 정제(관련성 필터)는 TF-IDF 를 끌어올리나 RoBERTa 엔 무이득.** 2024 소표본
   (EXP-C: KOSDAQ h5 0.25→0.45)·다년(EXP-E: KOSPI h1 0.30→**0.380**) 모두에서 필터가
   TF-IDF 를 개선해 무작위를 상회. 그러나 RoBERTa 는 필터로 나아지지 않고 market 필터는
   헤드라인 수를 줄여 오히려 해로움 → **TF-IDF 는 입력 정제를, RoBERTa 는 입력 볼륨을
   선호.** 가장 강한 예측기 = **시장 카테고리 정제 TF-IDF**(KOSPI 단기).
4. h21/h252 는 짧은 test 창(2024-only)에선 단일클래스로 무의미했으나, **test 를
   2024 전체로 늘리면(EXP-D) 의미 있는 값**을 가진다 → 평가 설계가 결론을 좌우.
5. **신호는 'IT 주제'가 아니라 '경제 프레이밍'에 있다**(EXP-G·N): 경제뉴스 *안의* IT/반도체
   부분집합(EXP-G `it`, 9%)은 전체에 준하는 예측력을 냈지만, **순수 IT_section 코퍼스 단독
   (EXP-N)은 약하다**(macro-F1 0.18, IC +0.035 vs 본체 +0.146). 즉 IT 비중 효과는 *경제·
   금융 맥락을 동반한 IT* 에 한정.
6. **방법론 caveat — RoBERTa 단일 시드 변동성**: market_it 을 4개 시드로 재학습 시 best
   val 0.196~0.269. **RoBERTa 단일 셀 수치는 ±0.04+ 불확실**하므로 필터 간 미세 비교는
   TF-IDF(결정적)와 '경향'에 무게를 둔다. 향후 다중 시드 평균이 정도.
7. **Attention 집중도가 '학습 성공'의 지표**(케이스 11개, `results/attention_summary.*`):
   소표본은 모두 top-1 lift≈1(평균 pooling 붕괴), 대표본은 6~8×(차별화). 유일한 예외
   multiyear market_it(lift 1.01)은 seed 42 학습 실패와 정확히 일치 → attention map 이
   붕괴/시드변동을 그대로 드러낸다. (attention 은 단일 query 라 horizon 공통.)

> 산출물 정리: 각 실험 케이스 폴더(`phase2/results*/`)에 attention map·heatmap·entropy
> 분포·통계(`attention_*`)와 8셀 metric(`test_metrics`, `metrics.json`)이 모두 포함.
> 케이스 간 요약은 `results/attention_summary.{md,png}`, `results*/filter_ablation.*`,
> `results/exp_f_datasize.*`.

### 방법론 사다리 — 2024-only (train 406, test 40; h=1 / h=5)

| 방법 | KOSPI | KOSDAQ |
|---|---|---|
| random (이론) | 0.333 / 0.333 | 0.333 / 0.333 |
| wordcount (비-ML 렉시콘) | 0.123 / 0.268 | 0.161 / 0.140 |
| RoBERTa (ft, mh30) — 붕괴 | 0.111 / 0.207 | 0.032 / 0.173 |
| RoBERTa (freeze, mh100) | 0.087 / 0.190 | 0.207 / 0.173 |
| TF-IDF (전체) | 0.279 / 0.356 | 0.246 / 0.250 |
| **TF-IDF (시장/거시 정제, EXP-C)** | 0.251 / **0.411** | **0.399** / **0.448** |

### 방법론 사다리 — 다년 EXP-D (train 1358, test 2024 전체 488; h=1 / h=5)

| 방법 | KOSPI | KOSDAQ |
|---|---|---|
| wordcount | 0.306 / 0.324 | 0.318 / 0.279 |
| TF-IDF | 0.300 / 0.350 | 0.265 / 0.279 |
| **RoBERTa (ft)** — 붕괴 해소 | **0.318** / 0.298 | 0.293 / 0.275 |

(2024-only 에선 정제 TF-IDF 만 0.333 상회. 다년에선 RoBERTa 가 대등해지나 모두
무작위 부근 — 신호 자체가 약함. 자세한 표·그림·confusion 은 각 실험 보고서 참조.)
