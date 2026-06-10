# EXP-V: 삼성전자 단일종목 예측 (전체 헤드라인 × 30 epoch)

| 항목 | 내용 |
|---|---|
| 상태 | 완료 |
| 날짜 | 2026-06-10 |
| 커밋 | `<this>` |
| 데이터 | `samsung` 프로필: 삼성전자(005930) 단독, 뉴스 data/Samsung_Electronics(2021–2025, 24357건) |
| 산출물 | `phase2/results_samsung_ordtime{,_bin}/`, 가격 `prices_samsung.parquet` |

## 1. 동기 / 설정
사용자 제공 삼성전자 단독 뉴스로 **개별 종목 방향**을 예측(분산된 지수 대신 *회사 뉴스 →
그 회사 주가*, 인과가 더 직접적일 가능성).
- 단일 'index'=삼성전자, 가격=FDR 005930. split **train 2021–2023(739) / val 2024(244) /
  test 2025(242)** — 2025 는 완전 out-of-sample.
- **전체 헤드라인 사용**: 일평균 13건(최대 49) → `MAX_HEADLINES=64` 로 *모든* 헤드라인 포함.
  시각 정렬(EXP-S), batch 8(메모리), **30 epoch**, 로그 자동 저장.
- 3-class 와 binary(up/down) 둘 다. TF-IDF·wordcount 도 동일 입력(mh64)으로 비교.

## 2. 학습 곡선
- 3-class: val best **0.372 @ epoch 14**, 이후 과적합(train_loss→0.03, val_loss↑). 소표본
  (739)이라 ~14 epoch 이 정점 — best-val 체크포인트가 포착.
- binary: val best **0.540**(이진 random 0.5 상회).

## 3. 결과 (test=2025)

**3-class macro-F1 (무작위 0.333):**

| 지표 | RoBERTa | TF-IDF | wordcount |
|---|---|---|---|
| h1 | 0.286 | 0.347 | 0.350 |
| h5 | **0.351** | 0.308 | 0.317 |
| h21† | 0.255 | 0.290 | 0.323 |
| h252† | 0.261 | 0.211 | 0.193 |

**Binary macro-F1 (무작위 0.5):** RoBERTa h1 0.449, h5 0.432, h21† 0.429, h252† 0.392.

† h21·h252 는 test=2025 의 forward return 이 2026+ 가격을 요구해 **상당수 NaN**(유효 표본
적음) → 신뢰도 낮음.

## 4. 분석 — **단일종목도 OOS(2025)에서 신호 약함**
- **3-class: 모두 무작위(0.333) 부근.** TF-IDF·wordcount 가 h1 에서 RoBERTa 보다 약간 높고
  (0.35 vs 0.286), RoBERTa 는 h5 가 최고(0.351). 지수 실험과 동일한 그림(약신호, TF-IDF 경쟁력).
- **Binary: val 0.540 이지만 test 2025 는 0.43~0.45 로 무작위(0.5) 미달.** 즉 binary 모델이
  train/val(2021–2024)에 맞춰졌으나 **2025 로 일반화 실패** → 단일년 OOS 분포 이동
  (EXP-A 의 2024-Dec, EXP-F 의 단년 test 붕괴와 같은 구조적 문제).
- **회사 뉴스라도 단일종목 방향 예측은 쉽지 않다** — 직접 인과 기대와 달리 2025 OOS 에서
  약함. 삼성 주가는 반도체 업황·매크로(개별 뉴스 외 요인)에 크게 좌우되는 점도 작용.

## 5. 한계
- **test=2025 단일년 OOS** → 분포 이동에 취약(지수 실험 EXP-D 처럼 test 를 여러 해로
  넓히면 더 견고). h21·h252 는 NaN 多 → 사실상 h1·h5 만 유효.
- 단일 시드. 삼성 헤드라인도 일 13건으로 적어 일부 날 신호 희박.
- 가격이 2026-06 까지뿐이라 2025 의 장기 horizon forward 미정의(코드에 NaN ignore 추가).

## 6. 결론 / 다음
- **단일종목(삼성) 뉴스 → 방향 예측도 OOS 에서 약신호**. 3-class 는 무작위 부근, binary 는
  과적합/분포이동으로 무작위 미달. 지수와 정성적으로 동일(신호 약함).
- 다음: test 를 2024+2025 등 다년으로 넓혀 OOS 견고성↑, 또는 다른 대형주(SK하이닉스 등)로
  확장. 이진은 다중 시드·정칙화로 과적합 완화 검토.

## 7. 재현
```
EXP_PROFILE=samsung python phase1/build_prices.py     # FDR 005930
EXP_PROFILE=samsung HEADLINE_ORDER=time python phase1/build_dataset.py
EXP_PROFILE=samsung HEADLINE_ORDER=time python phase1/build_labels.py
EXP_PROFILE=samsung HEADLINE_ORDER=time python phase2/train.py --batch-size 8 --max-headlines 64 --epochs 30
EXP_PROFILE=samsung HEADLINE_ORDER=time python phase2/evaluate.py
EXP_PROFILE=samsung HEADLINE_ORDER=time BASELINE_TOPN=64 python phase2/baseline.py   # TF-IDF
EXP_PROFILE=samsung HEADLINE_ORDER=time BASELINE_TOPN=64 python phase2/wordcount_baseline.py
# binary: 위에 BINARY=1 추가
```
