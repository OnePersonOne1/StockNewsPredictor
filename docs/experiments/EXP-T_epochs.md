# EXP-T: 4 epoch 이 충분했나 — 학습 부족(under-training) 점검

| 항목 | 내용 |
|---|---|
| 상태 | 완료 |
| 날짜 | 2026-06-09 |
| 커밋 | `<this>` |
| 데이터 | multiyear (mh30) vs 2024(소표본). epochs 4 vs 10 |
| 산출물 | 본 보고서 + `train.py` per-epoch 로그 자동 저장(`RESULTS_DIR/trainlog_*.txt`) |

## 1. 동기 / 가설
EXP-A~S 의 모든 RoBERTa 학습은 **epochs=4**(config 기본). "4 epoch 이 충분했나?"
특히 데이터가 큰 multiyear 에서 더 학습하면 좋아질 가능성.

## 2. 방법
- multiyear(1358행, mh30, batch16)을 **epochs=10** 으로 재학습, per-epoch val 곡선 관찰.
- 2024(406행)의 기존 곡선과 대조. (이후 4ep 으로 복원.)

## 3. 결과

**multiyear best-val macro-F1 궤적(10 epoch):** 0.129 → 0.192 → 0.242 → **(epoch4)0.245** →
0.278 → 0.310 → 0.334 → **0.343** — **epoch 4 이후로도 계속 상승**(train_loss 도 1.08→0.94).

**test macro-F1 (multiyear, seed42):**

| | KOSPI h1 | KOSPI h5 | KOSDAQ h1 | KOSDAQ h5 |
|---|---|---|---|---|
| 4 epoch (EXP-D) | 0.318 | 0.298 | 0.293 | 0.275 |
| **10 epoch** | **0.389** | **0.339** | **0.372** | **0.383** |

(IC KOSPI h5: 10ep +0.159 vs 4ep +0.182 — 단일시드 노이즈 범위.)

**2024(소표본) 대조:** val 이 **epoch 2 에서 정점(0.251) 후 하락**(과적합) → 4 는 충분/과다.

## 4. 분석 — **4 epoch 은 대표본에 부족했다**
- **multiyear 는 epoch 4 에서 val 0.245 → epoch ~8 에서 0.343**(+0.10). test macro-F1 도
  KOSDAQ h5 0.275→**0.383** 등 전 셀 크게 개선. **즉 다년 실험들은 학습 부족(under-trained)
  이었고, 결과가 보수적(과소평가)이었다.**
- **소표본(2024)은 정반대**: epoch 2 정점 후 과적합 → 4 가 충분/과다. best-val 체크포인트가
  과적합은 막았다.
- best-val 체크포인트는 *과적합*은 막지만 *학습 부족*은 못 고친다(val 이 계속 오르면 best=
  마지막 epoch). → 대표본은 **더 긴 epoch 필요**.

## 5. 함의 (중요)
- **EXP-P 의 '헤드라인 더 넣으면 나빠진다'(mh100/200 열화)도 학습 부족일 수 있다** — 입력이
  큰 mh200 은 4 epoch 이 더욱 부족. **전체 헤드라인 × 30 epoch 재검(최종 과제, EXP-U)** 필요.
- 단일시드 IC 는 여전히 노이즈(EXP-M) — epoch↑ 의 macro-F1 개선은 확실하나 IC 개선은 다중
  시드로 재확인 권장.

## 6. 결론 / 다음
- **권고: 대표본(multiyear/단일종목) 학습은 epochs≈10~30 으로.** config 기본 4 는 소표본 기준.
- 모든 학습 로그를 `trainlog_*.txt` 로 자동 보존(이번에 train.py 에 추가).
- 다음: 전체 헤드라인 × 30 epoch(EXP-U, 최종) / 단일종목 30 epoch(EXP-V).

## 7. 재현
```
EXP_PROFILE=multiyear python phase2/train.py --batch-size 16 --epochs 10   # val 곡선
EXP_PROFILE=multiyear python phase2/evaluate.py
```
