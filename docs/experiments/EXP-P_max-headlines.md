# EXP-P: MAX_HEADLINES 확대 — 입력 규모와 TF-IDF 와의 공정성

| 항목 | 내용 |
|---|---|
| 상태 | 완료 |
| 날짜 | 2026-06-09 |
| 커밋 | `<this>` |
| 데이터 | 다년. main(본체) MAX_HEADLINES ∈ {30,100,200}, combined(본체∪IT_section) mh=200 |
| 산출물 | 본 보고서(수치는 stdout 캡처; 결과 폴더는 EXP-D mh30 으로 복원). 코드: model.py `GRAD_CKPT` |

## 0. mh(MAX_HEADLINES) 정의와 'top-30 을 쓴 이유'
- **한 학습 샘플 = (거래일 × 지수)**. 그 하루엔 헤드라인이 **일평균 ~430건**. RoBERTa 는
  그중 **시간순 최신 N개(mh)**만 [CLS] 인코딩→attention pooling 으로 하루 1개 벡터로 합친다
  (`dataset.py`: `headlines[:max_headlines]`). mh 는 *하루 안에서 읽는 헤드라인 수*다.
- **top-30 사유**: 원래 `config.py` 주석대로 **GPU 메모리 제약**(원본 ~430건 전부는 24GB
  초과). 즉 처음엔 *현실적 절충*이었다. 본 실험은 그 절충이 성능 면에서도 타당한지 검증.
- **중요한 비대칭(사용자 지적)**: **TF-IDF 는 하루 전체(~430건)를, RoBERTa 는 최신 30건만**
  소비한다. 두 모델은 *같은 조건이 아니었다.* 이를 mh 를 키워 RoBERTa 를 TF-IDF 쪽으로
  맞춰 검증한다. (메모리는 encoder gradient checkpointing=`GRAD_CKPT=1` 로 해결.)

## 1. 결과 (다년, seed 42, full fine-tune)

**main(본체) macro-F1:**

| mh | KOSPI h1 | KOSPI h5 | KOSDAQ h1 | KOSDAQ h5 | IC KOSPI h5 |
|---|---|---|---|---|---|
| **30** (EXP-D) | 0.318 | **0.298** | 0.293 | 0.275 | +0.182 |
| 100 | 0.271 | 0.274 | 0.296 | 0.274 | −0.010 |
| 200 | 0.170 | **0.181** | 0.186 | 0.168 | +0.003 |

**combined(본체∪IT_section) mh=200**: KOSPI 0.221/0.236, KOSDAQ 0.252/0.248, IC KOSPI h5 +0.024.

## 2. 분석 — **헤드라인을 더 넣을수록 RoBERTa 가 나빠진다**
macro-F1 KOSPI h5 가 mh 30→100→200 에서 **0.298 → 0.274 → 0.181 로 단조 하락**(IC 도
+0.182→~0 으로 붕괴, 단 IC 는 단일시드라 macro-F1 추세에 무게). 원인:
1. **평균 pooling 희석(주원인)**: N개 임베딩을 attention(거의 균일, EXP-A·G)으로 *평균*하므로,
   신호 헤드라인 비중이 30개면 ~1/30, 200개면 ~1/200 로 **묽어진다.** 소수 시장 관련
   헤드라인이 다수 잡음에 묻힘(SNR↓).
2. **최신성=관련성**: 거래일 직전 최신 30건이 다음날 움직임에 가장 시의성. 31~200번째는
   더 오래·덜 관련 → 잡음.
3. **소표본 + 큰 입력 = 학습 난이도↑**: 200개 중 신호를 고를 attention 을 1358행으로 학습
   하기 어려워 **균일(평균)로 후퇴** → 희석 악화(EXP-M 데이터 크기 한계와 연결).
4. **TF-IDF 와 반대인 이유**: TF-IDF 는 idf+희소 선형으로 잡음 토큰을 자동 하향 → 많을수록
   이득(EXP-C·E). RoBERTa 평균 pooling 엔 그 필터가 없어 많을수록 손해. → **두 모델의
   최적 입력 규모가 반대.**

**IT 가시성**: mh=200 이면 IT_section 이 18% 행의 top-200 에 들어오지만(중앙값 위치 370),
combined mh200(0.236) 과 main mh200(0.181) 모두 mh30 보다 나쁘다 → **IT 를 보여줘도 희석
열화를 못 되돌린다**(EXP-N·O 와 일관).

## 3. 함의 (사용자 질문에 대한 답)
- **"LLM 이 데이터 일부만 학습했나?" → 그렇다.** mh=30 은 하루 ~430건 중 **최신 30건(~7%)**만
  읽고 ~93%는 무시했다(모든 *날*은 봄). 이게 TF-IDF 와의 비대칭의 실체.
- **그러나 '그 일부(최신 30)'가 가장 유용**했다 — 더 주면(mh100·200) 오히려 나빠진다.
  → **top-30 은 메모리 절충이자 사실상 (근사) 최적.** RoBERTa 가 불리했던 게 아니라,
  평균 pooling 모델은 *적고 시의성 있는* 입력을, 선형 BoW(TF-IDF)는 *많은* 입력을 선호한다.

## 4. 한계
- 단일 시드(macro-F1 추세는 단조·robust 하나 IC 는 EXP-M 대로 노이즈). mh∈{30,100,200} 3점.
- gradient checkpointing 으로 학습은 정확하되 느림(mh200 1회 ~40분). mh≫370(IT 대부분
  포함)은 미검(메모리·시간). news_date 가 날짜 단위라 같은 날 내 순서는 concat 의존.

## 5. 결론 / 다음
- **더 많은 헤드라인 ≠ 더 좋음**(RoBERTa). top-30 정당화. TF-IDF 비대칭은 *핸디캡이 아니라*
  모델 특성 차이. 비여과 IT 보강(EXP-O)이 무효였던 이유도 재확인(top-30 절단 + 희석).
- 다음 후보: attention 이 균일 붕괴하지 않도록 하는 구조(예: 헤드라인 사전선별·중요도 가중)
  로 큰 mh 에서 희석을 막을 수 있는지.

## 6. 재현
```
# encoder gradient checkpointing 으로 큰 mh 학습 (메모리 절감)
GRAD_CKPT=1 EXP_PROFILE=multiyear python phase2/train.py --batch-size 8 --max-headlines 200
EXP_PROFILE=multiyear python phase2/evaluate.py     # ckpt 의 max_headlines 사용
EXP_PROFILE=multiyear python phase2/signal_test.py
# combined: 위에 INCLUDE_IT=1 추가
```
