# RoBERTa freeze 실험 — test macro-F1 비교

| index | method | h=1 | h=5 | h=21 | h=252 |
|---|---|---|---|---|---|
| KOSPI | TF-IDF | 0.279 | 0.356 | 0.000 | 0.333 |
| KOSPI | RoBERTa(ft,mh30) | 0.111 | 0.207 | 0.000 | 0.333 |
| KOSPI | RoBERTa(freeze,mh100) | 0.087 | 0.190 | 0.263 | 0.333 |
| KOSDAQ | TF-IDF | 0.246 | 0.250 | 0.000 | 0.032 |
| KOSDAQ | RoBERTa(ft,mh30) | 0.032 | 0.173 | 0.000 | 0.333 |
| KOSDAQ | RoBERTa(freeze,mh100) | 0.207 | 0.173 | 0.000 | 0.000 |

주: 주 비교는 h=1,5 (h21/h252 는 test 단일클래스 붕괴로 신뢰도 낮음).
ft=fine-tune, mh=MAX_HEADLINES.