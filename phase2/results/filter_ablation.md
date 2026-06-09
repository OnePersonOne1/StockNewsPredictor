# 헤드라인 관련성 필터 ablation — test macro-F1

all=전체, market=증권_증시, macro=금융·증시·거시. 주 결과는 h=1,5.

## TF-IDF

| index | filter | h=1 | h=5 | h=21 | h=252 |
|---|---|---|---|---|---|
| KOSPI | all | 0.279 | 0.356 | 0.000 | 0.333 |
| KOSPI | market | 0.259 | 0.378 | 0.330 | 0.325 |
| KOSPI | macro | 0.251 | 0.411 | 0.231 | 0.263 |
| KOSDAQ | all | 0.246 | 0.250 | 0.000 | 0.032 |
| KOSDAQ | market | 0.325 | 0.448 | 0.000 | 0.032 |
| KOSDAQ | macro | 0.399 | 0.365 | 0.032 | 0.000 |

## RoBERTa

| index | filter | h=1 | h=5 | h=21 | h=252 |
|---|---|---|---|---|---|
| KOSPI | all | 0.111 | 0.207 | 0.000 | 0.333 |
| KOSPI | market | 0.111 | 0.207 | 0.000 | 0.333 |
| KOSPI | macro | 0.111 | 0.207 | 0.000 | 0.333 |
| KOSDAQ | all | 0.032 | 0.173 | 0.000 | 0.333 |
| KOSDAQ | market | 0.032 | 0.173 | 0.000 | 0.333 |
| KOSDAQ | macro | 0.032 | 0.173 | 0.000 | 0.333 |
