# 헤드라인 관련성 필터 ablation — test macro-F1

all=전체, market=증권_증시, macro=금융·증시·거시. 주 결과는 h=1,5.

## TF-IDF

| index | filter | h=1 | h=5 | h=21 | h=252 |
|---|---|---|---|---|---|
| KOSPI | all | 0.300 | 0.350 | 0.304 | 0.224 |
| KOSPI | market | 0.380 | 0.328 | 0.332 | 0.348 |
| KOSPI | macro | 0.355 | 0.324 | 0.376 | 0.279 |
| KOSDAQ | all | 0.265 | 0.279 | 0.291 | 0.294 |
| KOSDAQ | market | 0.292 | 0.338 | 0.289 | 0.365 |
| KOSDAQ | macro | 0.311 | 0.327 | 0.295 | 0.327 |

## RoBERTa

| index | filter | h=1 | h=5 | h=21 | h=252 |
|---|---|---|---|---|---|
| KOSPI | all | 0.318 | 0.298 | 0.325 | 0.277 |
| KOSPI | market | 0.226 | 0.297 | 0.255 | 0.269 |
| KOSPI | macro | 0.291 | 0.324 | 0.302 | 0.249 |
| KOSDAQ | all | 0.293 | 0.275 | 0.317 | 0.288 |
| KOSDAQ | market | 0.233 | 0.277 | 0.290 | 0.358 |
| KOSDAQ | macro | 0.327 | 0.292 | 0.325 | 0.309 |
