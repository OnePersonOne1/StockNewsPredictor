# EXP-R 공정 비교 — 동일 top-30 입력 × {3-class, 이진}

TF-IDF·wordcount 를 RoBERTa 와 같은 최신 30개로 제한. 비교: all(전체) vs top30.
이진 random=0.5, IC=0; 3-class random macro-F1≈0.333.

## h = 1

| model | input | KOSPI 3cls/bin/IC | KOSDAQ 3cls/bin/IC |
|---|---|---|---|
| TF-IDF | top30 | 0.330/0.485/-0.034 | 0.315/0.491/+0.040 |
| TF-IDF | all | 0.300/0.521/+0.011 | 0.265/0.457/-0.012 |
| wordcount | top30 | 0.346/0.515/+0.061 | 0.317/0.543/+0.045 |
| wordcount | all | 0.306/0.461/-0.077 | 0.318/0.463/-0.062 |
| RoBERTa | top30 | 0.318/0.575/+0.120 | 0.293/0.537/+0.063 |

## h = 5

| model | input | KOSPI 3cls/bin/IC | KOSDAQ 3cls/bin/IC |
|---|---|---|---|
| TF-IDF | top30 | 0.368/0.531/+0.024 | 0.275/0.453/-0.002 |
| TF-IDF | all | 0.350/0.475/-0.035 | 0.279/0.484/-0.036 |
| wordcount | top30 | 0.275/0.497/+0.015 | 0.263/0.458/-0.023 |
| wordcount | all | 0.324/0.447/-0.063 | 0.279/0.380/-0.088 |
| RoBERTa | top30 | 0.298/0.542/+0.182 | 0.275/0.474/+0.112 |
