## PubMed 20k RCT dev, 10 abstracts (4,739 tokens, 36 switches, declared length 30)

| Mechanism | decoder | token acc. | boundary F1 ±8 | ±16 | switches matched | mean delay | spurious /1k | Brier | ECE | ms/token | passes per token |
|---|---|---|---|---|---|---|---|---|---|---|---|
| asking, full history, 1 ordering | raw argmax | 0.794 | 0.14 | 0.18 | 33/36 | 16.0 | 41.0 | 0.315 | 0.037 | 103 | 1 stream + 1 question (126 tok) |
| asking, full history, 1 ordering | persist 3 | 0.779 | 0.25 | 0.32 | 32/36 | 22.5 | 7.7 | 0.315 | 0.037 | 103 | 1 stream + 1 question (126 tok) |
| asking, full history, 1 ordering | forward filter η=1 | 0.774 | 0.21 | 0.33 | 32/36 | 25.9 | 3.5 | 0.423 | 0.205 | 103 | 1 stream + 1 question (126 tok) |
| asking, full history, 1 ordering | forward filter η=0.5 | 0.757 | 0.21 | 0.30 | 31/36 | 27.0 | 4.2 | 0.438 | 0.206 | 103 | 1 stream + 1 question (126 tok) |
| asking, full history, 1 ordering | forward filter η=0.25 | 0.734 | 0.15 | 0.28 | 31/36 | 30.9 | 2.5 | 0.454 | 0.199 | 103 | 1 stream + 1 question (126 tok) |
| asking, full history, 1 ordering | filter, ÷ set prior (unlabelled), η=1 | 0.758 | 0.25 | 0.32 | 32/36 | 26.6 | 4.2 | 0.454 | 0.221 | 103 | 1 stream + 1 question (126 tok) |
| asking, full history, 1 ordering | filter, ÷ running prior, η=1 | 0.834 | 0.30 | 0.41 | 30/36 | 13.2 | 12.5 | 0.283 | 0.122 | 103 | 1 stream + 1 question (126 tok) |
| asking, full history, 2 orderings (prototype default) | raw argmax | 0.800 | 0.12 | 0.16 | 34/36 | 17.4 | 41.6 | 0.304 | 0.035 | 150 | 1 stream + 2 questions, batched |
| asking, full history, 2 orderings (prototype default) | persist 3 | 0.784 | 0.26 | 0.36 | 34/36 | 22.2 | 5.0 | 0.304 | 0.035 | 150 | 1 stream + 2 questions, batched |
| asking, full history, 2 orderings (prototype default) | forward filter η=1 | 0.775 | 0.24 | 0.33 | 33/36 | 25.5 | 4.8 | 0.417 | 0.202 | 150 | 1 stream + 2 questions, batched |
| likelihood, hopped window W=2 | raw argmax | 0.338 | 0.02 | 0.02 | 36/36 | 2.1 | 735.2 | 0.765 | 0.026 | 55 | 11-token rebuild of 5+1 copies per token (hop 1) |
| likelihood, hopped window W=2 | forward filter η=1 | 0.587 | 0.10 | 0.11 | 26/36 | 10.6 | 108.4 | 0.569 | 0.114 | 55 | 11-token rebuild of 5+1 copies per token (hop 1) |
| asking, last 30 tokens only (1×), every 2nd token | raw argmax | 0.757 | 0.18 | 0.27 | 29/36 | 18.6 | 35.6 | 0.355 | 0.097 | 39 | 1 pass over opening + 30 tokens + question, per 2 tokens |
| asking, last 30 tokens only (1×), every 2nd token | forward filter η=1 | 0.754 | 0.23 | 0.36 | 24/36 | 17.2 | 12.7 | 0.443 | 0.210 | 39 | 1 pass over opening + 30 tokens + question, per 2 tokens |
| asking, last 60 tokens only (2×), every 2nd token | raw argmax | 0.798 | 0.20 | 0.28 | 28/36 | 15.9 | 17.9 | 0.321 | 0.106 | 45 | 1 pass over opening + 60 tokens + question, per 2 tokens |
| asking, last 60 tokens only (2×), every 2nd token | forward filter η=1 | 0.776 | 0.20 | 0.38 | 25/36 | 20.4 | 3.7 | 0.414 | 0.204 | 45 | 1 pass over opening + 60 tokens + question, per 2 tokens |
| asking, last 120 tokens only (4×), every 2nd token | raw argmax | 0.763 | 0.18 | 0.24 | 30/36 | 22.5 | 16.9 | 0.372 | 0.133 | 55 | 1 pass over opening + 120 tokens + question, per 2 tokens |
| asking, last 120 tokens only (4×), every 2nd token | forward filter η=1 | 0.734 | 0.17 | 0.29 | 28/36 | 30.9 | 3.7 | 0.501 | 0.245 | 55 | 1 pass over opening + 120 tokens + question, per 2 tokens |
| 3a activation directions from model-written passages | raw argmax | 0.608 | 0.05 | 0.06 | 29/36 | 16.2 | 250.1 | 0.536 | 0.142 | 0.51 | 0 extra (read from the stream pass) |
| 3a activation directions from model-written passages | forward filter η=1 | 0.717 | 0.15 | 0.28 | 18/36 | 53.1 | 16.9 | 0.476 | 0.197 | 0.51 | 0 extra (read from the stream pass) |
| reference: bge-m3 similarity to descriptions, last 32 tokens | raw argmax | 0.575 | 0.09 | 0.11 | 28/36 | 19.6 | 89.9 | 0.552 | 0.101 | 1 | 1 embedding pass |
| reference: bge-m3 similarity to descriptions, last 32 tokens | forward filter η=1 | 0.570 | 0.17 | 0.26 | 18/36 | 40.7 | 13.1 | 0.750 | 0.352 | 1 | 1 embedding pass |

3b (distilled from asking) and 3c (probe on gold labels) are fitted on abstracts 1–5 and scored on 6–10: see `results/activation_directions.md`.

## WikiSection en_disease validation, 2 streams (2,645 tokens, 15 switches, regimes recur, declared length 190.5)

| Mechanism | decoder | token acc. | boundary F1 ±8 | ±16 | switches matched | mean delay | spurious /1k | Brier | ECE | ms/token | passes per token |
|---|---|---|---|---|---|---|---|---|---|---|---|
| asking, full history, 2 orderings | raw argmax | 0.564 | 0.07 | 0.10 | 12/15 | 10.2 | 88.1 | 0.628 | 0.107 | 248 | 1 stream + 2 questions, batched |
| asking, full history, 2 orderings | persist 3 | 0.560 | 0.14 | 0.33 | 12/15 | 19.3 | 17.4 | 0.628 | 0.107 | 248 | 1 stream + 2 questions, batched |
| asking, full history, 2 orderings | forward filter η=1 | 0.558 | 0.08 | 0.28 | 10/15 | 17.3 | 9.5 | 0.857 | 0.423 | 248 | 1 stream + 2 questions, batched |
| asking, full history, 2 orderings | forward filter η=0.5 | 0.538 | 0.04 | 0.27 | 10/15 | 20.4 | 7.6 | 0.888 | 0.440 | 248 | 1 stream + 2 questions, batched |
| asking, full history, 2 orderings | forward filter η=0.25 | 0.519 | 0.00 | 0.15 | 10/15 | 27.6 | 6.0 | 0.910 | 0.447 | 248 | 1 stream + 2 questions, batched |
| asking, full history, 2 orderings | filter, ÷ set prior (unlabelled), η=1 | 0.536 | 0.07 | 0.29 | 12/15 | 22.0 | 10.6 | 0.885 | 0.436 | 248 | 1 stream + 2 questions, batched |
| asking, full history, 2 orderings | filter, ÷ running prior, η=1 | 0.563 | 0.22 | 0.36 | 12/15 | 12.0 | 17.0 | 0.820 | 0.397 | 248 | 1 stream + 2 questions, batched |
