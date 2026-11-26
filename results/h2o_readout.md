| Set | model | decoder | token acc. | boundary F1 ±8 | ±16 | matched | mean delay | spurious /1k | Brier | ECE | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S3test | Qwen3.5-4B (base) | persist 1 | 0.701 | 0.26 | 0.40 | 83/98 | 30.7 | 8.2 | 0.476 | 0.142 | 14 |
| S3test | Qwen3.5-4B (base) | forward filter η=1 | 0.647 | 0.23 | 0.33 | 77/98 | 42.8 | 3.0 | 0.604 | 0.279 | 14 |
| S3test | H2O-Lightning-4B | persist 1 | 0.757 | 0.43 | 0.54 | 85/98 | 22.8 | 7.2 | 0.390 | 0.059 | 13 |
| S3test | H2O-Lightning-4B | forward filter η=1 | 0.721 | 0.30 | 0.49 | 83/98 | 31.0 | 3.8 | 0.476 | 0.199 | 13 |
| S2null | Qwen3.5-4B (base) | persist 1 | 0.622 | 0.00 | 0.00 | 0/0 | nan | 24.5 | 0.528 | 0.080 | 21 |
| S2null | Qwen3.5-4B (base) | forward filter η=1 | 0.674 | 0.00 | 0.00 | 0/0 | nan | 5.3 | 0.568 | 0.260 | 21 |
| S2null | H2O-Lightning-4B | persist 1 | 0.705 | 0.00 | 0.00 | 0/0 | nan | 14.4 | 0.443 | 0.055 | 21 |
| S2null | H2O-Lightning-4B | forward filter η=1 | 0.719 | 0.00 | 0.00 | 0/0 | nan | 4.3 | 0.511 | 0.241 | 21 |
