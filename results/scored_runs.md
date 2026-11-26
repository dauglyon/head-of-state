| Set | decoder | token acc. | boundary F1 ±8 | ±16 | matched | mean delay | spurious /1k | Brier | ECE | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|
| S3test (30 streams, 13327 tokens) | persist 1 | 0.712 | 0.26 | 0.37 | 83/98 | 30.7 | 8.2 | 0.459 | 0.130 | 14 |
| S3test (30 streams, 13327 tokens) | forward filter η=1 | 0.655 | 0.19 | 0.26 | 77/98 | 43.5 | 3.0 | 0.590 | 0.274 | 14 |
| S2null (4 streams, 15039 tokens) | persist 1 | 0.622 | 0.00 | 0.00 | 0/0 | nan | 24.5 | 0.528 | 0.080 | 21 |
| S2null (4 streams, 15039 tokens) | forward filter η=1 | 0.676 | 0.00 | 0.00 | 0/0 | nan | 5.3 | 0.568 | 0.259 | 21 |
