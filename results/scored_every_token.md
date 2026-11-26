| Set | reading | decoder | token acc. | boundary F1 ±8 | ±16 | matched | mean delay | spurious /1k | Brier | ECE | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S3test (30 streams, 13327 tokens) | every token (frozen) | persist 3 | 0.737 | 0.31 | 0.42 | 86/99 | 22.4 | 9.0 | 0.400 | 0.066 | 101 |
| S3test (30 streams, 13327 tokens) | every token (frozen) | forward filter η=1 | 0.732 | 0.29 | 0.39 | 85/99 | 23.2 | 8.3 | 0.506 | 0.245 | 101 |
| S3test (30 streams, 13327 tokens) | every 8th token (earlier) | persist 1 | 0.763 | 0.42 | 0.52 | 85/98 | 22.8 | 7.2 | 0.382 | 0.054 | 13 |
| S3test (30 streams, 13327 tokens) | every 8th token (earlier) | forward filter η=1 | 0.729 | 0.29 | 0.44 | 83/98 | 31.9 | 3.8 | 0.461 | 0.194 | 13 |
| ACI (10 streams, 11921 tokens) | every token (frozen) | persist 3 | 0.815 | 0.48 | 0.56 | 204/547 | 5.2 | 3.2 | 0.208 | 0.042 | 95 |
| ACI (10 streams, 11921 tokens) | every token (frozen) | forward filter η=1 | 0.828 | 0.49 | 0.56 | 207/547 | 4.5 | 3.6 | 0.276 | 0.119 | 95 |
| ACI (10 streams, 11921 tokens) | every 8th token (earlier) | persist 1 | 0.842 | 0.51 | 0.54 | 190/545 | 5.9 | 1.8 | 0.225 | 0.041 | 13 |
| ACI (10 streams, 11921 tokens) | every 8th token (earlier) | forward filter η=1 | 0.839 | 0.50 | 0.52 | 188/545 | 6.5 | 1.5 | 0.227 | 0.041 | 13 |
| S2null (4 streams, 15039 tokens) | every token (frozen) | persist 3 | 0.707 | 0.00 | 0.00 | 0/0 | nan | 15.8 | 0.449 | 0.063 | 160 |
| S2null (4 streams, 15039 tokens) | every token (frozen) | forward filter η=1 | 0.702 | 0.00 | 0.00 | 0/0 | nan | 8.4 | 0.570 | 0.282 | 160 |
| S2null (4 streams, 15039 tokens) | every 8th token (earlier) | persist 1 | 0.705 | 0.00 | 0.00 | 0/0 | nan | 14.4 | 0.443 | 0.055 | 21 |
| S2null (4 streams, 15039 tokens) | every 8th token (earlier) | forward filter η=1 | 0.719 | 0.00 | 0.00 | 0/0 | nan | 4.3 | 0.511 | 0.241 | 21 |
