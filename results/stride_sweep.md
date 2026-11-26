| orderings | stride | ms/token (est.) | best causal acc. (decoder) | latency at cap, causal | latency at cap, fixed-lag |
|---|---|---|---|---|---|
| 1 | 1 | 100 | 0.811 (raw) | none | none |
| 1 | 2 | 50 | 0.813 (raw) | 31.8 (persist 5; 1.06/1k) | 31.3 (lag 4 η=1; 1.41/1k) |
| 1 | 4 | 25 | 0.817 (raw) | 26.3 (persist 2; 1.77/1k) | 28.2 (lag 1 η=2; 1.06/1k) |
| 1 | 8 | 12 | 0.814 (raw) | 27.8 (filter η=2; 1.77/1k) | 32.4 (lag 1 η=2; 1.06/1k) |
| 1 | 16 | 6 | 0.798 (raw) | 34.5 (filter η=0.25; 1.41/1k) | 39.6 (lag 1 η=1; 1.77/1k) |
| 2 | 1 | 149 | 0.817 (raw) | none | none |
| 2 | 2 | 74 | 0.820 (raw) | 29.9 (persist 5; 1.41/1k) | 27.7 (lag 2 η=1; 1.41/1k) |
| 2 | 4 | 37 | 0.824 (raw) | 28.6 (persist 3; 1.06/1k) | 29.9 (lag 2 η=2; 1.41/1k) |
| 2 | 8 | 19 | 0.817 (raw) | 29.5 (persist 2; 1.41/1k) | 38.7 (lag 2 η=2; 1.06/1k) |
| 2 | 16 | 9 | 0.809 (raw) | 31.1 (filter η=1; 1.41/1k) | 42.7 (lag 1 η=1; 1.41/1k) |
