| orderings | stride | ms/token (est.) | best causal acc. (decoder) | latency at cap, causal | latency at cap, fixed-lag |
|---|---|---|---|---|---|
| 1 | 1 | 100 | 0.811 (raw) | none | none |
| 1 | 2 | 50 | 0.813 (raw) | 32.5 (persist 5; 1.06/1k) | 32.0 (lag 4 η=1; 1.41/1k) |
| 1 | 4 | 25 | 0.817 (raw) | 27.9 (persist 2; 1.77/1k) | 29.8 (lag 1 η=2; 1.06/1k) |
| 1 | 8 | 12 | 0.814 (raw) | 30.8 (filter η=2; 1.77/1k) | 35.5 (lag 1 η=2; 1.06/1k) |
| 1 | 16 | 6 | 0.798 (raw) | 42.6 (filter η=0.25; 1.41/1k) | 47.7 (lag 1 η=1; 1.77/1k) |
| 2 | 1 | 149 | 0.817 (raw) | none | none |
| 2 | 2 | 74 | 0.820 (raw) | 30.6 (persist 5; 1.41/1k) | 28.4 (lag 2 η=1; 1.41/1k) |
| 2 | 4 | 37 | 0.824 (raw) | 30.2 (persist 3; 1.06/1k) | 31.5 (lag 2 η=2; 1.41/1k) |
| 2 | 8 | 19 | 0.817 (raw) | 32.5 (persist 2; 1.41/1k) | 41.8 (lag 2 η=2; 1.06/1k) |
| 2 | 16 | 9 | 0.809 (raw) | 39.0 (filter η=1; 1.41/1k) | 50.6 (lag 1 η=1; 1.41/1k) |
