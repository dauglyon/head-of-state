   Building head-of-state @ file:///home/dauglyon/work/head-of-state
      Built head-of-state @ file:///home/dauglyon/work/head-of-state
Uninstalled 1 package in 0.21ms
Installed 1 package in 0.26ms

## PubMed dev abstracts 11–40 (13932 tokens); directions fitted on abstracts 1–10 labelled by asking

| Readout | decoder | token acc. | boundary F1 ±8 | ±16 | matched | mean delay | spurious /1k | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| distilled directions | raw argmax | 0.639 | 0.05 | 0.06 | 94/97 | 5.8 | 238.6 | 0.518 | 0.153 |
| distilled directions | forward filter η=1 | 0.734 | 0.29 | 0.35 | 80/97 | 19.6 | 23.0 | 0.439 | 0.174 |
| distilled directions | forward filter η=0.5 | 0.742 | 0.34 | 0.44 | 78/97 | 17.2 | 13.5 | 0.396 | 0.121 |
| distilled directions | forward filter η=0.25 | 0.744 | 0.27 | 0.45 | 76/97 | 20.5 | 10.8 | 0.368 | 0.044 |
| distilled directions | persist 8 tokens | 0.725 | 0.25 | 0.42 | 72/97 | 27.4 | 4.8 | 0.518 | 0.153 |
| distilled directions | persist 16 tokens | 0.561 | 0.03 | 0.25 | 66/97 | 48.1 | 0.8 | 0.518 | 0.153 |
| distilled directions | fixed-lag Viterbi, lag 8 (+8 latency) | 0.756 | 0.37 | 0.46 | 65/97 | 19.7 | 13.7 | 0.518 | 0.153 |
| distilled + asked every 8th token, fused | forward filter, cheap η=1 | 0.790 | 0.21 | 0.27 | 82/97 | 13.5 | 29.1 | 0.351 | 0.135 |
| distilled + asked every 8th token, fused | forward filter, cheap η=0.25 | 0.743 | 0.15 | 0.32 | 81/97 | 25.6 | 6.7 | 0.415 | 0.150 |
| asking alone, every 8th token | forward filter η=1 | 0.694 | 0.19 | 0.35 | 74/97 | 34.5 | 2.0 | 0.535 | 0.243 |

## WikiSection validation streams 3–6 (5455 tokens, regimes recur); directions fitted on streams 0–2 labelled by asking (regimes seen in warm-up: 10 of 10)

| Readout | decoder | token acc. | boundary F1 ±8 | ±16 | matched | mean delay | spurious /1k | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| distilled directions | raw argmax | 0.277 | 0.03 | 0.03 | 26/27 | 14.1 | 317.0 | 0.837 | 0.043 |
| distilled directions | forward filter η=1 | 0.308 | 0.16 | 0.22 | 18/27 | 37.0 | 30.6 | 1.265 | 0.605 |
| distilled directions | forward filter η=0.5 | 0.307 | 0.07 | 0.28 | 14/27 | 48.9 | 19.1 | 1.229 | 0.582 |
| distilled directions | forward filter η=0.25 | 0.300 | 0.02 | 0.25 | 10/27 | 35.7 | 12.3 | 1.173 | 0.544 |
| distilled directions | persist 8 tokens | 0.330 | 0.22 | 0.41 | 13/27 | 69.1 | 7.9 | 0.837 | 0.043 |
| distilled directions | persist 16 tokens | 0.262 | 0.09 | 0.14 | 8/27 | 98.4 | 1.6 | 0.837 | 0.043 |
| distilled directions | fixed-lag Viterbi, lag 8 (+8 latency) | 0.308 | 0.23 | 0.25 | 16/27 | 43.4 | 22.9 | 0.837 | 0.043 |
| distilled + asked every 2th token, fused | forward filter, cheap η=1 | 0.556 | 0.18 | 0.24 | 20/27 | 19.9 | 25.5 | 0.839 | 0.410 |
| distilled + asked every 2th token, fused | forward filter, cheap η=0.25 | 0.582 | 0.27 | 0.39 | 22/27 | 33.4 | 9.0 | 0.803 | 0.391 |
| asking alone, every 2th token | forward filter η=1 | 0.599 | 0.26 | 0.38 | 21/27 | 34.8 | 8.4 | 0.765 | 0.374 |

## S1 lag streams (12): delay by previous-segment length, scored from 16 tokens before the switch

| Readout | decoder | token acc. | matched | mean delay | spurious /1k | delay at 50 / 200 / 800 / 3000 |
|---|---|---|---|---|---|---|
| distilled directions (fitted on WikiSection validation) | forward filter η=1 | 0.702 | 12/12 | 13.3 | 30.4 | 19 / 5 / 23 / 6 |
| asking, full history, every 2nd token | forward filter η=1 | 0.879 | 12/12 | 23.7 | 2.9 | 21 / 7 / 54 / 13 |
