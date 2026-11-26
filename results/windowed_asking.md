## 1a. Delay against window, PubMed dev (10 abstracts, 36 switches; readings every 2nd token, 1 ordering)

| Question sees | decoder | token acc. | boundary F1 ±8 | ±16 | matched | mean delay | spurious /1k | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| last 30 tokens (1×) | raw | 0.757 | 0.18 | 0.27 | 29/36 | 18.6 | 35.6 | 0.355 | 0.097 |
| last 30 tokens (1×) | persist | 0.745 | 0.27 | 0.41 | 25/36 | 22.0 | 10.4 | 0.355 | 0.097 |
| last 30 tokens (1×) | forward filter η=1 | 0.754 | 0.23 | 0.36 | 24/36 | 17.2 | 12.7 | 0.443 | 0.210 |
| last 60 tokens (2×) | raw | 0.798 | 0.20 | 0.28 | 28/36 | 15.9 | 17.9 | 0.321 | 0.106 |
| last 60 tokens (2×) | persist | 0.774 | 0.26 | 0.45 | 26/36 | 19.9 | 2.9 | 0.321 | 0.106 |
| last 60 tokens (2×) | forward filter η=1 | 0.776 | 0.20 | 0.38 | 25/36 | 20.4 | 3.7 | 0.414 | 0.204 |
| last 120 tokens (4×) | raw | 0.763 | 0.18 | 0.24 | 30/36 | 22.5 | 16.9 | 0.372 | 0.133 |
| last 120 tokens (4×) | persist | 0.733 | 0.27 | 0.38 | 25/36 | 26.0 | 2.5 | 0.372 | 0.133 |
| last 120 tokens (4×) | forward filter η=1 | 0.734 | 0.17 | 0.29 | 28/36 | 30.9 | 3.7 | 0.501 | 0.245 |
| full history | raw | 0.797 | 0.19 | 0.25 | 32/36 | 17.1 | 25.4 | 0.309 | 0.040 |
| full history | persist | 0.775 | 0.30 | 0.40 | 32/36 | 26.9 | 2.5 | 0.309 | 0.040 |
| full history | forward filter η=1 | 0.766 | 0.23 | 0.37 | 31/36 | 26.7 | 2.7 | 0.424 | 0.201 |

## 1c. S1 lag streams (12; scored from 16 tokens before the switch; readings every 2nd token, 1 ordering)

| Question sees | decoder | token acc. | matched | mean delay | spurious /1k | delay by previous segment 50 / 200 / 800 / 3000 |
|---|---|---|---|---|---|---|
| last 48 tokens (0.25×) | persist | 0.870 | 12/12 | 17.8 | 15.0 | 17 / 7 / 37 / 10 |
| last 48 tokens (0.25×) | forward filter η=1 | 0.878 | 12/12 | 18.8 | 11.8 | 17 / 7 / 39 / 13 |
| last 95 tokens (0.5×) | persist | 0.700 | 12/12 | 22.2 | 11.0 | 20 / 7 / 36 / 25 |
| last 95 tokens (0.5×) | forward filter η=1 | 0.704 | 12/12 | 22.5 | 7.4 | 20 / 7 / 36 / 27 |
| last 190 tokens (1×) | persist | 0.860 | 11/12 | 19.1 | 8.3 | 21 / 8 / 35 / 9 |
| last 190 tokens (1×) | forward filter η=1 | 0.864 | 11/12 | 20.0 | 4.4 | 21 / 9 / 37 / 10 |
| full history | persist | 0.916 | 12/12 | 22.7 | 5.9 | 21 / 7 / 53 / 10 |
| full history | forward filter η=1 | 0.919 | 12/12 | 23.7 | 2.9 | 21 / 7 / 54 / 13 |
