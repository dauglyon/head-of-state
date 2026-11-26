## 1a. Delay against window, PubMed dev (10 abstracts, 36 switches; readings every 2nd token, 1 ordering)

| Question sees | decoder | token acc. | boundary F1 ±8 | ±16 | matched | mean delay | spurious /1k | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|
| last 30 tokens (1×) | raw | 0.756 | 0.18 | 0.27 | 29/36 | 17.9 | 35.6 | 0.358 | 0.099 |
| last 30 tokens (1×) | persist | 0.743 | 0.27 | 0.41 | 25/36 | 21.3 | 10.4 | 0.358 | 0.099 |
| last 30 tokens (1×) | forward filter η=1 | 0.752 | 0.23 | 0.36 | 24/36 | 16.5 | 12.7 | 0.447 | 0.212 |
| last 60 tokens (2×) | raw | 0.796 | 0.20 | 0.28 | 28/36 | 15.2 | 17.9 | 0.325 | 0.108 |
| last 60 tokens (2×) | persist | 0.772 | 0.26 | 0.45 | 26/36 | 19.1 | 2.9 | 0.325 | 0.108 |
| last 60 tokens (2×) | forward filter η=1 | 0.773 | 0.20 | 0.38 | 25/36 | 19.7 | 3.7 | 0.419 | 0.206 |
| last 120 tokens (4×) | raw | 0.761 | 0.18 | 0.24 | 30/36 | 21.9 | 16.9 | 0.376 | 0.135 |
| last 120 tokens (4×) | persist | 0.731 | 0.27 | 0.38 | 25/36 | 25.3 | 2.5 | 0.376 | 0.135 |
| last 120 tokens (4×) | forward filter η=1 | 0.732 | 0.17 | 0.29 | 28/36 | 30.1 | 3.7 | 0.505 | 0.247 |
| full history | raw | 0.795 | 0.19 | 0.25 | 32/36 | 16.4 | 25.4 | 0.313 | 0.040 |
| full history | persist | 0.772 | 0.30 | 0.40 | 32/36 | 26.2 | 2.5 | 0.313 | 0.040 |
| full history | forward filter η=1 | 0.764 | 0.23 | 0.37 | 31/36 | 26.1 | 2.7 | 0.428 | 0.203 |

## 1c. S1 lag streams (12; scored from 16 tokens before the switch; readings every 2nd token, 1 ordering)

| Question sees | decoder | token acc. | matched | mean delay | spurious /1k | delay by previous segment 50 / 200 / 800 / 3000 |
|---|---|---|---|---|---|---|
| last 48 tokens (0.25×) | persist | 0.735 | 12/12 | 17.8 | 15.0 | 17 / 7 / 37 / 10 |
| last 48 tokens (0.25×) | forward filter η=1 | 0.767 | 12/12 | 18.8 | 11.8 | 17 / 7 / 39 / 13 |
| last 95 tokens (0.5×) | persist | 0.811 | 12/12 | 22.2 | 11.0 | 20 / 7 / 36 / 25 |
| last 95 tokens (0.5×) | forward filter η=1 | 0.826 | 12/12 | 22.5 | 7.4 | 20 / 7 / 36 / 27 |
| last 190 tokens (1×) | persist | 0.841 | 11/12 | 19.1 | 8.3 | 21 / 8 / 35 / 9 |
| last 190 tokens (1×) | forward filter η=1 | 0.857 | 11/12 | 20.0 | 4.4 | 21 / 9 / 37 / 10 |
| full history | persist | 0.867 | 12/12 | 22.7 | 5.9 | 21 / 7 / 53 / 10 |
| full history | forward filter η=1 | 0.879 | 12/12 | 23.7 | 2.9 | 21 / 7 / 54 / 13 |
