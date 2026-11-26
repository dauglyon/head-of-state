## 3a. Directions from model-written passages: layer and smoothing (PubMed dev, all 10 abstracts)

| Mechanism | decoder | token acc. | boundary F1 ±8 | ±16 | matched | mean delay | spurious /1k | Brier | ECE | cost per token |
|---|---|---|---|---|---|---|---|---|---|---|
| 3a, layer 16, own token | raw argmax | 0.651 | 0.06 | 0.07 | 32/36 | 15.6 | 205.3 | 0.486 | 0.121 | 0.51 ms (whole-abstract pass); 0 extra passes when the stream is fed for asking |
| 3a, layer 16, own token | forward filter η=1 | 0.709 | 0.18 | 0.24 | 20/36 | 34.2 | 20.0 | 0.484 | 0.211 | 0.51 ms (whole-abstract pass); 0 extra passes when the stream is fed for asking |
| 3a, layer 16, mean of last 16 tokens | raw argmax | 0.706 | 0.15 | 0.27 | 21/36 | 35.1 | 20.6 | 0.462 | 0.216 | 0.51 ms (whole-abstract pass); 0 extra passes when the stream is fed for asking |
| 3a, layer 16, mean of last 16 tokens | forward filter η=1 | 0.696 | 0.09 | 0.15 | 17/36 | 43.9 | 8.5 | 0.520 | 0.235 | 0.51 ms (whole-abstract pass); 0 extra passes when the stream is fed for asking |
| 3a, layer 20, own token | raw argmax | 0.631 | 0.06 | 0.07 | 29/36 | 18.3 | 209.7 | 0.514 | 0.134 | 0.51 ms (whole-abstract pass); 0 extra passes when the stream is fed for asking |
| 3a, layer 20, own token | forward filter η=1 | 0.708 | 0.22 | 0.28 | 18/36 | 47.5 | 18.7 | 0.489 | 0.206 | 0.51 ms (whole-abstract pass); 0 extra passes when the stream is fed for asking |
| 3a, layer 20, mean of last 16 tokens | raw argmax | 0.701 | 0.17 | 0.28 | 16/36 | 30.1 | 23.1 | 0.492 | 0.246 | 0.51 ms (whole-abstract pass); 0 extra passes when the stream is fed for asking |
| 3a, layer 20, mean of last 16 tokens | forward filter η=1 | 0.702 | 0.09 | 0.15 | 16/36 | 35.5 | 8.5 | 0.506 | 0.223 | 0.51 ms (whole-abstract pass); 0 extra passes when the stream is fed for asking |
| 3a, layer 24, own token | raw argmax | 0.608 | 0.05 | 0.06 | 29/36 | 16.2 | 250.1 | 0.536 | 0.142 | 0.51 ms (whole-abstract pass); 0 extra passes when the stream is fed for asking |
| 3a, layer 24, own token | forward filter η=1 | 0.717 | 0.15 | 0.28 | 18/36 | 53.1 | 16.9 | 0.476 | 0.197 | 0.51 ms (whole-abstract pass); 0 extra passes when the stream is fed for asking |
| 3a, layer 24, mean of last 16 tokens | raw argmax | 0.704 | 0.13 | 0.27 | 18/36 | 38.4 | 20.0 | 0.512 | 0.270 | 0.51 ms (whole-abstract pass); 0 extra passes when the stream is fed for asking |
| 3a, layer 24, mean of last 16 tokens | forward filter η=1 | 0.715 | 0.12 | 0.19 | 17/36 | 37.6 | 6.4 | 0.491 | 0.214 | 0.51 ms (whole-abstract pass); 0 extra passes when the stream is fed for asking |
| 3a, layer 28, own token | raw argmax | 0.567 | 0.05 | 0.05 | 27/36 | 14.2 | 285.8 | 0.553 | 0.090 | 0.51 ms (whole-abstract pass); 0 extra passes when the stream is fed for asking |
| 3a, layer 28, own token | forward filter η=1 | 0.678 | 0.19 | 0.24 | 17/36 | 34.1 | 19.8 | 0.558 | 0.239 | 0.51 ms (whole-abstract pass); 0 extra passes when the stream is fed for asking |
| 3a, layer 28, mean of last 16 tokens | raw argmax | 0.670 | 0.18 | 0.28 | 16/36 | 38.1 | 19.3 | 0.516 | 0.228 | 0.51 ms (whole-abstract pass); 0 extra passes when the stream is fed for asking |
| 3a, layer 28, mean of last 16 tokens | forward filter η=1 | 0.672 | 0.08 | 0.21 | 17/36 | 33.6 | 8.7 | 0.568 | 0.256 | 0.51 ms (whole-abstract pass); 0 extra passes when the stream is fed for asking |

Chosen on dev for the rows below: layer 24, own token (best filtered token accuracy).

## 3a–3c on held-out abstracts 6–10 (abstracts 1–5 used for fitting 3b and 3c)

| Mechanism | decoder | token acc. | boundary F1 ±8 | ±16 | matched | mean delay | spurious /1k | Brier | ECE | cost per token |
|---|---|---|---|---|---|---|---|---|---|---|
| asking, full history, 1 ordering | raw argmax | 0.775 | 0.15 | 0.21 | 16/18 | 18.1 | 37.4 | 0.356 | 0.061 | 103 ms; 1 stream + 1 question pass |
| asking, full history, 1 ordering | forward filter η=1 | 0.760 | 0.26 | 0.32 | 15/18 | 27.3 | 2.1 | 0.456 | 0.223 | 103 ms; 1 stream + 1 question pass |
| 3a model-written directions (layer 24) | raw argmax | 0.638 | 0.06 | 0.06 | 14/18 | 12.5 | 238.6 | 0.514 | 0.158 | 0.51 ms (whole-abstract pass); 0 extra passes when the stream is fed for asking |
| 3a model-written directions (layer 24) | forward filter η=1 | 0.743 | 0.14 | 0.25 | 10/18 | 57.7 | 18.3 | 0.426 | 0.168 | 0.51 ms (whole-abstract pass); 0 extra passes when the stream is fed for asking |
| 3b directions distilled from asking (layer 24) | raw argmax | 0.709 | 0.06 | 0.06 | 18/18 | 6.3 | 215.3 | 0.451 | 0.202 | 0.51 ms (whole-abstract pass); 0 extra passes when the stream is fed for asking |
| 3b directions distilled from asking (layer 24) | forward filter η=1 | 0.821 | 0.33 | 0.43 | 15/18 | 7.9 | 19.1 | 0.286 | 0.097 | 0.51 ms (whole-abstract pass); 0 extra passes when the stream is fed for asking |
| 3c probe on gold labels, upper bound (layer 24) | raw argmax | 0.801 | 0.09 | 0.09 | 18/18 | 4.1 | 149.2 | 0.287 | 0.056 | 0.51 ms (whole-abstract pass); 0 extra passes when the stream is fed for asking |
| 3c probe on gold labels, upper bound (layer 24) | forward filter η=1 | 0.847 | 0.29 | 0.31 | 16/18 | 24.6 | 28.7 | 0.269 | 0.122 | 0.51 ms (whole-abstract pass); 0 extra passes when the stream is fed for asking |
| reference: bge-m3 similarity, last 32 tokens | raw argmax | 0.585 | 0.12 | 0.12 | 13/18 | 21.0 | 71.1 | 0.558 | 0.125 | about 1 ms; 1 embedding pass |
| reference: bge-m3 similarity, last 32 tokens | forward filter η=1 | 0.599 | 0.22 | 0.25 | 7/18 | 30.3 | 12.5 | 0.711 | 0.331 | about 1 ms; 1 embedding pass |

## 3d. Cascade: cheap reading everywhere, asking where the filter's entropy exceeds a threshold

| Cheap reading | set | threshold (nats) | fraction asked | token acc. | matched | mean delay | spurious /1k | est. ms/token |
|---|---|---|---|---|---|---|---|---|
| 3a | 1–10 | 0.00 | 1.00 | 0.774 | 32/36 | 25.9 | 3.5 | 104 |
| 3a | 1–10 | 0.05 | 0.82 | 0.774 | 32/36 | 25.9 | 3.5 | 85 |
| 3a | 1–10 | 0.10 | 0.59 | 0.774 | 32/36 | 25.9 | 3.5 | 61 |
| 3a | 1–10 | 0.20 | 0.38 | 0.772 | 32/36 | 26.4 | 3.5 | 40 |
| 3a | 1–10 | 0.30 | 0.28 | 0.767 | 32/36 | 26.6 | 4.4 | 30 |
| 3a | 1–10 | 0.50 | 0.17 | 0.752 | 32/36 | 28.9 | 2.3 | 18 |
| 3a | 1–10 | 0.70 | 0.09 | 0.810 | 31/36 | 19.6 | 2.9 | 9 |
| 3a | 1–10 | 1.00 | 0.02 | 0.751 | 24/36 | 45.4 | 13.1 | 2 |
| 3a | 1–10 | 1.30 | 0.00 | 0.722 | 20/36 | 53.9 | 14.1 | 1 |
| 3a | 1–10 | 2.61 | 0.00 | 0.717 | 18/36 | 53.1 | 16.9 | 1 |
| 3b | 6–10 | 0.00 | 1.00 | 0.760 | 15/18 | 27.3 | 2.1 | 104 |
| 3b | 6–10 | 0.05 | 0.74 | 0.760 | 15/18 | 27.3 | 2.1 | 77 |
| 3b | 6–10 | 0.10 | 0.53 | 0.760 | 15/18 | 27.2 | 2.1 | 55 |
| 3b | 6–10 | 0.20 | 0.31 | 0.758 | 15/18 | 27.4 | 2.5 | 32 |
| 3b | 6–10 | 0.30 | 0.21 | 0.756 | 15/18 | 27.5 | 2.5 | 22 |
| 3b | 6–10 | 0.50 | 0.13 | 0.757 | 15/18 | 26.3 | 3.3 | 14 |
| 3b | 6–10 | 0.70 | 0.08 | 0.780 | 17/18 | 20.4 | 4.2 | 9 |
| 3b | 6–10 | 1.00 | 0.01 | 0.817 | 15/18 | 8.3 | 17.9 | 1 |
| 3b | 6–10 | 1.30 | 0.00 | 0.822 | 15/18 | 8.0 | 19.1 | 1 |
| 3b | 6–10 | 2.61 | 0.00 | 0.821 | 15/18 | 7.9 | 19.1 | 1 |
