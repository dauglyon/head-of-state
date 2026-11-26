# Design comparison

Delay figure: shortest mean delay (positions) among decoder settings with ≤2 spurious switches per 1k tokens and ≥80% of true switches matched; for fixed-lag decoders the decision latency is that delay plus the lag.

## E1

| Design | ms/token | Best committed acc. (decoder) | Shortest delay at the cap (decoder; spurious/1k, matched) |
|---|---|---|---|
| qafter (6 streams, 2830 tokens) | 149 | 0.835 (fixed-lag 32 L×1 η=1) | 19.0 (fixed-lag 32 L×1 η=1; 1.06, 90%) |
| embed (6 streams, 2830 tokens) | 1 | 0.554 (fixed-lag 16 L×1 η=2) | none meets the cap |
| gen (6 streams, 2830 tokens) | 40 | 0.293 (filter L×4 η=4) | none meets the cap |
| ↳ gen evidence gap by position | | 0: -0.063, 256: +0.080, 512: +0.163 | |
| ↳ gen evidence gap by tokens since switch (<) | | 8: +0.066, 16: -0.013, 32: +0.009, 64: -0.002, 128: -0.027, 256: +0.004, ∞: +0.263 | |
| genrefresh (6 streams, 2830 tokens) | 43 | 0.289 (surprisal-hazard γ=2 η=1) | none meets the cap |
| ↳ genrefresh evidence gap by position | | 0: -0.050, 256: +0.059, 512: +0.043 | |
| ↳ genrefresh evidence gap by tokens since switch (<) | | 8: +0.084, 16: +0.002, 32: +0.025, 64: +0.013, 128: -0.033, 256: -0.040, ∞: +0.038 | |
| qbefore (6 streams, 2830 tokens) | 105 | 0.505 (fixed-lag 16 L×1 η=4) | none meets the cap |

## E2

| Design | ms/token | Best committed acc. (decoder) | Shortest delay at the cap (decoder; spurious/1k, matched) |
|---|---|---|---|
| embed (2 streams, 2645 tokens) | 6 | 0.258 (fixed-lag 32 L×1 η=2) | none meets the cap |
| gen (2 streams, 2645 tokens) | 46 | 0.218 (surprisal-hazard γ=1 η=2) | none meets the cap |
| ↳ gen evidence gap by position | | 0: +0.029, 256: -0.001, 512: +0.007, 768: -0.010, 1024: +0.009, 1280: +0.009 | |
| ↳ gen evidence gap by tokens since switch (<) | | 8: +0.107, 16: +0.028, 32: +0.022, 64: +0.000, 128: +0.003, 256: -0.007, ∞: -0.011 | |
| qafter (2 streams, 2645 tokens) | 252 | 0.578 (fixed-lag 8 L×1 η=4) | none meets the cap |

