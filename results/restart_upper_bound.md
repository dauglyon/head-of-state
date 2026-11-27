True segment starts given; each state's context restarts there. Evidence for the true state per token (own minus mean of the other occurring states), by tokens since the segment start.

| Model | set | all | 0-8 | 8-32 | 32-128 | 128+ | positive | state predictions minus plain |
|---|---|---|---|---|---|---|---|---|
| Qwen--Qwen3.5-4B-Base | news | +0.007 | +0.037 | +0.014 | +0.007 | +0.004 | 52% | +0.03 |
| Qwen--Qwen3.5-4B-Base | articles | +0.011 | +0.209 | +0.070 | +0.003 | -0.004 | 52% | +0.03 |
| Qwen--Qwen3.5-4B-Base | pubmed | +0.012 | +0.072 | +0.015 | +0.007 | +0.001 | 52% | +0.07 |
| h2o-lightning-4b | news | +0.030 | +0.028 | +0.085 | +0.026 | +0.024 | 55% | +0.03 |
| h2o-lightning-4b | articles | +0.024 | +0.137 | +0.068 | -0.010 | +0.032 | 52% | +0.09 |
| h2o-lightning-4b | pubmed | -0.007 | +0.062 | +0.036 | -0.007 | -0.076 | 47% | +0.16 |
