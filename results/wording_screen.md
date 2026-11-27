Tuning documents only. Per set: D = mean pairwise evidence between occurring states (nats per token) · share of tokens whose evidence (true state against the mean of the others) is positive · that evidence's mean over its SD (scale-free across models) · pairs whose 95% interval is below zero · lift range (nats) · drop below the plain prediction (nats) · ms per token · paired difference in per-token evidence from the wording in PAIR_WITH, same model, same tokens, 95% interval by 16-token blocks. Leads under 0.1 nats per token count as ties (best-of-N selection). Cells marked mis-fit are not read (MISFIT).

| Wording (paired with) | model | news | articles | pubmed |
|---|---|---|---|---|
| per-state-cache (—) | Qwen--Qwen3.5-4B-Base | 0.00 · 52% · 0.04 · 0/12 · 0.00 · +0.0 · 158 | 0.00 · 50% · 0.02 · 1/42 · 0.00 · +0.0 · 347 | 0.01 · 52% · 0.07 · 0/20 · 0.00 · +0.0 · 188 |
| per-state-cache (—) | h2o-lightning-4b | 0.02 · 58% · 0.13 · 0/12 · 0.01 · +0.0 · 158 | 0.03 · 59% · 0.11 · 0/42 · 0.02 · +0.1 · 347 | 0.02 · 50% · 0.01 · 7/20 · 0.10 · +0.1 · 188 |
| ref-instruction (—) | h2o-lightning-4b | 0.75 · 87% · 0.98 · 0/12 · 0.59 · -5.5 · 135 | 0.64 · 85% · 0.91 · 0/42 · 0.95 · -5.6 · 276 | 0.27 · 75% · 0.67 · 3/20 · 0.65 · -4.6 · 148 |
