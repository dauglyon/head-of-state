Readings of the new segment's first tokens: in the stream (after the earlier text) and alone (nothing before). P(new state) is the mean probability the reading gives the new segment's state after k tokens.

| State list | switches | first reading naming the new state, in stream (median tokens) | alone | P(new) at 8: stream / alone | P(new) at 16: stream / alone | P(new) at 32: stream / alone | P(new) at 64: stream / alone | P(new) at 128: stream / alone |
|---|---|---|---|---|---|---|---|---|
| ag_news | 21 | 7 (1 never) | 2 | 0.46 / 0.76 (n=21) | 0.56 / 0.82 (n=21) | 0.61 / 0.82 (n=21) | 0.67 / 0.79 (n=21) | 0.65 / 0.83 (n=15) |
| wikisection | 8 | 4 (1 never) | 4 (1 never) | 0.35 / 0.33 (n=8) | 0.42 / 0.55 (n=8) | 0.41 / 0.55 (n=8) | 0.61 / 0.70 (n=8) | 0.46 / 0.46 (n=8) |
| imdb | 19 | 18 | 1 | 0.34 / 0.65 (n=19) | 0.46 / 0.74 (n=19) | 0.55 / 0.78 (n=19) | 0.71 / 0.89 (n=19) | 0.77 / 0.93 (n=19) |
