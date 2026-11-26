Tuning sample. Emission: P(actual next token | text so far, state) with the owner's wording; weight 1, declared length.

| Set | document | tokens | D (nats/token) | weakest pair | SD | lag-1 | calibration slope | state offsets | shared distortion | token acc. | segments right | placed within 16 | false /1k | time to notice | ms/token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wiki_articles | wiki-https://en.wikipedia.org/wiki/Emetophobia | 1568 | 0.340 | treatment vs symptom: -0.332 | 0.65 | 0.27 | 1.69 | -0.32 to +0.37 | -4.84 | 0.467 | 2/4 | 1/3 | 3.2 | 283 | 271 |
| wiki_articles | wiki-https://en.wikipedia.org/wiki/Hidradenitis_suppurativa | 1787 | 0.815 | cause vs mechanism: -0.492 | 0.93 | 0.22 | 1.55 | -0.50 to +0.45 | -5.24 | 0.560 | 3/5 | 2/4 | 2.8 | 58 | 274 |
| wiki_articles | wiki-https://en.wikipedia.org/wiki/Paroxysmal_sympathetic_hyperactivity | 1961 | 0.783 | prognosis vs symptom: 0.012 | 0.83 | 0.23 | 2.47 | -0.53 to +0.64 | -5.58 | 0.709 | 6/7 | 5/6 | 3.6 | 29 | 275 |
| wiki_articles, catalogue | wiki-https://en.wikipedia.org/wiki/Emetophobia | 1568 | 0.477 | history vs symptom: -0.237 | 0.77 | 0.23 | 1.67 | -0.39 to +0.47 | -5.27 | 0.569 | 2/4 | 1/3 | 5.7 | 81 | 210 |
| wiki_articles, catalogue | wiki-https://en.wikipedia.org/wiki/Hidradenitis_suppurativa | 1787 | 0.743 | mechanism vs diagnosis: -1.254 | 0.99 | 0.22 | 1.11 | -0.48 to +0.31 | -5.37 | 0.495 | 2/5 | 3/4 | 1.7 | 13 | 211 |
| wiki_articles, catalogue | wiki-https://en.wikipedia.org/wiki/Paroxysmal_sympathetic_hyperactivity | 1961 | 0.962 | prognosis vs diagnosis: -0.075 | 0.91 | 0.23 | 2.65 | -0.56 to +0.36 | -5.61 | 0.921 | 6/7 | 5/6 | 0.0 | 24 | 213 |
| swda | swda-4721 | 1616 | 0.207 | abandoned vs opinion: -0.614 | 0.73 | 0.21 | 0.60 | -0.27 to +0.39 | -4.47 | 0.154 | 14/113 | 9/112 | 1.9 | 15 | 279 |
| swda | swda-4928 | 1633 | 0.188 | statement vs opinion: -0.642 | 0.66 | 0.09 | 0.13 | -0.38 to +0.54 | -4.60 | 0.085 | 8/66 | 2/65 | 0.0 | 7 | 279 |
| swda, catalogue | swda-4721 | 1616 | 0.355 | abandoned vs statement: -0.609 | 0.82 | 0.21 | 1.47 | -0.56 to +0.44 | -4.32 | 0.500 | 33/113 | 6/112 | 0.0 | 10 | 170 |
| swda, catalogue | swda-4928 | 1633 | 0.275 | abandoned vs statement: -0.922 | 0.71 | 0.16 | 1.71 | -0.49 to +0.59 | -4.50 | 0.674 | 24/66 | 7/65 | 1.2 | 14 | 169 |
| tripod | tripod-Pretty Woman | 1016 | -0.032 | progress vs final push: -0.607 | 0.47 | 0.05 | -0.63 | -0.34 to +0.34 | -4.57 | 0.148 | 1/6 | 0/5 | 0.0 | nan | 208 |
| tripod | tripod-We Own the Night (film) | 904 | 0.156 | progress vs final push: -0.394 | 0.51 | 0.11 | 0.89 | -0.33 to +0.27 | -4.61 | 0.273 | 2/6 | 0/5 | 1.1 | 60 | 207 |
| tripod | tripod-Men in Black (film) | 1378 | 0.021 | progress vs final push: -1.205 | 0.60 | 0.12 | 0.26 | -0.53 to +0.43 | -4.55 | 0.226 | 2/6 | 0/5 | 0.7 | 70 | 210 |
| tripod, catalogue | tripod-Pretty Woman | 1016 | 0.209 | complications vs final push: -0.471 | 0.44 | 0.13 | 0.76 | -0.35 to +0.38 | -4.68 | 0.175 | 2/6 | 0/5 | 1.0 | nan | 137 |
| tripod, catalogue | tripod-We Own the Night (film) | 904 | 0.347 | complications vs final push: -0.385 | 0.55 | 0.19 | 1.62 | -0.42 to +0.38 | -4.74 | 0.273 | 2/6 | 0/5 | 1.1 | nan | 142 |
| tripod, catalogue | tripod-Men in Black (film) | 1378 | 0.156 | new situation vs final push: -0.669 | 0.50 | 0.20 | 0.36 | -0.27 to +0.40 | -4.56 | 0.190 | 1/6 | 0/5 | 0.7 | nan | 139 |
| newsdiscourse | nd-validation-NYT_ENG_20090825.0142 | 490 | 0.099 | previous event vs main event: -0.415 | 0.47 | 0.19 | 0.38 | -0.26 to +0.25 | -4.56 | 0.082 | 1/12 | 1/11 | 0.0 | 2 | 273 |
| newsdiscourse | nd-validation-nyt_corpus_data_2007_01_07_1817145 | 1426 | 0.059 | current context vs main event: -0.494 | 0.50 | 0.22 | -0.21 | -0.28 to +0.41 | -4.60 | 0.034 | 2/20 | 1/19 | 0.0 | 4 | 281 |
| newsdiscourse | nd-validation-PReuters_108 | 562 | 0.216 | previous event vs main event: -0.269 | 0.54 | 0.17 | 0.79 | -0.36 to +0.27 | -4.85 | 0.103 | 1/12 | 2/11 | 1.8 | 28 | 273 |
| newsdiscourse, catalogue | nd-validation-NYT_ENG_20090825.0142 | 490 | 0.390 | evaluation vs previous event: -0.289 | 0.68 | 0.28 | 1.20 | -0.56 to +0.46 | -4.79 | 0.143 | 1/12 | 1/11 | 2.0 | 7 | 170 |
| newsdiscourse, catalogue | nd-validation-nyt_corpus_data_2007_01_07_1817145 | 1426 | 0.251 | current context vs previous event: -0.415 | 0.66 | 0.26 | 0.50 | -0.61 to +0.35 | -4.95 | 0.251 | 5/20 | 6/19 | 3.5 | 31 | 176 |
| newsdiscourse, catalogue | nd-validation-PReuters_108 | 562 | 0.231 | not applicable vs previous event: -0.525 | 0.69 | 0.22 | 0.97 | -0.72 to +0.49 | -4.67 | 0.302 | 2/12 | 1/11 | 3.6 | 35 | 169 |
