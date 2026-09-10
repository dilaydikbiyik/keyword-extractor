| System | Top-1 | 95% CI | Top-3 | F1-macro | κ | P@5 | p vs. full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 5.0% | [2.7, 7.7] | 13.7% | 0.036 | -0.001 | — | 0.000 |
| *Majority class (oracle floor) | 21.7% | [17.1, 26.4] | 46.8% | 0.020 | 0.000 | — | 0.000 |
| TF-IDF → nearest NACE section | 44.5% | [38.8, 50.2] | 65.6% | 0.297 | 0.385 | — | 0.037 |
| Zero-shot embeddings (no taxonomy guidance) | 52.8% | [47.2, 58.5] | 81.9% | 0.371 | 0.475 | — | 1.000 |
| Unguided KeyBERT | 52.8% | [47.2, 58.5] | 81.9% | 0.371 | 0.475 | — | 1.000 |
| Ours: taxonomy-guided | 52.5% | [46.8, 58.2] | 81.6% | 0.356 | 0.474 | — | — |
