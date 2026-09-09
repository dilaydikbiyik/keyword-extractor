| System | Top-1 | 95% CI | Top-3 | F1-macro | κ | P@5 | p vs. full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 6.0% | [3.3, 8.7] | 13.7% | 0.048 | 0.010 | — | 0.000 |
| *Majority class (oracle floor) | 22.4% | [17.7, 27.4] | 47.5% | 0.022 | 0.000 | — | 0.001 |
| TF-IDF → nearest NACE section | 26.1% | [21.4, 31.1] | 40.5% | 0.203 | 0.214 | — | 0.004 |
| Zero-shot embeddings (no taxonomy guidance) | 29.8% | [24.7, 35.1] | 61.9% | 0.246 | 0.248 | — | 0.008 |
| Unguided KeyBERT | 29.8% | [24.7, 35.1] | 61.9% | 0.246 | 0.248 | — | 0.008 |
| Ours: taxonomy-guided | 36.1% | [30.8, 41.8] | 67.9% | 0.290 | 0.313 | — | — |
