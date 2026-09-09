| System | Top-1 | 95% CI | Top-3 | F1-macro | κ | P@5 | p vs. full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 5.7% | [3.0, 8.4] | 14.4% | 0.040 | 0.007 | — | 0.000 |
| *Majority class (oracle floor) | 21.4% | [16.7, 26.1] | 46.5% | 0.021 | 0.000 | — | 0.002 |
| TF-IDF → nearest NACE section | 25.8% | [21.1, 30.8] | 40.5% | 0.186 | 0.210 | — | 0.011 |
| Zero-shot embeddings (no taxonomy guidance) | 29.1% | [24.1, 34.4] | 62.2% | 0.239 | 0.240 | — | 0.023 |
| Unguided KeyBERT | 29.1% | [24.1, 34.4] | 62.2% | 0.239 | 0.240 | — | 0.023 |
| Ours: taxonomy-guided | 34.4% | [29.1, 39.8] | 66.9% | 0.276 | 0.296 | — | — |
