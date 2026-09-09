| System | Top-1 | 95% CI | Top-3 | F1-macro | κ | P@5 | p vs. full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 5.0% | [2.7, 7.7] | 13.7% | 0.036 | -0.001 | — | 0.000 |
| *Majority class (oracle floor) | 21.7% | [17.1, 26.4] | 46.8% | 0.020 | 0.000 | — | 0.002 |
| TF-IDF → nearest NACE section | 25.4% | [20.7, 30.4] | 40.1% | 0.184 | 0.207 | — | 0.007 |
| Zero-shot embeddings (no taxonomy guidance) | 28.4% | [23.4, 33.4] | 62.5% | 0.243 | 0.234 | — | 0.008 |
| Unguided KeyBERT | 28.4% | [23.4, 33.4] | 62.5% | 0.243 | 0.234 | — | 0.008 |
| Ours: taxonomy-guided | 34.8% | [29.4, 40.1] | 67.2% | 0.293 | 0.300 | — | — |
