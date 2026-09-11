# Coding the errors

Fill in `author_category` for every row of `author_error_coding.csv` with exactly one
of the category names below, and `author_note` if a row needs a word of explanation.
Read the English translation, the gold section and the predicted section, and ask why
the system chose the predicted one. Code each row on its own; do not look at
`error_analysis.csv` until you are done, or the coding is no longer blind.

| Category | Use it when |
| --- | --- |
| `ambiguous_sector_definition` | The NACE section boundary itself is unclear for this business; two sections are defensible. |
| `multi_sector_company` | The company genuinely operates in several sections; the gold label picks one. |
| `taxonomy_granularity` | The correct activity sits in a NACE division whose parent section is counterintuitive (e.g. repair, holdings). |
| `short_text` | Too little text to determine the sector. |
| `boilerplate_only` | The purpose field is legal boilerplate (holding, asset management, participation) with no activity signal. |
| `seed_gap` | The correct sector's seed list lacks the vocabulary in this text. |
| `seed_leakage` | A wrong sector's seed list contains a term that dominates this text. |
| `annotation_error` | The gold label is wrong; the prediction is defensible. |

Then run `make author-coding-score`.
