# Data

Three datasets sit under `data/`.  Only two of them are in version control, and
the difference matters for anyone trying to reproduce the reported numbers.

| Path | In git | What it is |
| --- | --- | --- |
| `data/taxonomy/sectors.json` | yes | 21 NACE Rev. 2 sections with hand-written seed keywords and descriptions. Authored for this project. |
| `data/evaluation/human_labels.json` | yes | The 30-document evaluation set: gold NACE section plus reference keywords. Authored for this project. |
| `data/raw/handelsregister_sample_10k.csv` | **no** | 9,993 German trade register business purposes. |

## The corpus is not distributed with this repository

`data/raw/handelsregister_sample_10k.csv` is excluded by `.gitignore`.  Its
redistribution licence has not been established, so it is not published here.
Two consequences:

1. A fresh clone can run the tests, but `make reproduce` will fail at the
   TF-IDF baseline, which fits its IDF statistics on this corpus.
2. Any submission built on these results needs the data question answered
   first — both for the artefact and for the paper's data statement.

**Open question for the supervisor:** may the sample be redistributed, and
under what terms?  Until that is answered, one of these has to happen:

- publish the sample under a stated licence, or
- publish the derived artefacts only (embeddings, TF-IDF vocabulary, the
  labelled evaluation set) and document how to rebuild the corpus, or
- ship a small synthetic stand-in corpus so the pipeline runs end to end, with
  the reported numbers clearly marked as coming from the real data.

## Evaluation set

`human_labels.json` currently holds 30 documents, annotated semi-manually by a
single annotator.  Both facts are load-bearing limitations — see
[`docs/paper_readiness.md`](../docs/paper_readiness.md) — and the queue built by
`python -m experiments.build_annotation_queue` exists to fix the first.

No inter-annotator agreement has been measured, because there is only one
annotator.  Cohen's κ is reported in the results tables, but it measures
*classifier vs. gold*, not annotator vs. annotator.
