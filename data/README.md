# Data

Four things live under `data/`. Three are in version control; the difference
matters for anyone trying to reproduce the reported numbers.

| Path | In git | What it is |
| --- | --- | --- |
| `data/taxonomy/sectors.json` | yes | 21 NACE Rev. 2 sections with hand-written seed keywords and descriptions. Authored for this project. |
| `data/evaluation/human_labels.json` | yes | The evaluation set: gold NACE section plus reference keywords per document. Authored for this project. |
| `data/derived/tfidf_corpus_stats.json` | yes | Vocabulary and IDF weights derived from the corpus below. |
| `data/derived/tfidf_snowball_corpus_stats.json` | yes | The same over German Snowball stems, for `experiments/run_references.py`. |
| `data/derived/tfidf_char_corpus_stats.json` | yes | The same over character 3–5-grams, for `experiments/run_references.py`. |
| `data/raw/handelsregister_sample_10k.csv` | **no** | 9,993 German trade register business purposes. |

## The corpus is not redistributed — and reproduction does not need it

The sample was collected for this project from the German trade register. The
underlying facts are public record, but a bulk redistribution licence for the
collected file has not been established, so the file is not published here.
The one exception is the evaluation set: its 299 records, about 3% of the
sample, are committed with their labels so that every reported result can be
checked. They are public register entries, and company names are kept because
they are part of that record; no classifier reads them.

Rather than leave the repository unreproducible, the statistics the pipeline
actually consumes from the corpus are committed instead. The TF-IDF baseline
needs a vocabulary and a set of IDF weights, not the records; those are
aggregate counts, they cannot be inverted back into business descriptions, and
publishing them is standard practice for corpus-derived resources.

The substitution is exact, not approximate. On all 30 evaluation documents the
committed statistics produce **identical section rankings** to a vectoriser
fitted on the live corpus, with a maximum score difference of 1.5 × 10⁻⁸ — an
artefact of rounding the stored IDF weights to six decimals.
`tests/test_experiments.py::TestCommittedStatisticsReproduceTheCorpus` asserts
this whenever the corpus is present.

So a fresh clone can run:

```bash
make reproduce
```

Everything else — the taxonomy, the evaluation set, the embedding model — is
either in the repository or fetched on first use.

**What a clone cannot do:** `python main.py`, which batch-processes the raw CSV;
regenerating the statistics themselves (`python -m experiments.corpus_stats`,
`python -m experiments.run_references --refit`); and the label-free studies
(`experiments/run_rocchio.py`), whose unlabelled NACE pool is the corpus itself.
All of them need the corpus, and the last stops with a message saying so.

If the licence question is later settled in favour of publication, add the CSV,
drop the `data/raw/` line from `.gitignore`, and this section becomes a
footnote.

## Evaluation set

`human_labels.json` holds **299 documents sampled from the corpus**, labelled
by a language model (Claude, Anthropic) applying
[`docs/annotation_guidelines.md`](../docs/annotation_guidelines.md). Every
sample carries `provenance: corpus_sample`; 249 carry
`annotation_method: model_assisted` and the 50 checked by a person carry
`annotation_method: human_verified`.

**Model-assisted, human-validated.** The author checked 50 documents, working
from OPUS-MT English translations because the author does not read German, and
agreed 80%
of the time (κ = 0.772), and 100% on the documents the labeller flagged as
high-confidence. Those 50 answers were promoted over the machine labels, so 50
of the 299 are human-verified. `results/verification_report.json` has the
detail; a blind pilot before the guideline existed scored κ = 0.542, and both
figures belong in any write-up.

### What this replaced, and why

The previous set was 30 documents, none of which appeared in the corpus: three
were shortened edits of corpus records and the rest were written by hand. They
were measurably easier than real register text — median 116 characters against
175, 90th percentile 138 against 494, and none carrying the legal
boilerplate found in 23.2% of corpus entries (`results/retired_eval_set.json`).

The pipeline scored 80.0% on that set and 36.1% on the corpus-sampled one. Both
are real measurements; only the second is a corpus accuracy. The old documents
are kept in `retired_handwritten.json`. (`human_labels.json.bak` is something
else: the silver labels of the current set as they stood before the verified
answers were promoted.)

### Rebuilding it again

```bash
make annotate                    # build the queue and open the labelling tool
make merge ARGS=--replace        # fold the answers back in
make reproduce                   # re-measure
```

A human answer in `true_sector` always beats the model's suggestion, and the
merge records which is which, so the two never become indistinguishable.

### Remaining limitations

- **Agreement is human-vs-model, not human-vs-human.** κ = 0.772 measures the
  labeller against a person reading translations; `make second-annotator`
  prepares the blind German sample that would give the
  figure reviewers usually ask for. Cohen's κ in the *results tables* is a
  third thing again — classifier against labels.
- **No keyword ground truth.** The set carries section labels only, so
  Precision@K is unmeasurable on it.
