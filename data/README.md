# Data

Four things live under `data/`. Three are in version control; the difference
matters for anyone trying to reproduce the reported numbers.

| Path | In git | What it is |
| --- | --- | --- |
| `data/taxonomy/sectors.json` | yes | 21 NACE Rev. 2 sections with hand-written seed keywords and descriptions. Authored for this project. |
| `data/evaluation/human_labels.json` | yes | The evaluation set: gold NACE section plus reference keywords per document. Authored for this project. |
| `data/derived/tfidf_corpus_stats.json` | yes | Vocabulary and IDF weights derived from the corpus below. |
| `data/raw/handelsregister_sample_10k.csv` | **no** | 9,993 German trade register business purposes. |

## The corpus is not redistributed — and reproduction does not need it

The sample was collected for this project from the German trade register. The
underlying facts are public record, but a bulk redistribution licence for the
collected file has not been established, so the file is not published here.

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

**What a clone cannot do:** `python main.py`, which batch-processes the raw CSV,
and regenerating the statistics themselves
(`python -m experiments.corpus_stats`). Both need the corpus.

If the licence question is later settled in favour of publication, add the CSV,
drop the `data/raw/` line from `.gitignore`, and this section becomes a
footnote.

## Evaluation set

`human_labels.json` holds **299 documents sampled from the corpus**, labelled
by a language model applying
[`docs/annotation_guidelines.md`](../docs/annotation_guidelines.md). Every
sample carries `provenance: corpus_sample` and
`annotation_method: model_assisted`.

**These are silver labels.** A human validation pass on 50 of them
(`results/verification_sample.csv`, scored with `make verify`) is what turns
them into something a paper can cite. Until then, quote no number from them.

### What this replaced, and why

The previous set was 30 documents, none of which appeared in the corpus: three
were shortened edits of corpus records and the rest were written by hand. They
were measurably easier than real register text — median 116 characters against
175, 90th percentile 138 against 494, and 3.3% carrying legal boilerplate
against 24.5%.

The pipeline scored 80.0% on that set and 36.1% on the corpus-sampled one. Both
are real measurements; only the second is a corpus accuracy. The old file is
kept as `human_labels.json.bak`.

### Rebuilding it again

```bash
make annotate                    # build the queue and open the labelling tool
make merge ARGS=--replace        # fold the answers back in
make reproduce                   # re-measure
```

A human answer in `true_sector` always beats the model's suggestion, and the
merge records which is which, so the two never become indistinguishable.

### Remaining limitations

- **No inter-annotator agreement.** Cohen's κ in the results tables measures
  *classifier vs. labels*, not annotator vs. annotator. A second annotator on
  an overlapping 100 documents gives the figure reviewers ask for.
- **No keyword ground truth.** The set carries section labels only, so
  Precision@K is unmeasurable on it.
