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

`human_labels.json` holds 30 documents labelled by a single annotator. Every
sample carries a `provenance` field, and right now all 30 read `authored`.
That is the most important thing on this page.

### The evaluation set was not sampled from the corpus

No document in it appears in `handelsregister_sample_10k.csv`. Three are
shortened, edited versions of corpus records; the rest were written by hand.
The two populations differ measurably:

| | Evaluation set | Corpus |
| --- | --- | --- |
| Median length | 116 characters | 175 characters |
| 90th percentile length | 138 characters | 494 characters |
| Carrying legal boilerplate or `§` | 3.3% | 24.5% |

So accuracy measured on this set is a real measurement of a real system, but on
text that is systematically shorter and cleaner than the corpus the system
exists to handle. **Do not read it as corpus accuracy.**

### Rebuilding it

`results/annotation_queue.csv` is drawn from the corpus itself, so labelling it
fixes representativeness and sample size together:

```bash
make annotate                    # build the queue (296 documents)
open tools/annotate.html         # label them — keyboard-driven, offline
python -m experiments.merge_annotations --replace
```

`--replace` is deliberate: appending a hand-authored set to a corpus-sampled one
reintroduces exactly the bias above. The merge backs up the old file first.

Merged rows are recorded as `provenance: corpus_sample`, so the two populations
stay distinguishable and can be reported separately.

### Two further limitations

- **Thirty documents is too few** for the comparisons the set is used for: the
  95% interval is roughly ±14 points.
- **One annotator, no agreement measured.** Cohen's κ in the results tables
  measures *classifier vs. gold*, not annotator vs. annotator. A second
  annotator on an overlapping 100 documents gives the inter-annotator figure
  reviewers ask for; re-labelling a blind subset after a week gives an
  intra-annotator one, which is weaker but available alone.

See [`docs/paper_readiness.md`](../docs/paper_readiness.md) for what each of
these does to the reported numbers.
