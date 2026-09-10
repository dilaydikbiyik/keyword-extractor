# Paper

A complete draft of the workshop submission, wired so that **no number is ever
typed into the prose**. `tests/test_reported_numbers.py` fails if a literal
figure appears in the text instead of a macro.

## Build

```bash
brew install tectonic   # once; any TeX distribution also works
make paper              # regenerate tables, fetch the ACL style, write paper/main.pdf
```

`make paper` pulls `acl.sty` and `acl_natbib.bst` from
[acl-org/acl-style-files](https://github.com/acl-org/acl-style-files) at a pinned
commit instead of vendoring them, because the upstream repository carries no
licence file. `\usepackage[review]{acl}` gives the anonymous review layout; drop
the option for the camera-ready.

CI builds the same PDF on every push and attaches it to the run. The build fails
if the committed tables differ from what `results/` produces, or if the log
reports an undefined reference, an undefined citation, or a character the font
cannot draw.

## Tables and numbers come from `results/`

```bash
make paper-tables
```

Regenerates `tables/*.tex` from `results/baselines.json`,
`results/ablation.json` and `results/error_analysis.json`:

| File | Contents |
| --- | --- |
| `tables/baselines.tex` | Main comparison, with intervals and McNemar p-values |
| `tables/ablation.tex` | Component ablation with deltas against the full system |
| `tables/errors.tex` | Automatically determined properties of the errors |
| `tables/macros.tex` | Every figure the prose cites, as a LaTeX macro |

Write `\OursTopOne`, not `80.0\%`. When the evaluation set grows and
`make reproduce` runs again, the paper follows on its own; a hardcoded number
would silently become a lie.

## What is left in it

Nothing in the prose. The author block, affiliation and email are filled in; the
acknowledgements are commented out and must stay that way until the paper is
accepted, because a named acknowledgement breaks review anonymity.

Before submitting, read [`../docs/paper_readiness.md`](../docs/paper_readiness.md)
§11 — the open items are an LLM baseline, a fourth corpus, and the
inter-annotator figure that needs a second person.

Every entry in `references.bib` has been checked against its source: the ACL
Anthology's own BibTeX where the paper is in the Anthology, Crossref or DataCite
for DOIs, and the publisher's record otherwise. The header of the file says
which source each entry came from. None was written from memory.

## Venues

Tracked in [`venues.md`](venues.md), with the deadlines to verify and the
reminder schedule to set.
