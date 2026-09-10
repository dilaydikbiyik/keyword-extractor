# Paper

A complete draft of the workshop submission, wired so that **no number is ever
typed into the prose**. `tests/test_reported_numbers.py` fails if a literal
figure appears in the text instead of a macro.

## Build

The skeleton targets the ACL style files — the same ones behind the Overleaf
"ACL Proceedings" template.

1. Start from the ACL template on Overleaf, or clone
   <https://github.com/acl-org/acl-style-files> next to `main.tex`.
2. Copy `main.tex`, `references.bib` and `tables/` into it.
3. Compile. `\usepackage[review]{acl}` gives the anonymous review layout;
   drop the option for the camera-ready.

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

Every entry in `references.bib` carries a `CHECK` note where its venue or page
range has not been confirmed against the ACL Anthology or the publisher. Verify
them before submission; bibliographies are where reviewers look first for
carelessness.

## Venues

Tracked in [`venues.md`](venues.md), with the deadlines to verify and the
reminder schedule to set.
