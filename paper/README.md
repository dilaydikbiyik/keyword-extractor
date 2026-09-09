# Paper

A skeleton for the workshop submission, wired so that **no number is ever typed
into the prose**.

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

## Before writing a word

The skeleton's `FIXME` markers are ordered the way the sections should be
written, and each carries its target length. But the paper is not ready to
write yet — see [`../docs/paper_readiness.md`](../docs/paper_readiness.md).
The evaluation set is 30 documents, no comparison in it is significant, and
the error analysis needs roughly 50 errors against the current 6.

Writing before that is writing a paper whose central table a reviewer will
reject in the first pass.

## Venues

Tracked in [`venues.md`](venues.md), with the deadlines to verify and the
reminder schedule to set.
