# Submitted to ARR October 2026

- Recorded 2026-10-12 19:32 UTC
- Commit `f179d0d3892711af10e2cf53c9f5480812b142f9`
- Branch `main`
- Working tree clean
- Tag `v1-arr-october-2026`

## Digests

Verify with `shasum -a 256 <file>`.

| File | SHA-256 | Bytes |
| --- | --- | --- |
| `dist/review.pdf` | `a581c01d3bb61ffea61a7d1e0785a71c22d382eb496c79a17a8f5085a108958f` | 163,706 |
| `dist/anonymous_code.zip` | `229978804cc1e329a6383ab9629edb17bf19bd31b6e53ff8d5ecc3527e8fbe4d` | 2,197,477 |
| `paper/main.tex` | `eb3f2278c849523a8f6811647c3aadea4d98ec32dac09415ecf7c4704a1244b4` | 57,322 |

## What the code archive contains

- Built from `git archive HEAD`: the tracked tree at this commit, every text file passed through the anonymiser, every file it could not read as text dropped.
- 213 files, against 214 tracked at this commit.
- Tracked here but absent from the archive, either dropped as unreadable or because the archive predates this commit: `docs/assets/demo.gif`
  Rebuild with `make submission` and record again if the second.

## What this does and does not establish

It fixes two separate things, and they are worth keeping apart. The digest
identifies the file: a reader can check that the PDF they hold is the one
that was uploaded. The commit identifies the source it was built from.

It does not show that rebuilding from that commit reproduces the file byte
for byte, and it will not: the build embeds a timestamp, so a later build
of the same source has a different digest. What a rebuild establishes is
that the same source produces the same paper, not the same bytes.

It does not establish that the work is correct, and reproducing the tables
from this commit shows only that the computation repeats -- not that the
design is sound. The preregistration claim rests on the content of the
files in `results/*_preregistration.json`, which state each prediction and
the fingerprint of the method it was made under, and on their position in
the history relative to the results they predict; the ordering is
checkable with `git log --diff-filter=A`.

Work done after this record is post-submission. It belongs in its own
preregistrations and must not be written back into the ones above.
