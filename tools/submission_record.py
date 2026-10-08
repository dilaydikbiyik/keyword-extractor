#!/usr/bin/env python3
"""Pin exactly what was submitted, so the claim survives the repository moving on.

    python tools/submission_record.py --venue "ARR October 2026" > docs/submission_v1.md
    python tools/submission_record.py --verify docs/submission_v1.md

This project's argument rests on an order of events: predictions committed
before outcomes existed. That order is checkable in the git history only while
the history means what it meant on the day. A tag marks the commit; this
records the rest -- the digest of the file actually uploaded, the digest of the
code archive beside it, and the state of the working tree when they were built
-- so that a reader with `shasum` can confirm the PDF they are holding is the
one the record describes.

It asserts nothing about whether the work is correct. It fixes what "the
submitted version" refers to.

``--verify`` reads a record back and checks it still describes the files on
disk, because a PDF rebuilt after the record was written is a different file
with the same name, and the record would go quietly wrong. It also checks that
the tag, if one is named, dereferences to the commit the record claims. A git
tag is movable after it is pushed, so the record and the tag confirm each
other rather than either being the authority; an external timestamp -- the
commit as GitHub holds it, or an arXiv posting -- is what neither can forge.

On the order of operations. The record names the commit the submitted files
were built from, and the tag must point at that same commit -- not at the later
commit that stores the record, which necessarily comes after it. So: build the
artefacts, write the record, tag the commit the record names, commit the record
on top, and verify. Committing the record does not invalidate it; the record
says which commit it describes, and ``--verify`` compares the tag against that
rather than against HEAD.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTEFACTS = ("dist/review.pdf", "dist/anonymous_code.zip", "paper/main.tex")
RECORD_PATH = "docs/submission_v1.md"
TAG_LINE = "- Tag `"
DIGEST_ROW = re.compile(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \| ([\d,]+) \|$", re.M)
COMMIT_LINE = re.compile(r"^- Commit `([0-9a-f]{40})`$", re.M)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def archive_scope(path: Path) -> list:
    """What the archive holds, since a digest of it describes only the bytes.

    The archive is `git archive HEAD` with every binary file dropped and every
    text file passed through the anonymiser, so its scope is "the tracked tree
    at this commit, minus what could not be read as text".
    """
    if not path.exists():
        return ["- *not built*"]
    with zipfile.ZipFile(path) as z:
        names = [i.filename for i in z.infolist() if not i.is_dir()]
    tracked = git("ls-files").splitlines()
    missing = sorted(set(tracked) - set(names))
    out = [
        "- Built from `git archive HEAD`: the tracked tree at this commit, "
        "every text file passed through the anonymiser, every file it could not "
        "read as text dropped.",
        f"- {len(names)} files, against {len(tracked)} tracked at this commit.",
    ]
    if missing:
        # Why a file is absent is not knowable from here: the anonymiser drops
        # what it cannot decode, and an archive built before the current commit
        # is simply older than the tree. Name them and say so.
        out.append("- Tracked here but absent from the archive, either dropped as "
                   "unreadable or because the archive predates this commit: "
                   + ", ".join(f"`{m}`" for m in missing))
        out.append("  Rebuild with `make submission` and record again if the second.")
    else:
        out.append("- Every tracked file at this commit is in the archive.")
    return out


def verify(record: Path) -> int:
    text = record.read_text(encoding="utf-8")
    commit = COMMIT_LINE.search(text)
    problems = []
    if not commit:
        return fail(["no commit line in the record"])
    for name, want, _ in DIGEST_ROW.findall(text):
        path = ROOT / name
        if not path.exists():
            problems.append(f"{name}: recorded but missing on disk")
        elif digest(path) != want:
            problems.append(f"{name}: on disk now differs from the record "
                            f"-- rebuilt after it was written?")
    for line in text.splitlines():
        if line.startswith(TAG_LINE):
            tag = line.split("`")[1]
            try:
                at = git("rev-parse", f"{tag}^{{}}")
            except subprocess.CalledProcessError:
                problems.append(f"tag {tag} does not exist")
            else:
                if at != commit.group(1):
                    problems.append(f"tag {tag} points at {at[:8]}, "
                                    f"record says {commit.group(1)[:8]}")
    if problems:
        return fail(problems)
    print(f"{record}: digests match and the tag agrees with the commit.")
    return 0


def fail(problems: list) -> int:
    print("VERIFICATION FAILED", file=sys.stderr)
    for p in problems:
        print(f"  {p}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--venue", help='e.g. "ARR October 2026"')
    parser.add_argument("--tag", help="the annotated tag this record is pinned by")
    parser.add_argument("--verify", type=Path, help="check a record against the files on disk")
    args = parser.parse_args()
    if args.verify:
        return verify(args.verify)
    if not args.venue:
        parser.error("--venue is required unless --verify is given")

    # The record is written by redirect, so by the time this runs its own file
    # already exists and would report the tree as dirty because of itself.
    dirty = "\n".join(
        line for line in git("status", "--porcelain").splitlines()
        if RECORD_PATH not in line)
    lines = [
        f"# Submitted to {args.venue}",
        "",
        f"- Recorded {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Commit `{git('rev-parse', 'HEAD')}`",
        f"- Branch `{git('rev-parse', '--abbrev-ref', 'HEAD')}`",
        f"- Working tree {'DIRTY -- see below' if dirty else 'clean'}",
    ]
    if args.tag:
        lines.append(f"- Tag `{args.tag}`")
    lines += [
        "",
        "## Digests",
        "",
        "Verify with `shasum -a 256 <file>`.",
        "",
        "| File | SHA-256 | Bytes |",
        "| --- | --- | --- |",
    ]
    for name in ARTEFACTS:
        path = ROOT / name
        if path.exists():
            lines.append(f"| `{name}` | `{digest(path)}` | {path.stat().st_size:,} |")
        else:
            lines.append(f"| `{name}` | *not built* | — |")

    lines += ["", "## What the code archive contains", ""]
    lines += archive_scope(ROOT / "dist" / "anonymous_code.zip")
    lines += [
        "",
        "## What this does and does not establish",
        "",
        "It fixes what \"the submitted version\" refers to: a reader can check that",
        "the PDF they hold is the one this record describes, and that it was built",
        "from the commit named above.",
        "",
        "It does not establish that the work is correct, and reproducing the tables",
        "from this commit shows only that the computation repeats -- not that the",
        "design is sound. The preregistration claim rests on the content of the",
        "files in `results/*_preregistration.json`, which state each prediction and",
        "the fingerprint of the method it was made under, and on their position in",
        "the history relative to the results they predict; the ordering is",
        "checkable with `git log --diff-filter=A`.",
        "",
        "Work done after this record is post-submission. It belongs in its own",
        "preregistrations and must not be written back into the ones above.",
    ]
    if dirty:
        lines += ["", "## Uncommitted at the time of recording", "", "```", dirty, "```"]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
