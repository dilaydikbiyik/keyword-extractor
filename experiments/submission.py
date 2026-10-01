#!/usr/bin/env python3
"""Build what a double-blind submission uploads, and check it is anonymous.

    python -m experiments.submission      # or: make submission

Writes ``dist/review.pdf`` (the paper in review mode, which prints
"Anonymous ACL submission" instead of the author block) and
``dist/anonymous_code.zip`` (the committed tree, with every identifying string
replaced). Then it scans both and refuses to finish if anything identifying is
left: a submission that names its author is rejected without review.

The archive is built from ``git archive``, so history, commit authors and
untracked files never enter it -- including ``private/identity.txt``, which is
where the strings to remove are kept, precisely so that they are not in the
tree being cleaned. Images and other binary media are left out,
because a scan cannot read text inside them.
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import sys
import zipfile
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from experiments.config import ROOT

DIST = ROOT / "dist"
PLACEHOLDER = "ANONYMOUS"
BINARY_SUFFIXES = {".gif", ".png", ".jpg", ".jpeg", ".pdf", ".ico", ".zip"}
# What counts as identifying is the author's own name, address and university, so
# the patterns live outside version control: written into a tracked file they
# would travel inside the very archive this script cleans. See the file itself,
# or data/README.md, for its two sections.
IDENTITY_FILE = ROOT / "private" / "identity.txt"


def read_patterns(text: str) -> Tuple[List[re.Pattern], re.Pattern]:
    """The patterns of an identity file: ones to replace, and one to detect with.

    Under ``[remove]``, one regex per line, most specific first: replacing a
    surname before a username would leave the first name beside the placeholder
    in every URL. Under ``[detect]``, regexes that must not match anywhere in the
    result; detection is deliberately looser than replacement, so that any
    fragment surviving inside a word still fails the check.
    """
    sections: Dict[str, List[str]] = {"remove": [], "detect": []}
    current = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            if current not in sections:
                raise SystemExit(f"{IDENTITY_FILE.name}: unknown section [{current}]")
            continue
        if current is None:
            raise SystemExit(f"{IDENTITY_FILE.name}: a pattern before any [section]")
        sections[current].append(line)
    for name, patterns in sections.items():
        if not patterns:
            raise SystemExit(f"{IDENTITY_FILE.name}: section [{name}] is empty")
    return ([re.compile(p, re.I) for p in sections["remove"]],
            re.compile("|".join(f"(?:{p})" for p in sections["detect"]), re.I))


@lru_cache(maxsize=1)
def load_identity() -> Tuple[List[re.Pattern], re.Pattern]:
    if not IDENTITY_FILE.exists():
        raise SystemExit(f"Write the patterns to remove in {IDENTITY_FILE.relative_to(ROOT)} "
                         "(sections [remove] and [detect], one regex per line). "
                         "Keep it out of version control.")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(IDENTITY_FILE.relative_to(ROOT))],
                             cwd=ROOT, capture_output=True, text=True)
    if tracked.returncode == 0:
        raise SystemExit(f"{IDENTITY_FILE.relative_to(ROOT)} is tracked by git, so the archive "
                         "would carry the strings it removes. Untrack it first.")
    return read_patterns(IDENTITY_FILE.read_text(encoding="utf-8"))


def anonymize_text(text: str, patterns: Sequence[re.Pattern] = None) -> str:
    """Replace every identifying string with a placeholder."""
    for pattern in load_identity()[0] if patterns is None else patterns:
        text = pattern.sub(PLACEHOLDER, text)
    return text


def find_identity(text: str, detector: re.Pattern = None) -> List[str]:
    """Identifying fragments still present, with context, for the refusal message."""
    detector = load_identity()[1] if detector is None else detector
    return sorted({text[max(0, m.start() - 20):m.end() + 20].replace("\n", " ")
                   for m in detector.finditer(text)})


def pdf_text(data: bytes) -> str:
    """The literal strings inside a PDF's content streams, joined."""
    chunks = []
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
        try:
            chunks.append(zlib.decompress(m.group(1)))
        except zlib.error:
            continue
    literals = re.findall(rb"\(((?:[^()\\]|\\.)*)\)", b"".join(chunks))
    return b"".join(literals).decode("latin-1") + data.decode("latin-1")


def build_archive() -> Path:
    raw = subprocess.check_output(["git", "archive", "--format=zip", "HEAD"], cwd=ROOT)
    out = DIST / "anonymous_code.zip"
    skipped = []
    with zipfile.ZipFile(io.BytesIO(raw)) as src, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            if info.is_dir():
                continue
            if Path(info.filename).suffix.lower() in BINARY_SUFFIXES:
                skipped.append(info.filename)
                continue
            data = src.read(info)
            try:
                dst.writestr(info.filename, anonymize_text(data.decode("utf-8")))
            except UnicodeDecodeError:
                skipped.append(info.filename)
    if skipped:
        print(f"Left out {len(skipped)} binary files: {', '.join(skipped)}")
    return out


def check(archive: Path, pdf: Path) -> List[str]:
    problems = []
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            for hit in find_identity(name + "\n" + z.read(name).decode("utf-8", errors="ignore")):
                problems.append(f"{archive.name}:{name}: {hit}")
    for hit in find_identity(pdf_text(pdf.read_bytes())):
        problems.append(f"{pdf.name}: {hit}")
    return problems


def main() -> int:
    pdf = ROOT / "paper" / "main.pdf"
    if not pdf.exists():
        print("Build the paper first: make paper", file=sys.stderr)
        return 1
    if r"\usepackage[review]{acl}" not in (ROOT / "paper" / "main.tex").read_text(encoding="utf-8"):
        print("paper/main.tex is not in review mode; the PDF would show the author.", file=sys.stderr)
        return 1
    DIST.mkdir(exist_ok=True)
    shutil.copyfile(pdf, DIST / "review.pdf")
    archive = build_archive()
    problems = check(archive, DIST / "review.pdf")
    if problems:
        print("NOT ANONYMOUS — do not upload:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print(f"Anonymous: {DIST / 'review.pdf'} and {archive} contain no identifying string.")
    print("Upload the PDF to OpenReview. For the code, use an anonymising mirror of the")
    print("repository, or attach the zip if the venue accepts supplementary material.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
