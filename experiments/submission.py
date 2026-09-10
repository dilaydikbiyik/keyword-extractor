#!/usr/bin/env python3
"""Build what a double-blind submission uploads, and check it is anonymous.

    python -m experiments.submission      # or: make submission

Writes ``dist/review.pdf`` (the paper in review mode, which prints
"Anonymous ACL submission" instead of the author block) and
``dist/anonymous_code.zip`` (the committed tree, with every identifying string
replaced). Then it scans both and refuses to finish if anything identifying is
left: a submission that names its author is rejected without review.

The archive is built from ``git archive``, so history, commit authors and
untracked files never enter it. Images and other binary media are left out,
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
from pathlib import Path
from typing import List

from experiments.config import ROOT

DIST = ROOT / "dist"
# Replaced in this order, most specific first: replacing the surname before the
# username would leave "dilay" + placeholder inside every URL and address.
IDENTITY = [
    re.compile(r"dilaydikbiyik(@gmail\.com)?", re.I),
    re.compile(r"Dilay\s+Dikb[ıi]y[ıi]k", re.I),
    re.compile(r"Dikb\\i\s*y\\i\s*k", re.I),
    re.compile(r"Dikb[ıi]y[ıi]k", re.I),
    re.compile(r"Dilay", re.I),
    re.compile(r"Kocaeli(\s+(University|Üniversitesi))?", re.I),
]
# Detection is deliberately looser than replacement: any fragment of the name
# that survives, in any case and inside any word, fails the check.
LEAK = re.compile(r"dilay|dikb[ıi]y|dikb\\i|kocaeli", re.I)
PLACEHOLDER = "ANONYMOUS"
BINARY_SUFFIXES = {".gif", ".png", ".jpg", ".jpeg", ".pdf", ".ico", ".zip"}


def anonymize_text(text: str) -> str:
    """Replace every identifying string with a placeholder."""
    for pattern in IDENTITY:
        text = pattern.sub(PLACEHOLDER, text)
    return text


def find_identity(text: str) -> List[str]:
    """Identifying fragments still present, with context, for the refusal message."""
    return sorted({text[max(0, m.start() - 20):m.end() + 20].replace("\n", " ")
                   for m in LEAK.finditer(text)})


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
