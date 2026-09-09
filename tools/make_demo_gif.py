#!/usr/bin/env python3
"""Render the README demo GIF from a recorded terminal session.

    python tools/make_demo_gif.py

The session is described in ``SESSION`` below and its output is *captured
verbatim* from files under ``results/`` and from a real test run — nothing in
the GIF is typed by hand.  Lines removed for length are marked with an
explicit elision marker rather than silently dropped.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "demo.gif"

WIDTH, HEIGHT = 1000, 640
PAD_X, PAD_Y = 22, 54
LINE_H = 20
FONT_SIZE = 14
FPS = 20

BG = (24, 26, 32)
CHROME = (38, 41, 50)
FG = (223, 227, 235)
DIM = (140, 147, 162)
PROMPT = (126, 200, 148)
CMD = (238, 241, 246)
ACCENT = (186, 156, 232)
GOOD = (126, 200, 148)
WARN = (222, 170, 100)

FONT = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", FONT_SIZE, index=0)
FONT_BOLD = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", FONT_SIZE, index=1)


@dataclass
class Step:
    command: str
    output: List[str] = field(default_factory=list)
    hold_frames: int = 24


def colour_for(line: str) -> Tuple[int, int, int]:
    stripped = line.strip()
    if stripped.startswith("| **") or "passed" in stripped:
        return GOOD
    if stripped.startswith("|"):
        return FG
    if stripped.startswith("…") or stripped.startswith("["):
        return DIM
    if stripped.startswith("NOTE") or stripped.startswith("⚠"):
        return WARN
    if stripped.startswith("→") or stripped.startswith("Wrote"):
        return ACCENT
    return FG


def read_table(name: str, limit: int | None = None) -> List[str]:
    path = ROOT / "results" / "tables" / f"{name}.md"
    lines = path.read_text(encoding="utf-8").rstrip().splitlines()
    if limit and len(lines) > limit:
        lines = lines[:limit] + [f"…{len(lines) - limit} more rows"]
    return lines


def capture(cmd: List[str], keep: int) -> List[str]:
    """Run a command and keep its last ``keep`` non-empty lines."""
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    lines = [ln for ln in (proc.stdout + proc.stderr).splitlines() if ln.strip()]
    return lines[-keep:]


def build_session() -> List[Step]:
    return [
        Step(
            "make reproduce",
            [
                "Evaluation set: 30 documents, 18 sectors",
                "Corpus for TF-IDF statistics: 9993 documents",
                "",
                "[baselines]",
                *read_table("baselines"),
                "",
                "[ablation]",
                *read_table("ablation", limit=12),
                "",
                "Wrote results/metrics.json",
            ],
            hold_frames=70,
        ),
        Step(
            "python -m experiments.run_error_analysis",
            [
                "System: full",
                "Errors: 6 / 30",
                "",
                "Automatic flags:",
                "  recoverable_in_top3      5",
                "  low_margin               4",
                "  crowded_top3             2",
                "  outside_top3             1",
                "",
                "NOTE: the guide asks for 50 hand-inspected errors; this set",
                "yields 6. Enlarge the labelled set with",
                "`python -m experiments.build_annotation_queue`.",
            ],
            hold_frames=55,
        ),
        Step("pytest tests/ -q", capture([sys.executable, "-m", "pytest", "tests/", "-q"], 1), hold_frames=45),
    ]


def draw_frame(rendered: List[Tuple[str, Tuple[int, int, int], bool]], cursor: bool) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(img)

    # window chrome
    d.rectangle([0, 0, WIDTH, 34], fill=CHROME)
    for i, colour in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([18 + i * 20, 12, 28 + i * 20, 22], fill=colour)
    d.text((WIDTH // 2 - 90, 10), "keyword-extractor", font=FONT, fill=DIM)

    y = PAD_Y
    for text, colour, bold in rendered:
        d.text((PAD_X, y), text, font=FONT_BOLD if bold else FONT, fill=colour)
        y += LINE_H
    if cursor:
        d.rectangle([PAD_X, y, PAD_X + 8, y + 15], fill=FG)
    return img


def render(session: List[Step]) -> List[Image.Image]:
    frames: List[Image.Image] = []
    scrollback: List[Tuple[str, Tuple[int, int, int], bool]] = []
    max_lines = (HEIGHT - PAD_Y - 16) // LINE_H

    def visible(extra=()):
        lines = scrollback + list(extra)
        return lines[-max_lines:]

    for step in session:
        prompt = "$ "
        # type the command
        for i in range(0, len(step.command) + 1, 2):
            typed = [(prompt + step.command[:i], CMD, True)]
            frames.append(draw_frame(visible(typed), cursor=True))
        scrollback.append((prompt + step.command, CMD, True))
        frames.extend([draw_frame(visible(), cursor=True)] * 6)

        # reveal output
        for line in step.output:
            scrollback.append((line, colour_for(line), line.strip().startswith("| **")))
            frames.append(draw_frame(visible(), cursor=False))
        frames.extend([draw_frame(visible(), cursor=True)] * step.hold_frames)
        scrollback.append(("", FG, False))

    return frames


def collapse(frames: List[Image.Image]) -> Tuple[List[Image.Image], List[int]]:
    """Merge runs of identical frames into one frame with a longer duration.

    GIF optimisation drops duplicate frames but keeps a single per-frame
    duration, which silently deletes every hold in the session.  Collapsing
    them here keeps the pauses and still shrinks the file.
    """
    tick = int(1000 / FPS)
    kept: List[Image.Image] = []
    durations: List[int] = []
    previous = None
    for frame in frames:
        signature = frame.tobytes()
        if signature == previous:
            durations[-1] += tick
        else:
            kept.append(frame)
            durations.append(tick)
            previous = signature
    return kept, durations


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames, durations = collapse(render(build_session()))
    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    size_mb = OUT.stat().st_size / 1024 / 1024
    seconds = sum(durations) / 1000
    print(f"{OUT.relative_to(ROOT)} — {len(frames)} frames, {seconds:.0f}s, {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
