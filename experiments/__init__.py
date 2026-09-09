"""Experiment harness: baselines, ablations, and error analysis.

Kept separate from ``src/`` so the shipped pipeline stays free of paper-only
code, while every number in the paper is produced by a single command.
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import utils.quiet  # noqa: F401,E402  (registers warning filters on import)
