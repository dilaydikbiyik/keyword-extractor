"""Shared experiment configuration and determinism helpers."""

from __future__ import annotations

import os
import random
import warnings
from pathlib import Path

import numpy as np

# Every experiment in this package is run through this seed.
SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"

CORPUS_CSV = DATA_DIR / "raw" / "handelsregister_sample_10k.csv"
# Corpus-derived statistics, committed so a clone without the raw records can
# still reproduce the tables. See data/README.md.
DERIVED_DIR = DATA_DIR / "derived"
TFIDF_STATS = DERIVED_DIR / "tfidf_corpus_stats.json"
LABELS_JSON = DATA_DIR / "evaluation" / "human_labels.json"
SECTORS_JSON = DATA_DIR / "taxonomy" / "sectors.json"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Number of bootstrap resamples used for the confidence intervals reported
# next to every accuracy figure.
N_BOOTSTRAP = 10_000


def silence_stale_fp_warnings() -> None:
    """Drop the spurious matmul warnings numpy raises after a model forward pass.

    The transformer leaves a floating-point flag set inside BLAS; numpy then
    attributes it to the next matmul it evaluates, including ones inside
    KeyBERT.  The similarity values are checked for finiteness where we
    compute them, so the warning carries no information here.
    """
    warnings.filterwarnings(
        "ignore", message=".*encountered in matmul", category=RuntimeWarning
    )


def set_seed(seed: int = SEED) -> None:
    """Pin every source of randomness we can reach.

    Sector classification is deterministic, but KeyBERT's MMR and the
    bootstrap resampling are not, so this is what makes ``make reproduce``
    byte-identical across runs.
    """
    silence_stale_fp_warnings()
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # torch is optional for the metric-only paths
        pass


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
