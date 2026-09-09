"""Loading and access for ``config/config.yaml``.

Every tunable the pipeline actually honours lives in that file and is read
through here.  Keys the code does not act on do not belong in the file: a
configuration option that changes nothing is worse than no option at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULT_CONFIG_PATH = "config/config.yaml"

# Mirrors config/config.yaml.  Used when a key is missing so a partial file
# still produces a working pipeline.
DEFAULTS: Dict[str, Any] = {
    "embedding": {
        "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
        "chunk_size": 256,
        "overlap": 32,
        "max_length": 512,
        "device": "cpu",
    },
    "preprocessing": {
        "remove_urls": True,
        "remove_emails": True,
        "lowercase": True,
        "remove_punctuation": True,
        "preserve_hyphens": True,
        "min_word_length": 3,
        "max_ngram_length": 3,
    },
    "classification": {
        "top_k_sectors": 3,
        "confidence_threshold": 0.5,
        "classify_preprocessed_text": False,
    },
    "extraction": {
        "guided_mode": True,
        "top_n_final": 10,
        "diversity": 0.7,
    },
    "filtering": {
        "min_score": 0.1,
    },
    "iteration": {
        "max_iterations": 5,
        "quality_threshold": 0.55,
        "max_seed_size": 80,
    },
    "logging": {
        "level": "WARNING",
    },
    "paths": {
        "taxonomy": "data/taxonomy/",
        "raw_data": "data/raw/",
        "results": "output/",
    },
    "llm": {
        "enabled": False,
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Read the YAML config, merged over :data:`DEFAULTS`.

    A missing file is not an error — the defaults are the shipped
    configuration — but a malformed one is.
    """
    config_path = Path(path or DEFAULT_CONFIG_PATH)
    if not config_path.exists():
        return dict(DEFAULTS)
    with open(config_path, encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    return _deep_merge(DEFAULTS, loaded)


def get(config: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Look up ``"section.key"`` in a loaded config."""
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def sectors_file(config: Dict[str, Any]) -> str:
    """Path to the taxonomy file the configured taxonomy directory holds."""
    return str(Path(get(config, "paths.taxonomy", "data/taxonomy/")) / "sectors.json")
