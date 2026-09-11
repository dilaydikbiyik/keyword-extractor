"""One place that turns ``config/config.yaml`` into a working pipeline.

``ExtractionController`` and the services keep their own defaults so they can
still be constructed by hand, but every entry point in this repository builds
them through :func:`build_controller`, which is what makes the configuration
file real rather than decorative.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from controllers.controller import ExtractionController
from services.classifier import SectorClassifier
from services.embedder import EmbeddingService
from services.extractor import KeywordExtractor
from services.filter import KeywordFilter
from utils.config import get, load_config, sectors_file
from utils.preprocessing import TextPreprocessor


def build_controller(
    config_path: Optional[str] = None,
) -> Tuple[ExtractionController, Dict[str, Any]]:
    """Build the full pipeline from a config file.

    Args:
        config_path: Path to the YAML config; defaults to ``config/config.yaml``.

    Returns:
        The controller and the resolved configuration dictionary.
    """
    config = load_config(config_path)
    taxonomy = sectors_file(config)

    embedder = EmbeddingService(
        model_name=get(config, "embedding.model_name"),
        device=get(config, "embedding.device", "cpu"),
    )
    classifier = SectorClassifier(
        embedder,
        sectors_file=taxonomy,
        confidence_threshold=get(config, "classification.confidence_threshold"),
    )
    extractor = KeywordExtractor(
        model_name=get(config, "embedding.model_name"),
        sectors_file=taxonomy,
        device=get(config, "embedding.device", "cpu"),
    )
    keyword_filter = KeywordFilter(sectors_file=taxonomy)
    preprocessor = TextPreprocessor(config=config.get("preprocessing"))

    controller = ExtractionController(
        embedding_service=embedder,
        classifier=classifier,
        extractor=extractor,
        keyword_filter=keyword_filter,
        preprocessor=preprocessor,
        config={
            "top_k_sectors": get(config, "classification.top_k_sectors", 3),
            "classify_preprocessed_text": get(
                config, "classification.classify_preprocessed_text", False
            ),
            "top_n_keywords": get(config, "extraction.top_n_final", 10),
            "guided_mode": get(config, "extraction.guided_mode", True),
            "diversity": get(config, "extraction.diversity", 0.7),
            "min_score": get(config, "filtering.min_score", 0.1),
            "log_level": get(config, "logging.level", "WARNING"),
        },
    )
    return controller, config
