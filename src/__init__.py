"""NACE section assignment for German trade register texts.

The library layer: a domain model (the taxonomy), services that embed,
classify, extract and filter, and a controller that runs them per document.
Research code lives in ``experiments/`` and imports this package, never the
other way round; ``tests/test_architecture.py`` enforces that.
"""

from .controllers.controller import ExtractionController
from .models.taxonomy import TaxonomyManager
from .services.classifier import SectorClassifier
from .services.embedder import EmbeddingService
from .services.extractor import KeywordExtractor
from .services.filter import KeywordFilter
from .utils.preprocessing import TextPreprocessor

__version__ = "1.0.0"
__all__ = [
    "ExtractionController",
    "TaxonomyManager",
    "EmbeddingService",
    "KeywordExtractor",
    "SectorClassifier",
    "KeywordFilter",
    "TextPreprocessor",
]
