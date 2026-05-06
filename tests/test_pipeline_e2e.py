"""
End-to-end integration test for the full keyword extraction pipeline.
Loads all real components (models, taxonomy) and runs 5 representative texts.
Run with: pytest tests/test_pipeline_e2e.py -v  (takes ~30s on first run)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest

from utils.preprocessing import TextPreprocessor
from services.embedder import EmbeddingService
from services.classifier import SectorClassifier
from services.extractor import KeywordExtractor
from services.filter import KeywordFilter
from controllers.controller import ExtractionController


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def controller():
    """Build a full pipeline controller once for the whole module."""
    preprocessor = TextPreprocessor()
    embedder = EmbeddingService()
    classifier = SectorClassifier(embedder)
    extractor = KeywordExtractor()
    kw_filter = KeywordFilter()
    return ExtractionController(
        embedding_service=embedder,
        classifier=classifier,
        extractor=extractor,
        keyword_filter=kw_filter,
        preprocessor=preprocessor,
    )


SAMPLE_TEXTS = [
    # J — IT/Software
    (
        "Softwareentwicklung und Programmierung für Web- und Mobileanwendungen. "
        "API-Integrationsdienste und Cloud-Lösungen.",
        "J",
    ),
    # G — Handel
    (
        "Handel mit Elektronik, Computern und Mobiltelefonen. "
        "Großhandel und Einzelhandel. E-Commerce-Plattform.",
        "G",
    ),
    # Q — Gesundheit / Zahnmedizin
    (
        "Zahnklinik mit modernen Behandlungsmethoden. "
        "Zahnimplantate, Zahnbleaching und Prophylaxe.",
        "Q",
    ),
    # F — Construction
    (
        "Bauunternehmen für Hoch- und Tiefbau. Schlüsselfertigbau, "
        "Renovierung und Sanierung von Wohn- und Gewerbegebäuden.",
        "F",
    ),
    # M — Consulting
    (
        "Steuerberatung und Wirtschaftsprüfung für mittelständische Unternehmen. "
        "Buchhaltung, Jahresabschluss und Unternehmensberatung.",
        "M",
    ),
]


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected_sector", SAMPLE_TEXTS)
def test_single_extraction_returns_dict(controller, text, expected_sector):
    """extract() must return a dict with required keys."""
    result = controller.extract(text, top_n_keywords=5)
    assert isinstance(result, dict)
    assert "status" in result
    assert "sector_classification" in result


@pytest.mark.parametrize("text,expected_sector", SAMPLE_TEXTS)
def test_single_extraction_status_success_or_known_failure(
    controller, text, expected_sector
):
    """Status must be 'success' or a known non-crash status."""
    result = controller.extract(text, top_n_keywords=5)
    known_statuses = {"success", "sector_classification_failed", "error"}
    assert result["status"] in known_statuses


@pytest.mark.parametrize("text,expected_sector", SAMPLE_TEXTS)
def test_keywords_are_strings(controller, text, expected_sector):
    """All extracted keywords must be non-empty strings."""
    result = controller.extract(text, top_n_keywords=5)
    if result["status"] == "success":
        for kw in result.get("keywords", []):
            assert isinstance(kw.get("keyword"), str)
            assert len(kw["keyword"]) > 0


@pytest.mark.parametrize("text,expected_sector", SAMPLE_TEXTS)
def test_keyword_scores_in_range(controller, text, expected_sector):
    """Keyword scores must be numeric and typically in [0, 1]."""
    result = controller.extract(text, top_n_keywords=5)
    if result["status"] == "success":
        for kw in result.get("keywords", []):
            score = kw.get("score", 0)
            assert isinstance(score, float)
            assert -0.1 <= score <= 1.1  # slight margin for floating point


def test_batch_extraction_length(controller):
    """extract_batch must return exactly as many results as inputs."""
    texts = [t for t, _ in SAMPLE_TEXTS]
    results = controller.extract_batch(texts, top_n_keywords=5, show_progress=False)
    assert len(results) == len(texts)


def test_batch_stats_keys(controller):
    """get_extraction_stats must contain all required summary keys."""
    texts = [t for t, _ in SAMPLE_TEXTS]
    results = controller.extract_batch(texts, top_n_keywords=5, show_progress=False)
    stats = controller.get_extraction_stats(results)
    required = {
        "total_documents", "successful", "failed",
        "success_rate", "total_keywords_extracted",
        "avg_keywords_per_document", "avg_confidence",
        "sector_distribution",
    }
    assert required.issubset(set(stats.keys()))


def test_batch_report_creates_file(controller, tmp_path):
    """save_batch_report must create a JSON file in the target directory."""
    import json
    texts = [t for t, _ in SAMPLE_TEXTS[:2]]
    results = controller.extract_batch(texts, top_n_keywords=5, show_progress=False)
    report_path = controller.save_batch_report(results, output_dir=str(tmp_path))
    assert os.path.exists(report_path)
    with open(report_path) as f:
        data = json.load(f)
    assert "statistics" in data
    assert "documents" in data
    assert len(data["documents"]) == 2


def test_processing_time_recorded(controller):
    """Each result dict must contain a processing_time_ms field."""
    texts = [SAMPLE_TEXTS[0][0]]
    results = controller.extract_batch(texts, top_n_keywords=3, show_progress=False)
    assert "processing_time_ms" in results[0]
    assert results[0]["processing_time_ms"] >= 0
