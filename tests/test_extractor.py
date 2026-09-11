"""
Unit tests for KeywordExtractor (src/services/extractor.py)
Uses a lightweight mock to avoid loading the full SentenceTransformer model.
Run with: pytest tests/test_extractor.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import MagicMock, patch
from services.extractor import KeywordExtractor


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_extractor():
    """
    Return a KeywordExtractor with KeyBERT and sector data mocked out
    so no GPU / model download is required during unit tests.
    """
    with patch("services.extractor.KeyBERT") as MockKeyBERT:
        mock_kw_model = MagicMock()
        # Default: return 3 plausible (keyword, score) tuples
        mock_kw_model.extract_keywords.return_value = [
            ("software", 0.85),
            ("cloud", 0.72),
            ("api", 0.65),
        ]
        MockKeyBERT.return_value = mock_kw_model

        # Patch sector loading so no file I/O is needed
        with patch.object(
            KeywordExtractor,
            "_load_sector_keywords",
            return_value={
                "J": ["Software", "Cloud", "API", "Datenbank"],
                "G": ["Handel", "Einzelhandel", "E-Commerce"],
                "Q": ["Zahnarzt", "Zahnklinik", "Therapie"],
            },
        ):
            extractor = KeywordExtractor(model_name="dummy-model")

    return extractor, mock_kw_model


# ── extract_keywords ──────────────────────────────────────────────────────────

def test_extract_keywords_returns_list(mock_extractor):
    extractor, _ = mock_extractor
    result = extractor.extract_keywords("Some business text", top_n=5)
    assert isinstance(result, list)


def test_extract_keywords_tuple_structure(mock_extractor):
    """Each element must be a (str, float) tuple."""
    extractor, _ = mock_extractor
    result = extractor.extract_keywords("Some business text", top_n=3)
    for item in result:
        assert len(item) == 2, f"Expected 2-tuple, got: {item}"
        keyword, score = item
        assert isinstance(keyword, str)
        assert isinstance(score, float)


def test_extract_keywords_empty_text_no_crash(mock_extractor):
    extractor, mock_kw = mock_extractor
    mock_kw.extract_keywords.return_value = []
    result = extractor.extract_keywords("", top_n=5)
    assert isinstance(result, list)


def test_extract_keywords_with_seed_keywords(mock_extractor):
    """Seed keywords must be passed through to KeyBERT."""
    extractor, mock_kw = mock_extractor
    seeds = ["software", "cloud"]
    mock_kw.extract_keywords.return_value = [("software", 0.9)]
    result = extractor.extract_keywords("text", seed_keywords=seeds, top_n=3)

    _, kwargs = mock_kw.extract_keywords.call_args
    assert kwargs["seed_keywords"] == seeds
    assert kwargs["top_n"] == 3
    assert kwargs["use_mmr"] is True
    assert result == [("software", 0.9)]


# ── extract_keywords_guided_by_sector ─────────────────────────────────────────

def test_guided_extraction_known_sector(mock_extractor):
    extractor, mock_kw = mock_extractor
    mock_kw.extract_keywords.return_value = [("softwareentwicklung", 0.88)]
    result = extractor.extract_keywords_guided_by_sector(
        "Softwareentwicklung für Web", sector_code="J", top_n=5
    )
    assert isinstance(result, list)


def test_guided_extraction_accepts_language_kwarg(mock_extractor):
    """Passing language= must not raise TypeError."""
    extractor, mock_kw = mock_extractor
    mock_kw.extract_keywords.return_value = [("cloud", 0.8)]
    result = extractor.extract_keywords_guided_by_sector(
        "Cloud-Dienste", sector_code="J", top_n=3, language="de"
    )
    assert isinstance(result, list)


def test_guided_extraction_unknown_sector_falls_back(mock_extractor):
    """Unknown sector code → unguided extraction (no crash)."""
    extractor, mock_kw = mock_extractor
    mock_kw.extract_keywords.return_value = [("generic", 0.5)]
    result = extractor.extract_keywords_guided_by_sector(
        "Some text", sector_code="UNKNOWN", top_n=3
    )
    assert isinstance(result, list)
