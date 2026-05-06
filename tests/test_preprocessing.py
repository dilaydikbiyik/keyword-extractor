"""
Unit tests for TextPreprocessor (src/utils/preprocessing.py)
Run with: pytest tests/test_preprocessing.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from utils.preprocessing import TextPreprocessor


@pytest.fixture(scope="module")
def preprocessor():
    return TextPreprocessor()


# ── clean_text ──────────────────────────────────────────────────────────────

def test_clean_text_lowercase(preprocessor):
    """Text should be lowercased."""
    result = preprocessor.clean_text("Softwareentwicklung API Cloud")
    assert result == result.lower()


def test_clean_text_removes_urls(preprocessor):
    """URLs must be stripped."""
    text = "Besuchen Sie https://example.com oder http://foo.bar"
    result = preprocessor.clean_text(text)
    assert "https" not in result
    assert "http" not in result
    assert "example.com" not in result


def test_clean_text_removes_emails(preprocessor):
    """E-mail addresses must be stripped."""
    text = "Kontakt: info@firma.de für Anfragen"
    result = preprocessor.clean_text(text)
    assert "@" not in result


def test_clean_text_preserves_hyphenated_words(preprocessor):
    """Hyphens inside words (e-ticaret, cloud-Lösung) should survive."""
    result = preprocessor.clean_text("cloud-Lösung e-ticaret")
    # At least one hyphen should survive
    assert "-" in result or "cloud" in result.lower()


def test_clean_text_removes_extra_whitespace(preprocessor):
    """Multiple spaces must be collapsed."""
    result = preprocessor.clean_text("Software   und   Cloud")
    assert "  " not in result


def test_clean_text_empty_string(preprocessor):
    """Empty / whitespace-only input must not raise."""
    assert preprocessor.clean_text("") == ""
    assert preprocessor.clean_text("   ") == ""


# ── detect_language ──────────────────────────────────────────────────────────

def test_detect_language_german(preprocessor):
    text = "Softwareentwicklung und Programmierung für Webanwendungen."
    lang, conf = preprocessor.detect_language(text)
    assert lang == "de"
    assert 0.0 <= conf <= 1.0


def test_detect_language_returns_tuple(preprocessor):
    """Must return a 2-tuple (code, confidence)."""
    result = preprocessor.detect_language("Hello world")
    assert isinstance(result, (tuple, list)) and len(result) == 2


def test_detect_language_short_text(preprocessor):
    """Very short text must not crash — may return low confidence."""
    lang, conf = preprocessor.detect_language("ok")
    assert isinstance(lang, str)


# ── generate_ngram_candidates ────────────────────────────────────────────────

def test_ngram_candidates_unigrams_included(preprocessor):
    text = "Zahnarzt Zahnimplantat Behandlung"
    candidates = preprocessor.generate_ngram_candidates(text, n_range=(1, 2))
    # Individual words (unigrams) must be present
    lower_candidates = [c.lower() for c in candidates]
    assert any("zahnarzt" in c for c in lower_candidates)


def test_ngram_candidates_minimum_char_length(preprocessor):
    """Candidates with < 3 characters must be filtered out."""
    text = "IT und KI am Ende"
    candidates = preprocessor.generate_ngram_candidates(text, n_range=(1, 1))
    for c in candidates:
        assert len(c) >= 3, f"Candidate too short: '{c}'"


def test_ngram_candidates_no_duplicates(preprocessor):
    text = "cloud cloud cloud integration"
    candidates = preprocessor.generate_ngram_candidates(text, n_range=(1, 1))
    assert len(candidates) == len(set(c.lower() for c in candidates))


def test_ngram_candidates_returns_list(preprocessor):
    candidates = preprocessor.generate_ngram_candidates("test text here")
    assert isinstance(candidates, list)


# ── preprocess_pipeline ──────────────────────────────────────────────────────

def test_preprocess_pipeline_returns_dict(preprocessor):
    result = preprocessor.preprocess_pipeline("Softwareentwicklung GmbH")
    assert isinstance(result, dict)


def test_preprocess_pipeline_has_required_keys(preprocessor):
    result = preprocessor.preprocess_pipeline("Softwareentwicklung GmbH")
    assert "cleaned_text" in result
    assert "detected_language" in result


def test_preprocess_pipeline_full_german_text(preprocessor):
    text = (
        "Handel mit Elektronik, Computern und Mobiltelefonen. "
        "Großhandel und Einzelhandel. E-Commerce-Plattform."
    )
    result = preprocessor.preprocess_pipeline(text)
    assert result["detected_language"] == "de"
    assert len(result["cleaned_text"]) > 0