"""
Unit tests for the sector classifier (src/services/classifier.py)
Run with: pytest tests/test_classifier.py -v
"""

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from services.classifier import SectorClassifier


class TestThresholdFiltering:
    """The threshold cut is the trap that once inflated reported Top-3 accuracy."""

    class _StubEmbedder:
        """Scores sectors by a fixed table so the threshold is the only variable."""

        def __init__(self, scores):
            self.scores = scores

        def embed_text(self, text):
            return np.array([1.0, 0.0], dtype=np.float32)

        def similarity(self, a, b, metric="cosine"):
            return self.scores[int(b[0])]

    def _classifier(self, scores, threshold, tmp_path):
        sectors = {"sectors": {
            str(i): {"name": f"S{i}", "description": "d", "seed_keywords": []}
            for i in range(len(scores))
        }}
        path = tmp_path / "sectors.json"
        path.write_text(json.dumps(sectors), encoding="utf-8")

        embedder = self._StubEmbedder(scores)
        classifier = SectorClassifier.__new__(SectorClassifier)
        classifier.embedding_service = embedder
        classifier.sectors_file = str(path)
        classifier.confidence_threshold = threshold
        classifier.sectors_info = json.loads(path.read_text())["sectors"]
        classifier.sector_embeddings = {
            code: np.array([float(code)], dtype=np.float32)
            for code in classifier.sectors_info
        }
        return classifier

    def test_details_always_reports_a_best_match(self, tmp_path):
        """The method the pipeline actually uses must never come back empty."""
        classifier = self._classifier([0.4, 0.3, 0.2], threshold=0.5, tmp_path=tmp_path)
        details = classifier.classify_with_details("text", top_k=3)

        assert details["top_sector"] == "0"
        assert len(details["classifications"]) == 1
        assert details["classifications"][0]["confidence"] == pytest.approx(0.4)


class TestSectorEmbeddings:
    """The sector vector is a documented average; pin it so it stays one."""

    class _WordEmbedder:
        """Maps a text to a one-hot count vector so averaging is checkable by hand."""

        VOCAB = ["alpha", "beta", "gamma"]

        def embed_text(self, text):
            lowered = text.lower()
            return np.array([float(lowered.count(w)) for w in self.VOCAB])

    def _write(self, tmp_path, sectors):
        path = tmp_path / "sectors.json"
        path.write_text(json.dumps({"sectors": sectors}), encoding="utf-8")
        return str(path)

    def test_description_and_seeds_are_averaged(self, tmp_path):
        path = self._write(tmp_path, {
            "A": {"name": "alpha", "description": "alpha", "seed_keywords": ["beta", "beta"]},
        })
        classifier = SectorClassifier(self._WordEmbedder(), sectors_file=path)

        # description "alpha. alpha" -> [2,0,0]; seeds "beta, beta" -> [0,2,0].
        assert classifier.sector_embeddings["A"].tolist() == [1.0, 1.0, 0.0]

    def test_seedless_sector_uses_the_description_alone(self, tmp_path):
        path = self._write(tmp_path, {
            "B": {"name": "gamma", "description": "gamma", "seed_keywords": []},
        })
        classifier = SectorClassifier(self._WordEmbedder(), sectors_file=path)

        assert classifier.sector_embeddings["B"].tolist() == [0.0, 0.0, 2.0]

    def test_a_missing_taxonomy_is_logged_not_swallowed(self, tmp_path, caplog):
        with caplog.at_level("ERROR"):
            classifier = SectorClassifier(
                self._WordEmbedder(), sectors_file=str(tmp_path / "absent.json")
            )

        assert classifier.sectors_info == {}
        assert "not found" in caplog.text
