"""
Unit tests for the experiment harness (experiments/)
Run with: pytest tests/test_experiments.py -v
"""

import pytest

from experiments import error_analysis, metrics, report
from experiments.data import load_labeled_samples, load_taxonomy, sector_names


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestBootstrapCI:
    def test_bounds_contain_point_estimate(self):
        correct = [True] * 24 + [False] * 6  # 80%
        lo, hi = metrics.bootstrap_ci(correct, n_resamples=2000)
        assert lo <= 0.8 <= hi

    def test_interval_narrows_with_more_data(self):
        small = metrics.bootstrap_ci([True] * 24 + [False] * 6, n_resamples=2000)
        large = metrics.bootstrap_ci([True] * 240 + [False] * 60, n_resamples=2000)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_deterministic_for_a_fixed_seed(self):
        args = ([True, False, True, True], 2000)
        assert metrics.bootstrap_ci(*args) == metrics.bootstrap_ci(*args)

    def test_empty_input(self):
        assert metrics.bootstrap_ci([]) == (0.0, 0.0)


class TestMcNemar:
    def test_identical_systems_are_not_significant(self):
        a = [True, False, True, True]
        result = metrics.mcnemar_exact(a, a)
        assert result["discordant"] == 0
        assert result["p_value"] == 1.0

    def test_counts_discordant_pairs(self):
        a = [True, True, False, False]
        b = [True, False, True, False]
        result = metrics.mcnemar_exact(a, b)
        assert result["a_only_correct"] == 1
        assert result["b_only_correct"] == 1

    def test_clear_win_is_significant(self):
        a = [True] * 12 + [False] * 12
        b = [False] * 12 + [False] * 12
        assert metrics.mcnemar_exact(a, b)["p_value"] < 0.01


class TestSectorEvaluation:
    def test_perfect_predictions(self):
        result = metrics.evaluate_sector_predictions(["J", "C"], [["J", "M"], ["C", "G"]])
        assert result["top1_accuracy"] == 1.0
        assert result["f1_macro"] == 1.0
        assert result["correct_top1"] == [True, True]

    def test_top3_credits_a_recoverable_miss(self):
        result = metrics.evaluate_sector_predictions(["J"], [["M", "K", "J"]])
        assert result["top1_accuracy"] == 0.0
        assert result["top3_accuracy"] == 1.0

    def test_reports_a_confidence_interval(self):
        result = metrics.evaluate_sector_predictions(["J", "C", "G"], [["J"], ["C"], ["M"]])
        lo, hi = result["top1_ci95"]
        assert 0.0 <= lo <= result["top1_accuracy"] <= hi <= 1.0


class TestKeywordEvaluation:
    def test_precision_at_k(self):
        result = metrics.evaluate_keywords([["a", "b", "c"]], [["a", "b"]], k_values=(3,))
        assert result["precision_at_3"] == pytest.approx(2 / 3)

    def test_documents_without_labels_are_skipped(self):
        result = metrics.evaluate_keywords([["a"], ["b"]], [["a"], []], k_values=(1,))
        assert result["n_with_keyword_labels"] == 1
        assert result["precision_at_1"] == 1.0

    def test_no_labels_at_all(self):
        result = metrics.evaluate_keywords([["a"]], [[]], k_values=(5,))
        assert result["precision_at_5"] is None


class TestConfusionPairs:
    def test_only_errors_are_counted(self):
        pairs = metrics.confusion_pairs(["J", "J", "C"], ["J", "M", "M"])
        assert ("J", "M", 1) in pairs
        assert ("C", "M", 1) in pairs
        assert all(t != p for t, p, _ in pairs)


# ─────────────────────────────────────────────────────────────────────────────
# Error analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoFlags:
    def test_short_text(self):
        flags = error_analysis.auto_flags("Handel.", [0.6, 0.2, 0.1], "G", ["J", "M", "K"])
        assert "short_text" in flags

    def test_low_margin(self):
        flags = error_analysis.auto_flags("x" * 200, [0.51, 0.50, 0.30], "G", ["J", "M", "K"])
        assert "low_margin" in flags
        assert "crowded_top3" not in flags

    def test_crowded_top3(self):
        flags = error_analysis.auto_flags("x" * 200, [0.52, 0.51, 0.50], "G", ["J", "M", "K"])
        assert "crowded_top3" in flags

    def test_boilerplate_shape(self):
        text = "Erwerb und Verwaltung von Beteiligungen an anderen Unternehmen." * 2
        flags = error_analysis.auto_flags(text, [0.6, 0.2, 0.1], "K", ["L", "M", "K"])
        assert "boilerplate_shape" in flags

    def test_recoverable_versus_outside_top3(self):
        recoverable = error_analysis.auto_flags("x" * 200, [0.6, 0.2, 0.1], "K", ["L", "M", "K"])
        missed = error_analysis.auto_flags("x" * 200, [0.6, 0.2, 0.1], "Q", ["L", "M", "K"])
        assert "recoverable_in_top3" in recoverable
        assert "outside_top3" in missed

    def test_codebook_categories_are_documented(self):
        assert len(error_analysis.CODEBOOK) >= 5
        assert all(isinstance(v, str) and v for v in error_analysis.CODEBOOK.values())


class TestErrorRecords:
    def test_only_misclassified_documents_are_kept(self):
        samples = {s.id: s for s in load_labeled_samples()[:2]}
        ids = list(samples)
        predictions = [
            {"id": ids[0], "true": samples[ids[0]].true_sector,
             "predicted": samples[ids[0]].true_sector, "top3": ["J"], "scores": [0.6, 0.2, 0.1]},
            {"id": ids[1], "true": samples[ids[1]].true_sector,
             "predicted": "ZZZ", "top3": ["ZZZ", "M", "K"], "scores": [0.6, 0.2, 0.1]},
        ]
        records = error_analysis.build_error_records(predictions, samples)
        assert len(records) == 1
        assert records[0]["id"] == ids[1]
        assert records[0]["manual_category"] == ""  # left for the manual pass


# ─────────────────────────────────────────────────────────────────────────────
# Data and reporting
# ─────────────────────────────────────────────────────────────────────────────

class TestData:
    def test_labelled_samples_have_valid_sectors(self):
        codes = set(load_taxonomy())
        samples = load_labeled_samples()
        assert samples
        assert all(s.true_sector in codes for s in samples)

    def test_sector_names_cover_the_taxonomy(self):
        assert set(sector_names()) == set(load_taxonomy())


class TestReport:
    def test_comparison_table_has_a_row_per_system(self):
        results = [
            {
                "key": "full", "label": "Ours", "is_oracle": False,
                "sector": {"top1_accuracy": 0.8, "top1_ci95": [0.63, 0.93],
                           "top3_accuracy": 0.97, "f1_macro": 0.83, "cohen_kappa": 0.79},
                "keywords": {"precision_at_5": 0.33}, "p_value_vs_full": None,
            }
        ]
        table = report.comparison_table(results)
        assert "| Ours |" in table
        assert "80.0%" in table

    def test_oracle_rows_are_marked(self):
        results = [
            {
                "key": "majority", "label": "Majority", "is_oracle": True,
                "sector": {"top1_accuracy": 0.13, "top1_ci95": [0.03, 0.27],
                           "top3_accuracy": 0.33, "f1_macro": 0.01, "cohen_kappa": 0.0},
                "keywords": {}, "p_value_vs_full": 0.001,
            }
        ]
        assert "*Majority" in report.comparison_table(results)

    def test_ablation_table_reports_deltas_against_full(self):
        results = [
            {"key": "full", "label": "Full", "sector": {"top1_accuracy": 0.8, "f1_macro": 0.83},
             "keywords": {"precision_at_5": 0.33}, "p_value_vs_full": None},
            {"key": "no-seeds", "label": "− seeds", "sector": {"top1_accuracy": 0.733, "f1_macro": 0.744},
             "keywords": {"precision_at_5": 0.34}, "p_value_vs_full": 0.5},
        ]
        table = report.ablation_table(results)
        assert "+0.0 pp" in table
        assert "-6.7 pp" in table


# ─────────────────────────────────────────────────────────────────────────────
# The harness must measure the shipped pipeline, not a lookalike
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestHarnessMatchesShippedPipeline:
    def test_embedding_ranker_reproduces_sector_classifier(self):
        st = pytest.importorskip("sentence_transformers")  # noqa: F841
        from services.classifier import SectorClassifier
        from experiments.systems import EmbeddingRanker, get_embedder

        embedder = get_embedder()
        shipped = SectorClassifier(embedder)
        harness = EmbeddingRanker(use_description=True, use_seeds=True)

        texts = [
            "Entwicklung und Vertrieb von Software und IT-Dienstleistungen.",
            "Handel mit Elektronik und Betrieb einer E-Commerce-Plattform.",
            "Zahnklinik mit Implantaten und Prophylaxe.",
        ]
        for text in texts:
            assert harness.rank(text)[0][0] == shipped.classify_with_details(text)["top_sector"]


# ─────────────────────────────────────────────────────────────────────────────
# Annotation loop
# ─────────────────────────────────────────────────────────────────────────────

class TestSampleSizing:
    def test_ci_narrows_with_n(self):
        from experiments.build_annotation_queue import ci_halfwidth

        assert ci_halfwidth(0.8, 300) < ci_halfwidth(0.8, 30)

    def test_thirty_documents_give_a_wide_interval(self):
        from experiments.build_annotation_queue import ci_halfwidth

        assert ci_halfwidth(0.8, 30) == pytest.approx(0.143, abs=0.005)

    def test_sizing_table_is_monotonic(self):
        from experiments.build_annotation_queue import sizing_table

        widths = [row["ci_halfwidth_pp"] for row in sizing_table()]
        assert widths == sorted(widths, reverse=True)


class TestMergeAnnotations:
    def _labels_file(self, tmp_path):
        import json as _json

        path = tmp_path / "human_labels.json"
        path.write_text(
            _json.dumps(
                {
                    "metadata": {"total": 1},
                    "samples": [
                        {
                            "id": 0,
                            "legal_name": "Existing GmbH",
                            "purpose": "Bestehender Eintrag.",
                            "true_sector": "J",
                            "keywords_ground_truth": ["software"],
                            "annotation_method": "semi_manual",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    def _queue_file(self, tmp_path, rows):
        import csv as _csv

        path = tmp_path / "queue.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(
                fh,
                fieldnames=[
                    "queue_id", "legal_name", "purpose",
                    "true_sector", "keywords_ground_truth",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _run(self, monkeypatch, labels, queue, extra_args=()):
        import sys as _sys

        from experiments import merge_annotations

        monkeypatch.setattr(merge_annotations, "LABELS_JSON", labels)
        monkeypatch.setattr(
            _sys, "argv", ["merge", "--queue", str(queue), *extra_args]
        )
        return merge_annotations.main()

    def test_merges_annotated_rows_only(self, tmp_path, monkeypatch):
        import json as _json

        labels = self._labels_file(tmp_path)
        queue = self._queue_file(
            tmp_path,
            [
                {"queue_id": 0, "legal_name": "Neu GmbH", "purpose": "Bau von Häusern.",
                 "true_sector": "F", "keywords_ground_truth": "bau|hochbau"},
                {"queue_id": 1, "legal_name": "Offen GmbH", "purpose": "Noch nicht codiert.",
                 "true_sector": "", "keywords_ground_truth": ""},
            ],
        )
        assert self._run(monkeypatch, labels, queue) == 0

        payload = _json.loads(labels.read_text(encoding="utf-8"))
        assert payload["metadata"]["total"] == 2
        new = payload["samples"][1]
        assert new["true_sector"] == "F"
        assert new["keywords_ground_truth"] == ["bau", "hochbau"]
        assert new["id"] == 1

    def test_rejects_an_invalid_sector_code(self, tmp_path, monkeypatch):
        labels = self._labels_file(tmp_path)
        queue = self._queue_file(
            tmp_path,
            [{"queue_id": 0, "legal_name": "X", "purpose": "Text.",
              "true_sector": "ZZ", "keywords_ground_truth": ""}],
        )
        assert self._run(monkeypatch, labels, queue) == 1

    def test_is_idempotent(self, tmp_path, monkeypatch):
        import json as _json

        labels = self._labels_file(tmp_path)
        queue = self._queue_file(
            tmp_path,
            [{"queue_id": 0, "legal_name": "Neu GmbH", "purpose": "Bau von Häusern.",
              "true_sector": "F", "keywords_ground_truth": ""}],
        )
        self._run(monkeypatch, labels, queue)
        self._run(monkeypatch, labels, queue)
        payload = _json.loads(labels.read_text(encoding="utf-8"))
        assert payload["metadata"]["total"] == 2

    def test_dry_run_writes_nothing(self, tmp_path, monkeypatch):
        labels = self._labels_file(tmp_path)
        before = labels.read_text(encoding="utf-8")
        queue = self._queue_file(
            tmp_path,
            [{"queue_id": 0, "legal_name": "Neu GmbH", "purpose": "Bau von Häusern.",
              "true_sector": "F", "keywords_ground_truth": ""}],
        )
        assert self._run(monkeypatch, labels, queue, ["--dry-run"]) == 0
        assert labels.read_text(encoding="utf-8") == before


# ─────────────────────────────────────────────────────────────────────────────
# Reproducing without the raw corpus
# ─────────────────────────────────────────────────────────────────────────────

class TestCorpusStatistics:
    """The committed statistics must stand in for the corpus exactly."""

    DOCS = [
        "Softwareentwicklung und Vertrieb von IT-Dienstleistungen für Unternehmen.",
        "Handel mit Elektronik und Betrieb einer E-Commerce-Plattform.",
        "Zahnklinik mit Implantaten, Prophylaxe und Zahnbehandlung.",
        "Errichtung und Betrieb von Photovoltaikanlagen und Windkraftanlagen.",
        "Unternehmensberatung, Managementberatung und Wirtschaftsprüfung.",
        "Vermietung und Verwaltung von eigenen Grundstücken und Wohnungen.",
    ]

    def _fitted(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True
        )
        vec.fit(self.DOCS)
        return vec

    def test_transform_matches_scikit_learn(self):
        import numpy as np

        from experiments.corpus_stats import CorpusStatistics

        vec = self._fitted()
        stats = CorpusStatistics.from_vectorizer(vec, n_documents=len(self.DOCS))
        for doc in self.DOCS:
            expected = vec.transform([doc]).toarray()[0]
            assert np.allclose(stats.transform(doc), expected, atol=1e-9)

    def test_unknown_terms_are_ignored(self):
        from experiments.corpus_stats import CorpusStatistics

        stats = CorpusStatistics.from_vectorizer(self._fitted(), n_documents=6)
        vector = stats.transform("völlig unbekanntes vokabular xyzzy")
        assert vector.shape == (len(stats.vocabulary),)
        assert float(vector.sum()) == 0.0

    def test_round_trip_through_json(self, tmp_path):
        import numpy as np

        from experiments.corpus_stats import CorpusStatistics

        stats = CorpusStatistics.from_vectorizer(self._fitted(), n_documents=6)
        path = tmp_path / "stats.json"
        stats.save(path)
        restored = CorpusStatistics.load(path)
        assert restored.vocabulary == stats.vocabulary
        assert restored.n_documents == 6
        assert np.allclose(restored.transform(self.DOCS[0]), stats.transform(self.DOCS[0]), atol=1e-6)

    def test_missing_file_explains_the_fix(self, tmp_path):
        from experiments.corpus_stats import CorpusStatistics

        with pytest.raises(FileNotFoundError, match="data/README.md"):
            CorpusStatistics.load(tmp_path / "absent.json")

    def test_feature_names_line_up_with_indices(self):
        from experiments.corpus_stats import CorpusStatistics

        stats = CorpusStatistics.from_vectorizer(self._fitted(), n_documents=6)
        names = stats.feature_names
        for term, index in stats.vocabulary.items():
            assert names[index] == term


@pytest.mark.slow
class TestCommittedStatisticsReproduceTheCorpus:
    def test_stored_statistics_rank_like_the_fitted_vectorizer(self):
        from experiments.data import load_corpus, load_labeled_samples
        from experiments.systems import TfidfRanker

        corpus = load_corpus()
        if not corpus:
            pytest.skip("raw corpus not present in this checkout")

        fitted = TfidfRanker()
        fitted.fit(corpus)
        stored = TfidfRanker()
        stored.fit([])  # forces the committed statistics

        for sample in load_labeled_samples():
            assert [c for c, _ in fitted.rank(sample.purpose)] == [
                c for c, _ in stored.rank(sample.purpose)
            ]
