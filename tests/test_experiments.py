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


class TestManualCoding:
    """The one part of the error analysis no code can recompute must survive a rerun."""

    def _record(self, i, true="F", predicted="B"):
        return {"id": i, "true_sector": true, "predicted_sector": predicted,
                "manual_category": "", "manual_note": ""}

    def test_carries_coding_for_the_same_error(self):
        records = [self._record(1), self._record(2)]
        previous = [{"id": "1", "true_sector": "F", "predicted_sector": "B",
                     "manual_category": "seed_leakage", "manual_note": "Boden"}]
        kept = error_analysis.carry_manual_coding(records, previous)
        assert kept == {"carried": 1, "dropped": 0}
        assert records[0]["manual_category"] == "seed_leakage"
        assert records[0]["manual_note"] == "Boden"
        assert records[1]["manual_category"] == ""

    def test_drops_coding_when_the_error_changed(self):
        records = [self._record(1, predicted="C")]
        previous = [{"id": "1", "true_sector": "F", "predicted_sector": "B",
                     "manual_category": "seed_leakage", "manual_note": ""},
                    {"id": "9", "true_sector": "J", "predicted_sector": "M",
                     "manual_category": "multi_sector_company", "manual_note": ""}]
        kept = error_analysis.carry_manual_coding(records, previous)
        # One error now fails differently; the other is no longer an error at all.
        assert kept == {"carried": 0, "dropped": 2}
        assert records[0]["manual_category"] == ""

    def test_summary_counts_categories_and_names_the_half(self):
        records = [dict(self._record(i), manual_category=c)
                   for i, c in enumerate(["seed_leakage", "seed_leakage", "boilerplate_only"])]
        records.append(self._record(7))
        summary = error_analysis.summarise_manual_coding(records, dev_ids={0, 1, 2})
        assert summary["n_coded"] == 3
        assert summary["half"] == "dev"
        assert summary["distribution"] == {"seed_leakage": 2, "boilerplate_only": 1}
        assert error_analysis.summarise_manual_coding([self._record(1)], dev_ids=set()) is None


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

    def test_replace_writes_only_the_new_documents(self, tmp_path, monkeypatch):
        """--replace must swap the sample list, not just the metadata."""
        import json as _json

        labels = self._labels_file(tmp_path)
        queue = self._queue_file(
            tmp_path,
            [{"queue_id": 0, "legal_name": "Neu GmbH", "purpose": "Bau von Häusern.",
              "true_sector": "F", "keywords_ground_truth": ""}],
        )
        assert self._run(monkeypatch, labels, queue, ["--replace"]) == 0

        payload = _json.loads(labels.read_text(encoding="utf-8"))
        assert len(payload["samples"]) == 1
        assert payload["metadata"]["total"] == len(payload["samples"])
        assert payload["samples"][0]["true_sector"] == "F"
        assert payload["samples"][0]["id"] == 0

    def test_replace_does_not_skip_documents_already_in_the_set(self, tmp_path, monkeypatch):
        """--replace rebuilds the set, so a corrected label must not dedup away."""
        import json as _json

        labels = self._labels_file(tmp_path)
        # The queue repeats the document already present, with a different label.
        queue = self._queue_file(
            tmp_path,
            [{"queue_id": 0, "legal_name": "Existing GmbH", "purpose": "Bestehender Eintrag.",
              "true_sector": "C", "keywords_ground_truth": ""}],
        )
        assert self._run(monkeypatch, labels, queue, ["--replace"]) == 0

        payload = _json.loads(labels.read_text(encoding="utf-8"))
        assert len(payload["samples"]) == 1
        # The corrected label wins; the stale J is gone.
        assert payload["samples"][0]["true_sector"] == "C"

    def test_silver_labels_are_recorded_as_such(self, tmp_path, monkeypatch):
        import csv as _csv
        import json as _json

        labels = self._labels_file(tmp_path)
        path = tmp_path / "silver.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(
                fh,
                fieldnames=["queue_id", "legal_name", "purpose", "true_sector",
                            "model_assisted_sector", "keywords_ground_truth"],
            )
            writer.writeheader()
            writer.writerow({"queue_id": 0, "legal_name": "A", "purpose": "Softwareentwicklung.",
                             "true_sector": "", "model_assisted_sector": "J",
                             "keywords_ground_truth": ""})
            writer.writerow({"queue_id": 1, "legal_name": "B", "purpose": "Bau von Häusern.",
                             "true_sector": "F", "model_assisted_sector": "C",
                             "keywords_ground_truth": ""})
        assert self._run(monkeypatch, labels, path, ["--replace"]) == 0

        payload = _json.loads(labels.read_text(encoding="utf-8"))
        by_purpose = {s["purpose"]: s for s in payload["samples"]}
        silver = by_purpose["Softwareentwicklung."]
        human = by_purpose["Bau von Häusern."]
        assert silver["true_sector"] == "J"
        assert silver["annotation_method"] == "model_assisted"
        # A human answer wins over the silver one.
        assert human["true_sector"] == "F"
        assert human["annotation_method"] == "manual"

    def test_verified_rows_stay_marked_as_human(self, tmp_path, monkeypatch):
        """apply_verified() promotes into the silver column; merge must keep saying so."""
        import csv as _csv
        import json as _json

        labels = self._labels_file(tmp_path)
        path = tmp_path / "queue.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(
                fh,
                fieldnames=["queue_id", "legal_name", "purpose", "true_sector",
                            "model_assisted_sector", "human_verified", "keywords_ground_truth"],
            )
            writer.writeheader()
            writer.writerow({"queue_id": 0, "legal_name": "A", "purpose": "Softwareentwicklung.",
                             "true_sector": "", "model_assisted_sector": "J",
                             "human_verified": "", "keywords_ground_truth": ""})
            writer.writerow({"queue_id": 1, "legal_name": "B", "purpose": "Verwaltung eigenen Vermögens.",
                             "true_sector": "", "model_assisted_sector": "M",
                             "human_verified": "yes", "keywords_ground_truth": ""})
        assert self._run(monkeypatch, labels, path, ["--replace"]) == 0

        payload = _json.loads(labels.read_text(encoding="utf-8"))
        methods = {s["purpose"]: s["annotation_method"] for s in payload["samples"]}
        assert methods["Softwareentwicklung."] == "model_assisted"
        assert methods["Verwaltung eigenen Vermögens."] == "human_verified"


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


class TestSecondAnnotator:
    """The blind sample must carry nothing that hints at anyone else's answer."""

    def _sample(self, tmp_path):
        import csv as _csv

        path = tmp_path / "verification_sample.csv"
        rows = [
            {"queue_id": str(i), "legal_name": f"Firma {i}", "purpose_en": "English text",
             "purpose": f"Gegenstand {i}.", "silver_sector": silver, "labeller_confidence": "high",
             "top3": "J|M|K", "true_sector": human, "annotator_note": ""}
            for i, (silver, human) in enumerate([("J", "J"), ("M", "K"), ("F", "F"), ("G", "G")])
        ]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _patch(self, monkeypatch, tmp_path):
        from experiments import verify_labels

        monkeypatch.setattr(verify_labels, "SAMPLE_CSV", self._sample(tmp_path))
        monkeypatch.setattr(verify_labels, "SECOND_CSV", tmp_path / "second.csv")
        monkeypatch.setattr(verify_labels, "SECOND_REPORT_JSON", tmp_path / "second.json")
        return verify_labels

    def test_blind_sample_is_german_only(self, tmp_path, monkeypatch):
        import csv as _csv

        vl = self._patch(monkeypatch, tmp_path)
        assert vl.build_second() == 0
        with open(tmp_path / "second.csv", newline="", encoding="utf-8") as fh:
            rows = list(_csv.DictReader(fh))
        assert len(rows) == 4
        # No translation, no suggestion, no earlier answer.
        assert set(rows[0]) == {"queue_id", "legal_name", "purpose", "true_sector", "annotator_note"}
        assert all(r["true_sector"] == "" for r in rows)

    def test_refuses_to_overwrite_answers(self, tmp_path, monkeypatch):
        vl = self._patch(monkeypatch, tmp_path)
        assert vl.build_second() == 0
        path = tmp_path / "second.csv"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("Gegenstand 0.,,", "Gegenstand 0.,J,"), encoding="utf-8")
        assert vl.build_second() == 1

    def test_scores_second_against_first_and_silver(self, tmp_path, monkeypatch):
        import csv as _csv
        import json as _json

        vl = self._patch(monkeypatch, tmp_path)
        assert vl.build_second() == 0
        path = tmp_path / "second.csv"
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(_csv.DictReader(fh))
        answers = {"0": "J", "1": "K", "2": "F", "3": "C"}
        for r in rows:
            r["true_sector"] = answers[r["queue_id"]]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        assert vl.score_second() == 0
        report = _json.loads((tmp_path / "second.json").read_text(encoding="utf-8"))
        # The first human answered J, K, F, G: the second agrees on three of four.
        assert report["second_vs_first_human"]["raw_agreement"] == 0.75
        # The silver labels were J, M, F, G: agreement on two of four.
        assert report["second_vs_silver"]["raw_agreement"] == 0.5


class TestRobustness:
    """The checks that answer "your labels are silver" and "one class carries rho"."""

    def test_perfectly_monotone_classes_give_rho_one(self, monkeypatch):
        from experiments import robustness

        monkeypatch.setattr(robustness, "N_PERMUTATIONS", 300)
        rows = [{"class": str(k), "recall_terse": 0.5, "recall_rich": 0.5 + 0.04 * k,
                 "alignment_gain": 0.01 * k} for k in range(10)]
        out = robustness.mechanism_check({"classes": rows})
        assert out["rho"] == pytest.approx(1.0)
        assert out["rho_ci95"] == pytest.approx([1.0, 1.0])
        assert out["leave_one_out_min"] == pytest.approx(1.0)
        assert out["permutation_p"] < 0.01

    def test_refuses_predictions_made_against_other_labels(self):
        from experiments import robustness

        preds = {"full": [{"id": 0, "true": "A", "predicted": "A", "top3": ["A"]}]}
        samples = [{"id": 0, "true_sector": "B", "annotation_method": "human_verified"}]
        with pytest.raises(SystemExit):
            robustness.label_check(preds, samples)

    def test_splits_verified_from_the_rest(self):
        from experiments import robustness

        preds = {
            "full": [{"id": i, "true": "A", "predicted": "A" if i < 3 else "B", "top3": ["A", "B"]}
                     for i in range(4)],
            "other": [{"id": i, "true": "A", "predicted": "B", "top3": ["B", "A"]} for i in range(4)],
        }
        samples = [{"id": i, "true_sector": "A",
                    "annotation_method": "human_verified" if i < 2 else "model_assisted"}
                   for i in range(4)]
        out = robustness.label_check(preds, samples)
        assert out["n_verified"] == 2 and out["n_rest"] == 2
        assert out["systems"]["full"]["top1_verified"] == 1.0
        assert out["systems"]["full"]["top1_rest"] == 0.5
        assert out["systems"]["other"]["top3_verified"] == 1.0


class TestLLMReplyParsing:
    """The parser decides what an LLM baseline scores; it must not guess silently."""

    def _taxonomy(self):
        return {c: {} for c in "ABCDEFGHIJKLMNOPQRSTU"}

    def test_reads_a_reply_in_the_requested_format(self):
        from experiments.systems import parsed_letters

        assert parsed_letters("M, J, N", self._taxonomy()) == (["M", "J", "N"], True)

    def test_recovers_but_flags_a_reply_out_of_format(self):
        from experiments.systems import parsed_letters

        assert parsed_letters("Section M, then J.", self._taxonomy()) == (["M", "J"], False)

    def test_repeated_letters_count_once(self):
        from experiments.systems import parsed_letters

        assert parsed_letters("M, M, J", self._taxonomy()) == (["M", "J"], True)

    def test_ranking_puts_the_answer_first_and_keeps_every_section(self):
        from experiments.systems import ranking_from_reply

        taxonomy = self._taxonomy()
        ranking = ranking_from_reply("Q", taxonomy, sorted(taxonomy))
        assert ranking[0][0] == "Q"
        assert len(ranking) == len(taxonomy)
        assert [score for _, score in ranking] == sorted((score for _, score in ranking), reverse=True)


class TestSubmissionAnonymity:
    """A submission that names its author is rejected without review."""

    def test_every_form_of_the_name_is_replaced(self):
        from experiments.submission import anonymize_text, find_identity

        text = ("Dilay Dikbıyık, Dikb\\i y\\i k, dilaydikbiyik@gmail.com, "
                "github.com/dilaydikbiyik/keyword-extractor, Kocaeli University")
        cleaned = anonymize_text(text)
        assert find_identity(cleaned) == []
        # No fragment may survive inside a URL or an address.
        assert "dilay" not in cleaned.lower() and "kocaeli" not in cleaned.lower()
        assert "github.com/ANONYMOUS/keyword-extractor" in cleaned

    def test_a_surviving_fragment_is_detected(self):
        from experiments.submission import find_identity

        assert find_identity("https://github.com/dilayANONYMOUS/x")

    def test_ordinary_text_is_left_alone(self):
        from experiments.submission import anonymize_text

        text = "Zero-shot NACE classification of German trade register texts."
        assert anonymize_text(text) == text


class TestRocchio:
    """The update the causal test rests on, and the order it has to happen in."""

    def test_moves_each_vector_toward_its_nearest_documents(self):
        import numpy as np

        from experiments.run_rocchio import rocchio, unit_rows

        vectors = unit_rows(np.array([[1.0, 0.0], [0.0, 1.0]]))
        pool = unit_rows(np.array([[1.0, 0.2], [1.0, 0.3], [0.2, 1.0], [-1.0, 0.0]]))
        moved = rocchio(vectors, pool, k=2, beta=1.0)
        assert np.allclose(np.linalg.norm(moved, axis=1), 1.0)
        # Each vector ends closer to the documents it was already nearest to.
        assert moved[0] @ pool[0] > vectors[0] @ pool[0]
        assert moved[1] @ pool[2] > vectors[1] @ pool[2]

    def test_predictions_follow_the_alignment_changes(self):
        from experiments.run_rocchio import predictions_from

        preds = predictions_from({"A": 0.05, "B": -0.01, "C": 0.02})
        signs = {p["corpus"]: p["expected_sign"] for p in preds if "corpus" in p}
        assert signs == {"A": 1, "B": -1, "C": 1}
        assert next(p for p in preds if p["id"] == "ordering")["order"] == ["A", "C", "B"]

    def test_a_preregistration_is_never_overwritten(self, tmp_path, monkeypatch):
        from experiments import run_rocchio

        existing = tmp_path / "prereg.json"
        existing.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(run_rocchio, "PREREGISTRATION", existing)
        assert run_rocchio.preregister() == 1
        assert existing.read_text(encoding="utf-8") == "{}"

    def test_accuracy_refuses_to_run_without_a_preregistration(self, tmp_path, monkeypatch):
        from experiments import run_rocchio

        monkeypatch.setattr(run_rocchio, "PREREGISTRATION", tmp_path / "missing.json")
        assert run_rocchio.evaluate() == 1
