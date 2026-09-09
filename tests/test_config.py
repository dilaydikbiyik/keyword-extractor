"""
Unit tests for configuration loading and the pipeline factory
(src/utils/config.py, src/pipeline.py)
Run with: pytest tests/test_config.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import yaml

from utils.config import DEFAULTS, get, load_config, sectors_file


class TestLoadConfig:
    def test_reads_the_shipped_file(self):
        config = load_config("config/config.yaml")
        assert config["embedding"]["model_name"] == "paraphrase-multilingual-MiniLM-L12-v2"
        assert config["extraction"]["top_n_final"] == 10

    def test_missing_file_falls_back_to_defaults(self, tmp_path):
        assert load_config(str(tmp_path / "absent.yaml")) == DEFAULTS

    def test_partial_file_is_merged_over_defaults(self, tmp_path):
        path = tmp_path / "partial.yaml"
        path.write_text(yaml.safe_dump({"extraction": {"top_n_final": 3}}), encoding="utf-8")
        config = load_config(str(path))
        assert config["extraction"]["top_n_final"] == 3
        # Untouched keys in the same section survive the merge.
        assert config["extraction"]["diversity"] == DEFAULTS["extraction"]["diversity"]
        assert config["embedding"]["model_name"] == DEFAULTS["embedding"]["model_name"]

    def test_non_mapping_file_is_an_error(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- just\n- a list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_config(str(path))

    def test_empty_file_is_not_an_error(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert load_config(str(path)) == DEFAULTS


class TestGet:
    def test_dotted_lookup(self):
        assert get({"a": {"b": 7}}, "a.b") == 7

    def test_missing_key_returns_default(self):
        assert get({"a": {}}, "a.b", "fallback") == "fallback"
        assert get({}, "a.b.c") is None

    def test_does_not_walk_into_non_mappings(self):
        assert get({"a": 5}, "a.b", "fallback") == "fallback"


class TestSectorsFile:
    def test_derived_from_the_taxonomy_directory(self):
        path = sectors_file({"paths": {"taxonomy": "custom/dir/"}})
        assert path.endswith("sectors.json")
        assert "custom/dir" in path


class TestEveryConfigKeyIsHonoured:
    """A configuration option that changes nothing is worse than none at all."""

    def test_shipped_file_has_no_keys_the_code_ignores(self):
        shipped = load_config("config/config.yaml")
        # DEFAULTS enumerates exactly what the code reads.
        for section, values in shipped.items():
            assert section in DEFAULTS, f"config.yaml has unread section '{section}'"
            if isinstance(values, dict):
                unknown = set(values) - set(DEFAULTS[section])
                assert not unknown, f"config.yaml has unread keys in {section}: {unknown}"


@pytest.mark.slow
class TestBuildController:
    def test_builds_from_the_shipped_config(self):
        from pipeline import build_controller

        controller, config = build_controller("config/config.yaml")
        assert controller.config["top_n_keywords"] == config["extraction"]["top_n_final"]
        assert controller.classifier.confidence_threshold == (
            config["classification"]["confidence_threshold"]
        )

    def test_config_drives_the_keyword_count(self, tmp_path):
        from pipeline import build_controller

        path = tmp_path / "small.yaml"
        path.write_text(yaml.safe_dump({"extraction": {"top_n_final": 3}}), encoding="utf-8")
        controller, _ = build_controller(str(path))
        result = controller.extract("Handel mit Elektronik und Computern. Großhandel.")
        assert result["status"] == "success"
        assert len(result["keywords"]) <= 3

    def test_logging_handlers_are_not_duplicated(self):
        from pipeline import build_controller

        first, _ = build_controller("config/config.yaml")
        before = len(first.logger.handlers)
        second, _ = build_controller("config/config.yaml")
        assert len(second.logger.handlers) == before
