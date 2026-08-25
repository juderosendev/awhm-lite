"""Tests for AWHMConfig."""

import os

from awhm.config import AWHMConfig


def test_data_dir_expands_home():
    config = AWHMConfig(data_dir="~/awhm-test")
    assert "~" not in config.data_dir
    assert config.data_dir == os.path.join(os.path.expanduser("~"), "awhm-test")


def test_ner_labels_normalised_to_upper():
    config = AWHMConfig(ner_labels={"person", "org"})
    assert config.ner_labels == frozenset({"PERSON", "ORG"})
