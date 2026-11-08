"""Pytest fixtures for P2 tests."""

from __future__ import annotations

import pytest

from ml.p2_boosting.tests.fixtures import fast_config, make_dataset


@pytest.fixture
def synthetic_dataset():
    return make_dataset()


@pytest.fixture
def p2_config():
    return fast_config()
