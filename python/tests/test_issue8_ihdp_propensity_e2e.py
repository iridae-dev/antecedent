"""Earning test: IHDP-style propensity / AIPW example (GitHub issue #8)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("antecedent")


def test_issue8_ihdp_propensity_e2e():
    """Running the example asserts identification + estimate + refuter succeed."""
    import runpy

    example = (
        Path(__file__).resolve().parents[2] / "examples/python/ihdp_propensity_e2e.py"
    )
    assert example.is_file(), f"missing example script: {example}"
    runpy.run_path(str(example), run_name="__main__")
