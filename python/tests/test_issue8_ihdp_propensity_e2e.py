"""Earning test: IHDP-style propensity / AIPW example (GitHub issue #8)."""

from __future__ import annotations

from _repo_text import REPO_ROOT


def test_issue8_ihdp_propensity_e2e():
    """Running the example asserts identification + estimate + refuter succeed."""
    import runpy

    example = REPO_ROOT / "examples/python/ihdp_propensity_e2e.py"
    assert example.is_file(), f"missing example script: {example}"
    runpy.run_path(str(example), run_name="__main__")
