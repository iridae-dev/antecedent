"""Earning test for issue #15: EconML adjustment-set handoff example."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

pytest.importorskip("antecedent")

EXAMPLE = (
    Path(__file__).resolve().parents[2] / "examples" / "python" / "econml_cate_handoff.py"
)


def test_issue15_econml_cate_handoff_example() -> None:
    """Run the example via runpy; EconML fitting is optional."""
    ns = runpy.run_path(str(EXAMPLE), run_name="__main__")
    handoff = ns["LAST_HANDOFF"]
    assert handoff["confounders"] == ("z",)
    assert abs(float(handoff["ate"]) - float(ns["TRUE_ATE"])) < 0.35
    # EconML half is soft: present only when econml is installed.
    if "econml_cate" in handoff:
        assert abs(float(handoff["econml_cate"]) - float(ns["TRUE_ATE"])) < 0.5
