"""Earning test for issue #9: PCMCI → PulseEffect benchmark example.

Runs ``examples/python/pcmci_pulse_benchmark.py``, which asserts Exact lag-1
parent recovery (``conformance/discovery/pcmci_lag1``) and pulse ≈ 0.8.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("antecedent")


def test_issue9_pcmci_pulse_benchmark_example() -> None:
    import runpy

    runpy.run_path(
        str(
            Path(__file__).resolve().parents[2] / "examples" / "python" / "pcmci_pulse_benchmark.py"
        ),
        run_name="__main__",
    )
