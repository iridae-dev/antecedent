"""Earning test for issue #9: PCMCI → PulseEffect benchmark example.

Runs ``examples/python/pcmci_pulse_benchmark.py``, which asserts Exact lag-1
parent recovery (``conformance/discovery/pcmci_lag1``) and pulse ≈ 0.8.
"""

from __future__ import annotations

from _repo_text import REPO_ROOT


def test_issue9_pcmci_pulse_benchmark_example() -> None:
    import runpy

    runpy.run_path(
        str(REPO_ROOT / "examples" / "python" / "pcmci_pulse_benchmark.py"),
        run_name="__main__",
    )
