"""Earn issue #13: Python↔Rust ingest overhead bench script exists and smokes.

Absence of ``examples/python/bench_python_overhead.py`` (or a broken smoke
mode / missing table columns) fails this test. Absolute timings are not
asserted — machine noise must not fail CI.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

pytest.importorskip("antecedent")

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "examples" / "python" / "bench_python_overhead.py"

EXPECTED_COLUMNS = {
    "workload",
    "format",
    "iters",
    "ingest_ms",
    "total_ms",
    "native_est_ms",
}
EXPECTED_WORKLOADS = {"analyze", "pcmci"}
EXPECTED_FORMATS = {"dict"}  # pandas / arrow optional


def _load_bench():
    import importlib.util
    import sys

    assert BENCH.is_file(), f"missing earning artifact: {BENCH}"
    name = "bench_python_overhead"
    spec = importlib.util.spec_from_file_location(name, BENCH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass/string annotations resolve.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_issue13_bench_overhead_smoke_table():
    mod = _load_bench()
    buf = io.StringIO()
    with redirect_stdout(buf):
        rows = mod.main(["--smoke"])
    text = buf.getvalue()

    assert rows, "smoke mode must return structured timing rows"
    for col in EXPECTED_COLUMNS:
        assert col in text, f"printed table missing column {col!r}:\n{text}"

    formats = {r.format for r in rows}
    workloads = {r.workload for r in rows}
    assert formats >= EXPECTED_FORMATS, formats
    assert workloads >= EXPECTED_WORKLOADS, workloads

    header = next(line for line in text.splitlines() if line.startswith("workload"))
    for col in EXPECTED_COLUMNS:
        assert col in header.split(), header

    # One row per (workload, format); dict always present for both workloads.
    dict_rows = [r for r in rows if r.format == "dict"]
    assert {r.workload for r in dict_rows} == EXPECTED_WORKLOADS
    for r in rows:
        assert r.iters >= 1
        assert r.ingest_ms >= 0.0
        assert r.total_ms >= 0.0
        assert set(r.as_dict()) >= EXPECTED_COLUMNS
