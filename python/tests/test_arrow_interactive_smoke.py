"""gate: Arrow CDI is the interactive estimate ingest path (BACKLOG E).

Requires pyarrow. Skipped when unavailable.
"""

from __future__ import annotations

import math

import pytest
from antecedent.outcomes import require_success

from _scm import confounded_scm

pytest.importorskip("pyarrow")

import antecedent
import pyarrow as pa


def test_arrow_cdi_borrow_and_owned_snapshot_estimate():
    data, edges = confounded_scm(n=600, seed=7)
    table = pa.table(
        {
            "t": pa.array(data["t"], type=pa.float64()),
            "y": pa.array(data["y"], type=pa.float64()),
            "z": pa.array(data["z"], type=pa.float64()),
        }
    )
    n_rows = table.num_rows
    n_cols = table.num_columns

    info = antecedent.data.load_float64_arrow_c_columns(
        list(table.column_names),
        [table.column(i).combine_chunks() for i in range(n_cols)],
    )
    assert info.bytes_borrowed >= n_rows * n_cols * 8

    # Arrow may alias writable NumPy buffers. The common input contract owns
    # those bytes once; its receipt must describe the copy rather than borrowing.
    owned = antecedent.data.table(table)
    assert owned.bytes_copied == n_rows * n_cols * 8
    assert owned.bytes_borrowed == 0
    first_treatment = float(owned["t"][0])
    previous_treatment = data["t"][0]
    data["t"][0] += 10.0
    assert float(owned["t"][0]) == first_treatment
    data["t"][0] = previous_treatment

    arrow_result = require_success(
        antecedent.analyze(
            owned,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Frequentist(bootstrap=0),
            refute="cheap",
            execution=antecedent.Execution(seed=1),
        )
    )
    assert math.isfinite(arrow_result.ate)
    assert abs(arrow_result.ate - 2.0) < 0.5
    assert arrow_result.identification.status
    assert arrow_result.plan.validation_suite == "overlap+evalue"
    assert arrow_result.performance.bootstrap_replicates_requested == 0
    assert arrow_result.performance.bytes_borrowed in (None, 0)

    # Pandas / dict twin remains correct; CDI must not diverge silently.
    dict_result = require_success(
        antecedent.analyze(
            data,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Frequentist(bootstrap=0),
            refute="cheap",
            execution=antecedent.Execution(seed=1),
        )
    )
    assert abs(arrow_result.ate - dict_result.ate) < 1e-9


def test_arrow_interactive_on_stage_streams_payloads():
    # Regression: the Arrow C entry point silently lacked `on_stage`, so the
    # interactive click path (Arrow table + progressive stages) raised
    # TypeError while the dict path worked.
    data, edges = confounded_scm(n=400, seed=5)
    table = pa.table(
        {
            "t": pa.array(data["t"], type=pa.float64()),
            "y": pa.array(data["y"], type=pa.float64()),
            "z": pa.array(data["z"], type=pa.float64()),
        }
    )
    stages: list[str] = []

    def on_stage(stage: str, payload: dict) -> None:
        stages.append(stage)

    result = require_success(
        antecedent.analyze(
            table,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Frequentist(bootstrap=0),
            refute=False,
            execution=antecedent.Execution(on_stage=on_stage),
        )
    )
    assert math.isfinite(result.ate)
    assert stages, "on_stage must fire on the Arrow path"


def test_arrow_interactive_cancel_and_progress():
    data, edges = confounded_scm(n=400, seed=11)
    table = pa.table(
        {
            "t": pa.array(data["t"], type=pa.float64()),
            "y": pa.array(data["y"], type=pa.float64()),
            "z": pa.array(data["z"], type=pa.float64()),
        }
    )
    requested = 80
    token = antecedent.state.CancellationToken()
    seen = {"bootstrap": False}

    def on_progress(fraction: float, stage: str) -> None:
        if stage == "bootstrap" and not seen["bootstrap"]:
            seen["bootstrap"] = True
            token.cancel()

    partial = require_success(
        antecedent.analyze(
            table,
            graph=edges,
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Frequentist(bootstrap=requested),
            refute=False,
            execution=antecedent.Execution(
                seed=3, threads=1, cancel=token, on_progress=on_progress
            ),
        )
    )
    assert math.isfinite(partial.ate)
    assert partial.performance.cancelled
    ok = partial.performance.bootstrap_replicates_ok or 0
    assert ok < requested
