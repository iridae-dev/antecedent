#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Deterministic lifecycle counter comparison on fixed workloads.

Counters (native operation attempts, fit counts, ingestion copies, retained and admitted
bytes) are exact and reproducible, so they are always recorded. Wall-clock medians are
local indicative values only and are recorded solely under ``--timing``. Nothing here is a
reference-host performance or calibration claim.

Every workload is defined once, by data seed, graph and call order, so the same definition
can be replayed against another checkout. A ``--side`` label records which build produced
the counters; the baseline side needs its own built native extension and is not built here.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

WORKLOADS = (
    "prepare_estimate_rebind",
    "batch_shared_fit",
    "nuisance_replay",
    "trial_transport",
    "cyclic_family_branching",
    "retained_refresh",
    "ingestion_controls",
)


def _modules() -> dict[str, Any]:
    import antecedent as ant
    import numpy as np
    from antecedent.execution_attempt import observe_native_attempts
    from antecedent.outcomes import require_success

    return {
        "ant": ant,
        "np": np,
        "observe": observe_native_attempts,
        "require": require_success,
    }


def _ops(report: Any) -> dict[str, dict[str, int]]:
    """Operations that were begun, in sorted order."""
    return {
        key: {
            "attempted": row.attempted,
            "completed": row.completed,
            "failed": row.failed,
            "unfinished": row.unfinished,
        }
        for key, row in sorted(report.operations.items())
        if row.attempted
    }


def _facts(facts: Any) -> dict[str, Any] | None:
    if facts is None:
        return None
    names = (
        "observed_scans",
        "observed_copies",
        "observed_copied_bytes",
        "observed_retained_bytes",
        "observed_model_fits",
        "peak_admitted_bytes",
        "observed_spill_bytes",
        "observed_allocations",
        "observed_allocated_bytes",
    )
    return {name: getattr(facts, name) for name in names}


class Recorder:
    def __init__(self, mods: dict[str, Any]) -> None:
        self.mods = mods
        self.steps: list[dict[str, Any]] = []

    def step(self, name: str, call: Callable[[], Any], **extra: Any) -> Any:
        """Run ``call`` once under the native operation observer and record its counters."""
        attempt = self.mods["observe"](call)
        value = attempt.unwrap()
        record: dict[str, Any] = {"step": name, "native_operations": _ops(attempt.report)}
        record["native_fits"] = sum(
            row["attempted"]
            for key, row in record["native_operations"].items()
            if key.endswith("_fit")
        )
        for key, item in extra.items():
            record[key] = item(value) if callable(item) else item
        self.steps.append(record)
        return value


def _inputs(np: Any, n: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-0.3 * z))).astype(float)
    y = 2 * t + z + rng.normal(scale=0.2, size=n)
    return {"z": z, "t": t, "y": y, "y2": 0.5 * y - z}


def _estimate_facts(mods: dict[str, Any]) -> Callable[[Any], dict[str, Any]]:
    def read(outcome: Any) -> dict[str, Any]:
        result = mods["require"](outcome)
        perf = result.performance
        return {
            "bytes_borrowed": perf.bytes_borrowed,
            "design_reused": any(d.startswith("exec.design.cached:") for d in result.diagnostics),
            "cancelled": perf.cancelled,
        }

    return read


def workload_prepare_estimate_rebind(mods: dict[str, Any]) -> list[dict[str, Any]]:
    ant, np = mods["ant"], mods["np"]
    rec = Recorder(mods)
    data = _inputs(np, 400, 3)
    graph = [("z", "t"), ("z", "y"), ("t", "y")]
    prepared = rec.step(
        "prepare",
        lambda: ant.prepare(
            {k: data[k] for k in ("z", "t", "y")},
            graph=graph,
            query=ant.AverageEffect("t", "y"),
            refute=False,
            inference=ant.Frequentist(bootstrap=0),
            execution=ant.Execution(seed=3, threads=1),
        ),
    )
    first = rec.step("estimate_first", lambda: ant.estimate(prepared), facts=_estimate_facts(mods))
    study = first.study
    rec.step("estimate_repeat", lambda: ant.estimate(study), facts=_estimate_facts(mods))
    other = _inputs(np, 300, 9)
    branch = rec.step(
        "rebind", lambda: ant.rebind(study, data={k: other[k] for k in ("z", "t", "y")})
    )
    rec.step("estimate_rebound", lambda: ant.estimate(branch), facts=_estimate_facts(mods))
    rec.step(
        "estimate_original_after_branch", lambda: ant.estimate(study), facts=_estimate_facts(mods)
    )
    return rec.steps


def workload_batch_shared_fit(mods: dict[str, Any]) -> list[dict[str, Any]]:
    ant, np = mods["ant"], mods["np"]
    from antecedent.estimation import estimate_many, prepare_many

    rec = Recorder(mods)
    data = _inputs(np, 260, 3)
    study = rec.step(
        "prepare_many",
        lambda: prepare_many(
            data,
            graph=[("z", "t"), ("z", "y"), ("z", "y2"), ("t", "y"), ("t", "y2")],
            queries=[ant.AverageEffect("t", "y"), ant.AverageEffect("t", "y2")],
            estimator="aipw",
            inference=ant.Frequentist(bootstrap=0),
            refute=False,
        ),
        materialized_plans=lambda s: s._native.materialized_plan_count(),
    )
    first = rec.step(
        "estimate_many",
        study.estimate,
        materialized_plans=lambda o: mods["require"](o).study._native.materialized_plan_count(),
    )
    done = mods["require"](first).study
    rec.step(
        "estimate_many_repeat",
        lambda: estimate_many(done),
        materialized_plans=lambda o: mods["require"](o).study._native.materialized_plan_count(),
        shared_propensity_notes=lambda o: sum(
            any("propensity shared" in n for n in mods["require"](m).diagnostics)
            for m in mods["require"](o).members
        ),
    )
    rec.step("rebind", lambda: ant.rebind(done, data=_inputs(np, 190, 9)))
    return rec.steps


def workload_nuisance_replay(mods: dict[str, Any]) -> list[dict[str, Any]]:
    ant, np = mods["ant"], mods["np"]
    from antecedent import nuisance_scores as ns

    rec = Recorder(mods)
    data = {
        "z": np.r_[np.zeros(100), np.ones(100)],
        "t": np.r_[np.ones(30), np.zeros(70), np.ones(60), np.zeros(40)],
        "y": np.arange(200, dtype=float) % 7,
    }
    study = rec.step(
        "prepare",
        lambda: ant.prepare(
            data,
            graph=[("z", "t"), ("z", "y"), ("t", "y")],
            query=ant.AverageEffect("t", "y"),
            inference=ant.Frequentist(bootstrap=0),
            execution=ant.Execution(seed=13, threads=1),
            refute=False,
        ),
    )
    product = rec.step(
        "nuisance_fit",
        lambda: ns.treatment_score_posterior(study, draws=500, summary_rows=[0, 150]),
    )
    rec.step(
        "nuisance_repeat",
        lambda: ns.treatment_score_posterior(study, draws=500, summary_rows=[0, 150]),
    )
    artifact = bytes(mods["require"](product).export())
    rec.step("load_artifact", lambda: ns.load_artifact(artifact), artifact_bytes=len(artifact))
    return rec.steps


def workload_trial_transport(mods: dict[str, Any]) -> list[dict[str, Any]]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "tests"))
    from dataclasses import replace

    from antecedent.execution import Execution
    from antecedent.transport import advanced

    from test_learned_continuous_transport import fixture, options

    rec = Recorder(mods)
    query, data = fixture()
    study = rec.step(
        "prepare",
        lambda: advanced.prepare_learned_continuous(
            query, data, options=options(), execution=Execution(seed=5)
        ),
    )
    first = rec.step(
        "estimate",
        study.estimate,
        estimate=lambda o: round(mods["require"](o).estimate, 9),
    )
    shifted = replace(
        data, outcome=tuple(y + 2.0 * a for y, a in zip(data.outcome, data.treatment, strict=True))
    )
    branch = rec.step("bind_snapshot", lambda: study.bind_snapshot(shifted))
    rec.step(
        "estimate_branch",
        branch.estimate,
        estimate=lambda o: round(mods["require"](o).estimate, 9),
    )
    rec.step("estimate_original_again", study.estimate)
    rec.step("export", lambda: bytes(mods["require"](first).export()), artifact_bytes=len)
    return rec.steps


def workload_cyclic_family_branching(mods: dict[str, Any]) -> list[dict[str, Any]]:
    from antecedent.cyclic import Equation, EquilibriumModel

    rec = Recorder(mods)
    model = EquilibriumModel(
        (Equation("A", 0.0, (0.0, 0.4), "identity"), Equation("B", 0.0, (0.5, 0.0), "identity")),
        (1.0, 0.0, 0.0, 1.0),
    )
    rows = [{"values": [x, 0.7 * x], "clamps": [{"variable": 0, "value": x}]} for x in (-1.0, 1.0)]
    source = rec.step(
        "prepare",
        lambda: model.prepare(
            [[]],
            [[0.0, 0.0]],
            parameters=[{"kind": "weight", "equation": 1, "parent": 0}],
            rows=rows,
        ),
    )
    family = rec.step("to_family", source.to_family)
    facts = lambda o: _facts(mods["require"](o).execution_facts)  # noqa: E731
    rec.step("execute_family", family.execute, execution_facts=facts)
    rec.step("execute_source_after_family", source.execute, execution_facts=facts)
    branch = rec.step(
        "reconfigure", lambda: source.reconfigure([[{"variable": 0, "value": 2.0}]], [[0.0, 0.0]])
    )
    rec.step("execute_reconfigured", branch.execute, execution_facts=facts)
    rebound = rec.step(
        "rebind",
        lambda: source.rebind(
            [{"values": [x, 0.6 * x], "clamps": [{"variable": 0, "value": x}]} for x in (-1.0, 1.0)]
        ),
    )
    rec.step("execute_rebound", rebound.execute, execution_facts=facts)
    rec.step(
        "inspect_source",
        source.inspect,
        fits_performed=lambda r: r["fits_performed"],
    )
    return rec.steps


def workload_retained_refresh(mods: dict[str, Any]) -> list[dict[str, Any]]:
    ant, np = mods["ant"], mods["np"]
    from antecedent.prepared import prepare
    from antecedent.state import (
        RefreshMutation,
        RefreshObservation,
        RetainedRefresh,
        SpillRegistry,
    )

    def row(i: int) -> Any:
        t = float(i % 7) / 3.0
        y = 1.0 + 2.0 * t + 0.1 * np.sin(1.7 * i)
        return RefreshObservation(i, (1.0, t), float(y), 1.0, False)

    rec = Recorder(mods)
    t = np.tile([0.0, 1.0], 32)
    z = np.repeat(np.linspace(-1.0, 1.0, 32), 2)
    study = prepare(
        {"t": t, "z": z, "y": 2.0 * t + z},
        graph=ant.Dag.from_edges(["t", "z", "y"], [("z", "t"), ("z", "y"), ("t", "y")]),
        query=ant.AverageEffect("t", "y"),
        estimator="linear.adjustment.ate",
        refute=False,
    )
    refresh = rec.step(
        "start",
        lambda: RetainedRefresh.start(
            study, source="s0", width=2, rows=[row(i) for i in range(20)]
        ),
        frame_bytes=lambda r: r._native.frame_bytes(),
    )
    for name, mutation in (
        ("append", RefreshMutation.append("s0", [row(i) for i in range(20, 30)])),
        ("window", RefreshMutation.window(15)),
        ("evict", RefreshMutation.evict(20)),
        ("replace", RefreshMutation.replace("s1", [row(100 + i) for i in range(12)])),
    ):
        rec.step(
            f"apply_{name}",
            lambda m=mutation: refresh.apply(m),
            outcome=lambda r: r,
            frame_bytes=lambda _r: refresh._native.frame_bytes(),
        )
    rec.step("verify", refresh.verify, frame_bytes=lambda _r: refresh._native.frame_bytes())
    summary = rec.step(
        "summary_only", refresh.summary_only, frame_bytes=lambda r: r._native.frame_bytes()
    )
    _ = summary
    with tempfile.TemporaryDirectory() as directory:
        registry = SpillRegistry(directory, 1 << 20)
        spilled = rec.step("spill", lambda: refresh.spill(registry, "k0"))
        rec.step("reload", spilled.reload, frame_bytes=lambda r: r._native.frame_bytes())
    return rec.steps


def workload_ingestion_controls(mods: dict[str, Any]) -> list[dict[str, Any]]:
    np = mods["np"]
    from antecedent._data import ingest_columns_with_receipt, ingestion_receipt

    n = 1000
    rng = np.random.default_rng(7)
    tables = {
        "float64_columns": {"a": rng.normal(size=n), "b": rng.normal(size=n)},
        "int64_and_bool_columns": {
            "a": np.arange(n, dtype=np.int64),
            "b": (np.arange(n) % 2).astype(bool),
            "c": rng.normal(size=n),
        },
        "float32_column": {"a": rng.normal(size=n).astype(np.float32)},
    }
    steps = []
    for name, table in tables.items():
        receipt = ingestion_receipt(table)
        steps.append(
            {
                "step": name,
                "rows": receipt.row_count,
                "python_copies": [
                    {"column": c.column, "kind": c.kind, "nbytes": c.nbytes}
                    for c in receipt.python_copies
                ],
                "python_copy_count": len(receipt.python_copies),
                "python_copy_bytes": sum(c.nbytes for c in receipt.python_copies),
                "widening_ingest_copies": [
                    {"column": c.column, "kind": c.kind, "nbytes": c.nbytes}
                    for c in ingest_columns_with_receipt(table)[2]
                ],
            }
        )
    return steps


def _timed(
    function: Callable[[dict[str, Any]], Any], mods: dict[str, Any], repeats: int
) -> dict[str, Any]:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        function(mods)
        samples.append(time.perf_counter_ns() - started)
    return {
        "indicative_only": True,
        "repeats": repeats,
        "median_ns": int(statistics.median(samples)),
        "min_ns": min(samples),
        "max_ns": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", default="current", help="label of the build under test")
    parser.add_argument("--workload", action="append", choices=WORKLOADS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--timing", action="store_true", help="also record local wall-clock medians"
    )
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 5:
        parser.error("timing needs at least five repeats")
    mods = _modules()
    functions = {name: globals()[f"workload_{name}"] for name in WORKLOADS}
    record: dict[str, Any] = {
        "side": args.side,
        "scope": "lifecycle_counters_local_not_reference_host_not_calibration",
        "native_fit_definition": "sum of attempted native operations whose kind ends in _fit",
        "workloads": {},
    }
    for name in args.workload or WORKLOADS:
        entry: dict[str, Any] = {"steps": functions[name](mods)}
        if args.timing:
            entry["timing"] = _timed(functions[name], mods, args.repeats)
        record["workloads"][name] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
