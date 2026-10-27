#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Before/after lifecycle counter comparison on three fixed workloads.

The same data seeds, graph, estimator, seeds and call order run on the predecessor API
(mutable prepared study: ``estimate(data)``, ``refresh``, ``PreparedBatch``) and on the
current API (immutable prepared branches: ``rebind``, ``prepare_many``). Only call names
change. Each side runs in its own interpreter against its own built native extension::

    # predecessor side (its own package and extension, nothing installed)
    env PYTHONPATH=<predecessor checkout>/python python \
        compare_lifecycle_baseline.py run --side baseline --output baseline.json
    # current side
    python compare_lifecycle_baseline.py run --side current --output current.json
    # pure-JSON merge and verdicts, no antecedent import
    python compare_lifecycle_baseline.py merge baseline.json current.json \
        --into parity/performance_lanes/lifecycle_counters.json

Counters are exact native operation attempts from ``observe_native_attempts`` (present on
both sides). Phases align the two call sequences: a predecessor call that replaces data and
re-estimates in one step is compared with the current rebind plus estimate. Wall-clock
medians are indicative only, need at least five repeats and a quiet host, and are never
reference-host performance, capacity or calibration claims.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

WORKLOADS = ("prepare_estimate_rebind", "batch_shared_fit", "nuisance_replay")
SEED = 3
NUISANCE_SEED = 13
LOAD_LIMIT = 3.0
# Phases whose native work the predecessor does not report: its nuisance-score fit and refit on
# load sit outside the instrumented components, so an empty predecessor count is absence of
# observation, not absence of work.
UNOBSERVED_BASELINE = {
    "nuisance_replay": ("nuisance_fit", "nuisance_repeat", "load_artifact"),
}


def _inputs(np: Any, n: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-0.3 * z))).astype(float)
    y = 2 * t + z + rng.normal(scale=0.2, size=n)
    return {"z": z, "t": t, "y": y, "y2": 0.5 * y - z}


def _point(result: Any) -> list[float]:
    """Point estimate and analytic standard error of one result."""
    return [float(result.estimate.ate), float(result.estimate.se_analytic)]


def _ops(report: Any) -> dict[str, dict[str, int]]:
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


class Recorder:
    """Runs each step once under the native operation observer."""

    def __init__(self, observe: Callable[..., Any]) -> None:
        self.observe = observe
        self.steps: list[dict[str, Any]] = []

    def step(self, name: str, call: Callable[[], Any], **extra: Any) -> Any:
        attempt = self.observe(call)
        value = attempt.unwrap()
        record: dict[str, Any] = {
            "step": name,
            "native_operations": _ops(attempt.report),
        }
        record["native_fits"] = sum(
            row["attempted"]
            for key, row in record["native_operations"].items()
            if key.endswith("_fit")
        )
        for key, item in extra.items():
            record[key] = item(value) if callable(item) else item
        self.steps.append(record)
        return value


def _modules(side: str) -> dict[str, Any]:
    import antecedent as ant
    import numpy as np
    from antecedent.execution_attempt import observe_native_attempts

    mods: dict[str, Any] = {
        "ant": ant,
        "np": np,
        "observe": observe_native_attempts,
        "side": side,
        "package": str(Path(ant.__file__).resolve().parent),
    }
    if side == "current":
        from antecedent.outcomes import require_success

        mods["require"] = require_success
    return mods


def workload_prepare_estimate_rebind(mods: dict[str, Any]) -> dict[str, Any]:
    ant, np, side = mods["ant"], mods["np"], mods["side"]
    rec = Recorder(mods["observe"])
    data = _inputs(np, 400, 3)
    columns = {k: data[k] for k in ("z", "t", "y")}
    other = _inputs(np, 300, 9)
    other_columns = {k: other[k] for k in ("z", "t", "y")}
    graph = [("z", "t"), ("z", "y"), ("t", "y")]
    query = ant.AverageEffect("t", "y")
    answers: dict[str, float] = {}
    if side == "baseline":
        study = rec.step(
            "prepare",
            lambda: ant.prepare(
                columns,
                graph=graph,
                query=query,
                refute=False,
                bootstrap=0,
                seed=SEED,
                threads=1,
            ),
        )
        answers["estimate_first"] = _point(rec.step("estimate_first", study.estimate))
        answers["estimate_repeat"] = _point(rec.step("estimate_repeat", study.estimate))
        answers["rebound"] = _point(rec.step("refresh", lambda: study.refresh(other_columns)))
        answers["original_after_branch"] = _point(
            rec.step("estimate_original_data", lambda: study.estimate(columns))
        )
    else:
        require = mods["require"]
        study = rec.step(
            "prepare",
            lambda: ant.prepare(
                columns,
                graph=graph,
                query=query,
                refute=False,
                inference=ant.Frequentist(bootstrap=0),
                execution=ant.Execution(seed=SEED, threads=1),
            ),
        )
        first = rec.step("estimate_first", lambda: ant.estimate(study))
        answers["estimate_first"] = _point(require(first))
        done = require(first).study
        answers["estimate_repeat"] = _point(
            require(rec.step("estimate_repeat", lambda: ant.estimate(done)))
        )
        branch = rec.step("rebind", lambda: ant.rebind(done, data=other_columns))
        answers["rebound"] = _point(
            require(rec.step("estimate_rebound", lambda: ant.estimate(branch)))
        )
        answers["original_after_branch"] = _point(
            require(rec.step("estimate_original_after_branch", lambda: ant.estimate(done)))
        )
    phases = {
        "prepare": {"baseline": ["prepare"], "current": ["prepare"]},
        "estimate_first": {
            "baseline": ["estimate_first"],
            "current": ["estimate_first"],
        },
        "estimate_repeat": {
            "baseline": ["estimate_repeat"],
            "current": ["estimate_repeat"],
        },
        "rebound": {"baseline": ["refresh"], "current": ["rebind", "estimate_rebound"]},
        "original_after_branch": {
            "baseline": ["estimate_original_data"],
            "current": ["estimate_original_after_branch"],
        },
    }
    return {"steps": rec.steps, "answers": answers, "phases": phases}


def workload_batch_shared_fit(mods: dict[str, Any]) -> dict[str, Any]:
    ant, np, side = mods["ant"], mods["np"], mods["side"]
    rec = Recorder(mods["observe"])
    data = _inputs(np, 260, 3)
    other = _inputs(np, 190, 9)
    graph = [("z", "t"), ("z", "y"), ("z", "y2"), ("t", "y"), ("t", "y2")]
    queries = [ant.AverageEffect("t", "y"), ant.AverageEffect("t", "y2")]
    answers: dict[str, list[float]] = {}
    if side == "baseline":
        from antecedent._batch import PreparedBatch

        batch = rec.step(
            "prepare_many",
            lambda: PreparedBatch.prepare(
                data,
                graph=graph,
                queries=queries,
                estimator="aipw",
                bootstrap=0,
                refute=False,
                seed=SEED,
                threads=1,
            ),
        )

        def points(results: Any) -> list[float]:
            return [_point(r) for r in results]

        first = rec.step(
            "estimate_many",
            lambda: batch.estimate(data, seed=SEED, threads=1),
            points=points,
        )
        answers["estimate_many"] = points(first)
        repeat = rec.step(
            "estimate_many_repeat",
            lambda: batch.estimate(data, seed=SEED, threads=1),
            points=points,
        )
        answers["estimate_many_repeat"] = points(repeat)
        branch = rec.step(
            "estimate_new_data",
            lambda: batch.estimate(other, seed=SEED, threads=1),
            points=points,
        )
        answers["new_data"] = points(branch)
    else:
        from antecedent.estimation import estimate_many, prepare_many

        require = mods["require"]

        def points(outcome: Any) -> list[float]:
            return [_point(require(m)) for m in require(outcome).members]

        execution = ant.Execution(seed=SEED, threads=1)
        study = rec.step(
            "prepare_many",
            lambda: prepare_many(
                data,
                graph=graph,
                queries=queries,
                estimator="aipw",
                inference=ant.Frequentist(bootstrap=0),
                execution=execution,
                refute=False,
            ),
        )
        first = rec.step("estimate_many", study.estimate, points=points)
        answers["estimate_many"] = points(first)
        done = require(first).study
        repeat = rec.step("estimate_many_repeat", lambda: estimate_many(done), points=points)
        answers["estimate_many_repeat"] = points(repeat)
        rebound = rec.step("rebind", lambda: ant.rebind(done, data=other))
        branch = rec.step("estimate_rebound", lambda: estimate_many(rebound), points=points)
        answers["new_data"] = points(branch)
    phases = {
        "prepare": {"baseline": ["prepare_many"], "current": ["prepare_many"]},
        "estimate_many": {"baseline": ["estimate_many"], "current": ["estimate_many"]},
        "estimate_many_repeat": {
            "baseline": ["estimate_many_repeat"],
            "current": ["estimate_many_repeat"],
        },
        "new_data": {
            "baseline": ["estimate_new_data"],
            "current": ["rebind", "estimate_rebound"],
        },
    }
    return {"steps": rec.steps, "answers": answers, "phases": phases}


def workload_nuisance_replay(mods: dict[str, Any]) -> dict[str, Any]:
    ant, np, side = mods["ant"], mods["np"], mods["side"]
    from antecedent import nuisance_scores as ns

    rec = Recorder(mods["observe"])
    data = {
        "z": np.r_[np.zeros(100), np.ones(100)],
        "t": np.r_[np.ones(30), np.zeros(70), np.ones(60), np.zeros(40)],
        "y": np.arange(200, dtype=float) % 7,
    }
    graph = [("z", "t"), ("z", "y"), ("t", "y")]
    query = ant.AverageEffect("t", "y")
    answers: dict[str, Any] = {}
    if side == "baseline":
        study = rec.step(
            "prepare",
            lambda: ant.prepare(
                data,
                graph=graph,
                query=query,
                bootstrap=0,
                seed=NUISANCE_SEED,
                threads=1,
                refute=False,
            ),
        )

        def fit() -> Any:
            return study.treatment_score_posterior(
                draws=500, summary_rows=[0, 150], seed=NUISANCE_SEED
            )

        product = rec.step("nuisance_fit", fit)
        repeated = rec.step("nuisance_repeat", fit)
    else:
        require = mods["require"]
        study = rec.step(
            "prepare",
            lambda: ant.prepare(
                data,
                graph=graph,
                query=query,
                inference=ant.Frequentist(bootstrap=0),
                execution=ant.Execution(seed=NUISANCE_SEED, threads=1),
                refute=False,
            ),
        )

        def fit() -> Any:
            return require(ns.treatment_score_posterior(study, draws=500, summary_rows=[0, 150]))

        product = rec.step("nuisance_fit", fit)
        repeated = rec.step("nuisance_repeat", fit)
    answers["nuisance_fit"] = dict(sorted(product.values.items()))
    answers["nuisance_repeat"] = dict(sorted(repeated.values.items()))
    artifact = bytes(product.export())
    loaded = rec.step(
        "load_artifact",
        lambda: ns.load_artifact(artifact),
        artifact_bytes=len(artifact),
    )
    answers["load_artifact"] = dict(sorted(loaded.values.items()))
    phases = {
        "prepare": {"baseline": ["prepare"], "current": ["prepare"]},
        "nuisance_fit": {"baseline": ["nuisance_fit"], "current": ["nuisance_fit"]},
        "nuisance_repeat": {
            "baseline": ["nuisance_repeat"],
            "current": ["nuisance_repeat"],
        },
        "load_artifact": {"baseline": ["load_artifact"], "current": ["load_artifact"]},
    }
    return {"steps": rec.steps, "answers": answers, "phases": phases}


def _load_average() -> float:
    return os.getloadavg()[0]


def _timed(function: Callable[[dict[str, Any]], Any], mods: dict[str, Any], repeats: int) -> dict:
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


def run(args: argparse.Namespace) -> None:
    mods = _modules(args.side)
    functions = {name: globals()[f"workload_{name}"] for name in WORKLOADS}
    record: dict[str, Any] = {"side": args.side, "workloads": {}}
    timing_note = None
    if args.timing:
        load = _load_average()
        if load >= LOAD_LIMIT:
            timing_note = f"timing skipped: host load average was not below {LOAD_LIMIT}"
        elif args.repeats < 5:
            raise SystemExit("timing needs at least five repeats")
    for name in WORKLOADS:
        entry = functions[name](mods)
        if args.timing and timing_note is None:
            entry["timing"] = _timed(functions[name], mods, args.repeats)
        record["workloads"][name] = entry
    if timing_note:
        record["timing_note"] = timing_note
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"wrote {args.output}")


def _verdict(baseline: Any, current: Any) -> str:
    if baseline is None or current is None:
        return "not_comparable"
    if current == baseline:
        return "equal"
    return "improved" if current < baseline else "WORSE"


def _phase_counters(steps: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    by_name = {s["step"]: s for s in steps}
    kinds: dict[str, int] = {}
    fits = 0
    for name in names:
        step = by_name[name]
        fits += step["native_fits"]
        for kind, row in step["native_operations"].items():
            kinds[kind] = kinds.get(kind, 0) + row["attempted"]
    return {
        "native_fits": fits,
        "native_attempted_total": sum(kinds.values()),
        "attempted_by_kind": dict(sorted(kinds.items())),
    }


def _flatten(answer: Any) -> list[float]:
    if isinstance(answer, dict):
        return [x for key in sorted(answer) for x in _flatten(answer[key])]
    if isinstance(answer, list):
        return [x for item in answer for x in _flatten(item)]
    return [float(answer)]


def compare(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    workloads: dict[str, Any] = {}
    for name in WORKLOADS:
        b, c = baseline["workloads"][name], current["workloads"][name]
        phases = c["phases"]
        rows: dict[str, Any] = {}
        excluded: list[str] = []
        total_b: dict[str, Any] = {
            "native_fits": 0,
            "native_attempted_total": 0,
            "kinds": {},
        }
        total_c: dict[str, Any] = {
            "native_fits": 0,
            "native_attempted_total": 0,
            "kinds": {},
        }
        for phase, mapping in phases.items():
            pb = _phase_counters(b["steps"], mapping["baseline"])
            pc = _phase_counters(c["steps"], mapping["current"])
            kinds = sorted(set(pb["attempted_by_kind"]) | set(pc["attempted_by_kind"]))
            unobserved = phase in UNOBSERVED_BASELINE.get(name, ())
            if unobserved:
                pb = {
                    "native_fits": None,
                    "native_attempted_total": None,
                    "attempted_by_kind": {},
                    "unobserved_by_predecessor": True,
                }
            verdicts = {
                "native_fits": _verdict(pb["native_fits"], pc["native_fits"]),
                "native_attempted_total": _verdict(
                    pb["native_attempted_total"], pc["native_attempted_total"]
                ),
                "attempted_by_kind": {
                    kind: "not_comparable"
                    if unobserved
                    else _verdict(
                        pb["attempted_by_kind"].get(kind, 0),
                        pc["attempted_by_kind"].get(kind, 0),
                    )
                    for kind in kinds
                },
            }
            rows[phase] = {
                "baseline_steps": mapping["baseline"],
                "current_steps": mapping["current"],
                "baseline": pb,
                "current": pc,
                "verdict": verdicts,
            }
            if unobserved:
                excluded.append(phase)
                continue
            for total, part in ((total_b, pb), (total_c, pc)):
                total["native_fits"] += part["native_fits"]
                total["native_attempted_total"] += part["native_attempted_total"]
                for kind, count in part["attempted_by_kind"].items():
                    total["kinds"][kind] = total["kinds"].get(kind, 0) + count
        kinds = sorted(set(total_b["kinds"]) | set(total_c["kinds"]))
        answers_b, answers_c = _flatten(b["answers"]), _flatten(c["answers"])
        max_diff = max(abs(x - y) for x, y in zip(answers_b, answers_c, strict=True))
        extra_b = {key: [s.get(key) for s in b["steps"] if key in s] for key in ("artifact_bytes",)}
        extra_c = {key: [s.get(key) for s in c["steps"] if key in s] for key in ("artifact_bytes",)}
        workloads[name] = {
            "baseline": {"steps": b["steps"], "answers": b["answers"]},
            "current": {"steps": c["steps"], "answers": c["answers"]},
            "phases": rows,
            "total": {
                "phases_excluded_unobserved_by_predecessor": excluded,
                "baseline": {
                    "native_fits": total_b["native_fits"],
                    "native_attempted_total": total_b["native_attempted_total"],
                    "attempted_by_kind": dict(sorted(total_b["kinds"].items())),
                },
                "current": {
                    "native_fits": total_c["native_fits"],
                    "native_attempted_total": total_c["native_attempted_total"],
                    "attempted_by_kind": dict(sorted(total_c["kinds"].items())),
                },
                "verdict": {
                    "native_fits": _verdict(total_b["native_fits"], total_c["native_fits"]),
                    "native_attempted_total": _verdict(
                        total_b["native_attempted_total"],
                        total_c["native_attempted_total"],
                    ),
                    "attempted_by_kind": {
                        kind: _verdict(total_b["kinds"].get(kind, 0), total_c["kinds"].get(kind, 0))
                        for kind in kinds
                    },
                    "artifact_bytes": _verdict(
                        (extra_b["artifact_bytes"] or [None])[0],
                        (extra_c["artifact_bytes"] or [None])[0],
                    ),
                },
            },
            "answer_equality": {
                "values_compared": len(answers_b),
                "max_abs_difference": max_diff,
                "within_1e-9": max_diff <= 1e-9,
            },
        }
        if "timing" in b and "timing" in c:
            workloads[name]["timing"] = {
                "baseline": b["timing"],
                "current": c["timing"],
            }
    return workloads


def merge(args: argparse.Namespace) -> None:
    baseline = json.loads(args.baseline.read_text())
    current = json.loads(args.current.read_text())
    target = json.loads(args.into.read_text())
    target["baseline_comparison"] = {
        "scope": "lifecycle_counters_local_not_reference_host_not_calibration",
        "verdicts": "equal | improved (current attempts fewer) | WORSE (current attempts more) | not_comparable",
        "seed": {"workloads_1_2": SEED, "workload_3": NUISANCE_SEED},
        "workloads": compare(baseline, current),
    }
    notes = {
        k: v["timing_note"]
        for k, v in (("baseline", baseline), ("current", current))
        if "timing_note" in v
    }
    if notes:
        target["baseline_comparison"]["timing_notes"] = notes
    args.into.write_text(json.dumps(target, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"updated {args.into}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--side", choices=("baseline", "current"), required=True)
    run_p.add_argument("--output", type=Path, required=True)
    run_p.add_argument("--timing", action="store_true")
    run_p.add_argument("--repeats", type=int, default=5)
    run_p.set_defaults(func=run)
    merge_p = sub.add_parser("merge")
    merge_p.add_argument("baseline", type=Path)
    merge_p.add_argument("current", type=Path)
    merge_p.add_argument("--into", type=Path, required=True)
    merge_p.set_defaults(func=merge)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
