#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Measure complete Python first answers and prepared estimates; retain every repeat.

This diagnostic probe is not the full N20 hardware/control/capacity grid.
An import failure or incomplete uncertainty is an error, never a skipped lane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from functools import partial
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--bootstrap", type=int, default=199)
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows, args.threads, args.repeats, args.bootstrap, args.draws) < 1:
        parser.error("rows, threads, repeats and uncertainty budgets must be positive")
    try:
        import antecedent as ant
        import numpy as np
        from antecedent import _native
    except ImportError as error:
        parser.exit(
            1,
            f"Python benchmark preflight failed: {error}\n"
            "Repair/install a matching Python package and native extension before measuring. "
            "For an editable checkout, run maturin develop --release from python/ in its virtualenv.\n",
        )
    rng = np.random.default_rng(7)
    z = rng.normal(size=args.rows)
    t = (rng.random(args.rows) < 1 / (1 + np.exp(-0.9 * z))).astype(float)
    y = 2 * t + z + rng.normal(scale=0.4, size=args.rows)
    data = {"t": t, "y": y, "z": z}
    common = {
        "graph": [("z", "t"), ("z", "y"), ("t", "y")],
        "query": ant.AverageEffect("t", "y"),
        "refute": False,
        "seed": 11,
        "threads": args.threads,
    }
    cells = []
    for family in ("bootstrap", "gaussian_bayes"):
        options = dict(common, bootstrap=args.bootstrap if family == "bootstrap" else 0)
        if family == "gaussian_bayes":
            options["inference"] = ant.inference.Bayesian(n_draws=args.draws)
        prepared = ant.prepare(data, **options)
        for mode in ("first_answer", "prepared_estimate"):
            call = (
                partial(ant.analyze, data, **options)
                if mode == "first_answer"
                else prepared.estimate
            )
            call()  # Declared single warmup, outside the retained repeats.
            samples = []
            for repeat in range(args.repeats):
                started = time.perf_counter_ns()
                result = call()
                nanos = time.perf_counter_ns() - started
                perf = result.performance
                if not math.isfinite(result.effect) or abs(result.effect - 2) > 0.3:
                    raise RuntimeError("known-truth response failed")
                if perf.cancelled or perf.early_stopped:
                    raise RuntimeError("incomplete numerical effort")
                if family == "bootstrap":
                    if (
                        perf.bootstrap_replicates_ok != args.bootstrap
                        or not math.isfinite(result.estimate.se_bootstrap)
                    ):
                        raise RuntimeError("incomplete bootstrap uncertainty")
                elif perf.n_draws != args.draws or result.posterior is None:
                    raise RuntimeError("incomplete posterior uncertainty")
                samples.append(
                    {
                        "repeat": repeat,
                        "elapsed_ns": nanos,
                        "effect": result.effect,
                        "analytic_se": result.estimate.se_analytic,
                        "bootstrap_se": result.estimate.se_bootstrap,
                        "native_wall_ns": perf.wall_time_ns,
                        "stage_times_ns": dict(perf.stage_timings),
                        "bootstrap_requested": perf.bootstrap_replicates_requested,
                        "bootstrap_ok": perf.bootstrap_replicates_ok,
                        "posterior_draws": perf.n_draws,
                        "design_reused": any(
                            d.startswith("exec.design.cached:") for d in result.diagnostics
                        ),
                        "cancelled": perf.cancelled,
                        "early_stopped": perf.early_stopped,
                    }
                )
            times = [r["elapsed_ns"] for r in samples]
            cells.append(
                {
                    "family": family,
                    "mode": mode,
                    "samples": samples,
                    "median_ns": statistics.median(times),
                }
            )
    record = {
        "scope": "diagnostic_subset_not_full_N20_acceptance",
        "rows": args.rows,
        "width": 3,
        "threads_requested": args.threads,
        "warmups": 1,
        "replicates": args.bootstrap,
        "draws": args.draws,
        "seed": 7,
        "package": ant.__file__,
        "native": _native.__file__,
        "native_sha256": hashlib.sha256(
            Path(_native.__file__).read_bytes()
        ).hexdigest(),
        "platform": platform.platform(),
        "unavailable_native_counters": [
            "source_passes",
            "fit_count",
            "copy_bytes",
            "retained_outputs",
            "spill_bytes",
        ],
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cells": [
                    {k: c[k] for k in ("family", "mode", "median_ns")} for c in cells
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
