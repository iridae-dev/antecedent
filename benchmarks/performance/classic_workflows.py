#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Classic execution probes; no timing without --measure and a retained output file.

Run identical arguments against preserved baseline and candidate packages. Round
ordering belongs to the external paired-run driver, allowing package isolation.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import resource
import time
from pathlib import Path


def build(args):
    import antecedent as ant
    import numpy as np

    rng = np.random.default_rng(18)
    if args.case == "dose":
        from antecedent.dose_grid import DoseClaims, dose_functional

        dose = np.linspace(0, 4, args.rows)
        outcome = 1 + dose / 2 + dose**2 / 4
        kwargs = {
            "bandwidth": 0.4,
            "bandwidth_range": (0.1, 1.0),
            "grid": np.linspace(0.5, 3.5, args.grid).tolist(),
            "functional": args.functional,
        }
        if args.functional == "derivative":
            kwargs["claims"] = DoseClaims(derivative=True)
        if args.functional == "contrast":
            kwargs.pop("grid")
            kwargs["contrast"] = (1.0, 3.0)
        supports_workers = "threads" in inspect.signature(dose_functional).parameters
        if supports_workers:
            kwargs["threads"] = args.threads

        def call():
            return dose_functional(dose, outcome, **kwargs)

        def check(result):
            if args.functional == "contrast":
                assert abs(result.contrast.estimate - 3.0) < 1e-5
                return {"estimate": result.contrast.estimate}
            points = result.derivatives if args.functional == "derivative" else result.levels
            assert len(points) == args.grid
            values = [point.value for point in points]
            for point in points:
                truth = (
                    0.5 + point.dose / 2
                    if args.functional == "derivative"
                    else 1 + point.dose / 2 + point.dose**2 / 4
                )
                assert abs(point.value - truth) < 1e-5
            return {"values": values, "points": len(values)}

        return call, check, {"workers": args.threads if supports_workers else 1}
    if args.case == "pc":
        from antecedent.discovery import PC

        z = rng.normal(size=(args.columns, args.rows))
        for i in range(1, args.columns):
            z[i] += 0.55 * z[i - 1]
        data = {f"v{i}": column for i, column in enumerate(z)}

        def call():
            return PC(alpha=0.01, fdr=False, max_cond_size=2).run(
                data, seed=18, threads=args.threads
            )

        def decisions(result):
            links = [
                {
                    name: getattr(link, name)
                    for name in (
                        "source",
                        "source_lag",
                        "target",
                        "target_lag",
                        "statistic",
                        "p_value",
                        "adjusted_p_value",
                        "conditioning_set",
                    )
                }
                for link in result.links
            ]
            edges = [
                {
                    name: getattr(edge, name)
                    for name in (
                        "source",
                        "source_lag",
                        "target",
                        "target_lag",
                        "at_source",
                        "at_target",
                    )
                }
                for edge in result.graph_edges
            ]
            return {"links": links, "edges": edges, "ci_tests": result.ci_tests}

        reference = decisions(call())

        def check(result):
            actual = decisions(result)
            assert actual == reference
            return actual

        return call, check, {"workers": args.threads}
    if args.case == "cyclic":
        from antecedent.cyclic import Equation, EquilibriumModel, PreparedCyclicSystem

        d = args.columns
        model = EquilibriumModel(
            tuple(
                Equation(
                    f"v{i}",
                    0.0,
                    tuple(0.2 if j == (i - 1) % d else 0.0 for j in range(d)),
                )
                for i in range(d)
            ),
            tuple(float(i == j) for i in range(d) for j in range(d)),
        )
        noise = rng.normal(scale=0.1, size=(args.rows, d))
        if args.strided:
            noise = np.repeat(noise, 2, axis=1)[:, ::2]
        noise_input = noise if args.strided else noise.tolist()
        typed_available = hasattr(PreparedCyclicSystem, "_from_request")
        worlds = [[], [{"variable": 0, "value": 1.0}]]

        def prepare():
            return model.prepare(
                worlds,
                noise_input if typed_available or not args.strided else noise_input.tolist(),
                factual=[0.0] * d,
            )

        prepared = prepare()
        call = (lambda: prepare().execute()) if args.mode == "cold" else prepared.execute

        def check(result):
            answer = result.report
            assert len(answer["worlds"]) == 2
            worst = 0.0
            for actions, rows in zip(worlds, answer["worlds"], strict=True):
                assert len(rows) == args.rows
                clamps = {a["variable"]: a["value"] for a in actions}
                for u, row in zip(noise, rows, strict=True):
                    assert row["globally_unique"] and row["numerical_converged"]
                    values = row["values"]
                    worst = max(
                        worst,
                        max(
                            abs(
                                values[j]
                                - clamps.get(j, math.tanh(0.2 * values[(j - 1) % d]) + u[j])
                            )
                            for j in range(d)
                        ),
                    )
            assert worst < 1e-7
            return {
                "residual": worst,
                "values": answer["worlds"][0][0]["values"],
                "units": args.rows,
            }

        return call, check, {"workers": 1}
    z = rng.normal(size=(args.columns, args.rows))
    t = (rng.random(args.rows) < 0.5).astype(float)
    y = 2 * t + z.sum(axis=0) / math.sqrt(args.columns) + rng.normal(scale=0.3, size=args.rows)
    truth = 2.0
    if args.case in ("logit", "probit", "poisson"):
        baseline = 0.2 + 0.3 * z.sum(axis=0) / math.sqrt(args.columns)
        if args.case == "probit":
            link = np.vectorize(lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))))
        elif args.case == "logit":

            def link(x):
                return 1.0 / (1.0 + np.exp(-x))
        else:
            link = np.exp
        delta = 0.4 if args.case == "poisson" else 1.0
        mu = link(baseline + delta * t)
        y = (
            rng.poisson(mu).astype(float)
            if args.case == "poisson"
            else (rng.random(args.rows) < mu).astype(float)
        )
        truth = float(np.mean(link(baseline + delta) - link(baseline)))
    data = {"t": t, "y": y, **{f"z{i}": c for i, c in enumerate(z)}}
    if args.masked:
        data["z0"] = data["z0"].copy()
        data["z0"][::10] = np.nan
    if args.strided:
        data = {name: np.repeat(column, 2)[::2] for name, column in data.items()}
    kwargs = {
        "graph": [(f"z{i}", target) for i in range(args.columns) for target in ("t", "y")]
        + [("t", "y")],
        "query": ant.AverageEffect("t", "y"),
        "bootstrap": 0,
        "refute": False,
        "seed": 18,
        "threads": args.threads,
    }
    if args.case in ("logit", "probit", "poisson"):
        kwargs["inference"] = ant.inference.Bayesian(
            backend="laplace", likelihood=args.case, n_draws=400
        )
    prepared = ant.prepare(data, **kwargs)
    call = (lambda: ant.analyze(data, **kwargs)) if args.mode == "cold" else prepared.estimate

    def check(result):
        assert abs(result.effect - truth) < 0.3
        assert not result.performance.cancelled and not result.performance.early_stopped
        if args.case in ("logit", "probit", "poisson"):
            assert result.performance.n_draws == 400 and result.posterior is not None
        return {"effect": result.effect, "draws": result.performance.n_draws}

    return call, check, {"workers": args.threads}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("design", "dose", "pc", "cyclic", "logit", "probit", "poisson"),
        required=True,
    )
    parser.add_argument("--mode", choices=("cold", "prepared"), default="cold")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--grid", type=int, default=129)
    parser.add_argument(
        "--functional", choices=("level", "derivative", "contrast"), default="level"
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--masked", action="store_true")
    parser.add_argument("--strided", action="store_true")
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.rows, args.columns, args.grid, args.threads, args.repeats) <= 0:
        parser.error("sizes, workers and repeat count must be positive")
    if args.measure and args.output is None:
        parser.error("measurement requires --output")
    from antecedent import _native

    call, check, execution = build(args)
    check(call())
    samples = []
    for repeat in range(args.repeats if args.measure else 1):
        start = time.perf_counter_ns() if args.measure else None
        result = call()
        elapsed = time.perf_counter_ns() - start if args.measure else None
        samples.append(
            dict(
                repeat=repeat,
                observed=check(result),
                **({"elapsed_ns": elapsed} if args.measure else {}),
            )
        )
    report = {
        "timed": args.measure,
        "specification": {k: v for k, v in vars(args).items() if k != "output"},
        "execution": execution,
        "native": _native.__file__,
        "native_sha256": hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
        # macOS reports bytes. This process includes fixture + preparation + execution.
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "samples": samples,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"case": args.case, "timed": args.measure, "samples": len(samples)}))


if __name__ == "__main__":
    main()
