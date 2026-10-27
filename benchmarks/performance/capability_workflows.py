#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Capability-specific workflow probes. No timings without explicit --measure.

Fixture construction, scientific checks and profiles are separate from timed
complete calls. This is a diagnostic matrix, not full N20 or calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NAMES = (
    "logit_laplace",
    "gaussian_hmc",
    "crossfit_dml",
    "dose_grid",
    "cyclic_worlds",
    "cyclic_fit",
    "cyclic_identify",
    "cyclic_discover",
    "ode",
    "ou",
    "constraints",
    "artifact_replay",
    "artifact_fresh_process",
)


@dataclass
class Case:
    capability: str
    mode: str
    call: Callable[[], Any]
    check: Callable[[Any], dict[str, Any]]
    specification: dict[str, Any]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def build_cases(name: str, args: argparse.Namespace, scratch: Path) -> list[Case]:
    import antecedent as ant
    import numpy as np
    from antecedent.cyclic import (
        CyclicGraph,
        Equation,
        EquilibriumModel,
        OdeSystem,
        OrnsteinUhlenbeck,
        PolynomialSystem,
        consume_artifact,
    )
    from antecedent.outcomes import require_success

    spec = {"thread_policy": "public_api_default"}
    cases = []
    if name in ("logit_laplace", "gaussian_hmc", "crossfit_dml"):
        rng = np.random.default_rng(args.seed)
        z = rng.normal(size=args.rows)
        t = (rng.random(args.rows) < 1 / (1 + np.exp(-0.4 * z))).astype(float)

        def sigmoid(value):
            return 1 / (1 + np.exp(-value))

        if name == "logit_laplace":
            y = (rng.random(args.rows) < sigmoid(0.2 + 1.5 * t + 0.6 * z)).astype(float)
            truth = float(np.mean(sigmoid(1.7 + 0.6 * z) - sigmoid(0.2 + 0.6 * z)))
        else:
            y = 2 * t + z + rng.normal(scale=0.3, size=args.rows)
            truth = 2.0
        data = {"t": t, "y": y, "z": z}
        options = {
            "graph": [("z", "t"), ("z", "y"), ("t", "y")],
            "query": ant.AverageEffect("t", "y"),
            "refute": False,
            "inference": ant.Frequentist(bootstrap=0),
            "execution": ant.Execution(seed=args.seed, threads=args.threads),
        }
        if name == "crossfit_dml":
            from antecedent.estimators import DML

            options["estimator"] = DML(learner="ridge", folds=4)
        else:
            options["inference"] = ant.Bayesian(
                backend="hmc" if name == "gaussian_hmc" else "laplace",
                likelihood="gaussian" if name == "gaussian_hmc" else "logit",
                n_draws=args.draws,
            )
        prepared = ant.prepare(data, **options)
        spec.update(
            seed=args.seed,
            threads_requested=args.threads,
            thread_policy="explicit_request",
            rows=args.rows,
            variables=3,
            truth=truth,
            fixture_sha256=hashlib.sha256(b"".join(v.tobytes() for v in data.values())).hexdigest(),
            requested_draws=0 if name == "crossfit_dml" else args.draws,
            folds=4 if name == "crossfit_dml" else None,
        )

        def check(result):
            require(
                math.isfinite(result.effect) and abs(result.effect - truth) < 0.3,
                "effect truth failed",
            )
            require(
                not result.performance.cancelled and not result.performance.early_stopped,
                "incomplete execution",
            )
            if name == "crossfit_dml":
                require(result.estimate.crossfit_folds == 4, "crossfit effort changed")
                return {
                    "effect": result.effect,
                    "folds": 4,
                    "learner_provenance": list(result.estimate.learner_provenance or ()),
                }
            require(
                result.posterior is not None and result.posterior.n_draws >= args.draws,
                "posterior draws missing",
            )
            return {
                "effect": result.effect,
                "draws": result.posterior.n_draws,
                "backend": result.posterior.backend,
            }

        cases.extend(
            [
                Case(
                    name,
                    "cold",
                    lambda: require_success(ant.analyze(data, **options)),
                    check,
                    spec,
                ),
                Case(
                    name,
                    "prepared",
                    lambda: require_success(ant.estimate(prepared)),
                    check,
                    spec,
                ),
            ]
        )
    elif name == "dose_grid":
        from antecedent.dose_grid import dose_functional

        dose = np.linspace(0, 4, args.rows)
        outcome = 1 + 0.5 * dose + 0.25 * dose * dose
        grid = np.linspace(0.5, 3.5, args.grid_points).tolist()

        def call():
            return dose_functional(
                dose,
                outcome,
                grid=grid,
                functional="level",
                bandwidth=0.4,
                bandwidth_range=(0.1, 1.0),
                minimum_local_ess=10,
            )

        def check(result):
            require(len(result.levels) == len(grid), "grid points omitted")
            require(
                all(
                    abs(p.value - (1 + 0.5 * p.dose + 0.25 * p.dose * p.dose)) < 1e-5
                    for p in result.levels
                ),
                "quadratic response truth failed",
            )
            return {
                "grid_points": len(result.levels),
                "max_abs_influence_sum": result.max_abs_influence_sum,
            }

        cases.append(Case(name, "cold", call, check, dict(spec, rows=args.rows, grid=grid)))
    elif name in ("cyclic_worlds", "artifact_replay", "artifact_fresh_process"):
        d = args.dimensions
        equations = tuple(
            Equation(f"v{i}", 0.0, tuple(0.2 if j == (i - 1) % d else 0.0 for j in range(d)))
            for i in range(d)
        )
        covariance = tuple(float(i == j) for i in range(d) for j in range(d))
        model = EquilibriumModel(equations, covariance)
        worlds = [
            [],
            [{"variable": 0, "value": 1.0}],
            [{"variable": 0, "value": 1.0}, {"variable": d - 1, "value": -0.5}],
        ]
        noise = [[0.1 * math.sin(i * 0.31 + j) for j in range(d)] for i in range(args.units)]

        def make():
            return model.prepare(worlds, noise, factual=[0.0] * d)

        prepared = make()
        spec.update(
            variables=d,
            units=args.units,
            worlds=len(worlds),
            selection="unique contraction",
            factual=[0.0] * d,
        )

        def check(result):
            report = result.to_dict()
            require(len(report["worlds"]) == len(worlds), "worlds omitted")
            max_residual = 0.0
            for world, rows in zip(worlds, report["worlds"], strict=True):
                require(len(rows) == args.units, "units omitted")
                clamps = {a["variable"]: a["value"] for a in world}
                for disturbance, row in zip(noise, rows, strict=True):
                    require(
                        row["globally_unique"] and row["numerical_converged"],
                        "uncertified or failed world",
                    )
                    values = row["values"]
                    for j in range(d):
                        expected = clamps.get(
                            j, math.tanh(0.2 * values[(j - 1) % d]) + disturbance[j]
                        )
                        max_residual = max(max_residual, abs(values[j] - expected))
            require(max_residual < 1e-7, "independent mechanism residual failed")
            require(
                len(report["counterfactuals"]) == len(worlds),
                "counterfactual worlds omitted",
            )
            require(
                max(abs(v) for v in report["counterfactuals"][0]["values"]) < 1e-9,
                "factual coupling failed",
            )
            return {
                "worlds": len(worlds),
                "units": args.units,
                "max_residual": max_residual,
                "counterfactuals": len(worlds),
            }

        if name == "cyclic_worlds":
            cases.extend(
                [
                    Case(name, "cold", lambda: require_success(make().execute()), check, spec),
                    Case(
                        name, "prepared", lambda: require_success(prepared.execute()), check, spec
                    ),
                ]
            )
        else:
            result = require_success(prepared.execute())
            artifact = bytes(result.export())
            expected = result.identity
            spec["artifact_bytes"] = len(artifact)
            if name == "artifact_replay":
                cases.append(
                    Case(
                        name,
                        "load_and_recompute",
                        lambda: require_success(
                            consume_artifact(artifact, expected_identity=expected)
                        ),
                        check,
                        spec,
                    )
                )
            else:
                path = scratch / "equilibrium.ant"
                path.write_bytes(artifact)
                script = (
                    "import json,pathlib; from antecedent.cyclic import consume_artifact; "
                    "from antecedent.outcomes import require_success; "
                    f"r=require_success(consume_artifact(pathlib.Path({str(path)!r}).read_bytes(),expected_identity={expected!r})); "
                    "print(json.dumps({'identity':r.identity,'report':r.to_dict()}))"
                )

                def fresh():
                    return json.loads(
                        subprocess.check_output(
                            [sys.executable, "-c", script], cwd=scratch, text=True
                        )
                    )

                def check_fresh(value):
                    from types import SimpleNamespace

                    require(
                        value["identity"] == expected,
                        "fresh consumer identity mismatch",
                    )
                    observed = check(SimpleNamespace(to_dict=lambda: value["report"]))
                    return dict(observed, identity=expected)

                cases.append(Case(name, "startup_load_and_recompute", fresh, check_fresh, spec))
    elif name == "cyclic_fit":
        model = EquilibriumModel(
            (
                Equation("a", 0.0, (0.0, 0.4), "identity"),
                Equation("b", 0.0, (0.5, 0.0)),
            ),
            (1.0, 0.0, 0.0, 1.0),
        )
        xs = np.linspace(-2, 2, args.units).tolist()

        def rows(weight):
            return [
                {
                    "values": [x, math.tanh(weight * x)],
                    "clamps": [{"variable": 0, "value": x}],
                }
                for x in xs
            ]

        def make():
            return model.prepare(
                [[{"variable": 0, "value": 2.0}]],
                [[0.0, 0.0]],
                parameters=[{"kind": "weight", "equation": 1, "parent": 0}],
                rows=rows(0.7),
            )

        prepared = make()

        def check_fit(value, expected=1.4):
            require(value.to_dict()["fit"]["converged"], "fit did not converge")
            require(
                abs(value.to_dict()["worlds"][0][0]["values"][1] - math.tanh(expected)) < 1e-4,
                "fitted intervention truth failed",
            )
            return {
                "fit_converged": True,
                "training_rows": args.units,
                "intervention_response": value.to_dict()["worlds"][0][0]["values"][1],
                "expected_response": math.tanh(expected),
            }

        spec.update(
            variables=2,
            training_rows=args.units,
            parameters=1,
            known_interventions=True,
        )
        cases.extend(
            [
                Case(
                    name,
                    "cold_fit_and_query",
                    lambda: require_success(make().execute()),
                    check_fit,
                    spec,
                ),
                Case(
                    name,
                    "prepared_query",
                    lambda: require_success(prepared.execute()),
                    check_fit,
                    spec,
                ),
                Case(
                    name,
                    "refresh_fit_and_query",
                    lambda: require_success(ant.rebind(prepared, data=rows(0.6)).execute()),
                    lambda r: check_fit(r, 1.2),
                    spec,
                ),
            ]
        )
    elif name in ("cyclic_identify", "cyclic_discover"):
        graph = CyclicGraph(3, ((0, 1), (1, 2), (2, 1)))
        make = (
            (lambda: graph.identify([2], [0], io_scm=True, scc_product_density=True))
            if name == "cyclic_identify"
            else (lambda: graph.oracle_discovery(simple_scm=True, sigma_faithful=True))
        )
        prepared = make()

        def check(result):
            answer = result.to_dict()["answer"]["result"]
            if name == "cyclic_identify":
                require(
                    answer["status"] == "identified" and bool(answer["expression"]),
                    "identification failed",
                )
            else:
                require(answer["ci_source"] == "sigma_oracle", "discovery oracle changed")
                # Conditioning inside the {1,2} SCC cannot block the inducing
                # walks here. Every pair remains sigma-connected, yielding the
                # complete, unoriented equivalence class rather than direct edges.
                require(
                    sorted(answer["edges"])
                    == [
                        [0, 1, "circle", "circle"],
                        [0, 2, "circle", "circle"],
                        [1, 2, "circle", "circle"],
                    ]
                    and answer["conclusions"] == [],
                    "sigma equivalence-class truth failed",
                )
            return {
                "status": answer.get("status"),
                "ci_source": answer.get("ci_source"),
            }

        cases.extend(
            [
                Case(
                    name,
                    "cold",
                    lambda: require_success(make().execute()),
                    check,
                    dict(spec, variables=3),
                ),
                Case(
                    name,
                    "prepared",
                    lambda: require_success(prepared.execute()),
                    check,
                    dict(spec, variables=3),
                ),
            ]
        )
    elif name in ("ode", "ou", "constraints"):
        if name == "ode":
            model = OdeSystem(
                (
                    Equation("a", 0.0, (0.0, -1.0), "identity"),
                    Equation("b", 0.0, (1.0, 0.0), "identity"),
                ),
                (0.0, 0.0),
            )
            times = np.linspace(0, 1, args.grid_points).tolist()

            def make():
                return model.prepare([1.0, 0.0], times)

            def check(result):
                states = result.to_dict()["answer"]["result"]["states"]
                require(len(states) == len(times), "trajectory points omitted")
                error = max(
                    abs(a - b)
                    for t, state in zip(times, states, strict=True)
                    for a, b in zip(state, [math.cos(t), math.sin(t)], strict=True)
                )
                require(error < 1e-6, "ODE analytic truth failed")
                return {"time_points": len(times), "max_trajectory_error": error}

            spec.update(variables=2, time_points=len(times), horizon=1.0)
        elif name == "ou":
            d = args.dimensions
            model = OrnsteinUhlenbeck(
                # A=-2I+K with skew-symmetric, dense K: A+A^T=-4I.
                # With diffusion 4I the stationary covariance is exactly I,
                # while the solver still handles a coupled linear system.
                tuple(
                    -2.0 if i == j else 0.15 * math.sin((i + 1) * (j + 1)) * (i - j) / d
                    for i in range(d)
                    for j in range(d)
                ),
                tuple(3.0 for _ in range(d)),
                tuple(4.0 if i == j else 0.0 for i in range(d) for j in range(d)),
            )
            make = model.stationary

            def check(result):
                values = result.to_dict()["answer"]["result"]
                require(len(values) == d * d, "covariance entries omitted")
                require(
                    max(abs(v - float(i // d == i % d)) for i, v in enumerate(values)) < 1e-8,
                    "OU Lyapunov truth failed",
                )
                return {"covariance_entries": len(values)}

            spec.update(
                variables=d,
                law="stationary stochastic dynamics",
                drift="negative identity plus dense skew-symmetric coupling",
            )
        else:
            model = PolynomialSystem(
                ("x",),
                (
                    {
                        "id": "cubic",
                        "terms": [
                            {"coefficient": 1.0, "powers": [3]},
                            {"coefficient": -1.0, "powers": [1]},
                        ],
                    },
                ),
                ({"lower": -2.0, "upper": 2.0},),
            )
            make = model.prepare

            def check(result):
                answer = result.to_dict()["answer"]["result"]
                require(
                    len(answer["roots"]) == 3 and answer["unresolved"] == [],
                    "bounded roots incomplete",
                )
                roots = sorted(answer["roots"], key=lambda root: root[0]["lower"])
                for root, truth in zip(roots, (-1.0, 0.0, 1.0), strict=True):
                    interval = root[0]
                    require(
                        interval["lower"] <= truth <= interval["upper"]
                        and interval["upper"] - interval["lower"] < 1e-8,
                        "root enclosure truth failed",
                    )
                return {"roots": 3, "unresolved_regions": 0}

            spec.update(variables=1, degree=3, domain=[-2.0, 2.0])
        prepared = make()
        cases.extend(
            [
                Case(name, "cold", lambda: require_success(make().execute()), check, spec),
                Case(name, "prepared", lambda: require_success(prepared.execute()), check, spec),
            ]
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=NAMES, action="append")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--measure", action="store_true")
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--units", type=int, default=16)
    parser.add_argument("--dimensions", type=int, default=6)
    parser.add_argument("--grid-points", type=int, default=33)
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not (args.validate or args.measure):
        print("\n".join(NAMES))
        return
    if (
        args.rows < 128
        or min(args.units, args.dimensions, args.grid_points) < 2
        or min(args.draws, args.threads, args.repeats) < 1
    ):
        parser.error("rows >=128; units, dimensions, grid-points >=2; other budgets positive")
    if args.measure and args.output is None:
        parser.error("--measure requires --output to retain every sample")
    import antecedent as ant
    from antecedent import _native

    selected = args.case or list(NAMES)
    cells = []
    with tempfile.TemporaryDirectory(prefix="antecedent-capabilities-") as folder:
        for name in selected:
            for case in build_cases(name, args, Path(folder)):
                samples = []
                if args.measure:
                    case.check(case.call())
                for repeat in range(args.repeats if args.measure else 1):
                    if args.measure:
                        started = time.perf_counter_ns()
                    value = case.call()
                    elapsed = time.perf_counter_ns() - started if args.measure else None
                    observed = case.check(value)
                    samples.append(
                        {
                            "repeat": repeat,
                            "observed": observed,
                            **({"elapsed_ns": elapsed} if args.measure else {}),
                        }
                    )
                cell = {
                    "capability": case.capability,
                    "mode": case.mode,
                    "specification": case.specification,
                    "samples": samples,
                }
                if args.measure:
                    cell["median_ns"] = statistics.median(s["elapsed_ns"] for s in samples)
                cells.append(cell)
    report = {
        "scope": "diagnostic_capability_matrix_not_full_N20_or_calibration",
        "timed": args.measure,
        "warmups": 1 if args.measure else 0,
        "package": ant.__file__,
        "native": _native.__file__,
        "native_sha256": hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "cells": cells,
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(json.dumps({"timed": args.measure, "cells": len(cells), "capabilities": selected}))


if __name__ == "__main__":
    main()
