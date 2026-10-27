#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Compare public transport preparation with typed and legacy JSON inputs.

Uses the independently checked intervention fixtures in the repository tests.
No timings are collected without --measure and a raw output destination.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import resource
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("trial", "continuous", "smoothed"), required=True)
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--columns", type=int, default=1)
    parser.add_argument("--route", choices=("default", "legacy"), default="default")
    parser.add_argument("--output-route", choices=("default", "legacy"), default="default")
    parser.add_argument(
        "--mode", choices=("prepare", "cold", "prepared", "refresh"), default="prepare"
    )
    parser.add_argument("--strided", action="store_true")
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.rows, args.columns, args.repeats) < 1:
        parser.error("sizes and repeats must be positive")
    if args.columns != 1:
        parser.error("transport fixtures currently have one certified baseline covariate")
    if args.measure and args.output is None:
        parser.error("measurement requires --output")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "tests"))
    import numpy as np
    from antecedent import _native
    from antecedent.transport import advanced

    n_trial = args.rows * 3 // 5
    n_target = args.rows - n_trial
    if args.case == "smoothed":
        from test_smoothed_dose_transport import fixture, options

        query, data = fixture(n_trial=n_trial, n_target=n_target)

        def prepare(value):
            return advanced.prepare_smoothed_dose(query, value, options=options(), seed=5)

    else:
        from test_learned_continuous_transport import fixture, options

        query, data = fixture(n_trial=n_trial, n_target=n_target)
        if args.case == "continuous":

            def prepare(value):
                return advanced.prepare_learned_continuous(query, value, options=options(), seed=5)

        else:
            from antecedent import transport as tr
            from antecedent.learners import Linear

            def prepare(value):
                return advanced.prepare(
                    query,
                    value,
                    provider=tr.TrialAipw(outcome=Linear(), folds=3),
                    inference=tr.TransportInference(0),
                )

    typed_available = hasattr(data, "_buffers")
    if args.route == "legacy":

        class Legacy(type(data)):
            def __getattribute__(self, name):
                if name == "_buffers":
                    raise AttributeError(name)
                return super().__getattribute__(name)

        data = Legacy(
            **{field.name: getattr(data, field.name) for field in dataclasses.fields(data)}
        )
    if args.strided:

        def stride(values):
            array = np.asarray(values)
            if array.dtype == np.dtype(bool):
                return [bool(value) for value in array]
            return np.repeat(array, 2)[::2]

        data = dataclasses.replace(
            data,
            covariates={name: stride(values) for name, values in data.covariates.items()},
            **{
                field.name: stride(getattr(data, field.name))
                for field in dataclasses.fields(data)
                if field.name not in ("covariates", "sampling")
            },
        )
    if args.output_route == "legacy" and args.case != "trial":
        parser.error("only trial transport has a bulk-output compatibility payload")

    def estimate(study):
        if args.output_route == "legacy":
            from antecedent.transport._impl.trial import _learned_trial

            return _learned_trial(study._native, study._native.estimate())
        return study.estimate()

    def refresh(study):
        if args.output_route == "legacy":
            study._native.refresh(data)
            return study
        import antecedent as ant

        return ant.rebind(study, data=data)

    prepared = prepare(data)
    expected = estimate(prepared)

    def exported(study, result):
        return study.export() if hasattr(study, "export") else result.export()

    expected_bytes = exported(prepared, expected)
    if args.mode == "prepare":

        def call():
            return prepare(data)

        def check(value):
            return exported(value, estimate(value))

    else:
        if args.mode == "refresh":

            def call():
                branch = refresh(prepared)
                return branch, estimate(branch)
        elif args.mode == "cold":

            def call():
                study = prepare(data)
                return study, estimate(study)
        else:

            def call():
                return prepared, estimate(prepared)

        def check(value):
            return exported(*value)

    assert check(call()) == expected_bytes
    samples = []
    for repeat in range(args.repeats if args.measure else 1):
        started = time.perf_counter_ns() if args.measure else None
        result = call()
        elapsed = time.perf_counter_ns() - started if args.measure else None
        actual = check(result)
        assert actual == expected_bytes
        samples.append(
            {
                "repeat": repeat,
                "artifact_sha256": hashlib.sha256(actual).hexdigest(),
                **({"elapsed_ns": elapsed} if args.measure else {}),
            }
        )
    report = {
        "timed": args.measure,
        "specification": {k: v for k, v in vars(args).items() if k != "output"},
        "route_used": "typed" if typed_available and args.route == "default" else "json",
        "native_sha256": hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "samples": samples,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"case": args.case, "timed": args.measure, "route": report["route_used"]}))


if __name__ == "__main__":
    main()
