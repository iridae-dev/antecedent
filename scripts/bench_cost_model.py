#!/usr/bin/env python3
"""Micro-benchmark behind ``estimate_cost`` seconds: fit a monotone cost model, write it down.

    python3 scripts/bench_cost_model.py            # writes parity/cost_model.toml
    python3 scripts/bench_cost_model.py --out /tmp/cost_model.toml --repeats 5

Run it once on a quiet machine, on the python package built from this tree (for example
``maturin develop --release`` in ``python/``), then commit the generated
``parity/cost_model.toml``. Nothing else produces that file, and ``estimate_cost`` reports
seconds only when it exists and declares its machine.

What is timed. For each estimator (``linear.adjustment.ate`` and ``aipw``) and each point of a
small ``(rows, design columns)`` grid, ``antecedent.analyze`` runs end to end (prepare, the
nuisance fits, the cross-fit scores) once with no bootstrap and once with ``--bootstrap``
replicates, ``--repeats`` times each; the fastest repeat is kept. The number of nuisance fits
those runs performed is the count ``estimate_cost`` itself reports for the same plan
(``nuisance_fits_upper_bound``), so one sample is

    seconds per counted fit = best wall time / nuisance_fits_upper_bound

and the cross-fit and bootstrap steps are timed through the fits they add.

The model, per estimator, is ``seconds per counted fit = a + b * rows * columns**2`` with
``a, b >= 0`` (an ordinary least squares line in ``x = rows * columns**2``, then clamped to
non-negative so it is monotone in rows and columns). ``estimate_cost`` multiplies it by the
fit count, so predictions are monotone in folds, replicates, rows and columns. It is a planning
hint for this machine only: the file records the benchmark name, the grid and the machine
descriptor, and the label ``estimate_cost`` attaches says so.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import re
import sys
import time
from pathlib import Path

BENCHMARK_NAME = "scripts/bench_cost_model.py"
ESTIMATORS = ("linear.adjustment.ate", "aipw")
DEFAULT_ROWS = (400, 800, 1600)
DEFAULT_COLUMNS = (4, 8, 16)
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "parity" / "cost_model.toml"


def _frame(rows: int, covariates: int, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((rows, max(covariates, 1)))
    propensity = 1.0 / (1.0 + np.exp(-0.3 * z[:, 0]))
    t = (rng.random(rows) < propensity).astype(np.float64)
    y = t + 0.2 * z.sum(axis=1) + 0.5 * rng.standard_normal(rows)
    data = {"t": t, "y": y}
    names = [f"z{j}" for j in range(covariates)]
    for j, name in enumerate(names):
        data[name] = z[:, j]
    edges = [("t", "y")] + [(name, side) for name in names for side in ("t", "y")]
    return data, edges


def _plan(antecedent, data, edges, estimator: str, bootstrap: int, seed: int):
    return dict(
        graph=edges,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        estimator=estimator,
        refute=False,
        bootstrap=bootstrap,
        seed=seed,
    )


def sample_grid(antecedent, rows_grid, columns_grid, bootstrap: int, repeats: int, seed: int):
    """Per estimator, ``(rows, columns, seconds per counted fit)`` samples."""
    samples = {estimator: [] for estimator in ESTIMATORS}
    for estimator in ESTIMATORS:
        for rows in rows_grid:
            for columns in columns_grid:
                data, edges = _frame(rows, columns - 1, seed)
                for replicates in (0, bootstrap):
                    kwargs = _plan(antecedent, data, edges, estimator, replicates, seed)
                    prepared = antecedent.prepare(data, **kwargs)
                    cost = prepared.estimate_cost()
                    fits = cost.nuisance_fits_upper_bound
                    if not fits or cost.design_columns != columns:
                        raise SystemExit(
                            f"{estimator}: plan reports {fits} fits and "
                            f"{cost.design_columns} design columns, expected {columns}"
                        )
                    best = math.inf
                    for _ in range(repeats):
                        start = time.perf_counter()
                        antecedent.analyze(data, **kwargs)
                        best = min(best, time.perf_counter() - start)
                    samples[estimator].append((rows, columns, best / fits))
                    print(
                        f"{estimator:>24} rows={rows:<5} cols={columns:<3} "
                        f"boot={replicates:<3} fits={fits:<6} per-fit={best / fits:.3e}s",
                        flush=True,
                    )
    return samples


def fit_coefficients(samples) -> tuple[float, float]:
    """Least squares ``per_fit = a + b * rows * columns**2``, clamped to ``a, b >= 0``."""
    xs = [rows * columns**2 for rows, columns, _ in samples]
    ys = [per_fit for _, _, per_fit in samples]
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    b = sxy / sxx if sxx > 0 else 0.0
    a = mean_y - b * mean_x
    if b < 0:
        a, b = mean_y, 0.0
    elif a < 0:
        a, b = 0.0, sum(x * y for x, y in zip(xs, ys, strict=True)) / sum(x * x for x in xs)
    return a, b


def _ascii(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+ ():/-]", "-", text)


def format_toml(coefficients, rows_grid, columns_grid, bootstrap, repeats, samples) -> str:
    """The strict TOML subset ``CostModel::parse`` reads (see ``cost.rs``)."""
    lines = [
        "# Written by scripts/bench_cost_model.py; regenerate rather than edit.",
        "# estimate_cost reports seconds from this file only because it declares its machine.",
        f'benchmark = "{BENCHMARK_NAME}"',
        'model = "seconds per counted fit = a + b * rows * columns^2"',
        "",
        "[machine]",
        f'platform = "{_ascii(platform.platform())}"',
        f"cpu_count = {os.cpu_count() or 1}",
        f'python = "{_ascii(platform.python_version())}"',
        "",
        "[grid]",
        f"rows = [{', '.join(str(r) for r in rows_grid)}]",
        f"columns = [{', '.join(str(c) for c in columns_grid)}]",
        f"bootstrap_replicates = {bootstrap}",
        f"repeats = {repeats}",
    ]
    for estimator, (a, b) in coefficients.items():
        if not (math.isfinite(a) and math.isfinite(b) and a >= 0 and b >= 0):
            raise SystemExit(f"{estimator}: non-monotone or non-finite fit a={a} b={b}")
        lines += [
            "",
            f'[coefficients."{estimator}"]',
            f"a = {a!r}",
            f"b = {b!r}",
            f"samples = {len(samples[estimator])}",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rows", type=int, nargs="+", default=list(DEFAULT_ROWS))
    parser.add_argument("--columns", type=int, nargs="+", default=list(DEFAULT_COLUMNS))
    parser.add_argument("--bootstrap", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    if min(args.columns) < 2 or min(args.rows) < 50 or args.repeats < 1 or args.bootstrap < 1:
        parser.error("need columns >= 2 (intercept plus a covariate), rows >= 50, repeats >= 1")

    import antecedent

    samples = sample_grid(
        antecedent, args.rows, args.columns, args.bootstrap, args.repeats, args.seed
    )
    coefficients = {estimator: fit_coefficients(rows) for estimator, rows in samples.items()}
    text = format_toml(coefficients, args.rows, args.columns, args.bootstrap, args.repeats, samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
