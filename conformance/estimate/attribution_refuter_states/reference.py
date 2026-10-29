# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the attribution refuter-state fixture.

The table has 24 rows: x_i = sin(1.37 i) + 0.5 cos(0.91 i) and y_i = 1 + 2 x_i + 0.3 sin(2.17 i)
(the clean table). The planted table moves row 7 to x_7 = 40 and y_7 = 1 + 2 x_7 + 0.3 sin(2.17 * 7) + 60:
one row that sits far from the others on the design and far above the structural line.

A refuter that refits the attribution on random subsets of the rows can only see a violation when a
single row decides the fit. The leverage of row i in the regression of y on (1, x) is
h_i = 1/n + (x_i - mean x)^2 / S_xx. This script computes every h_i, and the least-squares slope with and
without the planted row, in exact rational arithmetic on the float inputs. The clean table has no row
above leverage 0.3; the planted row has leverage above 0.9 and carries the fitted slope with it, so a
subset that drops it (one in five) refits a different mechanism. That is the planted violation the
subset-stability refuter must detect, and the clean table is the case it must pass. The same derivation holds for the series: the lag-aligned design
of a 20-period series with one period at x = 40 (the outcome of the next period raised by 60) has one row of
leverage above 0.9, and the clean series has none above 0.3.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/attribution_refuter_states/reference.py --check
    uv run --frozen --script conformance/estimate/attribution_refuter_states/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "expected.json"


def table(planted: bool, n: int, row: int, lift: float, planted_x: float):
    x = [math.sin(1.37 * i) + 0.5 * math.cos(0.91 * i) for i in range(n)]
    if planted:
        x[row] = planted_x
    y = [1.0 + 2.0 * x[i] + 0.3 * math.sin(2.17 * i) for i in range(n)]
    if planted:
        y[row] += lift
    return x, y


def slope(x, y):
    n = len(x)
    xs = [F(v) for v in x]
    ys = [F(v) for v in y]
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((a - mx) ** 2 for a in xs)
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sxx


def leverages(x):
    n = len(x)
    xs = [F(v) for v in x]
    mx = sum(xs) / n
    sxx = sum((a - mx) ** 2 for a in xs)
    return [F(1, n) + (a - mx) ** 2 / sxx for a in xs]


def series_design(planted: bool, d):
    """Lag-aligned design of the series: the outcome on its parent one period back (row 0 repeats
    the first period, as the lag-aligned table does)."""
    n, at = d["rows"], d["planted_position"]
    x = [math.sin(0.37 * t) + 0.5 * math.cos(1.1 * t) for t in range(n)]
    if planted:
        x[at] = d["planted_x"]
    y = []
    for t in range(n):
        lag = x[t - 1] if t > 0 else 0.0
        v = 1.0 + 2.0 * lag + 0.3 * math.sin(2.17 * t)
        if planted and t == at + 1:
            v += d["lift"]
        y.append(v)
    lagged = [x[max(t - 1, 0)] for t in range(n)]
    return lagged, y


def check_series(d):
    problems = []
    cx, _ = series_design(False, d)
    px, py = series_design(True, d)
    at = d["planted_position"]
    clean_max = max(leverages(cx))
    planted = leverages(px)
    if not clean_max < F(d["clean_max_leverage_below"]):
        problems.append(f"series: clean max leverage {float(clean_max)} is not below {d['clean_max_leverage_below']}")
    if not planted[at + 1] > F(d["row_leverage_above"]):
        problems.append(f"series: planted row leverage {float(planted[at + 1])} is not above {d['row_leverage_above']}")
    if max(range(len(px)), key=lambda i: planted[i]) != at + 1:
        problems.append("series: the planted period is not the period of largest leverage")
    return problems


def check(document):
    problems = []
    d = document["dgp"]
    n, row, lift, planted_x = d["rows"], d["planted_row"], d["lift"], d["planted_x"]
    cx, cy = table(False, n, row, lift, planted_x)
    px, py = table(True, n, row, lift, planted_x)
    clean_max = max(leverages(cx))
    planted = leverages(px)
    if not clean_max < F(document["clean"]["max_leverage_below"]):
        problems.append(f"clean max leverage {float(clean_max)} is not below {document['clean']['max_leverage_below']}")
    if not planted[row] > F(document["planted"]["row_leverage_above"]):
        problems.append(f"planted row leverage {float(planted[row])} is not above {document['planted']['row_leverage_above']}")
    if max(range(n), key=lambda i: planted[i]) != row:
        problems.append("the planted row is not the row of largest leverage")
    keep = [i for i in range(n) if i != row]
    with_row = slope(px, py)
    without_row = slope([px[i] for i in keep], [py[i] for i in keep])
    if not abs(float(without_row) - 2.0) < document["planted"]["slope_without_row_within"]:
        problems.append(f"slope without the planted row {float(without_row)} is not near the structural 2")
    if not abs(float(with_row) - float(without_row)) > document["planted"]["slope_shift_above"]:
        problems.append(f"the planted row moves the slope by only {abs(float(with_row) - float(without_row))}")
    if not abs(float(slope(cx, cy)) - 2.0) < document["clean"]["slope_within"]:
        problems.append("the clean slope is not near the structural 2")
    problems += check_series(document["series"])
    expected = document["expected_verdicts"]
    if expected != {"clean_subset_stability_passes": True, "planted_subset_stability_passes": False}:
        problems.append("the declared verdicts are not clean passes / planted fails")
    return problems


MUTATIONS = [
    ("planted leverage bar above its leverage", lambda d: d["planted"].__setitem__("row_leverage_above", 0.9999)),
    ("clean leverage bar too low", lambda d: d["clean"].__setitem__("max_leverage_below", 0.05)),
    ("planted row moved off the extreme", lambda d: d["dgp"].__setitem__("planted_x", 0.2)),
    ("series leverage bar above its leverage", lambda d: d["series"].__setitem__("row_leverage_above", 0.99999)),
    ("series planted position moved off the extreme", lambda d: d["series"].__setitem__("planted_x", 0.1)),
    ("verdicts swapped", lambda d: d["expected_verdicts"].__setitem__("clean_subset_stability_passes", False)),
]


def self_test(document):
    failures = []
    if check(document):
        failures.append("the unmodified fixture is rejected")
    for label, mutate in MUTATIONS:
        wrong = copy.deepcopy(document)
        mutate(wrong)
        if not check(wrong):
            failures.append(f"accepted: {label}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    problems = self_test(document) if args.self_test else check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
