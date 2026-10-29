# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the refuter-state fixture of anomaly attribution over a structure class.

A refuter that refits the attribution on random subsets of the rows can only see a violation when a
single row decides a fitted mechanism. The leverage of row i in the regression of a node on its
parents is h_i = x_i' (X'X)^-1 x_i with x_i = (1, parents of row i). This script builds each table
from the stated formulas, computes every leverage and the least-squares coefficients with and
without the planted row in exact rational arithmetic on the float inputs, and asserts what the
planted row does: it has leverage above the stated bar, no clean row exceeds the clean bar, and the
coefficient it decides moves by more than the stated amount when it is dropped. For a mechanism
that has no parent (an intercept-only root, as in the identified completions of a PAG) the leverage
of every row is 1/n and the planted violation is instead one row far from the bulk: its robust
z-score (|y - median| / (1.4826 MAD)) is above the stated bar, and no clean row's is.

Tables (each with a clean and a planted variant):

* `xy`: x_i = sin(1.37 i) + 0.5 cos(0.91 i), y_i = 1 + 2 x_i + 0.3 sin(2.17 i), 24 rows; planted
  row 7 moves x to 40 and adds 60 to y.
* `abt`: a_i = sin(1.37 i) + 0.5 cos(0.91 i), b_i = sin(0.77 i + 0.4) - 0.3 cos(1.9 i),
  t_i = 1 + a_i + 2 b_i + 0.3 sin(2.17 i), 24 rows; planted row 7 moves a to 40 and b to 30 and adds
  150 to t.
* `series_xy`: x_t = sin(0.37 t) + 0.5 cos(1.1 t), y_t = 1 + 2 x_{t-1} + 0.3 sin(2.17 t) (x_{-1} = 0),
  20 periods; the planted series moves x_9 to 40 and adds 60 to y_10.
* `series_abt`: a_t = sin(0.37 t) + 0.5 cos(1.1 t), b_t = sin(0.53 t + 0.4) - 0.3 cos(1.9 t),
  t_t = 1 + a_{t-1} + 2 b_{t-1} + 0.3 sin(2.17 t) (a_{-1} = b_{-1} = 0), 24 periods; the planted
  series moves a_11 to 40 and adds 300 to t_12.

The lag-aligned design of a series reads a lagged column at row max(t - 1, 0), the route's stated
alignment.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/class_attribution_refuter_states/reference.py --check
    uv run --frozen --script conformance/estimate/class_attribution_refuter_states/reference.py --self-test
    uv run --frozen --script conformance/estimate/class_attribution_refuter_states/reference.py --write

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


def sx(i, f1, f2):
    return math.sin(f1 * i) + 0.5 * math.cos(f2 * i)


def table_xy(planted):
    n = 24
    x = [sx(i, 1.37, 0.91) for i in range(n)]
    if planted:
        x[7] = 40.0
    y = [1.0 + 2.0 * x[i] + 0.3 * math.sin(2.17 * i) for i in range(n)]
    if planted:
        y[7] += 60.0
    return {"x": x, "y": y}


def table_abt(planted):
    n = 24
    a = [sx(i, 1.37, 0.91) for i in range(n)]
    b = [math.sin(0.77 * i + 0.4) - 0.3 * math.cos(1.9 * i) for i in range(n)]
    if planted:
        a[7], b[7] = 40.0, 30.0
    t = [1.0 + a[i] + 2.0 * b[i] + 0.3 * math.sin(2.17 * i) for i in range(n)]
    if planted:
        t[7] += 150.0
    return {"a": a, "b": b, "t": t}


def table_series_xy(planted):
    n = 20
    x = [sx(t, 0.37, 1.1) for t in range(n)]
    if planted:
        x[9] = 40.0
    y = []
    for t in range(n):
        lag = x[t - 1] if t > 0 else 0.0
        v = 1.0 + 2.0 * lag + 0.3 * math.sin(2.17 * t)
        y.append(v + 60.0 if planted and t == 10 else v)
    return {"x": x, "y": y}


def table_series_abt(planted):
    n = 24
    a = [sx(t, 0.37, 1.1) for t in range(n)]
    b = [math.sin(0.53 * t + 0.4) - 0.3 * math.cos(1.9 * t) for t in range(n)]
    if planted:
        a[11] = 40.0
    t_ = []
    for t in range(n):
        la, lb = (a[t - 1], b[t - 1]) if t > 0 else (0.0, 0.0)
        v = 1.0 + la + 2.0 * lb + 0.3 * math.sin(2.17 * t)
        t_.append(v + 300.0 if planted and t == 12 else v)
    return {"a": a, "b": b, "t": t_}


# name: (builder, regressors of the checked mechanism, response, lagged regressors, planted row)
DESIGNS = {
    "xy": (table_xy, ["x"], "y", [], 7),
    "abt": (table_abt, ["a", "b"], "t", [], 7),
    "series_xy": (table_series_xy, ["x"], "y", ["x"], 10),
    "series_abt": (table_series_abt, ["a", "b"], "t", ["a", "b"], 12),
}


def solve(a, b):
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = next(r for r in range(col, n) if m[r][col] != 0)
        m[col], m[piv] = m[piv], m[col]
        p = m[col][col]
        m[col] = [v / p for v in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col]
                m[r] = [v - f * w for v, w in zip(m[r], m[col])]
    return [m[i][n] for i in range(n)]


def design_rows(columns, regressors, lagged):
    n = len(next(iter(columns.values())))
    rows = []
    for i in range(n):
        row = [F(1)]
        for name in regressors:
            index = max(i - 1, 0) if name in lagged else i
            row.append(F(columns[name][index]))
        rows.append(row)
    return rows


def gram(rows):
    p = len(rows[0])
    return [[sum(r[i] * r[j] for r in rows) for j in range(p)] for i in range(p)]


def leverages(rows):
    g = gram(rows)
    p = len(g)
    out = []
    for r in rows:
        sol = solve([row[:] for row in g], r)
        out.append(sum(a * b for a, b in zip(r, sol)))
    return out


def coefficients(rows, y):
    g = gram(rows)
    xty = [sum(r[j] * yv for r, yv in zip(rows, y)) for j in range(len(g))]
    return solve(g, xty)


def median(values):
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def robust_z(values):
    vals = [F(v) for v in values]
    c = median(vals)
    mad = median([abs(v - c) for v in vals])
    scale = F(1.482602218505602) * mad
    return [float(abs(v - c) / scale) for v in vals]


def derive(name):
    build, regressors, response, lagged, planted_row = DESIGNS[name]
    clean, planted = build(False), build(True)
    out = {"planted_row": planted_row}
    c_rows = design_rows(clean, regressors, lagged)
    p_rows = design_rows(planted, regressors, lagged)
    c_lev, p_lev = leverages(c_rows), leverages(p_rows)
    out["clean_max_leverage"] = float(max(c_lev))
    out["planted_row_leverage"] = float(p_lev[planted_row])
    out["planted_is_max_leverage_row"] = max(range(len(p_lev)), key=lambda i: p_lev[i]) == planted_row
    keep = [i for i in range(len(p_rows)) if i != planted_row]
    with_row = coefficients(p_rows, [F(v) for v in planted[response]])
    without = coefficients([p_rows[i] for i in keep], [F(planted[response][i]) for i in keep])
    clean_fit = coefficients(c_rows, [F(v) for v in clean[response]])
    out["coefficient_shift"] = float(max(abs(a - b) for a, b in zip(with_row[1:], without[1:])))
    out["clean_coefficients"] = [float(v) for v in clean_fit]
    out["coefficients_without_planted_row"] = [float(v) for v in without]
    out["clean_response_max_robust_z"] = float(max(robust_z(clean[response])))
    out["planted_response_max_robust_z"] = float(max(robust_z(planted[response])))
    out["planted_response_robust_z_row"] = max(range(len(planted[response])), key=lambda i: robust_z(planted[response])[i])
    return {"clean": clean, "planted": planted, "derived": out}


def build_document(old_oracle=None):
    document = {
        "fixture_id": "class_attribution_refuter_states",
        "tables": {name: derive(name) for name in DESIGNS},
        "expected_verdicts": {
            "clean_all_refuters_pass": True,
            "planted_subset_stability_passes": False,
        },
    }
    if old_oracle is not None:
        document["oracle"] = old_oracle
    return document


def check(document):
    problems = []
    for name, table in document["tables"].items():
        derived = derive(name)
        for key in ("clean", "planted"):
            for column, values in derived[key].items():
                if table[key][column] != values:
                    problems.append(f"{name}.{key}.{column}: the stored table differs from the stated formula")
        got, want = derived["derived"], table["derived"]
        for key, value in got.items():
            if isinstance(value, (int, bool)) or isinstance(value, list):
                same = (want[key] == value) if not isinstance(value, list) else all(
                    math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(want[key], value)
                )
            else:
                same = math.isclose(want[key], value, rel_tol=0.0, abs_tol=1e-12)
            if not same:
                problems.append(f"{name}.{key}: derived {value!r} != fixture {want[key]!r}")
        bars = table["bars"]
        d = got
        if not d["clean_max_leverage"] < bars["clean_max_leverage_below"]:
            problems.append(f"{name}: clean max leverage {d['clean_max_leverage']} is not below the bar")
        if len(DESIGNS[name][1]) > 0 and not d["planted_row_leverage"] > bars["planted_row_leverage_above"]:
            problems.append(f"{name}: planted row leverage {d['planted_row_leverage']} is not above the bar")
        if not d["planted_is_max_leverage_row"]:
            problems.append(f"{name}: the planted row is not the row of largest leverage")
        if not d["coefficient_shift"] > bars["coefficient_shift_above"]:
            problems.append(f"{name}: dropping the planted row moves a coefficient by only {d['coefficient_shift']}")
        if not d["planted_response_max_robust_z"] > bars["planted_response_robust_z_above"]:
            problems.append(f"{name}: planted response robust z {d['planted_response_max_robust_z']} is not above the bar")
        if not d["clean_response_max_robust_z"] < bars["clean_response_robust_z_below"]:
            problems.append(f"{name}: clean response robust z {d['clean_response_max_robust_z']} is not below the bar")
        if d["planted_response_robust_z_row"] != DESIGNS[name][4]:
            problems.append(f"{name}: the planted response is not the response of largest robust z")
    if document["expected_verdicts"] != {
        "clean_all_refuters_pass": True,
        "planted_subset_stability_passes": False,
    }:
        problems.append("the declared verdicts are not clean passes / planted fails")
    return problems


BARS = {
    "xy": {"clean_max_leverage_below": 0.3, "planted_row_leverage_above": 0.9, "coefficient_shift_above": 1.0, "planted_response_robust_z_above": 20.0, "clean_response_robust_z_below": 4.0},
    "abt": {"clean_max_leverage_below": 0.35, "planted_row_leverage_above": 0.9, "coefficient_shift_above": 2.0, "planted_response_robust_z_above": 20.0, "clean_response_robust_z_below": 4.0},
    "series_xy": {"clean_max_leverage_below": 0.4, "planted_row_leverage_above": 0.8, "coefficient_shift_above": 1.0, "planted_response_robust_z_above": 20.0, "clean_response_robust_z_below": 6.0},
    "series_abt": {"clean_max_leverage_below": 0.4, "planted_row_leverage_above": 0.9, "coefficient_shift_above": 5.0, "planted_response_robust_z_above": 20.0, "clean_response_robust_z_below": 6.0},
}


MUTATIONS = [
    ("planted leverage bar above its leverage", lambda d: d["tables"]["xy"]["bars"].__setitem__("planted_row_leverage_above", 0.9999)),
    ("clean leverage bar too low", lambda d: d["tables"]["abt"]["bars"].__setitem__("clean_max_leverage_below", 0.05)),
    ("coefficient shift bar too high", lambda d: d["tables"]["series_abt"]["bars"].__setitem__("coefficient_shift_above", 500.0)),
    ("planted value moved off the extreme", lambda d: d["tables"]["xy"]["planted"]["x"].__setitem__(7, 0.2)),
    ("robust z bar too high", lambda d: d["tables"]["xy"]["bars"].__setitem__("planted_response_robust_z_above", 1.0e6)),
    ("verdicts swapped", lambda d: d["expected_verdicts"].__setitem__("clean_all_refuters_pass", False)),
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
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        old = None
        if FIXTURE.exists():
            old = json.loads(FIXTURE.read_text(encoding="utf-8")).get("oracle")
        document = build_document(old)
        for name, table in document["tables"].items():
            table["bars"] = BARS[name]
        FIXTURE.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
        print("written")
        return 0
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    problems = self_test(document) if args.self_test else check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
