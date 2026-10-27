# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["dit==2.3", "numpy", "scipy"]
# ///
"""Retained black-box regeneration of the expected-information-gain reference values.

Reads only the fixture's declared posterior weights, feature labels and observation
reliabilities.  The reference value of each case is I(G;Y) of the joint law
p(g) p(y|g) with p(y|g) = r*1[y = feature(g)] + (1-r)/K, computed by the public
``dit.shannon.mutual_information`` API (natural log) and cross-checked by the direct
definition with ``scipy.stats.entropy``.  No Antecedent code is imported.

    uv run --frozen --script conformance/design/expected_information_gain/reference.py --check
    uv run --frozen --script conformance/design/expected_information_gain/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import sys
from pathlib import Path

import dit
import numpy as np
from dit.shannon import mutual_information
from scipy.stats import entropy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from reference_compare import values_close  # noqa: E402

ATOL = 1e-12


def joint(weights, features, reliability):
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    cats = sorted(set(features))
    k = len(cats)
    rows = {}
    for g, (pg, f) in enumerate(zip(w, features)):
        for y in cats:
            p = reliability * (1.0 if y == f else 0.0) + (1.0 - reliability) / k
            rows[(g, y)] = pg * p
    return w, cats, rows


def gain_dit(weights, features, reliability) -> float:
    _, cats, rows = joint(weights, features, reliability)
    outcomes = [f"{g}{y}" for (g, y) in rows]
    # Single-character symbols per variable keep dit's string outcomes unambiguous.
    d = dit.Distribution(outcomes, list(rows.values()))
    d.set_base("e")
    return float(mutual_information(d, [0], [1]))


def gain_direct(weights, features, reliability) -> float:
    w, cats, rows = joint(weights, features, reliability)
    h_g = entropy(w)
    total = 0.0
    for y in cats:
        py = sum(rows[(g, y)] for g in range(len(w)))
        post = np.array([rows[(g, y)] for g in range(len(w))]) / py
        total += py * (h_g - entropy(post))
    return float(total)


def outputs(document: dict) -> list[dict]:
    post = document["posterior"]
    out = []
    for case in document["cases"]:
        args = (post["weights"], post["features"], case["reliability"])
        a, b = gain_dit(*args), gain_direct(*args)
        if abs(a - b) > 1e-12:
            raise SystemExit(f"{case['name']}: dit and direct definition disagree {a} {b}")
        out.append({"name": case["name"], "expected_information_gain": a})
    return out


def check(document: dict) -> list[str]:
    problems = []
    if not document["cases"]:
        problems.append("empty case selection")
    for case, actual in zip(document["cases"], outputs(document), strict=True):
        if not values_close(
            case["expected_information_gain"],
            actual["expected_information_gain"],
            rtol=0.0,
            atol=ATOL,
        ):
            problems.append(f"{case['name']} differs from dit")
    prior = entropy(np.asarray(document["posterior"]["weights"]) / sum(document["posterior"]["weights"]))
    if not values_close(document["posterior"]["prior_entropy"], float(prior), rtol=0.0, atol=ATOL):
        problems.append("prior entropy differs")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    wrong = copy.deepcopy(document)
    wrong["cases"][0]["expected_information_gain"] += 1e-6
    if not check(wrong):
        failures.append("shifted gain accepted")
    swapped = copy.deepcopy(document)
    swapped["cases"][0]["reliability"] = swapped["cases"][1]["reliability"]
    if not check(swapped):
        failures.append("changed reliability accepted")
    reweighted = copy.deepcopy(document)
    reweighted["posterior"]["weights"] = [0.4, 0.4, 0.2]
    if not check(reweighted):
        failures.append("changed posterior accepted")
    empty = copy.deepcopy(document)
    empty["cases"] = []
    if not check(empty):
        failures.append("empty selection accepted")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    print("dit", importlib.metadata.version("dit"), "numpy", np.__version__)
    problems = check(document) if not args.self_test else self_test(document)
    for p in problems:
        print("FAIL", p)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
