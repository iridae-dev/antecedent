# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["dowhy==0.14", "numpy", "pandas"]
# ///
"""Retained black-box check of the refuter acceptance bands against DoWhy 0.14.

The fixture's bands describe the expected behavior of five refuters on a confounded linear
model Z->T, Z->Y, T->Y with true ATE 2.  The fixture carries no data, so this script draws a
seeded n=400 sample of that model, fits the public DoWhy linear-regression backdoor estimate,
runs the public placebo (permute), dummy-outcome, random-common-cause, data-subset and
bootstrap refuters, and checks that DoWhy's own results satisfy the same bands.  It does not
run Antecedent and it does not reproduce the product's inputs: it establishes that the bands
are consistent with an independent implementation on one comparable model, not that the product
output equals DoWhy's.  The unobserved-common-cause band is not checked because DoWhy's
confounder-strength parameterization is not in residual-SD units.

    uv run --managed-python --frozen --script conformance/validate/refuters/reference.py --check
    uv run --managed-python --frozen --script conformance/validate/refuters/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dowhy import CausalModel

HERE = Path(__file__).resolve().parent
SEED = 20260727
N = 400
SIMULATIONS = 100


def sample() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    z = rng.normal(size=N)
    t = 0.8 * z + rng.normal(size=N)
    y = 2.0 * t + 1.5 * z + rng.normal(size=N)
    return pd.DataFrame({"z": z, "t": t, "y": y})


def mean_new_effect(result) -> float:
    items = result if isinstance(result, list) else [result]
    return float(np.mean([r.new_effect for r in items]))


def outputs() -> dict:
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    model = CausalModel(
        data=sample(), treatment="t", outcome="y", graph="digraph{z->t;z->y;t->y}"
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        estimand, method_name="backdoor.linear_regression", test_significance=False
    )
    kwargs = {"random_seed": 1, "num_simulations": SIMULATIONS}
    placebo = model.refute_estimate(
        estimand, estimate, method_name="placebo_treatment_refuter", placebo_type="permute", **kwargs
    )
    dummy = model.refute_estimate(
        estimand, estimate, method_name="dummy_outcome_refuter", **kwargs
    )
    rcc = model.refute_estimate(
        estimand, estimate, method_name="random_common_cause", **kwargs
    )
    subset = model.refute_estimate(
        estimand, estimate, method_name="data_subset_refuter", subset_fraction=0.8, **kwargs
    )
    boot = model.refute_estimate(
        estimand, estimate, method_name="bootstrap_refuter", **kwargs
    )
    original = float(estimate.value)
    return {
        "original_ate": original,
        "placebo": mean_new_effect(placebo),
        "dummy_outcome": mean_new_effect(dummy),
        "random_common_cause_delta": mean_new_effect(rcc) - original,
        "subset_delta": mean_new_effect(subset) - original,
        "bootstrap_new_effect": mean_new_effect(boot),
    }


def check(document: dict, got: dict | None = None) -> list[str]:
    expected = document["expected"]
    got = got if got is not None else outputs()
    problems = []
    if document["oracle"]["version"] != "0.14":
        problems.append("reference package version changed")
    if abs(got["original_ate"] - 2.0) > 0.25:
        problems.append(f"original ATE {got['original_ate']} is not near the true value 2")
    if abs(got["placebo"]) >= expected["placebo_abs_max"]:
        problems.append(f"placebo effect {got['placebo']} outside band")
    if abs(got["dummy_outcome"]) >= expected["dummy_outcome_abs_max"]:
        problems.append(f"dummy-outcome effect {got['dummy_outcome']} outside band")
    if abs(got["random_common_cause_delta"]) >= expected["random_common_cause_abs_delta_max"]:
        problems.append("random-common-cause shift outside band")
    if abs(got["subset_delta"]) >= expected["subset_abs_delta_max"]:
        problems.append("data-subset shift outside band")
    if expected["bootstrap_contains_original"] and abs(got["bootstrap_new_effect"] - got["original_ate"]) > 0.25:
        problems.append("bootstrap refuter does not stay near the original estimate")
    return problems


def self_test(document: dict) -> list[str]:
    got = outputs()
    failures = []
    tight = copy.deepcopy(document)
    tight["expected"]["placebo_abs_max"] = 1e-6
    if not check(tight, got):
        failures.append("an unattainably tight placebo band accepted")
    shifted = dict(got, placebo=1.0)
    if not check(document, shifted):
        failures.append("a falsified placebo effect accepted")
    shifted = dict(got, random_common_cause_delta=1.0)
    if not check(document, shifted):
        failures.append("a large random-common-cause shift accepted")
    shifted = dict(got, original_ate=0.0)
    if not check(document, shifted):
        failures.append("a wrong original ATE accepted")
    wrong = copy.deepcopy(document)
    wrong["oracle"]["version"] = "0.13"
    if not check(wrong, got):
        failures.append("wrong reference version accepted")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    if args.print:
        print(json.dumps(outputs(), indent=2))
        return 0
    problems = self_test(document) if args.self_test else check(document)
    for p in problems:
        print("FAIL", p)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
