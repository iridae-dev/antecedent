# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.1.3", "scipy==1.14.1"]
# ///
"""Black-box distribution-domain reference; repository policy is not SciPy truth.

Only the frozen distribution inputs are read. Normalization is context about
the upstream distributions: the product tests implement support validation and
coefficient precision, not numerical density integration.

uv run --frozen --script conformance/bayesian/prior_support/reference.py --check
SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import warnings
import sys
from pathlib import Path

from scipy.integrate import quad
from scipy.special import gammaln
from scipy.stats import invgamma, norm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from reference_compare import values_close

PATH = Path(__file__).with_name("expected.json")


def numeric(value):
    return float(value)


def normal_valid(mean, variance):
    if not math.isfinite(mean) or not math.isfinite(variance):
        return False
    with warnings.catch_warnings():
        # Invalid scale is deliberately part of the frozen domain grid.
        warnings.simplefilter("ignore", RuntimeWarning)
        return math.isfinite(float(norm(loc=mean, scale=math.sqrt(max(variance, 0))).logpdf(mean)))


def check(document):
    problems = []
    if document.get("tolerance") != {"normalization_atol": 2e-8}:
        problems.append("normalization tolerance differs from reviewed 2e-8")
    for family in ("coefficient_priors", "inverse_gamma_priors", "effect_priors", "known_variances"):
        if not document.get(family):
            problems.append(f"empty {family} grid")
    for case in document["coefficient_priors"]:
        means = list(map(numeric, case["mean"]))
        variances = list(map(numeric, case["variance"]))
        # Nonempty aligned coordinates are repository container policy; scalar
        # domain decisions below are checked through the public distribution API.
        valid = bool(means) and len(means) == len(variances) and all(
            normal_valid(m, v) for m, v in zip(means, variances)
        ) and all(v > 0 for v in variances)
        if case["valid"] != valid:
            problems.append(f"{case['name']} domain differs")
        if valid:
            precision = [1 / v for v in variances]
            if not values_close(case["precision"], precision, rtol=1e-12, atol=0):
                problems.append(f"{case['name']} precision differs")
            integral = quad(norm.pdf, -math.inf, math.inf, epsabs=1e-11)[0]
            if not values_close(case["normal_integrals"], [integral] * len(means), rtol=0, atol=2e-8):
                problems.append(f"{case['name']} normalization differs")
    for i, case in enumerate(document["inverse_gamma_priors"]):
        shape, scale = case["shape"], case["scale"]
        distribution = invgamma(a=shape, scale=scale)
        valid = math.isfinite(float(distribution.cdf(scale)))
        if case["valid"] != valid:
            problems.append(f"inverse_gamma[{i}] domain differs")
        if valid:
            # y = shape * log(x / scale) avoids a vanishingly narrow peak and
            # the huge x tail for shape 0.001. This integrates the public pdf
            # in log space, without importing a density implementation.
            def transformed(y):
                log_x = math.log(scale) + y / shape
                if -700 < log_x < 700:
                    return math.exp(float(distribution.logpdf(math.exp(log_x))) + log_x - math.log(shape))
                # Outside representable x, use the documented log-density
                # identity. This is context-only numerical integration.
                if -y / shape > 700:
                    return 0.0
                return math.exp(-y - math.exp(-y / shape) - gammaln(shape) - math.log(shape))
            integral = quad(transformed, -math.inf, 0, epsabs=1e-10)[0] + quad(transformed, 0, math.inf, epsabs=1e-10)[0]
            if not values_close(case["integral"], integral, rtol=0, atol=2e-8):
                problems.append(f"inverse_gamma[{i}] normalization differs")
            mean = None if shape <= 1 else float(distribution.mean())
            if not values_close(case["mean"], mean, rtol=1e-12, atol=0):
                problems.append(f"inverse_gamma[{i}] mean differs")
    for i, case in enumerate(document["effect_priors"]):
        mean, sd = numeric(case["mean"]), numeric(case["sd"])
        valid = math.isfinite(mean) and math.isfinite(sd) and sd > 0 and normal_valid(mean, sd * sd)
        if case["valid"] != valid:
            problems.append(f"effect[{i}] domain differs")
    for i, case in enumerate(document["known_variances"]):
        value = numeric(case["value"])
        if case["valid"] != (math.isfinite(value) and value > 0 and normal_valid(0, value)):
            problems.append(f"variance[{i}] domain differs")
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    for package, version in {"numpy": "2.1.3", "scipy": "1.14.1"}.items():
        if importlib.metadata.version(package) != version:
            raise SystemExit(f"{package} must be {version}")
    document = json.loads(PATH.read_text(encoding="utf-8"))
    problems = check(document)
    if problems:
        raise SystemExit("\n".join(problems))
    if args.self_test:
        for mutate in (
            lambda d: d["coefficient_priors"][0]["precision"].__setitem__(0, 2.0),
            lambda d: d["inverse_gamma_priors"][0].__setitem__("valid", False),
            lambda d: d["tolerance"].__setitem__("normalization_atol", 1.0),
            lambda d: d.__setitem__("coefficient_priors", []),
            lambda d: d["inverse_gamma_priors"][2].__setitem__("mean", 3.0),
        ):
            altered = copy.deepcopy(document)
            mutate(altered)
            if not check(altered):
                raise SystemExit("reference failed to reject mutation")
    print("Prior-support reference: 24 domain cases and normalization context reproduced")


if __name__ == "__main__":
    main()
