# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the horizon-one trial-to-target fixture (n = 1000).

Rebuilds the rows of the test `horizon_one_transport_is_the_static_contrast_with_a_long_run_variance`
(splitmix64 streams, arm coin, outcome a + (u - 1/2), trial coin, recorded s = e = 1/2) and evaluates
the Dahabreh et al. inverse-odds contrast as defined by the provenance record: on trial rows
sign * ((1 - s) / s) / P(arm) * y, summed and divided by the NUMBER OF NON-PARTICIPANT (target) rows.
That is a Horvitz-Thompson style sum, not a normalised mean: with s = 1/2 recorded while the realised
trial share is n_trial / n, the estimate is the arm-mean contrast times (n_trial / n_target) up to the
arm-count imbalance, which is why it sits above the true effect 1 without any defect.

Exact Fractions throughout (the splitmix doubles are exact dyadic rationals). Standard library only;
imports no Antecedent code.

    python3 conformance/response/horizon_one_transport/reference.py --check
    python3 conformance/response/horizon_one_transport/reference.py --write

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
MASK = (1 << 64) - 1
N = 1000


def unit(i: int, salt: int) -> Fraction:
    z = ((i + salt) & MASK) * 0x9E3779B97F4A7C15 & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return Fraction((z ^ (z >> 31)) >> 11, 1 << 53)


def derive() -> dict:
    half = Fraction(1, 2)
    a = [1 if unit(i, 1) < half else 0 for i in range(N)]
    y = [a[i] + half * (unit(i, 2) - half) * 2 for i in range(N)]
    trial = [1 if unit(i, 3) < half else 0 for i in range(N)]
    s = e = half
    n_target = sum(1 for t in trial if not t)
    n_trial = N - n_target
    ipw_sum = Fraction(0)
    for i in range(N):
        if trial[i]:
            odds = (1 - s) / s
            arm = e if a[i] else 1 - e
            ipw_sum += (1 if a[i] else -1) * odds / arm * y[i]
    ipw = ipw_sum / n_target

    def arm_mean(rows, arm):
        v = [y[i] for i in rows if a[i] == arm]
        return sum(v) / len(v)

    trial_rows = [i for i in range(N) if trial[i]]
    return {
        "n": N,
        "n_trial": n_trial,
        "n_target": n_target,
        "ipw": float(ipw),
        "ipw_exact": f"{ipw.numerator}/{ipw.denominator}",
        "trial_arm_mean_difference": float(arm_mean(trial_rows, 1) - arm_mean(trial_rows, 0)),
        "all_rows_arm_mean_difference": float(arm_mean(range(N), 1) - arm_mean(range(N), 0)),
        "horvitz_thompson_over_n": float(ipw_sum / N),
        "trial_share_inflation": n_trial / n_target,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    doc = derive()
    path = HERE / "expected.json"
    if args.write:
        path.write_text(json.dumps(doc, indent=2) + "\n")
    if args.check:
        have = json.loads(path.read_text())
        bad = [k for k in doc if k == "ipw_exact" and have[k] != doc[k] or k != "ipw_exact" and abs(have[k] - doc[k]) > 1e-12]
        if bad:
            print("mismatch:", bad, file=sys.stderr)
            return 1
    print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
