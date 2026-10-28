# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Standalone references for adaptive (history-dependent) policy search.

Two closed forms, derived by hand and cross-checked by an independent Monte Carlo that holds
its own random generator and dynamics.  No Antecedent code is imported.

1. Treatment threshold.  A stationary Gaussian state z_t ~ N(0, s^2) is observed before the
   action u_t in [0, 1]; the stage utility is u_t (z_t - k).  The optimal rule treats when
   z_t >= k and earns T * (s*phi(k/s) - k*(1 - Phi(k/s))).  Any fixed schedule earns
   -k * u_t per step on average, so the best fixed schedule treats never and earns 0.
2. Linear-quadratic feedback.  x_{t+1} = a x_t + u_t + w_t, w_t ~ N(0, sigma^2), stage cost
   q x_t^2 + r u_t^2 and terminal cost P x_T^2 with P the steady-state Riccati solution.  The
   Riccati gain is then stationary, K = a P / (r + P), the optimal expected cost is
   P x0^2 + T P sigma^2, and the best open-loop schedule costs the deterministic optimum
   P x0^2 plus the input-independent state-variance cost.

    uv run --frozen --script conformance/design/adaptive_policy_search/reference.py --check

`--write` regenerates the numeric blocks only and keeps the recorded `oracle` block, including its
review, so a rewrite cannot erase it; `--check` compares every block except `oracle`.

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

EXPECTED = Path(__file__).with_name("expected.json")


def phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def big_phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def treatment_reference(horizon: int, s: float, k: float) -> dict[str, float]:
    return {
        "adaptive_value": horizon * (s * phi(k / s) - k * (1.0 - big_phi(k / s))),
        "best_fixed_schedule_value": 0.0,
    }


def riccati_p(a: float, q: float, r: float) -> float:
    p = q
    for _ in range(100_000):
        nxt = q + a * a * p - (a * p) ** 2 / (r + p)
        if abs(nxt - p) < 1e-15:
            break
        p = nxt
    return p


def lq_reference(a: float, q: float, r: float, sigma: float, x0: float, horizon: int) -> dict[str, float]:
    p = riccati_p(a, q, r)
    noise_cost = 0.0
    for step in range(1, horizon + 1):
        variance = sigma**2 * sum(a ** (2 * j) for j in range(step))
        noise_cost += (q if step < horizon else p) * variance
    return {
        "P": p,
        "gain": a * p / (r + p),
        "closed_loop_value": -(p * x0**2 + horizon * p * sigma**2),
        "open_loop_value": -(p * x0**2 + noise_cost),
    }


def monte_carlo_treatment(horizon: int, rho: float, sigma: float, k: float, n: int, seed: int) -> float:
    rng = random.Random(seed)
    s = sigma / math.sqrt(1.0 - rho * rho)
    total = 0.0
    for _ in range(n):
        z = s * rng.gauss(0.0, 1.0)
        for _ in range(horizon):
            total += max(z - k, 0.0)
            z = rho * z + sigma * rng.gauss(0.0, 1.0)
    return total / n


def monte_carlo_lq(a: float, q: float, r: float, sigma: float, x0: float, horizon: int, gain: float, n: int, seed: int) -> float:
    rng = random.Random(seed)
    p = riccati_p(a, q, r)
    total = 0.0
    for _ in range(n):
        x, cost = x0, 0.0
        for _ in range(horizon):
            u = -gain * x
            cost += q * x * x + r * u * u
            x = a * x + u + sigma * rng.gauss(0.0, 1.0)
        total += cost + p * x * x
    return -total / n


def document() -> dict:
    treat = treatment_reference(10, 1.0, 0.5)
    lq = lq_reference(0.9, 1.0, 0.5, 1.0, 2.0, 12)
    return {
        "schema_version": 1,
        "tolerance_class": "StableFloat",
        "oracle": {
            "kind": "closed_form",
            "method": "treat-above-threshold optimum for a stationary Gaussian state and the linear-quadratic Riccati feedback optimum with the closed-form best open-loop schedule",
            "independence_basis": "reference.py is standard-library Python that imports no Antecedent code; each closed form is cross-checked by its own Monte Carlo with a separate generator. The author of the reference is the repository author and independence from the production implementation has not been reviewed.",
            "generator": "conformance/design/adaptive_policy_search/reference.py",
            "historical_exposure": "Historical author exposure is unrecorded and remains unknown.",
        },
        "treatment": {
            "horizon": 10, "stationary_sd": 1.0, "cost": 0.5, **treat,
        },
        "lq": {
            "a": 0.9, "q": 1.0, "r": 0.5, "sigma": 1.0, "x0": 2.0, "horizon": 12, **lq,
        },
    }


def close(a, b) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def _numeric(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "oracle"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify closed forms and expected.json")
    parser.add_argument("--write", action="store_true", help="regenerate expected.json")
    args = parser.parse_args()
    doc = document()
    treat, lq = doc["treatment"], doc["lq"]
    recorded = json.loads(EXPECTED.read_text(encoding="utf-8")) if EXPECTED.exists() else {}
    if "oracle" in recorded:
        doc["oracle"] = recorded["oracle"]
    if args.write:
        EXPECTED.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if args.check:
        mc = monte_carlo_treatment(10, 0.6, 0.8, 0.5, 40_000, 1)
        assert abs(mc - treat["adaptive_value"]) < 0.05, (mc, treat)
        mc_lq = monte_carlo_lq(0.9, 1.0, 0.5, 1.0, 2.0, 12, lq["gain"], 40_000, 2)
        assert abs(mc_lq - lq["closed_loop_value"]) < 0.3, (mc_lq, lq)
        assert lq["closed_loop_value"] - lq["open_loop_value"] > 10.0
        if not close(_numeric(json.loads(EXPECTED.read_text(encoding="utf-8"))), _numeric(doc)):
            print("expected.json differs from the reference; rerun with --write", file=sys.stderr)
            raise SystemExit(1)
        print("adaptive policy search reference check OK")
    elif not args.write:
        print(json.dumps(doc, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
