# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Independent reference for a feedback system fitted with a registered external mechanism.

The system has ``x0 = gain * tanh(x1) + u0`` (the external mechanism) and
``x1 = tanh(0.2 * x0) + u1`` (a built-in: activation of the weighted sum). The truth is derived without the production code:

* With ``x1`` hard-clamped in every row, the maximum-likelihood gain of the first equation is
  the least-squares slope of ``x0`` on ``tanh(x1)`` (the free Jacobian is the identity).
* The fitted equilibrium is the fixed point of the two equations, found by bisection on ``x1``
  (the map is a contraction because the slope bounds ``0.9`` and ``0.2`` give row sums below one).
* The unclamped joint log-likelihood at a state is ``-0.5 z'z - log(2 pi) + log|det(I - J)|``
  with ``J = [[0, g sech^2(x1)], [0.2 sech^2(x0), 0]]``, evaluated here by an explicit 2x2
  determinant.

    uv run --script conformance/gcm/external_mechanism_feedback/reference.py          # write
    uv run --frozen --script conformance/gcm/external_mechanism_feedback/reference.py --check  # compare
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED = HERE / "expected.json"

CLAMPED = (-2.0, -1.0, 0.5, 1.0, 2.0)
NOISE = ((0.1, 0.2), (-0.1, -0.2))


def least_squares_gain() -> float:
    rows = [(0.7 * math.tanh(x) + e, x) for x in CLAMPED for e in (-0.1, 0.1)]
    t = [math.tanh(x) for _, x in rows]
    y = [v for v, _ in rows]
    return sum(a * b for a, b in zip(t, y, strict=True)) / sum(a * a for a in t)


def fixed_point(gain: float, u: tuple[float, float]) -> list[float]:
    def residual(x1: float) -> float:
        x0 = gain * math.tanh(x1) + u[0]
        return x1 - (math.tanh(0.2 * x0) + u[1])

    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if residual(mid) < 0 else (lo, mid)
    x1 = 0.5 * (lo + hi)
    return [gain * math.tanh(x1) + u[0], x1]


def joint_log_density(gain: float, state: tuple[float, float]) -> float:
    x0, x1 = state
    r0 = x0 - gain * math.tanh(x1)
    r1 = x1 - math.tanh(0.2 * x0)
    j01 = gain * (1.0 - math.tanh(x1) ** 2)
    j10 = 0.2 * (1.0 - math.tanh(0.2 * x0) ** 2)
    det = 1.0 - j01 * j10
    return -0.5 * (r0 * r0 + r1 * r1) - math.log(2.0 * math.pi) + math.log(abs(det))


def committed_review() -> dict | None:
    """The reviewed-artifact record lives in the committed fixture, not in this derivation."""
    if not EXPECTED.is_file():
        return None
    oracle = json.loads(EXPECTED.read_text(encoding="utf-8")).get("oracle", {})
    return oracle.get("reference_review")


def build() -> dict:
    gain = least_squares_gain()
    document = {
        "oracle": {
            "kind": "closed_form",
            "method": (
                "least-squares slope, bisection fixed point and explicit 2x2 Jacobian "
                "determinant of the two-equation saturating feedback system"
            ),
            "independence_basis": (
                "Standard-library arithmetic only; nothing is shared with the production "
                "solver, likelihood or resolver. The retained generator and the executing "
                "assertion were read for the review recorded in reference_review."
            ),
            "generator": "conformance/gcm/external_mechanism_feedback/reference.py",
            "generator_retained": True,
            "environment_lock": "conformance/gcm/external_mechanism_feedback/reference.py.lock",
            "command": "uv run --frozen --script conformance/gcm/external_mechanism_feedback/reference.py --check",
            "asserted_outputs": ["fitted_gain", "fixed_points"],
            "context_only_outputs": ["joint_log_density_at_first_fixed_point"],
            "executing_assertion": "python/tests/test_external_cyclic_mechanisms.py::test_fit_and_fixed_points_match_the_committed_independent_reference",
            "historical_exposure": "Historical author exposure is unrecorded and remains unknown.",
        },
        "schema_version": 1,
        "fitted_gain": gain,
        "fixed_points": [fixed_point(gain, u) for u in NOISE],
        "joint_log_density_at_first_fixed_point": joint_log_density(
            gain, tuple(fixed_point(gain, NOISE[0]))
        ),
    }
    review = committed_review()
    if review is not None:
        document["oracle"]["reference_review"] = review
    return document


def main() -> int:
    document = build()
    if "--check" in sys.argv[1:]:
        committed = json.loads(EXPECTED.read_text(encoding="utf-8"))
        problems = []
        stored_oracle = {k: v for k, v in committed["oracle"].items() if k != "reference_review"}
        fresh_oracle = {k: v for k, v in document["oracle"].items() if k != "reference_review"}
        if fresh_oracle != stored_oracle:
            problems.append("oracle record differs")
        for key in ("fitted_gain", "joint_log_density_at_first_fixed_point"):
            if not math.isclose(document[key], committed[key], rel_tol=1e-12):
                problems.append(key)
        for fresh, stored in zip(
            document["fixed_points"], committed["fixed_points"], strict=True
        ):
            if not all(
                math.isclose(a, b, rel_tol=1e-12)
                for a, b in zip(fresh, stored, strict=True)
            ):
                problems.append("fixed_points")
        for problem in problems:
            print(f"external mechanism reference mismatch: {problem}", file=sys.stderr)
        return 1 if problems else 0
    EXPECTED.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
