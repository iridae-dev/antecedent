#!/usr/bin/env python3
"""First answer for a randomized survival contrast.

One `analyze` call estimates the Kaplan-Meier curves, restricted mean survival time (RMST)
and fixed-horizon survival by arm, and requests 399 subject-stratified bootstrap replicates.
The pointwise intervals, the replicate accounting and the simultaneous-band decision are
inside the call. Data synthesis sits outside the timed section.

    python parity/performance_lanes/entry_points/survival.py --role control
    python parity/performance_lanes/entry_points/survival.py --role candidate
"""

from __future__ import annotations

from typing import Any

import antecedent
import numpy as np
from _entry import arguments, digest_arrays, run_entry
from antecedent.observation import IndependentGiven
from antecedent.survival import SurvivalOutcome

DATA_SEED = 29
EXECUTION_SEED = 17
ARM_ROWS = 1500
HORIZON = 2.0
BOOTSTRAP = 399


def synthesize(arm_rows: int) -> dict[str, np.ndarray]:
    """Randomized two-arm study: exponential event times, uniform administrative censoring."""
    rng = np.random.default_rng(DATA_SEED)
    treatment = np.repeat([0.0, 1.0], arm_rows)
    rate = np.where(treatment == 1.0, 0.35, 0.5)
    event_time = rng.exponential(1.0 / rate)
    censor_time = rng.uniform(0.5, 4.0, size=treatment.size)
    return {
        "duration": np.minimum(event_time, censor_time),
        "event": (event_time <= censor_time).astype(np.float64),
        "treatment": treatment,
    }


def main() -> None:
    args = arguments(__doc__ or "")
    data = synthesize((args.rows or 2 * ARM_ROWS) // 2)
    query = SurvivalOutcome(
        "duration",
        "event",
        "treatment",
        HORIZON,
        randomized=True,
        observation_assumption=IndependentGiven(()),
    )

    def section() -> Any:
        return antecedent.analyze(
            data, query=query, bootstrap=BOOTSTRAP, seed=EXECUTION_SEED, threads=args.threads
        )

    def fields(result: Any) -> dict[str, Any]:
        curve = result.survival
        band = None if curve is None else curve.difference_band
        return {
            "rmst_difference": None if curve is None else curve.rmst_difference,
            "rmst_difference_interval": None if curve is None else curve.rmst_difference_interval,
            "survival_difference_interval": None
            if curve is None
            else curve.survival_at_tau_difference_interval,
            "uncertainty": None if curve is None else curve.uncertainty,
            "replicates_requested": None if curve is None else curve.bootstrap_replicates_requested,
            "replicates_ok": None if curve is None else curve.bootstrap_replicates_ok,
            "support_status": None if curve is None else curve.support_status,
            "band_replicates_ok": None if band is None else band.replicates_ok,
            "band_unavailable_reason": None if curve is None else curve.band_unavailable_reason,
            "evidence_status": result.evidence_status,
        }

    run_entry(
        "survival",
        args,
        digest_arrays(data),
        section,
        fields,
        required=("rmst_difference_interval", "survival_difference_interval", "replicates_ok"),
    )


if __name__ == "__main__":
    main()
