#!/usr/bin/env python3
"""Randomized survival contrast: RMST and the survival curve by arm.

`antecedent.survival.SurvivalOutcome` carries an individually randomized
survival study through the retained `analyze` / `prepare` API and reports the
Kaplan-Meier curve, restricted mean survival time (RMST), and fixed-horizon
survival for each arm. The observation assumption is explicit: here censoring is
independent given the (empty) conditioning set. The retained study exports and
loads like any other; the contrast is point-only.

In the four-row fixture the treated arm has no events, so its survival stays at 1 and its RMST
equals the horizon 2; the script checks that and that the control RMST is smaller. The
independent-censoring assumption is declared, not tested.

Install with `python -m pip install antecedent`; see examples/README.md."""

from __future__ import annotations

import antecedent
from antecedent.observation import IndependentGiven
from antecedent.outcomes import require_success
from antecedent.survival import SurvivalOutcome


def main() -> None:
    data = {
        "duration": [1.0, 2.0, 2.0, 2.0],
        "event": [1.0, 0.0, 0.0, 0.0],
        "treatment": [0.0, 0.0, 1.0, 1.0],
    }
    query = SurvivalOutcome(
        "duration",
        "event",
        "treatment",
        2.0,  # RMST / survival horizon
        randomized=True,
        observation_assumption=IndependentGiven(()),
    )

    prepared = antecedent.prepare(data, query=query)
    outcome = antecedent.estimate(prepared)
    curve = require_success(outcome).survival
    print("times:", curve.times)
    print("control survival:", curve.control_survival)
    print("treated survival:", curve.treated_survival)
    print(f"RMST control={curve.rmst_control} treated={curve.rmst_treated}")
    print("uncertainty:", curve.uncertainty)
    # The treated arm has no events, so its survival is flat at 1 and RMST equals
    # the horizon; the control arm sees an event and so has strictly smaller RMST.
    assert abs(curve.rmst_treated - 2.0) < 1e-9, curve.rmst_treated
    assert curve.rmst_control < curve.rmst_treated

    # The result exports and loads; loading checks it is intact and recomputes nothing.
    artifact = outcome.export(artifact_id="survival-study")
    loaded = antecedent.load(artifact, expected=artifact.expectation)
    print("loaded answer kind:", loaded.answer.kind)


if __name__ == "__main__":
    main()
