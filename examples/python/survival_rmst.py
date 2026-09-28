#!/usr/bin/env python3
"""Randomized survival contrast: RMST and the survival curve by arm.

`antecedent.survival.SurvivalOutcome` carries an individually randomized
survival study through the retained `analyze` / `prepare` API and reports the
Kaplan-Meier curve, restricted mean survival time (RMST), and fixed-horizon
survival for each arm. The observation assumption is explicit: here censoring is
independent given the (empty) conditioning set. The retained study exports,
loads, and refreshes like any other; the contrast is point-only.

Install with `python -m pip install antecedent`; see examples/README.md."""

from __future__ import annotations

import antecedent
from antecedent.observation import IndependentGiven
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
    result = prepared.estimate()
    curve = result.survival
    print("times:", curve.times)
    print("control survival:", curve.control_survival)
    print("treated survival:", curve.treated_survival)
    print(f"RMST control={curve.rmst_control} treated={curve.rmst_treated}")
    print("uncertainty:", curve.uncertainty)

    # The frozen study round-trips through the durable artifact.
    loaded = antecedent.load(prepared.export(artifact_id="survival-study"))
    print("loaded answer kind:", loaded.answer.kind)


if __name__ == "__main__":
    main()
