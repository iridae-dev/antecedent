"""Enumerate a declared tiny procedure under supplied Bernoulli truths.

Two units are each assigned to one of two arms with probability 0.5, and the arms have
supplied success probabilities 0.2 and 0.6. The script checks the exact law of the contrast
against hand-computed masses (0.04, 0.22 and 0.24 at -1, 0 and 1) and keeps the 0.5 mass
where both units land in one arm visible as undefined.

This is a derived procedure report, not empirical interval coverage or a fitted
posterior for real trial data.
"""

import math

from antecedent import adaptive_inference as ai


def main():
    procedure = ai.AdaptiveProcedure.declare(
        ai.ProcedureDeclaration(
            arms=2,
            assignment=ai.AssignmentRule.fixed((0.5, 0.5)),
            enrolment=(2,),
            observation=ai.OutcomeObservation(),
            exposure=ai.ExposureModel((0.0, 0.0), 0, "delivered"),
            stopping=ai.ProcedureStopping.no_stopping(),
            selection=ai.FinalSelection.no_selection(),
        )
    )
    report = procedure.report(ai.FixedSampleProcedureReport(truth=(0.2, 0.6)))
    law = report.body["law"]
    assert math.isclose(law["undefined_mass"], 0.5)
    masses = {round(value, 12): probability for value, probability in law["support"]}
    for value, truth in [(-1.0, 0.04), (0.0, 0.22), (1.0, 0.24)]:
        assert math.isclose(masses[value], truth)
    print("Standing:", report.claim)
    print("Defined contrast masses:", masses, "undefined mass:", law["undefined_mass"])
    print("Report:", report.inspect())


if __name__ == "__main__":
    main()
