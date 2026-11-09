"""Refuse a changed randomized design, then explicitly prepare it again.

The simulated randomized effect is 2. Rebinding a table whose assignment column changed is
refused, because the design is fixed at preparation; the refusal names ``ant.prepare`` and
leaves the original study and outcome unchanged. Preparing again on the flipped assignments
gives exactly the negated effect. A one-component mixture with a declared mean and variance
then refuses an interval (``uncertainty_representation_insufficient``) while its declared mean
stays usable. Nothing here is a claim about real data.
"""

import antecedent as ant
import numpy as np
from antecedent import joint_law_propagation as jl
from antecedent.interference import BernoulliAssignment
from antecedent.outcomes import require_success


def main():
    rng = np.random.default_rng(24)
    assigned = rng.binomial(1, 0.5, 160).astype(bool)
    ids = np.array([f"unit-{i}" for i in range(len(assigned))])
    data = {
        "arm": assigned,
        "id": ids,
        "outcome": 2 * assigned + rng.normal(scale=0.2, size=len(ids)),
    }
    query = ant.experiment.RandomizedEffect(
        "outcome",
        ant.experiment.ExperimentDesign(BernoulliAssignment(0.5), "arm", "id", "id"),
    )
    study = ant.prepare(data, query=query, refute=False)
    original = ant.estimate(study)
    changed = {**data, "arm": ~assigned}
    try:
        ant.rebind(study, data=changed)
    except ant.CausalError as error:
        assert "arm" in str(error) and "ant.prepare" in str(error)
        print("Design refusal:", error)
    else:
        raise AssertionError("Changed assignments were accepted")
    # The refused branch left the study and its outcome as they were.
    assert ant.estimate(study).answer.value == original.answer.value
    revised = ant.estimate(ant.prepare(changed, query=query, refute=False))
    assert np.isclose(revised.answer.value, -original.answer.value)
    print("Effect after preparing again:", revised.answer.value)
    claim = require_success(jl.mixture([jl.MixtureComponent("model", "evidence", 1.0, 2.0, 3.0)]))
    try:
        claim.interval()
    except jl.JointLawPropagationRefusal as error:
        assert error.reason_code == "uncertainty_representation_insufficient"
        print("Interval refusal:", error.reason_code, error.detail, error.remedy)
    else:
        raise AssertionError("A point-only mixture supplied an interval")
    print(
        "Use the declared mixture point:",
        claim.as_mean_source(
            quantity=ant.joint_distribution.ScientificQuantity.of(
                "outcome", "outcome", units="score", population="target", regime="model"
            )
        ).means,
    )


if __name__ == "__main__":
    main()
