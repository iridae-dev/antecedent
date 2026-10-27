"""Refuse a changed randomized design, then explicitly reprepare it.

Also distinguish a composed point/spread from a licensed interval.
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
    print("Reprepared effect:", revised.answer.value)
    claim = require_success(jl.mixture([jl.MixtureComponent("model", "evidence", 1.0, 2.0, 3.0)]))
    try:
        claim.interval()
    except jl.JointLawPropagationRefusal as error:
        assert error.reason_code == "cell_not_licensed"
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
