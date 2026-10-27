"""Compute a product's spread from columns of one declared aligned law.

The ten rows are an enumerated synthetic distribution, not independent draws
from two marginals. A spread does not grant interval coverage.
"""

import math

import numpy as np
from antecedent import joint_law_propagation as jl
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)
from antecedent.outcomes import require_success


def _artifact(
    snapshot: str, columns: list[np.ndarray], alignment: str = "joint"
) -> JointDistributionArtifact:
    names = ("x", "y", "z")[: len(columns)]
    quantities = tuple(
        ScientificQuantity.outcome(name, units="units", population="target", regime="do(a=1)")
        for name in names
    )
    identity = DistributionIdentity(
        semantic="interventional_predictive",
        quantities=quantities,
        alignment=alignment,  # type: ignore[arg-type]
        source_id=f"study-{snapshot}",
        provider_id="engine",
        rng_id=f"seed-{snapshot}",
        snapshot_id=snapshot,
        causal_contract_id="checked-contract",
    )
    return JointDistributionArtifact(identity, np.column_stack(columns).astype(np.float64))


def _enumerated(scale: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rows = [(0.0, 0.0, 1), (0.0, 2.0, 4), (1.0, 0.0, 3), (1.0, 2.0, 2)]
    x = np.array([a for a, _, c in rows for _ in range(c * scale)])
    y = np.array([b for _, b, c in rows for _ in range(c * scale)])
    return x, y


def _pair(snapshot: str, scale: int = 1) -> tuple[jl.PropagationDraws, jl.PropagationDraws]:
    x, y = _enumerated(scale)
    art = _artifact(snapshot, [x, y])
    return (
        jl.PropagationDraws.from_artifact_column("x", art, 0),
        jl.PropagationDraws.from_artifact_column("y", art, 1),
    )


def main():
    x, y = _pair("tutorial")
    spread = require_success(jl.spread("product", x, y, "product"))
    assert spread.label == "uncalibrated_spread"
    assert math.isclose(spread.value("mean"), 0.4)
    assert math.isclose(spread.value("variance"), 6.4 / 9)
    print("Product mean:", spread.value("mean"))
    print("Product variance:", spread.value("variance"), "standing:", spread.label)
    try:
        require_success(jl.aligned_functional("product", x, y, "product")).interval()
    except jl.JointLawPropagationRefusal as error:
        assert error.reason_code == "cell_not_licensed"
        print("Interval request:", error.reason_code)
    else:
        raise AssertionError("A point producer supplied an interval")


if __name__ == "__main__":
    main()
