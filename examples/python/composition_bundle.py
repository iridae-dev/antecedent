"""Hand a composed decision to someone else as one bundle they can check against an identity.

A composition bundle links the parts of a decision (a joint distribution, the decision contract and
its result) as a typed graph of embedded artifacts and dependency edges. ``build`` seals it and
``export`` writes the bytes. A consumer who retained the bundle identity separately from the bytes
calls ``consume_bundle(..., expected_identity=...)``: every embedded artifact is decoded through its
own consumer and recomputed, so a changed or swapped part is refused or reported on its node.

Hand oracle: four equally likely rows with ``p = [1, 3, 2, 0]`` and ``q = [4, 0, 2, 6]``; the
risky action is worth ``E[p * q] = 2`` and the safe action 3, so the safe action is uniquely best."""

from __future__ import annotations

import numpy as np
from antecedent import composition_bundle as cb
from antecedent import decision
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)

P = [1.0, 3.0, 2.0, 0.0]
Q = [4.0, 0.0, 2.0, 6.0]


def quantity(variable: str, regime: str) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=variable,
        variable_name=variable,
        role="outcome",
        units="units",
        population_id="target",
        regime_id=regime,
        horizon=0,
        functional_id="outcome",
    )


p, q, safe = quantity("p", "do(a=1)"), quantity("q", "do(a=1)"), quantity("safe", "do(a=0)")
law = JointDistributionArtifact(
    DistributionIdentity(
        semantic="interventional_predictive",
        quantities=(p, q, safe),
        alignment="joint",
        source_id="enumerated",
        provider_id="exact-law",
        rng_id="deterministic_exact",
        snapshot_id="enumeration-1",
        causal_contract_id="checked-contract",
    ),
    np.array([[a, b, 3.0] for a, b in zip(P, Q, strict=True)], dtype=np.float64),
    calibration="exact",
)
contract = decision.Contract(
    actions=(
        decision.Action("risky", inputs=(p, q), utility=decision.x(0) * decision.x(1)),
        decision.Action(
            "safe", inputs=(safe,), utility=decision.maximum(decision.x(0), 0.0), kind="policy"
        ),
    ),
    utility_units="units",
    criterion=decision.Criterion.expected_utility(),
    target_population="target",
)
result = contract.evaluate(law)

# Build: embed each artifact from its bytes and declare what depends on what.
builder = cb.Bundle.builder()
parts = {
    "contract": ("decision_contract", contract.export(artifact_id="contract")),
    "law": ("distribution", law.export("law")),
    "result": ("decision_result", result.export(artifact_id="result")),
}
for node_id, (kind, artifact) in parts.items():
    builder.add_artifact(kind, artifact, node_id=node_id)
bundle = builder.connect("law", "result").connect("contract", "result").build()
print(bundle.explain())

# Export: the bytes travel; the identity is retained separately (a message, a ledger, a ticket).
data, identity = bundle.export(), bundle.identity

# Consume under the retained identity. Nothing is trusted from the bytes alone.
consumed = cb.consume_bundle(data, expected_identity=identity)
consumed.require_verified()
print(consumed.explain())
assert consumed.all_verified and consumed.identity == identity
assert abs(consumed.value("result", "risky.expected_utility") - 2.0) < 1e-9
assert abs(consumed.value("result", "safe.expected_utility") - 3.0) < 1e-9
assert consumed.claim_label == "joint_draw"

# Another identity refuses the same bytes.
try:
    cb.consume_bundle(data, expected_identity="0" * len(identity))
except cb.ExpectedIdentityMismatchRefusal as refusal:
    assert refusal.bundle_stage == "expected_identity_mismatch"
else:
    raise AssertionError("a bundle must not consume under another identity")
