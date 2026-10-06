"""Build the Python half of the Rust/Python cross-surface fixtures.

Run from the repository root::

    python python/tests/generate_cross_surface_fixtures.py

This writes Python-built artifacts into ``conformance/cross_surface/`` for the
Rust consumers in ``crates/antecedent-io/tests/cross_surface_external_claim.rs``
and ``crates/antecedent-design/tests/cross_surface_decision.rs``. The Rust
consumers never trust the ``*.identity.json`` files for loading; they rebuild
every expected identity from constants and only cross-check these files.

The module also holds the shared closed-form fixture builders that
``test_cross_surface.py`` imports, so the Python side declares each fixture in
exactly one place. The two fixtures are enumerated, not simulated:

* external claim: ``E[Y | do(a)] = 1 + 2a`` over ``a`` in ``0, 1, 2`` -> ``[1, 3, 5]``;
* decision: risky utility ``P * Q`` over rows ``p=[1,3,2,0]``, ``q=[4,0,2,6]``
  against a safe action worth 3, so ``EU = 2`` vs ``3``, ``EVPI = 0.5`` and the
  verdict is uniquely optimal ``safe``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import antecedent as ac
import numpy as np
from antecedent import decision, external
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "conformance" / "cross_surface"
GRID = [0.0, 1.0, 2.0]
EDGES = [("x", "a"), ("x", "y"), ("a", "y")]
NAMES = ["x", "a", "y"]
EXTERNAL_VALUES = [1.0, 3.0, 5.0]
SUPPORT = ("supported", "supported", "outside_empirical_support")
P = [1.0, 3.0, 2.0, 0.0]
Q = [4.0, 0.0, 2.0, 6.0]


def external_spec() -> external.ExternalSpec:
    """The spec of the Rust fixture: contract ``checked-contract`` over graph ``graph-1``."""
    ident = ac.identify(graph=EDGES, names=NAMES, query=ac.ResponseCurve("a", "y", grid=GRID))
    return external.response(
        ident,
        outcome_units="mmHg",
        population="target",
        graph_id="graph-1",
        contract_id="checked-contract",
        require_evidence=("factor:z",),
        require_assumptions=("ignorability",),
    )


def external_provider(**kwargs: Any) -> external.ProviderObject:
    fields: dict[str, Any] = {
        "provider_id": "lab",
        "object_id": "curve",
        "version": "v3",
        "snapshot": "snap-9",
        "request": "req-1",
        "meaning": "interventional_predictive",
        "capabilities": ("mean",),
    }
    return external.ProviderObject(**{**fields, **kwargs})


def external_response(**kwargs: Any) -> external.Response:
    fields: dict[str, Any] = {
        "provider": external_provider(),
        "values": EXTERNAL_VALUES,
        "evidence": ("factor:z",),
        "assumptions": ("ignorability",),
        "attested_by": "lab",
        "support": SUPPORT,
    }
    return external.Response(**{**fields, **kwargs})


def external_claim() -> external.BoundExternalClaim:
    return external_spec().bind(external_response())


def _quantity(variable: str, regime: str) -> ScientificQuantity:
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


COLUMNS = (_quantity("p", "do(a=1)"), _quantity("q", "do(a=1)"), _quantity("safe", "do(a=0)"))


def source_identity() -> DistributionIdentity:
    return DistributionIdentity(
        semantic="interventional_predictive",
        quantities=COLUMNS,
        alignment="joint",
        source_id="enumerated",
        provider_id="exact-law",
        rng_id="deterministic_exact",
        snapshot_id="enumeration-1",
        causal_contract_id="checked-contract",
    )


def decision_source() -> JointDistributionArtifact:
    draws = np.array([[a, b, 3.0] for a, b in zip(P, Q, strict=True)], dtype=np.float64)
    return JointDistributionArtifact(source_identity(), draws, calibration="exact")


def decision_contract() -> decision.Contract:
    """The ``P*Q`` contract of ``crates/antecedent-design/tests/decision_artifact.rs``."""
    return decision.Contract(
        actions=(
            decision.Action(
                "risky", inputs=(COLUMNS[0], COLUMNS[1]), utility=decision.x(0) * decision.x(1)
            ),
            decision.Action(
                "safe",
                inputs=(COLUMNS[2],),
                utility=decision.maximum(decision.x(0), 0.0),
                kind="policy",
            ),
        ),
        utility_units="units",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
        constraints=(
            decision.Constraint(
                "q-cap",
                decision.x(1),
                bound=5.0,
                units="units",
                min_probability=0.75,
                applies_to=("risky",),
            ),
        ),
    )


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    claim = external_claim()
    contract = decision_contract()
    source = decision_source()
    result = contract.evaluate(source)
    assert result.selected == ("safe",)

    files: dict[str, bytes] = {
        "py_external_claim.bin": claim.export(artifact_id="claim"),
        "py_external_claim.identity.json": (
            json.dumps(claim.identity, sort_keys=True, indent=2) + "\n"
        ).encode(),
        "py_decision_contract.bin": contract.export(artifact_id="decision-contract"),
        "py_decision_result.bin": result.export(artifact_id="decision-result"),
        "py_decision_source.bin": source.export("source"),
        "py_decision.identities.json": (
            json.dumps(
                {
                    "contract_identity": contract.identity,
                    "source_digest": decision.source_digest(source),
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode(),
    }
    for name, data in files.items():
        (FIXTURE_DIR / name).write_bytes(data)
        print(f"wrote {FIXTURE_DIR / name} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
