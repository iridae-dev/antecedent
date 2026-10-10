"""Build the Python half of the cross-surface composition-bundle fixtures (C3).

Run from the repository root::

    python python/tests/generate_cross_surface_bundle_fixtures.py

This writes ``py_composition_bundle.bin`` and ``py_composition_bundle.identity.json``
into ``conformance/cross_surface/`` for the Rust consumer in
``crates/antecedent-design/tests/cross_surface_bundle.rs``. The Rust consumer never
trusts the identity file: it rebuilds the bundle from constants and only
cross-checks the file. ``test_cross_surface_bundle.py`` imports the builders here so
the Python side declares the bundle in exactly one place.

The bundle holds two enumerated decisions whose values are derived by hand:

* joint-law decision (nodes ``contract``, ``law``, ``result``): four equally likely
  rows ``p = [1, 3, 2, 0]``, ``q = [4, 0, 2, 6]``, ``safe = 3``, so
  ``E[p * q] = (4 + 0 + 4 + 0) / 4 = 2`` and ``safe = max(3, 0) = 3``;
* point-only decision (nodes ``claim``, ``mean_contract``, ``mean_result``): the
  external claim ``E[Y | do(a)] = 1 + 2a`` on ``a = 0, 1, 2`` gives ``[1, 3, 5]``; with
  utility ``2 * mean - 1`` the action ``wait`` (``a = 0``) is worth ``2 * 1 - 1 = 1`` and
  ``treat`` (``a = 2``) is worth ``2 * 5 - 1 = 9``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from antecedent import composition_bundle as cb  # noqa: E402
from antecedent import decision, external  # noqa: E402

import generate_cross_surface_fixtures as gen  # noqa: E402

FIXTURE_DIR = gen.FIXTURE_DIR
PY_BUNDLE = "py_composition_bundle.bin"
PY_IDENTITY = "py_composition_bundle.identity.json"
RUST_BUNDLE = "rust_composition_bundle.bin"

PY_COMMAND = "python python/tests/generate_cross_surface_bundle_fixtures.py"
RUST_COMMAND = (
    "ANTECEDENT_WRITE_FIXTURES=1 cargo test -p antecedent-design --test cross_surface_bundle "
    "-- --ignored regenerate_rust_fixtures"
)

#: Hand-derived values every consumer reads from the verified result nodes.
EXPECTED_VALUES: dict[tuple[str, str], float] = {
    ("result", "risky.expected_utility"): 2.0,
    ("result", "safe.expected_utility"): 3.0,
    ("mean_result", "wait.expected_utility"): 1.0,
    ("mean_result", "treat.expected_utility"): 9.0,
}
NODE_KINDS: dict[str, str] = {
    "claim": "external_claim",
    "contract": "decision_contract",
    "law": "distribution",
    "mean_contract": "decision_contract",
    "mean_result": "decision_result",
    "result": "decision_result",
}
EDGES: tuple[tuple[str, str], ...] = (
    ("claim", "mean_result"),
    ("contract", "result"),
    ("law", "result"),
    ("mean_contract", "mean_result"),
)


def mean_contract(claim: external.BoundExternalClaim | None = None) -> decision.Contract:
    """``wait`` reads ``a = 0`` and ``treat`` reads ``a = 2``; utility ``2 * mean - 1``."""
    bound = claim if claim is not None else gen.external_claim()
    q = bound.quantities
    return decision.Contract(
        actions=(
            decision.Action("wait", inputs=(q[0],), utility=decision.x(0) * 2.0 - 1.0),
            decision.Action("treat", inputs=(q[2],), utility=decision.x(0) * 2.0 - 1.0),
        ),
        utility_units="utility",
        criterion=decision.Criterion.expected_utility(),
        target_population="target",
    )


def bundle_parts() -> dict[str, bytes]:
    """Exported container bytes of every node, keyed by node id."""
    contract = gen.decision_contract()
    law = gen.decision_source()
    claim = gen.external_claim()
    means = mean_contract(claim)
    return {
        "contract": contract.export(artifact_id="contract"),
        "law": law.export("law"),
        "result": contract.evaluate(law).export(artifact_id="result"),
        "claim": claim.export(artifact_id="claim"),
        "mean_contract": means.export(artifact_id="mean-contract"),
        "mean_result": cb.mean_decision(means, claim, artifact_id="mean-result"),
    }


def build_bundle() -> cb.Bundle:
    """The bundle both surfaces declare: two decisions, each beneath its sources."""
    parts = bundle_parts()
    builder = cb.Bundle.builder()
    for node_id, kind in NODE_KINDS.items():
        builder.add_artifact(kind, parts[node_id], node_id=node_id)
    for upstream, dependent in EDGES:
        builder.connect(upstream, dependent)
    return builder.build()


def identity_document(bundle: cb.Bundle) -> bytes:
    """The cross-check document: never trusted for loading, only compared."""
    document = {
        "bundle_identity": bundle.identity,
        "claim_label": "point_only_attested",
        "edges": [list(edge) for edge in EDGES],
        "node_kinds": NODE_KINDS,
        "values": {f"{node}.{key}": value for (node, key), value in EXPECTED_VALUES.items()},
    }
    return (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    bundle = build_bundle()
    consumed = cb.consume_bundle(bundle.export(), expected_identity=bundle.identity)
    consumed.require_verified()
    for (node, key), want in EXPECTED_VALUES.items():
        got = consumed.value(node, key)
        assert got is not None
        assert abs(got - want) < 1e-9, (node, key, got, want)
    assert consumed.claim_label == "point_only_attested"

    files = {
        PY_BUNDLE: bundle.export(artifact_id="composition-bundle"),
        PY_IDENTITY: identity_document(bundle),
    }
    for name, data in files.items():
        (FIXTURE_DIR / name).write_bytes(data)
        print(f"wrote {FIXTURE_DIR / name} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
