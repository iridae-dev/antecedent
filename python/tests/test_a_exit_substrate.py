"""2.3 A exit gate, box 2: the F-substrate is used by the A scientific paths.

One NATIVE result and one EXTERNAL result are each shown to carry typed quantities
(``ScientificQuantity`` coordinates), per-coordinate support, a trust label, provenance and a
portable artifact that a fresh interpreter consumes without the producer's objects.

Native results exercised
------------------------
* ``program_claims.native_claim`` (C1 over the A response path): typed coordinates, per-coordinate
  support (the dose-2 coordinate is outside empirical support and is *withheld*), trust
  ``native_licensed``, calibration ``point_only`` and a provenance id. Consumed downstream by the
  support-aware decision evaluation (``composition.evaluate_with_support``).
* The A joint-draw artifact (F1/F15), committed as ``conformance/cross_surface/py_joint_law.bin``:
  typed quantities, trust ``unverified``, calibration ``exact``, identity provenance
  (source/provider/snapshot/contract ids); consumed in a fresh process under constants.
* The A CPDAG-completion report (X2), exported and consumed in a fresh process: counts, per
  completion status and the structural envelope survive the artifact.

External result exercised
-------------------------
* ``external.response`` bound claim (F3), the Python-built and the Rust-built committed
  fixtures: typed quantities, per-coordinate support, trust ``externally_attested``, a Merkle
  lineage, consumed in a fresh process under an identity written from constants.

Hand-derived values: E[Y | do(a)] = 1 + 2a on a = 0, 1, 2 is (1, 3, 5); the native mean response is
(2, 4, 9) with dose 2 unsupported, so ``wait`` is worth 2, ``treat`` is 4 - 1 = 3 and ``extend``
has no supported coordinate. The joint law of two aligned draws (0, 0), (1, 2) has means (1/2, 1)
and covariance 1/2. The CPDAG chain a - b - c has three completions with ``P(c=1 | do(b=1))`` equal
to 0.8, 0.8 and 0.38.

Gaps (asserted only as absences, not worked around): the native response claim has no artifact
and no lineage chain (only ``provenance_id``); the CPDAG and temporal results carry no
``ScientificQuantity`` coordinates, trust label or provenance chain of their own; the joint draw
artifact carries no per-coordinate support unless a ``supported`` mask is supplied.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from antecedent import Cpdag
from antecedent import composition as comp
from antecedent.extensibility import ProviderTrust
from antecedent.joint_distribution import JointDistributionArtifact
from antecedent.transport import advanced as transport

import _c4_fixtures as fx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_cross_surface_fixtures as gen  # noqa: E402

TESTS_DIR = str(Path(__file__).resolve().parent)

JOINT_SCRIPT = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    import generate_cross_surface_fixtures as gen
    from antecedent.joint_distribution import JointDistributionArtifact

    loaded = JointDistributionArtifact.load(
        open(sys.argv[2], "rb").read(), expected_identity=gen.joint_law_identity()
    )
    ident = loaded.identity
    print(json.dumps({
        "means": [loaded.mean(0), loaded.mean(1)],
        "covariance": loaded.covariance(0, 1),
        "trust": loaded.trust,
        "calibration": loaded.calibration,
        "semantic": loaded.semantic,
        "regimes": [q.regime_id for q in loaded.quantities],
        "units": sorted({q.units for q in loaded.quantities}),
        "provenance": [ident.source_id, ident.provider_id, ident.snapshot_id,
                       ident.causal_contract_id],
        "supported": loaded.supported,
    }))
    """
)

EXTERNAL_SCRIPT = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    import generate_cross_surface_fixtures as gen

    claim = gen.external_spec().load(
        open(sys.argv[2], "rb").read(), expected=json.loads(sys.argv[3])
    )
    print(json.dumps({
        "values": [float(v) for v in claim.values],
        "native": claim.native,
        "trust": claim.trust.value,
        "support": list(claim.support),
        "support_status": claim.support_status,
        "regimes": [q.regime_id for q in claim.quantities],
        "units": sorted({q.units for q in claim.quantities}),
        "label": claim.provenance_label,
        "lineage": [link.id for link in claim.lineage],
        "stages": sorted(claim.stages_behind()),
    }))
    """
)

CPDAG_SCRIPT = textwrap.dedent(
    """
    import json, sys
    from antecedent.transport import advanced as transport

    consumed = transport.consume_cpdag_scenarios_artifact(open(sys.argv[1], "rb").read())
    counts = consumed.counts
    print(json.dumps({
        "counts": [counts.identified, counts.unidentified, counts.unevaluated,
                   counts.not_enumerated, counts.total],
        "statuses": sorted(c.status for c in consumed.completions),
        "range": list(consumed.envelope.mean_range("c")),
        "identity_len": len(consumed.cpdag_identity),
        "digests": bool(consumed.premises_digest) and bool(consumed.data_digest),
    }))
    """
)


def fresh(script: str, *args: str) -> dict[str, Any]:
    done = subprocess.run(
        [sys.executable, "-c", script, *args], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    loaded: dict[str, Any] = json.loads(done.stdout)
    return loaded


# ------------------------------------------------------------------------ native response


def test_a_exit_substrate_native_claim_carries_quantities_support_trust_and_provenance() -> None:
    claim = fx.native_claim()
    expected = tuple(fx.coordinate(dose) for dose in fx.GRID)
    # Typed quantities: written by hand, independent of the production derivation.
    assert claim.coordinates == expected
    assert {q.units for q in claim.coordinates} == {"mmHg"}
    assert [q.regime_id for q in claim.coordinates] == ["do(a=0)", "do(a=1)", "do(a=2)"]
    assert claim.means == pytest.approx((2.0, 4.0, 6.0), abs=1e-9)
    # Coordinate-level support, not one pooled flag.
    assert claim.support == ("supported", "supported", fx.OUTSIDE)
    assert claim.support_status == fx.OUTSIDE
    # Trust and provenance.
    assert claim.trust is ProviderTrust.NATIVE_LICENSED
    assert claim.calibration in {"point_only", "unmeasured"}
    assert claim.provenance_id == fx.native_view().provenance["operation_id"]
    assert claim.program_identity == fx.program().identity
    assert len(claim.program_identity) == 64
    # The support travels into the decision input: the unsupported coordinate is withheld.
    source = claim.as_decision_source(fx.mean_contract())
    assert [w.index for w in source.withheld] == [2]
    assert source.withheld[0].coordinate == expected[2]
    assert source.withheld[0].status == fx.OUTSIDE
    assert source.point_status == ("supported", "supported")
    assert source.trust is ProviderTrust.NATIVE_LICENSED
    # And into the decision: wait 2, treat 4 - 1 = 3, extend has no supported coordinate.
    decided = comp.evaluate_with_support(fx.mean_contract(), [fx.native_input()])
    assert fx.near(decided.outcome("wait").expected_utility, 2.0)
    assert fx.near(decided.outcome("treat").expected_utility, 3.0)
    assert decided.verdict.selected == "treat"
    assert decided.unsupported_actions == ("extend",)
    # Gap, asserted as an absence: the native claim exports no artifact of its own.
    assert not hasattr(claim, "export")


# ------------------------------------------------------------------------ native artifacts


def test_a_exit_substrate_native_joint_artifact_is_consumed_in_a_fresh_process() -> None:
    path = gen.FIXTURE_DIR / "py_joint_law.bin"
    assert path.is_file(), "regenerate with python python/tests/generate_cross_surface_fixtures.py"
    report = fresh(JOINT_SCRIPT, TESTS_DIR, str(path))
    # Hand values: draws (0, 0), (1, 2): means (1/2, 1), population covariance 1/2.
    assert report["means"] == pytest.approx([0.5, 1.0], abs=1e-12)
    assert report["covariance"] == pytest.approx(0.5, abs=1e-12)
    assert report["trust"] == "unverified"
    assert report["calibration"] == "exact"
    assert report["semantic"] == "interventional_predictive"
    assert report["regimes"] == ["do(a=1)", "do(a=1)"]
    assert report["units"] == ["units"]
    assert report["provenance"] == [
        "enumerated-law",
        "exact-law",
        "law-snapshot",
        "checked-contract",
    ]
    # Gap: no per-coordinate support unless the producer supplied a mask.
    assert report["supported"] is None
    # The same bytes in this process give the same numbers (producer-independent consumption).
    local = JointDistributionArtifact.load(
        path.read_bytes(), expected_identity=gen.joint_law_identity()
    )
    assert local.mean(1) == pytest.approx(report["means"][1], abs=1e-12)


def _chain_completions() -> transport.CpdagScenarioResult:
    names = ["a", "b", "c"]

    def joint(a: int, b: int, c: int) -> float:
        def p(one: bool, p_one: float) -> float:
            return p_one if one else 1.0 - p_one

        return p(a == 1, 0.4) * p(b == 1, (0.2, 0.7)[a]) * p(c == 1, (0.1, 0.8)[b])

    table = tuple(joint(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1))
    axes = tuple((n, (0.0, 1.0)) for n in names)
    graph = Cpdag.from_directed_undirected(names, [], [("a", "b"), ("b", "c")])
    return transport.cpdag_completion_scenarios(
        graph,
        outcomes=["c"],
        treatments=["b"],
        source="src_pop",
        target="target",
        coordinates=[transport.VariableCoordinate(n, "binary") for n in names],
        evidence=transport.CompletionEvidence(
            transport.EvidenceCatalog(
                regimes=[transport.EvidenceRegime("obs", "target", measured=names)]
            ),
            "target-law",
        ),
        laws=transport.ExactTransportData(
            (transport.ExactDiscreteLaw("target", "obs", axes, table, "target"),)
        ),
        at={"b": 1.0},
    )


def test_a_exit_substrate_native_cpdag_report_is_consumed_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    produced = _chain_completions()
    artifact = tmp_path / "cpdag.bin"
    artifact.write_bytes(produced.export())
    report = fresh(CPDAG_SCRIPT, str(artifact))
    # By hand: a->b->c and a<-b->c give P(c=1 | b=1) = 0.8 (c is independent of a given b);
    # a<-b<-c gives P(c=1) = P(b=0) * 0.1 + P(b=1) * 0.8 = 0.6 * 0.1 + 0.4 * 0.8 = 0.38, with
    # P(b=1) = 0.6 * 0.2 + 0.4 * 0.7 = 0.4. The structural envelope is [0.38, 0.8].
    assert report["counts"] == [3, 0, 0, 0, 3]
    assert report["statuses"] == ["identified"] * 3
    assert report["range"] == pytest.approx([0.38, 0.8], abs=1e-12)
    assert report["identity_len"] == 64
    assert report["digests"] is True
    # Gap, asserted as absences: this report carries counts and statuses, but no typed
    # ScientificQuantity coordinate, trust label or provenance chain of its own.
    for absent in ("trust", "lineage", "quantities", "provenance"):
        assert not hasattr(produced, absent), absent


# ------------------------------------------------------------------------ external result


@pytest.mark.parametrize("name", ["py_external_claim.bin", "rust_external_claim.bin"])
def test_a_exit_substrate_external_claim_carries_the_substrate_through_a_fresh_process(
    name: str,
) -> None:
    path = gen.FIXTURE_DIR / name
    assert path.is_file(), f"missing committed fixture {path}"
    identity = gen.external_claim().identity
    report = fresh(EXTERNAL_SCRIPT, TESTS_DIR, str(path), json.dumps(identity))
    # E[Y | do(a)] = 1 + 2a on a = 0, 1, 2.
    assert report["values"] == pytest.approx([1.0, 3.0, 5.0], abs=1e-12)
    assert report["native"] is False
    assert report["trust"] == "externally_attested"
    assert report["support"] == list(gen.SUPPORT)
    assert report["support_status"] == "outside_empirical_support"
    assert report["regimes"] == ["do(a=0)", "do(a=1)", "do(a=2)"]
    assert report["units"] == ["mmHg"]
    assert report["label"] == "external:lab/curve@v3#snap-9"
    assert report["lineage"] == [
        "contract:checked-contract",
        "evidence:factor:z",
        "provider:external:lab/curve@v3#snap-9",
        "claim",
    ]
    assert {"causal_contract", "evidence", "external_provider"} <= set(report["stages"])


def test_a_exit_substrate_external_claim_refuses_a_foreign_identity_in_a_fresh_process() -> None:
    path = gen.FIXTURE_DIR / "py_external_claim.bin"
    changed = {**gen.external_claim().identity, "snapshot_id": "other-snapshot"}
    done = subprocess.run(
        [sys.executable, "-c", EXTERNAL_SCRIPT, TESTS_DIR, str(path), json.dumps(changed)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode != 0
    assert "differs" in done.stderr or "verification receipt" in done.stderr, done.stderr
