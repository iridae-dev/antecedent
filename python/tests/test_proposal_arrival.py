"""Actual proposal-arrival counts through the original checked transport evaluator."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from antecedent import Admg, proposals, repair
from antecedent.errors import CausalSerializationError, CausalValueError
from antecedent.proposal_arrival import ArrivedStudy, consume_arrival
from antecedent.transport import advanced as tr

from test_proposals import experiment, rank, semantic_ids, transport_contract

EXPECTED = json.loads(
    (Path(__file__).resolve().parents[2] / "conformance/proposals/arrival/expected.json").read_text(
        encoding="utf-8"
    )
)


def fixture(family="transport"):
    base = transport_contract()
    graph = base.graph
    contract = base
    if family == "z_transport":
        graph = Admg.from_edges(["x", "y"], [("x", "y")], [("x", "y")])
        identified = tr.identify_z_transport(
            graph=graph,
            query=tr.ZTransportQuery(
                tr.SelectionDiagram("source", "target", ["x"]),
                outcomes=["y"],
                treatments=["x"],
                controllable=["x"],
                experiment_assignment={},
            ),
        )
        contract = repair.ZTransportContract.from_identification(
            identified, catalog=base.catalog, names=["x", "y"]
        )
    candidate = experiment()
    repaired = repair.repair(contract, [candidate])
    ranked = rank(repaired, [candidate], {candidate.label: 0.75})
    bundle = proposals.ProposalBundle.build(repaired.export(), ranked.export(), contract=contract)
    identifier = semantic_ids(repaired)[candidate.label]
    catalog = tr.EvidenceCatalog(
        environments=base.catalog.environments,
        regimes=(
            tr.EvidenceRegime(
                "arrived", "source", kind="experimental", interventions=("x",), measured=("y",)
            ),
        ),
        bindings=(
            tr.RegimeBinding(
                "arrived",
                "trial-snapshot",
                schema_names=("y",),
                sampling="independent",
                dependence="independent_studies",
            ),
        ),
    )
    laws = tuple(
        tr.ExactDiscreteLaw(
            "source",
            "arrived",
            (("y", (0.0, 1.0)),),
            probabilities,
            "trial-snapshot",
            interventions=(("x", active),),
            empirical_counts=counts,
        )
        for active, probabilities, counts in (
            (0.0, (0.8, 0.2), (200, 50)),
            (1.0, (0.2, 0.8), (50, 200)),
        )
    )
    return bundle, identifier, ArrivedStudy(graph, catalog, laws, {"x": 1.0})


@pytest.mark.parametrize("family", ["transport", "z_transport"])
def test_arrival_public_actual_count_truth_and_original_artifact_replay(family):
    bundle, identifier, study = fixture(family)
    result = bundle.estimate_arrival(identifier, study)
    assert result.mean == pytest.approx(EXPECTED["active_mean"])
    assert result.to_dict()["point"]["probabilities"] == pytest.approx([0.2, 0.8])
    assert result.to_dict()["sample_size"] == EXPECTED["python_sample_size"]
    assert result.to_dict()["inference"] == "empirical_point_only"
    assert result.to_dict()["calibration"] == "unmeasured"
    assert result.to_dict()["work"]["program_compilations"] > 0
    assert result.to_dict()["work"]["provider_calls"] > 0
    replay = consume_arrival(result.export(), expected_proposal=bundle.identity, seed=77)
    assert replay.to_dict() == result.to_dict()
    control = bundle.estimate_arrival(identifier, replace(study, assignment={"x": 0.0}))
    assert control.mean == pytest.approx(EXPECTED["control_mean"])
    assert result.mean - control.mean == pytest.approx(EXPECTED["contrast"])


def test_arrival_fresh_process_replays_original_artifacts_and_complete_raw_providers(tmp_path):
    bundle, identifier, study = fixture()
    result = bundle.estimate_arrival(identifier, study)
    artifact = tmp_path / "arrival.bin"
    artifact.write_bytes(result.export())
    script = """
import json, sys
from pathlib import Path
from antecedent.proposal_arrival import consume_arrival
value=consume_arrival(Path(sys.argv[1]).read_bytes(), expected_proposal=sys.argv[2], seed=19)
print(json.dumps(value.to_dict(), sort_keys=True))
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(artifact), bundle.identity],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(child.stdout) == result.to_dict()


@pytest.mark.parametrize(
    "change", ["counts", "snapshot", "population", "assignment", "exact", "budget", "schema"]
)
def test_arrival_public_counts_identity_scope_and_sample_refusals(change):
    bundle, identifier, study = fixture()
    laws = list(study.arrived_laws)
    if change == "counts":
        laws[0] = replace(laws[0], probabilities=(0.5, 0.5), empirical_counts=(25, 25))
    elif change == "snapshot":
        laws[0] = replace(laws[0], snapshot_identity="wrong")
    elif change == "population":
        laws[0] = replace(laws[0], population="target")
    elif change == "exact":
        laws[0] = replace(laws[0], empirical_counts=None)
    elif change == "assignment":
        study = replace(study, assignment={"y": 1.0})
    elif change == "schema":
        study = replace(study, graph=Admg.from_edges(["y", "x"], [("x", "y")]))
    else:
        study = replace(study, operations=0)
    detail = {
        "counts": "proposal_arrival.sample_size_mismatch",
        "snapshot": "proposal_arrival.provider_binding_mismatch",
        "population": "exact law names an unknown population/regime",
        "assignment": "proposal_arrival.assignment_mismatch",
        "exact": "proposal_arrival.raw_counts_required",
        "budget": "proposal_arrival.bounds_exceeded",
        "schema": "repair coordinate names differ from the supplied graph",
    }[change]
    with pytest.raises(CausalValueError, match=detail) as refused:
        bundle.estimate_arrival(identifier, replace(study, arrived_laws=laws))
    assert refused.value.reason_code == (
        "route_not_supported" if change == "budget" else "invalid_argument"
    )


def test_arrival_loaded_bundle_requires_original_artifacts_and_expected_identity():
    bundle, identifier, study = fixture()
    loaded = proposals.ProposalBundle.from_dict(bundle.to_dict())
    with pytest.raises(CausalValueError, match="original_artifacts_required"):
        loaded.estimate_arrival(identifier, study)
    result = bundle.estimate_arrival(identifier, study)
    with pytest.raises(CausalValueError, match="expected_proposal_mismatch"):
        consume_arrival(result.export(), expected_proposal="other-proposal")
    with pytest.raises(CausalSerializationError):
        consume_arrival(result.export()[:-1], expected_proposal=bundle.identity)
    with pytest.raises(CausalValueError, match="bounds_exceeded"):
        consume_arrival(b"x" * (32 * 1024 * 1024 + 1), expected_proposal=bundle.identity)
