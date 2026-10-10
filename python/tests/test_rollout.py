"""Finite source state/terminal decision/ranking producer-consumer acceptance."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from antecedent import composition_bundle as cb
from antecedent import decision
from antecedent import design as dr
from antecedent.errors import CausalSerializationError, CausalValueError
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)

PIN = json.loads(
    (
        Path(__file__).resolve().parents[2] / "conformance/composition/rollout/expected.json"
    ).read_text(encoding="utf-8")
)


def quantity(name: str = "theta", functional: str = "state") -> ScientificQuantity:
    return ScientificQuantity(
        name, name, "outcome", "dimensionless", "target", "do(t=1)", 0, functional
    )


def law(**kwargs: object) -> JointDistributionArtifact:
    identity = DistributionIdentity(
        "interventional_predictive",
        (quantity(), quantity("other")),
        "joint",
        "finite-law",
        "external-lab",
        "deterministic_exact",
        "law-1",
        "checked-source",
    )
    return JointDistributionArtifact(
        identity,
        np.array([[0.25, 1.0], [0.75, 2.0]], dtype=np.float64),
        calibration="unmeasured",
        trust="external_attested",
        **kwargs,
    )  # type: ignore[arg-type]


def terminal(*, slope: float = 1.0) -> dr.DesignDecision:
    return dr.DesignDecision(
        "bet-terminal",
        (dr.ActionUtility("wait", 0.0, 0.0), dr.ActionUtility("bet", -0.5, slope)),
        dr.StatePrior.draws(PIN["state_draws"]),
        "utility",
    )


def ranking(
    source: JointDistributionArtifact, problem: dr.DesignDecision | None = None
) -> dr.DesignRankingResult:
    return dr.rank_designs(
        [dr.Candidate("sample", 2, dr.BinomialSignal(), 0.0, "utility")],
        decision=terminal() if problem is None else problem,
        signal=dr.SignalSpec("law-1", quantity(), quantity("signal"), ("law-1",), rng_seed=3),
        source_digests=(decision.source_digest(source),),
        rng_seed=3,
    )


def fixture() -> tuple[JointDistributionArtifact, dr.DesignRankingResult, dr.RolloutResult]:
    source = law()
    ranked = ranking(source)
    rollout = terminal().export_rollout(
        source, quantity(), ranked, interpretation="interventional_state"
    )
    return source, ranked, rollout


def test_rollout_public_source_prior_terminal_and_ranking_match_independent_truth() -> None:
    _, ranked, rollout = fixture()
    assert ranked.prior_expected_utility == pytest.approx(PIN["prior_expected_utility"], abs=1e-12)
    assert ranked.evpi == pytest.approx(PIN["evpi"], abs=1e-12)
    assert ranked.candidates[0].evsi == pytest.approx(PIN["evsi"], abs=1e-12)
    assert ranked.candidates[0].rank == PIN["rank"]
    assert rollout.source_trust == "external_attested"
    assert rollout.calibration == PIN["calibration"]
    assert rollout.native_execution_authority is False
    expected = rollout.expectation()
    assert expected["decision"]["prior"]["draws"]["states"] == PIN["state_draws"]
    assert expected["decision"]["utility"]["table"]["rows"] == PIN["utilities"]
    assert expected["decision"]["action_ids"] == PIN["terminal_actions"]
    consumed = dr.consume_rollout(rollout.export(), expected_identity=expected)
    assert consumed.identity == rollout.identity
    assert consumed.ranking_identity == ranked.identity
    assert consumed.native_execution_authority is False


def test_rollout_refuses_changed_prior_action_utility_and_original_ranking() -> None:
    source, ranked, _ = fixture()
    changed = terminal(slope=2.0)
    with pytest.raises(CausalValueError, match="terminal_decision_mismatch"):
        changed.export_rollout(source, quantity(), ranked, interpretation="interventional_state")
    new_ranking = ranking(source, changed)
    with pytest.raises(CausalValueError, match="terminal_decision_mismatch"):
        terminal().export_rollout(
            source, quantity(), new_ranking, interpretation="interventional_state"
        )
    reversed_prior = dr.DesignDecision(
        "bet-terminal", terminal().actions, dr.StatePrior.draws([0.75, 0.25]), "utility"
    )
    with pytest.raises(CausalValueError, match="ordered_prior_mismatch"):
        reversed_prior.export_rollout(
            source, quantity(), ranked, interpretation="interventional_state"
        )


@pytest.mark.parametrize(
    "field", ["state", "trust", "calibration", "admissible", "utility", "source", "ranking"]
)
def test_rollout_consumer_requires_full_independent_scientific_expectation(field: str) -> None:
    _, _, rollout = fixture()
    expected = copy.deepcopy(rollout.expectation())
    if field == "state":
        expected["state"]["units"] = "another-unit"
    elif field in ("trust", "calibration"):
        expected["source"][field] = "native_licensed" if field == "trust" else "measured"
    elif field == "admissible":
        expected["decision"][field][0] = False
    elif field == "utility":
        expected["decision"][field]["table"]["rows"][1][0] = 999.0
    elif field == "source":
        expected["source"]["identity"]["provider_id"] = "another-provider"
    else:
        expected["ranking_identity"] = "another-ranking"
    with pytest.raises(CausalValueError, match="expected_binding_mismatch"):
        dr.consume_rollout(rollout.export(), expected_identity=expected)


def test_rollout_unsupported_state_meaning_support_weighting_and_missing_source_refuse() -> None:
    source, ranked, _ = fixture()
    with pytest.raises(CausalValueError, match="state_interpretation_mismatch"):
        terminal().export_rollout(source, quantity(), ranked)
    for source in [law(weights=(0.5, 0.5)), law(supported=(False, True))]:
        with pytest.raises((CausalValueError, CausalSerializationError)):
            terminal().export_rollout(
                source, quantity(), ranking(source), interpretation="interventional_state"
            )
    with pytest.raises(CausalValueError, match="state_coordinate_mismatch"):
        terminal().export_rollout(
            law(), quantity("not-present"), ranked, interpretation="interventional_state"
        )


def test_rollout_bundle_verifies_law_rollout_ranking_and_original_source_provider() -> None:
    source, ranked, rollout = fixture()
    builder = cb.Bundle.builder()
    builder.add_artifact("distribution", source.export("law"), node_id="law")
    builder.add_artifact("rollout", rollout.export(), node_id="rollout")
    builder.add_artifact("study_ranking", ranked.export(), node_id="ranking")
    builder.connect("law", "rollout").connect("law", "ranking").connect("rollout", "ranking")
    bundle = builder.build()
    consumed = cb.consume_bundle(bundle.export(), expected_identity=bundle.identity)
    assert consumed.all_verified
    assert next(node for node in bundle.nodes if node.id == "rollout").kind == "rollout"


def test_rollout_fresh_process_independently_reads_original_source_and_full_expectation(
    tmp_path: Path,
) -> None:
    _, _, rollout = fixture()
    artifact = tmp_path / "rollout.bin"
    expected = tmp_path / "expected.json"
    artifact.write_bytes(rollout.export())
    expected.write_text(json.dumps(rollout.expectation()), encoding="utf-8")
    script = """
import json, sys
from pathlib import Path
from antecedent import design as dr
value = dr.consume_rollout(Path(sys.argv[1]).read_bytes(), expected_identity=json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
assert value.native_execution_authority is False
print(json.dumps({'identity':value.identity, 'trust':value.source_trust, 'calibration':value.calibration}))
"""
    output = subprocess.run(
        [sys.executable, "-c", script, str(artifact), str(expected)],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    result = json.loads(output.stdout)
    assert result == {
        "identity": rollout.identity,
        "trust": "external_attested",
        "calibration": "unmeasured",
    }


def test_rollout_truncated_and_oversized_artifacts_refuse_before_replay() -> None:
    _, _, rollout = fixture()
    with pytest.raises(CausalSerializationError):
        dr.consume_rollout(rollout.export()[:100], expected_identity=rollout.expectation())
    with pytest.raises(CausalValueError):
        dr.consume_rollout(bytes(32 * 1024 * 1024 + 1), expected_identity=rollout.expectation())


def test_rollout_bundle_refuses_new_valid_ranking_with_changed_terminal_utilities() -> None:
    source, _, rollout = fixture()
    changed = ranking(source, terminal(slope=2.0))
    builder = cb.Bundle.builder()
    builder.add_artifact("distribution", source.export("law"), node_id="law")
    builder.add_artifact("rollout", rollout.export(), node_id="rollout")
    builder.add_artifact("study_ranking", changed.export(), node_id="ranking")
    builder.connect("law", "rollout").connect("law", "ranking").connect("rollout", "ranking")
    bundle = builder.build()
    consumed = cb.consume_bundle(bundle.export(), expected_identity=bundle.identity)
    assert not consumed.all_verified


def test_rollout_native_byte_budget_refuses_before_owned_copy_and_parse() -> None:
    from antecedent import _native

    _, _, rollout = fixture()
    expected = json.dumps(rollout.expectation())
    with pytest.raises(ValueError, match="exceeds its byte bound"):
        _native.consume_rollout(bytes(32 * 1024 * 1024 + 1), expected)
    with pytest.raises(ValueError, match="exceeds its byte bound"):
        _native.export_rollout(bytes(16 * 1024 * 1024 + 1), b"", "{}", "rollout")
