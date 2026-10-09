"""Original F18 evidence feeds actual downstream engines without inference promotion."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from antecedent.decision import Action, Contract, Criterion, const, x
from antecedent.effect_constancy_review import (
    policy_review,
    rank_prior_sources,
    transport_diagnostic,
)
from antecedent.errors import CausalValueError
from antecedent.external import ScientificQuantity
from antecedent.priors import (
    DesignVariable,
    EstimandFingerprint,
    PriorCatalog,
    PriorSource,
    PriorSourceMeta,
)
from antecedent.temporal import EffectEstimand, effect_constancy

ORACLE = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "conformance/population_time/effect_constancy_consumers/expected.json"
    ).read_text(encoding="utf-8")
)


def source(effects=(1.0, 2.0), **kwargs):
    return effect_constancy(
        [(f"p{i}", effect, 0.5) for i, effect in enumerate(effects)],
        estimand=EffectEstimand(
            "ate_difference", "outcome_units", "treat_vs_control", "all_observed_h2"
        ),
        **kwargs,
    )


def quantity():
    return ScientificQuantity(
        variable_id="y_effect",
        variable_name="Y effect",
        role="outcome",
        units="outcome_units",
        population_id="all_observed_h2",
        regime_id="treat_vs_control",
        horizon=0,
        functional_id="ate_difference",
        transform_id="identity",
    )


def contract():
    q = quantity()
    return Contract(
        actions=(
            Action("baseline", (q,), const(0) * x(0), kind="regime"),
            Action("treat", (q,), x(0), kind="intervention"),
        ),
        utility_units="outcome_units",
        criterion=Criterion.expected_utility(),
        target_population="all_observed_h2",
    )


def test_original_covariance_contrast_retains_adjustment_and_separate_theorem_requirement():
    result = source(covariance=[[0.25, 0.125], [0.125, 0.25]])
    out = transport_diagnostic(result, left="p0", right="p1")
    assert out.contrast.difference == pytest.approx(-1)
    assert out.contrast.se == pytest.approx(0.5)
    assert out.contrast.p_holm == pytest.approx(ORACLE["transport"]["holm_p_single_contrast"])
    assert out.identity == result.identity
    assert out.calibration == "unmeasured"
    assert out.separate_transport_identification_required
    with pytest.raises(CausalValueError, match="contrast_missing"):
        transport_diagnostic(result, left="p1", right="p0")


def test_original_catalog_filtering_does_not_turn_preference_into_transfer_authority():
    target = EstimandFingerprint("ate", "A", "Y")
    entries = []
    for name in ("far", "near", "wrong"):
        outcome = "Z" if name == "wrong" else "Y"
        entries.append(
            PriorSource(
                PriorSourceMeta(
                    name,
                    EstimandFingerprint("ate", "A", outcome),
                    "nonparametrically_identified",
                    design=(DesignVariable("A", "treatment"), DesignVariable(outcome, "outcome")),
                )
            )
        )
    out = rank_prior_sources(
        source((1, 2, 2.1)),
        catalog=PriorCatalog(entries),
        target=target,
        target_partition="p2",
        partitions={"far": "p0", "near": "p1", "wrong": "p2"},
    )
    assert [r.artifact_id for r in out.ranked] == ["near", "far"]
    assert dict(out.scores)["far"] == pytest.approx(ORACLE["prior"]["scores"][0])
    assert out.data_dependent_selection
    assert not out.posterior_transfer_licensed
    assert out.calibration == "unmeasured"


def test_original_point_policy_handles_disagreement_ties_and_semantic_refusals():
    out = policy_review(source((-1, 2)), contract=contract(), effect=quantity())
    assert out.common_leaders == ()
    assert [p.decision.actions[1].value for p in out.partitions] == [-1, 2]
    assert not out.generalization_guarantee
    assert all(p.decision.evpi is None for p in out.partitions)
    ties = policy_review(source((0, 0)), contract=contract(), effect=quantity())
    assert ties.common_leaders == ("baseline", "treat")
    assert ties.partitions[0].source.snapshot_id != ties.partitions[1].source.snapshot_id
    # The original mean digest represents numeric quantity/value content only.
    assert ties.partitions[0].decision.source_digest == ties.partitions[1].decision.source_digest
    assert all(a.standard_error is None for p in ties.partitions for a in p.decision.actions)
    with pytest.raises(CausalValueError, match="quantity_mismatch"):
        policy_review(source(), contract=contract(), effect=replace(quantity(), units="wrong"))
    # Mutable/reconstructed projections cannot substitute for the actual original evidence.
    projected = replace(source(), partitions=())
    assert policy_review(projected, contract=contract(), effect=quantity()).common_leaders == (
        "treat",
    )


def test_fresh_process_recomputes_original_artifact_before_policy_consumption(tmp_path):
    actual = source((-1, 2))
    artifact = tmp_path / "source.bin"
    artifact.write_bytes(actual.artifact)
    payload = tmp_path / "identity.json"
    payload.write_text(json.dumps(actual.identity._wire()), encoding="utf-8")
    script = """
import json, sys
from pathlib import Path
from antecedent.temporal import consume_effect_constancy_artifact
from antecedent.effect_constancy_review import transport_diagnostic
s = consume_effect_constancy_artifact(Path(sys.argv[1]).read_bytes(), expected=json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
o = transport_diagnostic(s, left="p0", right="p1")
print(json.dumps({"difference": o.contrast.difference, "se": o.contrast.se, "calibration": o.calibration, "separate": o.separate_transport_identification_required}))
"""
    out = json.loads(
        subprocess.run(
            [sys.executable, "-c", script, str(artifact), str(payload)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    assert out["difference"] == -3
    assert out["se"] == pytest.approx(2**-0.5)
    assert out["calibration"] == "unmeasured"
    assert out["separate"]
