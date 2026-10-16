"""Family envelopes preserve consumed types and reject misspelled arguments."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_checked_artifact_envelopes_preserve_family_types(tmp_path):
    fixture = tmp_path / "artifact_types.py"
    # The changed fixture is checked again; unchanged imported package types
    # can reuse their analysis without a shared repository cache.
    mypy = [sys.executable, "-m", "mypy", "--cache-dir", str(tmp_path / "mypy-cache")]
    fixture.write_text("""from typing import assert_type
import antecedent as ant
from antecedent.artifacts import ExportedArtifact
from antecedent import compact_export as ce, cohort_decision as cd, provider_callbacks as pc
from antecedent import sequential_evaluation as se, selective_discovery as sd
from antecedent import adaptive_inference as ai, cross_world_execution as cw

def checked(compact: ce.CompactExport, cohort: cd.CohortDecisionSource,
            callback: pc.CallbackExecution, sequential: se.EvaluationArtifact,
            discovery: sd.SelectiveDiscovery, adaptive: ai.AdaptiveInferenceSession,
            crossworld: cw.CrossWorldExecution) -> None:
    assert_type(compact.artifact(), ExportedArtifact[ce.CompactExport])
    assert_type(ant.load(compact.artifact(), expected=compact.identity), ce.CompactExport)
    assert_type(cohort.artifact(), ExportedArtifact[cd.CohortDecisionSource])
    assert_type(ant.load(cohort.artifact(), expected=cohort.identity), cd.CohortDecisionSource)
    assert_type(callback.artifact(), ExportedArtifact[pc.CallbackConsumed])
    assert_type(ant.load(callback.artifact()), pc.CallbackConsumed)
    assert_type(sequential.artifact(), ExportedArtifact[se.EvaluationArtifact])
    assert_type(ant.load(sequential.artifact()), se.EvaluationArtifact)
    assert_type(discovery.artifact(), ExportedArtifact[sd.ConsumedSelectiveDiscovery])
    assert_type(adaptive.artifact(ai.BayesianPosterior(1)), ExportedArtifact[ai.ConsumedAdaptiveResult])
    assert_type(adaptive.analyze(ai.BayesianPosterior(1)).artifact(), ExportedArtifact[ai.ConsumedAdaptiveResult])
    assert_type(crossworld.artifact(), ExportedArtifact[cw.ConsumedCrossWorldExecution])
""")
    result = subprocess.run(
        [*mypy, str(fixture)],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    fixture.write_text("""from antecedent import mechanism_composition as mc
from antecedent import scientific_bridge as sb

def unchecked(model: mc.PreparedMechanismComposition, bridge: sb.BridgedClaim) -> None:
    model.artifact(query_typo=mc.MeanContrast.mean_at("y", {}))
    bridge.artifact(artifact_typo="id")
""")
    result = subprocess.run(
        [*mypy, str(fixture)],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert 'Unexpected keyword argument "query_typo"' in result.stdout
    assert 'Unexpected keyword argument "artifact_typo"' in result.stdout
