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
from antecedent.outcomes import Outcome, require_success
from antecedent import compact_export as ce, cohort_decision as cd, provider_callbacks as pc
from antecedent import sequential_evaluation as se, selective_discovery as sd
from antecedent import adaptive_inference as ai, cross_world_execution as cw

def checked(compact: ce.CompactExport, cohort: cd.CohortDecisionSource,
            callback: pc.CallbackExecution, sequential: se.EvaluationArtifact,
            discovery: sd.SelectiveDiscovery, adaptive: ai.AdaptiveInferenceSession,
            crossworld: cw.CrossWorldExecution) -> None:
    assert_type(compact.export(), ExportedArtifact[ce.CompactExport])
    assert_type(ant.load(compact.export(), expected=compact.identity), Outcome[ce.CompactExport])
    assert_type(cohort.export(), ExportedArtifact[cd.CohortDecisionSource])
    assert_type(ant.load(cohort.export(), expected=cohort.identity), Outcome[cd.CohortDecisionSource])
    assert_type(callback.export(), ExportedArtifact[pc.CallbackConsumed])
    assert_type(ant.load(callback.export()), Outcome[pc.CallbackConsumed])
    assert_type(sequential.export(), ExportedArtifact[se.EvaluationArtifact])
    assert_type(ant.load(sequential.export()), Outcome[se.EvaluationArtifact])
    assert_type(discovery.export(), ExportedArtifact[sd.ConsumedSelectiveDiscovery])
    assert_type(adaptive.export(ai.BayesianPosterior(1)), ExportedArtifact[ai.ConsumedAdaptiveResult])
    assert_type(require_success(adaptive.analyze(ai.BayesianPosterior(1))).export(), ExportedArtifact[ai.ConsumedAdaptiveResult])
    assert_type(crossworld.export(), ExportedArtifact[cw.ConsumedCrossWorldExecution])
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
    model.export(query_typo=mc.MeanContrast.mean_at("y", {}))
    bridge.export(artifact_typo="id")
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
