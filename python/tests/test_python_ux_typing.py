"""Both import styles retain checked constructors and method-specific results."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_ux_types_through_lazy_root_and_explicit_imports(tmp_path):
    fixture = tmp_path / "positive.py"
    fixture.write_text("""from typing import assert_type
import antecedent as ant
from antecedent import adaptive_inference as ai
from antecedent import scientific_bridge as sb
from antecedent.artifacts import ExportedArtifact
from antecedent.results import MeanSource

def checked(session: ai.AdaptiveInferenceSession, claim: sb.BridgedClaim) -> None:
    posterior = session.analyze(ai.BayesianPosterior(1))
    assert_type(posterior, ai.AdaptiveResult[ai.PosteriorSummary])
    assert_type(posterior.payload.arms[0].mean, float)
    sequential = session.analyze(ant.adaptive_inference.GroupSequential(2))
    assert_type(sequential, ai.AdaptiveResult[ai.SequentialSummary])
    assert_type(sequential.payload.stopped_at, int | None)
    artifact = claim.artifact()
    assert_type(artifact, ExportedArtifact[sb.ConsumedBridge])
    assert_type(ant.load(artifact, expected=artifact.expectation), sb.ConsumedBridge)
    assert_type(claim.as_mean_source(), MeanSource)
    ant.transport.advanced.estimate_augmented_response_grid
""")
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-incremental", str(fixture)],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    fixture.write_text("""import antecedent as ant
from antecedent import adaptive_inference as ai
ant.adaptive_inference.BayesianPosterior(look_typo=1)
ai.BayesianPosterior(look_typo=1)
bad: ai.StoppingName = "invented"
""")
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-incremental", str(fixture)],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert result.stdout.count('Unexpected keyword argument "look_typo"') == 2, result.stdout
    assert "incompatible" in result.stdout.lower(), result.stdout
