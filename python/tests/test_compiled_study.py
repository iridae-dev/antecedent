"""A saved prepared study refits its estimator in an independent process."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import antecedent as ant
import pytest
from antecedent import compiled_study


def test_saved_study_replays_its_estimator_in_a_fresh_process(tmp_path: Path) -> None:
    treatment = [0.0, 1.0] * 32
    covariate = [index / 32.0 for index in range(32) for _ in range(2)]
    outcome = ant.prepare(
        {
            "t": treatment,
            "y": [1.0 + 3.0 * t + z for t, z in zip(treatment, covariate, strict=True)],
            "z": covariate,
        },
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=0),
        refute=False,
        execution=ant.Execution(seed=37, threads=1),
    ).estimate()
    assert outcome.refusal is None
    assert outcome.answer.value == pytest.approx(3.0)
    artifact = outcome.export()
    recorded = ant.load(artifact.data, expected=artifact.expectation)
    assert recorded.refusal is None
    assert recorded.value.claim_id == outcome.value.claim_id
    saved = outcome.study.save_compiled_study()
    path = tmp_path / "compiled-study.causal"
    path.write_bytes(saved.data)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """from pathlib import Path
import sys
from antecedent.compiled_study import open_study
reopened = open_study(Path(sys.argv[1]).read_bytes(), sys.argv[2])
assert reopened.resumable
assert reopened.layer5_bytes > 0
assert reopened.claim_intact
assert reopened.estimate == reopened.claim_estimate
print(reopened.estimate)
""",
            str(path),
            saved.expected_identity,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert float(child.stdout) == outcome.answer.value
    assert float(child.stdout) != sum(treatment) / len(treatment)


def test_an_opaque_invariant_without_bodies_is_not_resumable() -> None:
    saved = compiled_study.write(b"opaque", ())
    reopened = compiled_study.open_study(saved.data, saved.expected_identity)
    assert not reopened.resumable
    assert reopened.estimate is None


def test_exact_fit_keeps_its_point_claim_on_load_and_resume() -> None:
    outcome = ant.analyze(
        {"a": [0.0, 0.0, 1.0, 1.0], "y": [0.0, 0.0, 2.0, 2.0]},
        graph=[("a", "y")],
        query=ant.AverageEffect("a", "y"),
        inference=ant.Frequentist(bootstrap=0),
        refute=False,
    )
    artifact = outcome.export()
    recorded = ant.load(artifact.data, expected=artifact.expectation)
    assert recorded.refusal is None
    assert recorded.answer.value == pytest.approx(2.0)
    saved = outcome.study.save_compiled_study()
    resumed = compiled_study.open_study(saved.data, saved.expected_identity)
    assert resumed.resumable
    assert resumed.estimate == resumed.claim_estimate == recorded.answer.value
