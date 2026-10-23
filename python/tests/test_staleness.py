"""Staleness and drift diagnostics end to end: a prepared study, its exported baseline, and
new data with no change, a covariate shift, lost positivity and a schema change."""

from __future__ import annotations

import json

import antecedent as ant
import numpy as np
import pytest
from antecedent.drift import StalenessBaseline, StalenessPolicy, StalenessReport

GRAPH = [("z", "t"), ("z", "y"), ("t", "y")]
QUERY = ant.AverageEffect("t", "y")
POLICY = StalenessPolicy(subsample=60, permutations=99)


def sample(
    seed: int, n: int = 800, shift: float = 0.0, slope: float = 0.5
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n) + shift
    t = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-slope * z))).astype(float)
    y = 1.0 + 2.0 * t + z + rng.normal(size=n)
    return {"t": t, "y": y, "z": z}


@pytest.fixture(scope="module")
def baseline_data() -> dict[str, np.ndarray]:
    return sample(1)


@pytest.fixture(scope="module")
def study(baseline_data):
    return ant.prepare(baseline_data, graph=GRAPH, query=QUERY, bootstrap=0, refute="none")


def test_the_retained_data_is_reused(study, baseline_data):
    report = study.staleness(baseline_data, policy=POLICY)
    assert isinstance(report, StalenessReport)
    assert report.recommendation == "reuse"
    assert report.findings == ()
    assert report.changed_keys == () and report.invalidated == ()
    assert len(report.evaluations) > 5
    json.dumps(report.to_dict())


def test_a_covariate_shift_asks_for_a_refresh_naming_the_snapshot(study):
    report = study.staleness(sample(2, shift=0.35), policy=POLICY)
    assert report.recommendation == "refresh", report.to_dict()
    kinds = {finding.kind for finding in report.findings}
    assert "covariate_shift" in kinds
    shift = next(f for f in report.findings if f.subject == "z:mean")
    assert shift.threshold_name == "smd_threshold"
    assert shift.adjusted_p_value is not None and shift.adjusted_p_value <= 0.01
    assert "data_snapshot" in report.changed_keys
    assert "identification" not in report.invalidated
    assert {"score_artifact", "law", "decision"} <= set(report.invalidated)
    assert report.next_attempt.startswith("refresh")


def test_lost_positivity_asks_to_reidentify(study):
    report = study.staleness(sample(3, slope=9.0), policy=POLICY)
    assert report.recommendation == "reidentify", report.to_dict()
    assert "evidence" in report.changed_keys
    assert "identification" in report.invalidated


def test_a_schema_change_is_refused_like_refresh_refuses_it(study):
    wider = {**sample(4), "w": np.ones(800)}
    report = study.staleness(wider, policy=POLICY)
    assert report.recommendation == "refuse"
    assert any(f.kind == "schema_change" for f in report.findings)
    assert "project the new data" in report.next_attempt


def test_an_exported_baseline_is_assessed_without_the_study(study):
    new = sample(5, shift=0.3)
    direct = study.staleness(new, policy=POLICY)
    baseline = study.staleness_baseline(policy=POLICY)
    assert isinstance(baseline, StalenessBaseline)
    raw = baseline.to_bytes()
    via_bytes = ant.staleness(raw, new, policy=POLICY)
    via_object = ant.staleness(StalenessBaseline.from_bytes(raw), new, policy=POLICY)
    assert via_bytes.identity == direct.identity == via_object.identity
    assert via_bytes.to_dict() == direct.to_dict()
    with pytest.raises(ant.CausalError):
        ant.staleness(raw[: len(raw) // 2], new, policy=POLICY)


def test_the_policy_is_declared_validated_and_part_of_the_identity(study, baseline_data):
    resolved = POLICY.resolved()
    assert resolved["subsample"] == 60 and resolved["alpha"] == 0.01
    assert len(POLICY.identity) == 64
    other = study.staleness(
        baseline_data, policy={"subsample": 60, "permutations": 99, "alpha": 0.02}
    )
    same = study.staleness(baseline_data, policy=POLICY)
    assert other.policy_identity != same.policy_identity
    assert other.identity != same.identity
    with pytest.raises(ant.CausalError, match="alpha"):
        study.staleness(baseline_data, policy=StalenessPolicy(alpha=2.0))
    with pytest.raises(TypeError, match="no field"):
        study.staleness(baseline_data, policy={"not_a_field": 1})
