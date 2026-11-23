"""Staleness and drift diagnostics end to end: a prepared study, its exported baseline, and
new data with no change, a covariate shift, lost positivity and a schema change."""

from __future__ import annotations

import json

import antecedent as ant
import numpy as np
import pytest
from antecedent import drift
from antecedent.drift import StalenessBaseline, StalenessPolicy, StalenessReport
from antecedent.outcomes import Outcome, require_success

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
    return ant.prepare(
        baseline_data,
        graph=GRAPH,
        query=QUERY,
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
        execution=ant.Execution(seed=1, threads=2),
    )


def test_the_retained_data_is_reused(study, baseline_data):
    report = require_success(study.staleness(baseline_data, policy=POLICY))
    assert isinstance(report, StalenessReport)
    assert report.recommendation == "reuse"
    assert report.findings == ()
    assert report.changed_keys == () and report.invalidated == ()
    assert len(report.evaluations) > 5
    json.dumps(report.to_dict())


def test_a_covariate_shift_recommends_rebinding_naming_the_snapshot(study):
    report = require_success(study.staleness(sample(2, shift=0.35), policy=POLICY))
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
    report = require_success(study.staleness(sample(3, slope=9.0), policy=POLICY))
    assert report.recommendation == "reidentify", report.to_dict()
    assert "evidence" in report.changed_keys
    assert "identification" in report.invalidated


def test_a_schema_change_is_reported_before_rebinding(study):
    wider = {**sample(4), "w": np.ones(800)}
    report = require_success(study.staleness(wider, policy=POLICY))
    assert report.recommendation == "refuse"
    assert any(f.kind == "schema_change" for f in report.findings)
    assert "project the new data" in report.next_attempt


def test_an_exported_baseline_is_assessed_without_the_study(study):
    new = sample(5, shift=0.3)
    direct = require_success(drift.staleness(study, new, policy=POLICY))
    prepared_baseline = drift.baseline(study, policy=POLICY)
    assert isinstance(prepared_baseline, Outcome)
    baseline = require_success(prepared_baseline)
    assert isinstance(baseline, StalenessBaseline)
    artifact = baseline.export()
    loaded_outcome = ant.load(artifact, expected=artifact.expectation)
    loaded = require_success(loaded_outcome)
    assert isinstance(loaded, StalenessBaseline)
    assert loaded_outcome.study is None
    assert loaded.acceptance.status == "loaded"
    assert loaded.acceptance.identity_bound
    assert not loaded.acceptance.replayable and not loaded.acceptance.resumable
    assert loaded.identity == baseline.identity
    assert loaded.rows == 800 and loaded.complete_rows == 800
    raw = bytes(artifact)
    via_bytes = require_success(drift.staleness(raw, new, policy=POLICY, execution=study.execution))
    via_object = require_success(loaded.assess(new, policy=POLICY, execution=study.execution))
    via_artifact = require_success(
        drift.staleness(artifact, new, policy=POLICY, execution=study.execution)
    )
    assert via_bytes.identity == direct.identity == via_object.identity == via_artifact.identity
    assert via_bytes.to_dict() == direct.to_dict() == via_object.to_dict() == via_artifact.to_dict()
    with pytest.raises(ant.CausalError, match=r"truncated\ staleness\ baseline"):
        drift.staleness(raw[: len(raw) // 2], new, policy=POLICY, execution=study.execution)


def test_the_policy_is_declared_validated_and_part_of_the_identity(study, baseline_data):
    resolved = POLICY.resolved()
    assert resolved["subsample"] == 60 and resolved["alpha"] == 0.01
    assert len(POLICY.identity) == 64
    other = require_success(
        study.staleness(baseline_data, policy={"subsample": 60, "permutations": 99, "alpha": 0.02})
    )
    same = require_success(study.staleness(baseline_data, policy=POLICY))
    assert other.policy_identity != same.policy_identity
    assert other.identity != same.identity
    with pytest.raises(ant.CausalError, match="alpha"):
        require_success(study.staleness(baseline_data, policy=StalenessPolicy(alpha=2.0)))
    with pytest.raises(TypeError, match="no field"):
        require_success(study.staleness(baseline_data, policy={"not_a_field": 1}))


def test_study_conveniences_delegate_to_the_same_explicit_drift_operations(study, baseline_data):
    direct = drift.staleness(study, baseline_data, policy=POLICY, execution=study.execution)
    convenience = study.staleness(baseline_data, policy=POLICY)
    assert isinstance(direct, Outcome) and isinstance(convenience, Outcome)
    assert direct.accepted and convenience.accepted
    assert require_success(direct).to_dict() == require_success(convenience).to_dict()
    direct_baseline = require_success(drift.baseline(study, policy=POLICY))
    convenience_baseline = require_success(study.staleness_baseline(policy=POLICY))
    assert direct_baseline.identity == convenience_baseline.identity
    assert bytes(direct_baseline.export()) == bytes(convenience_baseline.export())


def test_staleness_baseline_consumes_in_a_fresh_process(study, tmp_path):
    import subprocess
    import sys
    import textwrap

    baseline = require_success(drift.baseline(study, policy=POLICY))
    artifact_path = tmp_path / "staleness.ant"
    artifact_path.write_bytes(bytes(baseline.export()))
    direct = require_success(drift.staleness(study, sample(5, shift=0.3), policy=POLICY))
    script = textwrap.dedent("""
        import json
        import sys
        from pathlib import Path
        import antecedent as ant
        import numpy as np
        from antecedent import drift
        from antecedent.artifacts import ArtifactExpectation
        from antecedent.outcomes import require_success

        loaded = ant.load(
            Path(sys.argv[1]).read_bytes(),
            expected=ArtifactExpectation("staleness_baseline_v1", sys.argv[2]),
        )
        assert loaded.study is None
        baseline = require_success(loaded)
        rng = np.random.default_rng(5)
        z = rng.normal(size=800) + 0.3
        t = (rng.uniform(size=800) < 1 / (1 + np.exp(-0.5 * z))).astype(float)
        y = 1 + 2 * t + z + rng.normal(size=800)
        report = require_success(drift.staleness(
            baseline, {"t": t, "y": y, "z": z},
            policy=drift.StalenessPolicy(subsample=60, permutations=99),
            execution=ant.Execution(seed=1, threads=1),
        ))
        print(json.dumps({
            "baseline": baseline.identity, "report": report.identity,
            "standing": loaded.acceptance.status,
            "identity_bound": loaded.acceptance.identity_bound,
            "replayable": loaded.acceptance.replayable,
        }))
    """)
    child = subprocess.run(
        [sys.executable, "-c", script, str(artifact_path), baseline.identity],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(child.stdout)
    assert result == {
        "baseline": baseline.identity,
        "report": direct.identity,
        "standing": "loaded",
        "identity_bound": True,
        "replayable": False,
    }


def test_staleness_guide_executes_the_declared_diagnostic_and_artifact_journey():
    import re
    from pathlib import Path

    guide = Path(__file__).resolve().parents[2] / "docs/guides/staleness.md"
    blocks = re.findall(r"```python\n(.*?)```", guide.read_text(encoding="utf-8"), flags=re.DOTALL)
    namespace = {}
    exec(compile(blocks[0], str(guide), "exec"), namespace)
    assert namespace["report"].recommendation == "refresh"
    assert namespace["report"].identity == namespace["independent"].identity
    assert namespace["study"] is not namespace["branch"]


def test_a_baseline_built_under_a_custom_policy_is_not_assessed_under_the_implicit_default(
    study, baseline_data
):
    baseline = require_success(drift.baseline(study, policy=POLICY))
    assert baseline.policy_identity == POLICY.identity != StalenessPolicy().identity
    with pytest.raises(ant.CausalError, match="baseline_policy_required"):
        require_success(baseline.assess(baseline_data))
    with pytest.raises(ant.CausalError, match="baseline_policy_required"):
        require_success(drift.staleness(baseline.export(), baseline_data))
    declared = require_success(baseline.assess(baseline_data, policy=POLICY))
    assert declared.policy_identity == POLICY.identity


def test_a_default_policy_baseline_may_be_assessed_without_a_declared_policy(study, baseline_data):
    baseline = require_success(drift.baseline(study))
    assert baseline.policy_identity == StalenessPolicy().identity
    report = require_success(baseline.assess(baseline_data))
    assert report.policy_identity == baseline.policy_identity


def test_execution_seed_does_not_reach_the_diagnostic_streams(study, baseline_data):
    first = require_success(
        drift.staleness(study, baseline_data, policy=POLICY, execution=ant.Execution(seed=1))
    )
    second = require_success(
        drift.staleness(study, baseline_data, policy=POLICY, execution=ant.Execution(seed=99))
    )
    assert first.to_dict()["evaluations"] == second.to_dict()["evaluations"]


def customers(seed: int, *, hidden: bool, n: int = 4000) -> dict[str, np.ndarray]:
    """`price -> engagement <- tenure`, `price -> support`; `hidden` adds an unmeasured cause
    of engagement, support and cancellation."""
    rng = np.random.default_rng(seed)
    price, tenure = rng.normal(size=n), rng.normal(size=n)
    u = rng.normal(size=n) if hidden else np.zeros(n)
    engagement = -0.6 * price + 0.5 * tenure + 0.6 * u + rng.normal(size=n)
    support = 0.5 * price + 0.8 * u + rng.normal(size=n)
    cancellation = -0.8 * engagement + 0.5 * support + 0.3 * price + 0.8 * u + rng.normal(size=n)
    return {
        "price": price,
        "tenure": tenure,
        "engagement": engagement,
        "support": support,
        "cancellation": cancellation,
    }


CUSTOMER_GRAPH = [
    ("price", "engagement"),
    ("tenure", "engagement"),
    ("price", "support"),
    ("price", "cancellation"),
    ("engagement", "cancellation"),
    ("support", "cancellation"),
]


def _customer_study(data, graph):
    return ant.prepare(
        data,
        graph=graph,
        query=ant.AverageEffect("engagement", "cancellation"),
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
        execution=ant.Execution(seed=1, threads=2),
    )


def test_an_independence_the_graph_implies_and_new_data_contradict_asks_to_reidentify():
    study = _customer_study(customers(1, hidden=False), CUSTOMER_GRAPH)
    unchanged = require_success(study.staleness(customers(3, hidden=False)))
    assert unchanged.recommendation == "reuse" and unchanged.findings == ()
    implied = [f for f in unchanged.evaluations if f.kind == "implied_independence"]
    assert {f.subject for f in implied} == {
        "price _||_ tenure",
        "tenure _||_ support",
        "tenure _||_ cancellation | price, engagement",
        "engagement _||_ support | price",
    }

    changed = require_success(study.staleness(customers(2, hidden=True)))
    assert changed.recommendation == "reidentify", changed.to_dict()
    assert {f.kind for f in changed.findings} == {"implied_independence"}
    failed = {f.subject: f for f in changed.findings}["engagement _||_ support | price"]
    # Given price: covariance 0.48 over variances 1.61 and 1.64, a partial correlation of 0.295.
    assert failed.statistic == pytest.approx(0.48 / np.sqrt(1.61 * 1.64), abs=0.05)
    assert failed.threshold_name == "implication_threshold" and failed.forces == "reidentify"
    assert "identification" in changed.invalidated
    assert "review the graph" in changed.next_attempt

    # The exported baseline carries the statements: the same report without the study.
    artifact = require_success(drift.baseline(study)).export()
    again = require_success(drift.staleness(bytes(artifact), customers(2, hidden=True)))
    assert again.identity == changed.identity


def test_a_graph_that_implies_nothing_names_the_check_it_could_not_make():
    keep = ("price", "engagement", "cancellation")
    graph = [edge for edge in CUSTOMER_GRAPH if set(edge) <= set(keep)]
    before = {name: column for name, column in customers(1, hidden=False).items() if name in keep}
    after = {name: column for name, column in customers(2, hidden=True).items() if name in keep}
    report = require_success(_customer_study(before, graph).staleness(after))
    assert report.recommendation == "reuse"
    reason = dict(report.not_evaluated)["implied_independence"]
    assert "implies no conditional independence" in reason


def test_the_implication_policy_fields_resolve_validate_and_enter_the_identity():
    default = StalenessPolicy().resolved()
    assert default["implication_test"] == "parcorr"
    assert default["implication_threshold"] == 0.1 and default["max_implications"] == 512
    assert StalenessPolicy(implication_threshold=0.2).identity != StalenessPolicy().identity
    with pytest.raises(ant.errors.CausalError, match="implication_test"):
        StalenessPolicy(implication_test="gpdc").resolved()
