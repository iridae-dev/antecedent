"""The conformance gate covers every capability it names and is wired into the release gates."""

from __future__ import annotations

import sys

from _repo_text import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_gcm_conformance as conformance  # noqa: E402

CAPABILITIES = (
    "policy_floor_downstream",
    "policy_route_agreement",
    "callback_validation",
    "dependency_failures",
    "portability_restrictions",
    "mechanism_selection_known_truth",
    "falsification_correct",
    "falsification_misspecified",
    "falsification_uninformative",
    "falsification_budget_limited",
    "intrinsic_influence_linear_variance",
    "shapley_allocation_identities",
    "artifact_round_trip",
    "replay_reproducible",
)


def test_the_conformance_gate_covers_every_capability_and_is_registered():
    import tomllib

    registry = tomllib.loads((REPO_ROOT / conformance.REGISTRY).read_text(encoding="utf-8"))
    rows_by_clause: dict[str, list[str]] = {}
    for row in registry["conformance"]:
        rows_by_clause.setdefault(row["clause"], []).append(row["id"])
    for clause in CAPABILITIES:
        assert rows_by_clause.get(clause), f"{clause} has no conformance test"
    # Every benchmark of batch execution is in the registry and carries its own bound.
    assert {row["bench"] for row in registry["performance"]} == {
        "policy_sampling",
        "noise_replay",
        "counterfactual_batch",
    }
    # The registry is consistent: tests exist, benches are registered, owned cases are bound
    # to tests the gate runs.
    assert conformance.problems() == []

    # The gate runs in the release gate and is documented beside its neighbours.
    release = (REPO_ROOT / "scripts" / "gate_release.sh").read_text(encoding="utf-8")
    assert "bash scripts/gate_gcm_conformance.sh" in release
    selftests = (REPO_ROOT / "scripts" / "gate_selftests.sh").read_text(encoding="utf-8")
    assert "scripts/gate_gcm_conformance.sh --self-test" in selftests
    development = (REPO_ROOT / "docs" / "development.md").read_text(encoding="utf-8")
    assert "scripts/gate_gcm_conformance.sh" in development


def test_the_registry_checker_rejects_a_clause_without_evidence():
    import copy
    import tomllib

    registry = tomllib.loads((REPO_ROOT / conformance.REGISTRY).read_text(encoding="utf-8"))
    broken = copy.deepcopy(registry)
    broken["conformance"] = [
        row for row in broken["conformance"] if row["clause"] != "falsification_budget_limited"
    ]
    problems = conformance.registry_problems(broken)
    assert any("falsification_budget_limited has no cited test" in p for p in problems)
