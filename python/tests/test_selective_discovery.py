"""Selective discovery through the Python surface, against independent known-graph oracles.

Oracles (computed in this file, not by the library):

* d-separation by the moralization criterion on a hand-written DAG, and the exact partial
  correlations of its linear-Gaussian SCM from the population covariance
  ``Sigma = (I - B)^-1 D (I - B)^-T``: an exact partial correlation is zero exactly when the
  d-separation oracle says the pair is separated (the SCM is faithful by construction);
* the true CPDAG, derived by hand from those separations (skeleton, v-structures, Meek rule 1).

The discovered structure is compared with that CPDAG on a large sample. Selective runs are
compared with full runs of the same request bit for bit. No test asserts that a discovered
graph is the true graph in general.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from antecedent import selective_discovery as sd
from antecedent.errors import CausalDiscoveryError, CausalError
from antecedent.state import CancellationToken

from _refusal import assert_registered_refusal

# -- the known graph ----------------------------------------------------------------------

NAMES = ("a", "b", "c", "d")
# a -> c <- b, c -> d with these structural coefficients and noise standard deviations.
COEF = {("a", "c"): 0.8, ("b", "c"): 0.8, ("c", "d"): 0.9}
NOISE = {"a": 1.0, "b": 1.0, "c": 0.6, "d": 0.5}
PARENTS = {"a": (), "b": (), "c": ("a", "b"), "d": ("c",)}


def d_separated(parents: dict[str, tuple[str, ...]], x: str, y: str, z: set[str]) -> bool:
    """Moralization criterion (Lauritzen et al. 1990), hand-written."""
    ancestral: set[str] = set()
    stack = [x, y, *z]
    while stack:
        v = stack.pop()
        if v in ancestral:
            continue
        ancestral.add(v)
        stack.extend(parents[v])
    adjacent: dict[str, set[str]] = {v: set() for v in ancestral}
    for v in ancestral:
        for p in parents[v]:
            adjacent[v].add(p)
            adjacent[p].add(v)
        for p, q in itertools.combinations(parents[v], 2):
            adjacent[p].add(q)
            adjacent[q].add(p)
    seen, frontier = {x}, [x]
    while frontier:
        v = frontier.pop()
        if v == y:
            return False
        for u in adjacent[v]:
            if u not in seen and u not in z:
                seen.add(u)
                frontier.append(u)
    return True


def population_partial_correlation(x: str, y: str, z: tuple[str, ...]) -> float:
    """Exact partial correlation of x, y given z in the SCM, from the precision matrix."""
    index = {name: i for i, name in enumerate(NAMES)}
    b = np.zeros((4, 4))
    for (src, dst), w in COEF.items():
        b[index[dst], index[src]] = w
    inv = np.linalg.inv(np.eye(4) - b)
    sigma = inv @ np.diag([NOISE[n] ** 2 for n in NAMES]) @ inv.T
    keep = [index[x], index[y], *[index[v] for v in z]]
    precision = np.linalg.inv(sigma[np.ix_(keep, keep)])
    return float(-precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1]))


def true_cpdag() -> set[tuple[str, str]]:
    """Directed edges of the true CPDAG by hand: skeleton, v-structures, then Meek rule 1."""
    others = {n: [m for m in NAMES if m != n] for n in NAMES}
    sepset: dict[frozenset[str], set[str]] = {}
    adjacent: set[frozenset[str]] = set()
    for x, y in itertools.combinations(NAMES, 2):
        rest = [v for v in NAMES if v not in (x, y)]
        found = None
        for size in range(len(rest) + 1):
            for z in itertools.combinations(rest, size):
                if d_separated(PARENTS, x, y, set(z)):
                    found = set(z)
                    break
            if found is not None:
                break
        if found is None:
            adjacent.add(frozenset((x, y)))
        else:
            sepset[frozenset((x, y))] = found
    directed: set[tuple[str, str]] = set()
    for mid in NAMES:
        for x, y in itertools.combinations(others[mid], 2):
            if (
                frozenset((x, mid)) in adjacent
                and frozenset((y, mid)) in adjacent
                and frozenset((x, y)) not in adjacent
                and mid not in sepset[frozenset((x, y))]
            ):
                directed |= {(x, mid), (y, mid)}
    changed = True
    while changed:  # Meek rule 1: x -> y - z with x, z non-adjacent orients y -> z.
        changed = False
        for x, y in list(directed):
            for z in NAMES:
                if (
                    z not in (x, y)
                    and frozenset((y, z)) in adjacent
                    and frozenset((x, z)) not in adjacent
                    and (z, y) not in directed
                    and (y, z) not in directed
                ):
                    directed.add((y, z))
                    changed = True
    return directed


def sample(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for name in NAMES:
        value = NOISE[name] * rng.normal(size=n)
        for parent in PARENTS[name]:
            value = value + COEF[(parent, name)] * out[parent]
        out[name] = value
    return out


def chain_sample(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 0.9 * a + 0.5 * rng.normal(size=n)
    c = 0.9 * b + 0.5 * rng.normal(size=n)
    return {"a": a, "b": b, "c": c}


def contract(**changes: object) -> sd.StaticContract:
    base = sd.StaticContract(NAMES, algorithm="pc", alpha=0.01, seed=0)
    return sd.replace(base, **changes) if changes else base


def bits(outcome: sd.Outcome) -> list[tuple[object, ...]]:
    return sorted(
        (
            e["source"],
            e["source_lag"],
            e["target"],
            e["target_lag"],
            repr(e["statistic"]),
            repr(e["p_value"]),
        )
        for e in outcome.evidence
    )


def run_full(c: sd.StaticContract | sd.TemporalContract, data: dict, kind: str = "static"):
    adapter = sd.SelectiveDiscovery(kind)
    report = adapter.run(c, data)
    outcome = adapter.outcome
    assert outcome is not None
    return adapter, report, outcome


DATA = sample(6000, 11)


# -- known-graph oracle ----------------------------------------------------------------------


def test_oracle_exact_partial_correlations_agree_with_d_separation() -> None:
    for x, y in itertools.combinations(NAMES, 2):
        rest = [v for v in NAMES if v not in (x, y)]
        for size in range(len(rest) + 1):
            for z in itertools.combinations(rest, size):
                zero = abs(population_partial_correlation(x, y, z)) < 1e-12
                assert zero == d_separated(PARENTS, x, y, set(z)), (x, y, z)


def test_discovered_structure_is_the_hand_derived_cpdag() -> None:
    assert true_cpdag() == {("a", "c"), ("b", "c"), ("c", "d")}
    _, report, outcome = run_full(contract(), DATA)
    found = {(p.variable, c.variable) for p, c in outcome.directed_edges}
    assert found == true_cpdag()
    assert len(outcome.edges) == 3 and not outcome.unresolved_edges
    assert outcome.structure_class == "cpdag"
    assert outcome.fully_directed and not outcome.search_incomplete
    work = report.work
    assert work.ci_tests_attempted == work.ci_tests_completed > 0
    assert work.ci_tests_attempted >= work.ci_tests_consumed > 0
    assert work.search_passes >= 1 and work.orientation_runs >= 1
    assert work.orientations_applied >= 3 and work.score_evaluations == 0
    assert work.algorithm_executions == 1


# -- selective vs full -------------------------------------------------------------------------


def test_threshold_change_reuses_statistics_and_equals_a_full_run() -> None:
    adapter = sd.SelectiveDiscovery("static")
    adapter.run(contract(alpha=0.2), DATA)
    report = adapter.run(contract(alpha=0.01), DATA)
    assert report.stage("statistics").status == "reused"
    assert report.stage("search").status == "recomputed"
    assert report.stage("search").because == ("thresholds",)
    _, full_report, full = run_full(contract(alpha=0.01), DATA)
    selective = adapter.outcome
    assert selective is not None
    assert selective.result_identity == full.result_identity
    assert selective.structure_identity == full.structure_identity
    assert bits(selective) == bits(full)
    assert report.work.ci_tests_reused > 0
    assert report.work.ci_tests_attempted < full_report.work.ci_tests_attempted
    assert report.work.ci_tests_consumed == full_report.work.ci_tests_consumed


def test_background_knowledge_variable_order_and_test_changes_equal_full_runs() -> None:
    adapter = sd.SelectiveDiscovery("static")
    adapter.run(contract(), DATA)
    forbidden = (sd.Link("a", "d"), sd.Link("d", "a"))
    knowledge = contract(forbidden=forbidden)
    report = adapter.run(knowledge, DATA)
    assert report.stage("statistics").status == "reused"
    assert report.stage("search").because == ("background_knowledge",)
    _, _, full = run_full(knowledge, DATA)
    assert adapter.outcome is not None and adapter.outcome.result_identity == full.result_identity

    permuted = contract(variables=("d", "b", "a", "c"))
    report = adapter.run(permuted, DATA)
    assert report.stage("statistics").status == "recomputed"
    _, _, full = run_full(permuted, DATA)
    assert adapter.outcome is not None and adapter.outcome.result_identity == full.result_identity

    other_test = contract(ci_test="robust_parcorr")
    report = adapter.run(other_test, DATA)
    assert "ci_test" in report.stage("statistics").because
    assert report.work.ci_tests_reused == 0
    _, _, full = run_full(other_test, DATA)
    assert adapter.outcome is not None and bits(adapter.outcome) == bits(full)


def test_a_repeated_request_is_pure_reuse() -> None:
    adapter = sd.SelectiveDiscovery("static")
    adapter.run(contract(), DATA)
    before = adapter.outcome
    retained = adapter.retained_statistics
    plan = adapter.plan(contract(), DATA)
    assert all(entry.status == "reused" for entry in plan)
    report = adapter.run(contract(), DATA)
    assert report.pure_reuse and report.work.is_pure_reuse
    assert not report.outcome_changed
    assert adapter.outcome == before and adapter.retained_statistics == retained


# -- cache, identity and atomicity ------------------------------------------------------------


def test_changed_input_at_unchanged_shape_retains_nothing() -> None:
    adapter = sd.SelectiveDiscovery("static")
    adapter.run(contract(), DATA)
    changed = {k: v.copy() for k, v in DATA.items()}
    changed["b"][17] += 0.5
    assert changed["b"].shape == DATA["b"].shape
    report = adapter.run(contract(), changed)
    assert report.stage("statistics").because == ("data",)
    _, fresh_report, fresh = run_full(contract(), changed)
    assert adapter.outcome is not None
    assert adapter.outcome.result_identity == fresh.result_identity
    assert report.work.ci_tests_reused == fresh_report.work.ci_tests_reused
    assert report.work.ci_tests_attempted == fresh_report.work.ci_tests_attempted


def test_failed_or_cancelled_update_leaves_the_retained_state_untouched() -> None:
    adapter = sd.SelectiveDiscovery("static")
    adapter.run(contract(), DATA)
    identity = adapter.outcome.result_identity  # type: ignore[union-attr]
    retained = adapter.retained_statistics
    token = CancellationToken()
    token.cancel()
    with pytest.raises(CausalError):
        adapter.run(contract(alpha=0.001), DATA, cancel=token)
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as refused:
        adapter.run(contract(additional_populations=("elsewhere",)), DATA)
    assert refused.value.detail == "selective_discovery.multi_population"
    assert refused.value.reason_code == "route_not_supported"
    assert_registered_refusal(refused.value)
    assert refused.value.remedy
    assert adapter.outcome is not None and adapter.outcome.result_identity == identity
    assert adapter.retained_statistics == retained
    assert adapter.run(contract(), DATA).work.is_pure_reuse


def test_a_second_population_and_a_kind_mismatch_are_refused() -> None:
    adapter = sd.SelectiveDiscovery("temporal")
    with pytest.raises(Exception, match="temporal"):
        adapter.run(contract(), DATA)
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as refused:
        sd.SelectiveDiscovery("temporal").run(
            sd.TemporalContract(("a",), algorithm="pcmci", min_lag=0, max_lag=1),
            {"a": np.arange(30.0)},
        )
    assert refused.value.detail == "selective_discovery.contemporaneous_links"


# -- closed claims, graph uncertainty -----------------------------------------------------------


def test_closed_claims_and_score_search_are_refused() -> None:
    adapter, _, _ = run_full(contract(), DATA)
    for request, detail in (
        (adapter.request_recovery_claim, "selective_discovery.recovery_claim_not_licensed"),
        (
            adapter.request_downstream_interval,
            "selective_discovery.downstream_interval_not_licensed",
        ),
    ):
        with pytest.raises(sd.SelectiveDiscoveryRefusal) as refused:
            request()
        assert refused.value.reason_code == "cell_not_licensed"
        assert refused.value.detail == detail
        assert_registered_refusal(refused.value)
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as refused:
        sd.request_score_search_reuse()
    assert refused.value.reason_code == "route_not_supported"
    assert refused.value.detail == "selective_discovery.score_search_closed"


def test_a_discovered_structure_never_supplies_a_selection_diagram() -> None:
    chain = sd.StaticContract(("a", "b", "c"), algorithm="pc", alpha=0.01, seed=0)
    adapter, _, outcome = run_full(chain, chain_sample(4000, 5))
    # a - b - c: no v-structure, so both edges stay undirected within the class.
    assert len(outcome.unresolved_edges) == 2
    assert {e.unresolved_reason for e in outcome.unresolved_edges} == {
        "undirected_in_equivalence_class"
    }
    assert not outcome.fully_directed
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as refused:
        adapter.request_selection_diagram(["b"])
    err = refused.value
    assert err.reason_code == "discovered_structure_unresolved"
    assert err.detail == "selective_discovery.unresolved_structure"
    assert len(err.unresolved_edges) == 2  # type: ignore[attr-defined]
    assert err.offending is not None and err.offending.count(";") == 1
    assert_registered_refusal(err)

    resolved, _, _ = run_full(contract(), DATA)
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as refused:
        resolved.request_selection_diagram(["c"])
    assert refused.value.detail == "selective_discovery.unaccepted_discovery"


# -- portable state, fresh process ------------------------------------------------------------------


def test_export_consume_inspect_and_resume() -> None:
    adapter, _, outcome = run_full(contract(), DATA)
    exported = adapter.export_state("run-0001")
    consumed = sd.consume_selective_discovery_artifact(exported.artifact, exported.identity)
    assert consumed["identity"] == exported.identity
    meta = sd.inspect_selective_discovery_artifact(exported.artifact)
    assert meta["inference"] == "algorithmic_match_only"
    assert meta["result_identity"] == outcome.result_identity
    assert len(meta["receipts"]) == adapter.retained_statistics

    resumed, report = sd.SelectiveDiscovery.resume(
        contract(), DATA, exported.artifact, exported.identity
    )
    assert report.work.ci_tests_attempted == 0 and report.work.ci_tests_reused > 0
    assert (
        resumed.outcome is not None and resumed.outcome.result_identity == outcome.result_identity
    )
    assert resumed.run(contract(), DATA).work.is_pure_reuse


def test_resume_refusals_and_artifact_tampering() -> None:
    adapter, _, _ = run_full(contract(), DATA)
    exported = adapter.export_state()
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as refused:
        sd.SelectiveDiscovery.resume(contract(), None, exported.artifact)
    assert refused.value.detail == "selective_discovery.raw_data_missing"
    assert refused.value.reason_code == "discovery_statistics_invalid"
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as refused:
        sd.SelectiveDiscovery.resume(contract(), sample(6000, 12), exported.artifact)
    assert refused.value.detail == "selective_discovery.data_mismatch"
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as refused:
        sd.SelectiveDiscovery.resume(contract(alpha=0.2), DATA, exported.artifact)
    assert refused.value.detail == "selective_discovery.contract_mismatch"
    with pytest.raises(sd.SelectiveDiscoveryRefusal):
        sd.consume_selective_discovery_artifact(exported.artifact, "0" * 64)
    flipped = bytearray(exported.artifact)
    flipped[-1] ^= 0xFF
    with pytest.raises(sd.SelectiveDiscoveryRefusal):
        sd.consume_selective_discovery_artifact(bytes(flipped))
    with pytest.raises(sd.SelectiveDiscoveryRefusal):
        sd.SelectiveDiscovery.resume(contract(), DATA, bytes(flipped))


_CHILD = """
import json, sys
import numpy as np
from antecedent import selective_discovery as sd

mode, folder = sys.argv[1], sys.argv[2]
artifact = open(folder + "/state.bin", "rb").read()
identity = open(folder + "/identity.txt").read().strip()
spec = json.load(open(folder + "/contract.json"))
contract = sd.StaticContract(tuple(spec["variables"]), algorithm="pc", alpha=spec["alpha"], seed=0)
data = None
if mode == "with_data":
    loaded = np.load(folder + "/data.npz")
    data = {name: loaded[name] for name in spec["variables"]}
try:
    adapter, report = sd.SelectiveDiscovery.resume(contract, data, artifact, identity)
except sd.SelectiveDiscoveryRefusal as refusal:
    print(json.dumps({"refused": refusal.detail, "code": refusal.reason_code}))
else:
    print(json.dumps({
        "attempted": report.work.ci_tests_attempted,
        "reused": report.work.ci_tests_reused,
        "result": adapter.outcome.result_identity,
    }))
"""


def test_a_fresh_process_resumes_only_with_the_raw_data(tmp_path: Path) -> None:
    adapter, _, outcome = run_full(contract(), DATA)
    exported = adapter.export_state()
    (tmp_path / "state.bin").write_bytes(exported.artifact)
    (tmp_path / "identity.txt").write_text(exported.identity)
    (tmp_path / "contract.json").write_text(json.dumps({"variables": list(NAMES), "alpha": 0.01}))
    np.savez(tmp_path / "data.npz", **DATA)

    def child(mode: str) -> dict[str, object]:
        done = subprocess.run(
            [sys.executable, "-c", _CHILD, mode, str(tmp_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout.strip().splitlines()[-1])

    with_data = child("with_data")
    assert with_data["attempted"] == 0 and with_data["reused"] > 0
    assert with_data["result"] == outcome.result_identity
    without = child("without_data")
    assert without == {
        "refused": "selective_discovery.raw_data_missing",
        "code": "discovery_statistics_invalid",
    }


# -- temporal --------------------------------------------------------------------------------------------


def series(n: int, seed: int, missing: tuple[int, ...] = ()) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x, y, z = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.normal()
        y[t] = 0.7 * x[t - 1] + 0.3 * y[t - 1] + 0.7 * rng.normal()
        z[t] = rng.normal()
    for t in missing:
        y[t] = np.nan
    return {"x": x, "y": y, "z": z}


def temporal(**changes: object) -> sd.TemporalContract:
    base = sd.TemporalContract(
        ("x", "y", "z"), algorithm="pcmci", alpha=0.001, min_lag=1, max_lag=2, seed=0
    )
    return sd.replace(base, **changes) if changes else base


def test_temporal_planted_link_and_selective_equals_full_across_lag_and_mask_changes() -> None:
    data = series(500, 21, missing=(40, 41, 120, 200))
    adapter, report, outcome = run_full(temporal(), data, "temporal")
    assert (sd.Node("x", 1), sd.Node("y", 0)) in outcome.directed_edges
    assert outcome.structure_class == "temporal_dag"
    assert report.work.ci_tests_attempted > 0 and report.work.search_passes >= 1

    for change, reused in (
        (dict(alpha=0.0001), True),
        (dict(alpha=0.0001, min_lag=2), True),
        (dict(alpha=0.0001, min_lag=2, max_lag=3), False),
        (dict(alpha=0.0001, min_lag=2, max_lag=3, mask_type="y"), False),
    ):
        changed = temporal(**change)
        step = adapter.run(changed, data)
        assert (step.stage("statistics").status == "reused") is reused, change
        _, _, full = run_full(changed, data, "temporal")
        assert adapter.outcome is not None
        assert adapter.outcome.result_identity == full.result_identity, change
        assert bits(adapter.outcome) == bits(full), change


def test_temporal_resume_and_new_data() -> None:
    data = series(300, 31)
    adapter, _, outcome = run_full(temporal(), data, "temporal")
    exported = adapter.export_state()
    resumed, report = sd.SelectiveDiscovery.resume(
        temporal(), data, exported.artifact, exported.identity
    )
    assert report.work.ci_tests_attempted == 0
    assert (
        resumed.outcome is not None and resumed.outcome.result_identity == outcome.result_identity
    )
    with pytest.raises(sd.SelectiveDiscoveryRefusal) as refused:
        sd.SelectiveDiscovery.resume(temporal(), series(300, 32), exported.artifact)
    assert refused.value.detail == "selective_discovery.data_mismatch"
    other = series(300, 32)
    step = adapter.run(temporal(), other)
    _, fresh_report, fresh = run_full(temporal(), other, "temporal")
    assert adapter.outcome is not None and adapter.outcome.result_identity == fresh.result_identity
    assert step.work.ci_tests_reused == fresh_report.work.ci_tests_reused
    assert step.work.ci_tests_attempted == fresh_report.work.ci_tests_attempted


def test_default_seed_is_zero_and_lazy_import() -> None:
    assert sd.StaticContract(("a",)).seed == 0
    assert sd.TemporalContract(("a",)).seed == 0
    import antecedent

    assert antecedent.selective_discovery is sd
    assert "selective_discovery" not in antecedent.__all__


def test_the_native_failure_is_a_discovery_error_not_a_refusal() -> None:
    adapter = sd.SelectiveDiscovery("static")
    # Fewer than three rows: `prepare_static_columns` fails with "insufficient rows for PC"
    # (a DiscoveryError), which is not a typed refusal. A constant column is accepted.
    data = {name: np.array([0.0, 1.0]) for name in NAMES}
    with pytest.raises(CausalDiscoveryError):
        adapter.run(contract(), data)
    assert adapter.outcome is None
