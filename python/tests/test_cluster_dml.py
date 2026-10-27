"""Cluster-aware cross-fitted ``Aipw`` and the entity-owned screen split.

The route publishes a cross-fitted point, the score table and the cluster-sandwich SE; few clusters,
unsupported dyadic structures and cluster options on flexible-learner estimators are closed with
their registered reason codes. The numerical oracles for the cluster and two-way variances and for
fold ownership are in the Rust integration test ``crates/antecedent-estimate/tests/
cluster_dml_aipw.rs`` (sums and closed forms written there), not this file.
"""

from __future__ import annotations

import math

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalError, CausalValueError
from antecedent.estimation import CandidateScreen, RetargetClaim
from antecedent.estimators import Aipw, ClusterDml
from antecedent.outcomes import require_success

from _native_ate import native_ate
from _refusal import assert_registered_refusal

QUERY = ant.AverageEffect("t", "y")
GRAPH = [("z0", "t"), ("z1", "t"), ("z0", "y"), ("z1", "y"), ("t", "y")]


def clustered(groups: int = 40, size: int = 10, seed: int = 7):
    rng = np.random.default_rng(seed)
    cluster = np.repeat(np.arange(groups), size)
    n = groups * size
    shift = np.repeat(0.7 * rng.normal(size=groups), size)
    shock = np.repeat(1.5 * rng.normal(size=groups), size)
    effect = np.repeat(2.0 + rng.normal(size=groups), size)
    z0 = shift + rng.normal(size=n)
    z1 = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.6 * z0 - 0.4 * z1)))).astype(float)
    y = effect * t + z0 + 0.5 * z1 + shock + 0.5 * rng.normal(size=n)
    return {"z0": z0, "z1": z1, "t": t, "y": y}, cluster


def refusal_text(error: BaseException) -> str:
    return f"{getattr(error, 'reason_code', '')} {error}"


def test_the_dataclass_wires_the_declared_unit_and_validates_it():
    cfg = Aipw(bootstrap=0, cluster_dml=ClusterDml(cluster_ids=[0, 0, 1], min_clusters=12))
    assert cfg._wire() == {
        "bootstrap_replicates": 0,
        "cluster_dml": {"cluster_ids": [0, 0, 1], "min_clusters": 12},
    }
    assert ClusterDml(
        cluster_ids=[1, 2], second_cluster_ids=[7, 8], unit="dyad", min_components_per_fold=3
    )._wire() == {
        "cluster_ids": [1, 2],
        "unit": "dyad",
        "second_cluster_ids": [7, 8],
        "min_components_per_fold": 3,
    }
    for bad in (
        lambda: ClusterDml(cluster_ids=[]),
        lambda: ClusterDml(cluster_ids=[-1, 0]),
        lambda: ClusterDml(cluster_ids=[0, 1], min_clusters=9),
        lambda: ClusterDml(cluster_ids=[0, 1], unit="pair"),  # type: ignore[arg-type]
        # A dyadic unit needs one second label per row; a cluster unit takes none.
        lambda: ClusterDml(cluster_ids=[0, 1], unit="dyad"),
        lambda: ClusterDml(cluster_ids=[0, 1], unit="dyad", second_cluster_ids=[5]),
        lambda: ClusterDml(cluster_ids=[0, 1], unit="dyad", second_cluster_ids=[5, -1]),
        lambda: ClusterDml(
            cluster_ids=[0, 1], unit="dyad", second_cluster_ids=[5, 6], min_components_per_fold=1
        ),
        lambda: ClusterDml(cluster_ids=[0, 1], second_cluster_ids=[5, 6]),
        lambda: ClusterDml(cluster_ids=[0, 1], min_components_per_fold=4),
        # No interval is published, so an interval request is refused up front.
        lambda: Aipw(cluster_dml=ClusterDml(cluster_ids=[0, 1])),
        lambda: Aipw(bootstrap=20, cluster_dml=ClusterDml(cluster_ids=[0, 1])),
        lambda: Aipw(bootstrap=0, se="hc1", cluster_dml=ClusterDml(cluster_ids=[0, 1])),
        lambda: Aipw(
            bootstrap=0, se="cluster", cluster_ids=[0, 1], cluster_dml=ClusterDml(cluster_ids=[0])
        ),
    ):
        with pytest.raises(CausalValueError):
            bad()


def test_a_cluster_dml_aipw_reports_the_cluster_sandwich_se_and_the_scores():
    data, cluster = clustered()
    cfg = Aipw(bootstrap=0, cluster_dml=ClusterDml(cluster_ids=cluster.tolist(), min_clusters=20))
    result = require_success(
        ant.analyze(
            data,
            graph=GRAPH,
            query=QUERY,
            estimator=cfg,
            refute=False,
            execution=ant.Execution(seed=7),
        )
    )
    assert result.effect == pytest.approx(2.0, abs=0.7)
    estimate = result.estimate
    assert math.isfinite(estimate.se_analytic) and estimate.se_analytic > 0.0
    assert estimate.se_bootstrap is None
    assert estimate.joint_covariance is None and estimate.score_inference is None
    assert estimate.score_table is not None
    again = require_success(
        ant.analyze(
            data,
            graph=GRAPH,
            query=QUERY,
            estimator=cfg,
            refute=False,
            execution=ant.Execution(seed=7),
        )
    )
    assert again.effect == result.effect


def test_a_cluster_batch_band_uses_the_cluster_reference():
    data, cluster = clustered(groups=12, size=20)
    cfg = Aipw(bootstrap=0, cluster_dml=ClusterDml(cluster_ids=cluster.tolist(), min_clusters=10))
    batch = require_success(
        ant.estimation.prepare_many(
            data,
            graph=GRAPH,
            queries=[QUERY],
            estimator=cfg,
            refute=False,
            execution=ant.Execution(seed=7),
        ).estimate()
    )
    rows = batch.retarget_rows()
    assert rows is not None
    report = require_success(
        batch.retarget(
            [RetargetClaim("effect", 0, np.ones(len(rows)), depends_on=[])],
            simultaneous_level=0.95,
            simultaneous_seed=29,
            simultaneous_draws=20_000,
        )
    )
    assert report.scores_source == "estimated"
    band = report.simultaneous_interval()
    assert report.complete and len(band.members) == 1
    assert band.critical_value > 2.1  # t_11, above the iid normal critical 1.96
    member = band.members[0]
    assert member.lower == pytest.approx(member.value - band.critical_value * member.std_error)


def test_few_clusters_are_refused_with_a_registered_reason_code():
    data, cluster = clustered(groups=12, size=20)
    cfg = Aipw(bootstrap=0, cluster_dml=ClusterDml(cluster_ids=cluster.tolist(), min_clusters=20))
    with pytest.raises(CausalError) as error:
        require_success(
            ant.analyze(
                data,
                graph=GRAPH,
                query=QUERY,
                estimator=cfg,
                refute=False,
                execution=ant.Execution(seed=1),
            )
        )
    assert "too_few_clusters" in refusal_text(error.value)
    assert_registered_refusal(error.value)


def two_way(blocks: int = 30, na: int = 3, nb: int = 4, seed: int = 11):
    """Blocks of a full ``na`` by ``nb`` grid of endpoints; shocks per first and second endpoint."""
    rng = np.random.default_rng(seed)
    first, second, cols = [], [], {"z0": [], "z1": [], "t": [], "y": []}
    for k in range(blocks):
        u = 1.2 * rng.normal(size=na)
        v = 1.2 * rng.normal(size=nb)
        for i in range(na):
            for j in range(nb):
                z0 = 0.5 * u[i] + rng.normal()
                z1 = rng.normal()
                t = float(rng.uniform() < 1 / (1 + np.exp(-(0.6 * z0 - 0.4 * z1))))
                y = 2.0 * t + z0 + 0.5 * z1 + u[i] + v[j] + 0.5 * rng.normal()
                for name, value in zip(("z0", "z1", "t", "y"), (z0, z1, t, y), strict=True):
                    cols[name].append(value)
                first.append(k * na + i)
                second.append(1_000_000 + k * nb + j)
    return {name: np.array(values) for name, values in cols.items()}, first, second


def test_a_two_way_cluster_dml_aipw_reports_the_two_way_se_and_the_scores():
    data, first, second = two_way()
    cfg = Aipw(
        bootstrap=0,
        cluster_dml=ClusterDml(
            cluster_ids=first, second_cluster_ids=second, unit="dyad", min_clusters=20
        ),
    )
    result = require_success(
        ant.analyze(
            data,
            graph=GRAPH,
            query=QUERY,
            estimator=cfg,
            refute=False,
            execution=ant.Execution(seed=7),
        )
    )
    assert result.effect == pytest.approx(2.0, abs=0.7)
    estimate = result.estimate
    assert math.isfinite(estimate.se_analytic) and estimate.se_analytic > 0.0
    assert estimate.se_bootstrap is None
    assert estimate.joint_covariance is None and estimate.score_inference is None
    assert estimate.score_table is not None
    again = require_success(
        ant.analyze(
            data,
            graph=GRAPH,
            query=QUERY,
            estimator=cfg,
            refute=False,
            execution=ant.Execution(seed=7),
        )
    )
    assert again.effect == result.effect
    # Other second-endpoint labels with the same structure are another declaration, but the
    # same components give the same folds and so the same point.
    relabeled = Aipw(
        bootstrap=0,
        cluster_dml=ClusterDml(
            cluster_ids=first,
            second_cluster_ids=[label + 5 for label in second],
            unit="dyad",
            min_clusters=20,
        ),
    )
    other = require_success(
        ant.analyze(
            data,
            graph=GRAPH,
            query=QUERY,
            estimator=relabeled,
            refute=False,
            execution=ant.Execution(seed=7),
        )
    )
    assert other.effect == result.effect


def test_unsupported_dyadic_structures_stay_closed_with_a_registered_reason_code():
    data, first, second = two_way()
    shared = list(second)
    shared[0] = first[5]  # an entity in both endpoint roles
    bridged = list(second)
    for k in range(1, 30):  # a bridge row per block joins all blocks into one component
        bridged[k * 12] = second[(k - 1) * 12]
    cases = ((shared, "dyadic_shared_namespace"), (bridged, "dyadic_giant_component"))
    for labels, detail in cases:
        cfg = Aipw(
            bootstrap=0,
            cluster_dml=ClusterDml(cluster_ids=first, second_cluster_ids=labels, unit="dyad"),
        )
        with pytest.raises(CausalError) as error:
            require_success(
                ant.analyze(
                    data,
                    graph=GRAPH,
                    query=QUERY,
                    estimator=cfg,
                    refute=False,
                    execution=ant.Execution(seed=1),
                )
            )
        assert "dyadic_dependence_not_licensed" in refusal_text(error.value)
        assert detail in refusal_text(error.value)
        assert_registered_refusal(error.value)


def test_few_components_per_fold_are_refused_with_a_registered_reason_code():
    data, first, second = two_way(blocks=12)
    cfg = Aipw(
        bootstrap=0,
        cluster_dml=ClusterDml(cluster_ids=first, second_cluster_ids=second, unit="dyad"),
    )
    with pytest.raises(CausalError) as error:
        require_success(
            ant.analyze(
                data,
                graph=GRAPH,
                query=QUERY,
                estimator=cfg,
                refute=False,
                execution=ant.Execution(seed=1),
            )
        )
    assert "too_few_clusters" in refusal_text(error.value)
    assert_registered_refusal(error.value)


@pytest.mark.parametrize(
    ("estimator", "option"),
    [
        (estimator, option)
        for estimator in ("dml", "dr.learner", "causal.forest")
        for option in ("cluster_ids", "cluster_dml", "multiway_ids")
        if (estimator, option) not in {("dml", "cluster_dml"), ("dr.learner", "cluster_dml")}
        # dml and dr.learner declare units (positive tests: test_column_projection_and_identifiers)
    ],
)
def test_flexible_learners_refuse_cluster_options(estimator, option):
    data, cluster = clustered(groups=30, size=10)
    names = ["z0", "z1", "t", "y"]
    columns = [data[name] for name in names]
    with pytest.raises(ValueError, match="cluster_dml.flexible_learner_closed") as error:
        native_ate(
            names,
            columns,
            GRAPH,
            "t",
            "y",
            refute=False,
            bootstrap=0,
            estimator=estimator,
            estimator_config={option: [int(c) for c in cluster]},
        )
    assert "route_not_supported" in refusal_text(error.value)
    assert "cluster_dml.flexible_learner_closed" in str(error.value)
    assert "unknown estimator_config key" not in str(error.value)
    assert_registered_refusal(error.value)


def test_the_entity_screen_split_never_shares_an_entity_and_records_seed_and_digest():
    entities = [e for e in range(30) for _ in range(1 + e % 3)]
    screen = CandidateScreen.from_units("s", "max_t", entity_ids=entities, seed=5)
    screen_entities = {entities[i] for i in screen.screen_rows}
    estimate_entities = {entities[i] for i in screen.estimate_rows}
    assert screen_entities.isdisjoint(estimate_entities)
    assert len(screen.screen_rows) + len(screen.estimate_rows) == len(entities)
    assert ";seed=0000000000000005;units=" in screen.screen_id
    again = CandidateScreen.from_units("s", "max_t", entity_ids=entities, seed=5)
    assert again == screen
    assert CandidateScreen.from_units("s", "max_t", entity_ids=entities, seed=6) != screen
    # The receipt states what was dealt: 30 distinct entities split 15/15 by the default
    # fraction, the seed, and the same unit digest the recorded screen id carries.
    receipt = screen.receipt
    assert receipt is not None
    assert (receipt.seed, receipt.unit_kind) == (5, "entity")
    assert (receipt.n_units, receipt.screen_units, receipt.estimate_units) == (30, 15, 15)
    assert screen.screen_id.endswith(f";units={receipt.unit_digest}")
    assert len(receipt.unit_digest) == 64
    assert len(set(entities[i] for i in screen.screen_rows)) == receipt.screen_units
    assert len(set(entities[i] for i in screen.estimate_rows)) == receipt.estimate_units
    # The digest names the set of units, not their order: permuting the rows keeps it.
    permuted = CandidateScreen.from_units("s", "max_t", entity_ids=entities[::-1], seed=5)
    assert permuted.receipt is not None
    assert permuted.receipt.unit_digest == receipt.unit_digest
    # A split declared directly from row lists records no receipt and compares equal on rows.
    declared = CandidateScreen("s", "max_t", screen.screen_rows, screen.estimate_rows)
    assert declared.receipt is None


def test_the_dyad_screen_split_keeps_connected_endpoints_together_and_refuses_one_component():
    first = [0, 1, 2, 10, 10, 10] + [100 + 2 * k for k in range(12)]
    second = [1, 2, 3, 11, 12, 13] + [101 + 2 * k for k in range(12)]
    screen = CandidateScreen.from_units("d", "bh", first=first, second=second, seed=3)

    def endpoints(rows):
        return {first[i] for i in rows} | {second[i] for i in rows}

    assert endpoints(screen.screen_rows).isdisjoint(endpoints(screen.estimate_rows))
    # One chain {0..3}, one star {10..13} and twelve isolated pairs: 1 + 1 + 12 = 14 units.
    receipt = screen.receipt
    assert receipt is not None
    assert (receipt.seed, receipt.unit_kind) == (3, "dyad_component")
    assert receipt.n_units == 14
    assert receipt.screen_units + receipt.estimate_units == receipt.n_units
    with pytest.raises(CausalError) as error:
        CandidateScreen.from_units("d", "bh", first=[0, 1, 2], second=[1, 2, 0], seed=1)
    assert "invalid_argument" in refusal_text(error.value)
    with pytest.raises(CausalError):
        CandidateScreen.from_units("d", "bh", entity_ids=[0, 1], first=[0], second=[1], seed=1)
