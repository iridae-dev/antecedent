"""2.2 E4: cluster-aware cross-fitted ``Aipw`` and the entity-owned screen split.

The route publishes a cross-fitted point and the score table and no interval; few clusters and a
dyadic unit are closed with their registered reason codes. The numerical oracle for the cluster
variance and for fold ownership is the Rust integration test ``crates/antecedent-estimate/tests/
cluster_dml_aipw.rs`` (cluster sums written there), not this file.
"""

from __future__ import annotations

import math

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalValueError
from antecedent.estimation import CandidateScreen
from antecedent.estimators import Aipw, ClusterDml

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
    assert ClusterDml(cluster_ids=[1, 2], unit="dyad")._wire() == {
        "cluster_ids": [1, 2],
        "unit": "dyad",
    }
    for bad in (
        lambda: ClusterDml(cluster_ids=[]),
        lambda: ClusterDml(cluster_ids=[-1, 0]),
        lambda: ClusterDml(cluster_ids=[0, 1], min_clusters=9),
        lambda: ClusterDml(cluster_ids=[0, 1], unit="pair"),  # type: ignore[arg-type]
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


def test_a_cluster_dml_aipw_reports_a_point_and_the_scores_but_no_interval():
    data, cluster = clustered()
    cfg = Aipw(bootstrap=0, cluster_dml=ClusterDml(cluster_ids=cluster.tolist(), min_clusters=20))
    result = ant.analyze(data, graph=GRAPH, query=QUERY, estimator=cfg, refute=False, seed=7)
    assert result.effect == pytest.approx(2.0, abs=0.7)
    estimate = result.estimate
    assert math.isnan(estimate.se_analytic)
    assert estimate.se_bootstrap is None
    assert estimate.joint_covariance is None and estimate.score_inference is None
    assert estimate.score_table is not None
    again = ant.analyze(data, graph=GRAPH, query=QUERY, estimator=cfg, refute=False, seed=7)
    assert again.effect == result.effect


def test_few_clusters_are_refused_with_a_registered_reason_code():
    data, cluster = clustered(groups=12, size=20)
    cfg = Aipw(bootstrap=0, cluster_dml=ClusterDml(cluster_ids=cluster.tolist(), min_clusters=20))
    with pytest.raises(Exception) as error:
        ant.analyze(data, graph=GRAPH, query=QUERY, estimator=cfg, refute=False, seed=1)
    assert "too_few_clusters" in refusal_text(error.value)
    assert_registered_refusal(error.value)


def test_a_dyadic_unit_is_declarable_and_closed():
    data, cluster = clustered()
    cfg = Aipw(
        bootstrap=0, cluster_dml=ClusterDml(cluster_ids=cluster.tolist(), unit="dyad")
    )
    with pytest.raises(Exception) as error:
        ant.analyze(data, graph=GRAPH, query=QUERY, estimator=cfg, refute=False, seed=1)
    assert "dyadic_dependence_not_licensed" in refusal_text(error.value)
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


def test_the_dyad_screen_split_keeps_connected_endpoints_together_and_refuses_one_component():
    first = [0, 1, 2, 10, 10, 10] + [100 + 2 * k for k in range(12)]
    second = [1, 2, 3, 11, 12, 13] + [101 + 2 * k for k in range(12)]
    screen = CandidateScreen.from_units("d", "bh", first=first, second=second, seed=3)

    def endpoints(rows):
        return {first[i] for i in rows} | {second[i] for i in rows}

    assert endpoints(screen.screen_rows).isdisjoint(endpoints(screen.estimate_rows))
    with pytest.raises(Exception) as error:
        CandidateScreen.from_units("d", "bh", first=[0, 1, 2], second=[1, 2, 0], seed=1)
    assert "invalid_argument" in refusal_text(error.value)
    with pytest.raises(Exception):
        CandidateScreen.from_units("d", "bh", entity_ids=[0, 1], first=[0], second=[1], seed=1)
