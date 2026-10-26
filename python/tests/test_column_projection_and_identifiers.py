"""Column projection by name, and named cluster identifiers aligned to analysed rows.

The analysis reads only the columns that the query, graph and estimator name: extra
string, date and identifier columns are never converted, columns are chosen by name
whatever their position, and a pandas index is not exported. A named ``cluster`` column
is encoded over the original rows and kept with its row when rows are dropped.
"""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from antecedent.errors import CausalError, CausalUnsupportedError, CausalValueError
from antecedent.estimation import analyze_many
from antecedent.estimators import (
    DML,
    Aipw,
    ClusterDml,
    DRLearner,
    FrontdoorLinearTwoStage,
    GlmAdjustment,
    LinearAdjustment,
)
from antecedent.graph import Admg
from antecedent.outcomes import require_success

EDGES = [("z", "t"), ("z", "y"), ("t", "y")]
QUERY = ant.AverageEffect(treatment="t", outcome="y")
ONE_SHOT = {"bootstrap": 0, "refute": False, "seed": 1}


def _numeric(n: int = 120, seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (0.7 * z + rng.normal(size=n) * 0.5 > 0).astype(float)
    y = 1.5 * t + z + rng.normal(size=n) * 0.4
    return {"z": z, "t": t, "y": y}


def _run(data, graph=EDGES, estimator=None, **options):
    return require_success(
        ant.analyze(data, graph=graph, query=QUERY, estimator=estimator, **ONE_SHOT, **options)
    )


def _same(left, right) -> None:
    assert left.estimate.ate == pytest.approx(right.estimate.ate, rel=1e-12)
    assert left.estimate.se_analytic == pytest.approx(right.estimate.se_analytic, rel=1e-12)


def _with_extras(data: dict[str, np.ndarray]) -> dict[str, object]:
    n = len(data["z"])
    return {
        "who": [f"person-{i}" for i in range(n)],
        **data,
        "when": np.datetime64("2024-01-01") + np.arange(n).astype("timedelta64[D]"),
        "note": ["free text"] * n,
    }


def _containers(columns: dict[str, object]):
    frame = pd.DataFrame(columns)
    return {"mapping": columns, "pandas": frame, "arrow": pa.Table.from_pandas(frame)}


# --- extra columns ---------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["mapping", "pandas", "arrow"])
def test_unused_string_date_and_identifier_columns_are_ignored(kind):
    numeric = _numeric()
    baseline = _run(numeric)
    table = _containers(_with_extras(numeric))[kind]
    _same(_run(table), baseline)


# --- reordered columns -----------------------------------------------------------------


@pytest.mark.parametrize("kind", ["mapping", "pandas", "arrow"])
def test_reordered_columns_are_matched_by_name(kind):
    numeric = _numeric()
    baseline = _run(numeric)
    reordered = {"y": numeric["y"], "who": ["a"] * 120, "z": numeric["z"], "t": numeric["t"]}
    _same(_run(_containers(reordered)[kind]), baseline)


@pytest.mark.parametrize("kind", ["mapping", "pandas", "arrow"])
def test_a_typed_graph_binds_columns_by_name_and_graph_order(kind):
    numeric = _numeric()
    graph = Admg.from_edges(["z", "t", "y"], EDGES, [])
    baseline = _run(numeric, graph=graph)
    shuffled = {"y": numeric["y"], "who": ["a"] * 120, "t": numeric["t"], "z": numeric["z"]}
    _same(_run(_containers(shuffled)[kind], graph=graph), baseline)


def test_refresh_accepts_reordered_and_extra_columns():
    numeric = _numeric()
    study = ant.prepare(numeric, graph=EDGES, query=QUERY, **ONE_SHOT)
    require_success(study.estimate())
    changed = _numeric(seed=4)
    plain = require_success(study.refresh(changed))
    shuffled = {"note": ["x"] * 120, "y": changed["y"], "z": changed["z"], "t": changed["t"]}
    _same(require_success(study.refresh(shuffled)), plain)
    _same(require_success(study.refresh(pd.DataFrame(shuffled))), plain)
    _same(require_success(study.refresh(pa.table(shuffled))), plain)


# --- filtered frames and indexes -------------------------------------------------------


def test_a_filtered_frame_with_a_non_default_index_analyses_the_same_rows():
    frame = pd.DataFrame(_with_extras(_numeric()))
    frame.index = pd.Index(np.arange(len(frame)) * 3 + 100, name="row")
    kept = frame[frame["z"] > -0.5]
    baseline = _run(kept.reset_index(drop=True))
    _same(_run(kept), baseline)
    # An Arrow table built with the index keeps it as an ordinary extra column.
    _same(_run(pa.Table.from_pandas(kept, preserve_index=True)), baseline)


# --- duplicate and missing names -------------------------------------------------------


def test_a_duplicated_analysis_column_is_refused_in_pandas_and_arrow():
    numeric = _numeric()
    frame = pd.concat([pd.DataFrame(numeric), pd.DataFrame({"z": numeric["z"]})], axis=1)
    with pytest.raises(CausalValueError, match=r"duplicate column names: \['z'\]"):
        _run(frame)
    table = pa.table(numeric).append_column("z", pa.array(numeric["z"]))
    with pytest.raises(CausalValueError, match=r"duplicate column names: \['z'\]"):
        _run(table)


def test_a_duplicated_column_the_analysis_does_not_read_is_ignored():
    numeric = _numeric()
    baseline = _run(numeric)
    frame = pd.DataFrame({**numeric, "first": 1.0, "second": 2.0})
    frame.columns = [*numeric, "note", "note"]
    _same(_run(frame), baseline)
    table = pa.table(numeric).append_column("note", pa.array(["a"] * 120))
    _same(_run(table.append_column("note", pa.array(["b"] * 120))), baseline)


def test_missing_required_columns_are_refused_by_name():
    numeric = _numeric()
    graph = Admg.from_edges(["z", "t", "y"], EDGES, [])
    without_z = {name: numeric[name] for name in ("t", "y")}
    with pytest.raises(CausalValueError, match=r"columns are missing: \['z'\]"):
        _run(without_z, graph=graph)
    with pytest.raises(CausalError):
        _run(without_z)


# --- named cluster identifiers ---------------------------------------------------------


def _clustered(n: int = 72, seed: int = 11):
    """Confounded data with string site labels (first-seen order not sorted) and gaps in y."""
    data = _numeric(n, seed)
    rng = np.random.default_rng(seed)
    sites = np.array(["north", "east", "south", "west", "centre", "coast"])
    labels = sites[rng.permutation(np.arange(n) % len(sites))]
    y = data["y"].copy()
    y[[2, 3, 9, 20, 31, 44, 58]] = np.nan
    return {**data, "y": y, "site": labels.tolist()}


def _cluster_robust_se(data: dict[str, object]) -> tuple[float, float]:
    """OLS of y on [1, t, z] over complete rows with the CR1 cluster sandwich, in numpy."""
    y = np.asarray(data["y"], dtype=float)
    keep = np.isfinite(y)
    first_seen = {label: i for i, label in enumerate(dict.fromkeys(data["site"]))}
    codes = np.array([first_seen[label] for label in data["site"]])[keep]
    x = np.column_stack([np.ones(keep.sum()), data["t"][keep], data["z"][keep]])
    yk = y[keep]
    bread = np.linalg.inv(x.T @ x)
    beta = bread @ x.T @ yk
    resid = yk - x @ beta
    meat = np.zeros((3, 3))
    for code in np.unique(codes):
        score = x[codes == code].T @ resid[codes == code]
        meat += np.outer(score, score)
    n, p, g = len(yk), x.shape[1], len(np.unique(codes))
    scale = g / (g - 1) * (n - 1) / (n - p)
    return float(beta[1]), float(np.sqrt((bread @ meat @ bread * scale)[1, 1]))


def test_string_cluster_labels_stay_with_their_rows_when_the_outcome_is_missing():
    data = _clustered()
    ate, se = _cluster_robust_se(data)
    result = _run(data, estimator=LinearAdjustment(se="cluster", cluster="site"))
    assert result.estimate.ate == pytest.approx(ate, rel=1e-9)
    assert result.estimate.se_analytic == pytest.approx(se, rel=1e-9)
    # Labels aligned by hand to the complete-case rows give the same answer.
    keep = np.isfinite(data["y"])
    first_seen = {label: i for i, label in enumerate(dict.fromkeys(data["site"]))}
    aligned = [first_seen[label] for label, kept in zip(data["site"], keep, strict=True) if kept]
    by_ids = _run(data, estimator=LinearAdjustment(se="cluster", cluster_ids=aligned))
    _same(result, by_ids)
    # Codes taken in row order without regard to the dropped rows are a different design.
    original = [first_seen[label] for label in data["site"]]
    shifted = _run(
        data, estimator=LinearAdjustment(se="cluster", cluster_ids=original[: len(aligned)])
    )
    assert shifted.estimate.se_analytic != pytest.approx(result.estimate.se_analytic, rel=1e-6)


@pytest.mark.parametrize("kind", ["mapping", "pandas", "arrow"])
def test_a_named_cluster_column_reads_from_every_table_kind(kind):
    data = _clustered()
    expected = _run(data, estimator=LinearAdjustment(se="cluster", cluster="site"))
    table = _containers(data)[kind]
    _same(_run(table, estimator=LinearAdjustment(se="cluster", cluster="site")), expected)


def test_the_glm_estimator_aligns_a_named_cluster_the_same_way():
    data = _clustered()
    data["y"] = np.where(np.isfinite(data["y"]), (np.nan_to_num(data["y"]) > 0.5) * 1.0, np.nan)
    keep = np.isfinite(data["y"])
    first_seen = {label: i for i, label in enumerate(dict.fromkeys(data["site"]))}
    aligned = [first_seen[label] for label, kept in zip(data["site"], keep, strict=True) if kept]
    named = _run(data, estimator=GlmAdjustment(se="cluster", cluster="site"))
    by_ids = _run(data, estimator=GlmAdjustment(se="cluster", cluster_ids=aligned))
    _same(named, by_ids)


def test_one_shot_and_prepared_agree_on_a_named_cluster():
    data = _clustered()
    estimator = LinearAdjustment(se="cluster", cluster="site")
    one_shot = _run(data, estimator=estimator)
    study = ant.prepare(data, graph=EDGES, query=QUERY, estimator=estimator, **ONE_SHOT)
    _same(require_success(study.estimate()), one_shot)
    # Replay against the same table rebuilds the same identifiers.
    _same(require_success(study.reprepare(data).estimate()), one_shot)


def test_refresh_with_changed_identifiers_is_refused_and_leaves_the_study_unchanged():
    data = _clustered()
    study = ant.prepare(
        data,
        graph=EDGES,
        query=QUERY,
        estimator=LinearAdjustment(se="cluster", cluster="site"),
        **ONE_SHOT,
    )
    first = require_success(study.estimate())

    swapped = {**data, "site": list(data["site"])}
    swapped["site"][0], swapped["site"][1] = data["site"][1], data["site"][0]
    assert swapped["site"] != data["site"]
    with pytest.raises(CausalValueError, match="frozen identifier column 'site'"):
        require_success(study.refresh(swapped))
    renamed = {**data, "site": [label.upper() for label in data["site"]]}
    with pytest.raises(CausalValueError, match="frozen identifier column 'site'"):
        require_success(study.refresh(renamed))
    assert study.last_result is first
    _same(require_success(study.estimate()), first)

    # Unchanged identifiers, changed outcomes, extra and reordered columns: accepted.
    updated = _clustered(seed=12)
    updated["site"] = data["site"]
    refreshed = require_success(study.refresh({"note": ["x"] * 72, **updated}))
    assert refreshed.estimate.ate != pytest.approx(first.estimate.ate)
    _, expected_se = _cluster_robust_se(updated)
    assert refreshed.estimate.se_analytic == pytest.approx(expected_se, rel=1e-9)


def test_cluster_misuse_is_refused_with_a_precise_message():
    data = _clustered()
    with pytest.raises(ValueError, match="requires se='cluster'"):
        LinearAdjustment(cluster="site")
    with pytest.raises(ValueError, match="do not also pass cluster_ids"):
        LinearAdjustment(se="cluster", cluster="site", cluster_ids=[0])
    with pytest.raises(CausalUnsupportedError, match="carries no cluster-robust variance"):
        _run(data, estimator="iv.wald", estimator_config={"cluster": "site"})
    with pytest.raises(CausalValueError, match="missing"):
        _run(
            {**data, "site": [None, *data["site"][1:]]},
            estimator=LinearAdjustment(se="cluster", cluster="site"),
        )
    with pytest.raises(CausalValueError, match=r"columns are missing: \['region'\]"):
        _run(data, estimator=LinearAdjustment(se="cluster", cluster="region"))


def _aligned_codes(data: dict[str, object]) -> list[int]:
    keep = np.isfinite(data["y"])
    first_seen = {label: i for i, label in enumerate(dict.fromkeys(data["site"]))}
    return [first_seen[label] for label, kept in zip(data["site"], keep, strict=True) if kept]


def test_aipw_aligns_a_named_cluster_to_its_complete_case_rows():
    data = _clustered()
    named = _run(data, estimator=Aipw(se="cluster", cluster="site"))
    by_ids = _run(data, estimator=Aipw(se="cluster", cluster_ids=_aligned_codes(data)))
    _same(named, by_ids)


def test_frontdoor_aligns_a_named_cluster_to_its_complete_case_rows():
    rng = np.random.default_rng(5)
    n = 90
    t = rng.normal(size=n)
    m = 0.8 * t + rng.normal(size=n) * 0.5
    y = 1.2 * m + rng.normal(size=n) * 0.5
    y[[1, 8, 30, 55]] = np.nan
    site = [f"s{k}" for k in rng.permutation(np.arange(n) % 9)]
    data = {"t": t, "m": m, "y": y, "site": site}
    options = {"identifier": "frontdoor", "graph": [("t", "m"), ("m", "y")]}
    named = _run(data, estimator=FrontdoorLinearTwoStage(se="cluster", cluster="site"), **options)
    by_ids = _run(
        data,
        estimator=FrontdoorLinearTwoStage(se="cluster", cluster_ids=_aligned_codes(data)),
        **options,
    )
    _same(named, by_ids)


def test_estimators_without_a_cluster_variance_refuse_a_named_cluster_precisely():
    data = _clustered()
    for estimator, phrase in (
        ("propensity.matching", "no clustered influence function"),
        ("dml", "no cluster standard error that reads cluster labels per row"),
    ):
        with pytest.raises(CausalUnsupportedError, match=phrase):
            _run(data, estimator=estimator, estimator_config={"cluster": "site"})
    with pytest.raises(CausalUnsupportedError, match="names its own unit column"):
        _run(
            data,
            estimator="aipw",
            estimator_config={"cluster": "site", "cluster_dml": {"cluster_ids": [0]}},
        )


def test_batch_routes_refuse_a_named_cluster_before_reading_the_table():
    data = _clustered()
    with pytest.raises(CausalUnsupportedError, match="does not resolve cluster="):
        analyze_many(
            data,
            graph=EDGES,
            queries=[QUERY],
            estimator=LinearAdjustment(se="cluster", cluster="site"),
        )


def test_refresh_without_the_cluster_column_keeps_the_frozen_labels():
    data = _clustered()
    study = ant.prepare(
        data,
        graph=EDGES,
        query=QUERY,
        estimator=LinearAdjustment(se="cluster", cluster="site"),
        **ONE_SHOT,
    )
    require_success(study.estimate())
    updated = _clustered(seed=12)
    without = {name: updated[name] for name in ("z", "t", "y")}
    _, expected_se = _cluster_robust_se({**updated, "site": data["site"]})
    refreshed = require_success(study.refresh(without))
    assert refreshed.estimate.se_analytic == pytest.approx(expected_se, rel=1e-9)
    # A table with another number of rows cannot carry the frozen labels.
    shorter = {name: values[:-1] for name, values in without.items()}
    with pytest.raises(CausalValueError, match="frozen identifier column 'site' labels 72 rows"):
        require_success(study.refresh(shorter))


def test_participant_labels_are_preserved_and_identity_checked():
    data = _clustered()
    data["pid"] = [f"P{i // 2:03d}" for i in range(72)]
    baseline = _run({name: data[name] for name in ("z", "t", "y")})
    study = ant.prepare(data, graph=EDGES, query=QUERY, participant="pid", **ONE_SHOT)
    _same(require_success(study.estimate()), baseline)
    assert study.participants == tuple(data["pid"])
    assert ant.prepare(data, graph=EDGES, query=QUERY, **ONE_SHOT).participants is None

    first = study.last_result
    moved = {**data, "pid": ["P999", *data["pid"][1:]]}
    with pytest.raises(CausalValueError, match="frozen identifier column 'pid'"):
        require_success(study.refresh(moved))
    with pytest.raises(CausalValueError, match="frozen identifier column 'pid'"):
        require_success(study.refresh({name: values[:-1] for name, values in data.items()}))
    assert study.last_result is first
    require_success(study.refresh({name: data[name] for name in ("z", "t", "y")}))
    assert study.reprepare(moved).participants == tuple(moved["pid"])
    with pytest.raises(CausalValueError, match="identifier column .pid. is missing"):
        ant.prepare(
            {**data, "pid": [None, *data["pid"][1:]]},
            graph=EDGES,
            query=QUERY,
            participant="pid",
            **ONE_SHOT,
        )


def test_a_split_reads_participants_at_the_original_row_ids():
    from antecedent import selective_discovery as sd

    rng = np.random.default_rng(2)
    n = 60
    a = rng.normal(size=n)
    b = a + rng.normal(size=n) * 0.5
    c = b + rng.normal(size=n) * 0.5
    table = {"a": a, "b": b, "c": c, "pid": [f"P{i // 2}" for i in range(n)]}
    contract = sd.StaticContract(
        variables=("a", "b", "c"), ci_test="parcorr", alpha=0.001, fdr=None, seed=1
    )
    by_participant = sd.discover_on_split(contract, table, discovery_rows=list(range(30)))
    discovery, analysis = by_participant.participants(table, "pid")
    assert discovery == tuple(table["pid"][:30])
    assert analysis == tuple(table["pid"][30:])
    shared = sd.discover_on_split(contract, table, discovery_rows=list(range(1, 31)))
    with pytest.raises(CausalValueError, match="both the discovery and the analysis"):
        shared.participants(table, "pid")
    assert shared.participants(table, "pid", require_disjoint=False)[1] == tuple(
        table["pid"][i] for i in shared.analysis_row_ids
    )


def _many_clusters(n: int = 240, groups: int = 20, seed: int = 21):
    data = _numeric(n, seed)
    rng = np.random.default_rng(seed)
    site = [f"unit-{k:02d}" for k in rng.permutation(np.arange(n) % groups)]
    y = data["y"].copy()
    y[[3, 17, 40, 81, 100, 133, 170, 222]] = np.nan
    return {**data, "y": y, "site": site}


@pytest.mark.parametrize(
    "build",
    [
        lambda unit: Aipw(bootstrap=0, cluster_dml=unit),
        lambda unit: DML(cluster_dml=unit),
        lambda unit: DRLearner(cluster_dml=unit),
    ],
    ids=["aipw", "dml", "dr_learner"],
)
def test_a_named_cluster_dml_unit_matches_hand_aligned_unit_ids(build):
    data = _many_clusters()
    named = _run(data, estimator=build(ClusterDml(cluster="site", min_clusters=10)))
    first_seen = {label: i for i, label in enumerate(dict.fromkeys(data["site"]))}
    keep = np.isfinite(data["y"])
    aligned = [first_seen[s] for s, kept in zip(data["site"], keep, strict=True) if kept]
    by_ids = _run(data, estimator=build(ClusterDml(cluster_ids=aligned, min_clusters=10)))
    assert named.estimate.ate == pytest.approx(by_ids.estimate.ate, rel=1e-12)
    assert named.estimate.se_analytic == pytest.approx(by_ids.estimate.se_analytic, rel=1e-12)
    assert np.isfinite(named.estimate.se_analytic)


def test_cluster_dml_validation_names_one_kind_of_labels():
    with pytest.raises(CausalValueError, match="exactly one of cluster_ids and cluster"):
        ClusterDml(cluster_ids=[0, 1], cluster="site")
    with pytest.raises(CausalValueError, match="exactly one of cluster_ids and cluster"):
        ClusterDml()
    with pytest.raises(CausalValueError, match="requires second_cluster"):
        ClusterDml(cluster="a", unit="dyad")
    with pytest.raises(CausalValueError, match="AIPW score"):
        DML(score="partially_linear", cluster_dml=ClusterDml(cluster="site"))
