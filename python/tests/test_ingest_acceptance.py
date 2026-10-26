"""Ingest and named identifiers at the sizes of a practitioner study (2,000 and 3,000 rows).

Each test runs one public route (``analyze``, ``prepare`` / ``refresh``, ``discover_on_split``)
on a frame carrying what real tables carry besides the analysed columns: free-text, date and
identifier columns, a filtered pandas index, shuffled column order, string participant and
cluster labels, and missing outcomes. The expected values are computed independently: the
projected frame, a hand-built complete-case table and a numpy cluster-robust sandwich.
"""

from __future__ import annotations

import itertools

import antecedent as ant
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from antecedent import selective_discovery as sd
from antecedent._data import table_column_names, try_as_arrow_c_columns
from antecedent.errors import CausalError, CausalTypeError, CausalValueError
from antecedent.estimators import LinearAdjustment
from antecedent.graph import Admg, Dag
from antecedent.outcomes import require_success

QUERY = ant.AverageEffect(treatment="t", outcome="y")
ONE_SHOT = {"bootstrap": 0, "refute": False, "seed": 1}
COVARIATES = ("z1", "z2")
EDGES = [("z1", "t"), ("z2", "t"), ("z1", "y"), ("z2", "y"), ("t", "y")]
DECLARED = ("z1", "z2", "t", "y")


def declared_frame(
    n: int, seed: int, *, effect: float = 0.5, icc: float = 0.0, groups: int = 0
) -> dict[str, np.ndarray]:
    """Confounded study data: two covariates drive treatment and outcome (true effect ``effect``).

    With ``groups`` the outcome also carries a site effect of intraclass correlation ``icc`` and
    the returned ``site`` codes say which site each row belongs to.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    treated = rng.random(n) < 1.0 / (1.0 + np.exp(-(0.6 * z[:, 0] - 0.4 * z[:, 1])))
    noise = rng.normal(size=n)
    out: dict[str, np.ndarray] = {"z1": z[:, 0], "z2": z[:, 1]}
    t = treated.astype(float)
    y = effect * t + 0.5 * z[:, 0] + 0.3 * z[:, 1] + noise
    if groups:
        site = rng.permutation(np.arange(n) % groups)
        shared = rng.normal(size=groups) * np.sqrt(icc / (1.0 - icc))
        y = y + shared[site]
        out["site_code"] = site
    out["t"], out["y"] = t, y
    return out


def with_nuisance(columns: dict[str, np.ndarray], seed: int) -> pd.DataFrame:
    """The declared columns beside two free-text columns, a date and a string identifier."""
    n = len(columns["y"])
    rng = np.random.default_rng(seed + 1)
    frame = pd.DataFrame(
        {
            "who": [f"person-{i:05d}" for i in rng.permutation(n)],
            **{name: columns[name] for name in DECLARED},
            "note_a": rng.choice(["alpha", "beta", "gamma"], size=n),
            "note_b": ["free text, with: punctuation"] * n,
            "visit": np.datetime64("2024-01-01") + np.arange(n).astype("timedelta64[D]"),
        }
    )
    return frame


def run(data, graph=EDGES, **options):
    return require_success(ant.analyze(data, graph=graph, query=QUERY, **ONE_SHOT, **options))


def same(left, right) -> None:
    assert left.estimate.ate == pytest.approx(right.estimate.ate, rel=1e-12)
    assert left.estimate.se_analytic == pytest.approx(right.estimate.se_analytic, rel=1e-12)


def containers(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "pandas": frame,
        "arrow": pa.Table.from_pandas(frame, preserve_index=False),
        "mapping": {name: frame[name].to_numpy() for name in frame.columns},
    }


# --- ingest ----------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["pandas", "arrow", "mapping"])
def test_unused_string_date_and_identifier_columns_are_never_converted(kind):
    columns = declared_frame(2000, 2644802)
    projected = {name: columns[name] for name in DECLARED}
    baseline = run(projected)
    frame = with_nuisance(columns, 2644802)
    assert frame.shape == (2000, 8)
    assert (
        sum(frame[name].dtype == object or frame[name].dtype.kind in "OUM" for name in frame) >= 4
    )
    same(run(containers(frame)[kind]), baseline)
    assert baseline.estimate.ate == pytest.approx(0.5, abs=0.2)


def test_the_pandas_index_is_not_exported_as_a_column():
    columns = declared_frame(2000, 8525629)
    frame = pd.DataFrame({name: columns[name] for name in DECLARED})
    frame.index = pd.Index(np.arange(len(frame)) * 3 + 100, name="row")
    kept = frame[frame["z1"] > -0.5]
    assert 0 < len(kept) < 2000
    exported = try_as_arrow_c_columns(kept)
    assert exported is not None
    names, arrays = exported
    assert names == list(DECLARED)
    assert not any(name.startswith("__index_level") for name in names)
    assert {len(array) for array in arrays} == {len(kept)}
    study = ant.prepare(kept, graph=EDGES, query=QUERY, **ONE_SHOT)
    assert table_column_names(kept) == list(DECLARED)
    same(require_success(study.estimate()), run(kept.reset_index(drop=True)))
    # The row count the study analysed is the filtered frame's, not the original frame's.
    assert require_success(study.estimate()).missing_data.accounting.rows_total == len(kept)


@pytest.mark.parametrize("typed", ["dag", "admg"])
def test_every_column_permutation_gives_the_same_estimate_under_a_typed_graph(typed):
    columns = declared_frame(2000, 8773591)
    nodes = ["z1", "z2", "t", "y"]
    graph = Dag.from_edges(nodes, EDGES) if typed == "dag" else Admg.from_edges(nodes, EDGES, [])
    frame = with_nuisance(columns, 8773591)
    baseline = run(frame, graph=graph)
    for order in itertools.permutations(DECLARED):
        shuffled = frame[["who", *order, "note_a", "visit"]]
        same(run(shuffled, graph=graph), baseline)
        same(run(pa.Table.from_pandas(shuffled, preserve_index=False), graph=graph), baseline)
    assert baseline.estimate.ate == pytest.approx(0.5, abs=0.2)


def test_refresh_binds_columns_by_name_and_equals_the_one_shot_estimate():
    first = declared_frame(2000, 8987131)
    second = declared_frame(2000, 8987132)
    study = ant.prepare(
        pd.DataFrame({name: first[name] for name in DECLARED}),
        graph=EDGES,
        query=QUERY,
        **ONE_SHOT,
    )
    prepared = require_success(study.estimate())
    same(prepared, run({name: first[name] for name in DECLARED}))
    reversed_names = list(reversed(DECLARED))
    assert reversed_names == ["y", "t", "z2", "z1"]
    refreshed = require_success(
        study.refresh(pd.DataFrame({name: second[name] for name in reversed_names}))
    )
    same(refreshed, run({name: second[name] for name in DECLARED}))
    assert refreshed.estimate.ate != pytest.approx(prepared.estimate.ate, rel=1e-6)
    # Extra columns and Arrow input bind by name as well.
    extra = with_nuisance(second, 8987132)[["note_a", "y", "t", "visit", "z2", "z1"]]
    same(
        require_success(study.refresh(pa.Table.from_pandas(extra, preserve_index=False))), refreshed
    )


@pytest.mark.parametrize("kind", ["pandas", "arrow"])
def test_a_filtered_frame_with_a_non_default_index_equals_its_reset_index_frame(kind):
    columns = declared_frame(2000, 4364692)
    frame = with_nuisance(columns, 4364692)
    frame.index = pd.Index(np.arange(len(frame)) * 7 + 11, name="row")
    kept = frame[(frame["z2"] > -0.7) & (frame["note_a"] != "gamma")]
    assert 1000 < len(kept) < 2000
    baseline = run(kept.reset_index(drop=True))
    table = kept if kind == "pandas" else pa.Table.from_pandas(kept, preserve_index=True)
    result = run(table)
    same(result, baseline)
    assert result.missing_data.accounting.rows_total == len(kept)


def test_a_missing_declared_column_is_refused_by_name():
    columns = declared_frame(500, 2685029)
    without = pd.DataFrame({name: columns[name] for name in DECLARED if name != "z1"})
    typed = Admg.from_edges(list(DECLARED), EDGES, [])
    with pytest.raises(CausalValueError, match=r"columns are missing: \['z1'\]"):
        run(without, graph=typed)
    with pytest.raises(CausalError, match="z1"):
        run(without)
    with pytest.raises(CausalError, match="z1"):
        run({name: without[name].to_numpy() for name in without})


def test_a_duplicated_declared_column_name_is_refused_by_name():
    columns = declared_frame(500, 7926377)
    frame = pd.concat(
        [
            pd.DataFrame({name: columns[name] for name in DECLARED}),
            pd.DataFrame({"z1": columns["z1"]}),
        ],
        axis=1,
    )
    assert list(frame.columns).count("z1") == 2
    table = pa.table({name: columns[name] for name in DECLARED}).append_column(
        "z1", pa.array(columns["z1"])
    )
    typed = Admg.from_edges(list(DECLARED), EDGES, [])
    for graph in (EDGES, typed):
        with pytest.raises(CausalValueError, match=r"duplicate column names: \['z1'\]"):
            run(frame, graph=graph)
        with pytest.raises(CausalValueError, match=r"duplicate column names: \['z1'\]"):
            run(table, graph=graph)


@pytest.mark.parametrize("kind", ["pandas", "arrow", "mapping"])
def test_a_declared_numeric_column_holding_strings_is_refused_with_its_name_and_type(kind):
    columns = declared_frame(500, 1039316)
    frame = pd.DataFrame({name: columns[name] for name in DECLARED})
    frame["z2"] = [f"level-{i % 3}" for i in range(500)]
    with pytest.raises(CausalTypeError, match=r"'z2'.*(str|string|<U)") as raised:
        run(containers(frame)[kind])
    assert "numeric" in str(raised.value)
    # An unused column of the same kind is not an error.
    frame["z2"] = columns["z2"]
    frame["unused"] = [f"level-{i % 3}" for i in range(500)]
    same(run(containers(frame)[kind]), run({name: columns[name] for name in DECLARED}))


# --- identifiers -----------------------------------------------------------------------


def labelled_study(
    n: int = 3000,
    seed: int = 7880483,
    *,
    missing_outcomes: int = 0,
    masked_rows: int = 0,
    groups: int = 0,
    icc: float = 0.0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """String-labelled participants (shuffled, so label order is not row order).

    ``masked_rows`` rows are excluded by the mask before the analysis (the frame is filtered
    and keeps a non-default index); ``missing_outcomes`` of the remaining rows lose their
    outcome. Returns the filtered frame, the missing-outcome positions and the masked-out rows
    of the original table.
    """
    columns = declared_frame(n, seed, icc=icc, groups=groups)
    rng = np.random.default_rng(seed + 7)
    labels = np.array([f"P-{k:05d}" for k in rng.permutation(n)])
    frame = pd.DataFrame({"pid": labels, **{name: columns[name] for name in DECLARED}})
    if groups:
        frame["site"] = [f"site-{code:02d}" for code in columns["site_code"]]
    masked = (
        np.sort(rng.choice(n, size=masked_rows, replace=False)) if masked_rows else np.empty(0, int)
    )
    kept = frame.drop(index=masked)
    holes = np.sort(rng.choice(len(kept), size=missing_outcomes, replace=False))
    if missing_outcomes:
        kept.iloc[holes, kept.columns.get_loc("y")] = np.nan
    return kept, holes, masked


def complete_case(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    done = frame.dropna(subset=list(DECLARED))
    return {name: done[name].to_numpy() for name in DECLARED}


def test_string_participant_labels_follow_the_rows_of_the_final_analysis_mask():
    frame, holes, masked = labelled_study(masked_rows=300, missing_outcomes=150)
    assert len(frame) == 2700 and frame.index.is_monotonic_increasing
    assert not frame.index.equals(pd.RangeIndex(len(frame)))
    result = run(frame, participant="pid")
    accounting = result.missing_data.accounting
    # Row accounting by exclusion reason: the masked rows left the table, 150 outcomes are gone.
    assert accounting.rows_total == 2700 and accounting.rows_kept == 2550
    assert accounting.rows_dropped == 150
    assert accounting.dropped_by_reason == {"missing_value": 150}
    assert {k: v for k, v in accounting.dropped_by_variable.items() if v} == {"y": 150}
    assert accounting.dropped_rows == tuple(int(i) for i in holes)
    # The identifiers named are those of the rows in the table, in the table's order.
    labels = tuple(frame["pid"])
    assert accounting.dropped_labels == tuple(labels[i] for i in holes)
    final_mask = tuple(label for i, label in enumerate(labels) if i not in set(holes.tolist()))
    assert len(final_mask) == 2550 and len(set(final_mask)) == 2550
    assert not set(final_mask) & set(accounting.dropped_labels)
    masked_labels = {f"P-{k:05d}" for k in range(3000)} - set(labels)
    assert len(masked_labels) == 300
    assert not masked_labels & (set(final_mask) | set(accounting.dropped_labels))
    # The estimate is that of the final mask alone.
    same(result, run(complete_case(frame)))


def cluster_robust(frame: pd.DataFrame) -> tuple[float, float, int]:
    """OLS of y on [1, t, z1, z2] over complete rows with the CR1 cluster sandwich, in numpy."""
    done = frame.dropna(subset=list(DECLARED))
    codes = pd.factorize(done["site"])[0]
    x = np.column_stack([np.ones(len(done)), done["t"], done["z1"], done["z2"]])
    y = done["y"].to_numpy()
    bread = np.linalg.inv(x.T @ x)
    beta = bread @ x.T @ y
    resid = y - x @ beta
    meat = np.zeros((4, 4))
    for code in np.unique(codes):
        score = x[codes == code].T @ resid[codes == code]
        meat += np.outer(score, score)
    n, p, g = len(y), x.shape[1], len(np.unique(codes))
    scale = g / (g - 1) * (n - 1) / (n - p)
    return float(beta[1]), float(np.sqrt((bread @ meat @ bread * scale)[1, 1])), g


def test_string_cluster_labels_drive_the_clustered_variance_over_the_final_mask():
    frame, _, _ = labelled_study(seed=4593730, groups=60, icc=0.1)
    assert frame["site"].nunique() == 60
    # One whole site and scattered outcomes of the others have no outcome: 59 clusters remain.
    lost = frame.index[frame["site"] == "site-07"]
    frame.loc[lost, "y"] = np.nan
    frame.loc[frame.index[::97], "y"] = np.nan
    ate, se, clusters = cluster_robust(frame)
    assert clusters == 59 and frame.dropna(subset=["y"])["site"].nunique() == 59
    estimator = LinearAdjustment(se="cluster", cluster="site")
    result = run(frame, estimator=estimator)
    assert result.estimate.ate == pytest.approx(ate, rel=1e-9)
    assert result.estimate.se_analytic == pytest.approx(se, rel=1e-9)
    # The count that enters the small-sample factor is the 59 sites of the mask, not 60.
    miscounted = se * np.sqrt((60 / 59) / (59 / 58))
    assert result.estimate.se_analytic != pytest.approx(miscounted, rel=1e-6)
    # Identical labels in another row order are the same clusters.
    reordered = run(
        require_success(frame.sample(frac=1.0, random_state=3)).reset_index(drop=True),
        estimator=estimator,
    )
    assert reordered.estimate.se_analytic == pytest.approx(se, rel=1e-9)


def test_one_shot_and_prepared_routes_report_the_same_identifiers_and_estimate():
    frame, holes, _ = labelled_study(seed=3653141, missing_outcomes=120, groups=30, icc=0.1)
    estimator = LinearAdjustment(se="cluster", cluster="site")
    one_shot = run(frame, estimator=estimator, participant="pid")
    study = ant.prepare(
        frame, graph=EDGES, query=QUERY, estimator=estimator, participant="pid", **ONE_SHOT
    )
    prepared = require_success(study.estimate())
    assert prepared.estimate.ate == one_shot.estimate.ate
    assert prepared.estimate.se_analytic == one_shot.estimate.se_analytic
    left, right = one_shot.missing_data.accounting, prepared.missing_data.accounting
    assert left.dropped_labels == right.dropped_labels == tuple(frame["pid"].iloc[holes])
    assert left.dropped_labels_digest == right.dropped_labels_digest
    assert left.dropped_digest == right.dropped_digest
    assert left.dropped_rows == right.dropped_rows
    assert study.participants == tuple(frame["pid"])
    # Replay against the same table rebuilds the same identifiers.
    replayed = study.reprepare(frame)
    assert replayed.participants == study.participants
    assert require_success(replayed.estimate()).estimate.ate == prepared.estimate.ate


def test_a_refresh_with_changed_identifiers_lists_them_and_marks_the_claim_for_recompute():
    frame, _, _ = labelled_study(seed=5310927)
    study = ant.prepare(frame, graph=EDGES, query=QUERY, participant="pid", **ONE_SHOT)
    first = require_success(study.estimate())
    changed_rows = np.sort(np.random.default_rng(5).choice(3000, size=25, replace=False))
    changed = frame.copy()
    changed.iloc[changed_rows, changed.columns.get_loc("pid")] = [
        f"REPLACED-{row:05d}" for row in changed_rows
    ]
    with pytest.raises(CausalValueError, match="frozen identifier column 'pid'") as raised:
        require_success(study.refresh(changed))
    receipt = raised.value.receipt
    assert receipt.column == "pid" and receipt.rows == 3000
    assert receipt.standing == "recompute" and receipt.duplicated == ()
    assert [change.row for change in receipt.changes] == [int(row) for row in changed_rows]
    for change in receipt.changes:
        assert change.frozen == frame["pid"].iloc[change.row]
        assert change.supplied == f"REPLACED-{change.row:05d}"
    assert "25 row(s) differ" in str(raised.value) and "recompute" in str(raised.value)
    # Nothing moved: the last result stands, and the unchanged column is accepted.
    assert study.last_result is first
    assert study.participants == tuple(frame["pid"])
    same(require_success(study.refresh(frame)), first)
    # Declaring the new identifiers is the way forward, and it binds them.
    rebound = study.reprepare(changed)
    assert rebound.participants == tuple(changed["pid"])


def test_missing_outcomes_keep_the_identity_of_every_dropped_and_retained_row():
    frame, holes, _ = labelled_study(seed=3416502, missing_outcomes=300)
    result = run(frame, participant="pid")
    accounting = result.missing_data.accounting
    labels = list(frame["pid"])
    assert labels != sorted(labels), "the original order is not label order"
    assert accounting.dropped_by_reason == {"missing_value": 300}
    assert {k: v for k, v in accounting.dropped_by_variable.items() if v} == {"y": 300}
    assert accounting.dropped_rows == tuple(int(i) for i in holes)
    assert accounting.dropped_labels == tuple(labels[i] for i in holes)
    dropped = set(accounting.dropped_labels)
    retained = [label for label in labels if label not in dropped]
    assert len(retained) == 2700 == accounting.rows_kept
    assert retained == [labels[i] for i in range(3000) if i not in set(holes.tolist())]
    # The same analysis on a table holding only the retained rows, in their original order.
    complete = frame.iloc[[i for i in range(3000) if i not in set(holes.tolist())]]
    same(result, run(complete_case(complete)))
    assert list(complete["pid"]) == retained


def test_an_invalid_refresh_leaves_the_prepared_study_exactly_as_it_was():
    frame, _, _ = labelled_study(seed=9438851, missing_outcomes=40)
    study = ant.prepare(frame, graph=EDGES, query=QUERY, participant="pid", **ONE_SHOT)
    first = require_success(study.estimate())
    before = bytes(study.export())
    participants = study.participants
    duplicated = frame.copy()
    duplicated.iloc[7, duplicated.columns.get_loc("pid")] = frame["pid"].iloc[3]
    with pytest.raises(CausalValueError, match="duplicate identifier on refresh") as raised:
        require_success(study.refresh(duplicated))
    assert raised.value.receipt.duplicated == (frame["pid"].iloc[3],)
    assert [change.row for change in raised.value.receipt.changes] == [7]
    assert frame["pid"].iloc[3] in str(raised.value)
    assert study.last_result is first
    assert study.participants == participants
    assert bytes(study.export()) == before
    again = require_success(study.estimate())
    same(again, first)
    # The exported artifact records the native work of its own estimate (the second click
    # reuses the retained design), so its bytes differ; the scientific identities do not.
    assert (again.claim_id, again.program_id, again.data_snapshot_id) == (
        first.claim_id,
        first.program_id,
        first.data_snapshot_id,
    )
    # The study still refreshes with a valid table afterwards.
    new_outcomes = declared_frame(3000, 9438852)
    valid = frame.copy()
    valid["y"] = new_outcomes["y"]
    refreshed = require_success(study.refresh(valid))
    assert refreshed.estimate.ate != pytest.approx(first.estimate.ate, rel=1e-6)
    assert study.participants == participants


def test_participants_survive_the_split_and_its_replay():
    n = 3000
    columns = declared_frame(n, 6371941)
    rng = np.random.default_rng(6371941)
    owners = np.array([f"person-{k:04d}" for k in rng.permutation(n // 2)])
    labels = np.repeat(owners, 2)[rng.permutation(n)]
    table = {**{name: columns[name] for name in DECLARED}, "pid": labels.tolist()}
    discovery_participants = set(owners[: len(owners) // 2])
    discovery_rows = [i for i, label in enumerate(labels) if label in discovery_participants]
    contract = sd.StaticContract(
        variables=DECLARED, ci_test="parcorr", alpha=0.01, fdr=None, seed=3
    )
    split = sd.discover_on_split(contract, table, discovery_rows=discovery_rows)
    discovery, analysis = split.participants(table, "pid")
    assert split.receipt.n_discovery + split.receipt.n_analysis == n
    assert discovery == tuple(labels[i] for i in split.discovery_row_ids)
    assert analysis == tuple(labels[i] for i in split.analysis_row_ids)
    assert set(discovery) == discovery_participants and not set(discovery) & set(analysis)
    assert len(discovery) + len(analysis) == n
    # Replay: the source table verifies against the receipt, and a second split of the same
    # source under the same rule reproduces the receipt, the identities and the estimate.
    split.verify()
    split.verify_source({name: table[name] for name in DECLARED})
    again = sd.discover_on_split(contract, table, discovery_rows=discovery_rows)
    assert again.receipt.identity == split.receipt.identity
    assert again.participants(table, "pid") == (discovery, analysis)
    estimates = []
    for part in (split, again):
        graph_nodes = part.analysis
        estimates.append(run({name: graph_nodes[name] for name in DECLARED}).estimate.ate)
    assert estimates[0] == estimates[1]
    # A split that puts one participant on both sides is refused by name.
    mixed = sd.discover_on_split(contract, table, discovery_rows=list(range(0, n, 2)))
    with pytest.raises(CausalValueError, match="both the discovery and the analysis"):
        mixed.participants(table, "pid")
