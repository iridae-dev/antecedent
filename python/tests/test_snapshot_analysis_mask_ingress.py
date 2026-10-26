"""Exact native views retain row scope; transformed arrays declare a new table."""

import antecedent as ant
import numpy as np
import pytest
from antecedent.estimation import CandidateScreen
from antecedent.outcomes import require_success

from test_prepared_batch_lifecycle import _inputs, _prepare


def _screened_member():
    data = _inputs(400)
    data["y"] += 4 * data["t"] * (np.arange(400) >= 200)
    screen = CandidateScreen("split", "unrecorded", tuple(range(200)), tuple(range(200, 400)))
    batch = require_success(_prepare(data, candidate_screen=screen).estimate())
    assert batch.retarget_rows() == tuple(range(200, 400))
    return batch, batch.member(0)


def _analyze(data, batch):
    return require_success(
        ant.analyze(
            data,
            query=ant.AverageEffect("t", "y"),
            graph=batch.study._declarations["graph"],
            estimator="linear.adjustment.ate",
            inference=ant.Frequentist(bootstrap=0),
            refute=False,
        )
    )


@pytest.mark.parametrize("view", ["original", "full_slice", "numpy_view", "memoryview"])
def test_complete_native_views_retain_screened_metadata_without_copying(view):
    batch, member = _screened_member()
    table = member.study._data
    assert table.num_rows == 400  # Physical storage stays full; the mask selects analysis rows.
    transforms = {
        "original": lambda column: column,
        "full_slice": lambda column: column[:],
        "numpy_view": lambda column: column.view(),
        "memoryview": lambda column: np.asarray(memoryview(column)),
    }
    columns = {name: transforms[view](table[name]) for name in table.column_names}
    shared = ant.data.table(columns)
    assert shared.bytes_copied == 0
    assert shared.num_rows == 400
    assert all(np.shares_memory(shared[name], table[name]) for name in table.column_names)
    result = _analyze(shared, batch)
    assert result.effect == pytest.approx(6, abs=0.3)
    original = require_success(member)
    direct = _analyze(table, batch)
    # A canonical execution snapshot also binds its observation contract and
    # estimator-specific checked fit moments. Cross-estimator equality would
    # erase that authority; the source partition still identifies the exact
    # same physical rows, values, validity, weights and analysis mask.
    snapshot = result.inspect().contract["data_snapshot"]
    original_snapshot = original.inspect().contract["data_snapshot"]
    assert snapshot["partitions"] == original_snapshot["partitions"]
    assert snapshot["row_count"] == original_snapshot["row_count"] == 400
    assert snapshot["linear_fit_moments"]["complete_case_rows"] == 200
    assert result.data_snapshot_id == direct.data_snapshot_id
    assert result.effect == direct.effect
    # Re-running the old retained execution still has its exact row identities.
    assert batch.retarget_rows() == tuple(range(200, 400))


def test_native_projection_and_reordered_columns_preserve_analysis_mask():
    batch, member = _screened_member()
    table = member.study._data
    names = tuple(reversed(table.column_names))
    projected = ant.data.table({name: table[name] for name in names})
    assert projected.column_names == names
    assert projected.bytes_copied == 0
    assert all(np.shares_memory(projected[name], table[name]) for name in names)
    assert _analyze(projected, batch).effect == pytest.approx(6, abs=0.3)


@pytest.mark.parametrize("transform", ["copy", "slice", "reverse", "arithmetic"])
def test_transformed_arrays_never_inherit_native_table_mask(transform):
    batch, member = _screened_member()
    table = member.study._data
    transforms = {
        "copy": lambda column: column.copy(),
        "slice": lambda column: column[::2],
        "reverse": lambda column: column[::-1],
        "arithmetic": lambda column: column + 0.0,
    }
    columns = {name: transforms[transform](table[name]) for name in table.column_names}
    # Explicitly copied values provide the oracle for this new, unmasked dataset.
    expected = _analyze({name: column.copy() for name, column in columns.items()}, batch)
    result = _analyze(ant.data.table(columns), batch)
    assert result.effect == expected.effect
    assert result.effect == pytest.approx(4, abs=0.4)
    assert abs(result.effect - _analyze(table, batch).effect) > 1
    assert result.data_snapshot_id != require_success(member).data_snapshot_id


def test_mixed_native_and_materialized_columns_do_not_infer_a_table_mask():
    batch, member = _screened_member()
    table = member.study._data
    columns = {name: table[name] for name in table.column_names}
    columns["y"] = columns["y"].copy()
    result = _analyze(ant.data.table(columns), batch)
    expected = _analyze({name: column.copy() for name, column in columns.items()}, batch)
    assert result.effect == expected.effect
    assert result.effect == pytest.approx(4, abs=0.4)
    assert result.data_snapshot_id != require_success(member).data_snapshot_id


@pytest.mark.parametrize("rename", [False, True], ids=["mixed", "renamed"])
def test_nullable_native_columns_preserve_missing_rows_in_a_new_table(rename):
    import pyarrow as pa

    t = np.tile([0.0, 1.0], 120)
    z = np.repeat(np.arange(60, dtype=float), 4)
    y = 3 * t + 0.4 * z + np.sin(np.arange(240)) * 0.01
    valid = np.ones(240, dtype=bool)
    valid[[7, 111]] = False
    raw_z = z.copy()
    raw_z[~valid] = 1_000_000.0  # Arbitrary Arrow null-slot payloads are not observations.
    owned = ant.data.table(pa.table({"t": t, "y": y, "z": pa.array(raw_z, mask=~valid)}))
    covariate = "renamed_z" if rename else "z"
    mixed = ant.data.table({"t": owned["t"], "y": owned["y"].copy(), covariate: owned["z"]})
    assert np.isnan(mixed[covariate][~valid]).all()
    assert np.shares_memory(mixed[covariate], owned["z"])
    oracle_z = z.copy()
    oracle_z[~valid] = np.nan
    declarations = dict(
        graph=[(covariate, "t"), (covariate, "y"), ("t", "y")],
        query=ant.AverageEffect("t", "y"),
        estimator="linear.adjustment.ate",
        inference=ant.Frequentist(bootstrap=0),
        refute=False,
    )
    actual = require_success(ant.analyze(mixed, **declarations))
    oracle = require_success(ant.analyze({"t": t, "y": y, covariate: oracle_z}, **declarations))
    assert actual.effect == oracle.effect
    assert actual.effect == pytest.approx(3.0, abs=0.01)
    snapshot = actual.inspect().contract["data_snapshot"]
    assert snapshot["linear_fit_moments"]["complete_case_rows"] == 238
