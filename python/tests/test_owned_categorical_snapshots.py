"""Owned mixed-schema data preserves labels and zero-copy numeric consumers."""

import gc

import antecedent as ant
import numpy as np
import pytest
from antecedent import _native
from antecedent.data import table
from antecedent.outcomes import require_success


def test_categorical_snapshot_owns_labels_and_shares_numeric_native_allocations():
    values = np.arange(12.0)
    subjects = np.array([f"s{i}" for i in range(12)])
    groups = np.where(values < 6, "a", "b")
    snapshot = table({"value": values, "subject": subjects, "group": groups})
    assert snapshot.bytes_copied == values.nbytes + subjects.nbytes + groups.nbytes
    assert tuple(snapshot) == ("value", "subject", "group")
    values[:] = -1
    subjects[:] = "bad"
    groups[:] = "z"
    assert snapshot["value"].tolist() == list(range(12))
    assert snapshot["subject"].tolist() == [f"s{i}" for i in range(12)]
    assert snapshot["group"].tolist() == ["a"] * 6 + ["b"] * 6
    for name in snapshot:
        assert not snapshot[name].flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            snapshot[name].setflags(write=True)
    numeric = snapshot.select(["value"])
    assert np.shares_memory(numeric["value"], snapshot["value"])
    names, columns = numeric._native_columns()
    assert isinstance(columns[0], np.ndarray)
    assert _native.load_float64_columns(names, columns).bytes_copied == 0
    # The Python/NumPy view pins the native owner independently of every Table.
    retained = columns[0]
    del columns, snapshot, numeric
    gc.collect()
    assert retained.tolist() == list(range(12))


def test_exact_object_labels_and_all_categorical_selection_stay_immutable():
    labels = np.array(["one", "two"], dtype=object)
    exact_ids = np.array([2**60, 2**60 + 1], dtype=np.int64)
    snapshot = table({"label": labels, "id": exact_ids})
    labels[0] = "changed"
    exact_ids[0] = 0
    assert snapshot["label"] == ("one", "two")
    assert snapshot["id"].tolist() == [2**60, 2**60 + 1]
    selected = snapshot.select(["id", "label"])
    assert selected.num_rows == 2
    assert tuple(selected) == ("id", "label")
    assert selected["label"] is snapshot["label"]
    assert np.shares_memory(selected["id"], snapshot["id"])


def test_panel_preparation_retains_subject_labels_after_source_mutation():
    ids = np.array([f"s{i}" for i in range(8) for _ in range(2)])
    treated = np.repeat(np.arange(8) < 4, 2)
    post = np.tile([False, True], 8)
    outcome = 10.0 + np.repeat(np.arange(8), 2) + 3 * post + 4 * (treated & post)
    prepared = ant.prepare(
        {"outcome": outcome, "id": ids, "treated": treated, "post": post},
        query=ant.quasi.PanelDifferenceInDifferences("outcome", "id", "treated", "post"),
    )
    ids[:] = "bad"
    outcome[:] = -1
    result = require_success(prepared.estimate()).panel_did
    assert result.estimate == pytest.approx(4.0)
    assert result.clusters == 8
    assert "complete_pre_post_panel" in result.assumptions


def test_conditional_dose_policy_keeps_group_labels_for_native_scientific_route():
    rows = 800
    index = np.arange(rows)
    dose = ((index * 37) % 997) / 997.0
    group = np.where(index < rows // 2, "a", "b")
    outcome = (
        1.0
        + (index >= rows // 2).astype(float)
        + 0.35 * np.sin(0.13 * index)
        + np.where(index < rows // 2, 1.5, 2.0) * dose
    )
    prepared = ant.prepare(
        {"outcome": outcome, "dose": dose, "density": np.ones(rows), "group": group},
        query=ant.policy.ConditionalDoseResponse(
            "outcome",
            "dose",
            "group",
            "density",
            (),
            0.2,
            "known",
            min_local_support=80,
            policy_doses={"a": 0.7, "b": 0.8},
            reference_doses={"a": 0.3, "b": 0.4},
        ),
        refute="none",
    )
    group[:] = "z"
    response = require_success(prepared.estimate()).continuous_dose_response
    assert response.support_status == "licensed"
    assert response.fixed_policy.incremental_value == pytest.approx(0.7, abs=0.15)


def test_dataframe_labels_preserve_the_native_numeric_view_boundary():
    pd = pytest.importorskip("pandas")
    snapshot = table(pd.DataFrame({"value": [1.0, 2.0], "group": ["a", "b"]}))
    assert tuple(snapshot["group"]) == ("a", "b")
    names, columns = snapshot.select(["value"])._native_columns()
    assert _native.load_float64_columns(names, columns).bytes_copied == 0


def test_mixed_arrow_table_keeps_numeric_borrowing_explicit_and_labels_owned():
    pa = pytest.importorskip("pyarrow")
    values = np.arange(3.0)
    original = pa.table({"value": pa.array(values), "group": ["a", "b", "a"]})
    owned = table(original)
    borrowed = table(original, borrow=True)
    assert borrowed.bytes_borrowed == values.nbytes
    # Object labels own tuple slots; their immutable string scalars can be shared.
    assert borrowed.bytes_copied == len(values) * np.dtype(np.intp).itemsize
    assert owned.bytes_borrowed == 0
    assert owned.bytes_copied == values.nbytes + borrowed.bytes_copied
    assert tuple(owned["group"]) == ("a", "b", "a")
    values[:] = -1
    assert owned["value"].tolist() == [0.0, 1.0, 2.0]
