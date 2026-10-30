"""Cohort-checked projection and filter pushdown against the cohort worked out by hand.

The closure requires ``y`` (target), ``w`` (weight) and ``x`` (complete-case). Row by row:

====  ===  ===  ===  =====  =============================
row    y    w    x   extra  in the cohort (y, w, x seen)
====  ===  ===  ===  =====  =============================
0      1    1    1   nan    yes
1     nan   1    2    1     no
2      3    1   nan   1     no
3      4    1    4    1     yes
4      5   nan   5    1     no
5      6    1    6    1     yes
====  ===  ===  ===  =====  =============================
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import cohort_pushdown as cp
from antecedent.errors import CausalTypeError, CausalValueError

NAN = float("nan")
SOURCE = {
    "y": np.array([1.0, NAN, 3.0, 4.0, 5.0, 6.0]),
    "w": np.array([1.0, 1.0, 1.0, 1.0, NAN, 1.0]),
    "x": np.array([1.0, 2.0, NAN, 4.0, 5.0, 6.0]),
    "extra": np.array([NAN, 1.0, 1.0, 1.0, 1.0, 1.0]),
}
CLOSURE = cp.ColumnClosure(target=("y",), weights=("w",), complete_case=("x",))
COHORT = [0, 3, 5]


def test_object_missing_values_are_excluded_from_checked_cohorts():
    from decimal import Decimal

    values = np.array([1.0, np.float32("nan"), Decimal("NaN"), None, "present"], dtype=object)
    source = {"x": values, "extra": np.arange(len(values))}
    closure = cp.ColumnClosure(target=("x",))
    assert cp.checked_projection(source, closure, ["x"]).tolist() == [0, 4]
    assert cp.checked_push_filter(source, closure, "extra", [False] * len(values)).tolist() == [0, 4]


@pytest.mark.parametrize("dtype, values", [
    ("Int64", [1, None, 3]), ("Float64", [1.0, None, 3.0]),
    ("boolean", [True, None, False]), ("string", ["a", None, "b"]),
])
def test_pandas_nullable_missing_values_are_excluded(dtype, values):
    pd = pytest.importorskip("pandas")
    source = {"x": pd.Series(values, dtype=dtype), "extra": np.arange(3)}
    closure = cp.ColumnClosure(target=("x",))
    assert cp.checked_projection(source, closure, ["x"]).tolist() == [0, 2]
    assert cp.checked_push_filter(source, closure, "extra", [False] * 3).tolist() == [0, 2]
    assert cp._observed(np.array([pd.NaT, pd.NA, "ok"], dtype=object)) == [False, False, True]


def test_the_closure_lists_each_column_once_under_its_first_duty():
    closure = cp.ColumnClosure(target=("y", "x"), validators=("x",), complete_case=("w", "y"))
    assert closure.columns == ("y", "x", "w")


def test_a_projection_that_keeps_every_required_column_delivers_the_cohort():
    rows = cp.checked_projection(SOURCE, CLOSURE, ["y", "w", "x"])
    assert rows.tolist() == COHORT
    assert rows.dtype == np.int64


def test_a_projection_that_drops_a_required_column_is_refused():
    with pytest.raises(CausalValueError, match=r"pushdown refused: column 'w' must be retained"):
        cp.checked_projection(SOURCE, CLOSURE, ["y", "x"])


def test_a_projection_that_keeps_an_extra_column_changes_the_cohort_and_is_refused():
    # Completing cases on the extra column as well would drop row 0: [3, 5] is not [0, 3, 5].
    with pytest.raises(CausalValueError, match=r"pushdown rewrite: rewrite_legality\.sample_selection: column_filter_pushdown would alter sample selec"):
        cp.checked_projection(SOURCE, CLOSURE, ["y", "w", "x", "extra"])


def test_a_filter_that_drops_only_rows_outside_the_cohort_is_admitted():
    drop = [False, True, True, False, True, False]
    assert cp.checked_push_filter(SOURCE, CLOSURE, "extra", drop).tolist() == COHORT


def test_a_filter_that_drops_a_cohort_row_is_refused():
    with pytest.raises(CausalValueError, match=r"pushdown refused: column 'extra' must be retained"):
        cp.checked_push_filter(SOURCE, CLOSURE, "extra", [False, False, False, True, False, False])


def test_a_filter_mask_of_the_wrong_length_is_refused():
    with pytest.raises(CausalValueError, match=r"extra: expected length 6, got 2"):
        cp.checked_push_filter(SOURCE, CLOSURE, "extra", [False, True])


def test_missing_cells_are_read_from_nan_none_and_nat():
    source = {
        "y": np.array([1.0, 2.0, 3.0, 4.0]),
        "w": np.array([1, None, 3, 4], dtype=object),
        "x": np.array(["2020-01-01", "2020-01-02", "NaT", "2020-01-04"], dtype="datetime64[D]"),
    }
    rows = cp.checked_projection(source, CLOSURE, ["y", "w", "x"])
    assert rows.tolist() == [0, 3]


def test_unknown_columns_and_bad_declarations_are_refused():
    with pytest.raises(CausalValueError, match=r"source has no column 'w'"):
        cp.checked_projection({"y": SOURCE["y"]}, CLOSURE, ["y", "w", "x"])
    with pytest.raises(CausalTypeError, match=r"keep must be a sequence of column names, not one string"):
        cp.checked_projection(SOURCE, CLOSURE, "yw")
    with pytest.raises(CausalTypeError, match=r"closure must be a ColumnClosure"):
        cp.checked_push_filter(SOURCE, ("y",), "extra", [False] * 6)  # type: ignore[arg-type]
