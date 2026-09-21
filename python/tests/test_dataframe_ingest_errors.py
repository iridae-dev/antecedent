"""DataFrame / mapping ingest errors must name columns and suggest next steps."""

from __future__ import annotations

import numpy as np
import pytest
from antecedent._data import as_columns, to_f64
from antecedent.errors import CausalTypeError, CausalValueError


def test_object_dtype_names_column_and_suggests_encoding() -> None:
    data = {"x": np.array([1.0, 2.0]), "city": np.array(["a", "b"], dtype=object)}
    with pytest.raises(CausalTypeError, match=r"city") as excinfo:
        as_columns(data)
    msg = str(excinfo.value)
    assert "object-dtype columns are not supported" not in msg
    assert "categor" in msg or "one-hot" in msg or "exclude" in msg


def test_shape_error_names_column() -> None:
    data = {"ok": np.array([1.0, 2.0]), "wide": np.array([[1.0, 2.0], [3.0, 4.0]])}
    with pytest.raises(CausalValueError, match=r"wide") as excinfo:
        as_columns(data)
    assert "shape" in str(excinfo.value)


def test_length_mismatch_names_columns_and_lengths() -> None:
    data = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([1.0, 2.0])}
    with pytest.raises(ValueError, match=r"column length mismatch") as excinfo:
        as_columns(data)
    msg = str(excinfo.value)
    assert msg != "column length mismatch"
    assert "a=3" in msg and "b=2" in msg


def test_to_f64_rejects_object_before_cast() -> None:
    with pytest.raises(CausalTypeError, match=r"label"):
        to_f64(np.array(["x", "y"], dtype=object), name="label")
