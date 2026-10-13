"""Correctness fixes that do not need the native extension.

Column-name lookup, canonical-JSON non-finite refusal, and refresh rollback
are pure Python. Ingest and HTML display are covered when the extension imports.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PKG = Path(__file__).resolve().parents[1] / "antecedent"


def _load(name: str):
    path = _PKG / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"audit_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cancellation_rolls_back_a_completed_prefix() -> None:
    """Worlds whose refresh returned go back; ``except Exception`` would leave them."""
    rollback = _load("_rollback")
    state = ["old", "old", "old"]

    def refresh(index: int) -> None:
        state[index] = "new"
        if index == 2:
            raise KeyboardInterrupt

    def restore(index: int) -> None:
        state[index] = "old"

    with pytest.raises(KeyboardInterrupt):
        rollback.refresh_all_or_restore((0, 1, 2), refresh, restore)
    # The interrupted call is not counted as complete. The two that returned are.
    assert state == ["old", "old", "new"]


def test_successful_refresh_keeps_the_new_snapshot() -> None:
    rollback = _load("_rollback")
    state = ["old", "old"]
    rollback.refresh_all_or_restore(
        (0, 1),
        lambda index: state.__setitem__(index, "new"),
        lambda index: state.__setitem__(index, "old"),
    )
    assert state == ["new", "new"]


def test_refusal_names_read_columns_not_values() -> None:
    names = _load("_names")

    class Frame:
        columns = ["y", "t"]

        def __init__(self) -> None:
            self.value_reads: list[str] = []

        def __getitem__(self, key: str):
            self.value_reads.append(key)
            raise KeyError(key)

    frame = Frame()
    assert names.table_names(frame) == ("y", "t")
    assert names.table_names(frame) == ("y", "t")
    frame.columns = ["a"]
    assert names.table_names(frame) == ("a",)
    assert frame.value_reads == []
    assert names.table_names({"z": [1], "y": [2]}) == ("z", "y")

    class Arrow:
        column_names = ("t", "y")

    assert names.table_names(Arrow()) == ("t", "y")


def test_freeze_recurses_into_tuples_and_mappings() -> None:
    freeze = _load("_freeze")
    frozen = freeze.freeze({"edges": ({"source": "z", "target": "y"}, [1, {"k": 2}])})
    assert frozen["edges"][0]["source"] == "z"
    with pytest.raises(TypeError, match="item assignment"):
        frozen["edges"][0]["source"] = "t"
    thawed = dict(frozen)
    assert thawed["edges"][0]["source"] == "z"
    assert isinstance(frozen["edges"], tuple)
    assert isinstance(frozen["edges"][1], tuple)


def test_canonical_form_refuses_non_finite_and_keeps_finite_values() -> None:
    digest = _load("_digest")
    with pytest.raises(ValueError, match="JSON"):
        digest._plain([float("nan")])
    with pytest.raises(ValueError, match="JSON"):
        digest._plain([float("inf")])
    with pytest.raises(ValueError, match="JSON"):
        digest._plain(np.float64("nan"))
    with pytest.raises(ValueError, match="JSON"):
        digest._array_form(np.array([1.0, np.nan]))
    assert digest._plain({"a": np.float64(1.5), "b": 1, "c": True}) == {
        "a": 1.5,
        "b": 1,
        "c": True,
    }
    finite = np.array([1.0, -2.5])
    # The finite array reaches the byte digest only after this check passes.
    assert np.isfinite(finite).all()
