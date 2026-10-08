"""The 2.3 stage modules resolve lazily on the package root and are listed by ``dir``.

``import antecedent as at; at.recalc_cell`` works without an explicit submodule import (PEP 562
``__getattr__``), ``dir(at)`` lists every lazy module for tab completion, and none of them
widens the frozen ``__all__``.
"""

from __future__ import annotations

import importlib
import types
from pathlib import Path

import antecedent as at
import pytest

_PKG = Path(at.__file__).resolve().parent


def test_lazy_module_tuple_is_unique_and_names_real_modules() -> None:
    names = at._LAZY_MODULES
    assert len(names) == len(set(names))
    for name in names:
        assert (_PKG / f"{name}.py").is_file(), f"{name} is not a module in the package"


@pytest.mark.parametrize("name", at._LAZY_MODULES)
def test_lazy_attribute_access_returns_the_submodule(name: str) -> None:
    module = getattr(at, name)
    assert isinstance(module, types.ModuleType)
    assert module is importlib.import_module(f"antecedent.{name}")
    assert module.__name__ == f"antecedent.{name}"
    # Cached: the second access is the attribute set on the package, not a re-import.
    assert at.__dict__[name] is module


@pytest.mark.parametrize("name", at._LAZY_MODULES)
def test_lazy_modules_are_listed_by_dir_and_stay_out_of_all(name: str) -> None:
    assert name in dir(at)
    assert name not in at.__all__


def test_dir_hides_the_future_annotations_binding() -> None:
    listing = dir(at)
    assert "annotations" not in listing
    assert listing == sorted(listing)
    # The ordinary root surface is still listed.
    for name in ("analyze", "AverageEffect", "design", "decision", "__version__"):
        assert name in listing


def test_the_lazy_module_names_do_not_shadow_root_exports() -> None:
    # `identify` is a root function; the module of the same name must never be lazy.
    assert "identify" not in at._LAZY_MODULES
    assert set(at._LAZY_MODULES).isdisjoint(at.__all__)


def test_unknown_attribute_raises_the_standard_attribute_error() -> None:
    with pytest.raises(AttributeError) as caught:
        _ = at.no_such_stage
    assert str(caught.value) == "module 'antecedent' has no attribute 'no_such_stage'"
    assert getattr(at, "no_such_stage", None) is None
    assert not hasattr(at, "no_such_stage")


def test_retired_name_signposts_survive_the_lazy_hook() -> None:
    with pytest.raises(AttributeError, match="renamed to antecedent.priors"):
        _ = at.prior_bank
    with pytest.raises(AttributeError, match="discovery config dataclasses"):
        _ = at.discover_pc
