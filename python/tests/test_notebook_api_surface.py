"""Every `antecedent` symbol the README-linked notebooks use must still exist.

The notebooks are the top of the funnel and are not executed by CI (they need
matplotlib/pandas and a kernel). A deleted-or-moved name therefore survives in
them silently: the 0.4.0 namespace freeze left a mid-cell
`antecedent.discover_pc(...)` call behind that no gate caught, because import
lines were checked and attribute accesses were not.

This resolves, without executing anything, every dotted `antecedent.…` path and
every `from antecedent[.…] import …` name appearing in the notebooks' code cells.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("antecedent")
import antecedent

_NOTEBOOKS = sorted(
    (Path(__file__).resolve().parents[2] / "examples" / "notebooks").glob("*.ipynb")
)

# `antecedent.a.b.c` — capture the dotted tail so submodule paths resolve too.
_ATTR = re.compile(r"\bantecedent((?:\.[A-Za-z_][A-Za-z0-9_]*)+)")
_FROM = re.compile(
    r"^\s*from\s+(antecedent(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s+import\s+([^\n#]+)", re.M
)


def _code(nb: Path) -> str:
    doc = json.loads(nb.read_text())
    return "\n".join("".join(c["source"]) for c in doc["cells"] if c.get("cell_type") == "code")


def _resolve(path: str):
    """Walk a dotted path from the package root, importing submodules as needed."""
    import importlib

    obj = antecedent
    walked = "antecedent"
    for part in path.split("."):
        walked = f"{walked}.{part}"
        try:
            obj = getattr(obj, part)
        except AttributeError:
            obj = importlib.import_module(walked)
    return obj


def test_notebooks_exist():
    assert _NOTEBOOKS, "expected README-linked notebooks under examples/notebooks"


@pytest.mark.parametrize("nb", _NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_attribute_paths_resolve(nb):
    src = _code(nb)
    unresolved = []
    for match in _ATTR.finditer(src):
        dotted = match.group(1).lstrip(".")
        try:
            _resolve(dotted)
        except (AttributeError, ModuleNotFoundError):
            # Only the first segment must be a real package member; trailing
            # segments may be attributes of a *result* object, not of the package.
            head = dotted.split(".")[0]
            try:
                _resolve(head)
            except (AttributeError, ModuleNotFoundError):
                unresolved.append(f"antecedent.{dotted}")
    assert not unresolved, f"{nb.name} uses names that no longer exist: {sorted(set(unresolved))}"


@pytest.mark.parametrize("nb", _NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_from_imports_resolve(nb):
    src = _code(nb)
    missing = []
    for module, names in _FROM.findall(src):
        for raw in names.split(","):
            name = raw.strip().split(" as ")[0].strip().strip("()")
            if not name:
                continue
            try:
                _resolve(f"{module}.{name}".removeprefix("antecedent."))
            except (AttributeError, ModuleNotFoundError):
                missing.append(f"{module}.{name}")
    assert not missing, f"{nb.name} imports names that no longer exist: {sorted(set(missing))}"
