"""Every shipped example and README-linked notebook runs, and asserts something.

`examples/python/*.py` and the notebooks' code cells are the front door a reader copies.
Name resolution alone (`test_notebook_api_surface`) cannot see a wrong number, a panic or
a refusal, so each is executed here. An example or notebook with no `assert` would pass
by not raising while printing a wrong answer, so each must carry at least one.
"""

from __future__ import annotations

import ast
import os
import runpy
from pathlib import Path

import pytest

from _repo_text import load_json, read_text

pytest.importorskip("antecedent")

_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = sorted((_ROOT / "examples" / "python").glob("*.py"))
_NOTEBOOKS = sorted((_ROOT / "examples" / "notebooks").glob("*.ipynb"))


def _asserts(source: str) -> int:
    return sum(isinstance(node, ast.Assert) for node in ast.walk(ast.parse(source)))


def test_examples_are_found():
    assert len(_EXAMPLES) >= 16
    assert len(_NOTEBOOKS) >= 5


@pytest.mark.parametrize("path", _EXAMPLES, ids=lambda p: p.name)
def test_example_asserts_its_own_result(path):
    assert _asserts(read_text(path)) >= 1, f"{path.name} prints an answer but never checks it"


@pytest.mark.parametrize("path", _EXAMPLES, ids=lambda p: p.name)
def test_example_runs(path, monkeypatch):
    # `runpy.run_path` does not touch `sys.argv`: an example whose `__main__` block
    # reads it (`bench_python_overhead.py`'s `argparse.ArgumentParser`) would
    # otherwise inherit pytest's own argv (`-q`, the node id, `--tb=...`, ...) and
    # fail with "unrecognized arguments" under any normal pytest invocation. Each
    # example must run the way a reader invoking it directly would: with no
    # arguments beyond its own name.
    monkeypatch.setattr("sys.argv", [str(path)])
    runpy.run_path(str(path), run_name="__main__")


def _notebook_source(path: Path) -> str:
    """Code cells joined, without shell/magic lines (the kernel-only conveniences)."""
    cells = [c for c in load_json(path)["cells"] if c.get("cell_type") == "code"]
    lines = []
    for cell in cells:
        for line in "".join(cell["source"]).splitlines():
            lines.append("" if line.lstrip().startswith(("%", "!")) else line)
        lines.append("")
    return "\n".join(lines)


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_asserts_its_own_result(path):
    assert _asserts(_notebook_source(path)) >= 1, f"{path.name} never checks an answer"


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs(path, monkeypatch):
    pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.chdir(os.fspath(path.parent))
    exec(compile(_notebook_source(path), str(path), "exec"), {"__name__": "__main__"})  # noqa: S102
