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

from _repo_text import REPO_ROOT, load_json, read_text

_ROOT = REPO_ROOT
_EXAMPLES = sorted((_ROOT / "examples" / "python").glob("*.py"))
_NOTEBOOKS = sorted((_ROOT / "examples" / "notebooks").glob("*.ipynb"))


def _asserts(source: str) -> int:
    """Checks of the answer: `assert` statements and explicit `raise AssertionError`."""
    count = 0
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assert):
            count += 1
        elif isinstance(node, ast.Raise) and node.exc is not None:
            raised = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            count += isinstance(raised, ast.Name) and raised.id == "AssertionError"
    return count


def test_examples_are_found():
    assert len(_EXAMPLES) >= 16
    assert len(_NOTEBOOKS) >= 5


@pytest.mark.parametrize("path", _EXAMPLES, ids=lambda p: p.name)
def test_example_asserts_its_own_result(path):
    assert _asserts(read_text(path)) >= 1, f"{path.name} prints an answer but never checks it"


@pytest.mark.parametrize("path", _EXAMPLES, ids=lambda p: p.name)
def test_example_runs(path, monkeypatch, tmp_path):
    # `runpy.run_path` does not touch `sys.argv`: an example whose `__main__` block
    # reads it (`bench_python_overhead.py`'s `argparse.ArgumentParser`) would
    # otherwise inherit pytest's own argv (`-q`, the node id, `--tb=...`, ...) and
    # fail with "unrecognized arguments" under any normal pytest invocation. Each
    # example must run the way a reader invoking it directly would, using its
    # own arguments rather than pytest's.
    argv = [str(path)]
    # Exercise every benchmark workload and input format without repeating its
    # timing samples. Correctness belongs here; performance measurements use
    # the example's normal 25-iteration invocation.
    if path.name == "bench_python_overhead.py":
        argv.append("--smoke")
    # A protocol script reads its lane and output directory from the command line; one
    # reduced trial of the linear lane exercises fit, export and fresh-process replay.
    if path.name == "cyclic_paper_experiments.py":
        pytest.importorskip("scipy")
        argv += [
            "--lane",
            "nodags-linear",
            "--output-dir",
            str(tmp_path),
            "--trials",
            "1",
            "--max-iterations",
            "50",
        ]
    # The Parquet example writes its input with PyArrow; without it the example only
    # prints its skip, so the run is reported as skipped rather than as checked.
    if path.name == "parquet_e2e.py":
        pytest.importorskip("pyarrow")
    monkeypatch.setattr("sys.argv", argv)
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
