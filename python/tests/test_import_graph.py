"""The package's module-level import graph must stay acyclic.

CodeQL reports `py/cyclic-import` and `py/unsafe-cyclic-import` across this package. Those
findings are false positives *given the current structure*, and this module pins the
property that makes them false so the situation cannot silently change.

The hazard `py/unsafe-cyclic-import` describes is real but specific: module A imports B at
**module scope**, B imports A at module scope, and A refers to a name defined after B's
import — so some import order raises `NameError` on a half-initialised module. That
requires a cycle among *module-scope* imports.

This package has none. Its cycles exist only once you also count imports written inside
function bodies (which run at call time, long after every module is initialised) and inside
`if TYPE_CHECKING:` blocks (which never run at all). CodeQL does not distinguish those from
module-scope imports, so it reports a cycle where no order-dependence can exist.

`test_module_level_import_graph_is_acyclic` is the guard that matters: it fails the moment
someone promotes one of those deferred imports to module scope, which is exactly when the
hazard would stop being hypothetical.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

# Deliberately no `importorskip("antecedent")`: these checks are pure AST and subprocess
# work, and the acyclicity guard is most valuable exactly when the package has become
# unimportable. Requiring the import here would turn its explanatory failure into a bare
# collection error.

PKG = Path(__file__).resolve().parent.parent / "antecedent"


def _module_names() -> list[str]:
    return sorted(p.stem for p in PKG.glob("*.py") if p.stem != "__init__")


def _module_scope_edges() -> dict[str, set[str]]:
    """Intra-package imports written at module scope, excluding `TYPE_CHECKING` blocks."""
    mods = set(_module_names()) | {"__init__"}
    edges: dict[str, set[str]] = {}

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.depth = 0  # inside a function body
            self.guarded = 0  # inside `if TYPE_CHECKING:`
            self.found: set[str] = set()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

        def visit_If(self, node: ast.If) -> None:
            guard = "TYPE_CHECKING" in ast.unparse(node.test)
            self.guarded += guard
            self.generic_visit(node)
            self.guarded -= guard

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.level == 1 and node.module and not (self.depth or self.guarded):
                target = node.module.split(".")[0]
                if target in mods:
                    self.found.add(target)
            self.generic_visit(node)

    for path in sorted(PKG.glob("*.py")):
        visitor = Visitor()
        visitor.visit(ast.parse(path.read_text()))
        edges[path.stem] = visitor.found
    return edges


def test_module_level_import_graph_is_acyclic() -> None:
    edges = _module_scope_edges()
    cycles: set[tuple[str, ...]] = set()

    def walk(node: str, stack: list[str], seen: set[str]) -> None:
        for nxt in sorted(edges.get(node, ())):
            if nxt in stack:
                cycles.add(tuple(stack[stack.index(nxt) :] + [nxt]))
            elif nxt not in seen:
                seen.add(nxt)
                walk(nxt, [*stack, nxt], seen)

    for node in sorted(edges):
        walk(node, [node], {node})

    assert not cycles, (
        "module-scope import cycle(s) introduced: "
        + "; ".join(" -> ".join(c) for c in sorted(cycles))
        + ". A deferred (function-local or TYPE_CHECKING) import was promoted to module "
        "scope. That makes initialisation order load-bearing and turns CodeQL's "
        "py/unsafe-cyclic-import from a false positive into a real defect — move it back, "
        "or relocate the shared name to a leaf module."
    )


@pytest.mark.parametrize("module", _module_names())
def test_module_imports_standalone(module: str) -> None:
    """Importing any single module first, in a fresh interpreter, must succeed.

    Direct check that no module depends on another having been imported before it.
    """
    proc = subprocess.run(
        [sys.executable, "-c", f"import antecedent.{module}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"import antecedent.{module} failed:\n{proc.stderr}"
