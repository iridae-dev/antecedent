"""One closure walk for a cited Python test.

Both evidence gates use this walk. Membership comes from the AST, so a name
inside a string or a comment does not pull in another function. ``constants``
adds module-level ``UPPER_CASE`` assignments, which the fixture reader follows
and the assertion reader does not.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_CONST = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def _index(tree: ast.Module, *, constants: bool) -> dict[str, ast.AST]:
    defs: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs[node.name] = node
        elif constants and isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and _CONST.match(target.id):
                    defs[target.id] = node
        elif (
            constants
            and isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and _CONST.match(node.target.id)
        ):
            defs[node.target.id] = node
    return defs


def closure_nodes(
    tree: ast.Module, name: str, *, constants: bool = False
) -> list[ast.AST]:
    """The function ``name`` plus every module-level function its body names."""
    defs = _index(tree, constants=constants)
    seen: set[str] = set()
    todo = [name]
    out: list[ast.AST] = []
    while todo:
        current = todo.pop()
        if current in seen or current not in defs:
            continue
        seen.add(current)
        node = defs[current]
        out.append(node)
        todo.extend(n.id for n in ast.walk(node) if isinstance(n, ast.Name))
    return out


def closure_source(path: Path, name: str) -> str:
    """Source of ``closure_nodes`` for ``name``, including constant assignments."""
    text = path.read_text(errors="ignore", encoding="utf-8")
    tree = ast.parse(text)
    parts = []
    for node in closure_nodes(tree, name, constants=True):
        segment = ast.get_source_segment(text, node)
        if segment:
            parts.append(segment)
    return "\n".join(parts)
