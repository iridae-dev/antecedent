"""One closure walk for a cited Python test.

Both evidence gates use this walk. Membership comes from the AST, so a name
inside a string or a comment does not pull in another function. ``constants``
adds module-level ``UPPER_CASE`` assignments, which the fixture reader follows
and the assertion reader does not.
"""

from __future__ import annotations

import ast
import functools
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


def closure_nodes(tree: ast.Module, name: str, *, constants: bool = False) -> list[ast.AST]:
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


@functools.lru_cache(maxsize=512)
def _parsed(path: Path, mtime_ns: int, size: int):
    """(text, tree, lines) of one file revision; parsed and split once."""
    text = path.read_text(errors="ignore", encoding="utf-8")
    tree = ast.parse(text)
    split = getattr(ast, "_splitlines_no_ff", None)
    return text, tree, (split(text) if split else None)


def _segment(text: str, lines: list[str] | None, node: ast.AST) -> str | None:
    """``ast.get_source_segment`` without re-splitting the file for every node."""
    if lines is None or getattr(node, "end_lineno", None) is None:
        return ast.get_source_segment(text, node)
    first, last = node.lineno - 1, node.end_lineno - 1
    start, end = node.col_offset, node.end_col_offset
    if first == last:
        return lines[first].encode()[start:end].decode()
    head = lines[first].encode()[start:].decode()
    tail = lines[last].encode()[:end].decode()
    return "".join([head, *lines[first + 1 : last], tail])


def closure_source(path: Path, name: str) -> str:
    """Source of ``closure_nodes`` for ``name``, including constant assignments."""
    stat = path.stat()
    text, tree, lines = _parsed(path, stat.st_mtime_ns, stat.st_size)
    parts = []
    for node in closure_nodes(tree, name, constants=True):
        segment = _segment(text, lines, node)
        if segment:
            parts.append(segment)
    return "\n".join(parts)
