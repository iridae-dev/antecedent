"""Resolve generated reason-detail constants back to the ids they spell.

`parity/reason_details.toml` registers every refusal detail id once; the generated Rust module
(`reason_details::<family>::NAME`) and Python module (`<family>.NAME`) expose each as a constant.
Source proofs read refusal details as string literals, so they read source after this resolution:
a literal -> constant change alone leaves every literal scan and pinned fingerprint unchanged.
The replacement never contains a newline, so line numbers are preserved.
"""

import ast
import re

from _repo import REPO_ROOT
import _repo

_IDS: set[str] | None = None


def registered_detail_ids() -> set[str]:
    global _IDS
    if _IDS is None:
        path = REPO_ROOT / "parity/reason_details.toml"
        rows = _repo.load_toml(path).get("detail", []) if path.is_file() else []
        _IDS = {row["id"] for row in rows}
    return _IDS


_PATH = re.compile(
    r"(?<![\w])(?:\w+::)*_?reason_details(?:::|\.)([a-z][a-z0-9_]*)(?:::|\.)([A-Z][A-Z0-9_]*)\b"
)
_SHORT = re.compile(r"(?<![\w:.])([a-z][a-z0-9_]*)(?:::|\.)([A-Z][A-Z0-9_]*)\b")


def imported_families(code: str) -> set[str]:
    """Families a file imports from the generated modules (Rust `use`, Python `from`)."""
    out: set[str] = set()
    for m in re.finditer(r"\buse\s+[\w:]*reason_details::(\{[^}]*\}|\w+)", code):
        out.update(re.findall(r"[a-z][a-z0-9_]*", m.group(1)))
    for m in re.finditer(r"\bfrom\s+[\w.]*reason_details\s+import\s+([^\n]+)", code):
        out.update(re.findall(r"[a-z][a-z0-9_]*", m.group(1)))
    return out


def resolve_rust(code: str) -> str:
    """Rust text with registry constants replaced by the id literal they spell."""
    if "reason_details" not in code:
        return code
    code = _PATH.sub(lambda m: f'"{m[1]}.{m[2].lower()}"', code)
    families, ids = imported_families(code), registered_detail_ids()
    return _SHORT.sub(
        lambda m: f'"{m[1]}.{m[2].lower()}"'
        if m[1] in families and f"{m[1]}.{m[2].lower()}" in ids
        else m[0],
        code,
    )


class _Constants(ast.NodeTransformer):
    def __init__(self, families: set[str]):
        self.families = families

    def visit_Attribute(self, node: ast.Attribute):
        self.generic_visit(node)
        value = node.value
        ids = registered_detail_ids()
        if isinstance(value, ast.Attribute) and isinstance(value.value, (ast.Name, ast.Attribute)):
            holder = value.value.id if isinstance(value.value, ast.Name) else value.value.attr
            via_module = holder in ("reason_details", "_reason_details")
            family = value.attr
        elif isinstance(value, ast.Name):
            via_module, family = False, value.id
        else:
            return node
        detail = f"{family}.{node.attr.lower()}"
        if node.attr.isupper() and detail in ids and (via_module or family in self.families):
            return ast.copy_location(ast.Constant(detail), node)
        return node


def parse_python(text: str) -> ast.Module:
    """`ast.parse` with registry detail constants resolved to their id literals."""
    tree = ast.parse(text)
    if "reason_details" not in text:
        return tree
    return ast.fix_missing_locations(_Constants(imported_families(text)).visit(tree))
