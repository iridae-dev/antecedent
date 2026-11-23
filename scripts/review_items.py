"""Item-level bindings for reviews.

A review binds what it reviewed.  A source or test artifact names the items it covers and
binds a digest of the normalized text of just those items, so an edit elsewhere in the file
never stales it:

    {path = "crates/x/src/a.rs", items = ["fn solve", "impl Kernel", "const TOL"], sha256 = "..."}

Item names:

* Rust: ``fn name``, ``impl Type`` (every ``impl`` block whose self type is ``Type``),
  ``mod name``, ``const NAME``, ``static NAME``, ``struct``/``enum``/``trait``/``type`` ``Name``,
  ``macro name`` and ``module`` (all non-test code of the file).
* Python: ``fn name`` (any function or method of that name), ``class Name``, ``const NAME``.

The digest hashes the item text with comments and inter-token whitespace removed
(Rust, through ``capability_source._compact_rust_tokens``) or ``ast.unparse`` (Python), so
formatting and comment edits never stale a review either.  A name that matches several items
binds all of them.  Data artifacts (fixtures, oracle generators and references, environment
locks) are the reviewed content in full and bind the whole file.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

# Roles whose whole file is the reviewed data or reference.
WHOLE_FILE_ROLES = frozenset(
    {
        "reference",
        "environment",
        "fixture",
        "generator",
        "environment_lock",
        "comparison_helper",
        "input_fixture",
    }
)
SOURCE_SUFFIXES = (".rs", ".py")

_RUST_KINDS = {"fn", "impl", "mod", "const", "static", "struct", "enum", "trait", "type", "macro"}
_PY_KINDS = {"fn", "class", "const"}


def review_json_digest(payload: object) -> str:
    """Hash review payloads with the established compact UTF-8 JSON encoding."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_digest(path: Path, root: Path | None = None) -> str:
    """Digest of a data artifact.  A fixture with a reference review (embedded as
    ``oracle.reference_review`` or kept in ``parity/fixture_reviews``) is bound by its
    payload, so rebinding that review never stales the reviews that read it."""
    if path.suffix == ".json":
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            document = None
        oracle = document.get("oracle") if isinstance(document, dict) else None
        from evidence_kinds import fixture_payload_digest, review_path
        try:
            separate = root is not None and review_path(root, path).is_file()
        except ValueError:
            separate = False
        if isinstance(oracle, dict) and ("reference_review" in oracle or separate):
            return fixture_payload_digest(document)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_item(spec: object) -> tuple[str, str] | None:
    """``(kind, name)`` of an item spec, or None when malformed."""
    if not isinstance(spec, str):
        return None
    if spec == "module":
        return ("module", "")
    match = re.fullmatch(r"([a-z]+) ([A-Za-z_][A-Za-z0-9_]*)", spec)
    return (match.group(1), match.group(2)) if match else None


# ------------------------------------------------------------------------- Rust
def _boundary(skel: str, pos: int) -> int:
    """Start of the attributes and qualifiers that belong to the item keyword at ``pos``."""
    k = pos
    while k > 0 and skel[k - 1] not in ";{}":
        k -= 1
    return k


def _declaration_end(skel: str, pos: int) -> int:
    """End of a declaration that closes at its first ``{...}`` block or its ``;``."""
    from test_evidence import _match_bracket

    depth = 0
    for j in range(pos, len(skel)):
        ch = skel[j]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif depth <= 0 and ch == "{":
            return _match_bracket(skel, j) + 1
        elif depth <= 0 and ch == ";":
            return j + 1
    return len(skel)


def _impl_spans(skel: str, name: str) -> list[tuple[int, int]]:
    spans = []
    for m in re.finditer(r"\bimpl\b", skel):
        head_end = skel.find("{", m.end())
        semi = skel.find(";", m.end())
        if head_end < 0 or (0 <= semi < head_end):
            continue
        head = skel[m.end() : head_end]
        head = re.sub(r"^\s*<.*?>\s*(?=[A-Za-z_:&])", "", head, count=1, flags=re.S)
        self_type = head.split(" for ", 1)[-1]
        if re.match(rf"\s*(?:[A-Za-z_][A-Za-z0-9_]*::)*{re.escape(name)}\b", self_type):
            spans.append((_boundary(skel, m.start()), _declaration_end(skel, m.end())))
    return spans


def _rust_spans(path: Path, kind: str, name: str) -> list[tuple[int, int]]:
    from test_evidence import rust_items

    masked, items = rust_items(path.resolve())
    skel = masked.skeleton
    if kind == "impl":
        return _impl_spans(skel, name)
    if kind in ("fn", "const", "static", "mod", "macro"):
        want = "const" if kind in ("const", "static") else kind
        return [
            (_boundary(skel, item.start), item.body[1])
            for item in items
            if item.kind == want and item.name == name
        ]
    return [
        (_boundary(skel, m.start()), _declaration_end(skel, m.end()))
        for m in re.finditer(rf"\b{kind}\s+{re.escape(name)}\b", skel)
    ]


def _rust_text(path: Path, kind: str, name: str) -> list[str] | None:
    from capability_source import _compact_rust_tokens, nontest_pair
    from test_evidence import rust_items

    if kind == "module":
        code, _ = nontest_pair(path, in_crate=False)
        return [_compact_rust_tokens(code)]
    spans = _rust_spans(path, kind, name)
    if not spans:
        return None
    code = rust_items(path.resolve())[0].code
    return [_compact_rust_tokens(code[a:b]) for a, b in sorted(set(spans))]


# ----------------------------------------------------------------------- Python
def _py_text(path: Path, kind: str, name: str) -> list[str] | None:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    found: list[str] = []
    for node in ast.walk(tree):
        if kind == "fn" and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or kind == "class" and isinstance(node, ast.ClassDef):
            hit = node.name == name
        else:
            hit = False
        if hit:
            found.append(ast.unparse(node))
    if kind == "const":
        for node in tree.body:
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, ast.AnnAssign)
                else []
            )
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                found.append(ast.unparse(node))
    return found or None


# ----------------------------------------------------------------------- digest
def items_digest(path: Path, items: list[str]) -> tuple[str | None, str | None]:
    """``(digest, error)`` of the named items of a source file."""
    kinds = _RUST_KINDS if path.suffix == ".rs" else _PY_KINDS
    parts: list[str] = []
    for spec in sorted(set(items)):
        parsed = parse_item(spec)
        if parsed is None or (parsed[0] != "module" and parsed[0] not in kinds):
            return None, f"unknown item {spec!r}"
        if parsed[0] == "module" and path.suffix != ".rs":
            return None, f"unknown item {spec!r}"
        try:
            texts = (_rust_text if path.suffix == ".rs" else _py_text)(path, *parsed)
        except (SyntaxError, ValueError) as error:
            return None, f"cannot read items: {error}"
        if not texts:
            return None, f"item not found: {spec}"
        parts.extend(f"{spec}\n{text}" for text in texts)
    return hashlib.sha256("\n\0\n".join(parts).encode("utf-8")).hexdigest(), None


def artifact_digest(root: Path, artifact: dict) -> tuple[str | None, str | None]:
    """``(digest, error)`` an artifact's recorded ``sha256`` must equal."""
    rel = artifact.get("path")
    path = (root / str(rel)).resolve()
    if not path.is_file():
        return None, "missing"
    items = artifact.get("items")
    if path.suffix in SOURCE_SUFFIXES:
        if items is None:
            if artifact.get("role") in WHOLE_FILE_ROLES:
                return file_digest(path, root), None
            return None, "binds a whole source file; name the reviewed items"
        if (
            not isinstance(items, list)
            or not items
            or any(parse_item(item) is None for item in items)
        ):
            return None, "items must be a non-empty list of item names"
        return items_digest(path, items)
    if items is not None:
        return None, "items apply to source files; a data file binds whole"
    return file_digest(path, root), None


def artifact_problem(root: Path, artifact: dict, label: str) -> str | None:
    """The finding for one artifact, or None when its digest holds."""
    rel = artifact.get("path")
    digest, error = artifact_digest(root, artifact)
    if error == "missing":
        return f"{label} missing artifact: {rel}"
    if error is not None:
        return f"{label} artifact {rel}: {error}"
    if digest != artifact.get("sha256"):
        return f"{label} stale artifact: {rel}"
    return None
