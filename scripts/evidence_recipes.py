"""What a build recipe says, apart from how it is written.

A record names the Nix recipes, the flake lock and the policy code its build was
made under. Bound by raw bytes, a comment or a reformatting of any of them owes
every record. The checker reads the repository only (it never evaluates Nix), so a
recipe is bound by the content that decides what it builds:

* `nix`: the expression's tokens. Comments and whitespace between tokens are not
  part of it; the contents of strings (builder scripts included) are, exactly.
* `flake-lock`: the locked entries of the inputs the recipes read
  (`lock.nodes.<name>.locked`), as canonical JSON. Other nodes, key order and
  indentation are not part of it.
* `python`: the module's syntax tree without docstrings: every constant, name and
  statement the policy evaluates, not its comments, layout or documentation.
* `json`: the value, as canonical JSON.

What this does not prove: two recipes with different tokens may still build the
same derivation; telling that needs a Nix evaluation, which a currency check does
not run. Such a change owes the records the recipe builds.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re

FORMS = ("nix", "flake-lock", "python", "json")
_LOCK_NODE = re.compile(r"\block\s*\.\s*nodes\s*\.\s*([A-Za-z_][\w-]*)\s*\.\s*locked\b")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def form_of(path: str) -> str | None:
    """The form a recipe file is bound in; `None` keeps its bytes."""
    if path.endswith(".nix"):
        return "nix"
    if path.rsplit("/", 1)[-1] == "flake.lock":
        return "flake-lock"
    if path.endswith(".py"):
        return "python"
    if path.endswith(".json"):
        return "json"
    return None


def nix_tokens(text: str) -> list[str]:
    """The tokens of a Nix expression: comments dropped, strings kept whole.

    A string runs to its closing delimiter with `${ … }` interpolations kept as written
    (their braces are matched, and strings inside them skipped), so nothing inside a
    string is ever taken for a comment. Raises `ValueError` on an unterminated string or
    comment."""
    out: list[str] = []
    p, n = 0, len(text)

    def string_end(start: int) -> int:
        indented = text.startswith("''", start)
        q = start + (2 if indented else 1)
        while q < n:
            if indented:
                if text.startswith("'''", q) or text.startswith("''$", q):
                    q += 3
                elif text.startswith("''\\", q):
                    q += 4
                elif text.startswith("''", q):
                    return q + 2
                elif text.startswith("${", q):
                    q = interpolation_end(q + 2)
                else:
                    q += 1
            else:
                if text[q] == "\\":
                    q += 2
                elif text[q] == '"':
                    return q + 1
                elif text.startswith("${", q):
                    q = interpolation_end(q + 2)
                else:
                    q += 1
        raise ValueError("unterminated Nix string")

    def interpolation_end(q: int) -> int:
        depth = 1
        while q < n:
            c = text[q]
            if c == '"' or text.startswith("''", q):
                q = string_end(q)
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return q + 1
            q += 1
        raise ValueError("unterminated Nix interpolation")

    while p < n:
        c = text[p]
        if c.isspace():
            p += 1
        elif c == "#":
            end = text.find("\n", p)
            p = n if end == -1 else end
        elif text.startswith("/*", p):
            end = text.find("*/", p + 2)
            if end == -1:
                raise ValueError("unterminated Nix comment")
            p = end + 2
        elif c == '"' or text.startswith("''", p):
            end = string_end(p)
            out.append(text[p:end])
            p = end
        else:
            m = re.compile(r"[A-Za-z0-9_.'+~/<>:@?=!&|*-]+|.", re.S).match(text, p)
            # A path or URL may hold `/*` or `#` only inside a string; split a word there.
            word = m.group(0)
            cut = min((i for i in (word.find("#"), word.find("/*")) if i > 0), default=len(word))
            out.append(word[:cut])
            p += cut
    return out


def lock_nodes(recipes: list[str]) -> list[str]:
    """The flake inputs a set of recipe texts read from the lock, sorted."""
    return sorted({name for text in recipes for name in _LOCK_NODE.findall(text)})


def python_tree(text: str) -> str:
    """A module's syntax tree without its docstrings."""
    tree = ast.parse(text)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body \
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def digest(form: str, data: bytes, *, nodes: list[str] | None = None) -> str:
    """The digest of a recipe's content in `form`; `ValueError` when it is not of that form."""
    try:
        text = data.decode("utf-8")
        if form == "nix":
            return _sha(b"nix-tokens\0" + _canonical(nix_tokens(text)))
        if form == "flake-lock":
            lock = json.loads(text)
            if not isinstance(nodes, list) or not nodes:
                raise ValueError("no flake input named")
            return _sha(b"flake-lock\0" + _canonical({name: lock["nodes"][name]["locked"] for name in nodes}))
        if form == "python":
            return _sha(b"python-tree\0" + python_tree(text).encode("utf-8"))
        if form == "json":
            return _sha(b"json\0" + _canonical(json.loads(text)))
    except (UnicodeDecodeError, SyntaxError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"not a {form} recipe: {error}") from error
    raise ValueError(f"unknown recipe form {form!r}")


def entry(path: str, data: bytes, role: str, *, recipes: list[str] | None = None) -> dict | None:
    """The manifest input of one recipe file, or `None` when it has no content form.

    `recipes` are the texts of the recipe set, which say which lock entries are read."""
    form = form_of(path)
    if form is None:
        return None
    bound = {"kind": "build_recipe", "path": path, "role": role, "form": form, "raw": _sha(data)}
    try:
        if form == "flake-lock":
            bound["nodes"] = lock_nodes(recipes or [])
            bound["digest"] = digest(form, data, nodes=bound["nodes"])
        else:
            bound["digest"] = digest(form, data)
    except ValueError:
        return None
    return bound
