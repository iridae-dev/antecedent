#!/usr/bin/env python3
"""The one reader of test source for the evidence gates.

A cited test is evidence only when it executes and its own body, together with
the helpers it calls, does what the citation says. This module answers three
questions for the gates, the same way everywhere:

* which Rust test functions are compiled, non-ignored tests (`cargo test --
  --list`, minus `--list --ignored`, for the targets a citation names), and which
  Python tests pytest collects without a static skip;
* what a test function's *closure* is: its body with comments removed plus the
  bodies of every same-target function, constant and `macro_rules!` it reaches
  (the test file and the modules it declares with `mod`);
* whether a closure consumes a fixture (names `conformance/<category>/<name>` or
  the category directory plus the quoted name, parses it and asserts), and
  whether it exercises support-matrix axis values.

Used by gate_support_matrix.sh (cited evidence), gate_evidence_reachability.sh,
gate_parity_schema.sh and gate_metadata_consistency.sh (fixture consumers).

    python3 scripts/test_evidence.py consumers conformance/estimate/aipw
    python3 scripts/test_evidence.py closure crates/antecedent/tests/pag.rs lpcmci_chain
"""

from __future__ import annotations

import ast
import functools
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PARSE_MARKERS = re.compile(
    r"serde_json::from_str|serde_json::from_slice|from_str::<|json\.loads|"
    r"json\.load\(|tomllib\.loads|tomllib\.load\(|load_expected|load_json\(|toml::from_str"
)
ASSERT_MARKERS = re.compile(r"\bassert(?:_eq|_ne)?!|\bassert\s|pytest\.approx|\bpanic!")


# ------------------------------------------------------------------ Rust lexing


@dataclass(frozen=True)
class Masked:
    """`code` keeps string literals and blanks comments; `skeleton` also blanks
    literal contents, so brackets inside strings never count. Offsets agree."""

    code: str
    skeleton: str


_LEX = re.compile(
    r"//[^\n]*|/\*|(?<![\w])b?r(#*)\"|(?<![\w])b?\"(?:\\.|[^\"\\])*\"|'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]{1,6}\}|.)|[^\\'\n])'",
    re.S,
)


def mask_rust(text: str) -> Masked:
    code = list(text)
    skel = list(text)
    n = len(text)

    def blank(lo: int, hi: int, *, keep_code: bool) -> None:
        for k in range(lo, hi):
            if text[k] != "\n":
                skel[k] = " "
                if not keep_code:
                    code[k] = " "

    i = 0
    while True:
        m = _LEX.search(text, i)
        if not m:
            break
        tok = m.group(0)
        if tok.startswith("//"):
            blank(m.start(), m.end(), keep_code=False)
            i = m.end()
        elif tok == "/*":
            depth, j = 1, m.end()
            while j < n and depth:
                nxt_open = text.find("/*", j)
                nxt_close = text.find("*/", j)
                if nxt_close == -1:
                    j = n
                    break
                if nxt_open != -1 and nxt_open < nxt_close:
                    depth, j = depth + 1, nxt_open + 2
                else:
                    depth, j = depth - 1, nxt_close + 2
            blank(m.start(), j, keep_code=False)
            i = j
        elif m.group(1) is not None:
            hashes = m.group(1)
            end = text.find('"' + hashes, m.end())
            end = n if end == -1 else end + 1 + len(hashes)
            blank(m.end(), max(m.end(), end - 1 - len(hashes)), keep_code=True)
            i = end
        else:
            blank(m.start() + (2 if tok.startswith("b") else 1), m.end() - 1, keep_code=True)
            i = m.end()
    return Masked("".join(code), "".join(skel))


_BRACKETS = {"{": re.compile(r"[{}]"), "(": re.compile(r"[()]"), "[": re.compile(r"[\[\]]")}


def _match_bracket(skel: str, open_at: int) -> int:
    opener = skel[open_at]
    depth = 0
    for m in _BRACKETS[opener].finditer(skel, open_at):
        depth += 1 if m.group(0) == opener else -1
        if depth == 0:
            return m.start()
    return len(skel) - 1


@dataclass
class RustItem:
    kind: str  # fn | const | macro | mod
    name: str
    start: int
    body: tuple[int, int]
    attrs: list[str] = field(default_factory=list)


_FN = re.compile(r"\bfn\s+([A-Za-z_]\w*)\s*(?:<)?")
_CONST = re.compile(r"\b(?:const|static)\s+([A-Z_][A-Z0-9_]*)\s*:")
_MACRO = re.compile(r"\bmacro_rules!\s*([A-Za-z_]\w*)\s*[{(\[]")
_MOD_BLOCK = re.compile(r"\bmod\s+([A-Za-z_]\w*)\s*\{")
_MOD_DECL = re.compile(r"\bmod\s+([A-Za-z_]\w*)\s*;")


_PREFIX_WORD = re.compile(r"(?:pub|async|const|unsafe|extern|crate|super|self|in|\"C\")\Z")


def _attrs_before(skel: str, code: str, pos: int) -> list[str]:
    """Outer attributes (`#[...]`) directly before the item keyword at `pos`,
    skipping visibility and qualifier words."""
    attrs = []
    k = pos
    while True:
        while k > 0 and skel[k - 1].isspace():
            k -= 1
        if k == 0:
            break
        if skel[k - 1] == ")":
            # `pub(crate)` / `pub(in path)`
            j = k - 1
            depth = 0
            while j >= 0:
                if skel[j] == ")":
                    depth += 1
                elif skel[j] == "(":
                    depth -= 1
                    if depth == 0:
                        break
                j -= 1
            w = j
            while w > 0 and (skel[w - 1].isalnum() or skel[w - 1] == "_"):
                w -= 1
            if skel[w:j] == "pub":
                k = w
                continue
            break
        if skel[k - 1].isalnum() or skel[k - 1] in '_"':
            w = k
            while w > 0 and (skel[w - 1].isalnum() or skel[w - 1] in '_"'):
                w -= 1
            if _PREFIX_WORD.match(code[w:k]):
                k = w
                continue
            break
        if skel[k - 1] == "]":
            j, depth = k - 1, 0
            while j >= 0:
                if skel[j] == "]":
                    depth += 1
                elif skel[j] == "[":
                    depth -= 1
                    if depth == 0:
                        break
                j -= 1
            if j > 0 and skel[j - 1] == "#":
                attrs.append(code[j - 1 : k])
                k = j - 1
                continue
        break
    return list(reversed(attrs))


@functools.cache
def rust_items(path: Path) -> tuple[Masked, tuple[RustItem, ...]]:
    text = path.read_text(errors="ignore")
    masked = mask_rust(text)
    skel, code = masked.skeleton, masked.code
    items: list[RustItem] = []
    for m in _FN.finditer(skel):
        j, depth = m.end(), 0
        while j < len(skel):
            ch = skel[j]
            if ch in "(<[":
                depth += 1
            elif ch in ")>]":
                depth -= 1
            elif ch == "{" and depth <= 0:
                break
            elif ch == ";" and depth <= 0:
                j = -1
                break
            j += 1
        if j == -1 or j >= len(skel):
            continue
        end = _match_bracket(skel, j)
        items.append(
            RustItem(
                "fn", m.group(1), m.start(), (j, end + 1), _attrs_before(skel, code, m.start())
            )
        )
    for m in _CONST.finditer(skel):
        end, depth = m.end(), 0
        while end < len(skel):
            ch = skel[end]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == ";" and depth == 0:
                break
            end += 1
        items.append(RustItem("const", m.group(1), m.start(), (m.start(), end + 1)))
    for m in _MACRO.finditer(skel):
        end = _match_bracket(skel, m.end() - 1)
        items.append(RustItem("macro", m.group(1), m.start(), (m.start(), end + 1)))
    for m in _MOD_BLOCK.finditer(skel):
        end = _match_bracket(skel, m.end() - 1)
        items.append(
            RustItem(
                "mod",
                m.group(1),
                m.start(),
                (m.end() - 1, end + 1),
                _attrs_before(skel, code, m.start()),
            )
        )
    return masked, tuple(items)


def _disabling_cfg(attrs: list[str]) -> str | None:
    for attr in attrs:
        inner = re.sub(r"\s+", "", attr)
        if inner.startswith("#[cfg(") and inner != "#[cfg(test)]":
            return attr
        if inner.startswith("#[cfg_attr(") and "ignore" in inner:
            return attr
    return None


def _file_cfg_disabled(masked: Masked) -> bool:
    for m in re.finditer(r"#!\[cfg\(([^\]]*)\)\]", masked.code):
        if re.sub(r"\s+", "", m.group(1)) != "test":
            return True
    return False


# --------------------------------------------------------------- compilation units


def _is_cfg_test(attrs: list[str]) -> bool:
    return any(re.sub(r"\s+", "", a) == "#[cfg(test)]" for a in attrs)


def _declared_modules(path: Path) -> list[tuple[Path, bool]]:
    """(module file, declared test-only) for every enabled `mod name;` in `path`."""
    masked, items = rust_items(path)
    blocks = [it for it in items if it.kind == "mod"]
    out = []
    for m in _MOD_DECL.finditer(masked.skeleton):
        attrs = _attrs_before(masked.skeleton, masked.code, m.start())
        if _disabling_cfg(attrs):
            continue
        around = [b for b in blocks if b.body[0] < m.start() < b.body[1]]
        if any(_disabling_cfg(b.attrs) for b in around):
            continue
        test_only = _is_cfg_test(attrs) or any(_is_cfg_test(b.attrs) for b in around)
        if path.name in ("mod.rs", "lib.rs", "main.rs") or path.parent.name == "tests":
            base = path.parent
        else:
            base = path.with_suffix("")
        for cand in (base / f"{m.group(1)}.rs", base / m.group(1) / "mod.rs"):
            if cand.is_file():
                out.append((cand.resolve(), test_only))
                break
    return out


@functools.cache
def target_modules(root_file: Path) -> dict[Path, bool]:
    """Every file compiled into the target rooted at `root_file`, mapped to whether it
    is compiled only under `cfg(test)`."""
    seen: dict[Path, bool] = {}
    todo = [(root_file.resolve(), False)]
    while todo:
        p, test_only = todo.pop()
        if p in seen or not p.is_file():
            continue
        seen[p] = test_only
        todo.extend((child, test_only or flag) for child, flag in _declared_modules(p))
    return seen


def target_files(root_file: Path) -> tuple[Path, ...]:
    return tuple(target_modules(root_file))


@functools.cache
def target_root(path: Path) -> tuple[Path, str, list[str]] | None:
    """(crate root file, crate name, cargo target args) for a Rust source file."""
    path = path.resolve()
    rel = path.relative_to(ROOT.resolve()).parts
    if len(rel) < 4 or rel[0] != "crates":
        return None
    crate_dir = ROOT / "crates" / rel[1]
    crate = rel[1]
    manifest = (crate_dir / "Cargo.toml").read_text(errors="ignore")
    m = re.search(r'^name\s*=\s*"([^"]+)"', manifest, re.M)
    crate = m.group(1) if m else crate
    if rel[2] == "tests":
        tests_dir = crate_dir / "tests"
        if len(rel) == 4:
            return tests_dir / rel[3], crate, ["--test", Path(rel[3]).stem]
        main = tests_dir / rel[3] / "main.rs"
        if main.is_file():
            return main, crate, ["--test", rel[3]]
        # A file under tests/<dir>/ is compiled only as a module of some target.
        for top in sorted(tests_dir.glob("*.rs")):
            if path in target_files(top.resolve()):
                return top, crate, ["--test", top.stem]
        return None
    if rel[2] == "src":
        lib = crate_dir / "src" / "lib.rs"
        if lib.is_file():
            return lib, crate, ["--lib"]
    return None


@functools.cache
def compiled(path: Path) -> bool:
    root = target_root(path)
    return bool(root) and path.resolve() in target_files(root[0].resolve())


# ------------------------------------------------------------------ test functions


@dataclass
class StaticTest:
    name: str
    problems: list[str]


@functools.cache
def static_rust_test(path: Path, name: str) -> StaticTest:
    """Why `name` in `path` is not a compiled, non-ignored `#[test]` (empty = it is)."""
    problems: list[str] = []
    if not path.is_file():
        return StaticTest(name, [f"{path} does not exist"])
    masked, items = rust_items(path)
    fns = [it for it in items if it.kind == "fn" and it.name == name]
    if not fns:
        return StaticTest(name, [f"no fn {name} outside comments and string literals"])
    if len(fns) > 1:
        problems.append(f"fn {name} is defined {len(fns)} times")
    item = fns[0]
    attrs = [re.sub(r"\s+", "", a) for a in item.attrs]
    if not any(a in ("#[test]", "#[tokio::test]") or a.startswith("#[tokio::test(") for a in attrs):
        problems.append(f"fn {name} has no #[test] attribute")
    if any(a.startswith("#[ignore") for a in attrs):
        problems.append(f"fn {name} is #[ignore]d")
    if (cfg := _disabling_cfg(item.attrs)) is not None:
        problems.append(f"fn {name} is compiled only under {cfg.strip()}")
    for block in (it for it in items if it.kind == "mod"):
        if block.body[0] < item.start < block.body[1] and (cfg := _disabling_cfg(block.attrs)):
            problems.append(f"fn {name} sits in mod {block.name} under {cfg.strip()}")
    if _file_cfg_disabled(masked):
        problems.append(f"{path} is compiled out by an inner #![cfg(...)]")
    root = target_root(path)
    file_test_only = bool(root) and target_modules(root[0].resolve()).get(path.resolve(), False)
    if "/src/" in path.as_posix() and not file_test_only:
        in_test_mod = any(
            b.body[0] < item.start < b.body[1]
            and any(re.sub(r"\s+", "", a) == "#[cfg(test)]" for a in b.attrs)
            for b in items
            if b.kind == "mod"
        ) or re.search(r"#!\[cfg\(test\)\]", masked.code)
        if not in_test_mod:
            problems.append(f"fn {name} in library source is not inside a #[cfg(test)] module")
    if not compiled(path):
        problems.append(f"{path} is not compiled into any cargo target (no `mod` declares it)")
    return StaticTest(name, problems)


_CARGO_LIST_CACHE: dict[tuple[str, tuple[str, ...]], tuple[set[str], set[str], str | None]] = {}


def cargo_listed(
    crate: str, target: list[str], cwd: Path = ROOT
) -> tuple[set[str], set[str], str | None]:
    """(all listed test names, ignored test names, error) for one cargo target."""
    key = (crate, tuple(target))
    if key in _CARGO_LIST_CACHE:
        return _CARGO_LIST_CACHE[key]

    def listing(extra: list[str]) -> tuple[set[str], str | None]:
        proc = subprocess.run(
            ["cargo", "test", "-q", "-p", crate, *target, "--", "--list", *extra],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return set(), (proc.stdout + proc.stderr)[-2000:]
        return {
            ln[: -len(": test")] for ln in proc.stdout.splitlines() if ln.endswith(": test")
        }, None

    everything, err = listing([])
    ignored, err2 = listing(["--ignored"]) if err is None else (set(), None)
    out = (everything, ignored, err or err2)
    _CARGO_LIST_CACHE[key] = out
    return out


def module_path(path: Path, root_file: Path) -> str:
    """Module prefix (`a::b::`) of a source file inside its target."""
    rel = path.resolve().relative_to(root_file.resolve().parent).with_suffix("").parts
    parts = [p for p in rel if p not in ("mod", "lib", "main")]
    if root_file.parent.name == "tests" and parts and parts[0] == root_file.stem:
        parts = parts[1:]
    return "".join(f"{p}::" for p in parts)


def resolve_rust_test(path: Path, name: str, cwd: Path = ROOT) -> tuple[str | None, list[str]]:
    """Full libtest name of a cited Rust test, or the reasons it is not executing evidence."""
    problems = static_rust_test(path, name).problems
    root = target_root(path)
    if root is None:
        return None, problems + [f"{path} is not under a cargo test or library target"]
    root_file, crate, target = root
    listed, ignored, err = cargo_listed(crate, target, cwd)
    if err:
        return None, problems + [
            f"`cargo test -p {crate} {' '.join(target)} -- --list` failed:\n{err}"
        ]
    _, items = rust_items(path)
    fns = [it for it in items if it.kind == "fn" and it.name == name]
    enclosing = (
        [b for b in items if b.kind == "mod" and b.body[0] < fns[0].start < b.body[1]]
        if fns
        else []
    )
    full = (
        module_path(path, root_file)
        + "".join(f"{b.name}::" for b in sorted(enclosing, key=lambda b: b.start))
        + name
    )
    if full not in listed:
        return None, problems + [
            f"cargo does not list the test {full} in {crate} {' '.join(target)}"
        ]
    if full in ignored:
        return None, problems + [f"cargo lists {full} as ignored"]
    return full, problems


_SKIP_ATTRS = {"skip", "skipif", "xfail"}


def _mark_names(node: ast.AST) -> set[str]:
    """Names of the marks an expression applies: `pytest.mark.skipif(c)` -> {"skipif"}.

    Only the mark expression itself is read (through a call's callee, and through a
    list or tuple of marks), never a call's arguments."""
    if isinstance(node, (ast.List, ast.Tuple)):
        return set().union(*(_mark_names(e) for e in node.elts)) if node.elts else set()
    if isinstance(node, ast.Call):
        return _mark_names(node.func)
    if isinstance(node, ast.Attribute):
        return {node.attr} | _mark_names(node.value)
    return set()


def _is_pytest_call(node: ast.AST, attr: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pytest"
    )


@functools.cache
def static_python_test(path: Path, name: str) -> list[str]:
    """Why `name` in `path` is not a collected, runnable pytest function (empty = it is).

    Read from the syntax tree, so a decorator spread over several lines, a
    multi-line `pytestmark = [...]`, an unconditional `pytest.skip()` in the
    body and a parametrisation over an empty list are all seen."""
    problems = []
    if not path.is_file():
        return [f"{path} does not exist"]
    if not path.name.startswith("test_"):
        problems.append(f"{path} is not a pytest module (test_*.py)")
    if not name.startswith("test"):
        problems.append(f"{name} is not a collected pytest name (test*)")
    try:
        tree = ast.parse(path.read_text(errors="ignore"))
    except SyntaxError as error:
        return problems + [f"{path} does not parse: {error.msg}"]
    funcs = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    ]
    if not funcs:
        nested = [
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
        ]
        if nested:
            return problems + [f"def {name} is not a module-level test function"]
        return problems + [f"no def {name} outside comments and strings"]
    fn = funcs[-1]
    for dec in fn.decorator_list:
        if _mark_names(dec) & _SKIP_ATTRS:
            problems.append(f"def {name} carries a skip/xfail marker")
        if (
            isinstance(dec, ast.Call)
            and isinstance(dec.func, ast.Attribute)
            and dec.func.attr == "parametrize"
            and len(dec.args) >= 2
            and isinstance(dec.args[1], (ast.List, ast.Tuple))
            and not dec.args[1].elts
        ):
            problems.append(f"def {name} is parametrised over an empty list, so it never runs")
    for stmt in fn.body:
        if isinstance(stmt, ast.Expr) and _is_pytest_call(stmt.value, "skip"):
            problems.append(f"def {name} calls pytest.skip() unconditionally")
            break
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(tg, ast.Name) and tg.id == "pytestmark" for tg in node.targets
        ):
            if _mark_names(node.value) & _SKIP_ATTRS:
                problems.append(f"{path} skips the whole module")
        elif isinstance(node, ast.Expr) and _is_pytest_call(node.value, "skip"):
            problems.append(f"{path} skips the whole module")
    return problems


def resolve_python_test(path: Path, name: str, cwd: Path = ROOT) -> list[str]:
    problems = static_python_test(path, name)
    if problems:
        return problems
    rel = path.resolve().relative_to((cwd / "python").resolve())
    proc = subprocess.run(
        [
            "uv",
            "run",
            "--quiet",
            "--project",
            ".",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            f"{rel}::{name}",
        ],
        cwd=cwd / "python",
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or f"::{name}" not in proc.stdout:
        return [f"pytest does not collect {rel}::{name}: {(proc.stdout + proc.stderr)[-500:]}"]
    return []


# --------------------------------------------------------------------- closures


def _strip_python(text: str) -> str:
    text = re.sub(r'(?s)("""|\'\'\').*?\1', lambda m: " " * len(m.group(0)), text)
    return re.sub(r"(?m)#[^\n]*", "", text)


@functools.cache
def _rust_index(root_file: Path) -> dict[str, list[tuple[Path, RustItem]]]:
    index: dict[str, list[tuple[Path, RustItem]]] = {}
    for f in target_files(root_file):
        _, items = rust_items(f)
        for it in items:
            if it.kind != "mod":
                index.setdefault(it.name, []).append((f, it))
    return index


# A reference to another item: a bare call or macro (`name(`, `name!`, `name::<`;
# never a method `.name(` or path `Type::name(`), a
# constant (`ALL_CAPS`), or a function passed as a value (`f(x, helper)`).
_REF = re.compile(
    r"(?<![.:\w])([A-Za-z_]\w*)\s*(?:\(|!|::<)|\b([A-Z][A-Z0-9_]{2,})\b|(?<=[(,])\s*([a-z_]\w*)\s*(?=[,)])"
)


@functools.cache
def _item_refs(path: Path, start: int) -> tuple[str, str, frozenset[str]]:
    """(code with comments removed, skeleton, referenced identifiers) of one item."""
    masked, items = rust_items(path)
    item = next(it for it in items if it.start == start)
    lo, hi = item.body
    refs = {name for groups in _REF.findall(masked.skeleton[lo:hi]) for name in groups if name}
    return masked.code[lo:hi], masked.skeleton[lo:hi], frozenset(refs)


def rust_closure(path: Path, name: str) -> str:
    """The test function's code plus every same-target item it reaches."""
    return _rust_closure_parts(path, name)[0]


@functools.cache
def _rust_closure_parts(path: Path, name: str) -> tuple[str, str]:
    """(code, skeleton) of the closure; offsets agree, so a match in the code can be
    checked against the skeleton to tell code from string-literal contents."""
    path = path.resolve()
    root = target_root(path)
    root_file = root[0].resolve() if root else path
    index = _rust_index(root_file)
    if path not in target_files(root_file):
        index = dict(index)
        for it in rust_items(path)[1]:
            if it.kind != "mod":
                index.setdefault(it.name, []).append((path, it))
    out, skel_out, seen = [], [], set()
    todo = [(ff, it) for ff, it in index.get(name, []) if ff == path]
    while todo:
        ff, it = todo.pop()
        key = (ff, it.start)
        if key in seen:
            continue
        seen.add(key)
        code, skel, refs = _item_refs(ff, it.start)
        out.append(code)
        skel_out.append(skel)
        for ref in refs:
            if ref != it.name:
                todo.extend(index.get(ref, ()))
    return "\n".join(out), "\n".join(skel_out)


@functools.cache
def python_closure(path: Path, name: str) -> str:
    text = _strip_python(path.read_text(errors="ignore"))
    defs: dict[str, str] = {}
    for m in re.finditer(r"^(?:async\s+)?def\s+(\w+)\s*\(.*?(?=^\S|\Z)", text, re.M | re.S):
        defs[m.group(1)] = m.group(0)
    for m in re.finditer(r"^([A-Z_][A-Z0-9_]*)\s*=.*?(?=^\S|\Z)", text, re.M | re.S):
        defs[m.group(1)] = m.group(0)
    out, seen, todo = [], set(), [name]
    while todo:
        nm = todo.pop()
        if nm in seen or nm not in defs:
            continue
        seen.add(nm)
        out.append(defs[nm])
        todo.extend(re.findall(r"\b([A-Za-z_]\w*)\b", defs[nm]))
    return "\n".join(out)


def closure(path: Path, name: str) -> str:
    path = path.resolve()
    return python_closure(path, name) if path.suffix == ".py" else rust_closure(path, name)


# ------------------------------------------------------------ fixture consumption


def _names_category(text: str, category: str) -> bool:
    """The closure builds `conformance/<category>`: as one path, or by joining a
    `conformance` path with the quoted category (possibly one of several searched)."""
    if re.search(rf"conformance/{re.escape(category)}(?![\w-])", text):
        return True
    return bool(
        re.search(r"conformance[\"'/)]", text)
        and re.search(rf"[\"']{re.escape(category)}[\"']", text)
    )


def names_fixture(path: Path, test_name: str, body: str, fixture: str) -> bool:
    """`conformance/<category>/<name>` as a path in the closure, or the quoted
    `<name>` passed straight to a helper whose own closure names the category
    directory (`load_expected("aipw")` with `join("conformance/estimate")`). A quoted
    name used any other way (an estimator id that happens to match) does not count."""
    parts = Path(fixture).parts
    if len(parts) < 3 or parts[0] != "conformance":
        return False
    category, name = parts[1], parts[2]
    if name not in body:
        return False
    full = f"conformance/{category}/{name}"
    if full in body and re.search(rf"{re.escape(full)}(?![\w-])", body):
        return True
    call = re.compile(r"([A-Za-z_]\w*)\s*\(\s*&?\Z")
    # The helper call must be code, not text inside another string literal.
    code = _rust_closure_parts(path.resolve(), test_name)[1] if path.suffix == ".rs" else body
    for m in re.finditer(rf"[\"']{re.escape(name)}[\"']", body):
        helper = call.search(code, max(0, m.start() - 80), m.start())
        if helper and _names_category(closure(path, helper.group(1)), category):
            return True
    return False


@functools.cache
def _markers(path: Path, name: str) -> tuple[bool, bool]:
    body = closure(path, name)
    return bool(PARSE_MARKERS.search(body)), bool(ASSERT_MARKERS.search(body))


def consumption_problems(path: Path, name: str, fixture: str) -> list[str]:
    body = closure(path, name)
    parses, asserts = _markers(path, name)
    problems = []
    if not names_fixture(path, name, body, fixture):
        problems.append(f"{name} (with the helpers it calls) never names {fixture}")
    if not parses:
        problems.append(f"{name} (with the helpers it calls) parses no fixture")
    if not asserts:
        problems.append(f"{name} (with the helpers it calls) asserts nothing")
    return problems


@functools.cache
def _read(path: Path) -> str:
    return path.read_text(errors="ignore")


@functools.cache
def _candidate_files() -> tuple[Path, ...]:
    # Globbed from the resolved root without resolving each file, so a symlinked
    # `python/` (a gate self-test overlay) stays inside the tree being checked.
    base = ROOT.resolve()
    files = [p for p in base.glob("crates/*/tests/**/*.rs")]
    files += [p for p in base.glob("crates/*/src/**/*.rs")]
    files += [p for p in base.glob("python/tests/**/test_*.py")]
    return tuple(p for p in files if "target" not in p.parts)


def fixture_consumers(fixture: str) -> list[str]:
    """`file::test` for every executing test whose closure consumes `fixture`.

    Static: the test must be a compiled, non-ignored, non-cfg-disabled `#[test]`
    (or a collected pytest without a static skip); comments never count."""
    parts = Path(fixture).parts
    if len(parts) < 3:
        return []
    name = parts[2]
    out = []
    for f in _candidate_files():
        raw = _read(f)
        if name not in raw:
            continue
        if f.suffix == ".py":
            text = _strip_python(raw)
            tests = re.findall(r"^(?:async\s+)?def\s+(test\w*)\s*\(", text, re.M)
            for t in tests:
                if static_python_test(f, t) or name not in closure(f, t):
                    continue
                if not consumption_problems(f, t, fixture):
                    out.append(f"{f.relative_to(ROOT.resolve())}::{t}")
            continue
        masked, items = rust_items(f)
        if name not in masked.code:
            continue
        for it in items:
            if it.kind != "fn" or not any(re.sub(r"\s+", "", a) == "#[test]" for a in it.attrs):
                continue
            if static_rust_test(f, it.name).problems or name not in closure(f, it.name):
                continue
            if not consumption_problems(f, it.name, fixture):
                out.append(f"{f.relative_to(ROOT.resolve())}::{it.name}")
    return out


# ------------------------------------------------------------------- matrix axes


def _snake(camel: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel).lower()


# Tokens whose presence in a test closure shows it builds the axis value. Queries
# are named by their type, facade constructor or functional variant.
QUERY_TOKENS: dict[str, list[str]] = {
    "AverageEffect": [r"AverageEffectQuery", r"\bate\(", r"binary_ate"],
    "ConditionalEffect": [r"ConditionalEffectQuery", r"CausalQuery::ConditionalEffect"],
    "ResponseCurve": [r"MeanCurve", r"ResponseCurve", r"response_curve", r"curve_query"],
    "InterventionResponse": [r"InterventionResponse", r"intervention_response", r"joint_query"],
    "InterventionalDistribution": [
        r"InterventionalDistribution",
        r"interventional_distribution",
        r"DistributionQuery",
    ],
    "MediationEffect": [r"MediationQuery", r"MediationEffect", r"CausalQuery::Mediation"],
    "PathSpecificEffect": [r"PathSpecific", r"path_specific"],
    "Counterfactual": [r"Counterfactual"],
    "AverageDerivative": [r"AverageDerivative"],
    "PointDerivative": [r"PointDerivative"],
    "DirectionalDerivative": [r"DirectionalDerivative"],
    "Elasticity": [r"\bElasticity\b", r"elasticity", r"DerivativeScale::LogLog"],
    "SemiElasticity": [
        r"SemiElasticity",
        r"semi_elasticity",
        r"DerivativeScale::Log(?:Treatment|Outcome)",
    ],
    "ResponseJacobian": [r"ResponseJacobian", r"Jacobian"],
    "PulseEffect": [
        r"TemporalEffectQuery::pulse",
        r"pulse_query",
        r"PulseEffect",
        r"TemporalPolicy::pulse",
    ],
    "SustainedEffect": [
        r"TemporalEffectQuery::sustained",
        r"SustainedEffect",
        r"TemporalPolicy::sustained",
    ],
    # A MediationQuery on a temporal graph (the graph-class check binds the class).
    "TemporalMediationEffect": [r"with_horizons", r"TemporalMediation", r"MediationQuery"],
    "AnomalyAttribution": [r"AnomalyAttribution", r"anomaly"],
    "ChangeAttribution": [r"ChangeAttribution", r"distribution_change", r"change_attribution"],
    "TransportQuery": [r"Transport"],
    "InterferenceQuery": [r"Interference"],
}
GRAPH_TOKENS: dict[str, list[str]] = {
    # A static graph posterior's atoms are DAGs.
    "Dag": [
        r"(?<![A-Za-z])Dag(?![A-Za-z])",
        r"\bdag\(",
        r"AcceptedGraph::dag",
        r"_dag\b",
        r"\bdag_",
        r"\.graph_posterior\(",
    ],
    "Admg": [r"Admg", r"admg"],
    "Cpdag": [r"(?<![A-Za-z])Cpdag(?![A-Za-z])", r"\bcpdag", r"AcceptedGraph::cpdag"],
    "Pag": [r"(?<![A-Za-z])Pag(?![A-Za-z])", r"\bpag", r"AcceptedGraph::pag", r"Mag"],
    "TemporalDag": [r"TemporalDag", r"temporal_dag", r"DbnPosterior", r"dbn"],
    "TemporalCpdag": [r"TemporalCpdag", r"temporal_cpdag"],
    "TemporalPag": [r"TemporalPag", r"temporal_pag"],
    "CoDetermined": [r"CoDetermined", r"codetermined", r"WithinTier"],
    "Unknown": [r"Unknown", r"unknown"],
}
STRUCTURE_TOKENS: dict[str, list[str]] = {
    "explicit": [r"\.graph\(", r"explicit", r"\.tiered_background\("],
    "accepted": [r"AcceptedGraph", r"accepted", r"accept\("],
    "graph_posterior": [
        r"graph_posterior",
        r"GraphPosterior",
        r"DbnPosterior",
        r"dag_posterior",
        r"discover_",
    ],
}
INFERENCE_TOKENS: dict[str, list[str]] = {
    "Bayesian": [r"Bayesian", r"bayes"],
    "Frequentist": [
        r"Frequentist",
        r"frequentist",
        r"bootstrap_replicates",
        r"se_analytic",
        r"\.estimate\b",
    ],
}
VALIDATION_TOKENS: dict[str, list[str]] = {
    "none": [r"RefuteSuite::None", r"\"none\"", r"refute\(suite\)", r"Validation::None"],
    "cheap": [r"RefuteSuite::Cheap", r"\"cheap\"", r"RefuteSuite::ALL"],
    "full": [r"RefuteSuite::Full", r"\"full\"", r"RefuteSuite::ALL"],
}


def axis_problems(body: str, row: dict) -> list[str]:
    tables = (
        ("query", QUERY_TOKENS),
        ("graph_class", GRAPH_TOKENS),
        ("structure", STRUCTURE_TOKENS),
        ("inference", INFERENCE_TOKENS),
        ("validation", VALIDATION_TOKENS),
    )
    problems = []
    for key, table in tables:
        value = row.get(key)
        tokens = table.get(value)
        if tokens is None:
            problems.append(f"no exercise tokens are defined for {key} {value!r}")
            continue
        if not any(re.search(tok, body) for tok in tokens):
            problems.append(f"never exercises {key} {value!r}")
    return problems


KNOWN_TRUTH_KINDS = {"internal_known_truth", "frozen_external_oracle"}
AXIS_KEYS = ("query", "graph_class", "structure", "inference", "validation")


def row_evidence_problems(row: dict) -> list[str]:
    """Why a licensed row's `evidence_test` / `evidence_assertion` is not evidence for it.

    The cited test must be an executing test (Rust: listed by `cargo test -- --list`
    and not by `--list --ignored`; Python: collected by pytest with no static skip).
    When the row claims known truth, the cited function itself (with the helpers it
    calls) must name, parse and assert on its `known_truth_fixture`. It must build
    every axis value of the row."""
    test_rel, name = row.get("evidence_test"), row.get("evidence_assertion")
    path = ROOT / str(test_rel)
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return ["evidence_test must remain inside the repository"]
    if not path.is_file():
        return [f"evidence_test {test_rel!r} does not exist"]
    if path.suffix == ".rs":
        _, problems = resolve_rust_test(path, str(name))
    elif path.suffix == ".py":
        problems = resolve_python_test(path, str(name))
    else:
        return ["evidence_test must be Rust or Python test code"]
    problems = [f"evidence_assertion {name!r} in {test_rel}: {p}" for p in problems]
    if problems:
        return problems
    fixture = row.get("known_truth_fixture")
    if row.get("evidence_kind") in KNOWN_TRUTH_KINDS and isinstance(fixture, str):
        problems += consumption_problems(path.resolve(), str(name), fixture)
    if all(isinstance(row.get(k), str) for k in AXIS_KEYS):
        problems += [f"{name} {p}" for p in axis_problems(closure(path, str(name)), row)]
    return problems


def main(argv: list[str]) -> int:
    if argv[:1] == ["rows"] and len(argv) == 2:
        import tomllib

        failed = 0
        for i, row in enumerate(tomllib.loads(Path(argv[1]).read_text()).get("cell", []), 1):
            for problem in row_evidence_problems(row):
                failed += 1
                print(f"cell #{i}: {problem}")
        return 1 if failed else 0
    if argv[:1] == ["consumers"] and len(argv) == 2:
        for c in fixture_consumers(argv[1]):
            print(c)
        return 0
    if argv[:1] == ["closure"] and len(argv) == 3:
        print(closure(ROOT / argv[1], argv[2]))
        return 0
    if argv[:1] == ["static"] and len(argv) == 3:
        path = ROOT / argv[1]
        problems = (
            static_python_test(path, argv[2])
            if path.suffix == ".py"
            else static_rust_test(path, argv[2]).problems
        )
        print("\n".join(problems) or "ok")
        return 1 if problems else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
