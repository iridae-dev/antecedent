"""Restricted Rust source equivalence proofs; unsupported syntax fails closed.

These are token and name-resolution guards, not an arbitrary Rust AST theorem.
The shared lexer owns comments/literals. Literal bytes and compound punctuation
are retained, and callers are checked in both immutable source trees.
"""

from __future__ import annotations

import hashlib
import posixpath
import re
import tomllib
from functools import lru_cache
from pathlib import PurePosixPath

import _rust_source

_IDENT = re.compile(r"(?:r#)?[A-Za-z_][A-Za-z_0-9]*")
_LIFETIME = re.compile(r"'[A-Za-z_][A-Za-z_0-9]*")
_NUMBER = re.compile(
    r"(?:0[xob][A-Za-z0-9_]+|[0-9][0-9_]*(?:\.(?!\.)(?:[0-9_]*))?(?:[eE][+-]?[0-9_]+)?(?:[A-Za-z][A-Za-z0-9_]*)?)"
)
_RAW = re.compile(r'(?:br|r)(#*)"')
_CHAR = re.compile(r"(?:b)?'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]{1,6}\}|.)|[^\\'\n])'")
_PUNCT = re.compile(
    r"<<=|>>=|\.\.=|\.\.\.|::|->|=>|==|!=|<=|>=|&&|\|\||\+=|-=|\*=|/=|%=|\^=|&=|\|=|<<|>>|\.\."
)
_BUILTIN_DERIVES = {
    "Clone",
    "Copy",
    "Debug",
    "Default",
    "Eq",
    "PartialEq",
    "Ord",
    "PartialOrd",
    "Hash",
}
_BUILTIN_ATTRIBUTES = {
    "derive",
    "must_use",
    "allow",
    "warn",
    "deny",
    "forbid",
    "expect",
    "inline",
    "cold",
    "repr",
    "non_exhaustive",
    "deprecated",
    "doc",
    "test",
    "ignore",
}


@lru_cache(maxsize=4096)
def tokens(text: str) -> tuple[str, ...]:
    """Keep literal bytes and Rust compound-token boundaries; drop comments/space."""
    parsed = _rust_source.lex(text)
    kept, skeleton = parsed.kept, parsed.skeleton
    out = []
    i = 0
    while i < len(kept):
        if kept[i].isspace():
            i += 1
            continue
        start = i
        raw = _RAW.match(kept, i)
        char = _CHAR.match(kept, i)
        # The shared skeleton distinguishes a char from a lifetime/label and
        # a raw-string opener from an identifier ending in r or b.
        literal = skeleton[i].isspace() or (
            kept[i] == "b"
            and i + 1 < len(kept)
            and kept[i + 1] in "\"'"
            and skeleton[i + 1].isspace()
        )
        if literal and raw:
            closer = '"' + raw[1]
            end = kept.find(closer, raw.end())
            if end < 0:
                raise ValueError("unterminated raw literal")
            i = end + len(closer)
        elif literal and char:
            i = char.end()
        elif literal and (kept[i] == '"' or kept.startswith('b"', i)):
            i += 2 if kept.startswith('b"', i) else 1
            while i < len(kept) and kept[i] != '"':
                i += 2 if kept[i] == "\\" else 1
            if i >= len(kept):
                raise ValueError("unterminated string literal")
            i += 1
        elif literal:
            raise ValueError("unsupported literal")
        else:
            if ord(kept[i]) > 127:
                raise ValueError("non-ASCII code identifier needs a Rust parser")
            match = (
                _LIFETIME.match(kept, i)
                or _IDENT.match(kept, i)
                or _NUMBER.match(kept, i)
                or _PUNCT.match(kept, i)
            )
            i = match.end() if match else i + 1
        out.append(kept[start:i])
    return tuple(out)


def _uses(code: tuple[str, ...], public_only: bool = False):
    """Use-tree spans. Macro-generated trees are deliberately not interpreted."""
    found = []
    depth = 0
    i = 0
    while i < len(code):
        if code[i] == "use" and (
            not public_only or (i > 0 and code[i - 1] == "pub" and depth == 0)
        ):
            start = i - 1 if i > 0 and code[i - 1] == "pub" else i
            end = i + 1
            braces = 0
            while end < len(code):
                if code[end] == "{":
                    braces += 1
                if code[end] == "}":
                    braces -= 1
                if code[end] == ";" and braces == 0:
                    break
                end += 1
            if end == len(code):
                raise ValueError("unresolved use tree")
            found.append((start, end + 1, code[i + 1 : end]))
            i = end + 1
            continue
        depth += (code[i] == "{") - (code[i] == "}")
        i += 1
    return found


def _use_paths(tree: tuple[str, ...]) -> list[tuple[tuple[str, ...], str | None]]:
    """Expand ordinary grouped use paths, retaining aliases/globs for rejection."""
    result = []
    i = 0

    def walk(prefix):
        nonlocal i
        path = list(prefix)
        while i < len(tree):
            token = tree[i]
            if token == "::":
                if i == 0 and not path:
                    path.append("::")
                i += 1
            elif token == "{":
                i += 1
                while i < len(tree) and tree[i] != "}":
                    walk(tuple(path))
                    if i < len(tree) and tree[i] == ",":
                        i += 1
                    elif i < len(tree) and tree[i] != "}":
                        raise ValueError("unresolved use group")
                if i >= len(tree):
                    raise ValueError("unclosed use group")
                i += 1
                return
            elif token in (",", "}"):
                break
            elif token == "as":
                i += 1
                if i >= len(tree) or not _IDENT.fullmatch(tree[i]):
                    raise ValueError("invalid use alias")
                alias = tree[i]
                i += 1
                result.append((tuple(path), alias))
                return
            elif _IDENT.fullmatch(token) or token == "*":
                path.append(token)
                i += 1
            else:
                raise ValueError("unsupported use path")
        if not path:
            raise ValueError("empty use path")
        result.append((tuple(path), None))

    walk(())
    if i != len(tree):
        raise ValueError("unresolved use suffix")
    return result


def _module(path: str) -> tuple[str, ...] | None:
    parts = PurePosixPath(path).parts
    if len(parts) < 4 or parts[0] != "crates" or parts[2] != "src" or not parts[-1].endswith(".rs"):
        return None
    tail = list(parts[3:])
    tail[-1] = tail[-1][:-3]
    if tail[-1] in ("lib", "main", "mod"):
        tail.pop()
    return (parts[1].replace("-", "_"), *tail)


def _canonical(path, module):
    if not path:
        return ()
    if path[0] == "::":
        return tuple(path[1:])
    if path[0] == "crate":
        return (module[0], *path[1:])
    if path[0] == "self":
        return (*module, *path[1:])
    if path[0] == "super":
        parent = list(module)
        index = 0
        while index < len(path) and path[index] == "super":
            if len(parent) <= 1:
                raise ValueError("unresolved super path")
            parent.pop()
            index += 1
        return (*parent, *path[index:])
    # Explicit external crate paths remain absolute. Bare local paths are also
    # tested relative to the file's module below; no global terminal-name proof.
    return tuple(path)


def _outer_docs(text: str) -> tuple[str, ...]:
    parsed = _rust_source.lex(text)
    spans = list(parsed.line_docs)
    # Block doc comments also become #[doc] attributes in a derive input.
    for match in re.finditer(r"/\*\*(?!\*)", text):
        start = match.start()
        if parsed.kept[start : start + 3] != "   ":
            continue
        i, depth = start + 2, 1
        while i < len(text) and depth:
            if text.startswith("/*", i):
                depth += 1
                i += 2
            elif text.startswith("*/", i):
                depth -= 1
                i += 2
            else:
                i += 1
        if depth:
            raise ValueError("unterminated block doc comment")
        spans.append((start, i))
    return tuple(text[a:b] for a, b in sorted(spans))


def _hazard(text: str) -> str | None:
    code = tokens(text)
    # Any macro in the changed file may expand to line!/include!/stringify!.
    if any(
        code[i] == "!" and i > 0 and (_IDENT.fullmatch(code[i - 1]) or code[i - 1] == "macro_rules")
        for i in range(len(code))
    ):
        return "macro invocation or definition in changed source"
    derived = set()
    for i, token in enumerate(code):
        if token != "#":
            continue
        j = i + 1
        if j < len(code) and code[j] == "!":
            j += 1
        if j + 1 >= len(code) or code[j] != "[":
            continue
        name = code[j + 1]
        if name not in _BUILTIN_ATTRIBUTES:
            return "procedural or conditional attribute in changed source"
        if name == "derive":
            end = j + 2
            while end < len(code) and code[end] != "]":
                end += 1
            names = {t for t in code[j + 2 : end] if _IDENT.fullmatch(t)}
            if not names <= _BUILTIN_DERIVES:
                return "nonbuiltin derive may consume changed documentation"
            derived |= names
    for _, _, tree in _uses(code):
        for imported, alias in _use_paths(tree):
            if derived and ((alias or imported[-1]) in derived or "*" in imported):
                return "derive binding may be shadowed by imported procedural macro"
    return None


def _derive_scope(path: str, sources: dict[str, str], names: set[str]) -> str | None:
    """Builtin derive spellings need an unambiguous macro-import namespace."""
    if not names:
        return None
    module = _module(path)
    if module is None:
        return "unresolved derive module context"
    root = "/".join(path.split("/")[:2]) + "/src/"

    def source_for(parts):
        if not parts or parts[0] != module[0]:
            return None
        tail = "/".join(parts[1:])
        candidates = [root + tail + ".rs", root + tail + "/mod.rs"] if tail else [root + "lib.rs"]
        return next((p for p in candidates if p in sources), None)

    def resolve(parts, current):
        canonical = _canonical(parts, current)
        for candidate in (canonical, (*current, *parts), (module[0], *parts)):
            if found := source_for(candidate):
                return found, candidate
        return None

    visited = set()

    def imported_names_are_safe(rel, current, exports_only):
        key = (rel, exports_only)
        if key in visited:
            return False  # unresolved reexport cycles fail closed
        visited.add(key)
        try:
            for _, _, tree in _uses(tokens(sources[rel]), public_only=exports_only):
                for imported, alias in _use_paths(tree):
                    if (alias or imported[-1]) in names:
                        return False
                    if imported[-1] == "*":
                        target = resolve(imported[:-1], current)
                        if target is None or not imported_names_are_safe(*target, False):
                            return False
            return True
        finally:
            visited.remove(key)

    for size in range(1, len(module) + 1):
        current = module[:size]
        rel = source_for(current)
        if rel is None or not imported_names_are_safe(rel, current, False):
            return "ancestor imports may shadow builtin derive with procedural macro"
    return None


def _source_literals(code: tuple[str, ...]) -> list[str]:
    # Retain concat! pieces too. Decode Rust's scalar escapes so spelling a
    # filename with \xNN/\u{...} cannot hide a source-introspection dependency.
    literals = []
    for token in code:
        if '"' not in token:
            continue
        first, last = token.find('"'), token.rfind('"')
        if first == last:
            continue
        value = token[first + 1 : last]
        if token[:first].startswith(("r", "br")):
            literals.append(value)
            continue

        def escape(match):
            value = match.group(1)
            if value.startswith("x"):
                return chr(int(value[1:], 16))
            if value.startswith("u{"):
                return chr(int(value[2:-1].replace("_", ""), 16))
            return {"n": "\n", "r": "\r", "t": "\t", "0": "\0", "\\": "\\", '"': '"', "'": "'"}.get(
                value, value
            )

        value = re.sub(r"\\(x[0-9a-fA-F]{2}|u\{[0-9a-fA-F_]+\}|.)", escape, value)
        literals.append(value)
    return literals


def _source_inclusion(code: tuple[str, ...], path: str) -> bool:
    if not any(
        t
        in (
            "include",
            "include_str",
            "include_bytes",
            "path",
            "read",
            "read_to_string",
            "read_to_end",
            "open",
        )
        for t in code
    ):
        return False
    return PurePosixPath(path).name in "".join(_source_literals(code))


def _build_context(root: str, sources: dict[str, str]) -> str | None:
    """Reject arbitrary workspace build code in the entire local compile closure."""
    try:
        workspace = (
            tomllib.loads(sources.get("Cargo.toml", ""))
            .get("workspace", {})
            .get("dependencies", {})
        )
        seen, pending = set(), [root]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            manifest = sources.get(current + "/Cargo.toml")
            if manifest is None:
                return "missing local compile-dependency manifest"
            config = tomllib.loads(manifest)
            build = config.get("package", {}).get("build")
            if build not in (None, False) or (
                current + "/build.rs" in sources and build is not False
            ):
                return "compile dependency build script may inspect changed source"
            tables = [config]
            tables.extend(config.get("target", {}).values())
            for table in tables:
                for key in ("dependencies", "build-dependencies", "dev-dependencies"):
                    for name, dependency in table.get(key, {}).items():
                        owner = current
                        if not isinstance(dependency, dict):
                            continue
                        if dependency.get("workspace"):
                            if name not in workspace:
                                return "unresolved workspace compile dependency"
                            dependency, owner = workspace[name], ""
                            if not isinstance(dependency, dict):
                                continue
                        if "path" in dependency:
                            resolved = posixpath.normpath(posixpath.join(owner, dependency["path"]))
                            if resolved.startswith("../") or resolved.startswith("/"):
                                return "compile dependency escapes immutable workspace"
                            pending.append(resolved)
        # A different workspace script could consume this source through a
        # static direct read or a recursive scanner. Such scripts fail closed;
        # unrelated scripts that only emit linker metadata remain harmless.
        scripts = {rel for rel in sources if rel.endswith("/build.rs")}
        for rel, text in sources.items():
            if not rel.endswith("/Cargo.toml"):
                continue
            build = tomllib.loads(text).get("package", {}).get("build")
            if isinstance(build, str):
                scripts.add(
                    posixpath.normpath(posixpath.join(str(PurePosixPath(rel).parent), build))
                )
        for rel in scripts:
            if rel not in sources:
                return "missing immutable custom workspace build script"
            code = tokens(sources[rel])
            if set(code) & {
                "read",
                "read_to_string",
                "read_to_end",
                "open",
                "read_dir",
                "WalkDir",
                "walkdir",
                "glob",
                "glob_with",
            }:
                return "workspace build script has unresolved filesystem/source scanning"
    except (tomllib.TOMLDecodeError, AttributeError, TypeError):
        return "unresolved local build compile context"
    return None


def _compile_context(path: str, sources: dict[str, str]) -> str | None:
    module = _module(path)
    if module is None:
        return "unsupported conventional crate module context"
    root = "/".join(path.split("/")[:2])
    manifest = sources.get(root + "/Cargo.toml")
    if manifest is None:
        return "missing immutable Cargo compile context"
    try:
        config = tomllib.loads(manifest)
    except tomllib.TOMLDecodeError:
        return "unresolved Cargo compile context"
    library = config.get("lib", {})
    package = config.get("package", {})
    name = library.get("name", package.get("name", "").replace("-", "_"))
    if (
        name != module[0]
        or library.get("path", "src/lib.rs") != "src/lib.rs"
        or library.get("proc-macro")
    ):
        return "custom crate identity/path/procedural compile context"
    return _build_context(root, sources)


def _caller_proof(
    path: str, added: set[str], sources: dict[str, str], *, old_references: bool
) -> str | None:
    target = _module(path)
    if target is None or path not in sources:
        return "unsupported module identity or missing immutable source tree"
    # The target must be a conventional out-of-line module, not a path attribute
    # or macro-generated inclusion whose compile context we cannot resolve.
    child = PurePosixPath(path)
    leaf = target[-1]
    while child.parent.name != "src" or child.name not in ("lib.rs", "main.rs"):
        parent_path = str(child.parent / "mod.rs")
        if child.name == "mod.rs":
            parent_path = str(child.parent.parent / "mod.rs")
            leaf = child.parent.name
        if parent_path not in sources:
            parent_path = str(PurePosixPath(parent_path).parent / "lib.rs")
        if parent_path == str(child) or parent_path not in sources:
            return "unresolved module declaration context"
        parent = tokens(sources[parent_path])
        declaration = ("pub", "mod", leaf, ";")
        matches = [i for i in range(len(parent)) if parent[i : i + 4] == declaration]
        if len(matches) != 1:
            return "target lacks unique conventional public module declaration"
        if matches[0] > 0 and parent[matches[0] - 1] == "]":
            return "conditional/procedural module declaration"
        child = PurePosixPath(parent_path)
    for rel, text in sources.items():
        if not rel.endswith(".rs"):
            continue
        code = tokens(text)
        module = _module(rel) or (
            PurePosixPath(rel).parts[1].replace("-", "_")
            if rel.startswith("crates/")
            else "external",
        )
        imports = _uses(code)
        masked = list(code)
        bound = set()
        aliases = {}
        glob = False
        for start, end, tree in imports:
            try:
                paths = _use_paths(tree)
            except ValueError:
                if added & set(code) or target[-1] in code:
                    return f"{rel}: unresolved caller import syntax"
                continue
            for imported, alias in paths:
                try:
                    normalized = _canonical(imported, module)
                except ValueError:
                    if added & set(code) or target[-1] in code:
                        return f"{rel}: unresolved caller module context"
                    continue
                relative = (*module, *imported)
                possibilities = (normalized, relative)
                if imported[-1] == "self":
                    possibilities = tuple(p[:-1] for p in possibilities)
                if alias:
                    aliases[alias] = normalized
                if imported[-1] == "*":
                    glob = True
                # Importing a target module/ancestor under an alias, or globbing
                # it, can expose the added name indirectly. Refuse this scope.
                module_paths = tuple(p[:-1] if p[-1] == "*" else p for p in possibilities)
                if any(p == target for p in module_paths):
                    return f"{rel}: ambiguous target module import/alias/glob"
                if any(target[: len(p)] == p for p in module_paths) and (
                    alias or imported[-1] == "self" or added & set(code)
                ):
                    return f"{rel}: ambiguous target ancestor import/alias/glob"
                if imported[-1] == "*" and target[-1] in imported:
                    return f"{rel}: unresolved target-module glob context"
                if old_references and (
                    any(p == (*target, name) for p in possibilities for name in added)
                    or (target[-1] in imported and imported[-1] in added)
                ):
                    return f"{rel}: old explicit import resolves through added export"
                if imported[-1] in added:
                    # Only a fully qualified unrelated import establishes a
                    # same-name binding. Bare or relative aliases are ambiguous.
                    if len(normalized) < 3 or (
                        imported[0] in ("self", "super") and _module(rel) is None
                    ):
                        return f"{rel}: unresolved same-name import"
                    bound.add(alias or imported[-1])
            masked[start:end] = [""] * (end - start)
        rest = tuple(masked)
        # Expand local module aliases when checking qualified references. This
        # catches `use antecedent::analysis as a; a::target::NEW` conservatively.
        for i, token in enumerate(rest):
            if token not in added:
                continue
            chain = [token]
            j = i
            while j >= 2 and rest[j - 1] == "::" and _IDENT.fullmatch(rest[j - 2]):
                chain.insert(0, rest[j - 2])
                j -= 2
            if chain[0] in aliases:
                chain = [*aliases[chain[0]], *chain[1:]]
            if old_references and target[-1] in chain:
                return f"{rel}: old target-module qualified export reference"
            canonical = _canonical(tuple(chain), module)
            if old_references and (
                canonical == (*target, token) or (*module, *chain) == (*target, token)
            ):
                return f"{rel}: old qualified reference to added export"
            if len(chain) == 1:
                local_definition = i > 0 and rest[i - 1] in (
                    "const",
                    "static",
                    "fn",
                    "struct",
                    "enum",
                    "type",
                    "mod",
                    "let",
                )
                declared = any(
                    rest[k] in ("const", "static", "fn", "struct", "enum", "type", "mod")
                    and k + 1 < len(rest)
                    and rest[k + 1] == token
                    for k in range(len(rest))
                )
                if old_references and (
                    module == target
                    or (not local_definition and not declared and token not in bound)
                    or (glob and token not in bound)
                ):
                    return f"{rel}: unresolved old lexical reference to added export"
        # Any source inclusion naming this file, even through concat!, invalidates
        # a source-only proof. Literal contents were intentionally retained.
        if _source_inclusion(code, path):
            return f"{rel}: source inclusion may consume changed source"
        # An alias/reexport exposing an ancestor makes downstream resolution
        # nonlocal; reject it even if this particular file never uses NEW.
        if aliases and any(target[: len(p)] == p for p in aliases.values()):
            return f"{rel}: target ancestor alias needs AST resolution"
    return None


# This is deliberately NOT a generic logging/AST equivalence rule. Each entry
# binds a reviewed complete historical token stream, the exact ignored test and
# its single closed-format observer of existing primitive locals. The two streams
# are identical at c30/e15/70; no arbitrary numerical-owner review bypass exists.
_TERMINAL_TEST_OBSERVERS = {
    "crates/antecedent-stats/src/ci/calibration.rs": (
        "scalable_gpdc_replacement_validated_separately",
        "57fa2318405a7bbc5af71225827ae39b888a7e7678926ffe87c6e9b487f9458a",
        "d1d9e4fa7b0eaa7d8e11f115284a7f7473e56d5b56eedecb817bcfbc1fbd6a80",
    ),
    "crates/antecedent-validate/src/sensitivity.rs": (
        "scalable_sensitivity_replacement_validated_separately",
        "e791b04d7c9c112cf14a40e070bf0386eaac0874b4dd6636b980eac5b1ebde7b",
        "9363d3d5523c24aefa2514a3f5617d0b01c895ca5f45e928ecd774f5f2501737",
    ),
}


def _isolated_python_defaults_scanner(rel, code, sources):
    """One exact reviewed scanner whose closed root is python/src, not crates/.

    Its private test helper traverses only CARGO_MANIFEST_DIR/src. Reject any
    alternate inclusion of that source so another crate cannot change the root.
    This is a fixture-specific proof, not a cfg(test) or scanner exemption.
    """
    if rel != "python/src/defaults.rs":
        return False
    digest = hashlib.sha256("\0".join(code).encode()).hexdigest()
    if digest != "d7a7ff566c2a2d73114a55f5bd9d0561874a053fdd2612d7d3243a951304fbc7":
        return False
    for other, text in sources.items():
        if not other.endswith(".rs") or other == rel:
            continue
        candidate = tokens(text)
        if any("defaults.rs" in value for value in _source_literals(candidate)):
            return False
        if not other.startswith("python/") and any(
            candidate[i : i + 2] == ("mod", "defaults") for i in range(len(candidate))
        ):
            return False
    return True


def _terminal_test_observer(
    path: str,
    old: str,
    new: str,
    old_sources: dict[str, str],
    new_sources: dict[str, str],
) -> str | None:
    specification = _TERMINAL_TEST_OBSERVERS.get(path)
    if specification is None:
        return "test observer path is not independently reviewed"
    name, expected_digest, expected_format = specification
    before, after = tokens(old), tokens(new)
    if hashlib.sha256("\0".join(before).encode()).hexdigest() != expected_digest:
        return "test observer original full token stream is not the reviewed primitive fixture"
    starts = [i for i in range(len(before)) if before[i : i + 5] == ("fn", name, "(", ")", "{")]
    if len(starts) != 1:
        return "test observer lacks its unique reviewed parameterless function"
    opening = starts[0] + 4
    depth, closing = 1, opening + 1
    while closing < len(before) and depth:
        depth += (before[closing] == "{") - (before[closing] == "}")
        closing += 1
    if depth:
        return "unbalanced test observer function"
    closing -= 1
    body = before[opening + 1 : closing]
    # Establish that the new observer is terminal AFTER the original final
    # scientific assertion, rather than before it or in a numerical branch.
    depth, start, final = 0, 0, ()
    for i, token in enumerate(body):
        depth += (token in ("(", "[", "{")) - (token in (")", "]", "}"))
        if token == ";" and depth == 0:
            final, start = body[start : i + 1], i + 1
    if start != len(body) or final[:3] != ("assert", "!", "("):
        return "test observer is not after the reviewed terminal assertion"
    if len(after) != len(before) + 9:
        return "test observer has extra tokens or missing original tokens"
    literal = after[closing + 6]
    if hashlib.sha256(literal.encode()).hexdigest() != expected_format:
        return "test observer format is not the reviewed closed primitive capture literal"
    observer = ("::", "std", "::", "eprintln", "!", "(", literal, ")", ";")
    if after != before[:closing] + observer + before[closing:]:
        return (
            "only the exact qualified terminal observer is permitted; all other tokens must match"
        )
    return _restricted_test_source_context(path, before, old, new, old_sources, new_sources)


def _reviewed_test_lint(path, old, new, old_sources, new_sources):
    if path != "crates/antecedent/tests/sample_split_parent_set_effect_calibration.rs":
        return "lint-only test path is not independently reviewed"
    before, after = tokens(old), tokens(new)
    if hashlib.sha256("\0".join(before).encode()).hexdigest() != (
        "61d2d1d1af4d6f2d9dd58b52547c22ce1d4352e02a416fa34303177ac08b07de"
    ):
        return "lint-only original full token stream is not the reviewed fixture"
    attribute = (
        "#",
        "[",
        "allow",
        "(",
        "clippy",
        "::",
        "too_many_lines",
        ",",
        "reason",
        "=",
        '"keep one complete discovery, admission, estimation and recovery audit together per replicate"',
        ")",
        "]",
    )
    # Fixed insertion immediately before the reviewed non-macro replicate item.
    index = 862
    if before[index : index + 2] != ("fn", "replicate"):
        return "lint-only reviewed item location changed"
    if after != before[:index] + attribute + before[index:]:
        return "only the exact builtin Clippy allow attribute may be inserted"
    return _restricted_test_source_context(
        path, before, old, new, old_sources, new_sources, lint_only=True
    )


def _restricted_test_source_context(
    path, before, old, new, old_sources, new_sources, *, lint_only=False
):
    crate = "/".join(path.split("/")[:2]) + "/"
    for sources, text in ((old_sources, old), (new_sources, new)):
        if sources.get(path) != text:
            return "test observer lacks its exact immutable source tree"
        context = crate + "src/lib.rs" if lint_only else path
        if problem := _compile_context(context, sources):
            return problem
        manifest = tomllib.loads(sources[crate + "Cargo.toml"])
        tables = [manifest, *manifest.get("target", {}).values()]
        if any(
            "std" in table.get(key, {})
            for table in tables
            for key in ("dependencies", "dev-dependencies", "build-dependencies")
        ):
            return "test observer crate renames a dependency into the std namespace"
        derived = set()
        for i, token in enumerate(before):
            if token == "derive" and before[i + 1 : i + 2] == ("(",):
                j = i + 2
                while j < len(before) and before[j] != ")":
                    if _IDENT.fullmatch(before[j]):
                        derived.add(before[j])
                    j += 1
        if not lint_only:
            if problem := _derive_scope(path, sources, derived):
                return problem
        for rel, source in sources.items():
            if not rel.endswith(".rs"):
                continue
            code = tokens(source)
            # A reviewed build script writes only a link declaration here; its
            # exact token binding excludes source-derived generated Rust input.
            generated_link = (
                "include",
                "!",
                "(",
                "concat",
                "!",
                "(",
                "env",
                "!",
                "(",
                '"OUT_DIR"',
                ")",
                ",",
                '"/python_test_link.rs"',
                ")",
                ")",
                ";",
            )
            if rel == "python/src/lib.rs":
                build = tokens(sources.get("python/build.rs", ""))
                if hashlib.sha256("\0".join(build).encode()).hexdigest() == (
                    "6601e98ed8a06a7642685135da2aa450caa3c7553e5f3bf76d80ca676b2c908e"
                ):
                    for i in range(len(code)):
                        if code[i : i + len(generated_link)] == generated_link:
                            code = code[:i] + code[i + len(generated_link) :]
                            break
            # Only actual read/include operations count, not a local variable
            # named `read`. Ordinary module #[path] attributes are compiler
            # declarations, not data-source reads.
            module_literals = set()
            for i in range(len(code) - 5):
                if code[i : i + 4] == ("#", "[", "path", "=") and code[i + 5] == "]":
                    module_literals.add(i + 4)
            imports = list(_uses(code))
            operations = {
                "include",
                "include_str",
                "include_bytes",
                "read",
                "read_to_string",
                "read_to_end",
                "open",
            }
            active = {
                i
                for i, token in enumerate(code)
                if token in operations
                and (
                    code[i + 1 : i + 2] in (("(",), ("!",))
                    or i > 0
                    and code[i - 1] in ("::", ".")
                    or any(start <= i < end for start, end, _ in imports)
                )
            }
            data_code = tuple(
                token
                for i, token in enumerate(code)
                if i not in module_literals
                and token != "path"
                and (token not in operations or i in active)
            )
            literals = _source_literals(data_code)
            leaf = re.escape(PurePosixPath(path).name)
            named_source = re.compile(r"(?:^|[/\\])" + leaf + r"(?:$|[:#])")
            reads = bool(active)
            if reads and any(named_source.search(value) for value in literals):
                return f"{rel}: source introspection may consume the observer change"
            # A dynamically assembled source filename is outside this restricted
            # direct-path proof, as is a directory scanner selecting Rust source.
            source_literals = any(
                value in {"src", "rs", ".rs"} or value.endswith(".rs") for value in literals
            )
            if reads and "concat" in code and source_literals:
                return f"{rel}: dynamically assembled source input is unresolved"
            if source_literals and any(
                code[i] in {"read_dir", "WalkDir", "walkdir", "glob", "glob_with"}
                and code[i + 1 : i + 2] in (("(",), ("::",))
                for i in range(len(code))
            ):
                if not _isolated_python_defaults_scanner(rel, code, sources):
                    return f"{rel}: unresolved source scanning forbids test observer proof"
            if rel == path and any(
                code[i] in {"line", "column", "file", "stringify"} and code[i + 1 : i + 2] == ("!",)
                for i in range(len(code))
            ):
                return "test observer source has location/token introspection"
            if (
                not lint_only
                and rel.startswith(crate)
                and (
                    any(
                        code[i : i + 2] in (("as", "std"), ("mod", "std")) for i in range(len(code))
                    )
                    or any(
                        (alias or imported[-1]) == "std"
                        for _, _, use in _uses(code)
                        for imported, alias in _use_paths(use)
                    )
                )
            ):
                return "test observer crate has an unresolved std namespace binding"
            if rel.startswith(crate) and {"register_tool", "register_attr"} & set(code):
                return "reviewed test crate has unresolved attribute/tool registration"
            if rel.startswith(crate) and "macro_use" in code:
                return "test observer crate has unresolved macro-use scope"
            if rel.startswith(crate) and any(
                code[i : i + 2] == ("macro_rules", "!")
                and i + 2 < len(code)
                and code[i + 2] in {"eprintln", "assert", "assert_eq", "assert_ne", "vec", "format"}
                for i in range(len(code))
            ):
                return "test observer crate shadows a reviewed builtin macro"
    return None


def prove(
    kind: str,
    path: str,
    old: str,
    new: str,
    old_sources: dict[str, str],
    new_sources: dict[str, str],
) -> str | None:
    """None only when the restricted equivalence proof succeeds."""
    try:
        if kind == "rust_reviewed_test_lint":
            return _reviewed_test_lint(path, old, new, old_sources, new_sources)
        if kind == "rust_terminal_test_observer":
            return _terminal_test_observer(path, old, new, old_sources, new_sources)
        for sources in (old_sources, new_sources):
            if context := _compile_context(path, sources):
                return context
        for text in (old, new):
            if hazard := _hazard(text):
                return hazard
        before, after = tokens(old), tokens(new)
        derived = set()
        for i, token in enumerate(before):
            if token == "derive" and i + 1 < len(before) and before[i + 1] == "(":
                end = i + 2
                while end < len(before) and before[end] != ")":
                    if _IDENT.fullmatch(before[end]):
                        derived.add(before[end])
                    end += 1
        for sources in (old_sources, new_sources):
            if problem := _derive_scope(path, sources, derived):
                return problem
        if "derive" in before and _outer_docs(old) != _outer_docs(new):
            return "changed item documentation may be consumed by a derive"
        if "derive" in before and any(
            "macro_use" in tokens(text)
            for rel, text in old_sources.items()
            if rel.endswith(".rs") and rel.startswith("/".join(path.split("/")[:2]) + "/")
        ):
            return "crate macro-use scope may shadow builtin derive"
        if kind == "rust_comments":
            if before != after:
                return "noncomment Rust tokens changed"
            # Includes can inspect otherwise inert changed docs from another file.
            for sources in (old_sources, new_sources):
                if path not in sources:
                    return "missing immutable source tree"
                for rel, text in sources.items():
                    if not rel.endswith(".rs"):
                        continue
                    code = tokens(text)
                    if _source_inclusion(code, path):
                        return f"{rel}: source inclusion may consume changed source"
            return None
        if kind != "rust_pub_use_addition":
            return "unsupported Rust equivalence kind"

        def exports(code):
            retained = list(code)
            paths = set()
            for start, end, tree in _uses(code, public_only=True):
                for item, alias in _use_paths(tree):
                    if alias is not None or "*" in item or "self" in item:
                        raise ValueError("aliases/globs/self exports need AST proof")
                    paths.add(item)
                retained[start:end] = [""] * (end - start)
            return tuple(t for t in retained if t), paths

        old_body, old_exports = exports(before)
        new_body, new_exports = exports(after)
        if old_body != new_body:
            return "non-export Rust tokens changed"
        if not old_exports < new_exports:
            return "exports removed, retargeted, or no strict addition"
        additions = new_exports - old_exports
        names = {p[-1] for p in additions}
        if len(names) != len(additions) or names & {p[-1] for p in old_exports}:
            return "added export name collision"
        if names & set(before):
            return "added export already has old lexical binding/reference"
        for sources, old_references in ((old_sources, True), (new_sources, False)):
            if problem := _caller_proof(path, names, sources, old_references=old_references):
                return problem
        return None
    except (ValueError, IndexError) as error:
        return f"unsupported Rust syntax: {error}"
