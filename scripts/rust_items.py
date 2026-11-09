"""Rust items of one source file, found by structure and hashed by their tokens.

Evidence binds a record to the items its test executed or depends on, not to whole
files. An item is named by its *locator*: the kind and name of each enclosing item
from the file's root, with an ordinal among siblings of the same kind and name
(`mod:tests#0/fn:fits#0`; an impl is named by its header). Its parts are hashed
from tokens, so whitespace and comment edits never change them; literal contents
are hashed exactly:

* `full`: the whole item with its outer attributes;
* `sig`: a function's attributes and header, up to its body;
* `head`: an impl's or inline module's attributes and header;
* `inner`: a file's or inline module's inner attributes.

Doc comments are text only to a procedural macro that reads them, so they are
hashed for items under a derive or attribute this module does not know to ignore
them. A file this parser cannot read raises `ParseError`; the caller keeps the
whole file.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

import _rust_source

_WORD = re.compile(r"(?:r#)?[A-Za-z_][A-Za-z0-9_]*")
_BRACKET = re.compile(r"[()\[\]{}]")
_PAIRS = {")": "(", "]": "[", "}": "{"}

# Attributes that never read the item's documentation.
_INERT_ATTRIBUTES = {
    "allow", "automatically_derived", "cfg", "cfg_attr", "cold", "coverage", "deny", "deprecated",
    "derive", "doc", "error", "expect", "export_name", "forbid", "from", "global_allocator", "ignore",
    "inline", "link", "link_name", "link_section", "macro_export", "macro_use", "must_use",
    "no_mangle", "non_exhaustive", "panic_handler", "path", "repr", "serde", "should_panic", "source",
    "target_feature", "test", "track_caller", "unsafe", "used", "warn", "clippy", "rustfmt",
    "diagnostic", "allow_internal_unstable", "backtrace", "display", "debug", "into", "as_ref",
    "deref", "add", "mul",
}
# Derives whose expansions ignore documentation.
_INERT_DERIVES = {
    "Debug", "Clone", "Copy", "PartialEq", "Eq", "PartialOrd", "Ord", "Hash", "Default",
    "Serialize", "Deserialize", "Error", "Display", "From", "Into", "Deref", "DerefMut", "AsRef",
    "Add", "Sub", "Mul", "Div", "Neg", "AddAssign", "SubAssign", "MulAssign", "DivAssign", "Sum",
    "Constructor", "IsVariant", "Unwrap", "TryInto", "FromStr", "Index", "IndexMut",
}


class ParseError(ValueError):
    pass


@dataclass(eq=False)
class Node:
    kind: str
    name: str
    start: int
    end: int
    head_end: int
    attrs: list[tuple[int, int]] = field(default_factory=list)
    children: list["Node"] = field(default_factory=list)
    inner: list[tuple[int, int]] = field(default_factory=list)
    parent: "Node | None" = None
    locator: str = ""


class File:
    """One parsed file: its item tree, locators and token hashes."""

    def __init__(self, text: str):
        self.text = text
        views = _rust_source.lex(text)
        self.kept = views.kept
        self.skel = views.kept_skeleton
        self.literals = sorted(views.literals)
        self.docs = views.line_docs
        self.close = self._brackets()
        self._doc_order = sorted(range(len(self.docs)), key=lambda i: self.docs[i][1])
        self._doc_ends = [self.docs[i][1] for i in self._doc_order]
        self.root = Node("root", "", 0, len(text), len(text))
        self._items(0, len(text), self.root)
        self.by_locator: dict[str, Node] = {}
        self._name(self.root)
        self._starts = None
        self._ascii: bool | None = None
        self._bytes: bytes | None = None

    # ------------------------------------------------------------------ parsing

    def _brackets(self) -> dict[int, int]:
        close: dict[int, int] = {}
        stack: list[int] = []
        for m in _BRACKET.finditer(self.skel):
            c = m.group(0)
            if c in "([{":
                stack.append(m.start())
            else:
                if not stack or self.skel[stack[-1]] != _PAIRS[c]:
                    raise ParseError(f"unbalanced {c!r} at {m.start()}")
                close[stack.pop()] = m.start()
        if stack:
            raise ParseError(f"unclosed bracket at {stack[-1]}")
        return close

    def _ws(self, p: int, hi: int) -> int:
        s = self.skel
        while p < hi and s[p].isspace():
            p += 1
        return p

    def _word(self, p: int) -> tuple[str, int]:
        m = _WORD.match(self.skel, p)
        return (m.group(0), m.end()) if m else ("", p)

    def _group_end(self, p: int) -> int:
        if p not in self.close:
            raise ParseError(f"expected a group at {p}")
        return self.close[p] + 1

    def _to_semicolon(self, p: int, hi: int) -> int:
        s = self.skel
        while p < hi:
            c = s[p]
            if c in "([{":
                p = self._group_end(p)
            elif c == ";":
                return p + 1
            elif c in ")]}":
                break
            else:
                p += 1
        raise ParseError(f"expected ';' before {p}")

    def _to_body(self, p: int, hi: int, stops: str = "{;") -> int:
        """The first stop outside brackets and generic arguments.

        Headers hold no comparisons, so `<` and `>` (but not `->` or `=>`) nest.
        """
        s = self.skel
        angle = 0
        while p < hi:
            c = s[p]
            if c == "<":
                angle += 1
                p += 1
                continue
            if c == ">" and p > 0 and s[p - 1] not in "-=":
                angle -= 1
                p += 1
                continue
            if c in stops and angle <= 0:
                return p
            if c in "([{":
                p = self._group_end(p)
            elif c in ")]}":
                break
            else:
                p += 1
        raise ParseError(f"expected an item body before {p}")

    def _doc_start(self, start: int, lo: int) -> int:
        """`start`, moved back over the `///` comments directly above it."""
        import bisect
        while True:
            i = bisect.bisect_left(self._doc_ends, start) - 1
            if i < 0:
                return start
            a, b = self.docs[self._doc_order[i]]
            if a < lo or self.skel[b:start].strip():
                return start
            start = a

    def _items(self, lo: int, hi: int, parent: Node) -> None:
        s = self.skel
        p = lo
        while True:
            p = self._ws(p, hi)
            if p >= hi:
                return
            if s[p] == ";":
                p += 1
                continue
            start, attrs = p, []
            while s.startswith("#", p):
                q = self._ws(p + 1, hi)
                inner = s.startswith("!", q)
                if inner:
                    q = self._ws(q + 1, hi)
                if not s.startswith("[", q):
                    raise ParseError(f"expected an attribute at {p}")
                end = self._group_end(q)
                if inner:
                    if attrs:
                        raise ParseError(f"inner attribute after outer at {p}")
                    parent.inner.append((p, end))
                    p = self._ws(end, hi)
                    start = p
                    continue
                attrs.append((p, end))
                p = self._ws(end, hi)
            if p >= hi:
                if attrs:
                    raise ParseError("attributes without an item")
                return
            # Doc comments directly above belong to the item.
            start = self._doc_start(start, lo)
            node = self._item(start, p, hi, attrs)
            node.parent = parent
            parent.children.append(node)
            p = node.end

    def _item(self, start: int, p: int, hi: int, attrs: list) -> Node:
        s = self.skel
        word, after = self._word(p)
        if word == "pub":
            p = self._ws(after, hi)
            if s.startswith("(", p):
                p = self._ws(self._group_end(p), hi)
            word, after = self._word(p)
        while True:
            nxt, nxt_after = self._word(self._ws(after, hi))
            if word in ("default", "async", "unsafe", "safe", "auto") and nxt:
                p, (word, after) = self._ws(after, hi), (nxt, nxt_after)
            elif word == "const" and nxt in ("fn", "unsafe", "async", "extern"):
                p, (word, after) = self._ws(after, hi), (nxt, nxt_after)
            elif word == "extern" and nxt != "crate":
                q = self._ws(after, hi)
                if s.startswith('"', q):
                    q = self._ws(s.index('"', q + 1) + 1, hi)
                if s.startswith("{", q):
                    end = self._group_end(q)
                    return Node("foreign", "", start, end, q, attrs)
                p = q
                word, after = self._word(q)
                if word not in ("fn", "unsafe", "safe"):
                    raise ParseError(f"unexpected extern item at {q}")
            else:
                break
        if word == "fn":
            name, q = self._word(self._ws(after, hi))
            body = self._to_body(q, hi)
            end = body + 1 if s[body] == ";" else self._group_end(body)
            return Node("fn", name, start, end, body, attrs)
        if word in ("struct", "union", "enum"):
            name, q = self._word(self._ws(after, hi))
            if not name or (word == "union" and name in ("", "fn")):
                raise ParseError(f"unexpected {word} at {p}")
            stop = self._to_body(q, hi, "{(;" if word != "enum" else "{")
            if s[stop] == "(":
                end = self._to_semicolon(self._group_end(stop), hi)
            elif s[stop] == "{":
                end = self._group_end(stop)
            else:
                end = stop + 1
            return Node(word, name, start, end, end, attrs)
        if word == "trait":
            name, q = self._word(self._ws(after, hi))
            body = self._to_body(q, hi)
            if s[body] == ";":
                return Node("trait", name, start, body + 1, body + 1, attrs)
            node = Node("trait", name, start, self._group_end(body), body, attrs)
            self._items(body + 1, node.end - 1, node)
            return node
        if word == "impl":
            body = self._to_body(after, hi, "{")
            node = Node("impl", self._norm(after, body), start, self._group_end(body), body, attrs)
            self._items(body + 1, node.end - 1, node)
            return node
        if word == "mod":
            name, q = self._word(self._ws(after, hi))
            q = self._ws(q, hi)
            if s.startswith(";", q):
                return Node("mod", name, start, q + 1, q + 1, attrs)
            if not s.startswith("{", q):
                raise ParseError(f"unexpected module at {q}")
            node = Node("mod", name, start, self._group_end(q), q, attrs)
            self._items(q + 1, node.end - 1, node)
            return node
        if word == "use":
            end = self._to_semicolon(after, hi)
            return Node("use", self._norm(after, end), start, end, end, attrs)
        if word in ("const", "static"):
            q = self._ws(after, hi)
            name, r = self._word(q)
            if name == "mut":
                name, r = self._word(self._ws(r, hi))
            if not name and s.startswith("_", q):
                name = "_"
            end = self._to_semicolon(r, hi)
            return Node(word, name, start, end, end, attrs)
        if word == "type":
            name, q = self._word(self._ws(after, hi))
            end = self._to_semicolon(q, hi)
            return Node("type", name, start, end, end, attrs)
        if word == "extern":
            _, q = self._word(self._ws(after, hi))  # crate
            end = self._to_semicolon(q, hi)
            tail = self._norm(q, end - 1).split()
            return Node("extern_crate", tail[-1], start, end, end, attrs)
        if word == "macro_rules":
            q = self._ws(after, hi)
            if not s.startswith("!", q):
                raise ParseError(f"expected macro_rules! at {q}")
            name, r = self._word(self._ws(q + 1, hi))
            r = self._ws(r, hi)
            end = self._group_end(r)
            if s[r] in "([":
                end = self._ws(end, hi)
                if not s.startswith(";", end):
                    raise ParseError(f"expected ';' after macro_rules! at {end}")
                end += 1
            return Node("macro_rules", name, start, end, end, attrs)
        if word:
            # An item-position macro invocation: `path!` [ident] group [;].
            path = [word]
            q = self._ws(after, hi)
            while s.startswith("::", q):
                part, q = self._word(self._ws(q + 2, hi))
                path.append(part)
                q = self._ws(q, hi)
            if s.startswith("!", q):
                r = self._ws(q + 1, hi)
                _, r2 = self._word(r)
                r = self._ws(r2, hi)
                end = self._group_end(r)
                if s[r] in "([":
                    after_group = self._ws(end, hi)
                    if s.startswith(";", after_group):
                        end = after_group + 1
                return Node("macro", "::".join(path), start, end, end, attrs)
        raise ParseError(f"unrecognized item at {p}: {s[p:p + 30]!r}")

    def _name(self, node: Node) -> None:
        seen: dict[str, int] = {}
        for child in node.children:
            base = f"{child.kind}:{child.name}"
            ordinal = seen.get(base, 0)
            seen[base] = ordinal + 1
            child.locator = f"{node.locator}/{base}#{ordinal}" if node.locator else f"{base}#{ordinal}"
            self.by_locator[child.locator] = child
            self._name(child)

    # ------------------------------------------------------------------ hashing

    def _norm(self, a: int, b: int) -> str:
        """Tokens of [a, b): literals exactly, whitespace only where it separates words."""
        import bisect
        out = []
        p = a
        i = bisect.bisect_left(self.literals, (a, -1))
        if i > 0 and self.literals[i - 1][1] > a:
            i -= 1
        while p < b:
            lit = self.literals[i] if i < len(self.literals) else None
            stop = min(lit[0], b) if lit and lit[0] < b else b
            if lit and lit[0] <= p:
                stop = min(lit[1], b)
                out.append(self.kept[p:stop])
                i += 1
            else:
                piece = re.sub(r"\s+", " ", self.kept[p:stop])
                out.append(piece)
            p = max(stop, p + 1) if stop == p else stop
        text = "".join(out)
        return re.sub(r"(?<![\w'\"#]) | (?![\w'\"#])", "", text).strip()

    def _attr_paths(self, node: Node) -> list[str]:
        return [self._norm(a, b) for a, b in node.attrs]

    def reads_docs(self, node: Node) -> bool:
        """Whether a procedural macro may read this item's documentation."""
        current = node
        while current is not None:
            for attr in self._attr_paths(current):
                m = re.match(r"#\[(?:unsafe\()?([A-Za-z_][A-Za-z0-9_]*)", attr)
                head = m.group(1) if m else ""
                if head == "derive":
                    names = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:,|\))", attr)
                    if any(n not in _INERT_DERIVES for n in names):
                        return True
                elif head == "cfg_attr":
                    if re.search(r"derive\s*\(", attr) or not all(
                            w in _INERT_ATTRIBUTES or w in ("all", "any", "not", "test", "feature", "target_os")
                            for w in re.findall(r"[,(]\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?=[(,=)\]])", attr)):
                        return True
                elif head not in _INERT_ATTRIBUTES:
                    return True
            current = current.parent
        return False

    def _docs(self, a: int, b: int) -> str:
        return "\n".join(self.text[x:y].strip() for x, y in self.docs if a <= x < b)

    def part(self, node: Node, part: str) -> str:
        if part == "full":
            text = self._norm(node.start, node.end)
            if self.reads_docs(node):
                text += "\n" + self._docs(node.start, node.end)
            return text
        if part in ("sig", "head"):
            return self._norm(node.start, node.head_end)
        if part == "inner":
            return "\n".join(self._norm(a, b) for a, b in node.inner)
        raise ValueError(part)

    def digest(self, locator: str, part: str) -> str | None:
        node = self.root if locator == "" else self.by_locator.get(locator)
        if node is None:
            return None
        return hashlib.sha256(self.part(node, part).encode()).hexdigest()

    # ------------------------------------------------------------------ positions

    def innermost(self, offset: int) -> Node:
        """The deepest item containing a character offset (the root when none does)."""
        node = self.root
        while True:
            for child in node.children:
                if child.start <= offset < child.end:
                    node = child
                    break
            else:
                return node

    def offset(self, line: int, column: int) -> int:
        """Character offset of a 1-based line and 1-based UTF-8 byte column."""
        if self._starts is None:
            starts = [0]
            for m in re.finditer("\n", self.text):
                starts.append(m.end())
            self._starts = starts
        start = self._starts[line - 1]
        end = self._starts[line] if line < len(self._starts) else len(self.text)
        raw = self.text[start:end].encode()
        return start + len(raw[: max(column - 1, 0)].decode("utf-8", "ignore"))

    def char_offset(self, byte_offset: int) -> int:
        """Character offset of a UTF-8 byte offset (compiler spans are in bytes)."""
        if self._ascii is None:
            self._ascii = self.text.isascii()
        if self._ascii:
            return byte_offset
        if self._bytes is None:
            self._bytes = self.text.encode()
        return len(self._bytes[:byte_offset].decode("utf-8", "ignore"))

    def module(self, chain: list[str]) -> Node | None:
        node = self.root
        for name in chain:
            found = [c for c in node.children if c.kind == "mod" and c.name == name and c.head_end < c.end]
            if len(found) != 1:
                return None
            node = found[0]
        return node


# ---------------------------------------------------------------------- module scope

_USE_TOKEN = re.compile(r"::|\*|\{|\}|,|(?:r#)?[A-Za-z_][A-Za-z0-9_]*")


def use_bindings(text: str) -> tuple[set[str], bool]:
    """Names a `use` declaration binds, and whether it has a glob."""
    tokens = _USE_TOKEN.findall(text)
    names: set[str] = set()
    glob = False

    def tree(i: int, prefix: list[str]) -> int:
        nonlocal glob
        path = list(prefix)
        while i < len(tokens) and tokens[i] not in ("::", "*", "{", "}", ","):
            if tokens[i] == "as":
                alias = tokens[i + 1] if i + 1 < len(tokens) else "_"
                if alias != "_":
                    names.add(alias)
                return i + 2
            path.append(tokens[i])
            i += 1
            if i < len(tokens) and tokens[i] == "::":
                i += 1
            else:
                last = path[-1] if path else ""
                if last == "self" and len(path) >= 2:
                    last = path[-2]
                if last and last not in ("self", "super", "crate"):
                    if i < len(tokens) and tokens[i] == "as":
                        continue
                    names.add(last)
                return i
        if i < len(tokens) and tokens[i] == "*":
            glob = True
            return i + 1
        if i < len(tokens) and tokens[i] == "{":
            i += 1
            while i < len(tokens) and tokens[i] != "}":
                i = tree(i, path)
                if i < len(tokens) and tokens[i] == ",":
                    i += 1
            return i + 1
        return i

    start = 1 if tokens[:1] == ["use"] else 0
    if start < len(tokens) and tokens[start] == "::":
        start += 1
    tree(start, [])
    return names, glob


def scope_entries(file: File, module: Node, names, traits: bool) -> list:
    """What decides how `names` (and, with `traits`, which traits) resolve in `module`.

    `names` is any container of identifiers.
    """
    entries = []
    for child in module.children:
        if child.kind == "use":
            bound, glob = use_bindings(file.part(child, "full"))
            if glob or traits or any(name in names for name in bound):
                entries.append(["use", file.part(child, "full")])
        elif child.kind in ("macro", "extern_crate"):
            entries.append([child.kind, file.part(child, "full")])
        elif child.name in names:
            entries.append([child.kind, child.name, child.locator])
        if traits and child.kind == "trait":
            entries.append(["trait", child.name, child.locator])
    return entries


# ---------------------------------------------------------------------- impl heads

def impl_parts(header: str) -> tuple[str | None, str, set[str]]:
    """(trait's last identifier or None, self type head, impl generic parameter names)."""
    text = header.strip()
    params: set[str] = set()
    if text.startswith("<"):
        depth = 0
        for i, c in enumerate(text):
            depth += (c == "<") - (c == ">")
            if depth == 0:
                params = set(re.findall(r"(?:^|[<,])\s*(?:const\s+)?([A-Za-z_][A-Za-z0-9_]*)", text[1:i]))
                text = text[i + 1:].strip()
                break
    text = re.split(r"\bwhere\b", text)[0].strip()
    trait = None
    depth, split = 0, None
    for m in re.finditer(r"[<>()\[\]]|\bfor\b", text):
        tok = m.group(0)
        if tok in "<([":
            depth += 1
        elif tok in ">)]":
            depth -= 1
        elif depth == 0:
            split = m
            break
    if split is not None:
        trait_text = text[:split.start()].strip().lstrip("!").strip()
        trait = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", re.split(r"<", trait_text)[0])
        trait = trait[-1] if trait else None
        text = text[split.end():].strip()
    return trait, type_head(text, params), params


def type_head(text: str, params: set[str] = frozenset()) -> str:
    text = text.strip()
    while text.startswith(("&", "mut ")) or re.match(r"'[A-Za-z_]\w*\s", text):
        text = re.sub(r"^&\s*|^mut\s+|^'[A-Za-z_]\w*\s+", "", text, count=1).strip()
    if text.startswith("dyn ") or text.startswith("dyn<"):
        words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", re.split(r"[<+]", text[4:])[0])
        return words[-1] if words else "*"
    if text.startswith("["):
        return "[;]" if ";" in text else "[]"
    if text.startswith("("):
        return "()"
    if text.startswith("*"):
        return "*ptr"
    if text.startswith(("fn", "unsafe fn", "extern")):
        return "fn"
    if text.startswith("impl "):
        return "*"
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", re.split(r"<", text)[0])
    if not words:
        return "*"
    head = words[-1]
    return "*" if head in params or head == "Self" else head


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
