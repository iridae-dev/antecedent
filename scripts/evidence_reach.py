"""Which of the functions a measurement's code names can run in it at all.

A calibration measurement runs its replicates on an uninstrumented binary and a
slice of them on an instrumented one, so its record binds the whole body of every
function the executed code names, transitively (`evidence_items.Selection`,
`callee_bodies`). Through a dispatcher (`match route { A => run_a(), B => run_b() }`)
that closure reaches every route, and a record of one route binds the functions of
all the others. This module says what the measurement cannot enter, by a rapid-type
analysis over the compiler inventory and the source text of the build:

* a `match` arm, the block of an `if let` (alone or in an `&&` chain) and the
  statements after a `let … else` run only when the pattern matched. A pattern that
  names a variant `Enum::Variant` of a workspace enum matches only a value some code
  built, and a variant is built only where its constructor is written;
* a function with a parameter (or `self`) of a workspace struct type is entered only
  with a value of that struct, and a struct is built only where a `Struct { … }`,
  `Struct(…)` or (inside its impls) `Self { … }` is written. A value of an alias
  `type A = G<P>` of a generic struct that stores its parameter in a field holds a
  value of `P`.

A name closes a region only where the compiler resolved it: the item (or the impl
around it) refers, in the inventory, to the workspace definition of that name. A
type of another crate with the same name closes nothing.

`reach` is the least fixed point of these from the roots: a variant or a struct
counts as built as soon as any code that can run writes its constructor, whatever
the order or the condition it is written under. What a function names only inside
regions that cannot run is not followed, and a named function that cannot be
entered is a dependency by its signature only.

The roots are the items the instrumented slice executed (earning) or, for a record
whose executed regions are gone, every bound function no bound item names and the
record's test (`narrow_evidence_binding`). Code the inventory cannot see enters
impls of traits the workspace does not define (`From`, `Display`, `Drop`, operators,
iterators): every such method of the crates' sources is a root too, bound or not,
entered when its parameters exist. So is every such method a macro invocation
writes, and the whole of an invocation this module cannot expand. A call the
compiler resolved to a trait's own method reaches every impl of that method.

What the analysis does not prove, and how each case is kept broad:

* It does not follow values. A variant built anywhere in code that can run opens
  every arm that matches it, in every function: two routes that one helper builds
  under a flag are both open.
* A value built without a constructor in source: an enum or struct with a derive
  other than the value-preserving ones (`VALUE_PRESERVING_DERIVES`), a `#[repr]`, a
  `#[default]` variant or any attribute this module does not know, a unit struct, a
  name defined twice (anywhere, a function body included), a definition written by a
  macro this module cannot expand (`rust_macros`), is *open*: nothing is ever closed
  on it. A second name (a type alias, an import's `as`, in a function body too)
  stands for what it renames. A function whose body holds an `impl` counts in full.
* Text it cannot attribute: the arguments of a macro it does not know, a macro's
  definition, a constant or a static count in full, and every capitalised name
  there counts as built. A variant written bare, or under a name that is no type's,
  counts as built for every enum that has it.
* Only patterns it parses close a region. A `match` whose arms it cannot split, a
  pattern with anything but paths, bindings, literals and pattern punctuation, a
  comparison (`route == Route::A`), an `if` without `let` and an `||` chain close
  nothing, and a variant a comparison names counts as built.
* A dependency whose name is written nowhere in a function (an operator, a
  coercion) is a dependency of the whole function; what a macro the function invokes
  writes counts as written where it is invoked.
* A workspace file this parser cannot read, or a file outside the inventory, may
  define or build anything: no type is then closable at all.

At earning, an executed position inside a function the analysis did not enter, or
a region it closed, withdraws the analysis for that record: the whole closure is
kept (`Selection.settle`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rust_items import File, Node, impl_parts

WORD = re.compile(r"(?:r#)?[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*\Z|[A-Z]\Z")
# Derives that only read or copy a value that already exists.
VALUE_PRESERVING_DERIVES = frozenset({
    "Debug", "Clone", "Copy", "PartialEq", "Eq", "PartialOrd", "Ord", "Hash", "Serialize",
})
# Standard macros whose arguments are ordinary expressions.
EXPRESSION_MACROS = frozenset({
    "assert", "assert_eq", "assert_ne", "debug_assert", "debug_assert_eq", "debug_assert_ne",
    "format", "format_args", "print", "println", "eprint", "eprintln", "write", "writeln",
    "panic", "unreachable", "todo", "unimplemented", "vec", "dbg", "matches", "concat", "stringify",
    "include_str", "include_bytes", "env", "option_env", "line", "column", "file", "module_path", "cfg",
})
# Item macros of the standard library and the test harness that define no type.
DEFINES_NO_TYPE = frozenset({"thread_local", "include", "compile_error", "criterion_group", "criterion_main"})
_PATTERN_OK = re.compile(r"[\w\s:(){}\[\],|&@'\"#\-]*\Z")
# An `impl … {` in expression text (not `impl Trait` in a type: that has no body).
_LOCAL_IMPL = re.compile(r"\bimpl\b[^;=(){}]*\{")
_PATH = re.compile(r"((?:(?:r#)?[A-Za-z_]\w*\s*::\s*)+)((?:r#)?[A-Za-z_]\w*)")
_KEYWORDS = frozenset({
    "as", "break", "const", "continue", "crate", "else", "enum", "extern", "false", "fn", "for", "if",
    "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref", "return", "self", "static",
    "struct", "super", "trait", "true", "type", "unsafe", "use", "where", "while", "async", "await", "dyn",
})


@dataclass
class Region:
    """A stretch of one item's text that runs together. `gate` lists the alternatives of
    what guards it: each a set of `(enum, variant)` (or `("type", struct)`) that must all
    be built."""
    parent: int | None
    gate: tuple[frozenset, ...] | None = None
    idents: set[str] = field(default_factory=set)
    variants: set[tuple[str, str]] = field(default_factory=set)
    builds: set[str] = field(default_factory=set)
    macros: set[str] = field(default_factory=set)
    span: tuple[int, int] | None = None  # where it is written, when it is in the item's own file
    # For a function a macro invocation expands to: that it is one, and the trait of the
    # impl it is a method of (`None` for an inherent method or a free function).
    written: bool = False
    trait: str | None = None


class Body:
    """The regions of one item and what each names, builds and invokes."""

    def __init__(self, file: File, node: Node, self_type: str | None, *, expansion: File | None = None):
        self.s = file.skel
        self.close = file.close
        self.self_type = self_type
        self.regions: list[Region] = [Region(None)]
        self._holes = sorted((c.start, c.end) for c in node.children) if node.kind == "fn" else []
        self.gate_types: list[str] = []
        if node.kind == "fn":
            self._text(node.start, node.head_end, 0, mentions=False)
            if node.head_end < node.end and _LOCAL_IMPL.search(self.s, node.head_end, node.end):
                # An impl written inside the body: its `Self` is not the function's, and
                # this module does not read it. Everything the body names counts.
                self.gate_types = self._parameters(node)
                self._text(node.head_end, node.end, 0, loose=True)
            elif node.head_end < node.end:
                self.gate_types = self._parameters(node)
                self._block(node.head_end + 1, node.end - 1, 0)
        elif expansion is not None:
            # What the invocation is written with stays visible; what it builds is read
            # from the items it expands to, each function gated by its own parameters.
            self._text(node.start, node.end, 0, mentions=False)
            for inner in _walk(expansion.root):
                if inner.kind in ("const", "static", "macro") or (inner.kind == "fn" and inner.head_end < inner.end):
                    self._graft(Body(expansion, inner, globals()["self_type"](inner)),
                                impl_trait(inner) if inner.kind == "fn" else None)
        else:
            self._text(node.start, node.end, 0, loose=True)

    def region_at(self, offset: int) -> int:
        """The innermost region written around a character offset of the item's file."""
        found = 0
        for index, region in enumerate(self.regions):
            if region.span is not None and region.span[0] <= offset < region.span[1]:
                found = index  # regions are listed outermost first
        return found

    def _graft(self, other: Body, trait: str | None) -> None:
        base = len(self.regions)
        for index, region in enumerate(other.regions):
            if index == 0:
                gate = (frozenset(("type", name) for name in other.gate_types),) if other.gate_types else None
                self.regions.append(Region(0, gate, region.idents, region.variants, region.builds, region.macros,
                                           written=True, trait=trait))
            else:
                self.regions.append(Region(base + region.parent, region.gate, region.idents, region.variants,
                                           region.builds, region.macros))

    # --- plain text

    def _text(self, lo: int, hi: int, region: int, *, mentions: bool = True, loose: bool = False) -> None:
        """Record what `[lo, hi)` names. `loose` (macro, constant, static text): every
        capitalised name is built and mentioned bare."""
        s, target = self.s, self.regions[region]
        for m in WORD.finditer(s, lo, hi):
            word = m.group(0)
            if any(a <= m.start() < b for a, b in self._holes):
                continue
            target.idents.add(word)
            if not mentions or not word[:1].isupper():
                continue
            before = s[max(lo, m.start() - 120):m.start()].rstrip()
            qualifier = "*"
            if before.endswith("::"):
                previous = WORD.findall(before[-80:-2].rstrip()[-60:])
                tail = before[:-2].rstrip()
                qualifier = previous[-1] if previous and tail.endswith(previous[-1]) else "*"
                if qualifier == "Self":
                    qualifier = self.self_type or "*"
            after = s[m.end():min(hi + 1, len(s))].lstrip()
            if qualifier != "*" or not after.startswith("::") or after.startswith("::<"):
                # A bare name followed by `::` is the head of a path, not a value.
                target.variants.add((qualifier, word))
            if loose or after[:1] in ("{", "(") or after.startswith("::<"):
                # `Self` outside an impl of a named type builds no named struct.
                name = self.self_type if word == "Self" else word
                if name:
                    target.builds.add(name)

    def _span(self, lo: int, hi: int, region: int) -> None:
        """Expression text `[lo, hi)`: split at the constructs that open regions."""
        s, close = self.s, self.close
        p = start = lo
        while p < hi:
            c = s[p]
            if c == "{":
                end = close.get(p, hi)
                self._text(start, p, region)
                self._block(p + 1, min(end, hi), region)
                p = start = min(end + 1, hi)
                continue
            m = WORD.match(s, p) if (c.isalpha() or c == "_") else None
            if m is None:
                p += 1
                continue
            word, q = m.group(0), m.end()
            if word == "match" and not s[lo:p].rstrip().endswith((".", "::")):
                found = self._match(q, hi, region)
                if found is not None:
                    self._text(start, p, region)
                    p = start = found
                    continue
            elif word in ("if", "while"):
                found = self._conditional(q, hi, region)
                if found is not None:
                    self._text(start, p, region)
                    p = start = found
                    continue
            elif s.startswith("!", self._ws(q, hi)) and not s.startswith("!=", self._ws(q, hi)) \
                    and word not in _KEYWORDS:
                r = self._ws(self._ws(q, hi) + 1, hi)
                if r < hi and s[r] in "([{" and r in close:
                    self._text(start, p, region)
                    self.regions[region].macros.add(word)
                    self.regions[region].idents.add(word)
                    end = min(close[r], hi)
                    if word == "matches" and self._matches(r + 1, end, region):
                        pass
                    elif word == "stringify":
                        self._text(r + 1, end, region, mentions=False)  # the tokens as text
                    elif word in EXPRESSION_MACROS:
                        self._span(r + 1, end, region)
                    else:
                        self._text(r + 1, end, region, loose=True)
                    p = start = min(end + 1, hi)
                    continue
            p = q
        self._text(start, hi, region)

    def _matches(self, lo: int, hi: int, region: int) -> bool:
        """`matches!(EXPR, PATTERN [if GUARD])`: the pattern tests a value and builds none."""
        comma = self._find(lo, hi, ",")
        if comma is None:
            return False
        guard = re.search(r"\bif\b", self.s[comma + 1:hi])
        end = comma + 1 + guard.start() if guard and self._depth_zero(comma + 1, comma + 1 + guard.start()) else hi
        pattern = self.s[comma + 1:end].rstrip().rstrip(",")
        if self._gate(comma + 1, comma + 1 + len(pattern)) is None:
            return False
        self._span(lo, comma, region)
        self._text(comma + 1, end, region, mentions=False)
        self._span(end, hi, region)
        return True

    def _block(self, lo: int, hi: int, region: int) -> None:
        """The statements of one `{ … }`: a `let … else` gates what follows it."""
        s = self.s
        p = lo
        while p < hi:
            for m in re.finditer(r"\blet\b", s[p:hi]):
                at = p + m.start()
                if self._depth_zero(p, at) and not re.search(r"\b(?:if|while)\s*\Z|&&\s*\Z|\|\|\s*\Z", s[p:at]):
                    break
            else:
                break
            found = self._let_else(at + 3, hi)
            if found is None:
                # Not a `let … else`: its text, pattern included, is ordinary text.
                end = self._statement_end(at + 3, hi)
                self._span(p, end, region)
                p = end
                continue
            pattern, else_lo, else_hi, after = found
            gate = self._gate(pattern[0], pattern[1])
            self._span(p, at, region)
            self._text(pattern[0], pattern[1], region, mentions=gate is None)
            self._span(pattern[1], else_lo, region)
            self._block(else_lo + 1, else_hi, region)
            if gate is not None:
                self.regions.append(Region(region, gate, span=(after, hi)))
                region = len(self.regions) - 1
            p = after
        self._span(p, hi, region)

    # --- constructs

    def _ws(self, p: int, hi: int) -> int:
        s = self.s
        while p < hi and s[p].isspace():
            p += 1
        return p

    def _depth_zero(self, lo: int, at: int) -> bool:
        """Whether `at` is outside every group opened in `[lo, at)`."""
        s, close = self.s, self.close
        p = lo
        while p < at:
            if s[p] in "([{":
                end = close.get(p)
                if end is None or end >= at:
                    return False
                p = end + 1
            else:
                p += 1
        return True

    def _find(self, lo: int, hi: int, stops: str, *, braces: bool = True) -> int | None:
        """The first of `stops` outside groups in `[lo, hi)`; with `braces` off a `{` is a stop."""
        s, close = self.s, self.close
        p = lo
        while p < hi:
            c = s[p]
            if c in stops:
                return p
            if c in "([" or (c == "{" and braces):
                end = close.get(p)
                if end is None:
                    return None
                p = end + 1
            elif c in ")]}":
                return None
            else:
                p += 1
        return None

    def _statement_end(self, lo: int, hi: int) -> int:
        end = self._find(lo, hi, ";")
        return hi if end is None else end + 1

    def _assign(self, lo: int, hi: int) -> int | None:
        """The `=` of a `let`, outside groups: not part of `==`, `=>`, `<=`, `>=`, `!=`, `..=`."""
        s = self.s
        p = lo
        while True:
            p = self._find(p, hi, "=;", braces=False)
            if p is None or s[p] == ";" or s[p] == "{":
                return None
            if s[p + 1:p + 2] not in ("=", ">") and s[p - 1] not in "=<>!.+-*/%&|^":
                return p
            p += 2 if s[p + 1:p + 2] in ("=", ">") else 1

    def _let_else(self, lo: int, hi: int):
        """`let PATTERN [: TYPE] = EXPR else { … };` -> (pattern span, else block, after)."""
        s = self.s
        eq = self._assign(lo, hi)
        if eq is None:
            return None
        end = self._find(eq + 1, hi, ";")
        if end is None:
            return None
        m = None
        for m in re.finditer(r"\belse\s*\{", s[eq:end]):
            pass
        if m is None:
            return None
        brace = eq + m.end() - 1
        if not self._depth_zero(eq, eq + m.start()) or self.close.get(brace) is None \
                or s[self.close[brace] + 1:end].strip():
            return None
        colon = lo
        while True:
            colon = self._find(colon, eq, ":", braces=False)
            if colon is None or (s[colon + 1:colon + 2] != ":" and s[colon - 1] != ":"):
                break
            colon += 2 if s[colon + 1:colon + 2] == ":" else 1
        return (lo, eq if colon is None else colon), brace, self.close[brace], end + 1

    def _conditional(self, lo: int, hi: int, region: int) -> int | None:
        """`if`/`while` whose condition holds `let PATTERN = EXPR`, alone or in a chain.

        A pattern tests a value and builds none. The block runs only when every pattern
        of an `&&` chain matches; a chain with `||` gates nothing."""
        s = self.s
        brace = self._find(lo, hi, "{", braces=False)
        if brace is None or s[brace] != "{" or self.close.get(brace) is None or self.close[brace] > hi:
            return None
        lets = [lo + m.start() for m in re.finditer(r"\blet\b", s[lo:brace]) if self._depth_zero(lo, lo + m.start())]
        if not lets:
            return None
        alternation = any(self._depth_zero(lo, lo + m.start()) for m in re.finditer(r"\|\|", s[lo:brace]))
        spans = []
        for at in lets:
            eq = self._assign(at + 3, brace)
            if eq is None:
                return None
            spans.append((at, eq, self._gate(at + 3, eq)))
        cursor, combined = lo, (frozenset(),)
        for at, eq, gate in spans:
            self._span(cursor, at, region)
            self._text(at + 3, eq, region, mentions=gate is None)
            cursor = eq + 1
            if combined is not None and gate is not None and len(combined) * len(gate) <= 64:
                combined = tuple(a | b for a in combined for b in gate)
            else:
                combined = None
        self._span(cursor, brace, region)
        inside = region
        if combined is not None and not alternation and any(combined):
            # Up to, not including, the closing brace: coverage reports the path that
            # skips the block as a region starting there.
            self.regions.append(Region(region, combined, span=(brace, self.close[brace])))
            inside = len(self.regions) - 1
        self._block(brace + 1, self.close[brace], inside)
        return self.close[brace] + 1

    def _match(self, lo: int, hi: int, region: int) -> int | None:
        """`match SCRUTINEE { arms }` -> the offset after it, or `None` when not split."""
        s = self.s
        brace = self._find(lo, hi, "{", braces=False)
        if brace is None or s[brace] != "{" or self.close.get(brace) is None or self.close[brace] > hi:
            return None
        end = self.close[brace]
        arms = []
        p = self._ws(brace + 1, end)
        while p < end:
            while s[p] == "#" and s[self._ws(p + 1, end)] == "[":
                p = self._ws(self.close[self._ws(p + 1, end)] + 1, end)
            arrow = p
            while True:
                arrow = self._find(arrow, end, "=")
                if arrow is None:
                    return None
                if s[arrow + 1:arrow + 2] == ">" and s[arrow - 1] not in "=<>!":
                    break
                arrow += 1
            guard = re.search(r"\bif\b", s[p:arrow])
            pattern_end = arrow
            if guard is not None and self._depth_zero(p, p + guard.start()):
                pattern_end = p + guard.start()
            body = self._ws(arrow + 2, end)
            if body < end and s[body] == "{" and self.close.get(body) is not None:
                body_end = self.close[body] + 1
                q = self._ws(body_end, end)
                nxt = q + 1 if q < end and s[q] == "," else q
            else:
                comma = self._find(body, end, ",")
                body_end = end if comma is None else comma
                nxt = end if comma is None else comma + 1
            gate = self._gate(p, pattern_end, strict=True)
            if gate is False:
                return None
            arms.append((p, pattern_end, arrow, body, body_end, gate))
            p = self._ws(nxt, end)
        if not arms:
            return None
        self._span(lo, brace, region)
        for p, pattern_end, arrow, body, body_end, gate in arms:
            self._text(p, pattern_end, region, mentions=False)
            closing = body_end - 1 if self.s[body:body + 1] == "{" else body_end
            self.regions.append(Region(region, gate or (frozenset(),), span=(pattern_end, closing)))
            child = len(self.regions) - 1
            self._span(pattern_end, arrow, child)
            self._span(body, body_end, child)
        return end + 1

    def _gate(self, lo: int, hi: int, *, strict: bool = False):
        """The alternatives of a pattern, each the `(enum, variant)` pairs it requires.

        `None` when the pattern is not one this module reads (`False` under `strict`, for
        a match arm: the split of the arms is then not trusted)."""
        text = self.s[lo:hi]
        reduced = text.replace("..=", " ").replace("..", " ")
        if not _PATTERN_OK.match(reduced) or not text.strip():
            return False if strict else None
        alternatives, depth, start = [], 0, 0
        for i, c in enumerate(text):
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
            elif depth == 0 and c == ",":
                return False if strict else None
            elif depth == 0 and c == "|":
                alternatives.append(text[start:i])
                start = i + 1
        alternatives.append(text[start:])
        return tuple(frozenset(self._required(alternative)) for alternative in alternatives)

    def _required(self, text: str) -> set[tuple[str, str]]:
        """The `(enum, variant)` every value matching the pattern `text` holds: the paths
        outside its groups, and what each element of each group requires; an alternation
        requires only what all its sides do."""
        pieces, groups, depth, start, opened = [], [], 0, 0, 0
        for i, c in enumerate(text):
            if c in "([{":
                if depth == 0:
                    opened = i
                depth += 1
            elif c in ")]}":
                depth -= 1
                if depth == 0:
                    groups.append((opened, i))
            elif depth == 0 and c == "|":
                pieces.append(text[start:i])
                start = i + 1
        if pieces:
            pieces.append(text[start:])
            sides = [self._required(piece) for piece in pieces]
            return set.intersection(*sides)
        required = set()
        flat = text
        for a, b in reversed(groups):
            flat = flat[:a + 1] + " " * (b - a - 1) + flat[b:]
        for m in _PATH.finditer(flat):
            qualifier = WORD.findall(m.group(1))[-1]
            variant = m.group(2)
            if qualifier == "Self":
                qualifier = self.self_type or ""
            if qualifier[:1].isupper() and _CAMEL.match(variant):
                required.add((qualifier, variant))
        for a, b in groups:
            element, depth, start = [], 0, a + 1
            for i in range(a + 1, b):
                c = text[i]
                depth += (c in "([{") - (c in ")]}")
                if c == "," and depth == 0:
                    element.append(text[start:i])
                    start = i + 1
            element.append(text[start:b])
            for part in element:
                required |= self._required(part)
        return required

    def _parameters(self, node: Node) -> list[str]:
        """The struct types a caller must hold a value of to enter this function."""
        s = self.s
        m = re.compile(r"\bfn\b").search(s, node.start, node.head_end)
        if m is None:
            return []
        p, angle = m.end(), 0
        while p < node.head_end and not (s[p] == "(" and angle == 0):
            angle += (s[p] == "<") - (s[p] == ">" and s[p - 1] != "-")
            p += 1
        if p >= node.head_end or self.close.get(p) is None:
            return []
        parameters, depth, start = [], 0, p + 1
        for i in range(p + 1, self.close[p]):
            c = s[i]
            depth += (c in "([{<") - (c in ")]}" or (c == ">" and s[i - 1] != "-"))
            if c == "," and depth == 0:
                parameters.append(s[start:i])
                start = i + 1
        parameters.append(s[start:self.close[p]])
        found = []
        for parameter in parameters:
            text = re.sub(r"#\s*\[[^\]]*\]", " ", parameter).strip()
            if re.fullmatch(r"(?:&\s*(?:'\w+\s+)?)?(?:mut\s+)?self", text):
                if self.self_type:
                    found.append(self.self_type)
                continue
            if ":" not in text:
                continue
            kind = text.split(":", 1)[1].strip()
            kind = re.sub(r"^(?:&\s*(?:'\w+\s+)?(?:mut\s+)?)+", "", kind).strip()
            wrapped = re.fullmatch(r"(?:\w+\s*::\s*)*(?:Box|Arc|Rc)\s*<\s*([\w:\s]+?)\s*>", kind)
            if wrapped:
                kind = wrapped.group(1)
            kind = re.sub(r"\s*<[^<>]*>\s*\Z", "", kind)  # `Name<'a, T>` is still a `Name`
            if re.fullmatch(r"(?:\w+\s*::\s*)*\w+", kind):
                name = WORD.findall(kind)[-1]
                found.append(self.self_type if name == "Self" and self.self_type else name)
        return [name for name in found if name]


# ---------------------------------------------------------------------- definitions

@dataclass
class Types:
    """The workspace enums and structs a region may be closed on."""
    enums: dict[str, set[str]] = field(default_factory=dict)  # closable enum -> its variants
    structs: set[str] = field(default_factory=set)            # closable structs
    nominal: set[str] = field(default_factory=set)            # names of defined types, never a second name
    renames: dict[str, set[str]] = field(default_factory=dict)  # a second name -> what it may name
    tuples: set[str] = field(default_factory=set)             # closable structs built by `Name(…)`
    # Where each closable type is defined: the item the compiler inventory names for it (the
    # definition, or the macro invocation that writes it).
    home: dict[str, object] = field(default_factory=dict)
    # `type A = G<P, …>` of a generic struct that stores its parameter in a field: a value
    # of `A` holds a value of the closable struct `P`. The alias, and where it is defined.
    contains: dict[str, set[str]] = field(default_factory=dict)
    alias_home: dict[str, object] = field(default_factory=dict)

    @classmethod
    def read(cls, files, expanded=lambda file, node: None) -> Types:
        """From every parsed workspace file (`{path: file}`, or a list). A name defined twice,
        aliased, renamed by an import, or defined where this parser cannot see its
        attributes is not closable. `expanded(file, node)` is the items of a macro
        invocation this module can expand."""
        defined: dict[str, list[tuple[str, File, Node, object]]] = {}
        aliased: set[str] = set()
        macro_text: dict[str, list[str]] = {}
        renames: dict[str, set[str]] = {}
        unexpanded: list[tuple[str, str]] = []
        expandable: set[str] = set()
        work = [(file, rel, None) for rel, file in (files.items() if isinstance(files, dict) else
                                                    ((None, file) for file in files))]
        written: dict[str, int] = {}
        while work:
            file, rel, owner = work.pop()
            # Read from the whole text, so that a declaration inside a function body (which
            # is no item of the file) counts: every definition of a type name, every
            # `type` alias and every `as` that gives a name a second one.
            text = file.skel
            for node in _walk(file.root):
                # What a macro writes is read from its expansion, or opened below.
                if node.kind in ("macro_rules", "macro"):
                    text = text[:node.start] + " " * (node.end - node.start) + text[node.end:]
            for name in re.findall(r"\b(?:struct|enum|union|trait|type)\s+((?:r#)?[A-Za-z_]\w*)", text):
                written[name] = written.get(name, 0) + 1
            for declaration in re.finditer(r"\buse\b[^;]*;", text):
                for m in re.finditer(r"\b(\w+)\s+as\s+(\w+)", declaration.group(0)):
                    renames.setdefault(m.group(2), set()).add(m.group(1))
            for m in re.finditer(r"\btype\s+(\w+)\s*(?:<[^=;{}]*>)?\s*(=[^;]*;)", text):
                # `type A = B<…>` gives B a second name. A reference, pointer, tuple, slice,
                # function or trait-object type names no constructor; any other shape may
                # name anything it mentions.
                target = re.fullmatch(r"=\s*((?:\w+\s*::\s*)*\w+)\s*(?:<.*>)?\s*;", m.group(2), re.S)
                if target and WORD.findall(target.group(1))[-1] not in ("fn", "dyn", "impl", "unsafe", "extern"):
                    renames.setdefault(m.group(1), set()).add(WORD.findall(target.group(1))[-1])
                elif re.match(r"=\s*(?:[&*(\[]|(?:unsafe\s+)?(?:extern\s+)?fn\b|dyn\b|impl\b)", m.group(2)):
                    aliased.add(m.group(1))
                else:
                    aliased.update(WORD.findall(m.group(0)))
            for node in _walk(file.root):
                if node.kind == "macro":
                    expansion = expanded(file, node)
                    if expansion is not None:
                        expandable.add(node.name.rsplit("::", 1)[-1])
                        work.append((expansion, rel, owner or (rel, node.locator)))
                        continue
                if node.kind in ("struct", "enum", "union", "trait", "type"):
                    home = owner or (rel, node.locator)
                    defined.setdefault(node.name, []).append((node.kind, file, node, home if rel else None))
                if node.kind == "macro_rules":
                    macro_text.setdefault(node.name, []).append(file.skel[node.start:node.end])
                elif node.kind == "macro":
                    unexpanded.append((node.name.rsplit("::", 1)[-1], file.skel[node.start:node.end]))
        # A definition a macro writes is open, and so is every name handed to a macro that
        # names a definition after its arguments (or that this workspace does not define).
        for name, texts in macro_text.items():
            if name in unexpanded_names(unexpanded) or name not in expandable:
                for text in texts:
                    aliased.update(re.findall(r"\b(?:struct|enum|union|type)\s+(\w+)", text))
        for name, text in unexpanded:
            texts = macro_text.get(name)
            if texts is None and name in DEFINES_NO_TYPE:
                continue
            if texts is None or any(re.search(r"\b(?:struct|enum|union|type)\s+\$", t) for t in texts):
                aliased.update(w for w in WORD.findall(text) if w[:1].isupper())
        types = cls()
        types.renames = renames
        # A name some item defines as a type, and that is no second name, is never another
        # type's: a path written under it names its own variants only.
        types.nominal = {name for name, definitions in defined.items()
                         if name not in renames and any(kind != "type" for kind, _, _, _ in definitions)}
        for name, definitions in defined.items():
            # A second name is what it renames, never a type of its own; a name written as a
            # definition a second time (inside a function body) is two types.
            if len(definitions) != 1 or name in aliased or name in renames or written.get(name) != 1:
                continue
            kind, file, node, home = definitions[0]
            if kind not in ("struct", "enum") or not _closed(file, node):
                continue
            types.home[name] = home
            text = file.skel[node.start:node.end]
            brace = text.find("{")
            if kind == "struct":
                # A unit struct's name is its constructor; a tuple or record struct needs a literal.
                paren = text.find("(")
                if brace != -1 and text[brace + 1:text.rfind("}")].strip():
                    types.structs.add(name)
                elif brace == -1 and paren != -1 and text[paren + 1:text.rfind(")")].strip():
                    # Its name alone is a constructor (`.map(Name)`).
                    types.structs.add(name)
                    types.tuples.add(name)
            elif brace != -1:
                variants = _variants(file, node.start + brace)
                if variants is not None:
                    types.enums[name] = variants
        for name, definitions in defined.items():
            if len(definitions) != 1 or definitions[0][0] != "type" or written.get(name) != 1 or name in aliased:
                continue
            _, file, node, home = definitions[0]
            alias = re.search(r"=\s*((?:\w+\s*::\s*)*\w+)\s*<(.*)>\s*;\s*\Z", file.skel[node.start:node.end], re.S)
            generic = defined.get(WORD.findall(alias.group(1))[-1]) if alias else None
            if home is None or not generic or len(generic) != 1 or generic[0][0] != "struct" \
                    or written.get(generic[0][2].name) != 1:
                continue
            stored = _stored_parameters(generic[0][1], generic[0][2])
            arguments = [a for a in _split(alias.group(2)) if not a.strip().startswith("'")]
            held = {WORD.findall(argument)[-1] for argument, kept in zip(arguments, stored)
                    if kept and re.fullmatch(r"\s*(?:\w+\s*::\s*)*\w+\s*", argument)} & types.structs
            if held:
                types.contains[name], types.alias_home[name] = held, home
        return types


def _split(text: str) -> list[str]:
    """`text` at its commas outside every bracket and generic argument list."""
    parts, depth, start = [], 0, 0
    for i, c in enumerate(text):
        if c in "([{<":
            depth += 1
        elif c in ")]}" or (c == ">" and text[i - 1:i] != "-"):
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [part for part in parts if part.strip()]


def _stored_parameters(file: File, node: Node) -> list[bool]:
    """For each type or const parameter of a struct, in order: whether a field has exactly
    that parameter as its type (so every value of the struct holds a value of it)."""
    text = file.skel[node.start:node.end]
    head = re.search(r"\bstruct\s+(?:r#)?\w+\s*<", text)
    if head is None:
        return []
    depth, end = 1, head.end()
    while end < len(text) and depth:
        depth += (text[end] == "<") - (text[end] == ">" and text[end - 1] != "-")
        end += 1
    parameters = [p.strip() for p in _split(text[head.end():end - 1]) if not p.strip().startswith("'")]
    names = [None if p.startswith("const ") else WORD.match(p).group(0) for p in parameters if WORD.match(p)]
    body = text[end:]
    opener = min((i for i in (body.find("{"), body.find("(")) if i != -1), default=-1)
    if opener == -1 or len(names) != len(parameters):
        return [False] * len(parameters)
    closer = body.rfind("}" if body[opener] == "{" else ")")
    stored = set()
    for element in _split(body[opener + 1:closer]):
        element = re.sub(r"#\s*\[[^\]]*\]", " ", element)
        kind = element.split(":", 1)[1] if body[opener] == "{" and ":" in element else re.sub(
            r"^\s*pub\s*(?:\([^)]*\))?", "", element)
        stored.add(kind.strip())
    return [name is not None and name in stored for name in names]


def unexpanded_names(unexpanded: list[tuple[str, str]]) -> set[str]:
    return {name for name, _ in unexpanded}


def _closed(file: File, node: Node) -> bool:
    """Whether only source constructors build a value of this type."""
    for a, b in node.attrs:
        text = file.skel[a:b]
        head = re.match(r"#\s*\[\s*(\w+)", text)
        name = head.group(1) if head else ""
        if name == "derive":
            if not set(re.findall(r"(\w+)\s*(?:,|\))", text.replace("::", " "))) <= VALUE_PRESERVING_DERIVES:
                return False
        elif name not in ("doc", "allow", "expect", "warn", "deny", "must_use", "non_exhaustive", "cfg_attr",
                          "deprecated"):
            return False
        if name == "cfg_attr" and "derive" in text:
            return False
    return "#[default]" not in re.sub(r"\s+", "", file.skel[node.start:node.end])


def _variants(file: File, brace: int) -> set[str] | None:
    s, close = file.skel, file.close
    end = close.get(brace)
    if end is None:
        return None
    variants, p = set(), brace + 1
    while p < end:
        while p < end and (s[p].isspace() or s[p] == ","):
            p += 1
        while p < end and s[p] == "#":
            q = s.find("[", p)
            if q == -1 or close.get(q) is None:
                return None
            p = close[q] + 1
            while p < end and s[p].isspace():
                p += 1
        if p >= end:
            break
        m = WORD.match(s, p)
        if m is None:
            return None
        variants.add(m.group(0))
        p = m.end()
        while p < end and s[p] != ",":
            if s[p] in "([{":
                if close.get(p) is None:
                    return None
                p = close[p]
            p += 1
    return variants


def _walk(node: Node):
    for child in node.children:
        yield child
        yield from _walk(child)


def self_type(node: Node) -> str | None:
    """The head of the type whose impl holds `node`, when it names one type."""
    current = node.parent
    while current is not None:
        if current.kind == "impl":
            head = impl_parts(current.name)[1]
            return head if head[:1].isupper() else None
        current = current.parent
    return None


def impl_trait(node: Node) -> str | None:
    """The trait of the impl that directly holds `node`."""
    parent = node.parent
    return impl_parts(parent.name)[0] if parent is not None and parent.kind == "impl" else None


# ---------------------------------------------------------------------- reachability

Key = tuple[str, str]  # (file, locator)


class Graph:
    """The items of one build, what each names (compiler inventory), and their text."""

    def __init__(self, files, inventory):
        """`files(rel)` parses a workspace file of the build's source (or `None`)."""
        self.files = files
        self.inventory = inventory
        self._key: dict[str, Key | None] = {}
        self._edges: dict[Key, list[tuple[str, Key]]] = {}
        self._bodies: dict[Key, Body] = {}
        self._types: Types | None = None
        self._impls: dict[tuple[str, str], list[Key]] | None = None
        self._macros: dict[str, list[Key]] | None = None
        self._by_node: dict[str, dict[str, list[dict]]] = {}
        self._expansions: dict[Key, File | None] = {}
        self._nodes: dict[Key, Node | None] = {}
        self._names: dict[Key, frozenset[str]] = {}
        self._seconds: dict[str, set[str]] | None = None
        self._macro_idents: dict[str, frozenset[str]] = {}
        self._traits: set[str] | None = None

    def key(self, inventory_key: str) -> Key | None:
        if inventory_key not in self._key:
            rel, offset = inventory_key.rsplit(":", 1)
            file = self.files(rel)
            node = file.innermost(file.char_offset(int(offset))) if file is not None else None
            self._key[inventory_key] = (rel, node.locator) if node is not None and node is not file.root else None
        return self._key[inventory_key]

    def node(self, key: Key) -> Node | None:
        if key not in self._nodes:
            file = self.files(key[0])
            self._nodes[key] = file.by_locator.get(key[1]) if file is not None else None
        return self._nodes[key]

    def names(self, key: Key) -> frozenset[str]:
        """The names a reference to the item is written with, its second names included."""
        if key not in self._names:
            node = self.node(key)
            if self._seconds is None:
                self._seconds = {}
                for second, firsts in self.types().renames.items():
                    for first in firsts:
                        self._seconds.setdefault(first, set()).add(second)
            names = target_names(node) if node is not None else set()
            self._names[key] = frozenset(names | {s for n in names for s in self._seconds.get(n, ())})
        return self._names[key]

    def items(self, rel: str) -> dict[str, list[dict]]:
        if rel not in self._by_node:
            index: dict[str, list[dict]] = {}
            for inventory_key in self.inventory.by_file.get(rel, []):
                key = self.key(inventory_key)
                if key is not None:
                    index.setdefault(key[1], []).append(self.inventory.items[inventory_key])
            self._by_node[rel] = index
        return self._by_node[rel]

    def edges(self, key: Key) -> list[tuple[str, Key]]:
        """`(relation, item)` for everything the item names, from its signature and body."""
        if key not in self._edges:
            found = set()
            for item in self.items(key[0]).get(key[1], []):
                for relation, target in item["deps"] | item["bdeps"]:
                    resolved = self.key(target)
                    if resolved is not None and resolved != key:
                        found.add((relation, resolved))
            self._edges[key] = sorted(found)
        return self._edges[key]

    def body(self, key: Key) -> Body | None:
        if key not in self._bodies:
            file, node = self.files(key[0]), self.node(key)
            self._bodies[key] = None if node is None else Body(
                file, node, self_type(node), expansion=self.expansion(key) if node.kind == "macro" else None)
        return self._bodies[key]

    def macro_idents(self, name: str) -> frozenset[str]:
        """Every name the definitions of a workspace macro write, and those of the macros
        they invoke: what an invocation may name without the caller writing it."""
        if name not in self._macro_idents:
            self._macro_idents[name] = frozenset()  # a macro that invokes itself
            found: set[str] = set()
            for key in self.macros().get(name, ()):
                body = self.body(key)
                for region in body.regions if body is not None else ():
                    found |= region.idents
                    for inner in region.macros:
                        found |= self.macro_idents(inner)
            self._macro_idents[name] = frozenset(found)
        return self._macro_idents[name]

    def expansion(self, key: Key) -> File | None:
        """The items an item-position macro invocation expands to, when this module can
        expand it (`rust_macros`); `None` keeps its text whole."""
        if key not in self._expansions:
            self._expansions[key] = self._expand(key)
        return self._expansions[key]

    def _expand(self, key: Key) -> File | None:
        import rust_macros
        from rust_items import ParseError
        file, node = self.files(key[0]), self.node(key)
        definitions = {self.files(rel).kept[self.node((rel, locator)).start:self.node((rel, locator)).end]
                       for rel, locator in self.macros().get(node.name.rsplit("::", 1)[-1], ())}
        if len(definitions) != 1:
            return None  # not a workspace macro, or one name with two meanings
        text = rust_macros.expand(definitions.pop(), file.kept[node.start:node.end])
        if text is None:
            return None
        try:
            return File(text)
        except ParseError:
            return None

    def types(self) -> Types:
        if self._types is None:
            rels = set(self.inventory.by_file) | {m["file"] for m in self.inventory.modules.values()}
            parsed = {rel: self.files(rel) for rel in sorted(rels)}
            if any(file is None for file in parsed.values()):
                # A file this parser cannot read may define or build anything.
                self._types = Types()
            else:
                by_file = {id(file): rel for rel, file in parsed.items()}
                self._types = Types.read(
                    parsed,
                    lambda file, node: self.expansion((by_file[id(file)], node.locator)) if id(file) in by_file else None)
        return self._types

    def impls(self) -> dict[tuple[str, str], list[Key]]:
        """(trait, method) -> the impl functions of that trait's method, in every file; for an
        impl a macro writes, the invocation."""
        if self._impls is None:
            self._impls = {}
            for rel in self.inventory.by_file:
                file = self.files(rel)
                if file is None:
                    continue
                for node in _walk(file.root):
                    if node.kind == "fn" and impl_trait(node):
                        self._impls.setdefault((impl_trait(node), node.name), []).append((rel, node.locator))
                    elif node.kind == "macro":
                        expansion = self.expansion((rel, node.locator))
                        for inner in _walk(expansion.root) if expansion is not None else ():
                            if inner.kind == "fn" and impl_trait(inner):
                                self._impls.setdefault((impl_trait(inner), inner.name), []).append((rel, node.locator))
        return self._impls

    def macros(self) -> dict[str, list[Key]]:
        if self._macros is None:
            self._macros = {}
            for rel in sorted(set(self.inventory.by_file) | {m["file"] for m in self.inventory.modules.values()}):
                file = self.files(rel)
                for node in _walk(file.root) if file is not None else ():
                    if node.kind == "macro_rules":
                        self._macros.setdefault(node.name, []).append((rel, node.locator))
        return self._macros


def target_names(node: Node) -> set[str]:
    """The name a reference to this item is written with (an impl has none)."""
    if node.kind == "impl":
        return set()
    name = node.name.rsplit("::", 1)[-1].removeprefix("r#") if node.name else ""
    return {name} if name else set()


@dataclass
class Reach:
    """What can run: the functions entered, what each item names only where it cannot run,
    and the regions closed in each item."""
    entered: set[Key]
    named: set[Key]
    dead: dict[Key, set[Key]]
    closed: dict[Key, set[int]]
    built: set[str]
    variants: set[tuple[str, str]]

    def enters(self, key: Key) -> bool:
        return key in self.entered

    def follows(self, source: Key, target: Key) -> bool:
        return target not in self.dead.get(source, ())


_SCANNED = ("fn", "const", "static", "macro", "macro_rules")


def reach(graph: Graph, roots: set[Key], *, executed: set[Key] = frozenset(), unseen: set[Key] = frozenset(),
          trace: dict | None = None) -> Reach:
    """Least fixed point of what can run from `roots`, over the whole build.

    `executed` are functions known to have run: entered whatever their parameters.
    `unseen` are macro invocations no code is known to name (`unseen_macros`): of what
    one expands to, only the methods of impls of traits the workspace does not define can
    run, until some code that runs names the invocation. `trace` receives, for each fact,
    the item that first established it."""
    def note(fact: tuple, key: Key) -> None:
        if trace is not None:
            trace.setdefault(fact, key)

    types = graph.types()
    impls = graph.impls()
    macros = graph.macros()
    named: set[Key] = set()
    entered: set[Key] = set()
    built: set[str] = set()
    variants: set[tuple[str, str]] = set()
    names_built: set[str] = set()      # variant names written bare or under a name that is no type's
    dead: dict[Key, set[Key]] = {}
    closed: dict[Key, set[int]] = {}
    waiting: set[Key] = set()          # items a new fact could change
    work: list[Key] = []
    facts = [0]

    traits = workspace_traits(graph)
    soft = set(unseen) - set(roots)    # invocations nothing that runs has named yet

    def name(target: Key, source: Key) -> None:
        if target in soft and source != target:
            soft.discard(target)
            work.append(target)
        if target not in named:
            named.add(target)
            note(("named", target), source)
            work.append(target)

    def constructible(requirement: tuple[str, str]) -> bool:
        enum, variant = requirement
        if enum == "type":
            return variant not in types.structs or variant in built or (
                variant in types.tuples and variant in names_built)
        if enum not in types.enums or variant not in types.enums[enum]:
            return True
        return requirement in variants or variant in names_built

    def visit(key: Key) -> None:
        node = graph.node(key)
        if node is None or node.kind in ("use", "mod", "extern_crate"):
            return  # a declaration: what it imports runs only where it is called
        body = graph.body(key) if node.kind in _SCANNED else None
        edges = graph.edges(key)
        # A name closes a region only where the compiler resolved it to the workspace
        # type of that name: the item (or the impl around it) refers to the definition.
        referred = {target for _, target in edges} | {key}
        if node.parent is not None and node.parent.kind == "impl":
            referred.update(target for _, target in graph.edges((key[0], node.parent.locator)))

        def holds(requirement: tuple[str, str]) -> bool:
            name = requirement[1] if requirement[0] == "type" else requirement[0]
            if requirement[0] == "type" and types.alias_home.get(name) in referred:
                # A value of the alias holds a value of each struct it stores.
                return all(constructible(("type", held)) for held in types.contains[name])
            return types.home.get(name) not in referred or constructible(requirement)

        if node.kind == "fn":
            if key not in executed and not all(holds(("type", t)) for t in body.gate_types):
                waiting.add(key)
                return
            entered.add(key)
        live_idents: set[str] = set()
        dead_idents: set[str] = set()
        regions = body.regions if body is not None else []
        live = [False] * len(regions)
        shut: set[int] = set()
        softly = key in soft
        for index, region in enumerate(regions):
            live[index] = (region.parent is None or live[region.parent]) and (
                region.gate is None or any(all(map(holds, alt)) for alt in region.gate)) and not (
                softly and region.written and (region.trait is None or region.trait in traits))
            if not live[index]:
                shut.add(index)
                dead_idents |= region.idents
                continue
            live_idents |= region.idents
            for qualifier, variant in region.variants:
                # A second name (an import's `as`, a type alias) stands for what it renames.
                for first in types.renames.get(qualifier, (qualifier,)):
                    if first in types.nominal:
                        if (first, variant) not in variants:
                            variants.add((first, variant))
                            note(("variant", first, variant), key)
                            facts[0] += 1
                    elif variant not in names_built:
                        names_built.add(variant)
                        note(("bare", variant, qualifier), key)
                        facts[0] += 1
            for written in region.builds:
                for first in types.renames.get(written, (written,)):
                    if first not in built:
                        built.add(first)
                        note(("built", first), key)
                        facts[0] += 1
            for macro in region.macros:
                for target in macros.get(macro, ()):
                    name(target, key)
                # What the macro writes is written here.
                live_idents |= graph.macro_idents(macro)
        if shut:
            waiting.add(key)
            closed[key] = shut
        else:
            waiting.discard(key)
            closed.pop(key, None)
        unreached: set[Key] = set()
        for relation, target in edges:
            target_node = graph.node(target)
            if target_node is None:
                continue
            names = graph.names(target)
            if shut and names and not names & live_idents and names & dead_idents:
                unreached.add(target)
                continue
            name(target, key)
            # A call the compiler resolved to a trait's method reaches every impl of it.
            if target_node.kind == "fn" and target_node.parent is not None and target_node.parent.kind == "trait":
                for impl in impls.get((target_node.parent.name, target_node.name), ()):
                    name(impl, key)
        if unreached:
            # An impl header is named with its methods: dead when every method named is.
            for relation, target in edges:
                target_node = graph.node(target)
                if target_node is not None and target_node.kind == "impl":
                    children = {t for _, t in edges if t[0] == target[0] and t[1].startswith(target[1] + "/")}
                    if children and children <= unreached:
                        unreached.add(target)
            dead[key] = unreached
        else:
            dead.pop(key, None)

    for root in sorted(set(roots) | soft):
        name(root, root)
    while work:
        seen = facts[0]
        while work:
            visit(work.pop())
        if facts[0] != seen or waiting:
            # New facts may open a region or a function: look again at what was held back.
            before = (facts[0], len(named), len(entered), sum(map(len, closed.values())))
            for key in sorted(waiting):
                visit(key)
            while work:
                visit(work.pop())
            if before != (facts[0], len(named), len(entered), sum(map(len, closed.values()))):
                work.extend(sorted(waiting))
    return Reach(entered, named, dead, closed, built, variants | {("*", v) for v in names_built})


def workspace_traits(graph: Graph) -> set[str]:
    """Names of the traits the build's own files define."""
    if graph._traits is not None:
        return graph._traits
    found = graph._traits = set()
    for rel in sorted(set(graph.inventory.by_file) | {m["file"] for m in graph.inventory.modules.values()}):
        file = graph.files(rel)
        found.update(node.name for node in _walk(file.root) if node.kind == "trait") if file is not None else None
    return found


_OTHER_TARGETS = re.compile(r"/(?:tests|benches|examples)/")


def unseen_roots(graph: Graph, files: set[str]) -> set[Key]:
    """Items that code outside the inventory enters: the methods of impls of traits the
    workspace does not define (`From`, `Display`, `Drop`, operators) or defines by a macro,
    in the crates' sources and in `files` (the test target's own). Each is still entered
    only when its parameters exist."""
    traits = workspace_traits(graph)
    roots = set()
    for (trait, _), keys in graph.impls().items():
        if trait in traits:
            continue
        roots.update(key for key in keys
                     if graph.node(key).kind == "fn" and (key[0] in files or not _OTHER_TARGETS.search("/" + key[0])))
    # A macro invocation this module cannot expand may write any impl: all of it can run.
    return roots | {key for key in _invocations(graph, files) if graph.expansion(key) is None}


def unseen_macros(graph: Graph, files: set[str]) -> set[Key]:
    """The macro invocations of the same files that this module expands: what one writes
    may be such an impl (`reach`, `unseen`)."""
    return {key for key in _invocations(graph, files) if graph.expansion(key) is not None}


def _invocations(graph: Graph, files: set[str]) -> set[Key]:
    found = set()
    for rel in sorted(set(graph.inventory.by_file) | {m["file"] for m in graph.inventory.modules.values()}):
        file = graph.files(rel) if rel in files or not _OTHER_TARGETS.search("/" + rel) else None
        if file is not None:
            found.update((rel, node.locator) for node in _walk(file.root)
                         if node.kind == "macro" and node.name != "include")
    return found
