"""Compiler-selected body holes with exact guards over all other source bytes.

This projection does not infer a call graph. The compiler supplies complete,
non-expanded, ordinary-function body spans; earning supplies execution evidence.
Only bodies proven unexecuted may be holes. Everything else stays a dependency.
The checker permits a deliberately small expression-only body language. Adding
items, attributes, macros or unsafe constructs invalidates the projection.
"""
from __future__ import annotations

import functools
import hashlib
import json
import re

from _rust_source import lex

VERSION = 1
# Ordinary body expressions cannot alter module lookup. Nested impls, exported
# macros and attributes can; never allow them to hide behind a body hole.
FORBIDDEN = re.compile(r"\b(?:fn|impl|trait|struct|enum|union|mod|const|static|type|extern|unsafe|macro_rules)\b|#|\b[A-Za-z_][A-Za-z_0-9:]*\s*!")


class ProjectionChanged(ValueError):
    pass

BUILTIN_ATTRIBUTES = frozenset({
    'test', 'cfg', 'allow', 'warn', 'deny', 'forbid', 'inline', 'cold', 'must_use',
    'repr', 'non_exhaustive', 'no_mangle', 'export_name', 'path', 'doc',
    'track_caller', 'deprecated', 'automatically_derived'})
STANDARD_DERIVES = frozenset({'Debug', 'Clone', 'Copy', 'Eq', 'PartialEq', 'Ord', 'PartialOrd', 'Default', 'Hash'})
_DATA_ITEM = re.compile(
    r'\s*(?:#\s*\[[^\[\]]*\]\s*)*'
    r'(?:pub(?:\s*\([^()]*\))?\s+)?(?:struct|enum|union)\s+[A-Za-z_][A-Za-z_0-9]*\s*(?P<end>[{;(])')


def _attributes(skeleton: str) -> list[tuple[int, int, str]] | None:
    """(start, end, text) of every attribute, or None when one is malformed."""
    found = []
    for match in re.finditer(r'#\s*!?\s*\[', skeleton):
        depth, end = 1, match.end()
        while end < len(skeleton) and depth:
            depth += (skeleton[end] == '[') - (skeleton[end] == ']')
            end += 1
        if depth:
            return None
        found.append((match.start(), end, skeleton[match.end():end - 1].strip()))
    return found


def _data_item_end(skeleton: str, after: int) -> int | None:
    """End of the struct/enum/union that an attribute ending at `after` decorates."""
    item = _DATA_ITEM.match(skeleton, after)
    if item is None:
        return None
    delimiter = item.start('end')
    if skeleton[delimiter] == ';':
        return delimiter + 1
    if skeleton[delimiter] == '(':
        # Tuple struct: balanced parentheses, then optional where clause, then `;`.
        depth = 0
        for pos in range(delimiter, len(skeleton)):
            depth += (skeleton[pos] == '(') - (skeleton[pos] == ')')
            if depth == 0:
                close = skeleton.find(';', pos)
                return None if close < 0 or '{' in skeleton[pos:close] else close + 1
        return None
    try:
        return end_of_body(skeleton, delimiter)
    except ProjectionChanged:
        return None


def has_opaque_attributes(source: str, *, scoped_derives: bool = False,
                          candidate_start: int | None = None) -> bool:
    """Can an attribute macro or derive receive tokens that include the candidate body?

    Without `scoped_derives`, any custom derive or attribute makes the file opaque.
    With it (only under an audited compile-time policy whose derive macros read
    nothing beyond their input tokens), a custom derive and the helper attributes
    inside its data item are inputs to that item alone, so unrelated function
    bodies stay projectable. Their positions are preserved by the layout guard.
    """
    opaque, derived = _attribute_scopes(source, scoped_derives)
    if opaque:
        return True
    if not derived:
        return False
    # A custom derive is an input to its own data item only.
    return candidate_start is None or any(lo <= candidate_start < hi for lo, hi in derived)


@functools.lru_cache(maxsize=512)
def _attribute_scopes(source: str, scoped_derives: bool) -> tuple[bool, tuple[tuple[int, int], ...]]:
    """(any attribute opaque to every body, custom-derive item spans), once per file.

    Equivalent to scanning the attributes for each candidate: whether an attribute
    is opaque never depends on the candidate, and a custom derive makes a body
    opaque exactly when the body lies inside the derive's item."""
    skeleton = lex(source).skeleton
    attributes = _attributes(skeleton)
    if attributes is None:
        return True, ()
    derived: list[tuple[int, int]] = []
    for start, end, attribute in attributes:
        if any(lo <= start < hi for lo, hi in derived):
            continue  # a helper or further attribute inside a scoped derive input
        name = re.match(r'[A-Za-z_][A-Za-z_0-9]*', attribute)
        if name is None:
            return True, ()
        tail = attribute[name.end():].lstrip()
        if tail and not tail.startswith(('(', '=')):
            return True, ()
        if name[0] == 'derive':
            arguments = set(re.findall(r'[A-Za-z_][A-Za-z_0-9]*', attribute)) - {'derive'}
            if ':' not in attribute and arguments <= STANDARD_DERIVES:
                continue
            item_end = _data_item_end(skeleton, end) if scoped_derives else None
            if item_end is None:
                return True, ()
            derived.append((start, item_end))
        elif name[0] not in BUILTIN_ATTRIBUTES:
            return True, ()
    return False, tuple(derived)


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def end_of_body(skeleton: str, start: int) -> int:
    if start >= len(skeleton) or skeleton[start] != "{":
        raise ProjectionChanged("expected compiler-selected function body")
    depth = 0
    for pos in range(start, len(skeleton)):
        if skeleton[pos] == "{":
            depth += 1
        elif skeleton[pos] == "}":
            depth -= 1
            if depth == 0:
                return pos + 1
    raise ProjectionChanged("unbalanced function body")


def check_body(body: str) -> None:
    view = lex(body)
    if FORBIDDEN.search(view.skeleton):
        raise ProjectionChanged("body introduces a compile-time/lookup-sensitive construct")
    # Macro invocations whose names use raw identifiers or unusual token shapes
    # also need to be excluded. Keep unary boolean negation, reject every other !.
    if re.search(r"[A-Za-z_0-9:)\]}]\s*!", view.skeleton):
        raise ProjectionChanged("macro-like body construct")
    # The lexer is intentionally not a Rust parser. Literal content is allowed,
    # but source-reading macros/attributes cannot be introduced through literals.


def layout(body: str) -> list[int]:
    # Preserve the line/column position of every retained byte. Rust embeds
    # source locations in generated code far beyond explicit line!/column!:
    # every #[track_caller] call (unwrap, expect, indexing), overflow and bounds
    # panics, and any macro reading an input span. None of these is visible
    # lexically, so a hole may never shift later source. Changed byte lengths
    # within a hole's earlier lines move no retained column. Columns are
    # counted in characters by panic locations and in bytes elsewhere: keep both.
    last = body.rsplit("\n", 1)[-1]
    return [body.count("\n"), len(last.encode()), len(last)]


def capture(source: str, spans: list[tuple[int, int]]) -> dict:
    """Construct a projection from compiler body spans in Python character offsets.

    The adapter converts compiler byte offsets before calling this function.
    Selection/execution provenance is verified by the earning adapter, not guessed
    here. This function cannot by itself qualify a manifest.
    """
    view = lex(source)
    segments, layouts, cursor = [], [], 0
    for start, end in sorted(spans):
        if start < cursor or end_of_body(view.skeleton, start) != end:
            raise ProjectionChanged("overlapping or incorrect compiler body span")
        check_body(source[start:end])
        segments.append(source[cursor:start])
        layouts.append(layout(source[start:end]))
        cursor = end
    segments.append(source[cursor:])
    return {"version": VERSION, "segments": segments, "layouts": layouts}


def matches(source: str, projection: dict, skeleton: str | None = None) -> bool:
    if projection.get("version") != VERSION:
        raise ProjectionChanged("unsupported source projection")
    segments, layouts = projection.get("segments"), projection.get("layouts")
    if (not isinstance(segments, list) or not all(isinstance(s, str) for s in segments)
            or not isinstance(layouts, list) or len(segments) != len(layouts) + 1):
        raise ProjectionChanged("malformed source projection")
    skeleton = lex(source).skeleton if skeleton is None else skeleton
    cursor = 0
    for prefix, expected_layout in zip(segments, layouts, strict=False):
        if not source.startswith(prefix, cursor):
            return False
        cursor += len(prefix)
        end = end_of_body(skeleton, cursor)
        body = source[cursor:end]
        check_body(body)
        if layout(body) != expected_layout:
            return False
        cursor = end
    return source[cursor:] == segments[-1]


def from_compiler_coverage(source: str, candidates: dict[str, tuple[int, int]],
                           covered_lines: set[int], *, opaque: bool = True,
                           coverage_complete: bool = False,
                           scoped_derives: bool = False) -> tuple[dict, list[str]]:
    """Conservative admission of compiler body spans using instrumented line coverage.

    `opaque` must be cleared by the earning adapter only after checking the whole
    compilation for procedural macros, source-reading generators and span-sensitive
    consumers. The default deliberately admits no body holes. Empty/missing
    compiler coverage is not evidence that every function was unexecuted.
    """
    if opaque:
        return capture(source, []), ['opaque compilation: complete source retained']
    if not covered_lines and not coverage_complete:
        return capture(source, []), ['no instrumented source coverage: complete source retained']
    # The supported subset permits only the compiler's built-in test attribute.
    # Parent/module attributes and doc attributes may be inputs to a procedural
    # macro; retain the entire source until the adapter resolves their scope.
    skeleton = lex(source).skeleton
    import bisect
    if not scoped_derives and has_opaque_attributes(source):
        return capture(source, []), ['unresolved attribute input: complete source retained']
    newlines = [pos for pos, char in enumerate(source) if char == '\n']
    # A body written inside a function-like macro's input keeps its own spans
    # (it is not marked as an expansion) yet is that macro's input: the macro
    # may stringify, duplicate or otherwise consume it without executing it.
    invocations = []
    for invocation in re.finditer(r'[A-Za-z_][A-Za-z_0-9]*\s*!\s*(?:[A-Za-z_][A-Za-z_0-9]*\s*)?([(\[{])', skeleton):
        close = {'(': ')', '[': ']', '{': '}'}[invocation[1]]
        depth = 0
        for pos in range(invocation.start(1), len(skeleton)):
            depth += (skeleton[pos] == invocation[1]) - (skeleton[pos] == close)
            if depth == 0:
                invocations.append((invocation.start(1), pos + 1))
                break
        else:
            invocations.append((invocation.start(1), len(skeleton)))
    spans, reasons = [], []
    for name, (start, end) in candidates.items():
        if any(lo <= start < hi for lo, hi in invocations):
            reasons.append(f'{name}: inside a macro invocation; body retained')
            continue
        if scoped_derives and has_opaque_attributes(source, scoped_derives=True, candidate_start=start):
            reasons.append(f'{name}: unresolved attribute input; body retained')
            continue
        first = bisect.bisect_left(newlines, start) + 1
        last = bisect.bisect_left(newlines, end - 1) + 1
        if any(line in covered_lines for line in range(first, last + 1)):
            continue
        try:
            check_body(source[start:end])
        except ProjectionChanged as error:
            reasons.append(f'{name}: {error}; body retained')
            continue
        spans.append((start, end))
    return capture(source, spans), reasons
