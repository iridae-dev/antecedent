"""Compiler-selected body holes with exact guards over all other source bytes.

This projection does not infer a call graph. The compiler supplies complete,
non-expanded, ordinary-function body spans; earning supplies execution evidence.
Only bodies proven unexecuted may be holes. Everything else stays a dependency.
The checker permits a deliberately small expression-only body language. Adding
items, attributes, macros or unsafe constructs invalidates the projection.
"""
from __future__ import annotations

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

def has_opaque_attributes(source: str) -> bool:
    skeleton = lex(source).skeleton
    builtins = {'test','cfg','allow','warn','deny','forbid','inline','cold','must_use',
                'repr','non_exhaustive','no_mangle','export_name','path','doc',
                'track_caller','deprecated','automatically_derived'}
    derives = {'Debug','Clone','Copy','Eq','PartialEq','Ord','PartialOrd','Default','Hash'}
    for match in re.finditer(r'#\s*!?\s*\[', skeleton):
        depth, end = 1, match.end()
        while end < len(skeleton) and depth:
            depth += (skeleton[end]=='[') - (skeleton[end]==']')
            end += 1
        if depth:
            return True
        attribute=skeleton[match.end():end-1].strip()
        name=re.match(r'[A-Za-z_][A-Za-z_0-9]*',attribute)
        if name is None:
            return True
        tail = attribute[name.end():].lstrip()
        if tail and not tail.startswith(('(', '=')):
            return True
        if name[0]=='derive':
            if ':' in attribute:
                return True
            arguments=set(re.findall(r'[A-Za-z_][A-Za-z_0-9]*',attribute)) - {'derive'}
            if not arguments <= derives:
                return True
        elif name[0] not in builtins:
            return True
    return False


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
    # Preserve line/column positions of subsequent source. Changed byte lengths
    # within earlier lines are harmless; later source byte offsets are not used
    # as identity. Opaque span consumers require a full-source fallback at earning.
    return [body.count("\n"), len(body.rsplit("\n", 1)[-1].encode())]


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
    for prefix, expected_layout in zip(segments, layouts):
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
                           covered_lines: set[int], *, opaque: bool = True) -> tuple[dict, list[str]]:
    """Conservative admission of compiler body spans using instrumented line coverage.

    `opaque` must be cleared by the earning adapter only after checking the whole
    compilation for procedural macros, source-reading generators and span-sensitive
    consumers. The default deliberately admits no body holes. Empty/missing
    compiler coverage is not evidence that every function was unexecuted.
    """
    if opaque:
        return capture(source, []), ['opaque compilation: complete source retained']
    if not covered_lines:
        return capture(source, []), ['no instrumented source coverage: complete source retained']
    # The supported subset permits only the compiler's built-in test attribute.
    # Parent/module attributes and doc attributes may be inputs to a procedural
    # macro; retain the entire source until the adapter resolves their scope.
    skeleton = lex(source).skeleton
    import bisect
    if has_opaque_attributes(source):
        return capture(source, []), ['unresolved attribute input: complete source retained']
    newlines = [pos for pos, char in enumerate(source) if char == '\n']
    spans, reasons = [], []
    for name, (start, end) in candidates.items():
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
