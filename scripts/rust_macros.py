"""Expand an item-position `macro_rules!` invocation to the text of its items.

`evidence_reach` reads what a workspace macro defines (an enum's variants, the
constructors and impls it writes) from the expansion rather than treating the
invocation as opaque. Only what this module expands exactly is used: an invocation
none of whose rules matches, a fragment kind it does not parse, a repetition whose
bindings disagree in length, or a transcriber it cannot follow gives `None`, and the
caller keeps the invocation's text whole.

Supported: literal tokens, delimited groups, `$name:fragment` for `ident`,
`lifetime`, `literal`, `tt`, `ty`, `path`, `expr`, `pat`, `pat_param`, `block`,
`meta`, `vis` and `item`, and `$( … ) sep? (*|+|?)` repetitions, nested. Rules are
tried in order and the first that matches the whole input is transcribed, as the
compiler does. `$crate` becomes `crate`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN = re.compile(
    r"""\s+
    |(?P<lit>b?r\#*"|b?"|b?'(?:\\.|[^\\'])'|\d[\w.]*)
    |(?P<life>'[A-Za-z_]\w*)
    |(?P<word>(?:r\#)?[A-Za-z_]\w*)
    |(?P<punct>::|=>|->|\.\.=|\.\.\.|\.\.|==|!=|<=|>=|&&|\|\||[-+*/%^&|]=|[^\s\w])
    """, re.X)
_OPEN = {"(": ")", "[": "]", "{": "}"}
FRAGMENTS = {"ident", "lifetime", "literal", "tt", "ty", "path", "expr", "pat", "pat_param", "block", "meta",
             "vis", "item"}


class Unsupported(ValueError):
    pass


@dataclass(frozen=True)
class Tok:
    text: str
    kind: str  # word, lit, life, punct


@dataclass(frozen=True)
class Group:
    open: str
    inner: tuple


def lex(text: str) -> list[Tok]:
    """Tokens of comment-free source (`rust_items.File.kept`), string literals whole."""
    out, p = [], 0
    while p < len(text):
        m = _TOKEN.match(text, p)
        if m is None:
            raise Unsupported(f"token at {p}")
        if m.lastgroup is None:
            p = m.end()
            continue
        kind, value = m.lastgroup, m.group(m.lastgroup)
        if kind == "lit" and value.endswith('"'):
            hashes = value.count("#")
            if "r" in value[:-1]:
                end = text.find('"' + "#" * hashes, m.end())
                if end == -1:
                    raise Unsupported("unterminated raw string")
                end += 1 + hashes
            else:
                end = m.end()
                while end < len(text) and text[end] != '"':
                    end += 2 if text[end] == "\\" else 1
                end += 1
            value = text[m.start():end]
            p = end
        else:
            p = m.end()
        out.append(Tok(value, kind))
    return out


def trees(tokens: list[Tok]) -> tuple:
    def read(i: int, close: str | None):
        out = []
        while i < len(tokens):
            t = tokens[i]
            if t.kind == "punct" and t.text in _OPEN:
                inner, i = read(i + 1, _OPEN[t.text])
                out.append(Group(t.text, inner))
            elif t.kind == "punct" and t.text in ")]}":
                if t.text != close:
                    raise Unsupported("unbalanced group")
                return tuple(out), i + 1
            else:
                out.append(t)
                i += 1
        if close is not None:
            raise Unsupported("unclosed group")
        return tuple(out), i
    return read(0, None)[0]


def render(tts) -> str:
    out = []
    for t in tts:
        if isinstance(t, Group):
            inner = render(t.inner)
            out.append(f"{t.open} {inner} {_OPEN[t.open]}" if inner else f"{t.open} {_OPEN[t.open]}")
        else:
            out.append(t.text)
    return " ".join(out)


# ---------------------------------------------------------------------- matchers

@dataclass(frozen=True)
class Var:
    name: str
    fragment: str


@dataclass(frozen=True)
class Rep:
    inner: tuple
    separator: Tok | None
    op: str


def _is(t, text: str) -> bool:
    return isinstance(t, Tok) and t.text == text


def pattern(tts: tuple, *, matcher: bool) -> tuple:
    """A matcher or transcriber: `$name[:frag]` and `$( … ) sep? op` recognised."""
    out, i = [], 0
    while i < len(tts):
        t = tts[i]
        if _is(t, "$") and i + 1 < len(tts):
            nxt = tts[i + 1]
            if isinstance(nxt, Group) and nxt.open == "(":
                j = i + 2
                separator = None
                if j < len(tts) and isinstance(tts[j], Tok) and tts[j].text not in ("*", "+", "?"):
                    separator = tts[j]
                    j += 1
                if j >= len(tts) or not isinstance(tts[j], Tok) or tts[j].text not in ("*", "+", "?"):
                    raise Unsupported("repetition without an operator")
                out.append(Rep(pattern(nxt.inner, matcher=matcher), separator, tts[j].text))
                i = j + 1
                continue
            if isinstance(nxt, Tok) and nxt.kind == "word":
                if nxt.text == "crate":
                    out.append(Tok("crate", "word"))
                    i += 2
                    continue
                if matcher:
                    if i + 3 >= len(tts) or not _is(tts[i + 2], ":") or not isinstance(tts[i + 3], Tok) \
                            or tts[i + 3].text not in FRAGMENTS:
                        raise Unsupported("metavariable without a supported fragment")
                    out.append(Var(nxt.text, tts[i + 3].text))
                    i += 4
                else:
                    out.append(Var(nxt.text, ""))
                    i += 2
                continue
            raise Unsupported("unsupported `$` form")
        if isinstance(t, Group):
            out.append(Group(t.open, pattern(t.inner, matcher=matcher)))
        else:
            out.append(t)
        i += 1
    return tuple(out)


_TYPE_STOP = {",", ";", "=", "=>", "|", ":", "{"}
_EXPR_STOP = {",", ";", "=>"}


def _fragment(kind: str, tts: tuple, i: int) -> int | None:
    """The end of one fragment of `kind` starting at `i`, or `None`."""
    if kind == "vis":
        if i < len(tts) and _is(tts[i], "pub"):
            return i + 2 if i + 1 < len(tts) and isinstance(tts[i + 1], Group) and tts[i + 1].open == "(" else i + 1
        return i
    if i >= len(tts):
        return None
    t = tts[i]
    if kind == "tt":
        return i + 1
    if kind == "ident":
        return i + 1 if isinstance(t, Tok) and t.kind == "word" else None
    if kind == "lifetime":
        return i + 1 if isinstance(t, Tok) and t.kind == "life" else None
    if kind == "literal":
        if _is(t, "-") and i + 1 < len(tts) and isinstance(tts[i + 1], Tok) and tts[i + 1].kind == "lit":
            return i + 2
        return i + 1 if isinstance(t, Tok) and (t.kind == "lit" or t.text in ("true", "false")) else None
    if kind == "block":
        return i + 1 if isinstance(t, Group) and t.open == "{" else None
    if kind == "meta":
        # `path`, `path = literal` or `path(…)`.
        j = i
        while j < len(tts) and isinstance(tts[j], Tok) and (tts[j].kind == "word" or tts[j].text == "::"):
            j += 1
        if j == i:
            return None
        if j < len(tts) and isinstance(tts[j], Group):
            return j + 1
        if j + 1 < len(tts) and _is(tts[j], "="):
            return _fragment("expr", tts, j + 1)
        return j
    if kind in ("ty", "path"):
        j, angle = i, 0
        while j < len(tts):
            u = tts[j]
            if isinstance(u, Tok):
                if u.text == "<":
                    angle += 1
                elif u.text == ">":
                    if angle == 0:
                        break
                    angle -= 1
                elif u.text == ">>":
                    if angle < 2:
                        break
                    angle -= 2
                elif angle == 0 and (u.text in _TYPE_STOP or (kind == "path" and u.kind != "word" and u.text != "::")):
                    break
            elif angle == 0 and (u.open == "{" or kind == "path"):
                break
            j += 1
        return j if j > i and angle == 0 else None
    if kind in ("expr", "pat", "pat_param"):
        stops = _EXPR_STOP | ({"=", "if"} if kind != "expr" else set()) | ({"|"} if kind == "pat_param" else set())
        j = i
        while j < len(tts) and not (isinstance(tts[j], Tok) and tts[j].text in stops):
            j += 1
        return j if j > i else None
    if kind == "item":
        # Attributes, then tokens up to and including the first `{ … }` group or `;`.
        j = i
        while j < len(tts):
            u = tts[j]
            j += 1
            if (isinstance(u, Group) and u.open == "{") or _is(u, ";"):
                return j
        return None
    raise Unsupported(kind)


def match(matcher: tuple, tts: tuple, i: int = 0, depth: int = 0):
    """Bindings when `matcher` matches `tts[i:]` to its end, else `None`."""
    bindings: dict = {}
    end = _match(matcher, 0, tts, i, bindings)
    return bindings if end == len(tts) else None


def _match(matcher: tuple, k: int, tts: tuple, i: int, bindings: dict) -> int | None:
    while k < len(matcher):
        element = matcher[k]
        if isinstance(element, Tok):
            if i >= len(tts) or tts[i] != element:
                return None
            i += 1
        elif isinstance(element, Group):
            if i >= len(tts) or not isinstance(tts[i], Group) or tts[i].open != element.open:
                return None
            inner: dict = {}
            if _match(element.inner, 0, tts[i].inner, 0, inner) != len(tts[i].inner):
                return None
            bindings.update(inner)
            i += 1
        elif isinstance(element, Var):
            end = _fragment(element.fragment, tts, i)
            if end is None:
                return None
            bindings[element.name] = tts[i:end]
            i = end
        else:
            rounds = []
            while True:
                start = i
                if rounds and element.separator is not None:
                    if i >= len(tts) or tts[i] != element.separator:
                        break
                    start = i + 1
                one: dict = {}
                end = _match(element.inner, 0, tts, start, one)
                if end is None or end == start:
                    break
                rounds.append(one)
                i = end
                if element.op == "?":
                    break
            if element.op == "+" and not rounds:
                return None
            for name in _names(element.inner):
                bindings[name] = [one.get(name, []) for one in rounds]
        k += 1
    return i


def _names(matcher: tuple) -> list[str]:
    out = []
    for element in matcher:
        if isinstance(element, Var):
            out.append(element.name)
        elif isinstance(element, (Group, Rep)):
            out.extend(_names(element.inner))
    return out


def transcribe(transcriber: tuple, bindings: dict) -> tuple:
    out = []
    for element in transcriber:
        if isinstance(element, Tok):
            out.append(element)
        elif isinstance(element, Group):
            out.append(Group(element.open, transcribe(element.inner, bindings)))
        elif isinstance(element, Var):
            value = bindings.get(element.name)
            if value is None:
                # Not a metavariable of this rule: `$name` is written as is.
                raise Unsupported(f"unbound ${element.name}")
            if isinstance(value, list):
                raise Unsupported(f"${element.name} used outside its repetition")
            out.extend(value)
        else:
            names = [n for n in _names(element.inner) if isinstance(bindings.get(n), list)]
            if not names:
                raise Unsupported("repetition without a repeated metavariable")
            lengths = {len(bindings[n]) for n in names}
            if len(lengths) != 1:
                raise Unsupported("repetition lengths disagree")
            for index in range(lengths.pop()):
                if index and element.separator is not None:
                    out.append(element.separator)
                out.extend(transcribe(element.inner, {**bindings, **{n: bindings[n][index] for n in names}}))
    return tuple(out)


def _argument(tts: tuple) -> Group | None:
    """The group after the `!` (and a `macro_rules!` name): attributes before it are skipped."""
    for i, t in enumerate(tts):
        if _is(t, "!"):
            return next((u for u in tts[i + 1:i + 3] if isinstance(u, Group)), None)
    return None


def rules(definition: str) -> list[tuple[tuple, tuple]]:
    """The `(matcher, transcriber)` rules of a `macro_rules! name { … }` item's text."""
    tts = trees(lex(definition))
    body = _argument(tts)
    if body is None:
        raise Unsupported("macro without a body")
    out, i, inner = [], 0, body.inner
    while i < len(inner):
        if i + 2 >= len(inner) or not isinstance(inner[i], Group) or not _is(inner[i + 1], "=>") \
                or not isinstance(inner[i + 2], Group):
            raise Unsupported("malformed macro rule")
        out.append((pattern(inner[i].inner, matcher=True), pattern(inner[i + 2].inner, matcher=False)))
        i += 3
        if i < len(inner) and _is(inner[i], ";"):
            i += 1
    return out


def expand(definition: str, invocation: str) -> str | None:
    """The text an invocation (`name! { … }`, as written) expands to, or `None`."""
    try:
        tts = trees(lex(invocation))
        argument = _argument(tts)
        if argument is None:
            return None
        for matcher, transcriber in rules(definition):
            bindings = match(matcher, argument.inner)
            if bindings is not None:
                return render(transcribe(transcriber, bindings))
    except (Unsupported, RecursionError):
        return None
    return None
