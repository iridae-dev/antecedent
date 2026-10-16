"""One Rust lexer for the release gates.

A single scan produces the three views the gates already consume:

* ``code`` drops comments and blanks string and character literal contents,
  which is what name searches want.
* ``skeleton`` blanks comments and literals to spaces and keeps every offset,
  which is what brace matching wants.
* ``kept`` blanks comments but leaves literal contents in place, which is what
  a cited test's closure wants when it must still read a string.
* ``line_docs`` are the ``///`` spans (not ``////``).

Byte raw strings, raw strings, character literals, and lifetimes are the same
tokens in every view.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SPECIAL = re.compile(r"//|/\*|b?r#*\"|\"|'")
_CHAR = re.compile(r"'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]{1,6}\}|.)|[^\\'\n])'")


@dataclass(frozen=True)
class RustSource:
    """Views of one source file. ``skeleton``, ``kept``, and ``kept_skeleton`` share offsets.

    ``kept_skeleton`` blanks comment tokens and literal interiors, and leaves the
    literal delimiters, so ``extern "C"`` is still a prefix word.
    """

    code: str
    skeleton: str
    kept: str
    kept_skeleton: str
    line_docs: tuple[tuple[int, int], ...]


def _spaces(text: str, a: int, b: int) -> str:
    return "".join("\n" if c == "\n" else " " for c in text[a:b])


def lex(text: str) -> RustSource:
    """Lex ``text``. The three views are built from the same token boundaries."""
    code: list[str] = []
    skel: list[str] = []
    kept: list[str] = []
    kept_skel: list[str] = []
    docs: list[tuple[int, int]] = []
    i, n = 0, len(text)

    def emit(head: str) -> None:
        code.append(head)
        skel.append(head)
        kept.append(head)
        kept_skel.append(head)

    def blank_comment(a: int, b: int) -> None:
        blank = _spaces(text, a, b)
        skel.append(blank)
        kept.append(blank)
        kept_skel.append(blank)

    def blank_literal(a: int, b: int, opener: int, closer: int) -> None:
        code.append('""' + "\n" * text.count("\n", a, b))
        skel.append(_spaces(text, a, b))
        kept.append(text[a:b])
        interior = _spaces(text, a + opener, b - closer)
        kept_skel.append(text[a : a + opener] + interior + text[b - closer : b])

    while i < n:
        m = _SPECIAL.search(text, i)
        if m is None:
            emit(text[i:])
            break
        start = m.start()
        tok = m.group(0)
        if (
            tok.startswith(("r", "b"))
            and start > 0
            and (text[start - 1].isalnum() or text[start - 1] == "_")
        ):
            # `for"` / `abr"`: an identifier ending in r or b, then a quote.
            emit(text[i : start + len(tok) - 1])
            i = start + len(tok) - 1
            continue
        emit(text[i:start])
        if tok == "//":
            end = text.find("\n", start)
            end = n if end < 0 else end
            if text.startswith("///", start) and not text.startswith("////", start):
                docs.append((start, end))
            blank_comment(start, end)
            i = end
        elif tok == "/*":
            depth, j = 1, start + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    if text[j] == "\n":
                        code.append("\n")
                    j += 1
            j = min(j, n)
            blank_comment(start, j)
            i = j
        elif tok == '"':
            j = start + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            closed = j < n and text[j] == '"'
            end = j + 1 if closed else n
            blank_literal(start, end, 1, 1 if closed else 0)
            i = end
        elif tok == "'":
            ch = _CHAR.match(text, start)
            if ch:
                blank_literal(start, ch.end(), 1, 1)
                code[-1] = "' '"
                i = ch.end()
            else:
                emit("'")
                i = start + 1
        else:
            hashes = tok.count("#")
            closer = '"' + "#" * hashes
            found = text.find(closer, m.end())
            end = n if found < 0 else found + len(closer)
            opener = len(tok)
            blank_literal(start, end, opener, len(closer) if found >= 0 else 0)
            i = end
    return RustSource(
        "".join(code),
        "".join(skel),
        "".join(kept),
        "".join(kept_skel),
        tuple(docs),
    )


def rust_code(text: str) -> str:
    """Comments dropped; string and character literal contents blanked."""
    return lex(text).code


def has_lint_attribute(
    text: str, level: str, lint: str, *, inner: bool | None = None
) -> bool:
    """Find a direct lint attribute, ignoring comments, literals and reason text.

    Conditional ``cfg_attr`` attributes do not establish an unconditional lint
    policy. ``inner=True`` requires a crate/module attribute; ``False`` requires
    an item attribute, and ``None`` accepts either.
    """
    marker = "!" if inner is True else "" if inner is False else "!?"
    pattern = rf"#\s*{marker}\s*\[\s*{re.escape(level)}\s*\(([^()]*)\)\s*\]"
    item = re.compile(rf"(?:^|,)\s*{re.escape(lint)}\s*(?:,|$)")
    return any(
        item.search(match.group(1)) for match in re.finditer(pattern, rust_code(text))
    )


def rust_mask(text: str, line_docs: list[tuple[int, int]] | None = None) -> str:
    """Comments and literals blanked to spaces. Offsets and newlines stay."""
    parsed = lex(text)
    if line_docs is not None:
        line_docs.extend(parsed.line_docs)
    return parsed.skeleton
