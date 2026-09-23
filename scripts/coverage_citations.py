#!/usr/bin/env python3
"""Coverage figures in licensed-row limitations come from coverage records.

    python3 scripts/coverage_citations.py check   # used by gate_coverage_citations.sh
    python3 scripts/coverage_citations.py cite    # add record citations to parity/support_licensed.toml

A *coverage figure* is a three- or four-decimal rate in [0.1, 1] written in a
sentence about coverage (`cover`, `covers`, `coverage`) that is not a location,
bias, standard error or pinned value (see `EXEMPT_BEFORE`). Known-truth values,
standard errors and seeded pins are never coverage figures, and pins with five
or more decimals are never matched.

Every coverage figure must be attributed by the first attribution that follows
it in the same sentence:

* a record citation, one or more backticked ids from parity/coverage_records.toml
  (`` (record `cov.…`) `` or `` (records `cov.…`, `cov.…`) ``); the figure must equal
  the `observed` value of one cited record to three decimals, so a re-measured
  record forces the prose to follow it; or
* the disclosure `not a registry value`, for a figure a probe or an earlier run
  measured that no registry row carries at that value. The figure stays public; the
  text says where it comes from: a new disclosure names its source,
  `not a registry value (probe <module>::<test_fn>)`, which must resolve to a
  compiled, non-ignored test in crates/antecedent/tests/<module>.rs.

Two-decimal rates (`0.93`) and percentages (`93 %`, `93.5%`) in a coverage
sentence are coverage figures too, except the nominal levels 0.90/0.95/90%/95%.
Those forms predate the gate, and the figures written before it existed, together
with the bare disclosures written before disclosures named a source, are listed in
parity/_coverage_citation_backlog.txt. The backlog only shrinks: a figure or
bare disclosure that is not listed fails, and a listed one that no longer exists
must be removed (`python3 scripts/coverage_citations.py backlog` rewrites it).

`cite` inserts record citations after figures whose value matches a record of a
test the sentence already names, marks the remaining figures as disclosures, and
rewrites an already-cited figure whose record has moved so the prose follows it.
"""

from __future__ import annotations

import fnmatch
import itertools
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LICENSED = ROOT / "parity/support_licensed.toml"
RECORDS = ROOT / "parity/coverage_records.toml"

DISCLOSURE = "not a registry value"
BACKLOG = ROOT / "parity/_coverage_citation_backlog.txt"
# A source-naming disclosure: `not a registry value (probe <module>::<test_fn>)`.
PROBE_RE = re.compile(r"\s*\(probe ([a-z0-9_]+)::([A-Za-z0-9_]+)\)")
# Two-decimal rates and percentages; three- and four-decimal rates are FIGURE_RE.
WIDE_FIGURE_RE = re.compile(
    r"(?<![\w.])(?:0\.(?P<dec>\d{2})(?![\d])|(?P<pct>\d{2,3}(?:\.\d+)?)\s?%)"
)
NOMINAL_LEVELS = {0.9, 0.95, 1.0}
FIGURE_RE = re.compile(r"(?<![\w.])0\.(\d{3,4})(?![\d])")
RECORD_RE = re.compile(r"`(cov\.[A-Za-z0-9_.]+)`")
DISCLOSURE_RE = re.compile(re.escape(DISCLOSURE))
COVERAGE_WORD_RE = re.compile(r"(?i)\bcover(?:s|ed|age)?\b")
# A cited boundary record must be named as a boundary in the surrounding
# sentence, not only inside a record id or test name.
BOUNDARY_DISCLOSURE_RE = re.compile(r"(?i)\bboundary\b|below the band|unasserted|floor miss")
# A number right after one of these words is a location, bias, SE or pin, not a rate.
EXEMPT_BEFORE = re.compile(
    r"(?:\bat|\bbias|\bSE(?: is)?|\bSD|\bmean|\btotal|\beffect|\bMixture|draw's|=)\s*$"
)
SENTENCE_END_RE = re.compile(r"(?<=[a-z)\]`%])\.\s+(?=[A-Z0-9(`])")
TOLERANCE = 0.0005 + 1e-9


def sentences(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in SENTENCE_END_RE.finditer(text):
        spans.append((start, m.start() + 1))
        start = m.end()
    spans.append((start, len(text)))
    return spans


def figures(text: str, lo: int, hi: int) -> list[re.Match]:
    body = text[lo:hi]
    if not COVERAGE_WORD_RE.search(body):
        return []
    out = []
    for m in FIGURE_RE.finditer(text, lo, hi):
        if float(m.group(0)) < 0.1:
            continue
        if EXEMPT_BEFORE.search(text[max(lo, m.start() - 12) : m.start()]):
            continue
        out.append(m)
    return out


def wide_figures(text: str, lo: int, hi: int) -> list[tuple[re.Match, float, float]]:
    """Two-decimal rates and percentages of a coverage sentence: (match, value, tolerance).

    The value is the rate in [0.1, 1]; the tolerance is half a unit in the last
    written place. Nominal levels are not figures."""
    if not COVERAGE_WORD_RE.search(text[lo:hi]):
        return []
    out = []
    for m in WIDE_FIGURE_RE.finditer(text, lo, hi):
        if m.group("dec") is not None:
            value, tol = float(m.group(0)), 0.005 + 1e-9
        else:
            digits = m.group("pct")
            places = len(digits.split(".", 1)[1]) if "." in digits else 0
            value, tol = float(digits) / 100.0, 0.5 * 10 ** (-(places + 2)) + 1e-9
        if value < 0.1 or value > 1.0 or round(value, 6) in NOMINAL_LEVELS:
            continue
        if EXEMPT_BEFORE.search(text[max(lo, m.start() - 12) : m.start()]):
            continue
        out.append((m, value, tol))
    return out


def load_records() -> dict[str, dict]:
    return {r["id"]: r for r in tomllib.loads(RECORDS.read_text()).get("record", [])}


def attribution_after(text: str, pos: int, hi: int):
    """The first attribution after `pos` in its clause: ('records', [ids]) or ('disclosure', None).

    A clause ends at the sentence end or the next `;`, so an attribution written
    for a later clause never covers an earlier figure."""
    semi = text.find(";", pos, hi)
    if semi != -1:
        hi = semi
    rec = RECORD_RE.search(text, pos, hi)
    dis = DISCLOSURE_RE.search(text, pos, hi)
    if rec and (not dis or rec.start() < dis.start()):
        ids = [rec.group(1)]
        end = rec.end()
        while True:
            nxt = re.compile(r"(?:,\s*|\s+and\s+)`(cov\.[A-Za-z0-9_.]+)`").match(text, end)
            if not nxt:
                break
            ids.append(nxt.group(1))
            end = nxt.end()
        return "records", ids
    if dis:
        return "disclosure", None
    return None, None


def row_label(cell: dict) -> str:
    return "/".join(
        cell[k] for k in ("query", "graph_class", "structure", "inference", "validation")
    )


def probe_problem(module: str, name: str) -> str | None:
    """Why `<module>::<name>` is not a compiled, non-ignored test of crates/antecedent/tests."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import test_evidence  # noqa: PLC0415  (only the disclosures that name a source need it)

    path = ROOT / "crates/antecedent/tests" / f"{module}.rs"
    if not path.is_file():
        return f"{path.relative_to(ROOT)} does not exist"
    why = test_evidence.static_rust_test(path, name).problems
    return why[0] if why else None


def open_items() -> tuple[set[str], list[str]]:
    """Backlog-eligible items now in the prose, and the problems that are never backlogged.

    An item is `kind|row|figure`: an `unattributed` two-decimal or percent figure, or a
    `disclosure` that names no probe source. Attributed and probe-sourced ones are checked
    here and never listed."""
    records = load_records()
    items: set[str] = set()
    problems: list[str] = []
    for cell in tomllib.loads(LICENSED.read_text()).get("cell", []):
        text = str(cell.get("limitations", ""))
        label = row_label(cell)
        for lo, hi in sentences(text):
            for fig in figures(text, lo, hi):
                kind, _ = attribution_after(text, fig.end(), hi)
                if kind == "disclosure":
                    dis = DISCLOSURE_RE.search(text, fig.end(), hi)
                    probe = PROBE_RE.match(text, dis.end())
                    if probe is None:
                        items.add(f"disclosure|{label}|{fig.group(0)}")
                    elif (why := probe_problem(*probe.groups())) is not None:
                        problems.append(
                            f"{label}: disclosure of {fig.group(0)} names probe "
                            f"{probe.group(1)}::{probe.group(2)}, which is not a test: {why}"
                        )
            for fig, value, tol in wide_figures(text, lo, hi):
                kind, ids = attribution_after(text, fig.end(), hi)
                if kind == "records":
                    known = [records[i] for i in ids if i in records]
                    if known and not any(abs(float(r["observed"]) - value) <= tol for r in known):
                        observed = ", ".join(f"{float(r['observed']):.4f}" for r in known)
                        problems.append(
                            f"{label}: coverage figure {fig.group(0)} does not match its cited "
                            f"record(s) ({observed})"
                        )
                elif kind == "disclosure":
                    dis = DISCLOSURE_RE.search(text, fig.end(), hi)
                    probe = PROBE_RE.match(text, dis.end())
                    if probe is None:
                        items.add(f"disclosure|{label}|{fig.group(0)}")
                    elif (why := probe_problem(*probe.groups())) is not None:
                        problems.append(
                            f"{label}: disclosure of {fig.group(0)} names probe "
                            f"{probe.group(1)}::{probe.group(2)}, which is not a test: {why}"
                        )
                else:
                    items.add(f"unattributed|{label}|{fig.group(0)}")
    return items, problems


def read_backlog() -> set[str]:
    if not BACKLOG.is_file():
        return set()
    return {
        line
        for line in BACKLOG.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


def backlog_problems() -> list[str]:
    items, problems = open_items()
    listed = read_backlog()
    for item in sorted(items - listed):
        kind, label, figure = item.split("|", 2)
        what = (
            "is a disclosure that names no source; write "
            f"'{DISCLOSURE} (probe <module>::<test_fn>)'"
            if kind == "disclosure"
            else "is a two-decimal or percent coverage figure with no record citation and no "
            f"'{DISCLOSURE} (probe ...)' disclosure"
        )
        problems.append(f"{label}: {figure} {what}")
    for item in sorted(listed - items):
        problems.append(
            f"{BACKLOG.name}: {item!r} no longer occurs; remove it (the backlog only shrinks)"
        )
    return problems


def write_backlog() -> int:
    items, _ = open_items()
    BACKLOG.write_text(
        "# Coverage figures and disclosures that predate scripts/coverage_citations.py's\n"
        "# two-decimal/percent scan and source-naming disclosures: kind|row|figure.\n"
        "# The list only shrinks (scripts/gate_coverage_citations.sh fails on a new or a\n"
        "# stale line). Regenerate with `python3 scripts/coverage_citations.py backlog`\n"
        "# after attributing figures; it must never be used to admit a new one.\n"
        + "".join(f"{item}\n" for item in sorted(items))
    )
    return len(items)


def check() -> list[str]:
    records = load_records()
    problems = backlog_problems()
    for cell in tomllib.loads(LICENSED.read_text()).get("cell", []):
        text = str(cell.get("limitations", ""))
        label = row_label(cell)
        for rid in RECORD_RE.findall(text):
            if rid not in records:
                problems.append(f"{label}: cites unknown coverage record {rid}")
        for lo, hi in sentences(text):
            for fig in figures(text, lo, hi):
                kind, ids = attribution_after(text, fig.end(), hi)
                value = float(fig.group(0))
                if kind is None:
                    problems.append(
                        f"{label}: coverage figure {fig.group(0)} has no record citation and is not "
                        f"marked '{DISCLOSURE}'"
                    )
                elif kind == "records":
                    known = [records[i] for i in ids if i in records]
                    if known and not any(
                        abs(float(r["observed"]) - value) <= TOLERANCE for r in known
                    ):
                        observed = ", ".join(f"{float(r['observed']):.4f}" for r in known)
                        problems.append(
                            f"{label}: coverage figure {fig.group(0)} does not match its cited "
                            f"record(s) ({observed})"
                        )
                    if known and any(r.get("boundary") for r in known) and not _discloses_boundary(
                        text[lo:hi]
                    ):
                        problems.append(
                            f"{label}: coverage figure {fig.group(0)} cites a boundary record "
                            "without naming it a boundary"
                        )
    return problems


def _discloses_boundary(sentence: str) -> bool:
    """True when the sentence names a boundary outside citations and test paths."""
    stripped = RECORD_RE.sub("", sentence)
    stripped = re.sub(r"`[^`]+`", "", stripped)
    stripped = re.sub(r"::[A-Za-z0-9_{},*]+", "", stripped)
    return BOUNDARY_DISCLOSURE_RE.search(stripped) is not None


# ------------------------------------------------------------------ cite

CITE_RE = re.compile(
    r"(?:crates/antecedent/tests/)?(v1(?:9|10)_[a-z0-9_]+)(?:\.rs)?(?:::([A-Za-z0-9_{},*]+)"
    r"((?:,\s*::[A-Za-z0-9_{},*]+)*))?"
)
CONT_RE = re.compile(r"::([A-Za-z0-9_{},*]+)")


def _expand(pattern: str) -> list[str]:
    parts = re.split(r"(\{[^}]*\})", pattern)
    alts = [p[1:-1].split(",") if p.startswith("{") else [p] for p in parts]
    return ["".join(c) for c in itertools.product(*alts)]


def cited_records(sentence: str, records: dict[str, dict]) -> list[dict]:
    pats: list[tuple[str, str]] = []
    for m in CITE_RE.finditer(sentence):
        if not m.group(2):
            continue  # a file-wide citation names no test precisely enough to bind a figure
        names = [m.group(2)] + CONT_RE.findall(m.group(3) or "")
        for name in names:
            for pat in _expand(name.rstrip(".,;")):
                pats.append((m.group(1), pat))
    bare = set(
        re.findall(r"\b([a-z][a-z0-9_]*_(?:coverage|boundary|probe|within_band))\b", sentence)
    )
    out = []
    for rec in records.values():
        path, fn = rec["test"].split("::")
        module = Path(path).stem
        if fn in bare or any(module == mod and fnmatch.fnmatchcase(fn, pat) for mod, pat in pats):
            out.append(rec)
    levels = set(re.findall(r"\b(90|95)%", sentence)) | {
        str(int(round(float(x) * 100))) for x in re.findall(r"nominal (0\.9|0\.95)\b", sentence)
    }
    if len(levels) == 1:
        (level,) = levels
        narrowed = [r for r in out if abs(float(r["nominal"]) * 100 - int(level)) < 1e-6]
        out = narrowed or out
    return out


def _format_observed(observed: float, template: str) -> str:
    decimals = len(template.split(".", 1)[1])
    rendered = f"{observed:.{decimals}f}"
    if abs(float(rendered) - observed) <= TOLERANCE:
        return rendered
    return f"{observed:.4f}"


def rebind_text(text: str, records: dict[str, dict]) -> str:
    """Move already-cited figures onto their records' current `observed` values."""
    edits: list[tuple[int, int, str]] = []
    for lo, hi in sentences(text):
        for fig in figures(text, lo, hi):
            kind, ids = attribution_after(text, fig.end(), hi)
            if kind != "records" or not ids:
                continue
            known = [records[i] for i in ids if i in records]
            if not known:
                continue
            value = float(fig.group(0))
            if any(abs(float(r["observed"]) - value) <= TOLERANCE for r in known):
                continue
            observed_vals = [float(r["observed"]) for r in known]
            before = text[max(lo, fig.start() - 2) : fig.start()]
            after = text[fig.end() : fig.end() + 1]
            if before.endswith(("–", "-")):
                target = max(observed_vals)
            elif after in {"–", "-"}:
                target = min(observed_vals)
            else:
                target = min(observed_vals, key=lambda obs: abs(obs - value))
            edits.append((fig.start(), fig.end(), _format_observed(target, fig.group(0))))
    for start, end, new in sorted(edits, reverse=True):
        text = text[:start] + new + text[end:]
    return text


def cite_text(text: str, records: dict[str, dict]) -> str:
    edits: list[tuple[int, str]] = []
    for lo, hi in sentences(text):
        figs = [f for f in figures(text, lo, hi) if attribution_after(text, f.end(), hi)[0] is None]
        if not figs:
            continue
        pool = cited_records(text[lo:hi], records)
        tagged = []
        for fig in figs:
            value = float(fig.group(0))
            ids = sorted(r["id"] for r in pool if abs(float(r["observed"]) - value) <= TOLERANCE)
            # Bind only an unambiguous match: one test (a curve's grid points share it).
            if len({records[i]["test"] for i in ids}) != 1:
                ids = []
            tagged.append((fig, ids))
        # Consecutive figures of the same kind share one attribution after the last.
        i = 0
        while i < len(tagged):
            j = i
            recorded = bool(tagged[i][1])
            while j + 1 < len(tagged) and bool(tagged[j + 1][1]) == recorded:
                gap = text[tagged[j][0].end() : tagged[j + 1][0].start()]
                if "(" in gap or ";" in gap or len(gap) > 40:
                    break
                j += 1
            last = tagged[j][0]
            end = last.end()
            tail = re.compile(r" at 0\.\d+").match(text, end)
            if tail:
                end = tail.end()
            if recorded:
                ids = sorted({rid for _, group in tagged[i : j + 1] for rid in group})
                noun = "record" if len(ids) == 1 else "records"
                note = f" ({noun} " + ", ".join(f"`{rid}`" for rid in ids) + ")"
            else:
                note = f" ({DISCLOSURE})"
            edits.append((end, note))
            i = j + 1
    for pos, note in sorted(edits, reverse=True):
        text = text[:pos] + note + text[pos:]
    return text


def cite() -> int:
    records = load_records()
    raw = LICENSED.read_text()
    changed = 0

    def repl(m: re.Match) -> str:
        nonlocal changed
        value = tomllib.loads(m.group(0))["limitations"]
        new = cite_text(rebind_text(value, records), records)
        if new == value:
            return m.group(0)
        changed += 1
        escaped = new.replace("\\", "\\\\").replace('"', '\\"')
        return f'limitations = "{escaped}"'

    out = re.sub(r'(?m)^limitations = ".*"$', repl, raw)
    LICENSED.write_text(out)
    return changed


def main(argv: list[str]) -> int:
    if argv == ["check"]:
        problems = check()
        if problems:
            print("Coverage figure attribution FAILED:")
            for p in problems:
                print(" -", p)
            return 1
        print("Coverage figures OK (every coverage figure cites a matching record or is disclosed)")
        return 0
    if argv == ["backlog"]:
        print(f"wrote {write_backlog()} backlog line(s) to {BACKLOG.relative_to(ROOT)}")
        return 0
    if argv == ["cite"]:
        print(f"cited coverage records in {cite()} rows")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
