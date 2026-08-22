#!/usr/bin/env python3
"""Generate docs/support-matrix.md from parity/support_*.toml.

Idempotent: re-running with no matrix changes must leave a clean git tree.

The licensed-cell block is rewritten only in the current workspace version's
release notes (`docs/release-notes/vX.Y.Z.md`). Historical notes use frozen
markers so a later regen cannot overwrite a shipped cut. `set_version.sh`
freezes the previous notes when the version bumps.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "support-matrix.md"
RUST_OUT = ROOT / "crates" / "antecedent" / "src" / "support_matrix_data.rs"
NOTES_DIR = ROOT / "docs" / "release-notes"
RN_BEGIN = "<!-- generated:support-matrix:licensed:begin -->"
RN_END = "<!-- generated:support-matrix:licensed:end -->"
FROZEN_BEGIN = "<!-- frozen:support-matrix:licensed:begin -->"
FROZEN_END = "<!-- frozen:support-matrix:licensed:end -->"


def workspace_version() -> str:
    cargo = (ROOT / "Cargo.toml").read_text()
    m = re.search(
        r"(?ms)^\[workspace\.package\]\n.*?^version\s*=\s*\"([^\"]+)\"",
        cargo,
    )
    if not m:
        raise SystemExit("Cargo.toml: [workspace.package] version not found")
    return m.group(1)


def release_notes_path(version: str) -> Path:
    return NOTES_DIR / f"v{version}.md"


RELEASE_NOTES = release_notes_path(workspace_version())


def git_tag_exists(version: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/v{version}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def freeze_licensed_block(path: Path) -> bool:
    """Rewrite live generated markers to frozen. Returns whether the file changed."""
    text = path.read_text()
    if RN_BEGIN not in text and RN_END not in text:
        return False
    path.write_text(
        text.replace(RN_BEGIN, FROZEN_BEGIN).replace(RN_END, FROZEN_END)
    )
    return True


def assert_historical_notes_are_frozen(version: str) -> None:
    """Only the current cut may keep live generated licensed-block markers."""
    current = release_notes_path(version).resolve()
    live: list[Path] = []
    for path in sorted(NOTES_DIR.glob("v*.md")):
        if path.resolve() == current:
            continue
        text = path.read_text()
        if RN_BEGIN in text or RN_END in text:
            live.append(path)
    if live:
        names = ", ".join(str(p.relative_to(ROOT)) for p in live)
        raise SystemExit(
            f"{names}: still have live generated licensed-block markers. "
            f"Freeze them (`{FROZEN_BEGIN}` / `{FROZEN_END}`) so a later "
            f"matrix regen cannot overwrite a shipped cut. Only "
            f"{current.relative_to(ROOT)} may keep live markers."
        )


def load(rel: str) -> dict:
    return tomllib.loads((ROOT / rel).read_text())


def matches(rule: dict, cell: dict) -> bool:
    mapping = {
        "queries": "query",
        "graph_classes": "graph_class",
        "structures": "structure",
        "inferences": "inference",
        "validations": "validation",
    }
    for rule_key, cell_key in mapping.items():
        allowed = rule.get(rule_key)
        if allowed is not None and cell[cell_key] not in allowed:
            return False
    return True


def main() -> int:
    axes = load("parity/support_axes.toml")
    na_rules = load("parity/support_n_a.toml").get("n_a") or []
    closed_rules = load("parity/support_closed.toml").get("closed") or []
    allowed_rules = load("parity/support_allowlist.toml").get("allowed") or []
    cells = load("parity/support_licensed.toml").get("cell") or []
    queries = list(axes["queries"]) + list(axes.get("stage_queries") or [])
    graphs = list(axes["graph_classes"])
    structures = list(axes["structures"])
    inferences = list(axes["inferences"])
    validations = list(axes["validations"])

    def is_n_a(cell: dict) -> bool:
        return any(matches(rule, cell) for rule in na_rules)

    def is_closed(cell: dict) -> bool:
        return (not is_n_a(cell)) and any(matches(rule, cell) for rule in closed_rules)

    def is_allowed(cell: dict) -> bool:
        return (
            (not is_n_a(cell))
            and (not is_closed(cell))
            and any(matches(rule, cell) for rule in allowed_rules)
        )

    cartesian = 0
    n_a_count = 0
    reason_backed_refused_count = 0
    allowed_count = 0
    for q, g, s, inf, v in product(queries, graphs, structures, inferences, validations):
        cartesian += 1
        cell = {
            "query": q,
            "graph_class": g,
            "structure": s,
            "inference": inf,
            "validation": v,
        }
        if is_n_a(cell):
            n_a_count += 1
        elif is_closed(cell):
            reason_backed_refused_count += 1
        elif is_allowed(cell):
            allowed_count += 1
    refused = cartesian - n_a_count - len(cells)
    unreasoned_refused_count = refused - reason_backed_refused_count - allowed_count

    def bullets(xs: list[str]) -> str:
        return "\n".join(f"- `{x}`" for x in xs)

    def rule_lines(rules: list[dict]) -> list[str]:
        lines = []
        for rule in rules:
            bits = []
            for key in ("queries", "graph_classes", "structures", "inferences", "validations"):
                if key in rule:
                    bits.append(f"{key} ∈ {{{', '.join(rule[key])}}}")
            lines.append(f"- {' ∧ '.join(bits)} — {rule['reason']}")
        return lines

    na_lines = rule_lines(na_rules)
    closed_lines = rule_lines(closed_rules)

    def allowed_lines(rules: list[dict]) -> list[str]:
        lines = []
        for rule in rules:
            bits = []
            for key in ("queries", "graph_classes", "structures", "inferences", "validations"):
                if key in rule:
                    bits.append(f"{key} ∈ {{{', '.join(rule[key])}}}")
            lines.append(
                f"- {' ∧ '.join(bits)} — {rule['reason']} (parent: {rule['parent']})"
            )
        return lines

    allowed_md_lines = allowed_lines(allowed_rules)

    if cells:
        def md_cell(value: str) -> str:
            return " ".join(value.split()).replace("|", r"\|")

        lic_rows = [
            "| query | graph | structure | inference | validation | evidence | limitations |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in cells:
            ev = row.get("evidence_kind", "")
            fix = row.get("known_truth_fixture", "")
            ev_s = f"{ev}" + (f" (`{fix}`)" if fix else "")
            limitations = md_cell(row.get("limitations", ""))
            lic_rows.append(
                f"| `{row['query']}` | `{row['graph_class']}` | `{row['structure']}` | "
                f"`{row['inference']}` | `{row['validation']}` | {ev_s} | {limitations} |"
            )
        licensed_md = "\n".join(lic_rows)
    else:
        licensed_md = (
            "No licensed cells yet. Every remaining combination is **refused** "
            "until it runs on the staged path (`identify` → prepare → estimate) "
            "with recorded evidence."
        )

    text = f"""# Support matrix

Generated by `scripts/generate_support_matrix_docs.py` from
`parity/support_axes.toml`, `parity/support_n_a.toml`,
`parity/support_closed.toml`, `parity/support_allowlist.toml`, and
`parity/support_licensed.toml`.
Do not edit this page by hand.

This page is the public **license**. `docs/capabilities.md` is an inventory
of what exists in the codebase; it does not license a cell.
See [ADR 0020](../adr/0020-support-matrix-and-prepared-workflow.md).

The Cartesian product (query × graph class × structure source × inference ×
validation) is **{cartesian}** cells. That denominator is not a feature count.
Of those cells, **{n_a_count}** are typed impossibilities and
**{cartesian - n_a_count}** are meaningful combinations.

Every cell is in exactly **one of three runtime states**: **licensed** (a
result), **n/a** (the coordinate does not denote — a typed impossibility),
or **refused** (`SupportRefusal::Refused`). Refusal is the *default*: any
cell that is not licensed and not n/a is refused, whether or not a rule
names a reason for it. `support_closed.toml` does not close anything — it
is the **reason table** for refused cells, not a fourth state. A refused
cell either has a documented reason on file or it doesn't; both refuse
identically at runtime. `allowed_unlicensed` is retained as a compatibility
wire value, but 0.9 has no active allowlist entries.

| Status | Count | How to read it |
|---|---|---|
| Cartesian product | {cartesian} | Axis product, not a coverage score |
| n/a | {n_a_count} | Typed impossibilities (temporal query on a static graph, static query on a temporal graph, ATE-shaped cheap/full on a function-valued estimand, and similar). These are not holes. |
| Meaningful remainder | {cartesian - n_a_count} | Combinations that could in principle be a claim |
| Licensed | {len(cells)} | Staged path plus the row's recorded evidence contract and limitations |
| `allowed_unlicensed` compatibility entries | {allowed_count} | Retained wire value; 0.9 requires this count to remain zero |
| Refused — reason on file | {reason_backed_refused_count} | Same runtime outcome as any other refused cell; documented in legacy-named `support_closed.toml`, including mislabeled-inference laundering |
| Refused — no reason on file yet | {unreasoned_refused_count} | Same runtime outcome; no rule in `support_closed.toml` names it yet |

Do not read "{len(cells)} / {cartesian}" as coverage. Read: **{len(cells)} cells
carry their recorded evidence contracts**; no cells run through the retained
`allowed_unlicensed` compatibility path; the rest are n/a or refused.

A missing cell is refused, not unspecified. `analyze` is sugar over the
staged path; a combination that only works inside `analyze` cannot be
licensed. A cell is exactly one of licensed / n/a / refused.
`allowed_unlicensed` remains a readable wire value for compatibility with
older artifacts and clients, but no 0.9 matrix cell can produce it.

## Axes

**Queries** (root `__all__`):

{bullets(list(axes["queries"]))}

**Stage queries** (not folded into `analyze`):

{bullets(list(axes.get("stage_queries") or []))}

**Graph classes** (`GraphClass`):

{bullets(graphs)}

**Structure source:** {", ".join(f"`{x}`" for x in structures)}.

**Inference:** {", ".join(f"`{x}`" for x in inferences)}.

**Validation:** {", ".join(f"`{x}`" for x in validations)}.

## n/a

{chr(10).join(na_lines) if na_lines else "_None._"}

## Refusal reasons

These cells are refused (wire id `refused`) — the default runtime state for
any cell that is not licensed and not n/a — and each row below is the
documented reason for that refusal. This is a reason table, not a fourth
state: an undocumented refused cell behaves identically, it just has no
row here yet.

{chr(10).join(closed_lines) if closed_lines else "_None._"}

## `allowed_unlicensed` compatibility entries

The wire value is retained for compatibility with older artifacts and clients.
The 0.9 gate requires this list to be empty: every active cell is licensed,
n/a, or refused.

{chr(10).join(allowed_md_lines) if allowed_md_lines else "_None._"}

## Licensed cells

{licensed_md}
"""
    OUT.write_text(text)
    RUST_OUT.write_text(render_rust(na_rules, closed_rules, allowed_rules, cells))
    write_release_notes_block(
        cells,
        {
            "cartesian": cartesian,
            "licensed": len(cells),
            "n_a": n_a_count,
            "reason_backed_refused": reason_backed_refused_count,
            "allowed": allowed_count,
            "unreasoned_refused": (
                cartesian
                - n_a_count
                - reason_backed_refused_count
                - allowed_count
                - len(cells)
            ),
        },
        axes,
    )
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"Wrote {RUST_OUT.relative_to(ROOT)}")
    print(f"Wrote {RELEASE_NOTES.relative_to(ROOT)} (licensed block)")
    return 0


def render_release_licensed(cells: list[dict], axes: dict) -> list[str]:
    """Compact only graph classes with identical rectangular cell products."""
    from itertools import product as iproduct

    order_q = list(axes["queries"]) + list(axes.get("stage_queries") or [])
    order_g = list(axes["graph_classes"])
    order_s = list(axes["structures"])
    order_v = list(axes["validations"])
    order_i = list(axes["inferences"])
    lines: list[str] = []
    for q in order_q:
        for inf in order_i:
            rows = [
                c
                for c in cells
                if c["query"] == q and c["inference"] == inf
            ]
            if not rows:
                continue
            expected = {
                (c["graph_class"], c["structure"], c["validation"]) for c in rows
            }
            emitted: set[tuple[str, str, str]] = set()
            profiles: dict[frozenset[tuple[str, str]], list[str]] = {}
            for graph in order_g:
                profile = frozenset(
                    (c["structure"], c["validation"])
                    for c in rows
                    if c["graph_class"] == graph
                )
                if profile:
                    profiles.setdefault(profile, []).append(graph)

            for profile, graphs in profiles.items():
                structs = [s for s in order_s if any(pair[0] == s for pair in profile)]
                vals = [v for v in order_v if any(pair[1] == v for pair in profile)]
                rectangular = set(profile) == set(iproduct(structs, vals))
                if not rectangular:
                    for graph in graphs:
                        for structure in structs:
                            for validation in vals:
                                key = (graph, structure, validation)
                                if key not in expected:
                                    continue
                                lines.append(
                                    f"- `{q}` × `{graph}` × `{structure}` × "
                                    f"`{inf}` × validation `{validation}`"
                                )
                                emitted.add(key)
                    continue

                s_part = " / ".join(f"`{s}`" for s in structs)
                v_part = " / ".join(f"`{v}`" for v in vals)
                g_part = " / ".join(f"`{g}`" for g in graphs)
                lines.append(
                    f"- `{q}` × {g_part} × {s_part} × `{inf}` × validation {v_part}"
                )
                emitted.update(iproduct(graphs, structs, vals))

            # Focused invariant: compaction must preserve the exact licensed
            # graph × structure × validation set for this query/inference pair.
            if emitted != expected:
                missing = sorted(expected - emitted)
                extra = sorted(emitted - expected)
                raise AssertionError(
                    f"release-note compaction changed {q}/{inf}: "
                    f"missing={missing}, extra={extra}"
                )
    return lines


def write_release_notes_block(cells: list[dict], counts: dict, axes: dict) -> None:
    """Replace the marked licensed block in the *current* release notes.

    Fails if the live markers are missing, if a historical notes file still
    has live markers, or if this workspace version is already a git tag —
    bump the version and put live markers on the new notes file first.
    """
    version = workspace_version()
    assert_historical_notes_are_frozen(version)
    if git_tag_exists(version):
        raise SystemExit(
            f"git tag v{version} already exists; refusing to regenerate "
            f"{RELEASE_NOTES.relative_to(ROOT)}. Bump the workspace version "
            "and add live generated markers on the new notes file first."
        )
    text = RELEASE_NOTES.read_text()
    if RN_BEGIN not in text or RN_END not in text:
        raise SystemExit(
            f"{RELEASE_NOTES}: missing generated-block markers "
            f"{RN_BEGIN!r} / {RN_END!r}"
        )
    body_lines = [
        f"{counts['licensed']} licensed of {counts['cartesian'] - counts['n_a']} meaningful "
        f"cells ({counts['n_a']} n/a typed impossibilities are not a coverage gap). "
        f"{counts['reason_backed_refused']} refused with a reason on file; "
        f"{counts['unreasoned_refused']} refused without a reason; "
        f"{counts['allowed']} active `allowed_unlicensed` compatibility entries.",
        "",
    ] + render_release_licensed(cells, axes)
    block = RN_BEGIN + "\n" + "\n".join(body_lines) + "\n" + RN_END
    head, rest = text.split(RN_BEGIN, 1)
    _, tail = rest.split(RN_END, 1)
    RELEASE_NOTES.write_text(head + block + tail)


def rust_list(values: list[str] | None) -> str:
    if values is None:
        return "None"
    if not values:
        return "Some(&[])"
    inner = ", ".join(f'"{v}"' for v in values)
    return f"Some(&[{inner}])"


def rust_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def render_rules(rules: list[dict]) -> str:
    items = []
    for rule in rules:
        items.append(
            "    NaRule {\n"
            f"        queries: {rust_list(rule.get('queries'))},\n"
            f"        graph_classes: {rust_list(rule.get('graph_classes'))},\n"
            f"        structures: {rust_list(rule.get('structures'))},\n"
            f"        inferences: {rust_list(rule.get('inferences'))},\n"
            f"        validations: {rust_list(rule.get('validations'))},\n"
            f"        reason: \"{rust_escape(rule['reason'])}\",\n"
            "    }"
        )
    return ",\n".join(items) if items else ""


def render_allowed_rules(rules: list[dict]) -> str:
    items = []
    for rule in rules:
        items.append(
            "    AllowedRule {\n"
            f"        queries: {rust_list(rule.get('queries'))},\n"
            f"        graph_classes: {rust_list(rule.get('graph_classes'))},\n"
            f"        structures: {rust_list(rule.get('structures'))},\n"
            f"        inferences: {rust_list(rule.get('inferences'))},\n"
            f"        validations: {rust_list(rule.get('validations'))},\n"
            f"        reason: \"{rust_escape(rule['reason'])}\",\n"
            f"        parent: \"{rust_escape(rule['parent'])}\",\n"
            "    }"
        )
    return ",\n".join(items) if items else ""


def render_rust(
    na_rules: list[dict],
    closed_rules: list[dict],
    allowed_rules: list[dict],
    cells: list[dict],
) -> str:
    na_block = render_rules(na_rules)
    closed_block = render_rules(closed_rules)
    allowed_block = render_allowed_rules(allowed_rules)
    lic_items = []
    for row in cells:
        lic_items.append(
            "    LicensedCell {\n"
            f"        query: \"{rust_escape(row['query'])}\",\n"
            f"        graph_class: \"{rust_escape(row['graph_class'])}\",\n"
            f"        structure: \"{rust_escape(row['structure'])}\",\n"
            f"        inference: \"{rust_escape(row['inference'])}\",\n"
            f"        validation: \"{rust_escape(row['validation'])}\",\n"
            "    }"
        )
    lic_block = ",\n".join(lic_items) if lic_items else ""
    return f"""//! Generated from `parity/support_*.toml`. Do not edit.

#![allow(missing_docs)]
#![cfg_attr(rustfmt, rustfmt::skip)]

#[derive(Clone, Copy, Debug)]
pub struct NaRule {{
    pub queries: Option<&'static [&'static str]>,
    pub graph_classes: Option<&'static [&'static str]>,
    pub structures: Option<&'static [&'static str]>,
    pub inferences: Option<&'static [&'static str]>,
    pub validations: Option<&'static [&'static str]>,
    pub reason: &'static str,
}}

#[derive(Clone, Copy, Debug)]
pub struct AllowedRule {{
    pub queries: Option<&'static [&'static str]>,
    pub graph_classes: Option<&'static [&'static str]>,
    pub structures: Option<&'static [&'static str]>,
    pub inferences: Option<&'static [&'static str]>,
    pub validations: Option<&'static [&'static str]>,
    pub reason: &'static str,
    pub parent: &'static str,
}}

#[derive(Clone, Copy, Debug)]
pub struct LicensedCell {{
    pub query: &'static str,
    pub graph_class: &'static str,
    pub structure: &'static str,
    pub inference: &'static str,
    pub validation: &'static str,
}}

pub static NA_RULES: &[NaRule] = &[
{na_block}
];

pub static CLOSED_RULES: &[NaRule] = &[
{closed_block}
];

pub static ALLOWED_RULES: &[AllowedRule] = &[
{allowed_block}
];

pub static LICENSED: &[LicensedCell] = &[
{lic_block}
];
"""


def _freeze_cli(version: str) -> int:
    path = release_notes_path(version)
    if not path.is_file():
        print(f"no notes to freeze: {path.relative_to(ROOT)}")
        return 0
    if freeze_licensed_block(path):
        print(f"Froze licensed block in {path.relative_to(ROOT)}")
    else:
        print(f"{path.relative_to(ROOT)} has no live licensed-block markers")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--freeze":
        sys.exit(_freeze_cli(sys.argv[2]))
    if len(sys.argv) > 1:
        print("usage: generate_support_matrix_docs.py [--freeze X.Y.Z]", file=sys.stderr)
        sys.exit(2)
    sys.exit(main())
