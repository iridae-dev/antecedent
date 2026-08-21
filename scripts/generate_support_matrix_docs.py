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
    closed_count = 0
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
            closed_count += 1
        elif is_allowed(cell):
            allowed_count += 1
    refused = cartesian - n_a_count - len(cells)
    default_refused = refused - closed_count - allowed_count

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
        lic_rows = [
            "| query | graph | structure | inference | validation | evidence |",
            "|---|---|---|---|---|---|",
        ]
        for row in cells:
            ev = row.get("evidence_kind", "")
            fix = row.get("known_truth_fixture", "")
            ev_s = f"{ev}" + (f" (`{fix}`)" if fix else "")
            lic_rows.append(
                f"| `{row['query']}` | `{row['graph_class']}` | `{row['structure']}` | "
                f"`{row['inference']}` | `{row['validation']}` | {ev_s} |"
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
Most of it is typed impossibility, not missing work.

| Status | Count | How to read it |
|---|---|---|
| Cartesian product | {cartesian} | Axis product, not a coverage score |
| n/a | {n_a_count} | Typed impossibilities (temporal query on a static graph, static query on a temporal graph, ATE-shaped cheap/full on a function-valued estimand, and similar). These are not holes. |
| Meaningful remainder | {cartesian - n_a_count} | Combinations that could in principle be a claim |
| Licensed | {len(cells)} | Staged path plus executing known-truth evidence — the strongest contract |
| Allowlisted (running, unlicensed) | {allowed_count} | Executes end-to-end; a successful number is **not** a licensed claim |
| Refused (enforced closed rules) | {closed_count} | Fail shut, including mislabeled-inference laundering |
| Refused (no allowlist match) | {default_refused} | Fail shut by default |

Do not read "{len(cells)} / {cartesian}" as coverage. Read: **{len(cells)} cells
carry the evidence contract**; {allowed_count} more run without that contract;
the rest are n/a or refused.

A missing cell is refused, not unspecified. `analyze` is sugar over the
staged path; a combination that only works inside `analyze` cannot be
licensed. A cell is exactly one of licensed / n/a / closed / allowlisted; any
refused cell not matched by the allowlist fails closed. Successful studies
record `licensed` vs `allowed_unlicensed` on the result (`evidence_status` in
Python, `StudyResult.support_status` in Rust) so the distinction survives
dispatch.

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

## Enforced refusals

These default-refused cells fail closed with id `refused`.

{chr(10).join(closed_lines) if closed_lines else "_None._"}

## Allowlisted (running, unlicensed)

These cells are neither licensed nor closed but do genuinely run; each row
below states why it runs and which licensed/keep-running family it rides.
Every other refused cell fails closed.

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
            "closed": closed_count,
            "allowed": allowed_count,
            "refused": cartesian - n_a_count - closed_count - allowed_count - len(cells),
        },
        axes,
    )
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"Wrote {RUST_OUT.relative_to(ROOT)}")
    print(f"Wrote {RELEASE_NOTES.relative_to(ROOT)} (licensed block)")
    return 0


def render_release_licensed(cells: list[dict], axes: dict) -> list[str]:
    """One bullet per (query, inference), compact when rows form a product."""
    from itertools import product as iproduct

    order_q = list(axes["queries"]) + list(axes.get("stage_queries") or [])
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
            graphs = sorted({c["graph_class"] for c in rows})
            structs = [s for s in order_s if any(c["structure"] == s for c in rows)]
            vals = [v for v in order_v if any(c["validation"] == v for c in rows)]
            combos = {(c["structure"], c["validation"]) for c in rows}
            if combos == set(iproduct(structs, vals)):
                s_part = " / ".join(f"`{s}`" for s in structs)
                v_part = " / ".join(f"`{v}`" for v in vals)
                g_part = " / ".join(f"`{g}`" for g in graphs)
                lines.append(
                    f"- `{q}` × {g_part} × {s_part} × `{inf}` × validation {v_part}"
                )
            else:
                for c in rows:
                    lines.append(
                        f"- `{q}` × `{c['graph_class']}` × `{c['structure']}` × "
                        f"`{inf}` × validation `{c['validation']}`"
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
        f"{counts['allowed']} allowlisted (running, unlicensed); "
        f"{counts['closed']} closed; {counts['refused']} refused with no allowlist match.",
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
