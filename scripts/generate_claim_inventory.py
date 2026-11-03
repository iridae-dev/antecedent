#!/usr/bin/env python3
"""Render the claim registry and the per-query support inventory.

Facts live in parity/claims.toml and parity/support_{axes,supported,n_a}.toml.
This script writes the text between generated markers. `--check` fails when a
marker disagrees with those files. Hand-written prose outside the markers is
not read.

Markers:
  <!-- generated:claim:<id>:begin -->text<!-- generated:claim:<id>:end -->
  <!-- generated:claims:begin --> ... <!-- generated:claims:end -->
  <!-- generated:query-inventory:begin --> ... <!-- generated:query-inventory:end -->
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections import defaultdict
from itertools import product

from _repo import REPO_ROOT as ROOT

CLAIM_FILES = ("README.md", "docs/capabilities.md", "docs/comparison.md")
CATALOG_FILE = "docs/capabilities.md"
INVENTORY_FILES = ("docs/capabilities.md", "docs/comparison.md")

CLAIM_MARK = re.compile(
    r"<!-- generated:claim:([A-Za-z0-9_]+):begin -->(.*?)<!-- generated:claim:\1:end -->",
    re.S,
)
CATALOG_BEGIN = "<!-- generated:claims:begin -->"
CATALOG_END = "<!-- generated:claims:end -->"
INVENTORY_BEGIN = "<!-- generated:query-inventory:begin -->"
INVENTORY_END = "<!-- generated:query-inventory:end -->"

CLAIM_FIELDS = ("id", "text", "supported_by")


def load_toml(rel: str) -> dict:
    return tomllib.loads((ROOT / rel).read_text(encoding="utf-8"))


def load_claims() -> list[dict]:
    rows = load_toml("parity/claims.toml").get("claim") or []
    if not rows:
        raise SystemExit("parity/claims.toml: no [[claim]] rows")
    seen: set[str] = set()
    for row in rows:
        extra = set(row) - set(CLAIM_FIELDS)
        if extra:
            raise SystemExit(f"parity/claims.toml {row.get('id')}: unknown fields {sorted(extra)}")
        for field in CLAIM_FIELDS:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise SystemExit(f"parity/claims.toml {row.get('id')}: missing {field}")
        if row["id"] in seen:
            raise SystemExit(f"parity/claims.toml: duplicate claim id {row['id']}")
        seen.add(row["id"])
        if not re.fullmatch(r"[a-z][a-z0-9_]*", row["id"]):
            raise SystemExit(f"parity/claims.toml: claim id {row['id']} is not snake_case")
        supported = ROOT / row["supported_by"]
        if not supported.is_file():
            raise SystemExit(f"claims.toml {row['id']}: supported_by {row['supported_by']} missing")
    return rows


def check_reusable_headline(claims: list[dict]) -> None:
    if not any(row["id"] == "reusable_headline" for row in claims):
        return
    products = load_toml("parity/python_products.toml")
    for row in products.get("route") or []:
        if not row.get("retains") and not row.get("reason"):
            raise SystemExit(
                "python_products.toml route "
                f"{row.get('kind')}/{row.get('data')}/{row.get('structure')}: "
                "retains=false without reason"
            )


def render_catalog(claims: list[dict]) -> str:
    lines = [
        "Rendered from `parity/claims.toml` by `scripts/generate_claim_inventory.py`.",
        "Do not edit this block by hand.",
        "",
    ]
    for row in claims:
        lines.extend(
            [
                f"### `{row['id']}`",
                "",
                f"Supported by `{row['supported_by']}`.",
                "",
                "```text",
                row["text"],
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


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


def ordered(values: set[str], axis: list[str]) -> list[str]:
    return [name for name in axis if name in values]


def cover(triples: list[tuple[str, str, str]], axes: dict) -> list[tuple[str, str, str, int]]:
    """Group (structure, inference, validation) triples into exact products."""
    structures = ordered({s for s, _, _ in triples}, axes["structures"])
    inferences = ordered({i for _, i, _ in triples}, axes["inferences"])
    validations = ordered({v for _, _, v in triples}, axes["validations"])
    have = set(triples)
    if len(triples) != len(have):
        raise SystemExit("duplicate supported coordinate")
    full = {(s, i, v) for s in structures for i in inferences for v in validations}
    if have == full:
        return [(", ".join(structures), ", ".join(inferences), ", ".join(validations), len(have))]
    rows: list[tuple[str, str, str, int]] = []
    for structure in structures:
        sub = [(i, v) for s, i, v in triples if s == structure]
        inf = ordered({i for i, _ in sub}, axes["inferences"])
        val = ordered({v for _, v in sub}, axes["validations"])
        have_sub = set(sub)
        full_sub = {(i, v) for i in inf for v in val}
        if have_sub == full_sub:
            rows.append((structure, ", ".join(inf), ", ".join(val), len(have_sub)))
            continue
        for inference, validation in sorted(
            have_sub, key=lambda pair: (inf.index(pair[0]), val.index(pair[1]))
        ):
            rows.append((structure, inference, validation, 1))
    return rows


def render_inventory() -> str:
    axes = load_toml("parity/support_axes.toml")
    na_rules = load_toml("parity/support_n_a.toml").get("n_a") or []
    import gate_support

    # One support count: the cells counted below are the matrix units that pass the one
    # support rule (scripts/gate_support.py); a listed cell that fails it is not supported.
    units = gate_support.summarize(gate_support.evaluate())
    cells = [
        c
        for c in load_toml("parity/support_cells.toml").get("cell") or []
        if gate_support.cell_ident(c) in units["matrix_ok"]
    ]
    queries = list(axes["queries"])
    graphs = list(axes["graph_classes"])
    structures = list(axes["structures"])
    inferences = list(axes["inferences"])
    validations = list(axes["validations"])

    supported: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for cell in cells:
        key = (cell["graph_class"], cell["structure"], cell["inference"], cell["validation"])
        supported[cell["query"]].append(key)
        if cell["query"] not in queries:
            raise SystemExit(f"support_cells.toml: query {cell['query']} is not on the axes")

    lines = [
        "Generated from `parity/support_axes.toml`, `parity/support_cells.toml`,",
        "and `parity/support_n_a.toml` by `scripts/generate_claim_inventory.py`.",
        "Supported coordinates are grouped only where the cell lists the full product",
        "of the structures, inferences, and validations shown. A coordinate that is",
        "not listed is n/a or refused. Refusal is the default;",
        "`parity/support_refusal_reasons.toml` names reasons and does not create a fourth state.",
        "The [support matrix](support-matrix.md) is the support.",
        "Do not edit this block by hand.",
        "",
        "| Query | Supported | n/a | Refused |",
        "|---|---:|---:|---:|",
    ]
    detail = [
        "",
        "| Query | Graph class | Structures | Inferences | Validations | Cells |",
        "|---|---|---|---|---|---:|",
    ]
    supported_total = 0
    for query in queries:
        coordinates = supported[query]
        if len(set(coordinates)) != len(coordinates):
            raise SystemExit(f"{query}: duplicate supported coordinate")
        coordinate_set = set(coordinates)
        n_supported = n_na = n_refused = 0
        for graph, structure, inference, validation in product(
            graphs, structures, inferences, validations
        ):
            cell = {
                "query": query,
                "graph_class": graph,
                "structure": structure,
                "inference": inference,
                "validation": validation,
            }
            if any(matches(rule, cell) for rule in na_rules):
                if (graph, structure, inference, validation) in coordinate_set:
                    raise SystemExit(f"{query}: supported cell is also n/a ({graph})")
                n_na += 1
            elif (graph, structure, inference, validation) in coordinate_set:
                n_supported += 1
            else:
                n_refused += 1
        if n_supported != len(coordinates):
            raise SystemExit(
                f"{query}: {len(coordinates)} supported rows but {n_supported} sit on the axes"
            )
        supported_total += n_supported
        lines.append(f"| {query} | {n_supported} | {n_na} | {n_refused} |")
        by_graph: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for graph, structure, inference, validation in coordinates:
            by_graph[graph].append((structure, inference, validation))
        for graph in graphs:
            if graph not in by_graph:
                continue
            for structures_cell, inferences_cell, validations_cell, count in cover(
                by_graph[graph], axes
            ):
                detail.append(
                    f"| {query} | {graph} | {structures_cell} | {inferences_cell} | "
                    f"{validations_cell} | {count} |"
                )
    if supported_total != len(cells):
        raise SystemExit(
            f"inventory covers {supported_total} supported cells; support_cells.toml has {len(cells)}"
        )
    lines.append("")
    lines.append(f"**{gate_support.headline(units)}.**")
    lines.append("")
    lines.append(
        f"The {supported_total} matrix cells below are the matrix part of that count; "
        "graphless rows (`parity/support_graphless.toml`) and capabilities outside the matrix "
        "(`parity/capability_records.toml`) make up the rest. Entry-point routes are not counted."
    )
    lines.extend(detail)
    return "\n".join(lines)


def block(text: str, begin: str, end: str) -> tuple[str, str]:
    start = text.find(begin)
    stop = text.find(end)
    if start < 0 or stop < 0 or stop < start:
        raise SystemExit(f"missing {begin} / {end}")
    if text.find(begin, start + len(begin)) >= 0:
        raise SystemExit(f"duplicate {begin}")
    inner_start = start + len(begin)
    return text[inner_start:stop], text[:start] + text[stop + len(end) :]


def replace_block(text: str, begin: str, end: str, body: str) -> str:
    start = text.find(begin)
    stop = text.find(end)
    if start < 0 or stop < 0 or stop < start:
        raise SystemExit(f"missing {begin} / {end}")
    inner = f"{begin}\n{body.rstrip()}\n{end}"
    return text[:start] + inner + text[stop + len(end) :]


def sync_claims(text: str, claims: list[dict], *, write: bool) -> str:
    by_id = {row["id"]: row["text"] for row in claims}

    def repl(match: re.Match[str]) -> str:
        claim_id = match.group(1)
        if claim_id not in by_id:
            raise SystemExit(f"generated claim {claim_id} is not in parity/claims.toml")
        return (
            f"<!-- generated:claim:{claim_id}:begin -->"
            f"{by_id[claim_id]}"
            f"<!-- generated:claim:{claim_id}:end -->"
        )

    found = {match.group(1) for match in CLAIM_MARK.finditer(text)}
    unknown = found - set(by_id)
    if unknown:
        raise SystemExit(f"generated claim ids not in parity/claims.toml: {sorted(unknown)}")
    if write:
        return CLAIM_MARK.sub(repl, text)
    for match in CLAIM_MARK.finditer(text):
        claim_id, body = match.group(1), match.group(2)
        if body != by_id[claim_id]:
            raise SystemExit(f"generated claim {claim_id} does not match parity/claims.toml")
    return text


def sync_region(text: str, begin: str, end: str, body: str, *, write: bool, label: str) -> str:
    if begin not in text or end not in text:
        raise SystemExit(f"missing {label} markers {begin} / {end}")
    if write:
        return replace_block(text, begin, end, body)
    inner, _rest = block(text, begin, end)
    if inner.strip("\n") != body.rstrip():
        raise SystemExit(f"generated {label} does not match the registry")
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    write = not args.check
    claims = load_claims()
    check_reusable_headline(claims)
    catalog = render_catalog(claims)
    inventory = render_inventory()
    rendered: set[str] = set()

    for rel in CLAIM_FILES:
        path = ROOT / rel
        text = path.read_text(encoding="utf-8")
        try:
            updated = sync_claims(text, claims, write=write)
            if rel == CATALOG_FILE:
                updated = sync_region(
                    updated, CATALOG_BEGIN, CATALOG_END, catalog, write=write, label="claims"
                )
            if rel in INVENTORY_FILES:
                updated = sync_region(
                    updated,
                    INVENTORY_BEGIN,
                    INVENTORY_END,
                    inventory,
                    write=write,
                    label="query-inventory",
                )
        except SystemExit as exc:
            message = exc.code if isinstance(exc.code, str) else str(exc.code)
            if not message.startswith(f"{rel}:"):
                message = f"{rel}: {message}"
            raise SystemExit(message) from exc
        rendered.update(match.group(1) for match in CLAIM_MARK.finditer(updated))
        if rel == CATALOG_FILE and CATALOG_BEGIN in updated:
            rendered.update(row["id"] for row in claims)
        if write and updated != text:
            path.write_text(updated, encoding="utf-8", newline="\n")

    missing = [row["id"] for row in claims if row["id"] not in rendered]
    if missing:
        raise SystemExit("parity/claims.toml claims are not rendered: " + ", ".join(missing))
    if args.check:
        print("claim inventory OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
