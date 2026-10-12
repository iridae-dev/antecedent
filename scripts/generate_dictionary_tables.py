#!/usr/bin/env python3
"""Render the identity dictionary and the producer/consumer matrix from TOML.

Facts live in parity/identity.toml and parity/producer_consumer.toml. The
markdown pages keep hand-written prose; the tables sit between generated
markers. `python3 scripts/generate_dictionary_tables.py --check` fails when a
marker block does not match the TOML render.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from _repo import REPO_ROOT as ROOT

IDENTITY = ROOT / "parity" / "identity.toml"
MATRIX = ROOT / "parity" / "producer_consumer.toml"
NAMING = ROOT / "docs" / "api_naming.md"
PRODUCER_CONSUMER = ROOT / "docs" / "producer-consumer-matrix.md"

IDENTITY_FIELDS = (
    "domain",
    "rust",
    "contract_key",
    "python",
    "availability",
    "naming_row",
    "covers",
    "doc_rust",
    "doc_python",
)
STATUSES = ("DIRECT", "ADAPTER", "REFUSED")
IDENTITY_BEGIN = "<!-- generated:identity-dictionary:begin -->"
IDENTITY_END = "<!-- generated:identity-dictionary:end -->"
MATRIX_BLOCKS = (
    ("producers", "<!-- generated:producer-consumer:producers:begin -->", "<!-- generated:producer-consumer:producers:end -->"),
    ("consumers", "<!-- generated:producer-consumer:consumers:begin -->", "<!-- generated:producer-consumer:consumers:end -->"),
    ("matrix", "<!-- generated:producer-consumer:matrix:begin -->", "<!-- generated:producer-consumer:matrix:end -->"),
    ("refusals", "<!-- generated:producer-consumer:refusals:begin -->", "<!-- generated:producer-consumer:refusals:end -->"),
)


def _cell(text: object, label: str) -> str:
    if not isinstance(text, str) or not text.strip() or "\n" in text or "|" in text:
        raise ValueError(f"{label} is not a single markdown cell")
    return text


def load_identity() -> list[dict]:
    rows = tomllib.loads(IDENTITY.read_text(encoding="utf-8")).get("identity", [])
    if not rows:
        raise ValueError("parity/identity.toml has no [[identity]] rows")
    seen: set[str] = set()
    for row in rows:
        domain = row.get("domain")
        for field in IDENTITY_FIELDS:
            if field == "availability":
                value = row.get(field)
                if not isinstance(value, str) or not value.strip() or "\n" in value:
                    raise ValueError(f"identity {domain} availability is empty")
                continue
            _cell(row.get(field), f"identity {domain} {field}")
        if row["domain"] in seen:
            raise ValueError(f"duplicate identity domain {row['domain']}")
        seen.add(row["domain"])
        if row["python"] not in row["doc_python"]:
            raise ValueError(f"identity {domain}: doc_python must contain python")
        if "|" in row["availability"]:
            parts = row["availability"].split("|")
            if not parts or any(not part.strip() for part in parts):
                raise ValueError(f"identity {domain}: empty availability")
    return rows


def load_matrix() -> dict:
    data = tomllib.loads(MATRIX.read_text(encoding="utf-8"))
    producers = data.get("producer", [])
    consumers = data.get("consumer", [])
    cells = data.get("cell", [])
    refusals = data.get("refusal", [])
    producer_keys = [_cell(row.get("key"), "producer key") for row in producers]
    consumer_keys = [_cell(row.get("key"), "consumer key") for row in consumers]
    if len(producer_keys) != len(set(producer_keys)):
        raise ValueError("duplicate producer key")
    if len(consumer_keys) != len(set(consumer_keys)):
        raise ValueError("duplicate consumer key")
    for row in producers:
        _cell(row.get("object"), f"producer {row.get('key')} object")
        _cell(row.get("built_from"), f"producer {row.get('key')} built_from")
    for row in consumers:
        _cell(row.get("python"), f"consumer {row.get('key')} python")
        _cell(row.get("rust"), f"consumer {row.get('key')} rust")
    seen: set[tuple[str, str]] = set()
    for row in cells:
        pair = (_cell(row.get("producer"), "cell producer"), _cell(row.get("consumer"), "cell consumer"))
        if pair in seen:
            raise ValueError(f"duplicate cell {pair}")
        seen.add(pair)
        if pair[0] not in producer_keys or pair[1] not in consumer_keys:
            raise ValueError(f"cell {pair} is not a declared producer/consumer")
        if row.get("status") not in STATUSES:
            raise ValueError(f"cell {pair} status must be one of {STATUSES}")
        _cell(row.get("detail"), f"cell {pair} detail")
    expected = {(producer, consumer) for producer in producer_keys for consumer in consumer_keys}
    if seen != expected:
        raise ValueError(f"matrix is missing {sorted(expected - seen)}")
    for row in refusals:
        _cell(row.get("code"), "refusal code")
        _cell(row.get("detail"), "refusal detail")
        _cell(row.get("where"), "refusal where")
    if not refusals:
        raise ValueError("producer/consumer matrix has no refusals")
    return data


def render_identity(rows: list[dict]) -> str:
    lines = [
        "| Capability | Rust | Python |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['naming_row']} | {row['doc_rust']} | {row['doc_python']} |"
        )
    return "\n".join(lines)


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_producers(data: dict) -> str:
    return _table(
        ["Row key", "Object", "Built from"],
        [[f"`{row['key']}`", row["object"], row["built_from"]] for row in data["producer"]],
    )


def render_consumers(data: dict) -> str:
    return _table(
        ["Column key", "Python entry", "Rust entry"],
        [[f"`{row['key']}`", row["python"], row["rust"]] for row in data["consumer"]],
    )


def render_matrix(data: dict) -> str:
    consumers = [row["key"] for row in data["consumer"]]
    by_pair = {(row["producer"], row["consumer"]): row for row in data["cell"]}
    header = ["Producer", *[f"`{key}`" for key in consumers]]
    rows = []
    counts = {status: 0 for status in STATUSES}
    for producer in data["producer"]:
        rendered = []
        for consumer in consumers:
            cell = by_pair[(producer["key"], consumer)]
            counts[cell["status"]] += 1
            rendered.append(f"{cell['status']}: {cell['detail']}")
        rows.append([f"`{producer['key']}`", *rendered])
    total = sum(counts.values())
    summary = (
        f"Counts: {total} cells, **{counts['DIRECT']} DIRECT**, "
        f"**{counts['ADAPTER']} ADAPTER**, **{counts['REFUSED']} REFUSED**, "
        "**0 UNVERIFIED**. Every cell is\n"
        "executed by `test_producer_consumer_matrix_<row>_row[<consumer>]`."
    )
    return _table(header, rows) + "\n\n" + summary


def render_refusals(data: dict) -> str:
    return _table(
        ["Code", "Detail", "Where"],
        [[row["code"], row["detail"], row["where"]] for row in data["refusal"]],
    )


def splice(text: str, begin: str, end: str, body: str) -> str:
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f"expected one {begin} / {end} pair")
    pre, rest = text.split(begin, 1)
    _mid, post = rest.split(end, 1)
    if not pre.endswith("\n") or not post.startswith("\n"):
        raise ValueError(f"{begin} markers must occupy their own lines")
    return f"{pre}{begin}\n{body.rstrip()}\n{end}{post}"


def rendered_pages() -> list[tuple[Path, str]]:
    identity = render_identity(load_identity())
    matrix = load_matrix()
    naming = splice(NAMING.read_text(encoding="utf-8"), IDENTITY_BEGIN, IDENTITY_END, identity)
    producer = PRODUCER_CONSUMER.read_text(encoding="utf-8")
    blocks = {
        "producers": render_producers(matrix),
        "consumers": render_consumers(matrix),
        "matrix": render_matrix(matrix),
        "refusals": render_refusals(matrix),
    }
    for name, begin, end in MATRIX_BLOCKS:
        producer = splice(producer, begin, end, blocks[name])
    return [(NAMING, naming), (PRODUCER_CONSUMER, producer)]


def staleness() -> list[str]:
    """Marker blocks that do not match the TOML render. Empty when fresh."""
    try:
        pages = rendered_pages()
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        return [f"dictionary tables: {exc}"]
    problems = []
    for path, expected in pages:
        if path.read_text(encoding="utf-8") != expected:
            problems.append(
                f"{path.relative_to(ROOT)}: generated table is stale; "
                "run python3 scripts/generate_dictionary_tables.py"
            )
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        problems = staleness()
        if problems:
            raise SystemExit("\n".join(problems))
        print("dictionary tables: ok")
        return
    for path, expected in rendered_pages():
        path.write_text(expected, encoding="utf-8")
    print("dictionary tables: wrote identity dictionary and producer/consumer matrix")


if __name__ == "__main__":
    main()
