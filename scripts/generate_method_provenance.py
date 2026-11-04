#!/usr/bin/env python3
"""Generate the method-provenance table results cite their sources from.

A result names the provenance records behind the route it executed: the identifier, the
estimator and any capability. The links already exist and this script only joins them:

- ``crates/antecedent/src/strategy_table/ids.rs``: each identifier and estimator row carries a
  ``provenance`` pair (family, step), resolved through ``scripts/provenance_links.py``
  (``ALIAS``, ``FAMILY``);
- ``parity/method_specs/specs.toml``: an admitted method's ``provenance`` list, keyed by the
  estimator id its ``estimator`` field names, or by the capability id its ``route`` names;
- ``parity/capability_records.toml``: a capability id whose provenance record has the same stem;
- ``provenance/*.toml``: per record, the published sources, the implementation deviations,
  what the bounded method review leaves out, and the record digest;
- ``provenance/citation_metadata.json``: authors and year retained from the DOI registry by
  ``scripts/verify_provenance_citations.py``, and whether the DOI resolved to the recorded title.

Only citations and short deviation text are embedded, never whole records. A wrapper record
(``non_algorithm_review.kind = "wrapper"``) brings in the records that carry its numerics.

    python3 scripts/generate_method_provenance.py           # write
    python3 scripts/generate_method_provenance.py --check   # fail when the snapshot is stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _repo import REPO_ROOT as ROOT  # noqa: E402
from provenance_links import ALIAS, FAMILY, method_review_digest  # noqa: E402
from verify_provenance_citations import verdict  # noqa: E402

SNAPSHOT = ROOT / "crates/antecedent-io-core/data/method_provenance.json"
IDS = ROOT / "crates/antecedent/src/strategy_table/ids.rs"
SPECS = ROOT / "parity/method_specs/specs.toml"
CAPABILITIES = ROOT / "parity/capability_records.toml"
CITATIONS = ROOT / "provenance/citation_metadata.json"

ROW = re.compile(r"\b(IdentifierData|EstimatorData)\s*\{(?P<body>[^{}]*)\}", re.S)
NAME = re.compile(r'\bname:\s*"([^"]+)"')
PAIR = re.compile(r'\bprovenance:\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)')
SAME = re.compile(r'\bprovenance:\s*same_step\(\s*"([^"]+)"\s*\)')
# The sentence of a bounded review's scope that names what the review does not cover.
NOT_COVERED = re.compile(r"(?:^|(?<=[.;]\s))Not (?P<rest>[^.]+(?:\.[0-9][^.]*)*)\.")


def load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def strategy_rows() -> dict[str, dict[str, tuple[str, str]]]:
    """``{"identifier": {name: (family, step)}, "estimator": {...}}`` from the strategy table."""
    rows: dict[str, dict[str, tuple[str, str]]] = {"identifier": {}, "estimator": {}}
    for match in ROW.finditer(IDS.read_text(encoding="utf-8")):
        body = match.group("body")
        name = NAME.search(body)
        pair = PAIR.search(body)
        same = SAME.search(body)
        if not name or not (pair or same):
            continue
        role = "identifier" if match.group(1) == "IdentifierData" else "estimator"
        rows[role][name.group(1)] = (
            (pair.group(1), pair.group(2)) if pair else (same.group(1), same.group(1))
        )
    return rows


def resolve_pair(family: str, step: str, stems: set[str]) -> list[str]:
    """The records a (family, step) pointer names: the step's own record, else the family's."""
    for candidate in (step, ALIAS.get(step, step)):
        if candidate in stems:
            return [candidate]
    resolved = ALIAS.get(family, family)
    if resolved in stems:
        return [resolved]
    return [member for member in FAMILY.get(family, ()) if member in stems]


def scope_limits(record: dict) -> list[str]:
    """What the bounded method review states it does not cover, one item per sentence."""
    scope = (record.get("method_review") or {}).get("scope") or ""
    return [
        f"not covered by the method review: {m.group('rest').strip()}"
        for m in NOT_COVERED.finditer(scope)
    ]


def sources(record: dict, cache: dict) -> list[dict]:
    out = []
    for paper in record.get("papers") or []:
        source: dict = {"title": paper["title"].strip()}
        doi = paper.get("doi")
        meta = cache.get(doi.lower(), {}) if doi else {}
        if meta.get("authors"):
            source["authors"] = list(meta["authors"])
        if meta.get("year"):
            source["year"] = int(meta["year"])
        if doi:
            source["doi"] = doi
            source["doi_verified"] = (
                verdict({"doi": doi, "title": paper["title"]}, cache)
                == "bibliographic_identity_verified"
            )
        if paper.get("url"):
            source["url"] = paper["url"]
        source["sections"] = [s.strip() for s in paper.get("sections") or []]
        out.append(source)
    return out


def review_kind(record: dict, digest: str) -> str:
    """Which bounded review still attests this record's text, if any."""
    for key in ("method_review", "non_algorithm_review"):
        review = record.get(key)
        if isinstance(review, dict) and review.get("record_sha256") == digest:
            return key
    return "none"


def record_entry(stem: str, record: dict, cache: dict) -> dict:
    digest = method_review_digest(record)
    review = record.get("non_algorithm_review") or {}
    entry: dict = {
        "id": stem,
        "digest": digest,
        "review": review_kind(record, digest),
        "sources": sources(record, cache),
        "deviations": [d.strip() for d in record.get("implementation_deviations") or []],
        "scope_limits": scope_limits(record),
    }
    if review.get("kind") == "wrapper":
        entry["covered_by"] = list(review.get("covered_by") or [])
    return entry


def spec_links(stems: set[str], estimators: set[str]) -> tuple[dict, dict]:
    """Records the method specifications link, by estimator id and by capability route."""
    by_estimator: dict[str, list[str]] = {}
    by_route: dict[str, list[str]] = {}
    for spec in load(SPECS).get("spec") or []:
        if spec.get("admission") not in ("supported", "not_supported"):
            continue
        links = [link for link in spec.get("provenance") or [] if link in stems]
        estimator = str(spec.get("estimator", "")).split(":", 1)[0].strip()
        if estimator in estimators:
            by_estimator.setdefault(estimator, []).extend(links)
        route = str(spec.get("route", ""))
        if route and not route.startswith(("supported.", "parity/")) and route != "none":
            by_route.setdefault(route, []).extend(links)
    return by_estimator, by_route


def dedupe(items: list[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


def build() -> dict:
    files = sorted(p for p in (ROOT / "provenance").glob("*.toml") if p.name != "_template.toml")
    records = {p.stem: load(p) for p in files}
    stems = set(records)
    cache = json.loads(CITATIONS.read_text(encoding="utf-8")) if CITATIONS.is_file() else {}

    rows = strategy_rows()
    by_estimator, by_route = spec_links(stems, set(rows["estimator"]))
    table: dict[str, dict[str, list[str]]] = {"identifier": {}, "estimator": {}, "capability": {}}
    unlinked: dict[str, list[str]] = {"identifier": [], "estimator": []}
    for role in ("identifier", "estimator"):
        for name, (family, step) in sorted(rows[role].items()):
            # The strategy-table pointer names the record of the executed estimator; the
            # specifications list every record covering the module, so they are read only when
            # the pointer names no record.
            linked = resolve_pair(family, step, stems) or (
                by_estimator.get(name, []) if role == "estimator" else []
            )
            if linked:
                table[role][name] = dedupe(linked)
            else:
                unlinked[role].append(name)
    capability_ids = {r["id"] for r in load(CAPABILITIES).get("record") or []}
    for capability in sorted(capability_ids | set(by_route)):
        own = ALIAS.get(capability, capability)
        linked = ([own] if own in stems else []) + by_route.get(capability, [])
        if linked:
            table["capability"][capability] = dedupe(linked)

    cited = {stem for keyed in table.values() for linked in keyed.values() for stem in linked}
    # A wrapper's numerics live in its covered_by records; cite those too.
    pending = sorted(cited)
    while pending:
        stem = pending.pop()
        review = records[stem].get("non_algorithm_review") or {}
        for covered in review.get("covered_by") or [] if review.get("kind") == "wrapper" else []:
            if covered in stems and covered not in cited:
                cited.add(covered)
                pending.append(covered)
    return {
        "format": 1,
        "records": [record_entry(stem, records[stem], cache) for stem in sorted(cited)],
        **table,
        # Strategy-table ids whose provenance pointer resolves to no record: a result reports
        # them as unlinked rather than citing a guess.
        "unlinked": unlinked,
    }


def render(table: dict) -> str:
    return json.dumps(table, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="fail when the snapshot is stale")
    args = parser.parse_args()
    expected = render(build())
    if args.check:
        if not SNAPSHOT.is_file() or SNAPSHOT.read_text(encoding="utf-8") != expected:
            print(
                f"{SNAPSHOT.relative_to(ROOT)} is stale; run python3 scripts/generate_method_provenance.py"
            )
            return 1
        print("method provenance table: current")
        return 0
    SNAPSHOT.write_text(expected, encoding="utf-8", newline="\n")
    print(f"wrote {SNAPSHOT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
