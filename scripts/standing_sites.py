#!/usr/bin/env python3
"""Standing-site inventory: every place a claim's standing is set, transformed, composed,
carried or restored.

`formal/standing_sites.toml` lists each site (a production function), the standing facets it
touches, what it does to them, and the checks that cover it on production code. This script
checks the inventory against the repository:

    python3 scripts/standing_sites.py check       # every site resolves and every check exists
    python3 scripts/standing_sites.py summary     # counts by status, action and facet
    python3 scripts/standing_sites.py --self-test # the check rejects deliberately broken inputs

A site's `item` must name a Rust item in its `file` (scripts/rust_items.py locators). A
`checked` site names at least one check, each of which must exist: `aeneas:<theorem>` declared
in formal/rust/AntecedentRust (a theorem about the Charon/Aeneas translation),
`kani:<harness>` declared as a #[kani::proof] in formal/kani/src, `adversarial:<case id>` in
parity/adversarial_campaign.toml, `assurance:<record id>` in parity/assurance_records.toml, or
`unit:<file>::<test fn>` for a #[test]. A `listed` site has no checks and says why.
"""

from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import rust_items
from _repo import REPO_ROOT, load_toml

INVENTORY = Path("formal/standing_sites.toml")
CAMPAIGN = Path("parity/adversarial_campaign.toml")
RECORDS = Path("parity/assurance_records.toml")
KANI = Path("formal/kani/src")
AENEAS = Path("formal/rust/AntecedentRust")
FACETS = {
    "kind", "identification", "domains", "support", "trust", "calibration", "revalidation", "obligations",
}
ACTIONS = {"set", "transform", "compose", "carry", "restore", "check"}
STATUSES = {"checked", "listed"}
SITE_KEYS = {"id", "file", "item", "facets", "action", "released", "status", "checks"}
OPTIONAL_SITE_KEYS = {"reason", "note", "concern"}


def _kani_harnesses(root: Path) -> set[str]:
    names: set[str] = set()
    for path in (root / KANI).glob("*.rs"):
        names |= set(re.findall(r"#\[kani::proof\][^{]*?fn\s+(\w+)", path.read_text(encoding="utf-8"), re.S))
    return names


def _aeneas_theorems(root: Path) -> set[str]:
    """Fully qualified theorems of the hand-written proofs about the translated Rust."""
    names: set[str] = set()
    for path in (root / AENEAS).glob("*.lean"):
        text = re.sub(r"/-.*?-/|--[^\n]*", " ", path.read_text(encoding="utf-8"), flags=re.S)
        spaces = re.findall(r"^namespace\s+([\w.]+)", text, re.M)
        prefix = f"{spaces[0]}." if len(spaces) == 1 else ""
        for name in re.findall(r"^\s*(?:@\[[^\]]*\]\s*)?(?:private\s+)?theorem\s+([\w.']+)", text, re.M):
            names.add(prefix + name)
    return names


def _unit_test_exists(root: Path, rel: str, test: str) -> bool:
    unit = root / rel
    return unit.is_file() and re.search(
        rf"#\[test\][^{{]*?fn\s+{re.escape(test)}\b", unit.read_text(encoding="utf-8"), re.S
    ) is not None


def validate(root: Path, inventory: dict, campaign: dict, records: dict) -> list[str]:
    errors: list[str] = []
    known = {
        "adversarial": {case["id"] for case in campaign.get("cases", [])},
        "assurance": {record["id"] for record in records.get("records", [])},
        "kani": _kani_harnesses(root),
        "aeneas": _aeneas_theorems(root),
    }
    parsed: dict[str, rust_items.File | None] = {}
    seen: set[str] = set()
    located: set[tuple[str, str]] = set()
    sites = inventory.get("sites", [])
    if not sites:
        errors.append("no standing sites")
    for site in sites:
        sid = site.get("id", "<no id>")
        missing = SITE_KEYS - set(site)
        extra = set(site) - SITE_KEYS - OPTIONAL_SITE_KEYS
        if missing or extra:
            errors.append(f"{sid}: missing {sorted(missing)} unexpected {sorted(extra)}")
            continue
        if sid in seen:
            errors.append(f"{sid}: duplicate id")
        seen.add(sid)
        if (site["file"], site["item"]) in located:
            errors.append(f"{sid}: {site['file']} {site['item']} is listed twice")
        located.add((site["file"], site["item"]))
        path = root / site["file"]
        if not path.is_file():
            errors.append(f"{sid}: no file {site['file']}")
        else:
            if site["file"] not in parsed:
                try:
                    parsed[site["file"]] = rust_items.File(path.read_text(encoding="utf-8"))
                except rust_items.ParseError:
                    parsed[site["file"]] = None
            source = parsed[site["file"]]
            if source is None:
                errors.append(f"{sid}: {site['file']} does not parse")
            elif source.digest(site["item"], "full") is None:
                errors.append(f"{sid}: no Rust item {site['item']} in {site['file']}")
        if not site["facets"] or set(site["facets"]) - FACETS:
            errors.append(f"{sid}: facets must be a nonempty subset of {sorted(FACETS)}")
        if site["action"] not in ACTIONS:
            errors.append(f"{sid}: action must be one of {sorted(ACTIONS)}")
        if not isinstance(site["released"], bool):
            errors.append(f"{sid}: released is true or false")
        if site["status"] not in STATUSES:
            errors.append(f"{sid}: status must be one of {sorted(STATUSES)}")
        if site["status"] == "checked" and not site["checks"]:
            errors.append(f"{sid}: a checked site names its checks")
        if site["status"] == "listed" and (site["checks"] or not site.get("reason")):
            errors.append(f"{sid}: a listed site has no checks and gives a reason")
        for check in site["checks"]:
            kind, _, name = check.partition(":")
            if kind == "unit":
                rel, _, test = name.rpartition("::")
                if not _unit_test_exists(root, rel, test):
                    errors.append(f"{sid}: check {check!r} names no #[test] fn")
            elif kind not in known:
                errors.append(f"{sid}: unknown check kind in {check!r}")
            elif name not in known[kind]:
                errors.append(f"{sid}: check {check!r} does not exist")
    return errors


def _load(root: Path) -> tuple[dict, dict, dict]:
    return load_toml(root / INVENTORY), load_toml(root / CAMPAIGN), load_toml(root / RECORDS)


def check(root: Path = REPO_ROOT) -> int:
    errors = validate(root, *_load(root))
    for error in errors:
        print(f"standing_sites: {error}", file=sys.stderr)
    if not errors:
        print(f"standing_sites: {len(_load(root)[0]['sites'])} sites resolve")
    return 1 if errors else 0


def summary(root: Path = REPO_ROOT) -> int:
    sites = _load(root)[0]["sites"]

    def count(key) -> dict[str, int]:
        out: dict[str, int] = {}
        for site in sites:
            for value in key(site):
                out[value] = out.get(value, 0) + 1
        return dict(sorted(out.items()))

    print(f"sites by status: {count(lambda s: [s['status']])}")
    print(f"released by status: {count(lambda s: [s['status']] if s['released'] else [])}")
    print(f"by action: {count(lambda s: [s['action']])}")
    print(f"by facet: {count(lambda s: s['facets'])}")
    print(f"by check kind: {count(lambda s: sorted({c.partition(':')[0] for c in s['checks']}))}")
    print(f"with a concern: {sum(bool(s.get('concern')) for s in sites)}; fixed: {sum(bool(s.get('note', '').startswith('Fixed')) for s in sites)}")
    return 0


def self_test(root: Path = REPO_ROOT) -> int:
    inventory, campaign, records = _load(root)
    if validate(root, inventory, campaign, records):
        print("standing_sites self-test: the inventory itself does not pass", file=sys.stderr)
        return 1
    checked = next(i for i, s in enumerate(inventory["sites"]) if s["status"] == "checked")
    listed = next(i for i, s in enumerate(inventory["sites"]) if s["status"] == "listed")

    def broken(edit, index: int = checked) -> dict:
        data = copy.deepcopy(inventory)
        edit(data["sites"][index])
        return data

    def duplicate() -> dict:
        data = copy.deepcopy(inventory)
        data["sites"].append(copy.deepcopy(data["sites"][checked]) | {"id": "duplicate.site"})
        return data

    cases = [
        ("missing item", broken(lambda s: s.update(item="fn:no_such_function#0"))),
        ("missing file", broken(lambda s: s.update(file="crates/no_such_crate/src/lib.rs"))),
        ("unknown case", broken(lambda s: s["checks"].append("adversarial:no.such.case"))),
        ("unknown harness", broken(lambda s: s["checks"].append("kani:no_such_harness"))),
        ("unknown theorem", broken(lambda s: s["checks"].append("aeneas:AntecedentRust.Standing.no_such_spec"))),
        ("unknown record", broken(lambda s: s["checks"].append("assurance:no.such.record"))),
        ("unknown unit test", broken(lambda s: s["checks"].append(f"unit:{s['file']}::no_such_test"))),
        ("unknown check kind", broken(lambda s: s["checks"].append("lean:Antecedent.Standing.relay_le"))),
        ("checked without checks", broken(lambda s: s.update(checks=[]))),
        ("listed with checks", broken(lambda s: s.update(status="listed"))),
        ("listed without a reason", broken(lambda s: s.pop("reason"), listed)),
        ("unknown facet", broken(lambda s: s["facets"].append("prestige"))),
        ("unknown action", broken(lambda s: s.update(action="raise"))),
        ("one item listed twice", duplicate()),
    ]
    failures = [name for name, data in cases if not validate(root, data, campaign, records)]
    if failures:
        print(f"standing_sites self-test: accepted broken inputs {failures}", file=sys.stderr)
        return 1
    print(f"standing_sites self-test: {len(cases)} broken inputs rejected")
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["check"]:
        return check()
    if argv[:1] == ["summary"]:
        return summary()
    if argv[:1] == ["--self-test"]:
        return self_test()
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
