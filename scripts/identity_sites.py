#!/usr/bin/env python3
"""Identity-site inventory: every place a collection is canonicalized into an identity.

`formal/identity_sites.toml` lists each site (a production function), the collections it
canonicalizes and how (`ordered`, `set`, `multiset`), and the checks that cover it on
production code. This script checks the inventory against the repository:

    python3 scripts/identity_sites.py check       # every site resolves and every check exists
    python3 scripts/identity_sites.py summary     # counts by status and class
    python3 scripts/identity_sites.py --self-test # the check rejects deliberately broken inputs

A site's `item` must name a Rust item in its `file` (scripts/rust_items.py locators). A
`checked` site names at least one check, each of which must exist: `adversarial:<case id>`
in parity/adversarial_campaign.toml, `kani:<harness>` declared as a #[kani::proof] in
formal/kani/src, `assurance:<record id>` in parity/assurance_records.toml, or
`unit:<file>::<test fn>` for an in-crate #[test] on a site no public API reaches. A `listed`
site has no checks and says why.
"""

from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import rust_items
from _repo import REPO_ROOT, load_toml

INVENTORY = Path("formal/identity_sites.toml")
CAMPAIGN = Path("parity/adversarial_campaign.toml")
RECORDS = Path("parity/assurance_records.toml")
KANI = Path("formal/kani/src")
CLASSES = {"ordered", "set", "multiset"}
STATUSES = {"checked", "listed"}
SITE_KEYS = {"id", "file", "item", "feeds", "released", "status", "checks", "collections"}
OPTIONAL_SITE_KEYS = {"reason", "note"}
COLLECTION_KEYS = {"name", "class", "canonical", "concern"}


def _kani_harnesses(root: Path) -> set[str]:
    names: set[str] = set()
    for path in (root / KANI).glob("*.rs"):
        names |= set(re.findall(r"#\[kani::proof\][^{]*?fn\s+(\w+)", path.read_text(encoding="utf-8"), re.S))
    return names


def validate(root: Path, inventory: dict, campaign: dict, records: dict) -> list[str]:
    errors: list[str] = []
    cases = {case["id"] for case in campaign.get("cases", [])}
    record_ids = {record["id"] for record in records.get("records", [])}
    harnesses = _kani_harnesses(root)
    parsed: dict[str, rust_items.File | None] = {}
    seen: set[str] = set()
    sites = inventory.get("sites", [])
    if not sites:
        errors.append("no identity sites")
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
                unit = root / rel
                if not unit.is_file() or not re.search(
                    rf"#\[test\][^{{]*?fn\s+{re.escape(test)}\b", unit.read_text(encoding="utf-8"), re.S
                ):
                    errors.append(f"{sid}: check {check!r} names no #[test] fn")
                continue
            known = {"adversarial": cases, "kani": harnesses, "assurance": record_ids}.get(kind)
            if known is None:
                errors.append(f"{sid}: unknown check kind in {check!r}")
            elif name not in known:
                errors.append(f"{sid}: check {check!r} does not exist")
        if not site["collections"]:
            errors.append(f"{sid}: no collections")
        for collection in site["collections"]:
            if set(collection) != COLLECTION_KEYS:
                errors.append(f"{sid}: collection fields must be exactly {sorted(COLLECTION_KEYS)}")
            elif collection["class"] not in CLASSES:
                errors.append(f"{sid}: class {collection['class']!r} is not one of {sorted(CLASSES)}")
    return errors


def _load(root: Path) -> tuple[dict, dict, dict]:
    return load_toml(root / INVENTORY), load_toml(root / CAMPAIGN), load_toml(root / RECORDS)


def check(root: Path = REPO_ROOT) -> int:
    errors = validate(root, *_load(root))
    for error in errors:
        print(f"identity_sites: {error}", file=sys.stderr)
    if not errors:
        print(f"identity_sites: {len(_load(root)[0]['sites'])} sites resolve")
    return 1 if errors else 0


def summary(root: Path = REPO_ROOT) -> int:
    inventory = _load(root)[0]
    status: dict[str, int] = {}
    classes: dict[str, int] = {}
    concerns = 0
    for site in inventory["sites"]:
        status[site["status"]] = status.get(site["status"], 0) + 1
        for collection in site["collections"]:
            classes[collection["class"]] = classes.get(collection["class"], 0) + 1
            concerns += bool(collection["concern"])
    print(f"sites by status: {status}; collections by class: {classes}; with a concern: {concerns}")
    return 0


def self_test(root: Path = REPO_ROOT) -> int:
    inventory, campaign, records = _load(root)
    if validate(root, inventory, campaign, records):
        print("identity_sites self-test: the inventory itself does not pass", file=sys.stderr)
        return 1
    checked = next(i for i, s in enumerate(inventory["sites"]) if s["status"] == "checked")

    def broken(edit) -> dict:
        data = copy.deepcopy(inventory)
        edit(data["sites"][checked])
        return data

    cases = [
        ("missing item", broken(lambda s: s.update(item="fn:no_such_function#0"))),
        ("missing file", broken(lambda s: s.update(file="crates/no_such_crate/src/lib.rs"))),
        ("unknown check", broken(lambda s: s["checks"].append("adversarial:no.such.case"))),
        ("unknown harness", broken(lambda s: s["checks"].append("kani:no_such_harness"))),
        ("unknown unit test", broken(lambda s: s["checks"].append(f"unit:{s['file']}::no_such_test"))),
        ("checked without checks", broken(lambda s: s.update(checks=[]))),
        ("bad class", broken(lambda s: s["collections"][0].update({"class": "bag"}))),
        ("listed with checks", broken(lambda s: s.update(status="listed"))),
    ]
    failures = [name for name, data in cases if not validate(root, data, campaign, records)]
    if failures:
        print(f"identity_sites self-test: accepted broken inputs {failures}", file=sys.stderr)
        return 1
    print(f"identity_sites self-test: {len(cases)} broken inputs rejected")
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
