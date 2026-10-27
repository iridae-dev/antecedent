#!/usr/bin/env python3
"""Check that every method child names a concrete acceptance fixture and artifact consumer.

Every leaf of parity/method_children/children.toml, whatever its gap, must carry

  fixtures   `;`-separated path[::test] tokens. Each path exists, a registered Cargo target,
             python test, gate script or evidence record runs it, and a named test is a
             real test function. At least one test file must mention the method it
             accepts (its target, module, estimator id or route) so a fixture cannot be
             borrowed from an unrelated method.
  consumer   `;`-separated path[::symbol] tokens. Each path exists and mentions its symbol,
             and at least one is a non-test source: the code that reads the artifact.

A family id, a prose description or `unlisted` is a placeholder and fails. A child whose route
is listed must name a route or method of the route manifest.

    python3 scripts/check_method_children.py
    python3 scripts/check_method_children.py --self-test
"""

from __future__ import annotations

import sys
import tomllib

from _repo import REPO_ROOT as ROOT
from method_evidence import UNLISTED, Evidence

CHILDREN = "parity/method_children/children.toml"
MANIFEST = "parity/capabilities/route_manifest.toml"
GAPS = {
    "none",
    "target",
    "implementation",
    "statistical",
    "validation",
    "temporal-identification",
    "theory",
}


def manifest_ids(ev: Evidence) -> set[str]:
    data = tomllib.loads(ev.text(MANIFEST))
    return {r["id"] for r in data.get("route", [])} | {m["id"] for m in data.get("method", [])}


def child_findings(ev: Evidence, child: dict, routes: set[str]) -> list[str]:
    cid = child["id"]
    out: list[str] = []
    if child["gap"] not in GAPS:
        out.append(f"{cid}: gap {child['gap']!r} is outside the vocabulary")
    route = child["route"]
    if route not in UNLISTED and route not in routes and not ev.repo.exists(route.split(" ")[0]):
        out.append(f"{cid}: route {route!r} is not in the route manifest")
    fixtures = ev.tokens(child["fixtures"])
    consumers = ev.tokens(child["consumer"])
    if not fixtures or any(f in UNLISTED for f in fixtures):
        out.append(f"{cid}: no acceptance fixture is named")
    if not consumers or any(c in UNLISTED for c in consumers):
        out.append(f"{cid}: no artifact consumer is named")
    for token in fixtures:
        if token in UNLISTED:
            continue
        out.extend(f"{cid}: {p}" for p in ev.fixture_findings(token))
    for token in consumers:
        if token in UNLISTED:
            continue
        out.extend(f"{cid}: {p}" for p in ev.consumer_findings(token))
    code = [t for t in consumers if t not in UNLISTED and not ev.is_test_path(ev.split(t)[0])]
    if consumers and not code:
        out.append(f"{cid}: every consumer is a test; name the source that reads the artifact")
    anchors = ev.anchors(child)
    sources = [ev.split(t)[0] for t in fixtures if ev.split(t)[0].endswith((".rs", ".py"))]
    if sources and anchors and not any(ev.mentions(s, anchors) for s in sources):
        out.append(
            f"{cid}: no fixture file mentions the method it accepts (looked for {sorted(anchors)[:4]})"
        )
    return out


def findings(ev: Evidence) -> list[str]:
    children = tomllib.loads(ev.text(CHILDREN))["child"]
    routes = manifest_ids(ev)
    out: list[str] = []
    seen: set[str] = set()
    for child in children:
        if child["id"] in seen:
            out.append(f"{child['id']}: duplicate child")
        seen.add(child["id"])
        out.extend(child_findings(ev, child, routes))
    return out


def self_test() -> int:
    import tempfile
    from pathlib import Path

    def write(root: Path, rel: str, body: str) -> None:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write(root, "crates/c/Cargo.toml", '[package]\nname = "c"\n')
        write(root, "crates/c/tests/t.rs", "// Widget\n#[test]\nfn works() {}\nfn helper() {}\n")
        write(root, "crates/c/src/reader.rs", "pub fn read_artifact() {}\n")
        write(root, "parity/v3_test_registry.toml", "schema = 1\n")
        write(root, MANIFEST, '[[route]]\nid = "r.one"\n')
        ev = Evidence(root)
        base = {
            "id": "P.x",
            "gap": "none",
            "route": "r.one",
            "target": "Widget",
            "module": "none",
            "inference": "x",
            "fixtures": "crates/c/tests/t.rs::works",
            "consumer": "crates/c/src/reader.rs::read_artifact",
        }
        routes = {"r.one"}
        assert child_findings(ev, base, routes) == [], child_findings(ev, base, routes)
        for field, value, needle in (
            ("fixtures", "unlisted", "no acceptance fixture"),
            ("consumer", "FAMILY", "not a path"),
            ("consumer", "crates/c/tests/t.rs", "every consumer is a test"),
            ("fixtures", "crates/c/tests/t.rs::helper", "not a test function"),
            ("fixtures", "crates/c/tests/t.rs::missing", "defines no"),
            ("consumer", "crates/c/src/reader.rs::absent", "does not mention"),
            ("route", "r.two", "not in the route manifest"),
            ("target", "Gadget", "mentions the method"),
        ):
            got = child_findings(ev, base | {field: value}, routes)
            assert any(needle in line for line in got), (field, value, got)
    print("check_method_children self-test passed")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    ev = Evidence(ROOT)
    out = findings(ev)
    for line in out:
        print(f"FAIL: {line}")
    count = len(tomllib.loads(ev.text(CHILDREN))["child"])
    print(f"method children: {count} checked, {len(out)} finding(s)")
    return 1 if out else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
