"""Validate the transport stage registry independently of the analyze axes."""

import sys
from pathlib import Path

import tomllib
from test_evidence import resolve_python_test

root = Path(__file__).resolve().parents[1]
registry = tomllib.loads((root / "parity/transport_stages.toml").read_text())
errors = []
if registry.get("version") != 1 or registry.get("default") != "closed":
    errors.append("transport stages require version 1 and default=closed")
seen = set()
for route in registry.get("routes", []):
    name = route.get("route")
    if not name or name in seen:
        errors.append(f"duplicate or missing transport route: {name}")
    seen.add(name)
    if route.get("stage") not in {
        "identify",
        "representation",
        "evaluate",
        "uncertainty",
    }:
        errors.append(f"invalid stage: {name}")
    if route.get("status") == "licensed":
        for key in (
            "graph",
            "evidence",
            "functional",
            "guarantee",
            "evidence_test",
            "evidence_assertion",
        ):
            if not route.get(key):
                errors.append(f"{name}: missing {key}")
        path, assertion = route.get("evidence_test"), route.get("evidence_assertion")
        if path and assertion:
            errors.extend(resolve_python_test(root / path, assertion))
        if (
            route.get("stage") == "identify"
            and route.get("guarantee") != "sound_incomplete"
        ):
            errors.append(f"{name}: implemented identifier must not claim completeness")
    elif route.get("status") != "closed" or not route.get("reason_code"):
        errors.append(f"{name}: expected licensed or a reason-backed closed contract")
required = {
    "antecedent.transport.identify",
    "antecedent.transport.reload_lowered_expression",
}
if not required.issubset(seen):
    errors.append("missing current public transport stage")
if errors:
    print("Transport stage gate FAILED:\n" + "\n".join(f" - {e}" for e in errors))
    sys.exit(1)
print(f"Transport stage contracts OK ({len(seen)} routes; unlisted routes closed)")
