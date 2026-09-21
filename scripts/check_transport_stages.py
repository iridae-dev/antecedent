"""Validate the transport stage registry independently of the analyze axes."""

import sys
from pathlib import Path

import tomllib
from test_evidence import resolve_python_test, resolve_rust_test

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
        "prepare",
        "consume",
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
            evidence_path = root / path
            if evidence_path.suffix == ".rs":
                _, problems = resolve_rust_test(evidence_path, assertion)
                errors.extend(problems)
            elif evidence_path.suffix == ".py":
                errors.extend(resolve_python_test(evidence_path, assertion))
            else:
                errors.append(f"{name}: evidence must be a collected Rust or Python test")
        if (
            route.get("stage") == "identify"
            and route.get("guarantee") != "sound_incomplete"
        ):
            pinned = {
                "antecedent.transport.identify_classical": (
                    "complete_in_classical_evidence_scope", "classical_complete_source_experimental_family",
                    "https://arxiv.org/abs/1312.7485v1"),
                "antecedent.transport.identify_meta": (
                    "complete_in_classical_meta_evidence_scope", "classical_complete_multi_source_experimental_families",
                    "https://proceedings.mlr.press/v31/bareinboim13a.pdf"),
            }
            if pinned.get(name) != (route.get("guarantee"), route.get("evidence"), route.get("reference")):
                errors.append(f"{name}: completeness requires its pinned classical evidence scope")
            path, assertion = route.get("conformance_test"), route.get("conformance_assertion")
            if not path or not assertion:
                errors.append(f"{name}: completeness requires consuming Rust branch conformance")
            else:
                _, problems = resolve_rust_test(root / path, assertion)
                errors.extend(problems)
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
