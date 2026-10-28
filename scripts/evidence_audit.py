#!/usr/bin/env python3
"""Join existing test, claim, fixture and provenance registries without licensing.

report is source/metadata inspection, never execution verification. --strict fails
on every reported gap. scientific-filter prints the nextest expression selecting
the cited Rust scientific assertions (test.sh runs it as one workspace run, after
checking it against a nextest listing); run-scientific executes exact registered
scientific tests through the existing evidence runner, or only the cited Python
ones with --python-only; ignored measurements remain forbidden.
No command runs calibration, replay or performance acceptance measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import check_evidence_ledger
import tomllib
from _repo import REPO_ROOT
from evidence_kinds import (
    EVIDENCE_KINDS,
    KNOWN_TRUTH_KINDS,
    evidence_roles,
    fixture_review_problems,
    inline_reference_problems,
    kind_problems,
    software_only_roles,
)
from provenance_links import method_review_problems, non_algorithm_review_problems
from scenario_matrix import build as build_scenario_matrix
from test_evidence import (
    registered_calibration_acceptance,
    registered_rust_assertions,
    nextest_selection,
    calibration_discovery_problems,
    python_test_markers,
    python_test_declarations,
    script_test_declaration_problems,
    rust_test_declaration_problems,
    static_python_test,
)
from verify_provenance_citations import CACHE as CITATION_CACHE
from verify_provenance_citations import verdict as citation_verdict

AXES = ("query", "graph_class", "structure", "inference", "validation", "estimators")


def disposition_problems(root: Path, cases: list[dict]) -> list[str]:
    problems = []
    seen = set()
    for case in cases:
        previous = case.get("previous")
        if not previous or previous in seen:
            problems.append(f"invalid or duplicated previous assertion: {previous}")
        seen.add(previous)
        additional = case.get("additional_replacements", [])
        if not isinstance(additional, list) or any(
            not isinstance(token, str) for token in additional
        ):
            problems.append(f"invalid additional replacements: {previous}")
            additional = []
        for token in [case.get("replacement", ""), *additional]:
            rel, _, name = token.partition("::")
            path = in_repo(root, rel)
            if not path or not path.is_file() or not name:
                problems.append(f"missing replacement assertion: {token}")
                continue
            findings = (
                rust_test_declaration_problems(path, name)
                if path.suffix == ".rs"
                else script_test_declaration_problems(path, name)
                if rel.startswith("scripts/")
                else static_python_test(path, name)
            )
            if case.get("disposition") == "explicit_calibration_only":
                if not any("ignore" in f for f in findings):
                    problems.append(
                        f"measurement replacement is not explicitly ignored: {token}"
                    )
                findings = [f for f in findings if "ignore" not in f]
            problems += [f"{token}: {finding}" for finding in findings]
        if not str(case.get("reason", "")).strip():
            problems.append(f"coverage disposition lacks a reason: {previous}")
    return problems


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tables(value: object, location: str = ""):
    """Walk nested TOML tables, retaining the exact registry coordinate."""
    if isinstance(value, dict):
        yield location, value
        for key, child in value.items():
            yield from tables(child, f"{location}.{key}".lstrip("."))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from tables(child, f"{location}[{index}]")


def in_repo(root: Path, rel: str) -> Path | None:
    path = (root / rel).resolve()
    return path if path.is_relative_to(root.resolve()) else None


def fixture_report(root: Path, path: Path) -> dict:
    root, path = root.resolve(), path.resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    oracle = data.get("oracle")
    oracle = oracle if isinstance(oracle, dict) else {}
    references = [
        v
        for loc, v in tables(data)
        if isinstance(v.get("command"), str) and not loc.startswith("historical")
    ]
    # Only explicit retained file paths count. Nearby scripts alone do not prove
    # that a frozen output can be reproduced, and a command is never executed here.
    candidates = set()
    for block in [oracle, *references, data.get("generation", {})]:
        for field in ("generator", "script", "command"):
            for name in re.findall(
                r"(?:[\w.-]+/)+[\w.-]+\.(?:py|R)\b", str(block.get(field, ""))
            ):
                candidate = in_repo(root, name)
                if candidate and candidate.is_file():
                    candidates.add(candidate)
    gaps = []
    gaps.extend(fixture_review_problems(root, data))
    roles = data.get("evidence_roles")
    if roles is not None and (
        not isinstance(roles, list)
        or not roles
        or any(
            role not in ("software", "scientific", "statistical_smoke", "process")
            for role in roles
        )
    ):
        gaps.append("invalid_fixture_evidence_roles")
    software_only = software_only_roles(roles)
    if software_only:
        source, _, assertion = str(data.get("evidence_source", "")).partition("::")
        source_path = in_repo(root, source)
        if not source_path or not source_path.is_file() or not assertion:
            gaps.append("software_fixture_missing_exact_assertion")
        else:
            gaps.extend(
                rust_test_declaration_problems(source_path, assertion)
                if source_path.suffix == ".rs"
                else static_python_test(source_path, assertion)
            )
        for field in ("evidence_proposition", "limitations"):
            if not isinstance(data.get(field), str) or not data[field].strip():
                gaps.append(f"software_fixture_missing_{field}")
    if not oracle.get("kind") and not software_only:
        gaps.append("oracle_origin_unrecorded")
    if oracle.get("kind") == "external_package" and not candidates:
        gaps.append("external_generator_not_retained")
    if any(b.get("generator_retained") is False for b in references):
        gaps.append("historical_generator_deleted")
    return {
        "path": (path.parent if path.name == "expected.json" else path)
        .relative_to(root)
        .as_posix(),
        "sha256": digest(path),
        "oracle": oracle,
        "evidence_roles": roles,
        "evidence_kind": data.get("evidence_kind"),
        "evidence_source": data.get("evidence_source"),
        "evidence_proposition": data.get("evidence_proposition"),
        "limitations": data.get("limitations"),
        "reference": data.get("reference"),
        "generation": data.get("generation"),
        "historical_reference": data.get("historical_reference"),
        "tolerances": {
            f"{location}.{key}".lstrip("."): v
            for location, row in tables(data)
            for key, v in row.items()
            if "tolerance" in key or key in {"atol", "rtol"}
        },
        "generators": {
            p.relative_to(root).as_posix(): digest(p) for p in sorted(candidates)
        },
        "independence_review": (
            "recorded_generator_and_assertion_review_with_limits_not_a_method_certification"
            if oracle.get("reference_review")
            and not fixture_review_problems(root, data)
            else "pending_source_and_transitive_helper_review"
        ),
        "gaps": gaps,
    }


def claim_rows(root: Path) -> list[dict]:
    claims = []
    for path in sorted(root.glob("parity/**/*.toml")):
        if path.name == "coverage_records.toml":
            continue
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        for loc, row in tables(data):
            if "evidence_kind" not in row:
                continue
            claims.append(
                {
                    "key": f"{path.relative_to(root).as_posix()}::{loc}",
                    "registry": path.relative_to(root).as_posix(),
                    "table": loc,
                    "id": row.get("id", row.get("route")),
                    **{
                        k: row[k]
                        for k in (
                            "status",
                            "evidence_kind",
                            "evidence_test",
                            "evidence_assertion",
                            "known_truth_fixture",
                            "external_oracle",
                            "limitations",
                            "coverage_records",
                            "calibration",
                            "calibration_reason",
                            "calibration_evidence",
                            "known_truth_reference",
                            "evidence_fields",
                            "evidence_activity",
                            *AXES,
                        )
                        if k in row
                    },
                    "roles": evidence_roles(
                        row["evidence_kind"], row.get("evidence_activity")
                    ),
                }
            )
    calibration_acceptance = {
        (path.relative_to(root).as_posix(), name)
        for path in root.glob("python/tests/test_*.py")
        for name, markers in python_test_markers(path).items()
        if "requires_calibration" in markers
    }
    calibration_acceptance.update(
        (r["evidence_test"], r["evidence_assertion"])
        for r in registered_calibration_acceptance(root)
    )
    for claim in claims:
        if (
            claim.get("evidence_test"),
            claim.get("evidence_assertion"),
        ) in calibration_acceptance:
            claim["evidence_activity"] = "calibration_acceptance"
            claim["roles"] = evidence_roles(claim["evidence_kind"], "calibration_acceptance")
    return claims


def cited_test_text(root: Path, test: str) -> str:
    """Cited test source plus its sibling module directory and shared test helpers."""
    path = in_repo(root, test)
    if not path or not path.is_file():
        return ""
    files = {path}
    if path.suffix == ".rs":
        if path.with_suffix("").is_dir():
            files.update(path.with_suffix("").glob("**/*.rs"))
        files.update((path.parent / "common").glob("**/*.rs"))
    return "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in sorted(files))


def fixture_load_problems(root: Path, claim: dict) -> list[str]:
    """A known-truth claim's cited test must actually name the fixture it cites."""
    fixture, test = claim.get("known_truth_fixture"), claim.get("evidence_test")
    if (
        claim.get("evidence_kind") not in KNOWN_TRUTH_KINDS
        or not isinstance(fixture, str)
        or not test
        or fixture.endswith((".rs", ".py"))
    ):
        return []
    fixture = fixture.removesuffix("/expected.json").removesuffix(".json")
    text = cited_test_text(root, test)
    if not text:
        return []
    if fixture in text or fixture.rsplit("/", 1)[-1] in text:
        return []
    return ["known_truth_fixture_not_named_by_cited_test"]


def software_claim_problems(claim: dict) -> list[str]:
    """A claim without a scientific role must say what its evidence does not show."""
    if (
        "scientific" not in claim["roles"]
        and claim["evidence_kind"] != "process_attested"
        and not str(claim.get("limitations", "")).strip()
    ):
        return ["software_claim_missing_limitations"]
    return []


def software_fixture_claim_problems(claim: dict, fixture: dict) -> list[str]:
    """A known-truth claim must not rest on a fixture declared software-only."""
    if claim.get("evidence_kind") in KNOWN_TRUTH_KINDS and software_only_roles(
        fixture.get("evidence_roles")
    ):
        return ["software_only_fixture_cannot_back_known_truth"]
    return []


def axis_coverage(claims: list[dict]) -> dict[str, dict[str, int]]:
    """Licensed rows per graph class and inference, split by evidence role.

    A cell with no scientific row is software coverage only; this is a map of
    where evidence is thin, not a verdict on any row."""
    table: dict[str, dict[str, int]] = {}
    for claim in claims:
        if claim["registry"] != "parity/support_licensed.toml":
            continue
        key = f"{claim.get('graph_class')}/{claim.get('inference')}"
        row = table.setdefault(key, {"rows": 0, "scientific": 0, "software_only": 0})
        row["rows"] += 1
        row["scientific" if "scientific" in claim["roles"] else "software_only"] += 1
    return dict(sorted(table.items()))


def build_report(root: Path) -> dict:
    root = root.resolve()
    claims = claim_rows(root)
    calibration_acceptance = {
        (path.relative_to(root).as_posix(), name)
        for path in root.glob("python/tests/test_*.py")
        for name, markers in python_test_markers(path).items()
        if "requires_calibration" in markers
    }
    calibration_acceptance.update(
        (r["evidence_test"], r["evidence_assertion"])
        for r in registered_calibration_acceptance(root)
    )
    fixture_paths = set(root.glob("conformance/**/expected.json"))
    for claim in claims:
        rel = claim.get("known_truth_fixture")
        if isinstance(rel, str) and rel.endswith(".json"):
            path = in_repo(root, rel)
            if path and path.is_file():
                fixture_paths.add(path)
    fixtures = [fixture_report(root, p) for p in sorted(fixture_paths)]
    fixture_by_path = {r["path"]: r for r in fixtures}
    claims_by_test = defaultdict(list)
    for claim in claims:
        problems = inline_reference_problems(claim)
        fixture = claim.get("known_truth_fixture")
        info = fixture_by_path.get(str(fixture).removesuffix("/expected.json"))
        if fixture and not info:
            source = in_repo(root, str(fixture))
            if source and source.is_file() and source.suffix in {".rs", ".py"}:
                claim["inline_reference"] = str(fixture)
                claim["reference_review"] = (
                    "source_inline_truth_requires_independence_review"
                )
            else:
                problems.append("missing_or_unrecognized_fixture")
        problems.extend(fixture_load_problems(root, claim))
        problems.extend(software_claim_problems(claim))
        if fixture and info:
            problems.extend(software_fixture_claim_problems(claim, info))
        if fixture and info:
            problems.extend(
                kind_problems(
                    claim["evidence_kind"],
                    info["oracle"],
                    claim.get("inference"),
                    claim.get("evidence_fields"),
                )
            )
        elif claim["evidence_kind"] not in EVIDENCE_KINDS:
            problems.append("unknown_evidence_kind")
        if "scientific" in claim["roles"]:
            for field in ("evidence_test", "evidence_assertion"):
                if not claim.get(field):
                    problems.append(f"missing_{field}")
            if claim.get("evidence_test"):
                path = in_repo(root, claim["evidence_test"])
                if not path or not path.is_file():
                    problems.append("missing_evidence_source")
                elif claim.get("evidence_assertion"):
                    if path.suffix == ".rs":
                        problems.extend(
                            rust_test_declaration_problems(
                                path, claim["evidence_assertion"]
                            )
                        )
                    elif path.suffix == ".py":
                        problems.extend(
                            static_python_test(path, claim["evidence_assertion"])
                        )
            if (
                claim["evidence_kind"] in KNOWN_TRUTH_KINDS
                and not fixture
                and not claim.get("known_truth_reference")
            ):
                problems.append("independent_reference_not_explicitly_linked")
        claim["gaps"] = problems
        claim["execution_verified"] = False
        if claim.get("evidence_test"):
            claims_by_test[claim["evidence_test"]].append(claim["key"])
    registry = root / "parity/v3_test_registry.toml"
    registered = tomllib.loads(registry.read_text(encoding="utf-8")).get("test", [])
    tests = []
    roles_by_test = defaultdict(set)
    for claim in claims:
        roles_by_test[claim.get("evidence_test")].update(claim["roles"])
    manifests = {}
    for row in registered:
        rel = row["path"]
        path = in_repo(root, rel)
        crate = (
            root / "python"
            if row["crate"] == "python"
            else root / "crates" / row["crate"]
        )
        if crate not in manifests:
            manifest = crate / "Cargo.toml"
            manifests[crate] = (
                tomllib.loads(manifest.read_text(encoding="utf-8"))
                if manifest.is_file()
                else {}
            )
        manifest = manifests[crate]
        target = row["cargo_target"]
        features = [
            f
            for t in manifest.get("test", [])
            if t.get("name") == target
            for f in t.get("required-features", [])
        ]
        tests.append(
            {
                **row,
                "source_sha256": digest(path) if path and path.is_file() else None,
                "claims": claims_by_test.get(rel, []),
                "roles": sorted(roles_by_test[rel] | {"software"}),
                "required_target_features": features,
                "assertion_declarations": (
                    python_test_declarations(path)
                    if path and path.suffix == ".py" and path.is_file()
                    else []
                ),
                "discovery_verified": False,
            }
        )
    provenance = []
    cache_path = root / CITATION_CACHE
    citation_cache = (
        json.loads(cache_path.read_text(encoding="utf-8"))
        if cache_path.is_file()
        else {}
    )
    for path in sorted(root.glob("provenance/*.toml")):
        if path.name == "_template.toml":
            continue
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        provenance.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": digest(path),
                "feature_id": data.get("feature_id"),
                "modules": data.get("modules", []),
                "papers": data.get("papers", []),
                "test_sources": data.get("test_sources", []),
                "implementation_deviations": data.get("implementation_deviations"),
                "exposure": data.get("upstream_implementations_observed", []),
                "citation_identity": [
                    citation_verdict(p, citation_cache) for p in data.get("papers", [])
                ],
                "algorithm_section_review": (
                    "reviewed_with_limits"
                    if isinstance(data.get("method_review"), dict)
                    and not method_review_problems(root, data)
                    else "justified_non_algorithm_mapping"
                    if isinstance(data.get("non_algorithm_review"), dict)
                    and not non_algorithm_review_problems(root, data)
                    else "pending_or_stale_implementation_correspondence_review"
                ),
                "non_algorithm_review": data.get("non_algorithm_review"),
                "non_algorithm_review_findings": non_algorithm_review_problems(root, data),
                "method_review": data.get("method_review"),
                "method_review_findings": method_review_problems(root, data),
            }
        )
    specs_path = root / "parity/method_specs/specs.toml"
    specs = (
        tomllib.loads(specs_path.read_text(encoding="utf-8")).get("spec", [])
        if specs_path.is_file()
        else []
    )
    routes_path = root / "parity/capabilities/route_manifest.toml"
    route_data = (
        tomllib.loads(routes_path.read_text(encoding="utf-8"))
        if routes_path.is_file()
        else {}
    )
    routes = {r["id"]: r for r in route_data.get("route", [])}
    for claim in claims:
        route = f"licensed.{claim.get('query')}.{claim.get('graph_class')}"
        linked = (
            [s for s in specs if s.get("route") == route] if route in routes else []
        )
        claim["route_registry_links"] = [route] if route in routes else []
        claim["method_spec_links"] = [s["id"] for s in linked]
        claim["provenance_registry_links"] = sorted(
            {p for s in linked for p in s.get("provenance", [])}
        )
        claim["link_basis"] = (
            "registry_coordinate_join; not independent source verification"
        )
    doctests = []
    for path in sorted(root.glob("crates/*/src/**/*.rs")):
        text = path.read_text(encoding="utf-8")
        blocks = re.findall(r"(?m)^\s*//[!/]\s*```([^\n]*)", text)
        rust_blocks = [
            label
            for label in blocks
            if label.strip()
            and label.strip().split(",")[0]
            in {"rust", "no_run", "compile_fail", "ignore"}
        ]
        # Unlabelled opening/closing fences need rustdoc to distinguish them.
        if blocks:
            doctests.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "labelled_rust_fences": len(rust_blocks),
                    "fence_count": len(blocks),
                    "discovery_verified": False,
                    "runner": "bash scripts/test.sh docs",
                }
            )
    proc = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    dispositions_path = root / "parity/test_refactor_dispositions.toml"
    dispositions = (
        tomllib.loads(dispositions_path.read_text(encoding="utf-8")).get("case", [])
        if dispositions_path.is_file()
        else []
    )
    def scientific_assertions():
        return {
            (c.get("evidence_test"), c.get("evidence_assertion"))
            for c in claims
            if "scientific" in c["roles"]
            and c.get("evidence_activity") != "calibration_acceptance"
            and c.get("evidence_test")
            and c.get("evidence_assertion")
        }

    partition = {
        # Each count answers one question; none is derived from another.
        "software_coverage": {
            "claims_with_software_role": sum("software" in c["roles"] for c in claims),
            "software_only_fixtures": sum(
                software_only_roles(f.get("evidence_roles")) for f in fixtures
            ),
        },
        "scientific_propositions": {
            "distinct_cited_assertions": len(scientific_assertions()),
            "claims_with_scientific_role": sum(
                "scientific" in c["roles"]
                and c.get("evidence_activity") != "calibration_acceptance"
                for c in claims
            ),
        },
        "licensed_claims": {
            "registry": "parity/support_licensed.toml",
            "claims": sum(c["registry"] == "parity/support_licensed.toml" for c in claims),
        },
        "licensed_axis_coverage": axis_coverage(claims),
        "pending_calibration": {
            "acceptance_assertions": len(calibration_acceptance),
            "status": "pending_frozen_candidate",
        },
        "evidence_gaps": {
            "claims_with_gaps": sum(bool(c["gaps"]) for c in claims),
            "fixtures_with_gaps": sum(bool(f["gaps"]) for f in fixtures),
            "oracle_origin_unrecorded": sum(
                "oracle_origin_unrecorded" in f["gaps"] for f in fixtures
            ),
        },
    }
    return {
        "schema": 1,
        "report_kind": "source_evidence_audit",
        "partition": partition,
        "commit": proc.stdout.strip(),
        "worktree_changes": status.stdout.splitlines(),
        "execution_verified": False,
        "calibration": {
            "branch_measurement": "pending_frozen_candidate",
            "measurements_run": False,
            "attestation_verified": False,
        },
        "release_certified": False,
        "calibration_acceptance_assertions": [
            {
                "evidence_test": path,
                "evidence_assertion": name,
                "status": "requires_attesting_frozen_branch_measurements",
            }
            for path, name in sorted(calibration_acceptance)
        ],
        "tests": tests,
        "claims": claims,
        "fixtures": fixtures,
        "provenance": provenance,
        "method_specs": specs,
        "route_registry": list(routes.values()),
        "doctest_sources": doctests,
        "test_refactor_dispositions": dispositions,
        "disposition_findings": disposition_problems(root, dispositions),
        "scenario_matrix": (
            build_scenario_matrix(root)
            if (root / "parity/scenario_matrix.toml").is_file()
            and (root / "parity/counterparts.toml").is_file()
            else None
        ),
        "route_evidence_ledger_findings": (
            check_evidence_ledger.problems(
                tomllib.loads(
                    (root / "parity/evidence_ledger.toml").read_text(encoding="utf-8")
                ).get("cell", []),
                tomllib.loads(
                    (root / "parity/promotion.toml").read_text(encoding="utf-8")
                ).get("record", []),
                tomllib.loads(
                    (root / "parity/transport_stages.toml").read_text(encoding="utf-8")
                ).get("routes", []),
            )
            if root == REPO_ROOT.resolve()
            else []
        ),
    }


def render(report: dict) -> str:
    lines = [
        "# Evidence and provenance audit",
        "",
        (
            "Source inventory only: execution, source independence, citations and calibration "
            "are not certified by this report."
        ),
        "",
        f"Commit: `{report['commit']}`; checkout changes: {len(report['worktree_changes'])}.",
        "",
        (
            f"Registered test sources: {len(report['tests'])}; claims: {len(report['claims'])}; "
            f"fixtures: {len(report['fixtures'])}; provenance records: {len(report['provenance'])}."
        ),
        "",
        "Calibration is pending the frozen candidate. No measurements or replays run.",
        "",
        "## Evidence kinds (registry declarations; execution is separate)",
        "",
        "| Evidence kind | Claims |",
        "| --- | ---: |",
    ]
    for kind, count in sorted(
        Counter(c["evidence_kind"] for c in report["claims"]).items()
    ):
        lines.append(f"| {kind} | {count} |")
    lines += ["", "| Evidence roles / activity | Claims |", "| --- | ---: |"]
    for classification, count in sorted(
        Counter(
            (", ".join(c["roles"]), c.get("evidence_activity", "ordinary"))
            for c in report["claims"]
        ).items()
    ):
        roles, activity = classification
        lines.append(f"| {roles} / {activity} | {count} |")
    partition = report["partition"]
    lines += [
        "",
        "## Evidence partition (separate questions; none implies another)",
        "",
        f"- Software coverage: {partition['software_coverage']['claims_with_software_role']} claims with a software role; "
        f"{partition['software_coverage']['software_only_fixtures']} software-only fixtures.",
        f"- Scientific propositions: {partition['scientific_propositions']['distinct_cited_assertions']} distinct cited assertions "
        f"({partition['scientific_propositions']['claims_with_scientific_role']} claims); declarations, not verified execution or independence.",
        f"- Licensed claims: {partition['licensed_claims']['claims']} rows in {partition['licensed_claims']['registry']}; a licensed row is not calibration.",
        f"- Pending calibration: {partition['pending_calibration']['acceptance_assertions']} acceptance assertions; "
        f"{partition['pending_calibration']['status']}; no measurements or attestations are made here.",
        f"- Evidence gaps: {partition['evidence_gaps']['claims_with_gaps']} claims and "
        f"{partition['evidence_gaps']['fixtures_with_gaps']} fixtures with gaps; "
        f"{partition['evidence_gaps']['oracle_origin_unrecorded']} with unrecorded oracle origin.",
    ]
    thin = [
        f"{key} ({v['software_only']} of {v['rows']})"
        for key, v in partition["licensed_axis_coverage"].items()
        if v["scientific"] == 0
    ]
    lines.append(
        "- Licensed graph-class/inference cells with no scientific row: "
        + (", ".join(thin) if thin else "none")
        + "."
    )
    software_fixtures = [
        f for f in report["fixtures"] if software_only_roles(f.get("evidence_roles"))
    ]
    reviewed = [
        f
        for f in report["fixtures"]
        if f["independence_review"].startswith("recorded_")
    ]
    lines += [
        "",
        "## Software-only fixture propositions",
        "",
        "These declarations supply no independent numerical truth or calibration license.",
        "",
    ]
    for f in software_fixtures:
        lines.append(
            f"- `{f['path']}` → `{f['evidence_source']}`: {f['evidence_proposition']} {f['limitations']}"
        )
    lines += [
        "",
        f"Retained generator/assertion reviews with current digests: {len(reviewed)} of {len(report['fixtures'])} fixtures. This is separate from primary theorem correspondence.",
        "",
        f"Measured-record acceptance assertions: {len(report.get('calibration_acceptance_assertions', []))}; pending frozen-branch measurements.",
    ]
    primary = [
        p
        for p in report["provenance"]
        if p["algorithm_section_review"] == "reviewed_with_limits"
    ]
    lines += [
        "",
        f"Bounded primary implementation reviews: {len(primary)} of {len(report['provenance'])} provenance records.",
        "",
        "Justified non-algorithm mappings (records with no primary paper mapped to the reviewed records and modules that carry their numerics, with digests): "
        f"{sum(1 for p in report['provenance'] if p['algorithm_section_review'] == 'justified_non_algorithm_mapping')} of {len(report['provenance'])}. "
        "A mapping is not an implementation review of a primary source.",
        "",
    ]
    for p in primary:
        lines.append(
            f"- `{p['path']}`: {p['method_review']['scope']} Limits: {p['method_review']['limitations']}"
        )
    lines += ["", "## Claim-link findings", ""]
    for c in report["claims"]:
        if c["gaps"]:
            lines.append(f"- `{c['key']}`: {'; '.join(c['gaps'])}")
    lines += ["", "## Fixture reproducibility findings", ""]
    for f in report["fixtures"]:
        if f["gaps"]:
            lines.append(f"- `{f['path']}`: {', '.join(f['gaps'])}")
    lines += ["", "## Route-to-evidence ledger findings", ""]
    lines += [
        f"- {finding}" for finding in report.get("route_evidence_ledger_findings", [])
    ]
    matrix = report.get("scenario_matrix")
    if matrix:
        lines += [
            "",
            "## Method-family scenario matrix (parity/scenario_matrix.toml; source inspection)",
            "",
            (
                f"{len(matrix['families'])} families, {len(matrix['cells'])} cells: "
                + ", ".join(f"{k} {v}" for k, v in sorted(matrix["counts"].items()))
                + ". Only an exact compiled, non-ignored test or fixture counts as delivered; "
                "unavailable cells state a reason; every other cell is a gap."
            ),
            "",
            "| Axis | Delivered | Unavailable | Gap |",
            "| --- | ---: | ---: | ---: |",
        ]
        for axis, counts in matrix["by_axis"].items():
            lines.append(
                f"| {axis} | {counts.get('delivered', 0)} | {counts.get('unavailable', 0)} "
                f"| {counts.get('gap', 0)} |"
            )
        lines += ["", "Gaps by kind:", ""]
        kinds = Counter(
            (c["axis"], c["gap_kind"]) for c in matrix["cells"] if c["status"] == "gap"
        )
        lines += [f"- {axis} / {kind}: {n}" for (axis, kind), n in sorted(kinds.items())]
        lines += ["", "Matrix findings (invalid curated cells):", ""]
        lines += [f"- {finding}" for finding in matrix["findings"]] or ["- none"]
        lines += ["", "Gap cells (family, axis, value):", ""]
        lines += [
            f"- `{c['family']}` {c['axis']} / {c['value']} [{c['gap_kind']}]"
            for c in matrix["cells"]
            if c["status"] == "gap"
        ]
    lines += ["", "## Test coverage dispositions", ""]
    lines += [
        f"- `{case['previous']}` → `{case['replacement']}`: {case['disposition']}. {case['reason']}"
        for case in report.get("test_refactor_dispositions", [])
    ]
    lines += [
        "",
        "## Required review",
        "",
        (
            "Bibliographic identity is reported separately from algorithm-section verification. "
            "Every oracle requires input, assertion, redistribution and transitive-helper "
            "independence review. A present generator is a reproducibility lead, not a completed review."
        ),
    ]
    return "\n".join(lines) + "\n"


def scientific_rows(claims: list[dict]) -> list[dict]:
    """Deduplicate assertions, not claims; each claim retains its own audit links."""
    found = {}
    for row in claims:
        if "scientific" not in row["roles"]:
            continue
        if row.get("evidence_activity") == "calibration_acceptance":
            continue
        path, assertion = row.get("evidence_test"), row.get("evidence_assertion")
        if path and assertion:
            found.setdefault(
                (path, assertion),
                {
                    "id": row["key"],
                    "evidence_test": path,
                    "evidence_assertion": assertion,
                },
            )
    if not found:
        raise ValueError("scientific selection is empty")
    return list(found.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("report", "run-scientific", "rust-calibration-filter", "scientific-filter"),
    )
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="run-scientific: run only the cited Python assertions (Rust ones run through scientific-filter)",
    )
    parser.add_argument("--listing", type=Path, help="Verify classified assertions in a nextest JSON listing")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.command == "rust-calibration-filter":
        rows = registered_calibration_acceptance(REPO_ROOT)
        if not rows:
            raise ValueError("calibration acceptance selection is empty")
        if args.listing:
            problems = calibration_discovery_problems(rows, json.loads(args.listing.read_text(encoding="utf-8")))
            if problems:
                print("\n".join(problems), file=sys.stderr)
                return 1
        expressions = []
        for row in rows:
            target = "kind(lib)" if row["cargo_target"] == "lib" else f"binary(={row['cargo_target']})"
            expressions.append(f"(package(={row['crate']}) and {target} and test(={row['libtest_name']}))")
        print(" or ".join(expressions))
        return 0
    if args.command == "scientific-filter":
        rows = scientific_rows(build_report(REPO_ROOT)["claims"])
        rust_rows = registered_rust_assertions(
            [(r["evidence_test"], r["evidence_assertion"]) for r in rows if r["evidence_test"].endswith(".rs")]
        )
        if not rust_rows:
            raise ValueError("scientific Rust selection is empty")
        if args.listing:
            problems = calibration_discovery_problems(rust_rows, json.loads(args.listing.read_text(encoding="utf-8")))
            if problems:
                print("\n".join(problems), file=sys.stderr)
                return 1
        print(nextest_selection(rust_rows))
        return 0
    report = build_report(REPO_ROOT)
    if args.command == "report":
        print(json.dumps(report, indent=2) if args.json else render(report), end="\n")
        return int(
            args.strict
            and (
                any(c["gaps"] for c in report["claims"])
                or any(f["gaps"] for f in report["fixtures"])
                or any(
                    p["algorithm_section_review"]
                    not in ("reviewed_with_limits", "justified_non_algorithm_mapping")
                    for p in report["provenance"]
                )
                or bool(report["route_evidence_ledger_findings"])
                or bool(report["disposition_findings"])
                or bool(
                    report.get("scenario_matrix")
                    and (
                        report["scenario_matrix"]["findings"]
                        or report["scenario_matrix"]["counts"].get("gap")
                    )
                )
            )
        )
    rows = scientific_rows(report["claims"])
    if args.python_only:
        rows = [row for row in rows if row["evidence_test"].endswith(".py")]
        if not rows:
            print("scientific: no cited Python assertions; nothing to run")
            return 0
    if any(row["evidence_test"].endswith(".py") for row in rows):
        interpreters = [
            REPO_ROOT / "python/.venv/bin/python",
            REPO_ROOT / "python/.venv/Scripts/python.exe",
        ]
        interpreter = next((p for p in interpreters if p.is_file()), None)
        if interpreter is None:
            print(
                "FAIL: the Python conformance environment is missing; run bash scripts/test.sh python"
            )
            return 1
        checked = subprocess.run(
            [
                str(interpreter),
                str(REPO_ROOT / "scripts/native_build_identity.py"),
                "check",
            ],
            check=False,
        )
        if checked.returncode:
            return checked.returncode
    missing = [
        c
        for c in report["claims"]
        if "scientific" in c["roles"]
        and (not c.get("evidence_test") or not c.get("evidence_assertion"))
    ]
    if missing:
        print(
            f"FAIL: {len(missing)} scientific declarations lack exact assertion citations; "
            "running the cited assertions does not close those gaps",
            flush=True,
        )
    # The existing runner rejects ignored tests and proves each exact assertion ran.
    with tempfile.TemporaryDirectory(prefix="antecedent-scientific-") as tmp:
        ledger = Path(tmp) / "selection.toml"
        ledger.write_text(
            "\n".join(
                "[[capabilities]]\n"
                + "\n".join(f"{k} = {json.dumps(v)}" for k, v in r.items())
                for r in rows
            ),
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/run_evidence_rows.py"),
                str(REPO_ROOT),
                str(ledger),
                str(REPO_ROOT),
                "capabilities",
                "scientific",
            ],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
        return result or int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
