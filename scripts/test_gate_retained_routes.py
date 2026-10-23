#!/usr/bin/env python3
"""Self-test: the retained-route check fails on each reopened bypass.

    python3 scripts/test_gate_retained_routes.py

Every case copies the sources and records the gate reads, injects one violation,
and requires a problem that the unmodified copy does not report.
"""

import shutil
import sys
import tempfile
from pathlib import Path

from gate_retained_routes import ANALYSIS, ROOT, SRC, retained_route_problems

COPIED_FILES = (
    "parity/capabilities/route_manifest.toml",
    "parity/licensed_routes.toml",
    "parity/reason_codes.toml",
)
LOWERING = f"{ANALYSIS}/lowering"
SEALERS = f"{ANALYSIS}/prepared/sealers"


def overlay(destination: Path) -> None:
    shutil.copytree(
        ROOT / SRC, destination / SRC,
        ignore=shutil.ignore_patterns("target", "*.json", "*.md"),
    )
    shutil.copytree(
        ROOT / "crates" / "antecedent" / "tests", destination / "crates" / "antecedent" / "tests",
        ignore=shutil.ignore_patterns("*.json", "*.csv", "*.parquet", "fixtures"),
    )
    for rel in COPIED_FILES:
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)


def replace(destination: Path, rel: str, old: str, new: str) -> None:
    path = destination / rel
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"FAIL: self-test anchor {old!r} is missing from {rel}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def append(destination: Path, rel: str, text: str) -> None:
    path = destination / rel
    path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")


MANIFEST = "parity/capabilities/route_manifest.toml"
CASES = {
    "compile entry lost its reentry refusal": lambda d: replace(
        d, f"{ANALYSIS}/execute/compile_planning.rs", "    refuse_reentry(ctx)?;", ""),
    "dispatch entry no longer requires the lowered program": lambda d: replace(
        d, f"{ANALYSIS}/execute/dispatch.rs",
        "        let lowered =\n            physical.require_lowered(crate::analysis::lowering::expected_cell(self))?.identity();\n"
        "        let ctx = &ctx.enter_lowered_dispatch(lowered);\n        let mut result = match data {",
        "        let mut result = match data {"),
    "reentry refusal code dropped": lambda d: replace(
        d, f"{LOWERING}/mod.rs", '"public_builder_reentry"', '"other_code"'),
    "route_not_lowered swallowed": lambda d: append(
        d, f"{ANALYSIS}/propagation_chain.rs",
        '\nfn swallow(error: &E) -> bool {\n    error.reason_code() == Some("route_not_lowered")\n}\n'),
    "second ordinary dispatch construction": lambda d: append(
        d, f"{ANALYSIS}/prepared/lifecycle.rs",
        "\nfn reroute(mut handle: Handle) {\n    handle.execution = PreparedExecution::OrdinaryDispatch;\n}\n"),
    "cell no sealer serves": lambda d: replace(
        d, f"{SEALERS}/design.rs", "Some(RouteInput::Survival", "Some(RouteInput::SurvivalGone"),
    "registered cell without a sealed operation": lambda d: replace(
        d, f"{LOWERING}/registry.rs", "    StaticAteDag => static_effect as StaticEffect,",
        "    NewRouteCell => static_effect as StaticEffect,\n    StaticAteDag => static_effect as StaticEffect,"),
    "sealed operation without a sealer": lambda d: replace(
        d, f"{SEALERS}/mod.rs", "    sealer(ExecutionKind::Survival, design::survival),\n", ""),
    "sealer that declares no cell": lambda d: replace(
        d, f"{SEALERS}/mod.rs", "    sealer(ExecutionKind::Survival, design::survival),",
        "    sealer(ExecutionKind::Survival, design::survival),\n"
        "    sealer(ExecutionKind::Randomized, posterior::undeclared_sealer),"),
    "licensed route lowers an unregistered cell": lambda d: replace(
        d, "parity/licensed_routes.toml", 'lowering = "StaticMediationDag"', 'lowering = "Nope"'),
    "retained operation names no sealed operation": lambda d: replace(
        d, MANIFEST, 'retained_operation = "attribution"', 'retained_operation = "no_such_operation"'),
    "retained operation does not serve the route's cell": lambda d: replace(
        d, MANIFEST, 'retained_operation = "attribution"', 'retained_operation = "survival"'),
    "graphless route without its retained test": lambda d: replace(
        d, MANIFEST,
        'retained_route_test = "crates/antecedent/tests/survival_route.rs::'
        'licensed_survival_scalar_intervals_round_trip_and_refuse_forgery"\n', ""),
    "retained test names no function": lambda d: replace(
        d, MANIFEST, "licensed_survival_scalar_intervals_round_trip_and_refuse_forgery",
        "no_such_test_function"),
    "executing route with no lowered program entry": lambda d: replace(
        d, MANIFEST, 'id = "bayesian.piecewise_survival"', 'id = "bayesian.unmapped_zz"'),
    "licensed route lost its manifest record": lambda d: replace(
        d, MANIFEST, 'id = "licensed.TransportQuery.Admg"\ntarget = "TransportQuery"',
        'id = "licensed.TransportQuery.Admg"\ntarget = "TransportQueryGone"'),
}


def main() -> None:
    failures = []
    with tempfile.TemporaryDirectory() as scratch:
        base = Path(scratch)
        overlay(base)
        baseline = set(retained_route_problems(base))
        if baseline != set(retained_route_problems(ROOT)):
            raise SystemExit("FAIL: the overlay does not reproduce the repository's findings")
        for label, inject in CASES.items():
            with tempfile.TemporaryDirectory() as case_scratch:
                case = Path(case_scratch)
                shutil.copytree(base, case, dirs_exist_ok=True)
                inject(case)
                if not set(retained_route_problems(case)) - baseline:
                    failures.append(label)
    if failures:
        raise SystemExit("FAIL: the retained-route check missed: " + ", ".join(failures))
    print(f"ok: the retained-route check catches {len(CASES)} injected violations "
          f"(repository baseline: {len(baseline)} finding(s))")


if __name__ == "__main__":
    sys.exit(main())
