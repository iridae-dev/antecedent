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
    "parity/estimator_route_residual.toml",
    "parity/support_graphless.toml",
    "crates/antecedent-estimate/src/lib.rs",
)
LOWERING = f"{ANALYSIS}/lowering"
SEALERS = f"{ANALYSIS}/prepared/sealers"


def overlay(destination: Path) -> None:
    shutil.copytree(
        ROOT / SRC,
        destination / SRC,
        ignore=shutil.ignore_patterns("target", "*.json", "*.md"),
    )
    shutil.copytree(
        ROOT / "crates" / "antecedent" / "tests",
        destination / "crates" / "antecedent" / "tests",
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


DISPATCH_GUARD = (
    "        let lowered =\n"
    "            physical.require_lowered(crate::analysis::lowering::expected_cell(self))?.identity();\n"
    "        let ctx = &ctx.enter_lowered_dispatch(lowered);\n"
)


def decoy_dispatch_guard(destination: Path, nested: bool) -> None:
    decoy = "physical.require_lowered(None)?; ctx.enter_lowered_dispatch(identity);"
    replacement = "        let _decoy = || { " + decoy + " };\n" if nested else ""
    replace(destination, f"{ANALYSIS}/execute/dispatch.rs", DISPATCH_GUARD, replacement)
    if not nested:
        append(
            destination,
            f"{ANALYSIS}/execute/dispatch.rs",
            "\nfn unrelated_guard() { " + decoy + " }\n",
        )


MANIFEST = "parity/capabilities/route_manifest.toml"
CASES = {
    "compile entry lost its reentry refusal": lambda d: replace(
        d, f"{ANALYSIS}/execute/compile_planning.rs", "    refuse_reentry(ctx)?;", ""
    ),
    "dispatch entry no longer requires the lowered program": lambda d: replace(
        d,
        f"{ANALYSIS}/execute/dispatch.rs",
        "        let lowered =\n            physical.require_lowered(crate::analysis::lowering::expected_cell(self))?.identity();\n"
        "        let ctx = &ctx.enter_lowered_dispatch(lowered);\n        let mut result = match data {",
        "        let mut result = match data {",
    ),
    "dispatch guard only appears in a neighboring helper": lambda d: decoy_dispatch_guard(d, False),
    "dispatch guard only appears in a nested closure": lambda d: decoy_dispatch_guard(d, True),
    "reentry refusal code dropped": lambda d: replace(
        d, f"{LOWERING}/mod.rs", '"public_builder_reentry"', '"other_code"'
    ),
    "route_not_lowered swallowed": lambda d: append(
        d,
        f"{ANALYSIS}/propagation_chain.rs",
        '\nfn swallow(error: &E) -> bool {\n    error.reason_code() == Some("route_not_lowered")\n}\n',
    ),
    "second ordinary dispatch construction": lambda d: append(
        d,
        f"{ANALYSIS}/prepared/lifecycle.rs",
        "\nfn reroute(mut handle: Handle) {\n    handle.execution = PreparedExecution::OrdinaryDispatch;\n}\n",
    ),
    "fresh run built outside the dispatcher": lambda d: append(
        d,
        f"{ANALYSIS}/prepared/lifecycle.rs",
        "\nfn reroute(op: CheckedRouteOperation) -> PreparedExecution {\n"
        "    PreparedExecution::FreshRun(op)\n}\n",
    ),
    "estimator missing from the family table": lambda d: replace(
        d, f"{SEALERS}/estimator.rs", "    EstimatorId::Dml,\n", ""
    ),
    "estimator a sealer serves added to the residual": lambda d: append(
        d,
        "parity/estimator_route_residual.toml",
        '\n[[estimator]]\nid = "CausalImpactGone"\nreason = "x"\n',
    ),
    "served estimator in the residual without the flag": lambda d: replace(
        d,
        "parity/estimator_route_residual.toml",
        'served_by_specialised_sealer_for_other_shapes = true\nreason = "the AIPW sealer',
        'reason = "the AIPW sealer',
    ),
    "estimator served by nothing and not in the residual": lambda d: replace(
        d,
        "parity/estimator_route_residual.toml",
        '[[estimator]]\nid = "Dml"',
        '[[estimator]]\nid = "DmlGone"',
    ),
    "estimator sealers not last": lambda d: replace(
        d,
        f"{SEALERS}/mod.rs",
        "    sealer(ExecutionKind::EstimatorRoute, estimator::estimator_route),\n",
        "",
    ),
    "cell no sealer serves": lambda d: replace(
        d, f"{SEALERS}/design.rs", "Some(RouteInput::Survival", "Some(RouteInput::SurvivalGone"
    ),
    "registered cell without a sealed operation": lambda d: replace(
        d,
        f"{LOWERING}/registry.rs",
        "    StaticAteDag => static_effect as StaticEffect,",
        "    NewRouteCell => static_effect as StaticEffect,\n    StaticAteDag => static_effect as StaticEffect,",
    ),
    "sealed operation without a sealer": lambda d: replace(
        d, f"{SEALERS}/mod.rs", "    sealer(ExecutionKind::Survival, design::survival),\n", ""
    ),
    "sealer that declares no cell": lambda d: replace(
        d,
        f"{SEALERS}/mod.rs",
        "    sealer(ExecutionKind::Survival, design::survival),",
        "    sealer(ExecutionKind::Survival, design::survival),\n"
        "    sealer(ExecutionKind::Randomized, posterior::undeclared_sealer),",
    ),
    "licensed route lowers an unregistered cell": lambda d: replace(
        d, "parity/licensed_routes.toml", 'lowering = "StaticMediationDag"', 'lowering = "Nope"'
    ),
    "retained operation names no sealed operation": lambda d: replace(
        d,
        MANIFEST,
        'retained_operation = "attribution"',
        'retained_operation = "no_such_operation"',
    ),
    "retained operation does not serve the route's cell": lambda d: replace(
        d, MANIFEST, 'retained_operation = "attribution"', 'retained_operation = "survival"'
    ),
    "graphless route without its retained test": lambda d: replace(
        d,
        MANIFEST,
        'retained_route_test = "crates/antecedent/tests/survival_route.rs::'
        'licensed_survival_scalar_intervals_round_trip_and_refuse_forgery"\n',
        "",
    ),
    "retained test names no function": lambda d: replace(
        d,
        MANIFEST,
        "licensed_survival_scalar_intervals_round_trip_and_refuse_forgery",
        "no_such_test_function",
    ),
    "temporal mixed family regresses to a program-only payload": lambda d: replace(
        d,
        f"{ANALYSIS}/prepared/execution.rs",
        "TemporalAdmgEffect(crate::analysis::execute::CheckedTemporalAdmgOperation)",
        "TemporalAdmgEffect(CheckedRouteOperation)",
    ),
    "closed coordinate id disagrees with its target and graph": lambda d: replace(
        d,
        MANIFEST,
        'id = "closed.Counterfactual.Admg"',
        'id = "closed.Counterfactual.NotARealGraph"',
    ),
    "closed coordinate names an unreviewed target": lambda d: replace(
        d,
        MANIFEST,
        'id = "closed.Counterfactual.Admg"\ntarget = "Counterfactual"',
        'id = "closed.UnreviewedTarget.Admg"\ntarget = "UnreviewedTarget"',
    ),
    "executing route with no lowered program entry": lambda d: replace(
        d, MANIFEST, 'id = "bayesian.piecewise_survival"', 'id = "bayesian.unmapped_zz"'
    ),
    "family operation click no longer requires its program": lambda d: replace(
        d,
        f"{ANALYSIS}/prepared/sealed/static_route.rs",
        "        self.require_program(&handle.plan)?;\n        let mut result = handle.analysis.execute_association_tabular",
        "        let mut result = handle.analysis.execute_association_tabular",
    ),
    "family operation wraps the generic dispatcher": lambda d: append(
        d,
        f"{ANALYSIS}/execute/checked_panel_did.rs",
        "\nfn wrap(study: &Study) {\n    study.execute_on();\n}\n",
    ),
    "cited test lost its retained-program assertion": lambda d: replace(
        d,
        "crates/antecedent/tests/interference_interval_route.rs",
        "retained_graphless::assert_retained_graphless_program(",
        "retained_graphless::skip_retained_program(",
    ),
    "axis names a route that does not exist": lambda d: replace(
        d,
        "parity/support_graphless.toml",
        'routes = ["interference.cluster_bayesian_bootstrap"]',
        'routes = ["interference.no_such_route"]',
    ),
    "family lost its temporal axis row": lambda d: replace(
        d,
        "parity/support_graphless.toml",
        'family = "survival"\naxis = "temporal"',
        'family = "survival"\naxis = "temporal_gone"',
    ),
    "license cites a test no graphless route records": lambda d: replace(
        d,
        "parity/support_graphless.toml",
        'retained_route_test = "crates/antecedent/tests/survival_route.rs::'
        'licensed_survival_scalar_intervals_round_trip_and_refuse_forgery"',
        'retained_route_test = "crates/antecedent/tests/survival_route.rs::unrecorded_test"',
    ),
    "graphless route owned by a module its operation never reaches": lambda d: replace(
        d,
        MANIFEST,
        'owning_module = "crates/antecedent-estimate/src/interference.rs"',
        'owning_module = "crates/antecedent-estimate/src/ols_gone_module.rs"',
    ),
    "licensed route lost its manifest record": lambda d: replace(
        d,
        MANIFEST,
        'id = "licensed.TransportQuery.Admg"\ntarget = "TransportQuery"',
        'id = "licensed.TransportQuery.Admg"\ntarget = "TransportQueryGone"',
    ),
}


def main() -> None:
    failures = []
    with tempfile.TemporaryDirectory() as scratch:
        base = Path(scratch)
        overlay(base)
        baseline = set(retained_route_problems(base))
        if baseline != set(retained_route_problems(ROOT)):
            raise SystemExit("FAIL: the overlay does not reproduce the repository's findings")
        if baseline:
            raise SystemExit(
                "FAIL: the repository has retained-route findings before mutation:\n"
                + "\n".join(sorted(baseline))
            )
        # A long legitimate entry must not become a baseline violation merely
        # because its admission guard moved past an arbitrary source window.
        with tempfile.TemporaryDirectory() as positive_scratch:
            positive = Path(positive_scratch)
            shutil.copytree(base, positive, dirs_exist_ok=True)
            replace(
                positive,
                f"{ANALYSIS}/execute/dispatch.rs",
                DISPATCH_GUARD,
                "        let _padding = ();\n" * 100 + DISPATCH_GUARD,
            )
            if set(retained_route_problems(positive)) != baseline:
                failures.append("long valid dispatch entry falsely refused")
        for label, inject in CASES.items():
            with tempfile.TemporaryDirectory() as case_scratch:
                case = Path(case_scratch)
                shutil.copytree(base, case, dirs_exist_ok=True)
                inject(case)
                if not set(retained_route_problems(case)) - baseline:
                    failures.append(label)
    if failures:
        raise SystemExit("FAIL: the retained-route check missed: " + ", ".join(failures))
    print(
        f"ok: the retained-route check catches {len(CASES)} injected violations "
        f"(repository baseline: {len(baseline)} finding(s))"
    )


if __name__ == "__main__":
    sys.exit(main())
