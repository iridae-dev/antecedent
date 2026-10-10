#!/usr/bin/env python3
"""Freeze the v2.2.0 promotion inputs and classify each closed promotion route.

The reviewed route-set digest makes an added or renamed v2.2.0 refusal a hard
failure instead of silently assigning it a disposition.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = "v2.2.0"
OUTPUT = ROOT / "parity/promotion_baseline.md"
REVIEWED_CLOSED_DIGEST = "f937e5bf987fa857c70e38df8788f2b8acea5b29f542009ef47bb58e635db968"
FILES = (
    # The registry was named per release at the tag; the path is the tag's path.
    "parity/promotion_2_2.toml",
    "parity/support_licensed.toml",
    "parity/support_closed.toml",
    "parity/transport_stages.toml",
    "parity/transport_coverage.md",
    "parity/counterfactual_coverage.md",
    "parity/coverage_records.toml",
    "parity/calibration_backlog.md",
)

# A candidate is a scoped investigation, not a license. Every other route in
# the reviewed 66-route set is intentionally retained as a deliberate refusal.
# No refusal is superseded merely because an adjacent method is planned.
CARRYOVER: dict[str, str] = {
    "antecedent.transport.multi_source_z_transport.uncertainty_joint_bootstrap": "multi-source limited-experiment transport",
    "antecedent.PreparedTransportScenarios.aggregate_interval": "finite scenario envelope",
    "antecedent.transport.advanced.PreparedTransportScenariosStage.aggregate_interval": "finite scenario envelope",
    "transport.scenarios.aggregate_inference": "finite scenario envelope",
    "transport.scenarios.equivalence_class_native": "finite scenario envelope",
    "antecedent.transport.advanced.PreparedTemporalTransportStage.interval": "two-step temporal transport",
    "transport.temporal.uncertainty": "two-step temporal transport",
    "antecedent.transport.advanced.MixedSourceStage.prepare_empirical": "mixed-source proof search",
    "antecedent.transport.advanced.AdmgConditionalTransportStage.prepare_empirical": "ADMG conditional transport",
    "antecedent.transport.advanced.ObservationRecoveryStage.prepare_empirical": "binary observation recovery",
    "antecedent.PreparedSmoothedDose.interval": "smoothed dose-response transport",
    "antecedent.transport.advanced.PreparedSmoothedDose.interval": "smoothed dose-response transport",
    "antecedent.transport.smoothed_dose.uncertainty_joint_outer_bootstrap": "smoothed dose-response transport",
    "antecedent.PreparedZTransport.joint_mechanism_sensitivity_interval": "joint mechanism sensitivity",
    "antecedent.transport.advanced.joint_mechanism_sensitivity_interval": "joint mechanism sensitivity",
    "antecedent.transport.joint_sensitivity.uncertainty_conservative_endpoint_bootstrap": "joint mechanism sensitivity",
    "antecedent.counterfactual_id.path_specific_admg": "path-specific counterfactual identification",
    "antecedent.inverse.probability_target": "finite-action inverse outcome",
    "antecedent.inverse.quantile_target": "finite-action inverse outcome",
    "antecedent.inverse.observational_scenarios": "finite-action inverse outcome",
    "antecedent_estimate.AipwAte.cluster_dml_interval": "clustered DML",
    "antecedent_estimate.ClusterDml.dyadic": "clustered DML",
    "antecedent.estimators.ClusterDml.dyad_structure": "clustered DML",
    "antecedent_estimate.refuse_joint_inference": "vector-treatment joint inference",
    "antecedent_estimate.refuse_joint_learner_inference": "vector-treatment joint inference",
    "antecedent.derived.interval": "vector-treatment joint inference",
    "antecedent.derived.ml_interval": "vector-treatment joint inference",
    "antecedent.PreparedBatch.partial_family": "vector-treatment joint inference",
    "antecedent.estimate_with_rank_drop.joint_cell": "vector-treatment joint inference",
}


# The tag's record ids encoded the development stream that produced them; the
# same records are listed under their current science-named ids.
RECORD_ID: dict[str, str] = {
    "2.2A.X1.multi_source_limited_experiment": "transport.multi_source_limited_experiment",
    "2.2A.X2.finite_scenario_envelope": "transport.finite_scenario_envelope",
    "2.2A.X4.learned_continuous_trial_transport": "transport.learned_continuous_trial",
    "2.2A.X5.two_step_temporal_transport": "transport.two_step_temporal",
    "2.2A.X8.path_specific_edge_intervention": "counterfactual.path_specific_edge",
    "2.2A.X9.mixed_source_proof_search": "identify.mixed_source_proof_search",
    "2.2B.X2.admg_conditional_transport": "transport.admg_conditional",
    "2.2B.X6.study_planning": "design.study_planning",
    "2.2B.X10.binary_observation_recovery": "recovery.binary_observation",
    "2.2B.X4.smoothed_dose_response_transport": "transport.smoothed_dose_response",
    "2.2B.X3.joint_mechanism_sensitivity": "sensitivity.joint_mechanism",
    "2.2B.X3.joint_sensitivity_uncertainty": "sensitivity.joint_uncertainty",
    "2.2B.X8.admg_counterfactual_id": "counterfactual.admg_identification",
    "2.2E.E7.matched_case_control_odds_ratio": "estimate.matched_case_control_odds_ratio",
    "2.2E.E6.finite_action_inverse_outcome": "decision.finite_action_inverse",
    "2.2E.E2.penalized_propensity_aipw": "estimate.penalized_propensity_aipw",
    "2.2E.E4.clustered_dml_aipw": "estimate.clustered_dml_aipw",
    "2.2E.E7.descriptive_comparison": "estimate.descriptive_comparison",
    "2.2E.E7.tier_diagnostics": "estimate.tier_diagnostics",
    "2.2E.E5.factorized_joint_cells": "estimate.factorized_joint_cells",
    "2.2E.E3.batch_retarget_covariance_contrasts": "estimate.batch_retarget_covariance",
    "2.2E.E3.dml_score_covariance": "estimate.dml_score_covariance",
    "2.2E.E1.preflight_rank_drop_estimation": "estimate.preflight_rank_drop",
}


def at_tag(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{TAG}:{path}"], cwd=ROOT)


def render() -> str:
    commit = subprocess.check_output(
        ["git", "rev-list", "-n", "1", TAG], cwd=ROOT, text=True
    ).strip()
    promotion = tomllib.loads(at_tag(FILES[0]).decode())
    closed = [
        (RECORD_ID.get(record["id"], record["id"]), route)
        for record in promotion["record"]
        for route in record["routes"]
        if route["status"] == "closed"
    ]
    names = {route["name"] for _, route in closed}
    digest = hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()
    if len(closed) != 66 or digest != REVIEWED_CLOSED_DIGEST:
        raise ValueError("v2.2.0 closed-route set changed; review every added or renamed refusal")
    if unknown := set(CARRYOVER) - names:
        raise ValueError(f"carryover references absent routes: {sorted(unknown)}")

    lines = [
        "# Frozen v2.2.0 baseline",
        "",
        f"Frozen source: annotated `{TAG}` at `{commit}`. This snapshot reads the tag, not the working tree.",
        "",
        "Release v2.2.0 is [published](https://github.com/iridae-dev/antecedent/releases/tag/v2.2.0).",
        "Its [CI run](https://github.com/iridae-dev/antecedent/actions/runs/37429694924) completed successfully on the tag commit, including required Rust, Python, dependency, wheel and domain jobs. The [publish-release run](https://github.com/iridae-dev/antecedent/actions/runs/37434348369) passed its tag/version/CI verification, wheel provenance, docs bundle and publication jobs on the same SHA. The workspace and Python package at the tag both declare `2.2.0`; the checked-in release notes describe the licensed analytic learned-continuous interval and the failed percentile method separately. These records establish the tag; they do not separately attest a local invocation of `gate_release_candidate.sh`.",
        "",
        "At the tag, `generate_calibration_backlog.py --check` reports **40 unmeasured cells / 18 distinct coordinates**. `check_release_claims.py --final` and `check_limits_agreement.py --strict` pass against the v2.2.0 registry and notes.",
        "",
        "## Immutable file snapshot",
        "",
        "| File at v2.2.0 | SHA-256 |",
        "| --- | --- |",
    ]
    lines += [
        f"| `{path}` | `{hashlib.sha256(at_tag(path)).hexdigest()}` |" for path in FILES
    ]
    lines += [
        "",
        "## Closed-route disposition",
        "",
        "A carryover candidate requires its own theorem, provider, value, refusal, artifact and calibration gates before opening. A deliberate refusal keeps its v2.2.0 reason code. No route is declared superseded. A listed candidate remains closed today.",
        "",
        f"Reviewed set: {len(closed)} routes; {len(CARRYOVER)} carryover candidates; {len(closed) - len(CARRYOVER)} deliberate refusals; 0 superseded.",
        "",
        "| Record | Closed route | Reason code | Disposition | Related capability |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record_id, route in closed:
        name = route["name"]
        disposition = "carryover candidate" if name in CARRYOVER else "deliberate refusal"
        lines.append(
            f"| `{record_id}` | `{name}` | `{route['reason_code']}` | {disposition} | {CARRYOVER.get(name, 'retain boundary')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = render()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text() != content:
            parser.error("promotion_baseline.md is stale")
    else:
        OUTPUT.write_text(content)
    print("v2.2.0 baseline: 66 closed routes classified")


if __name__ == "__main__":
    main()
