#!/usr/bin/env python3
"""Freeze the released 2.2 inputs and review each closed promotion route for 2.3.

The reviewed route-set digest makes an added or renamed 2.2 refusal a hard
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
OUTPUT = ROOT / "parity/baseline_2_2_for_2_3.md"
REVIEWED_CLOSED_DIGEST = "f937e5bf987fa857c70e38df8788f2b8acea5b29f542009ef47bb58e635db968"
FILES = (
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
# No 2.2 refusal is superseded merely because 2.3 plans an adjacent method.
CARRYOVER: dict[str, str] = {
    "antecedent.transport.multi_source_z_transport.uncertainty_joint_bootstrap": "C0 X1",
    "antecedent.PreparedTransportScenarios.aggregate_interval": "A1 X2",
    "antecedent.transport.advanced.PreparedTransportScenariosStage.aggregate_interval": "A1 X2",
    "transport.scenarios.aggregate_inference": "A1 X2",
    "transport.scenarios.equivalence_class_native": "A1 X2",
    "antecedent.transport.advanced.PreparedTemporalTransportStage.interval": "A4 X5",
    "transport.temporal.uncertainty": "A4 X5",
    "antecedent.transport.advanced.MixedSourceStage.prepare_empirical": "B2 X9",
    "antecedent.transport.advanced.AdmgConditionalTransportStage.prepare_empirical": "B1 X4",
    "antecedent.transport.advanced.ObservationRecoveryStage.prepare_empirical": "A6 X10",
    "antecedent.PreparedSmoothedDose.interval": "C0 X4",
    "antecedent.transport.advanced.PreparedSmoothedDose.interval": "C0 X4",
    "antecedent.transport.smoothed_dose.uncertainty_joint_outer_bootstrap": "C0 X4",
    "antecedent.PreparedZTransport.joint_mechanism_sensitivity_interval": "C0 X3",
    "antecedent.transport.advanced.joint_mechanism_sensitivity_interval": "C0 X3",
    "antecedent.transport.joint_sensitivity.uncertainty_conservative_endpoint_bootstrap": "C0 X3",
    "antecedent.counterfactual_id.path_specific_admg": "B2 X8",
    "antecedent.inverse.probability_target": "B4 F7",
    "antecedent.inverse.quantile_target": "B4 F7",
    "antecedent.inverse.observational_scenarios": "B4 F7",
    "antecedent_estimate.AipwAte.cluster_dml_interval": "B1 X4",
    "antecedent_estimate.ClusterDml.dyadic": "B1 X4",
    "antecedent.estimators.ClusterDml.dyad_structure": "B1 X4",
    "antecedent_estimate.refuse_joint_inference": "B4 vector treatment",
    "antecedent_estimate.refuse_joint_learner_inference": "B4 vector treatment",
    "antecedent.derived.interval": "B4 vector treatment",
    "antecedent.derived.ml_interval": "B4 vector treatment",
    "antecedent.PreparedBatch.partial_family": "B4 vector treatment",
    "antecedent.estimate_with_rank_drop.joint_cell": "B4 vector treatment",
}


def at_tag(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{TAG}:{path}"], cwd=ROOT)


def render() -> str:
    commit = subprocess.check_output(
        ["git", "rev-list", "-n", "1", TAG], cwd=ROOT, text=True
    ).strip()
    promotion = tomllib.loads(at_tag(FILES[0]).decode())
    closed = [
        (record["id"], route)
        for record in promotion["record"]
        for route in record["routes"]
        if route["status"] == "closed"
    ]
    names = {route["name"] for _, route in closed}
    digest = hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()
    if len(closed) != 66 or digest != REVIEWED_CLOSED_DIGEST:
        raise ValueError("2.2 closed-route set changed; review every added or renamed refusal")
    if unknown := set(CARRYOVER) - names:
        raise ValueError(f"carryover references absent routes: {sorted(unknown)}")

    lines = [
        "# 2.2 baseline for the 2.3 work",
        "",
        f"Frozen source: annotated `{TAG}` at `{commit}`. This snapshot reads the tag, not the 2.3 working tree.",
        "",
        "The 2.2 release is [published](https://github.com/iridae-dev/antecedent/releases/tag/v2.2.0).",
        "Its [CI run](https://github.com/iridae-dev/antecedent/actions/runs/37429694924) completed successfully on the tag commit, including required Rust, Python, dependency, wheel and domain jobs. The [publish-release run](https://github.com/iridae-dev/antecedent/actions/runs/37434348369) passed its tag/version/CI verification, wheel provenance, docs bundle and publication jobs on the same SHA. The workspace and Python package at the tag both declare `2.2.0`; the checked-in release notes describe the licensed analytic learned-continuous interval and the failed percentile method separately. These records establish the release cut; they do not separately attest a local invocation of `gate_release_candidate.sh`.",
        "",
        "At 2.3 kickoff, `generate_calibration_backlog.py --check` reports **40 unmeasured cells / 18 distinct coordinates**. `check_release_claims.py --final` and `check_limits_agreement.py --strict` pass against the 2.2 registry and notes.",
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
        "A carryover candidate requires its own 2.3 theorem, provider, value, refusal, artifact and calibration gates before opening. A deliberate refusal keeps its 2.2 reason code. No route is declared superseded at kickoff. A listed candidate remains closed today.",
        "",
        f"Reviewed set: {len(closed)} routes; {len(CARRYOVER)} carryover candidates; {len(closed) - len(CARRYOVER)} deliberate refusals; 0 superseded.",
        "",
        "| 2.2 record | Closed route | Reason code | Disposition | Next review |",
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
            parser.error("baseline_2_2_for_2_3.md is stale")
    else:
        OUTPUT.write_text(content)
    print("2.2 baseline: 66 closed routes classified")


if __name__ == "__main__":
    main()
