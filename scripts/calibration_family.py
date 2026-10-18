"""Freeze the complete gate roster and reviewed source-fixed coverage-label family.

The per-group cap is a fail-closed guard, not a justification for data-adaptive
hypothesis selection. The reviewed source footprint is hashed separately; changed
label-producing code requires another explicit review before a new family is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from functools import cache
from pathlib import Path

import calibration_decision as decision
from _repo import REPO_ROOT as ROOT
from _rust_source import lex

MANIFEST_PATH = "scripts/calibration_family_manifest.json"
AUDIT_PATH = "scripts/calibration_family_labels.json"
GATE_PATH = "scripts/gate_calibration.sh"
ACCEPTANCE_PATH = "parity/acceptance_manifest.toml"
SCHEMA = "source_fixed_calibration_family.v1"

# Reviewed producer provenance, not an automated proof from a runtime cell cap.
LABEL_REVIEW = {
    "discovered_transport_calibration.rs": "Literal protocol versions; fixed requested discovery/transport construction, never measured coverage.",
    "calibration_backlog.rs": "Prespecified mediation contrast labels and dose GRID/curve-dose arrays; helper labels passed from fixed query constructors.",
    "nested_markov_bayesian_calibration.rs": "Caller-declared functional labels from fixed nominal/boundary test configurations.",
    "sample_split_parent_set_effect_calibration.rs": "Literal producer-version labels; parent discovery outcomes do not choose a coverage hypothesis.",
    "temporal_response_calibration.rs": "Fixed test/horizon/dose procedures. Atom tallies are created after simulation from observed adjustment sets, but all labels belong to the fixed graph subset universe: CPDAG at most2 sets x3 cells x3 tests, PAG at most4 sets x3 cells x3 tests. Source-possible whole-file bound162, including empty/unobserved allowed subsets; observed historical135 is not the family definition.",
    "calibration_bayesian_static.rs": "Optional labels supplied by declared static query coordinates; finite source case/dose arrays.",
    "temporal_frequentist.rs": "Prespecified functional contrast labels at two source-fixed confidence levels.",
    "static_calibration.rs": "Constant dose GRID, fixed contrast tables, and literal linear-outcome labels.",
    "weighted_interval_calibration.rs": "Literal interval-construction versions; no labels selected using calibration outcomes.",
    "panel_calibration.rs": "Two fixed dose labels and deterministic sample-grid unit count.",
    "calibration_temporal_class.rs": "Optional labels from finite caller-declared query contrast cases.",
    "bayesian_temporal.rs": "Finite declared component-name tuples, deterministically lowercased before simulation.",
    "derivative_calibration.rs": "Fixed derivative-coordinate indices from the declared finite target dimension.",
    "calibration_response.rs": "Dose/shift/horizon labels from fixed caller query grids, constructed before replicate mapping.",
    "calibration_temporal.rs": "Caller-supplied declared temporal-query labels; no observed-statistic label generation.",
    "mixture_interval_calibration.rs": "Literal x=1, fixed dose grids, and finite direct/mediated/total contrast tuples.",
    "decision_reliability_interval_calibration.rs": "Finite declared coupling labels and literal shared-rho construction; selected action remains one fixed procedure target.",
    "calibration_counterfactual.rs": "Declared counterfactual-world labels from fixed query construction.",
    "practitioner_scale_calibration.rs": "Source-fixed named dose/horizon coordinates and two declared confidence levels.",
    "adaptive_whole_procedure_calibration.rs": "Literal selected_contrast scores the whole selection algorithm, rather than choosing a hypothesis from observed coverage.",
    "nested_markov_fisher_calibration.rs": "Debug rendering of a finite caller-declared functional enum.",
}


# Every additional producer/consumer receives a named review; there is no generic
# approval for a new unlabelled constructor file.
SOURCE_REVIEW = {
    "calibration_coverage.rs": "Runtime coverage consumer/tests only; no Monte Carlo hypothesis producer.",
    "adaptive_inference_calibration.rs": "Fixed declared adaptive procedure, arm truths, stopping rules and source-selected confidence levels; whole-procedure target is fixed.",
    "adaptive_procedure_prior_predictive_calibration.rs": "Fixed prior-predictive generators and procedure configurations; calibration outcomes do not select the prior or hypothesis.",
    "cohort_linear_effect_calibration.rs": "Fixed cohort test names, source-declared effect/truth and confidence levels.",
    "candidate_calibration.rs": "Shared candidate tally constructor takes caller static test/DGP/method identity and declared confidence; callers are source-fixed gate cases.",
    "count_fitted_posterior_calibration.rs": "Fixed count-family posterior construction, truths and test confidence.",
    "counterfactual_transport_sampled.rs": "Fixed sampled counterfactual transport test configurations and static record keys.",
    "coverage_determinism.rs": "Small explicit deterministic test constructors; no measured outcome selects a family.",
    "joint_bayesian_transport_calibration.rs": "Fixed declared joint Bayesian transport cases/levels and static producer keys.",
    "knn_matching_calibration.rs": "One fixed matching interval whole procedure, source-named tally and confidence.",
    "learned_joint_calibration.rs": "Fixed learned-joint/query cases and declared confidence; the learning algorithm is part of each fixed procedure.",
    "observational_effects_calibration.rs": "Fixed source-named effect/policy procedures; policy selection is inside the scored fixed procedure, not selected from empirical coverage.",
    "quadratic_dose_calibration.rs": "Fixed quadratic-dose contrasts and test-declared confidence.",
    "recovery_whole_method_calibration.rs": "Fixed whole-method recovery cases including refusals; static truths/producer identities.",
    "temporal_unit_calibration.rs": "Fixed temporal units, query configurations and confidence levels.",
    "coverage_records_data.rs": "Embedded registry consumer/tests only; no calibration hypothesis constructor.",
    "batch_retarget_calibration.rs": "Finite prespecified retarget rows and batch contrasts with static test keys.",
    "calibration_admg.rs": "Fixed ADMG estimators, generative models and caller static test/level arguments.",
    "calibration_anomaly_bayesian.rs": "Fixed anomaly queries and Bayesian posterior constructions; static record identities.",
    "calibration_attribution_bayesian.rs": "Fixed attribution functionals and declared posterior confidence.",
    "calibration_design.rs": "Fixed randomized-design procedures and named confidence targets.",
    "calibration_design_bayesian.rs": "Fixed randomized Bayesian-design procedures/priors and confidence targets.",
    "calibration_distribution_admg.rs": "Finite fixed distribution-query configurations and static record keys.",
    "calibration_estimate.rs": "Prespecified estimator/DGP/test tuples; constructor levels are declared constants.",
    "calibration_posterior_draws.rs": "Source-fixed posterior draw-count/row-count design cases and confidence.",
    "class_posterior_calibration.rs": "Fixed class-posterior procedures and declared source test configurations.",
    "cluster_dml_calibration.rs": "Fixed whole-cluster DML procedure configurations/targets; no empirical label selection.",
    "calibration_bind.rs": "Shared immutable construction/record-key binding, no calibration outcomes choose hypotheses.",
    "reported.rs": "Shared per-test reporting wrapper; one source-fixed test execution context, no empirical hypothesis selection.",
    "cross_world_interval_calibration.rs": "Fixed cross-world interval/query cases and static record identifiers.",
    "evidence_synthesis_calibration.rs": "Fixed6 study counts across4 declared heterogeneity regimes; distinct source-fixed legacy/new pooling methods and confidence.",
    "joint_law_composed_interval_calibration.rs": "Finite prespecified joint-law composed query constructions and confidence.",
    "joint_sensitivity_calibration.rs": "Fixed sensitivity parameters, model/grid families and named confidence targets.",
    "learned_continuous_calibration.rs": "Fixed learned continuous procedures and generative configurations; learning is inside each fixed scoring procedure.",
    "missing_outcome_calibration.rs": "Prespecified missingness/outcome constructors and static test identities.",
    "multilevel_interval_calibration.rs": "Fixed multilevel interval/query cases and source-declared confidence.",
    "mz_transport_calibration.rs": "Fixed limited-experiment transport queries/configurations and static producer identities.",
    "predictive_score_interval_calibration.rs": "Prespecified predictive-score methods/truths/levels and source-fixed test names.",
    "randomized_interval_calibration.rs": "Fixed randomized procedures and source-declared interval levels.",
    "sample_split_effect_calibration.rs": "Fixed sample-split procedures and named interval targets.",
    "sample_split_weighted_effect_calibration.rs": "Fixed legacy half-split and current three-quarter discovery procedures; current treatment-invariant conflict rule is a fixed method, not selected from coverage outcomes. Producer version labels are finite, with one declared confidence level.",
    "scientific_bridge_interval_calibration.rs": "Finite prespecified scientific bridge queries, models and confidence.",
    "smoothed_dose_calibration.rs": "Source-fixed smoothing/dose query cases and static interval identifiers.",
    "static_envelope_calibration.rs": "Fixed envelope query/functionals; endpoint selection is inside the scored fixed algorithm.",
    "static_mixture_calibration.rs": "Prespecified static mixture/posterior constructions and source-fixed confidence.",
    "temporal_class_calibration.rs": "Fixed temporal graph-class procedures, declared horizons and confidence.",
    "temporal_recalc_calibration.rs": "Fixed temporal recalculation scenarios/queries and static keys.",
    "transport_statistical_calibration.rs": "Finite source-declared transport construction/target cases and confidence.",
    "z_transport_surrogate_calibration.rs": "Fixed surrogate/transport procedures and static producer test identities.",
}
PATH_REVIEW = {
    "crates/antecedent-io-core/src/calibration.rs": "Runtime calibration lookup and its test fixtures; no Monte Carlo family producer.",
    "crates/antecedent/tests/common/calibration.rs": "Shared tally definition plus bounded unit fixtures; actual named hypothesis labels/levels come from audited caller sources.",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def producer_calls(text: str, labels: bool = True) -> list[dict]:
    source = lex(text)
    out = []
    pattern = (
        r"\.labelled\s*\("
        if labels
        else r"\bCoverageTally\s*::\s*(?:new|for_record)\s*\("
    )
    for match in re.finditer(pattern, source.skeleton):
        start = match.end()
        depth = 1
        end = start
        while end < len(source.skeleton) and depth:
            char = source.skeleton[end]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            end += 1
        if depth:
            raise ValueError("unclosed label producer")
        argument = source.kept[start : end - 1].strip()
        out.append(
            {
                "line": text.count("\n", 0, match.start()) + 1,
                "call": source.kept[match.start() : start].strip(),
                "argument": argument,
            }
        )
    return out


def labelled_calls(text: str) -> list[dict]:
    return producer_calls(text)


def _source_paths(root: Path, sha: str | None) -> list[str]:
    if sha:
        result = subprocess.run(
            ["git", "grep", "-l", "CoverageTally", sha, "--", "crates"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode not in (0, 1):
            raise ValueError(result.stderr.strip())
        return sorted(
            line.partition(":")[2]
            for line in result.stdout.splitlines()
            if line.endswith(".rs")
        )
    return sorted(
        path.relative_to(root).as_posix()
        for path in (root / "crates").rglob("*.rs")
        if "CoverageTally" in path.read_text(encoding="utf-8")
    )


def read_source(root: Path, path: str, sha: str | None = None) -> bytes:
    if sha:
        return subprocess.check_output(["git", "show", f"{sha}:{path}"], cwd=root)
    return (root / path).read_bytes()


def reviewed_label_audit(root: Path = ROOT) -> dict:
    sources = []
    for path in _source_paths(root, None):
        data = read_source(root, path)
        calls = labelled_calls(data.decode())
        review = (
            PATH_REVIEW.get(path)
            or LABEL_REVIEW.get(Path(path).name)
            or SOURCE_REVIEW.get(Path(path).name)
        )
        if not review:
            raise ValueError(
                f"new tally producer/consumer requires explicit source-fixed-family review: {path}"
            )
        sources.append(
            {
                "path": path,
                "sha256": _sha(data),
                "labelled_calls": calls,
                "constructor_calls": producer_calls(data.decode(), labels=False),
                "review": review,
            }
        )
    return {
        "schema": SCHEMA,
        "reviewed_cell_cap_per_group": decision.CELL_CAP,
        "review_scope": "Frozen coverage compatibility hypotheses; every level/label is a separate cell. Labels and procedures are source-fixed independently of Monte Carlo coverage outcomes.",
        "source_possible_cap_invariant": "Every group/grid has at most256 source-possible asserted cells across all branch outcomes and registered looks, including recordless, conditionally empty and alternate-confidence tallies. Runtime emitted-count enforcement only complements this source review.",
        "transitive_provider_review": "Caller-provided labels, target names and levels were reviewed at fixed test/query construction sites. Imported functional enums and numeric helper values describe fixed algorithms; none select a coverage hypothesis using aggregate calibration outcomes. The direct constructor/label footprint and named reviews are an audited invariant, not an automated transitive dataflow proof.",
        "prospective_validation_domain": decision.DOMAIN,
        "historical_cardinality_audit": {
            "source_sha": "c30e10304f4bd85a01ad81dd133a360ddc95c2f1",
            "group": "temporal_response_calibration",
            "maximum_distinct_tally_ids": 135,
            "source_possible_upper_bound": 162,
            "source_possible_derivation": "42 curve +24 persistent-treatment +6 intervention +6 sequence +6 Bayesian-intervention +8 selected-observation +6 observation-sequence +10 horizon-dependent +18 CPDAG atom +36 PAG atom cells; adjustment labels range over all fixed-graph eligible subsets, including unobserved subsets.",
            "basis": "Source-defined construction cardinality only; observed coverage values were not used to choose the cap.",
        },
        "sources": sources,
    }


def validate_label_audit(
    audit: dict, root: Path = ROOT, sha: str | None = None
) -> None:
    if (
        audit.get("schema") != SCHEMA
        or audit.get("reviewed_cell_cap_per_group") != decision.CELL_CAP
    ):
        raise ValueError("unknown source-fixed label audit")
    sources = audit.get("sources")
    if not isinstance(sources, list) or [
        row.get("path") for row in sources
    ] != _source_paths(root, sha):
        raise ValueError(
            "tally producer footprint changed; explicitly review the complete source family"
        )
    for row in sources:
        data = read_source(root, row["path"], sha)
        if (
            row.get("sha256") != _sha(data)
            or row.get("labelled_calls") != labelled_calls(data.decode())
            or row.get("constructor_calls")
            != producer_calls(data.decode(), labels=False)
            or not row.get("review")
        ):
            raise ValueError(f"source-fixed label review is stale: {row['path']}")


def gate_roster(gate: bytes, acceptance: bytes) -> list[dict]:
    # Execute only the gate's existing dry-run path in a scratch source tree.
    # This obtains the complete roster even when the caller selected a shard.
    with tempfile.TemporaryDirectory(prefix="calibration-family-") as directory:
        root = Path(directory)
        (root / "scripts").mkdir()
        (root / "parity").mkdir()
        (root / GATE_PATH).write_bytes(gate)
        (root / ACCEPTANCE_PATH).write_bytes(acceptance)
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("ANTECEDENT_CALIBRATION_")
        }
        env.update(
            ANTECEDENT_CALIBRATION_DRY_RUN="1",
            ANTECEDENT_CALIBRATION_NSIM="400",
            ANTECEDENT_CALIBRATION_RECHECK_NSIM="2000",
        )
        result = subprocess.run(
            ["bash", str(root / GATE_PATH)],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
    grid = {
        int(m.group(1))
        for m in re.finditer(r"^grid (\d+)$", result.stdout, re.MULTILINE)
    }
    groups = [
        {"index": int(m.group(1)), "label": m.group(2), "grid": int(m.group(1)) in grid}
        for m in re.finditer(r"^group (\d+): (.+)$", result.stdout, re.MULTILINE)
    ]
    if (
        not groups
        or [g["index"] for g in groups] != list(range(1, len(groups) + 1))
        or len({g["label"] for g in groups}) != len(groups)
        or len({g["label"].translate(str.maketrans(" /:", "___")) for g in groups})
        != len(groups)
    ):
        raise ValueError("gate dry run did not give a complete unique source roster")
    return groups


def build_manifest(root: Path = ROOT, audit: dict | None = None) -> dict:
    audit = audit or json.loads((root / AUDIT_PATH).read_text(encoding="utf-8"))
    validate_label_audit(audit, root)
    gate = read_source(root, GATE_PATH)
    acceptance = read_source(root, ACCEPTANCE_PATH)
    return {
        "schema": SCHEMA,
        "decision_protocol": decision.ID,
        "validation_domain": decision.DOMAIN,
        "family_alpha": decision.ALPHA,
        "family_grid_points": decision.GRID_POINTS,
        "cell_cap_per_group": decision.CELL_CAP,
        "looks": [dict(x) for x in decision.LOOKS],
        "gate_sha256": _sha(gate),
        "acceptance_sha256": _sha(acceptance),
        "label_audit_sha256": decision.digest(audit),
        "groups": gate_roster(gate, acceptance),
    }


def validate_manifest(
    manifest: dict, audit: dict, root: Path = ROOT, sha: str | None = None
) -> dict:
    validate_label_audit(audit, root, sha)
    gate = read_source(root, GATE_PATH, sha)
    acceptance = read_source(root, ACCEPTANCE_PATH, sha)
    expected = {
        "schema": SCHEMA,
        "decision_protocol": decision.ID,
        "validation_domain": decision.DOMAIN,
        "family_alpha": decision.ALPHA,
        "family_grid_points": decision.GRID_POINTS,
        "cell_cap_per_group": decision.CELL_CAP,
        "looks": [dict(x) for x in decision.LOOKS],
        "gate_sha256": _sha(gate),
        "acceptance_sha256": _sha(acceptance),
        "label_audit_sha256": decision.digest(audit),
        "groups": gate_roster(gate, acceptance),
    }
    if manifest != expected:
        raise ValueError(
            "family manifest differs from its complete frozen source roster/audit"
        )
    return decision.config_for_manifest(manifest)


@cache
def config_at_sha(sha: str, root: Path = ROOT) -> dict:
    if not re.fullmatch("[0-9a-f]{40}", sha):
        raise ValueError("family provenance requires a full source SHA")
    manifest = json.loads(read_source(root, MANIFEST_PATH, sha))
    audit = json.loads(read_source(root, AUDIT_PATH, sha))
    return validate_manifest(manifest, audit, root, sha)


def current_config(root: Path = ROOT) -> dict:
    return validate_manifest(
        json.loads((root / MANIFEST_PATH).read_text(encoding="utf-8")),
        json.loads((root / AUDIT_PATH).read_text(encoding="utf-8")),
        root,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--ack-source-fixed-label-review", action="store_true")
    args = parser.parse_args()
    if args.refresh:
        if not args.ack_source_fixed_label_review:
            raise SystemExit(
                "refresh requires explicit --ack-source-fixed-label-review after reviewing finite label/target producers"
            )
        audit = reviewed_label_audit()
        manifest = build_manifest(audit=audit)
        (ROOT / AUDIT_PATH).write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        (ROOT / MANIFEST_PATH).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    config = current_config()
    print(decision.canonical_json(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
