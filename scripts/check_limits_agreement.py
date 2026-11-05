#!/usr/bin/env python3
"""Declared integer bounds agree with the code that enforces them.

For every regex-mapped record of parity/capability_records.toml, each integer entry of its
`bounds` table must equal the value read from each place mapped to it in `MAPPING` below:

  rust      a Rust constant (`const NAME: ty = <int | OTHER_CONST>;`) or an integer literal picked
            out of a source file by a regex, in the cell's own source files;
  python    a Python module constant or `def f(x: int = N)` default of the cell's Python module.

Only code and structured data are read: no documentation, docstring, comment or refusal text,
and no number inside a string-valued `bounds` entry. `docs/supported-bounds.md` carries a
generated table of the exact-record bounds (marker `generated:supported-bounds`); the check
compares that marker to a fresh render. `parity/limits.toml` names implementation, guard,
refusal and fixture places in code.

The check fails, naming the record, the bound and the place, when a mapped constant, regex or
statement is missing or differs, and when a record carries an integer bound that has no mapping
entry at all (so a new record cannot ship a bound nobody compares). An unmapped bound of a
`not_supported` record is reported as a `gap`; `--strict` makes gaps failures.

Extending it for a new record: add `Bound(...)` rows for the record's id to `MAPPING` and list
the id in `CELLS`. `rust=` and `python=` take `C(path, NAME)` (a constant) or `L(path, regex)`
(first capture group is the number; every match in the file must agree).

    python3 scripts/check_limits_agreement.py             # check the tree
    python3 scripts/check_limits_agreement.py --strict    # gaps also fail
    python3 scripts/check_limits_agreement.py --self-test # synthetic disagreements must fail

Records outside `CELLS` are compared by scripts/check_limits.py (exact record ids, parity/limits.toml);
both run on every invocation.

Environment: LIMITS_AGREEMENT_ROOT (tree to read, default the repo; used by the self-test).
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

from _repo import REPO_ROOT as REPO

ROOT = Path(os.environ.get("LIMITS_AGREEMENT_ROOT", REPO))
RECORDS = "parity/capability_records.toml"
# Records whose bounds are compared here by regex-mapped places; every other record is
# compared by scripts/check_limits.py.
CELLS = frozenset(
    {
        "transport.multi_source_limited_experiment",
        "transport.finite_scenario_envelope",
        "transport.learned_continuous_trial",
        "transport.two_step_temporal",
        "counterfactual.path_specific_edge",
        "identify.mixed_source_proof_search",
        "transport.admg_conditional",
        "design.study_planning",
        "recovery.binary_observation",
        "transport.smoothed_dose_response",
        "sensitivity.joint_mechanism",
        "sensitivity.joint_uncertainty",
        "counterfactual.admg_identification",
        "estimate.matched_case_control_odds_ratio",
        "estimate.descriptive_comparison",
        "estimate.tier_diagnostics",
        "decision.finite_action_inverse",
        "estimate.penalized_propensity_aipw",
        "estimate.clustered_dml_aipw",
        "estimate.factorized_joint_cells",
        "estimate.batch_retarget_covariance",
        "estimate.dml_score_covariance",
        "estimate.preflight_rank_drop",
    }
)



@dataclass(frozen=True)
class Place:
    """A file and how to read one integer from it."""

    path: str
    kind: str  # "const", "regex", "snapshot_bound", or "reporting_scale_kinds"
    pattern: str


def C(path: str, name: str) -> Place:
    return Place(path, "const", name)


def L(path: str, regex: str) -> Place:
    return Place(path, "regex", regex)


def S(record: str, key: str) -> Place:
    return Place(
        "parity/release_snapshots/v2.2.0.toml", "snapshot_bound", f"{record}:{key}"
    )


@dataclass(frozen=True)
class Bound:
    record: str  # full record id in parity/capability_records.toml
    key: str  # key of the record's `bounds` table
    rust: tuple[Place, ...] = ()
    python: tuple[Place, ...] = ()
    note: str = field(default="", compare=False)


MZ = "crates/antecedent-identify/src/sid/mz_transport.rs"
MZ_PY = "python/antecedent/transport/_multi_source.py"
SCEN = "crates/antecedent-identify/src/sid/scenarios.rs"
SCEN_PY = "python/antecedent/transport/_scenarios.py"
LEARN = "crates/antecedent-estimate/src/learned_continuous.rs"
LEARN_IO = "crates/antecedent-io/src/learned_continuous_artifact.rs"
LEARN_PY = "python/antecedent/transport/_learned_continuous.py"
TEMP = "crates/antecedent-identify/src/sid/temporal_sequence.rs"
TEMP_PY = "python/antecedent/transport/_temporal.py"
XW_ID = "crates/antecedent-identify/src/cross_world.rs"
XW_IO = "crates/antecedent-io-core/src/cross_world_artifact.rs"
MIX = "crates/antecedent-identify/src/sid/mixed_source.rs"
MIX_PY = "python/antecedent/transport/_mixed_source.py"

# ---- cells
COND = "crates/antecedent-identify/src/sid/conditional.rs"
COND_PY = "python/antecedent/transport/_admg_conditional.py"
PLAN = "crates/antecedent-identify/src/sid/study_planning.rs"
PLAN_PY = "python/antecedent/transport/_study_planning.py"
REC = "crates/antecedent-identify/src/recovery.rs"
REC_PY = "python/antecedent/transport/_recovery.py"
DOSE = "crates/antecedent-estimate/src/smoothed_dose.rs"
DOSE_PY = "python/antecedent/transport/_smoothed_dose.py"
JOINT = "crates/antecedent-validate/src/joint_mechanism_sensitivity.rs"
JOINT_PY = "python/antecedent/transport/_joint_sensitivity.py"
CFID = "crates/antecedent-identify/src/counterfactual_id.rs"
CFID_PY = "python/antecedent/counterfactual_id.py"
MATCHED = "crates/antecedent-estimate/src/matched_case_control.rs"
MATCHED_PY = "python/antecedent/matched.py"
INVERSE = "crates/antecedent/src/analysis/inverse_outcome.rs"
PENALIZED = "crates/antecedent-estimate/src/propensity/penalized.rs"
FOLD_PLAN = "crates/antecedent-learn/src/crossfit.rs"
CLUSTER_DML = "crates/antecedent-estimate/src/cluster_dml_aipw.rs"
DESCRIPTIVE = "crates/antecedent-estimate/src/descriptive_comparison.rs"
JOINT_CELLS = "crates/antecedent-estimate/src/joint_cell_factorized.rs"
BATCH_RETARGET = "crates/antecedent/src/analysis/batch_retarget.rs"
JOINT_IF = "crates/antecedent-estimate/src/joint_if.rs"


# The mapping table: one row per (record, bounds key). Extend with additive rows for new records.
MAPPING: tuple[Bound, ...] = (
    # ---- estimator and decision cells
    Bound(
        "estimate.matched_case_control_odds_ratio",
        "max_iterations",
        rust=(C(MATCHED, "MAX_ITERATIONS"),),
    ),
    Bound(
        "estimate.matched_case_control_odds_ratio",
        "replay_max_rows",
        python=(L(MATCHED_PY, r"^_MAX_REPLAY_ROWS = ([\d_]+)"),),
    ),
    Bound(
        "estimate.penalized_propensity_aipw",
        "historical_evidence_max_grid",
        rust=(S("estimate.penalized_propensity_aipw", "max_grid"),),
        note="prior recorded scope; the per-declaration configuration cap is separate",
    ),
    Bound(
        "estimate.penalized_propensity_aipw",
        "historical_evidence_max_inner_folds",
        rust=(S("estimate.penalized_propensity_aipw", "max_inner_folds"),),
        note="prior recorded scope; the fold plan enforces its row and index limits",
    ),
    Bound(
        "estimate.penalized_propensity_aipw",
        "runtime_min_inner_folds",
        rust=(L(PENALIZED, r"if inner_folds < (\d+)"),),
    ),
    Bound(
        "estimate.penalized_propensity_aipw",
        "runtime_fold_index_capacity",
        rust=(Place(FOLD_PLAN, "fold_plan_capacity", "assign_folds"),),
    ),
    Bound(
        "estimate.clustered_dml_aipw",
        "default_min_clusters",
        rust=(C(CLUSTER_DML, "DEFAULT_MIN_CLUSTERS"),),
    ),
    Bound(
        "estimate.clustered_dml_aipw",
        "min_clusters_floor",
        rust=(C(CLUSTER_DML, "MIN_CLUSTERS_FLOOR"),),
    ),
    Bound(
        "estimate.clustered_dml_aipw",
        "default_min_components_per_fold",
        rust=(C(CLUSTER_DML, "DEFAULT_MIN_COMPONENTS_PER_FOLD"),),
    ),
    Bound(
        "estimate.clustered_dml_aipw",
        "min_components_per_fold_floor",
        rust=(C(CLUSTER_DML, "MIN_COMPONENTS_PER_FOLD_FLOOR"),),
    ),
    Bound(
        "estimate.descriptive_comparison",
        "reporting_scale_kinds",
        rust=(Place(DESCRIPTIVE, "reporting_scale_kinds", "ReportingScale"),),
        note="distinct supported kinds, not a maximum number of requested transforms",
    ),
    Bound(
        "estimate.factorized_joint_cells",
        "max_components",
        rust=(C(JOINT_CELLS, "MAX_FACTORIZED_COMPONENTS"),),
    ),
    Bound(
        "estimate.factorized_joint_cells",
        "runtime_min_folds",
        rust=(L(JOINT_CELLS, r"if self\.folds < (\d+)"),),
    ),
    Bound(
        "estimate.factorized_joint_cells",
        "runtime_fold_index_capacity",
        rust=(Place(FOLD_PLAN, "fold_plan_capacity", "assign_folds"),),
    ),
    Bound(
        "estimate.factorized_joint_cells",
        "historical_evidence_max_folds",
        rust=(S("estimate.factorized_joint_cells", "max_folds"),),
        note="prior recorded scope; the fold plan enforces its row and index limits",
    ),
    Bound(
        "estimate.factorized_joint_cells",
        "max_orderings",
        rust=(C(JOINT_CELLS, "MAX_ORDERINGS"),),
    ),
    # ---- mz-transportability
    Bound(
        "transport.multi_source_limited_experiment",
        "max_search_states",
        rust=(
            L(
                MZ,
                r"MZ_TRANSPORT_DEFAULT_LIMITS: SearchLimits = SearchLimits \{ operations: ([\d_]+),",
            ),
        ),
        python=(
            L(MZ_PY, r"catalog: EvidenceCatalog,\n\s+max_operations: int = ([\d_]+),"),
            L(MZ_PY, r"max_search_operations: int = ([\d_]+)"),
        ),
    ),
    Bound(
        "transport.multi_source_limited_experiment",
        "max_depth",
        rust=(
            L(
                MZ,
                r"MZ_TRANSPORT_DEFAULT_LIMITS: SearchLimits = SearchLimits \{ operations: [\d_]+, depth: ([\d_]+)",
            ),
        ),
        python=(
            L(
                MZ_PY,
                r"catalog: EvidenceCatalog,\n\s+max_operations: int = [\d_]+,\n\s+max_depth: int = ([\d_]+),",
            ),
            L(MZ_PY, r"max_search_depth: int = ([\d_]+),"),
        ),
    ),
    # ---- finite scenario envelope
    Bound(
        "transport.finite_scenario_envelope",
        "max_scenarios",
        rust=(C(SCEN, "SCENARIO_MAX_COUNT"),),
        python=(
            L(SCEN_PY, r"^MAX_SCENARIOS = (\d+)"),
        ),
    ),
    # ---- learned continuous trial transport
    Bound(
        "transport.learned_continuous_trial",
        "min_bootstrap",
        rust=(C(LEARN, "LEARNED_CONTINUOUS_MIN_BOOTSTRAP"),),
    ),
    Bound(
        "transport.learned_continuous_trial",
        "max_rows_consumed",
        rust=(L(LEARN_IO, r"max_rows: ([\d_]+), max_features"),),
    ),
    Bound(
        "transport.learned_continuous_trial",
        "max_features_consumed",
        rust=(L(LEARN_IO, r"max_rows: [\d_]+, max_features: ([\d_]+)"),),
    ),
    # ---- two-step temporal transport
    Bound(
        "transport.two_step_temporal",
        "horizon",
        python=(L(TEMP_PY, r"^HORIZON = (\d+)$"),),
        note="the Python two-step route; the native sequence type supports a general horizon",
    ),
    # ---- path-specific edge intervention (cross-world)
    Bound(
        "counterfactual.path_specific_edge",
        "max_variables",
        rust=(C(XW_ID, "CROSS_WORLD_MAX_NODES"),),
        note="no Python statement of the variable cap",
    ),
    Bound(
        "counterfactual.path_specific_edge",
        "max_worlds",
        rust=(L(XW_ID, r"if worlds\.len\(\) != (\d+) \|\|"),),
        note="the supported contrast requires exactly two worlds; the query type has a separate capacity",
    ),
    Bound(
        "counterfactual.path_specific_edge",
        "max_rows",
        rust=(C(XW_IO, "CROSS_WORLD_MAX_ROWS"),),
        note="no Python statement of the row cap",
    ),
    # ---- mixed-source proof search
    Bound(
        "identify.mixed_source_proof_search",
        "max_observed",
        rust=(C(MIX, "MIXED_SOURCE_MAX_OBSERVED"),),
        note="no Python statement of the observed-variable cap",
    ),
    Bound(
        "identify.mixed_source_proof_search",
        "max_distributions",
        rust=(C(MIX, "MIXED_SOURCE_MAX_DISTRIBUTIONS"),),
        note="no Python statement of the distribution cap",
    ),
    Bound(
        "identify.mixed_source_proof_search",
        "max_search_states",
        rust=(
            L(
                MIX,
                r"MIXED_SOURCE_DEFAULT_LIMITS: SearchLimits =\s*SearchLimits \{ operations: ([\d_]+),",
            ),
        ),
        python=(
            L(MIX_PY, r"catalog: EvidenceCatalog,\n\s+max_operations: int = ([\d_]+),"),
            L(MIX_PY, r"max_search_operations: int = ([\d_]+)"),
        ),
    ),
    Bound(
        "identify.mixed_source_proof_search",
        "max_depth",
        rust=(
            L(
                MIX,
                r"MIXED_SOURCE_DEFAULT_LIMITS: SearchLimits =\s*SearchLimits \{ operations: [\d_]+, depth: ([\d_]+)",
            ),
        ),
        python=(
            L(
                MIX_PY,
                r"catalog: EvidenceCatalog,\n\s+max_operations: int = [\d_]+,\n\s+max_depth: int = ([\d_]+),",
            ),
            L(MIX_PY, r"max_search_depth: int = ([\d_]+),"),
        ),
    ),
    Bound(
        "identify.mixed_source_proof_search",
        "max_move",
        rust=(C(MIX, "MIXED_SOURCE_MAX_MOVE"),),
    ),
    Bound(
        "identify.mixed_source_proof_search",
        "max_intervention_set",
        rust=(C(MIX, "MIXED_SOURCE_MAX_DO"),),
    ),
    Bound(
        "identify.mixed_source_proof_search",
        "max_sources",
        rust=(C(MZ, "MZ_TRANSPORT_MAX_SOURCES"),),
    ),
    # ---- ADMG conditional transport
    Bound(
        "transport.admg_conditional",
        "artifact_max_observed",
        rust=(C(COND, "ADMG_CONDITIONAL_MAX_OBSERVED"),),
    ),
    Bound(
        "transport.admg_conditional",
        "artifact_max_treatments",
        rust=(C(COND, "ADMG_CONDITIONAL_MAX_TREATMENTS"),),
    ),
    Bound(
        "transport.admg_conditional",
        "artifact_max_conditioned",
        rust=(C(COND, "ADMG_CONDITIONAL_MAX_CONDITIONED"),),
    ),
    Bound(
        "transport.admg_conditional",
        "max_search_states",
        rust=(
            L(
                COND,
                r"ADMG_CONDITIONAL_DEFAULT_LIMITS: SearchLimits =\s*SearchLimits \{ operations: ([\d_]+),",
            ),
        ),
        python=(
            L(
                COND_PY,
                r"max_operations: int = ([\d_]+),\n\s+max_depth: int = \d+,\n\s+max_support_rows: int = ",
            ),
            L(COND_PY, r"max_search_operations: int = ([\d_]+)"),
        ),
    ),
    Bound(
        "transport.admg_conditional",
        "max_depth",
        rust=(
            L(
                COND,
                r"ADMG_CONDITIONAL_DEFAULT_LIMITS: SearchLimits =\s*SearchLimits \{ operations: [\d_]+, depth: ([\d_]+)",
            ),
        ),
        python=(
            L(
                COND_PY,
                r"max_operations: int = [\d_]+,\n\s+max_depth: int = ([\d_]+),\n\s+max_support_rows: int = ",
            ),
            L(COND_PY, r"max_search_depth: int = ([\d_]+)"),
        ),
    ),
    # ---- study planning
    Bound(
        "design.study_planning",
        "max_candidates",
        rust=(C(PLAN, "STUDY_PLAN_MAX_CANDIDATES"),),
    ),
    Bound(
        "design.study_planning",
        "max_regimes_per_candidate",
        rust=(C(PLAN, "STUDY_PLAN_MAX_REGIMES_PER_CANDIDATE"),),
        note="no Python statement of the per-candidate regime cap",
    ),
    Bound(
        "design.study_planning",
        "max_subset_size",
        rust=(C(PLAN, "STUDY_PLAN_MAX_SUBSET"),),
    ),
    Bound(
        "design.study_planning",
        "max_proposals",
        rust=(C(PLAN, "STUDY_PLAN_MAX_PROPOSALS"),),
    ),
    Bound(
        "design.study_planning",
        "max_search_states",
        rust=(
            L(
                PLAN,
                r"STUDY_PLAN_DEFAULT_LIMITS: SearchLimits = SearchLimits \{ operations: ([\d_]+),",
            ),
        ),
        python=(L(PLAN_PY, r"^_MAX_OPERATIONS = ([\d_]+)"),),
    ),
    Bound(
        "design.study_planning",
        "max_depth",
        rust=(
            L(
                PLAN,
                r"STUDY_PLAN_DEFAULT_LIMITS: SearchLimits = SearchLimits \{ operations: [\d_]+, depth: ([\d_]+)",
            ),
            L(
                MZ,
                r"MZ_TRANSPORT_DEFAULT_LIMITS: SearchLimits = SearchLimits \{ operations: [\d_]+, depth: ([\d_]+)",
            ),
        ),
        python=(L(PLAN_PY, r'^_MAX_DEPTH = \{"mz": (\d+), "mixed": \d+\}'),),
    ),
    Bound(
        "design.study_planning",
        "max_depth_mixed",
        rust=(
            L(
                MIX,
                r"MIXED_SOURCE_DEFAULT_LIMITS: SearchLimits =\s*SearchLimits \{ operations: [\d_]+, depth: ([\d_]+)",
            ),
        ),
        python=(L(PLAN_PY, r'^_MAX_DEPTH = \{"mz": \d+, "mixed": (\d+)\}'),),
    ),
    # ---- binary observation recovery
    Bound(
        "recovery.binary_observation",
        "max_observed_cells",
        rust=(C(REC, "RECOVERY_MAX_OBSERVED_CELLS"),),
        note="no Python statement of the cell cap",
    ),
    Bound(
        "recovery.binary_observation",
        "max_search_operations",
        rust=(
            L(
                REC,
                r"RECOVERY_DEFAULT_LIMITS: RecoveryLimits = RecoveryLimits \{\n\s+search: SearchLimits \{ operations: ([\d_]+),",
            ),
        ),
        python=(
            L(
                REC_PY,
                r"effect_graph: Admg \| None = None,\n\s+max_operations: int = ([\d_]+),",
            ),
            L(REC_PY, r"max_search_operations: int = ([\d_]+)"),
        ),
    ),
    Bound(
        "recovery.binary_observation",
        "max_depth",
        rust=(
            L(
                REC,
                r"RECOVERY_DEFAULT_LIMITS: RecoveryLimits = RecoveryLimits \{\n\s+search: SearchLimits \{ operations: [\d_]+, depth: ([\d_]+)",
            ),
        ),
        python=(
            L(
                REC_PY,
                r"effect_graph: Admg \| None = None,\n\s+max_operations: int = [\d_]+,\n\s+max_depth: int = ([\d_]+),",
            ),
            L(REC_PY, r"max_search_depth: int = ([\d_]+)"),
        ),
    ),
    # ---- smoothed dose-response transport
    Bound(
        "transport.smoothed_dose_response",
        "max_grid_points",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_GRID"),),
        note="recorded reference-envelope size, not a fixed runtime ceiling",
    ),
    Bound(
        "transport.smoothed_dose_response",
        "max_folds",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_FOLDS"),),
        note="recorded reference-envelope size, not a fixed runtime ceiling",
    ),
    Bound(
        "transport.smoothed_dose_response",
        "max_rows",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_ROWS"),),
        note="recorded reference-envelope size, not a fixed runtime ceiling",
    ),
    Bound(
        "transport.smoothed_dose_response",
        "max_features",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_FEATURES"),),
        note="recorded reference-envelope size, not a fixed runtime ceiling",
    ),
    Bound(
        "transport.smoothed_dose_response",
        "max_basis_degree",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_BASIS_DEGREE"),),
        note="recorded reference-envelope size, not a fixed runtime ceiling",
    ),
    Bound(
        "transport.smoothed_dose_response",
        "max_knots",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_KNOTS"),),
        note="recorded reference-envelope size, not a fixed runtime ceiling",
    ),
    Bound(
        "transport.smoothed_dose_response",
        "min_bootstrap",
        rust=(C(DOSE, "SMOOTHED_DOSE_MIN_BOOTSTRAP"),),
    ),
    Bound(
        "transport.smoothed_dose_response",
        "max_bootstrap",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_BOOTSTRAP"),),
        note="recorded reference-envelope size, not a fixed runtime ceiling",
    ),
    Bound(
        "transport.smoothed_dose_response",
        "max_workspace_bytes",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_WORKSPACE_BYTES"),),
    ),
    # ---- joint mechanism sensitivity
    Bound(
        "sensitivity.joint_mechanism",
        "max_factors",
        rust=(C(JOINT, "JOINT_SENSITIVITY_MAX_FACTORS"),),
    ),
    Bound(
        "sensitivity.joint_mechanism",
        "historical_evidence_max_parent_levels",
        rust=(S("sensitivity.joint_mechanism", "max_parent_levels"),),
        note="historical evidence scope, not a current runtime ceiling",
    ),
    Bound(
        "sensitivity.joint_mechanism",
        "runtime_max_parent_levels",
        rust=(C(JOINT, "JOINT_SENSITIVITY_MAX_PARENT_LEVELS"),),
    ),
    Bound(
        "sensitivity.joint_mechanism",
        "historical_evidence_max_outcome_categories",
        rust=(S("sensitivity.joint_mechanism", "max_outcome_categories"),),
        note="historical evidence scope, not a current runtime ceiling",
    ),
    Bound(
        "sensitivity.joint_mechanism",
        "runtime_max_outcome_categories",
        rust=(C(JOINT, "JOINT_SENSITIVITY_MAX_OUTCOME_CATEGORIES"),),
    ),
    Bound(
        "sensitivity.joint_mechanism",
        "historical_evidence_max_frontier_points",
        rust=(S("sensitivity.joint_mechanism", "max_frontier_points"),),
        note="historical evidence scope, not a current runtime ceiling",
    ),
    Bound(
        "sensitivity.joint_mechanism",
        "runtime_max_frontier_points",
        rust=(C(JOINT, "JOINT_SENSITIVITY_MAX_FRONTIER_POINTS"),),
    ),
    Bound(
        "sensitivity.joint_mechanism",
        "max_operations",
        rust=(C(JOINT, "JOINT_SENSITIVITY_MAX_OPERATIONS"),),
        python=(
            L(JOINT_PY, r"^    max_operations: int = ([\d_]+)$"),
            L(JOINT_PY, r"max_search_operations: int = ([\d_]+)"),
        ),
    ),
    Bound(
        "sensitivity.joint_mechanism",
        "max_bisection_iterations",
        rust=(C(JOINT, "JOINT_SENSITIVITY_MAX_DEPTH"),),
        python=(
            L(JOINT_PY, r"^    max_depth: int = ([\d_]+)$"),
            L(JOINT_PY, r"max_search_depth: int = ([\d_]+)"),
        ),
    ),
    Bound(
        "sensitivity.joint_uncertainty",
        "max_bootstrap_internal",
        rust=(C(JOINT, "JOINT_SENSITIVITY_MAX_BOOTSTRAP"),),
    ),
    # ---- ADMG counterfactual identification
    Bound(
        "counterfactual.admg_identification",
        "artifact_max_variables",
        rust=(C(CFID, "COUNTERFACTUAL_ID_MAX_VARIABLES"),),
    ),
    Bound(
        "counterfactual.admg_identification",
        "artifact_max_levels",
        rust=(C(CFID, "COUNTERFACTUAL_ID_MAX_LEVELS"),),
    ),
    Bound(
        "counterfactual.admg_identification",
        "max_search_operations",
        rust=(C(CFID, "COUNTERFACTUAL_ID_MAX_OPERATIONS"),),
    ),
    Bound(
        "counterfactual.admg_identification",
        "max_search_depth",
        rust=(C(CFID, "COUNTERFACTUAL_ID_MAX_DEPTH"),),
    ),
)


def to_int(text: str) -> int | None:
    t = text.strip().lower().replace(",", "").replace("_", "")
    if t.isdigit():
        return int(t)
    return None


def to_num(text: str) -> int | float | None:
    """An integer (or number word), else a float such as `1e-12`."""
    v = to_int(text)
    if v is not None:
        return v
    if not re.search(r"\d", text):
        return None
    try:
        return float(text.strip().replace("_", ""))
    except ValueError:
        return None


class Reader:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: dict[str, str | None] = {}

    def text(self, rel: str) -> str | None:
        if rel not in self._cache:
            p = self.root / rel
            self._cache[rel] = p.read_text(encoding="utf-8") if p.is_file() else None
        return self._cache[rel]

    def find_const(
        self, name: str, near: str, depth: int = 0, local_only: bool = False
    ) -> tuple[int | float | None, str]:
        """Value of `const NAME: ty = <int|ALIAS>;`. The mapped constant must live in
        `near` (the cell's own file): a same-named constant of another cell would
        otherwise satisfy the mapping. An alias it names is followed from `near` into
        the rest of crates/*/src."""
        files = [near]
        if depth and not local_only:
            files += sorted(
                str(p.relative_to(self.root))
                for p in (self.root / "crates").glob("*/src/**/*.rs")
                if str(p.relative_to(self.root)) != near
            )
        pat = re.compile(
            rf"\bconst {re.escape(name)}\s*:\s*[\w<>\[\]]+\s*=\s*([\w.:\-]+)\s*;"
        )
        for rel in files:
            text = self.text(rel)
            if text is None:
                continue
            m = pat.search(text)
            if not m:
                continue
            raw = m.group(1)
            v = to_num(raw)
            if v is not None:
                return v, rel
            qualified = re.fullmatch(
                r"crate::([a-z_][a-z0-9_]*)::([A-Z][A-Z0-9_]*)", raw
            )
            if qualified and depth < 4:
                # Resolve the declared module, never an unrelated same-named
                # constant elsewhere in the workspace.
                source = self.root / rel
                src = next(
                    (parent for parent in source.parents if parent.name == "src"), None
                )
                if src:
                    for target in (
                        src / f"{qualified[1]}.rs",
                        src / qualified[1] / "mod.rs",
                    ):
                        if target.is_file():
                            return self.find_const(
                                qualified[2],
                                str(target.relative_to(self.root)),
                                depth + 1,
                                True,
                            )
                return None, rel
            if depth < 4 and re.fullmatch(r"[A-Z][A-Z0-9_]+", raw):
                return self.find_const(raw, rel, depth + 1)
            return None, rel
        return None, near


def read_place(rd: Reader, place: Place) -> tuple[list[int | float], str | None]:
    """All integers the place states (must be non-empty and, at the caller, all equal)."""
    text = rd.text(place.path)
    if text is None:
        return [], f"file {place.path} is missing"
    if place.kind == "fold_plan_capacity":
        function = re.search(r"pub fn assign_folds\(.*?\n\}", text, re.S)
        if function is None:
            return [], f"{place.path}: fold plan not found"
        body = function.group()
        capacity = re.findall(
            r"if folds > usize::from\(u(8|16|32)::MAX\) \+ (\d+)", body
        )
        if len(capacity) != 1 or "if n_rows < folds" not in body:
            return [], f"{place.path}: fold-plan capacity or row guard not found"
        bits, extra = capacity[0]
        return [(1 << int(bits)) - 1 + int(extra)], None
    if place.kind == "snapshot_bound":
        record, _, key = place.pattern.partition(":")
        try:
            records = tomllib.loads(text).get("record", [])
        except tomllib.TOMLDecodeError as error:
            return [], f"{place.path}: invalid snapshot: {error}"
        matches = [item for item in records if item.get("id") == record]
        if len(matches) != 1:
            return [], f"{place.path}: snapshot must contain exactly one {record}"
        value = matches[0].get("bounds", {}).get(key)
        if not _is_int(value):
            return [], f"{place.path}: {record} has no integer bound {key}"
        return [value], None
    if place.kind == "reporting_scale_kinds":
        # Count the public enum and independently verify that its parser names
        # exactly the same cases. Duplicates in a request are not a new kind.
        enum = re.search(r"pub enum ReportingScale\s*\{(.*?)^\}", text, re.M | re.S)
        parser = re.search(
            r"pub fn parse\(name: &str\).*?\[([^\]]+)\]\s*\.into_iter\(\)",
            text,
            re.S,
        )
        if enum is None or parser is None:
            return [], f"{place.path}: reporting-scale enum or parser not found"
        enum_lines = [
            line.strip()
            for line in enum.group(1).splitlines()
            if line.strip() and not line.strip().startswith("//")
        ]
        variants = []
        for line in enum_lines:
            variant = re.fullmatch(r"([A-Z][A-Za-z0-9_]*)\s*,", line)
            if variant is None:
                return [], f"{place.path}: unsupported reporting-scale enum declaration"
            variants.append(variant.group(1))
        parsed = re.findall(r"Self::([A-Z][A-Za-z0-9_]*)", parser.group(1))
        if (
            not variants
            or len(set(variants)) != len(variants)
            or sorted(variants) != sorted(parsed)
        ):
            return [], f"{place.path}: reporting-scale enum and parser disagree"
        return [len(variants)], None
    if place.kind == "const":
        v, where = rd.find_const(place.pattern, place.path)
        if v is None:
            return (
                [],
                f"constant {place.pattern} not found as an integer in {place.path} (or its aliases)",
            )
        return [v], None
    out: list[int | float] = []
    for m in re.finditer(place.pattern, text, re.M):
        v = to_int(m.group(1))
        if v is None:
            return (
                [],
                f"{place.path}: {m.group(1)!r} matched by /{place.pattern}/ is not a number",
            )
        out.append(v)
    if not out:
        return [], f"{place.path}: no statement matching /{place.pattern}/"
    return out, None


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


SUPPORTED_BOUNDS = "docs/supported-bounds.md"
BOUNDS_BEGIN = "<!-- generated:supported-bounds:begin -->"
BOUNDS_END = "<!-- generated:supported-bounds:end -->"
LIMITS_REGISTRY = "parity/limits.toml"


def render_supported_bounds_table(root: Path) -> str:
    """Table of exact-record bounds. Values come from capability_records.toml; paths from limits.toml."""
    records = tomllib.loads((root / RECORDS).read_text(encoding="utf-8"))["record"]
    registry = tomllib.loads((root / LIMITS_REGISTRY).read_text(encoding="utf-8"))
    values: dict[tuple[str, str], int] = {}
    for record in records:
        for key, value in record.get("bounds", {}).items():
            if _is_int(value):
                values[(record["id"], key)] = value
    lines = [
        "| Record | Bound | Value | Implementation |",
        "| --- | --- | ---: | --- |",
    ]
    for row in registry["bound"]:
        identity = (row["record"], row["key"])
        if identity not in values:
            raise ValueError(f"{identity}: numeric bound missing from {RECORDS}")
        path = row["implementation"][0]["path"]
        lines.append(
            f"| {row['record']} | {row['key']} | {values[identity]} | `{path}` |"
        )
    return "\n".join(lines) + "\n"


def supported_bounds_errors(root: Path) -> list[str]:
    """The generated marker must equal the render. Prose outside it is not read."""
    path = root / SUPPORTED_BOUNDS
    try:
        expected = render_supported_bounds_table(root)
    except (OSError, ValueError, KeyError, tomllib.TOMLDecodeError) as error:
        return [f"{SUPPORTED_BOUNDS}: {error}"]
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    begin = text.find(BOUNDS_BEGIN)
    end = text.find(BOUNDS_END)
    if begin < 0 or end < begin:
        return [f"{SUPPORTED_BOUNDS} is missing the generated bounds markers"]
    interior = text[begin + len(BOUNDS_BEGIN) : end]
    if interior != "\n" + expected:
        return [f"{SUPPORTED_BOUNDS} generated table does not match {RECORDS} [bounds]"]
    return []


# These are source-contract checks, not a memory admission or calibration claim.
# The 256 MiB policy is a conservative per-declaration cap; it does not reserve
# memory or charge later copies and concurrent declarations.
RIDGE_RESOURCE_FILES = (
    PENALIZED,
    "python/src/estimator_config.rs",
    "python/src/derived_treatment_api.rs",
    "python/antecedent/_penalty.py",
    "python/antecedent/estimators.py",
    "python/antecedent/derived.py",
)


def ridge_configuration_guard_errors(sources: dict[str, str]) -> list[str]:
    """Check bounded construction and Python boundaries; missing mappings fail closed."""
    errors: list[str] = []

    def scoped(path: str, start: str, end: str, tokens: tuple[str, ...]) -> None:
        source = sources.get(path, "")
        begin = source.find(start)
        finish = source.find(end, begin + len(start)) if begin >= 0 else -1
        part = source[begin:finish] if finish >= 0 else ""
        position = -1
        for token in tokens:
            position = part.find(token, position + 1)
            if position < 0:
                errors.append(f"{path}: missing or out-of-order penalty guard in {start}")
                return

    scoped(PENALIZED, "pub fn new(lambdas:", "    /// The default lasso tuning", (
        "Self::check_context_free_grid_len(lambdas.len())?;", "lambdas.iter().find",
        "allocate_penalty_grid(lambdas.len())?;", "pub const fn context_free_max_grid_len()",
        "CONTEXT_FREE_GRID_POLICY_BYTES / CONTEXT_FREE_GRID_BYTES_PER_PENALTY",
        "pub fn check_context_free_grid_len",
        "checked_mul(CONTEXT_FREE_GRID_BYTES_PER_PENALTY)",
        "bytes <= CONTEXT_FREE_GRID_POLICY_BYTES", 'reason_code!("execution_resource_limit")',
    ))
    scoped(PENALIZED, "const CONTEXT_FREE_GRID_POLICY_BYTES", "/// Inner cross-validation", (
        "256 * 1024 * 1024", "CONTEXT_FREE_GRID_BYTES_PER_PENALTY: u64 = 4096",
    ))
    scoped(PENALIZED, "    fn key(&self, label:", "    fn write_identity", (
        "self.lambdas.iter().enumerate()", 'write!(key, "{:016x}"',
    ))
    scoped(PENALIZED, "    fn label(&self, label:", "\n}", (
        "self.lambdas.iter().enumerate()", 'write!(result, "{value}"',
    ))
    native = RIDGE_RESOURCE_FILES[1]
    scoped(native, "fn bounded_penalty_values(", "\n}", (
        "raw.len()", "PyOverflowError", '"execution_resource_limit"',
        "check_context_free_grid_len(count)", "try_reserve_exact(count)",
        "raw.try_iter()", "values.len() == count", "values.push(",
    ))
    scoped(native, "fn penalty_declaration(", "\n}", (
        "bounded_penalty_values(&raw", "RidgeTuning::new(&lambdas",
    ))
    derived = RIDGE_RESOURCE_FILES[2]
    for method in ("factorized_joint_cells_json", "joint_cell_cost_json"):
        scoped(derived, f"fn {method}(", "\n}", (
            "penalties: Option<Bound<'_, PyAny>>", "bounded_penalty_values(&raw", "RidgeTuning::new(&grid",
        ))
    scoped(derived, "fn penalty_grid_max_items()", "\n}", (
        "RidgeTuning::context_free_max_grid_len()",
    ))
    scoped(derived, "pub(crate) fn register(", "\n}", (
        "wrap_pyfunction!(penalty_grid_max_items, module)",
    ))
    helper = RIDGE_RESOURCE_FILES[3]
    scoped(helper, "def bounded_penalty_list(", "\n    return result", (
        "_native._penalty_grid_max_items()", "len(values)", "except OverflowError",
        'reason_code="execution_resource_limit"', "declared > limit",
        "for value in values:", "len(result) >= limit", "result.append(value)",
    ))
    estimators = RIDGE_RESOURCE_FILES[4]
    scoped(estimators, "class PropensityPenalty:", "\n@dataclass", (
        "values = bounded_penalty_list(self.lambdas)",
        "for value in bounded_penalty_list(self.lambdas)",
    ))
    derived_py = RIDGE_RESOURCE_FILES[5]
    scoped(derived_py, "def factorized_joint_cells(", "\ndef estimate_joint_cell_cost(", (
        "for p in bounded_penalty_list(penalties)",
    ))
    # The final function is allowed to end at EOF.
    source = sources.get(derived_py, "")
    start = source.find("def estimate_joint_cell_cost(")
    if start < 0 or "for p in bounded_penalty_list(penalties)" not in source[start:]:
        errors.append(f"{derived_py}: missing estimate_joint_cell_cost penalty guard")
    return errors


def check(
    root: Path,
    strict: bool = False,
    mapping: tuple[Bound, ...] | None = None,
    cells: frozenset[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Return (errors, gaps, ok lines)."""
    mapping = MAPPING if mapping is None else mapping
    cells = CELLS if cells is None else cells
    rd = Reader(root)
    errors: list[str] = []
    gaps: list[str] = []
    ok: list[str] = []
    if "estimate.penalized_propensity_aipw" in cells and mapping is MAPPING:
        errors.extend(ridge_configuration_guard_errors(
            {path: rd.text(path) or "" for path in RIDGE_RESOURCE_FILES}
        ))
    rec_text = rd.text(RECORDS)
    if rec_text is None:
        return [f"{RECORDS} is missing"], gaps, ok
    records = tomllib.loads(rec_text).get("record", [])
    by_key = {(b.record, b.key): b for b in mapping}
    short_ids = set()
    records = [rec for rec in records if rec["id"] in cells]
    for rec in records:
        short = rec["id"]
        short_ids.add(short)
        bounds = rec.get("bounds", {})
        if short == "sensitivity.joint_mechanism":
            for dimension in ("parent_levels", "outcome_categories", "frontier_points"):
                historical = bounds.get(f"historical_evidence_max_{dimension}")
                runtime = bounds.get(f"runtime_max_{dimension}")
                if (
                    not _is_int(historical)
                    or not _is_int(runtime)
                    or historical < 1
                    or runtime < 1
                    or historical > runtime
                ):
                    errors.append(
                        f"{short}: historical evidence bound "
                        f"for {dimension} must be an integer no larger than the runtime ceiling"
                    )
        numeric = {k: v for k, v in bounds.items() if _is_int(v)}
        label = rec["id"]
        planned = rec.get("status") == "not_supported"
        for key in sorted(numeric):
            if (short, key) not in by_key and planned:
                gaps.append(
                    f"{label}: not_supported record's bound `{key}` = {numeric[key]} has no mapping yet "
                    "(required once the record is supported)"
                )
            elif (short, key) not in by_key:
                errors.append(
                    f"{label}: numeric bound `{key}` = {numeric[key]} has no entry in MAPPING "
                    f"(scripts/check_limits_agreement.py); add a Bound({short!r}, {key!r}, rust=...)"
                )
        for key, want in sorted(numeric.items()):
            b = by_key.get((short, key))
            if b is None:
                continue
            tag = f"{label} bound `{key}` = {want}"
            places = [("rust", p) for p in b.rust] + [("python", p) for p in b.python]
            if not b.rust and not b.python:
                errors.append(f"{tag}: mapping names no implementation bound")
            bad = False
            for kind, place in places:
                vals, err = read_place(rd, place)
                if err:
                    errors.append(f"{tag}: {kind} {err}")
                    bad = True
                    continue
                for v in vals:
                    if v != want:
                        errors.append(
                            f"{tag}: {kind} {place.path} states {v} ({place.pattern})"
                        )
                        bad = True
            if not bad:
                ok.append(f"{tag}: agrees ({len(places)} place(s))")
    declared = {
        (r["id"], k) for r in records for k, v in r.get("bounds", {}).items() if _is_int(v)
    }
    for b in mapping:
        if b.record in short_ids and (b.record, b.key) not in declared:
            errors.append(
                f"mapping {b.record}.{b.key} names an integer bound its record does not declare (stale mapping)"
            )
    if strict:
        errors.extend(f"gap: {g}" for g in gaps)
    return errors, gaps, ok


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        from check_limits import self_test as exact_record_self_test

        return self_test() or exact_record_self_test()
    strict = "--strict" in argv
    errors, gaps, ok = check(ROOT, strict)
    for error in supported_bounds_errors(ROOT):
        print(f"FAIL: {error}")
        errors.append(error)
    from check_limits import check as exact_record_check

    exact_errors, exact_checked = exact_record_check(ROOT)
    for error in exact_errors:
        print(f"FAIL: {error}")
    if exact_errors:
        errors = errors + exact_errors
    else:
        print(
            f"ok   {len(exact_checked)} exact-record bound(s) agree (scripts/check_limits.py)"
        )
    for line in ok:
        print(f"ok   {line}")
    for g in gaps:
        if not strict:
            print(f"gap  {g}")
    if errors:
        print("limits agreement FAILED:")
        for e in errors:
            print(f" - {e}")
        return 1
    print(
        f"limits agreement ok: {len(ok)} bound(s) agree, {len(gaps)} unmapped planned bound(s)"
    )
    return 0


# ------------------------------------------------------------------------------ self-test
# The self-test runs against a PINNED synthetic tree (never the live records, docs or sources), so an
# unrelated record or docs edit cannot break it. Every rule has a clean case and a mutant.
SYN_RECORDS = """
[[record]]
id = "synth.supported"
status = "supported"
bounds = { max_a = 4, max_b = 12, max_ops = 4096, depth = 24, nodes = "2 or 4, doubling check at 4 or 8", tol = "[1e-3, 1e-1]", ops_text = "4096 (cap): 32 ops in the fixture", memory_limit = "min(ctx, 64 MiB default)", contract = "SearchBudget" }

[[record]]
id = "synth.planned"
status = "not_supported"
bounds = { max_unmapped = 9, loose = "at most 7 things" }

[[record]]
id = "synth.carried"
status = "not_supported"
bounds = { max_c = 3 }
"""
SYN_RS = """
mod tipping;
pub const SYN_BASE_B: usize = 12;
pub const SYN_MAX_B: usize = SYN_BASE_B;
pub const SYN_MAX_A: usize = 4;
pub const SYN_MAX_C: usize = 3;
pub const SYN_DEFAULT: Limits = Limits { operations: 4_096, depth: 24 };
pub const SYN_NODES: [usize; 2] = [2, 4];
pub const SYN_MIN_TOL: f64 = crate::tipping::MIN_TOL;
pub const SYN_MAX_TOL: f64 = 1e-1;
"""
SYN_PY = "def f(max_ops: int = 4_096): ...\nMAX_SOURCES = 4\n"
SYN_RS_PATH = "crates/synth/src/lib.rs"
SYN_PY_PATH = "python/synth.py"
SYN_CELLS = frozenset({"synth.supported", "synth.planned", "synth.carried"})
SYN_MAPPING: tuple[Bound, ...] = (
    Bound(
        "synth.supported",
        "max_a",
        rust=(C(SYN_RS_PATH, "SYN_MAX_A"),),
        python=(L(SYN_PY_PATH, r"^MAX_SOURCES = (\d+)"),),
    ),
    Bound(
        "synth.supported",
        "max_b",
        rust=(C(SYN_RS_PATH, "SYN_MAX_B"),),  # alias of SYN_BASE_B
    ),
    Bound(
        "synth.supported",
        "max_ops",
        rust=(L(SYN_RS_PATH, r"operations: ([\d_]+),"),),
        python=(L(SYN_PY_PATH, r"max_ops: int = ([\d_]+)"),),
    ),
    Bound(
        "synth.supported",
        "depth",
        rust=(L(SYN_RS_PATH, r"depth: ([\d_]+) "),),
    ),
    Bound(
        "synth.carried",
        "max_c",
        rust=(C(SYN_RS_PATH, "SYN_MAX_C"),),
    ),
)
def _syn_tree(dest: Path) -> None:
    for rel, text in (
        (RECORDS, SYN_RECORDS),
        (SYN_RS_PATH, SYN_RS),
        ("crates/synth/src/tipping.rs", "pub const MIN_TOL: f64 = 1e-3;\n"),
        (SYN_PY_PATH, SYN_PY),
    ):
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).write_text(text, encoding="utf-8", newline="\n")


def self_test() -> int:
    failures: list[str] = []

    # The old evidence scope and the wider runtime caps are different claims.
    # Both sources and their ordering must fail closed.
    joint_id = "sensitivity.joint_mechanism"
    joint_rows = tuple(
        row for row in MAPPING
        if row.record == joint_id
        and row.key.startswith(("historical_evidence_max_", "runtime_max_"))
    )
    if len(joint_rows) != 6:
        failures.append("joint sensitivity must map all six historical and runtime limits")
    else:
        with tempfile.TemporaryDirectory() as joint_root:
            root = Path(joint_root)
            record_path = root / RECORDS
            snapshot_path = root / "parity/release_snapshots/v2.2.0.toml"
            rust_path = root / JOINT
            for path in (record_path, snapshot_path, rust_path):
                path.parent.mkdir(parents=True, exist_ok=True)
            record = '''[[record]]
id = "sensitivity.joint_mechanism"
status = "supported"
bounds = { historical_evidence_max_parent_levels = 64, historical_evidence_max_outcome_categories = 32, historical_evidence_max_frontier_points = 33, runtime_max_parent_levels = 4096, runtime_max_outcome_categories = 1024, runtime_max_frontier_points = 257 }
'''
            snapshot = '''[[record]]
id = "sensitivity.joint_mechanism"
bounds = { max_parent_levels = 64, max_outcome_categories = 32, max_frontier_points = 33 }
'''
            rust = '''pub const JOINT_SENSITIVITY_MAX_PARENT_LEVELS: usize = 4096;
pub const JOINT_SENSITIVITY_MAX_OUTCOME_CATEGORIES: usize = 1024;
pub const JOINT_SENSITIVITY_MAX_FRONTIER_POINTS: usize = 257;
'''
            record_path.write_text(record, encoding="utf-8")
            snapshot_path.write_text(snapshot, encoding="utf-8")
            rust_path.write_text(rust, encoding="utf-8")
            def joint_errors() -> list[str]:
                return check(root, mapping=joint_rows, cells=frozenset({joint_id}))[0]

            if joint_errors():
                failures.append("valid separate historical and runtime joint limits must pass")
            for dimension, historical, runtime in (
                ("parent_levels", 64, 4096),
                ("outcome_categories", 32, 1024),
                ("frontier_points", 33, 257),
            ):
                mutations = (
                    (record_path, f"historical_evidence_max_{dimension} = {historical}", f"historical_evidence_max_{dimension} = {historical + 1}", "historical evidence"),
                    (record_path, f"runtime_max_{dimension} = {runtime}", f"runtime_max_{dimension} = {runtime + 1}", "runtime record"),
                    (rust_path, f"JOINT_SENSITIVITY_MAX_{dimension.upper()}: usize = {runtime}", f"JOINT_SENSITIVITY_MAX_{dimension.upper()}: usize = {runtime + 1}", "runtime constant"),
                    (snapshot_path, f"max_{dimension} = {historical}", f"max_{dimension} = {historical + 1}", "historical snapshot"),
                )
                for path, before, after, label in mutations:
                    original = path.read_text(encoding="utf-8")
                    if before not in original:
                        failures.append(f"{label} mutation fixture for {dimension} is missing")
                        continue
                    path.write_text(original.replace(before, after), encoding="utf-8")
                    if not joint_errors():
                        failures.append(f"{label} mutation for {dimension} must fail closed")
                    path.write_text(original, encoding="utf-8")
                record_path.write_text(
                    record.replace(f"runtime_max_{dimension} = {runtime}", f"runtime_max_{dimension} = {historical - 1}"),
                    encoding="utf-8",
                )
                if not any("no larger than the runtime ceiling" in error for error in joint_errors()):
                    failures.append(f"historical {dimension} above the runtime ceiling must fail closed")
                record_path.write_text(record, encoding="utf-8")

    # The path-specific route requires exactly two worlds even though the
    # reusable query representation permits more. Detect changes to the route
    # check instead of reading the unrelated representation capacity.
    cross_world = next(
        row
        for row in MAPPING
        if row.record == "counterfactual.path_specific_edge" and row.key == "max_worlds"
    )
    route_pattern = cross_world.rust[0].pattern
    for source, expected in (
        ("if worlds.len() != 2 || other_condition {", "2"),
        ("if worlds.len() != 3 || other_condition {", "3"),
    ):
        matched = re.search(route_pattern, source)
        if matched is None or matched.group(1) != expected:
            failures.append(
                "cross-world route scope must track the actual world-count check"
            )

    # A supported scale kind and a requested transform are different quantities:
    # repeating one kind does not create another enum/parser case.
    with tempfile.TemporaryDirectory() as scale_root:
        scale_dir = Path(scale_root)
        scale_path = scale_dir / "scales.rs"
        scale_source = (
            "pub enum ReportingScale {\n    Mean,\n    Ratio,\n}\n"
            "impl ReportingScale {\n"
            "pub fn parse(name: &str) {\n"
            "[Self::Mean, Self::Ratio].into_iter();\n}\n}\n"
        )
        scale_place = Place("scales.rs", "reporting_scale_kinds", "ReportingScale")
        scale_path.write_text(scale_source, encoding="utf-8")
        values, error = read_place(Reader(scale_dir), scale_place)
        if error or values != [2]:
            failures.append("reporting-scale kind count must match the enum and parser")
        scale_path.write_text(
            scale_source.replace("    Ratio,\n", "    Risk,\n"), encoding="utf-8"
        )
        _, error = read_place(Reader(scale_dir), scale_place)
        if error is None:
            failures.append("reporting-scale enum/parser drift must be rejected")
        scale_path.write_text(
            scale_source.replace("    Ratio,\n", "    Ratio(u8),\n"), encoding="utf-8"
        )
        _, error = read_place(Reader(scale_dir), scale_place)
        if error is None:
            failures.append(
                "an unrecognized reporting-scale enum shape must be rejected"
            )
        scale_path.write_text(
            scale_source.replace("    Ratio,\n", "    Ratio,\n    Risk,\n").replace(
                "Self::Ratio]", "Self::Ratio, Self::Risk]"
            ),
            encoding="utf-8",
        )
        values, error = read_place(Reader(scale_dir), scale_place)
        if error or values != [3]:
            failures.append("a new reporting-scale kind must change the counted bound")

        fold_path = scale_dir / "fold.rs"
        fold_place = Place("fold.rs", "fold_plan_capacity", "assign_folds")
        fold_source = "pub fn assign_folds() {\n    if folds > usize::from(u16::MAX) + 1 {}\n    if n_rows < folds {}\n}\n"
        fold_path.write_text(fold_source, encoding="utf-8")
        values, error = read_place(Reader(scale_dir), fold_place)
        if error or values != [65536]:
            failures.append(
                "fold-plan index capacity must be derived from the runtime guard"
            )
        fold_path.write_text(
            fold_source.replace("u16::MAX", "u8::MAX"), encoding="utf-8"
        )
        values, error = read_place(Reader(scale_dir), fold_place)
        if error or values != [256]:
            failures.append("fold-plan index mutations must change the capacity")
        fold_path.write_text(
            fold_source.replace("if n_rows < folds", "if n_rows == folds"),
            encoding="utf-8",
        )
        _, error = read_place(Reader(scale_dir), fold_place)
        if error is None:
            failures.append("missing fold-plan row guard must be rejected")

        snapshot_path = scale_dir / "snapshot.toml"
        snapshot_place = Place("snapshot.toml", "snapshot_bound", "example:max_folds")
        snapshot = '[[record]]\nid = "example"\nbounds = { max_folds = 20 }\n'
        snapshot_path.write_text(snapshot, encoding="utf-8")
        values, error = read_place(Reader(scale_dir), snapshot_place)
        if error or values != [20]:
            failures.append(
                "historical scope must be read from the named snapshot bound"
            )
        snapshot_path.write_text(snapshot + snapshot, encoding="utf-8")
        _, error = read_place(Reader(scale_dir), snapshot_place)
        if error is None:
            failures.append("ambiguous historical snapshot records must be refused")
        snapshot_path.write_text(
            snapshot.replace("max_folds", "other"), encoding="utf-8"
        )
        _, error = read_place(Reader(scale_dir), snapshot_place)
        if error is None:
            failures.append("missing historical snapshot bounds must be refused")

    def run(d: Path, **kw: object) -> tuple[list[str], list[str], list[str]]:
        return check(
            d,
            mapping=kw.pop("mapping", SYN_MAPPING),
            cells=SYN_CELLS,
            **kw,  # type: ignore[arg-type]
        )

    with tempfile.TemporaryDirectory() as t:
        base = Path(t) / "base"
        _syn_tree(base)
        errors, gaps, ok = run(base)
        if errors:
            failures.append("clean fixture must pass: " + "; ".join(errors[:3]))
        if len(ok) < 5:
            failures.append(f"clean fixture checked only {len(ok)} bounds (vacuous)")
        if not any("max_unmapped" in g for g in gaps):
            failures.append("a not_supported record's unmapped bound must be a gap")
        if any("max_unmapped" in e for e in errors):
            failures.append("a not_supported record's unmapped bound must not be an error")
        if any("loose" in g for g in gaps + errors) or any("nodes" in e for e in errors):
            failures.append("a string-valued bound is not read, so it is neither a gap nor an error")
        if not any("max_c" in line for line in ok):
            failures.append("a second not_supported record's bound must be checked")
        if not run(base, strict=True)[0]:
            failures.append("--strict must turn the not_supported-record gaps into failures")

        def mutated(
            name: str,
            rel: str,
            old: str,
            new: str,
            expect: str,
            **kw: object,
        ) -> None:
            d = Path(t) / name
            shutil.copytree(base, d)
            p = d / rel
            text = p.read_text(encoding="utf-8")
            if old not in text:
                failures.append(f"{name}: mutation anchor {old!r} not found in {rel}")
                return
            p.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")
            errs, _, _ = run(d, **kw)
            if not any(expect in e for e in errs):
                failures.append(
                    f"{name}: expected an error containing {expect!r}, got {errs[:3]}"
                )

        mutated(
            "rust_const",
            SYN_RS_PATH,
            "SYN_MAX_A: usize = 4",
            "SYN_MAX_A: usize = 5",
            "max_a",
        )
        mutated(
            "rust_const_missing",
            SYN_RS_PATH,
            "pub const SYN_MAX_C: usize = 3;",
            "",
            "SYN_MAX_C",
        )
        mutated(
            "alias_followed",
            SYN_RS_PATH,
            "SYN_BASE_B: usize = 12",
            "SYN_BASE_B: usize = 13",
            "max_b",
        )
        mutated(
            "rust_literal",
            SYN_RS_PATH,
            "operations: 4_096",
            "operations: 4_097",
            "max_ops",
        )
        mutated(
            "python_default",
            SYN_PY_PATH,
            "max_ops: int = 4_096",
            "max_ops: int = 4_097",
            "max_ops",
        )
        mutated("record_value", RECORDS, "max_b = 12", "max_b = 13", "max_b")
        mutated(
            "unmapped_bound",
            RECORDS,
            "max_a = 4,",
            "max_a = 4, max_new = 3,",
            "max_new",
        )
        # stale mapping rows
        mutated(
            "stale_mapping",
            RECORDS,
            "max_c = 3 }",
            "max_q = 3 }",
            "stale mapping",
            mapping=SYN_MAPPING
            + (Bound("synth.carried", "max_c", rust=(C(SYN_RS_PATH, "SYN_MAX_C"),)),),
        )
        # A same-named constant in another file does not satisfy a mapping to this file.
        d = Path(t) / "const_elsewhere"
        shutil.copytree(base, d)
        rs = d / SYN_RS_PATH
        rs.write_text(
            rs.read_text(encoding="utf-8").replace(
                "pub const SYN_MAX_C: usize = 3;\n", "", 1
            ),
            encoding="utf-8",
            newline="\n",
        )
        (d / "crates/synth/src/other.rs").write_text(
            "pub const SYN_MAX_C: usize = 3;\n", encoding="utf-8", newline="\n"
        )
        errs, _, _ = run(d)
        if not any("SYN_MAX_C not found" in e for e in errs):
            failures.append(
                f"const_elsewhere: a constant moved to another file must not satisfy the mapping, got {errs[:3]}"
            )
        d = Path(t) / "stale_row"
        shutil.copytree(base, d)
        errs, _, _ = run(
            d,
            mapping=SYN_MAPPING
            + (
                Bound("synth.supported", "gone", rust=(C(SYN_RS_PATH, "SYN_MAX_A"),)),
            ),
        )
        if not any("stale mapping" in e for e in errs):
            failures.append(
                f"stale_row: a row for an undeclared bound must be stale, got {errs[:3]}"
            )
        table_root = Path(t) / "table"
        (table_root / "parity").mkdir(parents=True)
        (table_root / "docs").mkdir()
        (table_root / RECORDS).write_text(
            '[[record]]\nid = "synth.exact"\nbounds = { max_items = 4 }\n',
            encoding="utf-8",
        )
        (table_root / LIMITS_REGISTRY).write_text(
            "[[bound]]\n"
            'record = "synth.exact"\n'
            'key = "max_items"\n'
            'implementation = [{ path = "source.rs", constant = "MAX_ITEMS" }]\n'
            'documentation = [{ path = "source.rs", constant = "MAX_ITEMS" }]\n',
            encoding="utf-8",
        )
        rendered = render_supported_bounds_table(table_root)
        page = (
            "# Bounds\n\nHand prose may say at most 99 without failing.\n\n"
            f"{BOUNDS_BEGIN}\n{rendered}{BOUNDS_END}\n"
        )
        (table_root / SUPPORTED_BOUNDS).write_text(page, encoding="utf-8")
        if supported_bounds_errors(table_root):
            failures.append("generated bounds table must match the render")
        (table_root / SUPPORTED_BOUNDS).write_text(
            page.replace("at most 99", "at most twenty"), encoding="utf-8"
        )
        if supported_bounds_errors(table_root):
            failures.append("rewording prose outside the bounds marker must not fail")
        (table_root / SUPPORTED_BOUNDS).write_text(
            page.replace("| 4 |", "| 5 |"), encoding="utf-8"
        )
        if not supported_bounds_errors(table_root):
            failures.append("a hand-edited generated bounds value must fail")
    if failures:
        print("check_limits_agreement self-test FAILED:")
        for f in failures:
            print(f" - {f}")
        return 1
    print("check_limits_agreement self-test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
