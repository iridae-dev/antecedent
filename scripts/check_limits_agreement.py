#!/usr/bin/env python3
"""Declared bounds agree everywhere they are stated.

For every regex-mapped record of parity/promotion.toml, each numeric entry of its `bounds` table must
equal the value stated by each place mapped to it in `MAPPING` below:

  rust      a Rust constant (`const NAME: ty = <int | OTHER_CONST>;`) or an integer literal picked
            out of a source file by a regex, in the cell's own source files;
  guides    a `{{bound:record.key}}` placeholder under `docs/guides/` (the surrounding prose is
            not read; rewording it cannot fail this check);
  python    a constant or docstring/default of the cell's Python module (also the implementation
            bound for Python-only artifact replay);
  refusal   the numbers quoted by the record's `*.bounds_exceeded` refusal `when` text.

The numeric source is `parity/promotion.toml` `[bounds]`. `docs/supported-bounds.md` carries a
generated table of the exact-record bounds (marker `generated:supported-bounds`); the check
compares that marker to a fresh render and does not scrape the page. `parity/limits.toml`
names implementation, guard, refusal and fixture places. Its `documentation` place is the
same code place as `implementation`, because `scripts/check_limits.py` still requires one
and must not read the generated page.

The check fails, naming the record, the bound and the place, when a mapped constant, regex or
statement is missing or differs, when a documented bound has no `{{bound:record.key}}`
placeholder under `docs/guides/`, and when a record carries a numeric bound that has no mapping
entry at all (so a new record cannot ship a bound nobody compares). A bound with
`documented=False` is reported as a `gap` (a documentation statement owed, not a failure);
`--strict` makes gaps failures.

Extending it for a new record: add `Bound(...)` rows for the record's id (`"sensitivity.joint_mechanism"`, ...) to
`MAPPING` and list the id in `CELLS`. `rust=` and `python=` take `C(path, NAME)` (a constant) or `L(path, regex)` (first
capture group is the number; every match in the file must agree). Put `{{bound:record.key}}` in
the guide that states the bound. `refusal=r"regex"` anchors the bound in the record's
`*.bounds_exceeded` refusal text: the regex's first capture group is the number, every match
must equal the bound and at least one must match (a bare `True` is refused: a number found
anywhere in the text proves nothing).

A non-integer bound that carries digits (`"16 or 32, doubling check at 32 or 64"`, `"[1e-12, 1e-2]"`,
`"... (hard cap 100000)"`) must have a `NONINT` row: either the set of numbers it states equals the
set read from its places (`doubled=True` also admits twice each), or it is excluded with a recorded
reason. An unmapped one is an error (a gap on a `frozen` record). Works for any record status,
`carried_forward` included.

    python3 scripts/check_limits_agreement.py             # check the tree
    python3 scripts/check_limits_agreement.py --strict    # documentation gaps also fail
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
RECORDS = "parity/promotion.toml"
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

WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twelve": 12,
    "sixteen": 16,
    "twenty": 20,
}


@dataclass(frozen=True)
class Place:
    """A file and how to read one integer from it."""

    path: str
    kind: str  # "const" or "regex"
    pattern: str


def C(path: str, name: str) -> Place:
    return Place(path, "const", name)


def L(path: str, regex: str) -> Place:
    return Place(path, "regex", regex)


@dataclass(frozen=True)
class Bound:
    record: str  # full record id in parity/promotion.toml
    key: str  # key of the record's `bounds` table
    rust: tuple[Place, ...] = ()
    python: tuple[Place, ...] = ()
    refusal: str = ""
    documented: bool = True  # require {{bound:record.key}} in docs/guides
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
XW_CORE = "crates/antecedent-core/src/query/cross_world.rs"
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
CLUSTER_DML = "crates/antecedent-estimate/src/cluster_dml_aipw.rs"
DESCRIPTIVE = "crates/antecedent-estimate/src/descriptive_comparison.rs"
JOINT_CELLS = "crates/antecedent-estimate/src/joint_cell_factorized.rs"
BATCH_RETARGET = "crates/antecedent/src/analysis/batch_retarget.rs"
JOINT_IF = "crates/antecedent-estimate/src/joint_if.rs"


# The mapping table: one row per (record, bounds key). Extend with additive rows for new records.
MAPPING: tuple[Bound, ...] = (
    # ---- estimator and decision cells
    Bound("estimate.matched_case_control_odds_ratio", "max_iterations", rust=(C(MATCHED, "MAX_ITERATIONS"),)),
    Bound("estimate.matched_case_control_odds_ratio", "replay_max_rows", python=(L(MATCHED_PY, r"^_MAX_REPLAY_ROWS = ([\d_]+)"),)),
    Bound("estimate.penalized_propensity_aipw", "max_grid", rust=(C(PENALIZED, "MAX_RIDGE_GRID"),)),
    Bound("estimate.penalized_propensity_aipw", "max_inner_folds", rust=(C(PENALIZED, "MAX_RIDGE_INNER_FOLDS"),)),
    Bound("estimate.clustered_dml_aipw", "default_min_clusters", rust=(C(CLUSTER_DML, "DEFAULT_MIN_CLUSTERS"),)),
    Bound("estimate.clustered_dml_aipw", "min_clusters_floor", rust=(C(CLUSTER_DML, "MIN_CLUSTERS_FLOOR"),)),
    Bound("estimate.clustered_dml_aipw", "default_min_components_per_fold", rust=(C(CLUSTER_DML, "DEFAULT_MIN_COMPONENTS_PER_FOLD"),)),
    Bound("estimate.clustered_dml_aipw", "min_components_per_fold_floor", rust=(C(CLUSTER_DML, "MIN_COMPONENTS_PER_FOLD_FLOOR"),)),
    Bound("estimate.descriptive_comparison", "max_scales", rust=(C(DESCRIPTIVE, "MAX_REPORTING_SCALES"),)),
    Bound("estimate.factorized_joint_cells", "max_components", rust=(C(JOINT_CELLS, "MAX_FACTORIZED_COMPONENTS"),)),
    Bound("estimate.factorized_joint_cells", "max_folds", rust=(C(JOINT_CELLS, "MAX_FOLDS"),)),
    Bound("estimate.factorized_joint_cells", "max_orderings", rust=(C(JOINT_CELLS, "MAX_ORDERINGS"),)),
    # ---- mz-transportability
    Bound(
        "transport.multi_source_limited_experiment",
        "max_sources",
        rust=(C(MZ, "MZ_TRANSPORT_MAX_SOURCES"),),

        python=(L(MZ_PY, r"two to (\w+) sources, \w+ controllable variables per"),),
        refusal=r"Sources outside 2-(\d+)",
    ),
    Bound(
        "transport.multi_source_limited_experiment",
        "max_controllable_per_source",
        rust=(C(MZ, "MZ_TRANSPORT_MAX_CONTROLLABLE_PER_SOURCE"),),

        python=(L(MZ_PY, r"two to \w+ sources, (\w+) controllable variables per"),),
        refusal=r"controllables per source outside 1-(\d+)",
    ),
    Bound(
        "transport.multi_source_limited_experiment",
        "max_candidate_regimes",
        rust=(C(MZ, "MZ_TRANSPORT_MAX_CANDIDATE_REGIMES"),),

        python=(L(MZ_PY, r"source and (\d+) candidate regimes"),),
        refusal=r"more than (\d+) candidate regimes",
    ),
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
        refusal=r"above (\d+) operations / depth",
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
        refusal=r"operations / depth (\d+)",
    ),
    # ---- finite scenario envelope
    Bound(
        "transport.finite_scenario_envelope",
        "max_scenarios",
        rust=(C(SCEN, "SCENARIO_MAX_COUNT"),),

        python=(
            L(SCEN_PY, r"^MAX_SCENARIOS = (\d+)"),
            L(SCEN_PY, r"One to (\d+) scenarios"),
        ),
    ),
    # ---- learned continuous trial transport
    Bound(
        "transport.learned_continuous_trial",
        "max_folds",
        rust=(C(LEARN, "LEARNED_CONTINUOUS_MAX_FOLDS"),),

        refusal=r"More than (\d+) folds",
        note="no Python statement of the fold cap",
    ),
    Bound(
        "transport.learned_continuous_trial",
        "max_bootstrap",
        rust=(C(LEARN, "LEARNED_CONTINUOUS_MAX_BOOTSTRAP"),),

        python=(L(LEARN_PY, r"\(at most (\d+), floor \d+\)"),),
        refusal=r"more than (\d+) bootstrap replicates",
    ),
    Bound(
        "transport.learned_continuous_trial",
        "min_bootstrap",
        rust=(C(LEARN, "LEARNED_CONTINUOUS_MIN_BOOTSTRAP"),),

        python=(L(LEARN_PY, r"\(at most \d+, floor (\d+)\)"),),
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
        "max_actions",
        rust=(C(TEMP, "TEMPORAL_MAX_ACTIONS"),),

        python=(L(TEMP_PY, r"alphabet of at most (\w+) actions"),),
        refusal=r"More than (\d+) actions",
    ),
    Bound(
        "transport.two_step_temporal",
        "max_history_states",
        rust=(C(TEMP, "TEMPORAL_MAX_HISTORY_STATES"),),

        python=(
            L(TEMP_PY, r"at most ([\d,]+)\s+complete covariate histories"),
        ),
        refusal=r"more than (\d+) complete covariate histories",
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
        rust=(C(XW_CORE, "MAX_CROSS_WORLDS"),),

        note="no Python statement of the world cap",
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

        refusal=r"More than (\d+) observed variables",
        note="no Python statement of the observed-variable cap",
    ),
    Bound(
        "identify.mixed_source_proof_search",
        "max_distributions",
        rust=(C(MIX, "MIXED_SOURCE_MAX_DISTRIBUTIONS"),),

        refusal=r", (\d+) usable distributions",
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

        refusal=r"above (\d+) operations / depth",
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

        refusal=r"operations / depth (\d+)",
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

        python=(L(MIX_PY, r"declares at most (\w+) sources"),),
        refusal=r"(\d+) declared sources",
    ),
    # ---- ADMG conditional transport
    Bound(
        "transport.admg_conditional",
        "max_observed",
        rust=(C(COND, "ADMG_CONDITIONAL_MAX_OBSERVED"),),

        python=(L(COND_PY, r"more than (\d+) observed variables, \d+\s+treatments"),),
        refusal=r"More than (\d+) observed variables",
    ),
    Bound(
        "transport.admg_conditional",
        "max_treatments",
        rust=(C(COND, "ADMG_CONDITIONAL_MAX_TREATMENTS"),),

        python=(
            L(COND_PY, r"observed variables, (\d+)\s+treatments or \d+ conditioned"),
        ),
        refusal=r"observed variables, (\d+) treatments",
    ),
    Bound(
        "transport.admg_conditional",
        "max_conditioned",
        rust=(C(COND, "ADMG_CONDITIONAL_MAX_CONDITIONED"),),

        python=(L(COND_PY, r"treatments or (\d+) conditioned variables"),),
        refusal=r"treatments or (\d+) conditioned",
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
            L(COND_PY, r"Limits above ([\d,]+) operations / depth \d+,"),
            L(
                COND_PY,
                r"max_operations: int = ([\d_]+),\n\s+max_depth: int = \d+,\n\s+max_support_rows: int = ",
            ),
            L(COND_PY, r"max_search_operations: int = ([\d_]+)"),
        ),
        refusal=r"above (\d+) operations / depth",
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
            L(COND_PY, r"Limits above [\d,]+ operations / depth (\d+),"),
            L(
                COND_PY,
                r"max_operations: int = [\d_]+,\n\s+max_depth: int = ([\d_]+),\n\s+max_support_rows: int = ",
            ),
            L(COND_PY, r"max_search_depth: int = ([\d_]+)"),
        ),
        refusal=r"operations / depth (\d+)",
    ),
    # ---- study planning
    Bound(
        "design.study_planning",
        "max_candidates",
        rust=(C(PLAN, "STUDY_PLAN_MAX_CANDIDATES"),),

        python=(L(PLAN_PY, r"universe of at most (\w+) candidate studies"),),
        refusal=r"More than (\d+) candidates",
    ),
    Bound(
        "design.study_planning",
        "max_regimes_per_candidate",
        rust=(C(PLAN, "STUDY_PLAN_MAX_REGIMES_PER_CANDIDATE"),),

        refusal=r"more than (\d+) regimes",
        note="no Python statement of the per-candidate regime cap",
    ),
    Bound(
        "design.study_planning",
        "max_subset_size",
        rust=(C(PLAN, "STUDY_PLAN_MAX_SUBSET"),),

        python=(L(PLAN_PY, r"evaluates every subset of at most (\w+) studies"),),
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
        refusal=r"limits above (\d+) operations",
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
        refusal=r"\((\d+) Mz,",
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
        refusal=r", (\d+) Mixed\)",
    ),
    # ---- binary observation recovery
    Bound(
        "recovery.binary_observation",
        "max_partially_observed",
        rust=(C(REC, "RECOVERY_MAX_PARTIALLY_OBSERVED"),),

        refusal=r"More than (\d+) partially observed",
        note="no Python statement of the partially-observed cap",
    ),
    Bound(
        "recovery.binary_observation",
        "max_fully_observed",
        rust=(C(REC, "RECOVERY_MAX_FULLY_OBSERVED"),),

        refusal=r"or (\d+) fully observed",
        note="no Python statement of the fully-observed cap",
    ),
    Bound(
        "recovery.binary_observation",
        "max_observed_cells",
        rust=(C(REC, "RECOVERY_MAX_OBSERVED_CELLS"),),

        refusal=r"more than (\d+) observed-law cells",
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
        refusal=r"above (\d+) operations / depth",
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
        refusal=r"operations / depth (\d+)",
    ),
    # ---- smoothed dose-response transport
    Bound(
        "transport.smoothed_dose_response",
        "max_grid_points",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_GRID"),),

        python=(L(DOSE_PY, r"``grid`` holds at most (\d+) distinct doses"),),
        refusal=r"More than (\d+) grid doses",
    ),
    Bound(
        "transport.smoothed_dose_response",
        "max_folds",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_FOLDS"),),

        refusal=r"more than (\d+) folds",
        note="no Python statement of the fold cap",
    ),
    Bound(
        "transport.smoothed_dose_response",
        "max_rows",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_ROWS"),),

        refusal=r"more than (\d+) rows",
        note="no Python statement of the row cap",
    ),
    Bound(
        "transport.smoothed_dose_response",
        "max_features",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_FEATURES"),),

        refusal=r"more than (\d+) covariates",
        note="no Python statement of the covariate cap",
    ),
    Bound(
        "transport.smoothed_dose_response",
        "max_basis_degree",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_BASIS_DEGREE"),),

        refusal=r"outside 1\.\.=(\d+)",
        note="no Python statement of the basis degree cap",
    ),
    Bound(
        "transport.smoothed_dose_response",
        "max_knots",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_KNOTS"),),

        refusal=r"more than (\d+) knots",
        note="no Python statement of the knot cap",
    ),
    Bound(
        "transport.smoothed_dose_response",
        "min_bootstrap",
        rust=(C(DOSE, "SMOOTHED_DOSE_MIN_BOOTSTRAP"),),

        python=(L(DOSE_PY, r"\(at most \d+, floor (\d+)\)"),),
    ),
    Bound(
        "transport.smoothed_dose_response",
        "max_bootstrap",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_BOOTSTRAP"),),

        python=(L(DOSE_PY, r"\(at most (\d+), floor \d+\)"),),
        refusal=r"more than (\d+) bootstrap replicates",
    ),
    Bound(
        "transport.smoothed_dose_response",
        "max_workspace_bytes",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_WORKSPACE_BYTES"),),

        python=(L(DOSE_PY, r"mandatory 512 MiB \(([\d,]+) bytes\) cap"),),
    ),
    # ---- joint mechanism sensitivity
    Bound(
        "sensitivity.joint_mechanism",
        "max_factors",
        rust=(C(JOINT, "JOINT_SENSITIVITY_MAX_FACTORS"),),

    ),
    Bound(
        "sensitivity.joint_mechanism",
        "max_parent_levels",
        rust=(C(JOINT, "JOINT_SENSITIVITY_MAX_PARENT_LEVELS"),),

        refusal=r"More than (\d+) parent levels",
    ),
    Bound(
        "sensitivity.joint_mechanism",
        "max_outcome_categories",
        rust=(C(JOINT, "JOINT_SENSITIVITY_MAX_OUTCOME_CATEGORIES"),),

        refusal=r"or (\d+) outcome categories",
    ),
    Bound(
        "sensitivity.joint_mechanism",
        "max_frontier_points",
        rust=(C(JOINT, "JOINT_SENSITIVITY_MAX_FRONTIER_POINTS"),),

        refusal=r"grid outside 1-(\d+)",
    ),
    Bound(
        "sensitivity.joint_mechanism",
        "max_operations",
        rust=(C(JOINT, "JOINT_SENSITIVITY_MAX_OPERATIONS"),),

        python=(
            L(JOINT_PY, r"^    max_operations: int = ([\d_]+)$"),
            L(JOINT_PY, r"max_search_operations: int = ([\d_]+)"),
        ),
        refusal=r"above (\d+) operations / depth",
    ),
    Bound(
        "sensitivity.joint_mechanism",
        "max_bisection_iterations",
        rust=(C(JOINT, "JOINT_SENSITIVITY_MAX_DEPTH"),),

        python=(
            L(JOINT_PY, r"^    max_depth: int = ([\d_]+)$"),
            L(JOINT_PY, r"max_search_depth: int = ([\d_]+)"),
        ),
        refusal=r"operations / depth (\d+)",
    ),
    Bound(
        "sensitivity.joint_uncertainty",
        "max_bootstrap_internal",
        rust=(C(JOINT, "JOINT_SENSITIVITY_MAX_BOOTSTRAP"),),

        refusal=r"more than (\d+) bootstrap replicates",
    ),
    # ---- ADMG counterfactual identification
    Bound(
        "counterfactual.admg_identification",
        "max_variables",
        rust=(C(CFID, "COUNTERFACTUAL_ID_MAX_VARIABLES"),),

        python=(L(CFID_PY, r"at most (\w+) finite-discrete\s+variables"),),
        refusal=r"More than (\d+) variables",
    ),
    Bound(
        "counterfactual.admg_identification",
        "max_levels",
        rust=(C(CFID, "COUNTERFACTUAL_ID_MAX_LEVELS"),),

        python=(L(CFID_PY, r"at most (\w+) levels each"),),
        refusal=r"more than (\d+) levels",
    ),
    Bound(
        "counterfactual.admg_identification",
        "max_search_operations",
        rust=(C(CFID, "COUNTERFACTUAL_ID_MAX_OPERATIONS"),),

        python=(L(CFID_PY, r"the one search budget \(at most (\d+) and \d+\)"),),
        refusal=r"above (\d+) operations or depth",
    ),
    Bound(
        "counterfactual.admg_identification",
        "max_search_depth",
        rust=(C(CFID, "COUNTERFACTUAL_ID_MAX_DEPTH"),),

        python=(L(CFID_PY, r"the one search budget \(at most \d+ and (\d+)\)"),),
        refusal=r"operations or depth (\d+)",
    ),
)


@dataclass(frozen=True)
class NonInt:
    """A non-integer bound that states numbers: compared to its places, or excluded with a reason."""

    record: str
    key: str
    places: tuple[Place, ...] = ()
    doubled: bool = (
        False  # the record also states twice each place value (a doubling check)
    )
    exclude: str = ""  # recorded reason the numbers are not compared
    head: str = (
        ""  # regex (one group): compare only the numbers of that part of the text
    )


_MEMORY_REASON = (
    "memory default is an expression (N * 1024 * 1024, or min with the execution context) that no "
    "docs or Python page restates; the cell's memory-cap tests pin the behaviour"
)

NONINT: tuple[NonInt, ...] = (
    NonInt("estimate.descriptive_comparison", "cancellation_check", places=(C(DESCRIPTIVE, "POLL_EVERY_ROWS"),)),
    NonInt("estimate.tier_diagnostics", "cancellation_check", places=(C(DESCRIPTIVE, "POLL_EVERY_ROWS"),)),
    NonInt("estimate.descriptive_comparison", "work", exclude="2 x 2 names the dimension of the fixed delta-method covariance, not a tunable execution bound"),
    NonInt("estimate.tier_diagnostics", "work", exclude="2 x 2 names the dimension of the fixed delta-method covariance, not a tunable execution bound"),
    NonInt("estimate.batch_retarget_covariance", "max_t_poll", places=(C(JOINT_IF, "MAX_T_POLL_EVERY"),)),
    NonInt("transport.multi_source_limited_experiment", "memory_limit", exclude=_MEMORY_REASON),
    NonInt("transport.admg_conditional", "memory_limit", exclude=_MEMORY_REASON),
    NonInt("design.study_planning", "memory_limit", exclude=_MEMORY_REASON),
    NonInt("recovery.binary_observation", "memory_limit", exclude=_MEMORY_REASON),
    NonInt("counterfactual.admg_identification", "memory_limit", exclude=_MEMORY_REASON),
    NonInt("sensitivity.joint_mechanism", "memory_limit", exclude=_MEMORY_REASON),
    NonInt(
        "design.study_planning",
        "per_decision_operations",
        head=r"^([^:]*):",  # '4096 (Mz) / 20000 (Mixed): ...prose...'
        places=(
            L(
                MZ,
                r"MZ_TRANSPORT_DEFAULT_LIMITS: SearchLimits = SearchLimits \{ operations: ([\d_]+),",
            ),
            L(
                MIX,
                r"MIXED_SOURCE_DEFAULT_LIMITS: SearchLimits =\s*SearchLimits \{ operations: ([\d_]+),",
            ),
        ),
    ),
    NonInt(
        "transport.smoothed_dose_response",
        "quadrature_nodes",
        places=(
            L(DOSE, r"SMOOTHED_DOSE_QUADRATURE_NODES: \[usize; 2\] = \[(\d+), \d+\]"),
            L(DOSE, r"SMOOTHED_DOSE_QUADRATURE_NODES: \[usize; 2\] = \[\d+, (\d+)\]"),
        ),
        doubled=True,
    ),
    NonInt(
        "sensitivity.joint_mechanism",
        "tolerance",
        places=(
            C(JOINT, "JOINT_SENSITIVITY_MIN_TOLERANCE"),
            C(JOINT, "JOINT_SENSITIVITY_MAX_TOLERANCE"),
        ),
    ),
    NonInt(
        "sensitivity.joint_mechanism",
        "operation_limit",
        places=(C(JOINT, "JOINT_SENSITIVITY_MAX_OPERATIONS"),),
    ),
    NonInt("sensitivity.joint_mechanism", "depth_limit", places=(C(JOINT, "JOINT_SENSITIVITY_MAX_DEPTH"),)),
)


def to_int(text: str) -> int | None:
    t = text.strip().lower().replace(",", "").replace("_", "")
    if t.isdigit():
        return int(t)
    return WORDS.get(t)


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
            qualified = re.fullmatch(r"crate::([a-z_][a-z0-9_]*)::([A-Z][A-Z0-9_]*)", raw)
            if qualified and depth < 4:
                # Resolve the declared module, never an unrelated same-named
                # constant elsewhere in the workspace.
                source = self.root / rel
                src = next((parent for parent in source.parents if parent.name == "src"), None)
                if src:
                    for target in (src / f"{qualified[1]}.rs", src / qualified[1] / "mod.rs"):
                        if target.is_file():
                            return self.find_const(qualified[2], str(target.relative_to(self.root)), depth + 1, True)
                return None, rel
            if depth < 4 and re.fullmatch(r"[A-Z][A-Z0-9_]+", raw):
                return self.find_const(raw, rel, depth + 1)
            return None, rel
        return None, near


def read_place(
    rd: Reader, place: Place, num: bool = False
) -> tuple[list[int | float], str | None]:
    """All integers the place states (must be non-empty and, at the caller, all equal)."""
    text = rd.text(place.path)
    if text is None:
        return [], f"file {place.path} is missing"
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
        v = to_num(m.group(1)) if num else to_int(m.group(1))
        if v is None:
            return (
                [],
                f"{place.path}: {m.group(1)!r} matched by /{place.pattern}/ is not a number",
            )
        out.append(v)
    if not out:
        return [], f"{place.path}: no statement matching /{place.pattern}/"
    return out, None


NUM_TOKEN = re.compile(r"\d+(?:\.\d+)?(?:e[-+]?\d+)?")


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _refusal_problem(rx: str, text: str, want: int) -> str | None:
    """None when every match of the anchored regex quotes `want` (and one exists)."""
    if not text:
        return "record has no *.bounds_exceeded refusal to compare"
    found = [m.group(1) for m in re.finditer(rx, text)]
    if not found:
        return f"the record's bounds_exceeded refusal text has no statement matching /{rx}/"
    bad = [f for f in found if to_int(f) != want]
    if bad:
        return f"the record's bounds_exceeded refusal text states {bad[0]} (/{rx}/), not {want}"
    return None


def _check_nonint(
    rd: Reader, label: str, key: str, val: object, row: NonInt, errors: list[str]
) -> str | None:
    """Compare the numbers a non-integer bound states to its places; an ok line or None."""
    tag = f"{label} bound `{key}` = {val!r}"
    if row.exclude:
        return f"{tag}: numbers not compared ({row.exclude})"
    if not row.places:
        errors.append(f"{tag}: NONINT row has neither places nor an exclusion reason")
        return None
    text = str(val).lower()
    if row.head:
        m = re.search(row.head, text)
        if not m:
            errors.append(f"{tag}: no statement matching /{row.head}/")
            return None
        text = m.group(1)
    have = {to_num(t) for t in NUM_TOKEN.findall(text)}
    want: set[int | float] = set()
    for place in row.places:
        vals, err = read_place(rd, place, num=True)
        if err:
            errors.append(f"{tag}: {err}")
            return None
        want.update(vals)
    if row.doubled:
        want |= {2 * v for v in want}
    if have != want:
        errors.append(
            f"{tag}: states the numbers {sorted(have)} but the code states {sorted(want)}"
        )
        return None
    return f"{tag}: agrees ({len(row.places)} place(s))"


BOUND_PLACEHOLDER = re.compile(r"\{\{bound:([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+)\}\}")
SUPPORTED_BOUNDS = "docs/supported-bounds.md"
BOUNDS_BEGIN = "<!-- generated:supported-bounds:begin -->"
BOUNDS_END = "<!-- generated:supported-bounds:end -->"
LIMITS_REGISTRY = "parity/limits.toml"


def split_bound_token(token: str) -> tuple[str, str]:
    record, key = token.rsplit(".", 1)
    return record, key


def guide_placeholders(root: Path) -> set[tuple[str, str]]:
    """Bound identities named by `{{bound:record.key}}` under docs/guides."""
    found: set[tuple[str, str]] = set()
    guides = root / "docs" / "guides"
    if not guides.is_dir():
        return found
    for path in guides.rglob("*.md"):
        for match in BOUND_PLACEHOLDER.finditer(path.read_text(encoding="utf-8")):
            found.add(split_bound_token(match.group(1)))
    return found


def render_supported_bounds_table(root: Path) -> str:
    """Table of exact-record bounds. Values come from promotion.toml; paths from limits.toml."""
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
        lines.append(f"| {row['record']} | {row['key']} | {values[identity]} | `{path}` |")
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
        return [
            f"{SUPPORTED_BOUNDS} generated table does not match {RECORDS} [bounds]"
        ]
    return []


def check(
    root: Path,
    strict: bool = False,
    mapping: tuple[Bound, ...] | None = None,
    nonint: tuple[NonInt, ...] | None = None,
    cells: frozenset[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Return (errors, gaps, ok lines)."""
    mapping = MAPPING if mapping is None else mapping
    nonint = NONINT if nonint is None else nonint
    cells = CELLS if cells is None else cells
    rd = Reader(root)
    errors: list[str] = []
    gaps: list[str] = []
    ok: list[str] = []
    rec_text = rd.text(RECORDS)
    if rec_text is None:
        return [f"{RECORDS} is missing"], gaps, ok
    records = tomllib.loads(rec_text).get("record", [])
    integer_bounds = {
        (rec["id"], key)
        for rec in records
        for key, val in rec.get("bounds", {}).items()
        if _is_int(val)
    }
    stated = guide_placeholders(root)
    for record, key in sorted(stated - integer_bounds):
        errors.append(
            f"docs/guides placeholder {{{{bound:{record}.{key}}}}} does not name "
            f"an integer bound in {RECORDS}"
        )
    by_key = {(b.record, b.key): b for b in mapping}
    nonint_by_key = {(n.record, n.key): n for n in nonint}
    for b in mapping:
        if b.refusal:
            try:
                if re.compile(b.refusal).groups < 1:
                    errors.append(
                        f"mapping {b.record}.{b.key}: refusal regex /{b.refusal}/ has no capture group"
                    )
            except re.error as e:
                errors.append(f"mapping {b.record}.{b.key}: refusal regex invalid: {e}")
    short_ids = set()
    records = [rec for rec in records if rec["id"] in cells]
    for rec in records:
        short = rec["id"]
        short_ids.add(short)
        bounds = rec.get("bounds", {})
        numeric = {k: v for k, v in bounds.items() if _is_int(v)}
        label = rec["id"]
        frozen = rec.get("status") == "frozen"
        for key in sorted(numeric):
            if (short, key) not in by_key and frozen:
                gaps.append(
                    f"{label}: frozen record's bound `{key}` = {numeric[key]} has no mapping yet "
                    "(required once the record is in_progress)"
                )
            elif (short, key) not in by_key:
                errors.append(
                    f"{label}: numeric bound `{key}` = {numeric[key]} has no entry in MAPPING "
                    f"(scripts/check_limits_agreement.py); add a Bound({short!r}, {key!r}, rust=...) "
                    f"and a {{{{bound:{short}.{key}}}}} placeholder in docs/guides")
        for key, val in sorted(bounds.items()):
            if _is_int(val) or not re.search(r"\d", str(val)):
                continue
            row = nonint_by_key.get((short, key))
            if row is None:
                msg = (
                    f"{label}: non-integer bound `{key}` = {val!r} states numbers but has no "
                    "NONINT row (compare it or exclude it with a reason)"
                )
                (gaps if frozen else errors).append(msg)
                continue
            line = _check_nonint(rd, label, key, val, row, errors)
            if line:
                ok.append(line)
        refusal_when = " ".join(
            f["when"]
            for f in rec.get("refusals", [])
            if f.get("detail", "").endswith("bounds_exceeded")
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
            if b.refusal:
                problem = _refusal_problem(b.refusal, refusal_when, want)
                if problem:
                    errors.append(f"{tag}: {problem}")
                    bad = True
            if b.documented and (short, key) not in stated:
                errors.append(
                    f"{tag}: missing {{{{bound:{short}.{key}}}}} placeholder in docs/guides"
                )
                bad = True
            elif not b.documented:
                gaps.append(f"{tag}: no docs-page statement mapped ({b.note or 'owed'})")
            if not bad:
                ok.append(f"{tag}: agrees ({len(places)} place(s))")
    declared = {
        (r["id"], k)
        for r in records
        for k in r.get("bounds", {})
    }
    for b in mapping:
        if b.record in short_ids and (b.record, b.key) not in declared:
            errors.append(
                f"mapping {b.record}.{b.key} names a bound its record does not declare (stale mapping)"
            )
    for n in nonint:
        if n.record in short_ids and (n.record, n.key) not in declared:
            errors.append(
                f"NONINT row {n.record}.{n.key} names a bound its record does not declare (stale mapping)"
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
        print(f"ok   {len(exact_checked)} exact-record bound(s) agree (scripts/check_limits.py)")
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
        f"limits agreement ok: {len(ok)} bound(s) agree, {len(gaps)} documentation gap(s)"
    )
    return 0


# ------------------------------------------------------------------------------ self-test
# The self-test runs against a PINNED synthetic tree (never the live records, docs or sources), so an
# unrelated record or docs edit cannot break it. Every rule has a clean case and a mutant.
SYN_RECORDS = """
[[record]]
id = "synth.in_progress"
status = "in_progress"
bounds = { max_a = 4, max_b = 12, max_ops = 4096, depth = 24, nodes = "2 or 4, doubling check at 4 or 8", tol = "[1e-3, 1e-1]", ops_text = "4096 (cap): 32 ops in the fixture", memory_limit = "min(ctx, 64 MiB default)", contract = "SearchBudget" }
[[record.refusals]]
detail = "synth.bounds_exceeded"
when = "Sources outside 2-4, more than 12 b, or limits above 4096 operations / depth 24."

[[record]]
id = "synth.frozen"
status = "frozen"
bounds = { max_unmapped = 9, loose = "at most 7 things" }

[[record]]
id = "synth.carried"
status = "carried_forward"
bounds = { max_c = 3 }
[[record.refusals]]
detail = "carried.bounds_exceeded"
when = "More than 3 things."
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
SYN_DOC = (
    "Bounds: at most {{bound:synth.in_progress.max_a}} sources, "
    "{{bound:synth.in_progress.max_b}} b, {{bound:synth.in_progress.max_ops}} operations, "
    "depth {{bound:synth.in_progress.depth}}. At most {{bound:synth.carried.max_c}} c.\n"
)
SYN_PY = "def f(max_ops: int = 4_096): ...\n# at most 4 sources\n"
SYN_RS_PATH = "crates/synth/src/lib.rs"
SYN_DOC_PATH = "docs/guides/synth.md"
SYN_PY_PATH = "python/synth.py"
SYN_CELLS = frozenset({"synth.in_progress", "synth.frozen", "synth.carried"})
SYN_MAPPING: tuple[Bound, ...] = (
    Bound(
        "synth.in_progress",
        "max_a",
        rust=(C(SYN_RS_PATH, "SYN_MAX_A"),),

        python=(L(SYN_PY_PATH, r"at most (\d+) sources"),),
        refusal=r"Sources outside 2-(\d+)",
    ),
    Bound(
        "synth.in_progress",
        "max_b",
        rust=(C(SYN_RS_PATH, "SYN_MAX_B"),),  # alias of SYN_BASE_B

        refusal=r"more than (\d+) b",
    ),
    Bound(
        "synth.in_progress",
        "max_ops",
        rust=(L(SYN_RS_PATH, r"operations: ([\d_]+),"),),

        python=(L(SYN_PY_PATH, r"max_ops: int = ([\d_]+)"),),
        refusal=r"above (\d+) operations / depth",
    ),
    Bound(
        "synth.in_progress",
        "depth",
        rust=(L(SYN_RS_PATH, r"depth: ([\d_]+) "),),

        refusal=r"operations / depth (\d+)",
    ),
    Bound(
        "synth.carried",
        "max_c",
        rust=(C(SYN_RS_PATH, "SYN_MAX_C"),),

        refusal=r"More than (\d+) things",
    ),
)
SYN_NONINT: tuple[NonInt, ...] = (
    NonInt(
        "synth.in_progress",
        "nodes",
        places=(
            L(SYN_RS_PATH, r"SYN_NODES: \[usize; 2\] = \[(\d+), \d+\]"),
            L(SYN_RS_PATH, r"SYN_NODES: \[usize; 2\] = \[\d+, (\d+)\]"),
        ),
        doubled=True,
    ),
    NonInt(
        "synth.in_progress",
        "tol",
        places=(C(SYN_RS_PATH, "SYN_MIN_TOL"), C(SYN_RS_PATH, "SYN_MAX_TOL")),
    ),
    NonInt("synth.in_progress", "memory_limit", exclude="memory default is an expression"),
    NonInt(
        "synth.in_progress",
        "ops_text",
        head=r"^([^:]*):",
        places=(L(SYN_RS_PATH, r"operations: ([\d_]+),"),),
    ),
)


def _syn_tree(dest: Path) -> None:
    for rel, text in (
        (RECORDS, SYN_RECORDS),
        (SYN_RS_PATH, SYN_RS),
        ("crates/synth/src/tipping.rs", "pub const MIN_TOL: f64 = 1e-3;\n"),
        (SYN_DOC_PATH, SYN_DOC),
        (SYN_PY_PATH, SYN_PY),
    ):
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).write_text(text, encoding="utf-8", newline="\n")


def self_test() -> int:
    failures: list[str] = []

    def run(d: Path, **kw: object) -> tuple[list[str], list[str], list[str]]:
        return check(
            d,
            mapping=kw.pop("mapping", SYN_MAPPING),
            nonint=kw.pop("nonint", SYN_NONINT),
            cells=SYN_CELLS,
            **kw,  # type: ignore[arg-type]
        )

    with tempfile.TemporaryDirectory() as t:
        base = Path(t) / "base"
        _syn_tree(base)
        errors, gaps, ok = run(base)
        if errors:
            failures.append("clean fixture must pass: " + "; ".join(errors[:3]))
        if len(ok) < 8:
            failures.append(f"clean fixture checked only {len(ok)} bounds (vacuous)")
        if not any("max_unmapped" in g for g in gaps):
            failures.append("a frozen record's unmapped bound must be a gap")
        if any("max_unmapped" in e for e in errors):
            failures.append("a frozen record's unmapped bound must not be an error")
        if not any("loose" in g for g in gaps):
            failures.append(
                "a frozen record's unmapped non-integer bound must be a gap"
            )
        if not any("max_c" in line for line in ok):
            failures.append("a carried_forward record's bound must be checked")
        if not run(base, strict=True)[0]:
            failures.append("--strict must turn the frozen-record gaps into failures")

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
            "qualified_float_alias_followed",
            "crates/synth/src/tipping.rs",
            "MIN_TOL: f64 = 1e-3",
            "MIN_TOL: f64 = 2e-3",
            "tol",
        )
        mutated(
            "qualified_alias_wrong_module",
            SYN_RS_PATH,
            "crate::tipping::MIN_TOL",
            "crate::missing::MIN_TOL",
            "SYN_MIN_TOL",
        )
        mutated(
            "rust_literal",
            SYN_RS_PATH,
            "operations: 4_096",
            "operations: 4_097",
            "max_ops",
        )
        mutated(
            "docs_missing",
            SYN_DOC_PATH,
            "{{bound:synth.carried.max_c}}",
            "some c",
            "max_c",
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
        mutated(
            "unmapped_bound_carried",
            RECORDS,
            "bounds = { max_c = 3 }",
            "bounds = { max_c = 3, max_z = 1 }",
            "max_z",
        )
        # A rewording of one bound in the refusal text is caught even though the number
        # still appears elsewhere in the text ('4' remains in '2-4' and in the 4096 figure).
        mutated(
            "refusal_reword",
            RECORDS,
            "Sources outside 2-4",
            "Sources outside 2-5",
            "bounds_exceeded",
        )
        mutated(
            "refusal_reword_other_number_present",
            RECORDS,
            "more than 12 b",
            "more than 4 b",
            "max_b",
        )
        mutated("refusal_phrase_removed", RECORDS, "Sources outside 2-4, ", "", "max_a")
        mutated(
            "refusal_depth",
            RECORDS,
            "operations / depth 24",
            "operations / depth 25",
            "depth",
        )
        mutated(
            "refusal_carried",
            RECORDS,
            "More than 3 things",
            "More than 4 things",
            "max_c",
        )
        # non-integer bounds are compared, not ignored
        mutated(
            "nonint_quadrature",
            RECORDS,
            "2 or 4, doubling check at 4 or 8",
            "2 or 4, doubling check at 4 or 9",
            "nodes",
        )
        mutated(
            "nonint_rust",
            SYN_RS_PATH,
            "SYN_NODES: [usize; 2] = [2, 4]",
            "SYN_NODES: [usize; 2] = [2, 8]",
            "nodes",
        )
        mutated(
            "nonint_float",
            SYN_RS_PATH,
            "SYN_MAX_TOL: f64 = 1e-1",
            "SYN_MAX_TOL: f64 = 1e-2",
            "tol",
        )
        mutated(
            "nonint_unmapped",
            RECORDS,
            'memory_limit = "min(ctx, 64 MiB default)"',
            'memory_limit = "min(ctx, 64 MiB default)", extra = "at most 9 x"',
            "extra",
        )
        mutated(
            "nonint_head",
            RECORDS,
            "4096 (cap): 32 ops",
            "4097 (cap): 32 ops",
            "ops_text",
        )
        # stale mapping rows
        mutated(
            "stale_mapping",
            RECORDS,
            "max_c = 3 }",
            "max_q = 3 }",
            "stale mapping",
            mapping=SYN_MAPPING
            + (Bound("synth.carried", "max_c", rust=(C(SYN_RS_PATH, "SYN_MAX_C"),)),),)
        # A same-named constant in another file does not satisfy a mapping to this file.
        d = Path(t) / "const_elsewhere"
        shutil.copytree(base, d)
        rs = d / SYN_RS_PATH
        rs.write_text(rs.read_text(encoding="utf-8").replace("pub const SYN_MAX_C: usize = 3;\n", "", 1), encoding="utf-8", newline="\n")
        (d / "crates/synth/src/other.rs").write_text("pub const SYN_MAX_C: usize = 3;\n", encoding="utf-8", newline="\n")
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
            + (Bound("synth.in_progress", "gone", rust=(C(SYN_RS_PATH, "SYN_MAX_A"),)),),)
        if not any("stale mapping" in e for e in errs):
            failures.append(
                f"stale_row: a row for an undeclared bound must be stale, got {errs[:3]}"
            )
        errs, _, _ = run(
            d, nonint=SYN_NONINT + (NonInt("synth.in_progress", "gone", exclude="x"),)
        )
        if not any("stale mapping" in e for e in errs):
            failures.append("stale NONINT row must be reported")
        # a bare `refusal=True` is refused (no anchor)
        errs, _, _ = run(
            d,
            mapping=(replace(SYN_MAPPING[0], refusal=r"no group here"),)
            + SYN_MAPPING[1:],
        )
        if not any("capture group" in e for e in errs):
            failures.append("a refusal regex without a capture group must be an error")
        # a documentation gap is a gap, and --strict fails it
        gapless = tuple(
            replace(b, documented=False)
            if (b.record, b.key) == ("synth.carried", "max_c")
            else b
            for b in SYN_MAPPING
        )
        plain_errs, plain_gaps, _ = run(d, mapping=gapless)
        strict_errs, _, _ = run(d, mapping=gapless, strict=True)
        if plain_errs or not plain_gaps or len(strict_errs) <= len(plain_errs):
            failures.append("--strict must turn documentation gaps into failures")
        reword = Path(t) / "reword"
        shutil.copytree(base, reword)
        prose = reword / SYN_DOC_PATH
        prose.write_text(
            prose.read_text(encoding="utf-8").replace(
                "Bounds: at most",
                "Limits: no more than 4097 beside",
            )
        , encoding="utf-8", newline="\n")
        reword_errs, _, _ = run(reword)
        if reword_errs:
            failures.append(
                f"reword: surrounding prose must not fail, got {reword_errs[:3]}"
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
