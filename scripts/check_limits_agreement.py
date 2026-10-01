#!/usr/bin/env python3
"""2.2 release closure: the declared bounds of every 2.2 record agree everywhere they are stated.

For every record of parity/promotion_2_2.toml, each numeric entry of its `bounds` table must
equal the value stated by each place mapped to it in `MAPPING` below:

  rust      a Rust constant (`const NAME: ty = <int | OTHER_CONST>;`) or an integer literal picked
            out of a source file by a regex, in the cell's own source files;
  docs      a sentence of docs/guides/transport-scope.md (or another docs page) that states the
            bound, picked out by a regex (digits, digits with `,` or `_`, or number words);
  python    a constant or docstring/default of the cell's Python module;
  refusal   the numbers quoted by the record's `*.bounds_exceeded` refusal `when` text.

The check fails, naming the record, the bound and the place, when a mapped constant, regex or
statement is missing or differs, and when a record carries a numeric bound that has no mapping
entry at all (so a new record cannot ship a bound nobody compares). A bound with a rust entry but
no docs entry is reported as a `gap` (a documentation statement owed, not a failure); `--strict`
makes gaps failures.

Extending it for a 2.2B record: add `Bound(...)` rows for the record's id prefix (`"2.2B.X3"`, ...) to
`MAPPING`. `rust=` and `python=` take `C(path, NAME)` (a constant) or `L(path, regex)` (first
capture group is the number; every match in the file must agree); `docs=` takes `L(path, regex)`;
`refusal=r"regex"` anchors the bound in the record's `*.bounds_exceeded` refusal text: the
regex's first capture group is the number, every match must equal the bound and at least one must
match (a bare `True` is refused: a number found anywhere in the text proves nothing).

A non-integer bound that carries digits (`"16 or 32, doubling check at 32 or 64"`, `"[1e-12, 1e-2]"`,
`"... (hard cap 100000)"`) must have a `NONINT` row: either the set of numbers it states equals the
set read from its places (`doubled=True` also admits twice each), or it is excluded with a recorded
reason. An unmapped one is an error (a gap on a `frozen` record). Works for any record status,
`carried_forward` included.

    python3 scripts/check_limits_agreement.py             # check the tree
    python3 scripts/check_limits_agreement.py --strict    # documentation gaps also fail
    python3 scripts/check_limits_agreement.py --self-test # synthetic disagreements must fail

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

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("LIMITS_AGREEMENT_ROOT", REPO))
RECORDS = "parity/promotion_2_2.toml"
SCOPE = "docs/guides/transport-scope.md"

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
    record: str  # record id prefix "2.2<milestone>.<workstream>", e.g. "2.2A.X1" or "2.2B.X2"
    key: str  # key of the record's `bounds` table
    rust: tuple[Place, ...] = ()
    docs: tuple[Place, ...] = ()
    python: tuple[Place, ...] = ()
    refusal: str = ""
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
XW_IO = "crates/antecedent-io/src/cross_world_artifact.rs"
MIX = "crates/antecedent-identify/src/sid/mixed_source.rs"
MIX_PY = "python/antecedent/transport/_mixed_source.py"

# ---- 2.2B cells
COND = "crates/antecedent-identify/src/sid/conditional.rs"
COND_PY = "python/antecedent/transport/_admg_conditional.py"
PLAN = "crates/antecedent-identify/src/sid/study_planning.rs"
PLAN_PY = "python/antecedent/transport/_study_planning.py"
REC = "crates/antecedent-identify/src/recovery.rs"
REC_PY = "python/antecedent/transport/_recovery.py"
DOSE = "crates/antecedent-estimate/src/smoothed_dose.rs"
DOSE_PY = "python/antecedent/transport/_smoothed_dose.py"
DOSE_DOC = "docs/smoothed-dose-response-transport.md"
JOINT = "crates/antecedent-validate/src/joint_mechanism_sensitivity.rs"
JOINT_PY = "python/antecedent/transport/_joint_sensitivity.py"
JOINT_DOC = "docs/guides/joint-mechanism-sensitivity.md"
CFID = "crates/antecedent-identify/src/counterfactual_id.rs"
CFID_PY = "python/antecedent/counterfactual_id.py"


def scope_and(doc: str, regex: str) -> tuple[Place, ...]:
    """The same statement in the scope guide and in the cell's own docs page."""
    return (L(SCOPE, regex), L(doc, regex))


# The mapping table: one row per (record, bounds key). Extend with additive rows for 2.2B records.
MAPPING: tuple[Bound, ...] = (
    # ---- X1 mz-transportability
    Bound(
        "2.2A.X1",
        "max_observed",
        rust=(C(MZ, "MZ_TRANSPORT_MAX_OBSERVED"),),
        docs=(
            L(
                SCOPE,
                r"bounds \((\d+) observed variables, \d+ controllables per source",
            ),
        ),
        python=(L(MZ_PY, r"two to four sources, (\w+) observed variables"),),
        refusal=r"more than (\d+) observed",
    ),
    Bound(
        "2.2A.X1",
        "max_sources",
        rust=(C(MZ, "MZ_TRANSPORT_MAX_SOURCES"),),
        docs=(L(SCOPE, r"from two to (\w+) source populations"),),
        python=(L(MZ_PY, r"two to (\w+) sources, \w+ observed variables"),),
        refusal=r"Sources outside 2-(\d+)",
    ),
    Bound(
        "2.2A.X1",
        "max_controllable_per_source",
        rust=(C(MZ, "MZ_TRANSPORT_MAX_CONTROLLABLE_PER_SOURCE"),),
        docs=(L(SCOPE, r"observed variables, (\d+) controllables per source"),),
        python=(L(MZ_PY, r"observed variables, (\w+) controllable variables per"),),
        refusal=r"controllables per source outside 1-(\d+)",
    ),
    Bound(
        "2.2A.X1",
        "max_candidate_regimes",
        rust=(C(MZ, "MZ_TRANSPORT_MAX_CANDIDATE_REGIMES"),),
        docs=(L(SCOPE, r"controllables per source, (\d+) candidate regimes"),),
        python=(L(MZ_PY, r"source and (\d+) candidate regimes"),),
        refusal=r"more than (\d+) candidate regimes",
    ),
    Bound(
        "2.2A.X1",
        "max_search_states",
        rust=(
            L(
                MZ,
                r"MZ_TRANSPORT_DEFAULT_LIMITS: SearchLimits = SearchLimits \{ operations: ([\d_]+),",
            ),
        ),
        docs=(L(SCOPE, r"candidate regimes,\s+([\d,]+) operations, depth \d+"),),
        python=(
            L(MZ_PY, r"catalog: EvidenceCatalog,\n\s+max_operations: int = ([\d_]+),"),
            L(MZ_PY, r"max_search_operations: int = ([\d_]+)"),
        ),
        refusal=r"above (\d+) operations / depth",
    ),
    Bound(
        "2.2A.X1",
        "max_depth",
        rust=(
            L(
                MZ,
                r"MZ_TRANSPORT_DEFAULT_LIMITS: SearchLimits = SearchLimits \{ operations: [\d_]+, depth: ([\d_]+)",
            ),
        ),
        docs=(L(SCOPE, r"candidate regimes,\s+[\d,]+ operations, depth (\d+)\)"),),
        python=(
            L(
                MZ_PY,
                r"catalog: EvidenceCatalog,\n\s+max_operations: int = [\d_]+,\n\s+max_depth: int = ([\d_]+),",
            ),
            L(MZ_PY, r"max_search_depth: int = ([\d_]+),"),
        ),
        refusal=r"operations / depth (\d+)",
    ),
    # ---- X2 finite scenario envelope
    Bound(
        "2.2A.X2",
        "max_scenarios",
        rust=(C(SCEN, "SCENARIO_MAX_COUNT"),),
        docs=(L(SCOPE, r"list of one to (\d+) fixed selection ADMGs"),),
        python=(
            L(SCEN_PY, r"^MAX_SCENARIOS = (\d+)"),
            L(SCEN_PY, r"One to (\d+) scenarios"),
        ),
    ),
    Bound(
        "2.2A.X2",
        "max_observed",
        rust=(C(SCEN, "SCENARIO_MAX_OBSERVED"),),
        docs=(
            L(SCOPE, r"\(at most (\d+) observed variables\) is decided independently"),
        ),
    ),
    # ---- X4 learned continuous trial transport
    Bound(
        "2.2A.X4",
        "max_folds",
        rust=(C(LEARN, "LEARNED_CONTINUOUS_MAX_FOLDS"),),
        docs=(L(SCOPE, r"At most (\d+) cross-fitting folds and a bootstrap cap"),),
        refusal=r"More than (\d+) folds",
        note="no Python statement of the fold cap",
    ),
    Bound(
        "2.2A.X4",
        "max_bootstrap",
        rust=(C(LEARN, "LEARNED_CONTINUOUS_MAX_BOOTSTRAP"),),
        docs=(L(SCOPE, r"bootstrap cap of ([\d,]+) replicates\s+\(floor"),),
        python=(L(LEARN_PY, r"\(at most (\d+), floor \d+\)"),),
        refusal=r"more than (\d+) bootstrap replicates",
    ),
    Bound(
        "2.2A.X4",
        "min_bootstrap",
        rust=(C(LEARN, "LEARNED_CONTINUOUS_MIN_BOOTSTRAP"),),
        docs=(L(SCOPE, r"replicate floor (\d+)"),),
        python=(L(LEARN_PY, r"\(at most \d+, floor (\d+)\)"),),
    ),
    Bound(
        "2.2A.X4",
        "max_rows_consumed",
        rust=(L(LEARN_IO, r"max_rows: ([\d_]+), max_features"),),
        docs=(L(SCOPE, r"a consumer admits at most ([\d,]+) rows and \d+ features"),),
    ),
    Bound(
        "2.2A.X4",
        "max_features_consumed",
        rust=(L(LEARN_IO, r"max_rows: [\d_]+, max_features: ([\d_]+)"),),
        docs=(L(SCOPE, r"a consumer admits at most [\d,]+ rows and (\d+) features"),),
    ),
    # ---- X5 two-step temporal transport
    Bound(
        "2.2A.X5",
        "horizon",
        rust=(C(TEMP, "TEMPORAL_HORIZON"),),
        docs=(L(SCOPE, r"The licensed horizon is (\d+) steps"),),
        python=(L(TEMP_PY, r"and the horizon is (\w+)\."),),
    ),
    Bound(
        "2.2A.X5",
        "max_actions",
        rust=(C(TEMP, "TEMPORAL_MAX_ACTIONS"),),
        docs=(L(SCOPE, r"alphabet \(at most (\d+) actions\)"),),
        python=(L(TEMP_PY, r"alphabet of at most (\w+) actions"),),
        refusal=r"More than (\d+) actions",
    ),
    Bound(
        "2.2A.X5",
        "max_observed",
        rust=(C(TEMP, "TEMPORAL_MAX_OBSERVED"),),
        docs=(L(SCOPE, r"diagram of at most (\d+) coordinates"),),
        python=(L(TEMP_PY, r"at most (\w+) coordinates and [\d,]+\s+complete"),),
        refusal=r"more than (\d+) coordinates",
    ),
    Bound(
        "2.2A.X5",
        "max_history_states",
        rust=(C(TEMP, "TEMPORAL_MAX_HISTORY_STATES"),),
        docs=(L(SCOPE, r"coordinates and ([\d,]+) complete covariate histories"),),
        python=(
            L(TEMP_PY, r"coordinates and ([\d,]+)\s+complete covariate histories"),
        ),
        refusal=r"more than (\d+) complete covariate histories",
    ),
    # ---- X8 path-specific edge intervention (cross-world)
    Bound(
        "2.2A.X8",
        "max_variables",
        rust=(C(XW_ID, "CROSS_WORLD_MAX_NODES"),),
        docs=(L(SCOPE, r"The graph has at most (\d+) variables, a query"),),
        note="no Python statement of the variable cap",
    ),
    Bound(
        "2.2A.X8",
        "max_worlds",
        rust=(C(XW_CORE, "MAX_CROSS_WORLDS"),),
        docs=(L(SCOPE, r"a query declares at most (\d+) worlds"),),
        note="no Python statement of the world cap",
    ),
    Bound(
        "2.2A.X8",
        "max_rows",
        rust=(C(XW_IO, "CROSS_WORLD_MAX_ROWS"),),
        docs=(L(SCOPE, r"a consumer admits at most ([\d,]+) rows; the claim"),),
        note="no Python statement of the row cap",
    ),
    # ---- X9 mixed-source proof search
    Bound(
        "2.2A.X9",
        "max_observed",
        rust=(C(MIX, "MIXED_SOURCE_MAX_OBSERVED"),),
        docs=(L(SCOPE, r"Bounds: at most (\d+) observed variables, \d+ usable"),),
        refusal=r"More than (\d+) observed variables",
        note="no Python statement of the observed-variable cap",
    ),
    Bound(
        "2.2A.X9",
        "max_distributions",
        rust=(C(MIX, "MIXED_SOURCE_MAX_DISTRIBUTIONS"),),
        docs=(
            L(SCOPE, r"observed variables, (\d+) usable distributions and \d+ sources"),
        ),
        refusal=r", (\d+) usable distributions",
        note="no Python statement of the distribution cap",
    ),
    Bound(
        "2.2A.X9",
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
        docs=(L(SCOPE, r"search budget of ([\d,]+) operations at depth \d+"),),
        refusal=r"above (\d+) operations / depth",
    ),
    Bound(
        "2.2A.X9",
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
        docs=(L(SCOPE, r"search budget of [\d,]+ operations at depth (\d+)"),),
        refusal=r"operations / depth (\d+)",
    ),
    Bound(
        "2.2A.X9",
        "max_move",
        rust=(C(MIX, "MIXED_SOURCE_MAX_MOVE"),),
        docs=(L(SCOPE, r"each with at most (\w+)\s+moved variables"),),
    ),
    Bound(
        "2.2A.X9",
        "max_intervention_set",
        rust=(C(MIX, "MIXED_SOURCE_MAX_DO"),),
        docs=(L(SCOPE, r"at most (\w+)\s+intervened variables"),),
    ),
    Bound(
        "2.2A.X9",
        "max_sources",
        rust=(C(MZ, "MZ_TRANSPORT_MAX_SOURCES"),),
        docs=(L(SCOPE, r"usable distributions and (\d+) sources, under a"),),
        python=(L(MIX_PY, r"declares at most (\w+) sources"),),
        refusal=r"(\d+) declared sources",
    ),
    # ---- 2.2B X2 ADMG conditional transport
    Bound(
        "2.2B.X2",
        "max_observed",
        rust=(C(COND, "ADMG_CONDITIONAL_MAX_OBSERVED"),),
        docs=(L(SCOPE, r"at most (\d+)\s+observed variables, 0-\d+ treatments"),),
        python=(L(COND_PY, r"more than (\d+) observed variables, \d+\s+treatments"),),
        refusal=r"More than (\d+) observed variables",
    ),
    Bound(
        "2.2B.X2",
        "max_treatments",
        rust=(C(COND, "ADMG_CONDITIONAL_MAX_TREATMENTS"),),
        docs=(L(SCOPE, r"0-(\d+) treatments, 1-\d+ conditioned variables"),),
        python=(
            L(COND_PY, r"observed variables, (\d+)\s+treatments or \d+ conditioned"),
        ),
        refusal=r"observed variables, (\d+) treatments",
    ),
    Bound(
        "2.2B.X2",
        "max_conditioned",
        rust=(C(COND, "ADMG_CONDITIONAL_MAX_CONDITIONED"),),
        docs=(L(SCOPE, r"0-\d+ treatments, 1-(\d+) conditioned variables"),),
        python=(L(COND_PY, r"treatments or (\d+) conditioned variables"),),
        refusal=r"treatments or (\d+) conditioned",
    ),
    Bound(
        "2.2B.X2",
        "max_search_states",
        rust=(
            L(
                COND,
                r"ADMG_CONDITIONAL_DEFAULT_LIMITS: SearchLimits =\s*SearchLimits \{ operations: ([\d_]+),",
            ),
        ),
        docs=(L(SCOPE, r"\(at most ([\d,]+) operations, depth \d+, a memory cap"),),
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
        "2.2B.X2",
        "max_depth",
        rust=(
            L(
                COND,
                r"ADMG_CONDITIONAL_DEFAULT_LIMITS: SearchLimits =\s*SearchLimits \{ operations: [\d_]+, depth: ([\d_]+)",
            ),
        ),
        docs=(L(SCOPE, r"\(at most [\d,]+ operations, depth (\d+), a memory cap"),),
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
    # ---- 2.2B X6 study planning
    Bound(
        "2.2B.X6",
        "max_candidates",
        rust=(C(PLAN, "STUDY_PLAN_MAX_CANDIDATES"),),
        docs=(
            L(
                SCOPE,
                r"At most (\d+) candidates, \d+ regimes each; every subset of at most \d+",
            ),
        ),
        python=(L(PLAN_PY, r"universe of at most (\w+) candidate studies"),),
        refusal=r"More than (\d+) candidates",
    ),
    Bound(
        "2.2B.X6",
        "max_regimes_per_candidate",
        rust=(C(PLAN, "STUDY_PLAN_MAX_REGIMES_PER_CANDIDATE"),),
        docs=(
            L(
                SCOPE,
                r"At most \d+ candidates, (\d+) regimes each; every subset of at most",
            ),
        ),
        refusal=r"more than (\d+) regimes",
        note="no Python statement of the per-candidate regime cap",
    ),
    Bound(
        "2.2B.X6",
        "max_subset_size",
        rust=(C(PLAN, "STUDY_PLAN_MAX_SUBSET"),),
        docs=(L(SCOPE, r"regimes each; every subset of at most (\d+)"),),
        python=(L(PLAN_PY, r"evaluates every subset of at most (\w+) studies"),),
    ),
    Bound(
        "2.2B.X6",
        "max_proposals",
        rust=(C(PLAN, "STUDY_PLAN_MAX_PROPOSALS"),),
        docs=(L(SCOPE, r"at most (\d+) proposals retained"),),
    ),
    Bound(
        "2.2B.X6",
        "max_search_states",
        rust=(
            L(
                PLAN,
                r"STUDY_PLAN_DEFAULT_LIMITS: SearchLimits = SearchLimits \{ operations: ([\d_]+),",
            ),
        ),
        docs=(
            L(SCOPE, r"\(at most (\d+) operations, depth \d+, \d+ on the mixed route"),
        ),
        python=(L(PLAN_PY, r"^_MAX_OPERATIONS = ([\d_]+)"),),
        refusal=r"limits above (\d+) operations",
    ),
    Bound(
        "2.2B.X6",
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
        docs=(
            L(SCOPE, r"\(at most \d+ operations, depth (\d+), \d+ on the mixed route"),
        ),
        python=(L(PLAN_PY, r'^_MAX_DEPTH = \{"mz": (\d+), "mixed": \d+\}'),),
        refusal=r"\((\d+) Mz,",
    ),
    Bound(
        "2.2B.X6",
        "max_depth_mixed",
        rust=(
            L(
                MIX,
                r"MIXED_SOURCE_DEFAULT_LIMITS: SearchLimits =\s*SearchLimits \{ operations: [\d_]+, depth: ([\d_]+)",
            ),
        ),
        docs=(
            L(SCOPE, r"\(at most \d+ operations, depth \d+, (\d+) on the mixed route"),
        ),
        python=(L(PLAN_PY, r'^_MAX_DEPTH = \{"mz": \d+, "mixed": (\d+)\}'),),
        refusal=r", (\d+) Mixed\)",
    ),
    # ---- 2.2B X10 binary observation recovery
    Bound(
        "2.2B.X10",
        "max_partially_observed",
        rust=(C(REC, "RECOVERY_MAX_PARTIALLY_OBSERVED"),),
        docs=(L(SCOPE, r"at most (\d+) partially observed variables"),),
        refusal=r"More than (\d+) partially observed",
        note="no Python statement of the partially-observed cap",
    ),
    Bound(
        "2.2B.X10",
        "max_fully_observed",
        rust=(C(REC, "RECOVERY_MAX_FULLY_OBSERVED"),),
        docs=(L(SCOPE, r"at most (\d+) fully observed `O`"),),
        refusal=r"or (\d+) fully observed",
        note="no Python statement of the fully-observed cap",
    ),
    Bound(
        "2.2B.X10",
        "max_observed_cells",
        rust=(C(REC, "RECOVERY_MAX_OBSERVED_CELLS"),),
        docs=(L(SCOPE, r"at most (\d+)\s+observed cells"),),
        refusal=r"more than (\d+) observed-law cells",
        note="no Python statement of the cell cap",
    ),
    Bound(
        "2.2B.X10",
        "max_search_operations",
        rust=(
            L(
                REC,
                r"RECOVERY_DEFAULT_LIMITS: RecoveryLimits = RecoveryLimits \{\n\s+search: SearchLimits \{ operations: ([\d_]+),",
            ),
        ),
        docs=(L(SCOPE, r"\(at most (\d+) operations, depth \d+\) bounds every stage"),),
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
        "2.2B.X10",
        "max_depth",
        rust=(
            L(
                REC,
                r"RECOVERY_DEFAULT_LIMITS: RecoveryLimits = RecoveryLimits \{\n\s+search: SearchLimits \{ operations: [\d_]+, depth: ([\d_]+)",
            ),
        ),
        docs=(L(SCOPE, r"\(at most \d+ operations, depth (\d+)\) bounds every stage"),),
        python=(
            L(
                REC_PY,
                r"effect_graph: Admg \| None = None,\n\s+max_operations: int = [\d_]+,\n\s+max_depth: int = ([\d_]+),",
            ),
            L(REC_PY, r"max_search_depth: int = ([\d_]+)"),
        ),
        refusal=r"operations / depth (\d+)",
    ),
    # ---- 2.2B X4 smoothed dose-response transport
    Bound(
        "2.2B.X4",
        "max_grid_points",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_GRID"),),
        docs=(
            *scope_and(DOSE_DOC, r"\bAt most (\d+) grid doses"),
            L(DOSE_DOC, r"a grid of at most (\d+) doses"),
        ),
        python=(L(DOSE_PY, r"``grid`` holds at most (\d+) distinct doses"),),
        refusal=r"More than (\d+) grid doses",
    ),
    Bound(
        "2.2B.X4",
        "max_folds",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_FOLDS"),),
        docs=scope_and(DOSE_DOC, r"grid doses, (\d+) cross-fitting folds"),
        refusal=r"more than (\d+) folds",
        note="no Python statement of the fold cap",
    ),
    Bound(
        "2.2B.X4",
        "max_rows",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_ROWS"),),
        docs=scope_and(DOSE_DOC, r"cross-fitting folds, ([\d,]+) rows,"),
        refusal=r"more than (\d+) rows",
        note="no Python statement of the row cap",
    ),
    Bound(
        "2.2B.X4",
        "max_features",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_FEATURES"),),
        docs=scope_and(DOSE_DOC, r"rows, (\d+) covariates,"),
        refusal=r"more than (\d+) covariates",
        note="no Python statement of the covariate cap",
    ),
    Bound(
        "2.2B.X4",
        "max_basis_degree",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_BASIS_DEGREE"),),
        docs=scope_and(DOSE_DOC, r"covariates,\s+basis degree 1 to (\d+),"),
        refusal=r"outside 1\.\.=(\d+)",
        note="no Python statement of the basis degree cap",
    ),
    Bound(
        "2.2B.X4",
        "max_knots",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_KNOTS"),),
        docs=scope_and(DOSE_DOC, r"basis degree 1 to \d+,\s+at most (\d+) knots"),
        refusal=r"more than (\d+) knots",
        note="no Python statement of the knot cap",
    ),
    Bound(
        "2.2B.X4",
        "min_bootstrap",
        rust=(C(DOSE, "SMOOTHED_DOSE_MIN_BOOTSTRAP"),),
        docs=scope_and(DOSE_DOC, r"bootstrap request of (\d+) to [\d,]+ replicates"),
        python=(L(DOSE_PY, r"\(at most \d+, floor (\d+)\)"),),
    ),
    Bound(
        "2.2B.X4",
        "max_bootstrap",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_BOOTSTRAP"),),
        docs=scope_and(DOSE_DOC, r"bootstrap request of \d+ to ([\d,]+) replicates"),
        python=(L(DOSE_PY, r"\(at most (\d+), floor \d+\)"),),
        refusal=r"more than (\d+) bootstrap replicates",
    ),
    Bound(
        "2.2B.X4",
        "max_workspace_bytes",
        rust=(C(DOSE, "SMOOTHED_DOSE_MAX_WORKSPACE_BYTES"),),
        docs=(L(DOSE_DOC, r"mandatory cap of 512 MiB \(([\d,]+) bytes\)"),),
        python=(L(DOSE_PY, r"mandatory 512 MiB \(([\d,]+) bytes\) cap"),),
    ),
    # ---- 2.2B X3 joint mechanism sensitivity
    Bound(
        "2.2B.X3",
        "max_factors",
        rust=(C(JOINT, "JOINT_SENSITIVITY_MAX_FACTORS"),),
        docs=scope_and(JOINT_DOC, r"[Aa]t most (\d+) factors declared"),
    ),
    Bound(
        "2.2B.X3",
        "max_parent_levels",
        rust=(C(JOINT, "JOINT_SENSITIVITY_MAX_PARENT_LEVELS"),),
        docs=scope_and(JOINT_DOC, r"factors declared \([^)]*\), (\d+) parent levels"),
        refusal=r"More than (\d+) parent levels",
    ),
    Bound(
        "2.2B.X3",
        "max_outcome_categories",
        rust=(C(JOINT, "JOINT_SENSITIVITY_MAX_OUTCOME_CATEGORIES"),),
        docs=scope_and(JOINT_DOC, r"parent levels, (\d+) outcome categories"),
        refusal=r"or (\d+) outcome categories",
    ),
    Bound(
        "2.2B.X3",
        "max_frontier_points",
        rust=(C(JOINT, "JOINT_SENSITIVITY_MAX_FRONTIER_POINTS"),),
        docs=scope_and(JOINT_DOC, r"frontier grid of 1 to (\d+)\s+points"),
        refusal=r"grid outside 1-(\d+)",
    ),
    Bound(
        "2.2B.X3",
        "max_operations",
        rust=(C(JOINT, "JOINT_SENSITIVITY_MAX_OPERATIONS"),),
        docs=scope_and(JOINT_DOC, r"at most ([\d,]+)\s+search operations"),
        python=(
            L(JOINT_PY, r"^    max_operations: int = ([\d_]+)$"),
            L(JOINT_PY, r"max_search_operations: int = ([\d_]+)"),
        ),
        refusal=r"above (\d+) operations / depth",
    ),
    Bound(
        "2.2B.X3",
        "max_bisection_iterations",
        rust=(C(JOINT, "JOINT_SENSITIVITY_MAX_DEPTH"),),
        docs=scope_and(JOINT_DOC, r"bisection depth of\s+(\d+)\s+iterations"),
        python=(
            L(JOINT_PY, r"^    max_depth: int = ([\d_]+)$"),
            L(JOINT_PY, r"max_search_depth: int = ([\d_]+)"),
        ),
        refusal=r"operations / depth (\d+)",
    ),
    Bound(
        "2.2B.X3",
        "max_bootstrap_internal",
        rust=(C(JOINT, "JOINT_SENSITIVITY_MAX_BOOTSTRAP"),),
        docs=scope_and(JOINT_DOC, r"bootstrap request\s+cap of (\d+)"),
        refusal=r"more than (\d+) bootstrap replicates",
    ),
    # ---- 2.2B X8 ADMG counterfactual identification
    Bound(
        "2.2B.X8",
        "max_variables",
        rust=(C(CFID, "COUNTERFACTUAL_ID_MAX_VARIABLES"),),
        docs=(L(SCOPE, r"of at most (\d+) finite-discrete variables"),),
        python=(L(CFID_PY, r"at most (\w+) finite-discrete\s+variables"),),
        refusal=r"More than (\d+) variables",
    ),
    Bound(
        "2.2B.X8",
        "max_levels",
        rust=(C(CFID, "COUNTERFACTUAL_ID_MAX_LEVELS"),),
        docs=(L(SCOPE, r"at\s+most (\d+) levels each"),),
        python=(L(CFID_PY, r"at most (\w+) levels each"),),
        refusal=r"more than (\d+) levels",
    ),
    Bound(
        "2.2B.X8",
        "max_search_operations",
        rust=(C(CFID, "COUNTERFACTUAL_ID_MAX_OPERATIONS"),),
        docs=(L(SCOPE, r"\(at most (\d+) operations, depth \d+\); Python"),),
        python=(L(CFID_PY, r"the one search budget \(at most (\d+) and \d+\)"),),
        refusal=r"above (\d+) operations or depth",
    ),
    Bound(
        "2.2B.X8",
        "max_search_depth",
        rust=(C(CFID, "COUNTERFACTUAL_ID_MAX_DEPTH"),),
        docs=(L(SCOPE, r"\(at most \d+ operations, depth (\d+)\); Python"),),
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
    NonInt("2.2A.X1", "memory_limit", exclude=_MEMORY_REASON),
    NonInt("2.2B.X2", "memory_limit", exclude=_MEMORY_REASON),
    NonInt("2.2B.X6", "memory_limit", exclude=_MEMORY_REASON),
    NonInt("2.2B.X10", "memory_limit", exclude=_MEMORY_REASON),
    NonInt("2.2B.X8", "memory_limit", exclude=_MEMORY_REASON),
    NonInt("2.2B.X3", "memory_limit", exclude=_MEMORY_REASON),
    NonInt(
        "2.2B.X6",
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
        "2.2B.X4",
        "quadrature_nodes",
        places=(
            L(DOSE, r"SMOOTHED_DOSE_QUADRATURE_NODES: \[usize; 2\] = \[(\d+), \d+\]"),
            L(DOSE, r"SMOOTHED_DOSE_QUADRATURE_NODES: \[usize; 2\] = \[\d+, (\d+)\]"),
        ),
        doubled=True,
    ),
    NonInt(
        "2.2B.X3",
        "tolerance",
        places=(
            C(JOINT, "JOINT_SENSITIVITY_MIN_TOLERANCE"),
            C(JOINT, "JOINT_SENSITIVITY_MAX_TOLERANCE"),
        ),
    ),
    NonInt(
        "2.2B.X3",
        "operation_limit",
        places=(C(JOINT, "JOINT_SENSITIVITY_MAX_OPERATIONS"),),
    ),
    NonInt("2.2B.X3", "depth_limit", places=(C(JOINT, "JOINT_SENSITIVITY_MAX_DEPTH"),)),
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
            self._cache[rel] = p.read_text() if p.is_file() else None
        return self._cache[rel]

    def find_const(
        self, name: str, near: str, depth: int = 0
    ) -> tuple[int | float | None, str]:
        """Value of `const NAME: ty = <int|ALIAS>;`, searching `near` first then crates/*/src."""
        files = [near] + sorted(
            str(p.relative_to(self.root))
            for p in (self.root / "crates").glob("*/src/**/*.rs")
            if str(p.relative_to(self.root)) != near
        )
        pat = re.compile(
            rf"\bconst {re.escape(name)}\s*:\s*[\w<>\[\]]+\s*=\s*([\w.\-]+)\s*;"
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


def check(
    root: Path,
    strict: bool = False,
    mapping: tuple[Bound, ...] | None = None,
    nonint: tuple[NonInt, ...] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Return (errors, gaps, ok lines)."""
    mapping = MAPPING if mapping is None else mapping
    nonint = NONINT if nonint is None else nonint
    rd = Reader(root)
    errors: list[str] = []
    gaps: list[str] = []
    ok: list[str] = []
    rec_text = rd.text(RECORDS)
    if rec_text is None:
        return [f"{RECORDS} is missing"], gaps, ok
    records = tomllib.loads(rec_text).get("record", [])
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
    for rec in records:
        short = ".".join(
            rec["id"].split(".")[:3]
        )  # "2.2A.X1": milestone and workstream
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
                    f"(scripts/check_limits_agreement.py); add a Bound({short!r}, {key!r}, rust=..., docs=...)"
                )
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
            places = (
                [("rust", p) for p in b.rust]
                + [("docs", p) for p in b.docs]
                + [("python", p) for p in b.python]
            )
            if not b.rust:
                errors.append(f"{tag}: mapping names no Rust constant")
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
            if not b.docs and not b.python:
                gaps.append(
                    f"{tag}: no docs or Python statement mapped ({b.note or 'owed'})"
                )
            elif not b.docs:
                gaps.append(f"{tag}: no docs-page statement mapped")
            if not bad:
                ok.append(f"{tag}: agrees ({len(places)} place(s))")
    declared = {
        (".".join(r["id"].split(".")[:3]), k)
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
        return self_test()
    strict = "--strict" in argv
    errors, gaps, ok = check(ROOT, strict)
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
id = "2.2Z.A1.synth_in_progress"
status = "in_progress"
bounds = { max_a = 4, max_b = 12, max_ops = 4096, depth = 24, nodes = "2 or 4, doubling check at 4 or 8", tol = "[1e-3, 1e-1]", ops_text = "4096 (cap): 32 ops in the fixture", memory_limit = "min(ctx, 64 MiB default)", contract = "SearchBudget" }
[[record.refusals]]
detail = "synth.bounds_exceeded"
when = "Sources outside 2-4, more than 12 b, or limits above 4096 operations / depth 24."

[[record]]
id = "2.2Z.B1.synth_frozen"
status = "frozen"
bounds = { max_unmapped = 9, loose = "at most 7 things" }

[[record]]
id = "2.2Z.C1.synth_carried"
status = "carried_forward"
bounds = { max_c = 3 }
[[record.refusals]]
detail = "carried.bounds_exceeded"
when = "More than 3 things."
"""
SYN_RS = """
pub const SYN_BASE_B: usize = 12;
pub const SYN_MAX_B: usize = SYN_BASE_B;
pub const SYN_MAX_A: usize = 4;
pub const SYN_MAX_C: usize = 3;
pub const SYN_DEFAULT: Limits = Limits { operations: 4_096, depth: 24 };
pub const SYN_NODES: [usize; 2] = [2, 4];
pub const SYN_MIN_TOL: f64 = 1e-3;
pub const SYN_MAX_TOL: f64 = 1e-1;
"""
SYN_DOC = "Bounds: at most 4 sources, 12 b, 4096 operations, depth 24. At most 3 c.\n"
SYN_PY = "def f(max_ops: int = 4_096): ...\n# at most 4 sources\n"
SYN_RS_PATH = "crates/synth/src/lib.rs"
SYN_DOC_PATH = "docs/synth.md"
SYN_PY_PATH = "python/synth.py"
SYN_MAPPING: tuple[Bound, ...] = (
    Bound(
        "2.2Z.A1",
        "max_a",
        rust=(C(SYN_RS_PATH, "SYN_MAX_A"),),
        docs=(L(SYN_DOC_PATH, r"at most (\d+) sources"),),
        python=(L(SYN_PY_PATH, r"at most (\d+) sources"),),
        refusal=r"Sources outside 2-(\d+)",
    ),
    Bound(
        "2.2Z.A1",
        "max_b",
        rust=(C(SYN_RS_PATH, "SYN_MAX_B"),),  # alias of SYN_BASE_B
        docs=(L(SYN_DOC_PATH, r"sources, (\d+) b,"),),
        refusal=r"more than (\d+) b",
    ),
    Bound(
        "2.2Z.A1",
        "max_ops",
        rust=(L(SYN_RS_PATH, r"operations: ([\d_]+),"),),
        docs=(L(SYN_DOC_PATH, r"b, (\d+) operations"),),
        python=(L(SYN_PY_PATH, r"max_ops: int = ([\d_]+)"),),
        refusal=r"above (\d+) operations / depth",
    ),
    Bound(
        "2.2Z.A1",
        "depth",
        rust=(L(SYN_RS_PATH, r"depth: ([\d_]+) "),),
        docs=(L(SYN_DOC_PATH, r"operations, depth (\d+)\."),),
        refusal=r"operations / depth (\d+)",
    ),
    Bound(
        "2.2Z.C1",
        "max_c",
        rust=(C(SYN_RS_PATH, "SYN_MAX_C"),),
        docs=(L(SYN_DOC_PATH, r"At most (\d+) c\."),),
        refusal=r"More than (\d+) things",
    ),
)
SYN_NONINT: tuple[NonInt, ...] = (
    NonInt(
        "2.2Z.A1",
        "nodes",
        places=(
            L(SYN_RS_PATH, r"SYN_NODES: \[usize; 2\] = \[(\d+), \d+\]"),
            L(SYN_RS_PATH, r"SYN_NODES: \[usize; 2\] = \[\d+, (\d+)\]"),
        ),
        doubled=True,
    ),
    NonInt(
        "2.2Z.A1",
        "tol",
        places=(C(SYN_RS_PATH, "SYN_MIN_TOL"), C(SYN_RS_PATH, "SYN_MAX_TOL")),
    ),
    NonInt("2.2Z.A1", "memory_limit", exclude="memory default is an expression"),
    NonInt(
        "2.2Z.A1",
        "ops_text",
        head=r"^([^:]*):",
        places=(L(SYN_RS_PATH, r"operations: ([\d_]+),"),),
    ),
)


def _syn_tree(dest: Path) -> None:
    for rel, text in (
        (RECORDS, SYN_RECORDS),
        (SYN_RS_PATH, SYN_RS),
        (SYN_DOC_PATH, SYN_DOC),
        (SYN_PY_PATH, SYN_PY),
    ):
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).write_text(text)


def self_test() -> int:
    failures: list[str] = []

    def run(d: Path, **kw: object) -> tuple[list[str], list[str], list[str]]:
        return check(
            d,
            mapping=kw.pop("mapping", SYN_MAPPING),
            nonint=kw.pop("nonint", SYN_NONINT),
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
            text = p.read_text()
            if old not in text:
                failures.append(f"{name}: mutation anchor {old!r} not found in {rel}")
                return
            p.write_text(text.replace(old, new, 1))
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
            "docs_number", SYN_DOC_PATH, "4096 operations", "4097 operations", "max_ops"
        )
        mutated("docs_missing", SYN_DOC_PATH, "At most 3 c.", "Some c.", "max_c")
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
        # D2: a rewording of one bound in the refusal text is caught even though the number
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
            + (Bound("2.2Z.C1", "max_c", rust=(C(SYN_RS_PATH, "SYN_MAX_C"),)),),
        )
        d = Path(t) / "stale_row"
        shutil.copytree(base, d)
        errs, _, _ = run(
            d,
            mapping=SYN_MAPPING
            + (Bound("2.2Z.A1", "gone", rust=(C(SYN_RS_PATH, "SYN_MAX_A"),)),),
        )
        if not any("stale mapping" in e for e in errs):
            failures.append(
                f"stale_row: a row for an undeclared bound must be stale, got {errs[:3]}"
            )
        errs, _, _ = run(
            d, nonint=SYN_NONINT + (NonInt("2.2Z.A1", "gone", exclude="x"),)
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
            replace(b, docs=()) if (b.record, b.key) == ("2.2Z.C1", "max_c") else b
            for b in SYN_MAPPING
        )
        plain_errs, plain_gaps, _ = run(d, mapping=gapless)
        strict_errs, _, _ = run(d, mapping=gapless, strict=True)
        if plain_errs or not plain_gaps or len(strict_errs) <= len(plain_errs):
            failures.append("--strict must turn documentation gaps into failures")
    if failures:
        print("check_limits_agreement self-test FAILED:")
        for f in failures:
            print(f" - {f}")
        return 1
    print("check_limits_agreement self-test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
