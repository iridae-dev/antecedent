"""Links from implementation ids to provenance records.

An estimator or identifier row in ``crates/antecedent/src/strategy_table/ids.rs`` carries a
``provenance`` pair (a family and a step). A family name that is not a file stem resolves
through ``ALIAS``; a family that is several records resolves through ``FAMILY``. The provenance
gate and the route-method inventory read the same table.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _repo import REPO_ROOT


def method_review_digest(record: dict) -> str:
    payload = {key: value for key, value in record.items() if key != "method_review"}
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def method_review_problems(root: Path, record: dict) -> list[str]:
    """Check integrity of explicit bounded source reviews; never invent a review."""
    review = record.get("method_review")
    if review is None:
        return []
    if not isinstance(review, dict):
        return ["method_review must be a table"]
    problems = []
    if review.get("status") != "reviewed_with_limits":
        problems.append("method_review has unknown status")
    for key in ("scope", "historical_exposure", "redistribution", "limitations"):
        if not isinstance(review.get(key), str) or not review[key].strip():
            problems.append(f"method_review lacks {key}")
    for key in ("assumptions", "sources", "findings_resolved"):
        values = review.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
        ):
            problems.append(f"method_review lacks explicit {key}")
    if review.get("record_sha256") != method_review_digest(record):
        problems.append("method_review stale provenance/citation/deviation content")
    artifacts = review.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return problems + ["method_review requires bound artifacts"]
    roles, seen = set(), set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            problems.append("method_review artifact must be a table")
            continue
        role, rel = artifact.get("role"), artifact.get("path")
        if role not in (
            "implementation",
            "assertion",
            "reference",
            "environment",
            "fixture",
        ):
            problems.append("method_review artifact has unknown role")
        else:
            roles.add(role)
        if not isinstance(rel, str) or rel in seen:
            problems.append("method_review requires unique artifact paths")
            continue
        seen.add(rel)
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            problems.append(f"method_review missing artifact: {rel}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != artifact.get("sha256"):
            problems.append(f"method_review stale artifact: {rel}")
    if not {"implementation", "assertion"}.issubset(roles):
        problems.append(
            "method_review must bind implementation and executing assertions"
        )
    return problems


ALIAS = {
    "identify.general_id": "identify.id",
    "identify.transport.sid": "identify.transport_sid",
    "estimate.propensity": "estimate.propensity_weighting",
    "estimate.matching": "estimate.distance_matching",
    "estimate.glm_adjustment": "estimate.glm",
    "estimate.frontdoor": "estimate.frontdoor_two_stage",
    "estimate.iv": "estimate.iv_2sls",
    "estimate.temporal_linear": "estimate.temporal_linear_adjustment",
    "estimate.conditional_linear": "estimate.conditional",
    "estimate.transport.trial_ipw": "estimate.trial_to_target",
    "estimate.interference.ht_hajek": "stats.randomized_interference",
}
FAMILY = {
    "estimate.propensity": (
        "estimate.propensity_weighting",
        "estimate.propensity_matching",
        "estimate.propensity_stratification",
    ),
}


def stems(root: Path = REPO_ROOT) -> set[str]:
    return {
        p.stem
        for p in (root / "provenance").glob("*.toml")
        if p.name != "_template.toml"
    }


def resolve(feature: str, known: set[str]) -> str | None:
    """The provenance record a feature id names, or None when no record exists."""
    resolved = ALIAS.get(feature, feature)
    return resolved if resolved in known else None


# Modules that execute or orchestrate other methods and implement no algorithm of their own. A
# method whose implementing module is listed here records the reason instead of a provenance
# record; any other admitted method must link a record.
NOT_AN_ALGORITHM = {
    "crates/antecedent/src/analysis/builder/finish.rs": "assembles the lag-zero AverageEffect study from the temporal linear adjustment and its Bayesian counterpart; no estimator of its own",
    "crates/antecedent/src/analysis/execute/checked_panel_did.rs": "checked execution wrapper over the panel difference-in-differences estimators",
    "crates/antecedent/src/analysis/execute/checked_local_polynomial_ratio.rs": "checked execution wrapper over the local polynomial ratio estimators",
    "crates/antecedent/src/analysis/temporal_admg_contrast.rs": "lowers a temporal contrast to the unrolled window and counts it; the identification is the existing do-calculus routine",
    "scripts/closure_audit.py": "an audit over method records, not a statistical method",
    "crates/antecedent/src/analysis/execute/compile.rs": "the compiler dispatch that migrates routes onto the common contracts; no estimator of its own",
}


# Modules whose method an existing record documents without naming the file.
MODULE_RECORDS = {
    "crates/antecedent-design/src/temporal_policy_transport.rs": (
        "transport.period_shift_policy",
    ),
    "crates/antecedent-estimate/src/smoothed_dose_posterior.rs": (
        "transport.smoothed_dose_response.bayesian",
    ),
    "crates/antecedent-estimate/src/dose_grid_posterior.rs": (
        "estimate.dose_grid_functional.bayesian",
    ),
    "crates/antecedent-estimate/src/categorical_treatment_posterior.rs": (
        "estimate.categorical_treatment.bayesian",
    ),
    "crates/antecedent-stats/src/adaptive_inference/participant_posterior.rs": (
        "adaptive_inference.participant_family",
        "adaptive_inference.bayesian_posterior",
    ),
    "crates/antecedent-estimate/src/temporal_design/misclassification.rs": (
        "measurement.binary_misclassification",
    ),
}


def records_for(module: str, root: Path = REPO_ROOT) -> list[str]:
    """Provenance records that name a module: its path, else its file name, else a feature id."""
    files = sorted(
        p for p in (root / "provenance").glob("*.toml") if p.name != "_template.toml"
    )
    texts = {p.stem: p.read_text(encoding="utf-8") for p in files}
    linked = [stem for stem in MODULE_RECORDS.get(module, ()) if stem in texts]
    if linked:
        return linked
    by_path = [stem for stem, text in texts.items() if module in text]
    if by_path:
        return by_path
    base = Path(module)
    stem = base.parent.name if base.stem == "mod" else base.stem
    by_file = [
        s for s, text in texts.items() if f"{stem}.rs" in text or f"{stem}/" in text
    ]
    if by_file:
        return by_file
    return [s for s in texts if s.endswith(("." + stem, "." + stem.replace("_", ".")))]


def public_items(module: str, root: Path = REPO_ROOT) -> set[str]:
    """Names of the public items a module (a file, or every file beneath a directory) defines."""
    import re

    path = root / module
    files = (
        [path]
        if path.is_file()
        else sorted(path.rglob("*.rs"))
        if path.is_dir()
        else []
    )
    if path.is_file() and path.name == "mod.rs":
        files = sorted(path.parent.rglob("*.rs"))
    names: set[str] = set()
    for file in files:
        if file.suffix != ".rs":
            continue
        text = file.read_text(encoding="utf-8", errors="replace")
        names |= set(
            re.findall(
                r"^\s*pub\s+(?:const\s+)?(?:struct|enum|fn|trait|type)\s+([A-Za-z_][A-Za-z0-9_]*)",
                text,
                re.MULTILINE,
            )
        )
    return names


def api_review(module: str, root: Path = REPO_ROOT) -> str:
    """Whether the public surface of a module is in the reviewed API snapshot of its crate."""
    import re

    match = re.match(r"crates/([^/]+)/", module)
    if not match:
        return "not a library surface"
    snapshot = f"parity/public_api/{match.group(1)}.txt"
    snap_path = root / snapshot
    names = public_items(module, root)
    if not names:
        return "internal: the module defines no public item"
    if not snap_path.is_file():
        return "unreviewed: no API snapshot for the crate"
    text = snap_path.read_text(encoding="utf-8", errors="replace")
    reviewed = sorted(
        n
        for n in names
        if re.search(r"(?<![A-Za-z0-9_])" + re.escape(n) + r"(?![A-Za-z0-9_])", text)
    )
    if not reviewed:
        return (
            "internal: no public item of the module reaches the crate's public surface"
        )
    return f"{snapshot}: {len(reviewed)} of {len(names)} public items reviewed"
