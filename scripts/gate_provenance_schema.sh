#!/usr/bin/env bash
# Validate every machine-readable algorithm-provenance record and referenced path.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
from pathlib import Path
import re
import sys
import tomllib

root = Path(".")
records = sorted((root / "provenance").glob("*.toml"))
allowed_exposure = {
    "none",
    "previous familiarity",
    "black-box comparison only",
}
required_booleans = (
    "source_translation",
    "copied_code",
    "copied_comments",
    "copied_tests",
)
path_pattern = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"((?:crates|python|conformance|tests|parity|benches|docs)/[A-Za-z0-9_./-]+)"
)
problems: list[str] = []
checked = 0
path_refs = 0

for path in records:
    if path.name == "_template.toml":
        continue
    checked += 1
    rel = path.relative_to(root)
    try:
        record = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        problems.append(f"{rel}: cannot parse: {error}")
        continue

    if record.get("feature_id") != path.stem:
        problems.append(
            f"{rel}: feature_id must equal filename stem {path.stem!r}"
        )

    crate = record.get("implementation_crate")
    crate_exists = isinstance(crate, str) and (
        (root / "crates" / crate).is_dir()
        or (crate == "antecedent-py" and (root / "python").is_dir())
    )
    if not crate_exists:
        problems.append(f"{rel}: unknown implementation_crate {crate!r}")

    for field in required_booleans:
        if not isinstance(record.get(field), bool):
            problems.append(f"{rel}: {field} must be a boolean")

    papers = record.get("papers")
    if not isinstance(papers, list):
        problems.append(f"{rel}: papers must be an array")
    else:
        for index, paper in enumerate(papers):
            prefix = f"{rel}: papers[{index}]"
            if not isinstance(paper, dict):
                problems.append(f"{prefix} must be a table")
                continue
            if not isinstance(paper.get("title"), str) or not paper["title"].strip():
                problems.append(f"{prefix}.title must be non-empty")
            sections = paper.get("sections")
            if not isinstance(sections, list) or not sections or any(
                not isinstance(section, str) or not section.strip()
                for section in sections
            ):
                problems.append(f"{prefix}.sections must name exact locations")
            if "doi" in paper and (
                not isinstance(paper["doi"], str) or not paper["doi"].strip()
            ):
                problems.append(f"{prefix}.doi must be omitted rather than empty")

    upstream = record.get("upstream_implementations_observed")
    if not isinstance(upstream, list):
        problems.append(f"{rel}: upstream_implementations_observed must be an array")
    else:
        for index, implementation in enumerate(upstream):
            prefix = f"{rel}: upstream_implementations_observed[{index}]"
            if not isinstance(implementation, dict):
                problems.append(f"{prefix} must be a table, not a bare string")
                continue
            if not isinstance(implementation.get("project"), str) or not implementation[
                "project"
            ].strip():
                problems.append(f"{prefix}.project must be non-empty")
            if implementation.get("exposure") not in allowed_exposure:
                problems.append(f"{prefix}.exposure is not an allowed disclosure")

    test_sources = record.get("test_sources")
    if not isinstance(test_sources, list) or not test_sources or any(
        not isinstance(source, str) or not source.strip() for source in test_sources
    ):
        problems.append(f"{rel}: test_sources must be a non-empty string array")
        continue
    for source in test_sources:
        for match in path_pattern.finditer(source):
            path_refs += 1
            referenced = match.group(1).rstrip(".,);:]")
            if not (root / referenced).exists():
                problems.append(f"{rel}: missing test-source path {referenced}")

# Every closed-set identifier/estimator provenance pointer, and every named
# CI factory id, must resolve to a record. Family names that are not
# filenames are explicit aliases — a new closed-set id cannot ship without
# either a matching file or an alias to one.
stems = {
    p.stem
    for p in (root / "provenance").glob("*.toml")
    if p.name != "_template.toml"
}
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
ids_rs = (root / "crates/antecedent/src/strategy_table/ids.rs").read_text()
pointers = re.findall(r'provenance:\s*(?:\n\s*)?\(\s*"([^"]+)"', ids_rs)
for feat in pointers:
    resolved = ALIAS.get(feat, feat)
    if resolved not in stems:
        problems.append(
            f"strategy-table provenance {feat!r} does not resolve to a "
            f"provenance record (tried {resolved!r})"
        )
    for member in FAMILY.get(feat, ()):
        if member not in stems:
            problems.append(
                f"strategy-table family {feat!r} is missing member record {member!r}"
            )

CI_CLAIM = {
    "parcorr": "ci.partial_correlation",
    "partial_corr": "ci.partial_correlation",
    "partial_correlation": "ci.partial_correlation",
    "robust_parcorr": "ci.parcorr_variants",
    "robust_partial_corr": "ci.parcorr_variants",
    "weighted_parcorr": "ci.parcorr_variants",
    "weighted_partial_corr": "ci.parcorr_variants",
    "multivariate_parcorr": "ci.parcorr_variants",
    "multivariate_partial_corr": "ci.parcorr_variants",
    "pairwise_multivariate": "ci.parcorr_variants",
    "pairwise_mv": "ci.parcorr_variants",
    "gsquared": "ci.gsquared",
    "g_squared": "ci.gsquared",
    "regression": "ci.partial_correlation",
    "knn_dependence": "ci.knn_dependence",
    "mixed_knn_dependence": "ci.knn_dependence",
    "symbolic_cmi": "ci.symbolic_cmi",
    "gpdc": "ci.gpdc",
    "bayes_factor": "ci.bayes",
    "bayes_factor_ci": "ci.bayes",
    "posterior_dependence": "ci.bayes",
    "posterior_dependence_ci": "ci.bayes",
    "posterior_predictive_ci": "ci.bayes",
    "ppc_ci": "ci.bayes",
}
# Oracle CI is a synthetic fixture, not a published algorithm.
CI_SKIP = {"oracle"}
factory = (root / "crates/antecedent-stats/src/ci/factory.rs").read_text()
fn = re.search(r"pub fn ci_from_name\b.*?(?=\n#\[cfg|\nimpl |\Z)", factory, re.S)
factory_names = set(re.findall(r'"([a-z][a-z0-9_]*)"', fn.group(0) if fn else ""))
for name in sorted(factory_names):
    if name in CI_SKIP:
        continue
    claim = CI_CLAIM.get(name)
    if claim is None:
        problems.append(f"CI factory name {name!r} has no provenance mapping")
    elif claim not in stems:
        problems.append(
            f"CI factory name {name!r} maps to missing provenance record {claim!r}"
        )

for extra in (
    "estimate.identified_set",
    "estimate.graph_posterior.joint_if",
    "estimate.quantile",
    "identify.tiered",
    "ci.knn_dependence",
    "ci.symbolic_cmi",
):
    if extra not in stems:
        problems.append(f"required provenance record missing: {extra}")

if problems:
    print("provenance schema/path violations:")
    for problem in problems:
        print(f" - {problem}")
    sys.exit(1)

print(f"provenance schema/path audit: ok ({checked} records, {path_refs} path references)")
PY
