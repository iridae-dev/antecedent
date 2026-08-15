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

if problems:
    print("provenance schema/path violations:")
    for problem in problems:
        print(f" - {problem}")
    sys.exit(1)

print(f"provenance schema/path audit: ok ({checked} records, {path_refs} path references)")
PY
