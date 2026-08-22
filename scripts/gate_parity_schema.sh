#!/usr/bin/env bash
# Parity manifest schema gate: every [[capabilities]] row carries its required keys.
#
# Why this exists: the feature gates parse manifests with a regex `caps()` helper
# whose accessor takes an explicit default (`g("status", default=None)`). A row
# missing `status` therefore reads as None, matches none of the honesty checks,
# and passes every gate forever without ever being marked done/pending. Those
# parsers cannot detect an absent key by construction -- this gate is the
# schema-completeness check that closes that hole.
#
# It also pins the regex parser itself: the ids and statuses the gates' `caps()`
# recovers must agree with a real TOML parse, so a manifest whose layout drifts
# out from under the regex fails here instead of silently under-reporting.
#
# Run standalone, or via any feature gate / gate_release.sh, which all invoke it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
from pathlib import Path
import json
import re
import sys
import tomllib

root = Path(".")

STATUS_VALUES = {"pending", "in_progress", "done"}
# "none" is for Rust-only primitives. Recording them as "thin" would claim a Python
# surface that does not exist; omitting the row would leave a shipped capability, and
# its paper provenance, outside the inventory entirely.
FACADE_VALUES = {"full", "thin", "none"}
BASE_REQUIRED = ("id", "status")

# What kind of proposition the row's evidence actually demonstrates. Required on
# every `done` row (release.toml exempt — its rows are release infrastructure
# with their own evidence map in gate_release.sh, not scientific capabilities).
# The 2026-08 audit found rows whose notes read as external-package agreement
# while the fixture's own oracle block recorded clean-room enumeration; this
# field makes the claim strength machine-readable so wording cannot outrun it.
EVIDENCE_KINDS = {
    # code + ordinary unit tests; no numerical evidence against a truth
    "implementation_exists",
    # conformance fixture/test against closed-form, analytic, or clean-room truth
    "internal_known_truth",
    # agreement with another Antecedent estimator/module only
    "internal_cross_check",
    # frozen fixture recording an actual pinned upstream-package run, consumed
    # by an executing test
    "frozen_external_oracle",
    # agreement with an upstream package across a range of inputs
    "behavioral_parity",
    # theorem-level / method-contract argument
    "contract_equivalence",
}
# Kinds that assert an upstream package produced the truth being matched.
EXTERNAL_KINDS = {"frozen_external_oracle", "behavioral_parity"}

# An external claim is a three-link evidence contract, not just prose on an
# inventory row: immutable baseline metadata, a frozen JSON fixture, and an
# executing test that consumes that fixture.  Keep this stricter than the
# repository-wide reachability scan, which also accepts package code and gate
# scripts because it answers the broader question "is this artifact used?".
baseline_versions = {}
for path in sorted(root.glob("parity/baselines/*.toml")):
    baseline = tomllib.load(open(path, "rb"))
    project = str(baseline.get("project", "")).lower()
    if not project:
        continue
    versions = {
        str(value)
        for key, value in baseline.items()
        if key == "version" or key.endswith("_version")
    }
    baseline_versions.setdefault(project, set()).update(versions)

test_sources = set(root.glob("crates/**/tests/**/*.rs"))
test_sources.update(root.glob("python/tests/**/*.py"))
# Rust unit/conformance tests commonly live next to the implementation.  They
# count only when the fixture reference occurs below a cfg(test) marker.
rust_src = set(root.glob("crates/**/src/**/*.rs"))
PARSE_MARKERS = re.compile(
    r"serde_json::from_str|serde_json::Value|from_str::<|json\.loads|json\.load\(|"
    r"tomllib\.loads|tomllib\.load\(|load_expected"
)
ASSERT_MARKERS = re.compile(r"assert(?:_eq|_ne)?!|\bassert\s|pytest\.approx|approx::")


def consuming_test_file(text: str) -> bool:
    # Fixture-loader helpers commonly live at the top of a long conformance
    # test file while comparisons appear in several tests below. Requiring all
    # three signals in that same test source avoids accepting a prose mention
    # or bare existence check without imposing a brittle line-distance rule.
    return bool(PARSE_MARKERS.search(text) and ASSERT_MARKERS.search(text))


def has_consuming_test(fixture: str) -> bool:
    name = Path(fixture).name
    marker = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")
    for path in test_sources:
        text = path.read_text(errors="ignore")
        if marker.search(text) and consuming_test_file(text):
            return True
    for path in rust_src:
        text = path.read_text(errors="ignore")
        hits = list(marker.finditer(text))
        if path.name == "tests.rs" and hits and consuming_test_file(text):
            return True
        for hit in hits:
            if "#[cfg(test)]" in text[: hit.start()] and consuming_test_file(text):
                return True
    return False

# Inventory manifests and the extra keys each one requires beyond BASE_REQUIRED.
# Kept explicit rather than inferred from whichever keys the majority of rows
# happen to carry: the richer schema is what exposed the row that motivated this
# gate, so it has to be a stated contract, not a statistical accident.
# Second tuple element: whether `done` rows must carry `evidence_kind`.
# Only release.toml is exempt (infrastructure rows, evidence map in gate_release.sh).
MANIFESTS = {
    "parity/estimate.toml": (("group", "description", "owner"), True),
    "parity/discovery.toml": (("group", "description", "owner"), True),
    "parity/context.toml": (("group", "description", "owner"), True),
    "parity/bayesian.toml": ((), True),
    "parity/pag.toml": ((), True),
    "parity/gcm.toml": ((), True),
    "parity/attribution.toml": ((), True),
    "parity/design_state.toml": ((), True),
    "parity/release.toml": ((), False),
    "parity/response.toml": ((), True),
}

# The parser every feature gate embeds. Reproduced verbatim so this gate checks
# what those gates actually see, not an idealized reading of the file.
def regex_caps(text: str):
    blocks = re.split(r"\n\[\[capabilities\]\]\n", text)[1:]
    out = []
    for b in blocks:
        def g(k, default=None):
            m = re.search(rf'^{k}\s*=\s*"([^"]*)"', b, re.M)
            if m:
                return m.group(1)
            m = re.search(rf'^{k}\s*=\s*(\d+)', b, re.M)
            return m.group(1) if m else default
        out.append({"id": g("id"), "status": g("status")})
    return out


problems = []

# Any manifest carrying capability rows must be enrolled here; a new inventory
# cannot join the repo and skip the schema contract.
for path in sorted(root.glob("parity/*.toml")):
    rel = path.as_posix()
    if rel in MANIFESTS:
        continue
    if re.search(r"^\[\[capabilities\]\]", path.read_text(), re.M):
        problems.append(f"{rel}: has capability rows but is not enrolled in gate_parity_schema.sh")

for rel, (extra_required, requires_evidence) in MANIFESTS.items():
    path = root / rel
    if not path.exists():
        problems.append(f"{rel}: manifest missing")
        continue
    text = path.read_text()

    try:
        rows = tomllib.loads(text).get("capabilities", [])
    except tomllib.TOMLDecodeError as exc:
        problems.append(f"{rel}: not valid TOML: {exc}")
        continue

    if not rows:
        problems.append(f"{rel}: no [[capabilities]] rows")
        continue

    required = BASE_REQUIRED + tuple(extra_required)
    seen_ids = set()

    for i, row in enumerate(rows, 1):
        cid = row.get("id")
        label = cid if isinstance(cid, str) and cid.strip() else f"row #{i}"

        for key in required:
            if key not in row:
                problems.append(f"{rel}: {label} missing required key `{key}`")
            elif not isinstance(row[key], str) or not row[key].strip():
                problems.append(f"{rel}: {label} has empty/non-string `{key}`")

        status = row.get("status")
        if isinstance(status, str) and status.strip() and status not in STATUS_VALUES:
            problems.append(
                f"{rel}: {label} status={status!r} not in {sorted(STATUS_VALUES)}"
            )

        facade = row.get("python_facade")
        if facade is not None and facade not in FACADE_VALUES:
            problems.append(
                f"{rel}: {label} python_facade={facade!r} not in {sorted(FACADE_VALUES)}"
            )

        # ------------------------------------------- evidence-kind contract
        kind = row.get("evidence_kind")
        if kind is not None and kind not in EVIDENCE_KINDS:
            problems.append(
                f"{rel}: {label} evidence_kind={kind!r} not in {sorted(EVIDENCE_KINDS)}"
            )
        if requires_evidence and status == "done" and kind is None:
            problems.append(
                f"{rel}: {label} is done without evidence_kind — state what the "
                "evidence demonstrates (implementation_exists is a legal answer; "
                "an implied one is not)"
            )

        if kind == "internal_cross_check":
            limitations = row.get("limitations")
            if not isinstance(limitations, str) or not limitations.strip():
                problems.append(
                    f"{rel}: {label} is an internal_cross_check without limitations; "
                    "state explicitly that agreement between Antecedent paths is not "
                    "independent truth evidence"
                )

        oracle = row.get("external_oracle")
        fixture = row.get("known_truth_fixture")

        if fixture is not None:
            if not isinstance(fixture, str) or not (root / fixture).exists():
                problems.append(
                    f"{rel}: {label} known_truth_fixture {fixture!r} does not exist"
                )

        if kind in EXTERNAL_KINDS:
            if not isinstance(oracle, str) or not oracle.strip():
                problems.append(
                    f"{rel}: {label} claims {kind} without external_oracle "
                    "(project + pin)"
                )
            elif not isinstance(fixture, str):
                problems.append(
                    f"{rel}: {label} claims {kind} without known_truth_fixture "
                    "pointing at the frozen fixture"
                )
            else:
                oracle_parts = oracle.split(maxsplit=1)
                oracle_project = oracle_parts[0].lower()
                oracle_version = oracle_parts[1] if len(oracle_parts) == 2 else ""
                pins = baseline_versions.get(oracle_project)
                if not pins:
                    problems.append(
                        f"{rel}: {label} names external oracle {oracle!r} without "
                        f"parity/baselines metadata for {oracle_project!r}"
                    )
                elif oracle_version not in pins:
                    problems.append(
                        f"{rel}: {label} external oracle version {oracle_version!r} "
                        f"is not pinned by parity/baselines ({sorted(pins)})"
                    )

                fixture_path = root / fixture
                expected = fixture_path if fixture_path.is_file() else fixture_path / "expected.json"
                if not expected.is_file():
                    problems.append(
                        f"{rel}: {label} external fixture {fixture!r} has no expected.json"
                    )
                else:
                    try:
                        json.loads(expected.read_text())
                    except json.JSONDecodeError as exc:
                        problems.append(
                            f"{rel}: {label} external fixture {expected} is not valid JSON: {exc}"
                        )
                if not has_consuming_test(fixture):
                    problems.append(
                        f"{rel}: {label} external fixture {fixture!r} is not parsed and "
                        "compared by an executing Rust/Python conformance test"
                    )

                # Fixture-authoritative rule: the named project must actually
                # appear in the frozen fixture. The audit found ledger rows
                # claiming scikit-learn/pcalg/causaleffect against fixtures
                # whose oracle blocks recorded clean-room computation.
                fdir = root / fixture
                files = [fdir] if fdir.is_file() else sorted(fdir.glob("*"))
                blob = "\n".join(
                    f.read_text(errors="ignore").lower() for f in files if f.is_file()
                )
                token = oracle.split()[0].lower()
                if token not in blob:
                    problems.append(
                        f"{rel}: {label} names external oracle {oracle!r} but "
                        f"{fixture} never records {token!r} — the fixture's own "
                        "oracle block is authoritative"
                    )
        elif oracle is not None:
            problems.append(
                f"{rel}: {label} carries external_oracle but evidence_kind="
                f"{kind!r} does not claim an external comparison — drop one"
            )

        if isinstance(cid, str) and cid.strip():
            if cid in seen_ids:
                problems.append(f"{rel}: duplicate id `{cid}`")
            seen_ids.add(cid)

    # The gates' regex parser must recover the same rows the TOML parser sees.
    header_count = len(re.findall(r"^\[\[capabilities\]\]", text, re.M))
    scanned = regex_caps(text)
    if not (len(rows) == header_count == len(scanned)):
        problems.append(
            f"{rel}: row-count disagreement -- toml={len(rows)} "
            f"headers={header_count} gate-regex={len(scanned)}"
        )
    else:
        for row, seen in zip(rows, scanned):
            for key in ("id", "status"):
                if row.get(key) != seen[key]:
                    problems.append(
                        f"{rel}: gate regex reads {key}={seen[key]!r} for "
                        f"{row.get('id')!r} but TOML has {row.get(key)!r}"
                    )

if problems:
    print("parity manifest schema violations:")
    for p in problems:
        print(" -", p)
    sys.exit(1)

total = sum(
    len(tomllib.loads((root / rel).read_text())["capabilities"]) for rel in MANIFESTS
)
print(f"parity manifest schema: ok ({total} rows across {len(MANIFESTS)} manifests)")
PY
