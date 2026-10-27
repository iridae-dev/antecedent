#!/usr/bin/env python3
import json
import re
import sys
import tomllib
from pathlib import Path

from _repo import ensure_scripts_on_path

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
    # frozen output of this library itself: a change detector, never truth (no
    # known_truth_fixture; limitations name what is pinned)
    "regression_pin",
    # a release-process property enforced by a named gate or workflow, with no
    # test that could execute it (release.toml only; see the done-row rule below)
    "process_attested",
}
# An implementation_exists row that produces one of these is a statistical procedure
# a reader would take as validated; it must say that no numerical truth backs it.
# A term matches at a word start or inside a CamelCase name (`PosteriorWidth`,
# `ExpectedPosteriorWidth`); `SE` is matched as a whole word only.
STATISTICAL_QUANTITY = re.compile(
    r"(?:\b|(?-i:(?<=[a-z])(?=[A-Z])))"
    r"(interval|band|posterior|credible|p-value|EVPI|EVSI|decision|utility|coverage"
    r"|standard[ _-]errors?|bootstrap|prior|draws?|envelope|cluster(?:ed)?|probability)"
    r"|\bSEs?\b",
    re.I,
)
# Kinds that assert an upstream package produced the truth being matched.
EXTERNAL_KINDS = {"frozen_external_oracle", "behavioral_parity"}

# An external claim is a three-link evidence contract, not just prose on an
# inventory row: immutable baseline metadata, a frozen JSON fixture, and an
# executing test that consumes that fixture (scripts/test_evidence.py, the same
# reader gate_evidence_reachability.sh uses).
baseline_versions = {}
for path in sorted(root.glob("parity/baselines/*.toml")):
    with open(path, "rb") as handle:
        baseline = tomllib.load(handle)
    project = str(baseline.get("project", "")).lower()
    if not project:
        continue
    versions = {
        str(value)
        for key, value in baseline.items()
        if key == "version" or key.endswith("_version")
    }
    baseline_versions.setdefault(project, set()).update(versions)

ensure_scripts_on_path()
import test_evidence  # noqa: E402  (the one reader of test source for the gates)


def has_consuming_test(fixture: str) -> bool:
    """An executing test whose own function (with the helpers it calls) names the
    fixture's `conformance/<category>/<name>` path outside comments, parses it and
    asserts: not a file that merely mentions the basename and asserts elsewhere."""
    path = Path(fixture)
    directory = path.parent if path.suffix == ".json" else path
    return bool(test_evidence.fixture_consumers(directory.as_posix()))


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
    "parity/compiler.toml": (("group", "description", "owner"), True),
}

problems = []

# Any manifest carrying capability rows must be enrolled here; a new inventory
# cannot join the repo and skip the schema contract.
for path in sorted(root.glob("parity/*.toml")):
    rel = path.as_posix()
    if rel in MANIFESTS:
        continue
    if re.search(r"^\[\[capabilities\]\]", path.read_text(encoding="utf-8"), re.M):
        problems.append(f"{rel}: has capability rows but is not enrolled in gate_parity_schema.sh")

for rel, (extra_required, requires_evidence) in MANIFESTS.items():
    path = root / rel
    if not path.exists():
        problems.append(f"{rel}: manifest missing")
        continue
    text = path.read_text(encoding="utf-8")

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
            problems.append(f"{rel}: {label} status={status!r} not in {sorted(STATUS_VALUES)}")

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

        if kind == "regression_pin":
            if row.get("known_truth_fixture") is not None:
                problems.append(
                    f"{rel}: {label} is a regression_pin but names known_truth_fixture; a "
                    "frozen output of this library is not truth"
                )
            if not str(row.get("limitations", "")).strip():
                problems.append(f"{rel}: {label} is a regression_pin without limitations")
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

        if kind == "implementation_exists":
            # A fixture that only checks finiteness is a smoke fixture, never known truth.
            if fixture is not None:
                problems.append(
                    f"{rel}: {label} is implementation_exists but names known_truth_fixture; "
                    "name it smoke_fixture (no numerical truth) or claim a stronger kind"
                )
            smoke = row.get("smoke_fixture")
            if smoke is not None and (not isinstance(smoke, str) or not (root / smoke).exists()):
                problems.append(f"{rel}: {label} smoke_fixture {smoke!r} does not exist")
            described = " ".join(str(row.get(key, "")) for key in ("description", "notes"))
            if (
                status == "done"
                and STATISTICAL_QUANTITY.search(described)
                and not str(row.get("limitations", "")).strip()
            ):
                problems.append(
                    f"{rel}: {label} is a done implementation_exists row naming an interval, "
                    "test, posterior or decision quantity without limitations; state that no "
                    "numerical truth backs it"
                )
        elif row.get("smoke_fixture") is not None:
            problems.append(f"{rel}: {label} smoke_fixture is only legal on implementation_exists")

        if fixture is not None and (not isinstance(fixture, str) or not (root / fixture).exists()):
            problems.append(f"{rel}: {label} known_truth_fixture {fixture!r} does not exist")

        if kind in EXTERNAL_KINDS:
            if not isinstance(oracle, str) or not oracle.strip():
                problems.append(
                    f"{rel}: {label} claims {kind} without external_oracle (project + pin)"
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
                expected = (
                    fixture_path if fixture_path.is_file() else fixture_path / "expected.json"
                )
                if not expected.is_file():
                    problems.append(
                        f"{rel}: {label} external fixture {fixture!r} has no expected.json"
                    )
                else:
                    try:
                        json.loads(expected.read_text(encoding="utf-8"))
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
                    f.read_text(errors="ignore", encoding="utf-8").lower()
                    for f in files
                    if f.is_file()
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

        test_rel = row.get("evidence_test")
        assertion = row.get("evidence_assertion")
        # Every done row names the executing test that evidences it. The one exemption
        # is a release-process fact no test can execute: it says so (`process_attested`)
        # and names the gate or workflow that enforces it.
        if kind == "process_attested":
            if requires_evidence:
                problems.append(
                    f"{rel}: {label} process_attested is only for release-process rows "
                    "(parity/release.toml); a capability names an executing test"
                )
            elif not isinstance(row.get("notes"), str) or not re.search(
                r"(scripts/[\w./-]+|\.github/workflows/[\w./-]+)", row["notes"]
            ):
                problems.append(
                    f"{rel}: {label} is process_attested but its notes name no "
                    "scripts/... gate or .github/workflows/... file that enforces it"
                )
        elif status == "done" and (test_rel is None or assertion is None):
            problems.append(
                f"{rel}: {label} is done without evidence_test/evidence_assertion: name "
                "the executing test that evidences it"
            )
        if (test_rel is None) != (assertion is None):
            problems.append(
                f"{rel}: {label} must set evidence_test and evidence_assertion together"
            )
        elif isinstance(test_rel, str) and isinstance(assertion, str):
            test_path = root / test_rel
            if not test_path.is_file():
                problems.append(f"{rel}: {label} evidence_test {test_rel!r} does not exist")
            elif test_path.suffix not in {".rs", ".py"}:
                problems.append(f"{rel}: {label} evidence_test must be Rust or Python test code")
            else:
                # The one static resolver (comments/strings masked, #[ignore],
                # cfg-disabled and uncompiled modules rejected).
                if test_path.suffix == ".rs":
                    why = test_evidence.static_rust_test(test_path, assertion).problems
                else:
                    why = test_evidence.static_python_test(test_path, assertion)
                for reason in why:
                    problems.append(
                        f"{rel}: {label} evidence_assertion {assertion!r} in "
                        f"{test_rel} is not an executing test: {reason}"
                    )

        if isinstance(cid, str) and cid.strip():
            if cid in seen_ids:
                problems.append(f"{rel}: duplicate id `{cid}`")
            seen_ids.add(cid)

# Identity dictionary and producer/consumer tables are rendered from TOML.
# The marker blocks are the pages; this does not scrape hand-written prose.
import generate_dictionary_tables  # noqa: E402

problems.extend(generate_dictionary_tables.staleness())

if problems:
    print("parity manifest schema violations:")
    for p in problems:
        print(" -", p)
    sys.exit(1)

total = sum(
    len(tomllib.loads((root / rel).read_text(encoding="utf-8"))["capabilities"])
    for rel in MANIFESTS
)
print(f"parity manifest schema: ok ({total} rows across {len(MANIFESTS)} manifests)")
