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

# --self-test: each broken ledger, applied alone to an overlay of the repo,
# must fail this gate with the expected message (scripts/selftest_cases.py).
if [[ "${1:-}" == "--self-test" ]]; then
  exec python3 "$ROOT/scripts/selftest_cases.py" schema
fi

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
    "parity/compiler.toml": (("group", "description", "owner"), True),
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

        test_rel = row.get("evidence_test")
        assertion = row.get("evidence_assertion")
        if (test_rel is None) != (assertion is None):
            problems.append(
                f"{rel}: {label} must set evidence_test and evidence_assertion together"
            )
        elif isinstance(test_rel, str) and isinstance(assertion, str):
            test_path = root / test_rel
            if not test_path.is_file():
                problems.append(
                    f"{rel}: {label} evidence_test {test_rel!r} does not exist"
                )
            elif test_path.suffix not in {".rs", ".py"}:
                problems.append(
                    f"{rel}: {label} evidence_test must be Rust or Python test code"
                )
            else:
                text_src = test_path.read_text(errors="ignore")
                if test_path.suffix == ".rs":
                    test_pattern = re.compile(
                        rf"#\[test\][^\n]*\n((?:\s*#\[[^\n]*\n)*)\s*fn\s+{re.escape(assertion)}\s*\(",
                        re.M,
                    )
                else:
                    test_pattern = re.compile(
                        rf"()^\s*def\s+{re.escape(assertion)}\s*\(", re.M
                    )
                match = test_pattern.search(text_src)
                if not match:
                    problems.append(
                        f"{rel}: {label} evidence_assertion {assertion!r} is not "
                        f"an executing test function in {test_rel}"
                    )
                elif re.search(r"#\[\s*ignore\b", match.group(1)):
                    problems.append(
                        f"{rel}: {label} evidence_assertion {assertion!r} is #[ignore]d"
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

python3 - "$@" <<'PY'
from pathlib import Path
import json
import re
import subprocess
import sys
import tomllib

root = Path(".")
print_counts = "--print-counts" in sys.argv
problems = []

# --- reason codes ---
vocab_path = root / "parity/reason_codes.toml"
if not vocab_path.is_file():
    problems.append("parity/reason_codes.toml missing")
    vocab = {"code": []}
else:
    vocab = tomllib.loads(vocab_path.read_text())
codes = {row["id"]: row for row in vocab.get("code", [])}
uses = {cid: 0 for cid in codes}
applies_fields = ("reason", "calibration_reason", "reason_code")
# support_{closed,n_a,axes} `reason` fields are matrix prose, not vocabulary ids.
# Do not re-code those pre-existing typed refusals.
_REASON_PROSE = {
    "reason_codes.toml",
    "support_closed.toml",
    "support_n_a.toml",
    "support_axes.toml",
}
for path in sorted(root.glob("parity/*.toml")):
    if path.name in _REASON_PROSE:
        continue
    text = path.read_text()
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        problems.append(f"{path}: {exc}")
        continue
    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in applies_fields and isinstance(value, str) and value.strip():
                    cid = value.strip()
                    if cid not in codes:
                        problems.append(f"{path}: unknown reason {cid!r}")
                    else:
                        uses[cid] += 1
                        obligation = {
                            "calibration_reason": "calibration",
                            "reason_code": "runtime_refusal",
                            "reason": "python_product" if path.name == "python_products.toml" else "claim",
                        }[key]
                        if path.name == "support_licensed.toml" and key == "calibration_reason":
                            obligation = "calibration"
                        if obligation not in codes[cid].get("applies_to", []):
                            problems.append(
                                f"{path}: {cid} does not apply to {obligation}"
                            )
                else:
                    walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
    walk(data)

if print_counts:
    print("reason-code uses:")
    for cid, n in uses.items():
        print(f"  {cid}: {n} (max_uses={codes[cid]['max_uses']})")

for cid, n in uses.items():
    max_uses = int(codes[cid]["max_uses"])
    if n > max_uses:
        problems.append(f"parity/reason_codes.toml: {cid} uses={n} > max_uses={max_uses}")

# --- python_products ---
pp = root / "parity/python_products.toml"
if not pp.is_file():
    problems.append("parity/python_products.toml missing")
else:
    products = tomllib.loads(pp.read_text())
    route_tests = []
    for i, row in enumerate(products.get("route", []), 1):
        for key in ("kind", "data", "structure", "test"):
            if key not in row:
                problems.append(f"python_products.toml route #{i} missing {key}")
        if row.get("retains") is not True and not row.get("reason"):
            problems.append(f"python_products.toml route {row.get('kind')} retains=false without reason")
        test = row.get("test")
        if isinstance(test, str) and test not in route_tests:
            route_tests.append(test)

    def _collects(tests):
        return subprocess.run(
            ["uv", "run", "pytest", "--collect-only", "-q", *(t.removeprefix("python/") for t in tests)],
            cwd=root / "python",
            capture_output=True,
            text=True,
        ).returncode == 0

    # One collection for every route node; only on failure, one per node to
    # name the uncollectable ones.
    if route_tests and not _collects(route_tests):
        for test in route_tests:
            if not _collects([test]):
                problems.append(f"python_products.toml route test not collected: {test}")
    for i, row in enumerate(products.get("parameter", []), 1):
        for key in ("name", "entry_points", "binding", "test"):
            if key not in row:
                problems.append(f"python_products.toml parameter #{i} missing {key}")
        if row.get("binding") == "contract" and not row.get("contract_key"):
            problems.append(f"python_products.toml parameter {row.get('name')} missing contract_key")
        if row.get("binding") == "reason" and not row.get("reason"):
            problems.append(f"python_products.toml parameter {row.get('name')} missing reason")

# --- coverage records ---
#
# Every row must be a real measurement of a construction the runtime reports:
# emitted by a coverage test through `CoverageTally::for_record`, collected by
# scripts/collect_coverage_records.py, stamped with the commit it was measured
# at. The checks below are what a hand-written row cannot pass.
sys.path.insert(0, str(root / "scripts"))
import collect_coverage_records as collector  # noqa: E402


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _sanitize(label: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in label).strip("_")


def _resolves(spec: str) -> bool:
    """Does `<file>::<item>` name a function that file defines?

    A calibration suite may declare a grid of tests through a `macro_rules!`
    table, where the test name is an argument rather than a `fn` header. Such a
    file is allowed to resolve a name that appears in it as a whole word; a file
    with no macro table must spell the `fn` out.
    """
    if "::" not in spec:
        return False
    rel, fn = spec.rsplit("::", 1)
    path = root / rel
    if not path.is_file():
        return False
    text = path.read_text(errors="ignore")
    if re.search(rf"fn\s+{re.escape(fn)}\s*[(<]", text):
        return True
    return "macro_rules!" in text and bool(re.search(rf"\b{re.escape(fn)}\b", text))


def _band(nominal: float, replicates: int) -> tuple[float, float, float | None]:
    mcse = (nominal * (1.0 - nominal) / max(replicates, 1)) ** 0.5
    floor = nominal - 2.0 * mcse if replicates >= 1000 else None
    return (max(nominal - 3.0 * mcse, 0.0), min(nominal + 3.0 * mcse, 1.0), floor)


cr = root / "parity/coverage_records.toml"
record_ids = set()
records = []
if cr.is_file():
    records = tomllib.loads(cr.read_text()).get("record", [])
    for rec in records:
        rid = rec.get("id", "")
        record_ids.add(rid)
        label = f"coverage_records.toml {rid}"
        test = str(rec.get("test", ""))
        missing = [key for key in collector.FIELDS if key not in rec]
        if missing:
            problems.append(f"{label}: missing {', '.join(missing)}")
            continue
        test_fn = test.rsplit("::", 1)[-1]
        expected = (
            f"cov.{_snake(rec['query'])}.{_snake(rec['graph_class'])}."
            f"{str(rec['inference']).lower()}.{rec['interval_method']}."
            f"l{round(float(rec['nominal']) * 100)}.{test_fn}"
        )
        if rid != expected and not rid.startswith(expected + "."):
            problems.append(f"{label}: id does not match its fields ({expected}[.<label>])")
        if rid.startswith(expected + ".") and _sanitize(rid[len(expected) + 1 :]) != rid[len(expected) + 1 :]:
            problems.append(f"{label}: record label is not sanitized")
        sha = str(rec["calibration_sha"])
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            problems.append(f"{label}: calibration_sha must be 40 lowercase hex")
        elif set(sha) == {"0"}:
            problems.append(f"{label}: calibration_sha is the zero SHA; nothing was measured")
        for field in ("observed", "mcse", "nominal", "unidentified_mass_max"):
            val = rec[field]
            if not isinstance(val, (int, float)) or not 0 <= float(val) <= 1:
                problems.append(f"{label}: {field} not in [0,1]")
        if int(rec["replicates"]) < 1:
            problems.append(f"{label}: replicates must be positive")
        if int(rec["n_min"]) < 1 or int(rec["n_max"]) < int(rec["n_min"]):
            problems.append(f"{label}: measured row-count range is empty")
        if not _resolves(test):
            problems.append(f"{label}: test {test} does not resolve")
        if not _resolves(str(rec["dgp"])):
            problems.append(f"{label}: dgp {rec['dgp']} does not resolve")
        role = str(rec["role"])
        boundary = bool(rec["boundary"])
        nominal = float(rec["nominal"])
        lo, hi, floor = _band(nominal, int(rec["replicates"]))
        observed = float(rec["observed"])
        nominal_pass = lo <= observed <= hi and (floor is None or observed >= floor)
        if role == "gated":
            if boundary:
                problems.append(f"{label}: a gated record cannot be a boundary")
            if not nominal_pass:
                problems.append(
                    f"{label}: gated coverage {observed} is outside the {nominal} band"
                )
            named = re.search(r"nominal_(\d+)_coverage", test_fn)
            if named and abs(int(named.group(1)) / 100 - nominal) > 1e-9:
                problems.append(
                    f"{label}: nominal {nominal} contradicts the test name {test_fn}"
                )
        elif role == "named_boundary":
            if not boundary:
                problems.append(f"{label}: a named boundary must be boundary = true")
            if "boundary" not in test_fn:
                problems.append(
                    f"{label}: role named_boundary on {test_fn}, whose name claims no boundary"
                )
        elif role == "reported_level":
            if boundary == nominal_pass:
                problems.append(
                    f"{label}: boundary must be true exactly when the measured coverage "
                    f"{observed} misses the {nominal} band"
                )
        else:
            problems.append(f"{label}: unknown role {role}")
else:
    problems.append("parity/coverage_records.toml missing")

if len(record_ids) != len(records):
    problems.append("parity/coverage_records.toml: duplicate record ids")

by_id = {rec.get("id"): rec for rec in records}

# --- licensed cell calibration obligation ---
lic = tomllib.loads((root / "parity/support_licensed.toml").read_text()).get("cell", [])
by_coordinate = {}
for rid, rec in by_id.items():
    key = (rec["query"], rec["graph_class"], rec["inference"], rec["structure"])
    by_coordinate.setdefault(key, []).append(rid)
for cell in lic:
    label = f"{cell.get('query')}/{cell.get('graph_class')}/{cell.get('inference')}"
    has_cal = "calibration" in cell
    has_reason = bool(cell.get("calibration_reason"))
    if has_cal == has_reason:
        problems.append(f"support_licensed.toml {label}: exactly one of calibration / calibration_reason")
    structure = "graph_posterior" if cell.get("structure") == "graph_posterior" else "fixed"
    expected_ids = sorted(
        by_coordinate.get(
            (cell.get("query"), cell.get("graph_class"), cell.get("inference"), structure), []
        )
    )
    if expected_ids:
        if sorted(cell.get("calibration") or []) != expected_ids:
            problems.append(
                f"support_licensed.toml {label}: calibration is not the records measured for this "
                f"coordinate; re-run scripts/collect_coverage_records.py"
            )
    elif has_cal:
        problems.append(f"support_licensed.toml {label}: cites records that measure another coordinate")
    for rid in cell.get("calibration") or []:
        if rid not in record_ids:
            problems.append(f"support_licensed.toml {label}: unknown record {rid}")
    lim = str(cell.get("limitations", ""))
    if re.search(r"0\.\d{3}", lim):
        problems.append(f"support_licensed.toml {label}: limitations still contain a coverage figure")
    # ---- [gates] reason-code eligibility comes from the registry row ----
    cited = cell.get("calibration_reason")
    scope = codes.get(cited, {}).get("queries") if isinstance(cited, str) else None
    if scope is not None and cell.get("query") not in scope:
        problems.append(
            f"support_licensed.toml {label}: {cited} on a {cell.get('query')} cell; "
            f"reason_codes.toml limits it to {scope}"
        )

# ---- [gates] registry scopes name real queries ----
axes_queries = set(tomllib.loads((root / "parity/support_axes.toml").read_text()).get("queries", []))
for cid, code in codes.items():
    for query in code.get("queries") or []:
        if query not in axes_queries:
            problems.append(f"reason_codes.toml {cid}: queries names unknown query {query!r}")

# ---- [gates] estimator calibration obligation, keyed on group ----
# Every `group = "estimation"` row states its calibration; calibration fields on
# any other row would be an obligation no gate enforces.
est = tomllib.loads((root / "parity/estimate.toml").read_text()).get("capabilities", [])
for row in est:
    has_cal = "calibration" in row
    has_reason = "calibration_reason" in row
    if row.get("group") == "estimation":
        if has_cal == bool(row.get("calibration_reason")):
            problems.append(f"estimate.toml {row.get('id')}: exactly one of calibration / calibration_reason")
    elif has_cal or has_reason:
        problems.append(
            f"estimate.toml {row.get('id')}: calibration fields only on group = \"estimation\" rows"
        )
    row_id = row.get("id")
    if row_id not in collector.ESTIMATOR_ROW_IDS:
        continue
    expected_ids = sorted(
        rid for rid, rec in by_id.items() if rec["estimator"] in collector.ESTIMATOR_ROW_IDS[row_id]
    )
    if sorted(row.get("calibration") or []) != expected_ids and expected_ids:
        problems.append(
            f"estimate.toml {row_id}: calibration is not the records measured for its estimators; "
            f"re-run scripts/collect_coverage_records.py"
        )
    for rid in row.get("calibration") or []:
        if rid not in record_ids:
            problems.append(f"estimate.toml {row_id}: unknown record {rid}")

# ---- [gates] required composition rows ----
# Deleting a composition row must fail here, not silently shrink what
# gate_composition.sh executes. Additions are free; removals are a reviewed
# edit of this list.
REQUIRED_COMPOSITION_ROWS = {
    "compiler.inspect_licensed_cells", "compiler.e2e_licensed_cells",
    "compiler.inspect_does_not_identify", "compiler.metadata_read_no_identify",
    "compiler.validate_result_bindings", "compiler.claim_handoff",
    "compiler.request_lifecycle", "compiler.discovery_class",
    "compiler.prior_transfer_composition", "compiler.prepared_matches_fresh",
    "compiler.adversarial_boundaries", "compiler.panel_response_cluster_bands",
    "compiler.panel_class_pulse_completion_masses",
    "compiler.panel_response_bayesian_unit_surfaces", "compiler.panel_class_pulse_bayesian",
    "compiler.panel_class_response_surfaces", "compiler.panel_class_multi_step_sustained",
    "compiler.contract_section", "compiler.request_identity", "compiler.provenance_ancestry",
    "compiler.v110_contract_file", "compiler.licensed_family", "compiler.prepared_second_shot",
    "compiler.prepared_family", "compiler.identity_encoding", "compiler.dbn_atom_identity",
    "compiler.score_reuse_identity", "compiler.state_expected_version",
    "compiler.series_refresh_atomicity", "compiler.series_state_reuse",
    "compiler.score_batch_reuse", "compiler.capability_reports", "compiler.support_neighbors",
    "compiler.capability_prefix", "compiler.claim_core", "compiler.design_rank",
    "compiler.composition_prefix", "compiler.python_v110_smoke",
    "compiler.python_stub_conformance",
}
compiler_rows = {
    row.get("id"): row
    for row in tomllib.loads((root / "parity/compiler.toml").read_text()).get("capabilities", [])
}
for rid in sorted(REQUIRED_COMPOSITION_ROWS - compiler_rows.keys()):
    problems.append(f"compiler.toml: required composition row {rid} missing")
for rid, row in compiler_rows.items():
    if not row.get("evidence_test") or not row.get("evidence_assertion"):
        problems.append(f"compiler.toml {rid}: composition row without evidence_test/evidence_assertion")

# ---- [gates] runtime reason-code list matches the registry ----
# crates/antecedent-core/src/reason_codes_data.rs is generated from
# parity/reason_codes.toml; the Rust `unsupported_reason!` / `reason_code!`
# macros and Python CausalUnsupportedError validate against it.
data_rs = root / "crates/antecedent-core/src/reason_codes_data.rs"
if not data_rs.is_file():
    problems.append(f"{data_rs}: missing; run scripts/generate_support_matrix_docs.py")
else:
    generated = data_rs.read_text()
    def _const(name: str) -> list[str]:
        m = re.search(rf"pub const {name}: &\[&str\] = &\[(.*?)\];", generated, re.S)
        return re.findall(r'"([^"]+)"', m.group(1)) if m else []
    want_all = sorted(codes)
    want_runtime = sorted(c for c, row in codes.items() if "runtime_refusal" in row.get("applies_to", []))
    if _const("REASON_CODES") != want_all or _const("RUNTIME_REFUSAL_CODES") != want_runtime:
        problems.append(
            "crates/antecedent-core/src/reason_codes_data.rs is stale against "
            "parity/reason_codes.toml; run scripts/generate_support_matrix_docs.py"
        )
# A refusal message that carries a reason code must be built by the checked
# macro, never typed as a raw `"reason=` literal (comments / doc examples and
# `#[cfg(test)]` code below the first test marker are not emission sites).
for path in sorted(root.glob("crates/*/src/**/*.rs")) + sorted(root.glob("python/src/**/*.rs")):
    source = path.read_text(errors="ignore").split("#[cfg(test)]", 1)[0]
    for lineno, line in enumerate(source.splitlines(), 1):
        if line.lstrip().startswith("//"):
            continue
        if '"reason=' in line and "concat!" not in line:
            problems.append(f"{path}:{lineno}: raw \"reason=...\" literal; use antecedent::unsupported_reason!")

# --- release required_jobs ---
# ---- [gates] job ids from a YAML parse of ci.yml ----
# scripts/ci_workflow.py is the one reader of both ci.yml and release.toml's
# `required_jobs`; gate_release_candidate.sh calls the same subcommand.
listed = subprocess.run(
    ["uv", "run", "--quiet", "--project", ".", "--only-group", "dev", "python",
     str((root / "scripts/ci_workflow.py").resolve()), "required-jobs", "--json",
     "--workflow", str((root / ".github/workflows/ci.yml").resolve()),
     "--release", str((root / "parity/release.toml").resolve())],
    cwd=root / "python", capture_output=True, text=True,
)
if listed.returncode != 0:
    problems.append(f"scripts/ci_workflow.py required-jobs failed: {listed.stdout}{listed.stderr}")
else:
    problems.extend(json.loads(listed.stdout)["problems"])

# --- identity / claims files exist ---
for rel_path in ("parity/identity.toml", "parity/claims.toml"):
    if not (root / rel_path).is_file():
        problems.append(f"{rel_path} missing")

# --- every identity row names its layer, its bindings, and what it covers ---
identity_rows = tomllib.loads((root / "parity/identity.toml").read_text()).get("identity", [])
adr = (root / "adr/0022-causal-compiler-contract.md").read_text()
for row in identity_rows:
    for field in ("domain", "rust", "contract_key", "python", "availability", "naming_row", "covers"):
        if not row.get(field):
            problems.append(f"identity.toml {row.get('domain')}: missing {field}")
    if f"| {row.get('naming_row')} |" not in (root / "docs/api_naming.md").read_text():
        problems.append(f"identity.toml {row.get('domain')}: naming_row missing from api_naming.md")
if "antecedent.identity.v2" not in adr:
    problems.append("adr/0022: identity format tag does not match antecedent-core")

if problems:
    print("parity close-out schema violations:")
    for p in problems:
        print(" -", p)
    sys.exit(1)
print("parity close-out schema: ok")
PY
