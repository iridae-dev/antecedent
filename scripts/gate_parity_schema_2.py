#!/usr/bin/env python3
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from _repo import ensure_scripts_on_path

root = Path(".")
print_counts = "--print-counts" in sys.argv
problems = []

# --- reason codes ---
vocab_path = root / "parity/reason_codes.toml"
if not vocab_path.is_file():
    problems.append("parity/reason_codes.toml missing")
    vocab = {"code": []}
else:
    vocab = tomllib.loads(vocab_path.read_text(encoding="utf-8"))
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
    # Facade-surface `reason` explains how Python reaches an entry or why it has none.
    "python_facade_surface.toml",
    # An interval-level exception's `reason` explains why the runtime publishes no interval.
    "interval_level_exceptions.toml",
}
for path in sorted(root.glob("parity/*.toml")):
    if path.name in _REASON_PROSE:
        continue
    text = path.read_text(encoding="utf-8")
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        problems.append(f"{path}: {exc}")
        continue
    def walk(obj, path=path):
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
        print(f"  {cid}: {n} (max_uses={codes[cid].get('max_uses')})")

for cid, n in uses.items():
    # A runtime-refusal-only code is spent in Rust, where this gate cannot count it,
    # so a registry ratchet is meaningful only when the registries themselves cite it.
    if "max_uses" not in codes[cid]:
        if codes[cid].get("applies_to") != ["runtime_refusal"]:
            problems.append(f"parity/reason_codes.toml: {cid} needs max_uses")
        continue
    max_uses = int(codes[cid]["max_uses"])
    if n > max_uses:
        problems.append(f"parity/reason_codes.toml: {cid} uses={n} > max_uses={max_uses}")

# Refusal codes a Rust `fn code(&self)` returns are stable machine-readable strings
# callers switch on; each must be in the closed vocabulary.
for src in sorted(root.glob("crates/*/src/**/*.rs")):
    text = src.read_text(errors="ignore", encoding="utf-8")
    for fn_body in re.findall(r"fn code\(&self\) -> &'static str \{(.*?)\n    \}", text, re.S):
        for literal in re.findall(r'=>\s*"([a-z][a-z0-9_]+)"', fn_body):
            if literal not in codes:
                problems.append(f"{src}: fn code returns {literal!r}, absent from parity/reason_codes.toml")

# --- python_products ---
pp = root / "parity/python_products.toml"
if not pp.is_file():
    problems.append("parity/python_products.toml missing")
else:
    products = tomllib.loads(pp.read_text(encoding="utf-8"))
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
            ["uv", "run", "--no-sync", "pytest", "--collect-only", "-q", *(t.removeprefix("python/") for t in tests)],
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
ensure_scripts_on_path()
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
    text = path.read_text(errors="ignore", encoding="utf-8")
    if re.search(rf"fn\s+{re.escape(fn)}\s*[(<]", text):
        return True
    # The collector stamps `dgp` from the suite's `RecordKey`, which is a label
    # the suite writes, not necessarily the `fn` that builds the table. Two
    # suites do that: `calibration_distribution_admg` labels
    # `frontdoor_data` while the function is `data`, and
    # `calibration_design_bayesian` labels `transport_data_frozen_target`
    # while the function is `transport_frozen_data`. Renaming those functions
    # would drift the suite facet after the replay waivers.
    if re.search(r'dgp:\s*"' + re.escape(fn) + r'"', text):
        return True
    # A suite whose records share one helper passes the label as a string argument
    # (`measure(design, "...", "<label>", seed)`); the label is written in the suite.
    if re.search(r'"' + re.escape(fn) + r'"', text):
        return True
    if fn in collector.conditional_record_dgp_labels(path):
        return True
    # A suite that imports its data-generating function from a shared test module
    # (`use common::static_dgp::{path_data, ..}`) names it through that import: it
    # resolves when a file compiled into the same test target defines it.
    if re.search(rf"\buse\b[^;]*\b{re.escape(fn)}\b[^;]*;", text):
        ensure_scripts_on_path()
        import test_evidence

        target = test_evidence.target_root(path)
        if target and any(
            item.kind == "fn" and item.name == fn
            for module in test_evidence.target_files(target[0].resolve())
            for item in test_evidence.rust_items(module)[1]
        ):
            return True
    return "macro_rules!" in text and bool(re.search(rf"\b{re.escape(fn)}\b", text))


def _band(
    nominal: float, replicates: int
) -> tuple[float, float, float | None, float | None]:
    mcse = (nominal * (1.0 - nominal) / max(replicates, 1)) ** 0.5
    floor = nominal - 2.0 * mcse if replicates >= 1000 else None
    ceiling = nominal + 2.0 * mcse if replicates >= 1000 else None
    return (max(nominal - 3.0 * mcse, 0.0), min(nominal + 3.0 * mcse, 1.0), floor, ceiling)


cr = root / "parity/coverage_records.toml"
record_ids = set()
records = []
if cr.is_file():
    records = tomllib.loads(cr.read_text(encoding="utf-8")).get("record", [])
    # A record's facets decide which surface changes invalidate it, so they
    # must be exactly what scripts/calibration_facets.py derives from the
    # record, never a hand-narrowed list.
    surface = collector.facets.load_surface(root / "scripts/calibration_surface.list")
    if surface.errors:
        problems.extend(f"calibration_surface.list: {e}" for e in surface.errors)
    surface_refs = collector.facets.references(surface)
    for rec in records:
        rid = rec.get("id", "")
        record_ids.add(rid)
        label = f"coverage_records.toml {rid}"
        test = str(rec.get("test", ""))
        # `surface_list_blob` arrives with the next collection: a registry written by a
        # collector that stores it names it in its header, and only then is it required.
        stores_blob = "surface_list_blob" in cr.read_text(encoding="utf-8").split("[[record]]", 1)[0]
        missing = [
            key
            for key in collector.FIELDS
            if key not in rec and (key != "surface_list_blob" or stores_blob)
        ]
        if missing:
            problems.append(f"{label}: missing {', '.join(missing)}")
            continue
        tagged = rec["facets"]
        if not isinstance(tagged, list) or not tagged or not all(isinstance(f, str) for f in tagged):
            problems.append(f"{label}: facets must be a non-empty list of facet names")
        else:
            unknown = sorted(set(tagged) - surface.facets)
            if unknown:
                problems.append(f"{label}: unknown facets {', '.join(unknown)}")
            derived = collector.facets.record_facets(rec, surface, surface_refs)
            if sorted(tagged) != derived:
                problems.append(
                    f"{label}: facets {sorted(tagged)} are not the derived {derived}; "
                    "run python3 scripts/collect_coverage_records.py --retag"
                )
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
        elif "surface_list_blob" in rec:
            listed = subprocess.run(
                ["git", "rev-parse", f"{sha}:scripts/calibration_surface.list"],
                capture_output=True,
                text=True,
            )
            # A commit missing from a clone is the attestation gate's finding, not this one's.
            if listed.returncode == 0 and listed.stdout.strip() != rec["surface_list_blob"]:
                problems.append(
                    f"{label}: surface_list_blob {rec['surface_list_blob']} is not the surface "
                    f"list at calibration_sha ({listed.stdout.strip()})"
                )
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
        nominal = float(rec["nominal"])
        role = str(rec["role"])
        boundary = bool(rec["boundary"])

        def nominal_pass(point: dict, nominal=nominal) -> bool:
            lo, hi, floor, ceiling = _band(nominal, int(point["replicates"]))
            observed = float(point["observed"])
            # The harness reruns a point that shortfalls the level by more than
            # RECHECK_SHORTFALL at fewer than PRECISION_N_SIM replicates at
            # RECHECK_N_SIM, where the precision floor applies. A short-run point
            # that far under the level is unresolved for every role, not a pass.
            if int(point["replicates"]) < 1000 and observed < nominal - 0.02:
                return False
            if floor is not None and observed < floor:
                return False
            # Symmetric counterpart of the floor: a conservative interval that
            # over-covers must fail too, once measured precisely (see
            # precision_ceiling in crates/antecedent/tests/common/calibration.rs).
            if ceiling is not None and observed > ceiling:
                return False
            return lo <= observed <= hi

        def one_sided_pass(point: dict, nominal=nominal) -> bool:
            # A one-sided (at least nominal) target: the lower edge of the band and,
            # from PRECISION_N_SIM, the precision floor; no upper band and no ceiling,
            # since over-coverage is what a conservative interval claims
            # (passes_one_sided in crates/antecedent/tests/common/calibration.rs).
            lo, _, floor, _ = _band(nominal, int(point["replicates"]))
            observed = float(point["observed"])
            if int(point["replicates"]) < 1000 and observed < nominal - 0.02:
                return False
            if floor is not None and observed < floor:
                return False
            return observed >= lo

        # The measured range is the sample-size grid: every point present, in
        # order, strictly growing, and the row's summary exactly what
        # scripts/collect_coverage_records.py::merge_grid derives from it.
        grid = rec["grid"]
        if (
            not isinstance(grid, list)
            or len(grid) != collector.GRID_POINTS
            or not all(isinstance(p, dict) for p in grid)
        ):
            problems.append(
                f"{label}: grid must hold one measurement per sample-size grid point "
                f"({collector.GRID_POINTS}); re-measure with scripts/measure_calibration.sh"
            )
            continue
        if any(sorted(p) != sorted(collector.GRID_ENTRY_FIELDS) for p in grid):
            problems.append(
                f"{label}: grid points must carry exactly {', '.join(collector.GRID_ENTRY_FIELDS)}"
            )
            continue
        if [int(p["point"]) for p in grid] != list(range(collector.GRID_POINTS)):
            problems.append(f"{label}: grid points must be 0..{collector.GRID_POINTS - 1} in order")
        if any(
            int(lo["n_max"]) >= int(hi["n_min"])
            for lo, hi in zip(grid, grid[1:], strict=False)
        ):
            problems.append(f"{label}: grid sample sizes must strictly increase point to point")
        if (int(rec["n_min"]), int(rec["n_max"])) != (
            min(int(p["n_min"]) for p in grid),
            max(int(p["n_max"]) for p in grid),
        ):
            problems.append(f"{label}: n_min..n_max must span the grid points")
        failing = [p for p in grid if p["boundary"]]
        if boundary != bool(failing):
            problems.append(f"{label}: boundary must be true exactly when a grid point is")
        governing = min(failing or grid, key=lambda p: (float(p["observed"]), int(p["point"])))
        if (rec["observed"], rec["mcse"], rec["replicates"]) != (
            governing["observed"],
            governing["mcse"],
            governing["replicates"],
        ):
            problems.append(
                f"{label}: observed / mcse / replicates must be the governing grid point's "
                f"(point {governing['point']})"
            )
        for point in grid:
            where = f"{label} grid point {point['point']}"
            for field in ("observed", "mcse"):
                val = point[field]
                if not isinstance(val, (int, float)) or not 0 <= float(val) <= 1:
                    problems.append(f"{where}: {field} not in [0,1]")
            if int(point["replicates"]) < 1:
                problems.append(f"{where}: replicates must be positive")
            if int(point["n_min"]) < 1 or int(point["n_max"]) < int(point["n_min"]):
                problems.append(f"{where}: measured row-count range is empty")
            point_role = str(point["role"])
            passes = nominal_pass(point)
            if point_role == "gated":
                if point["boundary"]:
                    problems.append(f"{where}: a gated point cannot be a boundary")
                if not passes:
                    problems.append(
                        f"{where}: gated coverage {point['observed']} is outside the {nominal} band"
                    )
            elif point_role == "named_boundary":
                if not point["boundary"]:
                    problems.append(f"{where}: a named boundary point must be boundary = true")
            elif point_role == "one_sided":
                if point["boundary"]:
                    problems.append(f"{where}: a one-sided point cannot be a boundary")
                if not one_sided_pass(point):
                    problems.append(
                        f"{where}: one-sided coverage {point['observed']} is below the {nominal} "
                        "floor"
                    )
            elif point_role == "reported_level":
                if bool(point["boundary"]) == passes:
                    problems.append(
                        f"{where}: boundary must be true exactly when the measured coverage "
                        f"{point['observed']} misses the {nominal} band"
                    )
            else:
                problems.append(f"{where}: unknown role {point_role}")
        # An independent confirmation (fresh-seed re-measurement of the same design)
        # may recover the passing part of a boundary record's range; the range is
        # exactly what collect_coverage_records.confirmed_passing_range derives, and
        # the failing points stay on the record.
        confirmation = rec.get("confirmation")
        stated_range = rec.get("confirmed_passing_range")
        if confirmation is None and stated_range is not None:
            problems.append(f"{label}: confirmed_passing_range without the confirmation behind it")
        elif confirmation is not None:
            if not boundary or role == "named_boundary":
                problems.append(
                    f"{label}: a confirmation belongs to a boundary record that is not a "
                    "named boundary"
                )
            if (
                not isinstance(confirmation, list)
                or not confirmation
                or not all(isinstance(p, dict) for p in confirmation)
                or any(sorted(p) != sorted(collector.GRID_ENTRY_FIELDS) for p in confirmation)
            ):
                problems.append(
                    f"{label}: confirmation must list points carrying exactly "
                    f"{', '.join(collector.GRID_ENTRY_FIELDS)}"
                )
                continue
            indices = [int(p["point"]) for p in confirmation]
            if indices != sorted(set(indices)) or not set(indices) <= set(range(len(grid))):
                problems.append(f"{label}: confirmation points must be distinct grid points in order")
                continue
            for point in confirmation:
                where = f"{label} confirmation point {point['point']}"
                measured_point = grid[int(point["point"])]
                if (point["n_min"], point["n_max"], point["role"]) != (
                    measured_point["n_min"],
                    measured_point["n_max"],
                    measured_point["role"],
                ):
                    problems.append(f"{where}: must re-measure the grid point's sample sizes and role")
                passes = one_sided_pass(point) if point["role"] == "one_sided" else nominal_pass(point)
                if bool(point["boundary"]) == passes:
                    problems.append(
                        f"{where}: boundary must be true exactly when coverage {point['observed']} "
                        f"misses the {nominal} target"
                    )
            if stated_range != collector.confirmed_passing_range(grid, confirmation):
                problems.append(
                    f"{label}: confirmed_passing_range must be the contiguous passing run the "
                    "grid and confirmation establish (collect_coverage_records.confirmed_passing_range)"
                )
        # A one-sided record states the side whose true extremal bound it covers and
        # its target (CoverageTally::assert_one_sided); no other role carries one.
        one_sided = rec.get("one_sided")
        if role == "one_sided":
            if (
                not isinstance(one_sided, dict)
                or sorted(one_sided) != ["side", "target"]
                or one_sided.get("side") not in ("lower", "upper")
                or not isinstance(one_sided.get("target"), str)
                or not one_sided["target"].strip()
            ):
                problems.append(
                    f"{label}: a one_sided record must state one_sided = {{ side = "
                    '"lower"|"upper", target = "<non-empty>" }'
                )
        elif one_sided is not None:
            problems.append(f"{label}: one_sided on a {role} record")
        roles = {str(p["role"]) for p in grid}
        if len(roles) > 1 and roles != {"gated", "named_boundary"}:
            problems.append(f"{label}: grid points carry incompatible roles {sorted(roles)}")
        else:
            expected_role = "named_boundary" if len(roles) > 1 else next(iter(roles))
            if role != expected_role:
                problems.append(f"{label}: role {role} is not the grid points' {expected_role}")
        if role == "gated":
            named = re.search(r"nominal_(\d+)_coverage", test_fn)
            if named and abs(int(named.group(1)) / 100 - nominal) > 1e-9:
                problems.append(
                    f"{label}: nominal {nominal} contradicts the test name {test_fn}"
                )
        elif role == "named_boundary":
            # Role comes from assert_boundary / assert_boundary_at. Renaming the
            # test to include "boundary" would edit the suite and owe remesure.
            pass
else:
    problems.append("parity/coverage_records.toml missing")

if len(record_ids) != len(records):
    problems.append("parity/coverage_records.toml: duplicate record ids")

# --- record-less calibration groups: each ran, passed, at a commit in this clone ---
gates_path = root / "parity/calibration_gates.toml"
# The collector's registry header names the ledger; a registry written by it must
# come with the ledger, so deleting the ledger cannot pass.
ledger_owed = cr.is_file() and "calibration_gates.toml" in cr.read_text(encoding="utf-8")
if not gates_path.is_file():
    if ledger_owed:
        problems.append("parity/calibration_gates.toml missing")
else:
    gate_rows = tomllib.loads(gates_path.read_text(encoding="utf-8")).get("gate", [])
    if not gate_rows:
        problems.append("parity/calibration_gates.toml: no record-less group is attested")
    if len({g.get("group") for g in gate_rows}) != len(gate_rows):
        problems.append("parity/calibration_gates.toml: duplicate groups")
    for g in gate_rows:
        label = f"calibration_gates.toml {g.get('group')}"
        if not isinstance(g.get("group"), str) or not g["group"]:
            problems.append(f"{label}: missing group")
        if not isinstance(g.get("passed"), int) or g["passed"] < 1:
            problems.append(f"{label}: no passed test recorded")
        gate_sha = str(g.get("calibration_sha", ""))
        if not re.fullmatch(r"[0-9a-f]{40}", gate_sha) or set(gate_sha) == {"0"}:
            problems.append(f"{label}: calibration_sha must be a real 40 lowercase hex commit")

by_id = {rec.get("id"): rec for rec in records}

# --- licensed cell calibration obligation ---
lic = tomllib.loads((root / "parity/support_licensed.toml").read_text(encoding="utf-8")).get("cell", [])
for cell in lic:
    label = f"{cell.get('query')}/{cell.get('graph_class')}/{cell.get('inference')}"
    has_cal = "calibration" in cell
    has_reason = bool(cell.get("calibration_reason"))
    reported_records = [
        by_id[rid]
        for rid in cell.get("calibration") or []
        if rid in by_id and abs(float(by_id[rid]["nominal"]) - collector.REPORTED_LEVEL) < 1e-9
    ]
    # A record list says what was measured, not that it passed: a cell whose every
    # reported-level record is a boundary states `boundary_record` beside the list.
    boundary_only = bool(reported_records) and all(rec["boundary"] for rec in reported_records)
    if has_cal and has_reason:
        if cell.get("calibration_reason") != "boundary_record":
            problems.append(
                f"support_licensed.toml {label}: calibration and calibration_reason together "
                "are only legal as calibration_reason = \"boundary_record\""
            )
    elif has_cal == has_reason:
        problems.append(f"support_licensed.toml {label}: exactly one of calibration / calibration_reason")
    if has_cal and boundary_only != (cell.get("calibration_reason") == "boundary_record"):
        problems.append(
            f"support_licensed.toml {label}: calibration_reason = \"boundary_record\" must be "
            "present exactly when every record at the reported level is a boundary"
        )
    if not has_cal and cell.get("calibration_reason") == "boundary_record":
        problems.append(f"support_licensed.toml {label}: boundary_record without a record list")
    for evidence_error in collector.calibration_estimator_evidence_problems(cell, root):
        problems.append(f"support_licensed.toml {label}: {evidence_error}")
    for ownership_error in collector.licensed_cell_calibration_problems(cell, by_id):
        problems.append(f"support_licensed.toml {label}: {ownership_error}")
    # Coverage figures in `limitations` must cite a matching record or say they are
    # not a registry value: scripts/gate_coverage_citations.sh owns that check, so
    # known-truth values, SEs and disclosed probe figures stay in the license text.
    # ---- [gates] reason-code eligibility comes from the registry row ----
    cited = cell.get("calibration_reason")
    if cited == "no_interval_reported" and cell.get("interval_method") not in (None, "none", "None"):
        problems.append(
            f"support_licensed.toml {label}: no_interval_reported cannot declare an interval method"
        )
    scope = codes.get(cited, {}).get("queries") if isinstance(cited, str) else None
    if scope is not None and cell.get("query") not in scope:
        problems.append(
            f"support_licensed.toml {label}: {cited} on a {cell.get('query')} cell; "
            f"reason_codes.toml limits it to {scope}"
        )

# ---- [gates] registry scopes name real queries ----
axes_queries = set(tomllib.loads((root / "parity/support_axes.toml").read_text(encoding="utf-8")).get("queries", []))
for cid, code in codes.items():
    for query in code.get("queries") or []:
        if query not in axes_queries:
            problems.append(f"reason_codes.toml {cid}: queries names unknown query {query!r}")

# ---- [gates] estimator calibration obligation, keyed on group ----
# Every `group = "estimation"` row states its calibration; calibration fields on
# any other row would be an obligation no gate enforces.
est = tomllib.loads((root / "parity/estimate.toml").read_text(encoding="utf-8")).get("capabilities", [])
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
    "compiler.identity_contract_file", "compiler.licensed_family", "compiler.prepared_second_shot",
    "compiler.prepared_family", "compiler.identity_encoding", "compiler.dbn_atom_identity",
    "compiler.score_reuse_identity", "compiler.state_expected_version",
    "compiler.series_refresh_atomicity", "compiler.series_state_reuse",
    "compiler.score_batch_reuse", "compiler.capability_reports", "compiler.support_neighbors",
    "compiler.capability_prefix", "compiler.claim_core", "compiler.design_rank",
    "compiler.composition_prefix", "compiler.python_contract_smoke",
    "compiler.python_stub_conformance",
}
compiler_rows = {
    row.get("id"): row
    for row in tomllib.loads((root / "parity/compiler.toml").read_text(encoding="utf-8")).get("capabilities", [])
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
    generated = data_rs.read_text(encoding="utf-8")
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
    source = path.read_text(errors="ignore", encoding="utf-8").split("#[cfg(test)]", 1)[0]
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

# ---- [gates] licensed cells link to the external oracles their routes rest on ----
# parity/licensed_routes.toml (generated from executed plans) must route exactly
# the licensed cells, and every `oracle_components` entry must sit on an external
# oracle row and name a component a licensed cell actually runs
# (scripts/external_evidence.py; rendered by generate_support_matrix_docs.py).
import external_evidence  # noqa: E402

problems.extend(f"external evidence: {p}" for p in external_evidence.check())

# Identity-only migrations cannot rewrite their original measurements or become
# active merely because a metadata status is changed.
import calibration_key_aliases  # noqa: E402  (scripts/ is on sys.path above)

try:
    calibration_key_aliases.validated_aliases(root)
except (ValueError, KeyError, StopIteration, OSError, subprocess.CalledProcessError) as exc:
    problems.append(f"calibration identity mapping: {exc}")

# ---- [gates] publishing requires calibration attestation ----
# A tag must not ship calibration labels from a registry that no longer
# matches the code: the publish workflow attests before any build or upload.
gating = subprocess.run(
    ["uv", "run", "--quiet", "--project", ".", "--only-group", "dev", "python",
     str((root / "scripts/ci_workflow.py").resolve()), "publish-gating",
     str((root / ".github/workflows/publish-release.yml").resolve())],
    cwd=root / "python", capture_output=True, text=True,
)
if gating.returncode != 0:
    problems.append(f"scripts/ci_workflow.py publish-gating failed: {gating.stdout}{gating.stderr}")
else:
    problems.extend(json.loads(gating.stdout)["problems"])

# --- identity / claims files exist ---
for rel_path in ("parity/identity.toml", "parity/claims.toml"):
    if not (root / rel_path).is_file():
        problems.append(f"{rel_path} missing")

# --- every identity row names its layer, its bindings, and what it covers ---
identity_rows = tomllib.loads((root / "parity/identity.toml").read_text(encoding="utf-8")).get("identity", [])
adr = (root / "adr/0022-causal-compiler-contract.md").read_text(encoding="utf-8")
for row in identity_rows:
    for field in (
        "domain",
        "rust",
        "contract_key",
        "python",
        "availability",
        "naming_row",
        "covers",
        "doc_rust",
        "doc_python",
    ):
        if not row.get(field):
            problems.append(f"identity.toml {row.get('domain')}: missing {field}")
    if row.get("python") not in str(row.get("doc_python")):
        problems.append(f"identity.toml {row.get('domain')}: doc_python must contain python")
if "antecedent.identity.v2" not in adr:
    problems.append("adr/0022: identity format tag does not match antecedent-core")

if problems:
    print("parity close-out schema violations:")
    for p in problems:
        print(" -", p)
    sys.exit(1)
print("parity close-out schema: ok")
