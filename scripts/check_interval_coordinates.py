#!/usr/bin/env python3
"""Zero unmeasured interval coordinates, for every record of parity/promotion.toml.

This is scripts/check_interval_surfaces.py generalised to ALL records of
parity/promotion.toml, importing its helpers read-only. For each record:

  (a) no licensed route has an owning-registry row (parity/transport_stages.toml [[routes]],
      parity/support_licensed.toml cells) carrying `estimator_grid_not_measured`;
  (b) every route whose stage is `uncertainty` is `closed` (with an executed refusal test that
      names its reason code) or licensed with EVERY coverage record its record allocates present
      in parity/coverage_records.toml; a `nominal` claim is refused; a `calibrated` claim must
      allocate coverage record ids, and each allocated id must be emitted by a test under
      crates/*/tests (or python/tests) and registered in scripts/gate_calibration.sh, so a
      measurement can actually produce it;
      Emission means the id (or its last component, or a `...<component>` composed form) is a STRING
      LITERAL in the body of an `#[ignore]` test function (Rust; comments do not count) or of a
      function (Python), read through scripts/promotion_source.py; wiring means the component is a registered group name on a `run_*` call line
      of scripts/gate_calibration.sh (a comment does not count; a `carried_forward` record is not
      measured, so only its emission is required). The same holds for every record that
      allocates ids, whatever its claim or status (`carried_forward` included);
  (b2) hidden reservations: a `cov.<...>` id mentioned in any text field of a record but absent from
      that record's `coverage_records` is an error (an id reserved in prose is invisible to (b));
  (b3) orphan registrations: a `run_*` call of scripts/gate_calibration.sh whose test emits a
      coverage record (its own name is a literal in its body) must be owned by some record's
      `coverage_records` (else it is measured with no owner);
  (c) every public interval-bearing name on the record's surface files (its own `surface`,
      `surface_rust`, `surface_exports` and `surface_pyo3` files minus its `search_impl`) sits
      behind a closed or covered uncertainty route.

Calibration state is reported PER RECORD, for EVERY record that allocates coverage ids (not only
`calibrated` claims, so a `carried_forward` record's pending measurement stays visible), keyed by record id:
PENDING_CALIBRATION while an allocated record is absent,
PASS once all are present and attested (scripts/calibration_facets.py), FAIL otherwise.

    python3 scripts/check_interval_coordinates.py                 # static check
    python3 scripts/check_interval_coordinates.py --json          # machine-readable result
    python3 scripts/check_interval_coordinates.py --run-refusals  # also execute cited refusal tests
    python3 scripts/check_interval_coordinates.py --self-test

Exit status 1 on FAIL; PENDING_CALIBRATION is not a failure here (the exit gates decide).
Environment: INTERVAL_COORDINATES_ROOT (tree to read, default the repo), INTERVAL_COORDINATES_SKIP_ATTEST=1
(self-test only).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from _repo import REPO_ROOT as REPO
from _repo import ensure_scripts_on_path

ROOT = Path(os.environ.get("INTERVAL_COORDINATES_ROOT", REPO))
# check_interval_surfaces reads its tree from this variable at import; keep both on one tree.
os.environ["INTERVAL_SURFACES_ROOT"] = str(ROOT)
if os.environ.get("INTERVAL_COORDINATES_SKIP_ATTEST") == "1":
    os.environ["INTERVAL_SURFACES_SKIP_ATTEST"] = "1"
ensure_scripts_on_path()
import check_interval_surfaces as cai  # noqa: E402  (read-only reuse)
from promotion_source import (  # noqa: E402
    _scan_literals,
    assertion_texts,
    closure_code,
    ignored_test_emits_record,
    ignored_test_literals,
    nontest_pair,
    python_fn_literals,
    rust_items,
)

NOMINAL = cai.NOMINAL
WITHHOLD_REASONS = cai.WITHHOLD_REASONS
REGISTRY_FILES = [*cai.REGISTRY_FILES, "scripts/gate_calibration.sh"]


def load(rel: str) -> dict:
    p = ROOT / rel
    return tomllib.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def surface_interval_findings(rel: str) -> list[tuple[int, str]]:
    if not rel.endswith(".py"):
        lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
        # Crate/module-private builders are not public producers. Rust forbids
        # widening their visibility through a public reexport; actual public
        # interval declarations still require their own route or feature gate.
        return [
            (line, name)
            for line, name in cai.interval_findings(rel)
            if not re.match(r"\s*pub\((?:crate|super|self)\)\s+", lines[line - 1])
        ]
    # Python keyword arguments are uses, not new public interval declarations.
    found = []

    def declarations(nodes):
        for node in nodes:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                declarations(node.body)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_") and "interval" in node.name.lower():
                    found.append((node.lineno, node.name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
                if not name.startswith("_") and "interval" in name.lower():
                    found.append((node.lineno, name))

    declarations(ast.parse((ROOT / rel).read_text(encoding="utf-8")).body)
    return found


def has_hidden_calibration_interval(path: Path, line: int, name: str) -> bool:
    """Check the actual Rust attributes on this public interval item.

    The attribute parser strips comments before returning each item's attributes,
    so commented-out lookalikes cannot attest a production feature gate.
    """
    masked, items = rust_items(path.resolve())
    for item in items:
        if item.kind != "fn" or item.name != name:
            continue
        item_line = masked.code.count("\n", 0, item.start) + 1
        if item_line != line:
            continue
        attrs = {re.sub(r"\s+", "", attribute) for attribute in item.attrs}
        return '#[cfg(feature="calibration-internal")]' in attrs and "#[doc(hidden)]" in attrs
    return False


def surfaces_of(record: dict) -> list[str]:
    skip = set(record.get("search_impl") or [])
    files = [
        *(record.get("surface") or []),
        *(record.get("surface_rust") or []),
        *(record.get("surface_exports") or []),
        *(record.get("surface_pyo3") or []),
    ]
    return list(dict.fromkeys(f for f in files if f not in skip))


COV_ID = re.compile(r"cov\.[a-z_.0-9]+")


def label_component(text: str) -> str:
    """The id component CoverageTally::labelled makes of a label: lowercase, every
    non-alphanumeric character one `_` (`sanitize_label` of the shared harness, without
    the edge trim so a whole file can be searched for a component)."""
    return re.sub(r"[^A-Za-z0-9]", "_", text).lower()


def expand_shell_loops(gate_text: str) -> str:
    """`gate_text` with each `for v in a b c; do ... done` body repeated once per word, `$v`
    and `${v}` replaced, so a `run_*` call inside a loop names the groups it really runs.
    Only literal word lists expand; a loop over a command substitution or a glob is left
    as written, and a substituted `run_*` line drops its quotes."""
    lines = gate_text.splitlines()

    def expand(block: list[str]) -> list[str]:
        out: list[str] = []
        i = 0
        while i < len(block):
            m = re.match(r"\s*for\s+(\w+)\s+in\s+([^;$`(]+);\s*do\s*$", block[i])
            if not m:
                out.append(block[i])
                i += 1
                continue
            depth, j = 1, i + 1
            while j < len(block) and depth:
                if re.match(r"\s*for\s+\w+\s+in\b.*;\s*do\s*$", block[j]):
                    depth += 1
                elif re.match(r"\s*done\b", block[j]):
                    depth -= 1
                j += 1
            body = block[i + 1 : j - 1]
            for word in m.group(2).split():
                sub = [re.sub(rf"\${{{m.group(1)}}}|\${m.group(1)}\b", word, line) for line in body]
                for line in expand(sub):
                    out.append(
                        line.replace('"', "").strip() if line.lstrip().startswith("run_") else line
                    )
            i = j
        return out

    return "\n".join(expand(lines))


_RUST_LITERAL_CACHE: dict = {}


def emitted_by_a_test(coverage_id: str) -> bool:
    """The id, its last component (the claim/test name) or a `...<component>` composed form is a
    string literal in an `#[ignore]` test body (Rust) or a function body (Python); a comment or a
    mere substring of another token does not count."""
    comp = coverage_id.rsplit(".", 1)[-1]
    parts = coverage_id.split(".")
    components = {comp}
    if len(parts) >= 3 and re.fullmatch(r"l[0-9]+", parts[-3]):
        components.add(parts[-2])

    def hit(lit: str) -> bool:
        # A record label is the harness's label with every non-alphanumeric character
        # replaced by `_` and lowercased (CoverageTally::labelled), so the literal
        # `"shared_rho_0.0"` emits the id component `shared_rho_0_0`.
        return (
            lit in (comp, coverage_id) or lit.endswith("." + comp) or label_component(lit) == comp
        )

    # Some crates keep their ignored calibration tests in a #[cfg(test)]
    # module under src (antecedent-estimate does); both layouts emit records.
    for p in [*ROOT.glob("crates/*/tests/**/*.rs"), *ROOT.glob("crates/*/src/**/*.rs")]:
        stat = p.stat()
        cache_key = (p, stat.st_mtime_ns, stat.st_size)
        cached = _RUST_LITERAL_CACHE.get(cache_key)
        if cached is None:
            text = p.read_text(encoding="utf-8")
            cached = (text + "\n" + label_component(text), None)
            _RUST_LITERAL_CACHE[cache_key] = cached
        if not any(component in cached[0] for component in components):
            continue
        if cached[1] is None:
            cached = (cached[0], ignored_test_literals(p))
            _RUST_LITERAL_CACHE[cache_key] = cached
        if any(
            any(hit(lit) for lit in lits)
            or (
                registration_owns_id({"name": name}, set(lits), coverage_id)
                and any(hit(lit) for _, lit in _scan_literals(closure_code(p, name)))
            )
            for name, lits in cached[1].items()
        ):
            return True
    for p in ROOT.glob("python/tests/**/*.py"):
        if comp not in p.read_text(encoding="utf-8"):
            continue
        if any(hit(lit) for lits in python_fn_literals(p).values() for lit in lits):
            return True
    return False


def registered_groups(gate_text: str) -> set[str]:
    """Names on `run_*` call lines of gate_calibration.sh (comments and echoes excluded)."""
    names: set[str] = set()
    for line in expand_shell_loops(gate_text).splitlines():
        code = line.split("#", 1)[0].strip()
        if re.match(r"run_\w+\b", code):
            names.update(re.findall(r"[A-Za-z_][\w]*", code)[1:])
    return names


def registrations(gate_text: str) -> list[dict]:
    """`run_<fn> <name>` calls under each `echo "== <header>"`, with the wrapper's `--test` target."""
    targets: dict[str, str] = {}
    dynamic: set[str] = set()
    current_fn: str | None = None
    for line in gate_text.splitlines():
        m = re.match(r"(run_\w+)\(\) \{", line)
        if m:
            current_fn = m.group(1)
        elif current_fn and (t := re.search(r"--test (\w+)", line)):
            targets[current_fn] = t.group(1)
        elif line.startswith("}"):
            current_fn = None
    # Only this explicit positional wrapper binds a target supplied at its call.
    for match in re.finditer(r"(?m)^(run_\w+)\(\) \{\n(.*?)^\}", gate_text, re.DOTALL):
        body = "\n".join(line.split("#", 1)[0] for line in match.group(2).splitlines())
        if all(
            token in body
            for token in (
                'local file="$1"',
                'local filter="$2"',
                '--test "$file" "$filter"',
            )
        ):
            dynamic.add(match.group(1))
    out, header = [], ""
    for line in expand_shell_loops(gate_text).splitlines():
        h = re.match(r'echo "== (.*?) ==', line)
        if h:
            header = h.group(1)
        m = re.match(r"(run_\w+) +(?:\S+ +)?(\w+)\s*$", line)
        positional = re.fullmatch(r"(run_\w+) +(\w+) +(\w+)\s*", line)
        if positional and positional[1] in dynamic:
            out.append(
                {
                    "header": header,
                    "name": positional[3],
                    "target": positional[2],
                    "fn": positional[1],
                }
            )
            continue
        if m and m.group(1) in targets and not line.startswith("#"):
            out.append(
                {
                    "header": header,
                    "name": m.group(2),
                    "target": targets[m.group(1)],
                    "fn": m.group(1),
                }
            )
    return out


def measured_recordless_registration(registration: dict, path: Path) -> bool:
    """Own an original recordless gate without inventing a coverage ID.

    The source-wide restriction deliberately rejects mixed typed/recordless suites.
    Complete-tree attestation remains mandatory for these original ledger rows.
    """
    masked, _ = rust_items(path.resolve())
    if (
        "CoverageTally::new(" not in masked.code
        or re.search(r"\b(?:RecordKey|for_record)\b", masked.code)
        or registration["name"] not in ignored_test_literals(path).get(registration["name"], [])
    ):
        return False
    group = f"{registration['target']}__{registration['name']}"
    return any(
        row.get("group") == group
        and type(row.get("passed")) is int
        and row["passed"] > 0
        and re.fullmatch(r"[0-9a-f]{40}", row.get("calibration_sha", ""))
        for row in load("parity/calibration_gates.toml").get("gate", [])
    )


def reviewed_nonemitting_boundary(registration: dict, path: Path) -> bool:
    """The original cancelling-edge diagnostic deliberately emits no typed record.

    Reviewed literal-preserving function tokens prove its named-boundary branch
    returns before emission. `score(..., false)` only persists the unkeyed
    reporting/recovery tallies. This is diagnostic ownership, never calibration
    authority or an exemption for another mixed suite. All three original gate
    receipts and complete-tree attestation are still required.
    """
    from _rust_equivalence import tokens

    name = "sample_split_cancelling_edge_boundary_95"
    if (
        registration["target"] != "sample_split_effect_calibration"
        or registration["name"] != name
        or path != ROOT / "crates/antecedent/tests/sample_split_effect_calibration.rs"
        or name not in ignored_test_literals(path).get(name, [])
    ):
        return False
    expected = {
        name: "7b4cf337d050b09e1ad8ddce6c1c404ea4dfdf581fdc96eab9650f6293bac630",
        "named_boundary": "4ebe326283cc811eccaca2f87a41afd08ba2bc49d8dc393921769db8f76bf88e",
        "score": "503171f8d5a4096e09d6ffa02ae9159531f7c3a458e8e4f029392f285823c48d",
    }
    masked, items = rust_items(path.resolve())
    for function, digest in expected.items():
        matches = [item for item in items if item.kind == "fn" and item.name == function]
        if len(matches) != 1:
            return False
        item = matches[0]
        payload = json.dumps(tokens(masked.code[item.start : item.body[1]]), separators=(",", ":"))
        if hashlib.sha256(payload.encode()).hexdigest() != digest:
            return False
    rows = load("parity/calibration_gates.toml").get("gate", [])
    for point in range(3):
        group = f"sample_split_effect_calibration__{name}.p{point}"
        matching = [row for row in rows if row.get("group") == group]
        if (
            len(matching) != 1
            or type(matching[0].get("passed")) is not int
            or matching[0]["passed"] != 1
            or matching[0].get("calibration_sha") != "c30e10304f4bd85a01ad81dd133a360ddc95c2f1"
        ):
            return False
    return True


def text_fields(value: object, skip: tuple[str, ...] = ("coverage_records",)) -> list[str]:
    """Every string in a record (recursively) outside the skipped keys."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [t for k, v in value.items() if k not in skip for t in text_fields(v, skip)]
    if isinstance(value, list):
        return [t for v in value for t in text_fields(v, skip)]
    return []


def registered_output_ids(gate_text: str) -> dict[tuple[str, str], set[str]]:
    """Read each registered target once; keep literals bound to its actual test."""
    targets: dict[str, dict[str, set[str]]] = {}
    result = {}
    for registration in registrations(gate_text):
        target, name = registration["target"], registration["name"]
        if target not in targets:
            tests: dict[str, set[str]] = {}
            for source in ROOT.glob(f"crates/*/tests/**/{target}.rs"):
                for test, literals in ignored_test_literals(source).items():
                    tests.setdefault(test, set()).update(literals)
            targets[target] = tests
        result[(target, name)] = targets[target].get(name, set())
    return result


def registration_owns_id(registration: dict, literals: set[str], coverage_id: str) -> bool:
    """Match exact emitted IDs or CoverageTally's test[.label] suffix.

    The original tally composes its bound construction with the test string and
    optionally one sanitized effect label. The test string must occur in this
    actual registered target/test; a substring or another target's name is not
    evidence. Labels do not replace the emitting test's identity.
    """
    if coverage_id in literals:
        return True
    name = registration["name"]
    if name not in literals:
        return False
    parts = coverage_id.split(".")
    return parts[-1] == name or (
        len(parts) >= 3
        and re.fullmatch(r"l[0-9]+", parts[-3]) is not None
        and parts[-2] == name
        and bool(re.fullmatch(r"[a-z_0-9]+", parts[-1]))
    )


def untrusted_interval_input_problems(root: Path, record: dict, item: dict) -> list[str]:
    """Prove the one caller-declared report input is incapable of granting authority."""
    fields = {
        "name",
        "source",
        "class",
        "builder",
        "report_type",
        "authority_type",
        "authority_fixture",
        "why",
    }
    exact = {
        "name": "interval",
        "source": "crates/antecedent-io/src/measured_inference.rs",
        "class": "ScalarExecution",
        "builder": "build_report",
        "report_type": "MeasuredInferenceReport",
        "authority_type": "MeasuredInference",
    }
    if (
        not isinstance(item, dict)
        or set(item) != fields
        or any(item.get(key) != value for key, value in exact.items())
        or not isinstance(item.get("why"), str)
        or not item["why"].strip()
    ):
        return [
            "untrusted interval input needs the exact readable-report descriptor, never an estimator/output exemption"
        ]
    path = root / item["source"]
    code, _ = nontest_pair(path)
    _, items = rust_items(path)
    declarations = list(re.finditer(r"\bpub struct ScalarExecution\s*(\{[^}]*\})", code))
    builders = [i for i in items if i.kind == "fn" and i.name == item["builder"]]
    if len(declarations) != 1 or len(builders) != 1:
        return ["untrusted input declaration or readable report builder does not resolve uniquely"]
    field_body = declarations[0][1]
    # Exact bounded input fields. A new computed/authoritative field requires
    # another review and cannot silently inherit this metadata classification.
    if not re.fullmatch(
        r"\{\s*pub name: String,\s*pub point: f64,\s*pub interval: \(f64, f64\),\s*pub basis: CalibrationBasis,\s*\}",
        field_body,
    ):
        return ["untrusted input no longer consists solely of caller-declared scalar/basis fields"]
    builder = builders[0]
    header = code[builder.start : builder.body[0]]
    body = code[builder.body[0] : builder.body[1]]
    if (
        not re.search(r"executions:\s*Vec<ScalarExecution>", header)
        or not re.search(r"->\s*Result<MeasuredInferenceReport,\s*IoError>\s*$", header)
        or not re.search(r"lower:\s*execution\.interval\.0\s*,", body)
        or not re.search(r"upper:\s*execution\.interval\.1\s*,", body)
        or re.search(r"\bauthorize\s*\(|\bMeasuredInference\s*\{", body)
        or re.search(
            r"\bmut\s+executions?\b|execution\.interval\s*=|executions\s*\.\s*(?:iter_mut|sort|push|extend|retain)\b",
            header + body,
        )
        or not re.search(r"for\s+execution\s+in\s+executions\s*\{", body)
    ):
        return [
            "input builder must return readable metadata and project original endpoints without granting authority"
        ]
    authority = list(re.finditer(r"\bpub struct MeasuredInference\s*(\{[^}]*\})", code))
    factory = [i for i in items if i.kind == "fn" and i.name == "authorize"]
    if (
        len(authority) != 1
        or len(factory) != 1
        or not re.fullmatch(
            r"\{\s*report: MeasuredInferenceReport,\s*bytes: Arc<\[u8\]>,\s*source: Arc<\[u8\]>,\s*\}",
            authority[0][1],
        )
        or re.search(
            r"#\[derive\([^]]*Deserialize[^]]*\)\]\s*pub struct MeasuredInference\b",
            code,
        )
        or not re.search(r"pub\(crate\)\s*$", code[: factory[0].start])
    ):
        return [
            "readable input cannot expose, deserialize or publicly construct opaque measured authority"
        ]
    constructions = len(re.findall(r"\bMeasuredInference\s*\{", code)) - len(
        re.findall(r"\b(?:struct|impl)\s+MeasuredInference\s*\{", code)
    )
    if (
        constructions != 1
        or "MeasuredInference {" not in code[factory[0].body[0] : factory[0].body[1]]
    ):
        return ["readable metadata cannot gain an additional public measured-authority constructor"]
    fixture = next(
        (f for f in record.get("fixtures", []) if f.get("id") == item["authority_fixture"]),
        {},
    )
    if (
        fixture.get("role") != "negative"
        or not fixture.get("evidence_test")
        or not fixture.get("evidence_assertion")
    ):
        return ["untrusted input requires an actual ordinary authority-forgery negative fixture"]
    evidence = root / fixture["evidence_test"]
    assertion = fixture["evidence_assertion"]
    try:
        _, evidence_items = rust_items(evidence)
        tests = [i for i in evidence_items if i.kind == "fn" and i.name == assertion]
        observed = "\n".join(assertion_texts(evidence, assertion))
        if (
            len(tests) != 1
            or "#[test]" not in tests[0].attrs
            or any("ignore" in a for a in tests[0].attrs)
            or "MeasuredInference::consume" not in observed
            or ".is_err()" not in observed
            or "decode" not in observed
            or ".is_ok()" not in observed
        ):
            return [
                "authority evidence must exercise readable decode success and opaque consumer rejection"
            ]
    except (OSError, ValueError):
        return ["authority-forgery evidence does not resolve"]
    return []


def untrusted_interval_inputs(record: dict, findings: list, errors: list[str]) -> list:
    remaining = list(findings)
    for item in record.get("untrusted_interval_inputs", []):
        problems = untrusted_interval_input_problems(ROOT, record, item)
        matches = (
            [
                row
                for row in remaining
                if row[0] == item.get("source") and row[2] == item.get("name")
            ]
            if isinstance(item, dict)
            else []
        )
        if problems or len(matches) != 1:
            errors.extend(
                f"{record['id']}: {problem}"
                for problem in problems
                or ["untrusted input must match exactly one actual interval field"]
            )
        else:
            remaining = [row for row in remaining if row not in matches]
    return remaining


def delegated_interval_findings(
    record: dict, findings: list, records: dict, errors: list[str]
) -> list:
    """A verified scalar carrier exposes original endpoints, never a new estimator.

    Delegation names the exact prerequisite method owners; each still undergoes
    its own allocation, route, current attestation and producer/consumer gates.
    """
    remaining = list(findings)
    for item in record.get("delegated_interval_fields", []):
        if not isinstance(item, dict) or set(item) != {
            "name",
            "source",
            "class",
            "method_records",
            "authority_fixture",
            "why",
        }:
            errors.append(
                f"{record['id']}: delegated interval requires exact source/method/authority fields"
            )
            continue
        if any(
            not isinstance(item[key], str) or not item[key].strip()
            for key in ("name", "source", "class", "authority_fixture", "why")
        ):
            errors.append(
                f"{record['id']}: delegated interval needs nonempty source/method/authority fields"
            )
            continue
        methods = item["method_records"]
        fixtures = {f["id"]: f for f in record.get("fixtures", [])}
        authority = fixtures.get(item["authority_fixture"], {})
        valid = (
            isinstance(methods, list)
            and bool(methods)
            and all(isinstance(x, str) for x in methods)
            and len(methods) == len(set(methods))
            and set(methods) == set(record.get("prerequisite_records", []))
            and authority.get("role") == "negative"
            and bool(authority.get("evidence_assertion"))
            and isinstance(item["why"], str)
            and bool(item["why"].strip())
        )
        for method in methods if isinstance(methods, list) else []:
            owner = records.get(method, {}) if isinstance(method, str) else {}
            allocated = set(owner.get("coverage_records", [])) | set(
                owner.get("candidate_coverage_records", [])
            )
            outputs = owner.get("inference_outputs", [])
            if not allocated or not any(
                set(output.get("allocated_coverage_records", [])) & allocated for output in outputs
            ):
                valid = False
        source = item["source"]
        matches = [row for row in remaining if row[0] == source and row[2] == item["name"]]
        try:
            tree = ast.parse((ROOT / source).read_text(encoding="utf-8"))
            cls = next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == item["class"]
            )
            field = next(
                node
                for node in cls.body
                if isinstance(node, ast.FunctionDef) and node.name == item["name"]
            )
            # Exact original native report endpoint projection; no interval arithmetic,
            # fitting or caller callbacks can be disguised as inherited uncertainty.
            valid &= (
                len(matches) == 1
                and len(field.body) == 1
                and isinstance(field.body[0], ast.Return)
                and ast.unparse(field.body[0].value)
                == "(float(self._body['lower']), float(self._body['upper']))"
                and any(
                    isinstance(d, ast.Name) and d.id == "property" for d in field.decorator_list
                )
            )
        except (OSError, SyntaxError, StopIteration, TypeError):
            valid = False
        if not valid:
            errors.append(
                f"{record['id']}: delegated interval lacks original endpoint projection, exact method ownership or authority evidence"
            )
        else:
            remaining = [row for row in remaining if row not in matches]
    return remaining


def archived_emitter_literals(entry: dict) -> list[str]:
    """Read failed emitter evidence from its immutable original Git revision."""
    source = entry["harness"]
    if Path(source).is_absolute() or ".." in Path(source).parts:
        return []
    result = subprocess.run(
        ["git", "show", f"{entry['measurement_commit']}:{source}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        return []
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "archived_emitter.rs"
        path.write_text(result.stdout, encoding="utf-8")
        return ignored_test_literals(path).get(entry["test"], [])


def historical_allocations(
    record: dict, selected: list[str], present: set[str], errors: list[str]
) -> set[str]:
    """Own failed original emitters without selecting or licensing their intervals."""
    historical = set()
    rid = record["id"]
    declarations = record.get("historical_coverage_records", [])
    if not isinstance(declarations, list):
        errors.append(f"{rid}: historical coverage must be explicit failed emitter declarations")
        return historical
    for entry in declarations:
        if not isinstance(entry, dict):
            errors.append(
                f"{rid}: historical coverage must be explicit failed emitter declarations"
            )
            continue
        cid, source, test = entry.get("id"), entry.get("harness"), entry.get("test")
        if (
            not isinstance(cid, str)
            or not COV_ID.fullmatch(cid)
            or not isinstance(source, str)
            or not source.startswith("crates/")
            or not isinstance(test, str)
            or not test
            or entry.get("status") != "failed"
            or not isinstance(entry.get("measurement_commit"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", entry["measurement_commit"])
            or not isinstance(entry.get("failure"), str)
            or not entry["failure"].strip()
        ):
            errors.append(
                f"{rid}: historical coverage needs failed status, original source/test, commit and failure"
            )
            continue
        path = ROOT / source
        retired = entry.get("disposition") == "retired_superseded"
        if retired:
            replacement = entry.get("replacement_coverage_id")
            if replacement not in selected:
                errors.append(f"{rid}: retired history needs its selected replacement coverage id")
            literals = archived_emitter_literals(entry)
        else:
            literals = ignored_test_literals(path).get(test, []) if path.is_file() else []
        if not registration_owns_id({"name": test}, literals, cid):
            errors.append(
                f"{rid}: historical coverage id {cid} has no original ignored harness emitter"
            )
            continue
        if cid in selected or cid in historical:
            errors.append(
                f"{rid}: historical coverage must be disjoint from selected allocations and unique"
            )
        if cid in present:
            errors.append(f"{rid}: failed historical coverage cannot be a passing licensed record")
        matching_outputs = [
            output
            for output in record.get("inference_outputs", [])
            if cid in output.get("allocated_coverage_records", [])
        ]
        if retired:
            if matching_outputs:
                errors.append(f"{rid}: retired history cannot allocate an active inference output")
        elif not matching_outputs or any(
            output.get("allocation_status") != "historical_measurement_failed_route_closed"
            for output in matching_outputs
        ):
            errors.append(
                f"{rid}: historical coverage needs an explicitly failed output allocation"
            )
        historical.add(cid)
    return historical


def check() -> dict:
    # Every owner uses the same complete-tree attestation state during this check.
    # Keep the result local: a later invocation must observe newly collected evidence.
    attestation_result: tuple[bool, str] | None = None
    errors: list[str] = []
    notes: list[str] = []
    refusals: list[dict] = []
    calibration: dict[str, dict] = {}
    promotion_path = "parity/promotion.toml"
    promotion = load(promotion_path)
    stages = load("parity/transport_stages.toml")
    support = load("parity/support_licensed.toml")
    present = {r.get("id") for r in load("parity/coverage_records.toml").get("record", [])}
    gate_text = (
        (ROOT / "scripts/gate_calibration.sh").read_text(encoding="utf-8")
        if (ROOT / "scripts/gate_calibration.sh").is_file()
        else ""
    )
    groups = registered_groups(gate_text)
    registered_literals = registered_output_ids(gate_text)
    records = promotion.get("record", [])
    if not records:
        errors.append(f"no record found in {promotion_path}")
    stage_rows = {row.get("route"): row for row in stages.get("routes", [])}
    scoped_routes: set[str] = set()
    diagnostic_groups: set[tuple[str, str]] = set()
    allocations_by_record = {
        r["id"]: set(r.get("coverage_records") or [])
        | set(r.get("candidate_coverage_records") or [])
        for r in records
    }
    for record in records:
        rid = record["id"]
        for declaration in record.get("diagnostic_measurement_tests") or []:
            parts = declaration.split("::") if isinstance(declaration, str) else []
            if len(parts) != 2:
                errors.append(f"{rid}: diagnostic measurement needs an exact source::test")
                continue
            source, test = parts
            path = ROOT / source
            suites = {
                output.get("measurement_suite", "").split("::", 1)[0]
                for output in record.get("inference_outputs") or []
            }
            if (
                source not in suites
                or not path.is_file()
                or test not in ignored_test_literals(path)
                or "CoverageTally" in path.read_text(encoding="utf-8")
                or any(
                    COV_ID.search(literal) for literal in ignored_test_literals(path).get(test, [])
                )
            ):
                errors.append(
                    f"{rid}: diagnostic measurement {declaration!r} lacks its actual noncoverage ignored suite allocation"
                )
                continue
            key = (path.stem, test)
            if key not in registered_literals:
                errors.append(
                    f"{rid}: diagnostic measurement {declaration!r} is not registered at its actual target"
                )
            else:
                diagnostic_groups.add(key)
        active = list(record.get("coverage_records") or [])
        candidates = list(record.get("candidate_coverage_records") or [])
        allocated = active + candidates
        historical = historical_allocations(record, allocated, present, errors)
        if len(allocated) != len(set(allocated)):
            errors.append(f"{rid}: active and candidate coverage allocations must be distinct")
        missing_active = [cid for cid in active if cid not in present]
        missing = [c for c in allocated if c not in present]
        claim = record.get("inference_claim")
        if claim == "nominal":
            errors.append(f"{rid}: inference_claim 'nominal' ships an interval no record measures")
        if claim == "calibrated" and not active:
            errors.append(f"{rid}: a calibrated claim must allocate its coverage record ids")
        if active and claim != "calibrated" and record.get("status") != "carried_forward":
            errors.append(f"{rid}: allocates coverage records but claims {claim!r}, not calibrated")
        for cid in allocated:
            if not emitted_by_a_test(cid):
                errors.append(
                    f"{rid}: coverage id {cid} is emitted by no `#[ignore]` test body (string literal) "
                    "under crates/*/tests, crates/*/src, or python/tests"
                )
            # a carried_forward record is deliberately not measured, so it need not be
            # registered in gate_calibration.sh (its emitting test must still exist)
            if (
                record.get("status") != "carried_forward"
                and cid.rsplit(".", 1)[-1] not in groups
                and not any(
                    registration_owns_id(
                        registration,
                        registered_literals[(registration["target"], registration["name"])],
                        cid,
                    )
                    for registration in registrations(gate_text)
                )
            ):
                errors.append(
                    f"{rid}: coverage id {cid} is not a registered group name on a run_* line of "
                    "scripts/gate_calibration.sh"
                )
        inherited = set()
        for output in record.get("inference_outputs") or []:
            inherited_ids = set(output.get("inherited_coverage_records") or [])
            parents = output.get("inherited_from_records") or []
            parent_ids = set()
            for parent in parents:
                if parent == rid or parent not in allocations_by_record:
                    errors.append(
                        f"{rid}: inherited coverage names an invalid owning record {parent!r}"
                    )
                else:
                    parent_ids.update(allocations_by_record[parent])
            # Existing measured records are original methods with independently registered
            # emission. Unmeasured candidate inheritance must identify its actual owner.
            invalid = inherited_ids - (present | parent_ids)
            for cid in sorted(invalid):
                errors.append(
                    f"{rid}: inherited coverage id {cid} has no measured record or explicit candidate owner"
                )
            inherited.update(inherited_ids - invalid)
        hidden = sorted(
            {
                i.rstrip(".")
                for t in text_fields(record)
                for i in COV_ID.findall(t)
                if i.rstrip(".") not in allocated
                and i.rstrip(".") not in inherited
                and i.rstrip(".") not in historical
            }
        )
        for cid in hidden:
            errors.append(
                f"{rid}: coverage id {cid} is mentioned in the record's text but absent from its "
                "coverage_records (a hidden reserved id the calibration checks cannot see)"
            )
        open_uncertainty = closed_uncertainty = False
        for route in record.get("routes") or []:
            name, status, stage = (
                route.get("name"),
                route.get("status"),
                route.get("stage"),
            )
            scoped_routes.add(name)
            row = stage_rows.get(name)
            if (
                status == "licensed"
                and row is not None
                and (row.get("calibration_reason") == NOMINAL or NOMINAL in json.dumps(row))
            ):
                errors.append(
                    f"{rid}: licensed route {name} has a transport_stages row carrying {NOMINAL}"
                )
            if status == "licensed" and route.get("registry") == "support_licensed":
                for cell in support.get("cell", []):
                    if (
                        cell.get("query") == route.get("query")
                        and cell.get("contrast") == route.get("contrast")
                        and (
                            cell.get("calibration_reason") == NOMINAL or NOMINAL in json.dumps(cell)
                        )
                    ):
                        errors.append(
                            f"{rid}: licensed route {name} has a support cell carrying {NOMINAL}"
                        )
            if stage == "uncertainty":
                if status == "licensed":
                    open_uncertainty = True
                    if missing_active:
                        errors.append(
                            f"{rid}: uncertainty route {name} is licensed but coverage records are "
                            f"absent: {', '.join(missing_active)}"
                        )
                elif status == "closed":
                    closed_uncertainty = True
                else:
                    errors.append(f"{rid}: uncertainty route {name} has status {status!r}")
            if status == "closed" and (
                stage == "uncertainty" or route.get("reason_code") in WITHHOLD_REASONS
            ):
                test, assertion = (
                    route.get("refusal_test"),
                    route.get("refusal_assertion"),
                )
                if not test or not assertion:
                    # A frozen record has no executed test yet; in_progress/promoted must.
                    if record.get("status") in ("in_progress", "promoted"):
                        errors.append(f"{rid}: closed route {name} cites no executed refusal test")
                    continue
                path = ROOT / test
                if not path.is_file():
                    errors.append(f"{rid}: {name}: refusal test file {test} is missing")
                    continue
                body = cai.refusal_closure(path, assertion)
                if body is None:
                    errors.append(f"{rid}: {name}: refusal test {test}::{assertion} does not exist")
                elif not cai.names_reason(body, str(route.get("reason_code"))):
                    errors.append(
                        f"{rid}: {name}: refusal test {test}::{assertion} never names {route.get('reason_code')!r}"
                    )
                refusals.append(
                    {"record": rid, "route": name, "test": test, "assertion": assertion}
                )
        findings = []
        for rel in surfaces_of(record):
            if not (ROOT / rel).is_file():
                errors.append(f"{rid}: interval surface file {rel} is missing")
                continue
            findings += [(rel, line, n) for line, n in surface_interval_findings(rel)]
        # A wrapper may echo an interval licensed by its input route without
        # constructing its own uncertainty. The record must name those public
        # fields explicitly; any new interval-bearing name still needs a route.
        descriptors = record.get("noninferential_interval_fields") or []
        for descriptor in descriptors:
            name = descriptor.get("name")
            why = descriptor.get("why")
            if not isinstance(name, str) or not isinstance(why, str) or not why.strip():
                errors.append(f"{rid}: interval status descriptors need a name and explanation")
                continue
            matches = [
                row
                for row in findings
                if row[2] == name
                and row[0].endswith(".py")
                and (
                    re.match(
                        rf"\s*{re.escape(name)}\s*:\s*str\s*(?:#.*)?$",
                        (ROOT / row[0]).read_text(encoding="utf-8").splitlines()[row[1] - 1],
                    )
                    or (
                        re.match(
                            rf"\s*def {re.escape(name)}\(self\) -> str:",
                            (ROOT / row[0]).read_text(encoding="utf-8").splitlines()[row[1] - 1],
                        )
                        and (ROOT / row[0])
                        .read_text(encoding="utf-8")
                        .splitlines()[row[1] - 2]
                        .strip()
                        == "@property"
                    )
                )
            ]
            matches += [
                row
                for row in findings
                if row[2] == name
                and row[0].endswith(".rs")
                and re.match(
                    rf"\s*pub {re.escape(name)}\s*:\s*&'static str\s*,",
                    (ROOT / row[0]).read_text(encoding="utf-8").splitlines()[row[1] - 1],
                )
            ]
            if not matches:
                errors.append(f"{rid}: noninferential descriptor {name!r} is not a string field")
            findings = [row for row in findings if row not in matches]
        for name in record.get("internal_interval_fields") or []:
            matches = [
                row
                for row in findings
                if row[2] == name
                and row[0].endswith(".rs")
                and has_hidden_calibration_interval(ROOT / row[0], row[1], name)
            ]
            if not matches:
                errors.append(
                    f"{rid}: internal interval {name!r} lacks its explicit hidden calibration-only gate"
                )
            findings = [row for row in findings if row not in matches]
        findings = untrusted_interval_inputs(record, findings, errors)
        findings = delegated_interval_findings(
            record, findings, {r["id"]: r for r in records}, errors
        )
        inherited = set(record.get("inherited_interval_fields") or [])
        if inherited:
            seen = {name for _, _, name in findings}
            for name in sorted(inherited - seen):
                errors.append(
                    f"{rid}: inherited interval field {name!r} is absent from its surface"
                )
            findings = [row for row in findings if row[2] not in inherited]
        if findings and not (closed_uncertainty or (open_uncertainty and not missing_active)):
            first = findings[0]
            errors.append(
                f"{rid}: public interval-bearing output {first[2]} ({first[0]}:{first[1]}) has no "
                "closed uncertainty route and no coverage records"
            )
        if findings:
            notes.append(
                f"{rid}: {len(findings)} interval-bearing public names, all behind "
                f"{'a closed route' if closed_uncertainty else 'covered routes'}"
            )
        if allocated:
            if missing:
                calibration[rid] = {
                    "status": "PENDING_CALIBRATION",
                    "detail": f"{len(missing)} of {len(allocated)} coverage records absent",
                    "missing": missing,
                }
            else:
                if attestation_result is None:
                    attestation_result = cai.attested()
                ok, message = attestation_result
                calibration[rid] = {
                    "status": "PASS" if ok else "FAIL",
                    "detail": "records present and attested"
                    if ok
                    else f"records present, not attested: {message}",
                }
                if not ok:
                    errors.append(f"{rid}: coverage records present but not attested: {message}")
    allocated_ids = {
        cid
        for record in records
        for cid in [
            *(record.get("coverage_records") or []),
            *(record.get("candidate_coverage_records") or []),
        ]
    }
    # A registration is owned by a record that allocates the id it emits, or has been measured
    # already: a measured id is a row of parity/coverage_records.toml, which the licensed support
    # cells own. Only an unmeasured emitter no record allocates is an orphan.
    owned_ids = allocated_ids | present
    for reg in registrations(gate_text):
        if (reg["target"], reg["name"]) in diagnostic_groups or any(
            registration_owns_id(reg, registered_literals[(reg["target"], reg["name"])], cid)
            for cid in owned_ids
        ):
            continue
        for p in ROOT.glob(f"crates/*/tests/**/{reg['target']}.rs"):
            if measured_recordless_registration(reg, p) or reviewed_nonemitting_boundary(reg, p):
                if attestation_result is None:
                    attestation_result = cai.attested()
                if not attestation_result[0]:
                    errors.append(
                        f"{reg['name']}: recordless gate not attested: {attestation_result[1]}"
                    )
                continue
            literals = ignored_test_literals(p).get(reg["name"], [])
            names_itself = reg["name"] in literals or any(
                COV_ID.fullmatch(literal) for literal in literals
            )
            # A pass/fail gate scores through `CoverageTally::new` (no `for_record` key), so
            # it names itself without emitting a coverage record.
            if names_itself and ignored_test_emits_record(p, reg["name"]):
                errors.append(
                    f"scripts/gate_calibration.sh registers {reg['name']} (under '== {reg['header']}') "
                    f"and its test emits that coverage record, but no record lists it in "
                    "coverage_records (measured without an owner)"
                )
    legacy = [
        n
        for n, row in stage_rows.items()
        if n not in scoped_routes and NOMINAL in json.dumps(row) and row.get("status") == "licensed"
    ]
    if legacy:
        notes.append(f"{len(legacy)} licensed rows outside every record keep {NOMINAL} (backlog)")
    return {
        "intervals": "FAIL" if errors else "PASS",
        "calibration": calibration,
        "errors": errors,
        "notes": notes,
        "refusal_tests": [dict(t) for t in {tuple(sorted(r.items())) for r in refusals}],
        "records": [r["id"] for r in records],
    }


def report(result: dict) -> None:
    print(f"records checked: {len(result['records'])}")
    for note in result["notes"]:
        print(f"  note: {note}")
    for rid, state in sorted(result["calibration"].items()):
        print(f"  calibration {rid}: {state['status']} ({state['detail']})")
    for error in result["errors"]:
        print(f"  FAIL: {error}")
    print(f"interval coordinates (all records): {result['intervals']}")


# ----------------------------------------------------------------------------- self-test
SYNTH_SURFACE = "crates/synth/src/lib.rs"
SYNTH_TEST = "crates/synth/tests/refusal.rs"


def self_test() -> int:
    """Each synthetic violation, in an isolated synthetic tree, must fail; the baseline passes.
    No live record, source or gate line is read, so in-flight edits cannot break it."""
    global ROOT
    failures: list[str] = []
    real = REPO
    base = Path(tempfile.mkdtemp(prefix="interval_coordinates_"))
    try:
        synth = base / SYNTH_SURFACE
        synth.parent.mkdir(parents=True, exist_ok=True)
        synth.write_text(
            "pub struct R {\n    pub interval: f64,\n}\n"
            '#[cfg(feature = "calibration-internal")]\n'
            "// doc(hidden): internal calibration candidate\n"
            "#[doc(hidden)]\n"
            "pub fn candidate_interval_internal() {}\n",
            encoding="utf-8",
            newline="\n",
        )
        t = base / SYNTH_TEST
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(
            "#[test]\nfn refuses() {\n"
            '    let _ = "cell_not_licensed";\n}\n\n'
            '#[test]\n#[ignore = "coverage"]\nfn synth_alpha() {\n    record("synth_alpha");\n}\n\n'
            '#[test]\n#[ignore = "coverage"]\nfn synth_beta() {\n'
            '    let _key = "cov.synth.b.l95.synth_beta"; // a composed id also counts\n}\n',
            encoding="utf-8",
            newline="\n",
        )
        inherited_surface = base / "python/antecedent/synth_inherited.py"
        inherited_surface.parent.mkdir(parents=True, exist_ok=True)
        inherited_surface.write_text(
            "class Result:\n    interval: tuple[float | None, float | None] | None\n",
            encoding="utf-8",
            newline="\n",
        )
        synth_records = """
[[record]]
id = "synth.uncertainty_cell"
status = "in_progress"
inference_claim = "calibrated"
coverage_records = ["cov.synth.b.l95.synth_alpha", "cov.synth.b.l95.synth_beta"]
surface = ["crates/synth/src/lib.rs"]
internal_interval_fields = ["candidate_interval_internal"]
routes = [
  { name = "synth.prepare", stage = "prepare", status = "licensed", claim = "point_only" },
  { name = "synth.uncertainty", stage = "uncertainty", status = "closed", reason_code = "cell_not_licensed", refusal_test = "crates/synth/tests/refusal.rs", refusal_assertion = "refuses" },
]

[[record]]
id = "synth.inherited_interval"
status = "promoted"
inference_claim = "point_only"
surface = ["python/antecedent/synth_inherited.py"]
inherited_interval_fields = ["interval"]
"""
        promo = base / "parity/promotion.toml"
        promo.parent.mkdir(parents=True, exist_ok=True)
        promo.write_text("version = 1\n" + synth_records, encoding="utf-8", newline="\n")
        (base / "parity/transport_stages.toml").write_text(
            '[[routes]]\nroute = "synth.prepare"\nstatus = "licensed"\n',
            encoding="utf-8",
            newline="\n",
        )
        baseline_coverage: set[str] = set()
        (base / "parity/coverage_records.toml").write_text("", encoding="utf-8", newline="\n")
        gate = base / "scripts/gate_calibration.sh"
        gate.parent.mkdir(parents=True, exist_ok=True)
        gate.write_text(
            'echo "== synthetic coverage (antecedent) =="\n'
            "run_synth() {\n"
            '  local filter="$1"\n'
            '  check "synth: ${filter}" cargo test --release -p synth --test refusal "$filter" -- --ignored\n'
            "}\n"
            "run_synth synth_alpha\n"
            "run_synth synth_beta\n",
            encoding="utf-8",
            newline="\n",
        )
        os.environ["INTERVAL_COORDINATES_SKIP_ATTEST"] = "1"
        os.environ["INTERVAL_SURFACES_SKIP_ATTEST"] = "1"

        def run(edit=None, coverage: list[str] | None = None, edit_extra=None) -> dict:
            global ROOT
            case = Path(tempfile.mkdtemp(prefix="interval_coordinates_case_"))
            try:
                shutil.copytree(base, case, dirs_exist_ok=True)
                if coverage is not None:
                    (case / "parity/coverage_records.toml").write_text(
                        "".join(
                            f'[[record]]\nid = "{c}"\n'
                            for c in sorted(baseline_coverage | set(coverage))
                        ),
                        encoding="utf-8",
                        newline="\n",
                    )
                if edit is not None:
                    edit(case)
                if edit_extra is not None:
                    edit_extra(case)
                ROOT = case
                cai.ROOT = case
                rust_items.cache_clear()
                return check()
            finally:
                ROOT = real
                cai.ROOT = real
                shutil.rmtree(case, ignore_errors=True)

        def edit(rel: str, old: str, new: str):
            def apply(case: Path) -> None:
                p = case / rel
                text = p.read_text(encoding="utf-8")
                if old not in text:
                    raise LookupError(f"{old[:60]!r} not found in {rel}")
                p.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")

            return apply

        def expect(label: str, result: dict, needle: str | None) -> None:
            if needle is None:
                if result["intervals"] != "PASS":
                    failures.append(f"'{label}' should pass: {result['errors']}")
            elif result["intervals"] != "FAIL" or not any(needle in e for e in result["errors"]):
                failures.append(f"'{label}' did not report {needle!r}: {result['errors']}")

        promo_rel = "parity/promotion.toml"
        expect("baseline synthetic records", run(), None)
        expect(
            "commented cfg cannot attest internal interval gate",
            run(
                edit(
                    "crates/synth/src/lib.rs",
                    '#[cfg(feature = "calibration-internal")]\n',
                    '// #[cfg(feature = "calibration-internal")]\n',
                )
            ),
            "internal interval 'candidate_interval_internal' lacks its explicit hidden calibration-only gate",
        )
        expect(
            "internal interval without doc hidden",
            run(edit("crates/synth/src/lib.rs", "#[doc(hidden)]\n", "")),
            "internal interval 'candidate_interval_internal' lacks its explicit hidden calibration-only gate",
        )
        state = run()["calibration"].get("synth.uncertainty_cell", {}).get("status")
        if state != "PENDING_CALIBRATION":
            failures.append(f"absent coverage records must be PENDING_CALIBRATION, got {state}")
        closed_head = '{ name = "synth.uncertainty", stage = "uncertainty", status = "closed", reason_code = "cell_not_licensed", refusal_test = "crates/synth/tests/refusal.rs", refusal_assertion = "refuses" }'
        opened = '{ name = "synth.uncertainty", stage = "uncertainty", status = "licensed", claim = "calibrated" }'
        expect(
            "uncertainty licensed without records",
            run(edit(promo_rel, closed_head, opened)),
            "licensed but coverage records are absent",
        )
        covered = run(
            edit(promo_rel, closed_head, opened),
            coverage=["cov.synth.b.l95.synth_alpha", "cov.synth.b.l95.synth_beta"],
        )
        expect("uncertainty licensed with records", covered, None)
        if covered["calibration"].get("synth.uncertainty_cell", {}).get("status") != "PASS":
            failures.append(f"present coverage records must report PASS: {covered['calibration']}")
        expect(
            "record with a nominal claim",
            run(
                edit(
                    promo_rel,
                    'inference_claim = "calibrated"\ncoverage_records = ["cov.synth',
                    'inference_claim = "nominal"\ncoverage_records = ["cov.synth',
                )
            ),
            "ships an interval no record measures",
        )
        expect(
            "calibrated claim without ids",
            run(
                edit(
                    promo_rel,
                    'coverage_records = ["cov.synth.b.l95.synth_alpha", "cov.synth.b.l95.synth_beta"]',
                    "coverage_records = []",
                )
            ),
            "must allocate its coverage record ids",
        )
        expect(
            "coverage id emitted by no test",
            run(edit(SYNTH_TEST, 'record("synth_alpha")', 'record("other")')),
            "emitted by no `#[ignore]` test body",
        )
        expect(
            "coverage id only in a comment of its test is not emitted",
            run(edit(SYNTH_TEST, 'record("synth_alpha");', '// record("synth_alpha");')),
            "emitted by no `#[ignore]` test body",
        )
        expect(
            "coverage id literal in a test that is not #[ignore] is not emitted",
            run(
                edit(
                    SYNTH_TEST,
                    '#[ignore = "coverage"]\nfn synth_alpha',
                    "fn synth_alpha",
                )
            ),
            "emitted by no `#[ignore]` test body",
        )
        expect(
            "coverage id as a substring of another identifier is not emitted",
            run(edit(SYNTH_TEST, 'record("synth_alpha")', "let synth_alpha_v2 = 1")),
            "emitted by no `#[ignore]` test body",
        )
        generated = (
            "macro_rules! gen {\n    ($name:ident) => {\n        #[test]\n"
            '        #[ignore = "coverage"]\n        fn $name() {\n'
            "            record(stringify!($name));\n        }\n    };\n}\n"
        )
        alpha_fn = (
            '#[test]\n#[ignore = "coverage"]\nfn synth_alpha() {\n    record("synth_alpha");\n}\n'
        )
        expect(
            "an invocation of a test-generating macro emits its first identifier",
            run(edit(SYNTH_TEST, alpha_fn, generated + "gen!(synth_alpha);\n")),
            None,
        )
        labelled = (
            'const LABELS: &[(&str, u8)] = &[("rho_0.5", 1)];\n'
            + generated.replace("($name:ident)", "($name:ident, $labels:expr)")
            + "gen!(synth_alpha, LABELS);\n"
        )
        expect(
            "a dotted label is matched against the label table a generated test is passed",
            run(
                edit(SYNTH_TEST, alpha_fn, labelled),
                edit_extra=edit(
                    "parity/promotion.toml",
                    '"cov.synth.b.l95.synth_alpha"',
                    '"cov.synth.b.l95.synth_alpha.rho_0_5"',
                ),
            ),
            None,
        )
        expect(
            "a dotted label absent from the passed label table is not emitted",
            run(
                edit(SYNTH_TEST, alpha_fn, labelled.replace("rho_0.5", "rho_0.9")),
                edit_extra=edit(
                    "parity/promotion.toml",
                    '"cov.synth.b.l95.synth_alpha"',
                    '"cov.synth.b.l95.synth_alpha.rho_0_5"',
                ),
            ),
            "emitted by no `#[ignore]` test body",
        )
        expect(
            "a test-generating macro that is never invoked emits nothing",
            run(edit(SYNTH_TEST, alpha_fn, generated + "gen!(synth_other);\n")),
            "emitted by no `#[ignore]` test body",
        )
        expect(
            "a macro that does not generate an ignored test emits nothing",
            run(
                edit(
                    SYNTH_TEST,
                    alpha_fn,
                    generated.replace('#[ignore = "coverage"]\n', "") + "gen!(synth_alpha);\n",
                )
            ),
            "emitted by no `#[ignore]` test body",
        )
        expect(
            "a run_* call inside a literal shell loop registers each looped group",
            run(
                edit(
                    "scripts/gate_calibration.sh",
                    "run_synth synth_alpha\n",
                    'for n in synth_alpha synth_x; do\n  run_synth "${n}"\ndone\n',
                )
            ),
            None,
        )
        expect(
            "a shell loop over other names does not register the allocated group",
            run(
                edit(
                    "scripts/gate_calibration.sh",
                    "run_synth synth_alpha\n",
                    'for n in other; do\n  run_synth "${n}"\ndone\n',
                )
            ),
            "not a registered group name",
        )
        expect(
            "coverage id not registered in gate_calibration.sh",
            run(
                edit(
                    "scripts/gate_calibration.sh",
                    "run_synth synth_alpha",
                    "run_synth other",
                )
            ),
            "not a registered group name",
        )
        expect(
            "coverage id only in a comment of gate_calibration.sh is not registered",
            run(
                edit(
                    "scripts/gate_calibration.sh",
                    "run_synth synth_alpha",
                    "# run_synth synth_alpha",
                )
            ),
            "not a registered group name",
        )
        expect(
            "coverage id reserved only in the record's notes (hidden)",
            run(
                edit(
                    promo_rel,
                    'status = "in_progress"\ninference_claim = "calibrated"\ncoverage_records = ["cov.synth.b.l95.synth_alpha"',
                    'status = "in_progress"\ninference_notes = "also reserves cov.synth.b.l95.synth_gamma."\ninference_claim = "calibrated"\ncoverage_records = ["cov.synth.b.l95.synth_alpha"',
                )
            ),
            "hidden reserved id",
        )

        def register_orphan(case: Path) -> None:
            with (case / "scripts/gate_calibration.sh").open(
                "a", encoding="utf-8", newline="\n"
            ) as fh:
                fh.write("run_synth synth_gamma\n")
            with (case / SYNTH_TEST).open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(
                    '\n#[test]\n#[ignore = "coverage"]\nfn synth_gamma() {\n    record("synth_gamma");\n}\n'
                )

        expect(
            "a calibration registration that emits a record no record owns",
            run(register_orphan),
            "without an owner",
        )
        expect(
            "a carried_forward record's allocated ids are still checked and reported",
            run(
                edit(
                    promo_rel,
                    'status = "in_progress"\ninference_claim = "calibrated"',
                    'status = "carried_forward"\ninference_claim = "calibrated"',
                ),
                edit_extra=edit(SYNTH_TEST, 'record("synth_alpha")', 'record("other")'),
            ),
            "emitted by no `#[ignore]` test body",
        )
        expect(
            "a carried_forward record need not be registered in gate_calibration.sh",
            run(
                edit(
                    promo_rel,
                    'status = "in_progress"\ninference_claim = "calibrated"',
                    'status = "carried_forward"\ninference_claim = "calibrated"',
                ),
                edit_extra=edit(
                    "scripts/gate_calibration.sh",
                    "run_synth synth_alpha",
                    "# run_synth synth_alpha",
                ),
            ),
            None,
        )
        cf = run(
            edit(
                promo_rel,
                'status = "in_progress"\ninference_claim = "calibrated"',
                'status = "carried_forward"\ninference_claim = "calibrated"',
            )
        )
        if (
            cf["calibration"].get("synth.uncertainty_cell", {}).get("status")
            != "PENDING_CALIBRATION"
        ):
            failures.append(
                f"carried_forward record must report its calibration: {cf['calibration']}"
            )
        nc = run(
            edit(
                promo_rel,
                'inference_claim = "calibrated"\ncoverage_records',
                'inference_claim = "assumption_range"\ncoverage_records',
            )
        )
        if nc["calibration"].get("synth.uncertainty_cell", {}).get(
            "status"
        ) != "PENDING_CALIBRATION" or not any("not calibrated" in e for e in nc["errors"]):
            failures.append(
                f"non-calibrated claim with ids must error and still report calibration: {nc}"
            )
        expect(
            "interval surface without a closed route",
            run(
                edit(
                    promo_rel,
                    closed_head,
                    '{ name = "synth.point", stage = "evaluate", status = "licensed", claim = "point_only" }',
                )
            ),
            "has no closed uncertainty route",
        )
        expect(
            "an added interval field is not hidden by inherited interval metadata",
            run(
                edit(
                    "python/antecedent/synth_inherited.py",
                    "    interval: tuple[float | None, float | None] | None",
                    "    inverse_interval: tuple[float | None, float | None] | None\n"
                    "    interval: tuple[float | None, float | None] | None",
                )
            ),
            "has no closed uncertainty route",
        )
        expect(
            "refusal test never names the reason",
            run(edit(SYNTH_TEST, '"cell_not_licensed"', '"other"')),
            "never names",
        )

        def freeze_and_drop_test(case: Path) -> None:
            text = (case / promo_rel).read_text(encoding="utf-8")
            text = text.replace(
                ', refusal_test = "crates/synth/tests/refusal.rs", refusal_assertion = "refuses"',
                "",
            )
            text = text.replace(
                'status = "in_progress"\ninference_claim = "calibrated"',
                'status = "frozen"\ninference_claim = "calibrated"',
            )
            (case / promo_rel).write_text(text, encoding="utf-8", newline="\n")

        expect("frozen record may omit its refusal test", run(freeze_and_drop_test), None)
        expect(
            "in_progress record needs its refusal test",
            run(
                edit(
                    promo_rel,
                    ', refusal_test = "crates/synth/tests/refusal.rs", refusal_assertion = "refuses"',
                    "",
                )
            ),
            "cites no executed refusal test",
        )
        expect(
            "licensed row carrying estimator_grid_not_measured",
            run(
                edit(
                    "parity/transport_stages.toml",
                    'route = "synth.prepare"\n',
                    f'route = "synth.prepare"\ncalibration_reason = "{NOMINAL}"\n',
                )
            ),
            "carrying estimator_grid_not_measured",
        )
    finally:
        ROOT = real
        shutil.rmtree(base, ignore_errors=True)
    if failures:
        print("check_interval_coordinates self-test FAILED:")
        for f in failures:
            print(f" - {f}")
        return 1
    print("check_interval_coordinates self-test: ok")
    return 0


def self_test_exact_ownership() -> int:
    """Adversarial ownership checks on an isolated tree, without measurements."""
    global ROOT
    original_root, original_cai_root = ROOT, cai.ROOT
    failures = []
    with tempfile.TemporaryDirectory(prefix="intervals23_") as directory:
        ROOT = Path(directory)
        cai.ROOT = ROOT

        def write(path: str, contents: str) -> None:
            target = ROOT / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(contents, encoding="utf-8", newline="\n")

        coverage_id = "cov.synth.dag.frequentist.l95.mean0"
        surface = "python/antecedent/synth.py"
        write(
            surface,
            "class Result:\n    interval_status: str\n    def summarize(self):\n        return result(interval=(0, 1))\n",
        )
        write(
            "crates/synth/tests/calibration.rs",
            '#[test]\n#[ignore = "measurement"]\nfn whole_method() {\n'
            f'    emit("{coverage_id}");\n}}\n',
        )
        write(
            "scripts/gate_calibration.sh",
            'run_synth() {\n  cargo test --test calibration "$1" -- --ignored\n}\n'
            'echo "== synthetic =="\nrun_synth whole_method\n',
        )
        record = (
            '[[record]]\nid = "synth.diagnostic_candidate"\n'
            'status = "promoted"\ninference_claim = "point_only"\n'
            f'candidate_coverage_records = ["{coverage_id}"]\n'
            f'surface = ["{surface}"]\n'
            'noninferential_interval_fields = [{name="interval_status", why="status only"}]\n'
        )
        write("parity/promotion.toml", record)
        try:

            def expect(label: str, needle: str | None) -> dict:
                # Synthetic mutations deliberately reuse a path between checks.
                rust_items.cache_clear()
                result = check()
                if needle is None and result["errors"]:
                    failures.append(f"{label}: {result['errors']}")
                elif needle is not None and not any(needle in error for error in result["errors"]):
                    failures.append(f"{label}: missing {needle!r}: {result['errors']}")
                return result

            baseline = expect("scoped candidate allocation and typed status", None)
            if (
                baseline["calibration"].get("synth.diagnostic_candidate", {}).get("status")
                != "PENDING_CALIBRATION"
            ):
                failures.append("unmeasured diagnostic candidate lost its pending state")
            historical_id = "cov.synth.dag.frequentist.l95.old_method"
            historical_source = "crates/synth/tests/historical.rs"
            write(
                historical_source,
                '#[test]\n#[ignore = "measurement"]\nfn old_method() {\n'
                f'    emit("{historical_id}");\n}}\n',
            )
            history = (
                'historical_coverage_records = [{id="' + historical_id + '", '
                'harness="' + historical_source + '", test="old_method", status="failed", '
                'measurement_commit="'
                + "a"
                * 40
                + '", failure="below unchanged precision floor"}]\n'
                'inference_outputs = [{allocation_status="historical_measurement_failed_route_closed", '
                'allocated_coverage_records=["' + historical_id + '"]}]\n'
            )
            write("parity/promotion.toml", record + history)
            expect("failed historical emitter remains owned", None)
            write(
                "parity/promotion.toml",
                record + history.replace('status="failed"', 'status="passed"'),
            )
            expect("history cannot relabel passed evidence", "needs failed status")
            write(
                "parity/promotion.toml",
                record + history.replace('test="old_method"', 'test="invented"'),
            )
            expect("history binds actual original emitter", "no original ignored harness emitter")
            write(
                "parity/promotion.toml",
                record
                + history.replace(historical_id, coverage_id)
                .replace(historical_source, "crates/synth/tests/calibration.rs")
                .replace('test="old_method"', 'test="whole_method"'),
            )
            expect("historical cannot hide selected obligation", "disjoint from selected")
            write(
                "parity/promotion.toml",
                record
                + history.replace(
                    "historical_measurement_failed_route_closed",
                    "frozen_harness_compiled_measurement_pending",
                ),
            )
            expect(
                "history cannot be allocated as unmeasured candidate", "explicitly failed output"
            )
            write("parity/promotion.toml", record + history)
            write("parity/coverage_records.toml", '[[record]]\nid="' + historical_id + '"\n')
            expect(
                "failed history cannot supply passing record", "cannot be a passing licensed record"
            )
            write("parity/coverage_records.toml", "")
            write("parity/promotion.toml", record)
            multi_ids = [
                f"cov.synth.dag.frequentist.l95.whole_method.{label}"
                for label in ("mean0", "mean1", "contrast")
            ]
            multi_record = record.replace(
                f'candidate_coverage_records = ["{coverage_id}"]',
                "candidate_coverage_records = " + json.dumps(multi_ids),
            )
            write("parity/promotion.toml", multi_record)
            write(
                "crates/synth/tests/calibration.rs",
                '#[test]\n#[ignore = "measurement"]\nfn whole_method() {\n'
                '    measure("whole_method", ["mean0", "mean1", "contrast"]);\n}\n',
            )
            expect("composed multi-effect candidate IDs retain exact test ownership", None)
            write("parity/promotion.toml", multi_record.replace(json.dumps(multi_ids), "[]"))
            expect("multi-effect emitter without allocations remains orphaned", "without an owner")
            write(
                "parity/promotion.toml",
                multi_record.replace("l95.whole_method.", "l95.other_whole_method."),
            )
            expect("similar test-name substring cannot own composed IDs", "without an owner")
            write("parity/promotion.toml", record)
            write(
                "crates/synth/tests/calibration.rs",
                '#[test]\n#[ignore = "measurement"]\nfn whole_method() {\n'
                f'    emit("{coverage_id}");\n}}\n',
            )
            write(surface, "class Result:\n    interval_status: float\n")
            expect("numeric output cannot masquerade as a descriptor", "not a string field")
            write(
                surface,
                "class Result:\n    interval_status: str\n    interval: tuple[float, float]\n",
            )
            expect("real interval needs its own route", "has no closed uncertainty route")
            write(surface, "class Result:\n    interval_status: str\n")
            write(
                "parity/promotion.toml",
                record.replace('inference_claim = "point_only"', 'inference_claim = "calibrated"'),
            )
            expect(
                "candidate allocation does not license calibration",
                "must allocate its coverage record ids",
            )
            write("parity/promotion.toml", record)
            write(
                "scripts/gate_calibration.sh",
                'run_synth() {\n cargo test --test wrong_target "$1" -- --ignored\n}\n'
                'echo "== synthetic =="\nrun_synth whole_method\n',
            )
            expect(
                "same test name in another target is not registration",
                "not a registered group name",
            )
            write(
                "scripts/gate_calibration.sh",
                'run_synth() {\n cargo test --test calibration "$1" -- --ignored\n}\n'
                'echo "== synthetic =="\nrun_synth whole_method\n',
            )
            write(
                "crates/synth/tests/calibration.rs",
                '#[test]\n#[ignore = "measurement"]\nfn whole_method() {\n'
                f'    // emit("{coverage_id}");\n}}\n',
            )
            expect("comment is not actual candidate emission", "emitted by no")
            inherited_record = (
                '\n[[record]]\nid="synth.inherited_candidate"\n'
                'status="promoted"\ninference_claim="point_only"\n'
                'inference_outputs=[{inherited_from_records=["synth.diagnostic_candidate"],'
                f'inherited_coverage_records=["{coverage_id}"]}}]\n'
            )
            write("parity/promotion.toml", record + inherited_record)
            write(
                "crates/synth/tests/calibration.rs",
                '#[test]\n#[ignore = "measurement"]\nfn whole_method() {\n'
                f'    emit("{coverage_id}");\n}}\n',
            )
            expect("candidate inheritance names actual separate owner", None)
            write(
                "parity/promotion.toml",
                record
                + inherited_record.replace('"synth.diagnostic_candidate"', '"invented.owner"'),
            )
            expect("inherited reservation without owner refuses", "invalid owning record")
            write(
                "parity/promotion.toml",
                record
                + inherited_record.replace(
                    'inference_claim="point_only"', 'inference_claim="calibrated"'
                ),
            )
            expect(
                "candidate inheritance cannot license child calibration",
                "must allocate its coverage record ids",
            )
            diagnostic_record = record.replace(
                f'candidate_coverage_records = ["{coverage_id}"]\n',
                'diagnostic_measurement_tests = ["crates/synth/tests/calibration.rs::whole_method"]\n'
                'inference_outputs = [{measurement_suite="crates/synth/tests/calibration.rs"}]\n',
            )
            write("parity/promotion.toml", diagnostic_record)
            write(
                "crates/synth/tests/calibration.rs",
                '#[test]\n#[ignore = "precision"]\nfn whole_method() {\n'
                '    metric("whole_method");\n}\n',
            )
            expect("owned precision group does not fabricate a coverage record", None)
            write(
                "crates/synth/tests/calibration.rs",
                '#[test]\n#[ignore = "measurement"]\nfn whole_method() {\n'
                f'    emit("{coverage_id}");\n}}\n',
            )
            expect("coverage cannot be concealed as a diagnostic", "noncoverage ignored suite")
        finally:
            ROOT, cai.ROOT = original_root, original_cai_root
    if failures:
        print("check_interval_coordinates exact-ownership self-test FAILED:")
        for failure in failures:
            print(f" - {failure}")
        return 1
    print("check_interval_coordinates exact-ownership self-test: ok")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return max(self_test(), self_test_exact_ownership())
    result = check()
    if "--run-refusals" in argv:
        failures = cai.run_refusals([dict(t) for t in result["refusal_tests"]])
        result["errors"].extend(f"refusal test failed: {f}" for f in failures)
        result["intervals"] = "FAIL" if result["errors"] else "PASS"
    if "--json" in argv:
        print(json.dumps(result))
    else:
        report(result)
    return 1 if result["intervals"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
