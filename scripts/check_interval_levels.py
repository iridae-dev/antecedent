#!/usr/bin/env python3
"""Interval levels: runtime endpoints, reported levels, records and citations agree.

For every construction a coverage record describes (query, graph class, inference, interval
method and the emitting test) this audit states the level the runtime reports its interval at,
the levels the registry holds records for, and the support-matrix cells citing them, and fails
when they disagree:

  runtime_level_unmeasured   the runtime reports the construction at a level no record covers
                             (a 95% report with only a 90% record), and no design registered in
                             scripts/gate_calibration.sh scores that level
  record_level_unreachable   a record at a level the runtime never reports for a method that
                             reports one fixed level (identified-set intervals are 90%)
  misnamed_record            a record whose id states one level (`nominal_90`, `_l95`) but whose
                             level is another, unless it is the `reported_level` scoring of a
                             design gated at the named level on the same replicates, which then
                             has a record at the named level
  citation_level_mismatch    a support-matrix cell cites a construction's records at a level other
                             than the one the runtime reports while records at the reported level
                             exist
  citation_missing_record    a cell or estimator row cites an id that is not in the registry
  contrast_not_separate      a mediation design records the `mediated` contrast without separate
                             `total` and `direct` records at the same level

The level the runtime reports is read from the source, not restated here:
`REPORTED_SE_INTERVAL_LEVEL` (crates/antecedent/src/result.rs) for the interval methods that
report the facade default, and `IDENTIFIED_SET_INTERVAL_LEVEL`
(crates/antecedent/src/analysis/execute/identified_set_diagnostics.rs) for identified-set
intervals. The response default `confidence_level` must equal the first.

A finding that the next measurement will resolve is PENDING, not a failure: the registry lacks the
record, but the design registered for it scores the reported level (or separate contrasts), so the
collector will produce it. `--strict` turns PENDING into failure for use after a measurement.
A construction the runtime withholds the interval of is declared in
parity/interval_level_exceptions.toml with the reason; an exception that no longer applies
(its construction now has records at the reported level) fails.

    python3 scripts/check_interval_levels.py              # audit; exit 1 on FAIL
    python3 scripts/check_interval_levels.py --strict     # PENDING fails too
    python3 scripts/check_interval_levels.py --verbose    # list the pending findings
    python3 scripts/check_interval_levels.py --json       # machine-readable table
    python3 scripts/check_interval_levels.py --self-test

Environment: INTERVAL_LEVELS_ROOT (tree to read, default the repository).
"""

from __future__ import annotations

import json
import os
import re
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

from _repo import REPO_ROOT

ROOT = Path(os.environ.get("INTERVAL_LEVELS_ROOT", REPO_ROOT))
TOLERANCE = 1e-9

REPORTED_CONSTANT = (
    "crates/antecedent/src/result.rs",
    r"pub const REPORTED_SE_INTERVAL_LEVEL: f64 = ([0-9.]+);",
)
IDENTIFIED_SET_CONSTANT = (
    "crates/antecedent/src/analysis/execute/identified_set_diagnostics.rs",
    r"pub const IDENTIFIED_SET_INTERVAL_LEVEL: f64 = ([0-9.]+);",
)
RESPONSE_DEFAULT = (
    "crates/antecedent-estimate/src/response/mod.rs",
    r"confidence_level: ([0-9.]+),",
)
CITING_FILES = ("parity/support_licensed.toml", "parity/estimate.toml")

# `cov.<query>.<graph>.<inference>.<method>.l<NN>.<test>[.<label>]`
ID_RE = re.compile(r"^(cov\.[^.]+\.[^.]+\.[^.]+\.[^.]+)\.l(\d+)\.(.+)$")
COV_RE = re.compile(r"cov\.[a-z0-9_.]+")
# A level a record's name states: `nominal_90`, `_l95`.
NAMED_LEVEL_RE = re.compile(r"(?:nominal_|(?<=_)l)(90|95)(?=$|_|\.)")
MEDIATION_QUERIES = ("MediationEffect", "TemporalMediationEffect")
CONTRAST_LABELS = ("total", "direct", "mediated")


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def load(rel: str) -> dict:
    path = ROOT / rel
    return tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def constant(spec: tuple[str, str]) -> float:
    rel, pattern = spec
    match = re.search(pattern, read(rel))
    if not match:
        raise SystemExit(f"{rel}: no match for {pattern!r}; the audit reads the runtime level here")
    return float(match.group(1))


def runtime_levels() -> dict[str, float]:
    """Level per interval method, read from the runtime source."""
    reported = constant(REPORTED_CONSTANT)
    response = constant(RESPONSE_DEFAULT)
    if abs(reported - response) > TOLERANCE:
        raise SystemExit(
            f"the response default level {response} differs from the reported level {reported}"
        )
    return {"identified_set": constant(IDENTIFIED_SET_CONSTANT), "default": reported}


def level_of(method: str, levels: dict[str, float]) -> float:
    return levels.get(method, levels["default"])


def split_id(record_id: str) -> tuple[str, int, str]:
    match = ID_RE.match(record_id)
    if not match:
        raise SystemExit(f"coverage record id {record_id!r} is not cov.<q>.<g>.<i>.<m>.l<NN>.<test>")
    return match.group(1), int(match.group(2)), match.group(3)


def normalized(rest: str) -> str:
    """The test component with its stated level replaced, so `x_l90` pairs with `x_l95`."""
    return NAMED_LEVEL_RE.sub(lambda m: m.group(0).replace(m.group(1), "NN"), rest)


def method_of(prefix: str) -> str:
    return prefix.rsplit(".", 1)[-1]


# ---------------------------------------------------------------- design sources


def function_bodies(source: str) -> dict[str, str]:
    bodies: dict[str, str] = {}
    for declaration in re.finditer(r"\bfn\s+([A-Za-z_]\w*)\s*[<(]", source):
        opening = source.find("{", declaration.end())
        if opening < 0:
            continue
        depth = 0
        for end in range(opening, len(source)):
            depth += source[end] == "{"
            depth -= source[end] == "}"
            if depth == 0:
                bodies[declaration.group(1)] = source[opening : end + 1]
                break
    return bodies


def call_chain(source: str, entry: str) -> str | None:
    """Bodies of `entry` and every function of the file it reaches, or None when `entry` is not
    defined as a function (a macro-generated test)."""
    bodies = function_bodies(source)
    if entry not in bodies:
        return None
    pending, seen, chain = [entry], set(), []
    while pending:
        name = pending.pop()
        if name in seen or name not in bodies:
            continue
        seen.add(name)
        chain.append(bodies[name])
        pending.extend(
            other for other in bodies if re.search(rf"(?<![\w:]){re.escape(other)}\s*\(", bodies[name])
        )
    return "\n".join(chain)


class Designs:
    """What the design emitting a record scores, read from its test source."""

    def __init__(self) -> None:
        self.cache: dict[str, str] = {}

    def source_of(self, record: dict) -> str | None:
        path = str(record.get("test", "")).partition("::")[0]
        if not path or not (ROOT / path).is_file():
            return None
        if path not in self.cache:
            self.cache[path] = read(path)
        return self.cache[path]

    def unit(self, record: dict) -> str | None:
        """The call chain of the emitting test, or its whole file when the test is generated
        by a macro (the macro's helper is then somewhere in the file)."""
        source = self.source_of(record)
        if source is None:
            return None
        entry = str(record["test"]).partition("::")[2]
        return call_chain(source, entry) or source

    def scores_reported_level(self, record: dict) -> bool:
        unit = self.unit(record)
        return bool(unit and re.search(r"\bREPORTED_LEVEL\b|\bz_for\(|\bZ95\b", unit))

    def executes_contrasts(self, record: dict) -> bool:
        unit = self.unit(record)
        return bool(
            unit
            and "labelled(" in unit
            and re.search(r"\bCONTRASTS\b|\bcontrasts\(\)|MediationContrast::Total", unit)
        )


# ---------------------------------------------------------------- the audit


def audit(records: list[dict], cells: list[dict], estimator_rows: list[dict], exceptions: list[dict],
          levels: dict[str, float], designs: Designs) -> dict:
    by_id = {record["id"]: record for record in records}
    findings: list[dict] = []

    def finding(severity: str, code: str, subject: str, detail: str) -> None:
        findings.append({"severity": severity, "code": code, "subject": subject, "detail": detail})

    groups: dict[tuple[str, str], dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        prefix, level, rest = split_id(record["id"])
        groups[(prefix, normalized(rest))][level].append(record)

    used_exceptions: set[str] = set()
    table = []
    for (prefix, rest), by_level in sorted(groups.items()):
        method = method_of(prefix)
        expected = round(level_of(method, levels) * 100)
        subject = f"{prefix}|{rest}"
        exception = next((e for e in exceptions if prefix.startswith(e["construction"])), None)
        measured = sorted(by_level)
        table.append({"construction": prefix, "test": rest, "runtime_level": expected, "measured": measured})
        if expected not in by_level:
            sample = by_level[measured[0]][0]
            if exception is not None:
                used_exceptions.add(exception["construction"])
            elif designs.scores_reported_level(sample):
                finding("PENDING", "runtime_level_unmeasured", subject,
                        f"runtime reports {expected}%; the design scores it, the next measurement records it")
            else:
                finding("FAIL", "runtime_level_unmeasured", subject,
                        f"runtime reports {expected}% but the records are at {measured}%")
        elif exception is not None:
            finding("FAIL", "stale_exception", subject,
                    f"exception {exception['construction']!r} no longer applies: {expected}% is measured")
        if method in levels:
            stray = [level for level in measured if level != expected]
            if stray:
                finding("FAIL", "record_level_unreachable", subject,
                        f"{method} intervals are always reported at {expected}%; records at {stray}%")

    for record in records:
        prefix, level, rest = split_id(record["id"])
        for named in {int(n) for n in NAMED_LEVEL_RE.findall(rest)}:
            if named == level:
                continue
            gated = by_id.get(record["id"].replace(f".l{level}.", f".l{named}.", 1))
            if record.get("role") != "reported_level" or gated is None:
                finding("FAIL", "misnamed_record", record["id"],
                        f"named for {named}% but recorded at {level}%"
                        + ("" if record.get("role") == "reported_level"
                           else f" with role {record.get('role')!r}")
                        + ("" if gated is not None else f"; no gated {named}% record to be the reported scoring of"))

    for row_kind, rows in (("cell", cells), ("estimator row", estimator_rows)):
        for index, row in enumerate(rows):
            label = f"{row_kind} {row.get('query', '?')}/{row.get('graph_class', row.get('graph', '?'))}/" \
                    f"{row.get('structure', '?')}/{row.get('inference', '?')}#{index}"
            cited = list(row.get("calibration") or [])
            missing = [rid for rid in cited if rid not in by_id]
            for rid in missing:
                finding("FAIL", "citation_missing_record", label, rid)
            cited_levels: dict[tuple[str, str], set[int]] = defaultdict(set)
            for rid in cited:
                if rid in by_id:
                    prefix, level, rest = split_id(rid)
                    cited_levels[(prefix, normalized(rest))].add(level)
            for key, seen in cited_levels.items():
                expected = round(level_of(method_of(key[0]), levels) * 100)
                if expected not in seen and expected in groups.get(key, {}):
                    finding("FAIL", "citation_level_mismatch", label,
                            f"{key[0]}|{key[1]}: cites {sorted(seen)}%, runtime reports {expected}% and a record exists")

    contrast_groups: dict[tuple[str, int, str, str], dict[str, dict]] = defaultdict(dict)
    for record in records:
        contrast = str(record.get("functional", "")).partition(".")[0]
        if record.get("query") not in MEDIATION_QUERIES or contrast not in CONTRAST_LABELS:
            continue
        prefix, level, rest = split_id(record["id"])
        test, _, label = rest.partition(".")
        base = re.sub(r"(?:^|_)(?:%s)$" % "|".join(CONTRAST_LABELS), "", label)
        contrast_groups[(prefix, level, test, base)][contrast] = record
    for (prefix, level, test, base), present in sorted(contrast_groups.items()):
        absent = [label for label in CONTRAST_LABELS if label not in present]
        if absent:
            sample = next(iter(present.values()))
            severity = "PENDING" if designs.executes_contrasts(sample) else "FAIL"
            finding(severity, "contrast_not_separate", f"{prefix}.l{level}.{test}" + (f".{base}" if base else ""),
                    f"no separate {', '.join(absent)} record at {level}%")

    unused = [e["construction"] for e in exceptions if e["construction"] not in used_exceptions]
    for construction in unused:
        if not any(p.startswith(construction) for p, _ in groups):
            finding("FAIL", "stale_exception", construction, "matches no coverage record")
    for exception in exceptions:
        if not str(exception.get("reason", "")).strip():
            finding("FAIL", "exception_without_reason", exception["construction"], "state why")
    return {"table": table, "findings": findings}


def run() -> dict:
    records = load("parity/coverage_records.toml").get("record", [])
    cells = load("parity/support_licensed.toml").get("cell", [])
    estimator_rows = load("parity/estimate.toml").get("capabilities", [])
    exceptions = load("parity/interval_level_exceptions.toml").get("exception", [])
    return audit(records, cells, estimator_rows, exceptions, runtime_levels(), Designs())


def report(result: dict, strict: bool, verbose: bool) -> int:
    findings = result["findings"]
    fails = [f for f in findings if f["severity"] == "FAIL" or (strict and f["severity"] == "PENDING")]
    pending = [f for f in findings if f["severity"] == "PENDING" and not strict]
    for finding in findings:
        if finding in fails or verbose:
            print(f"{finding['severity']} {finding['code']}: {finding['subject']}: {finding['detail']}")
    by_code = defaultdict(int)
    for finding in pending:
        by_code[finding["code"]] += 1
    print(
        f"interval levels: {len(result['table'])} constructions, {len(fails)} failing, "
        f"{len(pending)} pending the next measurement"
        + "".join(f" ({n} {code})" for code, n in sorted(by_code.items()))
    )
    return 1 if fails else 0


# ---------------------------------------------------------------- self-test


def self_test() -> int:
    def record(rid: str, role: str = "gated", test: str = "t.rs::t") -> dict:
        functional = rid.rsplit(".", 1)[-1]
        return {
            "id": rid,
            "role": role,
            "test": test,
            "query": "TemporalMediationEffect",
            "functional": f"{functional}.all_observed",
        }

    class NoDesigns(Designs):
        def scores_reported_level(self, record: dict) -> bool:
            return False

        def executes_contrasts(self, record: dict) -> bool:
            return False

    class ScoringDesigns(NoDesigns):
        def scores_reported_level(self, record: dict) -> bool:
            return True

    levels = {"identified_set": 0.9, "default": 0.95}
    base = "cov.q.dag.frequentist.analytic_se"

    def codes(result: dict, severity: str | None = None) -> set[str]:
        return {f["code"] for f in result["findings"] if severity in (None, f["severity"])}

    def go(records, cells=(), exceptions=(), designs=None) -> dict:
        return audit(list(records), list(cells), [], list(exceptions), levels, designs or NoDesigns())

    failures = []

    def expect(name: str, got: set[str], want: set[str]) -> None:
        if got != want:
            failures.append(f"{name}: got {sorted(got)}, want {sorted(want)}")

    both = [record(f"{base}.l90.x_nominal_90_coverage"),
            record(f"{base}.l95.x_nominal_90_coverage", "reported_level")]
    expect("clean pair", codes(go(both)), set())
    expect("only 90 where runtime reports 95",
           codes(go(both[:1]), "FAIL"), {"runtime_level_unmeasured"})
    expect("pending when the design scores the level",
           codes(go(both[:1], designs=ScoringDesigns()), "PENDING"), {"runtime_level_unmeasured"})
    expect("excepted withheld interval",
           codes(go(both[:1], exceptions=[{"construction": base, "reason": "withheld"}])), set())
    expect("stale exception",
           codes(go(both, exceptions=[{"construction": base, "reason": "withheld"}])),
           {"stale_exception"})
    expect("exception without reason",
           codes(go(both[:1], exceptions=[{"construction": base, "reason": " "}])),
           {"exception_without_reason"})
    expect("misnamed gated record",
           codes(go([record(f"{base}.l95.x_nominal_90_coverage")])),
           {"misnamed_record"})
    expect("level pair named by suffix",
           codes(go([record(f"{base}.l90.x_l90"), record(f"{base}.l95.x_l95")])), set())
    ident = "cov.q.temporal_pag.frequentist.identified_set"
    expect("identified set reports 90",
           codes(go([record(f"{ident}.l90.y_nominal_90_coverage")])), set())
    expect("identified set never reports 95",
           codes(go([record(f"{ident}.l90.y"), record(f"{ident}.l95.y")])),
           {"record_level_unreachable"})
    cell = {"query": "q", "graph_class": "Dag", "structure": "fixed", "inference": "Frequentist"}
    expect("citation to a missing record",
           codes(go(both, [{**cell, "calibration": [*(r["id"] for r in both), f"{base}.l95.absent"]}])),
           {"citation_missing_record"})
    expect("citation at the wrong level",
           codes(go(both, [{**cell, "calibration": [both[0]["id"]]}])),
           {"citation_level_mismatch"})
    expect("citation at both levels",
           codes(go(both, [{**cell, "calibration": [r["id"] for r in both]}])), set())
    expect("mediated without total and direct",
           codes(go([record(f"{base}.l95.m.mediated"), record(f"{base}.l90.m.mediated")]), "FAIL"),
           {"contrast_not_separate"})
    expect("mediated labelled by design with separate contrasts",
           codes(go([record(f"{base}.l{lv}.m.k_0_{label}") for lv in (90, 95) for label in CONTRAST_LABELS])),
           set())
    full = [record(f"{base}.l{lv}.m.{label}") for lv in (90, 95) for label in CONTRAST_LABELS]
    expect("separate contrasts", codes(go(full)), set())
    for failure in failures:
        print("self-test FAIL:", failure)
    print("self-test:", "FAIL" if failures else "ok")
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    unknown = [a for a in argv if a not in ("--strict", "--json", "--verbose")]
    if unknown:
        print(f"unknown arguments {unknown}", file=sys.stderr)
        return 2
    result = run()
    if "--json" in argv:
        print(json.dumps(result, indent=1))
    return report(result, strict="--strict" in argv, verbose="--verbose" in argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
