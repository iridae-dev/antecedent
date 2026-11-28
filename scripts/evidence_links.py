#!/usr/bin/env python3
"""Link every scientific pass and coverage record to the units its test executed (ADR 0031).

    python3 scripts/evidence_links.py run [--only NAME ...] [--jobs N] [--batch N] [--again]
    python3 scripts/evidence_links.py link              # write parity/evidence_links.json
    python3 scripts/evidence_links.py link --check      # fail when the traces give another file
    python3 scripts/evidence_links.py residue           # baseline items no link or resolution covers
    python3 scripts/evidence_links.py resolve --item KIND:ID --evidence E --kind K --unit U --reason R

`run` builds the facade's test binaries with the test-only `evidence-trace` feature and
runs them with `ANTECEDENT_EVIDENCE_TRACE` set and a two-replicate calibration budget, so
each test pays for one pass through its support decisions, not for its measurement. Each
binary writes `target/evidence-trace/<binary>.tsv`, and its test list
`target/evidence-trace/<binary>.tests`; a line naming no listed test is an error, since its
decisions could be attributed to none:

    unit    <binary> <test> <kind> <record id> <status>    a support decision
    record  <binary> <test> <record id> [<kind> <unit>]... a coverage record and its units
    declare <binary> <test> <kind> <record id>             evidence_unit!(kind, id)
    interval <binary> <test> <kind> <record id> <method>  the interval a result of the unit
                                                           reported (`none` for no interval)
    route   <binary> <test> <route> <kind> <record id> [<kind> <record id>]...
                                                           a result bound to its compiled route,
                                                           the unit it carries (`- -` for none) and
                                                           the units decided since the last route

`link` joins the traces to `parity/scientific_records.toml` and
`parity/coverage_records.toml`. A test that declares a unit links only to what it
declares, and a declaration of a unit the test did not execute is an error. A pass whose
test lies below the facade never reaches a support decision; it links to the units whose
claim source cites it, marked `cited`, the one link that is not observed. A coverage record
whose test reaches no support decision (below the facade, or a kernel called directly)
links to the units its tally declares (`CoverageTally::measures`), each of which must name a
unit that is not an alias. A test that declares a unit records it under `declared`, so the
declaration settles which of several executed units it asserts about.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import json
import os
import re
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "parity" / "evidence_links.json"
TRACES = ROOT / "target" / "evidence-trace"
FACADE = "antecedent"
MANIFEST = ROOT / "crates" / FACADE / "Cargo.toml"
TRACE_FEATURE = "evidence-trace"
# Per-binary ceiling; a binary that exceeds it links what it traced and is reported incomplete.
TIMEOUT_SECONDS = 1800
# Optimized, with the test profile's assertions: a test reaches its support decisions in a
# fraction of the debug time. Defined on the command line, not in Cargo.toml, whose profile
# tables are an input of every evidence manifest; its artifacts sit in target/evidence-links/.
PROFILE = [
    "--config", 'profile.evidence-links.inherits="dev"',
    "--config", "profile.evidence-links.opt-level=2",
    "--config", "profile.evidence-links.debug=false",
    "--config", "profile.evidence-links.debug-assertions=true",
    "--config", "profile.evidence-links.overflow-checks=true",
    "--config", "profile.evidence-links.incremental=false",
    "--profile", "evidence-links",
]
# A link needs one replicate that reaches the unit; smoke tallies never gate.
RUN_ENV = {"ANTECEDENT_CALIBRATION_NSIM": "2", "ANTECEDENT_CALIBRATION_SMOKE": "1"}
NO_UNIT = "-\t-"
FAILED_TEST = re.compile(r"^test (\S+) \.\.\. FAILED$", re.M)
KINDS = ("matrix", "construction", "graphless", "capability")


def facade_targets() -> dict[str, str]:
    """Integration test target name -> its source path, relative to the repository."""
    return crate_targets(FACADE)


@functools.cache
def crate_targets(crate: str) -> dict[str, str]:
    """`crate`'s integration test targets and its `lib`, by name -> source path. A target of a
    crate below the facade is named `<crate>/<target>`."""
    cargo = tomllib.loads((ROOT / "crates" / crate / "Cargo.toml").read_text(encoding="utf-8"))
    prefix = "" if crate == FACADE else f"{crate}/"
    out = {f"{prefix}{t['name']}": f"crates/{crate}/" + t.get("path", f"tests/{t['name']}.rs")
           for t in cargo.get("test", [])}
    if prefix:
        out[f"{prefix}lib"] = f"crates/{crate}/src"
    return out


@functools.cache
def facade_paths() -> dict[str, str]:
    return {path: name for name, path in facade_targets().items()}


def crate_of(target: str) -> str:
    return target.split("/", 1)[0] if "/" in target else FACADE


def target_features(target: str) -> tuple[str, ...]:
    from _repo import scientific_target_feature_args

    crate = crate_of(target)
    name = binary_of(target)
    args = ["--lib"] if name == "lib" else ["--test", name]
    found = scientific_target_feature_args(ROOT / "crates" / crate / "Cargo.toml", args)
    features = set(found[1].split(",")) if found else set()
    if crate != FACADE:
        features |= workspace_features(crate)
    # Below the facade a test reaches no support decision; its records declare their units.
    return tuple(sorted(features | ({TRACE_FEATURE} if crate == FACADE else set())))


@functools.cache
def workspace_features(crate: str) -> frozenset[str]:
    """The features the workspace test build (`WORKSPACE_TEST_FEATURES`) unifies onto `crate`:
    a lower crate's calibration tests are gated on them."""
    from _repo import WORKSPACE_TEST_FEATURES

    pending, seen = list(WORKSPACE_TEST_FEATURES), set()
    while pending:
        owner, feature = pending.pop().split("/", 1)
        if (owner, feature) in seen:
            continue
        seen.add((owner, feature))
        cargo = tomllib.loads((ROOT / "crates" / owner / "Cargo.toml").read_text(encoding="utf-8"))
        pending += [f for f in cargo.get("features", {}).get(feature, []) if "/" in f]
        pending += [f"{owner}/{f}" for f in cargo.get("features", {}).get(feature, []) if "/" not in f]
    return frozenset(feature for owner, feature in seen if owner == crate)


@functools.cache
def below_facade_targets() -> dict[str, str]:
    """Source path -> target, for the crates below the facade that hold a coverage record's test:
    their records name the units they measure, which no trace observes."""
    coverage = tomllib.loads((ROOT / "parity/coverage_records.toml").read_text(encoding="utf-8"))
    crates = {path.split("/")[1] for r in coverage.get("record", [])
              if (path := str(r.get("test", ""))).startswith("crates/") and path.split("/")[1] != FACADE}
    out: dict[str, str] = {}
    for crate in sorted(crates):
        if (ROOT / "crates" / crate / "Cargo.toml").is_file():
            out.update({path: name for name, path in crate_targets(crate).items()})
    return out


def target_of(path: str, below: dict[str, str]) -> str | None:
    """The target whose source holds the test file `path`: a `[[test]]` path, else its crate's lib."""
    facade = facade_paths()
    if path in facade:
        return facade[path]
    if path in below:
        return below[path]
    parts = path.split("/")
    if len(parts) > 3 and parts[2] == "src":
        return below.get("/".join(parts[:3]))
    return None


def wanted_targets() -> list[str]:
    """The facade targets holding a recorded scientific pass or a coverage record's test."""
    wanted: set[str] = set()
    science = tomllib.loads((ROOT / "parity/scientific_records.toml").read_text(encoding="utf-8"))
    for record in science.get("record", []):
        crate, binary, _ = str(record["test"]).split("::", 2)
        if crate == FACADE:
            wanted.add(binary)
    coverage = tomllib.loads((ROOT / "parity/coverage_records.toml").read_text(encoding="utf-8"))
    below = below_facade_targets()
    for record in coverage.get("record", []):
        name = target_of(str(record.get("test", "")).split("::", 1)[0], below)
        if name:
            wanted.add(name)
    return sorted(wanted)


def batches(targets: list[str], size: int) -> list[tuple[tuple[str, tuple[str, ...]], list[str]]]:
    """Targets grouped by crate and feature set, then cut into batches of at most `size`."""
    groups: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for target in targets:
        groups.setdefault((crate_of(target), target_features(target)), []).append(target)
    return [(group, members[i:i + size])
            for group, members in sorted(groups.items())
            for i in range(0, len(members), size)]


def build(group: tuple[str, tuple[str, ...]], members: list[str]) -> dict[str, str]:
    """Build one batch; target -> test executable. A target that fails to build is absent."""
    crate, features = group
    executables: dict[str, str] = {}
    if members:
        names = {t.rpartition("/")[2]: t for t in members}
        selection = [arg for n in names for arg in (["--lib"] if n == "lib" else ["--test", n])]
        command = ["cargo", "test", "--no-run", "--no-fail-fast", "--message-format=json-render-diagnostics",
                   *PROFILE, "-p", crate, *(["--features", ",".join(features)] if features else []), *selection]
        print(f"evidence links: building {len(members)} {crate} target(s) with {','.join(features)}", flush=True)
        done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            errors = [line for line in done.stderr.splitlines() if line.startswith("error")]
            print("\n".join(errors[:40]) or done.stderr[-4000:], file=sys.stderr)
        for line in done.stdout.splitlines():
            if not line.startswith("{"):
                continue
            message = json.loads(line)
            if message.get("reason") != "compiler-artifact" or not message.get("executable"):
                continue
            target = message["target"]
            name = "lib" if "lib" in target.get("kind", []) else target["name"]
            if name in names:
                executables[names[name]] = message["executable"]
    return executables


def binary_of(target: str) -> str:
    """The binary name a trace line carries for `target` (a facade target, `<crate>/<target>`, or
    the file stem `<crate>.<target>` of its trace)."""
    crate, _, name = target.replace(".", "/").rpartition("/")
    if name == "lib":
        return (crate or FACADE).replace("-", "_")
    return name


def trace_stem(target: str) -> str:
    return target.replace("/", ".")


@functools.cache
def record_leaves() -> frozenset[str]:
    """The function names of the tests behind recorded passes and coverage records."""
    science = tomllib.loads((ROOT / "parity/scientific_records.toml").read_text(encoding="utf-8"))
    coverage = tomllib.loads((ROOT / "parity/coverage_records.toml").read_text(encoding="utf-8"))
    return frozenset({str(r["test"]).rsplit("::", 1)[-1] for r in science.get("record", [])}
                     | {str(r.get("test", "")).rsplit("::", 1)[-1] for r in coverage.get("record", [])})


def run_one(target: str, executable: str, timeout: int = TIMEOUT_SECONDS) -> dict:
    trace = TRACES / f"{trace_stem(target)}.tsv"
    trace.unlink(missing_ok=True)
    listed = subprocess.run([executable, "--list", "--format", "terse", "--include-ignored"],
                            capture_output=True, text=True, check=False)
    listed_names = [line.rsplit(": ", 1)[0] for line in listed.stdout.splitlines() if line.endswith(": test")]
    (TRACES / f"{trace_stem(target)}.tests").write_text("".join(name + "\n" for name in listed_names))
    if not listed_names:
        # A binary whose tests a feature gate compiled out would otherwise pass and link nothing.
        return {"status": "no_tests", "seconds": 0.0, "failed_tests": []}
    ignored = subprocess.run([executable, "--list", "--format", "terse", "--ignored"],
                             capture_output=True, text=True, check=False)
    # An ignored test no record names (a regeneration helper, a manual measurement) links
    # nothing and may need setup a trace run does not give it.
    skips = [arg for line in ignored.stdout.splitlines() if line.endswith(": test")
             for name in [line.rsplit(": ", 1)[0]] if name.rsplit("::", 1)[-1] not in record_leaves()
             for arg in ("--skip", name)]
    listed_names = [name for name in listed_names if name not in skips]
    env = {**os.environ, "ANTECEDENT_EVIDENCE_TRACE": str(trace), **RUN_ENV}
    started = time.monotonic()
    # A binary that outlives the timeout continues with the tests it did not finish, by exact
    # name, appending to the same trace; it stops once a round finishes none of them.
    filters: list[str] = []
    passed: set[str] = set()
    failed_set: set[str] = set()
    while True:
        try:
            done = subprocess.run([executable, "--include-ignored", "--exact", *filters, *skips],
                                  cwd=ROOT / "crates" / crate_of(target), env=env,
                                  capture_output=True, text=True, timeout=timeout, check=False)
            status = "ok" if done.returncode == 0 else "failed"
            failed_set |= set(FAILED_TEST.findall(done.stdout))
            if status == "failed":
                trace.with_suffix(".failures.txt").write_text(
                    done.stdout.partition("\nfailures:\n")[2] or done.stdout[-8000:] + done.stderr[-8000:])
            if failed_set:
                status = "failed"
            break
        except OSError:
            # A link that ran out of disk leaves an artifact that is not executable.
            return {"status": "build_failed", "seconds": 0, "failed_tests": []}
        except subprocess.TimeoutExpired as expired:
            out = expired.stdout.decode(errors="replace") if isinstance(expired.stdout, bytes) else expired.stdout or ""
            finished = set(re.findall(r"^test (\S+) \.\.\. ok$", out, re.M))
            failed_set |= set(re.findall(r"^test (\S+) \.\.\. FAILED$", out, re.M))
            remaining = set(listed_names) - passed - finished - failed_set
            if not finished or not remaining:
                status = "timeout"
                failed_set |= remaining
                break
            passed |= finished
            filters = sorted(remaining)
            print(f"  {target}: timeout, continuing with {len(filters)} test(s)", flush=True)
    return {"status": status, "seconds": round(time.monotonic() - started, 1), "failed_tests": sorted(failed_set)}


def cmd_run(args: argparse.Namespace) -> int:
    """Build, run and delete one batch at a time, so the run never holds more than one batch
    of test executables on disk. A target that already ran ok is skipped unless --again."""
    TRACES.mkdir(parents=True, exist_ok=True)
    runs_path = TRACES / "runs.json"
    runs = json.loads(runs_path.read_text()) if runs_path.is_file() else {}
    targets = args.only or wanted_targets()
    if not args.again:
        targets = [t for t in targets if runs.get(t, {}).get("status") != "ok"]
    for group, members in batches(targets, args.batch):
        executables = build(group, members)
        for target in sorted(set(members) - set(executables)):
            runs[target] = {"status": "build_failed", "seconds": 0}
            print(f"  {target}: build_failed", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run_one, t, executables[t], args.timeout): t for t in members if t in executables}
            for future in concurrent.futures.as_completed(futures):
                target = futures[future]
                runs[target] = future.result()
                Path(executables[target]).unlink(missing_ok=True)
                print(f"  {target}: {runs[target]['status']} in {runs[target]['seconds']}s", flush=True)
                write_runs(runs_path, {t: runs[t] for t in targets if t in runs})
        write_runs(runs_path, {t: runs[t] for t in targets if t in runs})
    runs = json.loads(runs_path.read_text())
    failed = sorted(t for t, r in runs.items() if r["status"] != "ok")
    if failed:
        print(f"evidence links: {len(failed)} target(s) not ok: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


def write_runs(path: Path, mine: dict) -> None:
    """Merge this run's targets into the file as it is now: a concurrent run owns the rest."""
    current = json.loads(path.read_text()) if path.is_file() else {}
    path.write_text(json.dumps({**current, **mine}, sort_keys=True, indent=1) + "\n")


def failed_tests() -> set[tuple[str, str]]:
    """(binary, test) of every test that failed or did not finish: it links nothing."""
    runs_path = TRACES / "runs.json"
    runs = json.loads(runs_path.read_text()) if runs_path.is_file() else {}
    return {(binary_of(target), test)
            for target, run in runs.items() for test in run.get("failed_tests", [])}


def listed_tests() -> dict[str, set[str]]:
    """Binary -> the tests its executable lists."""
    return {binary_of(path.stem): set(path.read_text(encoding="utf-8").split())
            for path in TRACES.glob("*.tests")}


def read_traces() -> tuple[dict, dict, dict, dict, dict, dict, list[str]]:
    """(binary, test) -> {unit: supported?}, record id -> units, (binary, test) -> declared units,
    compiled route -> the units its results carried or were decided under, and (binary, test) ->
    the units of the results the test was handed. Lines written by a test that failed are
    dropped."""
    executed: dict[tuple[str, str], dict[str, bool]] = {}
    records: dict[str, set[str]] = {}
    declared: dict[tuple[str, str], set[str]] = {}
    routes: dict[str, set[str]] = {}
    returned: dict[tuple[str, str], set[str]] = {}
    intervals: dict[str, set[str]] = {}
    failed = failed_tests()
    listed = listed_tests()
    unattributed: dict[str, int] = {}
    for trace in sorted(TRACES.glob("*.tsv")):
        for line in trace.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split("\t")
            if len(fields) > 2 and (fields[1], fields[2]) in failed:
                continue
            if len(fields) > 2 and fields[1] in listed and fields[2] not in listed[fields[1]]:
                unattributed[fields[1]] = unattributed.get(fields[1], 0) + 1
                continue
            if fields[0] == "unit" and len(fields) == 6:
                _, binary, test, kind, ident, status = fields
                units = executed.setdefault((binary, test), {})
                unit = f"{kind}\t{ident}"
                units[unit] = units.get(unit, False) or status == "supported"
            elif fields[0] == "record" and len(fields) >= 4:
                pairs = fields[4:]
                records.setdefault(fields[3], set()).update(
                    f"{pairs[i]}\t{pairs[i + 1]}" for i in range(0, len(pairs) - 1, 2))
            elif fields[0] == "interval" and len(fields) == 6:
                intervals.setdefault(f"{fields[3]}\t{fields[4]}", set()).add(fields[5])
            elif fields[0] == "declare" and len(fields) == 5:
                declared.setdefault((fields[1], fields[2]), set()).add(f"{fields[3]}\t{fields[4]}")
            elif fields[0] == "route" and len(fields) >= 6 and len(fields) % 2 == 0:
                pairs = [f"{fields[i]}\t{fields[i + 1]}" for i in range(4, len(fields), 2)]
                routes.setdefault(fields[3], set()).update(u for u in pairs if u != NO_UNIT)
                if pairs[0] != NO_UNIT:
                    returned.setdefault((fields[1], fields[2]), set()).add(pairs[0])
    problems = [f"{binary}: {n} trace line(s) name no test the binary lists; spawn test threads "
                "with antecedent::support::caller_thread()" for binary, n in sorted(unattributed.items())]
    return executed, records, declared, routes, returned, intervals, problems


def test_matches(selector_path: str, test: str) -> bool:
    """Whether the libtest name `test` is the selector path's test: the same function, inside
    the selector's module path (a test may sit in further inline modules)."""
    module, _, function = selector_path.rpartition("::")
    head, _, leaf = test.rpartition("::")
    return leaf == function and (not module or head == module or head.startswith(module + "::"))


def cited_units() -> dict[str, set[str]]:
    """Check `path::name` -> the units whose current claim source cites it."""
    import evidence_baseline as eb

    cells = tomllib.loads((ROOT / eb.CELLS).read_text(encoding="utf-8"))
    rows = tomllib.loads((ROOT / eb.GRAPHLESS).read_text(encoding="utf-8"))
    caps = tomllib.loads((ROOT / eb.CAPABILITIES).read_text(encoding="utf-8"))
    cited: dict[str, set[str]] = {}
    for cell in cells.get("cell", []):
        if cell.get("evidence_test") and cell.get("evidence_assertion"):
            check = f"{cell['evidence_test']}::{cell['evidence_assertion']}"
            cited.setdefault(check, set()).add(f"matrix\t{eb.cell_ident(cell)}")
    for row in rows.get("support", []):
        for field in ("independent_check", "known_truth_test"):
            check = row.get(field)
            if isinstance(check, str) and "::" in check:
                cited.setdefault(check, set()).add(f"graphless\t{eb.graphless_ident(row)}")
    for record in caps.get("record", []):
        for check in eb.capability_checks(record):
            cited.setdefault(check, set()).add(f"capability\t{record['id']}")
    return cited


def check_selector(check: str, selectors: dict[str, list[str]]) -> list[str]:
    import evidence_baseline as eb

    return eb.recorded_passes(check, selectors)


def claimable_units() -> set[str]:
    """`<kind>\t<id>` of every unit a record may name: a claim source that is not an alias."""
    import evidence_residue as er

    return {f"{kind}\t{ident}" for (kind, ident), row in er.claim_sources().items()
            if not er.is_alias((kind, ident), row)}


def build_links() -> tuple[dict, list[str]]:
    executed, record_units, declared, route_units, returned, intervals, problems = read_traces()
    runs_path = TRACES / "runs.json"
    runs = json.loads(runs_path.read_text()) if runs_path.is_file() else {}

    def tests_of(binary: str, path: str) -> list[tuple[str, str]]:
        return [key for key in set(executed) | set(declared)
                if key[0] == binary and test_matches(path, key[1])]

    def linked(keys: list[tuple[str, str]], what: str) -> dict[str, set[str]]:
        """`executed` and `refused` units, and the `returned` ones the test's results carried."""
        units: dict[str, bool] = {}
        for key in keys:
            for unit, supported in executed.get(key, {}).items():
                units[unit] = units.get(unit, False) or supported
        chosen = set().union(*(declared.get(key, set()) for key in keys)) if keys else set()
        if chosen:
            for unit in sorted(chosen - set(units)):
                problems.append(f"{what}: declares {unit.replace(chr(9), ' ')}, which it did not execute")
            units = {u: s for u, s in units.items() if u in chosen}
        handed = set().union(*(returned.get(key, set()) for key in keys)) if keys else set()
        return {"executed": {u for u, s in units.items() if s},
                "refused": {u for u, s in units.items() if not s},
                "returned": handed & set(units),
                "declared": chosen & {u for u, s in units.items() if s}}

    science = tomllib.loads((ROOT / "parity/scientific_records.toml").read_text(encoding="utf-8"))
    by_leaf: dict[str, list[str]] = {}
    for record in science.get("record", []):
        by_leaf.setdefault(str(record["test"]).rsplit("::", 1)[-1], []).append(str(record["test"]))
    cited: dict[str, set[str]] = {}
    for check, units in cited_units().items():
        for selector in check_selector(check, by_leaf):
            cited.setdefault(selector, set()).update(units)

    passes: dict[str, dict[str, set[str]]] = {}
    for record in science.get("record", []):
        selector = str(record["test"])
        crate, binary, path = selector.split("::", 2)
        entry: dict[str, set[str]] = {}
        if crate == FACADE and binary in runs:
            entry = linked(tests_of(binary_of(binary), path), selector)
        # A citation stands in for a link only where the test reaches no support decision.
        if selector in cited and not entry.get("executed") and not entry.get("refused"):
            entry["cited"] = cited[selector]
        passes[selector] = entry

    coverage = tomllib.loads((ROOT / "parity/coverage_records.toml").read_text(encoding="utf-8"))
    below = below_facade_targets()
    claimable = claimable_units()
    records: dict[str, dict[str, set[str]]] = {}
    for record in coverage.get("record", []):
        rid = str(record["id"])
        path, _, function = str(record.get("test", "")).partition("::")
        name = target_of(path, below)
        entry: dict[str, set[str]] = {}
        keys = tests_of(binary_of(name), function) if name is not None and name in runs else []
        if name is not None and name in runs and not any(executed.get(key) for key in keys):
            # A test that reaches no support decision (below the facade, or a kernel called
            # directly) names the units its record measures.
            named = record_units.get(rid, set())
            for unit in sorted(named - claimable):
                problems.append(f"{rid}: declares {unit.replace(chr(9), ' ')}, which names no unit")
            entry = {"declared": named & claimable}
        elif name is not None and name in runs:
            chosen = set().union(*(declared.get(key, set()) for key in keys)) if keys else set()
            if record_units.get(rid) and (not chosen or record_units[rid] & chosen):
                ran = {u for key in keys for u in executed.get(key, {})}
                for unit in sorted((chosen | record_units[rid]) - ran):
                    problems.append(f"{rid}: declares {unit.replace(chr(9), ' ')}, which it did not execute")
                reported = record_units[rid] & chosen if chosen else record_units[rid]
                entry = {"executed": reported & ran, "declared": reported & chosen}
            else:
                entry = linked(keys, rid)
        records[rid] = entry

    units = sorted({u for table in (passes, records) for e in table.values() for s in e.values() for u in s})
    position = {u: i for i, u in enumerate(units)}

    def pack(table: dict[str, dict[str, set[str]]]) -> dict:
        return {key: {k: sorted(position[u] for u in v) for k, v in sorted(e.items()) if v}
                for key, e in sorted(table.items())}

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    data = {
        "about": "ADR 0031 evidence links; generated by scripts/evidence_links.py; never edited",
        "traced_at": commit,
        "runs": {t: runs[t]["status"] for t in sorted(runs)},
        "units": units,
        "passes": pack(passes),
        "records": pack(records),
        "routes": {route: sorted(units) for route, units in sorted(route_units.items())},
        "unowned_routes": sorted(route for route, units in route_units.items() if not units),
        "intervals": {unit: sorted(methods) for unit, methods in sorted(intervals.items())},
    }
    return data, problems


def render(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n"


def load(root: Path = ROOT) -> dict:
    """The links with unit positions resolved to `(kind, id)` pairs."""
    data = json.loads((root / "parity" / "evidence_links.json").read_text(encoding="utf-8"))
    units = [tuple(u.split("\t", 1)) for u in data["units"]]
    for table in ("passes", "records"):
        for entry in data[table].values():
            for key in entry:
                entry[key] = [units[i] for i in entry[key]]
    return data


def cmd_link(args: argparse.Namespace) -> int:
    data, problems = build_links()
    for problem in problems:
        print(f"evidence links: {problem}", file=sys.stderr)
    if problems:
        return 1
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.is_file() else ""
        stored = json.loads(current) if current else {}
        data["traced_at"] = stored.get("traced_at", data["traced_at"])
        if render(data) != current:
            print("parity/evidence_links.json differs from the traces", file=sys.stderr)
            return 1
        print("evidence links: current")
        return 0
    OUT.write_text(render(data), encoding="utf-8")
    linked = sum(1 for table in ("passes", "records") for e in data[table].values() if e)
    total = len(data["passes"]) + len(data["records"])
    print(f"wrote {OUT.relative_to(ROOT)}: {linked}/{total} passes and records linked, "
          f"{len(data['units'])} units")
    return 0


def cmd_residue(args: argparse.Namespace) -> int:
    import evidence_residue

    return evidence_residue.main(args)


def cmd_resolve(args: argparse.Namespace) -> int:
    import evidence_residue as er

    er.append_resolutions([er.resolution(args.item, args.evidence, args.kind, args.unit, args.reason)])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--only", action="append", default=[], metavar="TARGET")
    run.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    run.add_argument("--batch", type=int, default=24, help="test executables built and held on disk at once")
    run.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS, metavar="SECONDS",
                     help="per round of one test executable; a round that times out continues with the "
                          "tests it did not finish")
    run.add_argument("--again", action="store_true", help="rerun targets that already ran ok")
    run.set_defaults(func=cmd_run)
    link = sub.add_parser("link")
    link.add_argument("--check", action="store_true")
    link.set_defaults(func=cmd_link)
    residue = sub.add_parser("residue")
    residue.add_argument("--json", action="store_true")
    residue.set_defaults(func=cmd_residue)
    resolve = sub.add_parser("resolve", help="append a digest-bound resolution")
    resolve.add_argument("--item", required=True, metavar="KIND:ID")
    resolve.add_argument("--evidence", action="append", required=True, metavar="pass:TEST|record:ID|*")
    resolve.add_argument("--kind", required=True, choices=("same", "split", "merged", "narrowed", "retired"))
    resolve.add_argument("--unit", action="append", default=[], metavar="KIND:ID")
    resolve.add_argument("--reason", required=True)
    resolve.set_defaults(func=cmd_resolve)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
