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
import fcntl
import json
import os
import re
import subprocess
import sys
import time
import tempfile
from collections import defaultdict
from contextlib import contextmanager
import tomllib
from pathlib import Path

from evidence_trace_receipts import TraceInputs, digest, observations
from _repo import test_target

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
    if (ROOT / "crates" / crate / "src/lib.rs").is_file():
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
    return tuple(sorted(features | ({TRACE_FEATURE, "development"} if crate == FACADE else set())))


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
    crates = {path.split("/")[1] for r in registry("coverage_records")
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
        return {**below, **facade}.get("/".join(parts[:3]))
    if len(parts) > 3 and parts[2] == "tests":
        crate = parts[1]
        name, _ = test_target(crate, parts[3].removesuffix(".rs"), ROOT)
        target = name if crate == FACADE else f"{crate}/{name}"
        if target in crate_targets(crate):
            return target
    return None


def wanted_targets() -> list[str]:
    """The facade targets holding a recorded scientific pass or a coverage record's test."""
    return sorted(selectors_by_target())


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
        done = subprocess.run(command, cwd=ROOT, env=clean_env(), capture_output=True, text=True, check=False)
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
def selectors_by_target() -> dict[str, set[str]]:
    """Tests whose ignored flag must be overridden, scoped to their Cargo target."""
    out: dict[str, set[str]] = defaultdict(set)
    for row in registry("scientific_records"):
        crate, binary, path = selector_parts(row["test"])
        if crate == FACADE:
            out[binary].add(path.rsplit("::", 1)[-1])
    for row in registry("coverage_records"):
        target, path = coverage_test(row["test"])
        if target is not None:
            out[target].add(path.rsplit("::", 1)[-1])
    return out


def clean_env() -> dict[str, str]:
    """Run the fixed trace profile independently of a caller's evidence/measurement mode."""
    from compile_time_environment import noncanonical_compile_environment
    env = {k: v for k, v in os.environ.items() if not k.startswith("ANTECEDENT_") and k not in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE")}
    overrides = noncanonical_compile_environment(ROOT, env)
    overrides += [name for name in ("RUSTFLAGS", "RUSTC", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_TARGET", "RUSTC_BOOTSTRAP") if env.get(name)]
    if overrides:
        raise ValueError("trace build has noncanonical compiler settings: " + ", ".join(overrides))
    return {**env, **RUN_ENV}


def source_key(target: str, inputs: TraceInputs) -> str:
    crate = crate_of(target)
    path = ROOT / crate_targets(crate)[target]
    if path.is_dir():
        path = path / "lib.rs"
    return inputs.target_key(crate, path, target_features(target),
                             sorted(selectors_by_target().get(target, ())), PROFILE, RUN_ENV)


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
             for name in [line.rsplit(": ", 1)[0]] if name.rsplit("::", 1)[-1] not in selectors_by_target().get(target, ())
             for arg in ("--skip", name)]
    listed_names = [name for name in listed_names if name not in skips]
    env = {**clean_env(), "ANTECEDENT_EVIDENCE_TRACE": str(trace)}
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
            passed.update(re.findall(r"^test (\S+) \.\.\. ok$", done.stdout, re.M))
            status = "ok" if done.returncode == 0 and passed == set(listed_names) else "failed"
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
    if status == "ok":
        return {"status": status, **observations(trace, binary_of(target), passed)}
    return {"status": status, "seconds": round(time.monotonic() - started, 1), "failed_tests": sorted(failed_set)}


def atomic_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as tmp:
        tmp.write(render(value))
    os.replace(tmp.name, path)


@contextmanager
def trace_lock():
    TRACES.mkdir(parents=True, exist_ok=True)
    with (TRACES / ".capture.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def receipt_path(target: str) -> Path:
    return TRACES / f"{trace_stem(target)}.receipt.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def cmd_run(args: argparse.Namespace) -> int:
    with trace_lock():
        return capture(args)


def capture(args: argparse.Namespace) -> int:
    targets = sorted(set(args.only or wanted_targets()))
    known = set(wanted_targets())
    if unknown := set(targets) - known:
        raise ValueError(f"unknown evidence targets: {sorted(unknown)}")
    inputs = TraceInputs(ROOT)
    keys = {t: source_key(t, inputs) for t in targets}
    if not args.again:
        committed = load_json(OUT).get("receipts", {})
        targets = [t for t in targets if not current_receipt(
            load_json(receipt_path(t)) or committed.get(t, {}), keys[t])]
    failures = 0
    for group, members in batches(targets, args.batch):
        executables = build(group, members)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run_one, t, binary, args.timeout): t for t, binary in executables.items()}
            results = {t: {"status": "build_failed"} for t in set(members) - set(executables)}
            for future in concurrent.futures.as_completed(futures):
                target = futures[future]
                results[target] = future.result()
                Path(executables[target]).unlink(missing_ok=True)
            inputs.check_unchanged()
            for target, result in results.items():
                receipt = {"inputs": keys[target], **result}
                receipt["digest"] = digest(receipt)
                atomic_json(receipt_path(target), receipt)
                failures += result["status"] != "ok"
                print(f"  {target}: {result['status']}", flush=True)
    return int(bool(failures))


def current_receipt(receipt: dict, source: str) -> bool:
    return (receipt.get("status") == "ok" and receipt.get("inputs") == source
            and isinstance(receipt.get("tests"), dict) and bool(receipt["tests"])
            and receipt.get("digest") == digest({k: v for k, v in receipt.items() if k != "digest"}))


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


def claimable_units() -> set[str]:
    """`<kind>\t<id>` of every unit a record may name: a claim source that is not an alias."""
    import evidence_residue as er

    return {f"{kind}\t{ident}" for (kind, ident), row in er.claim_sources().items()
            if not er.is_alias((kind, ident), row)}


@functools.cache
def registry(name: str) -> list[dict]:
    return tomllib.loads((ROOT / "parity" / f"{name}.toml").read_text(encoding="utf-8")).get("record", [])


def selector_parts(selector: str) -> tuple[str, str, str]:
    # Python selectors have no Cargo crate or target.
    parts = selector.split("::", 2)
    return tuple(parts) if len(parts) == 3 and not parts[0].endswith(".py") else ("", "", selector)


@functools.cache
def coverage_test(spec: str) -> tuple[str | None, str]:
    from evidence_fingerprint import lib_module

    path, _, function = spec.partition("::")
    target = target_of(path, below_facade_targets())
    parts = path.split("/")
    prefix = ""
    if target is not None and len(parts) > 3:
        if parts[2] == "src":
            prefix = lib_module(path)
        elif path not in facade_paths() and path not in below_facade_targets():
            _, prefix = test_target(parts[1], parts[3].removesuffix(".rs"), ROOT)
    return target, prefix + function


class TestIndex:
    """Index by Cargo target and libtest leaf once for the entire registry join."""

    def __init__(self, receipts: dict):
        self.by_name: dict[tuple[str, str], list[tuple[str, dict]]] = defaultdict(list)
        for target, receipt in receipts.items():
            for name, facts in receipt["tests"].items():
                self.by_name[target, name.rsplit("::", 1)[-1]].append((name, facts))

    def match(self, target: str | None, path: str) -> list[dict]:
        return [facts for name, facts in self.by_name.get((target, path.rsplit("::", 1)[-1]), [])
                if test_matches(path, name)]


def linked(facts: dict) -> dict[str, set[str]]:
    executed = set(facts.get("executed", ()))
    refused = set(facts.get("refused", ())) - executed
    chosen = set(facts.get("declared", ()))
    if missing := chosen - (executed | refused):
        raise ValueError(f"asserts about units it never executed: {sorted(missing)}")
    return {
        "executed": executed & chosen if chosen else executed,
        "refused": refused & chosen if chosen else refused,
        "declared": chosen & executed,
        "returned": set(facts.get("returned", ())) & (executed | refused),
    }


def record_link(facts: dict, rid: str, claimable: set[str]) -> tuple[dict, bool]:
    """A particular tally in a completed test must have emitted this very record."""
    emissions = facts.get("records", {})
    if rid not in emissions:
        return {}, False
    entry = linked(facts)
    measured = set(emissions[rid])
    executed, refused = set(facts.get("executed", ())), set(facts.get("refused", ()))
    entry["returned"] &= executed
    if measured:
        if executed or refused:
            if missing := measured - (executed | refused):
                raise ValueError(f"tally names units its test did not execute: {sorted(missing)}")
            chosen = set(facts.get("declared", ()))
            if chosen and not measured <= chosen:
                raise ValueError("the tally's measured unit conflicts with its assertion")
            # A tally bound directly to an identified result is authoritative for that record.
            entry = {"executed": measured & executed, "declared": measured & executed,
                     "refused": measured - executed}
        else:
            if unknown := measured - claimable:
                raise ValueError(f"tally names unknown evidence units: {sorted(unknown)}")
            entry = {"declared": measured}
    return entry, not executed and not entry.get("declared") and not facts.get("declared")


def project(receipts: dict) -> tuple[dict, list[str]]:
    """Deterministically join verified target observations to the current ledger selectors."""
    import evidence_fingerprint as ef
    tests = TestIndex(receipts)
    problems = []
    science = {row["test"] for row in registry("scientific_records") if row.get("passed") is True}
    cited: dict[str, set[str]] = defaultdict(set)
    for check, units in cited_units().items():
        path, _, name = check.partition("::")
        try:
            selector = ef.locate(ROOT, {"evidence_test": path, "evidence_assertion": name})["selector"]
        except (OSError, ValueError, KeyError):
            continue
        if selector in science:
            cited[selector].update(units)

    passes, records = {}, {}
    for selector in sorted(science):
        crate, target, path = selector_parts(selector)
        matches = tests.match(target, path) if crate == FACADE else []
        entry = {}
        if len(matches) > 1:
            problems.append(f"{selector}: ambiguous libtest selector")
        elif matches:
            try:
                entry = linked(matches[0])
            except ValueError as exc:
                problems.append(f"{selector}: {exc}")
        if (crate != FACADE or len(matches) == 1) and not any(entry.values()) and selector in cited:
            entry["cited"] = cited[selector]
        passes[selector] = entry

    claimable = claimable_units()
    unadmitted = []
    for row in registry("coverage_records"):
        rid = row["id"]
        target, path = coverage_test(row["test"])
        matches = tests.match(target, path)
        entry = {}
        if len(matches) > 1:
            problems.append(f"{rid}: ambiguous libtest selector {path}")
        elif matches:
            try:
                entry, witnessed = record_link(matches[0], rid, claimable)
                if witnessed:
                    unadmitted.append(rid)
            except ValueError as exc:
                problems.append(f"{rid}: {exc}")
        records[rid] = entry

    routes: dict[str, set[str]] = defaultdict(set)
    intervals: dict[str, set[str]] = defaultdict(set)
    for receipt in receipts.values():
        for field, merged in (("routes", routes), ("intervals", intervals)):
            for key, values in receipt[field].items():
                merged[key].update(values)
    units = sorted({u for table in (passes, records) for entry in table.values()
                    for values in entry.values() for u in values})
    positions = {u: i for i, u in enumerate(units)}

    def pack(table: dict) -> dict:
        return {key: {field: sorted(positions[u] for u in values) for field, values in entry.items() if values}
                for key, entry in table.items()}
    return {
        "about": "ADR 0031 evidence links; generated by scripts/evidence_links.py; never edited",
        "schema": 2, "receipts": receipts, "units": units,
        "passes": pack(passes), "records": pack(records),
        "unadmitted_records": sorted(unadmitted),
        "routes": {k: sorted(v) for k, v in routes.items()},
        "unowned_routes": sorted(k for k, v in routes.items() if not v),
        "intervals": {k: sorted(v) for k, v in intervals.items()},
    }, problems


def render(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n"


def unpack(packed: dict) -> dict:
    data = dict(packed)
    units = [tuple(u.split("\t", 1)) for u in data["units"]]
    for table in ("passes", "records"):
        data[table] = {key: {field: [units[i] for i in values] for field, values in entry.items()}
                       for key, entry in packed[table].items()}
    return data


def load(root: Path = ROOT) -> dict:
    return unpack(load_json(root / "parity/evidence_links.json"))


def verified(receipts: dict) -> tuple[dict, list[str]]:
    current, problems = {}, []
    targets = set(wanted_targets())
    inputs = TraceInputs(ROOT) if receipts else None
    for target in sorted(targets):
        receipt = receipts.get(target, {})
        if receipt and current_receipt(receipt, source_key(target, inputs)):
            current[target] = receipt
        else:
            problems.append(f"{target}: missing, failed or stale evidence trace; run evidence_links.py run --only {target}")
    return current, problems


def verified_projection(links: dict) -> tuple[dict, list[str]]:
    receipts, debt = verified(links.get("receipts", {}))
    projection, problems = project(receipts)
    return unpack(projection), debt + problems


def current_link_ids(links: dict) -> tuple[set[str], set[str]]:
    """Only attach evidence whose committed links equal the current verified projection."""
    projection, _ = verified_projection(links)
    return tuple({k for k, entry in projection[table].items() if entry and links[table].get(k) == entry}
                 for table in ("passes", "records"))


def provenance_problems(links: dict) -> list[str]:
    projection, problems = verified_projection(links)
    if projection != links:
        problems.append("parity/evidence_links.json differs from its verified trace receipts")
    return problems


def cmd_link(args: argparse.Namespace) -> int:
    with trace_lock():
        stored = load_json(OUT)
        receipts = dict(stored.get("receipts", {}))
        for target in wanted_targets():
            if local := load_json(receipt_path(target)):
                receipts[target] = local
        current, debt = verified(receipts)
        data, problems = project(current)
        if args.check and stored != data:
            problems.append("parity/evidence_links.json differs from the current verified traces")
        for issue in debt + problems:
            print(f"evidence links: {issue}", file=sys.stderr)
        if debt or problems:
            return 1
        if not args.check:
            atomic_json(OUT, data)
        print(f"evidence links: verified {len(current)} target receipts")
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
