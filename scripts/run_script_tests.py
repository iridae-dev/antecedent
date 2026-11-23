# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mkdocs==1.6.1", "PyYAML==6.0.3", "numpy==2.1.3", "pytest==9.1.1"]
# ///
"""Run discovered script unit tests in a retained, isolated tooling environment.

Test modules run concurrently, one worker process per module, slowest modules
first. Each module's output is buffered and printed when that module finishes;
the receipt aggregates every worker's outcome in discovery order and has the
same content as a serial run.

No calibration or performance campaign is launched. Two native performance
receipt consumers remain explicitly pending unless their real probe is supplied.
Every other skip, missing discovery or failed assertion fails this command.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from _repo import REPO_ROOT
from test_run_receipt import checkout

PENDING_PROBE_CONSUMERS = {
    "test_measurement_record.RustEmittedMeasurementTest.test_the_native_summary_is_accepted_and_shows_the_work_that_ran",
    "test_measurement_record.RustEmittedMeasurementTest.test_the_native_summary_rides_on_a_criterion_comparison",
}

JOBS_ENV = "ANTECEDENT_SCRIPT_TEST_JOBS"
# Set in every worker process. A runner started inside a worker (a self-test that drives
# the runner on its own suite) runs in that process, so it never spawns workers of its own.
WORKER_ENV = "ANTECEDENT_SCRIPT_TEST_WORKER"
MAX_DEFAULT_JOBS = 8

# Modules whose runtime dominates the suite; scheduled before all others.
KNOWN_SLOW_MODULES = ("test_evidence_binding_registry",)


def default_jobs() -> int:
    configured = os.environ.get(JOBS_ENV)
    if configured:
        return int(configured)
    return min(os.cpu_count() or 1, MAX_DEFAULT_JOBS)


def schedule(modules: list[str]) -> list[str]:
    """Known-slow modules first, then the remaining modules alphabetically."""
    slow = [module for module in KNOWN_SLOW_MODULES if module in modules]
    return slow + sorted(module for module in modules if module not in slow)


def flatten(group):
    for test in group:
        if isinstance(test, unittest.TestSuite):
            yield from flatten(test)
        else:
            yield test


def module_of(test_id: str) -> str:
    return test_id.split(".")[0]


def summarize(result: unittest.TestResult) -> dict:
    return {
        "tests_run": result.testsRun,
        "failures": [test.id() for test, _ in result.failures],
        "errors": [test.id() for test, _ in result.errors],
        "expected_failures": [test.id() for test, _ in result.expectedFailures],
        "unexpected_successes": [test.id() for test in result.unexpectedSuccesses],
        "skipped": [[test.id(), reason] for test, reason in result.skipped],
    }


def run_worker(module: str, result_path: Path, verbosity: int) -> int:
    """Run one module's tests and write its outcome as JSON to `result_path`."""
    suite = unittest.defaultTestLoader.discover(
        str(REPO_ROOT / "scripts"), pattern=f"{module}.py"
    )
    cases = [test for test in flatten(suite) if module_of(test.id()) == module]
    result = unittest.TextTestRunner(verbosity=verbosity).run(unittest.TestSuite(cases))
    outcome = summarize(result) | {"discovered": [test.id() for test in cases]}
    result_path.write_text(json.dumps(outcome), encoding="utf-8")
    return 0


def run_module(module: str, verbosity: int, workdir: Path) -> tuple[str, dict | None]:
    """Run one module in a worker process; return its buffered output and outcome."""
    result_path = workdir / f"{module}.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        module,
        "--worker-result",
        str(result_path),
    ]
    if verbosity > 1:
        command.append("--verbose")
    completed = subprocess.run(
        command,
        env={**os.environ, WORKER_ENV: "1"},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    output = completed.stdout
    if completed.returncode != 0 or not result_path.exists():
        output += f"\nworker for {module} exited with status {completed.returncode}\n"
        return output, None
    return output, json.loads(result_path.read_text(encoding="utf-8"))


def run_suite(cases: list[unittest.TestCase], jobs: int, verbosity: int) -> dict:
    """Run `cases` module by module in worker processes and aggregate the outcomes."""
    if jobs == 1 or os.environ.get(WORKER_ENV) == "1":
        result = unittest.TextTestRunner(verbosity=verbosity).run(unittest.TestSuite(cases))
        return summarize(result) | {"successful": result.wasSuccessful()}
    by_module: dict[str, list[str]] = {}
    for test in cases:
        by_module.setdefault(module_of(test.id()), []).append(test.id())
    # Modules that failed to import have no loadable tests for a worker; their
    # import-error cases run in this process so they surface as errors.
    local = [test for test in cases if type(test).__module__.startswith("unittest.")]
    local_ids = {test.id() for test in local}
    for module in by_module:
        by_module[module] = [i for i in by_module[module] if i not in local_ids]
    modules = schedule([module for module, ids in by_module.items() if ids])
    outcomes: dict[str, dict | None] = {}
    print_lock = threading.Lock()

    with tempfile.TemporaryDirectory(prefix="script-tests-") as workdir:

        def task(module: str) -> None:
            output, outcomes[module] = run_module(module, verbosity, Path(workdir))
            with print_lock:
                print(f"===== {module} =====", flush=True)
                print(output, end="" if output.endswith("\n") else "\n", flush=True)

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(task, modules))

    local_result = unittest.TestResult()
    if local:
        local_result = unittest.TextTestRunner(verbosity=verbosity).run(
            unittest.TestSuite(local)
        )
    merged = summarize(local_result)
    successful = local_result.wasSuccessful()
    # Aggregate in discovery order so the receipt lists match a serial run.
    for module in modules_in_discovery_order(by_module):
        outcome = outcomes[module]
        if outcome is None:
            merged["errors"].append(f"{module}.<worker did not complete>")
            successful = False
            continue
        if sorted(outcome["discovered"]) != sorted(by_module[module]):
            missing = sorted(set(by_module[module]) - set(outcome["discovered"]))
            print(f"FAIL: worker for {module} did not discover: {missing}")
            merged["errors"].append(f"{module}.<worker discovery mismatch>")
            successful = False
        merged["tests_run"] += outcome["tests_run"]
        for key in ("failures", "errors", "expected_failures", "skipped"):
            merged[key].extend(outcome[key])
        merged["unexpected_successes"].extend(outcome["unexpected_successes"])
        if outcome["failures"] or outcome["errors"] or outcome["unexpected_successes"]:
            successful = False
    merged["successful"] = successful
    return merged


def modules_in_discovery_order(by_module: dict[str, list[str]]) -> list[str]:
    return [module for module, ids in by_module.items() if ids]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="test_*.py")
    parser.add_argument(
        "--exclude-module",
        action="append",
        default=[],
        help="Explicitly deselect a module executed by another job; recorded in the receipt",
    )
    parser.add_argument(
        "--report", type=Path, default=REPO_ROOT / "target/script-tests/latest.json"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help=f"Concurrent module workers (default: min(cpu count, {MAX_DEFAULT_JOBS}), "
        f"or ${JOBS_ENV})",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    verbosity = 2 if args.verbose else 1
    if args.worker:
        return run_worker(args.worker, args.worker_result, verbosity)
    jobs = args.jobs if args.jobs is not None else default_jobs()
    if jobs < 1:
        parser.error("--jobs must be at least 1")
    before = checkout(REPO_ROOT)
    suite = unittest.defaultTestLoader.discover(
        str(REPO_ROOT / "scripts"), pattern=args.pattern
    )
    discovered_cases = list(flatten(suite))
    excluded = set(args.exclude_module)
    unknown = excluded - {module_of(test.id()) for test in discovered_cases}
    if unknown:
        parser.error(f"excluded modules were not discovered: {sorted(unknown)}")
    deselected = [
        test.id() for test in discovered_cases if module_of(test.id()) in excluded
    ]
    selected = [test for test in discovered_cases if test.id() not in deselected]
    discovered = len(selected)
    outcome = run_suite(selected, jobs, verbosity)
    allowed = (
        PENDING_PROBE_CONSUMERS
        if os.environ.get("ANTECEDENT_REQUIRE_NATIVE_MEASUREMENT") != "1"
        else set()
    )
    unexpected = [test_id for test_id, _ in outcome["skipped"] if test_id not in allowed]
    after = checkout(REPO_ROOT)
    report = {
        "schema": 1,
        "activity": "ordinary_script_unit_tests",
        "pattern": args.pattern,
        "discovered_before_selection": len(discovered_cases),
        "discovered": discovered,
        "explicit_deselections": deselected,
        "executed": outcome["tests_run"],
        "failures": outcome["failures"],
        "errors": outcome["errors"],
        "expected_failures": outcome["expected_failures"],
        "skipped": [
            {
                "assertion": test_id,
                "reason": reason,
                "activity": "pending_external_performance_probe"
                if test_id in allowed
                else "unexpected_skip",
            }
            for test_id, reason in outcome["skipped"]
        ],
        "unexpected_skips": unexpected,
        "checkout_before": before,
        "checkout_after": after,
        "checkout_unchanged": before == after,
        "calibration_measurements_run": False,
        "performance_acceptance_run": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not discovered:
        print("FAIL: script test discovery selected no tests")
    if unexpected:
        print(f"FAIL: required script assertions skipped: {unexpected}")
    if outcome["expected_failures"]:
        print(
            "FAIL: required script assertions must pass without expected-failure waivers"
        )
    print(f"Script test execution receipt: {args.report}")
    return (
        0
        if discovered
        and outcome["successful"]
        and not unexpected
        and not outcome["expected_failures"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
