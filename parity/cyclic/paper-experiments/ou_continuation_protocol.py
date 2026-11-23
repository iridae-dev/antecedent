# SPDX-License-Identifier: MIT OR Apache-2.0
"""Continue every budget-limited OU fit without changing data, parameters or score tolerance."""

import argparse
import concurrent.futures
import json
import runpy
import time
from pathlib import Path

from antecedent.cyclic import PreparedCyclicFamily
from antecedent.outcomes import require_success

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()
if not 1 <= args.workers <= 4:
    parser.error("workers must be in 1..4")
module = runpy.run_path(
    str(
        Path(__file__).resolve().parents[3]
        / "examples/python/cyclic_paper_experiments.py"
    )
)
base = args.base_dir.resolve()
out = args.output_dir.resolve()
out.mkdir(exist_ok=True, parents=True)
jobs = []
for seed in range(5):
    for path in sorted(
        (base / f"seed-{seed}").glob("single-contexts-*/native_report.json")
    ):
        report = json.loads(path.read_text())
        fit = report["answer"]["result"]["fit"]
        if not fit["converged"]:
            jobs.append((seed, path.parent, fit))


def fit_job(job):
    seed, path, old = job
    start = time.monotonic()
    request = json.loads((path / "request.json").read_text())
    request["model"] = old["model"]
    request["policy"]["max_iterations"] = 20000
    prepared = PreparedCyclicFamily(json.dumps(request, allow_nan=False))
    result = require_success(prepared.execute())
    report_dict = result.to_dict()
    fit = report_dict["answer"]["result"]["fit"]
    target = out / f"seed-{seed}-{path.name}"
    target.mkdir(exist_ok=True)
    module["write"](target / "request.json", request)
    module["write"](target / "native_report.json", report_dict)
    record = {
        "seed": seed,
        "case": path.name,
        "approach": "continuation_LBFGS_20000_steps_same_score_tolerance_no_heldout_selection",
        "original_identity": json.loads((path / "native_report.json").read_text())[
            "identity"
        ],
        "original_score_norm": old["score_norm"],
        "converged": fit["converged"],
        "score_norm": fit["score_norm"],
        "iterations": fit["iterations"],
        "line_search_exhausted": fit["line_search_exhausted"],
        "failed_worlds": sum(
            "failure" in law for law in report_dict["answer"]["result"]["laws"]
        ),
        "artifact": module["replay"](result, target),
        "wall_seconds": time.monotonic() - start,
    }
    module["write"](target / "metrics.json", record)
    print(json.dumps(record), flush=True)
    return record


records = []
approach = "bounded continuation of every budget-limited base fit, preserving the same training data and score tolerance"
with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
    for record in pool.map(fit_job, jobs):
        records.append(record)
        module["write"](out / "results.json", {"approach": approach, "cases": records})
# With no budget-limited base fit the record is empty, not absent.
module["write"](out / "results.json", {"approach": approach, "cases": records})
