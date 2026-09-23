#!/usr/bin/env python3
"""Relative bench regression check over a Criterion baseline comparison.

    python3 scripts/bench_regression.py [CRITERION_DIR]

Run after `cargo bench ... --save-baseline base` on the base commit and
`cargo bench ... --baseline base` on the head commit, on the same machine in the
same job, so absolute wall times (the Apple-M1 numbers in benches/baselines/) never
enter. Criterion writes `<id>/change/estimates.json` for each benchmark of the
comparison; `mean.confidence_interval.lower_bound` is the lower end of the
confidence interval of the relative change of the mean. The check fails when that
lower bound exceeds +25 % (the interval excludes noise: the head is slower by more
than a quarter) and warns when the point estimate is beyond +10 % with a lower
bound above zero. A run that compared nothing fails: an empty comparison is not a
pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FAIL_LOWER_BOUND = 0.25
WARN_POINT = 0.10


def main(argv: list[str]) -> int:
    criterion = Path(argv[0]) if argv else Path("target/criterion")
    estimates = sorted(criterion.glob("**/change/estimates.json"))
    if not estimates:
        print(f"FAIL: no baseline comparison under {criterion} (was --baseline base used?)")
        return 1
    failures, warnings = [], []
    for path in estimates:
        mean = json.loads(path.read_text())["mean"]
        lower = float(mean["confidence_interval"]["lower_bound"])
        point = float(mean["point_estimate"])
        bench = "/".join(path.parent.parent.relative_to(criterion).parts)
        if lower > FAIL_LOWER_BOUND:
            failures.append(f"{bench}: {point:+.1%} (lower bound {lower:+.1%})")
        elif point > WARN_POINT and lower > 0.0:
            warnings.append(f"{bench}: {point:+.1%} (lower bound {lower:+.1%})")
    for line in warnings:
        print(f"WARN: slower than the base: {line}")
    if failures:
        print(f"FAIL: {len(failures)} benchmark(s) more than {FAIL_LOWER_BOUND:.0%} slower:")
        for line in failures:
            print(f"  {line}")
        return 1
    print(f"bench regression: {len(estimates)} benchmark(s) compared, none beyond the bound")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
