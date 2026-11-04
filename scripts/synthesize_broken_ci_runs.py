#!/usr/bin/env python3
"""Write broken variants of a green CI run (one matrix leg missing, one required job absent or
failed, the wrong commit, jobs reported by id) so the release candidate's CI-run check can be
shown to reject each: `python3 scripts/synthesize_broken_ci_runs.py DIR` reads DIR/green.json."""

import json
import sys
from pathlib import Path

tmp = Path(sys.argv[1])
green = json.loads((tmp / "green.json").read_text(encoding="utf-8"))


def write(name, payload):
    (tmp / name).write_text(json.dumps(payload), encoding="utf-8", newline="\n")


# One matrix leg missing (the last wheel expansion).
wheels = [j for j in green["jobs"] if j["name"].startswith("Wheel ")]
write("missing_leg.json", {**green, "jobs": [j for j in green["jobs"] if j is not wheels[-1]]})
# A whole required job id absent.
write(
    "missing_job.json", {**green, "jobs": [j for j in green["jobs"] if j["name"] != "Static checks"]}
)
# A required job that failed.
write(
    "failed_job.json",
    {
        **green,
        "jobs": [
            {**j, "conclusion": "failure"} if j["name"] == "Python lint and types" else j
            for j in green["jobs"]
        ],
    },
)
# Green, but for a different commit.
write("wrong_sha.json", {**green, "headSha": "0" * 40})
# Jobs reported by workflow id instead of display name.
write(
    "ids_not_names.json",
    {
        **green,
        "jobs": [
            {"name": n, "conclusion": "success"}
            for n in ("static", "rust", "scientific", "python-lint", "python-wheels")
        ],
    },
)
