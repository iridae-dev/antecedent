#!/usr/bin/env python3
import json, sys
from pathlib import Path
tmp = Path(sys.argv[1])
green = json.loads((tmp / "green.json").read_text())
def write(name, payload):
    (tmp / name).write_text(json.dumps(payload))
# One matrix leg missing (the last wheel expansion).
wheels = [j for j in green["jobs"] if j["name"].startswith("Wheel ")]
write("missing_leg.json", {**green, "jobs": [j for j in green["jobs"] if j is not wheels[-1]]})
# A whole required job id absent.
write("missing_job.json", {**green, "jobs": [j for j in green["jobs"] if j["name"] != "Domain gates"]})
# A required job that failed.
write("failed_job.json", {**green, "jobs": [
    {**j, "conclusion": "failure"} if j["name"] == "Python lint and types" else j
    for j in green["jobs"]
]})
# Green, but for a different commit.
write("wrong_sha.json", {**green, "headSha": "0" * 40})
# Jobs reported by workflow id instead of display name.
write("ids_not_names.json", {**green, "jobs": [
    {"name": n, "conclusion": "success"} for n in ("rust", "gates", "python-lint", "python-wheels")
]})
