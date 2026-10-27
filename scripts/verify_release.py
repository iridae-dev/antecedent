#!/usr/bin/env python3
import json
import sys
from pathlib import Path

run = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (run.get("event"), run.get("headBranch"), run.get("conclusion"), run.get("workflowName")) != (
    "push",
    "main",
    "success",
    "ci",
):
    raise SystemExit("FAIL: release requires a successful main-push CI run")
