#!/usr/bin/env python3
"""CLI regressions: the decision-exit report reads the one scientific registry."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from _repo import REPO_ROOT as ROOT


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/decision_exit_report.py"), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> int:
    retained = run("--list-tests")
    selected = run("--promotion", "parity/promotion.toml", "--list-tests")
    assert retained.returncode == selected.returncode == 0, retained.stderr + selected.stderr
    assert retained.stdout == selected.stdout
    unknown = run("--promotion", "parity/promotion.toml", "--unrecognized")
    assert unknown.returncode == 2 and "unrecognized arguments" in unknown.stderr
    missing = run("--promotion")
    assert missing.returncode == 2 and "expected one argument" in missing.stderr
    with tempfile.TemporaryDirectory() as temporary:
        alternate = Path(temporary) / "different.toml"
        alternate.write_text((ROOT / "parity/promotion.toml").read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        refused = run("--promotion", str(alternate))
        assert refused.returncode == 2 and "parity/promotion.toml" in refused.stderr
    checked = run("--self-test")
    assert checked.returncode == 0, checked.stderr
    assert "decision_exit_report self-test: ok" in checked.stdout
    print("decision_exit_report CLI self-test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
