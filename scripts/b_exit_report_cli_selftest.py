#!/usr/bin/env python3
"""CLI regressions: never report a different registry or silently ignore flags."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/b_exit_report.py"), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> int:
    retained = run("--list-tests")
    selected = run("--promotion", "parity/promotion_2_2.toml", "--list-tests")
    assert retained.returncode == selected.returncode == 0
    assert retained.stdout == selected.stdout
    current = run("--promotion", "parity/promotion_2_3.toml")
    assert current.returncode == 0
    assert "2.3: parity/promotion_2_3.toml" in current.stdout
    assert "2.2 B exit gate" not in current.stdout
    unknown = run("--promotion", "parity/promotion_2_3.toml", "--unrecognized")
    assert unknown.returncode == 2 and "unrecognized arguments" in unknown.stderr
    missing = run("--promotion")
    assert missing.returncode == 2 and "expected one argument" in missing.stderr
    with tempfile.TemporaryDirectory() as temporary:
        alternate = Path(temporary) / "different.toml"
        alternate.write_text((ROOT / "parity/promotion_2_3.toml").read_text())
        refused = run("--promotion", str(alternate))
        assert refused.returncode == 2 and "canonical" in refused.stderr
    for flag in ("--release", "--require-calibrated", "--require-implemented"):
        refused = run("--promotion", "parity/promotion_2_3.toml", flag)
        assert refused.returncode == 1 and "does not certify" in refused.stderr
    incompatible = run("--promotion", "parity/promotion_2_3.toml", "--list-tests")
    assert incompatible.returncode == 2 and "cannot evidence 2.3" in incompatible.stderr
    checked = run("--promotion", "parity/promotion_2_3.toml", "--self-test")
    assert checked.returncode == 0
    assert "b_exit_report self-test: ok" in checked.stdout
    assert "release_evidence_report self-test: ok" in checked.stdout
    assert "Promotion evidence inventory" not in checked.stdout
    print("b_exit_report CLI self-test: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
