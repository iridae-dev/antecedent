"""Check grouped citation execution and its fail-closed cases with fake runners."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

runner = Path(__file__).with_name("run_evidence_rows.py")

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    bin_dir = root / "bin"
    bin_dir.mkdir()
    cargo = bin_dir / "cargo"
    cargo.write_text(
        """#!/usr/bin/env python3
import os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['FAKE_CALLS']).open('a') as out: out.write(' '.join(args) + '\\n')
if os.environ.get('CACHE_TIMEOUT') == '1' and os.environ.get('RUSTC_WRAPPER') == 'sccache':
    print('sccache: error: Timed out waiting for server startup. Maybe the remote service is unreachable?', file=sys.stderr)
    sys.exit(2)
if '--list' in args:
    if '--ignored' not in args and os.environ.get('MISSING') != '1': print('evidence_test: test')
    if '--ignored' in args and os.environ.get('IGNORED') == '1': print('evidence_test: test')
elif 'nextest' in args:
    count = 0 if os.environ.get('EMPTY') == '1' else 1
    if os.environ.get('COLOR') == '1':
        print(f'\x1b[32;1mSummary\x1b[0m \x1b[1m1\x1b[0m test run: \x1b[1m{count}\x1b[0m \x1b[32;1mpassed\x1b[0m')
    else:
        print(f'Summary 1 test run: {count} passed, {1-count} skipped')
""",
        encoding="utf-8",
        newline="\n",
    )
    cargo.chmod(0o755)
    (bin_dir / "cargo-nextest").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8", newline="\n")
    (bin_dir / "cargo-nextest").chmod(0o755)
    (root / "crates/example/src").mkdir(parents=True)
    (root / "crates/example/Cargo.toml").write_text(
        '[package]\nname = "example"\nversion = "0.1.0"\nedition = "2021"\n',
        encoding="utf-8",
        newline="\n",
    )
    (root / "crates/example/src/lib.rs").write_text(
        "#[test] fn evidence_test() {}\n", encoding="utf-8", newline="\n"
    )
    (root / "evidence.toml").write_text(
        """[[fixture_evidence]]
id = "first"
evidence_test = "crates/example/src/lib.rs"
evidence_assertion = "evidence_test"
[[fixture_evidence]]
id = "second"
evidence_test = "crates/example/src/lib.rs"
evidence_assertion = "evidence_test"
""",
        encoding="utf-8",
        newline="\n",
    )
    calls = root / "calls.log"
    base = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "FAKE_CALLS": str(calls),
    }

    def run(**changes):
        calls.write_text("", encoding="utf-8", newline="\n")
        return subprocess.run(
            [
                sys.executable,
                str(runner),
                str(root),
                "evidence.toml",
                str(root),
                "fixture_evidence",
                "self_test",
            ],
            env={**base, **changes},
            capture_output=True,
            text=True,
        )

    good = run()
    assert good.returncode == 0, good.stdout + good.stderr
    assert "ok: first" in good.stdout and "ok: second" in good.stdout
    assert (
        sum("nextest run" in line for line in calls.read_text(encoding="utf-8").splitlines()) == 1
    )
    colored = run(COLOR="1")
    assert colored.returncode == 0, colored.stdout + colored.stderr
    assert sum("--exact" in line for line in calls.read_text(encoding="utf-8").splitlines()) == 0
    recovered = run(CACHE_TIMEOUT="1", RUSTC_WRAPPER="sccache")
    assert recovered.returncode == 0 and "retrying without wrapper" in recovered.stdout
    for setting, expected in (
        ("IGNORED", "ignored"),
        ("MISSING", "no test"),
        ("EMPTY", "passed=0"),
    ):
        bad = run(**{setting: "1"})
        assert bad.returncode != 0 and expected in bad.stdout, (setting, bad.stdout, bad.stderr)
        if setting == "EMPTY":
            assert "self_test: 2 failing row(s): first, second" in bad.stdout, bad.stdout

    (root / "parity").mkdir()
    registry = root / "parity/v3_test_registry.toml"
    registry.write_text(
        '[[test]]\npath="crates/example/src/lib.rs"\ncrate="example"\n'
        'cargo_target="lib"\ncalibration_simulation_assertions=["evidence_test"]\n',
        encoding="utf-8",
    )
    excluded = run()
    assert excluded.returncode != 0 and "INCOMPLETE" in excluded.stdout, excluded.stdout + excluded.stderr
    assert "nextest run" not in calls.read_text(encoding="utf-8"), "excluded simulation was executed"
    opted_in = run(ANTECEDENT_RUN_CALIBRATION_SIMULATIONS="1")
    assert opted_in.returncode == 0, opted_in.stdout + opted_in.stderr
    assert "--ignore-default-filter" in calls.read_text(encoding="utf-8")
    (root / "crates/example/src/lib.rs").write_text(
        "#[test] fn evidence_test() {}\n#[test] fn simulation_test() {}\n", encoding="utf-8"
    )
    registry.write_text(
        '[[test]]\npath="crates/example/src/lib.rs"\ncrate="example"\n'
        'cargo_target="lib"\ncalibration_simulation_assertions=["simulation_test"]\n',
        encoding="utf-8",
    )
    (root / "evidence.toml").write_text(
        '[[fixture_evidence]]\nid="filtered"\nevidence_test="crates/example/src/lib.rs"\n'
        'evidence_assertion="evidence_test"\ncomposition_filter="simulation"\n', encoding="utf-8"
    )
    filtered = run()
    assert filtered.returncode != 0 and "filter 'simulation' includes a calibration simulation" in filtered.stdout
    assert "-E simulation" not in calls.read_text(encoding="utf-8")
    registry.unlink()

    uv = bin_dir / "uv"
    uv.write_text(
        """#!/usr/bin/env python3
import os
if os.environ.get('PY_SKIP') == '1':
    print('1 skipped in 0.01s')
else:
    print('PASSED tests/test_evidence.py::test_evidence')
    print('1 passed in 0.01s')
""",
        encoding="utf-8",
        newline="\n",
    )
    uv.chmod(0o755)
    (root / "python/tests").mkdir(parents=True)
    (root / "evidence.toml").write_text(
        """[[fixture_evidence]]
id = "python_first"
evidence_test = "python/tests/test_evidence.py"
evidence_assertion = "test_evidence"
[[fixture_evidence]]
id = "python_second"
evidence_test = "python/tests/test_evidence.py"
evidence_assertion = "test_evidence"
""",
        encoding="utf-8",
        newline="\n",
    )
    good_python = run()
    assert good_python.returncode == 0 and "ok: python_second" in good_python.stdout
    skipped = run(PY_SKIP="1")
    assert skipped.returncode != 0 and "python_first" in skipped.stdout
    (root / "python/tests/test_evidence.py").write_text(
        "import pytest\n@pytest.mark.calibration_simulation\ndef test_evidence(): pass\n", encoding="utf-8"
    )
    excluded_python = run()
    assert excluded_python.returncode != 0 and "INCOMPLETE" in excluded_python.stdout
    assert "cited Python test executed" not in excluded_python.stdout

print("grouped evidence self-test: ok")
