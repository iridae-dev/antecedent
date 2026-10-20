"""Performance probes must execute complete workflows and fail on a broken import."""

from __future__ import annotations

import json
import os
import subprocess
import sys


def test_python_performance_probe_refuses_import_failure(repo_root, tmp_path):
    package = tmp_path / "antecedent"
    package.mkdir()
    (package / "__init__.py").write_text("raise ImportError('stale_native_fixture')\n")
    output = tmp_path / "result.json"
    process = subprocess.run(
        [
            sys.executable,
            str(repo_root / "benchmarks/performance/python_workflows.py"),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert "preflight failed" in process.stderr
    assert "stale_native_fixture" in process.stderr
    assert "maturin develop --release" in process.stderr
    assert not output.exists()


def test_python_performance_probe_keeps_all_repeats_and_uncertainty(repo_root, tmp_path):
    output = tmp_path / "result.json"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "benchmarks/performance/python_workflows.py"),
            "--rows",
            "128",
            "--repeats",
            "2",
            "--bootstrap",
            "8",
            "--draws",
            "16",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["scope"] == "diagnostic_subset_not_full_N20_acceptance"
    assert len(report["native_sha256"]) == 64
    assert len(report["cells"]) == 4
    for cell in report["cells"]:
        assert [r["repeat"] for r in cell["samples"]] == [0, 1]
        for row in cell["samples"]:
            assert not row["cancelled"]
            assert not row["early_stopped"]
            if cell["family"] == "bootstrap":
                assert row["bootstrap_ok"] == 8
                assert row["bootstrap_se"] > 0
            else:
                assert row["posterior_draws"] == 16
                if cell["mode"] == "prepared_estimate":
                    assert row["design_reused"]
