"""Performance probes must execute complete workflows and fail on a broken import."""

from __future__ import annotations

import json
import os
import pstats
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


def test_wide_gaussian_probe_profiles_complete_prepared_calls(repo_root, tmp_path):
    output = tmp_path / "wide.json"
    profile = tmp_path / "prepared.prof"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "benchmarks/performance/python_workflows.py"),
            "--rows",
            "512",
            "--covariates",
            "30",
            "--family",
            "gaussian_bayes",
            "--draws",
            "16",
            "--repeats",
            "2",
            "--profile",
            str(profile),
            "--profile-repeats",
            "2",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["width"] == 32
    assert len(report["cells"]) == 2
    for cell in report["cells"]:
        assert len(cell["samples"]) == 2
        assert all(sample["posterior_draws"] == 16 for sample in cell["samples"])
    assert pstats.Stats(str(profile)).total_calls > 0


def test_capability_preflight_checks_science_without_collecting_timings(repo_root, tmp_path):
    script = repo_root / "benchmarks/performance/capability_workflows.py"
    names = subprocess.check_output([sys.executable, str(script)], text=True).splitlines()
    output = tmp_path / "capabilities.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--validate",
            "--rows",
            "512",
            "--units",
            "8",
            "--grid-points",
            "9",
            "--draws",
            "32",
            "--output",
            str(output),
            *[part for name in names for part in ("--case", name)],
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert report["timed"] is False
    assert report["warmups"] == 0
    assert {cell["capability"] for cell in report["cells"]} == set(names)
    assert len(report["cells"]) == 24
    for cell in report["cells"]:
        assert "median_ns" not in cell
        assert len(cell["samples"]) == 1
        assert "elapsed_ns" not in cell["samples"][0]
        assert cell["samples"][0]["observed"]
