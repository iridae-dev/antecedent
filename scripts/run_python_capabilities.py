#!/usr/bin/env python3
"""Execute manifest-selected journeys with a source-bound native build receipt.

Integration creates the receipt after its serialized native build (see the pass README).
This runner does not build, install, or infer native freshness from timestamps.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tomllib
from pathlib import Path

from check_python_capabilities import MANIFEST, ROOT, digest, source_digest


class JourneyRecorder:
    def __init__(self, selectors: list[str]) -> None:
        self.selectors = selectors
        self.collected: list[str] = []
        self.tests: dict[str, dict] = {}

    def pytest_collection_finish(self, session) -> None:
        self.collected = [item.nodeid for item in session.items]

    def pytest_runtest_logreport(self, report) -> None:
        selector = next(
            (
                item
                for item in self.selectors
                if report.nodeid == item or report.nodeid.startswith(item + "[")
            ),
            None,
        )
        if selector is None:
            return
        record = self.tests.setdefault(
            report.nodeid,
            {"nodeid": report.nodeid, "selector": selector, "outcome": "incomplete"},
        )
        if (
            report.failed
            or report.skipped
            or report.when == "call"
            and record["outcome"] == "incomplete"
        ):
            record["outcome"] = report.outcome
        if hasattr(report, "wasxfail"):
            record["outcome"] = "xfail"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = tomllib.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    selectors = sorted(
        {
            item["test"]
            for capability in manifest.get("capability", [])
            if capability["status"] in {"covered", "plumbing"}
            for item in capability.get("evidence", [])
        }
    )
    if not selectors:
        parser.error("manifest selects no executing tests")
    native = importlib.import_module("antecedent._native")
    native_path = Path(native.__file__).resolve()
    receipt = json.loads(args.build_receipt.read_text(encoding="utf-8"))
    source = source_digest(ROOT)
    native_hash = digest(native_path)
    if receipt.get("source_digest") != source or receipt.get("native_sha256") != native_hash:
        parser.error("stale source/native build receipt; rebuild and bind the actual extension")
    if Path(receipt.get("native_path", "")).resolve() != native_path:
        parser.error("receipt names a different loaded native extension")
    if receipt.get("python") != list(sys.version_info[:2]) or not receipt.get("build_command"):
        parser.error("receipt lacks matching Python runtime or concrete build command")
    import pytest

    os.chdir(ROOT)
    recorder = JourneyRecorder(selectors)
    manifest_hash = digest(ROOT / MANIFEST)
    os.environ.pop("PYTEST_ADDOPTS", None)
    code = int(pytest.main(["-q", "-o", "addopts=", *selectors], plugins=[recorder]))
    tests = list(recorder.tests.values())
    # Include every collected item; a deselected/unrun parameter cannot disappear.
    for nodeid in recorder.collected:
        if nodeid not in recorder.tests:
            selector = next((s for s in selectors if nodeid == s or nodeid.startswith(s + "[")), "")
            tests.append({"nodeid": nodeid, "selector": selector, "outcome": "incomplete"})
    changed = (
        source_digest(ROOT) != source
        or digest(native_path) != native_hash
        or digest(ROOT / MANIFEST) != manifest_hash
    )
    payload = {
        "schema": 1,
        "source_digest": source,
        "checker_sha256": digest(ROOT / "scripts/check_python_capabilities.py"),
        "runner_sha256": digest(ROOT / "scripts/run_python_capabilities.py"),
        "manifest_sha256": manifest_hash,
        "native_path": str(native_path),
        "native_sha256": native_hash,
        "build_receipt_verified": not changed,
        "build_receipt": receipt,
        "python": list(sys.version_info[:2]),
        "selected": selectors,
        "collected": recorder.collected,
        "tests": tests,
        "exit_code": code if not changed else 1,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload["exit_code"] if tests else 1


if __name__ == "__main__":
    sys.exit(main())
