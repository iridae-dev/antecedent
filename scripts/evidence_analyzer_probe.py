#!/usr/bin/env python3
"""Probe the pinned compiler interface using disposable engineering fixtures only.

Does not run Antecedent tests, write evidence, or qualify the analyzer. The
inventory driver deliberately reports incomplete dependency information.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from _repo import load_toml
from evidence_projection import from_compiler_coverage, matches

ROOT = Path(__file__).resolve().parents[1]


def inventory(driver: Path, sysroot: str, directory: Path, source: str) -> tuple[dict[str, str], list[str], dict]:
    path = directory / "fixture.rs"
    path.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [str(driver), "--sysroot", sysroot, "--test", str(path),
         "--crate-name", "fixture", "--out-dir", str(directory),
         "-C", f"incremental={directory / 'incremental'}", "-Z", "query-dep-graph"],
        env={**os.environ, "RUSTC_BOOTSTRAP": "1"},
        check=True, capture_output=True, text=True,
    )
    items = {}
    bodies = {}
    for line in result.stdout.splitlines():
        if line.startswith("item\t"):
            _, name, expanded, source_hex = line.split("\t")
            items[name] = bytes.fromhex(source_hex).decode("utf-8")
        elif line.startswith("body\t"):
            _, name, path_hex, start, end, _text = line.split("\t")
            if Path(bytes.fromhex(path_hex).decode()).resolve() == path.resolve():
                raw = source.encode()
                bodies[name] = (len(raw[:int(start)].decode()), len(raw[:int(end)].decode()))
    if "used" not in items:
        raise RuntimeError("compiler inventory did not report the fixture's used function")
    return items, [line for line in result.stdout.splitlines() if line.startswith("edge\t")], bodies


def probe(root: Path = ROOT) -> dict:
    pin = load_toml(root / "tools/evidence-analyzer/pin.toml")
    version = subprocess.check_output(["rustc", "-vV"], text=True)
    identity = dict(line.split(": ", 1) for line in version.splitlines() if ": " in line)
    if identity.get("release") != pin["release"] or identity.get("commit-hash") != pin["commit"]:
        raise RuntimeError("compiler identity differs from tools/evidence-analyzer/pin.toml")
    sysroot = subprocess.check_output(["rustc", "--print", "sysroot"], text=True).strip()
    with tempfile.TemporaryDirectory(prefix="antecedent-analyzer-probe-") as tmp:
        directory = Path(tmp)
        driver = directory / "inventory-driver"
        subprocess.run(
            ["rustc", str(root / "tools/evidence-analyzer/driver.rs"),
             "--edition=2021", "-C", "rpath=yes", "-o", str(driver)],
            env={**os.environ, "RUSTC_BOOTSTRAP": "1"}, check=True, capture_output=True, text=True,
        )
        source = (
            "macro_rules! value { () => {2} }\n"
            "fn used() -> i32 { value!() }\n"
            "fn irrelevant() -> i32 { 7 }\n"
            "#[test] fn check() { assert_eq!(used(), 2); }\n"
        )
        baseline, edges, bodies = inventory(driver, sysroot, directory, source)
        unrelated, _, _ = inventory(driver, sysroot, directory, source.replace("{ 7 }", "{ 8 }"))
        changed_macro, _, _ = inventory(driver, sysroot, directory, source.replace("=> {2}", "=> {3}"))
        # Run only this disposable engineering fixture, never an Antecedent
        # assertion. Obtain execution evidence from the same pinned compiler.
        fixture = directory / 'fixture.rs'
        fixture.write_text(source, encoding="utf-8")
        binary = directory / 'fixture-test'
        subprocess.run(['rustc', '--test', str(fixture), '--crate-name', 'fixture',
                        '-C', 'instrument-coverage', '-o', str(binary)],
                       check=True, capture_output=True, text=True)
        raw = directory / 'fixture.profraw'
        subprocess.run([str(binary), '--exact', 'check'],
                       env={**os.environ, 'LLVM_PROFILE_FILE': str(raw)},
                       check=True, capture_output=True, text=True)
        llvm = Path(sysroot) / 'lib/rustlib' / identity['host'] / 'bin'
        profdata = directory / 'fixture.profdata'
        subprocess.run([str(llvm / 'llvm-profdata'), 'merge', '-sparse', str(raw),
                        '-o', str(profdata)], check=True, capture_output=True, text=True)
        export = json.loads(subprocess.check_output(
            [str(llvm / 'llvm-cov'), 'export', str(binary), f'-instr-profile={profdata}'], text=True))
        covered = set()
        for block in export['data']:
            for file in block['files']:
                if Path(file['filename']).resolve() == fixture.resolve():
                    covered.update(s[0] for s in file['segments'] if s[2] > 0 and s[3])
        projection, _ = from_compiler_coverage(source, bodies, covered, opaque=False)
        assert projection['layouts'], 'no unexecuted ordinary body was admitted'
        assert matches(source.replace("{ 7 }", "{ 8 }"), projection)
        assert not matches(source.replace("=> {2}", "=> {3}"), projection)
        assert not matches(source.replace("fn irrelevant()", "fn irrelevant(x: i32)"), projection)
        assert edges, "compiler query graph was not enabled"
        unmapped_resolution = any("resolver_for_lowering_raw" in edge and "<unmapped>" in edge for edge in edges)
        assert unmapped_resolution, "expected crate-scoped resolver query was not observed"
        assert baseline["used"] == unrelated["used"]
        assert baseline["irrelevant"] != unrelated["irrelevant"]
        # A source inventory alone is not a dependency relation: the used item's
        # spelling remains unchanged when its macro's expansion changes.
        assert baseline["used"] == changed_macro["used"]
        assert baseline["value"] != changed_macro["value"]
    return {
        "compiler": identity,
        "driver": "tools/evidence-analyzer/driver.rs",
        "observations": {
            "compiler_item_inventory_available": True,
            "same_file_unrelated_body_is_distinguishable": True,
            "macro_change_leaves_calling_item_source_unchanged": True,
            "compiler_query_edges_available": len(edges),
            "crate_resolution_has_no_item_identity": unmapped_resolution,
            "compiler_body_projection_ignores_unrelated_edit": True,
            "whole_declaration_guard_catches_macro_and_signature_changes": True,
            "projection_uses_real_instrumented_fixture_coverage": True,
        },
        "query_graph_gaps": [
            "resolve non-item query nodes and cross-crate dependency boundaries",
            "map expansion provenance and constant evaluation to input items",
            "guard lookup scopes, imports, implementation sets, cfg and new definitions",
            "bind build scripts, procedural macros and generated inputs to bounded fallbacks",
        ],
    }


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        report = probe()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(json.dumps({"error": str(error), "detail": getattr(error, "stderr", "")}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
