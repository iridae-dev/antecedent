"""Portable receipts for the smoke execution that establishes evidence-to-unit links.

A receipt records successful libtest observations, including each tally's originating
test. Its source identity covers the target, transitive crate sources, fixtures, build
configuration and the trace reader. The release gate can replay the join without Cargo
test executables or a developer's target directory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import evidence_fingerprint as ef


def digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


class TraceInputs(ef.Fingerprinter):
    """Hash the bytes the trace build sees, including the embedded support decisions."""

    def __init__(self, root: Path):
        super().__init__(root, platform=False, scientific_build=True)
        self.libraries: dict[str, str] = {}
        self.file_states: dict[Path, tuple | None] = {}

    @staticmethod
    def file_state(path: Path) -> tuple | None:
        try:
            info = path.stat()
        except FileNotFoundError:
            return None
        return info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    def check_unchanged(self) -> None:
        for path, state in self.file_states.items():
            if self.file_state(path) != state:
                raise ValueError(f"{path}: trace inputs changed since the build started")

    def _digest(self, path: Path) -> str:
        if path not in self._fd:
            self.file_states[path] = self.file_state(path)
            if path.is_file():
                with path.open("rb") as src:
                    self._fd[path] = hashlib.file_digest(src, "sha256").hexdigest()
            else:
                self._fd[path] = "missing"
        return self._fd[path]

    def library_inputs(self, crate: str) -> str:
        if crate not in self.libraries:
            scan = self.crate_scan(crate)
            declared, _ = self.declared(scan.dynamic)
            self.libraries[crate] = digest([self.crate_digest(crate), declared])
        return self.libraries[crate]

    def target_key(self, crate: str, source: Path, features: tuple[str, ...],
                   selection: list[str], profile: list[str], env: dict[str, str]) -> str:
        scan = self.scan_rust([source], self.crate_dir(crate), test_build=True)
        declared, _ = self.declared(scan.dynamic)
        config = [self.root / p for p in (
            "Cargo.toml", "Cargo.lock", "rust-toolchain", "rust-toolchain.toml",
            ".cargo/config", ".cargo/config.toml", "scripts/evidence_links.py",
            "scripts/evidence_trace_receipts.py", "scripts/evidence_fingerprint.py",
        )]
        return digest({
            "workspace": self._tree(config),
            "crates": {c: self.library_inputs(c) for c in ef.closure(self.graph, crate)},
            "test": self._tree([source, *scan.files]), "markers": sorted(scan.markers),
            "declared_inputs": declared, "features": features, "profile": profile,
            "environment": env, "ignored_tests_selected": selection,
        })


def observations(path: Path, binary: str, passed: set[str]) -> dict:
    """Read one successful target. Distinct packages may use the same libtest binary name."""
    tests = {name: {} for name in sorted(passed)}
    routes: dict[str, set[str]] = {}
    intervals: dict[str, set[str]] = {}
    if path.is_file():
        with path.open(encoding="utf-8") as source:
            for line in source:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 4 or fields[1] != binary or fields[2] not in tests:
                    raise ValueError(f"{path.name}: event has no successful test in this binary: {fields[:3]}")
                op, _, name, *args = fields
                test = tests[name]
                if op in ("output", "output-v2") and len(args) == 1:
                    continue
                if op == "unit" and len(args) == 3 and args[2] in ("supported", "not_supported", "not_applicable"):
                    field = "executed" if args[2] == "supported" else "refused"
                    test.setdefault(field, set()).add("\t".join(args[:2]))
                elif op == "declare" and len(args) == 2:
                    test.setdefault("declared", set()).add("\t".join(args))
                elif op == "interval" and len(args) == 3:
                    intervals.setdefault("\t".join(args[:2]), set()).add(args[2])
                elif op == "record" and len(args) % 2 == 1:
                    units = test.setdefault("records", {}).setdefault(args[0], set())
                    units.update("\t".join(args[i:i + 2]) for i in range(1, len(args), 2))
                elif op == "route" and len(args) >= 3 and len(args) % 2 == 1:
                    units = ["\t".join(args[i:i + 2]) for i in range(1, len(args), 2)]
                    routes.setdefault(args[0], set()).update(u for u in units if u != "-\t-")
                    if units[0] != "-\t-":
                        test.setdefault("returned", set()).add(units[0])
                else:
                    raise ValueError(f"{path.name}: malformed {op} event for {name}")
    for test in tests.values():
        for key in test:
            if key == "records":
                test[key] = {rid: sorted(units) for rid, units in test[key].items()}
            else:
                test[key] = sorted(test[key])
    return {"tests": tests, "routes": {k: sorted(v) for k, v in routes.items()},
            "intervals": {k: sorted(v) for k, v in intervals.items()}}
