#!/usr/bin/env python3
"""Fail closed on the 3.0 test registry.

Every row in parity/v3_test_registry.toml must name a source file that exists.
A crate with autotests = false compiles only src/ unit tests, [[test]] paths,
and files a suite_NN.rs aggregator includes with #[path]; an included file must
also be listed in that crate's tests/suite_map.txt. A registry that points at
nothing does not pass.

Gates that already cite a path (checked_execution, support, evidence,
composition, transport, decision) require evidence_path, and that file must
exist. A stamped evidence file (a line ``# v3-evidence``, or any ``revision =
"<sha>"``) must carry revision equal to git HEAD; any other sha is stale, and
a stamp without revision is a missing required key. inventory consumes this
script. named consumes a compiled cargo test, so the source file is its evidence.

    python3 scripts/check_v3_test_registry.py
    python3 scripts/check_v3_test_registry.py --self-test
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import tomllib
from _repo import REPO_ROOT as ROOT
from test_evidence import mask_rust, source_masked, target_files, registered_calibration_acceptance

REGISTRY = "parity/v3_test_registry.toml"
KNOWN_GATES = (
    "inventory",
    "checked_execution",
    "support",
    "evidence",
    "composition",
    "transport",
    "decision",
    "named",
)
# These gates already refuse a citation that does not name a real path.
EVIDENCE_REQUIRED = {
    "checked_execution",
    "support",
    "evidence",
    "composition",
    "transport",
    "decision",
}
SHA = re.compile(r"[0-9a-f]{40}")
REVISION = re.compile(r'(?m)^revision\s*=\s*"(.*)"\s*$')
STAMP = re.compile(r"(?m)^#\s*v3-evidence\s*$")
PATH_ATTR = re.compile(r'#\[path\s*=\s*"([^"]+)"\]')
TEST_ATTR = re.compile(r"#\[(?:\w+::)?test\b|proptest!\s*\{|#\[rstest\]|#\[test_case\b")
IGNORED = re.compile(r"#\[ignore\b")
PY_TEST = re.compile(r"(?m)^\s*(?:async\s+)?def test_|^class Test")
# Directories whose top-level test files must each have a registry row when the
# registry declares ``complete = true``.
COMPLETE_GLOBS = (
    "crates/*/tests/*.rs",
    "crates/*/src/**/*.rs",
    "python/src/**/*.rs",
    "python/tests/test_*.py",
    "scripts/test_*.py",
)


def head_sha(repo: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    sha = proc.stdout.strip()
    if proc.returncode != 0 or not SHA.fullmatch(sha):
        return None
    return sha


def inside(root: Path, rel: str) -> Path | None:
    if not isinstance(rel, str) or not rel or rel.startswith(("/", "\\")):
        return None
    parts = Path(rel).parts
    if ".." in parts:
        return None
    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    return path


def suite_map(crate_dir: Path) -> dict[str, tuple[str, list[str]]]:
    """suite name -> (exact line, stems) from tests/suite_map.txt."""
    path = crate_dir / "tests" / "suite_map.txt"
    found: dict[str, tuple[str, list[str]]] = {}
    if not path.is_file():
        return found
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        name, rest = line.split(":", 1)
        stems = [part.strip() for part in rest.split(",") if part.strip()]
        found[name.strip()] = (line, stems)
    return found


def compiled_by(crate_dir: Path, manifest: dict) -> dict[str, tuple[str, str]]:
    """crate-relative path -> (cargo_target, kind) where kind is 'direct' or 'suite'.

    A ``#[path]`` include executes only when its ``suite_*.rs`` file is itself a
    ``[[test]]`` target. An unbuilt suite does not count.
    """
    found: dict[str, tuple[str, str]] = {}
    suites: list[tuple[str, str]] = []
    for entry in manifest.get("test", []):
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            continue
        rel = entry.get("path", f"tests/{name}.rs")
        if not isinstance(rel, str):
            continue
        found[rel] = (name, "direct")
        if rel.endswith(".rs"):
            suites.append((name, rel))
    for name, rel in suites:
        suite = crate_dir / rel
        if not suite.is_file():
            continue
        for included in target_files(suite.resolve(), crate_dir.resolve()):
            if included == suite.resolve() or not included.is_relative_to(crate_dir.resolve()):
                continue
            kind = "suite" if suite.name.startswith("suite_") else "included"
            found.setdefault(included.relative_to(crate_dir.resolve()).as_posix(), (name, kind))
    return found


def evidence_problems(root: Path, entry_id: str, rel: object, head: str | None) -> list[str]:
    if not isinstance(rel, str) or not rel:
        return [f"FAIL: {entry_id}: evidence_path is empty"]
    path = inside(root, rel)
    if path is None:
        return [f"FAIL: {entry_id}: evidence path {rel} escapes the repo"]
    if not path.is_file():
        return [f"FAIL: {entry_id}: evidence path {rel} does not exist"]
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".toml":
        try:
            parsed = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            if STAMP.search(text) or re.search(r"(?m)^[ \t]*revision[ \t]*=", text):
                return [f"FAIL: {entry_id}: evidence {rel} revision is stale"]
            return []
        revisions = []
        def collect(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "revision":
                        revisions.append(item)
                    else:
                        collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)
        collect(parsed)
    else:
        revisions = REVISION.findall(text)
    if STAMP.search(text) and not revisions:
        return [f"FAIL: {entry_id}: evidence {rel} missing required key revision"]
    if not revisions:
        return []
    sha = revisions[0]
    if len(revisions) != 1 or not isinstance(sha, str) or not SHA.fullmatch(sha) or sha != head:
        return [f"FAIL: {entry_id}: evidence {rel} revision is stale"]
    return []


def runnable_python(text: str) -> bool:
    """A pytest test, or a script that runs itself (a gate or a self-test)."""
    return bool(PY_TEST.search(text)) or "__main__" in text or "unittest" in text


def executes_tests(text: str, path: Path | None = None) -> bool:
    """True when the source carries at least one test that is not ignored."""
    skeleton = source_masked(path.resolve()).skeleton if path is not None else mask_rust(text).skeleton
    if len(TEST_ATTR.findall(skeleton)) > len(IGNORED.findall(skeleton)):
        return True
    return path is not None and any(
        len(TEST_ATTR.findall(s)) > len(IGNORED.findall(s))
        for p in target_files(path) if p != path.resolve()
        for s in [source_masked(p).skeleton]
    )


def unregistered(root: Path, registered: set[str]) -> list[str]:
    """Test files that exist but have no registry row."""
    found: list[str] = []
    for pattern in COMPLETE_GLOBS:
        for path in sorted(root.glob(pattern)):
            rel = path.relative_to(root).as_posix()
            if rel in registered:
                continue
            if path.name.startswith("suite_") or path.name == "__init__.py":
                continue
            text = (source_masked(path.resolve()).skeleton if path.suffix == ".rs"
                    else path.read_text(encoding="utf-8"))
            is_test = (
                bool(PY_TEST.search(text)) if rel.endswith(".py") else bool(TEST_ATTR.search(text))
            )
            if is_test and rel not in registered:
                found.append(rel)
    return found


def problems(root: Path, head: str | None) -> list[str]:
    registry = root / REGISTRY
    if not registry.is_file():
        return [f"FAIL: {REGISTRY} does not exist"]
    try:
        data = tomllib.loads(registry.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return [f"FAIL: {REGISTRY} is not valid TOML: {exc}"]
    errors: list[str] = []
    if data.get("schema") != 1:
        errors.append("FAIL: registry schema must be 1")
    entries = data.get("test")
    if not isinstance(entries, list) or not entries:
        errors.append("FAIL: registry points at nothing")
        return errors
    seen: set[str] = set()
    target_maps: dict[str, dict] = {}
    suite_maps: dict[str, dict] = {}
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            errors.append(f"FAIL: test #{index} is not a table")
            continue
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id.strip():
            errors.append(f"FAIL: test #{index} missing id")
            entry_id = f"#{index}"
        elif entry_id in seen:
            errors.append(f"FAIL: {entry_id}: duplicate id")
        seen.add(entry_id)
        crate = entry.get("crate")
        cargo_target = entry.get("cargo_target")
        rel = entry.get("path")
        if not isinstance(crate, str) or not crate:
            errors.append(f"FAIL: {entry_id}: crate is required")
            crate = ""
        if not isinstance(cargo_target, str) or not cargo_target:
            errors.append(f"FAIL: {entry_id}: cargo_target is required")
            cargo_target = ""
        gates = entry.get("gates")
        if (
            not isinstance(gates, list)
            or not gates
            or not all(isinstance(gate, str) for gate in gates)
        ):
            errors.append(f"FAIL: {entry_id}: gates must be a non-empty list")
            gates = []
        for gate in gates:
            if gate not in KNOWN_GATES:
                errors.append(f"FAIL: {entry_id}: unknown gate {gate}")
        needs_evidence = [gate for gate in gates if gate in EVIDENCE_REQUIRED]
        if needs_evidence and "evidence_path" not in entry:
            listed = ", ".join(needs_evidence)
            errors.append(f"FAIL: {entry_id}: gate {listed} requires evidence_path")
        if "evidence_path" in entry:
            errors.extend(evidence_problems(root, entry_id, entry.get("evidence_path"), head))
        path = inside(root, rel) if isinstance(rel, str) else None
        if not isinstance(rel, str) or not rel:
            errors.append(f"FAIL: {entry_id}: path is required")
            continue
        if path is None:
            errors.append(f"FAIL: {entry_id}: path {rel} escapes the repo")
            continue
        if not path.is_file():
            errors.append(f"FAIL: {entry_id}: path {rel} does not exist")
            continue
        suite_line = entry.get("suite_map_line")
        if (
            "named" in gates
            and cargo_target != "script"
            and not executes_tests(path.read_text(encoding="utf-8"), path)
        ):
            errors.append(f"FAIL: {entry_id}: gate named but {rel} has no non-ignored test")
        if (
            cargo_target == "script"
            and rel.endswith(".py")
            and not runnable_python(path.read_text(encoding="utf-8"))
        ):
            errors.append(f"FAIL: {entry_id}: {rel} defines no test function")
        if cargo_target == "script":
            if rel.startswith("crates/") or (root / "crates" / crate).is_dir():
                errors.append(f"FAIL: {entry_id}: cargo_target script is not a crate test")
            if "named" in gates:
                errors.append(f"FAIL: {entry_id}: gate named requires a compiled cargo test")
            if suite_line is not None:
                errors.append(f"FAIL: {entry_id}: suite_map_line set but test is not consolidated")
            continue
        # The native Python binding is a workspace crate outside crates/.
        # Its Rust unit tests must be registered as compiled tests, not scripts.
        crate_dir = root / "python" if crate == "python" else root / "crates" / crate
        manifest_path = crate_dir / "Cargo.toml"
        if not manifest_path.is_file():
            errors.append(f"FAIL: {entry_id}: crate {crate} has no Cargo.toml")
            continue
        prefix = "python/" if crate == "python" else f"crates/{crate}/"
        if not rel.startswith(prefix):
            errors.append(f"FAIL: {entry_id}: path {rel} is not in crate {crate}")
            continue
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("package", {}).get("autotests", True) is not False:
            if suite_line is not None:
                errors.append(f"FAIL: {entry_id}: suite_map_line set but test is not consolidated")
            continue
        crate_rel = rel[len(prefix) :]
        if crate_rel.startswith("src/"):
            if cargo_target != "lib":
                errors.append(f"FAIL: {entry_id}: src/ test must use cargo_target lib")
            if suite_line is not None:
                errors.append(f"FAIL: {entry_id}: suite_map_line set but test is not consolidated")
            continue
        if crate not in target_maps:
            target_maps[crate] = compiled_by(crate_dir, manifest)
        how = target_maps[crate]
        target = how.get(crate_rel)
        if target is None:
            errors.append(
                f"FAIL: {entry_id}: {rel} is compiled by no test target; "
                "add it to a [[test]] path or a suite_NN.rs #[path]"
            )
            continue
        expected, kind = target
        if cargo_target != expected:
            errors.append(
                f"FAIL: {entry_id}: cargo_target {cargo_target} does not compile {rel} ({expected})"
            )
        if kind != "suite":
            if suite_line is not None:
                errors.append(f"FAIL: {entry_id}: suite_map_line set but test is not consolidated")
            continue
        if crate not in suite_maps:
            suite_maps[crate] = suite_map(crate_dir)
        recorded = suite_maps[crate]
        stem = Path(crate_rel).stem
        line = recorded.get(expected)
        transitively_mapped = line is not None and any(
            path.resolve() in target_files(crate_dir / "tests" / f"{outer}.rs", crate_dir)
            for outer in line[1]
        ) if line is not None and stem not in line[1] else False
        if line is None or (stem not in line[1] and not transitively_mapped):
            errors.append(f"FAIL: {entry_id}: {rel} missing suite_map line for {expected}")
            continue
        if not isinstance(suite_line, str) or not suite_line.strip():
            errors.append(f"FAIL: {entry_id}: consolidated test requires suite_map_line")
        elif suite_line.strip() != line[0]:
            errors.append(f"FAIL: {entry_id}: suite_map_line does not match tests/suite_map.txt")
    if data.get("complete") is True:
        registered = {e.get("path") for e in entries if isinstance(e, dict)}
        for rel in unregistered(root, registered):
            errors.append(f"FAIL: {rel}: test file has no registry row")
    if any(entry.get("calibration_acceptance_assertions") for entry in entries if isinstance(entry, dict)):
        try:
            registered_calibration_acceptance(root)
        except (ValueError, OSError, KeyError) as exc:
            errors.append(f"FAIL: calibration acceptance registration: {exc}")
    return errors


def check(root: Path) -> int:
    found = problems(root, head_sha(ROOT))
    if found:
        print("\n".join(found))
        return 1
    count = len(tomllib.loads((root / REGISTRY).read_text(encoding="utf-8"))["test"])
    print(f"v3 test registry OK: {count} registered test")
    return 0


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _entry(fields: dict[str, object]) -> str:
    lines = ["[[test]]"]
    for key in ("id", "crate", "cargo_target", "path", "suite_map_line", "evidence_path"):
        if key in fields:
            lines.append(f'{key} = "{fields[key]}"')
    gates = fields.get("gates", [])
    assert isinstance(gates, list)
    rendered = ", ".join(f'"{gate}"' for gate in gates)
    lines.append(f"gates = [{rendered}]")
    return "\n".join(lines) + "\n"


def _registry(root: Path, entries: list[dict[str, object]]) -> None:
    body = "schema = 1\n\n" + "\n".join(_entry(entry) for entry in entries)
    _write(root, REGISTRY, body)


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )


def _expect_fail(name: str, root: Path, needle: str) -> None:
    proc = _run(root)
    text = proc.stdout + proc.stderr
    if proc.returncode == 0 or needle not in text or "Traceback" in text:
        sys.exit(
            f"self-test {name}: expected non-zero with {needle!r}, rc={proc.returncode}\n{text}"
        )
    print(f"ok  {name}")


def _expect_ok(name: str, root: Path) -> None:
    proc = _run(root)
    text = proc.stdout + proc.stderr
    if proc.returncode != 0 or "v3 test registry OK" not in text or "Traceback" in text:
        sys.exit(f"self-test {name}: expected pass, rc={proc.returncode}\n{text}")
    print(f"ok  {name}")


def _crate(root: Path, suite_map_text: str) -> None:
    crate = root / "crates" / "demo"
    _write(
        crate,
        "Cargo.toml",
        """[package]
name = "demo"
version = "0.0.0"
edition = "2021"
autotests = false

[[test]]
name = "suite_01"
path = "tests/suite_01.rs"
""",
    )
    _write(crate, "tests/suite_01.rs", '#[path = "foo.rs"]\nmod foo;\n')
    _write(crate, "tests/foo.rs", "#[test]\nfn runs() {}\n")
    _write(crate, "tests/suite_map.txt", suite_map_text)


def self_test() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "python/Cargo.toml", '[package]\nname = "antecedent-py"\n')
        _write(root, "python/src/callbacks.rs", "#[test]\nfn evidence_boundary() {}\n")
        _registry(
            root,
            [
                {
                    "id": "native-binding-unit",
                    "crate": "python",
                    "cargo_target": "lib",
                    "path": "python/src/callbacks.rs",
                    "gates": ["inventory", "named"],
                }
            ],
        )
        _expect_ok("native binding compiled unit test", root)
        _write(root, "python/src/callbacks.rs", "#[test]\n#[ignore]\nfn missing() {}\n")
        _expect_fail("native binding ignored-only unit", root, "has no non-ignored test")

    head = head_sha(ROOT)
    if head is None:
        print("FAIL: self-test cannot read git HEAD", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _registry(
            root,
            [
                {
                    "id": "missing",
                    "crate": "scripts",
                    "cargo_target": "script",
                    "path": "scripts/no_such_check.py",
                    "gates": ["inventory"],
                }
            ],
        )
        _expect_fail("missing source file", root, "path scripts/no_such_check.py does not exist")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _crate(root, "suite_01: other\n")
        _registry(
            root,
            [
                {
                    "id": "folded",
                    "crate": "demo",
                    "cargo_target": "suite_01",
                    "path": "crates/demo/tests/foo.rs",
                    "suite_map_line": "suite_01: foo",
                    "gates": ["named"],
                }
            ],
        )
        _expect_fail("missing suite_map line", root, "missing suite_map line")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        crate = root / "crates" / "demo"
        _write(
            crate,
            "Cargo.toml",
            """[package]
name = "demo"
version = "0.0.0"
edition = "2021"
autotests = false
""",
        )
        _write(crate, "tests/suite_01.rs", '#[path = "foo.rs"]\nmod foo;\n')
        _write(crate, "tests/foo.rs", "#[test]\nfn runs() {}\n")
        _write(crate, "tests/suite_map.txt", "suite_01: foo\n")
        _registry(
            root,
            [
                {
                    "id": "unbuilt",
                    "crate": "demo",
                    "cargo_target": "suite_01",
                    "path": "crates/demo/tests/foo.rs",
                    "suite_map_line": "suite_01: foo",
                    "gates": ["named"],
                }
            ],
        )
        _expect_fail("unbuilt suite include", root, "compiled by no test target")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "scripts/check.py", "# executable check\n")
        _registry(
            root,
            [
                {
                    "id": "absent",
                    "crate": "scripts",
                    "cargo_target": "script",
                    "path": "scripts/check.py",
                    "gates": ["checked_execution"],
                    "evidence_path": "parity/no_such_evidence.toml",
                }
            ],
        )
        _expect_fail(
            "absent evidence file",
            root,
            "evidence path parity/no_such_evidence.toml does not exist",
        )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "scripts/check.py", "# executable check\n")
        _write(root, "parity/stale.toml", f'# v3-evidence\nrevision = "{"0" * 40}"\n')
        _registry(
            root,
            [
                {
                    "id": "stale",
                    "crate": "scripts",
                    "cargo_target": "script",
                    "path": "scripts/check.py",
                    "gates": ["transport"],
                    "evidence_path": "parity/stale.toml",
                }
            ],
        )
        _expect_fail("stale evidence revision", root, "revision is stale")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "scripts/check.py", "# executable check\n")
        _write(root, "parity/unstamped.toml", "# v3-evidence\n")
        _registry(
            root,
            [
                {
                    "id": "unstamped",
                    "crate": "scripts",
                    "cargo_target": "script",
                    "path": "scripts/check.py",
                    "gates": ["evidence"],
                    "evidence_path": "parity/unstamped.toml",
                }
            ],
        )
        _expect_fail("evidence stamp missing revision", root, "missing required key revision")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "scripts/check.py", "# executable check\n")
        _registry(
            root,
            [
                {
                    "id": "no-evidence",
                    "crate": "scripts",
                    "cargo_target": "script",
                    "path": "scripts/check.py",
                    "gates": ["decision"],
                }
            ],
        )
        _expect_fail("evidence gate without evidence_path", root, "requires evidence_path")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, REGISTRY, "schema = 1\n")
        _expect_fail("empty registry", root, "points at nothing")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "scripts/check_v3_test_registry.py", 'if __name__ == "__main__":\n    pass\n')
        _registry(
            root,
            [
                {
                    "id": "v3-registry-sentinel",
                    "crate": "scripts",
                    "cargo_target": "script",
                    "path": "scripts/check_v3_test_registry.py",
                    "gates": ["inventory"],
                }
            ],
        )
        _expect_ok("positive script sentinel", root)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _crate(root, "suite_01: foo\n")
        _write(root, "parity/fresh.toml", f'# v3-evidence\nrevision = "{head}"\n')
        _registry(
            root,
            [
                {
                    "id": "folded",
                    "crate": "demo",
                    "cargo_target": "suite_01",
                    "path": "crates/demo/tests/foo.rs",
                    "suite_map_line": "suite_01: foo",
                    "gates": ["named", "composition"],
                    "evidence_path": "parity/fresh.toml",
                }
            ],
        )
        _expect_ok("positive consolidated test with fresh evidence", root)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _crate(root, "suite_01: foo\n")
        _write(root, "crates/demo/tests/foo.rs", "#[test]\n#[ignore]\nfn never_runs() {}\n")
        _registry(
            root,
            [
                {
                    "id": "ignored",
                    "crate": "demo",
                    "cargo_target": "suite_01",
                    "path": "crates/demo/tests/foo.rs",
                    "suite_map_line": "suite_01: foo",
                    "gates": ["named"],
                }
            ],
        )
        _expect_fail("named gate on an ignored-only file", root, "has no non-ignored test")

    for decoy in (
        "// #[test] fn only_a_comment() {}\n",
        'const DECOY: &str = r#"#[test] fn only_a_string() {}"#;\n',
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _crate(root, "suite_01: foo\n")
            _write(root, "crates/demo/tests/foo.rs", decoy)
            _registry(root, [{"id": "decoy", "crate": "demo", "cargo_target": "suite_01",
                              "path": "crates/demo/tests/foo.rs", "suite_map_line": "suite_01: foo",
                              "gates": ["named"]}])
            _expect_fail("named gate on comment/string decoy", root, "has no non-ignored test")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "scripts/test_orphan.py", "def test_runs():\n    pass\n")
        _write(root, "scripts/test_listed.py", "def test_runs():\n    pass\n")
        _write(
            root,
            REGISTRY,
            "schema = 1\ncomplete = true\n\n"
            + _entry(
                {
                    "id": "listed",
                    "crate": "scripts",
                    "cargo_target": "script",
                    "path": "scripts/test_listed.py",
                    "gates": ["inventory"],
                }
            ),
        )
        _expect_fail(
            "complete registry with an unregistered test",
            root,
            "scripts/test_orphan.py: test file has no registry row",
        )

    print("v3 test registry self-test: ok")
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--self-test"]:
        return self_test()
    if len(argv) == 2 and argv[0] == "--root":
        return check(Path(argv[1]).resolve())
    if not argv:
        return check(ROOT)
    print("usage: check_v3_test_registry.py [--self-test|--root DIR]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
