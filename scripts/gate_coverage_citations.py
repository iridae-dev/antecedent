#!/usr/bin/env python3
import fnmatch
import itertools
import re
import shlex
import sys
import tomllib
from pathlib import Path

from test_evidence import rust_items, target_files, rust_test_declaration_problems, _attrs_before, _disabling_cfg, _match_bracket
from promotion_source import _compact_rust_tokens
import hashlib

root = Path(".")
tests_dir = root / "crates/antecedent/tests"
with open(root / "parity/support_licensed.toml", "rb") as handle:
    lic = tomllib.load(handle)

# Only these reviewed repetition macros have this syntax. A changed template or
# a different macro must not turn an arbitrary match arm into a test identity.
CALIBRATION_MACROS = {
    "dag_coverage": "a02fb7f845f973ad94ac812530fcff78a84b5e7ecd248968d77b739836760c2c",
    "mediation_coverage": "d34d37538afe0c545855d9d082f21b63c57f5e67f189a46f7f9a1122a0040490",
}


def target_exists(base: Path, module: str) -> bool:
    """Resolve the cited source through an enabled Cargo test target."""
    crate = base / "crates/antecedent"
    try:
        data = tomllib.loads((crate / "Cargo.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return False
    source = (crate / "tests" / f"{module}.rs").resolve()
    targets = [t for t in data.get("test", []) if not t.get("required-features")]
    if data.get("package", {}).get("autotests", True):
        targets += [{"path": f"tests/{module}.rs"}]
    for target in targets:
        path = target.get("path", f"tests/{target.get('name', '')}.rs")
        root_file = (crate / path).resolve()
        if root_file.is_file() and source in target_files(root_file, crate.resolve()):
            return True
    return False


def nondeselecting_libtest_flags(words: list[str]) -> bool:
    """Recognize only options that do not filter the selected test set."""
    return all(word in {"--ignored", "--nocapture", "--exact"} for word in words) and "--ignored" in words


def calibration_lane_tests(base: Path, module: str) -> set[str]:
    """Find literal ignored-test registrations in the actual calibration gate."""
    path = base / "scripts/gate_calibration.sh"
    if not path.is_file():
        return set()
    text = path.read_text(encoding="utf-8").replace("\\\n", " ")
    helpers = {}
    for match in re.finditer(r"(?ms)^([A-Za-z_]\w*)\(\)\s*\{\s*\n(.*?)^\}", text):
        body = "\n".join(line for line in match[2].splitlines() if not line.lstrip().startswith("#"))
        variables = {m[1]: int(m[2]) for m in re.finditer(r'(?m)^\s*local\s+(\w+)="\$(\d+)"\s*$', body)}
        if not variables or re.search(r"(?m)^\s*(?:if|case|return|exit|for|while|until)\b", body):
            continue
        for line in body.splitlines():
            try:
                words = shlex.split(line, comments=True)
                pos = words.index("cargo")
                target = words.index("--test", pos + 1)
                separator = words.index("--", target + 2)
            except (ValueError, IndexError):
                continue
            if (words[0] not in {"cargo", "check"} or words[pos:pos + 2] != ["cargo", "test"]
                    or not any(words[i:i + 2] == ["-p", "antecedent"] for i in range(pos, target))
                    or not nondeselecting_libtest_flags(words[separator + 1:])):
                continue
            def arg(value):
                key = value.removeprefix("$").strip("{}")
                return variables.get(key) if value.startswith("$") else None
            target_value, filter_value = words[target + 1:target + 3]
            filter_arg = arg(filter_value)
            if filter_arg and (target_value == module or arg(target_value)):
                helpers[match[1]] = (target_value, arg(target_value), filter_arg)
    names = set()
    for line in text.splitlines():
        try:
            words = shlex.split(line, comments=True)
        except ValueError:
            continue
        if line[:1].isspace() or not words or words[0] not in helpers:
            continue
        target_value, target_arg, filter_arg = helpers[words[0]]
        if max(target_arg or 0, filter_arg) >= len(words):
            continue
        if (target_value == module or words[target_arg] == module) and re.fullmatch(r"[A-Za-z_]\w*", words[filter_arg]):
            names.add(words[filter_arg])
    # An unfiltered, direct target command runs all ignored tests in that target.
    for line in text.splitlines():
        try:
            words = shlex.split(line, comments=True)
            pos = words.index("cargo")
            target = words.index("--test", pos + 1)
        except (ValueError, IndexError):
            continue
        if (line.startswith(("check ", "cargo ")) and words[pos:pos + 2] == ["cargo", "test"]
                and any(words[i:i + 2] == ["-p", "antecedent"] for i in range(pos, target))
                and words[target + 1:target + 3] == [module, "--"]
                and nondeselecting_libtest_flags(words[target + 3:])):
            names.add("*")
    return names


def calibration_macro_tests(path: Path, module: str) -> set[str]:
    if module != "bayesian_temporal":
        return set()
    masked, items = rust_items(path.resolve())
    code, skel = masked.code, masked.skeleton
    ranges = [item.body for item in items if item.kind in {"fn", "mod", "macro"}]
    result = set()
    for name, expected in CALIBRATION_MACROS.items():
        definitions = [item for item in items if item.kind == "macro" and item.name == name]
        if len(definitions) != 1:
            continue
        item = definitions[0]
        digest = hashlib.sha256(_compact_rust_tokens(code[item.start:item.body[1]]).encode()).hexdigest()
        if digest != expected:
            continue
        for match in re.finditer(r"(?m)^\s*" + re.escape(name) + r"!\s*\{", skel):
            if any(lo < match.start() < hi for lo, hi in ranges):
                continue
            if _disabling_cfg(_attrs_before(skel, code, match.start())):
                continue
            opening = match.end() - 1
            closing = _match_bracket(skel, opening)
            cursor = opening + 1
            names = []
            while cursor < closing:
                entry = re.match(r"\s*([a-z_][a-z0-9_]*)\s*=>\s*\(", skel[cursor:closing])
                if entry is None:
                    if skel[cursor:closing].strip():
                        names = []
                    break
                end = _match_bracket(skel, cursor + entry.end() - 1)
                semi = re.match(r"\s*;", skel[end + 1:closing])
                if not semi:
                    names = []
                    break
                names.append(entry[1])
                cursor = end + 1 + semi.end()
            if len(names) == len(set(names)):
                result.update(names)
    return result


def test_fns(module: str):
    path = tests_dir / f"{module}.rs"
    if not path.is_file():
        return None
    if not target_exists(root, module):
        return set()
    _, items = rust_items(path.resolve())
    ignored_lane = calibration_lane_tests(root, module)
    result = set()
    for item in items:
        if item.kind != "fn":
            continue
        problems = rust_test_declaration_problems(path.resolve(), item.name)
        attrs = [re.sub(r"\s+", "", attr) for attr in item.attrs]
        if any(attr.startswith("#[should_panic") for attr in attrs):
            continue
        if not problems:
            result.add(item.name)
            continue
        # Ordinary #[ignore] is not evidence. Calibration tests are accepted
        # only when their exact identity is wired to this target's ignored lane.
        ignored = [attr for attr in attrs if attr.startswith("#[ignore")]
        calibration = len(ignored) == 1 and ignored[0].startswith('#[ignore="calibration:')
        if (calibration and (item.name in ignored_lane or "*" in ignored_lane)
                and all(problem.endswith("is #[ignore]d") for problem in problems)):
            result.add(item.name)
    generated = calibration_macro_tests(path, module)
    result.update(generated if "*" in ignored_lane else generated & ignored_lane)
    return result


fns_cache = {}

# ------------------------------------------------ citations in the prose
# `static_calibration::name`, `crates/antecedent/tests/derivative_calibration.rs::name`, and the
# shorthand `::name` that continues the last-named module inside one citation.
MODULES = sorted((p.stem for p in tests_dir.glob("*.rs")), key=len, reverse=True)
# Explicit test paths remain recognizable even after the target module is deleted.
# Shorthand names remain restricted to existing modules to avoid treating API
# references in prose as coverage-test citations.
module_alternatives = "|".join(re.escape(m) for m in MODULES) or r"(?!)"
CITE_RE = re.compile(
    r"(?:(?:crates/antecedent/tests/)(?P<explicit>[a-z][a-z0-9_]*)\.rs"
    r"|(?<![A-Za-z0-9_])(?P<short>" + module_alternatives + r")(?:\.rs)?)"
    r"::(?P<name>[A-Za-z0-9_{},*]+)(?P<continued>(?:,\s*::[A-Za-z0-9_{},*]+)*)"
)
CONT_RE = re.compile(r"::([A-Za-z0-9_{},*]+)")


def expand(pattern: str):
    """`a_{x,y}_*` -> the glob patterns `a_x_*`, `a_y_*`."""
    parts = re.split(r"(\{[^}]*\})", pattern)
    alts = [p[1:-1].split(",") if p.startswith("{") else [p] for p in parts]
    return ["".join(combo) for combo in itertools.product(*alts)]


fail = []
checked = 0
for cell in lic.get("cell", []):
    row = f"{cell['query']}/{cell['graph_class']}/{cell['structure']}/{cell['inference']}/{cell['validation']}"
    lim = str(cell.get("limitations", ""))
    for m in CITE_RE.finditer(lim):
        module = m.group("explicit") or m.group("short")
        names = [m.group("name")] + CONT_RE.findall(m.group("continued") or "")
        if module not in fns_cache:
            fns_cache[module] = test_fns(module)
        fns = fns_cache[module]
        if fns is None:
            fail.append(
                f"{row}: cites {module}::{names[0]} but crates/antecedent/tests/{module}.rs does not exist"
            )
            continue
        for name in names:
            checked += 1
            pats = expand(name.rstrip(".,;"))
            if not any(fnmatch.fnmatchcase(fn, pat) for pat in pats for fn in fns):
                fail.append(
                    f"{row}: cites {module}::{name} but no such test fn in crates/antecedent/tests/{module}.rs"
                )

if checked == 0:
    fail.append("no coverage test citations were checked")

if fail:
    print("Coverage citation gate FAILED:")
    for f in sorted(set(fail)):
        print(" -", f)
    sys.exit(1)

print(
    f"Coverage citations OK ({checked} citations across {len(fns_cache)} test modules resolve to existing test fns)"
)
