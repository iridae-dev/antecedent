#!/usr/bin/env python3
"""Fail when upstream package names appear outside the permitted files.

Scans code and TOML only. Hand-written markdown is not read, so rewording
CONTRIBUTING, ROADMAP, SECURITY, or a file under docs/guides cannot fail
this gate.
"""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

root = Path(".")
permitted_path = root / "parity/permitted_upstream_name_files.toml"
# Prose suffixes are refused even if the permitted files list them. This gate must
# not extract facts from hand-written markdown.
PROSE_SUFFIXES = {".md", ".markdown", ".rst", ".txt"}
MATCH_KEYS = {
    "prefix",
    "suffix",
    "contains",
    "name_contains",
    "path_contains",
    "path_contains_any",
}

skip_dirs = {
    ".git",
    "target",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".cursor",
    ".claude",
    ".codeql-db",
    ".codeql-results",
    "uv.lock",
}


def fail(message: str) -> None:
    print(message)
    sys.exit(1)


def string_list(data: dict, key: str, *, required: bool) -> list[str]:
    if key not in data:
        if required:
            fail(f"upstream-name permitted files: {permitted_path} missing `{key}`")
        return []
    value = data[key]
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        fail(f"upstream-name permitted files: `{key}` must be a list of non-empty strings")
    return value


def load_permitted_files(path: Path) -> dict:
    if not path.is_file():
        fail(f"upstream-name permitted files: missing {path}")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        fail(f"upstream-name permitted files: {path} is not valid TOML: {exc}")
    names = string_list(data, "names", required=True)
    suffixes = [item.lower() for item in string_list(data, "scan_suffixes", required=True)]
    prose = sorted(set(suffixes) & PROSE_SUFFIXES)
    if prose:
        fail(
            "upstream-name permitted files: refusing prose scan suffixes "
            f"{prose}; this gate reads code and TOML only"
        )
    rules = data.get("match", [])
    if not isinstance(rules, list):
        fail("upstream-name permitted files: `match` must be an array of tables")
    for i, rule in enumerate(rules, 1):
        if not isinstance(rule, dict) or not rule:
            fail(f"upstream-name permitted files: match #{i} must be a non-empty table")
        unknown = sorted(set(rule) - MATCH_KEYS)
        if unknown:
            fail(f"upstream-name permitted files: match #{i} has unknown keys {unknown}")
        any_of = rule.get("path_contains_any")
        if any_of is not None and (
            not isinstance(any_of, list)
            or not any_of
            or not all(isinstance(item, str) and item for item in any_of)
        ):
            fail(
                f"upstream-name permitted files: match #{i} path_contains_any must be a non-empty string list"
            )
        for key in ("prefix", "suffix", "contains", "name_contains", "path_contains"):
            if key in rule and (not isinstance(rule[key], str) or not rule[key]):
                fail(f"upstream-name permitted files: match #{i} `{key}` must be a non-empty string")
    return {
        "names": names,
        "suffixes": set(suffixes),
        "exact": set(string_list(data, "exact", required=False)),
        "prefix": string_list(data, "prefix", required=False),
        "match": rules,
    }


def rule_matches(path: str, name: str, rule: dict) -> bool:
    if "prefix" in rule and not path.startswith(rule["prefix"]):
        return False
    if "suffix" in rule and not path.endswith(rule["suffix"]):
        return False
    if "contains" in rule and rule["contains"] not in path:
        return False
    if "name_contains" in rule and rule["name_contains"] not in name:
        return False
    if "path_contains" in rule and rule["path_contains"].lower() not in path.lower():
        return False
    any_of = rule.get("path_contains_any")
    return any_of is None or any(item.lower() in path.lower() for item in any_of)


def allowed(path: Path, spec: dict) -> bool:
    text = path.as_posix()
    if text in spec["exact"]:
        return True
    if any(text.startswith(prefix) for prefix in spec["prefix"]):
        return True
    return any(rule_matches(text, path.name, rule) for rule in spec["match"])


spec = load_permitted_files(permitted_path)
pat = re.compile("|".join(re.escape(name) for name in spec["names"]), re.IGNORECASE)

hits = []
tracked = subprocess.run(
    ["git", "ls-files", "-z"],
    check=True,
    capture_output=True,
).stdout.split(b"\0")
for raw_path in tracked:
    if not raw_path:
        continue
    path = root / raw_path.decode()
    if not path.is_file():
        continue
    parts = set(path.parts)
    if parts & skip_dirs:
        continue
    # Skip hidden paths at every depth, except the explicitly scanned .github tree.
    if any(part.startswith(".") and part != ".github" for part in path.parts):
        continue
    if path.suffix.lower() not in spec["suffixes"]:
        continue
    if allowed(path, spec):
        continue
    try:
        text = path.read_text(errors="ignore", encoding="utf-8")
    except OSError:
        continue
    if pat.search(text):
        # Ignore lines that only cite permitted baseline pin paths.
        lines = []
        for i, line in enumerate(text.splitlines(), 1):
            if not pat.search(line):
                continue
            if "parity/baselines/" in line:
                continue
            lines.append(f"  L{i}: {line.strip()[:160]}")
            if len(lines) >= 3:
                break
        if lines:
            hits.append((path.as_posix(), lines))

if hits:
    print(f"upstream-name permitted files violations ({len(hits)} files):")
    for path, lines in hits[:60]:
        print(f"- {path}")
        for ln in lines:
            print(ln)
    if len(hits) > 60:
        print(f"... and {len(hits) - 60} more")
    sys.exit(1)
print("upstream-name permitted files: ok")
