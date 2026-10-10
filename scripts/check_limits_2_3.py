#!/usr/bin/env python3
"""Compare exact 2.3 record bounds to their original guards, docs and refusal evidence.

A mapping belongs to a complete record ID. Shared milestone/workstream prefixes
never select a sibling record's bounds. Missing/stale mappings are errors for every
status, including candidates; checking a candidate is not inference activation.
"""

from __future__ import annotations

import ast
import operator
import re
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = "parity/limits_2_3.toml"
RECORDS = "parity/promotion_2_3.toml"
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.LShift: operator.lshift,
}


def integer_expression(
    expression: str, constants: dict[str, str], active: frozenset[str] = frozenset()
) -> int:
    """Evaluate bounded integer arithmetic only; no calls, attributes or Python eval."""
    if len(expression) > 512 or len(active) > 64:
        raise ValueError("integer expression is too long")
    tree = ast.parse(expression.strip(), mode="eval")

    def visit(node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            value = node.value
        elif (
            isinstance(node, ast.Name)
            and node.id in constants
            and node.id not in active
        ):
            value = integer_expression(
                constants[node.id], constants, active | {node.id}
            )
        elif isinstance(node, ast.BinOp) and type(node.op) in OPS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.LShift) and not 0 <= right <= 63:
                raise ValueError("shift exceeds bounded integer arithmetic")
            value = OPS[type(node.op)](left, right)
        else:
            raise ValueError("not a bounded integer expression")
        if not 0 <= value <= (1 << 63):
            raise ValueError("integer value exceeds supported range")
        return value

    return visit(tree.body)


def constants_in(text: str) -> dict[str, str]:
    return dict(
        re.findall(
            r"\bconst\s+([A-Z][A-Z0-9_]*)\s*:\s*[\w<>;\[\]]+\s*=\s*([^;\n]+);", text
        )
    )


def read_values(root: Path, place: dict[str, str]) -> list[int]:
    text = (root / place["path"]).read_text(encoding="utf-8")
    constants = constants_in(text)
    if "constant" in place:
        name = place["constant"]
        if name not in constants:
            raise ValueError(f"constant {name} missing from its declared source")
        return [integer_expression(constants[name], constants, frozenset({name}))]
    pattern = re.compile(place["pattern"], re.MULTILINE)
    if pattern.groups != 1:
        raise ValueError("numeric place must have exactly one capture group")
    values = [
        integer_expression(match.group(1).replace(",", ""), constants)
        for match in pattern.finditer(text)
    ]
    if not values:
        raise ValueError("declared numeric statement is missing")
    return values


def check(root: Path = ROOT) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    checked: list[str] = []
    try:
        records = tomllib.loads((root / RECORDS).read_text(encoding="utf-8"))["record"]
        registry = tomllib.loads((root / REGISTRY).read_text(encoding="utf-8"))
        mappings = registry["bound"]
    except (OSError, ValueError, KeyError, re.error) as error:
        return [str(error)], checked
    records_by_id = {record["id"]: record for record in records}
    if len(records_by_id) != len(records):
        errors.append("duplicate complete record ID")
    rows: dict[tuple[str, str], dict] = {}
    for row in mappings:
        key = (row["record"], row["key"])
        if key in rows:
            errors.append(f"{key}: duplicate exact bound identity")
        rows[key] = row
    for record in records:
        for key, value in record.get("bounds", {}).items():
            if type(value) is not int:
                continue
            identity = (record["id"], key)
            row = rows.get(identity)
            if row is None:
                errors.append(f"{identity}: numeric bound has no exact-record mapping")
                continue
            failed = False
            for label in ("implementation", "documentation"):
                for place in row.get(label, []):
                    try:
                        values = read_values(root, place)
                        if any(got != value for got in values):
                            raise ValueError(f"states {values}, record states {value}")
                    except (
                        OSError,
                        ValueError,
                        SyntaxError,
                        KeyError,
                        re.error,
                    ) as error:
                        errors.append(f"{identity}: {label} {place}: {error}")
                        failed = True
                if not row.get(label):
                    errors.append(f"{identity}: missing {label} evidence")
                    failed = True
            for label in ("guard", "refusal"):
                place = row.get(label)
                try:
                    if not place or not re.search(
                        place["pattern"],
                        (root / place["path"]).read_text(encoding="utf-8"),
                        re.MULTILINE,
                    ):
                        raise ValueError(f"missing actual {label} evidence")
                except (OSError, ValueError, KeyError, re.error) as error:
                    errors.append(f"{identity}: {error}")
                    failed = True
            fixture = row.get("scope_fixture")
            if fixture is not None:
                try:
                    if not re.search(
                        fixture["pattern"],
                        (root / fixture["path"]).read_text(encoding="utf-8"),
                        re.MULTILINE,
                    ):
                        raise ValueError("original scope/refusal fixture is missing")
                except (OSError, ValueError, KeyError, re.error) as error:
                    errors.append(f"{identity}: {error}")
                    failed = True
            if not failed:
                checked.append(f"{identity[0]}.{key}={value}")
    prose_rows: dict[tuple[str, str], dict] = {}
    for row in registry.get("noninteger", []):
        identity = (row["record"], row["key"])
        if identity in prose_rows:
            errors.append(f"{identity}: duplicate noninteger bound identity")
        prose_rows[identity] = row
    for record in records:
        bounds = record.get("bounds", {})
        for key, value in bounds.items():
            if not isinstance(value, str) or not re.search(r"\d", value):
                continue
            identity = (record["id"], key)
            prose = prose_rows.get(identity)
            if prose is None:
                errors.append(
                    f"{identity}: numeric prose requires an exact noninteger mapping"
                )
                continue
            have = {
                int(token.replace("_", "").replace(",", ""))
                for token in re.findall(r"\d[\d_,]*", value)
            }
            expected = {
                bounds.get(source_key) for source_key in prose.get("numbers_from", [])
            }
            if (
                not expected
                or any(type(number) is not int for number in expected)
                or have != expected
            ):
                errors.append(
                    f"{identity}: numeric prose {have} differs from compared bounds {expected}"
                )
    for identity in prose_rows:
        record = records_by_id.get(identity[0])
        if record is None or not isinstance(
            record.get("bounds", {}).get(identity[1]), str
        ):
            errors.append(f"{identity}: stale noninteger bound mapping")
    for identity in rows:
        record = records_by_id.get(identity[0])
        if record is None or type(record.get("bounds", {}).get(identity[1])) is not int:
            errors.append(f"{identity}: stale exact-record bound mapping")
    return errors, checked


def self_test() -> int:
    """Independent synthetic sibling IDs and semantic mutations of each evidence place."""
    import copy
    import json
    import tempfile

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "parity").mkdir()
        (root / "source.rs").write_text(
            "const ALPHA: usize = 4;\nconst BETA: usize = 1 << 3;\n"
            "if n > ALPHA { return Err(bad); }\nif n > BETA { return Err(bad); }\n",
            encoding="utf-8",
        )
        (root / "scope.md").write_text("alpha:4\nbeta:8\n", encoding="utf-8")
        records = [
            {"id": "2.3C.C3.alpha", "bounds": {"max_items": 4}},
            {"id": "2.3C.C3.beta", "bounds": {"max_items": 8}},
        ]
        mappings = [
            {
                "record": record["id"],
                "key": "max_items",
                "implementation": [{"path": "source.rs", "constant": name.upper()}],
                "documentation": [{"path": "scope.md", "pattern": name + r":(\d+)"}],
                "guard": {"path": "source.rs", "pattern": "n > " + name.upper()},
                "refusal": {
                    "path": "source.rs",
                    "pattern": "n > " + name.upper() + r" \{ return Err",
                },
            }
            for record, name in zip(records, ["alpha", "beta"], strict=True)
        ]

        def write_rows(path: Path, label: str, values: list[dict]) -> None:
            def toml(value: object) -> str:
                if isinstance(value, dict):
                    return (
                        "{ "
                        + ", ".join(
                            key + " = " + toml(item) for key, item in value.items()
                        )
                        + " }"
                    )
                if isinstance(value, list):
                    return "[" + ", ".join(toml(item) for item in value) + "]"
                return json.dumps(value)

            path.write_text(
                "\n\n".join(
                    "[["
                    + label
                    + "]]\n"
                    + "\n".join(key + " = " + toml(value) for key, value in row.items())
                    for row in values
                ),
                encoding="utf-8",
            )

        def run(rows: list[dict], declared: list[dict] | None = None) -> list[str]:
            write_rows(root / REGISTRY, "bound", rows)
            write_rows(root / RECORDS, "record", declared or records)
            return check(root)[0]

        if run(mappings):
            failures.append("distinct full sibling IDs with different values must pass")
        for case in [
            "prefix",
            "duplicate",
            "missing",
            "stale",
            "implementation",
            "documentation",
            "guard",
            "refusal",
            "changed_record",
            "invalid_regex",
            "duplicate_record",
            "frozen_missing",
        ]:
            changed = copy.deepcopy(mappings)
            declared = copy.deepcopy(records)
            if case == "prefix":
                changed[1]["record"] = "2.3C.C3"
            elif case == "duplicate":
                changed.append(copy.deepcopy(changed[0]))
            elif case == "missing":
                changed.pop()
            elif case == "stale":
                changed[1]["key"] = "not_declared"
            elif case in ("implementation", "documentation"):
                changed[1][case][0] = {"path": "scope.md", "pattern": r"alpha:(\d+)"}
            elif case in ("guard", "refusal"):
                changed[1][case]["pattern"] = "MISSING ACTUAL EVIDENCE"
            elif case == "changed_record":
                declared[1]["bounds"]["max_items"] = 9
            elif case == "invalid_regex":
                changed[1]["guard"]["pattern"] = "["
            elif case == "duplicate_record":
                declared.append(copy.deepcopy(declared[0]))
            elif case == "frozen_missing":
                declared[1]["status"] = "frozen"
                changed.pop()
            if not run(changed, declared):
                failures.append(case + " must fail")
        prose_records = copy.deepcopy(records)
        prose_records[0]["bounds"]["operation_limit"] = "declared cap 4"
        if not run(mappings, prose_records):
            failures.append("numeric prose without a mapping must fail")
        prose_mapping = [
            {
                "record": records[0]["id"],
                "key": "operation_limit",
                "numbers_from": ["max_items"],
            }
        ]
        write_rows(root / "prose.toml", "noninteger", prose_mapping)
        registry_text = (root / REGISTRY).read_text(encoding="utf-8")
        (root / REGISTRY).write_text(
            registry_text + "\n" + (root / "prose.toml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        if check(root)[0]:
            failures.append(
                "numeric prose matching an independently compared cap must pass"
            )
        prose_records[0]["bounds"]["operation_limit"] = "declared cap 5"
        write_rows(root / RECORDS, "record", prose_records)
        if not check(root)[0]:
            failures.append("numeric prose disagreement must fail")
        for bad in ["__import__('os')", "1 << 1000", "BETA", "-1"]:
            try:
                integer_expression(bad, {"BETA": "BETA"})
            except (ValueError, SyntaxError):
                pass
            else:
                failures.append("unsafe or cyclic expression must fail: " + bad)
    for failure in failures:
        print("FAIL:", failure)
    if not failures:
        print(
            "2.3 limits agreement self-test PASS: exact sibling identities and evidence mutations"
        )
    return int(bool(failures))


def main() -> int:
    import sys

    if "--self-test" in sys.argv:
        return self_test()
    errors, checked = check()
    for error in errors:
        print(f"FAIL: {error}")
    if not errors:
        print(f"2.3 limits agreement PASS: {len(checked)} exact-record bounds")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
