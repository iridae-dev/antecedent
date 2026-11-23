#!/usr/bin/env python3
"""Assurance records: machine-checked properties of Antecedent operations.

`parity/assurance_records.toml` holds one record per property. A record states its
proposition and scope, what it trusts, the operation it is about (the operation-record
fields), and one or more checks. Each check has one kind, never merged into a single
"verified" label:

* `machine_checked_proof`: a Lean theorem about a specification;
* `verified_implementation`: a Lean theorem about the production Rust as Charon and
  Aeneas translate it (`formal/rust/extract.sh`);
* `bounded_model_check`: a Kani proof harness run on the production Rust.

A record binds its inputs by digest: the Lean and harness files, the Rust items it covers
(by token, so comment and whitespace edits are neutral), and the pinned toolchain. A
change to any of them makes the record stale until its checks run again.

    python3 scripts/assurance.py check            # schema, theorems, no sorry, digests current
    python3 scripts/assurance.py run [ID ...]     # run the checks, then rebind the records
    python3 scripts/assurance.py --self-test      # the check rejects deliberately broken inputs

`check` reads files only. `run` needs `lake` (formal/lean-toolchain), `cargo kani` at the
version in `formal/toolchain.toml`, and, for a `verified_implementation`, `AENEAS_DIR`
pointing to the pinned Aeneas release: it retranslates the Rust and fails if the committed
translation differs. CI rebuilds every proof and reruns every harness (the `proofs` and
`kani` jobs) and runs `check` in the static gate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from _repo import REPO_ROOT, load_toml

import rust_items

RECORDS = Path("parity/assurance_records.toml")
TOOLCHAIN = Path("formal/toolchain.toml")

CHECK_KINDS = {
    "machine_checked_proof": "lean",
    "verified_implementation": "lean+aeneas",
    "bounded_model_check": "kani",
}
# Pins every record depends on, whatever its kind.
TOOLCHAIN_FILES = [
    "formal/toolchain.toml",
    "formal/lean-toolchain",
    "formal/lakefile.toml",
    "formal/lake-manifest.json",
]
KIND_TOOLCHAIN_FILES = {
    "machine_checked_proof": [],
    "verified_implementation": [
        "formal/rust/lean-toolchain",
        "formal/rust/lakefile.toml",
        "formal/rust/lake-manifest.json",
        "formal/rust/extract.sh",
    ],
    "bounded_model_check": ["formal/kani/Cargo.toml", "formal/kani/Cargo.lock"],
}
RECORD_KEYS = {"id", "proposition", "scope", "trusted", "limitations", "operation", "checks", "result"}
OPERATION_KEYS = {
    "inputs", "output", "premises", "transformation", "justification", "implementation",
    "obligations", "limitations",
}
RESULT_KEYS = {"inputs_digest", "checked_commit"}
# Statements that end a proof without proving it, or assume what should be proved.
UNPROVED = re.compile(r"\b(sorry|admit|native_decide)\b|^\s*axiom\s", re.MULTILINE)
LEAN_COMMENT = re.compile(r"/-.*?-/|--[^\n]*", re.DOTALL)


class Problem(Exception):
    pass


def _sha(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


def _read(root: Path, rel: str) -> str:
    path = root / rel
    if not path.is_file():
        raise Problem(f"missing input {rel}")
    return path.read_text().replace("\r\n", "\n")


def _lean_code(text: str) -> str:
    return LEAN_COMMENT.sub(" ", text)


def check_files(check: dict) -> list[str]:
    files = list(check.get("files", []))
    if check["kind"] == "bounded_model_check":
        files += [h["file"] for h in check.get("harnesses", [])]
    return files


def inputs_digest(root: Path, record: dict) -> str:
    """Digest of everything the record's checks depend on."""
    parts: list[str] = [record["id"], record["proposition"], record["scope"]]
    pins = list(TOOLCHAIN_FILES)
    for check in record["checks"]:
        pins += KIND_TOOLCHAIN_FILES[check["kind"]]
        parts.append(json.dumps(
            {k: check[k] for k in sorted(check) if k not in {"files", "rust_items"}}, sort_keys=True
        ))
        for rel in check_files(check):
            parts += [rel, _read(root, rel)]
        for binding in check.get("rust_items", []):
            parsed = rust_items.File(_read(root, binding["file"]))
            for locator in binding["items"]:
                digest = parsed.digest(locator, "full")
                if digest is None:
                    raise Problem(f"{record['id']}: no Rust item {locator} in {binding['file']}")
                parts += [binding["file"], locator, digest]
    for rel in sorted(set(pins)):
        parts += [rel, _read(root, rel)]
    return _sha(*parts)


def _theorem_declared(text: str, name: str) -> bool:
    """Whether `name` (fully qualified) is declared as a theorem in `text`."""
    *namespace, short = name.split(".")
    code = _lean_code(text)
    if not re.search(rf"^\s*(?:@\[[^\]]*\]\s*)?(?:private\s+)?theorem\s+{re.escape(short)}\b", code, re.M):
        return False
    opened = re.findall(r"^namespace\s+([\w.]+)", code, re.M)
    return not namespace or ".".join(namespace) in opened


def validate(root: Path, data: dict, toolchain: dict) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    records = data.get("records", [])
    if not records:
        errors.append("no assurance records")
    for record in records:
        rid = record.get("id", "<no id>")
        missing = RECORD_KEYS - set(record)
        extra = set(record) - RECORD_KEYS
        if missing or extra:
            errors.append(f"{rid}: missing {sorted(missing)} unexpected {sorted(extra)}")
            continue
        if rid in seen:
            errors.append(f"{rid}: duplicate id")
        seen.add(rid)
        op = record["operation"]
        if set(op) != OPERATION_KEYS:
            errors.append(f"{rid}: operation fields must be exactly {sorted(OPERATION_KEYS)}")
        if not record["trusted"]:
            errors.append(f"{rid}: a check always trusts something; list it")
        if not record["checks"]:
            errors.append(f"{rid}: no checks")
        kinds_known = True
        for check in record["checks"]:
            kind = check.get("kind")
            if kind not in CHECK_KINDS:
                errors.append(f"{rid}: unknown check kind {kind!r}")
                kinds_known = False
                continue
            if check.get("tool") != CHECK_KINDS[kind]:
                errors.append(f"{rid}: {kind} runs on {CHECK_KINDS[kind]}, not {check.get('tool')!r}")
            if kind in {"verified_implementation", "bounded_model_check"} and not check.get("rust_items"):
                errors.append(f"{rid}: a {kind} names the production Rust items it covers")
            if kind == "machine_checked_proof" and check.get("rust_items"):
                errors.append(f"{rid}: a proof about a specification covers no Rust item; use verified_implementation")
            if kind == "bounded_model_check":
                for harness in check.get("harnesses", []):
                    text = _read(root, harness["file"])
                    if not re.search(rf"#\[kani::proof\][^{{]*?fn\s+{re.escape(harness['name'])}\b", text, re.S):
                        errors.append(f"{rid}: no #[kani::proof] fn {harness['name']} in {harness['file']}")
                if not check.get("harnesses"):
                    errors.append(f"{rid}: a bounded_model_check names its harnesses")
            else:
                texts = []
                for rel in check.get("files", []):
                    try:
                        texts.append(_read(root, rel))
                    except Problem as problem:
                        errors.append(f"{rid}: {problem}")
                for name in check.get("theorems", []):
                    if not any(_theorem_declared(t, name) for t in texts):
                        errors.append(f"{rid}: theorem {name} is not declared in {check.get('files')}")
                if not check.get("theorems"):
                    errors.append(f"{rid}: a {kind} names its theorems")
        result = record["result"]
        if set(result) != RESULT_KEYS:
            errors.append(f"{rid}: result fields must be exactly {sorted(RESULT_KEYS)}")
            continue
        if not kinds_known:
            continue
        if not re.fullmatch(r"[0-9a-f]{40}", result["checked_commit"]):
            errors.append(f"{rid}: checked_commit must be a full commit id")
        try:
            current = inputs_digest(root, record)
        except Problem as problem:
            errors.append(f"{rid}: {problem}")
            continue
        if current != result["inputs_digest"]:
            errors.append(
                f"{rid}: stale; an input changed since its checks ran at {result['checked_commit'][:12]}. "
                f"Rerun: python3 scripts/assurance.py run {rid}"
            )
    for path in sorted((root / "formal").rglob("*.lean")):
        if ".lake" in path.parts:
            continue
        if UNPROVED.search(_lean_code(path.read_text())):
            errors.append(f"{path.relative_to(root)}: sorry, admit, native_decide or axiom in a proof source")
    for key in ("lean", "aeneas", "charon", "kani"):
        if not toolchain.get(key):
            errors.append(f"{TOOLCHAIN}: no {key} pin")
    lean_pin = _read(root, "formal/lean-toolchain").strip()
    for rel in ("formal/lean-toolchain", "formal/rust/lean-toolchain"):
        if _read(root, rel).strip() != toolchain.get("lean"):
            errors.append(f"{rel} is {lean_pin!r}; {TOOLCHAIN} pins {toolchain.get('lean')!r}")
    extract = _read(root, "formal/rust/extract.sh")
    if f'CHARON_VERSION="{toolchain.get("charon")}"' not in extract or \
            f'AENEAS_RELEASE="{toolchain.get("aeneas")}"' not in extract:
        errors.append(f"formal/rust/extract.sh pins differ from {TOOLCHAIN}")
    return errors


def check(root: Path = REPO_ROOT) -> int:
    errors = validate(root, load_toml(root / RECORDS), load_toml(root / TOOLCHAIN))
    for error in errors:
        print(f"assurance: {error}", file=sys.stderr)
    if not errors:
        print(f"assurance: {len(load_toml(root / RECORDS)['records'])} records current")
    return 1 if errors else 0


def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
    print("+", " ".join(cmd), f"(in {cwd.relative_to(REPO_ROOT) if cwd != REPO_ROOT else '.'})", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def run(ids: list[str], root: Path = REPO_ROOT) -> int:
    data = load_toml(root / RECORDS)
    toolchain = load_toml(root / TOOLCHAIN)
    chosen = [r for r in data["records"] if not ids or r["id"] in ids]
    unknown = set(ids) - {r["id"] for r in chosen}
    if unknown:
        print(f"assurance: unknown record(s) {sorted(unknown)}", file=sys.stderr)
        return 2
    version = subprocess.run(["cargo", "kani", "--version"], capture_output=True, text=True).stdout
    if any(c["kind"] == "bounded_model_check" for r in chosen for c in r["checks"]) and \
            f"Kani Rust Verifier {toolchain['kani']}" not in version:
        print(f"assurance: cargo kani is not {toolchain['kani']}: {version.strip()}", file=sys.stderr)
        return 2
    built: set[str] = set()
    for record in chosen:
        for check in record["checks"]:
            kind = check["kind"]
            if kind == "verified_implementation" and "extract" not in built:
                if not os.environ.get("AENEAS_DIR"):
                    print("assurance: set AENEAS_DIR to the pinned Aeneas release", file=sys.stderr)
                    return 2
                _run(["bash", "formal/rust/extract.sh"], root)
                _run(["git", "diff", "--exit-code", "--", "formal/rust/AntecedentRust/Extracted"], root)
                built.add("extract")
            package = {"machine_checked_proof": "formal", "verified_implementation": "formal/rust"}.get(kind)
            if package and package not in built:
                _run(["lake", "build"], root / package)
                built.add(package)
            if kind == "bounded_model_check":
                for harness in check["harnesses"]:
                    _run(["cargo", "kani", "--harness", harness["name"]], root / "formal/kani")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                          check=True).stdout.strip()
    text = (root / RECORDS).read_text()
    for record in chosen:
        digest = inputs_digest(root, record)
        block = re.compile(
            rf'(id = "{re.escape(record["id"])}".*?\[records\.result\]\s*'
            r'inputs_digest = ")[0-9a-f]*("\s*checked_commit = ")[0-9a-f]*(")',
            re.S,
        )
        text, count = block.subn(rf"\g<1>{digest}\g<2>{head}\g<3>", text, count=1)
        if count != 1:
            print(f"assurance: could not rebind {record['id']}", file=sys.stderr)
            return 1
    (root / RECORDS).write_text(text)
    print(f"assurance: {len(chosen)} record(s) checked at {head[:12]} and rebound")
    return 0


def self_test(root: Path = REPO_ROOT) -> int:
    """Deliberately broken inputs must each be rejected."""
    data = load_toml(root / RECORDS)
    toolchain = load_toml(root / TOOLCHAIN)
    if validate(root, data, toolchain):
        print("assurance self-test: the committed records do not pass", file=sys.stderr)
        return 1
    first = data["records"][0]

    def broken(fn) -> dict:
        d = copy.deepcopy(data)
        fn(d["records"][0])
        return d

    rust_check = next(c for r in data["records"] for c in r["checks"] if c.get("rust_items"))
    rust_file = rust_check["rust_items"][0]["file"]
    rust_item = rust_check["rust_items"][0]["items"][0]
    original = (root / rust_file).read_text()
    cases: list[tuple[str, object]] = [
        ("unknown check kind", broken(lambda r: r["checks"][0].update(kind="verified"))),
        ("theorem not declared", broken(lambda r: r["checks"][0]["theorems"].append("Antecedent.Missing.no_such"))),
        ("missing operation field", broken(lambda r: r["operation"].pop("premises"))),
        ("nothing trusted", broken(lambda r: r.update(trusted=[]))),
        ("stale digest", broken(lambda r: r["result"].update(inputs_digest="0" * 64))),
    ]
    failures = 0
    for name, d in cases:
        if not validate(root, d, toolchain):
            print(f"assurance self-test: {name} was accepted", file=sys.stderr)
            failures += 1
    # A token change to a covered Rust item stales every record that binds it.
    item_kind, name = rust_item.rsplit("/", 1)[-1].split("#", 1)[0].split(":", 1)
    mutated = re.sub(rf"\b{item_kind} {re.escape(name)}\b", f"{item_kind} {name}Changed", original, count=1)
    try:
        (root / rust_file).write_text(mutated)
        if mutated == original or not validate(root, data, toolchain):
            print("assurance self-test: a changed covered Rust item kept its record current", file=sys.stderr)
            failures += 1
    finally:
        (root / rust_file).write_text(original)
    # A sorry in a proof source is rejected.
    lean = root / first["checks"][0]["files"][0]
    lean_text = lean.read_text()
    try:
        lean.write_text(lean_text + "\ntheorem selftest_unproved : False := by sorry\n")
        if not validate(root, data, toolchain):
            print("assurance self-test: sorry was accepted", file=sys.stderr)
            failures += 1
    finally:
        lean.write_text(lean_text)
    print(f"assurance self-test: {len(cases) + 2 - failures}/{len(cases) + 2} broken inputs rejected")
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["check"]:
        return check()
    if argv[:1] == ["run"]:
        return run(argv[1:])
    if argv[:1] == ["--self-test"]:
        return self_test()
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
