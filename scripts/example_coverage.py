#!/usr/bin/env python3
"""Account for every example-catalog entry: executed, compiled only, or skipped and why.

The accounting is read from what each runner selects, never from a list kept here:

* the Python suite (`python/tests/test_examples_run.py`) runs the examples and notebooks
  named by `python/tests/_example_selection.py`, skipping one whose optional module is
  absent;
* the documentation build runs the sources published pages embed
  (`run_doc_examples.embedded_sources`) and executes the notebooks
  (`render_notebooks.notebooks`) in real kernels;
* the optional-handoff check (`run_optional_doc_example.HANDOFFS`) requires the external
  half of a handoff example to execute;
* the Rust example suite (`crates/antecedent/tests/examples_run.rs`) includes an example
  and calls its `run`; a Cargo `[[example]]` target that the suite does not call is
  compiled only.

This reports selection: what a run of each lane would execute. It does not run anything
and is not a record that a lane ran.

    python3 scripts/example_coverage.py            # print the accounting
    python3 scripts/example_coverage.py --check    # also fail on an entry with no disposition
    python3 scripts/example_coverage.py --json OUT # also write it as JSON
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

from _repo import REPO_ROOT as ROOT
from _repo import load_toml, read_text, write_text
from render_notebooks import notebooks
from run_doc_examples import embedded_sources
from run_optional_doc_example import HANDOFFS

KINDS = ("python", "rust", "notebook")
EXECUTED = "executed"
CONDITIONAL = "executed when its optional dependency is installed, otherwise skipped"
COMPILED = "compiled only"
NONE = "no disposition"

RUST_SUITE = "crates/antecedent/tests/examples_run.rs"
RUST_MANIFEST = "crates/antecedent/Cargo.toml"
PYTHON_SUITE = "python/tests/test_examples_run.py"


def python_suite_selection(root: Path = ROOT):
    """The selection module the Python suite parametrizes over."""
    path = root / "python/tests/_example_selection.py"
    spec = importlib.util.spec_from_file_location("_example_selection", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rust_suite_runs(text: str) -> set[str]:
    """Examples the Rust suite includes and whose `run` a test calls."""
    included = dict(
        (module, f"examples/rust/{name}")
        for name, module in re.findall(
            r'#\[path = "(?:\.\./)+examples/rust/(\w+\.rs)"\]\s*mod (\w+);', text
        )
    )
    called = set(re.findall(r"\b(\w+)::run\(", text))
    return {path for module, path in included.items() if module in called}


def cargo_example_targets(text: str) -> set[str]:
    """Examples Cargo builds as `[[example]]` targets."""
    return {
        f"examples/rust/{name}"
        for name in re.findall(
            r'(?m)^\[\[example\]\]\n(?:(?!\[).*\n)*?path = "(?:\.\./)+examples/rust/(\w+\.rs)"',
            text,
        )
    }


def selections(root: Path = ROOT) -> dict[str, list[tuple[str, str, str]]]:
    """Every runner's claim on a path: path -> [(state, runner, condition)]."""
    claims: dict[str, list[tuple[str, str, str]]] = {}

    def claim(path: str, state: str, runner: str, condition: str = "") -> None:
        claims.setdefault(path, []).append((state, runner, condition))

    suite = python_suite_selection(root)
    for example in suite.EXAMPLES:
        needs = suite.EXAMPLE_REQUIRES.get(example.name)
        claim(
            example.relative_to(root).as_posix(),
            CONDITIONAL if needs else EXECUTED,
            PYTHON_SUITE,
            f"needs {needs}" if needs else "",
        )
    for notebook in suite.NOTEBOOKS:
        claim(
            notebook.relative_to(root).as_posix(),
            CONDITIONAL,
            PYTHON_SUITE,
            "needs " + ", ".join(suite.NOTEBOOK_REQUIRES),
        )
    for source in embedded_sources():
        claim(source, EXECUTED, "scripts/run_doc_examples.py", "embedded in a published page")
    for notebook in notebooks():
        claim(notebook.relative_to(ROOT).as_posix(), EXECUTED, "scripts/render_notebooks.py")
    for dependency, source in HANDOFFS.items():
        claim(
            source,
            EXECUTED,
            "scripts/run_optional_doc_example.py",
            f"external half, with {dependency} installed",
        )
    for path in sorted(rust_suite_runs(read_text(root / RUST_SUITE))):
        claim(path, EXECUTED, RUST_SUITE)
    for path in sorted(cargo_example_targets(read_text(root / RUST_MANIFEST))):
        claim(path, COMPILED, RUST_MANIFEST, "[[example]] target")
    return claims


def disposition(claims: list[tuple[str, str, str]]) -> str:
    states = {state for state, _, _ in claims}
    for state in (EXECUTED, CONDITIONAL, COMPILED):
        if state in states:
            return state
    return NONE


def account(entries: list[dict], claims: dict[str, list[tuple[str, str, str]]]) -> list[dict]:
    """One row per catalog artifact, with its disposition and the runners behind it."""
    rows = []
    for entry in entries:
        # An entry may declare, in the catalog itself, why an artifact is deliberately not
        # run: `unexecuted = { rust = "reason" }`.
        declared = entry.get("unexecuted", {})
        for kind in KINDS:
            if kind not in entry:
                continue
            found = claims.get(entry[kind], [])
            state = disposition(found)
            reason = declared.get(kind, "")
            if state == NONE and reason:
                state = "skipped"
            rows.append(
                {
                    "id": entry["id"],
                    "kind": kind,
                    "path": entry[kind],
                    "disposition": state,
                    "reason": reason,
                    "runners": [
                        {"state": s, "runner": runner, "condition": condition}
                        for s, runner, condition in found
                    ],
                }
            )
    return rows


def problems(rows: list[dict]) -> list[str]:
    out = [
        f"{row['id']}: {row['path']} is selected by no runner and declares no reason "
        "(run it, or state why not in the catalog entry's `unexecuted` table)"
        for row in rows
        if row["disposition"] == NONE
    ]
    out += [
        f"{row['id']}: {row['path']} declares an `unexecuted` reason but is "
        f"{row['disposition']}; remove the stale reason"
        for row in rows
        if row["reason"] and row["disposition"] != "skipped"
    ]
    return out


def render(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        detail = "; ".join(
            item["runner"] + (f" ({item['condition']})" if item["condition"] else "")
            for item in row["runners"]
        )
        lines.append(
            f"{row['id']} [{row['kind']}] {row['disposition']}: {row['reason'] or detail or '-'}"
        )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["disposition"]] = counts.get(row["disposition"], 0) + 1
    entries = len({row["id"] for row in rows})
    summary = ", ".join(f"{count} {state}" for state, count in sorted(counts.items()))
    lines.append(f"Example coverage: {entries} catalog entries, {len(rows)} artifacts: {summary}")
    return "\n".join(lines)


def current(root: Path = ROOT) -> list[dict]:
    return account(load_toml(root / "examples/catalog.toml")["example"], selections(root))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", type=Path, help="also write the accounting to this file")
    args = parser.parse_args(argv)
    rows = current()
    print(render(rows))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        write_text(args.json, json.dumps(rows, indent=2) + "\n")
    failures = problems(rows)
    if failures:
        print("Example coverage accounting FAILED:" if args.check else "Unaccounted examples:")
        for failure in failures:
            print(f" - {failure}")
        return 1 if args.check else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
