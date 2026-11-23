#!/usr/bin/env python3
"""Generate one compiler-path test and one replay test per route of the supported cells.

    python3 scripts/generate_supported_route_tests.py          # rewrite the generated blocks
    python3 scripts/generate_supported_route_tests.py --check  # fail when they drift

`parity/supported_routes.toml` assigns every supported cell to the route its execution ran:
an identifier, an estimator and a lowering. Cells of one route run the same code, so each
route gets its own tests in `crates/antecedent/tests/supported_compiler.rs`, and its own
scientific rows in `parity/scientific_tests.toml`. A record then binds only the code its
route's cells run, and a change to one route re-owes that route's tests alone.

Both files hold the generated text between BEGIN/END marker lines; everything else is
left as it is. After a rewrite, `python3 scripts/evidence_audit.py --write` regenerates the
nextest default filter from the scientific rows.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROUTES = ROOT / "parity/supported_routes.toml"
TESTS = ROOT / "crates/antecedent/tests/supported_compiler.rs"
SCIENTIFIC = ROOT / "parity/scientific_tests.toml"
RUST_BEGIN = "// BEGIN generated route tests (scripts/generate_supported_route_tests.py)"
RUST_END = "// END generated route tests"
TOML_BEGIN = "# BEGIN generated route tests (scripts/generate_supported_route_tests.py)"
TOML_END = "# END generated route tests"
# The fixtures the cell setups of crates/antecedent/tests/supported_compiler.rs read.
FIXTURES = [
    "conformance/estimate/cross_world_class_truth",
    "conformance/estimate/functional_graph_posterior_mixtures",
    "conformance/estimate/pag_ate_envelope_identified",
    "conformance/estimate/temporal_class_envelope/identified_pag.json",
    "conformance/estimate/temporal_window_truth",
    "conformance/response/class_aware_envelope/pag_identified.json",
    "conformance/response/temporal_admg_derivative_states",
]


def routes() -> list[tuple[str, str, str, int]]:
    """Distinct (identifier, estimator, lowering) in file order, with their cell counts."""
    counts: dict[tuple[str, str, str], int] = {}
    for row in tomllib.loads(ROUTES.read_text(encoding="utf-8"))["route"]:
        key = (row["identifier"], row["estimator"], row["lowering"])
        counts[key] = counts.get(key, 0) + 1
    return [(*key, n) for key, n in counts.items()]


def slug(identifier: str, estimator: str, lowering: str) -> str:
    camel = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", lowering)
    text = f"{estimator}_{identifier}_{camel}".lower()
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text)).strip("_")


def named() -> list[tuple[str, str, str, str, int]]:
    out, seen = [], set()
    for identifier, estimator, lowering, n in routes():
        name = slug(identifier, estimator, lowering)
        if name in seen:
            raise SystemExit(f"two routes share the test name {name}")
        seen.add(name)
        out.append((name, identifier, estimator, lowering, n))
    return out


def rust_block(rows) -> str:
    def lit(text: str) -> str:
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'

    lines = [RUST_BEGIN, "", "/// Every route of the committed route file, one generated test pair each.",
             "const ROUTE_TESTS: &[(&str, &str, &str)] = &["]
    for _, identifier, estimator, lowering, _ in rows:
        lines.append(f"    ({lit(identifier)}, {lit(estimator)}, {lit(lowering)}),")
    lines.append("];")
    for name, identifier, estimator, lowering, n in rows:
        args = f"{lit(identifier)}, {lit(estimator)}, {lit(lowering)}"
        cells = "cell" if n == 1 else "cells"
        lines += [
            "",
            f"/// {n} supported {cells}: `{identifier}` / `{estimator}` / `{lowering}`.",
            "#[test]",
            f"fn compiler_path_{name}() {{",
            f"    route_completes_the_compiler_path({args});",
            "}",
            "",
            "#[test]",
            f"fn replay_{name}() {{",
            f"    route_replays_to_the_published_number({args});",
            "}",
        ]
    lines.append(RUST_END)
    return "\n".join(lines)


def toml_block(rows) -> str:
    fixtures = "[" + ", ".join(f'"{f}"' for f in FIXTURES) + "]"
    lines = [TOML_BEGIN]
    for name, identifier, estimator, lowering, n in rows:
        cells = "cell" if n == 1 else "cells"
        lines += [
            "",
            "[[test]]",
            'evidence_test = "crates/antecedent/tests/supported_compiler.rs"',
            f'evidence_assertion = "compiler_path_{name}"',
            'kind = "route_sweep"',
            f"fixtures = {fixtures}",
            f'why = "The {n} supported {cells} of {identifier} / {estimator} / {lowering} run the compiler path with real estimators and the route the committed route file records."',
            "",
            "[[test]]",
            'evidence_test = "crates/antecedent/tests/supported_compiler.rs"',
            f'evidence_assertion = "replay_{name}"',
            'kind = "route_sweep"',
            f"fixtures = {fixtures}",
            f'why = "Every ninth supported cell of {identifier} / {estimator} / {lowering}, its first always, replays bit for bit; a fresh consumer reads back the number the route published."',
        ]
    lines.append(TOML_END)
    return "\n".join(lines)


def splice(text: str, begin: str, end: str, block: str, path: Path) -> str:
    try:
        a = text.index(begin)
        b = text.index(end, a) + len(end)
    except ValueError:
        raise SystemExit(f"{path.relative_to(ROOT)} lacks the generated-block markers") from None
    return text[:a] + block + text[b:]


def rustfmt(source: str) -> str:
    """The source as the repository's rustfmt configuration writes it."""
    return subprocess.run(
        ["rustfmt", "--edition", "2024", "--config-path", str(ROOT / "rustfmt.toml"), "--emit", "stdout"],
        input=source, capture_output=True, text=True, check=True,
    ).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="fail when a generated block drifts")
    args = parser.parse_args()
    rows = named()
    stale = []
    for path, begin, end, block in (
        (TESTS, RUST_BEGIN, RUST_END, rust_block(rows)),
        (SCIENTIFIC, TOML_BEGIN, TOML_END, toml_block(rows)),
    ):
        text = path.read_text(encoding="utf-8")
        new = splice(text, begin, end, block, path)
        if path == TESTS:
            new = rustfmt(new)
        if new != text:
            stale.append(path.relative_to(ROOT))
            if not args.check:
                path.write_text(new, encoding="utf-8")
    if args.check and stale:
        print("generated route tests drifted:", ", ".join(map(str, stale)), file=sys.stderr)
        return 1
    print(f"{len(rows)} routes; {'stale: ' + ', '.join(map(str, stale)) if stale else 'current'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
