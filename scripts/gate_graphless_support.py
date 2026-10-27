#!/usr/bin/env python3
import sys
from pathlib import Path

import generate_graphless_support as graphless
import test_evidence as te


def citations() -> list[tuple[Path, str]]:
    cited = sorted(
        {
            row[field]
            for row in graphless.load_rows()
            for field in ("known_truth_test", "retained_route_test")
        }
    )
    if not cited:
        sys.exit("graphless support evidence list is empty")
    out = []
    for citation in cited:
        path, function = citation.split("::")
        out.append((te.ROOT / path, function))
    return out


def problems(cited: list[tuple[Path, str]]) -> list[str]:
    found = []
    for path, function in cited:
        _, reasons = te.resolve_rust_test(path, function)
        found.extend(f"{path.relative_to(te.ROOT)}::{function}: {reason}" for reason in reasons)
    return found


cited = citations()

if sys.argv[1:] == ["--self-test"]:
    # Stub the cargo listing so the gate's fail-closed behaviour is checked without
    # compiling: a cited name listed only under another module, or listed as
    # ignored, must fail; the exact listing must pass.
    targets: dict[tuple[str, tuple[str, ...]], set[str]] = {}
    for path, function in cited:
        root_file, crate, target = te.target_root(path)
        targets.setdefault((crate, tuple(target)), set()).add(
            te.libtest_name(path, function, root_file)
        )
    path0, function0 = cited[0]
    root0, crate0, target0 = te.target_root(path0)
    key0 = (crate0, tuple(target0))
    full0 = te.libtest_name(path0, function0, root0)

    def run(name: str, listing: dict, ignored: set[str], want_fail: bool) -> None:
        te._CARGO_LIST_CACHE.clear()
        for key, names in listing.items():
            te._CARGO_LIST_CACHE[key] = (set(names), ignored if key == key0 else set(), None)
        got = problems(cited)
        if bool(got) != want_fail:
            sys.exit(
                f"graphless self-test {name}: expected {'failure' if want_fail else 'pass'}, got {got}"
            )
        print(f"ok  {name}")

    run("exact listing passes", targets, set(), False)
    moved = {k: set(v) for k, v in targets.items()}
    moved[key0] = (moved[key0] - {full0}) | {f"elsewhere::tests::{function0}"}
    run("same function name in another module fails", moved, set(), True)
    run("cited test listed as ignored fails", targets, {full0}, True)
    print("graphless support self-test: ok")
    sys.exit(0)

print(
    f"== resolving {len(cited)} graphless evidence citation(s) via cargo test -- --list", flush=True
)
found = problems(cited)
if found:
    print("graphless support evidence is not an executing test:", file=sys.stderr)
    for problem in found:
        print(f"  {problem}", file=sys.stderr)
    sys.exit(1)
print("graphless support evidence: ok")
