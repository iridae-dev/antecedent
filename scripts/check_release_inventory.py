#!/usr/bin/env python3
"""The release inventory checks, run as one gate.

Each section reads a registry of the release surface and fails when it disagrees with the
sources it describes. Every section runs even when an earlier one fails, so one run reports
every disagreement.

  acceptance manifest    every acceptance case is bound to a live test
  counterparts           every method records all four quadrants
  release scope          every scope row is matched to its route, owner, fixtures and consumer
  method children        every leaf names a registered fixture and a reading consumer
  route manifest         every method route is verified against its implementation
  method specifications  every specification is reconciled with its executing route
  closure report         the generated closure report is current

    python3 scripts/check_release_inventory.py                 # all sections
    python3 scripts/check_release_inventory.py --require-zero  # release: the closure report has
                                                               # no unresolved row either
    python3 scripts/check_release_inventory.py --self-test     # each section's own self-test
"""

from __future__ import annotations

import subprocess
import sys

from _repo import REPO_ROOT as ROOT

SECTIONS: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    # (label, script, check arguments, self-test arguments)
    ("acceptance manifest: every case bound to a live test",
     "check_acceptance_manifest.py", ("--require-bound",), ("--self-test",)),
    ("counterparts: every method records all four quadrants",
     "check_counterparts.py", (), ("--self-test",)),
    ("release scope: every row matched to its route, owner, fixtures and consumer",
     "reconcile_release_scope.py", ("--check",), ("--self-test",)),
    ("method children: every leaf names a registered fixture and a reading consumer",
     "check_method_children.py", (), ("--self-test",)),
    ("route manifest: every method verified against its implementation",
     "route_methods.py", ("--check",), ("--self-test",)),
    ("method specifications reconciled with the executing routes",
     "reconcile_method_specs.py", ("--check",), ("--self-test",)),
    ("closure report is current",
     "closure_audit.py", ("--check",), ("--self-test",)),
)

REQUIRE_ZERO = ("closure report has no unresolved row", "closure_audit.py", ("--require-zero",))


def run_section(label: str, script: str, args: tuple[str, ...]) -> bool:
    print(f"== {label} ==", flush=True)
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], cwd=ROOT)
    return result.returncode == 0


def main(argv: list[str]) -> int:
    unknown = [a for a in argv if a not in {"--require-zero", "--self-test"}]
    if unknown:
        print(f"usage: {sys.argv[0]} [--require-zero|--self-test]", file=sys.stderr)
        return 2
    if "--self-test" in argv:
        failed = [
            script
            for _, script, _, selftest in SECTIONS
            if not run_section(f"{script} self-test", script, selftest)
        ]
    else:
        sections = [(label, script, args) for label, script, args, _ in SECTIONS]
        if "--require-zero" in argv:
            sections.append(REQUIRE_ZERO)
        failed = [label for label, script, args in sections if not run_section(label, script, args)]
    if failed:
        print("release inventory FAILED in: " + "; ".join(failed), file=sys.stderr)
        return 1
    print("release inventory: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
