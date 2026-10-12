#!/usr/bin/env python3
"""Run conformance reference generators in ``--check`` mode.

Exact float documents are compared with ``numpy.isclose(..., rtol=0)`` inside
the generators (not ``dict !=``). ``generate_frontdoor.py`` compares the
committed CSV columns the same way. None of these rerun a Monte Carlo study.

    python3 scripts/check_conformance_references.py
"""

from __future__ import annotations

import subprocess
import sys

from _repo import REPO_ROOT

GENERATORS = (
    "conformance/estimate/rd_sharp/reference.py",
    "conformance/estimate/cpdag_ate_envelope/reference.py",
    "conformance/estimate/pag_ate_envelope_identified/reference.py",
    "conformance/estimate/path_specific_edge_gformula/reference.py",
    "conformance/estimate/temporal_class_envelope/identified_pag_reference.py",
    "conformance/bayesian/posterior_moments/generate.py",
    "conformance/estimate/noisy_estimators/generate_frontdoor.py",
)


def python_executable() -> str:
    venv = REPO_ROOT / "python" / ".venv" / "bin" / "python"
    if venv.is_file():
        return str(venv)
    return sys.executable


def main() -> int:
    executable = python_executable()
    failed = False
    for rel in GENERATORS:
        command = [executable, str(REPO_ROOT / rel), "--check"]
        print(" ".join(command))
        result = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if result.returncode != 0:
            failed = True
    if failed:
        print("conformance reference check FAILED", file=sys.stderr)
        return 1
    print("conformance reference check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
