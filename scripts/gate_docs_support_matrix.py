#!/usr/bin/env python3
"""Generated documentation blocks must match the support registry.

Runs the claim-inventory generator in --check mode: every marker block in the
docs equals what the generator renders from the registries. Hand-written text
outside the markers is not read.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

root = Path(".")
fail: list[str] = []

generated = subprocess.run(
    [
        sys.executable,
        str(Path(__file__).resolve().parent / "generate_claim_inventory.py"),
        "--check",
    ],
    cwd=root,
    capture_output=True,
    text=True,
)
if generated.returncode != 0:
    detail = (generated.stdout + generated.stderr).strip()
    fail.extend(detail.splitlines() or ["generate_claim_inventory.py --check failed"])

if fail:
    print("Docs support-matrix gate FAILED:")
    for item in fail:
        print(f" - {item}")
    sys.exit(1)

print(
    "Docs support-matrix OK (the per-query inventory and claim markers match "
    "the support registry and parity/claims.toml)"
)
