"""Preparatory Nix build adapter; this is not an evidence-earning boundary.

The diagnostic inspects *client* configuration. It does not prove what a Nix
daemon enforces. In particular, a
store path or a client-supplied option cannot attest to the daemon's sandbox,
local execution, or the input paths exposed inside a macOS sandbox. No caller
may treat these helpers as authorization to publish evidence.
"""
from __future__ import annotations

import json
import subprocess


def client_policy_diagnostics(config: dict) -> dict:
    """Report visible client settings without claiming to attest a daemon.

    ``nix config show --json`` reports the configuration of the calling Nix
    process; an untrusted client cannot assume daemon settings match it. The
    returned ``boundary_verified`` therefore remains false even when all
    client settings are conservative. Sandbox paths are returned for separate
    review: on macOS they can include mutable host directories.
    """
    def value(name: str):
        entry = config.get(name)
        return entry.get("value") if isinstance(entry, dict) else None

    failures = []
    for key, expected in (("sandbox", True), ("sandbox-fallback", False),
                          ("substitute", False)):
        actual = value(key)
        if type(actual) is not bool or actual is not expected:
            failures.append(f"{key} is not {str(expected).lower()}")
    if value("builders") not in ("", []):
        failures.append("remote builders are configured or unknown")
    paths = value("sandbox-paths")
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        failures.append("sandbox paths are unknown")
        paths = []
    return {
        "client_policy_ok": not failures,
        "problems": failures,
        "sandbox_paths_requiring_review": paths,
        "boundary_verified": False,
        "unverified": "daemon policy, local execution, sandbox paths and build inputs",
    }


def main() -> int:
    """Inspect client configuration; a zero exit cannot attest daemon policy."""
    try:
        proc = subprocess.run(
            ["nix", "--extra-experimental-features", "nix-command", "config", "show", "--json"],
            capture_output=True, text=True, check=True,
        )
        config = json.loads(proc.stdout)
        if not isinstance(config, dict):
            raise ValueError("Nix configuration is not an object")
        result = client_policy_diagnostics(config)
    except (OSError, subprocess.CalledProcessError, ValueError):
        print("Nix client configuration could not be read; evidence boundary is unverified.")
        return 1
    print(json.dumps(result, indent=2))
    # Client configuration alone never proves that the daemon enforced it.
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
