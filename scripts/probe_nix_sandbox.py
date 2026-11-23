#!/usr/bin/env python3
"""Probe selected Nix sandbox channels; never produces evidence records.

Negative results are not proof of a complete sandbox or daemon policy. The
program creates and reads only its own temporary sentinel and uses a temporary
loopback listener. It does not execute the project's scientific tests.
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import tempfile
from pathlib import Path

from _repo import REPO_ROOT


def probe_command(root: Path, sentinel: Path, workspace_sentinel: Path, port: str,
                  sandbox_paths: str | None = None) -> list[str]:
    """Build argv without evaluating source paths or accepting shell syntax."""
    if not sentinel.is_absolute() or not workspace_sentinel.is_absolute():
        raise ValueError("probe sentinels must be absolute paths")
    if not port.isdecimal() or not 0 < int(port) < 65536:
        raise ValueError("invalid loopback port")
    command = [
        "nix", "--extra-experimental-features", "nix-command flakes", "build",
        "--offline", "--no-link", "--print-out-paths", "--option", "substitute", "false",
        "--option", "builders", "",
    ]
    if sandbox_paths is not None:
        command.extend(["--option", "sandbox-paths", sandbox_paths])
    return command + ["--file", str(root / "nix/sandbox-probe.nix"),
        "--argstr", "hostProbe", str(sentinel),
        "--argstr", "workspaceProbe", str(workspace_sentinel),
        "--argstr", "loopbackPort", port,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--without-host-tmp", action="store_true", help="Test explicit Darwin paths excluding /private/tmp and /private/var/tmp; requires trusted Nix client")
    args = parser.parse_args()
    sandbox_paths = ("/System/Library/Frameworks /System/Library/PrivateFrameworks "
                     "/bin/bash /bin/sh /usr/lib") if args.without_host_tmp else None
    with tempfile.TemporaryDirectory(prefix="antecedent-nix-probe-", dir="/private/tmp") as tmp, \
         tempfile.TemporaryDirectory(prefix=".nix-probe-", dir=REPO_ROOT) as workspace_tmp:
        directory = Path(tmp)
        directory.chmod(0o755)
        sentinel = directory / "sentinel"
        sentinel.write_bytes(b"probe")
        sentinel.chmod(0o644)
        workspace_dir = Path(workspace_tmp)
        workspace_dir.chmod(0o755)
        workspace_sentinel = workspace_dir / "sentinel"
        workspace_sentinel.write_bytes(b"probe")
        workspace_sentinel.chmod(0o644)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = str(listener.getsockname()[1])
            result = subprocess.run(
                probe_command(REPO_ROOT, sentinel, workspace_sentinel, port, sandbox_paths),
                cwd=REPO_ROOT, capture_output=True, text=True, check=False,
            )
            if result.returncode:
                print("Sandbox probe build failed; no boundary conclusion can be drawn.")
                print(result.stderr[-3000:])
                return result.returncode
            paths = result.stdout.strip().splitlines()
            if len(paths) != 1 or not paths[0].startswith("/nix/store/"):
                print("Sandbox probe returned an unexpected output path.")
                return 1
            report = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
            print(json.dumps(report, indent=2, sort_keys=True))
            print("These probes do not establish a complete or reproducible build boundary.")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
