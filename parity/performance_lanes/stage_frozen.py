#!/usr/bin/env python3
"""Stage the frozen v2.4.1 controls and the candidate build for a lane run.

Everything a person used to edit by hand happens here, recorded, and can be repeated:

- a detached git worktree of the predecessor tag (`--frozen-root`);
- for every Rust entry point the tag does not ship, the bench or example file copied from the
  current tree and its `[[bench]]` stanza appended to the frozen crate's `Cargo.toml`;
- for every Python entry point or example the tag does not ship, the file copied beside it;
- one virtual environment with the published `antecedent==2.4.1` wheel (or the frozen tree
  built from source) and one with the current tree built by maturin.

    python parity/performance_lanes/stage_frozen.py            # stage
    python parity/performance_lanes/stage_frozen.py --dry-run  # print the plan only

The staging record is `<frozen-root>/.staged_entry_points.json`. Nothing is measured here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from lanes_common import (
    BENCH,
    EXAMPLE,
    REPO,
    ROW_AWARE_RUST,
    ROWS_ENV,
    entries_by_control,
    load_tables,
)

TAG = "v2.4.1"
PUBLISHED_VERSION = "2.4.1"
DEFAULT_ROOT = REPO / "target" / "performance_lanes"
# The second candidate build: the same source compiled for the host CPU, so the kernels the
# compiler can vectorize use the full instruction set of the machine the lane runs on.
DISPATCH_RUSTFLAGS = "-C target-cpu=native"


@dataclass(frozen=True)
class Layout:
    """Where the two builds live on this host."""

    repo: Path
    frozen_root: Path
    frozen_python: Path
    candidate_python: Path
    frozen_target: Path
    candidate_target: Path
    build: str = "portable"
    rustflags: str | None = None

    @classmethod
    def under(
        cls,
        work: Path,
        repo: Path = REPO,
        build: str = "portable",
        rustflags: str | None = None,
    ) -> Layout:
        """The `portable` build is the default compile. `runtime_dispatched` is the same source
        compiled with `rustflags` (see DISPATCH_RUSTFLAGS) into its own environment and target
        directory, so both candidate builds can sit side by side against one frozen build."""
        bindir = "Scripts" if os.name == "nt" else "bin"
        suffix = "" if build == "portable" else f"_{build}"
        return cls(
            repo=repo,
            frozen_root=work / "frozen",
            frozen_python=work / "frozen_venv" / bindir / "python",
            candidate_python=work / f"candidate_venv{suffix}" / bindir / "python",
            frozen_target=work / "frozen_target",
            candidate_target=work / f"candidate_target{suffix}",
            build=build,
            rustflags=rustflags if build != "portable" else None,
        )

    def candidate_environment(self) -> dict[str, str]:
        """Environment for compiling or running the candidate."""
        env = {**os.environ, "CARGO_TARGET_DIR": str(self.candidate_target)}
        if self.rustflags:
            env["RUSTFLAGS"] = " ".join(
                filter(None, [os.environ.get("RUSTFLAGS"), self.rustflags])
            )
        return env


def bench_stanza(name: str) -> str:
    return f'\n[[bench]]\nname = "{name}"\nharness = false\n'


def missing_in_frozen(layout: Layout, lanes: dict) -> list[dict]:
    """The files and bench registrations the tag lacks, one action each, in a stable order."""
    actions: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for control, entries in sorted(entries_by_control(lanes).items()):
        for entry in entries:
            command = entry["frozen_command"]
            for name in entry["files"]:
                frozen_file = layout.frozen_root / name
                if (
                    name in ROW_AWARE_RUST.values()
                    and frozen_file.is_file()
                    and ROWS_ENV not in frozen_file.read_text(encoding="utf-8")
                ):
                    # Same control, same registered size; the file only learns to read the row
                    # tier from the environment, so both builds run the same row sweep.
                    key = ("replace", name)
                    if key not in seen:
                        seen.add(key)
                        actions.append(
                            {"action": "copy", "file": name, "control": control}
                        )
                    continue
                if name.startswith("parity/") or frozen_file.is_file():
                    continue
                key = ("copy", name)
                if key not in seen:
                    seen.add(key)
                    actions.append({"action": "copy", "file": name, "control": control})
            match = BENCH.search(command)
            if match and not EXAMPLE.search(command):
                manifest = layout.frozen_root / "crates" / match["crate"] / "Cargo.toml"
                registered = manifest.is_file() and (
                    f'name = "{match["bench"]}"' in manifest.read_text(encoding="utf-8")
                )
                key = ("register", f"{match['crate']}/{match['bench']}")
                if not registered and key not in seen:
                    seen.add(key)
                    actions.append(
                        {
                            "action": "register_bench",
                            "crate": match["crate"],
                            "bench": match["bench"],
                            "control": control,
                        }
                    )
    return actions


def apply_action(layout: Layout, action: dict) -> None:
    if action["action"] == "copy":
        source = layout.repo / action["file"]
        target = layout.frozen_root / action["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    elif action["action"] == "register_bench":
        manifest = layout.frozen_root / "crates" / action["crate"] / "Cargo.toml"
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(bench_stanza(action["bench"]))
    else:  # pragma: no cover
        raise ValueError(action)


def run(argv: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print("+ " + " ".join(argv), flush=True)
    # A venv and a conda prefix together make maturin refuse to pick an interpreter; the
    # environments these builds target are the venvs staged here, so the conda one is dropped.
    base = dict(os.environ if env is None else env)
    if "VIRTUAL_ENV" in base or any("venv" in part for part in argv):
        base.pop("CONDA_PREFIX", None)
    subprocess.run(argv, cwd=cwd, env=base, check=True)


def ensure_worktree(layout: Layout, dry_run: bool) -> list[str]:
    root = layout.frozen_root
    if (root / ".git").exists():
        return []
    plan = [f"git worktree add --detach {root} {TAG}"]
    if not dry_run:
        root.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "git",
                "-C",
                str(layout.repo),
                "worktree",
                "add",
                "--detach",
                str(root),
                TAG,
            ]
        )
    return plan


def ensure_venv(python: Path, base: str, dry_run: bool) -> bool:
    if python.is_file():
        if not dry_run:
            # maturin installs dependency groups, which needs a recent pip.
            run([str(python), "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
        return False
    if not dry_run:
        python.parent.parent.parent.mkdir(parents=True, exist_ok=True)
        run([base, "-m", "venv", str(python.parent.parent)])
        run([str(python), "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
        run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--quiet",
                "numpy",
                "pydantic",
                "maturin",
            ]
        )
    return True


def installed_version(python: Path) -> str | None:
    if not python.is_file():
        return None
    probe = subprocess.run(
        [str(python), "-c", "import antecedent; print(antecedent.__version__)"],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.stdout.strip() if probe.returncode == 0 else None


def stage(
    layout: Layout,
    *,
    base_python: str = sys.executable,
    frozen_from_source: bool = False,
    build_candidate: bool = True,
    dry_run: bool = False,
) -> dict:
    """Make both builds ready. Returns the staging record (also printed in dry-run mode)."""
    lanes, _ = load_tables(layout.repo)
    record: dict = {"tag": TAG, "worktree": ensure_worktree(layout, dry_run)}
    actions = missing_in_frozen(layout, lanes)
    record["actions"] = actions
    if not dry_run:
        for action in actions:
            apply_action(layout, action)
        layout.frozen_root.mkdir(parents=True, exist_ok=True)
        (layout.frozen_root / ".staged_entry_points.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )

    env = layout.candidate_environment()
    record["frozen_venv_created"] = ensure_venv(
        layout.frozen_python, base_python, dry_run
    )
    record["candidate_venv_created"] = ensure_venv(
        layout.candidate_python, base_python, dry_run
    )
    if not dry_run:
        if installed_version(layout.frozen_python) is None:
            if frozen_from_source:
                run(
                    [
                        str(layout.frozen_python),
                        "-m",
                        "maturin",
                        "develop",
                        "--release",
                        "-m",
                        str(layout.frozen_root / "python" / "Cargo.toml"),
                    ],
                    env={
                        **os.environ,
                        "VIRTUAL_ENV": str(layout.frozen_python.parent.parent),
                        "CARGO_TARGET_DIR": str(layout.frozen_target),
                    },
                )
            else:
                run(
                    [
                        str(layout.frozen_python),
                        "-m",
                        "pip",
                        "install",
                        "--quiet",
                        f"antecedent=={PUBLISHED_VERSION}",
                    ]
                )
        # The development branch keeps the predecessor's version string, so a candidate
        # environment is always rebuilt (maturin is incremental) rather than trusted.
        if build_candidate:
            run(
                [
                    str(layout.candidate_python),
                    "-m",
                    "maturin",
                    "develop",
                    "--release",
                    "-m",
                    str(layout.repo / "python" / "Cargo.toml"),
                ],
                env={**env, "VIRTUAL_ENV": str(layout.candidate_python.parent.parent)},
            )
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--frozen-from-source",
        action="store_true",
        help="build the tag with maturin instead of installing the published wheel",
    )
    parser.add_argument(
        "--skip-candidate-build",
        action="store_true",
        help="use the candidate environment as it is",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan, change nothing"
    )
    args = parser.parse_args(argv)
    record = stage(
        Layout.under(args.work_dir.resolve()),
        frozen_from_source=args.frozen_from_source,
        build_candidate=not args.skip_candidate_build,
        dry_run=args.dry_run,
    )
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
