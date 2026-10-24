"""Shared reading of the lane and control tables for the lane harness.

Pure functions over `lanes.toml` and `controls/workloads.toml`. Nothing here imports
antecedent, runs cargo, or measures anything.
"""

from __future__ import annotations

import re
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ENTRY_DIR = "parity/performance_lanes/entry_points/"
LANE_IDS = ("first_answer", "reuse_throughput", "row_capacity", "exact_capacity")
TIERS = ("1e4", "1e5", "1e6", "1e7")
DEFAULT_TIER = "control_default"
ENV_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")
BENCH = re.compile(r"-p\s+(?P<crate>[\w-]+)\s+--bench\s+(?P<bench>[\w-]+)")
EXAMPLE = re.compile(r"-p\s+(?P<crate>[\w-]+)\s+--example\s+(?P<example>[\w-]+)")


def read_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_tables(repo: Path = REPO) -> tuple[dict, dict[str, dict]]:
    """The parsed lane table and the control rows by id."""
    lanes = read_toml(repo / "parity" / "performance_lanes" / "lanes.toml")
    workloads = read_toml(repo / "parity" / "controls" / "workloads.toml")
    return lanes, {row["id"]: row for row in workloads["control"]}


def lane_rows(lanes: dict) -> dict[str, dict]:
    return {row["id"]: row for row in lanes["lane"]}


def entries_by_control(lanes: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for entry in lanes["entry_point"]:
        grouped.setdefault(entry["control"], []).append(entry)
    return grouped


@dataclass(frozen=True)
class Command:
    """A registered command: leading `NAME=value` assignments, then the argument vector."""

    env: tuple[tuple[str, str], ...]
    argv: tuple[str, ...]

    @property
    def environment(self) -> dict[str, str]:
        return dict(self.env)


def parse_command(text: str) -> Command:
    tokens = shlex.split(text)
    env: list[tuple[str, str]] = []
    while tokens and ENV_ASSIGNMENT.fullmatch(tokens[0]):
        name, value = tokens.pop(0).split("=", 1)
        env.append((name, value))
    if not tokens:
        raise ValueError(f"command has no program: {text!r}")
    return Command(tuple(env), tuple(tokens))


def cargo_subcommand(argv: tuple[str, ...]) -> int | None:
    """Index of the cargo subcommand in `argv`, skipping a `+toolchain` token."""
    if argv[0] != "cargo":
        return None
    for index, token in enumerate(argv[1:], start=1):
        if not token.startswith("+"):
            return index
    return None


def command_kind(command: Command) -> str:
    """`entry` (a lane entry-point script), `example` (a repository Python example),
    `cargo_bench`, `cargo_run` or `other`."""
    argv = command.argv
    if argv[0] == "python" and len(argv) > 1:
        return "entry" if argv[1].startswith(ENTRY_DIR) else "example"
    index = cargo_subcommand(argv)
    if index is not None:
        return {"bench": "cargo_bench", "run": "cargo_run"}.get(argv[index], "other")
    return "other"


def prebuild_argv(command: Command) -> tuple[str, ...] | None:
    """The cargo invocation that compiles what `command` runs without running it."""
    kind = command_kind(command)
    argv = list(command.argv)
    index = cargo_subcommand(command.argv)
    if kind == "cargo_bench" and index is not None:
        if "--" in argv:
            argv.insert(argv.index("--"), "--no-run")
        else:
            argv.append("--no-run")
        return tuple(argv)
    if kind == "cargo_run" and index is not None:
        argv[index] = "build"
        return tuple(argv)
    return None


def rows_mechanism(command: Command) -> str | None:
    """How a command takes a row count: `arg` (--rows), `env:NAME`, or None (fixed size)."""
    if command_kind(command) == "entry" and "run_example.py" not in command.argv[1]:
        return "arg"
    for name, _ in command.env:
        if name == "USER_PATH_N":
            return f"env:{name}"
    return None


def tier_rows(tier: str) -> int:
    """`1e4` -> 10000."""
    return int(float(tier))


def with_rows(command: Command, tier: str) -> Command:
    """`command` with its row count set to `tier`; unchanged for `control_default`."""
    if tier == DEFAULT_TIER:
        return command
    mechanism = rows_mechanism(command)
    rows = tier_rows(tier)
    if mechanism == "arg":
        return Command(command.env, command.argv + ("--rows", str(rows)))
    if mechanism and mechanism.startswith("env:"):
        name = mechanism[4:]
        env = tuple((k, str(rows) if k == name else v) for k, v in command.env)
        return Command(env, command.argv)
    raise ValueError(f"{command.argv[0]} takes no row count; cannot run tier {tier}")


def primary_entries(lanes: dict, workloads: dict[str, dict]) -> dict[str, list[dict]]:
    """Per control, the registered rows, the one matching `workloads.toml` first."""
    ordered: dict[str, list[dict]] = {}
    for control, rows in entries_by_control(lanes).items():
        wanted = workloads[control].get("command")
        ordered[control] = sorted(rows, key=lambda row: row["frozen_command"] != wanted)
    return ordered
