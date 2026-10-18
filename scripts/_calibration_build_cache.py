"""Prevent stale Cargo artifacts when equal-version checkouts share a target directory."""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = "calibration_release_source_context.v1"


def source_inputs(root: Path) -> dict[str, str]:
    """Include Cargo metadata and crate sources/assets, including build-script inputs."""
    paths = [root / name for name in (
        "Cargo.toml", "Cargo.lock", "python/Cargo.toml", "python/build.rs",
        "python/pyproject.toml",
    )]
    # The PyO3 workspace member is outside crates/. Include its native inputs,
    # not its editable extension, Python bytecode or other packaging outputs.
    # build.rs generates its linker files in OUT_DIR, already under target/.
    for folder in (root / ".cargo", root / "crates", root / "python/src"):
        if folder.exists():
            paths.extend(folder.rglob("*"))
    result = {}
    for path in sorted(paths):
        relative = path.relative_to(root)
        if not path.is_file() or any(part in {"target", ".git"} for part in relative.parts):
            continue
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"build input escapes the measured checkout: {relative}")
        result[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


@contextmanager
def _exclusive(path: Path):
    """Serialize cooperating calibration prebuilds for the entire Cargo invocation."""
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def guarded_release_cache(root: Path, target: Path | None = None):
    """Invalidate changed inputs by mtime; never delete artifacts or change source bytes.

    Cargo can rebase relative dep-info onto another checkout whose files predate
    a recently produced artifact. A content receipt identifies those changed inputs
    even when Cargo's usual timestamp test would incorrectly call them fresh.
    """
    root = root.resolve()
    if target is None:
        target = Path(os.environ.get("CARGO_TARGET_DIR", root / "target"))
        if not target.is_absolute():
            target = root / target
    target.mkdir(parents=True, exist_ok=True)
    receipt = target / ".calibration-release-source-context.json"
    with _exclusive(target / ".calibration-release-source.lock"):
        inputs = source_inputs(root)
        previous = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else {}
        if previous and previous.get("schema") != SCHEMA:
            raise ValueError("unknown calibration Cargo cache context")
        changed = [
            name
            for name, digest in inputs.items()
            if previous.get("inputs", {}).get(name) != digest
        ]
        # Also invalidate any old dep-info referring to files removed from this tree.
        deleted = sorted(set(previous.get("inputs", {})) - inputs.keys())
        for name in changed:
            os.utime(root / name, None)
        for name in deleted:
            parent = (root / name).parent
            while not parent.exists() and parent != root:
                parent = parent.parent
            os.utime(parent, None)
        context = {
            "schema": SCHEMA,
            "root": str(root),
            "inputs": inputs,
            "source_inputs_sha256": hashlib.sha256(
                json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "invalidated_inputs": changed,
            "deleted_inputs": deleted,
            "started_ns": time.time_ns(),
            "status": "prebuild_started",
        }
        receipt.write_text(json.dumps(context, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"== Cargo source guard: {len(changed)} changed/unknown inputs invalidated; {len(inputs)} content hashes",
            flush=True,
        )
        try:
            yield context
        except BaseException:
            context["status"] = "prebuild_interrupted"
            raise
        else:
            context["status"] = "prebuild_returned"
        finally:
            context["finished_ns"] = time.time_ns()
            receipt.write_text(json.dumps(context, sort_keys=True) + "\n", encoding="utf-8")
