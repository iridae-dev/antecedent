"""Draft Nix build bindings for disposable fixtures, never evidence attestation.

The manifest checker refuses the reserved ``nix_build`` field until production
publication binds verified inputs, the actual executed artifact, and policy.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from nix_artifact_identity import recheck_artifact, target_binary
from nix_evidence_source import verify_committed_snapshot


class InvalidNixReceipt(ValueError):
    """A draft's inputs or declared trust assumptions are inconsistent."""


ASSUMPTIONS = frozenset({"daemon", "sandbox", "private_tmp", "host_runtime", "clock_entropy", "package_sources"})


def draft_receipt(build: dict, assumptions: dict[str, str]) -> dict:
    """Bind a build diagnostic and explicit assumptions without attesting them.

    Callers cannot turn a draft into an attestation by setting a boolean. Source
    verification and artifact rechecking remain explicit separate operations.
    """
    try:
        source, artifact, tools, graph = (build[k] for k in ("source", "artifact", "tools", "compiler_graph"))
        if (build["evidence_authorized"] is not False or build["sandbox_verified"] is not False
                or artifact["sandbox_verified"] is not False):
            raise InvalidNixReceipt("build must remain unverified")
        if (artifact["source"] != source["store_path"] or artifact["source_nar_hash"] != source["nar_hash"]
                or not source["commit"] or not artifact["binary_sha256"] or not artifact["output_nar_hash"]
                or not artifact["derivation"] or not artifact["recipe_sha256"]):
            raise InvalidNixReceipt("build source or artifact identity is incomplete")
        kind, name, crate, features = (artifact[k] for k in ("target_kind", "target_name", "crate", "features"))
        if (kind not in ("lib", "test") or not isinstance(name, str) or not isinstance(crate, str)
                or not crate or not isinstance(features, list)
                or any(not isinstance(feature, str) or not feature for feature in features)):
            raise InvalidNixReceipt("missing build target or features")
        if not isinstance(assumptions, dict) or set(assumptions) != ASSUMPTIONS or not all(
                isinstance(value, str) and value.strip() for value in assumptions.values()):
            raise InvalidNixReceipt("all boundary trust assumptions must be explicit")
        if not isinstance(tools, dict) or not all(isinstance(tools.get(k), str) and tools[k]
                                                  for k in ("rustc", "rustc_commit", "cargo")):
            raise InvalidNixReceipt("missing build tool identity")
        if not isinstance(graph, dict) or not graph.get("units") or not graph.get("roots"):
            raise InvalidNixReceipt("missing compiler unit graph")
        graph_bytes = json.dumps(graph, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        # JSON roundtrip detaches the record from mutable caller-owned objects.
        return json.loads(json.dumps({
            "schema": "nix-build-draft-v1", "status": "unverified", "publication_authorized": False,
            "source": source, "artifact": artifact, "tools": tools,
            "compiler_graph_sha256": hashlib.sha256(graph_bytes).hexdigest(), "assumptions": assumptions,
        }))
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        if isinstance(error, InvalidNixReceipt):
            raise
        raise InvalidNixReceipt("malformed build identity") from error


def exercise_fixture(recipe_root: Path, checkout: Path, snapshot: Path, build: dict,
                     assumptions: dict[str, str], run_fixture: Callable[[Path], None]) -> dict:
    """Recheck source and artifact before/after a disposable fixture callback.

    The returned draft cannot authorize publication even if every check passes.
    """
    receipt = draft_receipt(build, assumptions)

    def check() -> None:
        verify_committed_snapshot(checkout, snapshot, receipt["source"]["commit"])
        recheck_artifact(recipe_root, receipt["artifact"])

    check()
    try:
        # The production runner must itself guarantee which bytes it executes;
        # this fixture callback is supplied the independently rechecked path.
        run_fixture(target_binary(receipt["artifact"]))
    finally:
        check()
    return receipt
