#!/usr/bin/env python3
"""Reproducible snapshot of the tagged prior release's promotion baseline.

The snapshot answers one question for new work: *what did the last tagged
release license and keep closed, before this branch changed anything?* It reads
only git objects (never the working tree) for

* the promotion registry that tag shipped (detected, not hard-coded),
* its support, stage and carryover registries,
* its producer/consumer matrix and coverage records,

records a SHA-256 for each file, resolves the tag commit, records the base
commit the refactored branch is compared from (the SHA-256 of the same roles at
that base is listed beside the tag's so drift is visible), and lists every
closed inherited route so a classification pass can record, per route, whether
it is a prerequisite, a deliberate refusal, or reopened under a new record.

Outputs (both deterministic, no timestamps):

* ``parity/inherited_baseline.toml``: machine-readable; the per-route
  ``disposition`` fields are the only hand-maintained part and survive
  ``--write``.
* ``parity/inherited_baseline.md``: the same data rendered for review.

Usage::

    python3 scripts/snapshot_promotion_baseline.py            # --check
    python3 scripts/snapshot_promotion_baseline.py --write    # regenerate
    python3 scripts/snapshot_promotion_baseline.py --write --tag vX.Y.Z --base <commit>

``--check`` fails when the checked-in files differ from what git produces for
the recorded tag and base, when a classification is malformed, or when a
migration-decision pointer no longer resolves. It never reads the tag's files
from the working tree, so local edits cannot move the baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from _repo import REPO_ROOT, read_text, write_text

SNAPSHOT_TOML = "parity/inherited_baseline.toml"
SNAPSHOT_MD = "parity/inherited_baseline.md"

DISPOSITIONS = (
    "unclassified",
    "prerequisite",
    "deliberate_refusal",
    "reopened_under_new_record",
)

ABSENT = "absent"
SCHEMA_VERSION = 1

# (role, fixed path or None). Roles with None are discovered per commit.
ROLES: tuple[tuple[str, str | None], ...] = (
    ("promotion_registry", None),
    ("promotion_carryovers", "parity/promotion_carryovers.toml"),
    ("support_licensed", "parity/support_licensed.toml"),
    ("support_closed", "parity/support_closed.toml"),
    ("transport_stages", "parity/transport_stages.toml"),
    ("producer_consumer_registry", "parity/producer_consumer.toml"),
    ("producer_consumer_matrix", None),
    ("coverage_records", "parity/coverage_records.toml"),
    ("transport_coverage", "parity/transport_coverage.md"),
    ("counterfactual_coverage", "parity/counterfactual_coverage.md"),
)

_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
_REGISTRY_RE = re.compile(r"^parity/promotion[^/]*\.toml$")
_MATRIX_RE = re.compile(r"^docs/[^/]*producer-consumer-matrix\.md$")
_NOT_REGISTRY = {"parity/promotion_carryovers.toml"}


class SnapshotError(Exception):
    """A snapshot input is missing, inconsistent or malformed."""


# --------------------------------------------------------------------------
# git access
# --------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> bytes:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise SnapshotError(f"git {' '.join(args)} failed: {detail}")
    return proc.stdout


def latest_tag(repo: Path) -> str:
    """Highest ``vX.Y.Z`` tag by numeric version order."""
    names = _git(repo, "tag", "--list", "v[0-9]*").decode().split()
    versions = []
    for name in names:
        match = _TAG_RE.match(name)
        if match:
            versions.append((tuple(int(part) for part in match.groups()), name))
    if not versions:
        raise SnapshotError("no vX.Y.Z tag found; pass --tag")
    return max(versions)[1]


def resolve_commit(repo: Path, ref: str) -> str:
    return _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()


def tree_files(repo: Path, commit: str) -> set[str]:
    out = _git(repo, "ls-tree", "-r", "--name-only", "-z", commit)
    return {name for name in out.decode("utf-8").split("\0") if name}


def blob(repo: Path, commit: str, path: str) -> bytes:
    return _git(repo, "cat-file", "blob", f"{commit}:{path}")


def _parse(data: bytes, path: str) -> dict[str, Any]:
    try:
        return tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as err:
        raise SnapshotError(f"{path} is not valid TOML: {err}") from err


# --------------------------------------------------------------------------
# role discovery
# --------------------------------------------------------------------------


def _release_label(tag: str) -> str:
    match = _TAG_RE.match(tag)
    if not match:
        return ""
    return f"{match.group(1)}.{match.group(2)}"


def find_registry(repo: Path, commit: str, files: set[str], tag: str) -> str:
    """The promotion registry file a commit carries for ``tag``'s release."""
    candidates = sorted(p for p in files if _REGISTRY_RE.match(p) and p not in _NOT_REGISTRY)
    if not candidates:
        raise SnapshotError(f"no promotion registry file at {commit[:12]}")
    label = _release_label(tag)
    matching = []
    for path in candidates:
        doc = _parse(blob(repo, commit, path), path)
        if label and str(doc.get("release", "")) == label:
            matching.append(path)
    if len(matching) == 1:
        return matching[0]
    if len(matching) > 1:
        raise SnapshotError(f"several registries declare release {label}: {matching}")
    if "parity/promotion.toml" in candidates:
        return "parity/promotion.toml"
    if len(candidates) == 1:
        return candidates[0]
    raise SnapshotError(f"cannot choose a registry among {candidates}")


def find_matrix(files: set[str]) -> str | None:
    found = sorted(p for p in files if _MATRIX_RE.match(p))
    if len(found) > 1:
        raise SnapshotError(f"several producer/consumer matrix documents: {found}")
    return found[0] if found else None


def role_paths(repo: Path, commit: str, files: set[str], tag: str) -> dict[str, str | None]:
    paths: dict[str, str | None] = {}
    for role, fixed in ROLES:
        if role == "promotion_registry":
            paths[role] = find_registry(repo, commit, files, tag)
        elif role == "producer_consumer_matrix":
            paths[role] = find_matrix(files)
        else:
            assert fixed is not None
            paths[role] = fixed if fixed in files else None
    return paths


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


def _row_count(doc: dict[str, Any]) -> int:
    return sum(len(value) for value in doc.values() if isinstance(value, list))


def _status_counts(prefix: str, rows: list[dict[str, Any]], counts: dict[str, int]) -> None:
    for row in rows:
        key = f"{prefix}_{row.get('status', 'unspecified')}"
        counts[key] = counts.get(key, 0) + 1


def extract_closed_routes(registry: dict[str, Any], stages: dict[str, Any]) -> list[dict[str, Any]]:
    """Every route closed at the tag, from the stage rows and the registry."""
    routes: dict[str, dict[str, Any]] = {}

    def entry(name: str) -> dict[str, Any]:
        return routes.setdefault(
            name,
            {
                "route": name,
                "stage": "",
                "reason_code": "",
                "record": "",
                "sources": set(),
                "carryover": False,
            },
        )

    def merge(target: dict[str, Any], key: str, value: str, name: str) -> None:
        if not value:
            return
        if target[key] and target[key] != value:
            raise SnapshotError(f"route {name} has conflicting {key}: {target[key]!r} vs {value!r}")
        target[key] = value

    for row in stages.get("routes", []):
        if row.get("status") != "closed":
            continue
        name = str(row["route"])
        item = entry(name)
        item["sources"].add("stage_registry")
        merge(item, "stage", str(row.get("stage", "")), name)
        merge(item, "reason_code", str(row.get("reason_code", "")), name)

    for record in registry.get("record", []):
        for row in record.get("routes", []):
            if row.get("status") != "closed":
                continue
            name = str(row["name"])
            item = entry(name)
            item["sources"].add("promotion_registry")
            merge(item, "stage", str(row.get("stage", "")), name)
            merge(item, "reason_code", str(row.get("reason_code", "")), name)
            merge(item, "record", str(record["id"]), name)
            if record.get("status") == "carried_forward":
                item["carryover"] = True

    out = []
    for name in sorted(routes):
        item = routes[name]
        item["sources"] = sorted(item["sources"])
        out.append(item)
    return out


def file_table(
    repo: Path,
    tag_commit: str,
    base_commit: str,
    tag_paths: dict[str, str | None],
    base_paths: dict[str, str | None],
) -> list[dict[str, str]]:
    table = []
    for role, _fixed in ROLES:
        row = {"role": role}
        for prefix, commit, paths in (
            ("tag", tag_commit, tag_paths),
            ("base", base_commit, base_paths),
        ):
            path = paths[role]
            row[f"{prefix}_path"] = path or ""
            row[f"{prefix}_sha256"] = (
                hashlib.sha256(blob(repo, commit, path)).hexdigest() if path else ABSENT
            )
        table.append(row)
    return table


def collect_counts(
    repo: Path,
    tag_commit: str,
    tag_paths: dict[str, str | None],
    registry: dict[str, Any],
    stages: dict[str, Any],
    routes: list[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {"inherited_closed_routes": len(routes)}
    records = registry.get("record", [])
    counts["registry_records"] = len(records)
    _status_counts("registry_records", records, counts)
    registry_routes = [r for rec in records for r in rec.get("routes", [])]
    counts["registry_routes"] = len(registry_routes)
    _status_counts("registry_routes", registry_routes, counts)
    stage_rows = stages.get("routes", [])
    counts["stage_routes"] = len(stage_rows)
    _status_counts("stage_routes", stage_rows, counts)
    counts["carryover_closed_routes"] = sum(1 for r in routes if r["carryover"])
    for role, key in (
        ("support_licensed", "support_licensed_rows"),
        ("support_closed", "support_closed_rules"),
        ("promotion_carryovers", "carryover_cells"),
        ("coverage_records", "coverage_record_rows"),
        ("producer_consumer_registry", "producer_consumer_rows"),
    ):
        path = tag_paths[role]
        if path:
            counts[key] = _row_count(_parse(blob(repo, tag_commit, path), path))
    return dict(sorted(counts.items()))


# --------------------------------------------------------------------------
# document model
# --------------------------------------------------------------------------

_ROUTE_KEYS = ("disposition", "touched_by", "rationale", "new_record")


def build_document(
    repo: Path,
    tag: str,
    base_commit: str,
    previous: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    """Return the snapshot document and the names of dropped classifications."""
    tag_commit = resolve_commit(repo, tag)
    base_commit = resolve_commit(repo, base_commit)
    tag_files = tree_files(repo, tag_commit)
    base_files = tree_files(repo, base_commit)
    tag_paths = role_paths(repo, tag_commit, tag_files, tag)
    base_paths = role_paths(repo, base_commit, base_files, tag)
    for required in ("promotion_registry", "transport_stages"):
        if not tag_paths[required]:
            raise SnapshotError(f"{tag} carries no {required} file")
    registry_path = tag_paths["promotion_registry"]
    stages_path = tag_paths["transport_stages"]
    assert registry_path and stages_path
    registry = _parse(blob(repo, tag_commit, registry_path), registry_path)
    stages = _parse(blob(repo, tag_commit, stages_path), stages_path)
    routes = extract_closed_routes(registry, stages)

    prior_routes: dict[str, dict[str, Any]] = {}
    prior_decisions: list[dict[str, Any]] = []
    if previous:
        prior_routes = {str(r["route"]): r for r in previous.get("route", [])}
        prior_decisions = list(previous.get("decision", []))
    for item in routes:
        old = prior_routes.pop(item["route"], {})
        item["disposition"] = str(old.get("disposition", "unclassified"))
        item["touched_by"] = [str(v) for v in old.get("touched_by", [])]
        item["rationale"] = str(old.get("rationale", ""))
        item["new_record"] = str(old.get("new_record", ""))

    doc = {
        "version": SCHEMA_VERSION,
        "snapshot": {
            "tag": tag,
            "tag_commit": tag_commit,
            "base_commit": base_commit,
            "registry_path": registry_path,
            "base_registry_path": base_paths["promotion_registry"] or "",
        },
        "counts": collect_counts(repo, tag_commit, tag_paths, registry, stages, routes),
        "file": file_table(repo, tag_commit, base_commit, tag_paths, base_paths),
        "route": routes,
        "decision": prior_decisions,
    }
    return doc, sorted(prior_routes)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def _q(value: str) -> str:
    out = ['"']
    for ch in value:
        code = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif code < 0x20 or code == 0x7F:
            out.append(f"\\u{code:04x}")
        elif code > 0x7E:
            out.append(f"\\u{code:04x}" if code <= 0xFFFF else f"\\U{code:08x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _ql(values: list[str]) -> str:
    return "[" + ", ".join(_q(v) for v in values) + "]"


def render_toml(doc: dict[str, Any]) -> str:
    snap = doc["snapshot"]
    lines = [
        "# Inherited promotion baseline. Generated by scripts/snapshot_promotion_baseline.py;",
        "# `--write` regenerates everything except the hand-maintained route",
        "# classification fields (disposition, touched_by, rationale, new_record) and the",
        "# [[decision]] entries, which it preserves. `--check` verifies the rest against git.",
        f"version = {doc['version']}",
        "",
        "[snapshot]",
    ]
    for key in ("tag", "tag_commit", "base_commit", "registry_path", "base_registry_path"):
        lines.append(f"{key} = {_q(snap[key])}")
    lines += ["", "[counts]"]
    for key, value in doc["counts"].items():
        lines.append(f"{key} = {value}")
    for row in doc["file"]:
        lines += ["", "[[file]]"]
        for key in ("role", "tag_path", "tag_sha256", "base_path", "base_sha256"):
            lines.append(f"{key} = {_q(row[key])}")
    for item in doc["route"]:
        lines += ["", "[[route]]"]
        lines.append(f"route = {_q(item['route'])}")
        lines.append(f"stage = {_q(item['stage'])}")
        lines.append(f"reason_code = {_q(item['reason_code'])}")
        lines.append(f"record = {_q(item['record'])}")
        lines.append(f"sources = {_ql(item['sources'])}")
        lines.append(f"carryover = {'true' if item['carryover'] else 'false'}")
        lines.append(f"disposition = {_q(item['disposition'])}")
        lines.append(f"touched_by = {_ql(item['touched_by'])}")
        lines.append(f"rationale = {_q(item['rationale'])}")
        lines.append(f"new_record = {_q(item['new_record'])}")
    for decision in doc["decision"]:
        lines += ["", "[[decision]]"]
        for key in ("area", "decision", "reference", "anchor"):
            lines.append(f"{key} = {_q(str(decision.get(key, '')))}")
    return "\n".join(lines) + "\n"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _code(text: str) -> str:
    return f"`{text}`" if text else "-"


def render_markdown(doc: dict[str, Any]) -> str:
    snap = doc["snapshot"]
    counts = doc["counts"]
    routes = doc["route"]
    by_disposition: dict[str, int] = {name: 0 for name in DISPOSITIONS}
    for item in routes:
        by_disposition[item["disposition"]] = by_disposition.get(item["disposition"], 0) + 1

    out = [
        "# Inherited promotion baseline",
        "",
        "This page is generated by `scripts/snapshot_promotion_baseline.py` from git objects, not the working tree, "
        "and is verified by `--check`. It freezes what the tagged prior release licensed and kept closed, apart from "
        "the refactored base, so new work can say which inherited closed routes it depends on, keeps refusing, or "
        "reopens under a new record. Machine-readable twin: `parity/inherited_baseline.toml`.",
        "",
        "## Source",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Tag | `{snap['tag']}` |",
        f"| Tag commit | `{snap['tag_commit']}` |",
        f"| Refactored base commit | `{snap['base_commit']}` |",
        f"| Registry at the tag | `{snap['registry_path']}` |",
        f"| Registry at the base | {_code(snap['base_registry_path'])} |",
        "",
        "## Counts at the tag",
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
    ]
    for key, value in counts.items():
        out.append(f"| `{key}` | {value} |")
    out += [
        "",
        "## File digests",
        "",
        "SHA-256 of each role's file at the tag and at the base. A differing digest means the "
        "refactored base changed that file since the tag; `absent` means the commit carries no such file.",
        "",
        "| Role | Path at tag | SHA-256 at tag | Path at base | SHA-256 at base | Changed |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in doc["file"]:
        changed = "yes" if row["tag_sha256"] != row["base_sha256"] else "no"
        out.append(
            f"| `{row['role']}` | {_code(row['tag_path'])} | `{row['tag_sha256']}` "
            f"| {_code(row['base_path'])} | `{row['base_sha256']}` | {changed} |"
        )
    out += [
        "",
        "## Closed inherited routes",
        "",
        "Every route that was closed at the tag, from its stage rows and its registry. The reason code is the "
        "one the tag emitted and is kept until a new record licenses the route. Dispositions:",
        "",
        "- `prerequisite`: new work must consume this route's licensed form first; it stays closed until then.",
        "- `deliberate_refusal`: new work touches the route's neighbourhood and keeps refusing it, with its reason "
        "code unchanged.",
        "- `reopened_under_new_record`: a new promotion record owns the route's replacement; the inherited route "
        "stays closed until that record is promoted.",
        "- `unclassified`: no work in the current composition scope touches the route; it remains closed unchanged.",
        "",
        "| Disposition | Routes |",
        "| --- | ---: |",
    ]
    for name in DISPOSITIONS:
        out.append(f"| `{name}` | {by_disposition.get(name, 0)} |")

    classified = [r for r in routes if r["disposition"] != "unclassified"]
    out += ["", "### Classified routes", ""]
    if classified:
        out += [
            "| Route | Stage | Reason code | Record | Disposition | Touched by | New record | Rationale |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in classified:
            out.append(
                f"| `{r['route']}` | {r['stage']} | `{r['reason_code']}` | {_code(r['record'])} "
                f"| `{r['disposition']}` | {_cell(', '.join(r['touched_by']))} | {_code(r['new_record'])} "
                f"| {_cell(r['rationale'])} |"
            )
    else:
        out.append("No route is classified yet.")

    unclassified = [r for r in routes if r["disposition"] == "unclassified"]
    out += ["", "### Unclassified routes", ""]
    if unclassified:
        out += [
            "| Route | Stage | Reason code | Record |",
            "| --- | --- | --- | --- |",
        ]
        for r in unclassified:
            out.append(f"| `{r['route']}` | {r['stage']} | `{r['reason_code']}` | {_code(r['record'])} |")
    else:
        out.append("Every inherited closed route is classified.")

    out += ["", "## Identity and API migration decisions", ""]
    if doc["decision"]:
        out += [
            "Decisions the refactored base made to identities and public surfaces since the tag. "
            "`--check` verifies that each pointer's file exists and still contains its anchor text.",
            "",
            "| Area | Decision | Pointer |",
            "| --- | --- | --- |",
        ]
        for d in doc["decision"]:
            out.append(f"| {_cell(d['area'])} | {_cell(d['decision'])} | `{d['reference']}` |")
    else:
        out.append("No decisions are recorded.")

    out += [
        "",
        "## Reproduction",
        "",
        "```text",
        "python3 scripts/snapshot_promotion_baseline.py --check",
        "python3 scripts/snapshot_promotion_baseline.py --write",
        "python3 scripts/snapshot_promotion_baseline.py --write --tag <vX.Y.Z> --base <commit>",
        "```",
        "",
        "`--write` keeps existing route classifications and decisions and takes the base commit from the "
        "checked-in file unless `--base` is given (default: the current `HEAD`).",
        "",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def _working_record_ids(repo: Path) -> set[str]:
    path = repo / "parity" / "promotion.toml"
    if not path.is_file():
        return set()
    doc = tomllib.loads(read_text(path))
    return {str(rec.get("id", "")) for rec in doc.get("record", [])}


def validate(doc: dict[str, Any], repo: Path, strict: bool) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    known_records: set[str] | None = None
    for item in doc["route"]:
        name = item["route"]
        if name in seen:
            errors.append(f"duplicate route {name}")
        seen.add(name)
        disposition = item["disposition"]
        if disposition not in DISPOSITIONS:
            errors.append(f"{name}: unknown disposition {disposition!r}")
            continue
        if disposition == "unclassified":
            if item["rationale"] or item["new_record"] or item["touched_by"]:
                errors.append(f"{name}: an unclassified route carries classification fields")
            continue
        if not item["rationale"].strip():
            errors.append(f"{name}: {disposition} requires a rationale")
        if not item["touched_by"]:
            errors.append(f"{name}: {disposition} requires touched_by naming the work that touches it")
        if disposition == "reopened_under_new_record":
            if not item["new_record"]:
                errors.append(f"{name}: reopened_under_new_record requires new_record")
            else:
                if known_records is None:
                    known_records = _working_record_ids(repo)
                if item["new_record"] not in known_records:
                    message = f"{name}: new_record {item['new_record']!r} is not in parity/promotion.toml yet"
                    (errors if strict else warnings).append(message)
        elif item["new_record"]:
            errors.append(f"{name}: new_record is only valid for reopened_under_new_record")
    for index, decision in enumerate(doc.get("decision", [])):
        label = f"decision[{index}]"
        for key in ("area", "decision", "reference", "anchor"):
            if not str(decision.get(key, "")).strip():
                errors.append(f"{label}: missing {key}")
        reference = str(decision.get("reference", ""))
        anchor = str(decision.get("anchor", ""))
        if not reference or not anchor:
            continue
        target = repo / reference
        if not target.is_file():
            errors.append(f"{label}: pointer {reference} does not exist")
        elif anchor not in read_text(target):
            errors.append(f"{label}: anchor text not found in {reference}")
    return errors, warnings


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------


def _load_previous(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return tomllib.loads(read_text(path))


def run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT, help="repository root (default: this checkout)")
    parser.add_argument("--tag", help="release tag to snapshot (default: the recorded tag, else the latest vX.Y.Z)")
    parser.add_argument("--base", help="refactored base commit (default: the recorded base, else HEAD)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="regenerate the snapshot files")
    mode.add_argument("--check", action="store_true", help="verify the checked-in files (default)")
    parser.add_argument("--strict", action="store_true", help="--check: unallocated new_record is an error")
    parser.add_argument("--out-toml", default=SNAPSHOT_TOML)
    parser.add_argument("--out-md", default=SNAPSHOT_MD)
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    toml_path = repo / args.out_toml
    md_path = repo / args.out_md
    previous = _load_previous(toml_path)
    recorded = (previous or {}).get("snapshot", {})

    try:
        tag = args.tag or recorded.get("tag") or latest_tag(repo)
        base = args.base or recorded.get("base_commit") or "HEAD"
        if not args.write and previous is None:
            print(f"snapshot_promotion_baseline: {args.out_toml} is missing; run --write", file=sys.stderr)
            return 1
        doc, dropped = build_document(repo, tag, base, previous)
        tag_commit = doc["snapshot"]["tag_commit"]
        base_commit = doc["snapshot"]["base_commit"]
        proc = subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", tag_commit, base_commit],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise SnapshotError(f"base {base_commit[:12]} does not descend from tag {tag}")
    except SnapshotError as err:
        print(f"snapshot_promotion_baseline: {err}", file=sys.stderr)
        return 1

    toml_text = render_toml(doc)
    md_text = render_markdown(doc)
    errors, warnings = validate(doc, repo, strict=args.strict)

    if args.write:
        if errors:
            for message in errors:
                print(f"error: {message}", file=sys.stderr)
            return 1
        write_text(toml_path, toml_text)
        write_text(md_path, md_text)
        for name in dropped:
            print(f"warning: dropped classification of route absent at {tag}: {name}", file=sys.stderr)
        for message in warnings:
            print(f"warning: {message}", file=sys.stderr)
        print(f"wrote {args.out_toml} and {args.out_md} ({len(doc['route'])} routes, tag {tag})")
        return 0

    stale = []
    if not toml_path.is_file() or read_text(toml_path) != toml_text:
        stale.append(args.out_toml)
    if not md_path.is_file() or read_text(md_path) != md_text:
        stale.append(args.out_md)
    for name in stale:
        errors.append(f"{name} does not match git for {tag} at base {base_commit[:12]}; run --write")
    for name in dropped:
        errors.append(f"classified route absent at {tag}: {name}")
    for message in warnings:
        print(f"warning: {message}", file=sys.stderr)
    if errors:
        for message in errors:
            print(f"error: {message}", file=sys.stderr)
        return 1
    print(f"snapshot_promotion_baseline: ok ({len(doc['route'])} inherited closed routes, tag {tag})")
    return 0


def main() -> None:
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
