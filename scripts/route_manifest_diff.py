#!/usr/bin/env python3
"""Diff the support matrix against the published predecessor tag, coordinate by coordinate.

Every coordinate of the cartesian product of the support axes (query x graph class x structure x
inference x validation) is classified at the predecessor tag and in the working tree as
`licensed`, `n_a`, `executes_unlicensed` or `refused`, using the same precedence as the support
matrix: a licensed cell wins, then an n/a rule, then a closed rule (an `executes_unlicensed` rule
still builds and runs, its result reporting the support status `refused`).

* A lost capability is a coordinate that was licensed or executed at the predecessor and is
  neither licensed nor executing now. The release condition is zero of them.
* A graphless licence (family, design, method) at the predecessor that is absent now is a lost
  capability too.
* A changed refusal is a coordinate refused or n/a at the predecessor whose refusal class changed
  (refused to n/a or the reverse) while it still does not run; the reasons are listed so a
  reviewer reads each class change.
* A gained capability is a coordinate that did not run at the predecessor and does now.

`--require-zero` exits non-zero on any lost capability. The report is read-only and runs no test.
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import tomllib
from collections import Counter

from _repo import REPO_ROOT

TAG = "v2.4.1"
LICENSED, N_A, EXECUTES, REFUSED = "licensed", "n_a", "executes_unlicensed", "refused"
RUNS = (LICENSED, EXECUTES)
AXES = ("query", "graph_class", "structure", "inference", "validation")
RULE_AXES = {
    "queries": "query",
    "graph_classes": "graph_class",
    "structures": "structure",
    "inferences": "inference",
    "validations": "validation",
}
FILES = {
    "axes": "parity/support_axes.toml",
    "licensed": "parity/support_licensed.toml",
    "closed": "parity/support_closed.toml",
    "n_a": "parity/support_n_a.toml",
    "graphless": "parity/support_graphless.toml",
}


def _tag_text(path: str, tag: str) -> str:
    return subprocess.run(
        ["git", "show", f"{tag}:{path}"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout


def load_registries(tag: str | None = None) -> dict:
    """The four support registries from the working tree, or from a tag."""
    out = {}
    for name, path in FILES.items():
        text = _tag_text(path, tag) if tag else (REPO_ROOT / path).read_text(encoding="utf-8")
        out[name] = tomllib.loads(text)
    return out


def coordinates(axes: dict) -> list[tuple]:
    values = {
        "query": axes["queries"],
        "graph_class": axes["graph_classes"],
        "structure": axes["structures"],
        "inference": axes["inferences"],
        "validation": axes["validations"],
    }
    return list(itertools.product(*(values[a] for a in AXES)))


def _matches(rule: dict, coord: tuple) -> bool:
    by_axis = dict(zip(AXES, coord))
    return all(by_axis[axis] in rule[key] for key, axis in RULE_AXES.items() if key in rule)


def classify(registries: dict, coord: tuple) -> tuple[str, str]:
    """(class, reason) of one coordinate; the reason is empty for a licensed coordinate."""
    if coord in registries["_licensed_keys"]:
        return LICENSED, ""
    for rule in registries["n_a"].get("n_a", []):
        if _matches(rule, coord):
            return N_A, rule.get("reason", "")
    for rule in registries["closed"].get("closed", []):
        if _matches(rule, coord):
            kind = EXECUTES if rule.get("executes_unlicensed") else REFUSED
            return kind, rule.get("reason", "")
    return REFUSED, "refused by default"


def _prepare(registries: dict) -> dict:
    registries["_licensed_keys"] = {
        tuple(cell[a] for a in AXES) for cell in registries["licensed"].get("cell", [])
    }
    return registries


def _graphless_keys(registries: dict) -> set[tuple]:
    return {
        (row.get("family"), row.get("design"), row.get("method"))
        for row in registries.get("graphless", {}).get("license", [])
    }


def diff(old: dict, new: dict) -> dict:
    old, new = _prepare(old), _prepare(new)
    lost, changed, gained, counts = [], [], [], Counter()
    for key in sorted(_graphless_keys(old) - _graphless_keys(new), key=str):
        lost.append({"coordinate": key, "before": "graphless_license", "after": "absent", "reason": ""})
    for coord in coordinates(new["axes"]):
        before, reason_before = classify(old, coord)
        after, reason_after = classify(new, coord)
        counts[(before, after)] += 1
        if before in RUNS and after not in RUNS:
            lost.append({"coordinate": coord, "before": before, "after": after, "reason": reason_after})
        elif before not in RUNS and after in RUNS:
            gained.append({"coordinate": coord, "before": before, "after": after})
        elif before != after:
            changed.append(
                {
                    "coordinate": coord, "before": before, "after": after,
                    "reason_before": reason_before, "reason_after": reason_after,
                }
            )
    return {
        "tag": TAG,
        "coordinates": sum(counts.values()),
        "transitions": {f"{a}->{b}": n for (a, b), n in sorted(counts.items())},
        "lost": lost,
        "changed_refusals": changed,
        "gained": gained,
    }


def build(tag: str = TAG) -> dict:
    return diff(load_registries(tag), load_registries())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--require-zero", action="store_true")
    args = parser.parse_args(argv)
    result = build(args.tag)
    print(f"coordinates {result['coordinates']} against {args.tag}")
    for transition, n in result["transitions"].items():
        print(f"  {transition}: {n}")
    for entry in result["lost"]:
        print(f"LOST {entry['coordinate']}: {entry['before']} -> {entry['after']}: {entry['reason'][:160]}")
    for entry in result["changed_refusals"]:
        print(f"CHANGED {entry['coordinate']}: {entry['before']} -> {entry['after']}")
    print(
        f"{len(result['lost'])} lost capabilities, {len(result['changed_refusals'])} changed "
        f"refusals, {len(result['gained'])} gained"
    )
    return 1 if args.require_zero and result["lost"] else 0


if __name__ == "__main__":
    sys.exit(main())
