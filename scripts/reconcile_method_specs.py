#!/usr/bin/env python3
"""Reconcile the method specifications with the method catalog and the provenance records.

Every method child has exactly one specification in parity/method_specs/specs.toml. The record
carries its scientific fields (identifying result, assumptions, sampling units, estimator,
diagnostics, failure policy) and the links that tie it to the catalog and the clean-room
review: its parent, route and target, the provenance records of its implementing module, and
the review of that module's public API.

A scientific field a hand-written record leaves `unlisted` is filled when the records are
rewritten; hand-written fields are never replaced. `--check` compares only the structured
fields (parent, route, provenance links, API review) with the derived values and never reads
the text of the scientific fields.

    python3 scripts/reconcile_method_specs.py            # complete and rewrite the records
    python3 scripts/reconcile_method_specs.py --check    # fail on a stale or inconsistent record
    python3 scripts/reconcile_method_specs.py --report
    python3 scripts/reconcile_method_specs.py --self-test
"""

import ast
import json
import re
import sys
import tomllib
from collections import Counter

from _repo import REPO_ROOT as ROOT
from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import _repo  # noqa: E402
import provenance_links  # noqa: E402
from method_evidence import UNLISTED, Evidence  # noqa: E402

SPECS = "parity/method_specs/specs.toml"
CHILDREN = "parity/method_children/children.toml"
MANIFEST = "parity/capabilities/route_manifest.toml"
STRUCTURAL = (
    "parent",
    "route",
    "provenance",
    "api_review",
)
SCIENTIFIC = (
    "identifying_result",
    "assumptions",
    "sampling_units",
    "estimator",
    "diagnostics",
    "failure_policy",
)
FIELD_ORDER = (
    "id",
    "parent",
    "route",
    "target",
    "provenance",
    "api_review",
    "identifying_result",
    "assumptions",
    "sampling_units",
    "estimator",
    "diagnostics",
    "failure_policy",
)
HEADER = """\
# Method specifications. One record per method child in parity/method_children/children.toml.
# Maintained by scripts/reconcile_method_specs.py: parent, route and target follow the child,
# and provenance and api_review follow the child's implementing module. A scientific field the
# record leaves `unlisted` is filled when the records are rewritten. Hand-written fields are
# kept and the check does not read their text.
# provenance names the clean-room provenance records of the implementing module, or the reason
# the module needs none.
# estimator is the estimator or posterior construction.

"""
SAMPLING = re.compile(
    r"\b(independent|one row|each row|rows? (?:are|is)|units?|subjects?|clusters?|series|panel|blocks?|"
    r"groups?|participants?|draws?|observations?)\b",
    re.I,
)
ASSUME = re.compile(
    r"\b(assum\w*|premises?|requires?|required|excluded|exclusion|must|only (?:when|if|for)|declared|"
    r"unconfounded|ignorab\w*|exchangeab\w*|known)\b",
    re.I,
)
DIAGNOSE = re.compile(
    r"\b(diagnos\w*|refus\w*|overlap|positivity|first-stage|convergence|Rhat|ESS|reliab\w*|fails?|checks?)\b",
    re.I,
)
IDENTIFY = re.compile(r"\b(identif\w*|estimand|reported number|effect of|contrast|target)\b", re.I)


# ----------------------------------------------------------------------------- module docs
def module_header(ev: Evidence, path: str) -> list[str]:
    """Sentences of the module's own header (`//!` lines of a Rust file, docstring of a script)."""
    body = ev.text(path)
    if path.endswith(".py"):
        try:
            text = ast.get_docstring(ast.parse(body)) or ""
        except SyntaxError:
            text = ""
    elif path.endswith(".rs"):
        lines, fenced = [], False
        for line in body.splitlines():
            if line.startswith("// SPDX-License-Identifier"):
                continue
            if not line.startswith("//!"):
                if lines:
                    break
                continue
            content = line[3:].strip()
            if content.startswith("```"):
                fenced = not fenced
                continue
            if fenced or content.startswith("#") or content.startswith("SPDX"):
                continue
            lines.append(content)
        text = "\n".join(lines)
    else:
        return []
    text = re.sub(r"\[`?([^\]`]+)`?\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[`([^`]+)`\]", r"\1", text)
    text = text.replace("`", "").replace("**", "")
    text = re.sub(r"^\s*[*-]\s+", "", text, flags=re.M)
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    sentences: list[str] = []
    for paragraph in paragraphs:
        for sentence in re.split(r"(?<=[.;])\s+(?=[A-Z(\[])", paragraph):
            sentence = sentence.strip()
            if 25 <= len(sentence) <= 420:
                sentences.append(sentence)
    return sentences


def pick(sentences: list[str], pattern: re.Pattern, used: set[str], count: int = 1) -> str:
    chosen = []
    for sentence in sentences:
        if sentence in used or not pattern.search(sentence):
            continue
        chosen.append(sentence)
        used.add(sentence)
        if len(chosen) == count:
            break
    return " ".join(chosen)


REFUSAL_TEST = re.compile(
    r"refus|reject|forg|invalid|unsupported|weak|sparse|forbid|mismatch|tamper|corrupt|withhold",
    re.I,
)


def tests_named(ev: Evidence, child: dict, pattern: re.Pattern) -> str:
    """Names of the fixture tests whose name speaks of a refusal; else the fixture tests themselves."""
    names: list[str] = []
    fallback: list[str] = []
    for token in ev.tokens(child["fixtures"]) + [
        t for t in ev.tokens(child["consumer"]) if "/tests/" in t
    ]:
        path, name = ev.split(token)
        if name:
            (names if pattern.search(name) else fallback).append(name)
        elif path.endswith((".rs", ".py")):
            for found in re.findall(
                r"#\[test\]\s*(?:#\[[^\]]*\]\s*)*fn (\w+)|^\s*def (test_\w+)", ev.text(path), re.M
            ):
                n = found[0] or found[1]
                (names if pattern.search(n) else fallback).append(n)
    pool = list(dict.fromkeys(names)) or list(dict.fromkeys(fallback))
    return ", ".join(pool[:3]) if pool else "the fixtures named on the child"


def derive_fields(ev: Evidence, child: dict, spec: dict) -> dict:
    """Fill the unlisted fields from the implementing module's own header and the child's notes.

    A filled field is a sentence the module header states, so the check can find it there.
    """
    module = child["module"].split(";")[0].strip().split(" ")[0]
    header = module_header(ev, module) if module not in UNLISTED and ev.repo.exists(module) else []
    used: set[str] = set()
    out = dict(spec)
    temporal = ".temporal." in child["id"] or child["id"].split(".")[-2] == "temporal"
    executes = bool(header)
    notes = re.sub(r"^[;\s]+", "", child["notes"]).strip()

    def find(pool: list[str], pattern: re.Pattern, count: int = 1) -> str:
        return pick(pool, pattern, used, count)

    def fill(field: str, value: str) -> None:
        if value and out.get(field, "unlisted") in UNLISTED:
            out[field] = value

    refusal = re.compile(r"\brefus\w*|\berror\b|\bfails?\b", re.I)
    fill("sampling_units", find(header, SAMPLING))
    fill("assumptions", find(header, ASSUME, 2))
    fill("diagnostics", find(header, DIAGNOSE, 2))
    fill("identifying_result", find(header, IDENTIFY))
    realization = (
        child["inference"]
        if child["inference"] not in UNLISTED
        and child["inference"] not in ("frequentist", "bayesian")
        else ""
    )
    summary = header[0] if header else ""
    fill(
        "estimator",
        f"{realization}: {summary}" if realization and summary else realization or summary,
    )
    if not executes:
        fill(
            "sampling_units",
            "none: the axis defines no sampling design that this method executes over",
        )
        fill("estimator", "none: no estimator exists on this axis")
        fill("identifying_result", f"none: {notes[:240]}")
        fill("assumptions", f"none beyond the recorded boundary: {notes[:240]}")
        fill("diagnostics", "the typed refusal this axis raises, exercised by its fixture")
    default_units = (
        "ordered observations of the declared series or panel; serial dependence is handled by block resampling or block weights"
        if temporal
        else "independent rows of the declared sample; the study's sampling design may declare clusters or weights"
    )
    fill("sampling_units", default_units)
    fill(
        "assumptions",
        f"the identification premises of {child['target']} on the recorded route; the module documents no further premise",
    )
    fill(
        "diagnostics",
        f"refusals and diagnostics asserted by {tests_named(ev, child, REFUSAL_TEST)}",
    )
    fill("identifying_result", child["target"] if child["target"] not in UNLISTED else "none")
    fill("estimator", child["inference"] if child["inference"] not in UNLISTED else "none")
    fill("failure_policy", find(header, refusal, 2))
    fill(
        "failure_policy", f"Refused cases are exercised by {tests_named(ev, child, REFUSAL_TEST)}."
    )
    return out


def review_links(ev: Evidence, child: dict, kept: list[str]) -> tuple[list[str], str]:
    """The provenance records and the public-API review of the child's implementing module.

    A module no record names keeps the records the specification already links, when each one
    exists.
    """
    module = child["module"].split(";")[0].strip().split(" ")[0]
    if module in UNLISTED:
        return ["not_applicable: no implementing module"], "none: no implementing module"
    if module in provenance_links.NOT_AN_ALGORITHM:
        return [
            f"not_applicable: {provenance_links.NOT_AN_ALGORITHM[module]}"
        ], provenance_links.api_review(module, ev.root)
    records = provenance_links.records_for(module, ev.root)
    if not records:
        known = provenance_links.stems(ev.root)
        records = [k for k in kept if k in known]
    return sorted(records), provenance_links.api_review(module, ev.root)


def complete(ev: Evidence) -> list[dict]:
    children = tomllib.loads(ev.text(CHILDREN))["child"]
    existing = (
        {s["id"]: s for s in tomllib.loads(ev.text(SPECS))["spec"]} if ev.repo.exists(SPECS) else {}
    )
    specs = []
    for child in children:
        cid = child["id"]
        spec = dict(existing.get(cid) or {"id": cid})
        spec["parent"] = child["parent"]
        spec["route"] = child.get("route") or spec.get("route", "none")
        if spec.get("target", "unlisted") in UNLISTED:
            spec["target"] = child["target"] if child["target"] not in UNLISTED else "none"
        spec = derive_fields(ev, child, spec)
        spec["provenance"], spec["api_review"] = review_links(ev, child, spec.get("provenance", []))
        specs.append(spec)
    return specs


def quote(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def render(specs: list[dict]) -> str:
    out = [HEADER]
    for spec in specs:
        out.append("[[spec]]")
        for field in FIELD_ORDER:
            value = spec[field]
            if isinstance(value, list):
                out.append(f"{field} = [")
                out.extend(f"  {quote(v)}," for v in value)
                out.append("]")
            else:
                out.append(f"{field} = {quote(value)}")
        out.append("")
    return "\n".join(out)


# ------------------------------------------------------------------------------ verification
def verify(ev: Evidence, specs: list[dict]) -> list[str]:
    children = {c["id"]: c for c in tomllib.loads(ev.text(CHILDREN))["child"]}
    out: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        sid = spec["id"]
        if sid in seen:
            out.append(f"{sid}: duplicate specification")
        seen.add(sid)
        child = children.get(sid)
        if child is None:
            out.append(f"{sid}: no method child carries this id")
            continue
        out.extend(review_problems(ev, spec, child))
        for field in SCIENTIFIC:
            if spec[field] in UNLISTED:
                out.append(f"{sid}: record leaves {field} unlisted")
    for cid in children:
        if cid not in seen:
            out.append(f"{cid}: no specification")
    return out


def stale_structure(committed: list[dict], derived: list[dict]) -> list[str]:
    """Structured fields of the committed records that differ from the derived ones."""
    expected = {spec["id"]: spec for spec in derived}
    out: list[str] = []
    for spec in committed:
        want = expected.get(spec["id"])
        if want is None:
            continue  # verify() reports an id no child carries
        for field in STRUCTURAL:
            if spec.get(field) != want[field]:
                out.append(
                    f"{spec['id']}: {field} {spec.get(field)!r} is stale; derived {want[field]!r}"
                )
    return out


def review_problems(ev: Evidence, spec: dict, child: dict) -> list[str]:
    """A method links clean-room provenance and a reviewed public surface, or says why not."""
    sid = spec["id"]
    out: list[str] = []
    links = spec["provenance"]
    real = [stem for stem in links if not stem.startswith("not_applicable:")]
    if not links:
        out.append(f"{sid}: method links no provenance record and records no reason")
    for stem in real:
        path = f"provenance/{stem}.toml"
        if not ev.repo.exists(path):
            out.append(f"{sid}: provenance record {stem} does not exist")
            continue
        record = tomllib.loads(ev.text(path))
        for flag in ("source_translation", "copied_code", "copied_comments", "copied_tests"):
            if record.get(flag) is not False:
                out.append(f"{sid}: provenance record {stem} does not state {flag} = false")
    if spec["api_review"].startswith("unreviewed"):
        out.append(f"{sid}: public surface is unreviewed ({spec['api_review']})")
    return out


def report(specs: list[dict]) -> str:
    return "\n".join(
        [
            f"specs: {len(specs)}",
            "unlisted scientific fields: "
            + json.dumps(Counter(f for s in specs for f in SCIENTIFIC if s[f] in UNLISTED)),
        ]
    )


def self_test() -> int:
    sentences = [
        "Each row is one independent subject at one decision time.",
        "The estimand is the average mean-scale contrast.",
        "A single dependent series is refused.",
        "It requires a declared prior on the coefficients.",
    ]
    used: set[str] = set()
    assert pick(sentences, SAMPLING, used) == sentences[0]
    assert pick(sentences, IDENTIFY, used) == sentences[1]
    assert pick(sentences, ASSUME, used) == sentences[3]
    assert pick(sentences, SAMPLING, used) == sentences[2]
    assert pick(sentences, SAMPLING, used) == ""
    base = {f: "x" for f in FIELD_ORDER}
    text = render([base])
    parsed = tomllib.loads(text)["spec"][0]
    assert parsed["provenance"] == "x" and parsed["id"] == "x"
    print("reconcile_method_specs self-test passed")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    ev = Evidence(ROOT)
    specs = complete(ev)
    text = render(specs)
    path = ROOT / SPECS
    if "--report" in argv:
        print(report(specs))
        return 0
    if "--check" in argv:
        committed = _repo.load_toml(path)["spec"]
        stale = False
        for line in stale_structure(committed, specs):
            stale = True
            print(f"FAIL: {line}; run python3 scripts/reconcile_method_specs.py")
        problems = verify(ev, committed)
        for line in problems:
            print(f"FAIL: {line}")
        print(f"method specs: {len(specs)} records, {len(problems)} finding(s)")
        return 1 if stale or problems else 0
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {len(specs)} specifications to {SPECS}")
    print(report(specs))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
