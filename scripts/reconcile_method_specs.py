#!/usr/bin/env python3
"""Reconcile the method specifications with the executing routes and the counterpart inventory.

Every method child has exactly one specification in parity/method_specs/specs.toml. The record
carries its scientific fields (identifying result, assumptions, sampling units, estimator,
diagnostics, failure policy) and an explicit review decision:

  admission   licensed | executing_unlicensed | refused | not_admitted
  review_status  admitted | specification_only | blocked_research | not_admitted
  decision_basis the evidence behind the decision: the counterpart status, the executing
                 fixture, the artifact consumer, the uncertainty procedure and its basis

The decision is derived from the counterpart inventory (itself derived from the route
manifest's method records and the verified fixtures); an explicit `not_admitted` review is kept
and never overwritten. A scientific field a hand-written record leaves `unlisted` is filled
from the sentences the implementing module documents about itself (its `//!` header), so a
filled field is the module's own statement and the check can find it there. Hand-written
fields are never replaced.

    python3 scripts/reconcile_method_specs.py            # complete and rewrite the records
    python3 scripts/reconcile_method_specs.py --check    # fail on a stale or inconsistent record
    python3 scripts/reconcile_method_specs.py --report
    python3 scripts/reconcile_method_specs.py --self-test
"""

from __future__ import annotations

import ast
import json
import re
import sys
import tomllib
from collections import Counter

from _repo import REPO_ROOT as ROOT
from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import generate_counterparts as counterparts  # noqa: E402
import provenance_links  # noqa: E402
from method_evidence import UNLISTED, Evidence  # noqa: E402

SPECS = "parity/method_specs/specs.toml"
CHILDREN = "parity/method_children/children.toml"
MANIFEST = "parity/capabilities/route_manifest.toml"
REVIEW_STATUSES = ("admitted", "specification_only", "blocked_research", "not_admitted")
ADMISSIONS = ("licensed", "executing_unlicensed", "refused", "not_admitted")
SCIENTIFIC = ("identifying_result", "assumptions", "sampling_units", "estimator", "diagnostics", "failure_policy")
FIELD_ORDER = (
    "id", "parent", "gap", "route", "target", "review_status", "admission", "decision_basis",
    "provenance", "api_review", "identifying_result", "assumptions", "sampling_units", "estimator", "diagnostics",
    "failure_policy", "references",
)
HEADER = """\
# Method specifications. One record per method child in parity/method_children/children.toml.
# Maintained by scripts/reconcile_method_specs.py: the review decision (review_status,
# admission, decision_basis) is derived from the counterpart inventory and the executing
# evidence; a scientific field the record leaves `unlisted` is filled from the implementing
# module's own header. Hand-written fields are kept.
# estimator is the estimator or posterior construction.
# references are ROADMAP.md section titles.
# review_status admitted means the method is advertised at the admission it names and no
# further. An explicit not_admitted review is kept.

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


# ----------------------------------------------------------------------------- derivation
def spec_review(existing: str, status: str, gap: str) -> tuple[str, str]:
    """(review_status, admission) from the counterpart status; an explicit not_admitted is kept."""
    if existing == "not_admitted":
        return "not_admitted", "not_admitted"
    if status == "licensed":
        return "admitted", "licensed"
    if status == "executing_unlicensed":
        return "admitted", "executing_unlicensed"
    if status == "not_applicable_with_theory_reason":
        return "specification_only", "refused"
    kept = existing if existing in ("blocked_research", "specification_only") else ""
    return (kept or ("blocked_research" if gap in ("theory", "temporal-identification", "statistical") else "specification_only")), "refused"


def basis(row: dict) -> str:
    if row["status"] in counterparts.EXECUTING:
        return (
            f"{row['status']} per the counterpart inventory; execution {row['execution']}; "
            f"consumer {row['artifact']}; uncertainty {row['uncertainty']} ({row['uncertainty_basis']})"
        )
    if row["status"] == "not_applicable_with_theory_reason":
        return f"settled axis boundary with a typed refusal; fixture {row['execution']}; {row['reason'][:200]}"
    return f"closed with a named next attempt: {row['next_attempt']}"


REFUSAL_TEST = re.compile(r"refus|reject|forg|invalid|unsupported|weak|sparse|forbid|mismatch|tamper|corrupt|withhold", re.I)


def tests_named(ev: Evidence, child: dict, pattern: re.Pattern) -> str:
    """Names of the fixture tests whose name speaks of a refusal; else the fixture tests themselves."""
    names: list[str] = []
    fallback: list[str] = []
    for token in ev.tokens(child["fixtures"]) + [t for t in ev.tokens(child["consumer"]) if "/tests/" in t]:
        path, name = ev.split(token)
        if name:
            (names if pattern.search(name) else fallback).append(name)
        elif path.endswith((".rs", ".py")):
            for found in re.findall(r"#\[test\]\s*(?:#\[[^\]]*\]\s*)*fn (\w+)|^\s*def (test_\w+)", ev.text(path), re.M):
                n = found[0] or found[1]
                (names if pattern.search(n) else fallback).append(n)
    pool = list(dict.fromkeys(names)) or list(dict.fromkeys(fallback))
    return ", ".join(pool[:3]) if pool else "the fixtures named on the child"


GENERIC_COORDINATE = re.compile(r"^(static|temporal)\b")
STALE_STATUSES = ("blocked_research", "specification_only")


class Claims:
    """The claim text of the licensed cells that back a method: the support limitations."""

    def __init__(self, ev: Evidence) -> None:
        self.groups: dict[tuple[str, str, str, str], list[str]] = {}
        for cell in tomllib.loads(ev.text("parity/support_licensed.toml"))["cell"]:
            for estimator in cell["estimators"]:
                key = (cell["query"], cell["graph_class"], estimator, cell["inference"].lower())
                self.groups.setdefault(key, []).append(cell["limitations"])

    def sentences(self, methods: list[dict]) -> list[str]:
        out: list[str] = []
        for method in methods:
            key = (method["target"], method["graph_class"], method["estimator"], method["inference"])
            for text in self.groups.get(key, [])[:1]:
                clean = re.sub(r"\s+", " ", text.replace("`", ""))
                out += [s.strip() for s in re.split(r"(?<=[.;])\s+(?=[A-Z(])", clean) if 25 <= len(s.strip()) <= 420]
        return list(dict.fromkeys(out))


def derive_fields(ev: Evidence, child: dict, spec: dict, methods: list[dict], claims: Claims, stale: bool) -> dict:
    """Fill the unlisted (or, for a record written under an earlier gap, the superseded) fields.

    Sources, in order: the claim text of the licensed cells that back the method, the implementing
    module's own header, and the child's recorded boundary. A filled field is a sentence one of
    them states, so the check can find it there.
    """
    module = child["module"].split(";")[0].strip().split(" ")[0]
    header = module_header(ev, module) if module not in UNLISTED and ev.repo.exists(module) else []
    claim = claims.sentences([m for m in methods if m["kind"] == "licensed_estimator"])
    used: set[str] = set()
    out = dict(spec)
    temporal = ".temporal." in child["id"] or child["id"].split(".")[-2] == "temporal"
    executes = bool(methods) or bool(header)
    notes = re.sub(r"^[;\s]+", "", child["notes"]).strip()

    def find(pools: list[list[str]], pattern: re.Pattern, count: int = 1) -> str:
        for pool in pools:
            got = pick(pool, pattern, used, count)
            if got:
                return got
        return ""

    def fill(field: str, value: str, force: bool = False) -> None:
        if value and (force or out.get(field, "unlisted") in UNLISTED):
            out[field] = value

    refusal = re.compile(r"\brefus\w*|\berror\b|\bfails?\b", re.I)
    audit_like = child["target"] in ("RouteMigration", "ClosureAudit")
    designs = [m["design_time_model"] for m in methods if m["design_time_model"] not in UNLISTED]
    informative = next((d for d in designs if len(d) > 30 and not GENERIC_COORDINATE.match(d)), "")
    identifiers = sorted({i for m in methods if m["kind"] == "licensed_estimator" for i in m["identifier"]})
    if audit_like:
        fill("sampling_units", "none: an audit or migration over method records, not a statistical method")
        fill("identifying_result", "none: not an identification result")
    if identifiers:
        fill("identifying_result", f"{child['target']} identified by {', '.join(identifiers)}")
    fill("sampling_units", informative)
    fill("sampling_units", find([header], SAMPLING))
    fill("assumptions", find([header], ASSUME, 2))
    fill("diagnostics", find([header], DIAGNOSE, 2), force=stale)
    fill("identifying_result", find([header], IDENTIFY))
    realization = child["inference"] if child["inference"] not in UNLISTED and child["inference"] not in ("frequentist", "bayesian") else ""
    summary = header[0] if header else ""
    fill("estimator", f"{realization}: {summary}" if realization and summary else realization or summary, force=stale)
    if not executes:
        reason = child["missing"] if child["missing"] != "none" else notes
        fill("sampling_units", "none: the axis defines no sampling design that this method executes over")
        fill("estimator", "none: no estimator exists on this axis")
        fill("identifying_result", f"none: {reason[:240]}")
        fill("assumptions", f"none beyond the recorded boundary: {reason[:240]}")
        fill("diagnostics", "the typed refusal this axis raises, exercised by its fixture")
    default_units = (
        "ordered observations of the declared series or panel; serial dependence is handled by block resampling or block weights"
        if temporal else "independent rows of the declared sample; the study's sampling design may declare clusters or weights"
    )
    fill("sampling_units", default_units)
    fill("assumptions", f"the identification premises of {child['target']} on the recorded route; the module documents no further premise")
    suites = sorted({v for m in methods for v in m["validations"]})
    if suites:
        fill("diagnostics", f"validation suites {', '.join(suites)} on the licensed cells; refusals asserted by {tests_named(ev, child, REFUSAL_TEST)}")
    fill("diagnostics", f"refusals and diagnostics asserted by {tests_named(ev, child, REFUSAL_TEST)}")
    fill("identifying_result", child["target"] if child["target"] not in UNLISTED else "none")
    fill("estimator", child["inference"] if child["inference"] not in UNLISTED else "none")
    narrowed = child["missing"] if child["missing"] != "none" else ""
    stated = find([claim, header], refusal, 2)
    if stale:
        text = (f"Adjusted claim: {narrowed}. " if narrowed else "") + stated
        fill("failure_policy", text.strip() or "Execution refuses with a typed error naming the unmet premise.", force=True)
    fill("failure_policy", stated)
    fill("failure_policy", f"Refused cases are exercised by {tests_named(ev, child, REFUSAL_TEST)}.")
    return out


def review_links(ev: Evidence, child: dict, methods: list[dict], admission: str) -> tuple[list[str], str]:
    """The provenance records and the public-API review behind an admitted method."""
    module = child["module"].split(";")[0].strip().split(" ")[0]
    if admission not in ("licensed", "executing_unlicensed"):
        return ["not_applicable: no method executes on this axis"], "none: no executing module"
    if module in provenance_links.NOT_AN_ALGORITHM:
        return [f"not_applicable: {provenance_links.NOT_AN_ALGORITHM[module]}"], provenance_links.api_review(module, ev.root)
    records = provenance_links.records_for(module, ev.root)
    if not records:
        known = provenance_links.stems(ev.root)
        for method in methods:
            for feature in method["provenance"]:
                resolved = provenance_links.resolve(feature, known)
                if resolved and resolved not in records:
                    records.append(resolved)
    return sorted(records), provenance_links.api_review(module, ev.root)


def complete(ev: Evidence) -> list[dict]:
    children = tomllib.loads(ev.text(CHILDREN))["child"]
    existing = {s["id"]: s for s in tomllib.loads(ev.text(SPECS))["spec"]} if ev.repo.exists(SPECS) else {}
    rows = {r["id"]: r for r in counterparts.build() if r["kind"] == "method"}
    manifest = tomllib.loads(ev.text(MANIFEST))
    evidenced = counterparts.Evidenced(manifest, list(existing.values()), ev)
    by_child = evidenced.by_child
    specs = []
    claims = Claims(ev)
    for child in children:
        cid = child["id"]
        row = rows[cid]
        before = existing.get(cid, {})
        spec = dict(before or {"id": cid, "references": ["Static and temporal parity", "Frequentist and Bayesian parity"]})
        stale = bool(before) and (
            before.get("gap") != child["gap"]
            or (before.get("review_status") in STALE_STATUSES and row["status"] in counterparts.EXECUTING)
        )
        methods = evidenced.methods_of_child(child, cid.rsplit(".", 1)[-1])
        spec["parent"] = child["parent"]
        spec["gap"] = child["gap"]
        matched = by_child.get(cid) or [m for m in methods if m["route"] not in UNLISTED]
        spec["route"] = (
            child["route"] if child["route"] not in UNLISTED
            else by_child[cid][0]["id"] if cid in by_child
            else matched[0]["route"] if matched else "none"
        )
        if spec.get("target", "unlisted") in UNLISTED:
            spec["target"] = child["target"] if child["target"] not in UNLISTED else "none"
        review, admission = spec_review(existing.get(cid, {}).get("review_status", ""), row["status"], child["gap"])
        spec["review_status"], spec["admission"] = review, admission
        spec["decision_basis"] = basis(row)
        spec = derive_fields(ev, child, spec, methods, claims, stale)
        spec["provenance"], spec["api_review"] = review_links(ev, child, methods, spec["admission"])
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
    rows = {r["id"]: r for r in counterparts.build() if r["kind"] == "method"}
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
        row = rows[sid]
        if spec["gap"] != child["gap"]:
            out.append(f"{sid}: gap {spec['gap']!r} differs from the child's {child['gap']!r}")
        if spec["review_status"] not in REVIEW_STATUSES:
            out.append(f"{sid}: review_status {spec['review_status']!r} is outside the vocabulary")
        if spec["admission"] not in ADMISSIONS:
            out.append(f"{sid}: admission {spec['admission']!r} is outside the vocabulary")
        expected = spec_review(spec["review_status"] if spec["review_status"] == "not_admitted" else "", row["status"], child["gap"])
        if spec["review_status"] != "not_admitted" and (spec["admission"] != expected[1] or (expected[0] == "admitted") != (spec["review_status"] == "admitted")):
            out.append(
                f"{sid}: review {spec['review_status']}/{spec['admission']} contradicts the executing evidence "
                f"({row['status']}); expected {expected[0]}/{expected[1]}"
            )
        if spec["review_status"] == "admitted" and row["status"] not in counterparts.EXECUTING:
            out.append(f"{sid}: admitted but the counterpart row is {row['status']}")
        if not spec["decision_basis"].strip():
            out.append(f"{sid}: no decision basis is recorded")
        out.extend(review_problems(ev, spec))
        if row["status"] in counterparts.EXECUTING or spec["review_status"] == "admitted":
            for field in SCIENTIFIC:
                if spec[field] in UNLISTED:
                    out.append(f"{sid}: admitted record leaves {field} unlisted")
            if spec["route"] in UNLISTED:
                out.append(f"{sid}: admitted record names no route or method")
    for cid in children:
        if cid not in seen:
            out.append(f"{cid}: no specification")
    return out


def review_problems(ev: Evidence, spec: dict) -> list[str]:
    """An admitted method links clean-room provenance and a reviewed public surface."""
    sid = spec["id"]
    if spec["admission"] not in ("licensed", "executing_unlicensed"):
        return []
    out: list[str] = []
    links = spec["provenance"]
    real = [stem for stem in links if not stem.startswith("not_applicable:")]
    if not links:
        out.append(f"{sid}: admitted method links no provenance record and records no reason")
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
            "review_status: " + json.dumps(Counter(s["review_status"] for s in specs)),
            "admission: " + json.dumps(Counter(s["admission"] for s in specs)),
            "unlisted scientific fields: " + json.dumps(
                Counter(f for s in specs for f in SCIENTIFIC if s[f] in UNLISTED)
            ),
        ]
    )


def self_test() -> int:
    assert spec_review("", "licensed", "none") == ("admitted", "licensed")
    assert spec_review("", "executing_unlicensed", "validation") == ("admitted", "executing_unlicensed")
    assert spec_review("not_admitted", "executing_unlicensed", "none") == ("not_admitted", "not_admitted")
    assert spec_review("", "closed_with_next_attempt", "theory") == ("blocked_research", "refused")
    assert spec_review("specification_only", "closed_with_next_attempt", "theory") == ("specification_only", "refused")
    assert spec_review("", "not_applicable_with_theory_reason", "target") == ("specification_only", "refused")
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
    base = {f: "x" for f in FIELD_ORDER} | {"references": ["r"]}
    text = render([base])
    parsed = tomllib.loads(text)["spec"][0]
    assert parsed["references"] == ["r"] and parsed["id"] == "x"
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
        stale = path.read_text(encoding="utf-8") != text
        if stale:
            print(f"FAIL: {SPECS} is stale; run python3 scripts/reconcile_method_specs.py")
        problems = verify(ev, tomllib.loads(path.read_text(encoding="utf-8"))["spec"])
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
