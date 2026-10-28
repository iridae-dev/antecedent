# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the ID-status fixture by exhaustive finite-SCM enumeration.

Identification of P(Y | do(T = 1)) is decided from its definition on a finite family of
structural causal models compatible with the graph: every directed edge is a parent in a
structural function, every bidirected edge is a shared latent binary cause, every node has a
private binary noise term, all sources are uniform, and each structural function ranges over
every Boolean truth table of its inputs. For each model the reference computes the exact
observational distribution and the exact interventional probability by counting worlds. Only
models whose observational distribution gives every cell positive mass are kept, because
identification is a statement about positive distributions (without positivity even the
back-door functional is undefined).

* A case is *not identified* when two models in the family agree on the whole observational
  distribution and disagree on P(Y = 1 | do(T = 1)). The reference prints such a pair.
* A case is reported *identified* only when the family holds no such pair and, where the
  graph has a textbook identifying functional (back-door adjustment, front-door formula), that
  functional equals the interventional probability in every model where its conditioning events
  have positive probability.

The family is finite, so "identified" here means no witness exists in this family and the stated
functional is exact on it; it is not a proof for all models. Standard library only; it imports
no Antecedent code and shares nothing with the product's ID implementation.

    uv run --frozen --script conformance/identify/id_hedge/reference.py --check

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import itertools
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREATMENT, OUTCOME = "T", "Y"


def parse(edges):
    directed, bidirected = [], []
    for edge in edges:
        if "<->" in edge:
            bidirected.append(tuple(edge.split("<->")))
        else:
            directed.append(tuple(edge.split("->")))
    return directed, bidirected


def topological(nodes, parents):
    order, placed = [], set()
    while len(order) < len(nodes):
        for n in nodes:
            if n not in placed and set(parents[n]) <= placed:
                order.append(n)
                placed.add(n)
    return order


def build(edges):
    directed, bidirected = parse(edges)
    nodes = sorted({n for e in directed + bidirected for n in e})
    parents = {n: sorted({a for a, b in directed if b == n}) for n in nodes}
    latents = {n: [i for i, pair in enumerate(bidirected) if n in pair] for n in nodes}
    return nodes, parents, latents, len(bidirected)


def models(nodes, parents, latents, n_latent):
    order = topological(nodes, parents)
    inputs = {n: len(parents[n]) + len(latents[n]) + 1 for n in order}  # + private noise
    tables = {n: list(itertools.product((0, 1), repeat=2 ** inputs[n])) for n in order}
    sources = ["L%d" % i for i in range(n_latent)] + ["N_" + n for n in order]
    worlds = list(itertools.product((0, 1), repeat=len(sources)))
    return order, inputs, tables, sources, worlds


def evaluate(order, parents, latents, tables_for, sources, world, forced=None):
    value = dict(zip(sources, world, strict=True))
    out = {}
    for n in order:
        if forced and n in forced:
            out[n] = forced[n]
            continue
        bits = [out[p] for p in parents[n]] + [value["L%d" % i] for i in latents[n]]
        bits.append(value["N_" + n])
        index = 0
        for b in bits:
            index = index * 2 + b
        out[n] = tables_for[n][index]
    return out


def enumerate_family(edges):
    nodes, parents, latents, n_latent = build(edges)
    order, inputs, tables, sources, worlds = models(nodes, parents, latents, n_latent)
    results = []
    for combo in itertools.product(*(tables[n] for n in order)):
        tables_for = dict(zip(order, combo, strict=True))
        observed = {}
        do_count = 0
        for world in worlds:
            row = evaluate(order, parents, latents, tables_for, sources, world)
            key = tuple(row[n] for n in order)
            observed[key] = observed.get(key, 0) + 1
            forced = evaluate(order, parents, latents, tables_for, sources, world, {TREATMENT: 1})
            do_count += forced[OUTCOME]
        if len(observed) == 2 ** len(order):  # positivity: every observed cell has mass
            results.append((tuple(sorted(observed.items())), do_count))
    return order, len(worlds), results


def backdoor_formula(order, total, observed, z="Z"):
    index = {n: i for i, n in enumerate(order)}
    dist = dict(observed)

    def p(**fixed):
        return sum(c for k, c in dist.items() if all(k[index[n]] == v for n, v in fixed.items()))

    value = Fraction(0)
    for zv in (0, 1):
        pz = p(**{z: zv})
        pt = p(**{TREATMENT: 1, z: zv})
        if pz and not pt:
            return None
        if pz:
            value += Fraction(pz, total) * Fraction(p(**{TREATMENT: 1, z: zv, OUTCOME: 1}), pt)
    return value


def frontdoor_formula(order, total, observed, m="M"):
    index = {n: i for i, n in enumerate(order)}
    dist = dict(observed)

    def p(**fixed):
        return sum(c for k, c in dist.items() if all(k[index[n]] == v for n, v in fixed.items()))

    value = Fraction(0)
    for mv in (0, 1):
        pt1 = p(**{TREATMENT: 1})
        if not pt1:
            return None
        pm = Fraction(p(**{TREATMENT: 1, m: mv}), pt1)
        inner = Fraction(0)
        for tv in (0, 1):
            ptm = p(**{TREATMENT: tv, m: mv})
            pt = p(**{TREATMENT: tv})
            if not ptm:
                if pm:
                    return None
                continue
            inner += Fraction(p(**{TREATMENT: tv, m: mv, OUTCOME: 1}), ptm) * Fraction(pt, total)
        value += pm * inner
    return value


def derive_status(case):
    order, total, results = enumerate_family(case["edges"])
    by_observed = {}
    for observed, do_count in results:
        by_observed.setdefault(observed, set()).add(do_count)
    witness = any(len(values) > 1 for values in by_observed.values())
    if witness:
        return "not_identified", f"{len(results)} models, witness pair found"
    formula = {"backdoor_dag": backdoor_formula, "frontdoor_admg": frontdoor_formula}.get(case["id"])
    tested = 0
    if formula is not None:
        for observed, do_count in results:
            value = formula(order, total, observed)
            if value is None:
                continue
            tested += 1
            if value != Fraction(do_count, total):
                return "formula_fails", f"{case['id']} formula differs from the interventional value"
    return "identified", f"{len(results)} models, no witness; formula exact on {tested}"


def main() -> int:
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = []
    for case in document["cases"]:
        status, note = derive_status(case)
        print(f"{case['id']}: {status} ({note})")
        if status != case["status"]:
            problems.append(f"{case['id']}: derived {status} != fixture {case['status']}")
    for problem in problems:
        print("FAIL", problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
