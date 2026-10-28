# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the arrow-strength fixture for linear-Gaussian structural models.

Two quantities are derived for every edge, each in exact rational arithmetic from the declared
noise standard deviations and coefficients only:

* ``variance_contribution = beta^2 * Var(parent)``, with Var(parent) read off the exact joint
  covariance of the model (the covariance is built by solving the linear system node by node).
* ``strength``, the cited causal strength: the Kullback-Leibler divergence between the model's
  joint Gaussian law and the joint law of the model with the edge cut. Cutting parent -> child
  feeds the child an independent copy of the parent drawn from the parent's marginal. The cut
  law is the covariance of the model extended by a fresh independent root with the parent's
  variance, restricted to the original nodes, and the divergence is
  ``1/2 (tr(S_cut^-1 S) - n + ln(det S_cut / det S))`` evaluated with exact determinants and a
  60-digit logarithm. It is checked against the closed form
  ``1/2 ln(1 + beta^2 Var(parent) / sigma_child^2)`` that does not depend on how the parents are
  correlated; the two routes share nothing but the declared inputs.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/attribution/arrow_strength/reference.py --check

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
getcontext().prec = 60


def exact(value) -> Fraction:
    return Fraction(repr(value))


def ln(value: Fraction) -> Decimal:
    return (Decimal(value.numerator) / Decimal(value.denominator)).ln()


def covariance(nodes, sigma, edges):
    """Exact covariance of ``node = sum(beta * parent) + noise`` for nodes in topological order."""
    index = {name: i for i, name in enumerate(nodes)}
    n = len(nodes)
    cov = [[Fraction(0)] * n for _ in range(n)]
    for j, name in enumerate(nodes):
        betas = [(index[p], exact(b)) for (p, c), b in edges.items() if c == name]
        for k in range(j):
            cov[j][k] = cov[k][j] = sum((b * cov[p][k] for p, b in betas), Fraction(0))
        cov[j][j] = exact(sigma[name]) ** 2 + sum(
            (bp * bq * cov[p][q] for p, bp in betas for q, bq in betas), Fraction(0)
        )
    return cov


def determinant_and_inverse(matrix):
    n = len(matrix)
    work = [row[:] + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(matrix)]
    det = Fraction(1)
    for col in range(n):
        pivot = next(r for r in range(col, n) if work[r][col] != 0)
        if pivot != col:
            work[col], work[pivot] = work[pivot], work[col]
            det = -det
        det *= work[col][col]
        scale = work[col][col]
        work[col] = [v / scale for v in work[col]]
        for r in range(n):
            if r != col and work[r][col] != 0:
                factor = work[r][col]
                work[r] = [a - factor * b for a, b in zip(work[r], work[col])]
    return det, [row[n:] for row in work]


def cut_divergence(nodes, sigma, edges, parent, child):
    """KL(joint || joint with parent -> child cut) from exact covariances."""
    copy_name = f"{parent}__independent_copy"
    extended_nodes = [copy_name, *nodes]
    extended_sigma = dict(sigma)
    extended_edges = {}
    # The copy is a root with the parent's own marginal variance: sigma^2 = Var(parent).
    base = covariance(nodes, sigma, edges)
    extended_sigma[copy_name] = None
    parent_variance = base[nodes.index(parent)][nodes.index(parent)]
    for key, beta in edges.items():
        extended_edges[(copy_name, child) if key == (parent, child) else key] = beta
    index = {name: i for i, name in enumerate(extended_nodes)}
    n = len(extended_nodes)
    cov = [[Fraction(0)] * n for _ in range(n)]
    for j, name in enumerate(extended_nodes):
        betas = [(index[p], exact(b)) for (p, c), b in extended_edges.items() if c == name]
        for k in range(j):
            cov[j][k] = cov[k][j] = sum((b * cov[p][k] for p, b in betas), Fraction(0))
        own = parent_variance if name == copy_name else exact(sigma[name]) ** 2
        cov[j][j] = own + sum((bp * bq * cov[p][q] for p, bp in betas for q, bq in betas), Fraction(0))
    cut = [row[1:] for row in cov[1:]]
    det_full, _ = determinant_and_inverse(base)
    det_cut, inverse_cut = determinant_and_inverse(cut)
    m = len(nodes)
    trace = sum(
        (inverse_cut[i][k] * base[k][i] for i in range(m) for k in range(m)), Fraction(0)
    )
    return (Decimal(trace.numerator) / Decimal(trace.denominator) - m + ln(det_cut / det_full)) / 2


def derive(nodes, sigma, edges):
    """(parent, child) -> (Var(parent), variance_contribution, closed form, divergence)."""
    cov = covariance(nodes, sigma, edges)
    out = {}
    for (parent, child), beta in edges.items():
        var_parent = cov[nodes.index(parent)][nodes.index(parent)]
        contribution = exact(beta) ** 2 * var_parent
        closed = ln(1 + contribution / exact(sigma[child]) ** 2) / 2
        out[(parent, child)] = (
            var_parent,
            contribution,
            closed,
            cut_divergence(nodes, sigma, edges, parent, child),
        )
    return out


def close(got, want, tol, label, problems):
    if want is None or not math.isclose(float(got), want, rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {float(got)!r} != fixture {want!r} (tol {tol})")


def compare(label, nodes, sigma, edge_list, tol, problems):
    edges = {(e["parent"], e["child"]): e["coefficient"] for e in edge_list}
    derived = derive(nodes, sigma, edges)
    for edge in edge_list:
        key = (edge["parent"], edge["child"])
        var_parent, contribution, closed, divergence = derived[key]
        name = f"{label} edge {key[0]}->{key[1]}"
        close(var_parent, edge["parent_variance"], tol, f"{name} Var(parent)", problems)
        close(contribution, edge["variance_contribution"], tol, f"{name} variance_contribution", problems)
        close(divergence, edge["strength"], tol, f"{name} strength (cut divergence)", problems)
        close(closed, edge["strength"], tol, f"{name} strength (closed form)", problems)
        if abs(closed - divergence) > Decimal("1e-40"):
            problems.append(f"{name}: closed form {closed} and divergence {divergence} disagree")
    return derived


def check(document):
    problems = []
    tol = document["tolerance"]["absolute"]

    # Two root parents of one child with the child noise declared in the fixture.
    parents = {p["raw"]: p for p in document["parents"]}
    nodes = [f"n{p['raw']}" for p in document["parents"]] + [f"n{document['child']['raw']}"]
    sigma = {f"n{raw}": p["sigma"] for raw, p in parents.items()}
    sigma[f"n{document['child']['raw']}"] = document["child"]["sigma"]
    simple = [
        {
            "parent": f"n{e['parent_raw']}",
            "child": f"n{e['child_raw']}",
            "coefficient": e["coefficient"],
            "parent_variance": parents[e["parent_raw"]]["variance"],
            "variance_contribution": e["variance_contribution"],
            "strength": e["strength"],
        }
        for e in document["edges"]
    ]
    derived = compare("independent parents", nodes, sigma, simple, tol, problems)
    by_contribution = sorted(simple, key=lambda e: -float(derived[(e["parent"], e["child"])][1]))
    by_beta = sorted(simple, key=lambda e: -abs(e["coefficient"]))
    inverted = [e["parent"] for e in by_contribution] != [e["parent"] for e in by_beta]
    if inverted != document["abs_beta_ranking_is_inverted"]:
        problems.append("abs_beta_ranking_is_inverted disagrees with the derived rankings")

    # Correlated parents: the variance contribution can exceed the child's variance, where the
    # normalised form with Var(child) is undefined, while the cut divergence stays finite.
    block = document["correlated_parents"]
    derived = compare(
        "correlated parents", block["nodes"], block["sigma"], block["edges"], tol, problems
    )
    cov = covariance(
        block["nodes"], block["sigma"], {(e["parent"], e["child"]): e["coefficient"] for e in block["edges"]}
    )
    exceeding = 0
    for edge in block["edges"]:
        child_variance = cov[block["nodes"].index(edge["child"])][block["nodes"].index(edge["child"])]
        close(child_variance, edge["child_variance"], tol, f"correlated child {edge['child']} Var", problems)
        if derived[(edge["parent"], edge["child"])][1] >= child_variance:
            exceeding += 1
    if exceeding != block["edges_exceeding_child_variance"]:
        problems.append(f"{exceeding} edges exceed Var(child), fixture says {block['edges_exceeding_child_variance']}")

    # Rescaling the child leaves the divergence unchanged.
    scaled_nodes = block["nodes"]
    scaled_sigma = {k: v for k, v in block["sigma"].items()}
    scaled_edges = {(e["parent"], e["child"]): e["coefficient"] for e in block["edges"]}
    base = derive(scaled_nodes, scaled_sigma, scaled_edges)
    last = scaled_nodes[-1]
    scaled_sigma[last] = 2 * scaled_sigma[last]
    for key in list(scaled_edges):
        if key[1] == last:
            scaled_edges[key] = 2 * scaled_edges[key]
    scaled = derive(scaled_nodes, scaled_sigma, scaled_edges)
    for key in scaled_edges:
        if key[1] == last and abs(scaled[key][3] - base[key][3]) > Decimal("1e-40"):
            problems.append(f"edge {key}: divergence changes under rescaling the child")

    contrast = document["do_contrast"]
    close(contrast["coefficient"] * contrast["delta"], contrast["expected"], tol, "do_contrast", problems)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.parse_args()
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
