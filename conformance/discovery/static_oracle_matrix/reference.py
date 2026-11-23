# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["numpy==1.26.4", "scipy==1.12.0", "scikit-learn==1.4.2", "pandas==2.2.2", "causal-learn==0.1.4.3", "lingam==1.9.1"]
# ///
"""Retained black-box regeneration of the static discovery reference graphs.

Reads only the fixture's declared data, variable names and column permutations, and runs the
public ``causallearn`` FCI (Fisher-z, declared alpha) and GES (BIC) searches and the public
``lingam`` DirectLiNGAM fit on each permutation.  Edges are canonicalized as unordered pairs of
(node, endpoint); the GES score and DirectLiNGAM coefficients are compared numerically.  No
Antecedent code is imported and no upstream source is stored.

    uv run --managed-python --frozen --script conformance/discovery/static_oracle_matrix/reference.py --check
    uv run --managed-python --frozen --script conformance/discovery/static_oracle_matrix/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import lingam
import numpy as np
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges

HERE = Path(__file__).resolve().parent
SCORE_ATOL = 1e-6
COEF_ATOL = 1e-9
EXPECTED_TOLERANCES = {"lingam_coefficient_atol": 0.12, "lingam_coefficient_rtol": 0.18}


def canonical(edges: list[tuple[str, str, str, str]]) -> list[tuple]:
    return sorted(tuple(sorted([(a, ea), (b, eb)])) for a, ea, b, eb in edges)


def frozen_edges(edges: list[dict]) -> list[tuple]:
    return canonical(
        [(e["node1"], e["endpoint1"], e["node2"], e["endpoint2"]) for e in edges]
    )


def run_static(case: dict, perm: list[int]) -> dict:
    data = np.asarray(case["data"], dtype=float)[:, perm]
    names = [case["var_names"][i] for i in perm]
    graph, _ = fci(
        data,
        independence_test_method="fisherz",
        alpha=case["parameters"]["fci_alpha"],
        verbose=False,
        show_progress=False,
    )
    fci_edges = []
    for edge in graph.get_graph_edges():
        n1 = names[int(edge.get_node1().get_name()[1:]) - 1]
        n2 = names[int(edge.get_node2().get_name()[1:]) - 1]
        fci_edges.append((n1, edge.get_endpoint1().name, n2, edge.get_endpoint2().name))
    result = ges(data, score_func=case["parameters"]["ges_score"], maxP=None, parameters=None)
    adjacency = result["G"].graph
    ges_edges = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            # causal-learn encodes a CPDAG: G[j,i] == 1 and G[i,j] == -1 is i -> j; -1/-1 is i - j.
            if adjacency[i, j] == 0 and adjacency[j, i] == 0:
                continue
            if adjacency[i, j] == -1 and adjacency[j, i] == -1:
                ges_edges.append((names[i], "TAIL", names[j], "TAIL"))
            elif adjacency[j, i] == 1 and adjacency[i, j] == -1:
                ges_edges.append((names[i], "TAIL", names[j], "ARROW"))
            elif adjacency[i, j] == 1 and adjacency[j, i] == -1:
                ges_edges.append((names[j], "TAIL", names[i], "ARROW"))
            else:
                raise SystemExit(f"unexpected CPDAG entry {adjacency[i, j]}, {adjacency[j, i]}")
    return {"fci": canonical(fci_edges), "ges": canonical(ges_edges), "score": float(np.asarray(result["score"]).item())}


def run_lingam(case: dict, perm: list[int]) -> dict:
    data = np.asarray(case["data"], dtype=float)[:, perm]
    names = [case["var_names"][i] for i in perm]
    model = lingam.DirectLiNGAM()
    model.fit(data)
    order = [names[i] for i in model.causal_order_]
    coefficients = {}
    matrix = model.adjacency_matrix_
    for target in range(len(names)):
        for source in range(len(names)):
            if matrix[target, source] != 0.0:
                coefficients[(names[source], names[target])] = float(matrix[target, source])
    return {"order": order, "coefficients": coefficients}


def check(document: dict) -> list[str]:
    problems = []
    if document["tolerances"] != EXPECTED_TOLERANCES:
        problems.append("product comparison tolerances changed from the reviewed values")
    if not document["static_cases"] or not document["lingam_cases"]:
        problems.append("empty case selection")
    for case in document["static_cases"]:
        for ref in case["reference_by_permutation"]:
            got = run_static(case, ref["permutation"])
            tag = f"{case['name']}{ref['permutation']}"
            if got["fci"] != frozen_edges(ref["fci"]["edges"]):
                problems.append(f"{tag} FCI edges differ: {got['fci']}")
            if got["ges"] != frozen_edges(ref["ges"]["edges"]):
                problems.append(f"{tag} GES edges differ: {got['ges']}")
            if abs(got["score"] - ref["ges"]["score"]) > SCORE_ATOL:
                problems.append(f"{tag} GES score differs")
    for case in document["lingam_cases"]:
        for ref in case["reference_by_permutation"]:
            got = run_lingam(case, ref["permutation"])
            tag = f"lingam {case['name']}{ref['permutation']}"
            if got["order"] != ref["causal_order"]:
                problems.append(f"{tag} causal order differs: {got['order']}")
            frozen = {(c["source"], c["target"]): c["coefficient"] for c in ref["coefficients"]}
            if got["coefficients"].keys() != frozen.keys() or any(
                abs(got["coefficients"][k] - frozen[k]) > COEF_ATOL for k in frozen
            ):
                problems.append(f"{tag} coefficients differ")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    wrong = copy.deepcopy(document)
    wrong["static_cases"][0]["reference_by_permutation"][0]["ges"]["score"] += 1.0
    if not check(wrong):
        failures.append("shifted GES score accepted")
    wrong = copy.deepcopy(document)
    wrong["static_cases"][1]["reference_by_permutation"][0]["fci"]["edges"][0]["endpoint1"] = "TAIL"
    if not check(wrong):
        failures.append("wrong FCI endpoint accepted")
    wrong = copy.deepcopy(document)
    wrong["lingam_cases"][0]["reference_by_permutation"][0]["coefficients"][0]["coefficient"] += 0.05
    if not check(wrong):
        failures.append("shifted LiNGAM coefficient accepted")
    wrong = copy.deepcopy(document)
    wrong["lingam_cases"][2]["reference_by_permutation"][0]["causal_order"].reverse()
    if not check(wrong):
        failures.append("wrong causal order accepted")
    wrong = copy.deepcopy(document)
    wrong["tolerances"]["lingam_coefficient_atol"] = 1.0
    if not check(wrong):
        failures.append("widened tolerance accepted")
    wrong = copy.deepcopy(document)
    wrong["static_cases"][0]["data"][0][0] += 5.0
    if not check(wrong):
        failures.append("corrupted input accepted")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = self_test(document) if args.self_test else check(document)
    for p in problems:
        print("FAIL", p)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
