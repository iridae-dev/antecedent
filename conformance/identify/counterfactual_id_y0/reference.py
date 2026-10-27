# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["y0==0.2.11", "networkx==3.6.1", "numpy==2.4.6", "pandas==3.0.6", "scipy==1.17.1", "statsmodels==0.15.0"]
# ///
"""Retained black-box counterfactual identification reference (tooling only).

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

import networkx as nx
from y0.algorithm.identify import identify_outcomes
from y0.algorithm.identify.id_star import id_star
from y0.dsl import (
    CounterfactualVariable,
    Fraction,
    One,
    Probability,
    Product,
    Sum,
    Variable,
    Zero,
)
from y0.graph import NxMixedGraph
from y0.parser import parse_y0

HERE = Path(__file__).resolve().parent
INPUT_FIELDS = (
    "id",
    "nodes",
    "directed",
    "bidirected",
    "treatment",
    "outcome",
    "event",
)
INPUT_SHA256 = "988840bf89d2d974f14c0edcbcf9f64f09c80cdcd1dc414df6a08502f090a515"


def input_digest(cases: list[dict]) -> str:
    grid = [{key: case.get(key) for key in INPUT_FIELDS} for case in cases]
    return hashlib.sha256(
        json.dumps(grid, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def probabilities(expression):
    """Walk documented public algebra containers, not the ID* implementation."""
    if isinstance(expression, Probability):
        yield expression
    elif isinstance(expression, Sum):
        yield from probabilities(expression.expression)
    elif isinstance(expression, Product):
        for factor in expression.expressions:
            yield from probabilities(factor)
    elif isinstance(expression, Fraction):
        yield from probabilities(expression.numerator)
        yield from probabilities(expression.denominator)
    elif not isinstance(expression, (One, Zero)):
        raise TypeError(f"unreviewed expression container: {type(expression).__name__}")


def regenerate(case: dict) -> dict:
    variables = [Variable(f"V{i}") for i in range(case["nodes"])]
    graph = NxMixedGraph.from_edges(
        nodes=variables,
        directed=[(variables[a], variables[b]) for a, b in case["directed"]],
        undirected=[(variables[a], variables[b]) for a, b in case["bidirected"]],
    )
    treatment, outcome = variables[case["treatment"]], variables[case["outcome"]]
    declared_event = f"{{{outcome.name} @ -{treatment.name}: -{outcome.name}, {treatment.name}: +{treatment.name}}}"
    if case["event"] != declared_event:
        raise ValueError("event text does not match the declared query coordinates")
    event = {outcome @ -treatment: -outcome, treatment: +treatment}
    try:
        expression = id_star(graph, event)
    except Exception as error:
        if type(error).__name__ != "ConflictUnidentifiable":
            raise
        # The raw exception includes graph object addresses. The historical
        # harness rendered only its signed conflict list, not repr(error).
        if len(error.args) != 3 or not isinstance(error.args[2], list):
            raise ValueError("changed conflict exception representation") from error
        return {
            "y0_status": "conflict",
            "y0_id_star": f"ConflictUnidentifiable(conflicts={error.args[2]!r})",
        }
    result = {
        "y0_status": "zero" if isinstance(expression, Zero) else "expression",
        "y0_id_star": expression.to_y0(),
        "y0_id_star_repr": repr(expression),
    }
    if not isinstance(expression, Zero):
        terms = []
        for probability in probabilities(expression):
            if probability.parents:
                raise ValueError(
                    "unexpected conditional ID* probability requires review"
                )
            outcomes = {v.get_base() for v in probability.children}
            treatments = {
                intervention.get_base()
                for v in probability.children
                if isinstance(v, CounterfactualVariable)
                for intervention in v.interventions
            }
            identified = identify_outcomes(
                graph, treatments=treatments, outcomes=outcomes
            )
            if identified is None and not treatments:
                raise ValueError(
                    "observational marginal unexpectedly failed identification"
                )
            terms.append(
                {
                    "treatments": sorted(v.name for v in treatments),
                    "outcomes": sorted(v.name for v in outcomes),
                    "identify_outcomes": str(identified)
                    if treatments
                    else "observational",
                }
            )
        if not terms:
            raise ValueError("an ID* expression must supply its observational-ID terms")
        result["y0_terms"] = terms
    return result


def canonical_rendering(text: str) -> str:
    # ID* interventions and summation ranges are sets. Preserve every name and
    # multiplicity; normalize only their order, never factors or graph verdicts.
    return re.sub(
        r"(P|Sum)\[(V[0-9]+(?:, V[0-9]+|,V[0-9]+)*)\]",
        lambda match: (
            match[1]
            + "["
            + ",".join(sorted(part.strip() for part in match[2].split(",")))
            + "]"
        ),
        text,
    )


def latent_dag(case: dict) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(f"V{i}" for i in range(case["nodes"]))
    graph.add_edges_from((f"V{a}", f"V{b}") for a, b in case["directed"])
    for i, (a, b) in enumerate(case["bidirected"]):
        graph.add_edges_from(((f"latent{i}", f"V{a}"), (f"latent{i}", f"V{b}")))
    return graph


def observational_formula(expression, graph: nx.DiGraph):
    """Normalize only public algebra and graph-implied redundant conditioning.

    The declared ADMG is expanded with independent latent common causes.
    NetworkX d-separation is separate from y0's ID routines. No product code is
    used. This comparison assumes a positive law Markov to the declared graph.
    """
    if isinstance(expression, Probability):
        if any(
            isinstance(v, CounterfactualVariable)
            for v in (*expression.children, *expression.parents)
        ):
            raise ValueError("observational ID formula retained an intervention")
        children = {v.name for v in expression.children}
        parents = {v.name for v in expression.parents}
        for parent in sorted(parents):
            if nx.is_d_separator(graph, children, {parent}, parents - {parent}):
                parents.remove(parent)
        return ("P", tuple(sorted(children)), tuple(sorted(parents)))
    if isinstance(expression, Product):
        return (
            "product",
            tuple(
                sorted(observational_formula(e, graph) for e in expression.expressions)
            ),
        )
    if isinstance(expression, Sum):
        return (
            "sum",
            tuple(sorted(v.name for v in expression.ranges)),
            observational_formula(expression.expression, graph),
        )
    if isinstance(expression, Fraction):
        return (
            "fraction",
            observational_formula(expression.numerator, graph),
            observational_formula(expression.denominator, graph),
        )
    if isinstance(expression, (One, Zero)):
        return (type(expression).__name__,)
    raise ValueError(f"unsupported observational algebra: {type(expression).__name__}")


def parse_observational(text: str):
    tokens = re.findall(r"V[0-9]+|Sum|Zero|One|P|[\s\[\](),*/|]", text)
    if "".join(tokens) != text:
        raise ValueError("unsupported observational expression syntax")
    return parse_y0(text)


def normalized_terms(terms: list[dict], case: dict) -> list[str]:
    graph = latent_dag(case)
    result = []
    for term in terms:
        term = copy.deepcopy(term)
        text = term["identify_outcomes"]
        if text not in ("observational", "None"):
            term["identify_outcomes"] = observational_formula(
                parse_observational(text), graph
            )
        result.append(json.dumps(term, sort_keys=True))
    return sorted(result)


def check(document: dict, actual: dict[str, dict]) -> list[str]:
    cases = document.get("cases", [])
    problems = []
    if input_digest(cases) != INPUT_SHA256:
        problems.append("input graph/event grid changed or cases are missing")
    if document.get("oracle", {}).get("kind") != "external_package":
        problems.append("incorrect oracle origin")
    if len(cases) != 257 or len(actual) != 257:
        problems.append("missing or duplicate required comparator cases")
    for case in cases:
        regenerated = actual.get(case["id"])
        if regenerated is None:
            problems.append(f"{case['id']}: missing execution")
            continue
        frozen = copy.deepcopy({k: v for k, v in case.items() if k.startswith("y0_")})
        regenerated = copy.deepcopy(regenerated)
        for field in ("y0_id_star", "y0_id_star_repr"):
            if field in frozen:
                frozen[field] = canonical_rendering(frozen[field])
            if field in regenerated:
                regenerated[field] = canonical_rendering(regenerated[field])
        if "y0_terms" in frozen:
            frozen["y0_terms"] = normalized_terms(frozen["y0_terms"], case)
        if "y0_terms" in regenerated:
            regenerated["y0_terms"] = normalized_terms(regenerated["y0_terms"], case)
        if frozen != regenerated:
            problems.append(
                f"{case['id']}: frozen output differs: expected={frozen!r}, actual={regenerated!r}"
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    for name, version in (
        ("y0", "0.2.11"),
        ("networkx", "3.6.1"),
        ("numpy", "2.4.6"),
        ("pandas", "3.0.6"),
        ("scipy", "1.17.1"),
        ("statsmodels", "0.15.0"),
    ):
        if importlib.metadata.version(name) != version:
            raise ValueError(f"wrong pinned package version: {name}")
    document = json.loads((HERE / "expected.json").read_text())
    if input_digest(document.get("cases", [])) != INPUT_SHA256:
        raise ValueError("frozen input grid changed; refusing regeneration")
    actual = {case["id"]: regenerate(case) for case in document["cases"]}
    problems = check(document, actual)
    if args.self_test:
        collider = next(c for c in document["cases"] if c["id"] == "three.d1.b4.x1.y2")
        graph = latent_dag(collider)
        normalize = lambda text: observational_formula(parse_observational(text), graph)
        if normalize("P(V2 | V0)") != normalize("P(V2)"):
            problems.append("graph-implied redundant condition was not normalized")
        if normalize("P(V2 | V0, V1)") == normalize("P(V2 | V1)"):
            problems.append("fault escaped: conditioning opened the collider path")
        frontdoor = document["cases"][0]
        graph = latent_dag(frontdoor)
        if observational_formula(
            parse_y0("P(V1 | V0)"), graph
        ) == observational_formula(parse_y0("P(V1)"), graph):
            problems.append("fault escaped: a genuine causal parent was removed")
        if canonical_rendering("P[V0](V1)") == canonical_rendering("P[V2](V1)"):
            problems.append("fault escaped: intervention target changed")
        for field, value in (("y0_status", "wrong"), ("y0_id_star", "Zero()")):
            wrong = copy.deepcopy(document)
            wrong["cases"][0][field] = value
            if not check(wrong, actual):
                problems.append(f"fault escaped: wrong {field}")
        wrong = copy.deepcopy(document)
        wrong["cases"][0]["y0_terms"][0]["identify_outcomes"] = "P(V0)"
        if not check(wrong, actual):
            problems.append("fault escaped: wrong observational identification")
        wrong = copy.deepcopy(document)
        wrong["cases"][0]["y0_terms"][1]["identify_outcomes"] = "P(V1)"
        if not check(wrong, actual):
            problems.append("fault escaped: required causal conditioning removed")
        wrong = copy.deepcopy(document)
        wrong["cases"][0]["directed"] = []
        if not check(wrong, actual):
            problems.append("fault escaped: incorrect graph")
        wrong = copy.deepcopy(document)
        wrong["cases"] = []
        if not check(wrong, actual):
            problems.append("fault escaped: empty selection")
        if not check(document, {}):
            problems.append("fault escaped: missing execution")
        wrong = copy.deepcopy(document)
        wrong["oracle"]["kind"] = "closed_form"
        if not check(wrong, actual):
            problems.append("fault escaped: false origin")
    print(
        "\n".join(problems)
        if problems
        else "257 y0 ID* results, observational-ID terms and fault controls passed"
    )
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
