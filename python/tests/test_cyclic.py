# SPDX-License-Identifier: MIT OR Apache-2.0
"""Independent cyclic workflow and fresh-process artifact consumers."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
from antecedent.cyclic import (
    Equation,
    EquilibriumModel,
    FitOptions,
    OdeSystem,
    OrnsteinUhlenbeck,
    PolynomialSystem,
    SolvePolicy,
    consume_artifact,
    consume_family_artifact,
)


def model() -> EquilibriumModel:
    return EquilibriumModel(
        (
            Equation("a", 0.0, (0.0, 0.4)),
            Equation("b", 0.0, (0.5, 0.0)),
        ),
        (1.0, 0.3, 0.3, 1.0),
    )


def test_nonlinear_shared_worlds_counterfactual_and_fresh_artifact(tmp_path: Path) -> None:
    prepared = model().prepare(
        [[], [{"variable": 0, "value": 2.0}]],
        [[0.1, 0.2], [-0.1, -0.2]],
        factual=[0.3, 0.7],
    )
    result = prepared.execute()
    assert result.report["worlds"][1][0]["values"][1] == pytest.approx(math.tanh(1.0) + 0.2)
    assert result.report["worlds"][0][0]["globally_unique"] is True
    assert result.report["claim"] == "specified_or_fitted_model_conditional"
    assert result.report["calibration"] == "unmeasured"
    loaded = consume_artifact(result.export(), expected_identity=result.identity)
    assert loaded.report == result.report
    artifact = tmp_path / "cyclic.ant"
    artifact.write_bytes(result.export())
    script = (
        "import json, pathlib; from antecedent.cyclic import consume_artifact; "
        f"r=consume_artifact(pathlib.Path({str(artifact)!r}).read_bytes(),expected_identity={result.identity!r}); "
        "print(json.dumps(r.report,sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == result.report
    with pytest.raises(Exception, match="identity|mismatch"):
        consume_artifact(result.export(), expected_identity="0" * 64)


def test_linear_limit_joint_clamp_and_retained_fit() -> None:
    linear = EquilibriumModel(
        (Equation("a", 0.0, (0.0, 0.4), "identity"), Equation("b", 0.0, (0.5, 0.0), "identity")),
        (1.0, 0.0, 0.0, 1.0),
    )
    result = linear.prepare([[]], [[1.0, 0.0]]).execute()
    assert result.report["worlds"][0][0]["values"] == pytest.approx([1.25, 0.625])
    fitted = linear.prepare(
        [[{"variable": 0, "value": 2.0}]],
        [[0.0, 0.0]],
        parameters=[{"kind": "weight", "equation": 1, "parent": 0}],
        rows=[
            {"values": [1.0, 0.7], "clamps": [{"variable": 0, "value": 1.0}]},
            {"values": [-1.0, -0.7], "clamps": [{"variable": 0, "value": -1.0}]},
        ],
        fit_options=FitOptions(),
    ).execute()
    assert fitted.report["fit"]["converged"]
    assert fitted.report["worlds"][0][0]["values"][1] == pytest.approx(1.4, abs=1e-4)
    assert (
        consume_artifact(fitted.export(), expected_identity=fitted.identity).report == fitted.report
    )


def test_uncertified_roots_are_numerical_only() -> None:
    multiple = EquilibriumModel(
        (Equation("a", 0.0, (0.0, 2.0)), Equation("b", 0.0, (2.0, 0.0))), (1.0, 0.0, 0.0, 1.0)
    )
    with pytest.raises(Exception, match="contractive"):
        multiple.prepare([[]], [[0.0, 0.0]]).execute()
    numerical = multiple.prepare(
        [[]], [[0.0, 0.0]], initial=[1.0, 1.0], policy=SolvePolicy(require_contraction=False)
    ).execute()
    assert numerical.report["worlds"][0][0]["values"][0] > 0.9
    assert numerical.report["worlds"][0][0]["globally_unique"] is False


def test_continuous_time_families_and_replay() -> None:
    ode = OdeSystem(
        (Equation("a", 0.0, (0.0, -1.0), "identity"), Equation("b", 0.0, (1.0, 0.0), "identity")),
        (0.0, 0.0),
    )
    path = ode.prepare([1.0, 0.0], [0.0, 1.0]).execute()
    assert path.report["answer"]["result"]["states"][1] == pytest.approx(
        [math.cos(1.0), math.sin(1.0)], abs=1e-6
    )
    assert (
        consume_family_artifact(path.artifact, expected_identity=path.identity).report
        == path.report
    )
    ou = OrnsteinUhlenbeck((-2.0,), (3.0,), (4.0,))
    stationary = ou.stationary().execute()
    assert stationary.report["answer"]["result"] == pytest.approx([1.0])
    transient = ou.transient([0.0], [0.0], [0.0, 1.0]).execute()
    assert transient.report["answer"]["result"]["means"][1][0] == pytest.approx(
        3.0 * (1.0 - math.exp(-2.0)), abs=1e-6
    )
    with pytest.raises(ValueError, match="stability"):
        OrnsteinUhlenbeck((2.0,), (0.0,), (1.0,)).stationary().execute()


def test_polynomial_solution_sets_retain_unresolved_regions() -> None:
    system = PolynomialSystem(
        ("x",),
        (
            {
                "id": "cubic",
                "terms": [
                    {"coefficient": 1.0, "powers": [3]},
                    {"coefficient": -1.0, "powers": [1]},
                ],
            },
        ),
        ({"lower": -2.0, "upper": 2.0},),
    )
    result = system.prepare().execute()
    assert len(result.report["answer"]["result"]["roots"]) == 3
    assert result.report["answer"]["result"]["unresolved"] == []
    assert (
        consume_family_artifact(result.artifact, expected_identity=result.identity).report
        == result.report
    )
    short = system.prepare(max_boxes=1).execute()
    assert short.report["answer"]["result"]["unresolved"]
    replaced = system.replace_equation("cubic", 0, 0.5).prepare().execute()
    assert len(replaced.report["answer"]["result"]["roots"]) == 1


def test_prepared_fit_retarget_has_zero_refits_and_refresh_changes_identity() -> None:
    linear = EquilibriumModel(
        (Equation("a", 0.0, (0.0, 0.4), "identity"), Equation("b", 0.0, (0.5, 0.0), "identity")),
        (1.0, 0.0, 0.0, 1.0),
    )
    rows = [{"values": [x, 0.7 * x], "clamps": [{"variable": 0, "value": x}]} for x in [-1.0, 1.0]]
    prepared = linear.prepare(
        [[]], [[0.0, 0.0]], parameters=[{"kind": "weight", "equation": 1, "parent": 0}], rows=rows
    )
    assert prepared.inspect()["fits_performed"] == 1
    original = prepared.execute()
    retargeted = prepared.retarget([[{"variable": 0, "value": 2.0}]], [[0.0, 0.0]])
    assert retargeted.inspect()["fits_performed"] == 0
    result = retargeted.execute()
    assert result.report["fit"] == original.report["fit"]
    assert result.report["worlds"][0][0]["values"][1] == pytest.approx(1.4, abs=1e-4)
    assert (
        consume_artifact(result.export(), expected_identity=result.identity).report == result.report
    )
    refreshed = prepared.refresh(
        [{"values": [x, 0.6 * x], "clamps": [{"variable": 0, "value": x}]} for x in [-1.0, 1.0]]
    )
    assert refreshed.inspect()["fits_performed"] == 1
    assert refreshed.inspect()["identity"] != prepared.inspect()["identity"]


def test_graph_identification_fci_bipartite_and_domain_invariance_replay() -> None:
    from antecedent.cyclic import BipartiteCausalGraph, CyclicGraph

    graph = CyclicGraph(3, ((0, 1), (1, 2), (2, 1)))
    identified = graph.identify([2], [0], io_scm=True, scc_product_density=True).execute()
    assert identified.report["answer"]["result"]["status"] == "identified"
    assert identified.report["answer"]["result"]["expression"]
    assert (
        consume_family_artifact(identified.artifact, expected_identity=identified.identity).report
        == identified.report
    )
    failed = graph.identify([2], [1], io_scm=True, scc_product_density=True).execute()
    assert failed.report["answer"]["result"]["status"] == "not_identified_by_procedure"
    discovered = graph.oracle_discovery(simple_scm=True, sigma_faithful=True).execute()
    assert discovered.report["answer"]["result"]["ci_source"] == "sigma_oracle"
    assert "inducing path" in discovered.report["answer"]["result"]["interpretation"]
    bipartite = BipartiteCausalGraph(3, (2,), ((0, 2), (0, 1)))
    determined = bipartite.b_separation(0, 1, [2]).execute()
    assert determined.report["answer"]["result"]["separated"]
    invariant = bipartite.domain_invariance([1], [0], []).execute()
    assert invariant.report["answer"]["result"]["invariant"]
    changed = bipartite.domain_invariance([1], [1], []).execute()
    assert not changed.report["answer"]["result"]["invariant"]


def test_aligned_parameter_and_partial_counterfactual_posterior_artifacts() -> None:
    from antecedent.cyclic import SamplingPolicy, partial_counterfactual, posterior_responses

    linear = EquilibriumModel(
        (Equation("a", 0.0, (0.0, 0.4), "identity"), Equation("b", 0.0, (0.5, 0.0), "identity")),
        (1.0, 0.0, 0.0, 1.0),
    )
    sampling = SamplingPolicy(chains=2, warmup=16, draws=16, proposal_scale=0.2, seed=91)
    worlds = [[], [{"variable": 0, "value": 2.0}]]
    partial = partial_counterfactual(
        linear,
        [1.0, None],
        worlds,
        measurement_variances=[0.0, 0.0],
        initial=[1.0, 0.8],
        sampling=sampling,
    ).execute()
    result = partial.report["answer"]["result"]
    assert not result["abduction"]["diagnostics_pass"]
    for factual, counterfactual in zip(
        result["abduction"]["states"], result["worlds"][1], strict=True
    ):
        assert counterfactual["values"][1] == pytest.approx(factual[1] + 0.5, abs=1e-9)
    assert (
        consume_family_artifact(partial.artifact, expected_identity=partial.identity).report
        == partial.report
    )
    rows = [{"values": [x, 0.5 * x], "clamps": [{"variable": 0, "value": x}]} for x in [-1.0, 1.0]]
    posterior = posterior_responses(
        linear,
        [{"kind": "weight", "equation": 1, "parent": 0}],
        rows,
        worlds,
        [[0.0, 0.0]],
        prior_sd=1.0,
        sampling=sampling,
    ).execute()
    assert len(posterior.report["answer"]["result"]["worlds"][1]) == 32
    assert (
        consume_family_artifact(posterior.artifact, expected_identity=posterior.identity).report
        == posterior.report
    )


def test_dynamic_fitting_and_explicit_selection_equation_likelihood() -> None:
    from dataclasses import asdict

    from antecedent.cyclic import fit_polynomial_candidates

    ode = OdeSystem((Equation("drift", 0.0, (0.0,), "identity"),), (1.0,))
    observation = {
        "initial": [2.0],
        "times": [0.0, 0.5, 1.0],
        "values": [[2.0], [2.0 * math.exp(-1.0) + 0.01], [2.0 * math.exp(-2.0) - 0.01]],
        "variances": [0.01],
        "schedule": [],
    }
    fit = ode.fit([{"kind": "decay", "equation": 0}], [observation], score_tolerance=1e-5).execute()
    assert fit.report["answer"]["result"]["converged"]
    assert fit.report["answer"]["result"]["model"]["decay"][0] == pytest.approx(2.0, abs=0.05)
    ou = OrnsteinUhlenbeck((-1.0,), (0.0,), (4.0,))
    rows = [{"values": [value], "clamps": []} for value in [-0.5, 1.5] * 10]
    fitted_ou = ou.fit(
        [{"kind": "drift", "row": 0, "column": 0}, {"kind": "center", "variable": 0}], rows
    ).execute()
    assert fitted_ou.report["answer"]["result"]["converged"]
    assert (
        consume_family_artifact(fitted_ou.artifact, expected_identity=fitted_ou.identity).report
        == fitted_ou.report
    )
    candidates = []
    for a in [0.25, 0.81]:
        system = PolynomialSystem(
            ("x",),
            (
                {
                    "id": "square",
                    "terms": [
                        {"coefficient": 1.0, "powers": [2]},
                        {"coefficient": -a, "powers": [0]},
                    ],
                },
            ),
            ({"lower": -2.0, "upper": 2.0},),
        )
        candidates.append(
            {
                "id": f"a={a}",
                "model": asdict(system),
                "selection": [
                    {"bounds": [{"lower": -2.0, "upper": 0.0}], "probability": 0.0},
                    {"bounds": [{"lower": 0.0, "upper": 2.0}], "probability": 1.0},
                ],
            }
        )
    result = fit_polynomial_candidates(
        candidates, [{"values": [0.9], "variances": [0.01]}]
    ).execute()
    assert result.report["answer"]["result"]["best_candidate"] == "a=0.81"
    assert result.report["answer"]["result"]["finite_set_optimum_certified"]


def test_cyclic_execution_honors_resource_and_cancellation_controls() -> None:
    from antecedent import _native

    model = EquilibriumModel(
        (Equation("a", 0.0, (0.0, 0.2)), Equation("b", 0.0, (0.2, 0.0))), (1.0, 0.0, 0.0, 1.0)
    )
    prepared = model.prepare([[]], [[0.0, 0.0]])
    with pytest.raises(ValueError, match="(?i)(budget|memory|resource|limit|admission)"):
        prepared.execute(max_bytes=1)
    token = _native.CancellationToken()
    token.cancel()
    with pytest.raises(ValueError, match="cancel"):
        prepared.execute(cancel=token)
    family = OrnsteinUhlenbeck((-1.0,), (0.0,), (1.0,)).stationary()
    with pytest.raises(ValueError, match="cancel"):
        family.execute(cancel=token)


def test_all_cyclic_family_artifacts_replay_in_fresh_process(tmp_path: Path) -> None:
    from antecedent.cyclic import BipartiteCausalGraph, CyclicGraph

    requests = [
        OrnsteinUhlenbeck((-2.0,), (1.0,), (4.0,)).stationary(),
        OdeSystem((Equation("decay", 0.0, (0.0,), "identity"),), (1.0,)).prepare([2.0], [0.0, 1.0]),
        PolynomialSystem(
            ("x",),
            (
                {
                    "id": "linear",
                    "terms": [
                        {"coefficient": 1.0, "powers": [1]},
                        {"coefficient": -0.5, "powers": [0]},
                    ],
                },
            ),
            ({"lower": -1.0, "upper": 1.0},),
        ).prepare(),
        CyclicGraph(3, ((0, 1), (1, 2), (2, 1))).identify(
            [2], [0], io_scm=True, scc_product_density=True
        ),
        BipartiteCausalGraph(3, (0,), ((0, 1), (1, 2))).b_separation(0, 2, [1]),
    ]
    expected = []
    paths = []
    for i, request in enumerate(requests):
        result = request.execute()
        path = tmp_path / f"family-{i}.ant"
        path.write_bytes(result.artifact)
        expected.append(result.report)
        paths.append((str(path), result.identity))
    program = (
        "import json,pathlib; from antecedent.cyclic import consume_family_artifact; "
        + f"pairs={paths!r}; print(json.dumps([consume_family_artifact(pathlib.Path(p).read_bytes(),expected_identity=i).report for p,i in pairs],sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == expected


def test_joint_parameter_factual_abduction_preserves_coupling_and_replays() -> None:
    from antecedent.cyclic import SamplingPolicy, joint_counterfactual

    system = EquilibriumModel((Equation("x", 0.0, (0.0,), "identity"),), (1.0,))
    result = joint_counterfactual(
        system,
        [{"kind": "intercept", "equation": 0}],
        [{"values": [value], "clamps": []} for value in [0.0, 1.0]],
        [2.0],
        [0.5],
        [[], [{"variable": 0, "value": 3.0}]],
        prior_sd=1.0,
        initial=[1.5],
        sampling=SamplingPolicy(chains=2, warmup=32, draws=32, proposal_scale=0.5, seed=932),
    ).execute()
    answer = result.report["answer"]["result"]
    posterior = answer["posterior"]
    assert not posterior["factual"]["diagnostics_pass"]
    for theta, state, noise, factual, clamped in zip(
        posterior["parameter_draws"],
        posterior["factual"]["states"],
        posterior["factual"]["disturbances"],
        answer["worlds"][0],
        answer["worlds"][1],
        strict=True,
    ):
        assert theta[0] + noise[0] == pytest.approx(state[0])
        assert factual["values"][0] == pytest.approx(state[0])
        assert clamped["values"][0] == 3.0
    assert (
        consume_family_artifact(result.artifact, expected_identity=result.identity).report
        == result.report
    )


def test_named_structural_actions_keep_distinct_semantics_and_failed_mass() -> None:
    from antecedent.cyclic import action_worlds

    linear = EquilibriumModel(
        (Equation("a", 0.0, (0.0, 0.4), "identity"), Equation("b", 0.0, (0.5, 0.0), "identity")),
        (1.0, 0.0, 0.0, 1.0),
    )
    worlds = [
        {"id": "clamp", "actions": [{"kind": "clamp", "mechanism": "a", "value": 2.0}]},
        {"id": "shock", "actions": [{"kind": "additive_shock", "mechanism": "a", "delta": 1.0}]},
        {
            "id": "change",
            "actions": [{"kind": "change_mechanism", "mechanism": "b", "weights": [0.0, 0.0]}],
        },
        {
            "id": "replace",
            "actions": [
                {
                    "kind": "replace_mechanism",
                    "mechanism": "a",
                    "replacement": {
                        "id": "a",
                        "intercept": 3.0,
                        "weights": [0.0, 0.0],
                        "activation": "identity",
                    },
                }
            ],
        },
        {
            "id": "inconsistent",
            "actions": [
                {"kind": "change_mechanism", "mechanism": "a", "weights": [0.0, 1.0]},
                {"kind": "change_mechanism", "mechanism": "b", "weights": [1.0, 0.0]},
            ],
        },
    ]
    result = action_worlds(
        linear,
        worlds,
        [[1.0, 0.0]],
        factual=[1.25, 0.625],
        policy=SolvePolicy(require_contraction=False),
    ).execute()
    answers = result.report["answer"]["result"]
    for answer, expected in zip(
        answers[:4], [[2.0, 1.0], [2.5, 1.25], [1.0, 0.0], [4.0, 2.0]], strict=True
    ):
        assert answer["rows"][0]["values"] == pytest.approx(expected)
        assert answer["counterfactual"]["values"] == pytest.approx(expected)
    assert len(answers) == 5
    assert answers[4]["rows"][0]["failure"] == "singular"
    assert answers[4]["counterfactual"]["failure"] == "singular"
    assert (
        consume_family_artifact(result.artifact, expected_identity=result.identity).report
        == result.report
    )


def test_nonlinear_end_to_end_milestone_with_overlapping_feedback_loops(tmp_path: Path) -> None:
    import random

    random_source = random.Random(826)
    covariance = (0.0009, 0.00027, 0.0, 0.00027, 0.0009, 0.0, 0.0, 0.0, 0.0009)
    template = EquilibriumModel(
        (
            Equation("a", 0.0, (0.0, 0.1, 0.1)),
            Equation("b", 0.0, (0.1, 0.0, 0.0)),
            Equation("c", 0.0, (-0.1, 0.0, 0.0)),
        ),
        covariance,
    )
    rows = []
    for i in range(160):
        z = [random_source.gauss(0.0, 1.0) for _ in range(3)]
        disturbance = [0.03 * z[0], 0.03 * (0.3 * z[0] + math.sqrt(0.91) * z[1]), 0.03 * z[2]]
        clamps = (
            [] if i % 4 == 0 else [{"variable": i % 4 - 1, "value": 1.0 if i % 8 < 4 else -1.0}]
        )
        imposed = {clamp["variable"]: clamp["value"] for clamp in clamps}
        a, b, c = 0.0, 0.0, 0.0
        for _ in range(100):
            # Independent reference generator, no native solve/fitting calls.
            a, b, c = (
                imposed.get(0, math.tanh(0.3 * b + 0.2 * c) + disturbance[0]),
                imposed.get(1, math.tanh(0.25 * a) + disturbance[1]),
                imposed.get(2, math.tanh(-0.2 * a) + disturbance[2]),
            )
        rows.append({"values": [a, b, c], "clamps": clamps})
    joint = [{"variable": 1, "value": 0.8}, {"variable": 2, "value": -0.6}]
    prepared = template.prepare(
        [[], joint],
        [[0.0, 0.0, 0.0]],
        factual=[0.2, 0.3, -0.1],
        parameters=[
            {"kind": "weight", "equation": i, "parent": j}
            for i, j in [(0, 1), (0, 2), (1, 0), (2, 0)]
        ],
        rows=rows,
    )
    result = prepared.execute()
    assert result.report["fit"]["converged"]
    assert result.report["worlds"][1][0]["values"][0] == pytest.approx(math.tanh(0.12), abs=0.035)
    expected_counterfactual = math.tanh(0.12) + 0.2 - math.tanh(0.07)
    assert result.report["counterfactuals"][1]["values"][0] == pytest.approx(
        expected_counterfactual, abs=0.035
    )
    assert prepared.inspect()["fits_performed"] == 1
    path = tmp_path / "nonlinear-milestone.ant"
    path.write_bytes(result.artifact)
    program = (
        "import json,pathlib; from antecedent.cyclic import consume_artifact; "
        + f"r=consume_artifact(pathlib.Path({str(path)!r}).read_bytes(),expected_identity={result.identity!r});print(json.dumps(r.report,sort_keys=True))"
    )
    replay = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(replay.stdout) == result.report


def test_family_preparation_reuses_products_and_refresh_invalidates_binding() -> None:
    request = OrnsteinUhlenbeck((-2.0,), (1.0,), (4.0,)).stationary()
    assert request.inspect()["preparation_evaluations"] == 0
    first = request.execute()
    assert request.inspect()["preparation_evaluations"] == 1
    assert request.execute().report == first.report
    with pytest.raises(ValueError, match="(?i)(memory|limit|budget|admission)"):
        request.execute(max_bytes=1)
    replacement = json.loads(request.request_json)
    replacement["model"]["center"] = [2.0]
    refreshed = request.refresh(json.dumps(replacement))
    assert refreshed.inspect()["preparation_evaluations"] == 0
    assert refreshed.execute().identity != first.identity
    assert request.execute().identity == first.identity


def test_frequentist_bootstrap_responses_keep_failed_replicate_slots() -> None:
    from antecedent.cyclic import bootstrap_responses

    system = EquilibriumModel((Equation("x", 0.0, (0.0,), "identity"),), (1.0,))
    rows = [{"values": [value], "clamps": []} for value in [-1.0, -0.5, 0.5, 1.0]]
    inputs = (
        system,
        [{"kind": "intercept", "equation": 0}],
        rows,
        [[], [{"variable": 0, "value": 3.0}]],
        [[0.0]],
    )
    result = bootstrap_responses(*inputs, replicates=6, seed=91).execute()
    answer = result.report["answer"]["result"]
    assert len(answer["parameters"]) == 6
    for values, observational, clamped in zip(
        answer["parameters"], answer["worlds"][0], answer["worlds"][1], strict=True
    ):
        assert values is not None
        assert observational[0]["values"][0] == pytest.approx(values[0])
        assert clamped[0]["values"][0] == 3.0
    assert (
        consume_family_artifact(result.artifact, expected_identity=result.identity).report
        == result.report
    )
    failed = (
        bootstrap_responses(*inputs, replicates=4, seed=91, policy=FitOptions(max_sweeps=1))
        .execute()
        .report["answer"]["result"]
    )
    assert failed["parameters"] == [None] * 4
    assert failed["worlds"][0] == [None] * 4


def test_sampled_gaussian_ci_discovery_keeps_cyclic_interpretation() -> None:
    import random

    from antecedent.cyclic import discover_cyclic

    rng = random.Random(775)
    columns: list[list[float]] = [[], [], [], []]
    for _ in range(2000):
        ua, ub, uc, ud = [rng.gauss(0.0, 1.0) for _ in range(4)]
        a = (ua + 0.4 * ub) / 0.8
        b = 0.5 * a + ub
        values = [a, b, 0.7 * b + uc, ud]
        for column, value in zip(columns, values, strict=True):
            column.append(value)
    result = discover_cyclic(
        columns,
        simple_scm=True,
        sigma_faithful=True,
        independent_equilibrium_replicates=True,
        ci="parcorr",
        alpha=0.001,
    ).execute()
    answer = result.report["answer"]["result"]
    pairs = {(a, b) for a, b, _, _ in answer["edges"]}
    assert (0, 1) in pairs and (1, 2) in pairs
    assert all(3 not in pair for pair in pairs)
    assert answer["ci_source"] == "parcorr"
    assert "inducing path" in answer["interpretation"]
    assert (
        consume_family_artifact(result.artifact, expected_identity=result.identity).report
        == result.report
    )


def test_raw_and_convenience_cyclic_entries_honor_execution_controls() -> None:
    from antecedent import _native

    system = EquilibriumModel((Equation("x", 0.0, (0.0,), "identity"),), (1.0,))
    prepared = system.prepare(
        [[]],
        [[0.0]],
        parameters=[{"kind": "intercept", "equation": 0}],
        rows=[{"values": [x], "clamps": []} for x in [-0.5, 0.5]],
    )
    options = {
        "prior_sd": 1.0,
        "chains": 2,
        "warmup": 8,
        "draws": 8,
        "proposal_scale": 0.1,
        "seed": 82,
    }
    posterior = prepared.posterior(**options)
    assert len(posterior["samples"]) == 16
    with pytest.raises(ValueError, match="(?i)(admission|memory|limit|resource)"):
        prepared.posterior(**options, max_bytes=1)
    with pytest.raises(ValueError, match="(?i)(admission|memory|limit|resource)"):
        _native.execute_cyclic_system(prepared.request_json, max_bytes=1)
    token = _native.CancellationToken()
    token.cancel()
    with pytest.raises(ValueError, match="cancel"):
        prepared.posterior(**options, cancel=token)


def test_multiple_equilibrium_joint_selection_law_abduction_and_artifact(tmp_path: Path) -> None:
    from antecedent.cyclic import (
        PolynomialCausalCandidate,
        PolynomialCausalModel,
        PolynomialExogenousAtom,
        PolynomialObservation,
        PolynomialWorld,
    )

    def region(lower: float, upper: float) -> list[dict[str, float]]:
        return [{"lower": lower, "upper": upper}]

    system = PolynomialSystem(
        ("x",),
        (
            {
                "id": "balance",
                "terms": [
                    {"coefficient": 1.0, "powers": [3]},
                    {"coefficient": -1.0, "powers": [1]},
                ],
            },
        ),
        tuple(region(-3.0, 3.0)),
    )
    worlds = (
        PolynomialWorld("factual"),
        PolynomialWorld(
            "counterfactual",
            replacements=(
                {
                    "equation": "balance",
                    "replacement": {
                        "id": "balance",
                        "terms": [
                            {"coefficient": 1.0, "powers": [2]},
                            {"coefficient": -4.0, "powers": [0]},
                        ],
                    },
                    "clear_disturbance": True,
                },
            ),
        ),
    )

    def model(flipped: bool) -> PolynomialCausalModel:
        atoms = []
        for positive in [True, False]:
            selected_positive = positive != flipped
            atoms.append(
                PolynomialExogenousAtom(
                    "positive" if positive else "negative",
                    1.0,
                    (0.0,),
                    {
                        "factual": region(0.5, 1.5) if positive else region(-1.5, -0.5),
                        "counterfactual": region(1.5, 2.5)
                        if selected_positive
                        else region(-2.5, -1.5),
                    },
                    measurements={
                        "factual": {
                            "positive_sensor": 4.0 if positive else 1.0,
                            "negative_sensor": 1.0 if positive else 4.0,
                        }
                    },
                )
            )
        return PolynomialCausalModel(
            (PolynomialCausalCandidate("declared", system, tuple(atoms)),), worlds
        )

    answers = []
    for flipped in [False, True]:
        result = (
            model(flipped)
            .prepare(PolynomialObservation("factual", (None,), "positive_sensor"))
            .execute()
        )
        answer = result.report["answer"]["result"]
        assert answer["positive_evidence_certified"]
        assert not answer["impossible_evidence"]
        assert (
            answer["atoms"][0]["posterior_mass"]["lower"]
            <= 0.8
            <= answer["atoms"][0]["posterior_mass"]["upper"]
        )
        expected = 0.0
        for atom in answer["atoms"]:
            state = atom["states"]["counterfactual"]
            assert state["status"] == "selected"
            root = state["enclosure"][0]
            mass = atom["posterior_mass"]
            expected += (
                (root["lower"] + root["upper"]) * 0.5 * (mass["lower"] + mass["upper"]) * 0.5
            )
        assert expected == pytest.approx(-1.2 if flipped else 1.2, abs=1e-7)
        assert (
            consume_family_artifact(result.artifact, expected_identity=result.identity).report
            == result.report
        )
        answers.append(result)
    from antecedent import decision
    from antecedent.joint_distribution import ScientificQuantity

    impossible = model(False).prepare(PolynomialObservation("factual", tuple(region(0.2, 0.3))))
    impossible_evidence = impossible.evidence(
        {"id": "cubic", "population": "unit", "variables": [["x", "unit"]]},
        [{"kind": "polynomial_world", "world": "counterfactual", "variable": 0}],
    )
    q = ScientificQuantity.from_wire(impossible_evidence.report["columns"][0]["quantity"])
    c = decision.Contract(
        (
            decision.Action("act", inputs=(q,), utility=decision.x(0)),
            decision.Action("other", inputs=(q,), utility=decision.x(0) * 0),
        ),
        utility_units="unit",
        target_population="unit",
        criterion=decision.Criterion.maximin_over_structures(),
        structural_policy="maximin",
    )
    with pytest.raises(ValueError, match="positive"):
        impossible_evidence.decide_enclosures(c)
    path = tmp_path / "joint-selection.ant"
    path.write_bytes(answers[0].artifact)
    program = (
        "import json,pathlib;from antecedent.cyclic import consume_family_artifact;"
        + f"r=consume_family_artifact(pathlib.Path({str(path)!r}).read_bytes(),expected_identity={answers[0].identity!r});print(json.dumps(r.report,sort_keys=True))"
    )
    replay = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(replay.stdout) == answers[0].report


def test_compositional_feedback_gradient_fit_counterfactual_and_artifact(tmp_path):
    import math

    from antecedent.cyclic import Equation, EquilibriumModel, consume_family_artifact

    expression = {
        "op": "multiply",
        "factors": [
            {"op": "parameter", "parameter": 0},
            {"op": "tanh", "argument": {"op": "variable", "variable": 1}},
        ],
    }
    template = EquilibriumModel(
        (
            Equation.compositional("response", 2, expression, parameters=(0.0,)),
            Equation("return", 0.0, (0.2, 0.0)),
        ),
        (1.0, 0.0, 0.0, 1.0),
    )
    rows = [
        {"values": [0.7 * math.tanh(x) + noise, x], "clamps": [{"variable": 1, "value": x}]}
        for x in (-2.0, -1.0, 0.5, 1.0, 2.0)
        for noise in (-0.1, 0.1)
    ]
    prepared = template.fit_gradient(
        [{"kind": "expression_parameter", "equation": 0, "parameter": 0}], rows
    )
    result = prepared.execute()
    fit = result.report["answer"]["result"]
    assert fit["converged"]
    assert abs(fit["model"]["equations"][0]["parameters"][0] - 0.7) < 1e-6
    assert (
        consume_family_artifact(result.artifact, expected_identity=result.identity).report
        == result.report
    )
    fitted = EquilibriumModel(
        (
            Equation.compositional(
                "response",
                2,
                expression,
                parameters=(fit["model"]["equations"][0]["parameters"][0],),
            ),
            template.equations[1],
        ),
        template.noise_covariance,
    )
    answer = fitted.prepare(
        [[{"variable": 1, "value": 1.0}]], [[0.0, 0.0]], factual=[0.3, -0.2]
    ).execute()
    assert answer.report["worlds"][0][0]["globally_unique"]


def test_typed_identified_cyclic_functional_evaluation_and_replay():
    from antecedent.cyclic import CyclicGraph, consume_family_artifact

    source = CyclicGraph(3, ((0, 1), (1, 2), (2, 1))).identify(
        [2], [0], io_scm=True, scc_product_density=True, continuous=False
    )
    functional = source.identified_functional()
    weights = []
    for cell in range(8):
        t, a, b = bool(cell & 1), bool(cell & 2), bool(cell & 4)
        pt, pa, pb = (0.6 if t else 0.4), (0.8 if t else 0.2), (0.6 if a else 0.1)
        weights.append(pt * (pa if a else 1 - pa) * (pb if b else 1 - pb))
    result = functional.evaluate(
        [[0.0, 1.0]] * 3,
        weights,
        [[(0, 0.0), (2, 1.0)], [(0, 1.0), (2, 1.0)], [(0, 2.0), (2, 1.0)]],
    ).execute()
    values = result.report["answer"]["result"]["values"]
    assert abs(values[0]["value"] - 0.2) < 1e-12
    assert abs(values[1]["value"] - 0.5) < 1e-12
    assert values[2]["value"] is None and values[2]["failure"]
    assert (
        consume_family_artifact(result.artifact, expected_identity=result.identity).report
        == result.report
    )


def test_local_callbacks_scc_counterfactual_budget_and_portability():
    from antecedent.cyclic import LocalCyclicModel, LocalMechanism

    local = LocalCyclicModel(
        (
            LocalMechanism("a", (0, 1), lambda x: (1.0 + 0.2 * x[0] + 0.4 * x[1], [0.2, 0.4])),
            LocalMechanism("b", (0,), lambda x: (0.5 * x[0], [0.5])),
        )
    )
    answer = local.execute([0.0, 0.0], [])
    assert answer["result"]["values"][0] == pytest.approx(1.0 / 0.6)
    assert not answer["result"]["globally_unique"]
    assert not answer["inspection"]["portable"]
    cf = local.execute([0.0, 0.0], [{"variable": 0, "value": 2.0}], factual=[0.3, 0.7])
    assert cf["counterfactual"]["values"][1] == pytest.approx(1.55)
    with pytest.raises(ValueError, match="portable"):
        local.export()
    with pytest.raises(ValueError, match="budget"):
        local.execute([0.0, 0.0], [], max_evaluations=1)
    with pytest.raises(ValueError, match="derivatives"):
        LocalCyclicModel((LocalMechanism("bad", (0,), lambda x: (x[0], [])),)).execute([0.0], [])


def test_native_cyclic_evidence_decision_and_fresh_process_replay(tmp_path):
    from antecedent import decision
    from antecedent.cyclic import consume_evidence
    from antecedent.errors import CausalSerializationError
    from antecedent.joint_distribution import ScientificQuantity

    prepared = OrnsteinUhlenbeck((-2.0,), (3.0,), (4.0,)).stationary()
    schema = {"id": "physical-system", "population": "system", "variables": [["position", "meter"]]}
    evidence = prepared.evidence(schema, [{"kind": "ou_mean", "time": None, "variable": 0}])
    assert prepared.inspect()["preparation_evaluations"] == 1
    assert evidence.report["columns"][0]["samples"] == [3.0]
    assert evidence.report["columns"][0]["port"] == "mean"
    quantity = ScientificQuantity.from_wire(evidence.report["columns"][0]["quantity"])
    contract = decision.Contract(
        (
            decision.Action("keep", inputs=(quantity,), utility=decision.x(0)),
            decision.Action("constant", inputs=(quantity,), utility=decision.x(0) * 0 + 2),
        ),
        utility_units="meter",
        criterion=decision.Criterion.expected_utility(),
        target_population="system",
    )
    result = evidence.decide_means(contract)
    assert result["scope"] == "specified_or_fitted_model_conditional"
    assert result["evidence_identity"] == evidence.identity
    artifact = evidence.export()
    assert consume_evidence(artifact, expected_identity=evidence.identity).report == evidence.report
    path = tmp_path / "evidence.ant"
    path.write_bytes(artifact)
    code = (
        "import pathlib,json;from antecedent.cyclic import consume_evidence;"
        f"e=consume_evidence(pathlib.Path({str(path)!r}).read_bytes(),expected_identity={evidence.identity!r});"
        "print(json.dumps(e.report,sort_keys=True))"
    )
    process = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert json.loads(process.stdout) == evidence.report
    changed = prepared.evidence(
        {**schema, "population": "other-system"}, [{"kind": "ou_mean", "time": None, "variable": 0}]
    )
    assert changed.identity != evidence.identity
    with pytest.raises(CausalSerializationError, match="identity"):
        consume_evidence(artifact, expected_identity=changed.identity)
    with pytest.raises(ValueError, match="port"):
        evidence.joint_law()
    with pytest.raises(ValueError, match="schema"):
        prepared.evidence(
            {**schema, "variables": [["extra", "meter"], ["invalid", "meter"]]},
            [{"kind": "ou_mean", "time": None, "variable": 1}],
        )


def test_cyclic_evidence_retains_posterior_bootstrap_and_algebraic_meanings():
    from antecedent.cyclic import SamplingPolicy, bootstrap_responses, posterior_responses

    rows = [{"values": [x, 0.3 * x], "clamps": [{"variable": 0, "value": x}]} for x in (-1.0, 1.0)]
    schema = {"id": "feedback", "population": "model", "variables": [["a", "unit"], ["b", "unit"]]}
    selectors = [{"kind": "response", "world": 0, "noise_row": 0, "variable": 1}]
    posterior = posterior_responses(
        model(),
        [{"kind": "weight", "equation": 1, "parent": 0}],
        rows,
        [[{"variable": 0, "value": 1.0}]],
        [[0.0, 0.0]],
        prior_sd=0.2,
        sampling=SamplingPolicy(chains=2, warmup=8, draws=8, proposal_scale=0.01, seed=2),
    )
    evidence = posterior.evidence(schema, selectors)
    assert evidence.report["columns"][0]["port"] == "parameter_posterior"
    assert evidence.report["columns"][0]["uncertainty_axes"] == ["parameter"]
    law = evidence.joint_law()
    assert law.to_dict()["identity"]["semantic"] == "causal_functional_posterior"
    from antecedent import decision
    from antecedent.joint_distribution import ScientificQuantity

    q = ScientificQuantity.from_wire(evidence.report["columns"][0]["quantity"])
    contract = decision.Contract(
        (
            decision.Action("nonlinear", inputs=(q,), utility=decision.x(0) * decision.x(0)),
            decision.Action("constant", inputs=(q,), utility=decision.x(0) * 0 - 1),
        ),
        utility_units="unit_squared",
        target_population="model",
        criterion=decision.Criterion.expected_utility(),
    )
    selected = contract.evaluate(law)
    assert selected.selected == ("nonlinear",)
    assert (
        decision.replay(selected.export(), contract=contract, source=law).selected
        == selected.selected
    )

    bootstrap = bootstrap_responses(
        model(),
        [{"kind": "weight", "equation": 1, "parent": 0}],
        rows,
        [[{"variable": 0, "value": 1.0}]],
        [[0.0, 0.0]],
        replicates=2,
    )
    resampling = bootstrap.evidence(schema, selectors)
    assert resampling.report["columns"][0]["port"] == "sampling_law"
    with pytest.raises(ValueError, match="port"):
        resampling.joint_law()


def test_equilibrium_polynomial_bridge_and_robust_solution_decisions():
    from antecedent import decision
    from antecedent.joint_distribution import ScientificQuantity

    expression = {
        "op": "add",
        "terms": [
            {
                "op": "multiply",
                "factors": [{"op": "constant", "value": 2.0}, {"op": "variable", "variable": 0}],
            },
            {
                "op": "multiply",
                "factors": [{"op": "constant", "value": -1.0}]
                + [{"op": "variable", "variable": 0}] * 3,
            },
        ],
    }
    model = EquilibriumModel((Equation.compositional("cubic", 1, expression),), (1.0,))
    schema = {"id": "cubic", "population": "physical", "variables": [["x", "unit"]]}
    prepared = model.isolate([{"lower": -2.0, "upper": 2.0}], [0.0])
    answer = prepared.execute()
    assert len(answer.report["answer"]["result"]["roots"]) == 3
    evidence = prepared.evidence(schema, [{"kind": "solution_set", "variable": 0}])
    quantity = ScientificQuantity.from_wire(evidence.report["columns"][0]["quantity"])
    contract = decision.Contract(
        (
            decision.Action("root_action", inputs=(quantity,), utility=decision.x(0)),
            decision.Action("constant", inputs=(quantity,), utility=decision.x(0) * 0 - 1.5),
        ),
        utility_units="unit",
        criterion=decision.Criterion.maximin_over_structures(),
        target_population="physical",
        structural_policy="maximin",
    )
    result = evidence.decide_enclosures(contract)
    assert result["scope"] == "conditional_model_solution_outer_enclosure"
    assert "root_action" in result["verdict"]
    assert not result["selection_probabilities_inferred"]
    unresolved = model.isolate([{"lower": -2.0, "upper": 2.0}], [0.0], max_boxes=1)
    retained = unresolved.evidence(schema, [{"kind": "solution_set", "variable": 0}])
    assert retained.report["columns"][0]["diagnostics"]["unresolved_regions"] > 0
    other = ScientificQuantity.from_wire(retained.report["columns"][0]["quantity"])
    other_contract = decision.Contract(
        (
            decision.Action("root_action", inputs=(other,), utility=decision.x(0)),
            decision.Action("constant", inputs=(other,), utility=decision.x(0) * 0 - 1.5),
        ),
        utility_units="unit",
        criterion=decision.Criterion.maximin_over_structures(),
        target_population="physical",
        structural_policy="maximin",
    )
    assert "constant" in retained.decide_enclosures(other_contract)["verdict"]


def test_ou_equation_replacements_preserve_world_failures_and_artifact():
    from antecedent.cyclic import OuMechanisms

    model = OuMechanisms(("a", "b"), (-1.0, 0.3, -0.2, -1.0), (1.0, 0.7), (0.4, 0.0, 0.0, 0.3))
    changed = {
        "equation": "a",
        "drift": [-1.0, 0.0],
        "forcing": 0.1,
        "diffusion_factor": [0.001, 0.0],
    }
    unstable = {**changed, "drift": [1.0, 0.0]}
    prepared = model.prepare([[], [changed], [unstable]])
    answer = prepared.execute()
    laws = answer.report["answer"]["result"]["laws"]
    assert laws[1]["mean"][0] == pytest.approx(0.1)
    assert laws[1]["mean"][1] == pytest.approx(0.68)
    assert "failure" in laws[2]
    assert (
        consume_family_artifact(answer.artifact, expected_identity=answer.identity).report
        == answer.report
    )
    schema = {"id": "sde", "population": "physical", "variables": [["a", "unit"], ["b", "unit"]]}
    evidence = prepared.evidence(
        schema, [{"kind": "ou_regime", "world": 2, "variable": 0, "other": None}]
    )
    assert evidence.report["columns"][0]["samples"] == [None]
    assert evidence.report["columns"][0]["diagnostics"]["failure"]


def test_calculus_adjustment_functional_and_bounded_status_replay():
    from antecedent.cyclic import CyclicGraph

    graph = CyclicGraph(2, ((0, 1),))
    source = graph.calculus(
        "exchange", [0], [1], [], [], io_scm=True, standard_measurable_spaces=True
    )
    assert source.execute().report["answer"]["result"]["status"] == "verified_rule"
    result = (
        source.identified_functional()
        .evaluate(
            [[0.0, 1.0]] * 2, [0.4, 0.1, 0.1, 0.4], [[(0, 0.0), (1, 1.0)], [(0, 1.0), (1, 1.0)]]
        )
        .execute()
    )
    assert [x["value"] for x in result.report["answer"]["result"]["values"]] == pytest.approx(
        [0.2, 0.8]
    )
    assert (
        consume_family_artifact(result.artifact, expected_identity=result.identity).report
        == result.report
    )
    adjusted = graph.adjust(
        [1], [0], [], io_scm=True, standard_measurable_spaces=True, continuous=False
    )
    answer = (
        adjusted.identified_functional()
        .evaluate([[0.0, 1.0]] * 2, [0.4, 0.1, 0.1, 0.4], [[(0, 1.0), (1, 1.0)]])
        .execute()
    )
    assert answer.report["answer"]["result"]["values"][0]["value"] == pytest.approx(0.8)
    exhausted = graph.calculus(
        "exchange", [0], [1], [], [], io_scm=True, standard_measurable_spaces=True, max_checks=0
    ).execute()
    assert exhausted.report["answer"]["result"]["status"] == "budget_exhausted"
    failed = graph.calculus(
        "action", [0], [1], [], [], io_scm=True, standard_measurable_spaces=True
    ).execute()
    assert failed.report["answer"]["result"]["status"] == "not_licensed_by_criterion"


def test_adam_public_replay_and_policy_data_invalidation():
    from dataclasses import replace

    from antecedent.cyclic import AdamPolicy

    m = EquilibriumModel((Equation("one", 0.0, (0.0,), "relu"),), (1.0,))
    training = [{"values": [x], "clamps": []} for x in [-1.0, 0.0, 1.0, 2.0]]
    p = AdamPolicy(max_epochs=25, minimum_epochs=3, patience=3)
    prepared = m.fit_adam([{"kind": "intercept", "equation": 0}], training, training, policy=p)
    result = prepared.execute()
    fit = result.report["answer"]["result"]
    assert fit["epochs"] <= 25 and fit["nonsmooth_convention"]
    assert (
        consume_family_artifact(result.artifact, expected_identity=result.identity).report
        == result.report
    )
    assert (
        m.fit_adam(
            [{"kind": "intercept", "equation": 0}], training, training, policy=replace(p, seed=1)
        )
        .execute()
        .identity
        != result.identity
    )
    changed = [dict(row, values=[row["values"][0] + 0.1]) for row in training]
    assert (
        m.fit_adam([{"kind": "intercept", "equation": 0}], changed, training, policy=p)
        .execute()
        .identity
        != result.identity
    )


def test_whole_family_scientific_identity_invalidation_and_fresh_consumers(tmp_path):
    from copy import deepcopy

    from antecedent.cyclic import AdamPolicy, CyclicGraph, OuMechanisms, PreparedCyclicFamily

    scalar = EquilibriumModel((Equation("x", 0.0, (0.0,), "identity"),), (1.0,))
    row = [{"values": [0.0], "clamps": []}]
    requests = [
        scalar.isolate([{"lower": -1.0, "upper": 1.0}], [0.0], clamps=[]),
        scalar.fit_adam(
            [{"kind": "intercept", "equation": 0}],
            row,
            row,
            policy=AdamPolicy(max_epochs=2, minimum_epochs=1, patience=1),
        ),
        OuMechanisms(("x",), (-1.0,), (0.0,), (1.0,)).prepare([[]]),
        CyclicGraph(2, ((0, 1),)).calculus(
            "exchange", [0], [1], [], [], io_scm=True, standard_measurable_spaces=True
        ),
        OrnsteinUhlenbeck((-1.0,), (0.0,), (1.0,)).transient([0.0], [1.0], [0.0, 1.0]),
        OdeSystem((Equation("drift", 0.0, (0.0,), "identity"),), (1.0,)).prepare([1.0], [0.0, 1.0]),
    ]
    pairs, expected = [], []
    for i, prepared in enumerate(requests):
        result = prepared.execute()
        # Every top-level numerical policy change has a separately bound target;
        # retaining an old prepared result never reinterprets it as this new task.
        request = json.loads(prepared.request_json)
        changed = deepcopy(request)
        if changed["family"] == "equilibrium_roots":
            changed["domain"][0]["upper"] = 2.0
        elif changed["family"] == "adam_fit":
            changed["policy"]["seed"] += 1
        elif changed["family"] == "ou_mechanisms":
            changed["model"]["forcing"][0] = 0.2
        elif changed["family"] == "graph":
            changed["request"]["max_checks"] += 1
        elif changed["family"] == "ou_transient":
            changed["initial_covariance"][0] = 2.0
        else:
            changed["initial"][0] = 2.0
        fresh = PreparedCyclicFamily(json.dumps(changed)).execute()
        assert fresh.identity != result.identity
        assert prepared.execute().report == result.report
        with pytest.raises(Exception, match="identity|mismatch"):
            consume_family_artifact(fresh.artifact, expected_identity=result.identity)
        path = tmp_path / f"inventory-{i}.ant"
        path.write_bytes(result.artifact)
        pairs.append((str(path), result.identity))
        expected.append(result.report)
    script = (
        "import json,pathlib;from antecedent.cyclic import consume_family_artifact;"
        f"pairs={pairs!r};print(json.dumps([consume_family_artifact(pathlib.Path(p).read_bytes(),expected_identity=i).report for p,i in pairs],sort_keys=True))"
    )
    process = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert json.loads(process.stdout) == expected


def test_numerical_roots_never_acquire_solution_enclosure_decision_contract():
    from antecedent.cyclic import action_worlds

    m = EquilibriumModel((Equation("multi", 0.0, (2.0,), "tanh"),), (1.0,))
    p = action_worlds(
        m,
        [{"id": "numerical", "actions": []}],
        [[0.0]],
        policy=SolvePolicy(require_contraction=False),
    )
    e = p.evidence(
        {"id": "numeric", "population": "unit", "variables": [["x", "unit"]]},
        [{"kind": "action", "world": 0, "noise_row": 0, "variable": 0, "counterfactual": False}],
    )
    assert e.report["columns"][0]["numerical_enclosures"] == []
    assert e.report["columns"][0]["diagnostics"]["globally_unique"] is False
    with pytest.raises(ValueError, match="admission|resource|limit|budget"):
        e.export(max_bytes=1)


def test_continuous_identification_gaussian_fitting_evidence_and_fresh_replay(tmp_path):
    from antecedent.cyclic import CyclicGraph

    source = CyclicGraph(3, ((0, 1), (1, 2)), ((0, 2),)).identify(
        [2], [0], io_scm=True, scc_product_density=True, continuous=True
    )
    functional = source.identified_functional()
    rows = []
    for mask in range(16):
        u, e, v, w = [1.0 if mask & (1 << i) else -1.0 for i in range(4)]
        x = u + e
        m = 0.5 * x + v
        y = 0.8 * m + 0.7 * u + w
        rows.append([x, m, y])
    prepared = functional.evaluate_gaussian(
        {"kind": "fit", "rows": rows, "independent_observational_replicates": True},
        [[(0, 1.0), (2, 0.0)], [(0, 1.0)]],
    )
    result = prepared.execute()
    answer = result.report["answer"]["result"]
    truth = math.exp(-0.5 * 0.4**2 / 2.13) / math.sqrt(2 * math.pi * 2.13)
    assert answer["fitted_rows"] == 16
    assert answer["values"][0]["value"] == pytest.approx(truth, abs=1e-12)
    assert answer["values"][1]["value"] is None
    evidence = prepared.evidence(
        {
            "id": "continuous",
            "population": "population",
            "variables": [["x", "unit"], ["m", "unit"], ["y", "unit"]],
        },
        [{"kind": "functional", "assignment": 0}],
    )
    assert evidence.report["columns"][0]["port"] == "likelihood"
    assert evidence.report["columns"][0]["quantity"]["variable_name"] == "y"
    assert evidence.report["columns"][0]["quantity"]["units"] == "1/(unit)"
    path = tmp_path / "continuous.ant"
    path.write_bytes(result.artifact)
    code = (
        "import pathlib,json;from antecedent.cyclic import consume_family_artifact;"
        f"r=consume_family_artifact(pathlib.Path({str(path)!r}).read_bytes(),expected_identity={result.identity!r});print(json.dumps(r.report,sort_keys=True))"
    )
    replay = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert json.loads(replay.stdout) == result.report
    with pytest.raises(ValueError, match="admission|limit|resource"):
        prepared.execute(max_bytes=1)
    with pytest.raises(ValueError, match="IID|replicates"):
        functional.evaluate_gaussian(
            {"kind": "fit", "rows": rows, "independent_observational_replicates": False},
            [[(0, 1.0), (2, 0.0)]],
        ).execute()


def test_structure_and_discovery_semantic_seeds_bind_replay():
    from antecedent.cyclic import discover_cyclic, learn_structure

    m = EquilibriumModel(
        (Equation("x", 0.0, (0.0, 0.0), "identity"), Equation("y", 0.0, (0.0, 0.0), "identity")),
        (1.0, 0.0, 0.0, 1.0),
    )
    rows = [
        {"values": [x, 0.6 * x + e], "clamps": [{"variable": 0, "value": x}]}
        for x in [-2.0, -1.0, 1.0, 2.0]
        for e in [-0.1, 0.1]
    ]
    kwargs = {
        "policy": FitOptions(tolerance=0.001, max_sweeps=200, l1_penalty=0.1),
        "bootstrap_replicates": 2,
        "design": "randomized_within_targets",
    }
    prepared = learn_structure(m, rows, seed=71, **kwargs)
    result = prepared.execute()
    assert any(value is not None for value in result.report["answer"]["result"]["bootstrap"])
    assert (
        consume_family_artifact(result.artifact, expected_identity=result.identity).report
        == result.report
    )
    assert learn_structure(m, rows, seed=72, **kwargs).execute().identity != result.identity
    ci = {"simple_scm": True, "sigma_faithful": True, "independent_equilibrium_replicates": True}
    columns = [list(map(float, range(8))), [0.0, 0.5, 0.1, 2.0, 1.0, 0.5, 2.0, 1.5]]
    graph = discover_cyclic(columns, seed=8, **ci).execute()
    assert discover_cyclic(columns, seed=9, **ci).execute().identity != graph.identity
    assert (
        consume_family_artifact(graph.artifact, expected_identity=graph.identity).report
        == graph.report
    )


def test_cyclic_logical_retention_and_projection_resource_limits():
    from antecedent.cyclic import consume_evidence
    from antecedent.errors import CausalSerializationError

    prepared = OrnsteinUhlenbeck((-2.0,), (3.0,), (4.0,)).stationary()
    result = prepared.execute()
    assert prepared.inspect()["retained_bytes"] > len(result.artifact) * 4
    schema = {"id": "physical", "population": "system", "variables": [["position", "meter"]]}
    selector = {"kind": "ou_mean", "time": None, "variable": 0}
    with pytest.raises(ValueError, match="schema"):
        prepared.evidence({**schema, "id": "x" * 257}, [selector])
    evidence = prepared.evidence(schema, [selector] * 128)
    artifact = evidence.export()
    # Repeated columns compress well; wire size is not the retained heap budget.
    with pytest.raises(ValueError, match="admission|resource|limit|budget"):
        prepared.evidence(schema, [selector] * 128, max_bytes=len(artifact) * 4)
    with pytest.raises(CausalSerializationError, match="admission|resource|limit|budget"):
        consume_evidence(artifact, expected_identity=evidence.identity, max_bytes=len(artifact) * 4)
