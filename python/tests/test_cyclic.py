# SPDX-License-Identifier: MIT OR Apache-2.0
"""Independent cyclic workflow and fresh-process artifact consumers."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from antecedent.cyclic import (
    CyclicRefusal,
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
from antecedent.errors import (
    CausalCancelledError,
    CausalResourceError,
    CausalValueError,
    StructuredRefusal,
)
from antecedent.execution import Execution, ResourceLimits
from antecedent.outcomes import require_success


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
    result = require_success(prepared.execute())
    assert result.to_dict()["worlds"][1][0]["values"][1] == pytest.approx(math.tanh(1.0) + 0.2)
    assert result.to_dict()["worlds"][0][0]["globally_unique"] is True
    assert result.to_dict()["claim"] == "specified_or_fitted_model_conditional"
    assert result.to_dict()["calibration"] == "unmeasured"
    loaded = require_success(
        consume_artifact(bytes(result.export()), expected_identity=result.identity)
    )
    assert loaded.to_dict() == result.to_dict()
    artifact = tmp_path / "cyclic.ant"
    artifact.write_bytes(bytes(result.export()))
    script = f"from antecedent.outcomes import require_success\nimport json, pathlib\nfrom antecedent.cyclic import consume_artifact\nr = require_success(consume_artifact(pathlib.Path({str(artifact)!r}).read_bytes(), expected_identity={result.identity!r}))\nprint(json.dumps(r.to_dict(), sort_keys=True))\n"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == result.to_dict()
    with pytest.raises(Exception, match="identity|mismatch"):
        require_success(consume_artifact(bytes(result.export()), expected_identity="0" * 64))


def test_linear_limit_joint_clamp_and_retained_fit() -> None:
    linear = EquilibriumModel(
        (Equation("a", 0.0, (0.0, 0.4), "identity"), Equation("b", 0.0, (0.5, 0.0), "identity")),
        (1.0, 0.0, 0.0, 1.0),
    )
    result = require_success(linear.prepare([[]], [[1.0, 0.0]]).execute())
    assert result.to_dict()["worlds"][0][0]["values"] == pytest.approx([1.25, 0.625])
    fitted = require_success(
        linear.prepare(
            [[{"variable": 0, "value": 2.0}]],
            [[0.0, 0.0]],
            parameters=[{"kind": "weight", "equation": 1, "parent": 0}],
            rows=[
                {"values": [1.0, 0.7], "clamps": [{"variable": 0, "value": 1.0}]},
                {"values": [-1.0, -0.7], "clamps": [{"variable": 0, "value": -1.0}]},
            ],
            fit_options=FitOptions(),
        ).execute()
    )
    assert fitted.to_dict()["fit"]["converged"]
    assert fitted.to_dict()["worlds"][0][0]["values"][1] == pytest.approx(1.4, abs=1e-4)
    assert (
        require_success(
            consume_artifact(bytes(fitted.export()), expected_identity=fitted.identity)
        ).to_dict()
        == fitted.to_dict()
    )


def test_uncertified_roots_are_numerical_only() -> None:
    multiple = EquilibriumModel(
        (Equation("a", 0.0, (0.0, 2.0)), Equation("b", 0.0, (2.0, 0.0))), (1.0, 0.0, 0.0, 1.0)
    )
    with pytest.raises(Exception, match="contractive"):
        require_success(multiple.prepare([[]], [[0.0, 0.0]]).execute())
    numerical = require_success(
        multiple.prepare(
            [[]], [[0.0, 0.0]], initial=[1.0, 1.0], policy=SolvePolicy(require_contraction=False)
        ).execute()
    )
    # With the saturating link the equations are a = tanh(2 b), b = tanh(2 a): the symmetric
    # nonzero roots solve x = tanh(2 x), found here by iterating from the same start.
    root = 1.0
    for _ in range(200):
        root = math.tanh(2.0 * root)
    assert numerical.to_dict()["worlds"][0][0]["values"] == pytest.approx([root, root], abs=1e-9)
    assert numerical.to_dict()["worlds"][0][0]["globally_unique"] is False


def test_continuous_time_families_and_replay() -> None:
    ode = OdeSystem(
        (Equation("a", 0.0, (0.0, -1.0), "identity"), Equation("b", 0.0, (1.0, 0.0), "identity")),
        (0.0, 0.0),
    )
    path = require_success(ode.prepare([1.0, 0.0], [0.0, 1.0]).execute())
    assert path.to_dict()["answer"]["result"]["states"][1] == pytest.approx(
        [math.cos(1.0), math.sin(1.0)], abs=1e-6
    )
    assert (
        require_success(
            consume_family_artifact(bytes(path.export()), expected_identity=path.identity)
        ).to_dict()
        == path.to_dict()
    )
    ou = OrnsteinUhlenbeck((-2.0,), (3.0,), (4.0,))
    stationary = require_success(ou.stationary().execute())
    assert stationary.to_dict()["answer"]["result"] == pytest.approx([1.0])
    transient = require_success(ou.transient([0.0], [0.0], [0.0, 1.0]).execute())
    assert transient.to_dict()["answer"]["result"]["means"][1][0] == pytest.approx(
        3.0 * (1.0 - math.exp(-2.0)), abs=1e-6
    )
    with pytest.raises(ValueError, match="stability"):
        require_success(OrnsteinUhlenbeck((2.0,), (0.0,), (1.0,)).stationary().execute())


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
    result = require_success(system.prepare().execute())
    assert len(result.to_dict()["answer"]["result"]["roots"]) == 3
    assert result.to_dict()["answer"]["result"]["unresolved"] == []
    assert (
        require_success(
            consume_family_artifact(bytes(result.export()), expected_identity=result.identity)
        ).to_dict()
        == result.to_dict()
    )
    short = require_success(system.prepare(max_boxes=1).execute())
    assert short.to_dict()["answer"]["result"]["unresolved"]
    replaced = require_success(system.replace_equation("cubic", 0, 0.5).prepare().execute())
    assert len(replaced.to_dict()["answer"]["result"]["roots"]) == 1


def test_prepared_fit_retarget_has_zero_refits_and_refresh_changes_identity() -> None:
    linear = EquilibriumModel(
        (Equation("a", 0.0, (0.0, 0.4), "identity"), Equation("b", 0.0, (0.5, 0.0), "identity")),
        (1.0, 0.0, 0.0, 1.0),
    )
    rows = [{"values": [x, 0.7 * x], "clamps": [{"variable": 0, "value": x}]} for x in [-1.0, 1.0]]
    prepared = linear.prepare(
        [[]], [[0.0, 0.0]], parameters=[{"kind": "weight", "equation": 1, "parent": 0}], rows=rows
    )
    assert prepared.inspect()["fits_performed"] == 0
    original = require_success(prepared.execute())
    assert prepared.inspect()["fits_performed"] == 1
    retargeted = prepared.reconfigure([[{"variable": 0, "value": 2.0}]], [[0.0, 0.0]])
    assert retargeted.inspect()["fits_performed"] == 0
    result = require_success(retargeted.execute())
    assert result.to_dict()["fit"] == original.to_dict()["fit"]
    assert result.to_dict()["worlds"][0][0]["values"][1] == pytest.approx(1.4, abs=1e-4)
    assert (
        require_success(
            consume_artifact(bytes(result.export()), expected_identity=result.identity)
        ).to_dict()
        == result.to_dict()
    )
    refreshed = require_success(
        prepared.rebind(
            [{"values": [x, 0.6 * x], "clamps": [{"variable": 0, "value": x}]} for x in [-1.0, 1.0]]
        )
    )
    assert refreshed.inspect()["fits_performed"] == 0
    require_success(refreshed.execute())
    assert refreshed.inspect()["fits_performed"] == 1
    assert refreshed.inspect()["identity"] != prepared.inspect()["identity"]


def test_graph_identification_fci_bipartite_and_domain_invariance_replay() -> None:
    from antecedent.cyclic import BipartiteCausalGraph, CyclicGraph

    graph = CyclicGraph(3, ((0, 1), (1, 2), (2, 1)))
    identified = require_success(
        graph.identify([2], [0], io_scm=True, scc_product_density=True).execute()
    )
    assert identified.to_dict()["answer"]["result"]["status"] == "identified"
    assert identified.to_dict()["answer"]["result"]["expression"]
    assert (
        require_success(
            consume_family_artifact(
                bytes(identified.export()), expected_identity=identified.identity
            )
        ).to_dict()
        == identified.to_dict()
    )
    failed = require_success(
        graph.identify([2], [1], io_scm=True, scc_product_density=True).execute()
    )
    assert failed.to_dict()["answer"]["result"]["status"] == "not_identified_by_procedure"
    discovered = require_success(
        graph.oracle_discovery(simple_scm=True, sigma_faithful=True).execute()
    )
    assert discovered.to_dict()["answer"]["result"]["ci_source"] == "sigma_oracle"
    assert "inducing path" in discovered.to_dict()["answer"]["result"]["interpretation"]
    bipartite = BipartiteCausalGraph(3, (2,), ((0, 2), (0, 1)))
    determined = require_success(bipartite.b_separation(0, 1, [2]).execute())
    assert determined.to_dict()["answer"]["result"]["separated"]
    invariant = require_success(bipartite.domain_invariance([1], [0], []).execute())
    assert invariant.to_dict()["answer"]["result"]["invariant"]
    changed = require_success(bipartite.domain_invariance([1], [1], []).execute())
    assert not changed.to_dict()["answer"]["result"]["invariant"]


def test_aligned_parameter_and_partial_counterfactual_posterior_artifacts() -> None:
    from antecedent.cyclic import SamplingPolicy, partial_counterfactual, posterior_responses

    linear = EquilibriumModel(
        (Equation("a", 0.0, (0.0, 0.4), "identity"), Equation("b", 0.0, (0.5, 0.0), "identity")),
        (1.0, 0.0, 0.0, 1.0),
    )
    sampling = SamplingPolicy(chains=2, warmup=16, draws=16, proposal_scale=0.2)
    worlds = [[], [{"variable": 0, "value": 2.0}]]
    partial = require_success(
        partial_counterfactual(
            linear,
            [1.0, None],
            worlds,
            measurement_variances=[0.0, 0.0],
            initial=[1.0, 0.8],
            sampling=sampling,
            execution=Execution(seed=91),
        ).execute()
    )
    result = partial.to_dict()["answer"]["result"]
    assert not result["abduction"]["diagnostics_pass"]
    for factual, counterfactual in zip(
        result["abduction"]["states"], result["worlds"][1], strict=True
    ):
        assert counterfactual["values"][1] == pytest.approx(factual[1] + 0.5, abs=1e-9)
    assert (
        require_success(
            consume_family_artifact(bytes(partial.export()), expected_identity=partial.identity)
        ).to_dict()
        == partial.to_dict()
    )
    rows = [{"values": [x, 0.5 * x], "clamps": [{"variable": 0, "value": x}]} for x in [-1.0, 1.0]]
    posterior = require_success(
        posterior_responses(
            linear,
            [{"kind": "weight", "equation": 1, "parent": 0}],
            rows,
            worlds,
            [[0.0, 0.0]],
            prior_sd=1.0,
            sampling=sampling,
            execution=Execution(seed=91),
        ).execute()
    )
    assert len(posterior.to_dict()["answer"]["result"]["worlds"][1]) == 32
    assert (
        require_success(
            consume_family_artifact(bytes(posterior.export()), expected_identity=posterior.identity)
        ).to_dict()
        == posterior.to_dict()
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
    fit = require_success(
        ode.fit([{"kind": "decay", "equation": 0}], [observation], score_tolerance=1e-5).execute()
    )
    assert fit.to_dict()["answer"]["result"]["converged"]
    # Equal variances make the likelihood least squares in the decay k of x(t) = 2 exp(-k t):
    # minimise the sum of squared errors by ternary search (convex near the optimum).
    times = np.array([0.5, 1.0])
    seen = np.array([2.0 * math.exp(-1.0) + 0.01, 2.0 * math.exp(-2.0) - 0.01])
    low, high = 0.0, 5.0
    for _ in range(200):
        a, b = low + (high - low) / 3.0, high - (high - low) / 3.0
        if ((seen - 2.0 * np.exp(-a * times)) ** 2).sum() < (
            (seen - 2.0 * np.exp(-b * times)) ** 2
        ).sum():
            high = b
        else:
            low = a
    assert fit.to_dict()["answer"]["result"]["model"]["decay"][0] == pytest.approx(low, abs=1e-5)
    assert abs(low - 2.0) > 1e-3, "the noise moves the optimum off the generating value"
    ou = OrnsteinUhlenbeck((-1.0,), (0.0,), (4.0,))
    rows = [{"values": [value], "clamps": []} for value in [-0.5, 1.5] * 10]
    fitted_ou = require_success(
        ou.fit(
            [{"kind": "drift", "row": 0, "column": 0}, {"kind": "center", "variable": 0}], rows
        ).execute()
    )
    assert fitted_ou.to_dict()["answer"]["result"]["converged"]
    assert (
        require_success(
            consume_family_artifact(bytes(fitted_ou.export()), expected_identity=fitted_ou.identity)
        ).to_dict()
        == fitted_ou.to_dict()
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
    result = require_success(
        fit_polynomial_candidates(candidates, [{"values": [0.9], "variances": [0.01]}]).execute()
    )
    assert result.to_dict()["answer"]["result"]["best_candidate"] == "a=0.81"
    assert result.to_dict()["answer"]["result"]["finite_set_optimum_certified"]


def test_cyclic_execution_honors_resource_and_cancellation_controls() -> None:
    from antecedent import _native

    model = EquilibriumModel(
        (Equation("a", 0.0, (0.0, 0.2)), Equation("b", 0.0, (0.2, 0.0))), (1.0, 0.0, 0.0, 1.0)
    )
    prepared = model.prepare([[]], [[0.0, 0.0]])
    with pytest.raises(CausalResourceError, match="(?i)(budget|memory|resource|limit|admission)"):
        require_success(prepared.execute(execution=Execution(limits=ResourceLimits(1))))
    token = _native.CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelledError, match="cancel"):
        require_success(prepared.execute(execution=Execution(cancel=token)))
    family = OrnsteinUhlenbeck((-1.0,), (0.0,), (1.0,)).stationary()
    with pytest.raises(CausalCancelledError, match="cancel"):
        require_success(family.execute(execution=Execution(cancel=token)))


def test_recovered_linear_keeps_failed_draws_and_replays_in_a_fresh_process(tmp_path: Path) -> None:
    from antecedent.cyclic import prepare_recovered_linear

    def draw(into_from, weight, marked):
        return {
            "coefficients": [
                {"into": 0, "from": 1, "value": into_from[0]},
                {"into": 1, "from": 0, "value": into_from[1]},
            ],
            "intercept": [0.0, 0.0],
            "noise_mean": [1.0, 0.0],
            "noise_covariance": [1.0, 0.0, 0.0, 1.0],
            "unresolved": [],
            "weight": weight,
            "marked_inadmissible": marked,
        }

    prepared = prepare_recovered_linear(
        [0, 1],
        [draw((0.4, 0.5), 0.5, False), draw((0.4, 0.5), 0.25, True), draw((1.0, 1.0), 0.25, False)],
        [{"clamp_nodes": [], "clamp_values": []}, {"clamp_nodes": [0], "clamp_values": [2.0]}],
        outcome=1,
        source_id="clamp-experiments",
        certifying_evidence=["held-out-regime"],
    )
    assert prepared.inspect()["preparation_evaluations"] == 0
    assert prepared.inspect()["family"] == "recovered_linear"
    result = require_success(prepared.execute())
    assert prepared.inspect()["preparation_evaluations"] == 0
    worlds = result.to_dict()["answer"]["result"]["worlds"]
    open_draw, marked, failed = worlds[0]["draws"]
    assert open_draw["outcome"]["solved_by"]["algorithm"] == "cyclic"
    assert "condensation" in open_draw["outcome"]["solved_by"]["next_attempt"]
    assert open_draw["outcome"]["mean"] == pytest.approx([1.25, 0.625])
    assert marked["outcome"]["status"] == "inadmissible"
    assert marked["outcome"]["reason"] == "marked"
    assert failed["outcome"]["status"] == "not_certified"
    clamped = worlds[1]["draws"][0]["outcome"]
    assert clamped["solved_by"]["algorithm"] == "acyclic"
    assert clamped["mean"] == pytest.approx([2.0, 1.0])
    assert len(worlds[0]["draws"]) == 3
    path = tmp_path / "recovered-linear.ant"
    path.write_bytes(bytes(result.export()))
    program = (
        "from antecedent.outcomes import require_success; import json,pathlib; from antecedent.cyclic import consume_family_artifact; "
        + f"print(json.dumps(require_success(consume_family_artifact(pathlib.Path({str(path)!r}).read_bytes(),"
        + f"expected_identity={result.identity!r})).to_dict(),sort_keys=True))"
    )
    replay = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(replay.stdout) == result.to_dict()


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
        result = require_success(request.execute())
        path = tmp_path / f"family-{i}.ant"
        path.write_bytes(bytes(result.export()))
        expected.append(result.to_dict())
        paths.append((str(path), result.identity))
    program = (
        "from antecedent.outcomes import require_success; import json,pathlib; from antecedent.cyclic import consume_family_artifact; "
        + f"pairs={paths!r}; print(json.dumps([require_success(consume_family_artifact(pathlib.Path(p).read_bytes(),expected_identity=i)).to_dict() for p,i in pairs],sort_keys=True))"
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
    result = require_success(
        joint_counterfactual(
            system,
            [{"kind": "intercept", "equation": 0}],
            [{"values": [value], "clamps": []} for value in [0.0, 1.0]],
            [2.0],
            [0.5],
            [[], [{"variable": 0, "value": 3.0}]],
            prior_sd=1.0,
            initial=[1.5],
            sampling=SamplingPolicy(chains=2, warmup=32, draws=32, proposal_scale=0.5),
            execution=Execution(seed=932),
        ).execute()
    )
    answer = result.to_dict()["answer"]["result"]
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
        require_success(
            consume_family_artifact(bytes(result.export()), expected_identity=result.identity)
        ).to_dict()
        == result.to_dict()
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
    result = require_success(
        action_worlds(
            linear,
            worlds,
            [[1.0, 0.0]],
            factual=[1.25, 0.625],
            policy=SolvePolicy(require_contraction=False),
        ).execute()
    )
    answers = result.to_dict()["answer"]["result"]
    for answer, expected in zip(
        answers[:4], [[2.0, 1.0], [2.5, 1.25], [1.0, 0.0], [4.0, 2.0]], strict=True
    ):
        assert answer["rows"][0]["values"] == pytest.approx(expected)
        assert answer["counterfactual"]["values"] == pytest.approx(expected)
    assert len(answers) == 5
    assert answers[4]["rows"][0]["failure"] == "singular"
    assert answers[4]["counterfactual"]["failure"] == "singular"
    assert (
        require_success(
            consume_family_artifact(bytes(result.export()), expected_identity=result.identity)
        ).to_dict()
        == result.to_dict()
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
    result = require_success(prepared.execute())
    assert result.to_dict()["fit"]["converged"]
    assert result.to_dict()["worlds"][1][0]["values"][0] == pytest.approx(
        math.tanh(0.12), abs=0.035
    )
    expected_counterfactual = math.tanh(0.12) + 0.2 - math.tanh(0.07)
    assert result.to_dict()["counterfactuals"][1]["values"][0] == pytest.approx(
        expected_counterfactual, abs=0.035
    )
    assert prepared.inspect()["fits_performed"] == 1
    path = tmp_path / "nonlinear-milestone.ant"
    path.write_bytes(bytes(result.export()))
    program = (
        "from antecedent.outcomes import require_success; import json,pathlib; from antecedent.cyclic import consume_artifact; "
        + f"r=require_success(consume_artifact(pathlib.Path({str(path)!r}).read_bytes(),expected_identity={result.identity!r}));print(json.dumps(r.to_dict(),sort_keys=True))"
    )
    replay = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(replay.stdout) == result.to_dict()


def test_family_preparation_reuses_products_and_refresh_invalidates_binding() -> None:
    request = OrnsteinUhlenbeck((-2.0,), (1.0,), (4.0,)).stationary()
    assert request.inspect()["preparation_evaluations"] == 0
    first = require_success(request.execute())
    assert request.inspect()["preparation_evaluations"] == 0
    assert require_success(request.execute()).to_dict() == first.to_dict()
    # Refreshing an already compiled family preserves the same public contract.
    same = require_success(request.reconfigure())
    assert same.inspect()["preparation_evaluations"] == 0
    assert require_success(same.execute()).to_dict() == first.to_dict()
    with pytest.raises(CausalResourceError, match="(?i)(memory|limit|budget|admission)"):
        require_success(request.execute(execution=Execution(limits=ResourceLimits(1))))
    replacement = json.loads(request.request_json)
    replacement["model"]["center"] = [2.0]
    refreshed = require_success(request.reconfigure(scientific={"model": replacement["model"]}))
    assert refreshed.inspect()["preparation_evaluations"] == 0
    assert require_success(refreshed.execute()).identity != first.identity
    assert require_success(request.execute()).identity == first.identity


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
    result = require_success(
        bootstrap_responses(*inputs, replicates=6, execution=Execution(seed=91)).execute()
    )
    answer = result.to_dict()["answer"]["result"]
    assert len(answer["parameters"]) == 6
    for values, observational, clamped in zip(
        answer["parameters"], answer["worlds"][0], answer["worlds"][1], strict=True
    ):
        assert values is not None
        assert observational[0]["values"][0] == pytest.approx(values[0])
        assert clamped[0]["values"][0] == 3.0
    assert (
        require_success(
            consume_family_artifact(bytes(result.export()), expected_identity=result.identity)
        ).to_dict()
        == result.to_dict()
    )
    failed = require_success(
        bootstrap_responses(
            *inputs, replicates=4, execution=Execution(seed=91), policy=FitOptions(max_sweeps=1)
        ).execute()
    ).to_dict()["answer"]["result"]
    # ``to_dict`` is the explicit editable JSON boundary: sequences are lists, failed slots stay.
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
    result = require_success(
        discover_cyclic(
            columns,
            simple_scm=True,
            sigma_faithful=True,
            independent_equilibrium_replicates=True,
            ci="parcorr",
            alpha=0.001,
        ).execute()
    )
    answer = result.to_dict()["answer"]["result"]
    pairs = {(a, b) for a, b, _, _ in answer["edges"]}
    assert (0, 1) in pairs and (1, 2) in pairs
    assert all(3 not in pair for pair in pairs)
    assert answer["ci_source"] == "parcorr"
    assert "inducing path" in answer["interpretation"]
    assert (
        require_success(
            consume_family_artifact(bytes(result.export()), expected_identity=result.identity)
        ).to_dict()
        == result.to_dict()
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
    }
    posterior = require_success(prepared.posterior(**options, execution=Execution(seed=82)))
    assert len(posterior["samples"]) == 16
    with pytest.raises(CausalResourceError, match="(?i)(admission|memory|limit|resource)"):
        require_success(
            prepared.posterior(**options, execution=Execution(limits=ResourceLimits(1)))
        )
    with pytest.raises(CausalResourceError, match="(?i)(admission|memory|limit|resource)"):
        _native.execute_cyclic_system(prepared.request_json, max_bytes=1)
    token = _native.CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelledError, match="cancel"):
        require_success(prepared.posterior(**options, execution=Execution(cancel=token)))


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
        result = require_success(
            model(flipped)
            .prepare(PolynomialObservation("factual", (None,), "positive_sensor"))
            .execute()
        )
        answer = result.to_dict()["answer"]["result"]
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
            require_success(
                consume_family_artifact(bytes(result.export()), expected_identity=result.identity)
            ).to_dict()
            == result.to_dict()
        )
        answers.append(result)
    from antecedent import decision
    from antecedent.joint_distribution import ScientificQuantity

    impossible = model(False).prepare(PolynomialObservation("factual", tuple(region(0.2, 0.3))))
    impossible_evidence = require_success(
        impossible.evidence(
            {"id": "cubic", "population": "unit", "variables": [["x", "unit"]]},
            [{"kind": "polynomial_world", "world": "counterfactual", "variable": 0}],
        )
    )
    q = ScientificQuantity.from_wire(impossible_evidence.to_dict()["columns"][0]["quantity"])
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
        require_success(impossible_evidence.decide_enclosures(c))
    path = tmp_path / "joint-selection.ant"
    path.write_bytes(bytes(answers[0].export()))
    program = (
        "from antecedent.outcomes import require_success; import json,pathlib;from antecedent.cyclic import consume_family_artifact;"
        + f"r=require_success(consume_family_artifact(pathlib.Path({str(path)!r}).read_bytes(),expected_identity={answers[0].identity!r}));print(json.dumps(r.to_dict(),sort_keys=True))"
    )
    replay = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(replay.stdout) == answers[0].to_dict()


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
    result = require_success(prepared.execute())
    fit = result.to_dict()["answer"]["result"]
    assert fit["converged"]
    assert abs(fit["model"]["equations"][0]["parameters"][0] - 0.7) < 1e-6
    assert (
        require_success(
            consume_family_artifact(bytes(result.export()), expected_identity=result.identity)
        ).to_dict()
        == result.to_dict()
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
    answer = require_success(
        fitted.prepare(
            [[{"variable": 1, "value": 1.0}]], [[0.0, 0.0]], factual=[0.3, -0.2]
        ).execute()
    )
    assert answer.to_dict()["worlds"][0][0]["globally_unique"]


def test_typed_identified_cyclic_functional_evaluation_and_replay():
    from antecedent.cyclic import CyclicGraph, consume_family_artifact

    source = CyclicGraph(3, ((0, 1), (1, 2), (2, 1))).identify(
        [2], [0], io_scm=True, scc_product_density=True, continuous=False
    )
    functional = require_success(source.identified_functional())
    weights = []
    for cell in range(8):
        t, a, b = bool(cell & 1), bool(cell & 2), bool(cell & 4)
        pt, pa, pb = (0.6 if t else 0.4), (0.8 if t else 0.2), (0.6 if a else 0.1)
        weights.append(pt * (pa if a else 1 - pa) * (pb if b else 1 - pb))
    result = require_success(
        functional.evaluate(
            [[0.0, 1.0]] * 3,
            weights,
            [[(0, 0.0), (2, 1.0)], [(0, 1.0), (2, 1.0)], [(0, 2.0), (2, 1.0)]],
        ).execute()
    )
    values = result.to_dict()["answer"]["result"]["values"]
    assert abs(values[0]["value"] - 0.2) < 1e-12
    assert abs(values[1]["value"] - 0.5) < 1e-12
    assert values[2]["value"] is None and values[2]["failure"]
    assert (
        require_success(
            consume_family_artifact(bytes(result.export()), expected_identity=result.identity)
        ).to_dict()
        == result.to_dict()
    )


def test_local_callbacks_scc_counterfactual_budget_and_portability():
    from antecedent.cyclic import LocalCyclicModel, LocalMechanism

    local = LocalCyclicModel(
        (
            LocalMechanism("a", (0, 1), lambda x: (1.0 + 0.2 * x[0] + 0.4 * x[1], [0.2, 0.4])),
            LocalMechanism("b", (0,), lambda x: (0.5 * x[0], [0.5])),
        )
    )
    answer = require_success(local.execute([0.0, 0.0], []))
    assert answer["result"]["values"][0] == pytest.approx(1.0 / 0.6)
    assert not answer["result"]["globally_unique"]
    assert not answer["inspection"]["portable"]
    cf = require_success(
        local.execute([0.0, 0.0], [{"variable": 0, "value": 2.0}], factual=[0.3, 0.7])
    )
    assert cf["counterfactual"]["values"][1] == pytest.approx(1.55)
    with pytest.raises(CyclicRefusal, match="portable") as refused:
        bytes(local.export())
    assert refused.value.reason_code == "policy_callback_not_portable"
    with pytest.raises(CausalResourceError, match="budget"):
        require_success(local.execute([0.0, 0.0], [], max_evaluations=1))
    with pytest.raises(ValueError, match="derivatives"):
        require_success(
            LocalCyclicModel((LocalMechanism("bad", (0,), lambda x: (x[0], [])),)).execute(
                [0.0], []
            )
        )


def test_native_cyclic_evidence_decision_and_fresh_process_replay(tmp_path):
    from antecedent import decision
    from antecedent.cyclic import consume_evidence
    from antecedent.errors import CausalSerializationError
    from antecedent.joint_distribution import ScientificQuantity

    prepared = OrnsteinUhlenbeck((-2.0,), (3.0,), (4.0,)).stationary()
    schema = {"id": "physical-system", "population": "system", "variables": [["position", "meter"]]}
    evidence = require_success(
        prepared.evidence(schema, [{"kind": "ou_mean", "time": None, "variable": 0}])
    )
    assert prepared.inspect()["preparation_evaluations"] == 0
    assert evidence.to_dict()["columns"][0]["samples"] == [3.0]
    assert evidence.to_dict()["columns"][0]["port"] == "mean"
    quantity = ScientificQuantity.from_wire(evidence.to_dict()["columns"][0]["quantity"])
    contract = decision.Contract(
        (
            decision.Action("keep", inputs=(quantity,), utility=decision.x(0)),
            decision.Action("constant", inputs=(quantity,), utility=decision.x(0) * 0 + 2),
        ),
        utility_units="meter",
        criterion=decision.Criterion.expected_utility(),
        target_population="system",
    )
    result = require_success(evidence.decide_means(contract))
    assert result["scope"] == "specified_or_fitted_model_conditional"
    assert result["evidence_identity"] == evidence.identity
    artifact = bytes(evidence.export())
    assert (
        require_success(consume_evidence(artifact, expected_identity=evidence.identity)).to_dict()
        == evidence.to_dict()
    )
    path = tmp_path / "evidence.ant"
    path.write_bytes(artifact)
    code = f"from antecedent.outcomes import require_success\nimport pathlib, json\nfrom antecedent.cyclic import consume_evidence\ne = require_success(consume_evidence(pathlib.Path({str(path)!r}).read_bytes(), expected_identity={evidence.identity!r}))\nprint(json.dumps(e.to_dict(), sort_keys=True))\n"
    process = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert json.loads(process.stdout) == evidence.to_dict()
    changed = require_success(
        prepared.evidence(
            {**schema, "population": "other-system"},
            [{"kind": "ou_mean", "time": None, "variable": 0}],
        )
    )
    assert changed.identity != evidence.identity
    with pytest.raises(CausalSerializationError, match="identity"):
        require_success(consume_evidence(artifact, expected_identity=changed.identity))
    refused = evidence.joint_law()
    assert refused.refusal is not None
    assert refused.refusal.code == "joint_law_required"
    assert refused.refusal.detail == "cyclic_evidence.joint_law"
    with pytest.raises(StructuredRefusal, match="uncertainty_port"):
        require_success(refused)
    with pytest.raises(ValueError, match="schema"):
        require_success(
            prepared.evidence(
                {**schema, "variables": [["extra", "meter"], ["invalid", "meter"]]},
                [{"kind": "ou_mean", "time": None, "variable": 1}],
            )
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
        sampling=SamplingPolicy(chains=2, warmup=8, draws=8, proposal_scale=0.01),
        execution=Execution(seed=2),
    )
    evidence = require_success(posterior.evidence(schema, selectors))
    assert evidence.to_dict()["columns"][0]["port"] == "parameter_posterior"
    assert evidence.to_dict()["columns"][0]["uncertainty_axes"] == ["parameter"]
    law = require_success(evidence.joint_law())
    assert law.to_dict()["identity"]["semantic"] == "causal_functional_posterior"
    from antecedent import decision
    from antecedent.joint_distribution import ScientificQuantity

    q = ScientificQuantity.from_wire(evidence.to_dict()["columns"][0]["quantity"])
    contract = decision.Contract(
        (
            decision.Action("nonlinear", inputs=(q,), utility=decision.x(0) * decision.x(0)),
            decision.Action("constant", inputs=(q,), utility=decision.x(0) * 0 - 1),
        ),
        utility_units="unit_squared",
        target_population="model",
        criterion=decision.Criterion.expected_utility(),
    )
    selected = require_success(contract.evaluate(law))
    assert selected.selected == ("nonlinear",)
    assert (
        require_success(
            decision.replay(bytes(selected.export()), contract=contract, source=law)
        ).selected
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
    resampling = require_success(bootstrap.evidence(schema, selectors))
    assert resampling.to_dict()["columns"][0]["port"] == "sampling_law"
    refused = resampling.joint_law()
    assert refused.refusal is not None
    assert refused.refusal.code == "joint_law_required"
    with pytest.raises(StructuredRefusal, match="uncertainty_port"):
        require_success(refused)


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
    answer = require_success(prepared.execute())
    assert len(answer.to_dict()["answer"]["result"]["roots"]) == 3
    evidence = require_success(prepared.evidence(schema, [{"kind": "solution_set", "variable": 0}]))
    quantity = ScientificQuantity.from_wire(evidence.to_dict()["columns"][0]["quantity"])
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
    result = require_success(evidence.decide_enclosures(contract))
    assert result["scope"] == "conditional_model_solution_outer_enclosure"
    assert "root_action" in result["verdict"]
    assert not result["selection_probabilities_inferred"]
    unresolved = model.isolate([{"lower": -2.0, "upper": 2.0}], [0.0], max_boxes=1)
    retained = require_success(
        unresolved.evidence(schema, [{"kind": "solution_set", "variable": 0}])
    )
    assert retained.to_dict()["columns"][0]["diagnostics"]["unresolved_regions"] > 0
    other = ScientificQuantity.from_wire(retained.to_dict()["columns"][0]["quantity"])
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
    assert "constant" in require_success(retained.decide_enclosures(other_contract))["verdict"]


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
    answer = require_success(prepared.execute())
    laws = answer.to_dict()["answer"]["result"]["laws"]
    assert laws[1]["mean"][0] == pytest.approx(0.1)
    assert laws[1]["mean"][1] == pytest.approx(0.68)
    assert "failure" in laws[2]
    assert (
        require_success(
            consume_family_artifact(bytes(answer.export()), expected_identity=answer.identity)
        ).to_dict()
        == answer.to_dict()
    )
    schema = {"id": "sde", "population": "physical", "variables": [["a", "unit"], ["b", "unit"]]}
    evidence = require_success(
        prepared.evidence(schema, [{"kind": "ou_regime", "world": 2, "variable": 0, "other": None}])
    )
    assert evidence.to_dict()["columns"][0]["samples"] == [None]
    assert evidence.to_dict()["columns"][0]["diagnostics"]["failure"]


def test_calculus_adjustment_functional_and_bounded_status_replay():
    from antecedent.cyclic import CyclicGraph

    graph = CyclicGraph(2, ((0, 1),))
    source = graph.calculus(
        "exchange", [0], [1], [], [], io_scm=True, standard_measurable_spaces=True
    )
    assert (
        require_success(source.execute()).to_dict()["answer"]["result"]["status"] == "verified_rule"
    )
    result = require_success(
        require_success(source.identified_functional())
        .evaluate(
            [[0.0, 1.0]] * 2, [0.4, 0.1, 0.1, 0.4], [[(0, 0.0), (1, 1.0)], [(0, 1.0), (1, 1.0)]]
        )
        .execute()
    )
    assert [x["value"] for x in result.to_dict()["answer"]["result"]["values"]] == pytest.approx(
        [0.2, 0.8]
    )
    assert (
        require_success(
            consume_family_artifact(bytes(result.export()), expected_identity=result.identity)
        ).to_dict()
        == result.to_dict()
    )
    adjusted = graph.adjust(
        [1], [0], [], io_scm=True, standard_measurable_spaces=True, continuous=False
    )
    answer = require_success(
        require_success(adjusted.identified_functional())
        .evaluate([[0.0, 1.0]] * 2, [0.4, 0.1, 0.1, 0.4], [[(0, 1.0), (1, 1.0)]])
        .execute()
    )
    assert answer.to_dict()["answer"]["result"]["values"][0]["value"] == pytest.approx(0.8)
    exhausted = require_success(
        graph.calculus(
            "exchange", [0], [1], [], [], io_scm=True, standard_measurable_spaces=True, max_checks=0
        ).execute()
    )
    assert exhausted.to_dict()["answer"]["result"]["status"] == "budget_exhausted"
    failed = require_success(
        graph.calculus(
            "action", [0], [1], [], [], io_scm=True, standard_measurable_spaces=True
        ).execute()
    )
    assert failed.to_dict()["answer"]["result"]["status"] == "not_licensed_by_criterion"


@pytest.mark.parametrize(
    ("rule", "max_checks", "status"),
    [("exchange", 0, "budget_exhausted"), ("action", 1000, "not_licensed_by_criterion")],
)
def test_unavailable_cyclic_functional_is_a_scientific_refusal_with_native_evidence(
    rule, max_checks, status
):
    from antecedent.cyclic import CyclicGraph

    study = CyclicGraph(2, ((0, 1),)).calculus(
        rule,
        [0],
        [1],
        [],
        [],
        io_scm=True,
        standard_measurable_spaces=True,
        max_checks=max_checks,
    )
    original = require_success(study.execute())
    refused = study.identified_functional()
    assert refused.value is None
    assert refused.answer.kind == "refused"
    assert refused.refusal.code == "effect_not_identified"
    assert refused.refusal.detail == "cyclic.observational_functional_unavailable"
    assert refused.refusal.fields["native_identification"]["status"] == status
    assert refused.refusal.fields["source_identity"] == original.identity
    from antecedent._freeze import thaw

    assert (
        thaw(refused.refusal.fields["native_identification"])
        == (original.to_dict()["answer"]["result"])
    )
    assert refused.refusal.remedy
    assert require_success(study.execute()).to_dict() == original.to_dict()


def test_adam_public_replay_and_policy_data_invalidation():
    from antecedent.cyclic import AdamPolicy

    m = EquilibriumModel((Equation("one", 0.0, (0.0,), "relu"),), (1.0,))
    training = [{"values": [x], "clamps": []} for x in [-1.0, 0.0, 1.0, 2.0]]
    p = AdamPolicy(max_epochs=25, minimum_epochs=3, patience=3)
    prepared = m.fit_adam([{"kind": "intercept", "equation": 0}], training, training, policy=p)
    result = require_success(prepared.execute())
    fit = result.to_dict()["answer"]["result"]
    assert fit["epochs"] <= 25 and fit["nonsmooth_convention"]
    assert (
        require_success(
            consume_family_artifact(bytes(result.export()), expected_identity=result.identity)
        ).to_dict()
        == result.to_dict()
    )
    assert (
        require_success(
            m.fit_adam(
                [{"kind": "intercept", "equation": 0}],
                training,
                training,
                policy=p,
                execution=Execution(seed=1),
            ).execute()
        ).identity
        != result.identity
    )
    changed = [dict(row, values=[row["values"][0] + 0.1]) for row in training]
    assert (
        require_success(
            m.fit_adam(
                [{"kind": "intercept", "equation": 0}], changed, training, policy=p
            ).execute()
        ).identity
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
        result = require_success(prepared.execute())
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
        fresh = require_success(PreparedCyclicFamily(json.dumps(changed)).execute())
        assert fresh.identity != result.identity
        assert require_success(prepared.execute()).to_dict() == result.to_dict()
        with pytest.raises(Exception, match="identity|mismatch"):
            require_success(
                consume_family_artifact(bytes(fresh.export()), expected_identity=result.identity)
            )
        path = tmp_path / f"inventory-{i}.ant"
        path.write_bytes(bytes(result.export()))
        pairs.append((str(path), result.identity))
        expected.append(result.to_dict())
    script = f"from antecedent.outcomes import require_success\nimport json, pathlib\nfrom antecedent.cyclic import consume_family_artifact\npairs = {pairs!r}\nprint(json.dumps([require_success(consume_family_artifact(pathlib.Path(p).read_bytes(), expected_identity=i)).to_dict() for p, i in pairs], sort_keys=True))\n"
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
    e = require_success(
        p.evidence(
            {"id": "numeric", "population": "unit", "variables": [["x", "unit"]]},
            [
                {
                    "kind": "action",
                    "world": 0,
                    "noise_row": 0,
                    "variable": 0,
                    "counterfactual": False,
                }
            ],
        )
    )
    assert e.to_dict()["columns"][0]["numerical_enclosures"] == []
    assert e.to_dict()["columns"][0]["diagnostics"]["globally_unique"] is False
    with pytest.raises(CausalResourceError, match="admission|resource|limit|budget"):
        bytes(e.export(execution=Execution(limits=ResourceLimits(1))))


def test_continuous_identification_gaussian_fitting_evidence_and_fresh_replay(tmp_path):
    from antecedent.cyclic import CyclicGraph

    source = CyclicGraph(3, ((0, 1), (1, 2)), ((0, 2),)).identify(
        [2], [0], io_scm=True, scc_product_density=True, continuous=True
    )
    functional = require_success(source.identified_functional())
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
    result = require_success(prepared.execute())
    answer = result.to_dict()["answer"]["result"]
    truth = math.exp(-0.5 * 0.4**2 / 2.13) / math.sqrt(2 * math.pi * 2.13)
    assert answer["fitted_rows"] == 16
    assert answer["values"][0]["value"] == pytest.approx(truth, abs=1e-12)
    assert answer["values"][1]["value"] is None
    evidence = require_success(
        prepared.evidence(
            {
                "id": "continuous",
                "population": "population",
                "variables": [["x", "unit"], ["m", "unit"], ["y", "unit"]],
            },
            [{"kind": "functional", "assignment": 0}],
        )
    )
    assert evidence.to_dict()["columns"][0]["port"] == "likelihood"
    assert evidence.to_dict()["columns"][0]["quantity"]["variable_name"] == "y"
    assert evidence.to_dict()["columns"][0]["quantity"]["units"] == "1/(unit)"
    path = tmp_path / "continuous.ant"
    path.write_bytes(bytes(result.export()))
    code = f"from antecedent.outcomes import require_success\nimport pathlib, json\nfrom antecedent.cyclic import consume_family_artifact\nr = require_success(consume_family_artifact(pathlib.Path({str(path)!r}).read_bytes(), expected_identity={result.identity!r}))\nprint(json.dumps(r.to_dict(), sort_keys=True))\n"
    replay = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert json.loads(replay.stdout) == result.to_dict()
    with pytest.raises(CausalResourceError, match="admission|limit|resource"):
        require_success(prepared.execute(execution=Execution(limits=ResourceLimits(1))))
    with pytest.raises(ValueError, match="IID|replicates"):
        require_success(
            functional.evaluate_gaussian(
                {"kind": "fit", "rows": rows, "independent_observational_replicates": False},
                [[(0, 1.0), (2, 0.0)]],
            ).execute()
        )


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
    prepared = learn_structure(m, rows, execution=Execution(seed=71), **kwargs)
    result = require_success(prepared.execute())
    assert any(value is not None for value in result.to_dict()["answer"]["result"]["bootstrap"])
    assert (
        require_success(
            consume_family_artifact(bytes(result.export()), expected_identity=result.identity)
        ).to_dict()
        == result.to_dict()
    )
    assert (
        require_success(
            learn_structure(m, rows, execution=Execution(seed=72), **kwargs).execute()
        ).identity
        != result.identity
    )
    ci = {"simple_scm": True, "sigma_faithful": True, "independent_equilibrium_replicates": True}
    columns = [list(map(float, range(8))), [0.0, 0.5, 0.1, 2.0, 1.0, 0.5, 2.0, 1.5]]
    graph = require_success(discover_cyclic(columns, execution=Execution(seed=8), **ci).execute())
    assert (
        require_success(
            discover_cyclic(columns, execution=Execution(seed=9), **ci).execute()
        ).identity
        != graph.identity
    )
    assert (
        require_success(
            consume_family_artifact(bytes(graph.export()), expected_identity=graph.identity)
        ).to_dict()
        == graph.to_dict()
    )


def test_cyclic_logical_retention_and_projection_resource_limits():
    from antecedent.cyclic import consume_evidence

    prepared = OrnsteinUhlenbeck((-2.0,), (3.0,), (4.0,)).stationary()
    result = require_success(prepared.execute())
    # The prepared owner retains the compiled request, not the exported artifact.
    retained = prepared.inspect()["retained_bytes"]
    assert retained > 0
    assert bytes(require_success(prepared.execute()).export()) == bytes(result.export())
    assert prepared.inspect()["retained_bytes"] == retained
    schema = {"id": "physical", "population": "system", "variables": [["position", "meter"]]}
    selector = {"kind": "ou_mean", "time": None, "variable": 0}
    with pytest.raises(ValueError, match="schema"):
        require_success(prepared.evidence({**schema, "id": "x" * 257}, [selector]))
    evidence = require_success(prepared.evidence(schema, [selector] * 128))
    artifact = bytes(evidence.export())
    # Repeated columns compress well; wire size is not the retained heap budget.
    with pytest.raises(CausalResourceError, match="admission|resource|limit|budget"):
        require_success(
            prepared.evidence(
                schema,
                [selector] * 128,
                execution=Execution(limits=ResourceLimits(len(artifact) * 4)),
            )
        )
    with pytest.raises(CausalResourceError, match="admission|resource|limit|budget"):
        require_success(
            consume_evidence(
                artifact,
                expected_identity=evidence.identity,
                execution=Execution(limits=ResourceLimits(len(artifact) * 4)),
            )
        )


def test_cyclic_native_facts_survive_execution_cancellation_and_bad_artifact():
    from antecedent import _native
    from antecedent.cyclic import consume_artifact

    system = EquilibriumModel((Equation("x", 0.0, (0.0,), "identity"),), (1.0,))
    prepared = system.prepare([[]], [[0.0]])
    result = require_success(prepared.execute())
    assert result.execution_facts.is_native
    assert result.execution_facts.status() == "completed"
    assert result.execution_facts.peak_admitted_bytes > 0
    replay = require_success(
        consume_artifact(bytes(result.export()), expected_identity=result.identity)
    )
    assert replay.execution_facts.is_native
    assert replay.execution_facts.status() == "completed"

    token = _native.CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelledError, match="cancel") as cancelled:
        require_success(prepared.execute(execution=Execution(cancel=token)))
    assert cancelled.value.execution_facts.status() == "cancelled"
    with pytest.raises(_native.CausalSerializationError) as corrupt:
        require_success(consume_artifact(b"corrupt", expected_identity=result.identity))
    assert corrupt.value.execution_facts.is_native
    assert corrupt.value.execution_facts.status() == "failed"


def test_native_numeric_views_retain_their_execution_after_the_preparation_is_discarded():
    import gc

    system = EquilibriumModel((Equation("x", 0.0, (0.0,), "identity"),), (1.0,))
    prepared = system.prepare([[]], [[float(i)] for i in range(80)])
    execution, facts = prepared._handle.execute()
    rows = [memoryview(execution.world_values(0, row=row)) for row in range(5, 10)]
    assert all(view.readonly and view.format == "d" and view.shape == (1,) for view in rows)
    del prepared, facts, execution
    gc.collect()
    # Each borrowed row still reads its exact native values: the view owns the execution.
    assert [view[0] for view in rows] == [5.0, 6.0, 7.0, 8.0, 9.0]
    with pytest.raises(ValueError, match="out of range"):
        # The coordinate space belongs to the execution that produced it.
        system.prepare([[]], [[0.0]])._handle.execute()[0].world_values(0, row=1)


# Typed worlds, experimentally recovered laws and dynamic-fit inference. The feedback pair is
# `A = 0.4 B + U_A`, `B = 0.5 A + U_B` with independent unit disturbances: the observational
# B-on-A association is 1.40625 / 1.8125 (about 0.776), the hard-clamp derivative of B in A
# is 0.5, and a unit mean shift of U_A moves B by 0.625 (and A by 1.25).


def _fresh_replay(tmp_path: Path, program: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout


def _feedback_linear():
    from antecedent.cyclic import LinearCyclicModel

    return LinearCyclicModel.from_direct_effects(["A", "B"], {("A", "B"): 0.4, ("B", "A"): 0.5})


def _clamp_regimes(designs, *, rows=400, seed=20_251_023):
    """Randomized clamp experiments on the feedback pair, one regime per design."""
    import numpy as np
    from antecedent.cyclic import ExperimentRegime

    rng = np.random.default_rng(seed)
    regimes = []
    for source, clamped in enumerate(designs, start=1):
        ua, ub = rng.standard_normal(rows), rng.standard_normal(rows)
        if clamped is None:
            a = (ua + 0.4 * ub) / 0.8
            b = 0.5 * a + ub
            clamps = {}
        elif clamped == "A":
            a = 2.0 + 1.5 * rng.standard_normal(rows)
            b = 0.5 * a + ub
            clamps = {"A": 2.0}
        else:
            b = 3.0 + 1.5 * rng.standard_normal(rows)
            a = 0.4 * b + ua
            clamps = {"B": 3.0}
        regimes.append(ExperimentRegime({"A": a, "B": b}, source=source, clamps=clamps))
    return regimes


def _recovery(
    uncertainty, *, execution=None, designs=(None, "A", "B"), worlds=None, sampling_design=None
):
    from antecedent.cyclic import Set, World, prepare_experimental_recovery

    return prepare_experimental_recovery(
        ["A", "B"],
        _clamp_regimes(designs),
        worlds or [World(1), World(2, Set("A", 2.0))],
        outcome="B",
        uncertainty=uncertainty,
        source_id="clamp-experiments",
        certifying_evidence=["held-out-regime"],
        sampling_design=sampling_design,
        execution=execution,
    )


def test_linear_worlds_reproduce_the_analytic_feedback_pair_and_replay(tmp_path: Path) -> None:
    from antecedent.cyclic import Set, Shift, World, WorldAnswer, WorldSelector, consume_evidence

    prepared = _feedback_linear().worlds(
        [World(0), World(1, Set("A", 0.0)), World(2, Set("A", 1.0)), World(3, Shift("A", 1.0))],
        component_id="feedback",
        source_id="specification",
        certifying_evidence=["independent-bench-test"],
    )
    facts = prepared.inspect()
    assert facts["family"] == "linear_worlds"
    assert facts["preparation_evaluations"] == 0
    assert facts["worlds"] == [0, 1, 2, 3]
    assert facts["variables"] == ["A", "B"]
    assert facts["draw_origin"] is None
    worlds = require_success(prepared.interventions())
    assert list(worlds) == [0, 1, 2, 3]
    assert all(isinstance(world, WorldAnswer) for world in worlds.values())
    observational = worlds[0]
    association = observational.covariance("A", "B") / observational.covariance("A", "A")
    assert association == pytest.approx(1.40625 / 1.8125, abs=1e-10)
    assert association == pytest.approx(0.776, abs=5e-4)
    derivative = worlds[2].mean("B") - worlds[1].mean("B")
    assert derivative == pytest.approx(0.5, abs=1e-10)
    assert worlds[3].mean("B") == pytest.approx(0.625, abs=1e-10)
    assert worlds[3].mean("A") == pytest.approx(1.25, abs=1e-10)
    for world in worlds.values():
        assert world.certified
        assert world.certificate["basis"] == "linear_family_contraction"
        assert world.failed_mass == 0.0
        assert world.physical_stability == "unclaimed"
    assert worlds[1].solved_by["method"] == "acyclic_order"
    assert worlds[3].solved_by["method"] == "strongly_connected_blocks"
    assert worlds[1].actions[0]["kind"] == "hard_clamp"
    assert worlds[1].actions[0]["name"] == "A"
    assert worlds[3].actions[0]["kind"] == "additive_shock"

    result = require_success(prepared.execute())
    assert require_success(
        consume_family_artifact(bytes(result.export()), expected_identity=result.identity)
    ).to_dict() == (result.to_dict())
    schema = {"id": "feedback", "population": "model", "variables": [["A", "unit"], ["B", "unit"]]}
    evidence = require_success(prepared.evidence(schema, [WorldSelector(3, "B")]))
    column = evidence.to_dict()["columns"][0]
    assert column["port"] == "interventional_law"
    assert column["samples"][0] == pytest.approx(0.625, abs=1e-10)
    assert require_success(
        consume_evidence(bytes(evidence.export()), expected_identity=evidence.identity)
    ).to_dict() == (evidence.to_dict())
    path = tmp_path / "linear-worlds.ant"
    path.write_bytes(bytes(result.export()))
    replayed = _fresh_replay(
        tmp_path,
        "from antecedent.outcomes import require_success; import json,pathlib; from antecedent.cyclic import consume_family_artifact; "
        + f"print(json.dumps(require_success(consume_family_artifact(pathlib.Path({str(path)!r}).read_bytes(),"
        + f"expected_identity={result.identity!r})).to_dict(),sort_keys=True))",
    )
    assert json.loads(replayed) == result.to_dict()


def test_unsolvable_clamp_is_failed_mass_and_reading_it_names_the_next_attempt() -> None:
    from antecedent.cyclic import LinearCyclicModel, Set, World

    # A = B + C, B = A, C = A: the full system solves; do(C = 1) leaves 0 = 1 + U_A + U_B.
    model = LinearCyclicModel.from_direct_effects(
        ["A", "B", "C"],
        {("A", "B"): 1.0, ("A", "C"): 1.0, ("B", "A"): 1.0, ("C", "A"): 1.0},
    )
    worlds = require_success(
        model.worlds(
            [World(0), World(1, Set("C", 1.0))],
            component_id="loops",
            source_id="specification",
            certifying_evidence=["independent-bench-test"],
            certificate="report_uncertified",
        ).interventions()
    )
    assert not worlds[0].certified
    assert worlds[0].certificate["next_attempt"] is not None
    clamped = worlds[1]
    assert clamped.status == "no_solution"
    assert clamped.failed_mass == 1.0
    assert clamped.solved_by["method"] == "not_solved"
    with pytest.raises(CyclicRefusal, match="no gaussian law") as refused:
        clamped.mean("B")
    assert refused.value.next_attempt
    assert refused.value.reason_code == "route_not_supported"


def test_equilibrium_and_ode_worlds_share_draws_initial_laws_and_refuse_shocks() -> None:
    from antecedent.cyclic import InitialState, Set, Shift, World

    feedback = EquilibriumModel(
        (
            Equation("a", 0.0, (0.0, 0.4), "identity"),
            Equation("b", 0.0, (0.5, 0.0), "identity"),
        ),
        (1.0, 0.0, 0.0, 1.0),
    )
    worlds = require_success(
        feedback.worlds(
            [World(0), World(1, Set("a", 2.0))], [[0.0, 0.0], [1.0, 0.0]]
        ).interventions()
    )
    assert worlds[0].draws("b")[1] == pytest.approx(0.625, abs=1e-6)
    assert worlds[0].draws("a")[1] == pytest.approx(1.25, abs=1e-6)
    assert worlds[1].draws("b")[1] == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(CyclicRefusal, match="equilibrium_draws") as wrong_kind:
        worlds[0].mean("b")
    assert "draws()" in wrong_kind.value.next_attempt

    decay = OdeSystem((Equation("x", 0.0, (0.0,), "identity"),), (1.0,))
    trajectories = require_success(
        decay.worlds(
            [2.0], [0.0, 1.0], [World(0), World(1, initial_states=InitialState("x", 4.0))]
        ).interventions()
    )
    assert trajectories[0].trajectory("x")[1] == pytest.approx(2.0 * math.exp(-1.0), rel=1e-6)
    assert trajectories[1].trajectory("x")[1] == pytest.approx(4.0 * math.exp(-1.0), rel=1e-6)
    assert trajectories[1].certificate["gap"] == "numerical_integration_only"
    assert trajectories[1].actions[0]["kind"] == "initial_state_change"
    # A shock on a continuous-time world is a scientific refusal naming its next attempt,
    # not an invalid argument.
    shocked = decay.worlds([2.0], [0.0, 1.0], [World(0, Shift("x", 1.0))]).interventions()
    assert shocked.refusal is not None
    assert shocked.refusal.code == "route_not_supported"
    assert shocked.refusal.detail == "cyclic_world.shift"
    assert shocked.refusal.remedy
    with pytest.raises(StructuredRefusal, match="cyclic_world.shift"):
        require_success(shocked)


def test_experimental_recovery_keeps_each_draw_origin_through_worlds_evidence_and_replay() -> None:
    from antecedent.cyclic import (
        AsymptoticNormal,
        FlatCoefficients,
        KnownDisturbance,
        RecoveredDrawSelector,
        RecoveredResponseSelector,
        RecoveredWorldAnswer,
        RecoveryBootstrap,
        RecoveryPosterior,
        RecoveryPrior,
        consume_evidence,
    )

    posterior = RecoveryPosterior(
        RecoveryPrior(FlatCoefficients(), KnownDisturbance({"A": 1.0, "B": 1.0})), draws=32
    )
    cases = [
        (AsymptoticNormal(64), None, "asymptotic_normal", "sampling_law", 64, 7),
        (
            RecoveryBootstrap(12),
            "independent_subject_trajectories",
            "bootstrap",
            "sampling_law",
            12,
            4,
        ),
        (posterior, None, "posterior", "parameter_posterior", 32, 3),
    ]
    schema = {"id": "feedback", "population": "units", "variables": [["A", "unit"], ["B", "unit"]]}
    for uncertainty, design, origin, meaning, draws, seed in cases:
        prepared = _recovery(uncertainty, sampling_design=design, execution=Execution(seed=seed))
        facts = prepared.inspect()
        assert facts["family"] == "experimental_recovery"
        assert facts["preparation_evaluations"] == 0
        assert facts["draw_origin"]["origin"] == origin
        assert facts["law_meaning"] == meaning
        # The complete clamp design leaves nothing unresolved, known before any fit.
        assert facts["design_unresolved"] == []
        assert facts["unresolved_directions"] == []
        assert {"regimes", "samples", "inference"} <= set(facts["fit_dependencies"])
        assert "worlds" not in facts["fit_dependencies"]
        assert facts["worlds"] == [1, 2]
        if origin == "bootstrap":
            assert facts["draw_origin"]["design"] == "independent_subject_trajectories"
            assert facts["draw_origin"]["rule"] == "row_subject_iid"

        result = require_success(prepared.execute())
        law = result.to_dict()["answer"]["result"]
        assert law["law_meaning"] == meaning
        assert law["law"]["origin"]["origin"] == origin
        assert len(law["law"]["draws"]) == draws
        assert prepared.inspect()["unresolved_directions"] == []
        worlds = require_success(prepared.interventions())
        assert list(worlds) == [1, 2]
        clamped = worlds[2]
        assert isinstance(clamped, RecoveredWorldAnswer)
        assert clamped.law_meaning == meaning
        assert clamped.mean() == pytest.approx(1.0, abs=0.2)
        assert require_success(
            consume_family_artifact(bytes(result.export()), expected_identity=result.identity)
        ).to_dict() == (result.to_dict())
        evidence = require_success(
            prepared.evidence(
                schema,
                [RecoveredResponseSelector(1, 1), RecoveredDrawSelector(1, 0, 1)],
            )
        )
        response, within = evidence.to_dict()["columns"]
        assert response["port"] == meaning
        assert len(response["samples"]) == draws
        assert within["port"] == "interventional_law"
        assert require_success(
            consume_evidence(bytes(evidence.export()), expected_identity=evidence.identity)
        ).to_dict() == (evidence.to_dict())


def test_uncovered_recovery_rows_stay_unresolved_and_withhold_their_mass() -> None:
    from antecedent.cyclic import AsymptoticNormal

    # Without a regime that clamps B, the row into A is uncovered: no draw completes it.
    prepared = _recovery(AsymptoticNormal(16), designs=(None, "A"), execution=Execution(seed=9))
    before = prepared.inspect()
    # Read from the clamp design by the native compiler, before any fit (A is 0, B is 1).
    assert before["design_unresolved"] == [{"target": 0, "missing_clamps": [1]}]
    assert before["unresolved_directions"] == before["design_unresolved"]
    assert "draw_status" not in before
    result = require_success(prepared.execute())
    facts = prepared.inspect()
    assert facts["unresolved_directions"]
    assert facts["unresolved_directions"] == before["unresolved_directions"]
    assert "draw_status" not in facts
    statuses = result.law.draw_status()
    assert statuses.count == len(statuses.items) == 16
    assert {item["status"] for item in statuses.items} == {"unresolved"}
    assert result.law.unresolved(0).items
    clamped = require_success(prepared.interventions())[2]
    assert clamped.mixture_mean is None
    assert clamped.withheld_mass == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(CyclicRefusal, match="withheld") as refused:
        clamped.mean()
    assert "clamps each missing source" in refused.value.next_attempt


def test_recovery_and_world_refusals_name_their_next_attempt() -> None:
    from antecedent import _native
    from antecedent.cyclic import AsymptoticNormal, RecoveryBootstrap, Set, Shift, World

    # A valid question the recovered law cannot answer is retained as the native scientific
    # refusal on the immutable preparation; nothing is fitted.
    shifted = _recovery(
        AsymptoticNormal(4),
        worlds=[World(1), World(3, Shift("A", 1.0))],
        execution=Execution(seed=1),
    )
    assert shifted.refusal is not None
    assert shifted.refusal.code == "route_not_supported"
    assert shifted.refusal.code in _native.runtime_refusal_codes()
    assert shifted.refusal.detail == "cyclic_recovery.hard_clamp_required"
    assert "hard clamps only" in shifted.refusal.message
    assert shifted.refusal.remedy
    assert shifted.inspect()["stage"] == "preparation_refused"

    # A malformed declaration is an invalid argument, not a refusal.
    with pytest.raises(CausalValueError, match="sampling_design"):
        _recovery(RecoveryBootstrap(8), execution=Execution(seed=1))

    # Decided by Rust request compilation; Python passes the declared design through.
    for design in ("clustered_panel", "dependent_time_series"):
        clustered = _recovery(
            AsymptoticNormal(8), sampling_design=design, execution=Execution(seed=1)
        )
        assert clustered.refusal is not None
        assert clustered.refusal.code == "estimator_inference_mismatch"
        assert clustered.refusal.detail == "cyclic_recovery.row_independence_unlicensed"
        assert design in clustered.refusal.message
        assert "bootstrap under the declared" in clustered.refusal.remedy
    assert (
        _recovery(
            AsymptoticNormal(8),
            sampling_design="independent_subject_trajectories",
            execution=Execution(seed=1),
        ).inspect()["family"]
        == "experimental_recovery"
    )

    with pytest.raises(CausalValueError, match="not declared") as unknown:
        _recovery(
            AsymptoticNormal(8), worlds=[World(1, Set("C", 1.0))], execution=Execution(seed=1)
        )
    assert "'A'" in str(unknown.value)

    with pytest.raises(ValueError, match="requested twice") as repeated:
        _feedback_linear().worlds(
            [World(3), World(3, Set("A", 1.0))],
            component_id="feedback",
            source_id="specification",
            certifying_evidence=["independent-bench-test"],
        )
    assert "own id" in str(repeated.value)

    with pytest.raises(CausalValueError, match="sampling_design"):
        _recovery(AsymptoticNormal(8), sampling_design="rows", execution=Execution(seed=1))


def test_system_family_and_families_without_worlds_refuse_interventions() -> None:
    system = model().prepare([[], [{"variable": 0, "value": 2.0}]], [[0.1, 0.2]])
    family = system.to_family()
    facts = family.inspect()
    assert facts["family"] == "system"
    assert facts["preparation_evaluations"] == 0
    assert facts["fit_dependencies"] == ["request.model", "request.fit"]
    assert facts["variables"] == ["a", "b"]
    answer = require_success(family.execute())
    assert answer.to_dict()["answer"]["family"] == "system"
    assert (
        answer.to_dict()["answer"]["result"]["worlds"]
        == require_success(system.execute()).to_dict()["worlds"]
    )
    assert require_success(
        consume_family_artifact(bytes(answer.export()), expected_identity=answer.identity)
    ).to_dict() == (answer.to_dict())
    refused = family.interventions()
    assert refused.refusal is not None
    assert refused.refusal.code == "route_not_supported"
    assert refused.refusal.detail == "cyclic_family.intervention_worlds_unavailable"
    assert "linear_worlds" in refused.refusal.remedy
    with pytest.raises(StructuredRefusal, match="intervention_worlds_unavailable"):
        require_success(refused)


def _ou_rows(seed: int = 7):
    import numpy as np

    rng = np.random.default_rng(seed)
    return [{"values": [0.5 + float(value)], "clamps": []} for value in rng.standard_normal(300)]


def test_dynamic_inference_shares_its_fit_across_methods_and_binds_its_port(
    tmp_path: Path,
) -> None:
    from antecedent.cyclic import (
        DynamicBootstrap,
        DynamicPosterior,
        DynamicQuantitySelector,
        ObservedInformation,
        OuStationaryMean,
    )

    process = OrnsteinUhlenbeck((-1.0,), (0.0,), (4.0,))
    parameters = [{"kind": "drift", "row": 0, "column": 0}, {"kind": "center", "variable": 0}]
    rows = _ou_rows()
    quantities = [OuStationaryMean(0)]

    def infer(method, *, execution=None):
        return process.infer(
            parameters,
            rows,
            method=method,
            execution=execution,
            quantities=quantities,
            sampling_design="independent_subject_trajectories",
        )

    information = infer(ObservedInformation())
    facts = information.inspect()
    assert facts["family"] == "dynamic_inference"
    assert facts["uncertainty_method"] == "information"
    assert facts["law_meaning"] == "covariance_influence"
    assert facts["fit_dependencies"] == ["problem"]
    first = require_success(information.execute()).to_dict()["answer"]["result"]
    observed = first["inference"]["result"]
    assert first["inference"]["method"] == "information"
    # The rows are independent draws, so the stationary mean is the sample mean and its standard
    # error is the sample standard deviation over sqrt(n).
    values = np.array([row["values"][0] for row in rows])
    assert observed["quantity_estimates"][0] == pytest.approx(values.mean(), abs=1e-4)
    assert observed["quantity_standard_errors"][0] == pytest.approx(
        values.std() / math.sqrt(len(values)), abs=1e-6
    )

    bootstrap = require_success(
        information.reconfigure(method=DynamicBootstrap(20), execution=Execution(seed=11))
    )
    result = require_success(bootstrap.execute())
    second = result.to_dict()["answer"]["result"]
    assert second["fit"] == first["fit"]
    assert len(second["inference"]["result"]["replicates"]) == 20
    schema = {"id": "process", "population": "system", "variables": [["x", "unit"]]}
    evidence = require_success(bootstrap.evidence(schema, [DynamicQuantitySelector(0)]))
    assert evidence.to_dict()["columns"][0]["port"] == "sampling_law"
    assert len(evidence.to_dict()["columns"][0]["samples"]) == 20

    posterior = infer(
        DynamicPosterior((-1.0, 0.0), (10.0, 10.0), chains=2, warmup=50, draws=50),
        execution=Execution(seed=13),
    )
    draws = require_success(posterior.execute()).to_dict()["answer"]["result"]["inference"][
        "result"
    ]
    assert len(draws["quantity_draws"]) == 100
    assert require_success(posterior.evidence(schema, [DynamicQuantitySelector(0)])).to_dict()[
        "columns"
    ][0]["port"] == ("parameter_posterior")

    path = tmp_path / "dynamic.ant"
    path.write_bytes(bytes(result.export()))
    replayed = _fresh_replay(
        tmp_path,
        "from antecedent.outcomes import require_success; import json,pathlib; from antecedent.cyclic import consume_family_artifact; "
        + f"print(json.dumps(require_success(consume_family_artifact(pathlib.Path({str(path)!r}).read_bytes(),"
        + f"expected_identity={result.identity!r})).to_dict(),sort_keys=True))",
    )
    assert json.loads(replayed) == result.to_dict()

    with pytest.raises(CausalValueError, match="sampling_design"):
        process.infer(parameters, rows, method=ObservedInformation(), sampling_design=None)


def test_recovered_law_and_its_evidence_replay_in_a_fresh_process(tmp_path: Path) -> None:
    from antecedent.cyclic import RecoveredResponseSelector, RecoveryBootstrap

    prepared = _recovery(
        RecoveryBootstrap(8),
        sampling_design="independent_subject_trajectories",
        execution=Execution(seed=5),
    )
    result = require_success(prepared.execute())
    schema = {"id": "feedback", "population": "units", "variables": [["A", "unit"], ["B", "unit"]]}
    evidence = require_success(prepared.evidence(schema, [RecoveredResponseSelector(1, 1)]))
    family_path = tmp_path / "recovery.ant"
    family_path.write_bytes(bytes(result.export()))
    evidence_path = tmp_path / "recovery-evidence.ant"
    evidence_path.write_bytes(bytes(evidence.export()))
    replayed = _fresh_replay(
        tmp_path,
        "from antecedent.outcomes import require_success; import json,pathlib; from antecedent.cyclic import consume_evidence, "
        + "consume_family_artifact; "
        + f"f=require_success(consume_family_artifact(pathlib.Path({str(family_path)!r}).read_bytes(),"
        + f"expected_identity={result.identity!r})); "
        + f"e=require_success(consume_evidence(pathlib.Path({str(evidence_path)!r}).read_bytes(),"
        + f"expected_identity={evidence.identity!r})); "
        + "print(json.dumps([f.to_dict(),e.to_dict()],sort_keys=True))",
    )
    family_report, evidence_report = json.loads(replayed)
    assert family_report == result.to_dict()
    assert evidence_report == evidence.to_dict()
    assert evidence_report["columns"][0]["port"] == "sampling_law"


def test_prepared_cyclic_handles_expose_their_contract_without_fitting() -> None:
    from antecedent.cyclic import AsymptoticNormal, World

    worlds = _feedback_linear().worlds(
        [World(0)],
        component_id="feedback",
        source_id="specification",
        certifying_evidence=["independent-bench-test"],
    )
    contract = worlds.contract()
    assert contract["family"] == "linear_worlds"
    assert contract["identification_kind"] == "specified_model_conditional"
    assert contract["recovery"] is None
    assert contract["request_identity"] == worlds.inspect()["identity"]
    assert len(contract["lowered_program_identity"]) == 64
    for name in ("target", "identification", "inference_binding", "observation", "data_snapshot"):
        assert len(contract["identities"][name]) == 64
    assert len(contract["cyclic_solution"]) == 64
    assert set(contract["layers"]) == {
        "target",
        "identification",
        "program",
        "inference_binding",
        "observation",
        "data_snapshot",
        "solution",
    }

    recovery = _recovery(AsymptoticNormal(8), execution=Execution(seed=2))
    recovered = recovery.contract()
    assert recovered["identification_kind"] == "experimental_recovery"
    assert len(recovered["recovery"]) == 64
    assert recovered["layers"]["identification"]["unresolved"] == []
    assert recovery.inspect()["preparation_evaluations"] == 0
    assert "draw_status" not in recovery.inspect(), "reading the contract does not fit"

    system = model().prepare([[], [{"variable": 0, "value": 2.0}]], [[0.1, 0.2]])
    carried = system.contract()
    assert carried["family"] == "system"
    assert carried["identification_kind"] == "specified_model_conditional"
    assert system.inspect()["fits_performed"] == 0


def test_a_prepared_cyclic_product_composes_as_one_component() -> None:
    from antecedent import interface_contract as ic
    from antecedent.cyclic import World
    from antecedent.joint_distribution import ScientificQuantity

    def stage(name: str) -> ic.InterfaceContract:
        quantity = ScientificQuantity.outcome("b", units="unit", population="bench", regime="do(a)")
        support = ic.SupportDomain.real()
        return ic.InterfaceContract(
            name,
            produces=[
                ic.ProducedPort("out", quantity, "feedback:b", support, operations=("mean",))
            ],
            consumes=[ic.ConsumedPort("in", quantity, "feedback:b", support, operations=("mean",))],
        )

    product = _feedback_linear().worlds(
        [World(0)],
        component_id="feedback",
        source_id="specification",
        certifying_evidence=["independent-bench-test"],
    )
    parts = [stage("feedback"), stage("reader")]
    wires = [("feedback", "out", "reader", "in")]
    composed = ic.check_composition_with_cyclic(parts, wires, {"feedback": product})
    assert composed.order == ("feedback", "reader")
    # The composition identity binds the cyclic product's program and solve tag.
    assert composed.identity != ic.check_composition(parts, wires).identity
    system = model().prepare([[]], [[0.1, 0.2]])
    assert (
        ic.check_composition_with_cyclic(parts, wires, {"feedback": system}).identity
        != composed.identity
    )

    # The feedback inside the product stays inside its component; among components it is
    # refused with the path named.
    looped = [*wires, ("reader", "out", "feedback", "in")]
    with pytest.raises(ic.InterfaceContractRefusal) as refused:
        ic.check_composition_with_cyclic(parts, looped, {"feedback": product})
    assert refused.value.reason_code == "interface_contract_cyclic_composition"
    assert refused.value.remedy
    with pytest.raises(Exception, match="PreparedCyclicFamily"):
        ic.check_composition_with_cyclic(parts, wires, {"feedback": object()})


# Committed fixtures of parity/cyclic/, replayed in a fresh CPU process. The gate copies this
# file elsewhere and names the fixture directory in ANTECEDENT_CYCLIC_FIXTURES.
_FIXTURES = (
    "specified-feedback-pair",
    "overlapping-loops",
    "recovered-asymptotic-normal",
    "recovered-bootstrap",
    "recovered-unresolved-direction",
    "dynamic-inference-ou",
    "previous-record-format",
)


def _committed_fixtures() -> Path:
    configured = os.environ.get("ANTECEDENT_CYCLIC_FIXTURES")
    if configured:
        root = Path(configured)
        if not (root / _FIXTURES[0] / "family.ant").is_file():
            pytest.fail(f"ANTECEDENT_CYCLIC_FIXTURES={configured} holds no committed fixtures")
        return root
    root = Path(__file__).resolve().parents[2] / "parity" / "cyclic"
    if not (root / _FIXTURES[0] / "family.ant").is_file():
        pytest.skip("committed cyclic fixtures are not next to this copy of the tests")
    return root


def _replay_committed(tmp_path: Path, name: str) -> dict:
    directory = _committed_fixtures() / name
    identity = json.loads((directory / "expected-identity.json").read_text(encoding="utf-8"))
    replayed = _fresh_replay(
        tmp_path,
        "from antecedent.outcomes import require_success; import json,pathlib; from antecedent.cyclic import consume_family_artifact; "
        + f"p=pathlib.Path({str(directory)!r}); "
        + "r=require_success(consume_family_artifact((p/'family.ant').read_bytes(),"
        + f"expected_identity={identity['identity']!r})); "
        + "print(json.dumps(r.to_dict()))",
    )
    report = json.loads(replayed)
    assert report == json.loads((directory / "report.json").read_text(encoding="utf-8"))
    return report


@pytest.mark.parametrize("name", _FIXTURES)
def test_committed_cyclic_fixture_replays_in_a_fresh_process(tmp_path: Path, name: str) -> None:
    report = _replay_committed(tmp_path, name)
    manifest = json.loads(
        (_committed_fixtures() / name / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["fresh_process_cpu_replay"] is True
    expected = json.loads(
        (_committed_fixtures() / name / "expected-identity.json").read_text(encoding="utf-8")
    )
    assert report["identity"] == expected["identity"]


def test_committed_feedback_pair_replays_its_analytic_truths(tmp_path: Path) -> None:
    worlds = _replay_committed(tmp_path, "specified-feedback-pair")["answer"]["result"]
    law = {world["world"]: world["outcome"] for world in worlds}
    covariance = law[0]["covariance"]
    assert covariance[1] / covariance[0] == pytest.approx(1.40625 / 1.8125, abs=1e-12)
    assert covariance[1] / covariance[0] == pytest.approx(0.776, abs=5e-4)
    assert law[2]["mean"][1] - law[1]["mean"][1] == pytest.approx(0.5, abs=1e-12)
    assert law[3]["mean"][1] == pytest.approx(0.625, abs=1e-12)

    loops = _replay_committed(tmp_path, "overlapping-loops")["answer"]["result"]
    clamped = next(world for world in loops if world["world"] == 1)
    assert clamped["failed_mass"] == 1.0


def test_committed_records_refuse_another_identity(tmp_path: Path) -> None:
    root = _committed_fixtures()
    other = json.loads(
        (root / "overlapping-loops" / "expected-identity.json").read_text(encoding="utf-8")
    )["identity"]
    artifact = (root / "specified-feedback-pair" / "family.ant").read_bytes()
    with pytest.raises(Exception, match="identity|mismatch"):
        require_success(consume_family_artifact(artifact, expected_identity=other))


@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_local_cyclic_preserves_original_callback_exception(exception_type):
    from antecedent.cyclic import LocalCyclicModel, LocalMechanism

    original = exception_type("local cyclic callback identity")

    def failing_mechanism(_inputs):
        raise original

    local = LocalCyclicModel(
        (
            LocalMechanism("a", (1,), failing_mechanism),
            LocalMechanism("b", (0,), lambda inputs: (0.5 * inputs[0], [0.5])),
        )
    )
    expected = CausalValueError if exception_type is RuntimeError else exception_type
    with pytest.raises(expected) as raised:
        require_success(local.execute([0.0, 0.0], []))
    if exception_type is RuntimeError:
        assert raised.value.__cause__ is original
    else:
        assert raised.value is original
    assert original.__traceback__ is not None
