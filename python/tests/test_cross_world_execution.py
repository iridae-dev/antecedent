"""Cross-world selective execution through the Python surface.

Truth is enumerated in this file by a pure-Python brute force over the exogenous states
(independent of the Rust enumeration) and checked against hand values: the response-type model
(``q = [0.2, 0.5, 0.1, 0.2]``, ``P(U_X = 1) = 0.4``) gives ``P(Y_x0 = 0 | X = 1, Y = 1) = 5/7``
and ``P(Y_x1 = 1, Y_x0 = 0) = 0.5``; the mediation model gives NDE 0.25, NIE 0.225, total
0.475 and the nested joint 0.25.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
import textwrap

import pytest
from antecedent import cross_world_execution as cw
from antecedent.errors import CausalTypeError, CausalValueError
from antecedent.outcomes import require_success

R = cw.CrossWorldExecutionRefusal
Q = [0.2, 0.5, 0.1, 0.2]
RESPONSE = [0, 0, 0, 1, 1, 0, 1, 1]  # never, complier, defier, always; cfg = X


def _response_scm(q: list[float] = Q, population: str = "pop-a") -> cw.FiniteSCM:
    return cw.FiniteSCM(
        population=population,
        history="regime-1",
        exogenous=(cw.ExogenousTerm("u_x", (0.6, 0.4)), cw.ExogenousTerm("u_y", tuple(q))),
        variables=(
            cw.FiniteVariable("x", 2, (), "u_x", (0, 1)),
            cw.FiniteVariable("y", 2, ("x",), "u_y", tuple(RESPONSE)),
        ),
    )


def _do(name: str, level: int) -> cw.World:
    return cw.World(name, {"x": level})


def _mediation_scm() -> cw.FiniteSCM:
    return cw.FiniteSCM(
        population="pop-a",
        history="regime-1",
        exogenous=(
            cw.ExogenousTerm("u_x", (0.5, 0.5)),
            cw.ExogenousTerm("u_m", (0.3, 0.4, 0.1, 0.2)),
            cw.ExogenousTerm("u_y", (0.75, 0.25)),
        ),
        variables=(
            cw.FiniteVariable("x", 2, (), "u_x", (0, 1)),
            cw.FiniteVariable("m", 2, ("x",), "u_m", tuple(RESPONSE)),
            cw.FiniteVariable("y", 2, ("x", "m"), "u_y", (0, 1, 0, 1, 0, 0, 1, 1)),
        ),
    )


def _brute_force(scm: cw.FiniteSCM, evidence: dict[str, int], worlds, event) -> float:
    """Enumerate exogenous states: P(event | evidence) with every world on the same state."""
    pmfs = [e.pmf for e in scm.exogenous]
    total = hit = 0.0
    for states in itertools.product(*[range(len(p)) for p in pmfs]):
        weight = 1.0
        for p, s in zip(pmfs, states, strict=True):
            weight *= p[s]

        def run(interventions, routes, borrowed, states=states):
            vals: dict[str, int] = {}
            for var in scm.variables:
                if var.name in interventions:
                    vals[var.name] = interventions[var.name]
                    continue
                cfg = 0
                for parent in var.parents:
                    src = borrowed.get((parent, var.name))
                    pv = src[parent] if src is not None else vals[parent]
                    levels = next(v.levels for v in scm.variables if v.name == parent)
                    cfg = cfg * levels + pv
                u = states[[e.name for e in scm.exogenous].index(var.exogenous)]
                n_cfg = len(var.table) // len(
                    next(e.pmf for e in scm.exogenous if e.name == var.exogenous)
                )
                vals[var.name] = var.table[u * n_cfg + cfg]
            return vals

        factual = run({}, [], {})
        if any(factual[k] != v for k, v in evidence.items()):
            continue
        total += weight
        computed: dict[str, dict[str, int]] = {}
        for name, interventions, routes in worlds:
            borrowed = {(p, c): computed[src] for p, c, src in routes}
            computed[name] = run(interventions, routes, borrowed)
        hit += weight * event(computed)
    return hit / total


def _joint_request(coupling: str = "shared_abduced_exogenous") -> cw.CrossWorldRequest:
    return cw.CrossWorldRequest(
        scm=_response_scm(),
        units=(cw.FactualUnit("u"),),
        worlds=(_do("x0", 0), _do("x1", 1)),
        quantity=cw.joint_probability([("x1", "y", 1), ("x0", "y", 0)]),
        coupling=coupling,
    )


def test_probability_of_necessity_and_coupled_joint_event_match_enumerated_truth() -> None:
    executor = cw.CrossWorldExecutor()
    necessity = cw.CrossWorldRequest(
        scm=_response_scm(),
        units=(cw.FactualUnit("case", {"x": 1, "y": 1}),),
        worlds=(_do("x0", 0), _do("x1", 1)),
        quantity=cw.probability("x0", "y", 0),
    )
    out = require_success(executor.execute(necessity))
    brute = _brute_force(
        _response_scm(),
        {"x": 1, "y": 1},
        [("x0", {"x": 0}, []), ("x1", {"x": 1}, [])],
        lambda w: float(w["x0"]["y"] == 0),
    )
    assert out.value == pytest.approx(5 / 7, abs=1e-12)
    assert out.value == pytest.approx(brute, abs=1e-12)
    assert out.units[0].evidence_probability == pytest.approx(0.28, abs=1e-12)
    assert out.label == "point_only"
    assert out.work["mechanism_fits"] == 0 and out.work["shared_draws"] == 0

    joint = require_success(executor.execute(_joint_request()))
    assert joint.value == pytest.approx(0.5, abs=1e-12)
    assert abs(joint.value - 0.49) > 1e-3  # the product of the marginals is not the joint law


def test_independent_or_undeclared_coupling_is_never_a_joint_law() -> None:
    executor = cw.CrossWorldExecutor()
    with pytest.raises(R) as error:
        require_success(executor.execute(_joint_request("independent_world_noise")))
    assert error.value.reason_code == "joint_law_required"
    assert error.value.remedy
    with pytest.raises(R) as error:
        require_success(executor.execute(_joint_request("undeclared")))
    assert error.value.reason_code == "joint_law_required"
    assert executor.retained_len() == 0


def test_natural_effects_use_coupled_worlds_and_match_enumeration() -> None:
    route_from_x1 = cw.WorldRoute("m", "y", "x1")
    route_from_x0 = cw.WorldRoute("m", "y", "x0")
    worlds = (
        cw.World("x0_m_of_x1", {"x": 0}, (route_from_x1,)),
        _do("x1", 1),
        _do("x0", 0),
        cw.World("x1_m_of_x0", {"x": 1}, (route_from_x0,)),
    )
    executor = cw.CrossWorldExecutor()

    def run(quantity: cw._Quantity) -> float:
        request = cw.CrossWorldRequest(
            scm=_mediation_scm(), units=(cw.FactualUnit("u"),), worlds=worlds, quantity=quantity
        )
        return require_success(executor.execute(request)).value

    nde = run(require_success(cw.mean_contrast(("x1_m_of_x0", "y"), ("x0", "y"))))
    nie = run(require_success(cw.mean_contrast(("x0_m_of_x1", "y"), ("x0", "y"))))
    total = run(require_success(cw.mean_contrast(("x1", "y"), ("x0", "y"))))
    nested = run(cw.joint_probability([("x1_m_of_x0", "y", 1), ("x0", "y", 0)]))
    assert nde == pytest.approx(0.25, abs=1e-12)
    assert nie == pytest.approx(0.225, abs=1e-12)
    assert total == pytest.approx(0.475, abs=1e-12)
    assert nested == pytest.approx(0.25, abs=1e-12)
    brute = _brute_force(
        _mediation_scm(),
        {},
        [
            ("x0", {"x": 0}, []),
            ("x1_m_of_x0", {"x": 1}, [("m", "y", "x0")]),
        ],
        lambda w: float(w["x1_m_of_x0"]["y"] == 1 and w["x0"]["y"] == 0),
    )
    assert nested == pytest.approx(brute, abs=1e-12)
    # Five quantities, one unit, one abduction.
    assert executor.counters()["abductions"] == 1
    assert executor.counters()["abductions_reused"] == 3


def test_action_only_change_reuses_abduction_and_changed_evidence_rebuilds() -> None:
    evidence = (cw.FactualUnit("case", {"x": 1, "y": 1}),)
    first = cw.CrossWorldRequest(
        scm=_response_scm(),
        units=evidence,
        worlds=(_do("x0", 0), _do("x1", 1)),
        quantity=cw.probability("x0", "y", 0),
    )
    second = cw.CrossWorldRequest(
        scm=_response_scm(),
        units=evidence,
        worlds=(_do("x1", 1), _do("x0", 0), _do("x1b", 1)),
        quantity=require_success(cw.mean_contrast(("x1", "y"), ("x0", "y"))),
    )
    warm = cw.CrossWorldExecutor()
    a = require_success(warm.execute(first))
    assert a.work["abductions"] == 1
    b = require_success(warm.execute(second))
    assert b.work["abductions"] == 0 and b.work["abductions_reused"] == 1
    assert b.work["world_predictions"] == 3
    assert [s["status"] for s in b.stages][:3] == ["declared", "reused", "reused"]
    assert b.value == pytest.approx(1 - 2 / 7, abs=1e-12)
    fresh = require_success(cw.CrossWorldExecutor().execute(second))
    assert fresh.work["abductions"] == 1
    assert fresh.value == b.value  # bit for bit
    assert dict(fresh.receipt) == dict(b.receipt)

    # Changed evidence and a changed mechanism parameter at the same shape rebuild.
    other_evidence = cw.CrossWorldRequest(
        scm=_response_scm(),
        units=(cw.FactualUnit("case", {"x": 1, "y": 0}),),
        worlds=first.worlds,
        quantity=first.quantity,
    )
    c = require_success(warm.execute(other_evidence))
    assert c.work["abductions"] == 1
    assert c.receipt["abduction_digest"] != a.receipt["abduction_digest"]
    changed = cw.CrossWorldRequest(
        scm=_response_scm([0.2, 0.4, 0.2, 0.2]),
        units=(cw.FactualUnit("u"),),
        worlds=(_do("x0", 0), _do("x1", 1)),
        quantity=cw.joint_probability([("x1", "y", 1), ("x0", "y", 0)]),
    )
    d = require_success(warm.execute(changed))
    assert d.work["abductions"] == 1
    assert d.value == pytest.approx(0.4, abs=1e-12)


def test_a_measured_interval_needs_a_declared_sampling_and_an_attesting_record() -> None:
    execution = require_success(cw.seal(_joint_request()))
    with pytest.raises(CausalValueError):
        require_success(execution.measured_interval("undeclared"))  # type: ignore[arg-type]
    # One unit under an independent sample has no determined sampling variance, so the interval
    # is closed whatever the registry holds; the unqualified readings stay closed.
    with pytest.raises(R) as error:
        require_success(execution.measured_interval("independent_sample"))
    assert error.value.reason_code == "cell_not_licensed"
    assert error.value.detail == "cross_world_execution.interval_route_closed"
    with pytest.raises(R) as error:
        execution.interval()
    assert error.value.detail == "cross_world_execution.interval_route_closed"
    with pytest.raises(R) as error:
        require_success(execution.posterior())
    assert error.value.detail == "cross_world_execution.posterior_route_closed"

    # Sixty units under a declared sampling are inside the measured range: the interval is
    # bound to its attesting record and is conditional on the declared model.
    evidence = [{"x": 0, "y": 0}, {"x": 1, "y": 1}, {"x": 1, "y": 0}, {"x": 0, "y": 1}]
    units = tuple(cw.FactualUnit(f"u{i}", evidence[i % 4]) for i in range(60))
    request = cw.CrossWorldRequest(
        scm=_response_scm(),
        units=units,
        worlds=(_do("x0", 0), _do("x1", 1)),
        quantity=cw.probability("x0", "y", 0),
    )
    sealed = require_success(cw.seal(request))
    for sampling, scalar in (
        ("independent_sample", "population_mean_counterfactual_probability"),
        ("fixed_units", "realized_counterfactual_mean"),
    ):
        interval = require_success(sealed.measured_interval(sampling))
        assert interval.scalar == scalar
        assert interval.unit_sampling == sampling
        assert interval.lower <= interval.point <= interval.upper
        assert interval.level == 0.95
        assert interval.record_id.startswith("cov.counterfactual.dag.")
        assert interval.label == "measured_interval_under_the_declared_model"
    # Fewer units than the measured range have no attesting record.
    small = require_success(
        cw.seal(
            cw.CrossWorldRequest(
                scm=_response_scm(),
                units=units[:10],
                worlds=(_do("x0", 0), _do("x1", 1)),
                quantity=cw.probability("x0", "y", 0),
            )
        )
    )
    with pytest.raises(R) as error:
        require_success(small.measured_interval("independent_sample"))
    assert error.value.reason_code == "cell_not_licensed"
    assert error.value.detail == "cross_world_execution.interval_route_closed"


def test_unsupported_requests_are_scientific_refusals() -> None:
    executor = cw.CrossWorldExecutor()
    base = _joint_request()

    with pytest.raises(R) as error:
        require_success(
            executor.execute(
                cw.CrossWorldRequest(
                    scm=base.scm,
                    units=base.units,
                    worlds=base.worlds,
                    quantity=base.quantity,
                    target_population="pop-b",
                )
            )
        )
    assert error.value.reason_code == "cell_not_licensed"
    assert "transport_license_missing" in error.value.missing

    with pytest.raises(R) as error:
        require_success(
            executor.execute(
                cw.CrossWorldRequest(
                    scm=base.scm,
                    units=base.units,
                    worlds=base.worlds,
                    quantity=base.quantity,
                    evidence_history="regime-2",
                )
            )
        )
    assert error.value.detail == "cross_world_execution.incompatible_history"

    for reading in ("interval", "posterior"):
        with pytest.raises(R) as error:
            require_success(
                executor.execute(
                    cw.CrossWorldRequest(
                        scm=base.scm,
                        units=base.units,
                        worlds=base.worlds,
                        quantity=base.quantity,
                        reading=reading,
                    )
                )
            )
        assert error.value.reason_code == "cell_not_licensed"
    with pytest.raises(R) as error:
        require_success(
            executor.execute(
                cw.CrossWorldRequest(
                    scm=base.scm,
                    units=base.units,
                    worlds=base.worlds,
                    quantity=base.quantity,
                    exogenous_inference="posterior_sampling",
                )
            )
        )
    assert error.value.detail == "cross_world_execution.sampled_abduction_not_licensed"
    assert executor.retained_len() == 0
    assert executor.counters()["abductions"] == 0


def test_capability_matrix_and_decide() -> None:
    rows = {row.id: row for row in cw.capability_matrix()}
    assert rows["natural_path_specific.finite_markovian_scm"].status == "executed"
    assert rows["transported.cross_world_execution"].status == "closed"
    for row in rows.values():
        if row.status == "closed":
            assert row.refusal_code and row.remedy
    licensed = cw.decide(quantity="natural_path_specific")
    assert licensed["id"] == "natural_path_specific.finite_markovian_scm"
    shared = cw.decide(quantity="counterfactual_joint", model="finite_shared_exogenous_scm")
    assert shared["id"] == "counterfactual.finite_shared_exogenous_scm.joint"
    with pytest.raises(R) as error:
        cw.decide(quantity="counterfactual_joint", model="semi_markovian")
    assert error.value.reason_code == "cross_world_not_identified"
    assert error.value.detail == "cross_world_execution.latent_confounding_not_identified"
    with pytest.raises(R) as error:
        cw.decide(quantity="transported_cross_world", population="transported")
    assert error.value.missing


def test_artifact_round_trip_mutation_resume_and_fresh_process(tmp_path) -> None:  # type: ignore[no-untyped-def]
    sealed = require_success(cw.seal(_joint_request()))
    assert sealed.value == pytest.approx(0.5, abs=1e-12)
    assert sealed.result.label == "point_only"
    with pytest.raises(R) as error:
        sealed.interval()
    assert error.value.reason_code == "cell_not_licensed"
    with pytest.raises(R):
        require_success(sealed.posterior())
    data = sealed.to_bytes()
    consumed = require_success(
        cw.consume_cross_world_execution_artifact(data, expected_identity=sealed.identity)
    )
    assert consumed.result.value == sealed.value
    assert consumed.inference_claim == "point_only"
    assert dict(consumed.identity) == dict(sealed.identity)

    other = require_success(
        cw.seal(
            cw.CrossWorldRequest(
                scm=_response_scm([0.2, 0.4, 0.2, 0.2]),
                units=(cw.FactualUnit("u"),),
                worlds=(_do("x0", 0), _do("x1", 1)),
                quantity=cw.joint_probability([("x1", "y", 1), ("x0", "y", 0)]),
            )
        )
    )
    with pytest.raises(R) as error:
        require_success(
            cw.consume_cross_world_execution_artifact(data, expected_identity=other.identity)
        )
    assert error.value.detail == "cross_world_execution.artifact_changed"

    # Sampled single-byte edits never replay to a different accepted result.
    for position in range(0, len(data), max(1, len(data) // 64)):
        mutated = bytearray(data)
        mutated[position] ^= 0x55
        try:
            replay = require_success(cw.consume_cross_world_execution_artifact(bytes(mutated)))
        except (R, CausalValueError, CausalTypeError):
            continue
        assert replay.result.value == sealed.value
        assert dict(replay.identity) == dict(sealed.identity)

    # A licensed resume: an action-only change reuses the verified abduction.
    executor = cw.CrossWorldExecutor.from_artifact(data)
    resumed = require_success(
        executor.execute(
            cw.CrossWorldRequest(
                scm=_response_scm(),
                units=(cw.FactualUnit("u"),),
                worlds=(_do("x0", 0), _do("x1", 1), _do("x1b", 1)),
                quantity=cw.probability("x1b", "y", 1),
            )
        )
    )
    assert resumed.work["abductions"] == 0 and resumed.work["abductions_reused"] == 1
    assert resumed.value == pytest.approx(0.7, abs=1e-12)

    path = tmp_path / "execution.bin"
    path.write_bytes(data)
    script = textwrap.dedent(
        "from antecedent.outcomes import require_success\nimport json, sys\nfrom antecedent import cross_world_execution as cw\ndata = open(sys.argv[1], 'rb').read()\nok = require_success(cw.consume_cross_world_execution_artifact(data, expected_identity=json.loads(sys.argv[2])))\ntry:\n    require_success(cw.consume_cross_world_execution_artifact(data, expected_identity={**json.loads(sys.argv[2]), 'coupling': 'undeclared'}))\n    refused = False\nexcept cw.CrossWorldExecutionRefusal:\n    refused = True\nprint(json.dumps({'value': ok.result.value, 'label': ok.result.label, 'refused': refused}))\n"
    )
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-I", "-c", script, str(path), json.dumps(dict(sealed.identity))],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == {
        "value": sealed.value,
        "label": "point_only",
        "refused": True,
    }


def test_inputs_are_typed() -> None:
    executor = cw.CrossWorldExecutor()
    with pytest.raises(CausalTypeError):
        require_success(executor.execute("not a request"))  # type: ignore[arg-type]
    bad = cw.CrossWorldRequest(
        scm=_response_scm(),
        units=(cw.FactualUnit("u"),),
        worlds=(_do("x0", 0),),
        quantity=cw.probability("nowhere", "y", 0),
    )
    with pytest.raises(CausalValueError):
        require_success(executor.execute(bad))
    with pytest.raises(CausalTypeError):
        require_success(cw.consume_cross_world_execution_artifact("text"))  # type: ignore[arg-type]


def _shared_noise_scm(*, shared: bool = True) -> cw.FiniteSCM:
    """X = U and Y = X xor U, with U feeding both (or a separate U_y with the same law)."""
    u_y = "u" if shared else "u_y"
    terms = (cw.ExogenousTerm("u", (0.7, 0.3)),) + (
        () if shared else (cw.ExogenousTerm("u_y", (0.7, 0.3)),)
    )
    return cw.FiniteSCM(
        population="pop-a",
        history="regime-1",
        exogenous=terms,
        variables=(
            cw.FiniteVariable("x", 2, (), "u", (0, 1)),
            cw.FiniteVariable("y", 2, ("x",), u_y, (0, 1, 1, 0)),
        ),
    )


def test_a_shared_exogenous_term_is_executed_and_equals_independent_enumeration() -> None:
    executor = cw.CrossWorldExecutor()
    worlds = (_do("x0", 0), _do("x1", 1))
    # Factual X = 1 forces U = 1, and U is the same draw in the counterfactual world, so
    # Y under do(x0) is 0 xor 1 = 1 with certainty. The independent-noise model gives 0.3.
    evidence = (cw.FactualUnit("case", {"x": 1}),)
    shared = require_success(
        executor.execute(
            cw.CrossWorldRequest(
                scm=_shared_noise_scm(),
                units=evidence,
                worlds=worlds,
                quantity=cw.probability("x0", "y", 1),
            )
        )
    )
    brute = _brute_force(
        _shared_noise_scm(),
        {"x": 1},
        [("x0", {"x": 0}, []), ("x1", {"x": 1}, [])],
        lambda w: float(w["x0"]["y"] == 1),
    )
    assert shared.value == pytest.approx(1.0, abs=1e-12)
    assert shared.value == pytest.approx(brute, abs=1e-12)
    assert shared.units[0].evidence_probability == pytest.approx(0.3, abs=1e-12)

    independent = require_success(
        executor.execute(
            cw.CrossWorldRequest(
                scm=_shared_noise_scm(shared=False),
                units=evidence,
                worlds=worlds,
                quantity=cw.probability("x0", "y", 1),
            )
        )
    )
    assert independent.value == pytest.approx(0.3, abs=1e-12)
    assert independent.value == pytest.approx(
        _brute_force(
            _shared_noise_scm(shared=False),
            {"x": 1},
            [("x0", {"x": 0}, []), ("x1", {"x": 1}, [])],
            lambda w: float(w["x0"]["y"] == 1),
        ),
        abs=1e-12,
    )

    # One realization of U in both worlds: Y_x1 = 1 iff U = 0 and Y_x0 = 1 iff U = 1.
    joint = require_success(
        executor.execute(
            cw.CrossWorldRequest(
                scm=_shared_noise_scm(),
                units=(cw.FactualUnit("prior"),),
                worlds=worlds,
                quantity=cw.joint_probability([("x1", "y", 1), ("x0", "y", 1)]),
            )
        )
    )
    assert joint.value == pytest.approx(0.0, abs=1e-12)


def test_shared_exogenous_execution_replays_and_the_bound_is_not_unidentification() -> None:
    request = cw.CrossWorldRequest(
        scm=_shared_noise_scm(),
        units=(cw.FactualUnit("case", {"x": 1}),),
        worlds=(_do("x0", 0), _do("x1", 1)),
        quantity=cw.probability("x0", "y", 1),
    )
    sealed = require_success(cw.seal(request))
    consumed = require_success(
        cw.consume_cross_world_execution_artifact(
            sealed.to_bytes(), expected_identity=sealed.identity
        )
    )
    assert consumed.result.value == sealed.value == pytest.approx(1.0, abs=1e-12)
    assert sealed.result.matrix_row == "counterfactual.finite_shared_exogenous_scm.marginal"

    # A declared model whose joint exogenous enumeration is too large is unsupported here.
    terms = tuple(cw.ExogenousTerm(f"u{k}", (0.125,) * 8) for k in range(6))
    oversized = cw.FiniteSCM(
        population="pop-a",
        history="regime-1",
        exogenous=terms,
        variables=tuple(
            cw.FiniteVariable(f"v{k}", 2, (), f"u{k}", (0, 1, 0, 1, 0, 1, 0, 1)) for k in range(6)
        ),
    )
    with pytest.raises(R) as error:
        require_success(
            cw.CrossWorldExecutor().execute(
                cw.CrossWorldRequest(
                    scm=oversized,
                    units=(cw.FactualUnit("u"),),
                    worlds=(cw.World("w", {"v0": 0}),),
                    quantity=cw.probability("w", "v1", 1),
                )
            )
        )
    assert error.value.reason_code == "route_not_supported"
    assert error.value.detail == "cross_world_execution.joint_state_bound_exceeded"
