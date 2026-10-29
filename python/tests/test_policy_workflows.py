"""The fitted-model handle reaches every policy capability, with licences, reason codes and provenance.

One test per acceptance case of the public-workflow area, on simulated linear chains with known
structure (``a -> b -> c``) and the pricing chain. Results are compared with the library's own
shared execution (the retained-noise replay and the one-shot calls), with closed forms where the
model is linear, and with an independent encoding of the released artifact layout.
"""

from __future__ import annotations

import re
import struct

import antecedent
import numpy as np
import pytest
from antecedent.errors import CausalError, CausalUnsupportedError
from antecedent.gcm import fitted_intrinsic_influence, intrinsic_influence
from antecedent.graph import Dag
from antecedent.graph_falsification import evaluate_graph
from antecedent.intervention_callbacks import (
    GENERATED,
    CallbackProgram,
    InterventionCallback,
    required_callbacks,
)
from antecedent.intervention_programs import (
    PolicyProgramValue,
    assign,
    counterfactual_policy,
    counterfactual_temporal_policy,
    decode_policy_program,
    encode_policy_program,
    generated,
    lag,
    maximum,
    policy_program_identity,
    policy_support,
    policy_value,
    prepare_policy,
    sample_policy,
    sample_temporal_policy,
    transform,
    var,
)
from antecedent.outcomes import require_success
from antecedent.prepared import PreparedAnalysis

from _pricing_chain import (
    LINEAR_EDGES,
    PRICING_EDGES,
    fit,
    fit_pricing_chain,
    linear_chain,
    pricing_chain,
)
from _refusal import REGISTERED
from _repo_text import REPO_ROOT


def _bits(draws):
    return np.ascontiguousarray(draws).view(np.uint64)


def _code(error: BaseException) -> str | None:
    """The registered reason code an error carries, as an attribute or in its message."""
    code = getattr(error, "reason_code", None)
    if code is None:
        found = re.search(r"reason=([a-z0-9_]+)", str(error))
        code = found.group(1) if found else None
    return code


def test_the_fitted_handle_reaches_expression_policies_selection_influence_and_noise_replay():
    data = linear_chain(5000, 5286255)
    fitted = fit(data, LINEAR_EDGES)

    # Expression interventions: the one-shot call and the prepared handle are one executor.
    floor = transform("b", maximum(0, generated()))
    one_shot = sample_policy(fitted, [floor], 2000, execution=antecedent.Execution(seed=3))
    prepared = prepare_policy(fitted, [floor])
    np.testing.assert_array_equal(
        _bits(one_shot.draws),
        _bits(require_success(prepared.sample(2000, execution=antecedent.Execution(seed=3))).draws),
    )
    assert one_shot.draws[fitted.names.index("b")].min() >= 0.0
    # A hard level has the closed-form answer: do(a = 1) puts c at 1.5 * 2 = 3.
    level = fitted.sample_do({"a": 1.0}, 4000, execution=antecedent.Execution(seed=2))
    assert abs(level.draws[fitted.names.index("c")].mean() - 3.0) < 0.05  # observed 0.010

    # Mechanism selection: a candidate set and a constraint reach the fit, and the record says so.
    constrained = fit(
        data,
        LINEAR_EDGES,
        candidates=["linear", "additive_spline"],
        constraints={"c": {"required": "linear_gaussian"}},
    )
    record = next(r for r in constrained.mechanism_selection() if r["variable"] == "c")
    assert record["selected"] == "linear_gaussian"
    assert record["rule"] == "required_by_node_constraint"
    assert record["constraint"]["required"] == "linear_gaussian"
    # The named families in order, then `constant` (admissible only for a degenerate column).
    assert record["candidate_set"] == ["linear_gaussian", "linear_spline", "constant"]
    assert {c["family"] for c in record["candidates"]} == {"linear_gaussian"}
    # The spline-interaction family needs two parents; with one it is a recorded failure.
    assert "linear_spline" in {f["family"] for f in record["failed"]}
    assert record["provenance_id"] == "estimate.gcm.fit"

    # Intrinsic influence from the handle equals the standalone call on the same data and seed.
    options = {
        "independent_noise": True,
        "outer_draws": 600,
        "inner_draws": 30,
        "execution": antecedent.Execution(seed=3),
    }
    from_handle = fitted_intrinsic_influence(fitted, "c", **options)
    standalone = intrinsic_influence(data, LINEAR_EDGES, "c", **options)
    assert [s.contribution for s in from_handle.sources] == [
        s.contribution for s in standalone.sources
    ]
    by_source = {s.source: s.contribution for s in from_handle.sources}
    # Closed form: the noise terms of a, b and c reach c with coefficients 3, 2 and 0.5.
    # Observed relative errors 2.9%, 8.0% and 23% (the small source c is the noisiest at 600 outer
    # draws); the bounds are about three times those.
    for source, (truth, relative) in {"a": (9.0, 0.09), "b": (4.0, 0.25), "c": (0.25, 0.35)}.items():
        assert abs(by_source[source] - truth) < relative * truth, (source, by_source[source])

    # Noise replay: the retained noise reproduces the draws and gives the counterfactual.
    draws, noise = fitted.sample_noise({}, 1000, execution=antecedent.Execution(seed=4))
    again = fitted.replay_noise(noise, {})
    np.testing.assert_array_equal(_bits(draws.draws), _bits(again.draws))
    counterfactual = fitted.replay_noise(noise, {"a": 2.0})
    a, b = draws.draws[0], draws.draws[1]
    assert np.all(counterfactual.draws[0] == 2.0)
    assert np.abs((counterfactual.draws[1] - b) - 1.5 * (2.0 - a)).max() < 0.1  # observed 0.035
    with pytest.raises(CausalError):
        fitted.replay_noise(noise[:2], {})
    with pytest.raises(CausalError):
        fitted.replay_noise(
            np.where(np.arange(noise.size).reshape(noise.shape) == 3, np.nan, noise)
        )


def test_a_prepared_policy_reuses_one_licence_and_equals_the_one_shot_results():
    data = pricing_chain(5000, 6512894)
    fitted = fit_pricing_chain(5000, 6512894)
    policy = [transform("price", maximum(1, 0.85 * generated()))]
    handle = prepare_policy(fitted, policy)

    support = handle.support
    assert support.simulation.licensed and support.counterfactual.licensed
    assert not support.identification.licensed
    assert support.identification.reason_code == "policy_route_not_licensed"
    assert support.provenance_id == "model.intervention_programs"
    assert handle.identity == policy_program_identity(list(data), policy)

    for seed in (1, 2):
        np.testing.assert_array_equal(
            _bits(
                require_success(
                    handle.sample(1500, execution=antecedent.Execution(seed=seed))
                ).draws
            ),
            _bits(
                sample_policy(fitted, policy, 1500, execution=antecedent.Execution(seed=seed)).draws
            ),
        )
    outcomes = handle.counterfactual(["demand", "revenue"], execution=antecedent.Execution(seed=1))
    one_shot = counterfactual_policy(
        fitted, policy, ["demand", "revenue"], execution=antecedent.Execution(seed=1)
    )
    for name in outcomes:
        np.testing.assert_array_equal(outcomes[name], one_shot[name])

    value = handle.value("demand", n=4000, execution=antecedent.Execution(seed=5))
    assert value == policy_value(
        fitted, policy, "demand", n=4000, execution=antecedent.Execution(seed=5)
    )
    # Cutting price by 15% raises demand by 0.8 * 0.15 * mean(price): the closed form of the
    # generating model, within a few simulation errors and the fitted slope's error.
    expected = 0.8 * 0.15 * data["price"].mean()
    assert abs(value.mean_difference - expected) < 0.1, (value.mean_difference, expected)  # observed 0.032
    assert value.difference_simulation_stderr < 0.1
    assert value.model_uncertainty == "unquantified"
    assert value.provenance_id == "model.intervention_programs"
    assert value.support == support


def test_the_temporal_counterfactual_replays_a_panel_through_the_shared_rollout():
    rows, steps = 200, 4
    data = [pricing_chain(rows, 9000 + t) for t in range(steps)]
    fitted = fit_pricing_chain(rows, 9000)

    # Without a lag read the first step (the data the model was fitted on) is the static
    # counterfactual, and later steps replay their own noise through the same policy.
    floor = transform("price", maximum(10, 0.85 * generated()))
    path = counterfactual_temporal_policy(
        fitted, [floor], data, window=0, execution=antecedent.Execution(seed=1)
    )
    static = counterfactual_policy(
        fitted, [floor], ["price", "demand", "revenue"], execution=antecedent.Execution(seed=1)
    )
    for index, name in enumerate(["price", "demand", "revenue"]):
        np.testing.assert_array_equal(path[0].draws[index], static[name])
    assert len(path) == steps
    assert all(step.draws[0].min() >= 10.0 for step in path)
    assert not np.array_equal(path[0].draws, path[1].draws)

    # A lagged policy follows its hand recursion from the unit's last observed price.
    start = {name: values[:rows] for name, values in pricing_chain(rows, 77).items()}
    recursion = assign("price", maximum(10, 0.85 * lag("price", 1)))
    rolled = counterfactual_temporal_policy(
        fitted, [recursion], data, window=1, history=[start], execution=antecedent.Execution(seed=1)
    )
    previous = start["price"]
    for step in rolled:
        previous = np.maximum(10.0, 0.85 * previous)
        np.testing.assert_allclose(step.draws[0], previous, rtol=0, atol=1e-12)
    # The counterfactual and the sampled rollout are the same executor on different noise.
    drawn = sample_temporal_policy(
        fitted,
        [recursion],
        n_rows=rows,
        n_steps=steps,
        window=1,
        history=[start],
        execution=antecedent.Execution(seed=1),
    )
    for sampled, replayed in zip(drawn, rolled, strict=True):
        np.testing.assert_allclose(sampled.draws[0], replayed.draws[0], rtol=0, atol=1e-12)

    # A look-back before the first step, and a panel missing a variable, are refused.
    with pytest.raises(CausalError) as early:
        counterfactual_temporal_policy(
            fitted, [recursion], data, window=1, execution=antecedent.Execution(seed=1)
        )
    assert early.value.reason_code == "policy_history_unavailable"
    with pytest.raises(CausalError):
        counterfactual_temporal_policy(
            fitted,
            [floor],
            [{"price": data[0]["price"]}],
            window=0,
            execution=antecedent.Execution(seed=1),
        )


def test_every_new_workflow_refuses_with_a_registered_code_and_names_its_provenance():
    data = linear_chain(2000, 4100773)
    fitted = fit(data, LINEAR_EDGES)
    provenance = REPO_ROOT / "provenance"
    codes = {}

    # Policy: a transform has no observational identification; a read cycle is refused.
    support = policy_support(fitted, [transform("b", maximum(0, generated()))])
    codes["policy_licence"] = support.identification.reason_code
    with pytest.raises(CausalError) as cycle:
        sample_policy(fitted, [assign("a", var("b"))], 10, execution=antecedent.Execution(seed=1))
    codes["policy_cycle"] = _code(cycle.value)
    value = policy_value(
        fitted,
        [transform("b", maximum(0, generated()))],
        "c",
        n=500,
        execution=antecedent.Execution(seed=1),
    )

    # Callback: a runtime binding cannot leave the process.
    callback = InterventionCallback(lambda b: b, inputs=["b"], name="identity_b")
    program = CallbackProgram("b", callback, reads=[GENERATED], mode="transform")
    with pytest.raises(CausalUnsupportedError) as bound:
        encode_policy_program(list(data), [program])
    codes["callback"] = _code(bound.value)
    recoverable = InterventionCallback.from_source(
        "def keep(b):\n    return b\n", entry_point="keep", inputs=["b"]
    )
    record = required_callbacks(
        encode_policy_program(
            list(data), [CallbackProgram("b", recoverable, reads=[GENERATED], mode="transform")]
        )
    )[0]

    # Mechanism selection and graph falsification: invalid settings.
    with pytest.raises(CausalError) as selection:
        fit(data, LINEAR_EDGES, candidates=["not_a_family"])
    codes["mechanism_selection"] = _code(selection.value)
    with pytest.raises(CausalError) as falsification:
        evaluate_graph(data, LINEAR_EDGES, alpha=1.5)
    codes["graph_falsification"] = _code(falsification.value)

    # Intrinsic influence: independence must be declared.
    with pytest.raises(CausalError) as influence:
        fitted_intrinsic_influence(fitted, "c")
    codes["intrinsic_influence"] = _code(influence.value)

    assert codes == {
        "policy_licence": "policy_route_not_licensed",
        "policy_cycle": "policy_dependency_circular",
        "callback": "policy_callback_not_portable",
        "mechanism_selection": "mechanism_candidates_invalid",
        "graph_falsification": "invalid_argument",
        "intrinsic_influence": "route_not_supported",
    }
    for workflow, code in codes.items():
        assert code in REGISTERED, (workflow, code)

    names = {
        "policy": support.provenance_id,
        "policy_value": value.provenance_id,
        "callback": record.provenance_id,
        "mechanism_selection": fitted.mechanism_selection()[0]["provenance_id"],
        "graph_falsification": evaluate_graph(data, LINEAR_EDGES).provenance_id,
        "intrinsic_influence": fitted_intrinsic_influence(
            fitted, "c", independent_noise=True, outer_draws=50, inner_draws=10
        ).provenance_id,
    }
    for workflow, record_id in names.items():
        assert (provenance / f"{record_id}.toml").is_file(), (workflow, record_id)


def _cbor(value):
    """An independent CBOR encoder for the released layout: shortest heads, shortest floats."""

    def head(major, n):
        if n < 24:
            return bytes([major << 5 | n])
        for fmt, info in ((">B", 24), (">H", 25), (">I", 26), (">Q", 27)):
            if n < 1 << (8 * struct.calcsize(fmt)):
                return bytes([major << 5 | info]) + struct.pack(fmt, n)
        raise AssertionError(n)

    if isinstance(value, bool):
        raise AssertionError("no booleans in this layout")
    if isinstance(value, int):
        return head(0, value)
    if isinstance(value, float):
        for fmt, initial in ((">e", 0xF9), (">f", 0xFA), (">d", 0xFB)):
            try:
                packed = struct.pack(fmt, value)
            except OverflowError:
                continue
            if struct.unpack(fmt, packed)[0] == value:
                return bytes([initial]) + packed
        raise AssertionError(value)
    if isinstance(value, str):
        text = value.encode("utf-8")
        return head(3, len(text)) + text
    if isinstance(value, list):
        return head(4, len(value)) + b"".join(_cbor(item) for item in value)
    if isinstance(value, dict):
        return head(5, len(value)) + b"".join(_cbor(k) + _cbor(v) for k, v in value.items())
    raise AssertionError(type(value))


# Identity of the closed-form price floor `max(10, 0.85 * generated)` on variable 0, as pinned by
# the Rust test `the_price_floor_program_has_a_pinned_identity`.
RELEASED_FLOOR_IDENTITY = "4828012ed5bdb86f66847911617623031457a3373151fb18a2b4c54d3a051d04"


def _released_floor_artifact() -> bytes:
    wire = {
        "transform": {
            "variable": 0,
            "expr": {
                "arith": {
                    "op": "max",
                    "lhs": {"const": {"value": 10.0}},
                    "rhs": {
                        "arith": {
                            "op": "mul",
                            "lhs": {"const": {"value": 0.85}},
                            "rhs": "generated",
                        }
                    },
                }
            },
            "reads": {"current": [], "history": []},
        }
    }
    envelope = {
        "version": 1,
        "identity": list(bytes.fromhex(RELEASED_FLOOR_IDENTITY)),
        "interventions": [wire],
    }
    return b"ANTPOLICYPROG" + _cbor(envelope)


def test_an_artifact_in_the_released_layout_still_loads_and_replays_unchanged():
    data = linear_chain(2000, 2366583)
    names = list(data)
    fitted = fit(data, LINEAR_EDGES)
    saved = _released_floor_artifact()

    # Loaded by the current reader: no callback records, the pinned identity, the same program.
    program = transform("a", maximum(10, 0.85 * generated()))
    assert decode_policy_program(names, saved) == (program,)
    assert required_callbacks(saved) == ()
    assert policy_program_identity(names, [program]) == RELEASED_FLOOR_IDENTITY
    # Re-encoding it writes the very bytes that were stored: nothing about the layout moved.
    assert encode_policy_program(names, [program]) == saved

    # Replayed, it gives the draws of the program built directly.
    (loaded,) = decode_policy_program(names, saved)
    np.testing.assert_array_equal(
        _bits(sample_policy(fitted, [loaded], 2000, execution=antecedent.Execution(seed=9)).draws),
        _bits(sample_policy(fitted, [program], 2000, execution=antecedent.Execution(seed=9)).draws),
    )


def test_prepare_reruns_the_policy_workflow_without_recompiling_and_equals_the_one_shot_result():
    data = pricing_chain(5000, 6512894)
    policy = [transform("price", maximum(1, 0.85 * generated()))]
    query = PolicyProgramValue(policy, "demand", n_draws=4000)
    handle = antecedent.prepare(
        data,
        query=query,
        graph=Dag.from_edges(list(data), PRICING_EDGES),
        execution=antecedent.Execution(seed=5),
    )
    assert isinstance(handle, PreparedAnalysis)

    # The program is compiled once, at prepare; the fit is deferred to execution.
    prepared = handle.inspect()
    assert prepared.stage == "prepared"
    assert prepared.details["phase"] == "compiled" and not prepared.details["fit_materialized"]
    first_outcome = antecedent.estimate(handle)
    first = require_success(first_outcome)
    other_seed = require_success(
        antecedent.estimate(antecedent.reconfigure(handle, execution=antecedent.Execution(seed=9)))
    )
    third = require_success(
        antecedent.estimate(antecedent.reconfigure(handle, execution=antecedent.Execution(seed=2)))
    )
    # The preparation is immutable: executing its branches leaves it unexecuted.
    assert handle.inspect().stage == "prepared"
    for result in (first, other_seed, third):
        assert (
            result.preparation.program_compilations,
            result.preparation.model_bindings,
            result.preparation.runs,
        ) == (1, 1, 1)
    assert first.value != other_seed.value

    # Every rerun is the one-shot result on the same fitted model, program and seed, to the bit.
    fitted = fit_pricing_chain(5000, 6512894)
    assert first.value == policy_value(
        fitted, policy, "demand", n=4000, execution=antecedent.Execution(seed=5)
    )
    assert other_seed.value == policy_value(
        fitted, policy, "demand", n=4000, execution=antecedent.Execution(seed=9)
    )
    assert third.value == policy_value(
        fitted, policy, "demand", n=4000, execution=antecedent.Execution(seed=2)
    )
    # Cutting price by 15% raises demand by 0.8 * 0.15 * mean(price), the generating model's
    # closed form, within the fitted slope's error.
    expected = 0.8 * 0.15 * data["price"].mean()
    assert abs(first.value.mean_difference - expected) < 0.1, (first.value, expected)

    # Support decision, contract and provenance are reported on the result.
    support = first.support
    assert support.simulation.licensed and support.counterfactual.licensed
    assert not support.identification.licensed
    assert support.identification.reason_code == "policy_route_not_licensed"
    assert first.provenance_id == "model.intervention_programs"
    contract = first.contract
    assert contract.route == "simulation"
    assert contract.model_uncertainty == "unquantified"
    assert contract.program_identity == policy_program_identity(list(data), policy)
    assert any("independent" in assumption for assumption in contract.assumptions)
    assert set(contract.mechanisms) == set(data)
    report = first_outcome.study.inspect()
    assert report.stage == "executed" and report.program_id == contract.program_identity
    assert report.details["provenance_id"] == "model.intervention_programs"

    # A hard level is licensed for observational identification as well, with the same executor.
    level = antecedent.prepare(
        data,
        query=PolicyProgramValue([assign("price", 20.0)], "demand", n_draws=500),
        graph=PRICING_EDGES,
        execution=antecedent.Execution(seed=1),
    )
    assert require_success(antecedent.estimate(level)).support.identification.licensed

    # New observations refit the model on a branch and rebind the program: still one compilation,
    # and the result is the one-shot result on the refit model. The preparation is unchanged.
    fresh = pricing_chain(5000, 777)
    branch = antecedent.rebind(handle, data=fresh)
    refreshed = require_success(antecedent.estimate(branch))
    assert refreshed.preparation.program_compilations == 1
    assert refreshed.preparation.model_bindings == 1
    assert refreshed.value == policy_value(
        fit(fresh, PRICING_EDGES), policy, "demand", n=4000, execution=antecedent.Execution(seed=5)
    )
    assert require_success(antecedent.estimate(handle)).value == first.value
    # Branching on other data never alters the retained model of the original preparation.
    require_success(antecedent.estimate(antecedent.rebind(handle, data=pricing_chain(5000, 778))))
    assert require_success(antecedent.estimate(handle)).value == first.value

    # Operations that need an identified estimand are refused, as are options it does not take.
    for operation in (handle.refute, lambda: handle.identification):
        with pytest.raises(CausalUnsupportedError) as refused:
            operation()
        assert _code(refused.value) == "option_not_applicable"
    with pytest.raises(CausalUnsupportedError):
        antecedent.prepare(data, query=query, graph=PRICING_EDGES, estimator="linear")
    # The one-shot verb is exactly prepare followed by estimate, so it answers the same question.
    one_shot = require_success(
        antecedent.analyze(
            data,
            query=query,
            graph=Dag.from_edges(list(data), PRICING_EDGES),
            execution=antecedent.Execution(seed=5),
        )
    )
    assert one_shot.value == first.value
    with pytest.raises(CausalError):
        antecedent.prepare(
            data, query=PolicyProgramValue(policy, "missing", n_draws=100), graph=PRICING_EDGES
        )
    with pytest.raises(CausalError):
        antecedent.prepare(
            data,
            query=PolicyProgramValue(
                [transform("nowhere", maximum(1, generated()))], "demand", n_draws=100
            ),
            graph=PRICING_EDGES,
        )
