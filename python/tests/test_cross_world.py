"""Cross-world edge contrasts on a fixed Markovian DAG (2.2 A5, X8).

The mediation SCM is ``M = 0.8 X + U_M`` and ``Y = 1.7 X + 4 M (+ U_Y)`` with the
disturbances orthogonal to their regressors, so every coefficient is recovered
exactly and each edge set has an exact closed-form effect. The non-separable SCM
makes the outcome convex in the mediator and interacting with the treatment.
Consistency is a named assumption that holds by construction of abduction; it is
not tested from data, so no test here claims to detect its violation.
"""

import numpy as np
import pytest
from antecedent import Admg, Cpdag, Dag, Pag
from antecedent.accepted_graph import AcceptedGraph
from antecedent.cross_world import (
    EdgeIntervention,
    consume_cross_world_artifact,
    path_specific_effect,
)
from antecedent.errors import (
    CausalSerializationError,
    CausalUnsupportedError,
    CausalValueError,
)

EDGES = [("x", "m"), ("x", "y"), ("m", "y")]
CONTROL, ACTIVE = -1.0, 2.0
DELTA = ACTIVE - CONTROL


def mediation_dag():
    return Dag.from_edges(["x", "m", "y"], EDGES)


def linear_data(n=300):
    i = np.arange(n, dtype=float)
    x = np.sin(i * 0.37)
    raw = np.cos(i * 0.91)
    design = np.column_stack([np.ones(n), x])
    u = raw - design @ np.linalg.lstsq(design, raw, rcond=None)[0]
    m = 0.8 * x + u
    return {"x": x, "m": m, "y": 1.7 * x + 4.0 * m}


def orthogonal_noise(raw, *regressors):
    """Unit-variance part of ``raw`` orthogonal to the constant and ``regressors``."""
    design = np.column_stack([np.ones(len(raw)), *regressors])
    resid = raw - design @ np.linalg.lstsq(design, raw, rcond=None)[0]
    return resid / resid.std()


def noisy_linear_data(n=300):
    """``M = 0.8 X + U_M``, ``Y = 1.7 X + 4 M + U_Y``: both disturbances of unit sd."""
    i = np.arange(n, dtype=float)
    x = np.sin(i * 0.37)
    u_m = orthogonal_noise(np.cos(i * 0.91), x)
    m = 0.8 * x + u_m
    u_y = orthogonal_noise(np.sin(i * 1.7 + 0.3), x, m)
    return {"x": x, "m": m, "y": 1.7 * x + 4.0 * m + u_y}, u_m, u_y


def non_separable_data(n=600, outcome_noise=0.0):
    i = np.arange(n, dtype=float)
    x = np.sin(i * 0.37) * 3.0
    u = np.cos(i * 0.91)
    m = 0.8 * x + u
    u_y = outcome_noise * np.sin(i * 1.7 + 0.3)
    return {"x": x, "m": m, "y": 1.7 * x + 0.5 * m + 0.9 * x * m * m + u_y}, u


def intervention(edges):
    return EdgeIntervention("x", "y", CONTROL, ACTIVE, tuple(edges))


def test_exact_scm_truth_for_every_edge_set():
    data = linear_data()
    for mask in range(8):
        chosen = [e for k, e in enumerate(EDGES) if mask >> k & 1]
        truth = 1.7 * DELTA * (("x", "y") in chosen) + 4 * 0.8 * DELTA * (
            ("x", "m") in chosen and ("m", "y") in chosen
        )
        got = path_specific_effect(mediation_dag(), data, intervention(chosen)).point
        assert abs(got - truth) < 1e-9, (chosen, got, truth)


def test_natural_effects_are_edge_sets_and_edge_order_is_immaterial():
    data = linear_data()
    direct = EdgeIntervention.natural_direct("x", "m", "y", control=CONTROL, active=ACTIVE)
    indirect = EdgeIntervention.natural_indirect("x", "m", "y", control=CONTROL, active=ACTIVE)
    assert abs(path_specific_effect(mediation_dag(), data, direct).point - 1.7 * DELTA) < 1e-9
    assert abs(path_specific_effect(mediation_dag(), data, indirect).point - 3.2 * DELTA) < 1e-9
    a = intervention([("m", "y"), ("x", "m")])
    b = intervention([("x", "m"), ("m", "y"), ("x", "m")])
    assert a == b
    ra = path_specific_effect(mediation_dag(), data, a)
    rb = path_specific_effect(mediation_dag(), data, b)
    assert ra.query_text == rb.query_text and ra.point == rb.point


def test_non_separable_mean_answers_per_unit_truth_not_the_conditional_mean_plug_in():
    data, u = non_separable_data()
    direct = EdgeIntervention.natural_direct("x", "m", "y", control=CONTROL, active=ACTIVE)

    def outcome(x, m):
        return 1.7 * x + 0.5 * m + 0.9 * x * m * m

    m0 = 0.8 * CONTROL + u
    per_unit = float(np.mean(outcome(ACTIVE, m0) - outcome(CONTROL, m0)))
    plug_in = float(outcome(ACTIVE, m0.mean()) - outcome(CONTROL, m0.mean()))
    assert abs(per_unit - plug_in) > 1.0
    got = path_specific_effect(mediation_dag(), data, direct, mechanism="non_separable_basis").point
    assert abs(got - per_unit) < 0.25 and abs(got - plug_in) > 1.0
    # The separable fit of the same data cannot read the interaction.
    separable = path_specific_effect(mediation_dag(), data, direct).point
    assert abs(separable - per_unit) > 1.0


def test_edge_answer_differs_from_the_total_effect_under_interaction():
    data, _ = non_separable_data()
    dag = mediation_dag()
    kw = {"mechanism": "non_separable_basis"}
    direct = path_specific_effect(dag, data, intervention([("x", "y")]), **kw).point
    indirect = path_specific_effect(dag, data, intervention([("x", "m"), ("m", "y")]), **kw).point
    total = path_specific_effect(dag, data, intervention(EDGES), **kw).point
    assert abs(direct + indirect - total) > 0.5


def test_recanting_witness_refuses_with_cross_world_not_identified():
    edges = [("x", "w"), ("w", "m"), ("w", "y"), ("m", "y")]
    dag = Dag.from_edges(["x", "w", "m", "y"], edges)
    i = np.arange(80, dtype=float)
    data = {"x": np.sin(i), "w": np.cos(i * 1.3), "m": np.sin(i * 0.7), "y": np.cos(i * 2.1)}
    query = EdgeIntervention("x", "y", 0.0, 1.0, (("x", "w"), ("w", "y")))
    with pytest.raises(CausalUnsupportedError) as raised:
        path_specific_effect(dag, data, query)
    assert raised.value.reason_code == "cross_world_not_identified"
    assert "cross_world.recanting_witness" in str(raised.value)
    # Every edge intervened has no witness.
    path_specific_effect(dag, data, EdgeIntervention("x", "y", 0.0, 1.0, tuple(edges)))


def test_structures_and_inference_outside_the_cell_are_refused():
    data = linear_data(60)
    query = EdgeIntervention.natural_direct("x", "m", "y")
    latent = Admg.from_edges(["x", "m", "y"], EDGES, [("m", "y")])
    for graph in (
        latent,
        Cpdag.from_edges(["x", "m", "y"], [("x", "m", "directed")]),
        Pag.from_marked_edges(["x", "m", "y"], [("x", "m", "tail", "arrow")]),
        AcceptedGraph(mediation_dag()),
    ):
        with pytest.raises(CausalUnsupportedError) as raised:
            path_specific_effect(graph, data, query)
        assert raised.value.reason_code == "cell_not_licensed"
        assert "cross_world.graph_outside_contract" in str(raised.value)
    for uncertainty in ("interval", "bayesian"):
        with pytest.raises(CausalUnsupportedError) as raised:
            path_specific_effect(mediation_dag(), data, query, uncertainty=uncertainty)
        assert raised.value.reason_code == "estimator_inference_mismatch"
        assert "cross_world.interval_requested" in str(raised.value)
    with pytest.raises(CausalValueError):
        EdgeIntervention("x", "x", 0.0, 1.0, ())
    with pytest.raises(CausalValueError):
        EdgeIntervention("x", "y", 1.0, 1.0, ())
    with pytest.raises(CausalValueError):
        path_specific_effect(mediation_dag(), data, query, mechanism="gaussian_process")


def test_artifact_round_trip_matches_the_direct_result_and_is_recomputed():
    data = linear_data(120)
    query = intervention([("x", "m"), ("m", "y")])
    result = path_specific_effect(mediation_dag(), data, query)
    consumed = consume_cross_world_artifact(result.artifact)
    assert consumed.point == result.point
    assert consumed.query_text == result.query_text
    assert consumed.witness == result.witness
    assert consumed.witness["intervened_edges"] == [[0, 1], [1, 2]]
    assert "consistency" in consumed.witness["assumptions"]
    assert abs(consumed.point - 3.2 * DELTA) < 1e-9


def test_a_mutated_artifact_fails_consumption_with_a_typed_error():
    data = linear_data(120)
    result = path_specific_effect(mediation_dag(), data, intervention([("x", "y")]))
    artifact = bytearray(result.artifact)
    with pytest.raises(CausalSerializationError):
        consume_cross_world_artifact(b"not an artifact")
    # Flip bytes across the artifact: every survivor of decoding must still fail
    # digest, witness or point replay rather than return a different answer.
    for position in range(0, len(artifact), max(1, len(artifact) // 97)):
        mutated = bytearray(artifact)
        mutated[position] ^= 0x01
        try:
            replay = consume_cross_world_artifact(bytes(mutated))
        except CausalSerializationError:
            continue
        assert replay.point == result.point


def test_per_unit_effects_follow_each_units_own_mediator_and_outcome_noise():
    # Under the natural indirect edge set each unit's contrast is built from that
    # unit's own mediator disturbance in both worlds, and its outcome disturbance
    # cancels because both worlds carry it. A unit rotation (world 1 reading
    # another unit's disturbance) leaves the mean unchanged but moves every unit.
    indirect = EdgeIntervention.natural_indirect("x", "m", "y", control=CONTROL, active=ACTIVE)

    # Noisy linear SCM: every unit's contrast is the exact constant 4 * 0.8 * DELTA.
    data, _, u_y = noisy_linear_data()
    assert u_y.std() > 0.9
    result = path_specific_effect(mediation_dag(), data, indirect)
    per_unit = np.array(result.unit_effects)
    assert len(per_unit) == 300
    assert np.abs(per_unit - 3.2 * DELTA).max() < 1e-8
    # A rotated outcome disturbance would be off by U_Y(i+1) - U_Y(i) per unit.
    assert np.abs(np.roll(u_y, -1) - u_y).mean() > 0.9

    # Non-separable SCM with a noisy outcome: contrasts vary by unit and match the
    # structural per-unit truth built from the unit's own U_M.
    data, u = non_separable_data(2000, outcome_noise=1.0)

    def outcome(x, m):
        return 1.7 * x + 0.5 * m + 0.9 * x * m * m

    truth = outcome(CONTROL, 0.8 * ACTIVE + u) - outcome(CONTROL, 0.8 * CONTROL + u)
    got = np.array(
        path_specific_effect(
            mediation_dag(), data, indirect, mechanism="non_separable_basis"
        ).unit_effects
    )
    assert truth.std() > 1.0
    mean_error = float(np.abs(got - truth).mean())
    # An unshared outcome disturbance or a rotated mediator disturbance is far off.
    rotated_mediator = outcome(CONTROL, 0.8 * ACTIVE + np.roll(u, -1)) - outcome(
        CONTROL, 0.8 * CONTROL + u
    )
    assert mean_error < 0.2, mean_error
    assert np.abs(got - rotated_mediator).mean() > 4.0 * mean_error


def test_unknown_names_raise_the_invalid_query_detail():
    data = linear_data(60)
    bad_treatment = EdgeIntervention("nope", "y", 0.0, 1.0, (("nope", "y"),))
    bad_edge = EdgeIntervention("x", "y", 0.0, 1.0, (("x", "ghost"),))
    for query in (bad_treatment, bad_edge):
        with pytest.raises(CausalValueError) as raised:
            path_specific_effect(mediation_dag(), data, query)
        assert raised.value.reason_code == "invalid_argument"
        assert "cross_world.invalid_query" in str(raised.value)
    # A graph naming a variable the table lacks is the same malformed query.
    with pytest.raises(CausalValueError) as raised:
        path_specific_effect(
            Dag.from_edges(["x", "m", "y", "z"], EDGES + [("z", "y")]),
            data,
            EdgeIntervention.natural_direct("x", "m", "y"),
        )
    assert "cross_world.invalid_query" in str(raised.value)


def test_the_artifact_binds_its_data_and_reports_the_closed_form_cross_check():
    data, _, _ = noisy_linear_data(200)
    result = path_specific_effect(mediation_dag(), data, intervention([("x", "m"), ("m", "y")]))
    assert result.data_digest and not result.independently_verified
    consumed = consume_cross_world_artifact(result.artifact)
    assert consumed.data_digest == result.data_digest
    assert consumed.independently_verified
    assert abs(consumed.point - 3.2 * DELTA) < 1e-9
    # A different table has a different digest.
    other = {**data, "y": data["y"] + 1e-9 * np.arange(200)}
    assert (
        path_specific_effect(mediation_dag(), other, intervention([("x", "y")])).data_digest
        != result.data_digest
    )
    # The non-separable family replays but has no closed form to cross-check.
    ns, _ = non_separable_data(400, outcome_noise=1.0)
    produced = path_specific_effect(
        mediation_dag(), ns, intervention([("x", "y")]), mechanism="non_separable_basis"
    )
    replayed = consume_cross_world_artifact(produced.artifact)
    assert replayed.point == produced.point and not replayed.independently_verified


@pytest.mark.parametrize("mechanism", ["linear_gaussian", "non_separable_basis"])
def test_abduction_is_exact_inversion_and_no_consistency_claim_is_published(mechanism):
    # `consistency` is named in the witness as an assumption that holds by
    # construction of abduction; the result publishes no consistency residual
    # because none can be tested from data. What is enforced is well-posed
    # abduction, and these fits are exact inversions.
    data, _ = non_separable_data(600, outcome_noise=1.0)
    result = path_specific_effect(
        mediation_dag(), data, intervention([("x", "y")]), mechanism=mechanism
    )
    assert "consistency" in result.witness["assumptions"]
    assert not hasattr(result, "consistency_residual")
