"""Decision reliability through the Python surface, against truth computed here in pure Python.

Truth is closed form or enumerated and computed in this file (``math.erf`` for the Gaussian
probability, a bisected quantile for the CVaR, hand counts for the enumerated law); tolerances are
predeclared multiples of the theoretical standard error; every simulation runs at a fixed seed.
"""

from __future__ import annotations

import math
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from antecedent import decision_reliability as dr
from antecedent.errors import CausalTypeError

DR = dr.DecisionReliabilityRefusal


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _phi_inverse(p: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _stream(purpose: str = "selection", replicate: int = 0, seed: int = 11) -> dr.StreamId:
    return dr.StreamId(run_seed=seed, purpose=purpose, replicate=replicate, label="test")


def _draws(columns: dict[str, np.ndarray], **kwargs: object) -> dr.DecisionDraws:
    options: dict[str, object] = {
        "stream": _stream(),
        "source": "known_law",
        "coupling": "shared_simulation",
        "within_stream": "independent",
    }
    options.update(kwargs)
    return dr.DecisionDraws.from_arrays(columns, **options)  # type: ignore[arg-type]


def _gaussian(means: list[float], n: int, seed: int, rho: float = 0.0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(n)
    return {
        f"c{j:02d}": mean + math.sqrt(rho) * common + math.sqrt(1.0 - rho) * rng.standard_normal(n)
        for j, mean in enumerate(means)
    }


# ------------------------------------------------------------------ exact probabilities


def test_two_gaussian_candidates_match_phi_within_four_standard_errors() -> None:
    n = 20_000
    for rho in (0.0, 0.75):
        draws = _draws(_gaussian([1.0, 0.0], n, seed=3, rho=rho))
        report = dr.rank(draws, dr.RankingRequest(baseline="c01"))
        exact = _phi(1.0 / math.sqrt(2.0 * (1.0 - rho)))
        better = report.statements["c00"]["probability_better_than_baseline"]
        theory_se = math.sqrt(exact * (1.0 - exact) / n)
        assert abs(better["value"] - exact) < 4.0 * better["standard_error"], rho
        assert better["standard_error"] == pytest.approx(theory_se, rel=0.05)
        assert better["effective_sample_size"] == pytest.approx(n)
        assert report.label["calibration"] == "uncalibrated"
        assert report.label["paradigm"] == "known_law"
        assert report.label["coupling"] == "shared_simulation"


def test_enumerated_four_state_law_has_exact_ranking_probabilities() -> None:
    draws = _draws(
        {
            "A": np.array([3.0, 3.0, 2.0, 2.0]),
            "B": np.array([2.0, 1.0, 3.0, 2.0]),
            "C": np.array([1.0, 2.0, 1.0, 2.0]),
        }
    )
    report = dr.rank(draws, dr.RankingRequest(baseline="B", top_k=(2,)))
    best = {c: report.probability_best(c) for c in "ABC"}
    # The last state is a three-way tie shared 1/3 each.
    assert best == pytest.approx({"A": 7 / 12, "B": 1 / 3, "C": 1 / 12}, abs=1e-12)
    assert sum(best.values()) == pytest.approx(1.0, abs=1e-12)
    top2 = {c: report.statements[c]["probability_top_k"][0]["estimate"]["value"] for c in "ABC"}
    assert top2 == pytest.approx({"A": 11 / 12, "B": 2 / 3, "C": 5 / 12}, abs=1e-12)
    versus = report.statements["A"]["probability_better_than_baseline"]["value"]
    assert versus == pytest.approx(0.625, abs=1e-12)
    assert report.best_by_expected_utility == "A"


# ------------------------------------------------------------------ risk measures and error


def test_cvar_of_a_known_normal_has_the_influence_function_standard_error() -> None:
    alpha, n = 0.1, 200_000
    values = np.random.default_rng(5).standard_normal(n)
    draws = _draws({"x": values, "y": values - 5.0})
    request = dr.ReliabilityRequest(functional=dr.Functional.cvar(alpha), decision="x")
    estimate = dr.assess(draws, request).report.monte_carlo_error["decision"]
    q = _phi_inverse(alpha)
    truth = -_pdf(q) / alpha
    m1 = q * (1.0 - alpha) - _pdf(q)
    m2 = alpha - q * _pdf(q) + q * q * (1.0 - alpha)
    theory_se = math.sqrt(m2 - m1 * m1) / (alpha * math.sqrt(n))
    assert abs(estimate["value"] - truth) < 4.0 * theory_se
    assert estimate["standard_error"] == pytest.approx(theory_se, rel=0.03)
    assert estimate["method"] == "influence_function"


def test_shared_stream_removes_difference_variance_that_independent_streams_keep() -> None:
    n, rho = 20_000, 0.8
    shared = _gaussian([1.0, 0.0], n, seed=8, rho=rho)
    request = dr.ReliabilityRequest(decision="c00")
    crn = dr.assess(_draws(shared), request).report.monte_carlo_error["differences"][0]
    separate = _gaussian([1.0, 0.0], n, seed=9, rho=0.0)
    streams = {
        "c00": dr.StreamId(1, "selection", 0, "own/c00"),
        "c01": dr.StreamId(1, "selection", 0, "own/c01"),
    }
    independent_draws = dr.DecisionDraws.from_arrays(
        separate,
        stream=streams,
        source="known_law",
        coupling="declared_independent",
        within_stream="independent",
    )
    independent = dr.assess(independent_draws, request).report.monte_carlo_error["differences"][0]
    crn_diff, ind_diff = crn["difference"], independent["difference"]
    # Var(mean difference) = 2 (1 - rho) / n with common random numbers, 2 / n without.
    assert crn_diff["standard_error"] == pytest.approx(math.sqrt(2 * (1 - rho) / n), rel=0.05)
    assert ind_diff["standard_error"] == pytest.approx(math.sqrt(2 / n), rel=0.05)
    assert crn_diff["standard_error"] < 0.5 * ind_diff["standard_error"]
    assert crn_diff["variance_ratio_vs_independent"] == pytest.approx(1 - rho, abs=0.04)
    assert crn_diff["coupling"] == "shared_stream"
    assert ind_diff["coupling"] == "independent_streams"
    assert crn_diff["effective_sample_size"] == pytest.approx(n)


def test_importance_weights_use_kish_effective_sample_size() -> None:
    n = 1_000
    weights = np.where(np.arange(n) % 2 == 0, 1.0, 3.0)
    utilities = (np.arange(n) % 2 == 1).astype(float)
    draws = _draws({"a": utilities, "b": utilities - 1.0}, weights=weights)
    estimate = dr.assess(draws, dr.ReliabilityRequest()).report.monte_carlo_error["decision"]
    # Weighted mean 3/4; sum w^2 (y - m)^2 = 562.5; sum w = 2000; Kish ESS = 2000^2 / 5000.
    assert estimate["value"] == pytest.approx(0.75, abs=1e-12)
    assert estimate["standard_error"] == pytest.approx(math.sqrt(562.5) / 2000.0, abs=1e-12)
    assert estimate["effective_sample_size"] == pytest.approx(800.0, abs=1e-9)
    with pytest.raises(DR) as refused:
        dr.assess(draws, dr.ReliabilityRequest(functional=dr.Functional.quantile(0.5)))
    assert refused.value.reason_code == "route_not_supported"


# ------------------------------------------------------------------ selection


def test_in_sample_maximum_is_biased_up_and_held_out_value_is_unbiased() -> None:
    # 20 candidates of true utility 0, Monte Carlo standard error 0.1: the in-sample maximum
    # overstates the truth by E max of 20 N(0, 0.01) = 1.8675 * 0.1.
    n, candidates, reps = 100, 20, 150
    request = dr.ReliabilityRequest(validation="provided")
    in_sample, held_out, bound = [], [], []
    for replicate in range(reps):
        rng = np.random.default_rng(1_000 + replicate)
        selection = _draws(
            {f"c{j:02d}": rng.standard_normal(n) for j in range(candidates)},
            stream=_stream("selection", replicate),
        )
        validation = _draws(
            {f"c{j:02d}": rng.standard_normal(n) for j in range(candidates)},
            stream=_stream("validation", replicate),
        )
        report = dr.assess(selection, request, validation=validation).report
        chosen = report.selection_uncertainty["selected"]
        in_sample.append(chosen["in_sample"]["value"])
        held_out.append(chosen["held_out"]["held_out"]["estimate"]["value"])
        bound.append(chosen["winners_curse"]["bound"])
        assert chosen["in_sample_label"] == "in_sample"
        assert chosen["status"] == "held_out_validated"
        record = chosen["held_out"]["record"]
        assert record["validation_segment"]["stream"]["purpose"] == "validation"
        assert record["selection_segments"][0]["stream"]["purpose"] == "selection"
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    # Truth is 0 for every candidate. Mean-bias standard error: in-sample 0.054/sqrt(150).
    assert 0.16 <= mean(in_sample) <= 0.215
    assert abs(mean(held_out)) < 4.0 * 0.1 / math.sqrt(reps)
    assert mean(in_sample) <= mean(bound) + 0.02


def test_adaptively_generated_candidates_need_a_search_record() -> None:
    columns = _gaussian([0.0, 0.2, 0.4], 200, seed=2, rho=0.3)
    with pytest.raises(DR) as refused:
        dr.assess(_draws(columns, origin="adaptively_generated"))
    assert refused.value.reason_code == "candidate_search_record_required"
    assert refused.value.detail == "decision_reliability.search_record_required"
    with pytest.raises(DR) as ranking:
        dr.rank(_draws(columns, origin="adaptively_generated"))
    assert ranking.value.reason_code == "candidate_search_record_required"
    record = dr.CandidateSearchRecord(
        generator="hill_climb@1", budget=1_000, n_evaluated=500, generator_stream=_stream("source")
    )
    draws = _draws(columns, origin="adaptively_generated", search_record=record)
    chosen = dr.assess(draws).report.selection_uncertainty["selected"]
    assert chosen["n_considered"] == 500
    assert chosen["winners_curse"]["padded_candidates"] == 497
    small = dr.assess(_draws(columns)).report.selection_uncertainty["selected"]
    assert chosen["winners_curse"]["bound"] > 2.0 * small["winners_curse"]["bound"]
    with pytest.raises(DR) as inconsistent:
        bad = dr.CandidateSearchRecord("g", budget=10, n_evaluated=2)
        dr.assess(_draws(columns, origin="adaptively_generated", search_record=bad))
    assert inconsistent.value.reason_code == "invalid_argument"


def test_undeclared_or_contradicted_dependence_is_refused() -> None:
    columns = _gaussian([0.0, 1.0], 50, seed=4)
    for kwargs, code in (
        ({"coupling": "unknown"}, "sampling_dependence_unknown"),
        ({"within_stream": "unknown"}, "sampling_dependence_unknown"),
    ):
        with pytest.raises(DR) as refused:
            dr.assess(_draws(columns, **kwargs))
        assert refused.value.reason_code == code
    # Declared one shared simulation, but the candidates sit on different streams.
    streams = {"c00": _stream("selection", 0), "c01": _stream("selection", 1)}
    with pytest.raises(DR) as joint:
        dr.assess(_draws(columns, stream=streams))
    assert joint.value.reason_code == "joint_law_required"
    # The defaults are honest: nothing is assumed independent.
    bare = dr.DecisionDraws.from_arrays(columns, stream=_stream())
    with pytest.raises(DR) as default:
        dr.assess(bare)
    assert default.value.reason_code == "sampling_dependence_unknown"


def test_validation_must_be_independent_of_the_selection_draws() -> None:
    columns = _gaussian([0.0, 0.5, 1.0], 200, seed=6)
    selection = _draws(columns)
    request = dr.ReliabilityRequest(validation="provided")
    with pytest.raises(DR) as same:
        dr.assess(selection, request, validation=_draws(columns))
    assert same.value.reason_code == "validation_stream_not_independent"
    assert same.value.remedy
    other = _draws(_gaussian([0.0, 0.5, 1.0], 200, seed=7), stream=_stream("validation"))
    report = dr.assess(selection, request, validation=other).report
    assert report.dependence["validation_independent_of_selection"] is True
    reused = _draws(
        _gaussian([0.0, 0.5, 1.0], 200, seed=7),
        stream=_stream("validation"),
        repeated_evaluations=1,
    )
    with pytest.raises(DR):
        dr.assess(selection, request, validation=reused)


def test_sample_splitting_records_disjoint_segments() -> None:
    columns = _gaussian([0.0] * 10, 200, seed=12)
    request = dr.ReliabilityRequest(validation="split", selection_fraction=0.5, gap=10)
    chosen = dr.assess(_draws(columns), request).report.selection_uncertainty["selected"]
    assert chosen["status"] == "sample_split_validated"
    record = chosen["held_out"]["record"]
    selection = record["selection_segments"][0]
    assert (selection["start"], selection["len"]) == (0, 100)
    assert (record["validation_segment"]["start"], record["validation_segment"]["len"]) == (110, 90)


# ------------------------------------------------------------------ components and closed routes


def test_three_components_are_separate_and_a_combined_number_is_refused() -> None:
    columns = {k: 2.0 * v for k, v in _gaussian([0.0, 0.3], 500, seed=14, rho=0.5).items()}
    draws = _draws(columns, source="posterior")
    result = dr.assess(draws)
    report = result.report
    spread = report.source_uncertainty["spread"]["candidates"][1]["standard_deviation"]
    se = report.monte_carlo_error["candidates"][1]["estimate"]["standard_error"]
    # Spread is a property of the law (about 2); the standard error shrinks with the draws.
    assert spread == pytest.approx(2.0, abs=0.25)
    assert se == pytest.approx(spread / math.sqrt(500), rel=1e-9)
    assert "selected" in report.selection_uncertainty
    assert _draws(columns, source="known_law") is not None
    known = dr.assess(_draws(columns, source="known_law")).report
    assert known.source_uncertainty == "known_law"
    with pytest.raises(DR) as combined:
        dr.assess(draws, dr.ReliabilityRequest(reporting="combined"))
    assert combined.value.reason_code == "uncertainty_components_not_combined"
    with pytest.raises(DR) as again:
        result.combined_uncertainty()
    assert again.value.detail == "decision_reliability.combined_uncertainty"


def test_every_published_ranking_or_interval_route_is_closed() -> None:
    result = dr.assess(_draws(_gaussian([0.0, 1.0], 100, seed=1)))
    assert result.report.coordinate["candidate_set_size"] == 2
    assert result.report.coordinate["mc_budget"] == 100
    assert len(result.report.closed_routes) == 4
    for route in (
        "ranking_probability",
        "optimality_confidence_set",
        "utility_interval",
        "selection_corrected_interval",
    ):
        with pytest.raises(DR) as refused:
            result.publish(route)
        assert refused.value.reason_code == "cell_not_licensed"
        assert refused.value.detail == "decision_reliability.cell_not_licensed"
        with pytest.raises(DR):
            dr.publish(route, result.report.coordinate)
    with pytest.raises(ValueError):
        dr.publish("not_a_route", result.report.coordinate)


# ------------------------------------------------------------------ artifact


def _sealed() -> tuple[dr.DecisionReliability, dr.DecisionDraws]:
    columns = _gaussian([0.2, 0.0, 0.4], 300, seed=21, rho=0.4)
    draws = _draws(columns)
    request = dr.ReliabilityRequest(ranking=dr.RankingRequest(baseline="c01", top_k=(2,)))
    return dr.assess(draws, request, artifact_id="python-test"), draws


def test_artifact_round_trips_and_retained_expectations_are_enforced() -> None:
    result, draws = _sealed()
    digest, set_digest = draws.digests()
    consumed = dr.consume_decision_reliability_artifact(
        result.export(),
        draws_digest=digest,
        candidate_set_digest=set_digest,
        selection_stream=_stream(),
        decided_id=result.report.decided_id,
    )
    assert consumed.report.raw == result.report.raw
    assert consumed.inspection["standing"] == "uncalibrated_point_diagnostic"
    for kwargs in (
        {"decided_id": "not-the-decision"},
        {"draws_digest": "0" * 64},
        {"selection_stream": _stream("selection", 9)},
    ):
        with pytest.raises(DR):
            dr.consume_decision_reliability_artifact(result.export(), **kwargs)
    damaged = bytearray(result.export())
    damaged[len(damaged) // 2] ^= 0xFF
    with pytest.raises(DR):
        dr.consume_decision_reliability_artifact(bytes(damaged))
    with pytest.raises(DR):
        dr.consume_decision_reliability_artifact(b"not an artifact")


def test_a_fresh_process_consumes_the_artifact_and_recomputes_the_report(tmp_path) -> None:
    result, draws = _sealed()
    path = tmp_path / "reliability.bin"
    path.write_bytes(result.export())
    script = textwrap.dedent(
        """
        import json, sys
        from antecedent import decision_reliability as dr
        consumed = dr.consume_decision_reliability_artifact(
            open(sys.argv[1], "rb").read(), draws_digest=sys.argv[2]
        )
        print(json.dumps({
            "decided": consumed.report.decided_id,
            "value": consumed.report.monte_carlo_error["decision"]["value"],
        }))
        """
    )
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-I", "-c", script, str(path), draws.digests()[0]],
        capture_output=True,
        text=True,
        check=True,
    )
    import json

    seen = json.loads(completed.stdout)
    assert seen["decided"] == result.report.decided_id
    assert seen["value"] == result.report.monte_carlo_error["decision"]["value"]


def test_inputs_are_typed() -> None:
    with pytest.raises(CausalTypeError):
        dr.assess("not draws")  # type: ignore[arg-type]
    with pytest.raises(CausalTypeError):
        dr.DecisionDraws.from_arrays({}, stream=_stream())
    with pytest.raises(ValueError):
        dr.Functional.quantile(1.5)
    with pytest.raises(ValueError):
        dr.DecisionDraws.from_arrays(
            {"a": [1.0, float("nan"), 2.0, 3.0]}, stream=_stream(), coupling="shared_simulation"
        )
