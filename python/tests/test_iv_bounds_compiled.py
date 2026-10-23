"""Binary-instrument bounds through the compiled ``IvBounds`` study.

The identified set is an interval of average effects, never a point. The known truths are the
frozen Balke-Pearl table (reproduced by an exact-count sample and by the eight-row weighted law),
a perfect-compliance law whose effect is point identified, and the Imbens-Manski interval, which
covers the identified set. Each route is unlicensed and uncalibrated.
"""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalError, CausalUnsupportedError
from antecedent.identify import binary_iv_bounds
from antecedent.quasi import IvBounds

from _repo_text import REPO_ROOT, load_json
from _sealed_loads import assert_answer_kept

TABLE1 = load_json(
    REPO_ROOT / "conformance" / "response" / "binary_iv_balke_pearl_table1" / "expected.json"
)
CELL_ORDER = ((0, 0), (1, 0), (0, 1), (1, 1))
QUERY = IvBounds(outcome="y", treatment="d", instrument="z")


def _counts_sample(cells: list[list[float]], n_per_arm: int = 10_000) -> dict[str, np.ndarray]:
    """A sample whose empirical law is `cells` exactly (arm sizes are equal)."""
    z: list[float] = []
    d: list[float] = []
    y: list[float] = []
    for arm, probabilities in enumerate(cells):
        for (outcome, treatment), probability in zip(CELL_ORDER, probabilities, strict=True):
            count = round(probability * n_per_arm)
            z += [float(arm)] * count
            d += [float(treatment)] * count
            y += [float(outcome)] * count
    return {"z": np.asarray(z), "d": np.asarray(d), "y": np.asarray(y)}


def test_the_compiled_study_reproduces_the_frozen_balke_pearl_bounds() -> None:
    result = ant.analyze(_counts_sample(TABLE1["cells"]), query=QUERY, refute="none", seed=3)
    section = result.iv_bounds
    assert section is not None
    expected = TABLE1["expected"]
    assert section.lower == pytest.approx(expected["lower"], abs=1e-9)
    assert section.upper == pytest.approx(expected["upper"], abs=1e-9)
    assert section.licensed is False and section.calibration == "unmeasured"
    assert section.interval is None and section.posterior is None
    assert section.rows == 20_000 and section.rows_dropped == 0


def test_the_answer_is_an_identified_set_never_a_point() -> None:
    result = ant.analyze(_counts_sample(TABLE1["cells"]), query=QUERY, refute="none", seed=3)
    answer = result.answer
    assert answer.kind == "bounds"
    assert answer.bounds is not None
    assert answer.bounds[0] < answer.bounds[1]
    assert answer.detail is not None and answer.detail.startswith("unlicensed_iv_bounds")
    assert answer.value is None
    assert "partially identified" in result.claim()


def test_perfect_compliance_point_identifies_the_effect() -> None:
    cells = [[0.7, 0.3, 0.0, 0.0], [0.0, 0.0, 0.4, 0.6]]
    result = ant.analyze(_counts_sample(cells), query=QUERY, refute="none")
    section = result.iv_bounds
    assert section is not None
    assert section.lower == pytest.approx(0.3, abs=1e-9)
    assert section.upper == pytest.approx(0.3, abs=1e-9)


def test_the_bootstrap_interval_covers_the_identified_set() -> None:
    rng = np.random.default_rng(11)
    n = 4_000
    z = rng.integers(0, 2, n).astype(float)
    d = (rng.random(n) < 0.2 + 0.6 * z).astype(float)
    y = (rng.random(n) < 0.3 + 0.2 * d).astype(float)
    result = ant.analyze(
        {"z": z, "d": d, "y": y}, query=QUERY, refute="none", bootstrap=200, seed=5
    )
    section = result.iv_bounds
    assert section is not None and section.interval is not None
    interval = section.interval
    assert interval.method == "imbens_manski_row_bootstrap"
    assert interval.lower <= section.lower and interval.upper >= section.upper
    assert interval.replicates > 150
    assert section.lower <= 0.2 <= section.upper


def test_bayesian_inference_reports_the_endpoint_posterior() -> None:
    result = ant.analyze(
        _counts_sample(TABLE1["cells"], 2_000),
        query=QUERY,
        inference=ant.Bayesian(backend="conjugate", n_draws=400),
        refute="none",
        seed=7,
    )
    section = result.iv_bounds
    assert section is not None and section.posterior is not None
    assert section.posterior.n_draws == 400
    assert section.posterior.lower_mean < section.posterior.upper_mean
    assert result.answer.kind == "bounds"


def test_a_weighted_law_is_the_same_as_the_sample_it_describes() -> None:
    cells = TABLE1["cells"]
    sample = ant.analyze(_counts_sample(cells), query=QUERY, refute="none").iv_bounds
    weighted = binary_iv_bounds(cells)
    assert sample is not None
    assert weighted.lower == pytest.approx(sample.lower, abs=1e-9)
    assert weighted.upper == pytest.approx(sample.upper, abs=1e-9)


def test_a_declared_dependent_design_refuses_the_row_level_draw() -> None:
    data = _counts_sample(TABLE1["cells"], 1_000)
    with pytest.raises(CausalError) as refusal:
        ant.analyze(
            data,
            query=QUERY,
            refute="none",
            bootstrap=50,
            sampling_design="dependent_time_series",
        )
    assert "design" in str(refusal.value).lower()


def test_a_column_that_is_not_coded_zero_one_is_refused() -> None:
    data = _counts_sample(TABLE1["cells"], 1_000)
    data["d"] = data["d"] * 2.0
    with pytest.raises(CausalError):
        ant.analyze(data, query=QUERY, refute="none")


def test_graph_and_refutation_options_do_not_apply() -> None:
    data = _counts_sample(TABLE1["cells"], 1_000)
    with pytest.raises(CausalUnsupportedError) as graph:
        ant.analyze(data, query=QUERY, graph=[("z", "d"), ("d", "y")], refute="none")
    assert graph.value.reason_code == "option_not_applicable"
    with pytest.raises(CausalUnsupportedError) as refute:
        ant.analyze(data, query=QUERY, refute="full")
    assert refute.value.reason_code == "option_not_applicable"


def test_the_estimator_follows_the_inference_mode() -> None:
    data = _counts_sample(TABLE1["cells"], 1_000)
    with pytest.raises(CausalError):
        ant.analyze(
            data, query=QUERY, estimator="bayesian.iv_bounds.endpoint_bootstrap", refute="none"
        )


def test_an_exported_result_loads_with_the_same_identified_set() -> None:
    result = ant.analyze(_counts_sample(TABLE1["cells"]), query=QUERY, refute="none", seed=3)
    loaded = ant.load(result.export())
    assert_answer_kept(loaded)
    assert loaded.answer == result.answer
    assert loaded.answer.kind == "bounds"
