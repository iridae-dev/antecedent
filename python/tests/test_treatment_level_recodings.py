"""One confounded binary treatment coded four ways gives one answer.

The same 2000 units carry the treatment as ``0/1``, ``1/2``, ``10/20`` and the labels
``a/b``. Each coding declares its control and active level on the query; the estimate is
bit-identical across codings and the estimand text is identical up to the level labels it
names. Labels run the declared-level contrast of the numeric path (they are mapped to the arm
codes before preparation), so a refreshed table is encoded by the same declaration and a row at
any other label is refused instead of dropped.
"""

from __future__ import annotations

from typing import Any

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalError, CausalValueError
from antecedent.estimators import Aipw, PropensityMatching, PropensityWeighting
from antecedent.outcomes import require_success

from _refusal import assert_registered_refusal

STATIC_DAG = [("z", "t"), ("z", "y"), ("t", "y")]
SEED = 7866192
EFFECT = 0.5

#: (control, active) of each coding; a column is the 0/1 indicator mapped onto them.
CODINGS: dict[str, tuple[Any, Any]] = {
    "0/1": (0.0, 1.0),
    "1/2": (1.0, 2.0),
    "a/b": ("a", "b"),
    "10/20": (10.0, 20.0),
}


def _indicator(n: int = 2000) -> dict[str, np.ndarray]:
    """``P(T=1|z) = expit(-0.2 + 0.8 z)`` and ``Y = 0.5 T + 0.7 z + 0.5 noise``."""
    rng = np.random.default_rng(SEED)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(-0.2 + 0.8 * z)))).astype(float)
    y = EFFECT * t + 0.7 * z + 0.5 * rng.normal(size=n)
    return {"t": t, "y": y, "z": z}


def _coded(data: dict[str, np.ndarray], control: Any, active: Any) -> dict[str, np.ndarray]:
    treated = data["t"] == 1.0
    if isinstance(control, str):
        column = np.where(treated, active, control)
    else:
        column = np.where(treated, active, control).astype(float)
    return {**data, "t": column}


def _analyze(data: Any, estimator: Any, control: Any, active: Any) -> Any:
    return require_success(
        ant.analyze(
            data,
            graph=STATIC_DAG,
            query=ant.AverageEffect("t", "y", control_level=control, active_level=active),
            estimator=estimator,
            refute="none",
            inference=ant.Frequentist(bootstrap=0),
            execution=ant.Execution(seed=3),
        )
    )


def _neutral(text: str, control: Any, active: Any) -> str:
    """The estimand text with the two declared levels replaced by placeholders."""
    for level, placeholder in ((active, "ACTIVE"), (control, "CONTROL")):
        spelled = repr(level) if isinstance(level, str) else f"{float(level):g}"
        text = text.replace(f"t={spelled})", f"t={placeholder})")
    return text


@pytest.mark.parametrize("estimator", [Aipw, PropensityWeighting, PropensityMatching])
def test_equivalent_treatment_recodings_agree(estimator: Any) -> None:
    data = _indicator()
    results = {
        name: _analyze(_coded(data, control, active), estimator(bootstrap=0), control, active)
        for name, (control, active) in CODINGS.items()
    }
    base = results["0/1"]
    assert abs(base.answer.value - EFFECT) < 0.25
    for name, (control, active) in CODINGS.items():
        result = results[name]
        assert result.answer.value == base.answer.value, name
        # The estimand names the declared levels, and is the same sentence otherwise.
        text = result.estimand_text
        assert text is not None
        assert _neutral(text, control, active) == _neutral(base.estimand_text, *CODINGS["0/1"]), (
            name
        )
    assert results["a/b"].estimand_text == "E[y | do(t='b')] - E[y | do(t='a')]"
    assert results["10/20"].estimand_text == "E[y | do(t=20)] - E[y | do(t=10)]"


def test_a_pandas_table_of_labels_gives_the_same_answer() -> None:
    pd = pytest.importorskip("pandas")
    data = _indicator()
    labelled = _coded(data, "a", "b")
    frame = pd.DataFrame({**labelled, "t": pd.Categorical(labelled["t"])})
    numeric = _analyze(data, Aipw(bootstrap=0), 0.0, 1.0)
    from_labels = _analyze(frame, Aipw(bootstrap=0), "a", "b")
    assert from_labels.answer.value == numeric.answer.value


def test_swapping_labelled_levels_negates_the_estimate() -> None:
    labelled = _coded(_indicator(), "a", "b")
    forward = _analyze(labelled, PropensityWeighting(bootstrap=0), "a", "b")
    reversed_ = _analyze(labelled, PropensityWeighting(bootstrap=0), "b", "a")
    assert abs(forward.answer.value + reversed_.answer.value) < 1e-9
    assert reversed_.estimand_text == "E[y | do(t='a')] - E[y | do(t='b')]"


def test_refresh_encodes_a_new_table_by_the_same_declared_labels() -> None:
    data = _indicator()
    labelled = _coded(data, "a", "b")
    prepared = ant.prepare(
        labelled,
        graph=STATIC_DAG,
        query=ant.AverageEffect("t", "y", control_level="a", active_level="b"),
        estimator=PropensityWeighting(bootstrap=0),
        refute="none",
        inference=ant.Frequentist(bootstrap=0),
        execution=ant.Execution(seed=3),
    )
    first = require_success(prepared.estimate())
    again = require_success(prepared.refresh(labelled))
    assert again.answer.value == first.answer.value
    assert again.estimand_text == first.estimand_text == "E[y | do(t='b')] - E[y | do(t='a')]"
    # A refreshed table with other outcomes keeps the mapping and moves the estimate.
    shifted = {**labelled, "y": labelled["y"] + 1.0 * data["t"]}
    assert require_success(prepared.refresh(shifted)).answer.value > first.answer.value + 0.5


def test_a_row_at_another_label_is_refused_not_dropped() -> None:
    labelled = _coded(_indicator(), "a", "b")
    labelled["t"] = labelled["t"].astype(object)
    labelled["t"][:7] = "c"
    with pytest.raises(CausalError) as raised:
        _analyze(labelled, Aipw(bootstrap=0), "a", "b")
    assert raised.value.reason_code == "treatment_level_undeclared"
    assert_registered_refusal(raised.value)
    assert "7 complete-case rows" in str(raised.value)
    assert "observed treatment levels" in str(raised.value)


def test_levels_are_both_labels_or_both_numbers() -> None:
    labelled = _coded(_indicator(), "a", "b")
    with pytest.raises(CausalValueError, match="both be labels"):
        _analyze(labelled, Aipw(bootstrap=0), "a", 1.0)
    with pytest.raises(CausalValueError, match="must differ"):
        _analyze(labelled, Aipw(bootstrap=0), "a", "a")
