"""A retained fitted effect carried to another population's cohort through its compiled program.

``cohort_decision.transport_cohort_effect`` binds the retained result, its population, the cohort
and the selection-shift declaration into the ``TransportedCohortEffect`` program and executes it.
The program runs the route the direct native entry runs, so the summary, the decision source for
the same cohort, its artifact bytes and every refusal are the direct entry's, bit for bit; only
the compiled summary names its program.

Oracle (independent of both paths): the data of ``test_transported_cohort`` are
``y = t (2 + 0.4 z) + z``, so the cohort mean contrast is ``2 + 0.4 * sum(w z)`` within the
estimator's tolerance, and exactly the weighted sum of the model's own CATE predictions.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest
from antecedent import _native
from antecedent import cohort_decision as cd
from antecedent.outcomes import require_success

from test_transported_cohort import TOLERANCE, W, Z, cohort, fitted_result, shift, source_data

ARTIFACT_ID = "cohort_decision_source"


@pytest.fixture(scope="module")
def fitted_model():
    return fitted_result().fitted_model_from_source(source_data())


def native_arguments(model, *, declared="default", source="clinic_a"):
    declaration = shift() if declared == "default" else declared
    return (
        bytes(model.export()),
        model.parent_claim,
        source,
        json.dumps(cohort()._wire(), allow_nan=False),
        "mmHg",
        None if declaration is None else json.dumps(declaration._wire()),
        ARTIFACT_ID,
    )


def without_program(summary: str) -> dict:
    wire = json.loads(summary)
    wire.pop("program", None)
    return wire


def test_the_compiled_transport_returns_the_direct_entrys_summary_source_and_artifact(fitted_model):
    model = fitted_model
    arguments = native_arguments(model)
    direct = _native.transport_cohort_effect(*arguments)
    compiled = _native.transport_cohort_effect_compiled(*arguments)
    assert direct[3] is None and compiled[3] is None
    assert without_program(compiled[0]) == json.loads(direct[0])
    assert compiled[1] == direct[1]
    assert bytes(compiled[2]) == bytes(direct[2])
    program = json.loads(compiled[0])["program"]
    assert program["route"] == "TransportedCohortEffect"
    assert re.fullmatch(r"[0-9a-f]{64}", program["lowered_program_identity"])


def test_transport_cohort_effect_runs_through_the_compiled_program(fitted_model):
    model = fitted_model
    effect = cd.transport_cohort_effect(
        model,
        cohort=cohort(),
        source_population="clinic_a",
        shift=shift(),
        outcome_units="mmHg",
    )
    direct = json.loads(_native.transport_cohort_effect(*native_arguments(model))[0])
    assert effect.mean_contrast == direct["mean_contrast"]
    assert effect.row_contrasts == tuple(direct["row_contrasts"])
    assert effect.identity == direct["identity"]
    assert effect.program_identity is not None
    assert re.fullmatch(r"[0-9a-f]{64}", effect.program_identity)
    points = np.asarray(require_success(model.predict({"z": Z})).values)
    assert effect.mean_contrast == pytest.approx(float(np.dot(W, points)), abs=TOLERANCE)
    assert effect.mean_contrast == pytest.approx(2.0 + 0.4 * float(np.dot(W, Z)), abs=0.12)
    again = cd.transport_cohort_effect(
        model,
        cohort=cohort(),
        source_population="clinic_a",
        shift=shift(),
        outcome_units="mmHg",
    )
    assert again.program_identity == effect.program_identity
    # The result compares by identity (eq=False), so the repeat is compared field by field.
    assert again.identity == effect.identity
    assert again.mean_contrast == effect.mean_contrast
    assert again.row_contrasts == effect.row_contrasts
    assert again.receipt == effect.receipt


@pytest.mark.parametrize(
    "change",
    [
        {"declared": None},
        {"source": "clinic_c"},
        {"declared": "into_outcome"},
    ],
)
def test_refusals_are_the_direct_entrys_refusals(fitted_model, change):
    model = fitted_model
    if change.get("declared") == "into_outcome":
        change = {"declared": shift(children=("z", "y"))}
    arguments = native_arguments(model, **change)
    direct = _native.transport_cohort_effect(*arguments)
    compiled = _native.transport_cohort_effect_compiled(*arguments)
    assert direct[3] is not None
    assert compiled == direct
