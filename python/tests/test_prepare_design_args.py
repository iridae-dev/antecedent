"""Design and posterior preparation refuse arguments they would otherwise drop."""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import _native
from antecedent.errors import CausalError
from antecedent.quasi import StaggeredRollout

from known_truth import TEMPORAL, temporal_posterior, white_noise_pulse_series


def test_replicates_without_resampling_refused():
    with pytest.raises(CausalError, match="replicates"):
        StaggeredRollout("y", "unit", "period", "cohort", replicates=99)
    query = StaggeredRollout(
        "y", "unit", "period", "cohort", resampling="cluster_bootstrap"
    )
    assert query.replicates == 999


def test_accepted_flag_not_silently_dropped():
    columns = [np.array([float(i % 2) for i in range(8)]) for _ in range(4)]
    with pytest.raises(ValueError, match="accepted"):
        _native.PreparedAnalysis.prepare_panel_did(
            ["y", "treated", "post", "unit"],
            columns,
            "y",
            [True, False] * 4,
            [False, True] * 4,
            [f"u{i}" for i in range(8)],
            [f"c{i}" for i in range(8)],
            accepted=True,
        )


def test_foreign_randomized_design_argument_refused():
    with pytest.raises(CausalError, match="treated_units"):
        _native.PreparedAnalysis.prepare_randomized_effect(
            ["y"],
            [np.array([1.0, 2.0])],
            "y",
            [True, False],
            [0.5, 0.5],
            ["a", "b"],
            ["a", "b"],
            ("control", "treated"),
            "bernoulli",
            treated_units=1,
        )


def test_dbn_fitting_options_refused_beside_supplied_posterior():
    series = white_noise_pulse_series(int(TEMPORAL["n"]), int(TEMPORAL["seed"]))
    with pytest.raises(ValueError, match="supplied posterior"):
        _native.PreparedAnalysis.prepare_dbn_posterior_temporal(
            ["pressure", "defect"],
            [series["pressure"], series["defect"]],
            "pressure",
            "defect",
            max_lag=3,
            posterior=temporal_posterior(),
        )
