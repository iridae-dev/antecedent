"""Small audit items: integral identifiers, tau-indexed survival contrasts, raw estimator-config
companions, (PAG circle counts are covered in test_discover_pcmci)."""

from __future__ import annotations

import dataclasses

import antecedent as ant
import numpy as np
import pytest
from antecedent import data as frames
from antecedent.errors import CausalValueError
from antecedent.outcomes import require_success
from antecedent.population import PopulationRegistry, target_custom_distribution
from antecedent.state import RefreshMutation, RefreshObservation
from antecedent.survival import difference_at_tau


def _panel():
    unit = {"x": np.arange(6.0), "y": np.arange(6.0)[::-1].copy()}
    return frames.panel([unit, unit])


@pytest.mark.parametrize("bad", [0.5, True, -1, 2**32])
def test_panel_unit_ids_must_be_integers_in_range(bad):
    with pytest.raises(CausalValueError):
        dataclasses.replace(_panel(), unit_ids=(bad, 7))


def test_panel_unit_ids_accept_integral_values_and_normalise_them():
    rebuilt = dataclasses.replace(_panel(), unit_ids=(np.int64(3), 4.0))
    assert rebuilt.unit_ids == (3, 4)
    assert all(type(i) is int for i in rebuilt.unit_ids)


def test_distribution_and_refresh_identifiers_are_never_truncated():
    registry = PopulationRegistry()
    with pytest.raises(CausalValueError):
        registry.insert_distribution(1.5, [1.0])
    with pytest.raises(CausalValueError):
        registry.insert_distribution(1, [1.0], depends_on=[0.5])
    with pytest.raises(CausalValueError):
        target_custom_distribution(2.5)
    with pytest.raises(CausalValueError):
        RefreshMutation.evict(1.5)
    with pytest.raises(CausalValueError):
        RefreshMutation.window(True)
    with pytest.raises(CausalValueError):
        RefreshObservation(id=0.5, x=(1.0,))._wire()
    assert RefreshMutation.evict(3.0)._body["id"] == 3


def test_survival_contrast_is_read_at_tau_not_at_the_last_grid_point():
    times, control, treated = (0.0, 1.0, 2.0), (1.0, 0.8, 0.6), (1.0, 0.9, 0.7)
    assert difference_at_tau(times, treated, control, 1.0) == pytest.approx(0.1)
    assert difference_at_tau(times, treated, control, 2.0) == pytest.approx(0.1)
    with pytest.raises(CausalValueError, match="not a point"):
        difference_at_tau(times, treated, control, 1.5)
    with pytest.raises(CausalValueError, match="equal lengths"):
        difference_at_tau(times, treated[:2], control, 2.0)


def _linear_data(n=200):
    rng = np.random.default_rng(0)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 0.5).astype(float)
    return {"z": z, "t": t, "y": t + z + rng.normal(size=n)}


@pytest.mark.parametrize(
    "config",
    [
        {"se_lag": 2},
        {"fit_lambda": 0.1},
        {"fit_c": 1.3},
        {"fit_kind": "ols", "fit_lambda": 0.1},
        {"fit_kind": "ridge", "fit_lambda": 0.1, "fit_c": 1.3},
    ],
)
def test_raw_estimator_config_refuses_companion_keys_nothing_reads(config):
    with pytest.raises(ant.CausalError):
        require_success(
            ant.analyze(
                _linear_data(),
                graph=[("z", "t"), ("z", "y"), ("t", "y")],
                query=ant.AverageEffect("t", "y"),
                estimator_config=config,
            )
        )
