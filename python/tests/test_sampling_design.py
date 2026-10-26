"""The sampling design is declared once on the study and held by every resampling estimator."""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalError, CausalValueError
from antecedent.outcomes import require_success

EDGES = [("z", "t"), ("z", "y"), ("t", "y")]
QUERY = ant.AverageEffect(treatment="t", outcome="y")


def _rows(n: int = 120, seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (0.7 * z + rng.normal(size=n) * 0.5 > 0).astype(float)
    y = 1.5 * t + z + rng.normal(size=n) * 0.4
    return {"z": z, "t": t, "y": y}


def _run(**options):
    return require_success(
        ant.prepare(
            _rows(),
            graph=EDGES,
            query=QUERY,
            refute=False,
            execution=ant.Execution(seed=1),
            **options,
        ).estimate()
    )


def test_an_unknown_design_name_is_a_value_error():
    with pytest.raises(CausalValueError, match="sampling_design"):
        ant.prepare(
            _rows(),
            graph=EDGES,
            query=QUERY,
            inference=ant.Frequentist(bootstrap=0),
            sampling_design="row_iid",
        )


def test_a_declared_design_without_resampling_leaves_the_estimate_unchanged():
    baseline = _run(bootstrap=0)
    declared = _run(bootstrap=0, sampling_design="independent_subject_trajectories")
    assert declared.estimate.ate == pytest.approx(baseline.estimate.ate, rel=1e-12)


def test_a_row_bootstrap_is_refused_under_a_trajectory_design():
    with pytest.raises(CausalError):
        _run(bootstrap=20, sampling_design="independent_subject_trajectories")


def test_the_declaration_is_part_of_the_study_identity():
    def snapshot(**options):
        study = ant.prepare(
            _rows(),
            graph=EDGES,
            query=QUERY,
            inference=ant.Frequentist(bootstrap=0),
            refute=False,
            execution=ant.Execution(seed=1),
            **options,
        )
        return study.inspect_plan().data_snapshot_id

    assert snapshot() != snapshot(sampling_design="clustered_panel")
    assert snapshot(sampling_design="clustered_panel") == snapshot(
        sampling_design="clustered_panel"
    )
