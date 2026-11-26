"""Point-only mean readings of a trial contrast and a dose grid."""

from antecedent.joint_distribution import ScientificQuantity
from antecedent.transport._smoothed_dose import SmoothedDoseGridPoint
from antecedent.transport._smoothed_dose_learners import SmoothedDoseLearnersEstimate
from antecedent.transport._trial_aipw_learners import TrialAipwLearnersEstimate


def _quantity(functional: str = "mean") -> ScientificQuantity:
    return ScientificQuantity.of(
        "outcome",
        "y",
        units="outcome",
        population="target",
        regime="do(a=1)-do(a=0)",
        functional=functional,
    )


def test_a_trial_contrast_is_one_unverified_mean() -> None:
    estimate = TrialAipwLearnersEstimate(
        estimate=2.0,
        overlap={},
        diagnostics={},
        provenance=(),
        external_nuisances_unverified=True,
        route="TrialAipwLearnerTransport",
        program_identity="abc",
        claim="point_only",
        calibration="not_measured",
        receipts=(),
    )
    source = estimate.as_mean_source(quantity=_quantity("mean_difference"))
    assert source.means == (2.0,)
    assert source.provider_id == "abc"
    assert source.trust == "externally_attested"
    assert source.coordinates[0].functional_id == "mean_difference"


def test_a_dose_grid_is_one_mean_per_dose() -> None:
    point = SmoothedDoseGridPoint(0.0, 1.0, 1.0, 0.0, {}, {}, {}, 0.0)
    other = SmoothedDoseGridPoint(1.0, 3.0, 3.0, 0.0, {}, {}, {}, 9.0)
    estimate = SmoothedDoseLearnersEstimate(
        grid=(point, other),
        uncertainty={},
        overlap={},
        diagnostics={},
        provenance=(),
        folds={},
        external_nuisances_unverified=True,
        route="SmoothedDoseLearnerTransport",
        program_identity="dose",
        claim="point_only",
        calibration="not_measured",
        variable_names=("a", "y"),
        receipts=(),
    )
    source = estimate.as_mean_source(quantity=_quantity())
    assert source.means == (1.0, 3.0)
    assert [item.regime_id for item in source.coordinates] == [
        "do(a=1)-do(a=0):dose=0.0",
        "do(a=1)-do(a=0):dose=1.0",
    ]
    assert source.coordinates[0].units == "outcome"
