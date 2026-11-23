import pytest
from antecedent import Execution, ResourceLimits, mnar_fixed_shift
from antecedent import _reason_details as reason_details
from antecedent.state import CancellationToken

KWARGS = {"population": "target", "design": "independent_randomized_arms"}


def _design(
    control: tuple[int, int] = (30, 24), treated: tuple[int, int] = (40, 28)
) -> tuple[list[int], list[float], list[float | None]]:
    ids: list[int] = []
    arm: list[float] = []
    outcome: list[float | None] = []
    for flag, (assigned, observed) in ((0.0, control), (1.0, treated)):
        for i in range(assigned):
            ids.append(len(ids))
            arm.append(flag)
            outcome.append((i * 7 % 11) / 3.0 + flag if i < observed else None)
    return ids, arm, outcome


def _prepare(
    shift: float = 0.8, **design: tuple[int, int]
) -> mnar_fixed_shift.PreparedMnarFixedShift:
    ids, arm, outcome = _design(**design)
    return mnar_fixed_shift.prepare_fixed_shift(ids, arm, outcome, **KWARGS, treated_shift=shift)


def test_interval_is_released_only_under_a_covering_record() -> None:
    # Reference. Treated: 28 of 40 observed, observed mean 74/28, so the missing 12 are
    # shifted by 0.8 and the arm mean is ybar + (12/40) * 0.8; control: 24 of 30 observed, no
    # shift, mean 1.625. The variance is the sum of the observed-mean variances s^2 / n_obs
    # (sample variances of the observed outcomes) plus the variance of the treated missing
    # fraction times the squared shift, p (1 - p) / n * 0.8^2 with p = 12/40. The interval is
    # the Wald interval with z = 1.959964; the calibration status only annotates it. This
    # execution has 70 rows, a declared shift of 0.73 pooled standard deviations and a largest
    # arm missingness of 0.30, so it falls in the cell measured for shifts of 0.5 to 1.0
    # standard deviations and missingness of 0.250 to 0.375 at 50 to 10000 rows.
    ids, arm, outcome = _design()
    treated = [v for a, v in zip(arm, outcome, strict=True) if a == 1.0 and v is not None]
    control = [v for a, v in zip(arm, outcome, strict=True) if a == 0.0 and v is not None]

    def mean(v: list[float]) -> float:
        return sum(v) / len(v)

    def var(v: list[float]) -> float:
        m = mean(v)
        return sum((x - m) ** 2 for x in v) / (len(v) - 1)

    point = (mean(treated) + 0.3 * 0.8) - mean(control)
    variance = (
        var(treated) / len(treated) + var(control) / len(control) + (0.3 * 0.7 / 40.0) * 0.8**2
    )
    half = 1.959964 * variance**0.5

    scalar = _prepare().measured_interval().scalar("treated_minus_control_at_declared_shift")
    assert scalar.point == pytest.approx(point, abs=1e-12)
    assert scalar.level == 0.95
    assert scalar.interval[0] == pytest.approx(point - half, abs=1e-4)
    assert scalar.interval[1] == pytest.approx(point + half, abs=1e-4)
    assert scalar.calibration == "calibrated"
    assert scalar.calibration_reason is None
    assert scalar.record_id == (
        "cov.average_effect.not_applicable.frequentist.delta_normal.l95."
        "mnar_compiled_shift_delta_io_interval.v2_validation_not_applicable."
        "mnar_treated_shift_sd_0p5_1p0_missing_0p250_0p375"
    )


@pytest.mark.parametrize("shift", [-50.0, -1.5, 0.0, 0.8, 1.9, 40.0])
def test_declared_shifts_and_unbalanced_arms_prepare(shift: float) -> None:
    assert _prepare(shift) is not None
    assert _prepare(shift, control=(15, 9), treated=(120, 80)) is not None
    assert _prepare(shift, control=(30, 6), treated=(40, 5)) is not None


@pytest.mark.parametrize("shift", [float("nan"), float("inf")])
def test_non_finite_shift_is_refused(shift: float) -> None:
    ids, arm, outcome = _design()
    with pytest.raises(ValueError, match="finite number"):
        mnar_fixed_shift.prepare_fixed_shift(ids, arm, outcome, **KWARGS, treated_shift=shift)


def test_boolean_shift_is_not_a_number() -> None:
    ids, arm, outcome = _design()
    with pytest.raises(TypeError, match="must be a number"):
        mnar_fixed_shift.prepare_fixed_shift(ids, arm, outcome, **KWARGS, treated_shift=True)


@pytest.mark.parametrize(
    ("design", "shift", "detail"),
    [
        ({"control": (30, 1), "treated": (40, 28)}, 0.0, "insufficient_observations"),
        ({"control": (30, 0), "treated": (40, 28)}, 0.0, "insufficient_observations"),
    ],
)
def test_source_violations_are_refused_with_specific_details(
    design: dict[str, tuple[int, int]], shift: float, detail: str
) -> None:
    with pytest.raises(mnar_fixed_shift.MnarFixedShiftRefusal) as refusal:
        _prepare(shift, **design)
    assert refusal.value.code == "invalid_argument"
    assert refusal.value.detail == f"mnar_fixed_shift.{detail}"


def test_duplicate_subjects_are_refused_across_arms() -> None:
    ids, arm, outcome = _design()
    ids[40] = ids[0]
    with pytest.raises(mnar_fixed_shift.MnarFixedShiftRefusal) as refusal:
        mnar_fixed_shift.prepare_fixed_shift(ids, arm, outcome, **KWARGS, treated_shift=0.0)
    assert refusal.value.detail == reason_details.mnar_fixed_shift.DUPLICATE_SUBJECT


def test_declared_design_is_required() -> None:
    ids, arm, outcome = _design()
    with pytest.raises(ValueError, match="independent_randomized_arms"):
        mnar_fixed_shift.prepare_fixed_shift(
            ids, arm, outcome, population="target", design="observational", treated_shift=0.0
        )


def test_preparation_preserves_cancellation_and_resource_refusals() -> None:
    ids, arm, outcome = _design()
    with pytest.raises(mnar_fixed_shift.MnarFixedShiftRefusal) as resource:
        mnar_fixed_shift.prepare_fixed_shift(
            ids,
            arm,
            outcome,
            **KWARGS,
            treated_shift=0.8,
            execution=Execution(limits=ResourceLimits(memory_bytes=1)),
        )
    assert resource.value.code == "execution_resource_limit"
    token = CancellationToken()
    token.cancel()
    with pytest.raises(mnar_fixed_shift.MnarFixedShiftRefusal) as cancelled:
        mnar_fixed_shift.prepare_fixed_shift(
            ids, arm, outcome, **KWARGS, treated_shift=0.8, execution=Execution(cancel=token)
        )
    assert cancelled.value.code == "execution_cancelled"
