"""What is a target clinic's true prevalence when its records are misclassified?

Sensitivity 0.9 and specificity 0.8 come from a 500-unit validation sample in clinic A and are
carried to clinic B under a declared invariance premise, which is not learned from the target
counts and is not absorbed into the standard error. With 520 of 1,000 clinic B records positive,
the corrected prevalence is (0.52 + 0.8 - 1) / (0.9 + 0.8 - 1); the script asserts it and its
delta-method standard error against those closed forms, and that the standard error is labelled
uncalibrated.

Run: ``python examples/python/transported_measurement.py``.
"""

import math

from antecedent import measurement_models as mm
from antecedent.outcomes import require_success


def main():
    model = mm.MeasurementModel(
        indicator=mm.MeasurementVariable("recorded", "noisy_misclassified", "unit"),
        construct=mm.MeasurementVariable("latent", "latent_scientific", "unit"),
        parameters=mm.BinaryParameters((mm.BinaryRates(0.9, 0.8, 1e-4, 4e-4),), min_youden=0.1),
        population_id="clinic_a",
        regime_id="observational",
        evidence=mm.ParameterEvidence.validation_sample("source-validation", units=500),
        representation=mm.RepresentationVersion("scalar", 1),
    )
    analysis = mm.AnalysisBinding("latent", "clinic_b", "observational")
    invariance = mm.InvarianceDeclaration(
        "clinic_a", "clinic_b", "analyst", ("sensitivity", "specificity")
    )
    result = require_success(
        mm.transport_measurement_model(
            [model], analysis, mm.BinaryPrevalenceData(520, 1000), structure=invariance
        )
    )
    point = (0.52 + 0.8 - 1) / (0.9 + 0.8 - 1)
    variance = (0.52 * 0.48 / 1000 + (1 - point) ** 2 * 4e-4 + point**2 * 1e-4) / (
        0.9 + 0.8 - 1
    ) ** 2
    assert math.isclose(result.estimate.point, point)
    assert math.isclose(result.estimate.standard_error**2, variance)
    assert result.receipt.carried and not result.native_target_data
    print("Corrected target prevalence:", result.estimate.point)
    print("Inference:", result.estimate.inference, "receipt:", result.receipt)
    assert result.estimate.inference == "uncalibrated_standard_error"


if __name__ == "__main__":
    main()
