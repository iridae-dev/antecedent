"""Support verdicts of the direct (uncompiled) Python results.

Generated from ``parity/compiled_units.toml`` (``[[direct_status]]``) by
``scripts/generate_compiled_units.py``. Do not edit.
"""

from __future__ import annotations

# caller_supplied_conditional_outcome_predictions: A plug-in g-formula regime value from
#   caller-supplied sequential conditional outcome predictions; the library fits no nuisance and
#   makes no uncertainty claim.
# Reported by: GFormulaValue.
CALLER_SUPPLIED_CONDITIONAL_OUTCOME_PREDICTIONS = "caller_supplied_conditional_outcome_predictions"

# caller_supplied_sequential_Q_and_probabilities: A sequentially augmented regime value from
#   caller-supplied Q scores and sequential probabilities; the library fits no nuisance, does not
#   verify the cross-fitting, and makes no uncertainty claim.
# Reported by: DoublyRobustRegimeValue.
CALLER_SUPPLIED_SEQUENTIAL_Q_AND_PROBABILITIES = "caller_supplied_sequential_Q_and_probabilities"

# caller_supplied_sequential_probabilities: A longitudinal regime value weighted by caller-supplied
#   sequential treatment and censoring probabilities; the library fits no nuisance and makes no
#   uncertainty claim.
# Reported by: RegimeValue.
CALLER_SUPPLIED_SEQUENTIAL_PROBABILITIES = "caller_supplied_sequential_probabilities"

# not_assessed: A member of a batch retarget whose support was not decided: the default, and the
#   verdict of a member that failed for a reason other than a support refusal (an ok member reports
#   `supported`, a refused one `not_supported`).
# Reported by: RetargetMember.
NOT_ASSESSED = "not_assessed"

# off_axis_interval_evidence: A local polynomial ratio (fuzzy discontinuity or kink) interval
#   reported off the support matrix: no support row licenses the direct call, so the interval is
#   evidence, not a supported claim.
# Reported by: LocalPolynomialRatioEstimate.
OFF_AXIS_INTERVAL_EVIDENCE = "off_axis_interval_evidence"

# unsupported_direct_estimator_utility: A randomized-design estimate computed by a direct estimator
#   outside the compiler; no support row licenses it and no interval or calibration claim is made.
# Reported by: ANCOVAEstimate, CUPEDEstimate, ComplierEffectEstimate, SwitchbackEstimate.
UNSUPPORTED_DIRECT_ESTIMATOR_UTILITY = "unsupported_direct_estimator_utility"

# unsupported_direct_inference_utility: An exact randomization test computed directly outside the
#   compiler; the p-value is exact under the declared design, but no support row licenses the direct
#   call.
# Reported by: RandomizationTest.
UNSUPPORTED_DIRECT_INFERENCE_UTILITY = "unsupported_direct_inference_utility"

# unsupported_point_utility: A point estimate (with any reported standard error or variance bound)
#   computed directly outside the compiler; no support row licenses it and no interval or
#   calibration claim is made.
# Reported by: AugmentedPanelDiDEstimate, CategoricalDidEstimate, ConditionalDoseResponseEstimate,
#   CumulativeIncidenceEstimate, DifferenceInDifferencesEstimate, DoublyRobustPolicyEvaluation,
#   IPCWCumulativeIncidenceEstimate, IPCWSurvivalEstimate, InterferenceEstimate,
#   LongitudinalRegimeEstimate, MarginalStructuralModelResult, ObservationalNetworkExposureEstimate,
#   PanelDifferenceInDifferencesEstimate, SaturationEffects, StaggeredAdoptionEstimate,
#   StaggeredEventStudyEstimate, SurvivalEstimate, SyntheticControlEstimate,
#   SyntheticDifferenceInDifferencesEstimate.
UNSUPPORTED_POINT_UTILITY = "unsupported_point_utility"

# unsupported_simultaneous_band: A simultaneous survival-difference band whose arm counts match no
#   band row of the graphless support matrix; a matching band reports `supported` instead.
# Reported by: SurvivalDifferenceBand.
UNSUPPORTED_SIMULTANEOUS_BAND = "unsupported_simultaneous_band"

# unsupported_uncalibrated: A direct result whose interval is reported but whose coverage is not
#   measured; `supported` and `calibration_measured` are false.
# Reported by: PeriodHistoryMatchEstimate, SequentialContrastEstimate.
UNSUPPORTED_UNCALIBRATED = "unsupported_uncalibrated"
