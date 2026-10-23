"""Typed identifier / estimator / latency / refute wire ids (Pythonic enums)."""

from __future__ import annotations

from enum import StrEnum


class Identifier(StrEnum):
    """Identification strategies (wire ids match Rust ``IdentifierId``)."""

    BACKDOOR_ADJUSTMENT = "backdoor.adjustment"
    BACKDOOR_EFFICIENT = "backdoor.efficient"
    FRONTDOOR = "frontdoor"
    IV = "iv"
    RD_SHARP = "rd.sharp"
    TEMPORAL_BACKDOOR_UNFOLDED = "temporal.backdoor.unfolded"
    TEMPORAL_IV_UNFOLDED = "temporal.iv.unfolded"
    GENERALIZED_ADJUSTMENT = "generalized.adjustment"
    GENERAL_ID = "general.id"
    PATH_SPECIFIC_NATURAL = "path_specific.natural"
    RESPONSE_BACKDOOR = "response.backdoor"
    GCM_PARAMETRIC = "gcm.parametric"
    TRANSPORT_SID = "transport.sid"
    INTERFERENCE_DESIGN = "interference.design"
    RANDOMIZED_DESIGN = "randomized.design"
    POLICY_CONTINUOUS_DOSE_EXCHANGEABILITY = "policy.continuous_dose_exchangeability"
    AUTO = "auto"


class Estimator(StrEnum):
    """Estimation strategies (wire ids match Rust ``EstimatorId``)."""

    LINEAR_ADJUSTMENT_ATE = "linear.adjustment.ate"
    PROPENSITY_WEIGHTING = "propensity.weighting"
    PROPENSITY_MATCHING = "propensity.matching"
    PROPENSITY_STRATIFICATION = "propensity.stratification"
    DISTANCE_MATCHING = "distance.matching"
    AIPW = "aipw"
    GLM_ADJUSTMENT = "glm.adjustment"
    FRONTDOOR_FUNCTIONAL = "frontdoor.functional"
    FRONTDOOR_LINEAR_TWO_STAGE = "frontdoor.linear_two_stage"
    IV_WALD = "iv.wald"
    IV_2SLS = "iv.2sls"
    RD_SHARP = "rd.sharp"
    BAYESIAN_GCOMP = "bayesian.gcomp"
    BAYESIAN_BASIS_GCOMP = "bayesian.basis.gcomp"
    BAYESIAN_ROBUST_ATE = "bayesian.robust_ate"
    IV_BAYESIAN_JOINT_LINEAR = "iv.bayesian_joint_linear"
    RD_BAYESIAN_LOCAL_LINEAR = "rd.bayesian_local_linear"
    BAYESIAN_PANEL_DID = "bayesian.panel_did"
    BAYESIAN_SYNTHETIC_CONTROL = "bayesian.synthetic_control"
    BAYESIAN_LOCAL_RATIO = "bayesian.local_ratio"
    BAYESIAN_RANDOMIZED_FINITE_POPULATION = "bayesian.randomized_finite_population"
    BAYESIAN_RANDOMIZED_SUPERPOPULATION = "bayesian.randomized_superpopulation"
    BAYESIAN_SWITCHBACK = "bayesian.switchback_effect"
    BAYESIAN_COMPLIER_EFFECT = "bayesian.complier_effect"
    TEMPORAL_LINEAR_ADJUSTMENT = "temporal.linear.adjustment"
    BAYESIAN_TEMPORAL_GCOMP = "bayesian.temporal.gcomp"
    BAYESIAN_TEMPORAL_WHITE_INNOVATION = "bayesian.temporal.white_innovation_linear"
    TEMPORAL_GLM_LINK_CONTRAST = "temporal.glm_link_contrast"
    BAYESIAN_TEMPORAL_GLM_LINK_CONTRAST = "bayesian.temporal.glm_link_contrast"
    TEMPORAL_LAGGED_INSTRUMENT = "temporal.lagged_instrument"
    BAYESIAN_TEMPORAL_LAGGED_INSTRUMENT = "bayesian.temporal.lagged_instrument"
    FUNCTIONAL_DISTRIBUTION = "functional.distribution"
    FUNCTIONAL_EFFECT = "functional.effect"
    BAYESIAN_CONDITIONAL = "conditional.bayesian"
    BAYESIAN_TEMPORAL_MEDIATION = "temporal.mediation.bayesian"
    RESPONSE_BAYESIAN = "response.bayesian"
    TEMPORAL_RESPONSE_BAYESIAN = "response.temporal.bayesian"
    TEMPORAL_SEQUENTIAL_GCOMP = "temporal.sequential.gcomp"
    CONDITIONAL_LINEAR_ADJUSTMENT = "conditional.linear.adjustment"
    TEMPORAL_MEDIATION = "temporal.mediation"
    RESPONSE_KENNEDY_DR = "response.kennedy_dr"
    RESPONSE_RIESZ_ADE = "response.riesz_ade"
    RESPONSE_GAM_DERIVATIVE = "response.gam_derivative"
    RESPONSE_INTERVENTION_GCOMP = "response.intervention_gcomp"
    CELL_AIPW = "cell.aipw"
    TEMPORAL_RESPONSE_GCOMP = "temporal.response.gcomp"
    GCM_FIT = "gcm.fit"
    GCM_FIT_BAYESIAN = "gcm.fit.bayesian"
    GCM_ATTRIBUTION_BAYESIAN = "gcm.attribution.bayesian"
    MEDIATION_LINEAR = "mediation.linear"
    TRANSPORT_TRIAL_IPW = "transport.trial_ipw"
    TRANSPORT_TRIAL_BAYESIAN_BOOTSTRAP = "transport.trial_bayesian_bootstrap"
    INTERFERENCE_HT_HAJEK = "interference.ht_hajek"
    INTERFERENCE_CLUSTER_NEYMAN = "interference.cluster_neyman"
    INTERFERENCE_SATURATION_EXACT = "interference.saturation_exact"
    INTERFERENCE_OBSERVATIONAL_IPW = "interference.observational_ipw"
    INTERFERENCE_BAYESIAN_GAUSSIAN = "interference.bayesian_gaussian"
    INTERFERENCE_CLUSTER_BAYESIAN_BOOTSTRAP = "interference.cluster_bayesian_bootstrap"
    RANDOMIZED_HT_ITT = "randomized.ht_itt"
    RANDOMIZED_FIXED_CUPED_HT_ITT = "randomized.fixed_cuped_ht_itt"
    RANDOMIZED_ANCOVA_ITT = "randomized.ancova_itt"
    RANDOMIZED_WALD_CACE_LATE = "randomized.wald_cace_late"
    RANDOMIZED_SWITCHBACK_HT_ITT = "randomized.switchback_ht_itt"
    RANDOMIZED_NEYMAN_ITT = "randomized.neyman_itt"
    RANDOMIZED_DR_POLICY = "randomized.dr_policy"
    RANDOMIZED_IPW_POLICY = "randomized.ipw_policy"
    RANDOMIZED_MULTI_ACTION_IPW_POLICY = "randomized.multi_action_ipw_policy"
    RANDOMIZED_SURVIVAL_PRODUCT_LIMIT = "randomized.survival_product_limit"
    POLICY_TRIANGULAR_KERNEL_INVERSE_DENSITY = "policy.triangular_kernel_inverse_density"
    LONGITUDINAL_IPW_REGIME = "longitudinal.ipw_regime"
    LONGITUDINAL_G_FORMULA_REGIME = "longitudinal.g_formula_regime"
    LONGITUDINAL_SEQUENTIAL_DR_REGIME = "longitudinal.sequential_dr_regime"
    LONGITUDINAL_MARGINAL_STRUCTURAL_MODEL = "longitudinal.marginal_structural_model"
    DML = "dml"
    DR_LEARNER = "dr.learner"
    CAUSAL_FOREST = "causal.forest"
    BAYESIAN_PIECEWISE_SURVIVAL = "bayesian.piecewise_survival"
    BAYESIAN_DOSE_POLICY = "bayesian.continuous_dose_policy"
    BAYESIAN_RANDOMIZED_QUANTILE = "bayesian.randomized_quantile"
    BAYESIAN_BINARY_POLICY_VALUE = "bayesian.binary_policy_value"
    BAYESIAN_SEQUENTIAL_REGIME_VALUE = "bayesian.sequential_regime_value"
    SEQUENTIAL_CROSSFIT_CONTRAST_WALD = "sequential.crossfit_contrast.wald"
    BAYESIAN_SEQUENTIAL_CROSSFIT_CONTRAST = "bayesian.sequential_crossfit_contrast"
    PERIOD_HISTORY_MATCHING = "matching.period_history_contrast"
    BAYESIAN_PERIOD_HISTORY_OUTCOME_MODEL = "bayesian.period_history_outcome_model"
    TEMPORAL_DESIGN_EXCURSION = "temporal_design.excursion"
    BAYESIAN_TEMPORAL_DESIGN_EXCURSION = "bayesian.temporal_design.excursion"
    TEMPORAL_REGIME_SURVIVAL = "temporal_design.regime_survival"
    BAYESIAN_TEMPORAL_REGIME_SURVIVAL = "bayesian.temporal_design.regime_survival"
    IV_BOUNDS_SHARP = "iv_bounds.balke_pearl"
    BAYESIAN_IV_BOUNDS = "bayesian.iv_bounds.endpoint_bootstrap"


class Latency(StrEnum):
    """Latency tiers (wire ids match Rust ``LatencyMode``)."""

    INTERACTIVE = "interactive"
    STANDARD = "standard"
    REPORT = "report"


class Refute(StrEnum):
    """Refutation suite ids.

    Call sites also accept ``bool`` at the ``False`` value only (meaning "no
    refutation"); ``refute=True`` is rejected — it names no suite, so pass a
    member of this enum (or the equivalent string) instead. See
    :func:`antecedent._coerce.coerce_refute`.
    """

    FULL = "full"
    PLACEBO = "placebo"
    NONE = "none"
    CHEAP = "cheap"


__all__ = ["Estimator", "Identifier", "Latency", "Refute"]
