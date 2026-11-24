"""Named choices for ``prepare`` and ``analyze``: identification, estimator, profile, refutation.

Each class is a string enum whose values are the engine's stable ids, so a member and its string
are interchangeable (``Estimator.AIPW == "aipw"``). Pass them as ``identifier=``,
``estimator=`` and ``refute=`` to :func:`antecedent.prepare` or :func:`antecedent.analyze`, and
:class:`Latency` members as ``profile=`` on :class:`~antecedent.Frequentist` or
:class:`~antecedent.Bayesian`. A choice that does not fit the query or structure is refused when
the study is prepared, never silently replaced.

Example::

    outcome = antecedent.analyze(
        data,
        graph=graph,
        query=antecedent.AverageEffect(treatment="t", outcome="y"),
        estimator=antecedent.Estimator.AIPW,
        refute=antecedent.Refute.PLACEBO,
    )
"""

from __future__ import annotations

from enum import StrEnum


class Identifier(StrEnum):
    """The identification strategy: how the estimand is expressed in terms of observed data.

    Pass as ``identifier=``. Leaving it ``None`` lets the engine choose for the query and
    structure. The values are the ids of the Rust ``IdentifierId``.

    * ``BACKDOOR_ADJUSTMENT`` -- backdoor adjustment-set search.
    * ``BACKDOOR_EFFICIENT`` -- efficient (optimal) backdoor adjustment set.
    * ``FRONTDOOR`` -- front-door identification.
    * ``IV`` -- instrumental-variable identification.
    * ``RD_SHARP`` -- sharp regression discontinuity.
    * ``TEMPORAL_BACKDOOR_UNFOLDED`` -- backdoor adjustment on the unfolded temporal graph.
    * ``TEMPORAL_IV_UNFOLDED`` -- lagged instrument on the unfolded window graph.
    * ``GENERALIZED_ADJUSTMENT`` -- generalized adjustment, valid for a CPDAG or PAG class.
    * ``GENERAL_ID`` -- Shpitser-Pearl general identification on an ADMG.
    * ``PATH_SPECIFIC_NATURAL`` -- path-restricted natural effects.
    * ``RESPONSE_BACKDOOR`` -- pairwise backdoor identification of response functionals.
    * ``GCM_PARAMETRIC`` -- structural-mechanism identification of counterfactuals.
    * ``TRANSPORT_SID`` -- conservative single-source transport identification.
    * ``INTERFERENCE_DESIGN`` -- the assignment design identifies the exposure contrast.
    * ``RANDOMIZED_DESIGN`` -- randomized assignment identifies the intention-to-treat effect.
    * ``PARALLEL_TRENDS`` -- difference-in-differences under parallel untreated trends.
    * ``CONVEX_DONOR_COUNTERFACTUAL`` -- synthetic control: a convex combination of donor
      outcomes is the untreated counterfactual.
    * ``CONVEX_UNIT_TIME_TRENDS`` -- synthetic difference-in-differences: convex unit and time
      weights reproduce the untreated trend.
    * ``FUZZY_RD_LOCAL_RATIO`` -- fuzzy regression discontinuity: the outcome jump over the
      treatment-receipt jump at the cutoff.
    * ``REGRESSION_KINK_LOCAL_RATIO`` -- regression kink: the outcome slope change over the dose
      slope change at the cutoff.
    * ``POLICY_CONTINUOUS_DOSE_EXCHANGEABILITY`` -- conditional dose response under a supplied
      density and group exchangeability.
    * ``AUTO`` -- every applicable estimand, with no silent estimator choice.
    """

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
    PARALLEL_TRENDS = "quasi.parallel_trends"
    CONVEX_DONOR_COUNTERFACTUAL = "quasi.convex_donor_counterfactual"
    CONVEX_UNIT_TIME_TRENDS = "quasi.convex_unit_time_trends"
    FUZZY_RD_LOCAL_RATIO = "quasi.fuzzy_rd.local_ratio"
    REGRESSION_KINK_LOCAL_RATIO = "quasi.regression_kink.local_ratio"
    POLICY_CONTINUOUS_DOSE_EXCHANGEABILITY = "policy.continuous_dose_exchangeability"
    AUTO = "auto"


class Estimator(StrEnum):
    """The estimator that computes the identified estimand from data.

    Pass as ``estimator=``; ``None`` selects the engine's default for the identified estimand. A
    typed configuration from :mod:`antecedent.estimators` (for example ``Aipw(...)``) can be passed
    instead when the estimator takes options. The values are the ids of the Rust ``EstimatorId``
    and are the names results report as their estimator.

    The ids are grouped by prefix: ``linear.``, ``glm.``, ``propensity.``, ``aipw``, ``dml``,
    ``dr.learner`` and ``causal.forest`` for adjustment estimators; ``frontdoor.``, ``iv.``,
    ``rd.`` and ``local_ratio`` for front-door, instrumental and discontinuity designs;
    ``panel_did`` and ``synthetic_control`` for panel designs; ``randomized.`` for experiments;
    ``temporal.`` and ``longitudinal.`` for time series and sequential regimes; ``response.``
    for response curves and derivatives; ``interference.``, ``transport.`` and ``gcm.`` for those
    families. An id containing ``bayesian`` is the Bayesian counterpart and needs
    :class:`~antecedent.Bayesian` inference. Which estimator is supported for which query,
    structure and inference is listed in the support matrix; an unsupported combination is
    refused (``route_not_supported`` or ``estimator_inference_mismatch``).
    """

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
    PANEL_DID = "panel_did"
    SYNTHETIC_CONTROL = "synthetic_control"
    LOCAL_RATIO = "local_ratio"
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
    RANDOMIZED_ESTIMATED_CUPED_HT_ITT = "randomized.estimated_cuped_ht_itt"
    RANDOMIZED_CLUSTER_FIXED_CUPED_ITT = "randomized.cluster_fixed_cuped_itt"
    RANDOMIZED_ANCOVA_ITT = "randomized.ancova_itt"
    RANDOMIZED_WALD_CACE_LATE = "randomized.wald_cace_late"
    RANDOMIZED_SWITCHBACK_HT_ITT = "randomized.switchback_ht_itt"
    RANDOMIZED_NEYMAN_ITT = "randomized.neyman_itt"
    RANDOMIZED_DR_POLICY = "randomized.dr_policy"
    RANDOMIZED_IPW_POLICY = "randomized.ipw_policy"
    RANDOMIZED_MULTI_ACTION_IPW_POLICY = "randomized.multi_action_ipw_policy"
    RANDOMIZED_IPW_POLICY_NET = "randomized.ipw_policy_net"
    RANDOMIZED_MULTI_ACTION_IPW_POLICY_NET = "randomized.multi_action_ipw_policy_net"
    RANDOMIZED_FACTORIAL_BERNOULLI_HT = "randomized.factorial_bernoulli_ht"
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
    RANDOMIZED_QUANTILE = "randomized.quantile"
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
    """Scientific budget profile: how much computation default replicate counts may use.

    Pass a member (or its string) as ``profile=`` on :class:`~antecedent.Frequentist` or
    :class:`~antecedent.Bayesian`. The estimand, identification status and recorded assumptions
    are the same across profiles; only defaults you leave unset change (bootstrap replicates,
    posterior draws, validation depth, backends allowed), and the result reports what was used.
    Explicit counts always take precedence.

    * ``INTERACTIVE`` -- analytic standard errors or conjugate / Laplace posteriors with few
      draws; no bootstrap, cheap refutation, no HMC.
    * ``STANDARD`` -- the engine's standard defaults (199 bootstrap replicates, 1000 posterior
      draws, placebo and random-common-cause refutation).
    * ``REPORT`` -- more posterior draws and the full validation suite; HMC allowed. The bootstrap
      count (200) is effectively the standard one.

    The values are the ids of the Rust ``LatencyMode``.
    """

    INTERACTIVE = "interactive"
    STANDARD = "standard"
    REPORT = "report"


class Refute(StrEnum):
    """The refutation suite run alongside an estimate.

    Pass as ``refute=`` to :func:`antecedent.prepare` or :func:`antecedent.analyze`, or as
    ``suite=`` to :func:`antecedent.prepared.refute` after estimation. Refutation checks try to
    falsify the estimate (for example with a placebo treatment); passing them does not prove the
    identifying assumptions.

    * ``FULL`` -- the full suite.
    * ``PLACEBO`` -- placebo checks.
    * ``CHEAP`` -- the inexpensive checks.
    * ``NONE`` -- no refutation.

    ``refute=False`` also means no refutation. ``refute=True`` raises
    :class:`~antecedent.errors.CausalTypeError` (``reason_code="invalid_argument"``) because it
    names no suite.
    """

    FULL = "full"
    PLACEBO = "placebo"
    NONE = "none"
    CHEAP = "cheap"


__all__ = ["Estimator", "Identifier", "Latency", "Refute"]
