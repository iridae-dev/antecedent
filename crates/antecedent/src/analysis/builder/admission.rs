//! Admission checks for declared Bayesian controls and likelihoods.

use super::{
    CausalError, CausalQuery, DataInput, EstimatorId, GraphClass, IdentifierId, InferenceMode,
};

/// Refuse [`crate::PosteriorDrawControls`] where they would not be enforced or are not met.
///
/// The controls bound the draws behind a Bayesian g-computation interval of a tabular
/// average effect on one fixed `Dag`; any other route refuses them rather than ignore them.
/// A draw count below the backend's supported minimum is refused naming it, never rewritten.
pub(in crate::analysis) fn refuse_unsupported_draw_controls(
    query: &CausalQuery,
    data: &DataInput,
    class: GraphClass,
    structure_fixed: bool,
    estimator: Option<EstimatorId>,
    controls: Option<crate::PosteriorDrawControls>,
    inference: &InferenceMode,
) -> Result<(), CausalError> {
    if controls.is_none() {
        return Ok(());
    }
    let cfg = match inference {
        InferenceMode::Bayesian(cfg) => Some(cfg),
        InferenceMode::Frequentist => None,
    };
    let enforced = cfg.is_some()
        && matches!(query, CausalQuery::AverageEffect(_))
        && matches!(data, DataInput::Tabular(_))
        && class == GraphClass::Dag
        && structure_fixed
        && matches!(estimator, None | Some(EstimatorId::BayesianGcomp));
    if !enforced {
        return Err(crate::unsupported_reason!(
            "invalid_argument",
            "posterior draw controls bound the draws behind a Bayesian g-computation interval of \
             a tabular AverageEffect on one fixed Dag; this route does not enforce them, so it \
             refuses them rather than publish an interval they do not bound"
        ));
    }
    let Some(cfg) = cfg else {
        return Ok(());
    };
    if cfg.n_draws >= antecedent_estimate::supported_min_draws(cfg.backend) {
        return Ok(());
    }
    Err(if cfg.backend == antecedent_estimate::BayesianBackendKind::Hmc {
        crate::unsupported_reason!(
            "posterior_draws_below_supported",
            "the supported minimum is 3000 posterior draws for the hmc backend, the floor its \
             chain diagnostics need; request at least 3000 draws"
        )
    } else {
        crate::unsupported_reason!(
            "posterior_draws_below_supported",
            "the supported minimum is 2000 posterior draws; request at least 2000 draws"
        )
    })
}

/// Refuse retained predictive replicates where no executor builds them or a bound is zero.
///
/// Only a tabular `AverageEffect` on one fixed `Dag` under Bayesian g-computation retains
/// replicates; every other route refuses the declaration rather than return a result that
/// silently lacks them.
pub(in crate::analysis) fn refuse_unsupported_predictive_replicates(
    query: &CausalQuery,
    data: &DataInput,
    class: GraphClass,
    structure_fixed: bool,
    estimator: Option<EstimatorId>,
    inference: &InferenceMode,
) -> Result<(), CausalError> {
    let InferenceMode::Bayesian(cfg) = inference else {
        return Ok(());
    };
    let Some(bounds) = cfg.predictive_replicates else {
        return Ok(());
    };
    if bounds.n_sims == 0 || bounds.max_rows == 0 {
        return Err(crate::unsupported_reason!(
            "invalid_argument",
            "predictive replicates need at least one simulation (n_sims) and a positive row \
             bound (max_rows)"
        ));
    }
    let supported = matches!(query, CausalQuery::AverageEffect(_))
        && matches!(data, DataInput::Tabular(_))
        && class == GraphClass::Dag
        && structure_fixed
        && matches!(estimator, None | Some(EstimatorId::BayesianGcomp));
    if supported {
        return Ok(());
    }
    Err(crate::unsupported_reason!(
        "option_not_applicable",
        "retained predictive replicates are built by Bayesian g-computation of a tabular \
         AverageEffect on one fixed Dag; this route does not build them, so it refuses the \
         declaration rather than return a result without them"
    ))
}

/// Refuse a declared white-innovation premise where no executor reads it.
///
/// The untempered posterior is the full-likelihood posterior of one lag-aligned Gaussian
/// regression, so it exists for the temporal pulse, the one-step sustained effect and the
/// response curve at the single horizon of one step ahead, on one fixed `TemporalDag` under the
/// isotropic Gaussian prior only. Every other route would ignore the
/// premise, so it refuses it instead of publish a posterior the premise did not shape.
pub(in crate::analysis) fn refuse_unsupported_white_innovation(
    query: &CausalQuery,
    data: &DataInput,
    class: GraphClass,
    structure_fixed: bool,
    inference: &InferenceMode,
) -> Result<(), CausalError> {
    let InferenceMode::Bayesian(cfg) = inference else {
        return Ok(());
    };
    if cfg.white_innovation.is_none() {
        return Ok(());
    }
    // A response curve reads the premise at the single horizon of one step ahead.
    let single_horizon_response = matches!(
        query,
        CausalQuery::Response(q)
            if q.temporal.as_ref().is_some_and(|spec| &*spec.horizons == [1_u32].as_slice())
    );
    let single_window = single_horizon_response
        || matches!(
            query,
            CausalQuery::TemporalEffect(q) if !q.is_multi_step_sustained() && q.effect_modifiers.is_empty()
        );
    let supported = single_window
        && matches!(data, DataInput::Temporal(_))
        && class == GraphClass::TemporalDag
        && structure_fixed
        && !cfg.has_transfer()
        && cfg.likelihood == antecedent_prob::BayesLikelihood::GaussianIdentity;
    if supported {
        return Ok(());
    }
    Err(crate::unsupported_reason!(
        "option_not_applicable",
        "a declared white-innovation premise supports the untempered Gaussian posterior of a \
         single-step temporal pulse, one-step sustained effect or one-step single-horizon \
         response curve on one fixed TemporalDag under the isotropic Gaussian prior; this route \
         does not read it, so it refuses the premise \
         rather than publish a posterior it did not shape. Next attempt: drop the declaration to \
         use the tempered posterior, or compose per-mechanism posteriors for a longer horizon"
    ))
}

/// Refuse a selected lagged-instrument estimator where its construction does not apply.
///
/// The instrument is certified on the unfolded window graph of a single-step pulse or one-step
/// sustained effect over one series, so it needs the `temporal.iv.unfolded` identifier, one fixed
/// `TemporalDag` and series data; every other route would not read the selection.
pub(in crate::analysis) fn refuse_unsupported_temporal_instrument(
    query: &CausalQuery,
    data: &DataInput,
    class: GraphClass,
    structure_fixed: bool,
    identifier: Option<IdentifierId>,
    estimator: Option<EstimatorId>,
) -> Result<(), CausalError> {
    let selected = matches!(
        estimator,
        Some(EstimatorId::TemporalLaggedInstrument | EstimatorId::BayesianTemporalLaggedInstrument)
    );
    if !selected && identifier != Some(IdentifierId::TemporalIvUnfolded) {
        return Ok(());
    }
    let single_window = matches!(
        query,
        CausalQuery::TemporalEffect(q) if !q.is_multi_step_sustained() && q.effect_modifiers.is_empty()
    );
    if selected
        && identifier == Some(IdentifierId::TemporalIvUnfolded)
        && single_window
        && matches!(data, DataInput::Temporal(_))
        && class == GraphClass::TemporalDag
        && structure_fixed
    {
        return Ok(());
    }
    Err(crate::unsupported_reason!(
        "option_not_applicable",
        "the lagged-instrument estimators and the temporal.iv.unfolded identifier go together and \
         identify a single-step pulse or one-step sustained effect over one series on one fixed \
         TemporalDag; this route does not read the selection. Next attempt: select the temporal \
         linear adjustment, or declare the instrument's graph and select both the identifier and a \
         lagged-instrument estimator"
    ))
}

/// Refuse a selected temporal GLM link contrast where its construction does not apply.
///
/// The contrast is a mean-scale difference of a Bernoulli temporal model whose rows must be
/// independent subjects, so it exists for a single-step pulse or one-step sustained effect on one
/// fixed `TemporalDag` over a panel. One dependent series has no independent-trajectory
/// information matrix; every other route would not read the selection.
pub(in crate::analysis) fn refuse_unsupported_temporal_link(
    query: &CausalQuery,
    data: &DataInput,
    class: GraphClass,
    structure_fixed: bool,
    estimator: Option<EstimatorId>,
) -> Result<(), CausalError> {
    if !matches!(
        estimator,
        Some(EstimatorId::TemporalGlmLinkContrast | EstimatorId::BayesianTemporalGlmLinkContrast)
    ) {
        return Ok(());
    }
    if matches!(data, DataInput::Temporal(_)) {
        return Err(crate::unsupported_reason!(
            "route_not_supported",
            "a single dependent series has no independent-trajectory information matrix, so the \
             temporal GLM link contrast is not fitted as if its rows were iid. Next attempt: \
             supply a panel of independent subjects (one trajectory per subject), or use the \
             temporal linear adjustment, whose uncertainty accounts for serial dependence"
        ));
    }
    let single_window = matches!(
        query,
        CausalQuery::TemporalEffect(q) if !q.is_multi_step_sustained() && q.effect_modifiers.is_empty()
    );
    if single_window
        && matches!(data, DataInput::Panel(_))
        && class == GraphClass::TemporalDag
        && structure_fixed
    {
        return Ok(());
    }
    Err(crate::unsupported_reason!(
        "option_not_applicable",
        "the temporal GLM link contrast is the mean-scale contrast of a single-step pulse or \
         one-step sustained effect on one fixed TemporalDag over a panel of independent \
         subjects; this route does not read the selection. Next attempt: select the temporal \
         linear adjustment or its Bayesian g-computation for other temporal effects"
    ))
}

/// Refuse a non-Gaussian Bayesian likelihood where no executor fits it.
///
/// The one owner of which routes honour [`crate::BayesianConfig::likelihood`].
/// Bayesian g-computation of a tabular [`CausalQuery::AverageEffect`] mean on
/// one explicit or accepted [`GraphClass::Dag`] fits the declared Bernoulli or
/// Poisson GLM and averages the inverse link over the rows; that construction
/// has repeated-sampling coverage records. Every other Bayesian route
/// (conditional effects, mediation, temporal effects and responses, continuous
/// responses, panels, ADMG front door, class envelopes, tiered backgrounds and
/// graph-posterior mixtures) is either a Gaussian identity-link model or has no
/// coverage measurement under another link, the conjugate backend is Gaussian by
/// construction, and a transferred prior is mapped on the identity-link
/// coefficient scale. Those combinations refuse here rather than silently fit
/// a Gaussian model or report a `not_measured` construction.
pub(in crate::analysis) fn refuse_unsupported_likelihood(
    query: &CausalQuery,
    data: &DataInput,
    class: GraphClass,
    structure_fixed: bool,
    estimator: Option<EstimatorId>,
    inference: &InferenceMode,
) -> Result<(), CausalError> {
    let InferenceMode::Bayesian(cfg) = inference else {
        return Ok(());
    };
    // The temporal link contrast reads the Bernoulli likelihood itself and checks it at fit.
    if estimator == Some(EstimatorId::BayesianTemporalGlmLinkContrast) {
        return Ok(());
    }
    if cfg.likelihood == antecedent_prob::BayesLikelihood::GaussianIdentity {
        return Ok(());
    }
    if cfg.backend == antecedent_estimate::BayesianBackendKind::ConjugateGaussian {
        return Err(crate::unsupported_reason!(
            "likelihood_not_supported",
            "the conjugate backend fits a Gaussian identity-link model only; use the laplace or \
             hmc backend for a Bernoulli or Poisson likelihood"
        ));
    }
    if cfg.prior_artifact.is_some() || cfg.external_compose.is_some() {
        return Err(crate::unsupported_reason!(
            "likelihood_not_supported",
            "a transferred prior is mapped onto identity-link coefficients; a Bernoulli or \
             Poisson likelihood fits under the isotropic prior_scale only"
        ));
    }
    let supported_query = match query {
        CausalQuery::AverageEffect(q) => {
            matches!(q.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
        }
        CausalQuery::ConditionalEffect(q) => {
            matches!(q.inner.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
        }
        CausalQuery::Response(q) => matches!(
            q.functional,
            antecedent_core::ResponseFunctional::MeanCurve { .. }
                | antecedent_core::ResponseFunctional::InterventionResponse { .. }
        ),
        _ => false,
    };
    let supported = matches!(data, DataInput::Tabular(_))
        && class == GraphClass::Dag
        && structure_fixed
        && supported_query;
    if supported {
        return Ok(());
    }
    Err(crate::unsupported_reason!(
        "likelihood_not_supported",
        "a Bernoulli or Poisson likelihood is fitted by Bayesian g-computation of a tabular \
         AverageEffect or ConditionalEffect mean, or a static response level, on one Dag only; this route fits a Gaussian identity-link model or \
         mixes structures, so it refuses the likelihood rather than fit a different model than \
         the one declared"
    ))
}
