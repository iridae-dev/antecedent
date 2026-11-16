//! Refuters of an effect estimated for a declared weighted target population.
//!
//! The placebo-treatment, random-common-cause and data-subset refuters perturb the columns or the
//! analysed rows of the table the effect was estimated on, then re-estimate it. For a weighted
//! target the re-estimate is the targeted effect: the weights are per row of the analysed table,
//! so they apply unchanged to a table whose columns were perturbed, and a row subset keeps the
//! same rows' weights, which the estimator renormalises. The overlap gate of the declaration is
//! re-checked on the rows a replicate keeps, and a replicate whose weights fail it is reported
//! as a failed refuter with that reason, never dropped.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, PopulationRegistry, TargetPopulation,
    VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::{BayesianGCompWorkspace, BayesianGComputationAte, EffectEstimate};
use antecedent_expr::IdentifiedEstimand;
use antecedent_validate::ValidationError;
use antecedent_validate::common::EffectRefit;

use crate::estimator_spec::EstimatorSpec;
use crate::strategy_table::{
    StaticEstimateInput, StaticEstimateWorkspaces, estimate_static_effect,
};

/// How the targeted effect is re-estimated on a perturbed table.
#[derive(Clone, Debug)]
pub(crate) enum RefitEngine {
    /// The study's own static estimator.
    Frequentist { spec: EstimatorSpec, overlap_policy: Option<antecedent_estimate::OverlapPolicy> },
    /// Bayesian g-computation: the posterior mean of the model-based effect on the target.
    Bayesian(Box<BayesianGComputationAte>),
}

/// What every atom's refit shares: the targeted query, the registered weights and the engine.
#[derive(Clone, Debug)]
pub(crate) struct TargetedRefits {
    /// The study's query, naming the custom target distribution.
    query: AverageEffectQuery,
    registry: PopulationRegistry,
    min_ess_fraction: f64,
    engine: RefitEngine,
}

impl TargetedRefits {
    /// The refit of a study whose query names a custom target distribution; `None` for any
    /// other population, which the refuters already re-estimate.
    pub(crate) fn new(
        query: &AverageEffectQuery,
        registry: Option<&PopulationRegistry>,
        min_ess_fraction: Option<f64>,
        engine: RefitEngine,
    ) -> Option<Arc<Self>> {
        let TargetPopulation::CustomDistribution(id) = &query.target_population else {
            return None;
        };
        let registry = registry?;
        registry.distribution(*id)?;
        Some(Arc::new(Self {
            query: query.clone(),
            registry: registry.clone(),
            min_ess_fraction: min_ess_fraction.unwrap_or(antecedent_prob::TARGET_MIN_ESS_FRACTION),
            engine,
        }))
    }

    /// The query the refuters carry: the same effect over every analysed row. Their overlap
    /// and sensitivity diagnostics describe the analysed rows; the re-estimates are targeted.
    pub(crate) fn all_rows_query(&self) -> AverageEffectQuery {
        let mut query = self.query.clone();
        query.target_population = TargetPopulation::AllObserved;
        query
    }

    /// The refit of one atom's estimand.
    pub(crate) fn refit_for(
        self: &Arc<Self>,
        estimand: &IdentifiedEstimand,
    ) -> Box<dyn EffectRefit> {
        Box::new(TargetedRefit { shared: Arc::clone(self), estimand: estimand.clone() })
    }
}

#[derive(Debug)]
struct TargetedRefit {
    shared: Arc<TargetedRefits>,
    estimand: IdentifiedEstimand,
}

impl EffectRefit for TargetedRefit {
    fn refit(
        &self,
        data: &TabularData,
        extra_contemporaneous: &[VariableId],
        ctx: &ExecutionContext,
    ) -> Result<EffectEstimate, ValidationError> {
        let shared = &*self.shared;
        let TargetPopulation::CustomDistribution(id) = &shared.query.target_population else {
            return Err(ValidationError::estimation_msg(
                "the refit target is not a custom weighting",
            ));
        };
        let weights = shared.registry.distribution(*id).ok_or_else(|| {
            ValidationError::estimation_msg("the target's weights are not registered")
        })?;
        // The rows this replicate analyses keep their own weights; the overlap gate of the
        // declaration is checked on them.
        let kept: Vec<f64> = match data.storage().analysis_mask() {
            Some(mask) => weights
                .iter()
                .enumerate()
                .filter(|(row, _)| mask.is_valid(*row))
                .map(|(_, weight)| *weight)
                .collect(),
            None => weights.to_vec(),
        };
        antecedent_prob::check_weights_overlap(&kept, shared.min_ess_fraction)?;

        let mut estimand = self.estimand.clone();
        if !extra_contemporaneous.is_empty() {
            let mut adjustment: Vec<VariableId> = estimand.adjustment_set.to_vec();
            adjustment.extend(
                extra_contemporaneous.iter().copied().filter(|id| !adjustment.contains(id)),
            );
            estimand.adjustment_set = Arc::from(adjustment);
        }
        // A placebo treatment is continuous noise, which a binary-treatment estimator cannot
        // take: its targeted effect is read from the linear outcome model with the
        // interactions centred at the weighted covariate means (the Gaussian g-computation).
        let binary_treatment = data
            .float64_values(shared.query.treatment)
            .map_err(ValidationError::from)?
            .iter()
            .all(|value| !value.is_finite() || *value == 0.0 || *value == 1.0);
        if matches!(shared.engine, RefitEngine::Frequentist { .. }) && !binary_treatment {
            let mut linear = BayesianGComputationAte::conjugate();
            linear.n_draws = 64;
            linear.seed = ctx.rng.master_seed();
            return Self::linear_targeted(&linear, data, &estimand, shared, ctx);
        }
        match &shared.engine {
            RefitEngine::Frequentist { spec, overlap_policy } => estimate_static_effect(
                spec,
                StaticEstimateInput {
                    data,
                    estimand: &estimand,
                    query: &shared.query,
                    bootstrap_replicates: 0,
                    overlap_policy: *overlap_policy,
                    population_registry: Some(&shared.registry),
                    sampling_design: None,
                },
                AssumptionSet::new(),
                ctx,
                &mut StaticEstimateWorkspaces::default(),
            )
            .map_err(|error| ValidationError::estimation_msg(error.to_string())),
            RefitEngine::Bayesian(estimator) => {
                Self::linear_targeted(estimator, data, &estimand, shared, ctx)
            }
        }
    }
}

impl TargetedRefit {
    /// The posterior-mean model-based effect on the target under `estimator`.
    fn linear_targeted(
        estimator: &BayesianGComputationAte,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        shared: &TargetedRefits,
        ctx: &ExecutionContext,
    ) -> Result<EffectEstimate, ValidationError> {
        let prepared = estimator.prepare_for_population(
            data,
            estimand,
            &shared.query,
            Some(&shared.registry),
        )?;
        let posterior = estimator.fit(
            &prepared,
            antecedent_core::IdentificationStatus::NonparametricallyIdentified,
            &mut BayesianGCompWorkspace::default(),
            ctx,
        )?;
        crate::analysis::helpers::effect_from_posterior(&posterior)
            .map_err(|error| ValidationError::estimation_msg(error.to_string()))
    }
}
