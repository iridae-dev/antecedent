//! Standalone Bayesian kernel for a checked fixed-DAG temporal effect.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_data::{DiscoveryEstimationSplit, TimeSeriesData};
use antecedent_estimate::{
    BayesianGCompWorkspace, BayesianGComputationAte, BayesianTemporalGcomp, CausalPosterior,
    EffectEstimate, OverlapPolicy, PreparedBayesianProblem, TemporalLinearAdjustment,
};

use crate::{
    CausalError,
    analysis::checked_bayesian_temporal_effect::CheckedBayesianTemporalEffectOperation,
    inference::resolve_bayesian_prior_with_conflict,
};

/// Executable Bayesian fit for a pulse or one-step sustained temporal DAG effect.
///
/// The retained identification operation owns the target graph, temporal query,
/// unfolded proof, estimator choice, and indexer. This kernel only binds the
/// current series to that operation and fits the declared Bayesian model.
#[derive(Clone, Debug)]
pub(crate) struct CheckedBayesianTemporalDagFit {
    pub(crate) estimate: EffectEstimate,
    pub(crate) posterior: CausalPosterior,
    pub(crate) prepared: PreparedBayesianProblem,
    pub(crate) estimator: BayesianTemporalGcomp,
}

/// Fit one checked fixed-DAG temporal effect through lag-aligned Bayesian g-computation.
///
/// Multi-step schedules and temporal graph classes have separate execution
/// contracts and are refused here. Those operations need sequential g-formula
/// composition or atom-wise posterior mixing rather than one temporal design.
pub(crate) fn fit_temporal_dag_effect(
    operation: &CheckedBayesianTemporalEffectOperation,
    data: &TimeSeriesData,
    split: Option<&DiscoveryEstimationSplit>,
    context: &ExecutionContext,
) -> Result<CheckedBayesianTemporalDagFit, CausalError> {
    let target = operation.target();
    if target.query().is_multi_step_sustained()
        || target.procedure().1 != crate::strategy_table::EstimatorId::TemporalLinearAdjustment
    {
        return Err(CausalError::Unsupported {
            message: "standalone Bayesian temporal kernel supports pulse and one-step sustained effects only",
        });
    }

    let mut temporal = TemporalLinearAdjustment::new();
    temporal.inner.bootstrap_replicates = 0;
    temporal.inner.overlap = OverlapPolicy::ExplicitOverride;
    let prepared = temporal
        .prepare(
            data,
            target.estimand(),
            target.query(),
            target.indexer(),
            split,
            &context.kernel_policy,
        )
        .map_err(CausalError::from)?;
    let coefficient_names = antecedent_estimate::temporal_coefficient_names(
        data,
        target.estimand(),
        target.query(),
        target.indexer(),
    )
    .map_err(CausalError::from)?;
    let problem = BayesianGComputationAte::from_prepared_temporal(&prepared, coefficient_names)
        .map_err(CausalError::from)?;
    let (prior, conflict) =
        resolve_bayesian_prior_with_conflict(operation.config(), &problem, Some(context))?;
    let estimator = BayesianTemporalGcomp {
        inner: BayesianGComputationAte {
            backend: operation.config().backend,
            likelihood: operation.config().likelihood,
            n_draws: operation.config().n_draws,
            seed: context.rng.master_seed(),
            overlap: OverlapPolicy::ExplicitOverride,
            prior_scale: operation.config().prior_scale,
            prior,
        },
    };
    // Resolve once against the exact lag-named design, then fit with that prior.
    let mut workspace = BayesianGCompWorkspace::default();
    let mut posterior = estimator
        .fit(&problem, target.identification().status, &mut workspace, context)
        .map_err(CausalError::from)?;
    if let Some(summary) = conflict {
        posterior = antecedent_validate::with_conflict_summary(posterior, summary);
    }
    let estimate = crate::analysis::execute::effect_from_posterior(&posterior)?
        .with_n_obs(u64::try_from(prepared.design.nrows).unwrap_or(u64::MAX));
    Ok(CheckedBayesianTemporalDagFit { estimate, posterior, prepared: problem, estimator })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BayesianConfig, InferenceMode,
        analysis::checked_temporal_effect::CheckedTemporalEffectOperation,
    };
    use antecedent_core::{Lag, TemporalEffectQuery, TemporalPolicy, VariableId};
    use antecedent_data::TimeSeriesData;
    use antecedent_graph::{TemporalDag, ensure_lagged};
    use antecedent_identify::TemporalBackdoorIdentifier;

    #[test]
    fn checked_temporal_bayesian_kernel_recovers_independent_lagged_scm_truth() {
        let n = 2_400usize;
        let mut rng = antecedent_core::CausalRng::from_seed(332_810);
        let treatment =
            (0..n).map(|_| antecedent_kernels::standard_normal(&mut rng)).collect::<Vec<_>>();
        let outcome = (0..n)
            .map(|row| {
                let lag1 = row.checked_sub(1).map_or(0.0, |index| treatment[index]);
                let lag2 = row.checked_sub(2).map_or(0.0, |index| treatment[index]);
                0.4 + 2.0 * lag1 + 3.0 * lag2 + 0.15 * antecedent_kernels::standard_normal(&mut rng)
            })
            .collect::<Vec<_>>();
        let data = TimeSeriesData::from_f64_columns(
            [("t", treatment.as_slice()), ("y", outcome.as_slice())],
            1,
        )
        .unwrap();
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let mut graph = TemporalDag::empty();
        let t_lag1 = ensure_lagged(&mut graph, t, Lag::from_raw(1)).unwrap();
        let t_lag2 = ensure_lagged(&mut graph, t, Lag::from_raw(2)).unwrap();
        let y_now = ensure_lagged(&mut graph, y, Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(t_lag1, y_now).unwrap();
        graph.insert_directed(t_lag2, y_now).unwrap();
        let query = TemporalEffectQuery::pulse(t, y, 1.0).with_policy(TemporalPolicy::pulse(-1));
        let identified =
            TemporalBackdoorIdentifier::new().identify_temporal(&graph, &query).unwrap();
        let estimand = identified.result.estimands[0].clone();
        let target = CheckedTemporalEffectOperation::checked(
            &graph,
            &query,
            identified.result,
            estimand,
            identified.indexer,
            crate::strategy_table::IdentifierId::TemporalBackdoorUnfolded,
            crate::strategy_table::EstimatorId::TemporalLinearAdjustment,
            0,
            super::super::super::builder::RefuteSuite::None,
        )
        .unwrap();
        let operation = CheckedBayesianTemporalEffectOperation::checked(
            target,
            &InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(512).prior_scale(100.0)),
            super::super::super::builder::RefuteSuite::None,
        )
        .unwrap();
        let context = ExecutionContext::for_tests(81_002);
        let fit = fit_temporal_dag_effect(&operation, &data, None, &context).unwrap();
        assert!(
            (fit.estimate.ate - 2.0).abs() < 0.2,
            "posterior effect {} summaries {:?} coefficients {:?}",
            fit.estimate.ate,
            fit.posterior.summaries.mean,
            fit.prepared.coef_names
        );
        assert_eq!(fit.posterior.draws.n_draws, 512);
        assert_eq!(fit.prepared.design.nrows, n - 1);
    }
}
