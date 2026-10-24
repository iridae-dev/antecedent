//! Shared prepare-time predicates and score-table helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::analysis::builder::DataInput;
use crate::analysis::execute::Study;
use crate::analysis::query_license::{prepared_rule, unlicensed_combination};
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::result::StudyResult;
use antecedent_core::{CausalQuery, Intervention, OutcomeFunctional};
use antecedent_data::TableView;
use antecedent_estimate::ScoreTable;

/// Sampling regularity shared by every unit of a prepare-time panel.
///
/// Refresh requires every unit to match the frozen regularity, so prepare
/// refuses a panel whose units already disagree rather than freezing one
/// unit's grid and then refusing the same data on refresh.
pub(super) fn multi_env_regularity(
    multi: &antecedent_data::MultiEnvironmentData,
) -> Result<antecedent_data::SamplingRegularity, CausalError> {
    let first =
        multi.environment(0).map_err(CausalError::rejected)?.time_index().regularity.clone();
    if multi.environments().iter().any(|env| env.time_index().regularity != first) {
        return Err(CausalError::Compile {
            message: "PreparedStudy requires every multi-env series to share one time-index \
                      regularity; align the environments before preparing"
                .into(),
        });
    }
    Ok(first)
}

pub(super) fn discrete_set_treatments(
    interventions: &[Intervention],
) -> Vec<antecedent_core::VariableId> {
    interventions
        .iter()
        .filter_map(|iv| match iv {
            Intervention::Set { variable, .. } => Some(*variable),
            _ => None,
        })
        .collect()
}

/// The declared independence units of a cluster-DML plan.
#[derive(Clone, Debug)]
pub(crate) struct ClusterUnits {
    pub(crate) spec: antecedent_estimate::ClusterDml,
    pub(crate) first: Vec<u32>,
    pub(crate) second: Option<Vec<u32>>,
}

pub(super) fn score_table_treatment_col(analysis: &Study, table: &ScoreTable) -> Option<Vec<f64>> {
    let DataInput::Tabular(data) = &analysis.data else {
        return None;
    };
    let values = data.float64_values(table.treatment).ok()?;
    let mut col = Vec::with_capacity(table.n_rows);
    for &idx in table.row_index.iter() {
        col.push(*values.get(usize::try_from(idx.get()).expect("row id"))?);
    }
    Some(col)
}

/// Whether the query's outcome functional is computed from the frozen score table
/// (exceedance, exceedance grid, quantile) rather than by the estimator itself.
pub(super) fn query_reads_score_table(query: &CausalQuery) -> bool {
    let functional = match query {
        CausalQuery::AverageEffect(q) => &q.outcome_functional,
        CausalQuery::Response(q) => &q.outcome_functional,
        CausalQuery::ConditionalEffect(q) => &q.inner.outcome_functional,
        _ => return false,
    };
    matches!(
        functional,
        OutcomeFunctional::Exceedance(_)
            | OutcomeFunctional::ExceedanceGrid(_)
            | OutcomeFunctional::Quantile(_)
    )
}

pub(super) fn overlay_prepared_score_functional(
    query: &CausalQuery,
    table: Option<&ScoreTable>,
    result: &mut StudyResult,
) -> Result<(), CausalError> {
    let Some(table) = table else {
        return Ok(());
    };
    if matches!(query, CausalQuery::AverageEffect(_))
        && result.logical_plan.estimator.as_deref() != Some("aipw")
    {
        return Ok(());
    }
    if !query_reads_score_table(query) {
        return Ok(());
    }
    let functional = match query {
        CausalQuery::AverageEffect(q) => &q.outcome_functional,
        CausalQuery::Response(q) => &q.outcome_functional,
        CausalQuery::ConditionalEffect(q) => &q.inner.outcome_functional,
        _ => return Ok(()),
    };
    let effect = result.estimate.as_effect().cloned().ok_or_else(|| CausalError::Compile {
        message: "score functional overlay requires an effect estimate".into(),
    })?;
    let (estimate, diagnostics) = if let Some(tau) = functional.quantile_level() {
        if let CausalQuery::Response(q) = query {
            crate::analysis::helpers::attach_joint_quantile_from_table(
                effect,
                table.clone(),
                q,
                tau,
            )?
        } else {
            crate::analysis::helpers::attach_quantile_from_table(effect, table.clone(), tau)?
        }
    } else {
        crate::analysis::helpers::attach_score_functional_grid(effect, table.clone())?
    };
    if functional.quantile_level().is_some() {
        if let Some(response) = result.payload.response_mut() {
            response.estimate = antecedent_core::ResponseIdentification::PointIdentified(
                antecedent_core::ResponseValue::Scalar(estimate.ate),
            );
            response.uncertainty = antecedent_core::ResponseUncertainty::Scalar {
                standard_error: estimate.se_analytic(),
                lower: estimate.ate
                    - crate::result::reported_se_interval_z() * estimate.se_analytic(),
                upper: estimate.ate
                    + crate::result::reported_se_interval_z() * estimate.se_analytic(),
                level: 0.95,
                interpretation: antecedent_core::IntervalInterpretation::Confidence,
                draws: None,
            };
        }
    }
    result.estimate = crate::PrimaryEstimate::Effect(estimate);
    result.rebind_interval(result.posterior.is_some());
    result.diagnostics.extend(diagnostics);
    Ok(())
}

pub(super) fn ensure_prepared_supported(analysis: &Study) -> Result<(), CausalError> {
    if analysis.graph_posterior.is_some() {
        // Refuse here what every estimate click would refuse, before the
        // posterior identification cache is built.
        match (&analysis.data, &analysis.query) {
            (DataInput::Tabular(_), CausalQuery::Response(query)) => {
                // A derivative or multi-coordinate level is answered atom by atom, and each
                // atom's executor states its own refusals.
                if !crate::analysis::execute::atom_route_response(query) {
                    crate::analysis::execute::graph_posterior_response_supported(query)?;
                }
            }
            (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Mediation(query))
                if matches!(analysis.inference, InferenceMode::Frequentist)
                    && query.horizons.len() != 1 =>
            {
                return Err(CausalError::Unsupported {
                    message: "Frequentist DBN-posterior mediation is licensed for one horizon; \
                              multi-horizon grids need their own joint uncertainty contract",
                });
            }
            (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Response(query)) => {
                crate::analysis::execute::dbn_posterior_response_supported(query)?;
            }
            _ => {}
        }
        return match (&analysis.data, &analysis.query) {
            (
                DataInput::Tabular(_),
                CausalQuery::AverageEffect(_)
                | CausalQuery::Response(_)
                | CausalQuery::ConditionalEffect(_)
                | CausalQuery::Transport(_)
                | CausalQuery::Interference(_)
                | CausalQuery::Mediation(_)
                | CausalQuery::PathSpecific(_)
                | CausalQuery::NestedCounterfactual(_)
                | CausalQuery::Counterfactual(_)
                | CausalQuery::Distribution(_)
                | CausalQuery::AnomalyAttribution(_)
                | CausalQuery::ChangeAttribution(_),
            )
            | (
                DataInput::Temporal(_) | DataInput::Event(_),
                CausalQuery::TemporalEffect(_)
                | CausalQuery::Mediation(_)
                | CausalQuery::Response(_)
                | CausalQuery::NestedCounterfactual(_)
                | CausalQuery::PathSpecific(_)
                | CausalQuery::Counterfactual(_)
                | CausalQuery::Distribution(_)
                | CausalQuery::AnomalyAttribution(_)
                | CausalQuery::ChangeAttribution(_),
            ) => Ok(()),
            _ => Err(crate::support_reason!(
                "data_modality_not_licensed",
                "graph_posterior on the prepared handle is licensed only for tabular \
                 AverageEffect/Response/ConditionalEffect/Transport/Interference/Mediation/\
                 PathSpecific/NestedCounterfactual/Counterfactual/Anomaly and Change \
                 attribution and series \
                 TemporalEffect, TemporalMediationEffect, or temporal Response"
            )),
        };
    }
    let modality = crate::analysis::execute::data_modality(&analysis.data);
    let Some(rule) = prepared_rule(&analysis.query, modality) else {
        return Err(unlicensed_combination());
    };
    if !rule.admits(analysis.graph.class(), analysis.structure_source) {
        return Err(rule.refusal());
    }
    if let DataInput::Panel(panel) = &analysis.data {
        crate::analysis::builder::refuse_unlicensed_panel_route(
            &analysis.query,
            analysis.graph.class(),
            &analysis.inference,
            panel,
            analysis.split.as_ref(),
        )?;
    }
    Ok(())
}
