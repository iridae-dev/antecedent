//! Shared prepare-time predicates and score-table helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::accepted::GraphClass;
use crate::analysis::builder::DataInput;
use crate::analysis::execute::Study;
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
    let first = multi
        .environment(0)
        .map_err(|err| CausalError::Compile { message: err.to_string() })?
        .time_index()
        .regularity
        .clone();
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
        col.push(*values.get(idx as usize)?);
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
    let (estimate, diagnostics) = if let Some(tau) = functional.quantile_level() {
        if let CausalQuery::Response(q) = query {
            crate::analysis::helpers::attach_joint_quantile_from_table(
                result.estimate.as_effect().expect("effect query").clone(),
                table.clone(),
                q,
                tau,
            )?
        } else {
            crate::analysis::helpers::attach_quantile_from_table(
                result.estimate.as_effect().expect("effect query").clone(),
                table.clone(),
                tau,
            )?
        }
    } else {
        crate::analysis::helpers::attach_score_functional_grid(
            result.estimate.as_effect().expect("effect query").clone(),
            table.clone(),
        )?
    };
    if functional.quantile_level().is_some() {
        if let Some(response) = &mut result.response {
            response.estimate = antecedent_core::ResponseIdentification::PointIdentified(
                antecedent_core::ResponseValue::Scalar(estimate.ate),
            );
            response.uncertainty = antecedent_core::ResponseUncertainty::Scalar {
                standard_error: estimate.se_analytic,
                lower: estimate.ate
                    - crate::result::reported_se_interval_z() * estimate.se_analytic,
                upper: estimate.ate
                    + crate::result::reported_se_interval_z() * estimate.se_analytic,
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

/// Identification schedule of a sequence plan: `(variable, lag, optional level)` per step.

pub(super) fn ensure_prepared_supported(analysis: &Study) -> Result<(), CausalError> {
    if analysis.graph_posterior.is_some() {
        // Refuse here what every estimate click would refuse, before the
        // posterior identification cache is built.
        match (&analysis.data, &analysis.query) {
            (DataInput::Tabular(_), CausalQuery::Response(query)) => {
                crate::analysis::execute::graph_posterior_response_supported(query)?;
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
                | CausalQuery::ConditionalEffect(_),
            )
            | (
                DataInput::Temporal(_) | DataInput::Event(_),
                CausalQuery::TemporalEffect(_)
                | CausalQuery::Mediation(_)
                | CausalQuery::Response(_)
                | CausalQuery::NestedCounterfactual(_)
                | CausalQuery::PathSpecific(_)
                | CausalQuery::Counterfactual(_)
                | CausalQuery::Distribution(_),
            ) => Ok(()),
            _ => Err(crate::support_reason!(
                "data_modality_not_licensed",
                "graph_posterior on the prepared handle is licensed only for tabular \
                 AverageEffect/Response/ConditionalEffect and series TemporalEffect, \
                 TemporalMediationEffect, or temporal Response"
            )),
        };
    }
    match (&analysis.data, &analysis.query) {
        (DataInput::Tabular(_), CausalQuery::AverageEffect(_)) => {
            if !is_supplied_static_graph(analysis.graph.class()) {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy requires a static Dag/Cpdag/Pag/Admg structure \
                (temporal classes are not session-refreshable here)",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::Response(q)) if !q.is_temporal() => {
            // An Admg is licensed for a joint response with or without a tiered background
            // (the functional-effect estimator serves it), so no tier condition narrows it.
            if !matches!(
                analysis.graph.class(),
                GraphClass::Dag
                    | GraphClass::Cpdag
                    | GraphClass::Pag
                    | GraphClass::Admg
                    | GraphClass::TemporalAdmg
            ) {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports ResponseCurve on a supplied Dag, Cpdag, Pag, \
                              or Admg (or CoDetermined joint cells)",
                });
            }
        }
        (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Response(q))
            if q.is_temporal() =>
        {
            if !matches!(
                analysis.graph.class(),
                GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag
            ) {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports temporal ResponseCurve on TemporalDag, \
                              TemporalCpdag, or TemporalPag",
                });
            }
        }
        (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::TemporalEffect(_)) => {
            if !matches!(
                analysis.graph.class(),
                GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag
            ) {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports TemporalEffect on TemporalDag, \
                              TemporalCpdag, or TemporalPag",
                });
            }
        }
        (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Mediation(_)) => {
            if !matches!(
                analysis.graph.class(),
                GraphClass::TemporalDag | GraphClass::TemporalCpdag
            ) {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports TemporalMediationEffect on TemporalDag or \
                              TemporalCpdag",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::Counterfactual(_)) => {
            if analysis.graph.class() != GraphClass::Dag {
                return Err(CausalError::Unsupported { message: "counterfactual requires Dag" });
            }
        }
        (DataInput::Tabular(_), CausalQuery::NestedCounterfactual(_)) => {
            if analysis.graph.class() != GraphClass::Dag {
                return Err(crate::unsupported_reason!(
                    "cross_world_not_identified",
                    "nested effect requires a supplied DAG"
                ));
            }
        }
        (
            DataInput::Tabular(_),
            CausalQuery::AnomalyAttribution(_) | CausalQuery::ChangeAttribution(_),
        ) => {
            if analysis.graph.class() != GraphClass::Dag {
                return Err(CausalError::Unsupported {
                    message: "AnomalyAttribution and ChangeAttribution require a supplied Dag",
                });
            }
        }
        (
            DataInput::Temporal(_) | DataInput::Event(_),
            CausalQuery::AnomalyAttribution(_) | CausalQuery::ChangeAttribution(_),
        ) => {
            if analysis.graph.class() != GraphClass::TemporalDag {
                return Err(CausalError::Unsupported {
                    message: "AnomalyAttribution and ChangeAttribution on a time series require a TemporalDag",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::Transport(_)) => {
            if !matches!(analysis.graph.class(), GraphClass::Admg | GraphClass::TemporalAdmg) {
                return Err(CausalError::Unsupported {
                    message: "TransportQuery requires Admg or TemporalAdmg",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::Interference(_)) => {
            if analysis.graph.class() != GraphClass::Dag {
                return Err(CausalError::Unsupported { message: "InterferenceQuery requires Dag" });
            }
        }
        (DataInput::Tabular(_), CausalQuery::RandomizedEffect(_)) => {
            if analysis.graph.class() != GraphClass::RandomizedTrial
                || analysis.structure_source != crate::support::StructureSource::RandomizedTrial
            {
                return Err(crate::unsupported_reason!(
                    "data_modality_not_licensed",
                    "PreparedStudy supports randomized ITT only on a graphless Bernoulli trial"
                ));
            }
        }
        (DataInput::Tabular(_), CausalQuery::PolicyValue(_)) => {
            if analysis.graph.class() != GraphClass::RandomizedTrial
                || analysis.structure_source != crate::support::StructureSource::RandomizedTrial
            {
                return Err(CausalError::Unsupported {
                    message: "PolicyValue requires a graphless randomized trial",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::ContinuousDoseResponse(_)) => {
            if analysis.graph.class() != GraphClass::RandomizedTrial
                || analysis.structure_source != crate::support::StructureSource::RandomizedTrial
            {
                return Err(CausalError::Unsupported {
                    message: "continuous-dose response requires a graphless fixed baseline-group design",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::PanelDid(_)) => {
            if analysis.graph.class() != GraphClass::RandomizedTrial
                || analysis.structure_source != crate::support::StructureSource::RandomizedTrial
            {
                return Err(CausalError::Unsupported {
                    message: "PanelDid requires a graphless two-period panel design",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::SyntheticControl(_)) => {
            if analysis.graph.class() != GraphClass::RandomizedTrial
                || analysis.structure_source != crate::support::StructureSource::RandomizedTrial
            {
                return Err(CausalError::Unsupported {
                    message: "SyntheticControl requires a graphless balanced-panel design",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::LocalPolynomialRatio(_)) => {
            if analysis.graph.class() != GraphClass::RandomizedTrial
                || analysis.structure_source != crate::support::StructureSource::RandomizedTrial
            {
                return Err(CausalError::Unsupported {
                    message: "LocalPolynomialRatio requires a graphless fixed-cutoff design",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::Survival(_)) => {
            if analysis.graph.class() != GraphClass::RandomizedTrial
                || analysis.structure_source != crate::support::StructureSource::RandomizedTrial
            {
                return Err(CausalError::Unsupported {
                    message: "Survival requires a graphless randomized trial",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::LongitudinalRegime(_)) => {
            if analysis.graph.class() != GraphClass::RandomizedTrial
                || analysis.structure_source != crate::support::StructureSource::RandomizedTrial
            {
                return Err(CausalError::Unsupported {
                    message: "LongitudinalRegime requires a graphless sequential randomized trial",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::Mediation(_)) => {
            if analysis.graph.class() != GraphClass::Dag {
                return Err(CausalError::Unsupported { message: "static mediation requires Dag" });
            }
        }
        (DataInput::Tabular(_), CausalQuery::ConditionalEffect(_)) => {
            if !matches!(
                analysis.graph.class(),
                GraphClass::Dag | GraphClass::Cpdag | GraphClass::Pag
            ) {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports ConditionalEffect on a supplied Dag, Cpdag, or Pag",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::PathSpecific(_)) => {
            if analysis.graph.class() != GraphClass::Dag {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports PathSpecific only on a supplied Dag",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::Distribution(_)) => {
            if !matches!(
                analysis.graph.class(),
                GraphClass::Dag | GraphClass::Admg | GraphClass::TemporalAdmg
            ) {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports Distribution on a supplied Dag or Admg",
                });
            }
        }
        (DataInput::MultiEnv(_), CausalQuery::TemporalEffect(_)) => {
            if analysis.graph.class() != GraphClass::TemporalDag {
                return Err(crate::unsupported_reason!(
                    "data_modality_not_licensed",
                    "PreparedStudy supports multi-environment TemporalEffect on a TemporalDag"
                ));
            }
        }
        (DataInput::Panel(panel), query) => {
            if !matches!(
                (query, analysis.graph.class()),
                (
                    CausalQuery::TemporalEffect(_) | CausalQuery::Response(_),
                    GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag
                )
            ) {
                return Err(crate::unsupported_reason!(
                    "data_modality_not_licensed",
                    "PreparedStudy supports panel Pulse/Sustained and temporal response on \
                     TemporalDag, TemporalCpdag, or TemporalPag"
                ));
            }
            crate::analysis::builder::refuse_unlicensed_panel_route(
                query,
                analysis.graph.class(),
                &analysis.inference,
                panel,
                analysis.split.as_ref(),
            )?;
        }
        (
            DataInput::Temporal(_) | DataInput::Event(_),
            CausalQuery::Counterfactual(_)
            | CausalQuery::Distribution(_)
            | CausalQuery::PathSpecific(_)
            | CausalQuery::NestedCounterfactual(_),
        ) => {
            if !matches!(
                analysis.graph.class(),
                GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag
            ) {
                return Err(crate::unsupported_reason!(
                    "data_modality_not_licensed",
                    "temporal cross-world effects require a TemporalDag, TemporalCpdag, or TemporalPag"
                ));
            }
        }
        _ => {
            return Err(crate::unsupported_reason!(
                "data_modality_not_licensed",
                "PreparedStudy supports AverageEffect, ResponseCurve, ConditionalEffect, \
                 PathSpecific, Distribution, temporal ResponseCurve, TemporalEffect (Pulse / \
                 single-step Sustained), TemporalMediationEffect, panel Pulse/Sustained, \
                 Counterfactual, AnomalyAttribution, ChangeAttribution, TransportQuery, or \
                 InterferenceQuery, or graphless randomized ITT"
            ));
        }
    }
    Ok(())
}

pub(super) fn is_supplied_static_graph(class: GraphClass) -> bool {
    matches!(
        class,
        GraphClass::Dag
            | GraphClass::Cpdag
            | GraphClass::Pag
            | GraphClass::Admg
            | GraphClass::TemporalAdmg
    )
}
