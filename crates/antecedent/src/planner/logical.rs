//! Logical-plan compilers for each analysis cell.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, DataClassification, Intervention, LogicalAnalysisPlanRecord,
    ResponseFunctional, ResponseQuery, TargetPopulation, TemporalEffectQuery, VariableId,
};
use antecedent_data::{DiscoveryEstimationSplit, TableView, TabularData, TimeSeriesData};
use antecedent_graph::{Dag, Pag, TemporalDag};

use crate::error::CausalError;
use crate::strategy_table::{
    EstimatorId, IdentifierId, validate_distribution_pair, validate_path_specific_pair,
    validate_response_pair, validate_static_pair,
};

use super::LogicalAnalysisPlan;

/// Inputs needed to compile a logical plan for the static ATE path.
#[derive(Clone, Debug)]
pub struct StaticAteCompileInput<'a> {
    /// Tabular data (classification + row count).
    pub data: &'a TabularData,
    /// Graph.
    pub graph: &'a Dag,
    /// Query.
    pub query: &'a AverageEffectQuery,
    /// Validation suite id.
    pub validation_suite: Option<Arc<str>>,
    /// Identifier id selected by the builder (defaults to `backdoor.adjustment`).
    pub identifier: Arc<str>,
    /// Estimator id selected by the builder (defaults to `linear.adjustment.ate`).
    pub estimator: Arc<str>,
}

/// Compile logical plan for static ATE .
///
/// # Errors
///
/// Query validation failures, or an identifier/estimator pair not in the compile-time allowlist
/// (see [`crate::strategy_table::validate_static_pair`]).
pub fn compile_logical_static_ate(
    input: StaticAteCompileInput<'_>,
) -> Result<LogicalAnalysisPlan, CausalError> {
    input.query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
    validate_query_vars_in_dag(input.graph, input.query.treatment, input.query.outcome)?;
    let identifier: IdentifierId = input.identifier.parse()?;
    let estimator: EstimatorId = input.estimator.parse()?;
    validate_static_pair(identifier, estimator)?;
    if matches!(estimator, EstimatorId::LinearAdjustmentAte)
        && input.query.target_population != TargetPopulation::AllObserved
    {
        return Err(CausalError::Compile {
            message: format!(
                "estimator \"linear.adjustment.ate\" only supports TargetPopulation::AllObserved \
                 (got {:?}); use a propensity or AIPW estimator for ATT/ATC/Predicate",
                input.query.target_population
            ),
        });
    }
    let record = LogicalAnalysisPlanRecord {
        plan_id: Arc::from("static_ate"),
        data_classification: DataClassification::Tabular,
        discovery_algorithm: None,
        graph_review_required: false,
        identifier: Some(Arc::clone(&input.identifier)),
        estimator: Some(Arc::clone(&input.estimator)),
        validation_suite: input.validation_suite,
        query_variables: Arc::from([input.query.treatment, input.query.outcome]),
    };
    let plan = LogicalAnalysisPlan {
        record,
        query: CausalQuery::AverageEffect(input.query.clone()),
        split: None,
        row_count_hint: input.data.row_count() as u64,
    };
    plan.validate()?;
    Ok(plan)
}

/// Inputs for a static-DAG continuous-response plan.
#[derive(Clone, Debug)]
pub struct StaticResponseCompileInput<'a> {
    /// Tabular data.
    pub data: &'a TabularData,
    /// Accepted causal DAG.
    pub graph: &'a Dag,
    /// Response query.
    pub query: &'a ResponseQuery,
    /// Validation suite id.
    pub validation_suite: Option<Arc<str>>,
    /// Identifier selected by the builder.
    pub identifier: Arc<str>,
    /// Estimator selected by the builder.
    pub estimator: Arc<str>,
}

/// Compile a continuous-response query over tabular data and a static DAG.
///
/// # Errors
///
/// Invalid queries, out-of-range variables, or incompatible strategies.
pub fn compile_logical_static_response(
    input: StaticResponseCompileInput<'_>,
) -> Result<LogicalAnalysisPlan, CausalError> {
    input.query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
    let identifier: IdentifierId = input.identifier.parse()?;
    let estimator: EstimatorId = input.estimator.parse()?;
    validate_response_pair(identifier, estimator)?;
    let expected = match &input.query.functional {
        ResponseFunctional::MeanCurve { .. } | ResponseFunctional::PointDerivative { .. } => {
            EstimatorId::ResponseKennedyDr
        }
        ResponseFunctional::AverageDerivative { .. } => EstimatorId::ResponseRieszAde,
        ResponseFunctional::DirectionalDerivative { .. } | ResponseFunctional::Jacobian { .. } => {
            EstimatorId::ResponseGamDerivative
        }
        ResponseFunctional::InterventionResponse { .. } => EstimatorId::ResponseInterventionGcomp,
    };
    if estimator != expected {
        return Err(CausalError::Compile {
            message: format!(
                "response functional requires estimator {:?}; got {:?}",
                expected.as_str(),
                estimator.as_str()
            ),
        });
    }
    let (treatments, outcomes) = response_query_variables(&input.query.functional);
    for &treatment in &treatments {
        for &outcome in &outcomes {
            validate_query_vars_in_dag(input.graph, treatment, outcome)?;
        }
    }
    let query_variables: Arc<[VariableId]> =
        treatments.iter().chain(outcomes.iter()).copied().collect::<Vec<_>>().into();
    let plan = LogicalAnalysisPlan {
        record: LogicalAnalysisPlanRecord {
            plan_id: Arc::from("static_response"),
            data_classification: DataClassification::Tabular,
            discovery_algorithm: None,
            graph_review_required: false,
            identifier: Some(input.identifier),
            estimator: Some(input.estimator),
            validation_suite: input.validation_suite,
            query_variables,
        },
        query: CausalQuery::Response(input.query.clone()),
        split: None,
        row_count_hint: input.data.row_count() as u64,
    };
    plan.validate()?;
    Ok(plan)
}

fn response_query_variables(functional: &ResponseFunctional) -> (Vec<VariableId>, Vec<VariableId>) {
    (functional.treatment_ids(), functional.outcome_ids())
}

/// Inputs for PAG ATE compile (class-aware identification).
#[derive(Clone, Debug)]
pub struct StaticPagAteCompileInput<'a> {
    /// Tabular data.
    pub data: &'a TabularData,
    /// PAG.
    pub pag: &'a Pag,
    /// Query.
    pub query: &'a AverageEffectQuery,
    /// Validation suite id.
    pub validation_suite: Option<Arc<str>>,
    /// Identifier (must be generalized.adjustment).
    pub identifier: Arc<str>,
    /// Estimator id.
    pub estimator: Arc<str>,
}

/// Compile logical plan for static ATE on a PAG.
///
/// # Errors
///
/// Query validation or incompatible identifier/estimator.
pub fn compile_logical_static_pag_ate(
    input: StaticPagAteCompileInput<'_>,
) -> Result<LogicalAnalysisPlan, CausalError> {
    input.query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
    validate_query_vars_in_pag(input.pag, input.query.treatment, input.query.outcome)?;
    let identifier: IdentifierId = input.identifier.parse()?;
    let estimator: EstimatorId = input.estimator.parse()?;
    if !matches!(identifier, IdentifierId::GeneralizedAdjustment) {
        return Err(CausalError::Compile {
            message: format!(
                "PAG ATE requires identifier \"generalized.adjustment\"; got {:?}",
                identifier.as_str()
            ),
        });
    }
    validate_static_pair(identifier, estimator)?;
    let record = LogicalAnalysisPlanRecord {
        plan_id: Arc::from("static_pag_ate"),
        data_classification: DataClassification::Tabular,
        discovery_algorithm: None,
        graph_review_required: false,
        identifier: Some(Arc::clone(&input.identifier)),
        estimator: Some(Arc::clone(&input.estimator)),
        validation_suite: input.validation_suite,
        query_variables: Arc::from([input.query.treatment, input.query.outcome]),
    };
    let plan = LogicalAnalysisPlan {
        record,
        query: CausalQuery::AverageEffect(input.query.clone()),
        split: None,
        row_count_hint: input.data.row_count() as u64,
    };
    plan.validate()?;
    Ok(plan)
}

/// Compile logical plan for interventional-distribution queries.
#[derive(Clone, Debug)]
pub struct StaticDistributionCompileInput<'a> {
    /// Tabular data.
    pub data: &'a TabularData,
    /// Graph.
    pub graph: &'a Dag,
    /// Distribution query.
    pub query: &'a antecedent_core::InterventionalDistributionQuery,
    /// Validation suite id.
    pub validation_suite: Option<Arc<str>>,
    /// Identifier (`general.id` / `auto`).
    pub identifier: Arc<str>,
    /// Estimator (`functional.distribution`).
    pub estimator: Arc<str>,
}

/// Compile logical plan for an interventional distribution.
///
/// # Errors
///
/// Query validation or incompatible identifier/estimator.
pub fn compile_logical_distribution(
    input: StaticDistributionCompileInput<'_>,
) -> Result<LogicalAnalysisPlan, CausalError> {
    input.query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
    if input.query.target_population != TargetPopulation::AllObserved {
        return Err(CausalError::Compile {
            message: "functional.distribution only supports TargetPopulation::AllObserved".into(),
        });
    }
    let treatment =
        input.query.interventions.first().and_then(Intervention::primary_variable).ok_or_else(
            || CausalError::Compile {
                message:
                    "distribution query requires at least one intervention with a primary variable"
                        .into(),
            },
        )?;
    let outcome = *input.query.outcomes.first().ok_or_else(|| CausalError::Compile {
        message: "distribution query requires at least one outcome".into(),
    })?;
    validate_query_vars_in_dag(input.graph, treatment, outcome)?;
    let identifier: IdentifierId = input.identifier.parse()?;
    let estimator: EstimatorId = input.estimator.parse()?;
    validate_distribution_pair(identifier, estimator)?;
    let mut qvars = vec![treatment, outcome];
    for &z in input.query.conditioning.iter() {
        if !qvars.contains(&z) {
            qvars.push(z);
        }
    }
    let record = LogicalAnalysisPlanRecord {
        plan_id: Arc::from("static_distribution"),
        data_classification: DataClassification::Tabular,
        discovery_algorithm: None,
        graph_review_required: false,
        identifier: Some(Arc::clone(&input.identifier)),
        estimator: Some(Arc::clone(&input.estimator)),
        validation_suite: input.validation_suite,
        query_variables: Arc::from(qvars),
    };
    let plan = LogicalAnalysisPlan {
        record,
        query: CausalQuery::Distribution(input.query.clone()),
        split: None,
        row_count_hint: input.data.row_count() as u64,
    };
    plan.validate()?;
    Ok(plan)
}

/// Compile input for path-specific natural-effect queries.
#[derive(Clone, Debug)]
pub struct StaticPathSpecificCompileInput<'a> {
    /// Tabular data.
    pub data: &'a TabularData,
    /// Graph.
    pub graph: &'a Dag,
    /// Path-specific query.
    pub query: &'a antecedent_core::PathSpecificEffectQuery,
    /// Validation suite id.
    pub validation_suite: Option<Arc<str>>,
    /// Identifier.
    pub identifier: Arc<str>,
    /// Estimator.
    pub estimator: Arc<str>,
}

/// Compile logical plan for path-specific natural effects.
///
/// # Errors
///
/// Query validation or incompatible identifier/estimator.
pub fn compile_logical_path_specific(
    input: StaticPathSpecificCompileInput<'_>,
) -> Result<LogicalAnalysisPlan, CausalError> {
    input.query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
    if input.query.target_population != TargetPopulation::AllObserved {
        return Err(CausalError::Compile {
            message: "functional.effect only supports TargetPopulation::AllObserved".into(),
        });
    }
    validate_query_vars_in_dag(input.graph, input.query.treatment, input.query.outcome)?;
    let identifier: IdentifierId = input.identifier.parse()?;
    let estimator: EstimatorId = input.estimator.parse()?;
    validate_path_specific_pair(identifier, estimator)?;
    let mut qvars = vec![input.query.treatment, input.query.outcome];
    for &m in input.query.path_nodes.iter() {
        if !qvars.contains(&m) {
            qvars.push(m);
        }
    }
    let record = LogicalAnalysisPlanRecord {
        plan_id: Arc::from("static_path_specific"),
        data_classification: DataClassification::Tabular,
        discovery_algorithm: None,
        graph_review_required: false,
        identifier: Some(Arc::clone(&input.identifier)),
        estimator: Some(Arc::clone(&input.estimator)),
        validation_suite: input.validation_suite,
        query_variables: Arc::from(qvars),
    };
    let plan = LogicalAnalysisPlan {
        record,
        query: CausalQuery::PathSpecific(input.query.clone()),
        split: None,
        row_count_hint: input.data.row_count() as u64,
    };
    plan.validate()?;
    Ok(plan)
}

fn validate_query_vars_in_dag(
    dag: &Dag,
    treatment: antecedent_core::VariableId,
    outcome: antecedent_core::VariableId,
) -> Result<(), CausalError> {
    let mut has_t = false;
    let mut has_y = false;
    for node in dag.nodes() {
        if let antecedent_graph::NodeRef::Static(v) = node {
            if *v == treatment {
                has_t = true;
            }
            if *v == outcome {
                has_y = true;
            }
        }
    }
    if !has_t || !has_y {
        return Err(CausalError::Compile {
            message: format!(
                "query variables not in DAG (treatment present={has_t}, outcome present={has_y})"
            ),
        });
    }
    Ok(())
}

fn validate_query_vars_in_pag(
    pag: &Pag,
    treatment: antecedent_core::VariableId,
    outcome: antecedent_core::VariableId,
) -> Result<(), CausalError> {
    let mut has_t = false;
    let mut has_y = false;
    for node in pag.nodes() {
        if let antecedent_graph::NodeRef::Static(v) = node {
            if *v == treatment {
                has_t = true;
            }
            if *v == outcome {
                has_y = true;
            }
        }
    }
    if !has_t || !has_y {
        return Err(CausalError::Compile {
            message: format!(
                "query variables not in PAG (treatment present={has_t}, outcome present={has_y})"
            ),
        });
    }
    Ok(())
}

fn validate_query_vars_in_temporal_dag(
    dag: &TemporalDag,
    treatment: antecedent_core::VariableId,
    outcome: antecedent_core::VariableId,
) -> Result<(), CausalError> {
    // A node-less DAG is the placeholder the graph-posterior path supplies: the
    // structure lives in the `GraphPosterior` mixture, not here, so there is no
    // membership to check and rejecting would be a false negative.
    if dag.nodes().is_empty() {
        return Ok(());
    }
    let mut has_t = false;
    let mut has_y = false;
    for node in dag.nodes() {
        // Temporal graphs carry `Lagged` nodes, and `Context` nodes when an
        // environment is attached; both name a variable the query may reference.
        let variable = match node {
            antecedent_graph::NodeRef::Lagged { variable, .. }
            | antecedent_graph::NodeRef::Context { variable, .. } => *variable,
            antecedent_graph::NodeRef::Static(v) => *v,
        };
        if variable == treatment {
            has_t = true;
        }
        if variable == outcome {
            has_y = true;
        }
    }
    if !has_t || !has_y {
        return Err(CausalError::Compile {
            message: format!(
                "query variables not in temporal DAG (treatment present={has_t}, outcome \
                 present={has_y})"
            ),
        });
    }
    Ok(())
}

/// Compile logical plan for a temporal effect with a supplied temporal graph.
///
/// # Errors
///
/// Modality / query validation failures.
pub fn compile_logical_temporal_effect(
    data: &TimeSeriesData,
    graph: &TemporalDag,
    query: &TemporalEffectQuery,
    split: Option<DiscoveryEstimationSplit>,
    review_required: bool,
) -> Result<LogicalAnalysisPlan, CausalError> {
    compile_logical_temporal_effect_classified(
        data,
        graph,
        query,
        split,
        review_required,
        DataClassification::Temporal,
    )
}

/// Temporal effect plan with an explicit data classification (Event / Panel / Temporal).
///
/// # Errors
///
/// Query validation failures.
pub fn compile_logical_temporal_effect_classified(
    data: &TimeSeriesData,
    graph: &TemporalDag,
    query: &TemporalEffectQuery,
    split: Option<DiscoveryEstimationSplit>,
    review_required: bool,
    data_classification: DataClassification,
) -> Result<LogicalAnalysisPlan, CausalError> {
    query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
    validate_query_vars_in_temporal_dag(graph, query.treatment, query.outcome)?;
    if query.target_population != TargetPopulation::AllObserved {
        return Err(CausalError::Compile {
            message: format!(
                "temporal linear adjustment only supports TargetPopulation::AllObserved \
                 (got {:?})",
                query.target_population
            ),
        });
    }
    let row_count_hint =
        split.map_or_else(|| data.row_count() as u64, |s| s.estimation.len() as u64);
    let record = LogicalAnalysisPlanRecord {
        plan_id: Arc::from("temporal_effect"),
        data_classification,
        discovery_algorithm: None,
        graph_review_required: review_required,
        identifier: Some(Arc::from("temporal.backdoor.unfolded")),
        estimator: Some(Arc::from("temporal.linear.adjustment")),
        validation_suite: None,
        query_variables: Arc::from([query.treatment, query.outcome]),
    };
    let plan = LogicalAnalysisPlan {
        record,
        query: CausalQuery::TemporalEffect(query.clone()),
        split,
        row_count_hint,
    };
    plan.validate()?;
    Ok(plan)
}

