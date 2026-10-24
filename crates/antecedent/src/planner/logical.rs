//! Logical-plan compilers for each analysis cell.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_contract::{DataClassification, LogicalAnalysisPlanRecord};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, Intervention, ObservationSpec, ResponseFunctional,
    ResponseQuery, TargetPopulation, TemporalEffectQuery, VariableId,
};
use antecedent_data::{DiscoveryEstimationSplit, TableView, TabularData, TimeSeriesData};
use antecedent_graph::{Admg, Cpdag, Dag, Pag, TemporalDag};

use crate::error::CausalError;
use crate::strategy_table::{
    EstimatorId, IdentifierId, validate_class_response_pair, validate_distribution_pair,
    validate_path_specific_pair, validate_response_pair, validate_static_pair,
};

use super::LogicalAnalysisPlan;

/// Inputs shared by the static tabular compiles.
///
/// Each route alias parses `identifier` and `estimator` once inside its compile
/// and keeps that route's refusal text. The structure field is `graph` for a
/// DAG, a PAG, and a CPDAG.
#[derive(Clone, Debug)]
pub(crate) struct StaticCompileInput<'a, G, Q> {
    /// Tabular data (classification + row count).
    pub data: &'a TabularData,
    /// Accepted structure for this route.
    pub graph: &'a G,
    /// Query the plan records.
    pub query: &'a Q,
    /// Validation suite id.
    pub validation_suite: Option<Arc<str>>,
    /// Identifier id selected by the builder, still a string until this compile parses it.
    pub identifier: Arc<str>,
    /// Estimator id selected by the builder, still a string until this compile parses it.
    pub estimator: Arc<str>,
}

/// Static DAG average-effect compile input.
pub(crate) type StaticAteCompileInput<'a> = StaticCompileInput<'a, Dag, AverageEffectQuery>;
/// Static DAG response compile input.
pub(crate) type StaticResponseCompileInput<'a> = StaticCompileInput<'a, Dag, ResponseQuery>;
/// Static PAG average-effect compile input.
pub(crate) type StaticPagAteCompileInput<'a> = StaticCompileInput<'a, Pag, AverageEffectQuery>;
/// Static CPDAG average-effect compile input.
pub(crate) type StaticCpdagAteCompileInput<'a> = StaticCompileInput<'a, Cpdag, AverageEffectQuery>;
/// Static PAG response compile input.
pub(crate) type StaticPagResponseCompileInput<'a> = StaticCompileInput<'a, Pag, ResponseQuery>;
/// Static CPDAG response compile input.
pub(crate) type StaticCpdagResponseCompileInput<'a> = StaticCompileInput<'a, Cpdag, ResponseQuery>;
/// Interventional-distribution compile input.
pub(crate) type StaticDistributionCompileInput<'a> =
    StaticCompileInput<'a, Dag, antecedent_core::InterventionalDistributionQuery>;
/// Path-specific natural-effect compile input.
pub(crate) type StaticPathSpecificCompileInput<'a> =
    StaticCompileInput<'a, Dag, antecedent_core::PathSpecificEffectQuery>;
/// Path-specific natural-effect compile input on a mixed graph.
pub(crate) type StaticAdmgPathSpecificCompileInput<'a> =
    StaticCompileInput<'a, Admg, antecedent_core::PathSpecificEffectQuery>;
/// Path-specific natural-effect compile input on a CPDAG.
pub(crate) type StaticCpdagPathSpecificCompileInput<'a> =
    StaticCompileInput<'a, Cpdag, antecedent_core::PathSpecificEffectQuery>;
/// Path-specific natural-effect compile input on a PAG.
pub(crate) type StaticPagPathSpecificCompileInput<'a> =
    StaticCompileInput<'a, Pag, antecedent_core::PathSpecificEffectQuery>;

/// `linear.adjustment.ate` reports the treatment coefficient, the effect in the
/// `AllObserved` population. On every graph class it refuses another target
/// rather than reporting that coefficient under the target's name.
fn refuse_linear_adjustment_population(
    estimator: EstimatorId,
    query: &AverageEffectQuery,
) -> Result<(), CausalError> {
    if matches!(estimator, EstimatorId::LinearAdjustmentAte)
        && query.target_population != TargetPopulation::AllObserved
    {
        return Err(crate::unsupported_reason!(
            "population_not_estimable",
            "estimator \"linear.adjustment.ate\" only supports TargetPopulation::AllObserved; use \
             a propensity or AIPW estimator for ATT/ATC/Predicate/CustomDistribution targets"
        ));
    }
    Ok(())
}

/// Compile logical plan for static ATE.
///
/// # Errors
///
/// Query validation failures, or an identifier/estimator pair not in the compile-time allowlist
/// (see [`crate::strategy_table::validate_static_pair`]).
pub(crate) fn compile_logical_static_ate(
    input: StaticAteCompileInput<'_>,
) -> Result<LogicalAnalysisPlan, CausalError> {
    input.query.validate()?;
    validate_query_vars_in_dag(input.graph, input.query.treatment, input.query.outcome)?;
    let identifier: IdentifierId = input.identifier.parse()?;
    let estimator: EstimatorId = input.estimator.parse()?;
    validate_static_pair(identifier, estimator)?;
    refuse_linear_adjustment_population(estimator, input.query)?;
    finish_static_ate(
        input.data,
        input.query,
        input.validation_suite,
        input.identifier,
        input.estimator,
        "static_ate",
    )
}

/// Assemble and validate a plan; every compile entry point ends here.
fn finish_plan(
    record: LogicalAnalysisPlanRecord,
    query: CausalQuery,
    split: Option<DiscoveryEstimationSplit>,
    row_count_hint: u64,
) -> Result<LogicalAnalysisPlan, CausalError> {
    let plan = LogicalAnalysisPlan {
        design_columns_hint: record.query_variables.len().max(1) as u64,
        record,
        query,
        split,
        row_count_hint,
    };
    plan.validate()?;
    Ok(plan)
}

/// Record for a tabular plan with no discovery stage and no graph review.
fn tabular_record(
    plan_id: &'static str,
    identifier: Arc<str>,
    estimator: Arc<str>,
    validation_suite: Option<Arc<str>>,
    query_variables: Arc<[VariableId]>,
) -> LogicalAnalysisPlanRecord {
    LogicalAnalysisPlanRecord {
        plan_id: Arc::from(plan_id),
        data_classification: DataClassification::Tabular,
        discovery_algorithm: None,
        graph_review_required: false,
        identifier: Some(identifier),
        estimator: Some(estimator),
        validation_suite,
        query_variables,
    }
}

/// Plan assembly shared by the DAG and class-aware (PAG / CPDAG) static ATE compiles,
/// run after the caller has validated the strategy pair.
fn finish_static_ate(
    data: &TabularData,
    query: &AverageEffectQuery,
    validation_suite: Option<Arc<str>>,
    identifier: Arc<str>,
    estimator: Arc<str>,
    plan_id: &'static str,
) -> Result<LogicalAnalysisPlan, CausalError> {
    let record = tabular_record(
        plan_id,
        identifier,
        estimator,
        validation_suite,
        Arc::from([query.treatment, query.outcome]),
    );
    finish_plan(record, CausalQuery::AverageEffect(query.clone()), None, data.row_count() as u64)
}

/// Compile a continuous-response query over tabular data and a static DAG.
///
/// # Errors
///
/// Invalid queries, out-of-range variables, or incompatible strategies.
pub(crate) fn compile_logical_static_response(
    input: StaticResponseCompileInput<'_>,
) -> Result<LogicalAnalysisPlan, CausalError> {
    input.query.validate()?;
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
    if estimator != expected
        && !(estimator == EstimatorId::ResponseBayesian
            && matches!(
                input.query.functional,
                ResponseFunctional::MeanCurve { .. }
                    | ResponseFunctional::InterventionResponse { .. }
            ))
        && !(estimator == EstimatorId::CellAipw
            && matches!(input.query.functional, ResponseFunctional::InterventionResponse { .. }))
    {
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
    finish_plan(
        tabular_record(
            "static_response",
            input.identifier,
            input.estimator,
            input.validation_suite,
            query_variables,
        ),
        CausalQuery::Response(input.query.clone()),
        None,
        input.data.row_count() as u64,
    )
}

fn response_query_variables(functional: &ResponseFunctional) -> (Vec<VariableId>, Vec<VariableId>) {
    (functional.treatment_ids(), functional.outcome_ids())
}

/// Compile logical plan for static ATE on a PAG.
///
/// # Errors
///
/// Query validation or incompatible identifier/estimator.
pub(crate) fn compile_logical_static_pag_ate(
    input: StaticPagAteCompileInput<'_>,
) -> Result<LogicalAnalysisPlan, CausalError> {
    input.query.validate()?;
    validate_query_vars_in_pag(input.graph, input.query.treatment, input.query.outcome)?;
    compile_logical_static_class_ate(
        input.data,
        input.query,
        input.validation_suite,
        input.identifier,
        input.estimator,
        "static_pag_ate",
        "PAG",
    )
}

/// Compile logical plan for static ATE on a CPDAG.
///
/// # Errors
///
/// Query validation or incompatible identifier/estimator.
pub(crate) fn compile_logical_static_cpdag_ate(
    input: StaticCpdagAteCompileInput<'_>,
) -> Result<LogicalAnalysisPlan, CausalError> {
    input.query.validate()?;
    validate_query_vars_in_cpdag(input.graph, input.query.treatment, input.query.outcome)?;
    compile_logical_static_class_ate(
        input.data,
        input.query,
        input.validation_suite,
        input.identifier,
        input.estimator,
        "static_cpdag_ate",
        "CPDAG",
    )
}

/// Shared body of the static class-aware ATE compile (PAG / CPDAG): the query
/// and graph-membership checks are done by the caller; `label` names the class
/// in the identifier-refusal message.
fn compile_logical_static_class_ate(
    data: &TabularData,
    query: &AverageEffectQuery,
    validation_suite: Option<Arc<str>>,
    identifier: Arc<str>,
    estimator: Arc<str>,
    plan_id: &'static str,
    label: &str,
) -> Result<LogicalAnalysisPlan, CausalError> {
    let identifier_id: IdentifierId = identifier.parse()?;
    let estimator_id: EstimatorId = estimator.parse()?;
    if !matches!(identifier_id, IdentifierId::GeneralizedAdjustment) {
        return Err(CausalError::Compile {
            message: format!(
                "{label} ATE requires identifier \"generalized.adjustment\"; got {:?}",
                identifier_id.as_str()
            ),
        });
    }
    validate_static_pair(identifier_id, estimator_id)?;
    refuse_linear_adjustment_population(estimator_id, query)?;
    finish_static_ate(data, query, validation_suite, identifier, estimator, plan_id)
}

/// Compile a `MeanCurve` / `InterventionResponse` query over a supplied PAG.
///
/// # Errors
///
/// Invalid queries, unsupported functionals, or incompatible strategies.
pub(crate) fn compile_logical_static_pag_response(
    input: StaticPagResponseCompileInput<'_>,
) -> Result<LogicalAnalysisPlan, CausalError> {
    require_class_aware_response(input.query, "PAG")?;
    let (treatments, outcomes) = response_query_variables(&input.query.functional);
    for &treatment in &treatments {
        for &outcome in &outcomes {
            validate_query_vars_in_pag(input.graph, treatment, outcome)?;
        }
    }
    compile_logical_class_response(
        ClassResponseCompile {
            data: input.data,
            query: input.query,
            validation_suite: input.validation_suite,
            identifier: input.identifier,
            estimator: input.estimator,
        },
        "static_pag_response",
        &treatments,
        &outcomes,
    )
}

/// Compile a `MeanCurve` / `InterventionResponse` query over a supplied CPDAG.
///
/// # Errors
///
/// Invalid queries, unsupported functionals, or incompatible strategies.
pub(crate) fn compile_logical_static_cpdag_response(
    input: StaticCpdagResponseCompileInput<'_>,
) -> Result<LogicalAnalysisPlan, CausalError> {
    require_class_aware_response(input.query, "CPDAG")?;
    let (treatments, outcomes) = response_query_variables(&input.query.functional);
    for &treatment in &treatments {
        for &outcome in &outcomes {
            validate_query_vars_in_cpdag(input.graph, treatment, outcome)?;
        }
    }
    compile_logical_class_response(
        ClassResponseCompile {
            data: input.data,
            query: input.query,
            validation_suite: input.validation_suite,
            identifier: input.identifier,
            estimator: input.estimator,
        },
        "static_cpdag_response",
        &treatments,
        &outcomes,
    )
}

fn require_class_aware_response(query: &ResponseQuery, class_tag: &str) -> Result<(), CausalError> {
    query.validate()?;
    if query.temporal.is_some() {
        return Err(CausalError::Compile {
            message: format!("{class_tag} response does not accept a temporal attachment"),
        });
    }
    if query.observation != ObservationSpec::Complete {
        return Err(CausalError::Compile {
            message: format!("{class_tag} response currently requires complete observations"),
        });
    }
    // Every static response functional has a class-aware reading: curves and intervention
    // levels, and the derivative family (point, average, directional, Jacobian), which is
    // identified on each completion by the adjustment set the regression conditions on.
    Ok(())
}

/// The query, data and strategy names shared by every class-aware response compile.
struct ClassResponseCompile<'a> {
    data: &'a TabularData,
    query: &'a ResponseQuery,
    validation_suite: Option<Arc<str>>,
    identifier: Arc<str>,
    estimator: Arc<str>,
}

fn compile_logical_class_response(
    input: ClassResponseCompile<'_>,
    plan_id: &'static str,
    treatments: &[VariableId],
    outcomes: &[VariableId],
) -> Result<LogicalAnalysisPlan, CausalError> {
    let ClassResponseCompile { data, query, validation_suite, identifier, estimator } = input;
    let identifier_id: IdentifierId = identifier.parse()?;
    let estimator_id: EstimatorId = estimator.parse()?;
    if !matches!(identifier_id, IdentifierId::GeneralizedAdjustment) {
        return Err(CausalError::Compile {
            message: format!(
                "Cpdag/Pag response requires identifier \"generalized.adjustment\"; got {:?}",
                identifier_id.as_str()
            ),
        });
    }
    validate_class_response_pair(identifier_id, estimator_id)?;
    let expected = EstimatorId::default_for_response(&query.functional);
    if estimator_id != expected
        && estimator_id != EstimatorId::ResponseBayesian
        && !(estimator_id == EstimatorId::CellAipw
            && matches!(
                query.functional,
                antecedent_core::ResponseFunctional::InterventionResponse { .. }
            ))
    {
        return Err(CausalError::Compile {
            message: format!(
                "response functional requires estimator {:?} or response.bayesian; got {:?}",
                expected.as_str(),
                estimator_id.as_str()
            ),
        });
    }
    let query_variables: Arc<[VariableId]> =
        treatments.iter().chain(outcomes.iter()).copied().collect::<Vec<_>>().into();
    finish_plan(
        tabular_record(plan_id, identifier, estimator, validation_suite, query_variables),
        CausalQuery::Response(query.clone()),
        None,
        data.row_count() as u64,
    )
}

/// Compile logical plan for an interventional distribution.
///
/// # Errors
///
/// Query validation or incompatible identifier/estimator.
pub(crate) fn compile_logical_distribution(
    input: StaticDistributionCompileInput<'_>,
) -> Result<LogicalAnalysisPlan, CausalError> {
    input.query.validate()?;
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
    let record = tabular_record(
        "static_distribution",
        input.identifier,
        input.estimator,
        input.validation_suite,
        Arc::from(qvars),
    );
    finish_plan(
        record,
        CausalQuery::Distribution(input.query.clone()),
        None,
        input.data.row_count() as u64,
    )
}

/// Compile logical plan for path-specific natural effects.
///
/// # Errors
///
/// Query validation or incompatible identifier/estimator.
pub(crate) fn compile_logical_path_specific(
    input: StaticPathSpecificCompileInput<'_>,
) -> Result<LogicalAnalysisPlan, CausalError> {
    input.query.validate()?;
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
    let record = tabular_record(
        "static_path_specific",
        input.identifier,
        input.estimator,
        input.validation_suite,
        Arc::from(qvars),
    );
    finish_plan(
        record,
        CausalQuery::PathSpecific(input.query.clone()),
        None,
        input.data.row_count() as u64,
    )
}

/// Compile logical plan for a path-specific natural effect on a supplied ADMG.
///
/// The plan records `plan_id`; the caller may replace the recorded query by the query the
/// path-specific effect answers (a natural direct effect reads the mediator as its path node).
///
/// # Errors
///
/// Query validation or incompatible identifier/estimator.
pub(crate) fn compile_logical_path_specific_admg(
    input: StaticAdmgPathSpecificCompileInput<'_>,
    plan_id: &'static str,
) -> Result<LogicalAnalysisPlan, CausalError> {
    let nodes = input.graph.nodes();
    compile_logical_path_specific_on(nodes, "ADMG", input.into_parts(), plan_id)
}

/// Compile logical plan for a path-specific natural effect over the completions of a
/// supplied CPDAG. Same recording rules as [`compile_logical_path_specific_admg`].
///
/// # Errors
///
/// Query validation or incompatible identifier/estimator.
pub(crate) fn compile_logical_path_specific_cpdag(
    input: StaticCpdagPathSpecificCompileInput<'_>,
    plan_id: &'static str,
) -> Result<LogicalAnalysisPlan, CausalError> {
    let nodes = input.graph.nodes();
    compile_logical_path_specific_on(nodes, "CPDAG", input.into_parts(), plan_id)
}

/// Compile logical plan for a path-specific natural effect over the ADMGs a supplied PAG
/// leaves open. Same recording rules as [`compile_logical_path_specific_admg`].
///
/// # Errors
///
/// Query validation or incompatible identifier/estimator.
pub(crate) fn compile_logical_path_specific_pag(
    input: StaticPagPathSpecificCompileInput<'_>,
    plan_id: &'static str,
) -> Result<LogicalAnalysisPlan, CausalError> {
    let nodes = input.graph.nodes();
    compile_logical_path_specific_on(nodes, "PAG", input.into_parts(), plan_id)
}

/// What a path-specific compile reads besides its graph.
struct PathSpecificParts<'a> {
    data: &'a TabularData,
    query: &'a antecedent_core::PathSpecificEffectQuery,
    validation_suite: Option<Arc<str>>,
    identifier: Arc<str>,
    estimator: Arc<str>,
}

impl<'a, G> StaticCompileInput<'a, G, antecedent_core::PathSpecificEffectQuery> {
    fn into_parts(self) -> PathSpecificParts<'a> {
        PathSpecificParts {
            data: self.data,
            query: self.query,
            validation_suite: self.validation_suite,
            identifier: self.identifier,
            estimator: self.estimator,
        }
    }
}

fn compile_logical_path_specific_on(
    nodes: &[antecedent_graph::NodeRef],
    label: &str,
    parts: PathSpecificParts<'_>,
    plan_id: &'static str,
) -> Result<LogicalAnalysisPlan, CausalError> {
    parts.query.validate()?;
    if parts.query.target_population != TargetPopulation::AllObserved {
        return Err(CausalError::Compile {
            message: "functional.effect only supports TargetPopulation::AllObserved".into(),
        });
    }
    validate_query_vars_in_static_nodes(nodes, label, parts.query.treatment, parts.query.outcome)?;
    let identifier: IdentifierId = parts.identifier.parse()?;
    let estimator: EstimatorId = parts.estimator.parse()?;
    validate_path_specific_pair(identifier, estimator)?;
    let mut qvars = vec![parts.query.treatment, parts.query.outcome];
    for &m in parts.query.path_nodes.iter() {
        if !qvars.contains(&m) {
            qvars.push(m);
        }
    }
    let record = tabular_record(
        plan_id,
        parts.identifier,
        parts.estimator,
        parts.validation_suite,
        Arc::from(qvars),
    );
    finish_plan(
        record,
        CausalQuery::PathSpecific(parts.query.clone()),
        None,
        parts.data.row_count() as u64,
    )
}

/// Compile a `CoDetermined` joint `InterventionResponse` on the closure ADMG.
///
/// # Errors
/// Invalid query or identifier/estimator pair.
pub(crate) fn compile_logical_codetermined_joint(
    data: &TabularData,
    admg: &Admg,
    query: &ResponseQuery,
    validation_suite: Option<Arc<str>>,
    identifier: Arc<str>,
    estimator: Arc<str>,
) -> Result<LogicalAnalysisPlan, CausalError> {
    query.validate()?;
    let identifier_id: IdentifierId = identifier.parse()?;
    let estimator_id: EstimatorId = estimator.parse()?;
    if identifier_id != IdentifierId::GeneralizedAdjustment
        || !matches!(estimator_id, EstimatorId::CellAipw | EstimatorId::ResponseBayesian)
    {
        return Err(CausalError::Compile {
            message: format!(
                "CoDetermined joint cells require generalized.adjustment + cell.aipw (or \
                 response.bayesian under Bayesian inference); got {:?} / {:?}",
                identifier_id.as_str(),
                estimator_id.as_str()
            ),
        });
    }
    let (treatments, outcomes) = response_query_variables(&query.functional);
    for &treatment in &treatments {
        for &outcome in &outcomes {
            validate_query_vars_in_admg(admg, treatment, outcome)?;
        }
    }
    let query_variables: Arc<[VariableId]> =
        treatments.iter().chain(outcomes.iter()).copied().collect::<Vec<_>>().into();
    finish_plan(
        tabular_record(
            "codetermined_joint_response",
            identifier,
            estimator,
            validation_suite,
            query_variables,
        ),
        CausalQuery::Response(query.clone()),
        None,
        data.row_count() as u64,
    )
}

fn validate_query_vars_in_admg(
    admg: &Admg,
    treatment: antecedent_core::VariableId,
    outcome: antecedent_core::VariableId,
) -> Result<(), CausalError> {
    validate_query_vars_in_static_nodes(admg.nodes(), "ADMG", treatment, outcome)
}

fn validate_query_vars_in_dag(
    dag: &Dag,
    treatment: antecedent_core::VariableId,
    outcome: antecedent_core::VariableId,
) -> Result<(), CausalError> {
    validate_query_vars_in_static_nodes(dag.nodes(), "DAG", treatment, outcome)
}

fn validate_query_vars_in_cpdag(
    cpdag: &Cpdag,
    treatment: antecedent_core::VariableId,
    outcome: antecedent_core::VariableId,
) -> Result<(), CausalError> {
    validate_query_vars_in_static_nodes(cpdag.nodes(), "CPDAG", treatment, outcome)
}

fn validate_query_vars_in_pag(
    pag: &Pag,
    treatment: antecedent_core::VariableId,
    outcome: antecedent_core::VariableId,
) -> Result<(), CausalError> {
    validate_query_vars_in_static_nodes(pag.nodes(), "PAG", treatment, outcome)
}

/// Membership check shared by the static graph-class validators: the query's
/// treatment and outcome must each name a static node of the graph.
fn validate_query_vars_in_static_nodes(
    nodes: &[antecedent_graph::NodeRef],
    label: &str,
    treatment: antecedent_core::VariableId,
    outcome: antecedent_core::VariableId,
) -> Result<(), CausalError> {
    let mut has_t = false;
    let mut has_y = false;
    for node in nodes {
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
                "query variables not in {label} (treatment present={has_t}, outcome present={has_y})"
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
    validate_query_vars_in_temporal_nodes(dag.nodes(), treatment, outcome)
}

fn validate_query_vars_in_temporal_nodes(
    nodes: &[antecedent_graph::NodeRef],
    treatment: antecedent_core::VariableId,
    outcome: antecedent_core::VariableId,
) -> Result<(), CausalError> {
    // A node-less DAG is the placeholder the graph-posterior path supplies: the
    // structure lives in the `GraphPosterior` mixture, not here, so there is no
    // membership to check and rejecting would be a false negative.
    if nodes.is_empty() {
        return Ok(());
    }
    let mut has_t = false;
    let mut has_y = false;
    for node in nodes {
        // Temporal graphs carry `Lagged` nodes, and `Context` nodes when an
        // environment is attached; both name a variable the query may reference.
        let variable = match node {
            antecedent_graph::NodeRef::Lagged { variable, .. }
            | antecedent_graph::NodeRef::Unfolded { variable, .. }
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
                "query variables not in temporal graph (treatment present={has_t}, outcome \
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
pub(crate) fn compile_logical_temporal_effect(
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

/// Temporal dose-over-horizon / policy-path response (ADR 0021).
///
/// # Errors
///
/// Query validation failures or variables missing from the temporal DAG.
pub(crate) fn compile_logical_temporal_response(
    data: &TimeSeriesData,
    graph: &TemporalDag,
    query: &ResponseQuery,
    review_required: bool,
) -> Result<LogicalAnalysisPlan, CausalError> {
    query.validate()?;
    query.require_licensed_temporal_observation()?;
    let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
        message: "temporal response compile requires ResponseQuery.temporal".into(),
    })?;
    let (treatment, outcome) = response_primary_pair(&query.functional)?;
    for variable in query.functional.treatment_ids() {
        validate_query_vars_in_temporal_dag(graph, variable, outcome)?;
    }
    if query.target_population != TargetPopulation::AllObserved {
        return Err(CausalError::Compile {
            message: format!(
                "temporal response only supports TargetPopulation::AllObserved (got {:?})",
                query.target_population
            ),
        });
    }
    let _ = temporal;
    let record = LogicalAnalysisPlanRecord {
        plan_id: Arc::from("temporal_response"),
        data_classification: DataClassification::Temporal,
        discovery_algorithm: None,
        graph_review_required: review_required,
        identifier: Some(Arc::from("temporal.backdoor.unfolded")),
        estimator: Some(Arc::from("temporal.response.gcomp")),
        validation_suite: None,
        query_variables: Arc::from([treatment, outcome]),
    };
    finish_plan(record, CausalQuery::Response(query.clone()), None, data.row_count() as u64)
}

fn response_primary_pair(
    functional: &ResponseFunctional,
) -> Result<(VariableId, VariableId), CausalError> {
    crate::analysis::proof_support::require_primary_pair(functional)
}

/// Temporal effect plan with an explicit data classification (Event / Panel / Temporal).
///
/// # Errors
///
/// Query validation failures.
pub(crate) fn compile_logical_temporal_effect_classified(
    data: &TimeSeriesData,
    graph: &TemporalDag,
    query: &TemporalEffectQuery,
    split: Option<DiscoveryEstimationSplit>,
    review_required: bool,
    data_classification: DataClassification,
) -> Result<LogicalAnalysisPlan, CausalError> {
    query.validate()?;
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
    finish_plan(record, CausalQuery::TemporalEffect(query.clone()), split, row_count_hint)
}

/// Inputs for [`compile_logical_temporal_class_effect`].
#[derive(Clone, Debug)]
pub(crate) struct TemporalClassEffectCompileInput<'a> {
    /// Time-series data.
    pub data: &'a TimeSeriesData,
    /// Nodes of the supplied temporal class graph.
    pub nodes: &'a [antecedent_graph::NodeRef],
    /// Pulse or Sustained effect query.
    pub query: &'a TemporalEffectQuery,
    /// Discovery/estimation split, if any.
    pub split: Option<DiscoveryEstimationSplit>,
    /// Data classification recorded on the plan.
    pub data_classification: DataClassification,
    /// Plan id recorded on the plan.
    pub plan_id: &'static str,
    /// Identifier (must be generalized.adjustment).
    pub identifier: Arc<str>,
    /// Estimator id.
    pub estimator: Arc<str>,
}

/// Pulse or Sustained on an incomplete `TemporalCpdag` or `TemporalPag`.
///
/// # Errors
///
/// Query validation, multi-step Sustained, or incompatible identifier/estimator.
pub(crate) fn compile_logical_temporal_class_effect(
    input: TemporalClassEffectCompileInput<'_>,
) -> Result<LogicalAnalysisPlan, CausalError> {
    let TemporalClassEffectCompileInput {
        data,
        nodes,
        query,
        split,
        data_classification,
        plan_id,
        identifier,
        estimator,
    } = input;
    query.validate()?;
    validate_query_vars_in_temporal_nodes(nodes, query.treatment, query.outcome)?;
    if query.target_population != TargetPopulation::AllObserved {
        return Err(CausalError::Compile {
            message: format!(
                "temporal class-aware effect only supports TargetPopulation::AllObserved \
                 (got {:?})",
                query.target_population
            ),
        });
    }
    let identifier_id: IdentifierId = identifier.parse()?;
    let estimator_id: EstimatorId = estimator.parse()?;
    if !matches!(identifier_id, IdentifierId::GeneralizedAdjustment) {
        return Err(CausalError::Compile {
            message: format!(
                "TemporalCpdag/TemporalPag effect requires identifier \"generalized.adjustment\"; \
                 got {:?}",
                identifier_id.as_str()
            ),
        });
    }
    let multi_step = matches!(
        query.policy,
        antecedent_core::TemporalPolicy::Sustained { from, until } if from != until
    );
    if multi_step {
        if !matches!(
            estimator_id,
            EstimatorId::TemporalLinearAdjustment | EstimatorId::TemporalSequentialGcomp
        ) {
            return Err(CausalError::Compile {
                message: format!(
                    "TemporalCpdag/TemporalPag multi-step Sustained requires estimator \
                     \"temporal.sequential.gcomp\"; got {:?}",
                    estimator_id.as_str()
                ),
            });
        }
    } else if !matches!(estimator_id, EstimatorId::TemporalLinearAdjustment) {
        return Err(CausalError::Compile {
            message: format!(
                "TemporalCpdag/TemporalPag effect requires estimator \
                 \"temporal.linear.adjustment\"; got {:?}",
                estimator_id.as_str()
            ),
        });
    }
    let row_count_hint =
        split.map_or_else(|| data.row_count() as u64, |s| s.estimation.len() as u64);
    let record = LogicalAnalysisPlanRecord {
        plan_id: Arc::from(plan_id),
        data_classification,
        discovery_algorithm: None,
        graph_review_required: false,
        identifier: Some(identifier),
        estimator: Some(estimator),
        validation_suite: None,
        query_variables: Arc::from([query.treatment, query.outcome]),
    };
    finish_plan(record, CausalQuery::TemporalEffect(query.clone()), split, row_count_hint)
}
