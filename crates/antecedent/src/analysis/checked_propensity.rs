//! Checked target and procedure bindings for DAG propensity estimators.
//!
//! Weighting and matching share a prepared propensity design, but deliberately retain
//! different fit procedures and uncertainty products.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, Intervention, OutcomeFunctional,
    TargetPopulation, Value,
};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::{
    EffectEstimate, PropensityEstimationWorkspace, PropensityMatching, PropensityWeighting,
};
use antecedent_expr::{
    FactorRequirement, FunctionalProgram, IdentifiedEstimand, ProgramLimits, ProgramSchema,
    ProgramVariable,
};
use antecedent_identify::{IdentificationResult, IdentificationStatus};

use crate::accepted::GraphClass;
use crate::analysis::builder::RefuteSuite;
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::planner::PhysicalExecutionPlan;
use crate::strategy_table::{EstimatorId, IdentifierId};
use crate::support::{CellStatus, StructureSource};

/// Procedure identity for a checked propensity estimator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckedPropensityProcedure {
    /// Self-normalized Hajek weighting.
    HajekWeighting,
    /// Nearest-neighbor matching with Abadie–Imbens analytic uncertainty.
    AbadieImbensMatching,
}

/// Distinct uncertainty semantics retained by the checked procedure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckedPropensityUncertainty {
    /// The weighting estimator's analytic SE and optional propensity-refitted bootstrap SE.
    HajekAnalyticAndBootstrap { bootstrap_replicates: u32 },
    /// Matching's Abadie–Imbens analytic SE; nonparametric bootstrap is intentionally absent.
    AbadieImbensAnalytic { se_kind: antecedent_estimate::AnalyticSeKind },
}

/// Typed target roles and procedure bindings for a checked propensity execution.
#[derive(Clone, Debug)]
pub(crate) struct CheckedPropensityLowering {
    pub(crate) functional: antecedent_expr::ExprId,
    pub(crate) treatment: antecedent_core::VariableId,
    pub(crate) outcome: antecedent_core::VariableId,
    pub(crate) adjustment: Arc<[antecedent_core::VariableId]>,
    pub(crate) active: f64,
    pub(crate) control: f64,
    pub(crate) population: TargetPopulation,
    pub(crate) procedure: CheckedPropensityProcedure,
    pub(crate) uncertainty: CheckedPropensityUncertainty,
    pub(crate) overlap: antecedent_estimate::OverlapPolicy,
    pub(crate) source_rows: Arc<[u32]>,
}

/// Frozen semantic and analysis context used to construct a checked propensity operation.
#[derive(Clone, Debug)]
pub(crate) struct CheckedPropensityContext {
    pub(crate) source_query: CausalQuery,
    pub(crate) identification: IdentificationResult,
    pub(crate) estimand_index: usize,
    pub(crate) identifier: IdentifierId,
    pub(crate) physical: PhysicalExecutionPlan,
    pub(crate) graph_class: GraphClass,
    pub(crate) graph_version: u32,
    pub(crate) support_status: Option<CellStatus>,
    pub(crate) structure_source: StructureSource,
    pub(crate) inference: InferenceMode,
    pub(crate) refute: RefuteSuite,
    pub(crate) population_registry: Option<antecedent_core::PopulationRegistry>,
    pub(crate) latency_mode: Option<super::latency::LatencyMode>,
    pub(crate) custom_validator_names: Arc<[Arc<str>]>,
}

#[derive(Clone, Debug)]
enum CheckedPropensityFit {
    Weighting {
        fitter: PropensityWeighting,
        problem: antecedent_estimate::PreparedPropensityProblem,
    },
    Matching {
        fitter: PropensityMatching,
        problem: antecedent_estimate::PreparedPropensityProblem,
    },
}

/// Checked causal program, identified target, selected propensity procedure, and bound design.
#[derive(Clone, Debug)]
pub(crate) struct CheckedPropensityOperation {
    context: CheckedPropensityContext,
    query: AverageEffectQuery,
    target: IdentifiedEstimand,
    program: FunctionalProgram,
    factor_requirements: Arc<[FactorRequirement]>,
    lowering: CheckedPropensityLowering,
    assumptions: antecedent_core::AssumptionSet,
    fit: CheckedPropensityFit,
}

impl CheckedPropensityOperation {
    /// Validate a single identified backdoor target and prepare its selected propensity fit.
    pub(crate) fn prepare(
        data: &TabularData,
        mut context: CheckedPropensityContext,
        estimator: EstimatorId,
        fitter: impl Into<CheckedPropensityEstimator>,
    ) -> Result<Self, CausalError> {
        let query = match &context.source_query {
            CausalQuery::AverageEffect(query) => query.clone(),
            _ => return Err(compile_error("checked propensity requires an AverageEffect query")),
        };
        if context.identification.query != context.source_query
            || context.identification.status != IdentificationStatus::NonparametricallyIdentified
            || context.graph_class != GraphClass::Dag
            || !matches!(
                context.structure_source,
                StructureSource::Explicit | StructureSource::Accepted
            )
            || !matches!(context.inference, InferenceMode::Frequentist)
            || !matches!(
                context.refute,
                RefuteSuite::None
                    | RefuteSuite::Cheap
                    | RefuteSuite::PlaceboAndRcc
                    | RefuteSuite::Full
            )
            || !context.custom_validator_names.is_empty()
            || !matches!(query.outcome_functional, OutcomeFunctional::Mean)
        {
            return Err(compile_error(
                "checked propensity route premises are outside the licensed DAG frequentist mean-effect contract",
            ));
        }
        let target =
            context.identification.estimands.get(context.estimand_index).cloned().ok_or_else(
                || compile_error("checked propensity estimand index is out of range"),
            )?;
        let claim = context.identification.claim(context.estimand_index).ok_or_else(|| {
            compile_error("checked propensity estimand has no scoped identification claim")
        })?;
        if claim.status != IdentificationStatus::NonparametricallyIdentified
            || !target.is_adjustment_shaped()
            || !target.instruments.is_empty()
            || !target.mediators.is_empty()
        {
            return Err(compile_error(
                "checked propensity requires a nonparametrically identified backdoor adjustment target",
            ));
        }
        let mut adjustment = target.adjustment_set.to_vec();
        adjustment.sort_unstable();
        adjustment.dedup();
        if adjustment.len() != target.adjustment_set.len()
            || adjustment.contains(&query.treatment)
            || adjustment.contains(&query.outcome)
        {
            return Err(compile_error(
                "checked propensity adjustment roles must be unique and exclude treatment and outcome",
            ));
        }
        let active = set_value(&query.active, query.treatment)?;
        let control = set_value(&query.control, query.treatment)?;
        if (active - 1.0).abs() > 1e-12 || control.abs() > 1e-12 {
            return Err(compile_error(
                "checked propensity requires binary active=1 and control=0 treatment arms",
            ));
        }
        let fitter = fitter.into();
        let expected_estimator = estimator_id(&fitter);
        if estimator != expected_estimator
            || context.physical.logical.record.estimator.as_deref() != Some(estimator.as_str())
        {
            return Err(compile_error(
                "checked propensity estimator does not match the selected physical plan",
            ));
        }
        context.identifier = context
            .physical
            .logical
            .record
            .identifier
            .as_deref()
            .unwrap_or(context.identifier.as_str())
            .parse()?;

        let mut arena = context.identification.arena.clone();
        let expected = arena.backdoor_ate(
            query.treatment,
            query.outcome,
            &target.adjustment_set,
            Value::f64(active),
            Value::f64(control),
        );
        if expected != target.functional {
            return Err(compile_error(
                "identified propensity functional disagrees with treatment, arms, or adjustment roles",
            ));
        }
        let schema = ProgramSchema::new(
            data.schema()
                .variables()
                .iter()
                .map(|v| (v.id, ProgramVariable { name: Arc::clone(&v.name) })),
        );
        let program = FunctionalProgram::new(
            arena,
            schema,
            target.functional,
            target.functional,
            ProgramLimits::default(),
        )
        .map_err(|e| compile_error(&format!("checked propensity target program: {e}")))?;
        let factor_requirements = Arc::from(program.factor_requirements().to_vec());
        let fit = prepare_fit(data, &target, &query, fitter, &context)?;
        let problem = fit.problem();
        if problem.method != target.method
            || problem.adjustment_set != target.adjustment_set
            || problem.treatment_id != query.treatment
            || problem.target_population != query.target_population
            || problem.treatment.len() != problem.row_index.len()
        {
            return Err(compile_error(
                "prepared propensity design disagrees with the selected target roles or row binding",
            ));
        }
        let (procedure, uncertainty) = match &fit {
            CheckedPropensityFit::Weighting { fitter, .. } => (
                CheckedPropensityProcedure::HajekWeighting,
                CheckedPropensityUncertainty::HajekAnalyticAndBootstrap {
                    bootstrap_replicates: fitter.bootstrap_replicates,
                },
            ),
            CheckedPropensityFit::Matching { fitter, .. } => (
                CheckedPropensityProcedure::AbadieImbensMatching,
                CheckedPropensityUncertainty::AbadieImbensAnalytic { se_kind: fitter.se_kind },
            ),
        };
        let lowering = CheckedPropensityLowering {
            functional: target.functional,
            treatment: query.treatment,
            outcome: query.outcome,
            adjustment: Arc::clone(&target.adjustment_set),
            active,
            control,
            population: query.target_population.clone(),
            procedure,
            uncertainty,
            overlap: problem.overlap,
            source_rows: Arc::clone(&problem.row_index),
        };
        Ok(Self {
            context,
            query,
            target,
            program,
            factor_requirements,
            lowering,
            assumptions: claim.required_assumptions,
            fit,
        })
    }

    pub(crate) fn context(&self) -> &CheckedPropensityContext {
        &self.context
    }
    pub(crate) fn query(&self) -> &AverageEffectQuery {
        &self.query
    }
    pub(crate) fn target(&self) -> &IdentifiedEstimand {
        &self.target
    }
    pub(crate) fn lowering(&self) -> &CheckedPropensityLowering {
        &self.lowering
    }

    /// Rebuild only the row-bound design; target and procedure selection stay frozen.
    pub(crate) fn rebind(&self, data: &TabularData) -> Result<Self, CausalError> {
        let schema = ProgramSchema::new(
            data.schema()
                .variables()
                .iter()
                .map(|v| (v.id, ProgramVariable { name: Arc::clone(&v.name) })),
        );
        if self.program.schema() != &schema
            || self.program.mapping().source != self.target.functional
            || self.program.mapping().executable != self.target.functional
            || self.lowering.functional != self.target.functional
            || self.lowering.treatment != self.query.treatment
            || self.lowering.outcome != self.query.outcome
            || self.lowering.adjustment != self.target.adjustment_set
            || self.lowering.population != self.query.target_population
            || self.lowering.active != set_value(&self.query.active, self.query.treatment)?
            || self.lowering.control != set_value(&self.query.control, self.query.treatment)?
            || self.factor_requirements.as_ref() != self.program.factor_requirements()
        {
            return Err(compile_error("checked propensity target or schema changed during rebind"));
        }
        let fit = match &self.fit {
            CheckedPropensityFit::Weighting { fitter, .. } => {
                let problem =
                    fitter.prepare(data, &self.target, &self.query).map_err(CausalError::from)?;
                CheckedPropensityFit::Weighting { fitter: fitter.clone(), problem }
            }
            CheckedPropensityFit::Matching { fitter, .. } => {
                let problem =
                    fitter.prepare(data, &self.target, &self.query).map_err(CausalError::from)?;
                CheckedPropensityFit::Matching { fitter: fitter.clone(), problem }
            }
        };
        if fit.problem().overlap != self.lowering.overlap {
            return Err(compile_error(
                "checked propensity overlap procedure changed during rebind",
            ));
        }
        let mut rebound = self.clone();
        rebound.lowering.source_rows = Arc::clone(&fit.problem().row_index);
        rebound.fit = fit;
        Ok(rebound)
    }

    /// Execute the selected estimator and its own uncertainty procedure.
    pub(crate) fn execute(&self, ctx: &ExecutionContext) -> Result<EffectEstimate, CausalError> {
        let mut workspace = PropensityEstimationWorkspace::default();
        match &self.fit {
            CheckedPropensityFit::Weighting { fitter, problem } => fitter
                .fit(problem, &mut workspace, ctx, self.assumptions.clone())
                .map_err(CausalError::from),
            CheckedPropensityFit::Matching { fitter, problem } => fitter
                .fit(problem, &mut workspace, ctx, self.assumptions.clone())
                .map_err(CausalError::from),
        }
    }
}

/// Procedure configuration accepted by the shared checked operation.
#[derive(Clone, Debug)]
pub(crate) enum CheckedPropensityEstimator {
    Weighting(PropensityWeighting),
    Matching(PropensityMatching),
}

impl From<PropensityWeighting> for CheckedPropensityEstimator {
    fn from(value: PropensityWeighting) -> Self {
        Self::Weighting(value)
    }
}
impl From<PropensityMatching> for CheckedPropensityEstimator {
    fn from(value: PropensityMatching) -> Self {
        Self::Matching(value)
    }
}

impl CheckedPropensityFit {
    fn problem(&self) -> &antecedent_estimate::PreparedPropensityProblem {
        match self {
            Self::Weighting { problem, .. } | Self::Matching { problem, .. } => problem,
        }
    }
}

fn prepare_fit(
    data: &TabularData,
    target: &IdentifiedEstimand,
    query: &AverageEffectQuery,
    estimator: CheckedPropensityEstimator,
    context: &CheckedPropensityContext,
) -> Result<CheckedPropensityFit, CausalError> {
    match estimator {
        CheckedPropensityEstimator::Weighting(mut fitter) => {
            if fitter.population_registry.is_none() {
                fitter.population_registry = context.population_registry.clone();
            }
            let problem = fitter.prepare(data, target, query).map_err(CausalError::from)?;
            Ok(CheckedPropensityFit::Weighting { fitter, problem })
        }
        CheckedPropensityEstimator::Matching(mut fitter) => {
            if fitter.population_registry.is_none() {
                fitter.population_registry = context.population_registry.clone();
            }
            let problem = fitter.prepare(data, target, query).map_err(CausalError::from)?;
            Ok(CheckedPropensityFit::Matching { fitter, problem })
        }
    }
}

fn estimator_id(estimator: &CheckedPropensityEstimator) -> EstimatorId {
    match estimator {
        CheckedPropensityEstimator::Weighting(_) => EstimatorId::PropensityWeighting,
        CheckedPropensityEstimator::Matching(_) => EstimatorId::PropensityMatching,
    }
}

fn set_value(
    intervention: &Intervention,
    variable: antecedent_core::VariableId,
) -> Result<f64, CausalError> {
    match intervention {
        Intervention::Set { variable: id, value } if *id == variable => value
            .as_f64()
            .ok_or_else(|| compile_error("checked propensity treatment arms must be numeric")),
        _ => Err(compile_error(
            "checked propensity query arms must intervene on the declared treatment",
        )),
    }
}

fn compile_error(message: &str) -> CausalError {
    CausalError::Compile { message: message.into() }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
    use antecedent_data::TabularData;
    use antecedent_estimate::{PropensityMatching, PropensityWeighting};
    use antecedent_graph::{Dag, DenseNodeId};

    use super::*;
    use crate::analysis::builder::RefuteSuite;
    use crate::analysis::execute::Study;
    use crate::strategy_table::{EstimatorId, IdentifierId, identify_static_query};

    fn inputs(
        estimator: EstimatorId,
    ) -> (TabularData, CheckedPropensityContext, CheckedPropensityEstimator) {
        let n = 512;
        let z: Vec<f64> = (0..n).map(|i| ((i * 37 % 101) as f64 - 50.0) / 50.0).collect();
        let treatment: Vec<f64> = (0..n).map(|i| f64::from((i * 17 % 31) < 15)).collect();
        let outcome: Vec<f64> = treatment
            .iter()
            .zip(&z)
            .enumerate()
            .map(|(i, (t, z))| 2.0 * t + z + ((i * 13 % 47) as f64 - 23.0) / 100.0)
            .collect();
        let data = TabularData::from_f64_columns([
            ("t", treatment.as_slice()),
            ("y", outcome.as_slice()),
            ("z", z.as_slice()),
        ])
        .unwrap();
        let mut graph = Dag::with_variables(3);
        for (from, to) in [(2, 0), (2, 1), (0, 1)] {
            graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let selected_config = match estimator {
            EstimatorId::PropensityWeighting => CheckedPropensityEstimator::Weighting(
                PropensityWeighting::new().with_bootstrap_replicates(24),
            ),
            EstimatorId::PropensityMatching => CheckedPropensityEstimator::Matching(
                PropensityMatching::new().with_bootstrap_replicates(24),
            ),
            _ => unreachable!(),
        };
        let selected = Study::tabular(data.clone())
            .graph(graph.clone())
            .query(query.clone())
            .estimator(estimator)
            .build()
            .unwrap();
        let physical = selected.compile(&ExecutionContext::for_tests(90)).unwrap();
        let source_query = CausalQuery::AverageEffect(query);
        let identification =
            identify_static_query(IdentifierId::BackdoorAdjustment, &graph, &source_query).unwrap();
        let context = CheckedPropensityContext {
            source_query,
            identification,
            estimand_index: 0,
            identifier: IdentifierId::BackdoorAdjustment,
            physical,
            graph_class: GraphClass::Dag,
            graph_version: 0,
            support_status: None,
            structure_source: StructureSource::Explicit,
            inference: InferenceMode::Frequentist,
            refute: RefuteSuite::None,
            population_registry: None,
            latency_mode: None,
            custom_validator_names: Arc::from([]),
        };
        (data, context, selected_config)
    }

    #[test]
    fn weighting_and_matching_keep_distinct_uncertainty_and_known_truth() {
        for estimator in [EstimatorId::PropensityWeighting, EstimatorId::PropensityMatching] {
            let (data, context, config) = inputs(estimator);
            let query = context.identification.average_effect().unwrap().clone();
            let target = context.identification.estimands[0].clone();
            let operation =
                CheckedPropensityOperation::prepare(&data, context, estimator, config).unwrap();
            assert_eq!(operation.lowering().functional, target.functional);
            assert_eq!(operation.lowering().treatment, query.treatment);
            assert_eq!(
                operation.lowering().procedure,
                if estimator == EstimatorId::PropensityWeighting {
                    CheckedPropensityProcedure::HajekWeighting
                } else {
                    CheckedPropensityProcedure::AbadieImbensMatching
                }
            );
            let rebound = operation.rebind(&data).unwrap();
            assert_eq!(rebound.lowering().procedure, operation.lowering().procedure);
            let estimate = rebound.execute(&ExecutionContext::for_tests(91)).unwrap();
            assert!((estimate.ate - 2.0).abs() < 0.3, "{}: {}", estimator.as_str(), estimate.ate);
            match estimator {
                EstimatorId::PropensityWeighting => {
                    assert!(matches!(
                        operation.lowering().uncertainty,
                        CheckedPropensityUncertainty::HajekAnalyticAndBootstrap {
                            bootstrap_replicates: 24
                        }
                    ));
                    assert!(estimate.se_bootstrap.is_some());
                }
                EstimatorId::PropensityMatching => {
                    assert!(matches!(
                        operation.lowering().uncertainty,
                        CheckedPropensityUncertainty::AbadieImbensAnalytic { .. }
                    ));
                    assert!(estimate.se_analytic.is_finite());
                    assert!(estimate.se_bootstrap.is_none());
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn checked_propensity_refuses_mismatched_selected_estimator() {
        let (data, context, _) = inputs(EstimatorId::PropensityWeighting);
        let err = CheckedPropensityOperation::prepare(
            &data,
            context,
            EstimatorId::PropensityMatching,
            PropensityWeighting::new(),
        )
        .unwrap_err();
        assert!(matches!(err, CausalError::Compile { .. }));
    }
}
