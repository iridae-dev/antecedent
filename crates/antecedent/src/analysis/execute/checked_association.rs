//! Executor of the associational route: a noncausal association of an outcome with predictors
//! over the rows of a table or the points of one series.
//!
//! The route runs only from its retained lowered program. The plan must carry the program
//! lowered for the association cell, and the analysis the declared selection rule picks on the
//! data must be the one that program recorded in its model slot; a plan swapped for another
//! program, or for a program lowered on different data, is refused with `route_not_lowered`
//! before any work runs. The result is a [`crate::analysis::association::AssociationPayload`]:
//! never an intervention effect, never a ranking of actions, with the assumptions of a
//! description, and with every uncertainty labelled `unmeasured` because no coverage record
//! measures it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    AssociationQuery, Assumption, AssumptionRecord, AssumptionScope, AssumptionSet,
    AssumptionSource, AssumptionStatus,
};
use antecedent_data::OwnedColumnarStorage;
use antecedent_estimate::{
    AssociationEstimate, AssociationEstimator, BayesianSettings, Modality, descriptive_assumptions,
};

use super::bayesian_quasi_common::require_built_in_model;
use super::{
    Arc, AssembleArgs, CausalError, CausalExprArena, CausalQuery, DataInput, DerivationMeta,
    DerivationTrace, Diagnostic, DiagnosticKind, DiagnosticSeverity, DomainRef, ExecutionContext,
    ExprNode, IdentificationPerformanceRecord, IdentificationResult, IdentificationStatus,
    IdentifiedEstimand, InferenceMode, Instant, OutcomeExprId, PhysicalExecutionPlan,
    ProvenanceGraph, RefuteSuite, Study, StudyResult, assemble_result, elapsed_ns, nan_effect,
};
use crate::analysis::association::{AssociationPayload, score_predictions};
use crate::analysis::lowering::associational::{lowered_analysis, retained_refusal};
use crate::analysis::lowering::expected_cell;

/// Identifier an association records: it identifies no causal contrast.
pub(crate) const ASSOCIATION_IDENTIFIER: &str = "association.descriptive";
/// Stable code of the disclosure every association result carries.
const DESCRIPTIVE_DISCLOSURE: &str = "estimate.association.descriptive";

impl Study {
    /// The estimator the study's inference mode and sampling design declare.
    pub(crate) fn association_estimator(&self, modality: Modality) -> AssociationEstimator {
        let estimator = AssociationEstimator::new(modality)
            .with_sampling_design(crate::estimator_spec::declared_sampling(self.sampling_design));
        match &self.inference {
            InferenceMode::Frequentist => {
                estimator.with_bootstrap_replicates(self.bootstrap_replicates)
            }
            InferenceMode::Bayesian(config) => estimator.with_bayesian(BayesianSettings {
                draws: config.n_draws,
                prior_scale: config.prior_scale,
            }),
        }
    }

    /// Admission of an association study: a graphless table or series, built-in procedures
    /// only, and a sampling design the data can honour.
    pub(crate) fn ensure_association_admitted(
        &self,
        query: &AssociationQuery,
    ) -> Result<(), CausalError> {
        query.validate()?;
        let modality = match &self.data {
            DataInput::Tabular(_) => Modality::CrossSection,
            DataInput::Temporal(_) | DataInput::Event(_) => Modality::Series,
            DataInput::Panel(_) | DataInput::MultiEnv(_) => {
                return Err(crate::compile_reason!(
                    "data_modality_not_licensed",
                    "an association reads the rows of one table or the points of one series. Next \
                     attempt: pass a table, or one unit's series"
                ));
            }
        };
        if self.graph.class() != crate::accepted::GraphClass::RandomizedTrial
            || self.structure_source != crate::support::StructureSource::RandomizedTrial
            || self.refute != RefuteSuite::None
            || !self.custom_validators.is_empty()
            || self.graph_posterior.is_some()
            || self.tiered.is_some()
            || self.split.is_some()
        {
            return Err(CausalError::Unsupported {
                message: "an association reads the observed columns with no graph, refuter, \
                          split or custom validator",
            });
        }
        if self.identifier.is_some() || self.estimator.is_some() || self.estimator_spec.is_some() {
            return Err(crate::compile_reason!(
                "option_not_applicable",
                "an association picks its analysis by the declared selection rule, so it takes no \
                 identifier or estimator. Next attempt: declare the functional on the query"
            ));
        }
        if let InferenceMode::Bayesian(config) = &self.inference {
            require_built_in_model(config, "the Bayesian association")?;
            if self.bootstrap_replicates != 0 {
                return Err(crate::compile_reason!(
                    "option_not_applicable",
                    "the Bayesian association draws its posterior with n_draws and reads no \
                     bootstrap replicates. Next attempt: omit bootstrap_replicates, or use \
                     frequentist inference for a resampling spread"
                ));
            }
        }
        self.association_estimator(modality).check_design().map_err(CausalError::from)
    }

    /// Execute an association from the retained lowered program of `physical`.
    pub(super) fn execute_association(
        &self,
        storage: &OwnedColumnarStorage,
        modality: Modality,
        physical: &Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let CausalQuery::Association(query) = &self.query else {
            return Err(CausalError::Compile {
                message: "association execution was handed a study that asks no association".into(),
            });
        };
        self.ensure_association_admitted(query)?;
        let program = physical.require_lowered(expected_cell(self))?;
        let estimator = self.association_estimator(modality);
        let prepared = estimator.prepare(storage, query)?;
        let selected = prepared.selection().analysis.as_str();
        if lowered_analysis(program) != Some(selected) {
            return Err(crate::compile_reason!(
                "route_not_lowered",
                "the plan's lowered program fixed the analysis {:?} but the selection rule picks \
                 {selected} on this data; compile the study again",
                lowered_analysis(program)
            ));
        }
        let refused_after = retained_refusal(program);
        let estimate = estimator.estimate(&prepared, ctx)?;
        let scores = estimate
            .prediction
            .as_ref()
            .map(|fit| score_predictions(estimate.selection.outcome_kind, &fit.cases));
        let (identification, estimand) = association_identification(&self.query, query, &estimate);
        let diagnostics = vec![disclosure(&estimate, refused_after.as_deref())];
        let replicates_ok = estimate.resampling.map(|record| record.replicates_ok);
        let draws =
            estimate.posterior.map(|record| u32::try_from(record.draws).unwrap_or(u32::MAX));
        let mut result = assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical.record,
            identification,
            estimand,
            estimate: nan_effect(),
            posterior: None,
            payload: crate::result::ResultPayload::Association(Box::new(AssociationPayload {
                estimate,
                scores,
                refused_after,
            })),
            refutations: Vec::new(),
            diagnostics,
            provenance: ProvenanceGraph::new(),
            treatment: query.outcome,
            outcome: query.outcome,
            wall_time_ns: elapsed_ns(started),
            latency_mode: self.latency_mode.map(|mode| Arc::from(mode.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: (self.bootstrap_replicates > 0
                && !self.inference.is_bayesian())
            .then_some(self.bootstrap_replicates),
            bootstrap_replicates_ok: replicates_ok,
            n_draws: draws,
            cancelled: false,
            early_stopped: false,
            bayesian: self.inference.is_bayesian(),
        });
        result.estimate = crate::PrimaryEstimate::NotAnEffect;
        result.treatment = None;
        crate::analysis::helpers::stamp_result_context(
            &mut result,
            self.support_status,
            self.structure_source,
            self.population_registry.as_ref(),
            &[],
        );
        Ok(result)
    }

    /// Execute on the rows of a table.
    pub(super) fn execute_association_tabular(
        &self,
        data: &antecedent_data::TabularData,
        physical: &Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.execute_association(data.storage(), Modality::CrossSection, physical, ctx)
    }

    /// Execute on the points of one series.
    pub(super) fn execute_association_series(
        &self,
        data: &antecedent_data::TimeSeriesData,
        physical: &Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.execute_association(data.storage(), Modality::Series, physical, ctx)
    }
}

/// The identification record of an association: it claims none.
///
/// The record states the observational functional the association describes and the
/// descriptive premises; its status is `not_identified` because no causal contrast is
/// identified, and the claim the contract derives from it states identification as not claimed.
pub(crate) fn association_identification(
    study_query: &CausalQuery,
    query: &AssociationQuery,
    estimate: &AssociationEstimate,
) -> (IdentificationResult, IdentifiedEstimand) {
    let mut arena = CausalExprArena::new();
    let outcomes = arena.intern_var_set([query.outcome]);
    let given =
        arena.intern_var_set(query.predictors.iter().chain(query.conditioning.iter()).copied());
    let none = arena.intern_intervention_set([]);
    let distribution = arena.intern_distribution(outcomes, given, none, DomainRef::Observational);
    let functional = arena.intern(ExprNode::Expectation {
        function: OutcomeExprId::identity(query.outcome),
        distribution,
    });
    let rule = ASSOCIATION_IDENTIFIER;
    let note = "a descriptive association of the observed law: the outcome given the predictors; \
                no causal contrast is identified and none is claimed";
    arena.set_derivation(functional, DerivationMeta::rule(rule, Some(Arc::from(note))));
    let estimand = IdentifiedEstimand::backdoor(rule, Arc::from([]), functional);
    let mut derivation = DerivationTrace::default();
    derivation.push(rule, note);
    let mut assumptions = AssumptionSet::default();
    for description in descriptive_assumptions(&estimate.selection) {
        let (id, text) = description.split_once(": ").unwrap_or((description, description));
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from(format!("association.{id}")),
                description: Arc::from(text),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
    }
    let identification =
        IdentificationResult::from_parts(antecedent_identify::IdentificationResultParts {
            status: IdentificationStatus::NotIdentified,
            query: study_query.clone(),
            estimands: vec![estimand.clone()],
            arena,
            derivation,
            required_assumptions: assumptions,
            diagnostics: Vec::new(),
            performance: IdentificationPerformanceRecord::default(),
            hedge: None,
        });
    (identification, estimand)
}

fn disclosure(estimate: &AssociationEstimate, refused_after: Option<&str>) -> Diagnostic {
    let follows = refused_after.map_or_else(String::new, |reason| {
        format!("; it follows a causal query refused with `{reason}`, which it does not answer")
    });
    Diagnostic::new(
        DESCRIPTIVE_DISCLOSURE,
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "a noncausal description of the observed law: the {} picked by the {} for a {} \
             outcome on {} ({}) is not the effect of changing any variable and ranks no action{follows}; \
             its standard errors, spreads and posteriors are unlicensed and uncalibrated, and no \
             coverage record names them",
            estimate.selection.analysis.as_str(),
            estimate.selection.rule,
            estimate.selection.outcome_kind.as_str(),
            estimate.selection.modality.as_str(),
            estimate.selection.reason,
        ),
    )
    .with_fields([
        ("analysis", estimate.selection.analysis.as_str().to_owned()),
        ("rows", estimate.rows.to_string()),
        ("rows_dropped", estimate.rows_dropped.to_string()),
        ("calibration", "unmeasured".to_owned()),
    ])
}
