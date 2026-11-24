//! Compiled smoothed dose-response transport fitted with caller-supplied learners.
//!
//! [`Study::compile_smoothed_dose_learners`] checks the smoothed-dose query, decides the
//! transport by the sID identifier on the declared selection diagram (as
//! [`crate::StudyBuilder::smoothed_dose_transport`] does) and validates the rows and options
//! before anything is fitted; it binds the certificate, the query, the options, the declared
//! learners' identity and the rows into a lowered program. [`CompiledSmoothedDoseLearners::execute`]
//! cross-fits the outcome curve with the declared outcome-regression factory and source
//! membership with the declared binary-probability factory through
//! `antecedent_estimate::estimate_smoothed_dose_with_factories`, the kernel the direct entry
//! is, under the caller's execution context, and returns the kernel's grid unchanged.
//!
//! The learners are external code the library does not verify: the estimate records them as
//! unverified (`external_nuisances_unverified`). The kernel refuses serial (time-ordered) rows
//! and bootstrap replicates for caller learners, so the route is point-only.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ClaimKind;
use antecedent_core::{
    CanonicalWriter, ExecutionContext, SemanticDigest, SmoothedDoseTransportQuery,
};
use antecedent_estimate::{
    SmoothedDoseEstimate, SmoothedDoseInput, SmoothedDoseOptions, TrialSampling,
};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{TransportIdentification, TransportIdentifier};

use super::compiled_point_program::PointProgram;
use super::declared_learners::LearnerFactories;
use super::program_handle::{CompiledProgramHandle, gate_program};
use crate::analysis::execute::Study;
use crate::error::CausalError;
use crate::planner::lowered::{EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell};
use crate::planner::{PlanStage, PlanView};

const PLAN_ID: &str = "smoothed_dose_caller_learners.v1";
const EVALUATOR: &str = "antecedent_estimate::estimate_smoothed_dose_with_factories";
/// Whether `execute` passes the support gate of the unit the route is bound to
/// (`parity/compiled_units.toml`). It stays off until that unit's record has earned its support,
/// so no valid use refuses before then.
const GATE_ON_UNIT_SUPPORT: bool = false;

/// A smoothed dose-response transport request whose nuisances the caller's learners fit.
///
/// `learners` pairs the outcome-curve factory (`outcome`, a regression) with the source
/// membership factory (`treatment`, a binary probability); its identity enters the program.
#[derive(Clone, Debug)]
pub struct SmoothedDoseLearnersRequest {
    diagram: SelectionDiagram,
    query: SmoothedDoseTransportQuery,
    input: SmoothedDoseInput,
    options: SmoothedDoseOptions,
    learners: LearnerFactories,
}

impl SmoothedDoseLearnersRequest {
    /// Declare the diagram, query, rows, options and caller learners.
    #[must_use]
    pub fn new(
        diagram: SelectionDiagram,
        query: SmoothedDoseTransportQuery,
        input: SmoothedDoseInput,
        options: SmoothedDoseOptions,
        learners: LearnerFactories,
    ) -> Self {
        Self { diagram, query, input, options, learners }
    }
}

/// A smoothed-dose transport with caller learners, decided and bound to its lowered program.
#[derive(Clone, Debug)]
pub struct CompiledSmoothedDoseLearners {
    request: SmoothedDoseLearnersRequest,
    identification: TransportIdentification,
    lowered: Arc<LoweredProgram>,
}

/// The grid the bound request evaluates to, bound to the program that evaluated it.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SmoothedDoseLearnersRun {
    /// The grid, fold models, diagnostics and overlap, as the kernel returned them.
    pub estimate: SmoothedDoseEstimate,
    /// Identity of the lowered program.
    pub lowered_program_identity: SemanticDigest,
    /// Always `point_only`: caller learners run no bootstrap.
    pub claim: &'static str,
    /// Always `not_measured`.
    pub calibration: &'static str,
}

fn source_digest(input: &SmoothedDoseInput) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.smoothed_dose_caller_learners.rows.v1");
    writer.u32s(&input.features).word(input.covariates.len() as u64);
    for column in &input.covariates {
        writer.reals(column);
    }
    writer.reals(&input.outcome).reals(&input.dose).reals(&input.dose_density);
    writer.word(input.source.len() as u64);
    for &member in &input.source {
        writer.word(u64::from(member));
    }
    writer.word(match input.sampling {
        TrialSampling::NestedCohort => 0,
        TrialSampling::IndependentSamples => 1,
    });
    match &input.sampling_weights {
        Some(weights) => writer.word(1).reals(weights),
        None => writer.word(0),
    };
    match &input.cluster {
        Some(groups) => writer.word(1).u32s(groups),
        None => writer.word(0),
    };
    writer.opt_word(input.serial_lag.map(u64::from));
    writer.finish()
}

fn lower(
    request: &SmoothedDoseLearnersRequest,
    identification: &TransportIdentification,
) -> Result<Arc<LoweredProgram>, CausalError> {
    let mut question = CanonicalWriter::new("antecedent.smoothed_dose_caller_learners.question.v1");
    question
        .text(&format!("{:?}", request.query))
        .text(&format!("{:?}", request.options))
        .text(&format!("{identification:?}"))
        .text(request.learners.identity());
    let question = question.finish();
    let source = source_digest(&request.input);
    let rows = request.input.source.len() as u64;
    let columns = 4 + request.input.covariates.len() as u64;
    PointProgram {
        cell: ProgramCell::SmoothedDoseLearnerTransport,
        plan_id: PLAN_ID,
        kind: FunctionalKind::InterventionalDistribution,
        outcome: Some(request.query.outcome),
        graph_class: "selection_admg_with_a_smoothed_dose_transport_certificate",
        query_variables: Arc::from([request.query.dose, request.query.outcome]),
        question,
        source,
        source_role: EvidenceRole::Trial,
        evaluator: EVALUATOR,
        supplied: vec![
            ("caller_outcome_regression", Some(request.query.outcome)),
            ("caller_source_membership", None),
        ],
        rows,
        columns,
        input_bytes: rows.saturating_mul(columns).saturating_mul(8),
        claim: ClaimKind::Response,
    }
    .lower()
}

impl Study {
    /// Decide the transport and bind a smoothed-dose request with caller learners.
    ///
    /// # Errors
    ///
    /// A malformed query, an identification failure, a request the estimator would refuse
    /// before fitting (bounds, density, dose support, local support, rows), cancellation, or a
    /// physical operator that cannot be declared.
    pub fn compile_smoothed_dose_learners(
        request: SmoothedDoseLearnersRequest,
        ctx: &ExecutionContext,
    ) -> Result<CompiledSmoothedDoseLearners, CausalError> {
        request.query.validate()?;
        let identification = TransportIdentifier::new().identify_with_context(
            &request.diagram,
            &request.query.transport_query(),
            ctx,
        )?;
        antecedent_estimate::validate_smoothed_dose(
            &identification,
            &request.query,
            &request.input,
            &request.options,
        )?;
        let lowered = lower(&request, &identification)?;
        Ok(CompiledSmoothedDoseLearners { request, identification, lowered })
    }
}

impl CompiledProgramHandle for CompiledSmoothedDoseLearners {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledSmoothedDoseLearners {
    /// Inspect all six compiler layers.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the plan, including the support decision of its unit, without evaluating.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// The transport certificate the program was decided on.
    #[must_use]
    pub const fn identification(&self) -> &TransportIdentification {
        &self.identification
    }

    /// Cross-fit the caller learners and evaluate the grid.
    ///
    /// # Errors
    ///
    /// Time-ordered rows or bootstrap replicates (`route_not_supported`), a learner without the
    /// required task or sampling-weight capability, a portability, budget, fit or overlap
    /// refusal, cancellation, or the support gate once it is on.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<SmoothedDoseLearnersRun, CausalError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if GATE_ON_UNIT_SUPPORT {
            gate_program(&scoped, self)?;
        }
        let request = &self.request;
        let estimate = antecedent_estimate::estimate_smoothed_dose_with_factories(
            &self.identification,
            &request.query,
            &request.input,
            &request.options,
            &scoped,
            request.learners.outcome(),
            request.learners.treatment(),
        )?;
        Ok(SmoothedDoseLearnersRun {
            estimate,
            lowered_program_identity: self.lowered.identity(),
            claim: "point_only",
            calibration: "not_measured",
        })
    }
}
