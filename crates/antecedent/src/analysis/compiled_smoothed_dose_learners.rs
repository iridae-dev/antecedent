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
use antecedent_io::smoothed_dose_artifact::SmoothedDoseQueryWire;
use antecedent_io::{TransportIdentificationWire, transport_identification_to_wire};

use super::compiled_point_program::PointProgram;
use super::declared_learners::LearnerFactories;
use super::program_handle::{CompiledProgramHandle, gate_program};
use super::supplied_identity::{Canonical, SetOf, Wire, encoded};
use crate::analysis::execute::Study;
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell};

const PLAN_ID: &str = "smoothed_dose_caller_learners.v1";
const EVALUATOR: &str = "antecedent_estimate::estimate_smoothed_dose_with_factories";
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
    /// Calibration state of the route's evidence unit.
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

/// The declared question: the selection diagram (the graph's canonical identity and the
/// selection targets as a set), the query and the options in their wire forms (grid doses in
/// declared order, reals exact), the transport certificate's wire form, and the declared
/// identity of the caller's learner factories, which are runtime callbacks.
fn question_digest(
    diagram: &SelectionDiagram,
    query: &SmoothedDoseTransportQuery,
    options: &SmoothedDoseOptions,
    identification: &TransportIdentificationWire,
    learners: &str,
) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.smoothed_dose_caller_learners.question.v2");
    Canonical::encode(diagram.causal_graph(), &mut writer);
    // A set: identification reads the selection targets as a node set.
    Canonical::encode(&SetOf(diagram.selection_targets()), &mut writer);
    Canonical::encode(&Wire(&SmoothedDoseQueryWire::from_query(query)), &mut writer);
    Canonical::encode(&Wire(options), &mut writer);
    Canonical::encode(&Wire(identification), &mut writer);
    writer.note_runtime_callback();
    writer.text("learners").text(learners);
    writer.finish()
}

fn lower(
    request: &SmoothedDoseLearnersRequest,
    identification: &TransportIdentification,
) -> Result<Arc<LoweredProgram>, CausalError> {
    let identification = transport_identification_to_wire(identification);
    // A graph without a canonical identity refuses the compile (`encoded`).
    let question = encoded(PLAN_ID, || {
        question_digest(
            &request.diagram,
            &request.query,
            &request.options,
            &identification,
            request.learners.identity(),
        )
    })?;
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
        request.learners.admit_portable()?;
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
        CompiledProgramHandle::plan_view(self)
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
    /// refusal, cancellation, or a support refusal.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<SmoothedDoseLearnersRun, CausalError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let publication = gate_program(&scoped, self)?;
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
        publication.finish(Ok(SmoothedDoseLearnersRun {
            estimate,
            lowered_program_identity: self.lowered.identity(),
            claim: "point_only",
            calibration: self.unit_calibration(),
        }))
    }
}

#[cfg(test)]
mod identity_tests {
    use antecedent_core::{SmoothingKernel, VariableId};
    use antecedent_graph::{Admg, DenseNodeId};
    use antecedent_io::NotCertifiedCertificateWire;

    use super::*;
    use crate::identity_laws::{all_distinct, bit_exact, ordered_laws, set_laws};

    #[test]
    fn site_compiled_smoothed_dose_learners_question_sets_selections_and_orders_the_grid() {
        // `SelectionDiagram::try_new` refuses a repeated selection target.
        set_laws(&[0_u32, 2], |selections| {
            let mut unique = selections.to_vec();
            unique.sort_unstable();
            unique.dedup();
            (unique.len() == selections.len()).then(|| digest(selections, &[0.5, 1.0], "learners"))
        });
        ordered_laws(&[0.5, 1.0, 1.5], |grid| Some(digest(&[0, 2], grid, "learners")));
        bit_exact(|dose| Some(digest(&[0, 2], &[dose, 1.0], "learners")));
        all_distinct(&[
            digest(&[0, 2], &[0.5, 1.0], "learners"),
            digest(&[0, 2], &[0.5, 1.0], "other"),
            digest(&[0], &[0.5, 1.0], "learners"),
        ]);
    }

    fn rows(outcome: &[f64]) -> SmoothedDoseInput {
        SmoothedDoseInput {
            features: vec![0],
            covariates: vec![vec![0.1; outcome.len()]],
            outcome: outcome.to_vec(),
            dose: vec![0.5; outcome.len()],
            dose_density: vec![1.0; outcome.len()],
            source: vec![true; outcome.len()],
            sampling: TrialSampling::NestedCohort,
            sampling_weights: None,
            cluster: None,
            serial_lag: None,
        }
    }

    #[test]
    fn site_compiled_smoothed_dose_learners_source_orders_the_rows() {
        ordered_laws(&[1.0, 2.0, 3.0], |outcome| Some(source_digest(&rows(outcome))));
        bit_exact(|value| Some(source_digest(&rows(&[value, 2.0]))));
        let base = rows(&[1.0, 2.0]);
        let independent =
            SmoothedDoseInput { sampling: TrialSampling::IndependentSamples, ..base.clone() };
        let weighted = SmoothedDoseInput { sampling_weights: Some(vec![1.0, 2.0]), ..base.clone() };
        let clustered = SmoothedDoseInput { cluster: Some(vec![0, 1]), ..base.clone() };
        let serial = SmoothedDoseInput { serial_lag: Some(1), ..base.clone() };
        let no_features = SmoothedDoseInput { features: Vec::new(), ..base.clone() };
        all_distinct(&[
            source_digest(&base),
            source_digest(&independent),
            source_digest(&weighted),
            source_digest(&clustered),
            source_digest(&serial),
            source_digest(&no_features),
        ]);
    }

    fn diagram(selections: &[u32]) -> SelectionDiagram {
        let mut graph = Admg::with_variables(3);
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let selections = selections.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>();
        SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from(selections)).unwrap()
    }

    fn query(grid: &[f64]) -> SmoothedDoseTransportQuery {
        SmoothedDoseTransportQuery {
            outcome: VariableId::from_raw(2),
            dose: VariableId::from_raw(1),
            source_population: Arc::from("trial"),
            target_population: Arc::from("target"),
            grid: Arc::from(grid),
            bandwidth: 0.5,
            kernel: SmoothingKernel::Epanechnikov,
            dose_support: (0.0, 2.0),
            density_provenance: Arc::from("known"),
        }
    }

    fn digest(selections: &[u32], grid: &[f64], learners: &str) -> [u8; 32] {
        let identification =
            TransportIdentificationWire::NotCertified(NotCertifiedCertificateWire {
                reason: "fixture".into(),
                witness: Vec::new(),
                message: "fixture".into(),
            });
        question_digest(
            &diagram(selections),
            &query(grid),
            &SmoothedDoseOptions::default(),
            &identification,
            learners,
        )
    }

    #[test]
    fn the_smoothed_dose_question_identity_is_canonical() {
        let base = digest(&[0, 2], &[0.5, 1.0], "learners");
        assert_eq!(base, digest(&[0, 2], &[0.5, 1.0], "learners"), "repeatable");
        assert_ne!(base, digest(&[0, 2], &[0.5, 1.0], "other"), "changed learners");
        assert_ne!(base, digest(&[0, 2], &[1.0, 0.5], "learners"), "the grid is ordered");
        assert_eq!(base, digest(&[2, 0], &[0.5, 1.0], "learners"), "selections are a set");
        assert_ne!(
            digest(&[0], &[0.0, 1.0], "learners"),
            digest(&[0], &[-0.0, 1.0], "learners"),
            "reals by bits"
        );
    }
}
