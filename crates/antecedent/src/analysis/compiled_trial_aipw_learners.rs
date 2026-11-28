//! Compiled trial-to-target AIPW transport fitted with caller-supplied learners.
//!
//! [`Study::compile_trial_aipw_learners`] checks the binary trial contrast, decides the
//! transport by the sID identifier on the declared selection diagram (as
//! [`crate::StudyBuilder::learned_trial_transport`] does) and validates the rows and options
//! against the certificate before anything is fitted; it binds the diagram, the query, the
//! options, the declared learners and the rows into a lowered program.
//! [`CompiledTrialAipwLearners::execute`] cross-fits the source outcome with the declared
//! outcome-regression factory and source membership with the declared binary-probability
//! factory through `antecedent_estimate::estimate_trial_aipw_with_factories`, the kernel the
//! direct entry is, under the caller's execution context, and returns the kernel's estimate
//! unchanged.
//!
//! The learners are the study's caller-learner declaration ([`LearnerFactories`], what
//! [`crate::StudyBuilder::learner_factories`] declares): the outcome factory fits the source
//! outcome regression and the treatment factory the source membership probability. They are
//! external code the library does not verify: the estimate records them as unverified
//! (`external_nuisances_unverified`). The kernel refuses bootstrap replicates for caller
//! learners, so the route is point-only and reports calibration `not_measured`; the run carries
//! the declaration's own calibration label, so coverage measured with the library learners
//! (the `learned_trial_aipw` coverage records) never annotates a factory fit.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ClaimKind;
use antecedent_core::{ExecutionContext, SemanticDigest, TransportQuery, VariableId};
use antecedent_estimate::{TrialAipwEstimate, TrialAipwInput, TrialAipwOptions};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::{TransportIdentification, TransportIdentifier};

use super::compiled_point_program::PointProgram;
use super::declared_learners::{LearnerFactories, calibration_label};
use super::program_handle::{CompiledProgramHandle, gate_program};
use super::supplied_identity::{Declaration, SetOf, Wire, encoded};
use crate::analysis::execute::Study;
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell};

const PLAN_ID: &str = "trial_aipw_learner_transport.v1";
const EVALUATOR: &str = "antecedent_estimate::estimate_trial_aipw_with_factories";
/// A trial-to-target AIPW transport request whose nuisances the caller's learners fit.
///
/// `learners` pairs the source outcome factory (`outcome`, a regression) with the source
/// membership factory (`treatment`, a binary probability); its declared identity enters the
/// program.
#[derive(Clone, Debug)]
pub struct TrialAipwLearnersRequest {
    diagram: SelectionDiagram,
    query: TransportQuery,
    input: TrialAipwInput,
    options: TrialAipwOptions,
    learners: LearnerFactories,
}

impl TrialAipwLearnersRequest {
    /// Declare the diagram, query, rows, options and caller learners.
    #[must_use]
    pub fn new(
        diagram: SelectionDiagram,
        query: TransportQuery,
        input: TrialAipwInput,
        options: TrialAipwOptions,
        learners: LearnerFactories,
    ) -> Self {
        Self { diagram, query, input, options, learners }
    }
}

/// A trial AIPW transport with caller learners, decided and bound to its lowered program.
#[derive(Clone, Debug)]
pub struct CompiledTrialAipwLearners {
    request: TrialAipwLearnersRequest,
    identification: TransportIdentification,
    lowered: Arc<LoweredProgram>,
}

/// The contrast the bound request evaluates to, bound to the program that evaluated it.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TrialAipwLearnersRun {
    /// The contrast, scores, out-of-fold nuisances, diagnostics and overlap, as the kernel
    /// returned them.
    pub estimate: TrialAipwEstimate,
    /// Identity of the lowered program.
    pub lowered_program_identity: SemanticDigest,
    /// Always `point_only`: caller learners run no bootstrap.
    pub claim: &'static str,
    /// Calibration state of the route's evidence unit.
    pub calibration: &'static str,
    /// The calibration label of the declared learners (`learners=caller_factory:<digest of the
    /// declared identity>`): a record measured with other learners, the library's among them,
    /// never binds to it.
    pub calibration_label: String,
}

/// Digest of every row the kernel reads, bit for bit, with the declared sampling design.
fn source_digest(input: &TrialAipwInput) -> [u8; 32] {
    Declaration::new("antecedent.trial_aipw_caller_learners.rows.v1")
        .field("rows", &Wire(input))
        .finish()
}

/// Digest of the question: the diagram, the query, the options and the declared learners.
fn question_digest(request: &TrialAipwLearnersRequest) -> Result<[u8; 32], CausalError> {
    let query = antecedent_io::query_wire::transport_query_to_wire(&request.query)?;
    // A graph without a canonical identity refuses the compile (`encoded`).
    encoded(PLAN_ID, || {
        let mut declaration = Declaration::new("antecedent.trial_aipw_caller_learners.question.v1");
        declaration
            .field("causal_graph", request.diagram.causal_graph())
            .field("selection_targets", &SetOf(request.diagram.selection_targets()))
            .field("query", &Wire(&query))
            .field("options", &Wire(&request.options))
            .callback("caller_learner_pair", Some(request.learners.identity()));
        declaration.finish()
    })
}

fn lower(request: &TrialAipwLearnersRequest) -> Result<Arc<LoweredProgram>, CausalError> {
    let question = question_digest(request)?;
    let source = source_digest(&request.input);
    let rows = request.input.source.len() as u64;
    let columns = 4 + request.input.covariates.len() as u64;
    let (treatment, outcome) = contrast_variables(&request.query);
    PointProgram {
        cell: ProgramCell::TrialAipwLearnerTransport,
        plan_id: PLAN_ID,
        kind: FunctionalKind::InterventionalDistribution,
        outcome,
        graph_class: "selection_admg_with_a_trial_transport_certificate",
        query_variables: treatment.into_iter().chain(outcome).collect(),
        question,
        source,
        source_role: EvidenceRole::Trial,
        evaluator: EVALUATOR,
        supplied: vec![("caller_outcome_regression", outcome), ("caller_source_membership", None)],
        rows,
        columns,
        input_bytes: rows.saturating_mul(columns).saturating_mul(8),
        claim: ClaimKind::Point,
    }
    .lower()
}

/// The treatment and outcome of the binary mean contrast `validate_trial_query` admits.
fn contrast_variables(query: &TransportQuery) -> (Option<VariableId>, Option<VariableId>) {
    match &query.response.functional {
        antecedent_core::ResponseFunctional::MeanCurve { outcome, treatment } => {
            (Some(treatment.variable), Some(*outcome))
        }
        _ => (None, None),
    }
}

impl Study {
    /// Decide the transport and bind a trial AIPW request with caller learners.
    ///
    /// # Errors
    ///
    /// Bootstrap replicates (`route_not_supported`,
    /// `trial_aipw_caller_learners.bootstrap_not_supported`: the kernel refuses them for caller
    /// learners, so they are refused here before anything is fitted), a query other than the
    /// binary trial mean contrast, an identification failure, a request
    /// the estimator would refuse before fitting (certificate, features, rows, randomization,
    /// sampling, folds), an unencodable query, cancellation, or a physical operator that cannot
    /// be declared.
    pub fn compile_trial_aipw_learners(
        request: TrialAipwLearnersRequest,
        ctx: &ExecutionContext,
    ) -> Result<CompiledTrialAipwLearners, CausalError> {
        request.learners.admit_portable()?;
        antecedent_estimate::validate_trial_query(&request.query)?;
        if request.options.bootstrap != 0 {
            return Err(crate::unsupported_reason!(
                "route_not_supported",
                detail =
                    antecedent_core::reason_details::trial_aipw_caller_learners::BOOTSTRAP_NOT_SUPPORTED,
                "caller learners run no joint outer bootstrap, so the route is point-only. Next \
                 attempt: declare bootstrap 0"
            ));
        }
        let identification = TransportIdentifier::new().identify_with_context(
            &request.diagram,
            &request.query,
            ctx,
        )?;
        antecedent_estimate::validate_trial_aipw(
            &identification,
            &request.input,
            &request.options,
        )?;
        let lowered = lower(&request)?;
        Ok(CompiledTrialAipwLearners { request, identification, lowered })
    }
}

impl CompiledProgramHandle for CompiledTrialAipwLearners {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledTrialAipwLearners {
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

    /// Cross-fit the caller learners and evaluate the target contrast.
    ///
    /// # Errors
    ///
    /// Bootstrap replicates (`route_not_supported`), a learner without the required task, a
    /// budget, fit or overlap refusal, cancellation, or a support refusal.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<TrialAipwLearnersRun, CausalError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let publication = gate_program(&scoped, self)?;
        let request = &self.request;
        let estimate = antecedent_estimate::estimate_trial_aipw_with_factories(
            &self.identification,
            &request.input,
            &request.options,
            &scoped,
            request.learners.outcome(),
            request.learners.treatment(),
        )?;
        publication.finish(Ok(TrialAipwLearnersRun {
            estimate,
            lowered_program_identity: self.lowered.identity(),
            claim: "point_only",
            calibration: self.unit_calibration(),
            calibration_label: calibration_label(Some(&request.learners)).unwrap_or_default(),
        }))
    }
}

#[cfg(test)]
mod site_tests {
    use std::sync::Arc;

    use antecedent_core::{
        ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery, TransportQuery, VariableId,
    };
    use antecedent_estimate::{
        LearnerSpec, LinearSpec, LogisticSpec, TrialAipwInput, TrialAipwOptions, TrialSampling,
    };
    use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
    use antecedent_learn::{LearnerFactory, PredictionTask, resolve_for};

    use super::{LearnerFactories, TrialAipwLearnersRequest, question_digest, source_digest};
    use crate::identity_laws::{all_distinct, bit_exact, ordered_laws, set_laws};

    fn learners(identity: &str) -> LearnerFactories {
        let outcome: Arc<dyn LearnerFactory> = Arc::from(
            resolve_for(LearnerSpec::Linear(LinearSpec::default()), PredictionTask::Regression)
                .expect("a regression learner"),
        );
        let membership: Arc<dyn LearnerFactory> = Arc::from(
            resolve_for(
                LearnerSpec::Logistic(LogisticSpec::default()),
                PredictionTask::BinaryProbability,
            )
            .expect("a probability learner"),
        );
        LearnerFactories::new(outcome, membership, identity).expect("declared learners")
    }

    fn diagram(selections: &[u32]) -> Option<SelectionDiagram> {
        let mut graph = Admg::with_variables(3);
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).ok()?;
        }
        let selections: Vec<VariableId> =
            selections.iter().copied().map(VariableId::from_raw).collect();
        SelectionDiagram::try_new(graph, selections).ok()
    }

    fn query() -> TransportQuery {
        TransportQuery::new(
            ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: VariableId::from_raw(1),
                treatment: ContinuousDomain::new(
                    VariableId::from_raw(0),
                    GridSpec::Values(Arc::from([0.0, 1.0])),
                ),
            }),
            "trial",
            "target",
            [VariableId::from_raw(0)],
        )
    }

    fn input(outcome: &[f64]) -> TrialAipwInput {
        TrialAipwInput {
            features: vec![],
            covariates: vec![],
            outcome: outcome.to_vec(),
            treatment: outcome.iter().map(|value| *value > 1.0).collect(),
            source: vec![true; outcome.len()],
            randomization: vec![0.5; outcome.len()],
            sampling: TrialSampling::NestedCohort,
        }
    }

    fn question(selections: &[u32], identity: &str, folds: usize) -> Option<[u8; 32]> {
        let options = TrialAipwOptions { folds, ..TrialAipwOptions::default() };
        let request = TrialAipwLearnersRequest::new(
            diagram(selections)?,
            query(),
            input(&[1.0, 2.0]),
            options,
            learners(identity),
        );
        question_digest(&request).ok()
    }

    #[test]
    fn site_compiled_trial_aipw_learners_question_sets_the_selections() {
        set_laws(&[0_u32, 2], |selections| question(selections, "pair", 3));
        all_distinct(&[
            question(&[0, 2], "pair", 3).expect("encoded"),
            question(&[0, 2], "other pair", 3).expect("encoded"),
            question(&[0, 2], "pair", 4).expect("encoded"),
            question(&[0], "pair", 3).expect("encoded"),
        ]);
    }

    #[test]
    fn site_compiled_trial_aipw_learners_source_orders_the_rows() {
        ordered_laws(&[1.0, 2.0, 3.0], |outcome| Some(source_digest(&input(outcome))));
        bit_exact(|value| Some(source_digest(&input(&[value, 2.0]))));
        let independent =
            TrialAipwInput { sampling: TrialSampling::IndependentSamples, ..input(&[1.0, 2.0]) };
        all_distinct(&[source_digest(&input(&[1.0, 2.0])), source_digest(&independent)]);
    }
}
