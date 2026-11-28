//! Compiled likelihood fit of a feedback system with registered external mechanisms.
//!
//! [`Study::compile_external_feedback_fit`] binds an external-capable feedback system (its
//! built-in equations and the recorded identity of each registered mechanism: implementation
//! id, version, content digest, dependencies and declarations), the scalars to fit, the
//! known-regime rows and the bounded gradient-ascent policy into a lowered program: the
//! system, targets and policy are the program's question, the rows its source.
//! [`CompiledExternalFeedbackFit::execute`] fits through
//! `antecedent_estimate::external_cyclic_fit::fit_external_feedback`, the kernel the direct
//! entry is, under the caller's execution context. Like replay, fitting runs a mechanism only
//! after the caller's trusted resolver has supplied an implementation that the kernel verifies
//! against the record; the resolver is a trust declaration of the execution, not part of the
//! question, so it is passed to `execute` and never enters the program identity.
//!
//! The fit is model-conditional (graph, covariance and mechanism family are assumptions) and
//! local: convergence is not a global optimum, and no uncertainty is formed.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ClaimKind;
use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest};
use antecedent_estimate::external_cyclic_fit::{
    ExternalFeedbackFit, ExternalFitPolicy, fit_external_feedback,
};
use antecedent_model::external_mechanism::{
    ExternalFeedbackSystem, ExternalFitParameter, TrustedMechanismResolver,
};
use antecedent_model::nonlinear_equilibrium::{EquilibriumError, RegimeRow};

use super::compiled_point_program::PointProgram;
use super::program_handle::{CompiledProgramHandle, gate_program};
use crate::analysis::execute::Study;
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell};

const PLAN_ID: &str = "external_feedback_fit.v1";
const EVALUATOR: &str = "antecedent_estimate::external_cyclic_fit::fit_external_feedback";
/// The system, fitted scalars, known-regime rows and policy of one external feedback fit.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalFeedbackFitRequest {
    system: ExternalFeedbackSystem,
    parameters: Vec<ExternalFitParameter>,
    rows: Vec<RegimeRow>,
    policy: ExternalFitPolicy,
}

impl ExternalFeedbackFitRequest {
    /// Declare the template system, the scalars to fit, the rows and the policy (the arguments
    /// `fit_external_feedback` takes besides the resolver).
    #[must_use]
    pub fn new(
        system: ExternalFeedbackSystem,
        parameters: Vec<ExternalFitParameter>,
        rows: Vec<RegimeRow>,
        policy: ExternalFitPolicy,
    ) -> Self {
        Self { system, parameters, rows, policy }
    }

    /// The template system.
    #[must_use]
    pub const fn system(&self) -> &ExternalFeedbackSystem {
        &self.system
    }
}

/// Failure to compile or execute an external feedback fit.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ExternalFeedbackFitError {
    /// The kernel refused or failed: invalid policy or rows, a missing, unavailable, changed or
    /// failing resolver, a mechanism callback failure, a non-finite score, or cancellation.
    #[error(transparent)]
    Model(#[from] EquilibriumError),
    /// The program could not be lowered, or the support gate refused it.
    #[error(transparent)]
    Compile(#[from] CausalError),
}

/// An external feedback fit request bound to its lowered program.
#[derive(Clone, Debug)]
pub struct CompiledExternalFeedbackFit {
    request: ExternalFeedbackFitRequest,
    lowered: Arc<LoweredProgram>,
}

/// The fit the bound request produced, bound to the program that produced it.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ExternalFeedbackFitRun {
    /// The fitted system and its termination record, as the kernel returned them.
    pub fit: ExternalFeedbackFit,
    /// Identity of the lowered program.
    pub lowered_program_identity: SemanticDigest,
    /// Always `model_conditional_point`: no uncertainty is formed.
    pub claim: &'static str,
}

/// The system and the fitted scalars enter as their CBOR wire encodings (serde types of
/// ordered lists and scalars with no hash-ordered collections; reals in the shortest exact
/// width, so injective on their bits, unlike JSON, which writes every NaN as `null`). The
/// fitted scalars stay in declaration order: the kernel's parameter vector follows it.
fn question_digest(request: &ExternalFeedbackFitRequest) -> Result<[u8; 32], CausalError> {
    let encode = |what: &str, value: Result<Vec<u8>, antecedent_io::IoError>| {
        value.map_err(|error| CausalError::Compile {
            message: format!("{PLAN_ID}: the {what} cannot be encoded: {error}"),
        })
    };
    let mut writer = CanonicalWriter::new("antecedent.external_feedback_fit.question.v2");
    writer.bytes(&encode("system", antecedent_io::to_cbor(&request.system))?);
    writer.bytes(&encode("fitted parameters", antecedent_io::to_cbor(&request.parameters))?);
    let policy = request.policy;
    writer
        .real(policy.learning_rate)
        .real(policy.score_tolerance)
        .word(policy.max_iterations as u64)
        .real(policy.difference_step);
    Ok(writer.finish())
}

/// The known-regime rows in declared order (the likelihood sums them in that order), each row's
/// values in dense order and its clamps as a set: the solve replaces each clamped mechanism by
/// its value, so the clamps are written sorted by variable (a repeated variable is refused by
/// the kernel when the program runs).
fn source_digest(rows: &[RegimeRow]) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.external_feedback_fit.rows.v2");
    writer.word(rows.len() as u64);
    for row in rows {
        writer.reals(&row.values).word(row.clamps.len() as u64);
        let mut clamps: Vec<_> = row.clamps.iter().collect();
        clamps.sort_by_key(|clamp| clamp.variable);
        for clamp in clamps {
            writer.word(clamp.variable as u64).real(clamp.value);
        }
    }
    writer.finish()
}

fn lower(request: &ExternalFeedbackFitRequest) -> Result<Arc<LoweredProgram>, CausalError> {
    let question = question_digest(request)?;
    let source = source_digest(&request.rows);
    let rows = request.rows.len() as u64;
    let columns = request.system.equations.len() as u64;
    PointProgram {
        cell: ProgramCell::ExternalFeedbackFit,
        plan_id: PLAN_ID,
        kind: FunctionalKind::InterventionalDistribution,
        outcome: None,
        graph_class: "specified_feedback_system_with_registered_external_mechanisms",
        query_variables: Arc::from([]),
        question,
        source,
        source_role: EvidenceRole::Observations,
        evaluator: EVALUATOR,
        supplied: vec![("registered_external_mechanism", None)],
        rows,
        columns,
        input_bytes: rows.saturating_mul(columns).saturating_mul(8),
        claim: ClaimKind::Point,
    }
    .lower()
}

impl Study {
    /// Bind an external feedback fit request into a compiled program.
    ///
    /// # Errors
    ///
    /// A system or parameter list that cannot be encoded, or a physical operator that cannot
    /// be declared.
    pub fn compile_external_feedback_fit(
        request: ExternalFeedbackFitRequest,
    ) -> Result<CompiledExternalFeedbackFit, ExternalFeedbackFitError> {
        let lowered = lower(&request)?;
        Ok(CompiledExternalFeedbackFit { request, lowered })
    }
}

impl CompiledProgramHandle for CompiledExternalFeedbackFit {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledExternalFeedbackFit {
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

    /// The bound request.
    #[must_use]
    pub const fn request(&self) -> &ExternalFeedbackFitRequest {
        &self.request
    }

    /// Fit the declared scalars with the caller's trusted resolver.
    ///
    /// # Errors
    ///
    /// [`ExternalFeedbackFitError::Model`] for every refusal of the kernel, including fitting
    /// without a resolver (`mechanism_resolver_missing`) and cancellation;
    /// [`ExternalFeedbackFitError::Compile`] for a support refusal.
    pub fn execute(
        &self,
        resolver: Option<&dyn TrustedMechanismResolver>,
        ctx: &ExecutionContext,
    ) -> Result<ExternalFeedbackFitRun, ExternalFeedbackFitError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let publication = gate_program(&scoped, self).map_err(CausalError::from)?;
        let request = &self.request;
        let fit = fit_external_feedback(
            &request.system,
            resolver,
            &request.parameters,
            &request.rows,
            request.policy,
            &scoped,
        )?;
        publication.finish(Ok(ExternalFeedbackFitRun {
            fit,
            lowered_program_identity: self.lowered.identity(),
            claim: "model_conditional_point",
        }))
    }
}

#[cfg(test)]
mod identity_tests {
    use antecedent_estimate::external_cyclic_fit::ExternalFitPolicy;
    use antecedent_model::external_mechanism::{ExternalFeedbackSystem, ExternalFitParameter};

    use antecedent_model::nonlinear_equilibrium::{Clamp, RegimeRow};

    use super::{ExternalFeedbackFitRequest, question_digest, source_digest};
    use crate::identity_laws::{all_distinct, bit_exact, ordered_laws, set_laws};

    #[test]
    fn site_compiled_external_feedback_question_orders_the_fitted_scalars() {
        ordered_laws(&[0_usize, 1, 2], |equations| {
            question_digest(&request(vec![1.0], intercepts(equations), 0.1)).ok()
        });
        bit_exact(|rate| question_digest(&request(vec![1.0], intercepts(&[0]), rate)).ok());
        bit_exact(|noise| question_digest(&request(vec![noise], intercepts(&[0]), 0.1)).ok());
    }

    fn row(values: &[f64], clamps: &[(usize, f64)]) -> RegimeRow {
        RegimeRow {
            values: values.to_vec(),
            clamps: clamps.iter().map(|&(variable, value)| Clamp { variable, value }).collect(),
        }
    }

    #[test]
    fn site_compiled_external_feedback_source_orders_rows_and_sets_clamps() {
        let rows = [row(&[1.0, 2.0], &[]), row(&[3.0, 4.0], &[(1, 4.0)]), row(&[5.0, 6.0], &[])];
        ordered_laws(&rows, |ordered| Some(source_digest(ordered)));
        // The kernel refuses a row that clamps one variable twice.
        set_laws(&[(0_usize, 1.0), (1, 2.0)], |clamps| {
            let mut variables: Vec<usize> = clamps.iter().map(|clamp| clamp.0).collect();
            variables.sort_unstable();
            variables.dedup();
            (variables.len() == clamps.len()).then(|| source_digest(&[row(&[1.0, 2.0], clamps)]))
        });
        bit_exact(|value| Some(source_digest(&[row(&[1.0, 2.0], &[(0, value)])])));
        all_distinct(&[
            source_digest(&[]),
            source_digest(&[row(&[1.0, 2.0], &[])]),
            source_digest(&[row(&[1.0, 2.0], &[(0, 1.0)])]),
            source_digest(&[row(&[1.0, 2.0], &[(1, 1.0)])]),
        ]);
    }

    fn request(
        noise_covariance: Vec<f64>,
        parameters: Vec<ExternalFitParameter>,
        learning_rate: f64,
    ) -> ExternalFeedbackFitRequest {
        ExternalFeedbackFitRequest::new(
            ExternalFeedbackSystem { equations: Vec::new(), noise_covariance },
            parameters,
            Vec::new(),
            ExternalFitPolicy { learning_rate, ..ExternalFitPolicy::default() },
        )
    }

    fn intercepts(equations: &[usize]) -> Vec<ExternalFitParameter> {
        equations.iter().map(|&equation| ExternalFitParameter::Intercept { equation }).collect()
    }

    #[test]
    fn the_same_request_has_the_same_identity() {
        assert_eq!(
            question_digest(&request(vec![1.0], intercepts(&[0, 1]), 0.1)).unwrap(),
            question_digest(&request(vec![1.0], intercepts(&[0, 1]), 0.1)).unwrap()
        );
    }

    #[test]
    fn a_changed_field_changes_the_identity() {
        let base = question_digest(&request(vec![1.0], intercepts(&[0, 1]), 0.1)).unwrap();
        assert_ne!(base, question_digest(&request(vec![2.0], intercepts(&[0, 1]), 0.1)).unwrap());
        assert_ne!(base, question_digest(&request(vec![1.0], intercepts(&[0]), 0.1)).unwrap());
        assert_ne!(base, question_digest(&request(vec![1.0], intercepts(&[0, 1]), 0.2)).unwrap());
    }

    #[test]
    fn reordering_the_fitted_scalars_changes_the_identity() {
        assert_ne!(
            question_digest(&request(vec![1.0], intercepts(&[0, 1]), 0.1)).unwrap(),
            question_digest(&request(vec![1.0], intercepts(&[1, 0]), 0.1)).unwrap()
        );
    }

    #[test]
    fn negative_zero_and_nan_payloads_are_distinguished() {
        assert_ne!(
            question_digest(&request(vec![0.0], intercepts(&[0]), 0.1)).unwrap(),
            question_digest(&request(vec![-0.0], intercepts(&[0]), 0.1)).unwrap()
        );
        assert_ne!(
            question_digest(&request(vec![f64::NAN], intercepts(&[0]), 0.1)).unwrap(),
            question_digest(&request(vec![f64::INFINITY], intercepts(&[0]), 0.1)).unwrap()
        );
    }
}
