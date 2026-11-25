//! Compiled transport of a retained fitted effect to a cohort of another population.
//!
//! [`Study::compile_transported_cohort`] binds the exported analysis result of a native
//! DR-learner or causal-forest fit, the population it was fitted for, the target cohort and the
//! selection-shift declaration into a lowered program.
//! [`CompiledTransportedCohort::execute`] runs the route of
//! [`super::transported_cohort::transport_cohort_effect_from_result`]: it verifies the result,
//! takes its portable conditional-contrast map and issues, through
//! `antecedent_learn::transported_effect::transport_cohort_effect`, the weighted cohort mean of
//! the contrast only when the declaration states that the selection node points only into the
//! declared effect modifiers (exactly the model's features) and the cohort's overlap check
//! against the model's support box passes.
//!
//! The value is point-only under a declared premise and labelled extrapolative; no standard
//! error, interval or law is inherited from the model's fit.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_io::cohort_decision::CohortBinding;
use antecedent_learn::transported_effect::{SelectionShiftDeclaration, TransportRefusal};

use super::compiled_point_program::PointProgram;
use super::program_handle::{CompiledProgramHandle, gate_program};
use super::transported_cohort::{TransportedCohort, transport_cohort_effect_from_result};
use crate::analysis::execute::Study;
use crate::planner::lowered::{EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell};
use crate::planner::{PlanStage, PlanView};

/// Whether `execute` passes the support gate of the unit the route is bound to
/// (`parity/compiled_units.toml`). It stays off until that unit's record has earned its support,
/// so no valid use refuses before then.
const GATE_ON_UNIT_SUPPORT: bool = false;

/// A retained fitted effect, its population, a target cohort and the declared shift.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TransportedCohortRequest {
    /// The exported analysis result that retains the fitted effect.
    pub artifact: Vec<u8>,
    /// The executed claim the result must carry, when the caller retained one.
    pub expected_claim: Option<String>,
    /// The population the model was fitted for (the result carries no label of its own).
    pub source_population: String,
    /// The target cohort.
    pub cohort: CohortBinding,
    /// The selection-shift declaration; `None` declares nothing and is refused naming it.
    pub shift: Option<SelectionShiftDeclaration>,
}

impl TransportedCohortRequest {
    /// A request without an expected claim.
    #[must_use]
    pub fn new(
        artifact: Vec<u8>,
        source_population: impl Into<String>,
        cohort: CohortBinding,
        shift: Option<SelectionShiftDeclaration>,
    ) -> Self {
        Self {
            artifact,
            expected_claim: None,
            source_population: source_population.into(),
            cohort,
            shift,
        }
    }

    /// The same request, requiring the result to carry `claim`.
    #[must_use]
    pub fn expecting_claim(mut self, claim: impl Into<String>) -> Self {
        self.expected_claim = Some(claim.into());
        self
    }
}

/// A bound cohort transport.
#[derive(Clone, Debug)]
pub struct CompiledTransportedCohort {
    request: TransportedCohortRequest,
    lowered: Arc<LoweredProgram>,
}

fn refusal(subject: impl Into<String>) -> TransportRefusal {
    TransportRefusal {
        code: antecedent_core::reason_code!("invalid_argument"),
        detail: antecedent_core::reason_details::transported_effect::INVALID_REQUEST,
        subject: subject.into(),
        remedy: "supply a well-formed request",
    }
}

/// The question: the expected claim, the source population and the shift declaration, every
/// field in declaration order. Both name lists stay ordered: the kernel compares the effect
/// modifiers to the model's sorted features as a list, and the selection children are reported
/// in the value's assumptions in their declared order.
fn question_digest(
    expected_claim: Option<&str>,
    source_population: &str,
    shift: Option<&SelectionShiftDeclaration>,
) -> [u8; 32] {
    let mut question = CanonicalWriter::new("antecedent.transported_cohort.question.v2");
    question.opt_text(expected_claim).text(source_population);
    match shift {
        Some(shift) => {
            question
                .word(1)
                .text(&shift.source_population)
                .text(&shift.target_population)
                .text(&shift.selection_node)
                .texts(&shift.selection_children)
                .texts(&shift.effect_modifiers);
        }
        None => {
            question.word(0);
        }
    }
    question.finish()
}

/// The inputs the kernel reads: the exported result's bytes, then the target cohort's wire
/// encoding (its rows, features, columns and weights in their bound order), each framed.
fn source_digest(artifact: &[u8], cohort: &CohortBinding) -> Result<[u8; 32], TransportRefusal> {
    let mut source = CanonicalWriter::new("antecedent.transported_cohort.inputs.v1");
    source.bytes(artifact);
    source.bytes(
        &antecedent_io::to_cbor(cohort).map_err(|error| refusal(format!("cohort: {error}")))?,
    );
    Ok(source.finish())
}

impl Study {
    /// Bind a cohort transport of a retained fitted effect into a compiled program.
    ///
    /// The premises are checked when the program executes, in the route's order of refusal.
    ///
    /// # Errors
    ///
    /// A program that cannot be lowered (`transported_effect.invalid_request`).
    pub fn compile_transported_cohort(
        request: TransportedCohortRequest,
    ) -> Result<CompiledTransportedCohort, TransportRefusal> {
        let question = question_digest(
            request.expected_claim.as_deref(),
            &request.source_population,
            request.shift.as_ref(),
        );
        let source = source_digest(&request.artifact, &request.cohort)?;
        let lowered = PointProgram {
            cell: ProgramCell::TransportedCohortEffect,
            plan_id: "transported_cohort_effect.v1",
            kind: FunctionalKind::PropagatedChain,
            outcome: None,
            graph_class: "declared_selection_shift",
            query_variables: Arc::from([]),
            question,
            source,
            source_role: EvidenceRole::Observations,
            evaluator: "antecedent_learn::transported_effect::transport_cohort_effect",
            supplied: Vec::new(),
            rows: 1,
            columns: 1,
            input_bytes: request.artifact.len() as u64,
            claim: antecedent_contract::ClaimKind::Point,
        }
        .lower()
        .map_err(|error| refusal(error.to_string()))?;
        Ok(CompiledTransportedCohort { request, lowered })
    }
}

impl CompiledProgramHandle for CompiledTransportedCohort {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledTransportedCohort {
    /// Inspect all six compiler layers.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the plan, including the support decision of its unit, without transporting.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Verify the result, check the premises and issue the transported cohort mean.
    ///
    /// # Errors
    ///
    /// A [`TransportRefusal`] naming the first unmet premise, or the support gate once it is
    /// on.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<TransportedCohort, TransportRefusal> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if GATE_ON_UNIT_SUPPORT {
            gate_program(&scoped, self).map_err(|error| refusal(error.to_string()))?;
        }
        let request = &self.request;
        transport_cohort_effect_from_result(
            &request.artifact,
            request.expected_claim.as_deref(),
            &request.source_population,
            &request.cohort,
            request.shift.as_ref(),
            &scoped,
        )
    }
}

#[cfg(test)]
mod identity_tests {
    use antecedent_learn::transported_effect::SelectionShiftDeclaration;

    use super::{question_digest, source_digest};
    use crate::analysis::cohort_decision::{CohortBinding, CohortColumn};
    use crate::identity_laws::{ADVERSARIAL_TEXT, all_distinct, injective, ordered_laws};

    #[test]
    fn site_compiled_transported_cohort_question_orders_the_name_lists_and_frames_the_texts() {
        ordered_laws(&["age", "sex", "bmi"], |children| {
            Some(digest(&shift(children, &["age"], "S")))
        });
        ordered_laws(&["age", "sex", "bmi"], |modifiers| {
            Some(digest(&shift(&["age"], modifiers, "S")))
        });
        injective(
            &ADVERSARIAL_TEXT,
            |parts| *parts,
            |&(claim, population)| Some(question_digest(Some(claim), population, None)),
        );
    }

    fn cohort(rows: &[(f64, f64)]) -> Option<CohortBinding> {
        CohortBinding::new(
            "snapshot",
            "target",
            (0..rows.len()).map(|i| format!("row-{i}")).collect(),
            vec!["z".to_owned()],
            vec![CohortColumn::Numeric(rows.iter().map(|&(z, _)| z).collect())],
            rows.iter().map(|&(_, weight)| weight).collect(),
        )
        .ok()
    }

    #[test]
    fn site_compiled_transported_cohort_source_frames_the_artifact_and_orders_the_cohort() {
        ordered_laws(&[(-1.0, 0.2), (0.5, 0.3), (1.0, 0.5)], |rows| {
            source_digest(b"artifact", &cohort(rows)?).ok()
        });
        let base = cohort(&[(-1.0, 0.5), (1.0, 0.5)]).expect("a cohort");
        all_distinct(&[
            source_digest(b"artifact", &base).expect("encoded"),
            source_digest(b"artifacT", &base).expect("encoded"),
            source_digest(b"", &base).expect("encoded"),
            source_digest(b"artifact", &cohort(&[(-1.0, 0.25), (1.0, 0.75)]).expect("a cohort"))
                .expect("encoded"),
        ]);
    }

    fn shift(children: &[&str], modifiers: &[&str], node: &str) -> SelectionShiftDeclaration {
        SelectionShiftDeclaration {
            source_population: "source".to_owned(),
            target_population: "target".to_owned(),
            selection_node: node.to_owned(),
            selection_children: children.iter().map(|name| (*name).to_owned()).collect(),
            effect_modifiers: modifiers.iter().map(|name| (*name).to_owned()).collect(),
        }
    }

    fn digest(shift: &SelectionShiftDeclaration) -> [u8; 32] {
        question_digest(Some("claim"), "source", Some(shift))
    }

    #[test]
    fn the_same_declaration_has_the_same_identity() {
        assert_eq!(
            digest(&shift(&["age"], &["age", "sex"], "S")),
            digest(&shift(&["age"], &["age", "sex"], "S"))
        );
    }

    #[test]
    fn a_changed_field_changes_the_identity() {
        let base = digest(&shift(&["age"], &["age", "sex"], "S"));
        assert_ne!(base, digest(&shift(&["age"], &["age", "sex"], "T")));
        assert_ne!(base, digest(&shift(&["sex"], &["age", "sex"], "S")));
        assert_ne!(base, question_digest(Some("claim"), "source", None));
        assert_ne!(
            base,
            question_digest(None, "source", Some(&shift(&["age"], &["age", "sex"], "S")))
        );
        // A name moved from one list to the next is a different declaration.
        assert_ne!(
            digest(&shift(&["age", "sex"], &["bmi"], "S")),
            digest(&shift(&["age"], &["sex", "bmi"], "S"))
        );
    }

    #[test]
    fn reordering_a_name_list_changes_the_identity() {
        assert_ne!(
            digest(&shift(&["age"], &["age", "sex"], "S")),
            digest(&shift(&["age"], &["sex", "age"], "S"))
        );
    }
}
