//! Small shared pieces of the checked operations: the estimand-claim
//! comparisons every sealed route repeats, the re-identification guard, the
//! sealed-procedure record, and the validation-suite and study-admission
//! predicates several `admits` functions spell out.
//!
//! Each helper restates a fragment that routes previously inlined; none
//! widens or narrows what any route admits.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    Diagnostic, DiagnosticKind, DiagnosticSeverity, ResponseFunctional, VariableId,
};
use antecedent_expr::IdentifiedEstimand;
use antecedent_identify::result::IdentificationResult;

use super::builder::RefuteSuite;
use super::execute::Study;
use crate::error::CausalError;
use crate::strategy_table::{EstimatorId, IdentifierId};

/// The physical proof boundary and the caller funding any unsealed proof replay.
/// The context is used only while constructing an operation; it is never retained with it.
pub(crate) struct CheckedCompileContext<'a> {
    pub(crate) physical: std::sync::Arc<crate::planner::PhysicalExecutionPlan>,
    pub(crate) ctx: &'a antecedent_core::ExecutionContext,
}

/// Comparison of what two identified estimands claim.
///
/// `functional` is an [`ExprId`](antecedent_expr::ExprId) into the arena of
/// the result that produced the estimand, so comparing two estimands from
/// different results is meaningful only because the identifiers build their
/// arenas by deterministic hash-consing; the test below pins that property.
pub(crate) trait EstimandClaim {
    /// Same method, adjustment set, and functional.
    fn same_claim(&self, other: &IdentifiedEstimand) -> bool;

    /// [`Self::same_claim`] and the same mediators.
    fn same_claim_with_mediators(&self, other: &IdentifiedEstimand) -> bool;

    /// Same method and functional only (no adjustment set).
    fn same_method_and_functional(&self, other: &IdentifiedEstimand) -> bool;

    /// Every field of the estimand agrees.
    fn same_full_claim(&self, other: &IdentifiedEstimand) -> bool;
}

impl EstimandClaim for IdentifiedEstimand {
    fn same_claim(&self, other: &IdentifiedEstimand) -> bool {
        self.functional == other.functional
            && self.method == other.method
            && self.adjustment_set == other.adjustment_set
    }

    fn same_claim_with_mediators(&self, other: &IdentifiedEstimand) -> bool {
        self.same_claim(other) && self.mediators == other.mediators
    }

    fn same_method_and_functional(&self, other: &IdentifiedEstimand) -> bool {
        self.method == other.method && self.functional == other.functional
    }

    fn same_full_claim(&self, other: &IdentifiedEstimand) -> bool {
        self.same_claim_with_mediators(other)
            && self.instruments == other.instruments
            && self.rd_design == other.rd_design
    }
}

/// Whether `identification` supplies `estimand` (same method, adjustment set,
/// and functional) as one of its estimands.
#[must_use]
pub(crate) fn proof_supplies(
    identification: &IdentificationResult,
    estimand: &IdentifiedEstimand,
) -> bool {
    identification.estimands.iter().any(|candidate| candidate.same_claim(estimand))
}

/// [`proof_supplies`] that also requires the mediators to agree.
#[must_use]
pub(crate) fn proof_supplies_with_mediators(
    identification: &IdentificationResult,
    estimand: &IdentifiedEstimand,
) -> bool {
    identification.estimands.iter().any(|candidate| candidate.same_claim_with_mediators(estimand))
}

/// [`proof_supplies`] that compares only the method and the functional.
#[must_use]
pub(crate) fn proof_supplies_method_and_functional(
    identification: &IdentificationResult,
    estimand: &IdentifiedEstimand,
) -> bool {
    identification.estimands.iter().any(|candidate| candidate.same_method_and_functional(estimand))
}

/// [`proof_supplies`] that requires every estimand field to agree.
#[must_use]
pub(crate) fn proof_supplies_full(
    identification: &IdentificationResult,
    estimand: &IdentifiedEstimand,
) -> bool {
    identification.estimands.iter().any(|candidate| candidate.same_full_claim(estimand))
}

/// Refuse unless the claim re-derived from the retained graph agrees with the
/// retained one on status, query, and the selected estimand.
///
/// # Errors
///
/// [`CausalError::Compile`] carrying `message` when any of them differ.
pub(crate) fn verify_proof_against_graph(
    checked: &IdentificationResult,
    checked_estimand: &IdentifiedEstimand,
    identification: &IdentificationResult,
    estimand: &IdentifiedEstimand,
    message: &str,
) -> Result<(), CausalError> {
    if checked.status != identification.status
        || checked.query != identification.query
        || !checked_estimand.same_claim(estimand)
    {
        return Err(CausalError::Compile { message: message.into() });
    }
    Ok(())
}

/// What checking a proof against the compile-time seal found.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RetainedProof {
    /// The product equals the full identification result compile sealed: identification
    /// ran once, at compile.
    Sealed,
    /// Compile retained no product for this plan, so there is no seal to check against.
    NotRetained,
}

/// Verify the static identification an operation proves from against the digest compile
/// sealed over the full identification result.
///
/// The seal covers status, estimands, arena, derivation rules, required assumptions, hedge,
/// query and the selected estimand, so this is a stronger check than comparing status, query
/// and estimand, and it runs no identification.
///
/// # Errors
///
/// [`CausalError::Compile`] carrying `message` when the product differs from the seal.
pub(crate) fn verify_retained_static(
    plan: &crate::planner::PhysicalExecutionPlan,
    identification: &IdentificationResult,
    estimand: &IdentifiedEstimand,
    message: &str,
) -> Result<RetainedProof, CausalError> {
    let sealed = plan
        .lowered()
        .and_then(crate::planner::lowered::LoweredProgram::retained_identification)
        .and_then(crate::planner::lowered::RetainedIdentification::static_seal);
    let Some(sealed) = sealed else {
        return Ok(RetainedProof::NotRetained);
    };
    if super::identification_seal::seal_static(identification, estimand)? != sealed {
        return Err(CausalError::Compile { message: message.into() });
    }
    Ok(RetainedProof::Sealed)
}

/// [`verify_retained_static`] for the temporal identification of every requested horizon.
///
/// # Errors
///
/// [`CausalError::Compile`] carrying `message` when the product differs from the seal.
pub(crate) fn verify_retained_temporal(
    plan: &crate::planner::PhysicalExecutionPlan,
    cache: &super::prepared::CachedTemporalIdentification,
    message: &str,
) -> Result<RetainedProof, CausalError> {
    let sealed = plan
        .lowered()
        .and_then(crate::planner::lowered::LoweredProgram::retained_identification)
        .and_then(crate::planner::lowered::RetainedIdentification::temporal_seal);
    let Some(sealed) = sealed else {
        return Ok(RetainedProof::NotRetained);
    };
    if super::identification_seal::seal_temporal(cache)? != sealed {
        return Err(CausalError::Compile { message: message.into() });
    }
    Ok(RetainedProof::Sealed)
}

/// The note a graph-posterior route attaches when it withholds the aggregate
/// posterior because a graph- or completion-conditional posterior does not
/// describe the full weighted target.
#[must_use]
pub(crate) fn graph_posterior_withheld_diagnostic() -> Diagnostic {
    Diagnostic::new(
        "estimate.graph_posterior.posterior_withheld",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Warning,
        "aggregate posterior withheld: a graph- or completion-conditional posterior \
         does not describe the full weighted target; retained atom posteriors are conditional",
    )
}

/// The treatment/outcome pair a response functional is about.
///
/// # Errors
///
/// [`CausalError::Compile`] when the functional names no such pair.
pub(crate) fn require_primary_pair(
    functional: &ResponseFunctional,
) -> Result<(VariableId, VariableId), CausalError> {
    functional.primary_pair().ok_or_else(|| CausalError::Compile {
        message: "response query has no treatment/outcome pair".into(),
    })
}

/// The identifier, estimator, and execution settings a sealed temporal
/// operation fixed at preparation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Procedure {
    pub(crate) identifier: IdentifierId,
    pub(crate) estimator: EstimatorId,
    pub(crate) bootstrap_replicates: u32,
    pub(crate) validation: RefuteSuite,
}

impl RefuteSuite {
    /// One of the built-in point suites (`none`, `cheap`, or `full`).
    #[must_use]
    pub(crate) const fn is_point(self) -> bool {
        matches!(self, Self::None | Self::Cheap | Self::Full)
    }

    /// [`Self::is_point`] widened to the placebo-and-RCC suite.
    #[must_use]
    pub(crate) const fn is_point_or_placebo(self) -> bool {
        matches!(self, Self::None | Self::Cheap | Self::PlaceboAndRcc | Self::Full)
    }
}

impl Study {
    /// A tabular-or-series study on a fixed structure with no graph posterior,
    /// tier closure, or caller validators: the plain coordinate the sealed
    /// class routes admit before looking at their own query.
    #[must_use]
    pub(crate) fn plain_fixed_route(&self) -> bool {
        self.graph_posterior.is_none()
            && self.tiered.is_none()
            && self.custom_validators.is_empty()
            && self.fixed_structure()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{AverageEffectQuery, VariableId};
    use antecedent_graph::{Dag, DenseNodeId};

    fn confounded() -> Dag {
        let mut dag = Dag::with_variables(3);
        for (from, to) in [(2, 0), (2, 1), (0, 1)] {
            dag.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        dag
    }

    fn compiled_static() -> (
        crate::planner::PhysicalExecutionPlan,
        std::sync::Arc<crate::analysis::prepared::CachedStaticIdentification>,
    ) {
        let rows = 80_usize;
        let noise = |i: usize| ((i as f64 * 12.9898).sin() * 43758.5453).fract().abs() - 0.5;
        let z: Vec<f64> = (0..rows).map(|i| noise(i + 17) * 4.0).collect();
        let t: Vec<f64> = (0..rows).map(|i| f64::from(z[i] + 2.0 * noise(i + 31) > 0.0)).collect();
        let y: Vec<f64> =
            (0..rows).map(|i| 1.0 + 2.0 * t[i] + 0.5 * z[i] + 0.4 * noise(i + 99)).collect();
        let data = antecedent_data::TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("z", z.as_slice()),
        ])
        .unwrap();
        let study = crate::Study::tabular(data)
            .graph(confounded())
            .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let plan = study.compile(&antecedent_core::ExecutionContext::for_tests(1)).unwrap();
        let cache = plan
            .lowered()
            .unwrap()
            .retained_identification()
            .unwrap()
            .caches()
            .iter()
            .find_map(|cache| match cache {
                crate::analysis::IdentificationCache::Static(cache) => {
                    Some(std::sync::Arc::clone(cache))
                }
                _ => None,
            })
            .expect("compile retained the static identification");
        (plan, cache)
    }

    #[test]
    fn the_retained_identification_verifies_against_its_compile_seal() {
        let (plan, cache) = compiled_static();
        let sealed = verify_retained_static(
            &plan,
            &cache.identification,
            &cache.estimand,
            "the retained identification differs from its seal",
        )
        .unwrap();
        assert_eq!(sealed, RetainedProof::Sealed, "no identification ran to verify it");
    }

    #[test]
    fn a_tampered_retained_identification_is_refused() {
        let (plan, cache) = compiled_static();
        let message = "the retained identification differs from its seal";
        // A different selected estimand.
        let mut estimand = cache.estimand.clone();
        estimand.adjustment_set = std::sync::Arc::from([]);
        let refused =
            verify_retained_static(&plan, &cache.identification, &estimand, message).unwrap_err();
        assert!(refused.to_string().contains(message), "{refused}");
        // A different status, with the estimand untouched.
        let mut identification = cache.identification.clone();
        identification.status = antecedent_core::IdentificationStatus::PartiallyIdentified;
        assert!(verify_retained_static(&plan, &identification, &cache.estimand, message).is_err());
        // A different query.
        let mut identification = cache.identification.clone();
        identification.query = antecedent_core::CausalQuery::AverageEffect(
            AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(0)),
        );
        assert!(verify_retained_static(&plan, &identification, &cache.estimand, message).is_err());
        // A different arena: the claim is the same names over another functional.
        let mut identification = cache.identification.clone();
        identification.arena = antecedent_expr::CausalExprArena::new();
        assert!(verify_retained_static(&plan, &identification, &cache.estimand, message).is_err());
    }

    #[test]
    fn independent_identifications_hash_cons_to_the_same_functional() {
        let graph = confounded();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let first = crate::strategy_table::identify_static(
            IdentifierId::BackdoorAdjustment,
            &graph,
            &query,
        )
        .unwrap();
        let second = crate::strategy_table::identify_static(
            IdentifierId::BackdoorAdjustment,
            &graph,
            &query,
        )
        .unwrap();
        assert!(!first.estimands.is_empty());
        assert_eq!(first.estimands.len(), second.estimands.len());
        for (a, b) in first.estimands.iter().zip(second.estimands.iter()) {
            assert!(a.same_full_claim(b), "ExprIds from separate arenas must agree");
        }
        assert!(proof_supplies(&second, &first.estimands[0]));
        assert!(proof_supplies_full(&second, &first.estimands[0]));
        assert!(
            verify_proof_against_graph(
                &first,
                &first.estimands[0],
                &second,
                &second.estimands[0],
                "unused",
            )
            .is_ok()
        );
    }

    #[test]
    fn a_different_adjustment_set_is_a_different_claim() {
        let graph = confounded();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let identified = crate::strategy_table::identify_static(
            IdentifierId::BackdoorAdjustment,
            &graph,
            &query,
        )
        .unwrap();
        let mut other = identified.estimands[0].clone();
        other.adjustment_set = std::sync::Arc::from([]);
        assert!(!identified.estimands[0].same_claim(&other));
        assert!(!proof_supplies(&identified, &other));
    }

    #[test]
    fn suite_predicates_match_the_point_suites() {
        assert!(RefuteSuite::Cheap.is_point());
        assert!(!RefuteSuite::PlaceboAndRcc.is_point());
        assert!(RefuteSuite::PlaceboAndRcc.is_point_or_placebo());
    }
}
