//! The six layers of a measured-interval program family, and what its run returns.
//!
//! Several compiled routes take a fully declared retained source (a count-fitted model, a
//! native cohort fit with its cohort, one named result of an adaptive or a choice study, a
//! temporal unit panel), run the native measured consumer that replays it, and report the
//! interval as a measured scalar annotated with the calibration slot of its construction.
//! Their programs differ only in the cell, the question and source digests, the realization
//! and the claim, so each family states those and this module writes the layers. The run
//! carries the measured inference unchanged, the identity of the program that produced it and
//! the measured route whose coverage records annotate it.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{
    ByteEstimates, LikelihoodModel, OrdinaryBayesianLikelihood, PhysicalEffect, PhysicalOperator,
    Prior, ScoreFrequentist, ScoreModel, SemanticDigest,
};
pub use antecedent_io::measured_inference::MeasuredInference;

use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramCell, ProgramIdentityScope,
    ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};

/// The measured interval a compiled interval family evaluated, bound to its program.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct MeasuredIntervalRun {
    /// The measured inference, exactly as the native measured consumer returned it: every
    /// reported scalar with its point, interval, level and calibration slot.
    pub measured: MeasuredInference,
    /// Identity of the lowered program.
    pub lowered_program_identity: SemanticDigest,
    /// The measured route whose coverage records annotate the interval (the route the direct
    /// entry's measured coverage is recorded under).
    pub calibration_route: &'static str,
}

/// What a measured-interval program states about its question, source and construction.
pub(crate) struct MeasuredIntervalProgram {
    /// The program family.
    pub(crate) cell: ProgramCell,
    /// The route record plan id.
    pub(crate) plan_id: &'static str,
    /// The functional the interval is for.
    pub(crate) kind: FunctionalKind,
    /// The structure class the question was decided on.
    pub(crate) graph_class: &'static str,
    /// Digest of everything the question declares: the construction, the level, the request.
    pub(crate) question: [u8; 32],
    /// Digest of the retained numbers the measured consumer replays.
    pub(crate) source: [u8; 32],
    /// What the source is to the program.
    pub(crate) source_role: EvidenceRole,
    /// The interval's construction.
    pub(crate) realization: Realization,
    /// The law the interval reads.
    pub(crate) uncertainty: UncertaintyReading,
    /// Rows and columns of the retained source.
    pub(crate) rows: u64,
    pub(crate) columns: u64,
    /// What the program claims.
    pub(crate) claim: ClaimKind,
}

/// A frequentist construction with the named score.
pub(crate) fn frequentist(score: &str, plan_id: &str) -> Result<Realization, CausalError> {
    let score = ScoreModel::new(score).map_err(|_| CausalError::Compile {
        message: format!("{plan_id}: the score cannot be declared"),
    })?;
    Ok(Realization::Frequentist(ScoreFrequentist::new(score)))
}

/// A posterior construction under the named likelihood and prior.
pub(crate) fn bayesian(
    likelihood: &str,
    prior: String,
    plan_id: &str,
) -> Result<Realization, CausalError> {
    let invalid = |what: &str| CausalError::Compile {
        message: format!("{plan_id}: the {what} cannot be declared"),
    };
    let likelihood = LikelihoodModel::new(likelihood).map_err(|_| invalid("likelihood"))?;
    let prior = Prior::new(prior).map_err(|_| invalid("prior"))?;
    Ok(Realization::Bayesian(OrdinaryBayesianLikelihood::new(likelihood, prior)))
}

impl MeasuredIntervalProgram {
    /// The lowered program: the question as a shared evidence contract, the measured
    /// construction over the retained source, an interval claim, no decision.
    ///
    /// # Errors
    ///
    /// A physical operator that cannot be declared.
    pub(crate) fn lower(self) -> Result<Arc<LoweredProgram>, CausalError> {
        let design = DesignShape::quoted(self.rows, self.columns, true);
        let scientific = ScientificProblem {
            runtime_query: None,
            identity_scope: ProgramIdentityScope::Semantic,
            target: ProblemTarget::QueryFunctional {
                kind: self.kind,
                outcome: None,
                query: self.question,
            },
            plan_id: Arc::from(self.plan_id),
            data: DataClassification::Tabular,
            graph_class: self.graph_class,
            query_variables: Arc::from([]),
            source_digest: Some(self.source),
            support_coordinate: None,
            supported: false,
            support_status: None,
        };
        let statistical = StatisticalProgram {
            identifier: None,
            estimator: None,
            realization: self.realization,
            models: Arc::from([]),
            folds: None,
            resampling: None,
            posterior: None,
            design,
            operators: Arc::from([]),
        };
        let evidence = EvidenceDecisionProgram {
            sources: Arc::from([EvidenceSourceRef {
                role: self.source_role,
                digest: Some(self.source),
            }]),
            composition: Composition::Single,
            decision: None,
            planning: None,
            chain: None,
        };
        let plan_id = self.plan_id;
        let operator = |effect: PhysicalEffect, bytes: Option<u64>| {
            let declared = PhysicalOperator::declare(effect);
            let declared = match bytes {
                Some(bytes) => declared.estimates(ByteEstimates::absent().with_output(bytes)),
                None => declared,
            };
            declared.build().map_err(|_| CausalError::Compile {
                message: format!("{plan_id}: a physical operator cannot be declared"),
            })
        };
        let input_bytes = self.rows.saturating_mul(self.columns).saturating_mul(8);
        let physical = PhysicalLayer {
            operators: Arc::from([
                operator(PhysicalEffect::Read, Some(input_bytes))?,
                operator(PhysicalEffect::Fit, None)?,
            ]),
            design,
        };
        let identification = IdentificationLayer::NonGraphical(Arc::new(
            NonGraphicalIdentification::SharedEvidenceContract { contract: self.question },
        ));
        Ok(Arc::new(
            StandaloneCompositionLoweredParts {
                cell: self.cell,
                scientific,
                identification,
                statistical,
                evidence,
                physical,
                claim: ClaimDemand { uncertainty: self.uncertainty, kind: Some(self.claim) },
            }
            .finish(),
        ))
    }
}

#[cfg(test)]
mod site_tests {
    use antecedent_contract::ClaimKind;
    use antecedent_core::SemanticDigest;

    use super::{MeasuredIntervalProgram, frequentist};
    use crate::identity_laws::all_distinct;
    use crate::planner::lowered::{EvidenceRole, FunctionalKind, ProgramCell, UncertaintyReading};

    fn identity(question: [u8; 32], source: [u8; 32]) -> SemanticDigest {
        MeasuredIntervalProgram {
            cell: ProgramCell::CohortMeanInterval,
            plan_id: "cohort_mean_interval.v1",
            kind: FunctionalKind::AssignmentContrast,
            graph_class: "as_the_cohort_mean_contrast_route",
            question,
            source,
            source_role: EvidenceRole::Observations,
            realization: frequentist("ols_hc3_treatment_by_basis_student_t", "site")
                .expect("score"),
            uncertainty: UncertaintyReading::Sampling,
            rows: 3,
            columns: 2,
            claim: ClaimKind::Point,
        }
        .lower()
        .expect("a measured interval program lowers")
        .identity()
    }

    #[test]
    fn site_compiled_measured_interval_lower_binds_the_question_and_the_source_in_place() {
        all_distinct(&[
            identity([0; 32], [0; 32]),
            identity([1; 32], [0; 32]),
            identity([0; 32], [1; 32]),
            identity([1; 32], [2; 32]),
            identity([2; 32], [1; 32]),
        ]);
    }
}
