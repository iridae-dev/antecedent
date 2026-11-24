//! The six layers of a point-only program family that evaluates one kernel on retained inputs.
//!
//! Several compiled routes take a fully declared question (a transport certificate and the
//! caller's columns, a prepared scenario set, a measurement model and its premise, a retained
//! effect and a cohort), run one kernel on it and report a point, with no sampling law of
//! their own. Their programs differ only in the cell, the question and source digests, the
//! evaluator and the claim, so each states those and this module writes the layers.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{ByteEstimates, PhysicalEffect, PhysicalOperator, VariableId};

use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramCell, ProgramIdentityScope,
    ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{ModelRole, ModelSlot, Realization, StatisticalProgram};

/// What a point-only program states about its question, inputs and evaluator.
pub(crate) struct PointProgram {
    /// The program family.
    pub(crate) cell: ProgramCell,
    /// The route record plan id.
    pub(crate) plan_id: &'static str,
    /// The functional the program targets.
    pub(crate) kind: FunctionalKind,
    /// The outcome the functional reads, when it names one variable.
    pub(crate) outcome: Option<VariableId>,
    /// The structure class the question was decided on.
    pub(crate) graph_class: &'static str,
    /// The variables the question names.
    pub(crate) query_variables: Arc<[VariableId]>,
    /// Digest of everything the question declares: structure, query, premises, bindings.
    pub(crate) question: [u8; 32],
    /// Digest of the numbers the kernel reads.
    pub(crate) source: [u8; 32],
    /// What the source is to the program.
    pub(crate) source_role: EvidenceRole,
    /// The kernel the program evaluates.
    pub(crate) evaluator: &'static str,
    /// The supplied (never fitted) nuisances the kernel reads, by name, with their columns.
    pub(crate) supplied: Vec<(&'static str, Option<VariableId>)>,
    /// Rows and columns of the evaluated table.
    pub(crate) rows: u64,
    pub(crate) columns: u64,
    /// Bytes the program reads.
    pub(crate) input_bytes: u64,
    /// What the program claims.
    pub(crate) claim: ClaimKind,
}

impl PointProgram {
    /// The lowered program: the question as a shared evidence contract, an exact evaluation
    /// of the kernel, no decision.
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
                outcome: self.outcome,
                query: self.question,
            },
            plan_id: Arc::from(self.plan_id),
            data: DataClassification::Tabular,
            graph_class: self.graph_class,
            query_variables: self.query_variables,
            source_digest: Some(self.source),
            support_coordinate: None,
            supported: false,
            support_status: None,
        };
        let statistical = StatisticalProgram {
            identifier: None,
            estimator: None,
            realization: Realization::ExactPoint { evaluator: Arc::from(self.evaluator) },
            models: self
                .supplied
                .iter()
                .map(|&(family, output)| ModelSlot {
                    role: ModelRole::Nuisance,
                    family: Arc::from(family),
                    inputs: Arc::from([]),
                    output,
                })
                .collect(),
            folds: None,
            resampling: None,
            posterior: None,
            design,
            operators: Arc::from([]),
        };
        let evidence = EvidenceDecisionProgram {
            sources: Arc::from([
                EvidenceSourceRef { role: self.source_role, digest: Some(self.source) },
                EvidenceSourceRef {
                    role: EvidenceRole::TransportSource,
                    digest: Some(self.question),
                },
            ]),
            composition: Composition::Chained { stages: 2 },
            decision: None,
            planning: None,
            chain: None,
        };
        let operator = |effect: PhysicalEffect, bytes: Option<u64>| {
            let declared = PhysicalOperator::declare(effect);
            let declared = match bytes {
                Some(bytes) => declared.estimates(ByteEstimates::absent().with_output(bytes)),
                None => declared,
            };
            declared.build().map_err(|_| CausalError::Compile {
                message: format!("{}: a physical operator cannot be declared", self.plan_id),
            })
        };
        let physical = PhysicalLayer {
            operators: Arc::from([
                operator(PhysicalEffect::Read, Some(self.input_bytes))?,
                operator(PhysicalEffect::Transport, None)?,
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
                claim: ClaimDemand {
                    uncertainty: UncertaintyReading::None,
                    kind: Some(self.claim),
                },
            }
            .finish(),
        ))
    }
}

#[cfg(test)]
mod site_tests {
    use antecedent_contract::ClaimKind;
    use antecedent_core::{SemanticDigest, VariableId};

    use super::PointProgram;
    use crate::identity_laws::{all_distinct, ordered_laws};
    use crate::planner::lowered::{EvidenceRole, FunctionalKind, ProgramCell};

    fn identity(
        question: [u8; 32],
        source: [u8; 32],
        supplied: &[(&'static str, Option<VariableId>)],
    ) -> SemanticDigest {
        PointProgram {
            cell: ProgramCell::TransportScenarioSet,
            plan_id: "transport_scenario_set.v1",
            kind: FunctionalKind::IdentifiedSet,
            outcome: None,
            graph_class: "finite_selection_scenario_set",
            query_variables: std::sync::Arc::from([]),
            question,
            source,
            source_role: EvidenceRole::JointScenarios,
            evaluator: "site_evaluator",
            supplied: supplied.to_vec(),
            rows: 2,
            columns: 2,
            input_bytes: 64,
            claim: ClaimKind::Mixture,
        }
        .lower()
        .expect("a point program lowers")
        .identity()
    }

    #[test]
    fn site_compiled_point_program_lower_binds_the_parts_and_orders_the_supplied_nuisances() {
        all_distinct(&[
            identity([0; 32], [0; 32], &[]),
            identity([1; 32], [0; 32], &[]),
            identity([0; 32], [1; 32], &[]),
            identity([1; 32], [2; 32], &[]),
            identity([2; 32], [1; 32], &[]),
        ]);
        let supplied = [
            ("supplied_grid_regression", Some(VariableId::from_raw(7))),
            ("supplied_grid_regression", Some(VariableId::from_raw(8))),
            ("supplied_response_weight", Some(VariableId::from_raw(9))),
        ];
        ordered_laws(&supplied, |items| Some(identity([0; 32], [0; 32], items)));
    }
}
