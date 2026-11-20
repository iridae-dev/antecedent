//! Lower a design-native joint functional: a compiled joint design and its retained design
//! proof, independent of the nonlinear panel fitter.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{
    ByteEstimates, CanonicalWriter, EvidenceClaims, Operator, OperatorDeclaration, OperatorEffect,
    OperatorGraph, OperatorPorts, OperatorRole, PhysicalEffect, PhysicalOperator, ScientificTarget,
    SemanticDigest,
};
use antecedent_estimate::joint_design::compiled::JointDesignRecipe;
use antecedent_identify::CompiledIdentificationProduct;

use crate::analysis::joint_panel::compiled::support_status;
use crate::analysis::joint_panel::designs::NativeJointIdentification;
use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, JointPanelLoweredParts,
    JointPanelRoute, LoweredProgram, NonGraphicalIdentification, PhysicalLayer, ProblemTarget,
    ProgramIdentityScope, RetainedIdentification, ScientificProblem, UncertaintyReading,
};
use crate::planner::statistical::{ModelRole, ModelSlot, Realization, StatisticalProgram};
use crate::support::CellStatus;

/// What a joint-design program fits instead of one quoted design matrix.
const DESIGN_REASON: &str = "compiled joint design functional over the design source";
/// Evaluator name of the design's identifying functional.
const EVALUATOR: &str = "joint_design_functional_v1";

/// The compiled design a joint-design program lowers.
#[derive(Clone, Copy, Debug)]
pub(crate) enum DesignInput<'a> {
    /// A design bound to the intervention worlds of `target`, with its target-bound
    /// identification product.
    Worlds {
        /// The scientific target whose worlds the design binds.
        target: &'a ScientificTarget,
        /// The identification product compiled from the design proof.
        identification: &'a CompiledIdentificationProduct,
        /// The design recipe the proof declares.
        recipe: &'a JointDesignRecipe,
    },
    /// A native design whose target kind (a local, cohort, structural or donor-model effect)
    /// is its own and is never read as an intervention-world mean.
    Native {
        /// The native identifying functional, target and required assumptions.
        identification: &'a NativeJointIdentification,
        /// The design recipe the native identification summarises.
        recipe: &'a JointDesignRecipe,
    },
}

/// Lower the program of a compiled joint design (`JointDesign`).
///
/// The six layers:
///
/// 1. the design's target: the worlds of a bound design, or the native target of a native
///    design;
/// 2. the identification product retained with the program for a bound design, and the
///    native identifying functional and required assumptions for a native design;
/// 3. the design's own identifying functional and estimator; the inference mode chosen at
///    execution does not change the program;
/// 4. the design source as trial evidence when the recipe's design randomizes assignment,
///    and as observation evidence otherwise;
/// 5. the read of `source_bytes` and the design-functional evaluation;
/// 6. the claim of the design's target kind.
///
/// `source` is the design dependency: the identity of the source data, design recipe and
/// target the design was compiled from.
///
/// # Errors
/// A program layer the planner refuses to declare.
pub(crate) fn lower(
    design: DesignInput<'_>,
    source: SemanticDigest,
    source_bytes: u64,
) -> Result<LoweredProgram, CausalError> {
    let status = support_status(JointPanelRoute::Design);
    let (target, identification, family, recipe) = match design {
        DesignInput::Worlds { target, identification, recipe } => {
            let method = identification.identified().first().map_or_else(
                || Arc::<str>::from("joint_design"),
                |estimand| estimand.method.clone(),
            );
            let retained = RetainedIdentification::new(
                Some(identification.clone()),
                Vec::new(),
                Some(*source.as_bytes()),
            )?;
            (
                ProblemTarget::Target(target.clone()),
                IdentificationLayer::Retained(Arc::new(retained)),
                method,
                recipe,
            )
        }
        DesignInput::Native { identification: native, recipe } => {
            let model = native_digest(native);
            (
                ProblemTarget::QueryFunctional {
                    kind: native_kind(recipe),
                    outcome: None,
                    query: model,
                },
                IdentificationLayer::NonGraphical(Arc::new(
                    NonGraphicalIdentification::SpecifiedModelConditional { model },
                )),
                Arc::<str>::from(native.method.as_str()),
                recipe,
            )
        }
    };
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target,
        plan_id: Arc::from("joint_design"),
        data: DataClassification::Panel,
        graph_class: "design_native",
        query_variables: Arc::from([]),
        source_digest: Some(*source.as_bytes()),
        support_coordinate: None,
        supported: status == CellStatus::Supported,
        support_status: Some(status),
    };
    let shape = DesignShape::not_design_based(DESIGN_REASON);
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        // The inference mode chosen at execution changes the law, not the program.
        realization: Realization::ExactPoint { evaluator: Arc::from(EVALUATOR) },
        models: Arc::from([ModelSlot {
            role: ModelRole::Outcome,
            family,
            inputs: Arc::from([]),
            output: None,
        }]),
        folds: None,
        resampling: None,
        posterior: None,
        design: shape,
        operators: Arc::from(read_and_fit()?),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef {
            role: source_role(recipe),
            digest: Some(*source.as_bytes()),
        }]),
        composition: Composition::Single,
        decision: None,
        planning: None,
        chain: None,
    };
    let physical =
        PhysicalLayer { operators: Arc::from(physical_read_and_fit(source_bytes)?), design: shape };
    let claim =
        ClaimDemand { uncertainty: UncertaintyReading::None, kind: Some(ClaimKind::Response) };
    Ok(JointPanelLoweredParts {
        route: JointPanelRoute::Design,
        scientific,
        identification,
        statistical,
        evidence,
        physical,
        claim,
    }
    .finish())
}

/// The read and fit operators every panel-source program declares.
pub(super) fn read_and_fit() -> Result<[Operator; 2], CausalError> {
    let read_rows = Operator::new(
        OperatorDeclaration::new(
            OperatorEffect::Read,
            OperatorGraph::Execution,
            OperatorPorts::new([OperatorRole::Population], [OperatorRole::Outcome]),
        )
        .with_evidence_claims(EvidenceClaims::observation_read()),
    )
    .map_err(refuse)?;
    let fit = Operator::new(
        OperatorDeclaration::new(
            OperatorEffect::Fit,
            OperatorGraph::Execution,
            OperatorPorts::new([OperatorRole::Outcome], [OperatorRole::FitHandle]),
        )
        .with_evidence_claims(EvidenceClaims::nuisance_fit()),
    )
    .map_err(refuse)?;
    Ok([read_rows, fit])
}

/// The physical read of `source_bytes` followed by the fit.
pub(super) fn physical_read_and_fit(
    source_bytes: u64,
) -> Result<[PhysicalOperator; 2], CausalError> {
    let read = PhysicalOperator::declare(PhysicalEffect::Read)
        .estimates(ByteEstimates::absent().with_output(source_bytes))
        .build()
        .map_err(refuse)?;
    let fit = PhysicalOperator::declare(PhysicalEffect::Fit).build().map_err(refuse)?;
    Ok([read, fit])
}

/// A lowering refusal, as the compiler reports an invalid declaration.
pub(super) fn refuse(message: impl std::fmt::Display) -> CausalError {
    CausalError::Compile {
        message: format!(
            "{}{}: joint_panel.lowering: {message}",
            antecedent_core::reason_code::PREFIX,
            antecedent_core::reason_code!("invalid_argument")
        ),
    }
}

/// The evidence role of a design's source, from the assignment its recipe declares: a
/// complete, cluster, matched-pair, switchback, sequential-excursion, whole-program or
/// encouragement randomization is trial evidence; a design over recorded histories,
/// rollouts, donor panels or instrument features is observation evidence.
const fn source_role(recipe: &JointDesignRecipe) -> EvidenceRole {
    match recipe {
        JointDesignRecipe::DecisionCluster { .. }
        | JointDesignRecipe::DecisionExperiment { .. }
        | JointDesignRecipe::Excursion { .. }
        | JointDesignRecipe::Randomized { .. }
        | JointDesignRecipe::Switchback { .. }
        | JointDesignRecipe::InstrumentedLocal { .. } => EvidenceRole::Trial,
        JointDesignRecipe::AdaptiveRegime { .. }
        | JointDesignRecipe::StaticRegime { .. }
        | JointDesignRecipe::SyntheticControl { .. }
        | JointDesignRecipe::Longitudinal { .. }
        | JointDesignRecipe::GaussianLongitudinal { .. }
        | JointDesignRecipe::Instrumented { .. }
        | JointDesignRecipe::DifferenceInDifferences { .. } => EvidenceRole::Observations,
    }
}

/// The functional kind a native design's target reads as, from its recipe: never an
/// intervention-world mean unless the target is one.
const fn native_kind(recipe: &JointDesignRecipe) -> FunctionalKind {
    match recipe {
        JointDesignRecipe::InstrumentedLocal { .. } | JointDesignRecipe::Instrumented { .. } => {
            FunctionalKind::AssignmentContrast
        }
        JointDesignRecipe::DifferenceInDifferences { .. }
        | JointDesignRecipe::SyntheticControl { .. } => FunctionalKind::EffectOnTreated,
        JointDesignRecipe::AdaptiveRegime { .. }
        | JointDesignRecipe::StaticRegime { .. }
        | JointDesignRecipe::Longitudinal { .. }
        | JointDesignRecipe::GaussianLongitudinal { .. } => FunctionalKind::RegimeValue,
        JointDesignRecipe::DecisionCluster { .. }
        | JointDesignRecipe::DecisionExperiment { .. }
        | JointDesignRecipe::Excursion { .. }
        | JointDesignRecipe::Randomized { .. }
        | JointDesignRecipe::Switchback { .. } => FunctionalKind::InterventionalDistribution,
    }
}

/// Digest of a native design's identifying functional, target and required assumptions.
fn native_digest(native: &NativeJointIdentification) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.lowering.joint_design.native.v1");
    writer
        .word(u64::from(native.version))
        .text(&native.method)
        .text(&native.identifying_functional)
        .text(&native.target_kind)
        .text(&native.population)
        .text(&native.sampling_unit)
        .word(native.targets.len() as u64);
    for target in &native.targets {
        writer.text(target);
    }
    writer.word(native.required_assumptions.len() as u64);
    for assumption in &native.required_assumptions {
        writer.text(assumption);
    }
    writer.finish()
}
