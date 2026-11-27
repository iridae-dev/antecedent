//! Compiled program families that run one kernel computation over a declared input.
//!
//! Several capabilities are a single kernel call over an input the caller declares in full: a
//! measured source replay, a nested Markov fit, a learned joint transport fit, a scenario
//! covariance, an external feedback system. Each such family compiles here to a
//! [`KernelProgram`]: the declared input and a six-layer lowered program whose scientific
//! problem names the family's plan and functional and binds the digest of the input's canonical
//! bytes, whose identification is conditional on that declared input, whose statistical layer
//! names the kernel evaluator, and whose claim states how the published uncertainty must be
//! read. Every execution of the handle first passes the support gate of the unit its route is
//! bound to (`parity/compiled_units.toml`); the computation it then runs is the kernel function
//! the direct path calls, with the same inputs, so the two give the same numbers.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::DataClassification;
use antecedent_core::{
    ByteEstimates, CanonicalWriter, ExecutionContext, NotSupported, PhysicalEffect,
    PhysicalOperator, ScoreFrequentist, ScoreModel,
};

use super::program_handle::{CompiledProgramHandle, gate_program};
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramCell, ProgramIdentityScope,
    ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};

pub mod bridge_images;
pub mod chain_recovery;
pub mod counterfactual_seal;
pub mod discretized_bridge;
pub mod exposure_measurement;
pub mod external_feedback;
pub mod fitted_mechanism_transport;
pub mod learned_joint_transport;
pub mod measured_exposure_effect;
pub mod nested_markov;
pub mod patient_surrogate;
pub mod policy_transport;
pub mod power_prior_selection;
pub mod prior_composition;
pub mod sampled_counterfactual_transport;
pub mod sampled_recovery;
pub mod sampling_bridge;
pub mod scenario_covariance;
pub mod sequential_evaluation;
pub mod split_effect;
pub mod surrogate_bridge;
pub mod transported_synthesis;
pub mod z_transport;

/// How a family's kernel turns its input into the published answer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum KernelRealization {
    /// A score or estimating-equation estimate with a sampling law.
    Score,
    /// Sampling-law map, optionally with whole validation-trial refits.
    ReplicateMapping { validation_refits: Option<u32> },
    /// Deterministic endpoint or joint-row transformation.
    FixedImage,
    /// A posterior evaluated by the kernel under its declared likelihood and prior.
    Posterior,
    /// An exact evaluation of an identified functional of the declared input.
    Exact,
    /// A sharp compatibility set or its empirical counterpart.
    PartialIdentification,
}

/// The fixed shape of one kernel program family.
#[derive(Clone, Copy, Debug)]
pub(crate) struct KernelProgramShape {
    /// The family's program cell; its name is the route name.
    pub(crate) cell: ProgramCell,
    /// Stable plan id of the family.
    pub(crate) plan_id: &'static str,
    /// Functional the family answers.
    pub(crate) functional: FunctionalKind,
    /// Stable name of the kernel evaluator.
    pub(crate) evaluator: &'static str,
    /// How the kernel realizes the answer.
    pub(crate) realization: KernelRealization,
    /// How the published uncertainty must be read.
    pub(crate) uncertainty: UncertaintyReading,
    /// What the declared input contributes.
    pub(crate) role: EvidenceRole,
    /// Structure class the input declares, or `"none"`.
    pub(crate) graph_class: &'static str,
    /// Data modality of the input.
    pub(crate) data: DataClassification,
    /// What the family fits instead of a single quoted design.
    pub(crate) design: &'static str,
}

/// Digest of a family's canonical input bytes under the family's plan id.
pub(crate) fn input_digest(shape: &KernelProgramShape, canonical: &[u8]) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.kernel_program.input.v1");
    writer.text(shape.plan_id).bytes(canonical);
    writer.finish()
}

fn invalid(shape: &KernelProgramShape, message: &str) -> CausalError {
    crate::compile_reason!("invalid_argument", "{}.lowering: {message}", shape.plan_id)
}

fn kernel_realization(shape: &KernelProgramShape) -> Result<Realization, CausalError> {
    Ok(match shape.realization {
        KernelRealization::Score | KernelRealization::ReplicateMapping { .. } => {
            let score = ScoreModel::new(shape.evaluator)
                .map_err(|_| invalid(shape, "blank evaluator name"))?;
            Realization::Frequentist(ScoreFrequentist::new(score))
        }
        KernelRealization::Posterior => {
            Realization::DeclaredPosterior { evaluator: Arc::from(shape.evaluator) }
        }
        KernelRealization::FixedImage if shape.uncertainty == UncertaintyReading::Predictive => {
            Realization::DeclaredPredictive { evaluator: Arc::from(shape.evaluator) }
        }
        KernelRealization::Exact
        | KernelRealization::PartialIdentification
        | KernelRealization::FixedImage => {
            Realization::ExactPoint { evaluator: Arc::from(shape.evaluator) }
        }
    })
}

fn kernel_physical(
    shape: &KernelProgramShape,
    canonical: &[u8],
    design: DesignShape,
) -> Result<PhysicalLayer, CausalError> {
    let read = PhysicalOperator::declare(PhysicalEffect::Read)
        .estimates(ByteEstimates::absent().with_output(canonical.len() as u64))
        .build()
        .map_err(|_| invalid(shape, "invalid read operator"))?;
    let effect = match shape.realization {
        KernelRealization::PartialIdentification => PhysicalEffect::Condition,
        KernelRealization::FixedImage
        | KernelRealization::ReplicateMapping { validation_refits: None } => {
            PhysicalEffect::Transport
        }
        _ => PhysicalEffect::Fit,
    };
    let fit = PhysicalOperator::declare(effect)
        .build()
        .map_err(|_| invalid(shape, "invalid fit operator"))?;
    Ok(PhysicalLayer { operators: Arc::from([read, fit]), design })
}

/// Lower one kernel computation over the input whose canonical bytes are `canonical`.
pub(crate) fn lower(
    shape: &KernelProgramShape,
    canonical: &[u8],
) -> Result<LoweredProgram, CausalError> {
    let input = input_digest(shape, canonical);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: shape.functional,
            outcome: None,
            query: input,
        },
        plan_id: Arc::from(shape.plan_id),
        data: shape.data,
        graph_class: shape.graph_class,
        query_variables: Arc::from([]),
        source_digest: Some(input),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SpecifiedModelConditional { model: input },
    ));
    let realization = kernel_realization(shape)?;
    let design = DesignShape::not_design_based(shape.design);
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization,
        models: Arc::from([]),
        folds: None,
        resampling: match shape.realization {
            KernelRealization::ReplicateMapping { validation_refits: Some(replicates) } => {
                Some(crate::planner::statistical::ResamplingPlan {
                    method: crate::planner::statistical::ResamplingMethod::Bootstrap,
                    replicates,
                    unit: crate::planner::statistical::ResampleUnit::Row,
                })
            }
            _ => None,
        },
        posterior: None,
        design,
        operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef { role: shape.role, digest: Some(input) }]),
        composition: Composition::Single,
        decision: None,
        planning: None,
        chain: None,
    };
    let physical = kernel_physical(shape, canonical, design)?;
    Ok(StandaloneCompositionLoweredParts {
        cell: shape.cell,
        scientific,
        identification,
        statistical,
        evidence,
        physical,
        claim: ClaimDemand {
            uncertainty: shape.uncertainty,
            kind: match shape.realization {
                KernelRealization::PartialIdentification | KernelRealization::FixedImage
                    if shape.uncertainty == UncertaintyReading::Envelope =>
                {
                    Some(antecedent_contract::ClaimKind::Bounds)
                }
                KernelRealization::FixedImage | KernelRealization::ReplicateMapping { .. } => {
                    Some(antecedent_contract::ClaimKind::Mixture)
                }
                _ => None,
            },
        },
    }
    .finish())
}

/// A compiled kernel program: the declared input and the lowered program that binds it.
#[derive(Clone, Debug)]
pub(crate) struct KernelProgram<I> {
    input: I,
    lowered: Arc<LoweredProgram>,
}

impl<I> KernelProgram<I> {
    /// Compile `input`, whose canonical bytes are `canonical`, under `shape`. Compiling executes
    /// nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, cancellation, or a malformed program.
    pub(crate) fn compile(
        shape: &KernelProgramShape,
        input: I,
        canonical: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<Self, CausalError> {
        crate::analysis::lowering::refuse_reentry(ctx)?;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "kernel program compile" });
        }
        let lowered = Arc::new(lower(shape, canonical)?);
        Ok(Self { input, lowered })
    }

    /// The declared input.
    pub(crate) const fn input(&self) -> &I {
        &self.input
    }

    /// Pass the support gate of the family's unit before an execution.
    ///
    /// # Errors
    ///
    /// [`NotSupported`] in a release build when the unit is not supported.
    pub(crate) fn gate(&self, ctx: &ExecutionContext) -> Result<(), NotSupported> {
        gate_program(ctx, self)
    }

    /// The plan view of the compiled program.
    pub(crate) fn view(&self) -> PlanView {
        CompiledProgramHandle::plan_view(self)
    }

    /// Every layer of the compiled program.
    pub(crate) fn program_layers(&self) -> &LoweredProgram {
        &self.lowered
    }
}

impl<I> CompiledProgramHandle for KernelProgram<I> {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

/// The support refusal as an artifact error, for a family whose kernel reports artifact errors.
pub(crate) fn io_refusal(refusal: &NotSupported) -> antecedent_io::IoError {
    antecedent_io::IoError::Refused {
        code: antecedent_core::reason_code!("not_supported"),
        message: refusal.message().to_owned(),
    }
}

/// The canonical CBOR bytes of a declared wire input.
///
/// # Errors
///
/// A value that does not encode.
pub(crate) fn canonical_cbor<T: serde::Serialize>(
    value: &T,
) -> Result<Vec<u8>, antecedent_io::IoError> {
    antecedent_io::to_cbor(value)
}

#[cfg(test)]
mod site_tests {
    use super::*;
    use crate::identity_laws::{ADVERSARIAL_TEXT, all_distinct, injective};

    const fn shape(plan_id: &'static str) -> KernelProgramShape {
        KernelProgramShape {
            cell: ProgramCell::SampleSplitEffect,
            plan_id,
            functional: FunctionalKind::Association,
            evaluator: "site_test_evaluator",
            realization: KernelRealization::Exact,
            uncertainty: UncertaintyReading::None,
            role: EvidenceRole::Observations,
            graph_class: "none",
            data: DataClassification::Tabular,
            design: "site test",
        }
    }

    /// `input_digest`: the plan id and the family's canonical bytes are framed, so moving bytes
    /// across their boundary names another input; the lowered program's question, source digest
    /// and specified model are that one digest.
    #[test]
    fn site_kernel_programs_input_digest_frames_the_plan_and_the_bytes() {
        injective(
            &ADVERSARIAL_TEXT,
            |parts| *parts,
            |&(plan, bytes)| Some(input_digest(&shape(plan), bytes.as_bytes())),
        );
        let digest = input_digest(&shape("plan"), b"input");
        let lowered = lower(&shape("plan"), b"input").unwrap();
        assert_eq!(lowered.scientific().source_digest, Some(digest));
        all_distinct(&[
            lowered.identity(),
            lower(&shape("plan"), b"other").unwrap().identity(),
            lower(&shape("other"), b"input").unwrap().identity(),
        ]);
    }
}
