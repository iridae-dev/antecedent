//! Standalone compiled programs over evidence a caller supplies.
//!
//! A decision over supplied claims, a ranking of candidate studies and a plan of the evidence that
//! would repair a failed identification read no data table: their inputs are declared claims,
//! laws, artifacts, graphs or candidates. Each such computation compiles here to a six-layer
//! program of its own program family, so it has a route, a support unit read from
//! `parity/compiled_units.toml`, a program identity over every declared input, and a support gate
//! at execution. Execution calls the same kernel function as the direct path it replaces, on the
//! same inputs, so every number is the direct path's.
//!
//! [`TerminalSpec`] states the layers a family fills; [`lower`] seals them. The
//! `compiled_terminal!` macro declares a family's handle: the retained lowered program and the
//! declared inputs, with the shared plan view, identity and support gate.
//!
//! The program identity covers the declared inputs, written by
//! [`crate::analysis::supplied_identity::Declaration`] in the repository's canonical
//! length-prefixed encoding under a versioned tag per family. Nothing in it is a support
//! decision.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{ExecutionContext, PhysicalEffect, PhysicalOperator};

use crate::analysis::program_handle::{CompiledProgramHandle, gate_program};
use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, Composition, DecisionProgram, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramCell, ProgramIdentityScope,
    ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};

/// Why a supplied-evidence program quotes no design.
const NOT_DESIGN_BASED: &str =
    "a program over supplied evidence reads declared claims, laws or candidates, not a data table";

/// A refusal of a compiled program over supplied evidence.
///
/// The compiler refuses before the kernel runs (re-entry inside a lowered dispatch, or a
/// release build's support gate); otherwise a refusal is the kernel's own error, unchanged, so a
/// caller that matched the direct path's error matches it here. Cancellation is the kernel's to
/// refuse, with its own typed stop, as on the direct path.
#[derive(Debug)]
pub enum TerminalError<E> {
    /// The compiler refused the program.
    Refused(CausalError),
    /// The kernel the route calls refused, with its own error.
    Kernel(E),
}

impl<E> TerminalError<E> {
    /// The kernel's error, when the kernel refused.
    #[must_use]
    pub const fn kernel(&self) -> Option<&E> {
        match self {
            Self::Kernel(error) => Some(error),
            Self::Refused(_) => None,
        }
    }

    /// Map the kernel's error.
    pub fn map_kernel<F>(self, map: impl FnOnce(E) -> F) -> TerminalError<F> {
        match self {
            Self::Refused(error) => TerminalError::Refused(error),
            Self::Kernel(error) => TerminalError::Kernel(map(error)),
        }
    }
}

impl<E> From<CausalError> for TerminalError<E> {
    fn from(error: CausalError) -> Self {
        Self::Refused(error)
    }
}

impl<E: fmt::Display> fmt::Display for TerminalError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(error) => write!(formatter, "{error}"),
            Self::Kernel(error) => write!(formatter, "{error}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for TerminalError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Refused(error) => Some(error),
            Self::Kernel(error) => Some(error),
        }
    }
}

/// The layers a supplied-evidence family fills; [`lower`] seals them into a program.
pub(crate) struct TerminalSpec {
    /// The family's program cell.
    pub(crate) cell: ProgramCell,
    /// Versioned plan id of the family.
    pub(crate) plan_id: &'static str,
    /// What the program answers.
    pub(crate) functional: FunctionalKind,
    /// What structure the declared inputs carry.
    pub(crate) graph_class: &'static str,
    /// Identity of every declared input
    /// ([`crate::analysis::supplied_identity::Declaration::finish`]).
    pub(crate) question: [u8; 32],
    /// The kernel evaluator, by name.
    pub(crate) evaluator: &'static str,
    /// How the evaluator reads its inputs.
    pub(crate) reading: Reading,
    /// The role of the declared inputs.
    pub(crate) role: EvidenceRole,
    /// How the declared inputs combine.
    pub(crate) composition: Composition,
    /// The decision the program scores, when it scores one.
    pub(crate) decision: Option<DecisionProgram>,
    /// The kind of claim the result is.
    pub(crate) claim: Option<ClaimKind>,
}

/// How a supplied-evidence evaluator reads its inputs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Reading {
    /// An exact evaluation of declared inputs; no uncertainty is propagated.
    Exact,
    /// A declared predictive law read draw by draw.
    Predictive,
    /// A declared parameter posterior read draw by draw.
    Posterior,
    /// A structural or identified-set envelope; no probability is read.
    Envelope,
    /// Monte Carlo over declared laws under a seeded stream.
    MonteCarlo,
}

impl Reading {
    fn realization(self, evaluator: &'static str) -> Realization {
        let evaluator = Arc::<str>::from(evaluator);
        match self {
            Self::Exact | Self::Envelope | Self::MonteCarlo => {
                Realization::ExactPoint { evaluator }
            }
            Self::Predictive => Realization::DeclaredPredictive { evaluator },
            Self::Posterior => Realization::DeclaredPosterior { evaluator },
        }
    }

    const fn uncertainty(self) -> UncertaintyReading {
        match self {
            Self::Exact => UncertaintyReading::None,
            Self::Predictive => UncertaintyReading::Predictive,
            Self::Posterior => UncertaintyReading::Posterior,
            Self::Envelope => UncertaintyReading::Envelope,
            Self::MonteCarlo => UncertaintyReading::MonteCarlo,
        }
    }
}

/// Refuse a compile inside a lowered dispatch. A cancelled context compiles: the kernel refuses
/// it when the program runs, with the same typed stop the direct path returns.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub(crate) fn admit_compile(ctx: &ExecutionContext) -> Result<(), CausalError> {
    crate::analysis::lowering::refuse_reentry(ctx)?;
    Ok(())
}

/// Seal `spec` into a lowered program.
///
/// # Errors
/// A physical operator that cannot be declared.
pub(crate) fn lower(spec: TerminalSpec) -> Result<Arc<LoweredProgram>, CausalError> {
    let sources: Arc<[EvidenceSourceRef]> =
        Arc::from([EvidenceSourceRef { role: spec.role, digest: Some(spec.question) }]);
    lower_over(spec, sources)
}

/// Seal `spec` into a lowered program whose layer four reads `sources`: [`lower`] passes the one
/// declared source; a parent program over compiled child programs passes one source per child.
///
/// # Errors
/// A physical operator that cannot be declared.
pub(crate) fn lower_over(
    spec: TerminalSpec,
    sources: Arc<[EvidenceSourceRef]>,
) -> Result<Arc<LoweredProgram>, CausalError> {
    let operator = |effect: PhysicalEffect| {
        PhysicalOperator::declare(effect).build().map_err(|_| CausalError::Compile {
            message: format!("{}: the program's operator cannot be declared", spec.plan_id),
        })
    };
    let design = DesignShape::not_design_based(NOT_DESIGN_BASED);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: spec.functional,
            outcome: None,
            query: spec.question,
        },
        plan_id: Arc::from(spec.plan_id),
        data: DataClassification::Tabular,
        graph_class: spec.graph_class,
        query_variables: Arc::from(Vec::new()),
        source_digest: Some(spec.question),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: spec.question },
    ));
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: spec.reading.realization(spec.evaluator),
        models: Arc::from([]),
        folds: None,
        resampling: None,
        posterior: None,
        design,
        operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources,
        composition: spec.composition,
        decision: spec.decision,
        planning: None,
        chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            operator(PhysicalEffect::Read)?,
            operator(PhysicalEffect::Condition)?,
        ]),
        design,
    };
    Ok(Arc::new(
        StandaloneCompositionLoweredParts {
            cell: spec.cell,
            scientific,
            identification,
            statistical,
            evidence,
            physical,
            claim: ClaimDemand { uncertainty: spec.reading.uncertainty(), kind: spec.claim },
        }
        .finish(),
    ))
}

/// Enter `handle`'s execution: scope the context to its program and pass the support gate of
/// its route. Cancellation is left to the kernel, which refuses it with its own typed stop.
///
/// # Errors
/// A release build's refusal of a route whose unit is not supported.
pub(crate) fn enter<H: CompiledProgramHandle>(
    ctx: &ExecutionContext,
    handle: &H,
) -> Result<ExecutionContext, CausalError> {
    let scoped = ctx.enter_lowered_dispatch(handle.lowered().identity());
    gate_program(&scoped, handle).map_err(CausalError::from)?;
    Ok(scoped)
}

/// Declare a supplied-evidence program handle: the retained lowered program and the declared
/// inputs, with the plan view, identity and support decision every family shares.
macro_rules! compiled_terminal {
    (
        $(#[$meta:meta])*
        $name:ident { $($(#[$field_meta:meta])* $field:ident : $ty:ty),* $(,)? }
    ) => {
        $(#[$meta])*
        pub struct $name {
            lowered: std::sync::Arc<crate::planner::lowered::LoweredProgram>,
            $($(#[$field_meta])* $field: $ty,)*
        }

        impl crate::analysis::program_handle::CompiledProgramHandle for $name {
            fn program(&self) -> &std::sync::Arc<crate::planner::lowered::LoweredProgram> {
                &self.lowered
            }
        }

        impl $name {
            /// Every layer of the compiled program.
            #[must_use]
            pub fn lowered(&self) -> &crate::planner::lowered::LoweredProgram {
                &self.lowered
            }

            /// The plan view of the compiled program, with its route and the support decision
            /// of the route's unit. Reading it runs nothing.
            #[must_use]
            pub fn plan_view(&self) -> crate::planner::PlanView {
                crate::planner::PlanView::of_program(
                    std::sync::Arc::clone(&self.lowered),
                    crate::planner::PlanStage::Compiled,
                )
            }

            /// Identity of the compiled program: the family, its version and every declared
            /// input.
            #[must_use]
            pub fn identity(&self) -> antecedent_core::SemanticDigest {
                self.lowered.identity()
            }

            /// Enter execution: scope the context and pass the support gate.
            fn enter(
                &self,
                ctx: &antecedent_core::ExecutionContext,
            ) -> Result<antecedent_core::ExecutionContext, crate::error::CausalError> {
                crate::analysis::supplied_terminal::enter(ctx, self)
            }
        }
    };
}

pub(crate) use compiled_terminal;
