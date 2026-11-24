//! Lower a process-local feedback system of declared callback mechanisms into the common
//! six-layer program.
//!
//! The answer is conditional on the specified mechanisms, so the identification layer holds the
//! digest of the system: every mechanism's id, reads and callback declaration. The program's
//! identity scope is `RuntimeCallbackDeclarationOnly`: the declaration is committed to, the
//! callback's code and state are not, and the program is not portable.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::DataClassification;
use antecedent_core::{
    ByteEstimates, CallbackImplementation, CanonicalEncode, CanonicalWriter, PhysicalEffect,
    PhysicalOperator,
};

use crate::analysis::compiled_local_cyclic::{DeclaredCyclicMechanism, LocalCyclicRun};
use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, FunctionalKind,
    IdentificationLayer, LocalCyclicCallbackLoweredParts, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramIdentityScope,
    ScientificProblem, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};

/// Domain tag of the system's digest.
const SYSTEM_TAG: &str = "antecedent.local_cyclic_callback.system";
/// Domain tag of the run's digest.
const RUN_TAG: &str = "antecedent.local_cyclic_callback.run";
/// Plan id and evaluator.
const PLAN_ID: &str = "cyclic.local_callback_solve";

fn error(message: impl std::fmt::Display) -> CausalError {
    crate::compile_reason!("invalid_argument", "local_cyclic_callback.lowering: {message}")
}

/// Write every mechanism's id, reads and callback declaration.
fn write_system(writer: &mut CanonicalWriter, mechanisms: &[DeclaredCyclicMechanism]) {
    writer.word(mechanisms.len() as u64);
    for mechanism in mechanisms {
        let spec = mechanism.callback.spec();
        if matches!(spec.implementation, CallbackImplementation::Runtime) {
            writer.note_runtime_callback();
        }
        writer.text(&mechanism.id);
        writer.word(mechanism.reads.len() as u64);
        for &read in &mechanism.reads {
            writer.word(read as u64);
        }
        writer.text(&spec.name);
        writer.texts(spec.inputs.as_ref());
        spec.output.encode(writer);
        writer.word(spec.dependencies.len() as u64);
        for dependency in spec.dependencies.as_ref() {
            writer.text(&dependency.name).text(&dependency.version);
        }
        spec.implementation.encode(writer);
    }
}

fn system_digest(mechanisms: &[DeclaredCyclicMechanism]) -> [u8; 32] {
    let mut writer = CanonicalWriter::new(SYSTEM_TAG);
    write_system(&mut writer, mechanisms);
    writer.finish()
}

fn run_digest(system: [u8; 32], run: &LocalCyclicRun) -> [u8; 32] {
    let mut writer = CanonicalWriter::new(RUN_TAG);
    writer.bytes(&system);
    writer.reals(&run.noise);
    writer.word(run.clamps.len() as u64);
    for clamp in &run.clamps {
        writer.word(clamp.variable as u64).real(clamp.value);
    }
    writer.reals(&run.initial);
    match &run.factual {
        Some(factual) => {
            writer.word(1).reals(factual);
        }
        None => {
            writer.word(0);
        }
    }
    writer
        .real(run.policy.tolerance)
        .word(run.policy.max_iterations as u64)
        .word(run.policy.max_evaluations as u64);
    writer.finish()
}

/// Lower `run` of the system `mechanisms`.
///
/// # Errors
///
/// A system whose size overflows the resource quote.
pub(crate) fn lower(
    mechanisms: &[DeclaredCyclicMechanism],
    run: &LocalCyclicRun,
) -> Result<LoweredProgram, CausalError> {
    let system = system_digest(mechanisms);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::RuntimeCallbackDeclarationOnly,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::Simulation,
            outcome: None,
            query: run_digest(system, run),
        },
        plan_id: Arc::from(PLAN_ID),
        data: DataClassification::Tabular,
        graph_class: "cyclic",
        query_variables: Arc::from(Vec::new()),
        source_digest: None,
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SpecifiedModelConditional { model: system },
    ));
    let design = DesignShape::not_design_based(
        "specified process-local feedback mechanisms; numerical roots carry no uniqueness claim",
    );
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::ExactPoint { evaluator: Arc::from(PLAN_ID) },
        models: Arc::from(Vec::new()),
        folds: None,
        resampling: None,
        posterior: None,
        design,
        operators: Arc::from(Vec::new()),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from(Vec::new()),
        composition: Composition::Single,
        decision: None,
        planning: None,
        chain: None,
    };
    let n = mechanisms.len() as u64;
    let output = n
        .checked_mul(n)
        .and_then(|cells| cells.checked_mul(64))
        .and_then(|bytes| bytes.checked_add(n.saturating_mul(256)))
        .ok_or_else(|| error("system size overflow"))?;
    let read = PhysicalOperator::declare(PhysicalEffect::Read)
        .estimates(ByteEstimates::absent().with_output(output))
        .build()
        .map_err(error)?;
    let physical = PhysicalLayer { operators: Arc::from(vec![read]), design };
    let claim = ClaimDemand { uncertainty: UncertaintyReading::None, kind: None };
    Ok(LocalCyclicCallbackLoweredParts {
        scientific,
        identification,
        statistical,
        evidence,
        physical,
        claim,
    }
    .finish())
}
