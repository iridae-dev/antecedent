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

/// Digest of the run. The clamps are keyed by the variable they fix (the solve sets each
/// clamped coordinate and refuses a variable clamped twice), so they are written sorted.
fn run_digest(system: [u8; 32], run: &LocalCyclicRun) -> [u8; 32] {
    let mut writer = CanonicalWriter::new(RUN_TAG);
    writer.bytes(&system);
    writer.reals(&run.noise);
    let mut clamps: Vec<(usize, u64)> =
        run.clamps.iter().map(|clamp| (clamp.variable, clamp.value.to_bits())).collect();
    clamps.sort_unstable();
    writer.word(clamps.len() as u64);
    for (variable, value) in clamps {
        writer.word(variable as u64).word(value);
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

#[cfg(test)]
mod site_tests {
    use std::sync::Arc;

    use antecedent_core::{
        BatchCallback, CallbackError, CallbackImplementation, CallbackOutput, CallbackSpec,
        ExecutionContext,
    };
    use antecedent_model::local_cyclic::LocalCyclicPolicy;
    use antecedent_model::nonlinear_equilibrium::Clamp;

    use super::{run_digest, system_digest};
    use crate::analysis::compiled_local_cyclic::{
        DeclaredCyclicMechanism, LocalCyclicRun, compile_local_cyclic,
    };
    use crate::identity_laws::{
        ADVERSARIAL_TEXT, all_distinct, bit_exact, injective, ordered_laws, set_laws,
    };
    use crate::planner::lowered::ProblemTarget;

    #[derive(Debug)]
    struct Declared(CallbackSpec);

    impl BatchCallback for Declared {
        fn spec(&self) -> &CallbackSpec {
            &self.0
        }
        fn call(&self, _inputs: &[&[f64]], out: &mut [f64]) -> Result<(), CallbackError> {
            out.fill(0.0);
            Ok(())
        }
        fn value_and_derivatives(&self, inputs: &[f64]) -> Result<(f64, Vec<f64>), CallbackError> {
            Ok((0.0, vec![0.0; inputs.len()]))
        }
    }

    fn mechanism(id: &str, name: &str, reads: Vec<usize>) -> DeclaredCyclicMechanism {
        let inputs: Vec<String> = reads.iter().map(|read| format!("x{read}")).collect();
        DeclaredCyclicMechanism {
            id: id.to_owned(),
            reads,
            callback: Arc::new(Declared(CallbackSpec {
                name: name.to_owned(),
                inputs: Arc::from(inputs),
                output: CallbackOutput::Real,
                dependencies: Arc::from([]),
                implementation: CallbackImplementation::Runtime,
            })),
        }
    }

    /// A three-variable feedback loop: x0 reads x1, x1 reads x2, x2 reads x0.
    fn system() -> Vec<DeclaredCyclicMechanism> {
        vec![
            mechanism("x0", "f0", vec![1]),
            mechanism("x1", "f1", vec![2]),
            mechanism("x2", "f2", vec![0]),
        ]
    }

    fn run(clamps: Vec<Clamp>, noise: Vec<f64>, factual: Option<Vec<f64>>) -> LocalCyclicRun {
        LocalCyclicRun {
            noise,
            clamps,
            initial: vec![0.0; 3],
            factual,
            policy: LocalCyclicPolicy::default(),
        }
    }

    /// The question digest of the compiled run, `None` when compilation or the solve refuses it.
    fn question(run: &LocalCyclicRun) -> Option<[u8; 32]> {
        let ctx = ExecutionContext::for_tests(3);
        let compiled = compile_local_cyclic(&system(), run, &ctx).ok()?;
        match compiled.lowered().scientific().target() {
            ProblemTarget::QueryFunctional { query, .. } => Some(*query),
            _ => None,
        }
    }

    #[test]
    fn site_local_cyclic_callback_clamps_are_a_set() {
        // Each clamp fixes one variable, and the solve refuses a variable clamped twice.
        let clamps = [Clamp { variable: 0, value: 0.5 }, Clamp { variable: 1, value: -0.25 }];
        set_laws(&clamps, |clamps| question(&run(clamps.to_vec(), vec![0.4, -0.2, 0.1], None)));
    }

    #[test]
    fn site_local_cyclic_callback_system_and_run_are_ordered_and_framed() {
        // Mechanism i is variable i, and the reads are the callback's inputs in call order.
        let declared = [("x0", "f0", 1_usize), ("x1", "f1", 0)];
        ordered_laws(&declared, |order| {
            let mechanisms: Vec<_> =
                order.iter().map(|&(id, name, read)| mechanism(id, name, vec![read])).collect();
            Some(system_digest(&mechanisms))
        });
        ordered_laws(&[0_usize, 2], |reads| {
            Some(system_digest(&[mechanism("x", "f", reads.to_vec())]))
        });
        injective(
            &ADVERSARIAL_TEXT,
            |pair| *pair,
            |pair| Some(system_digest(&[mechanism(pair.0, pair.1, vec![0])])),
        );
        // Noise is in variable order; reals are written by their bits.
        let declared_system = system_digest(&system());
        ordered_laws(&[0.4, -0.2], |noise| {
            Some(run_digest(declared_system, &run(Vec::new(), noise.to_vec(), None)))
        });
        bit_exact(|value| {
            Some(run_digest(
                declared_system,
                &run(vec![Clamp { variable: 0, value }], vec![value, 0.0, 0.0], None),
            ))
        });
        // The factual state has a presence word.
        let noise = || vec![0.4, -0.2, 0.1];
        all_distinct(&[
            run_digest(declared_system, &run(Vec::new(), noise(), None)),
            run_digest(declared_system, &run(Vec::new(), noise(), Some(Vec::new()))),
            run_digest(declared_system, &run(Vec::new(), noise(), Some(vec![0.0; 3]))),
        ]);
    }
}
