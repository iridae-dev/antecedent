//! Transported synthesis as three compiled program cells.
//!
//! `transported_synthesis_response` registers a pool, a borrowing update or a transported prior
//! and builds the response. Compiling binds that declaration, including the ledger's state
//! digest, and selects the cell from the method. Executing passes that cell's support gate and
//! calls [`transported_synthesis_response_kernel`] with the bound declaration and the caller's
//! ledger. The kernel still replaces the ledger only when both the synthesis and the response
//! were built.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::evidence_ledger::{EvidenceLedger, SourceId};
use antecedent_contract::{DataClassification, ExternalRefusal, ScientificQuantity};
use antecedent_core::{CanonicalWriter, ExecutionContext, NotSupported, SemanticDigest};
use antecedent_design::transported_synthesis::{
    TransportedContribution, TransportedSynthesis, TransportedSynthesisMethod,
};
use antecedent_stats::synthesis::{MeanPrior, TauPrior};

use super::{KernelProgram, KernelProgramShape, KernelRealization};
use crate::analysis::Study;
use crate::analysis::supplied_identity::{Canonical, Declaration, encoded, variant};
use crate::analysis::transported_constructors::{
    TransportedConstruction, transported_synthesis_response_kernel,
};
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

const TAG: &str = "antecedent.kernel_program.transported_synthesis";

const POOLING: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::SynthesisPooling,
    plan_id: "transport.synthesis_pooling",
    functional: FunctionalKind::PropagatedChain,
    evaluator: "transported_synthesis_pooling",
    realization: KernelRealization::Exact,
    uncertainty: UncertaintyReading::None,
    role: EvidenceRole::TransportSource,
    graph_class: "evidence_ledger",
    data: DataClassification::MultiEnvironment,
    design: "generalized least squares of native and transported claims, transport error once",
};

const BORROWING: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::SynthesisBorrowing,
    plan_id: "transport.synthesis_borrowing",
    functional: FunctionalKind::PropagatedChain,
    evaluator: "transported_synthesis_borrowing",
    realization: KernelRealization::Exact,
    uncertainty: UncertaintyReading::None,
    role: EvidenceRole::TransportSource,
    graph_class: "evidence_ledger",
    data: DataClassification::MultiEnvironment,
    design: "hierarchical borrowing of native and transported claims under a declared prior",
};

const TRANSPORTED_PRIOR: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::SynthesisTransportedPrior,
    plan_id: "transport.synthesis_transported_prior",
    functional: FunctionalKind::PropagatedChain,
    evaluator: "transported_synthesis_transported_prior",
    realization: KernelRealization::Exact,
    uncertainty: UncertaintyReading::None,
    role: EvidenceRole::TransportSource,
    graph_class: "evidence_ledger",
    data: DataClassification::MultiEnvironment,
    design: "a transported claim as the prior of a native likelihood",
};

/// The declaration `transported_synthesis_response_kernel` reads, apart from the caller's ledger.
///
/// Contributions are sorted by source identity before they are written: the kernel sorts the
/// same way, and a repeated source stays so that listing a source twice does not share a digest
/// with listing it once. The sampling covariance is already in that sorted order.
#[derive(Clone, Debug)]
struct OwnedSynthesis {
    ledger_state: SemanticDigest,
    label: String,
    contributions: Vec<TransportedContribution>,
    method: TransportedSynthesisMethod,
    native_population: Option<String>,
    coordinate: ScientificQuantity,
    provider_id: String,
}

impl Canonical for SourceId {
    fn encode(&self, out: &mut CanonicalWriter) {
        self.digest().encode(out);
    }
}

impl Canonical for TransportedContribution {
    fn encode(&self, out: &mut CanonicalWriter) {
        self.source.encode(out);
        self.estimate.encode(out);
        self.sampling_variance.encode(out);
        self.transport_variance.encode(out);
    }
}

impl Canonical for MeanPrior {
    fn encode(&self, out: &mut CanonicalWriter) {
        match *self {
            Self::Flat => {
                variant(out, "flat");
            }
            Self::Normal { mean, variance } => {
                variant(out, "normal");
                mean.encode(out);
                variance.encode(out);
            }
        }
    }
}

impl Canonical for TauPrior {
    fn encode(&self, out: &mut CanonicalWriter) {
        // Index i is support point i; the probabilities are that same order.
        self.support.encode(out);
        self.probabilities.encode(out);
    }
}

impl Canonical for TransportedSynthesisMethod {
    fn encode(&self, out: &mut CanonicalWriter) {
        match self {
            Self::Pool { sampling_covariance } => {
                variant(out, "pool");
                sampling_covariance.encode(out);
            }
            Self::Borrow { mean_prior, tau_prior } => {
                variant(out, "borrow");
                mean_prior.encode(out);
                tau_prior.encode(out);
            }
            Self::TransportedPrior { prior } => {
                variant(out, "transported_prior");
                prior.encode(out);
            }
        }
    }
}

impl Canonical for OwnedSynthesis {
    fn encode(&self, out: &mut CanonicalWriter) {
        self.ledger_state.encode(out);
        self.label.encode(out);
        let mut contributions = self.contributions.clone();
        contributions.sort_by_key(|row| row.source);
        contributions.encode(out);
        self.method.encode(out);
        self.native_population.encode(out);
        self.coordinate.encode(out);
        self.provider_id.encode(out);
    }
}

fn shape_of(method: &TransportedSynthesisMethod) -> &'static KernelProgramShape {
    match method {
        TransportedSynthesisMethod::Pool { .. } => &POOLING,
        TransportedSynthesisMethod::Borrow { .. } => &BORROWING,
        TransportedSynthesisMethod::TransportedPrior { .. } => &TRANSPORTED_PRIOR,
    }
}

fn synthesis_bytes(input: &OwnedSynthesis) -> Result<[u8; 32], CausalError> {
    let shape = shape_of(&input.method);
    encoded(shape.plan_id, || {
        let mut declaration = Declaration::new(TAG);
        declaration.field("request", input);
        declaration.finish()
    })
}

fn support_refusal(refusal: &NotSupported) -> ExternalRefusal {
    ExternalRefusal::new(refusal.code(), "bind", refusal.detail).with_remedy(
        "run the route in a development build, which records the missing evidence, or use a \
         supported route",
    )
}

/// A compiled transported synthesis. The cell is the method's unit.
#[derive(Clone, Debug)]
pub struct CompiledTransportedSynthesis {
    program: KernelProgram<OwnedSynthesis>,
}

impl CompiledTransportedSynthesis {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// Pass the support gate, then call `transported_synthesis_response_kernel` with the bound
    /// declaration and `ledger`.
    ///
    /// The ledger is replaced only when both the synthesis and the response were built.
    ///
    /// # Errors
    /// The support refusal, and every refusal of `transported_synthesis_response_kernel`.
    pub fn execute(
        &self,
        ledger: &mut EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> Result<TransportedConstruction<TransportedSynthesis>, ExternalRefusal> {
        self.program.gate(ctx).map_err(|refusal| support_refusal(&refusal))?;
        let input = self.program.input();
        transported_synthesis_response_kernel(
            ledger,
            &input.label,
            &input.contributions,
            &input.method,
            input.native_population.as_deref(),
            input.coordinate.clone(),
            &input.provider_id,
        )
    }
}

impl Study {
    /// Compile `transported_synthesis_response` for this declaration. The method selects the
    /// cell: pooling, borrowing, or a transported prior. The ledger is borrowed for its state
    /// digest; executing passes the caller's ledger to the kernel. Compiling executes nothing.
    ///
    /// # Errors
    /// Cancellation, compiling from inside a lowered dispatch, or a declaration with no encoding.
    #[allow(
        clippy::too_many_arguments,
        reason = "the kernel's arguments, plus the context the support gate reads"
    )]
    pub fn compile_transported_synthesis(
        ledger: &EvidenceLedger,
        label: &str,
        contributions: &[TransportedContribution],
        method: &TransportedSynthesisMethod,
        native_population: Option<&str>,
        coordinate: ScientificQuantity,
        provider_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<CompiledTransportedSynthesis, CausalError> {
        let owned = OwnedSynthesis {
            ledger_state: ledger.state_digest(),
            label: label.to_owned(),
            contributions: contributions.to_vec(),
            method: method.clone(),
            native_population: native_population.map(str::to_owned),
            coordinate,
            provider_id: provider_id.to_owned(),
        };
        let bytes = synthesis_bytes(&owned)?;
        let program = KernelProgram::compile(shape_of(method), owned, &bytes, ctx)?;
        Ok(CompiledTransportedSynthesis { program })
    }
}

#[cfg(test)]
mod site_tests {
    use super::*;
    use crate::identity_laws::{
        ADVERSARIAL_TEXT, bit_exact, injective, multiset_laws, ordered_laws,
    };
    use antecedent_contract::QuantityRole;

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::from_bytes([byte; 32])
    }

    fn source(byte: u8) -> SourceId {
        SourceId::from_digest(digest(byte))
    }

    fn row(byte: u8, estimate: f64, transport: Option<f64>) -> TransportedContribution {
        TransportedContribution {
            source: source(byte),
            estimate,
            sampling_variance: 1.0,
            transport_variance: transport,
        }
    }

    fn coordinate() -> ScientificQuantity {
        ScientificQuantity {
            variable_id: "pooled".into(),
            variable_name: "pooled".into(),
            role: QuantityRole::Outcome,
            units: "units".into(),
            population_id: "registry".into(),
            regime_id: "do(treatment=1)".into(),
            horizon: 0,
            functional_id: "average_effect".into(),
            conditioning: Vec::new(),
            transform_id: "identity".into(),
        }
    }

    fn owned(
        rows: &[TransportedContribution],
        covariance: Option<Vec<f64>>,
        label: &str,
        population: Option<&str>,
    ) -> OwnedSynthesis {
        OwnedSynthesis {
            ledger_state: digest(9),
            label: label.to_owned(),
            contributions: rows.to_vec(),
            method: TransportedSynthesisMethod::Pool { sampling_covariance: covariance },
            native_population: population.map(str::to_owned),
            coordinate: coordinate(),
            provider_id: "synthesis".to_owned(),
        }
    }

    fn bytes(input: &OwnedSynthesis) -> Option<[u8; 32]> {
        synthesis_bytes(input).ok()
    }

    /// Contributions are a multiset keyed by source: the kernel sorts them and refuses a repeat.
    /// The covariance is the sorted-source order, each real by its bits. The label and the
    /// provider are framed.
    #[test]
    fn site_synthesis_contributions_are_a_multiset_and_the_covariance_is_ordered() {
        let rows = [row(1, 1.0, None), row(2, 2.0, Some(0.5))];
        multiset_laws(&rows, |items| bytes(&owned(items, None, "pooled", Some("registry"))));
        ordered_laws(&[0.1, 0.2], |covariance| {
            bytes(&owned(&rows, Some(covariance.to_vec()), "pooled", Some("registry")))
        });
        bit_exact(|estimate| {
            bytes(&owned(&[row(1, estimate, None)], None, "pooled", Some("registry")))
        });
        injective(
            &ADVERSARIAL_TEXT,
            |parts| *parts,
            |&(label, provider)| {
                let mut input = owned(&rows, None, label, Some("registry"));
                input.provider_id = provider.to_owned();
                bytes(&input)
            },
        );
        // A repeated source does not collapse, and the ledger digest is part of the declaration.
        let mut repeated = owned(&rows, None, "pooled", Some("registry"));
        repeated.contributions.push(row(1, 1.0, None));
        assert_ne!(bytes(&owned(&rows, None, "pooled", Some("registry"))), bytes(&repeated));
        let mut other_ledger = owned(&rows, None, "pooled", Some("registry"));
        other_ledger.ledger_state = digest(8);
        assert_ne!(bytes(&owned(&rows, None, "pooled", Some("registry"))), bytes(&other_ledger));
    }

    #[test]
    fn the_method_selects_the_synthesis_cell() {
        let ctx = ExecutionContext::for_tests(0);
        let ledger = EvidenceLedger::new();
        let coordinate = coordinate();
        let pool = TransportedSynthesisMethod::Pool { sampling_covariance: None };
        let pooled = Study::compile_transported_synthesis(
            &ledger,
            "pooled",
            &[],
            &pool,
            None,
            coordinate.clone(),
            "synthesis",
            &ctx,
        )
        .unwrap();
        assert_eq!(pooled.lowered().route(), "SynthesisPooling");
        crate::planner::lowered::round_trip(pooled.lowered()).unwrap();
        let borrow = TransportedSynthesisMethod::Borrow {
            mean_prior: MeanPrior::Flat,
            tau_prior: TauPrior { support: vec![0.0], probabilities: vec![1.0] },
        };
        let borrowed = Study::compile_transported_synthesis(
            &ledger,
            "borrowed",
            &[],
            &borrow,
            None,
            coordinate.clone(),
            "synthesis",
            &ctx,
        )
        .unwrap();
        assert_eq!(borrowed.lowered().route(), "SynthesisBorrowing");
        crate::planner::lowered::round_trip(borrowed.lowered()).unwrap();
        let prior = TransportedSynthesisMethod::TransportedPrior { prior: source(3) };
        let updated = Study::compile_transported_synthesis(
            &ledger,
            "prior",
            &[],
            &prior,
            None,
            coordinate,
            "synthesis",
            &ctx,
        )
        .unwrap();
        assert_eq!(updated.lowered().route(), "SynthesisTransportedPrior");
        crate::planner::lowered::round_trip(updated.lowered()).unwrap();
    }

    fn rows(start: u64, end: u64) -> Vec<antecedent_contract::evidence_ledger::ObservationId> {
        (start..end)
            .map(|i| {
                antecedent_contract::evidence_ledger::ObservationId::from_key(
                    "registry",
                    &i.to_le_bytes(),
                )
            })
            .collect()
    }

    fn carried() -> antecedent_contract::evidence_ledger::TransportIdentity {
        antecedent_contract::evidence_ledger::TransportIdentity {
            source_populations: vec!["trial_a".to_owned()],
            target_population: "registry".to_owned(),
            selection_diagram: digest(1),
            assumptions: vec!["outcome mechanism invariant".to_owned()],
        }
    }

    /// The compiled cell and the kernel register the same synthesis, and a refusal replaces nothing.
    #[test]
    fn execute_matches_the_kernel_and_a_refusal_keeps_the_ledger() {
        use antecedent_contract::evidence_ledger::{Lineage, SourceDeclaration, SourceKind};
        let mut ledger = EvidenceLedger::new();
        let cohort = ledger
            .add_source(
                SourceDeclaration::new("registry cohort", SourceKind::ObservationSet)
                    .with_observations(rows(0, 100)),
            )
            .unwrap();
        let carried_source = ledger
            .add_source(
                SourceDeclaration::new("trial a claim", SourceKind::ExternalClaim)
                    .with_observations(rows(1_000, 1_100))
                    .with_lineage(Lineage::Original)
                    .with_transport(carried()),
            )
            .unwrap();
        let rows = vec![row_of(cohort, 1.0, None), row_of(carried_source, 2.0, Some(0.5))];
        let method = TransportedSynthesisMethod::Pool { sampling_covariance: None };
        let coordinate = coordinate();
        let ctx = ExecutionContext::for_tests(0);
        let compiled = Study::compile_transported_synthesis(
            &ledger,
            "pooled effect",
            &rows,
            &method,
            Some("registry"),
            coordinate.clone(),
            "synthesis",
            &ctx,
        )
        .unwrap();
        let mut by_kernel = ledger.clone();
        let mut by_cell = ledger.clone();
        let kernel = transported_synthesis_response_kernel(
            &mut by_kernel,
            "pooled effect",
            &rows,
            &method,
            Some("registry"),
            coordinate.clone(),
            "synthesis",
        )
        .unwrap();
        let cell = compiled.execute(&mut by_cell, &ctx).unwrap();
        assert_eq!(kernel, cell);
        assert_eq!(by_kernel.state_digest(), by_cell.state_digest());
        assert!((cell.evaluation.estimate - 1.4).abs() < 1e-12);
        assert!((cell.evaluation.variance - 0.6).abs() < 1e-12);

        let before = ledger.state_digest();
        let mut wrong = coordinate;
        wrong.population_id = "trial_a".into();
        let mut kernel_ledger = ledger.clone();
        let kernel_refusal = transported_synthesis_response_kernel(
            &mut kernel_ledger,
            "pooled effect",
            &rows,
            &method,
            Some("registry"),
            wrong.clone(),
            "synthesis",
        )
        .unwrap_err();
        let refused = Study::compile_transported_synthesis(
            &ledger,
            "pooled effect",
            &rows,
            &method,
            Some("registry"),
            wrong,
            "synthesis",
            &ctx,
        )
        .unwrap();
        let cell_refusal = refused.execute(&mut ledger, &ctx).unwrap_err();
        assert_eq!(kernel_refusal, cell_refusal);
        assert_eq!(kernel_ledger.state_digest(), before);
        assert_eq!(ledger.state_digest(), before);
    }

    fn row_of(
        source: antecedent_contract::evidence_ledger::SourceId,
        estimate: f64,
        transport: Option<f64>,
    ) -> TransportedContribution {
        TransportedContribution {
            source,
            estimate,
            sampling_variance: 1.0,
            transport_variance: transport,
        }
    }
}
