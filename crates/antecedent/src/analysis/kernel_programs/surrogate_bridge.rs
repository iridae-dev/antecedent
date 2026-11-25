//! A transported surrogate bridge as a compiled program cell.
//!
//! `surrogate_bridge_response` derives the bridged claim and builds the response. Compiling binds
//! the bridge, the claim fields the derivation reads, and the provider. Executing passes the
//! `transport.surrogate_bridge_premise` gate and calls [`surrogate_bridge_response_kernel`] with
//! those inputs.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::bridge::BridgedClaim;
use antecedent_contract::claim::ClaimEnvelope;
use antecedent_contract::obligation::ObligationRecord;
use antecedent_contract::{
    DataClassification, ExternalRefusal, ObligationKind, ObligationScope, SlotAvailability,
};
use antecedent_core::{
    Assumption, AssumptionSource, AssumptionStatus, CanonicalEncode, CanonicalWriter,
    ContractIdentities, ExecutionContext,
};
use antecedent_design::transported_bridge::TransportedSurrogateBridge;

use super::{KernelProgram, KernelProgramShape, KernelRealization};
use crate::analysis::Study;
use crate::analysis::supplied_identity::{
    Canonical, Declaration, SetOf, encoded, unencodable, variant,
};
use crate::analysis::transported_constructors::{
    TransportedConstruction, surrogate_bridge_response_kernel, transport_support_refusal,
};
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

const TAG: &str = "antecedent.kernel_program.surrogate_bridge";

const SHAPE: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::SurrogateBridge,
    plan_id: "transport.surrogate_bridge_premise",
    functional: FunctionalKind::PropagatedChain,
    evaluator: "transported_surrogate_bridge",
    realization: KernelRealization::Exact,
    uncertainty: UncertaintyReading::None,
    role: EvidenceRole::TransportSource,
    graph_class: "bridge_selection_diagram",
    data: DataClassification::MultiEnvironment,
    design: "trial-level surrogate relation carried under a declared selection diagram",
};

/// The bridge, the claim, and the provider the response names.
#[derive(Clone, Debug)]
struct OwnedBridge {
    bridge: TransportedSurrogateBridge,
    claim: ClaimEnvelope,
    provider_id: String,
}

fn is_alignment_evidence(item: &str) -> bool {
    // The same predicate `derive_bridged_claim` uses to drop alignment evidence.
    item.starts_with("alignment:")
        || ["joint_draws", "covariance", "replicate_ids"]
            .iter()
            .any(|kind| item == *kind || item.starts_with(&format!("{kind}:")))
}

fn encode_bridge(bridge: &TransportedSurrogateBridge, out: &mut CanonicalWriter) {
    bridge.surrogate.encode(out);
    bridge.terminal.encode(out);
    // The contract identity sorts pairs by their bits and keeps a repeated pair. The fit's
    // sums do not depend on that order.
    let mut pairs: Vec<(u64, u64)> = bridge
        .validation
        .trials
        .iter()
        .map(|trial| (trial.surrogate_effect.to_bits(), trial.terminal_effect.to_bits()))
        .collect();
    pairs.sort_unstable();
    out.word(pairs.len() as u64);
    for (surrogate, terminal) in pairs {
        surrogate.encode(out);
        terminal.encode(out);
    }
    bridge.validation.min_trials.encode(out);
    bridge.validation.min_r_squared.encode(out);
    bridge.validated_in.population_id.encode(out);
    bridge.validated_in.regime_id.encode(out);
    bridge.validation_evidence.encode(out);
    bridge.diagram.source_population.encode(out);
    bridge.diagram.target_population.encode(out);
    // `identity_hex` sorts and drops a repeated selection target.
    SetOf(&bridge.diagram.selection_targets).encode(out);
    bridge.terminal_variable.encode(out);
    // Declaration order: the first contradicted modifier is the one the refusal names, and the
    // premise statement counts every declared modifier.
    bridge.relation_modifiers.encode(out);
}

impl Canonical for ObligationScope {
    fn encode(&self, out: &mut CanonicalWriter) {
        match *self {
            Self::Program => {
                variant(out, "program");
            }
            Self::Factor => {
                variant(out, "factor");
            }
            Self::Atom => {
                variant(out, "atom");
            }
            Self::Horizon { horizon } => {
                variant(out, "horizon");
                horizon.encode(out);
            }
            _ => unencodable(out, "obligation scope"),
        }
    }
}

impl Canonical for ObligationKind {
    fn encode(&self, out: &mut CanonicalWriter) {
        match *self {
            Self::UserAssertion => {
                variant(out, "user_assertion");
            }
            Self::GraphicalImplication => {
                variant(out, "graphical_implication");
            }
            Self::EmpiricalDiagnostic => {
                variant(out, "empirical_diagnostic");
            }
            Self::FailedCheck => {
                variant(out, "failed_check");
            }
            Self::Uncheckable => {
                variant(out, "uncheckable");
            }
            Self::CheckNotRun => {
                variant(out, "check_not_run");
            }
            _ => unencodable(out, "obligation kind"),
        }
    }
}

impl Canonical for AssumptionSource {
    fn encode(&self, out: &mut CanonicalWriter) {
        match self {
            Self::UserDeclared => {
                variant(out, "user_declared");
            }
            Self::AlgorithmDefault { algorithm } => {
                variant(out, "algorithm_default");
                algorithm.encode(out);
            }
            Self::Artifact => {
                variant(out, "artifact");
            }
            Self::Derived { from } => {
                variant(out, "derived");
                from.encode(out);
            }
        }
    }
}

impl Canonical for AssumptionStatus {
    fn encode(&self, out: &mut CanonicalWriter) {
        variant(out, self.as_str());
    }
}

impl Canonical for Assumption {
    fn encode(&self, out: &mut CanonicalWriter) {
        CanonicalEncode::encode(self, out);
    }
}

impl Canonical for ObligationRecord {
    fn encode(&self, out: &mut CanonicalWriter) {
        // Every field is copied onto the bridged claim's assumption slot.
        self.id.encode(out);
        self.scope.encode(out);
        self.source.encode(out);
        self.kind.encode(out);
        self.status.encode(out);
        self.required_check.encode(out);
        self.evidence.encode(out);
        self.assumption.encode(out);
        self.description.encode(out);
    }
}

fn encode_assumptions(claim: &ClaimEnvelope, out: &mut CanonicalWriter) {
    match &claim.reasoning.assumptions {
        SlotAvailability::Available(slot) => {
            variant(out, "available");
            // Encounter order, repeats kept: the bridged claim copies each unresolved record.
            // A resolved record is not copied.
            let unresolved: Vec<&ObligationRecord> =
                slot.obligations.iter().filter(|record| record.is_unresolved()).collect();
            unresolved.encode(out);
        }
        // The reason is not read: every unavailable slot stands for one fixed obligation.
        SlotAvailability::Unavailable { .. } => {
            variant(out, "unavailable");
        }
        _ => unencodable(out, "assumption slot"),
    }
}

fn encode_copied_identities(identities: &ContractIdentities, out: &mut CanonicalWriter) {
    // The bridge replaces the target and drops the program. The other layers are the source's.
    identities.identification.encode(out);
    identities.identification_product.encode(out);
    identities.inference_binding.encode(out);
    identities.observation.encode(out);
    identities.data_snapshot.encode(out);
}

fn encode_claim(claim: &ClaimEnvelope, out: &mut CanonicalWriter) {
    claim.claim_id.encode(out);
    encode_copied_identities(&claim.identities, out);
    claim.kind.as_str().encode(out);
    claim.value.encode(out);
    claim.outcome_units.encode(out);
    claim.execution.encode(out);
    let retained: Vec<&str> = claim
        .evidence
        .iter()
        .map(Arc::as_ref)
        .filter(|item| !is_alignment_evidence(item))
        .collect();
    out.word(retained.len() as u64);
    for item in retained {
        item.encode(out);
    }
    encode_assumptions(claim, out);
}

impl Canonical for OwnedBridge {
    fn encode(&self, out: &mut CanonicalWriter) {
        encode_bridge(&self.bridge, out);
        encode_claim(&self.claim, out);
        self.provider_id.encode(out);
    }
}

fn bridge_bytes(input: &OwnedBridge) -> Result<[u8; 32], CausalError> {
    encoded(SHAPE.plan_id, || {
        let mut declaration = Declaration::new(TAG);
        declaration.field("request", input);
        declaration.finish()
    })
}

/// A compiled surrogate bridge, bound to `transport.surrogate_bridge_premise`.
#[derive(Clone, Debug)]
pub struct CompiledSurrogateBridge {
    program: KernelProgram<OwnedBridge>,
}

impl CompiledSurrogateBridge {
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

    /// Pass the support gate, then call `surrogate_bridge_response_kernel` with the bound bridge,
    /// claim and provider.
    ///
    /// # Errors
    /// The support refusal, and every refusal of `surrogate_bridge_response_kernel`.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<TransportedConstruction<BridgedClaim>, ExternalRefusal> {
        self.program.gate(ctx).map_err(|refusal| transport_support_refusal(&refusal))?;
        let input = self.program.input();
        surrogate_bridge_response_kernel(&input.bridge, &input.claim, &input.provider_id)
    }
}

impl Study {
    /// Compile `surrogate_bridge_response` for this bridge and claim. Compiling executes nothing.
    ///
    /// # Errors
    /// Cancellation, compiling from inside a lowered dispatch, or a declaration with no encoding.
    pub fn compile_surrogate_bridge(
        bridge: &TransportedSurrogateBridge,
        claim: &ClaimEnvelope,
        provider_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<CompiledSurrogateBridge, CausalError> {
        let owned = OwnedBridge {
            bridge: bridge.clone(),
            claim: claim.clone(),
            provider_id: provider_id.to_owned(),
        };
        let bytes = bridge_bytes(&owned)?;
        let program = KernelProgram::compile(&SHAPE, owned, &bytes, ctx)?;
        Ok(CompiledSurrogateBridge { program })
    }
}

#[cfg(test)]
mod site_tests {
    use std::sync::Arc;

    use super::*;
    use crate::identity_laws::{bit_exact, multiset_laws, ordered_laws, set_laws};
    use antecedent_contract::bridge::{BridgeScope, TrialEffectPair, TrialLevelValidation};
    use antecedent_contract::reasoning::AssumptionSlot;
    use antecedent_contract::{
        ClaimDomains, ClaimKind, DomainStatus, ObligationKind, ObligationRecord, ObligationScope,
        QuantityRole, ReasoningView, ScientificQuantity,
    };
    use antecedent_core::{
        AssumptionSource, AssumptionStatus, ContractIdentities, ExecutionContext, SemanticDigest,
    };
    use antecedent_design::transported_bridge::BridgeSelectionDiagram;

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::from_bytes([byte; 32])
    }

    fn quantity(variable: &str, population: &str) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: variable.into(),
            variable_name: variable.into(),
            role: QuantityRole::Outcome,
            units: "units".into(),
            population_id: population.into(),
            regime_id: "do(treatment=1)".into(),
            horizon: 0,
            functional_id: "average_effect".into(),
            conditioning: Vec::new(),
            transform_id: "identity".into(),
        }
    }

    fn claim_with(
        evidence: &[String],
        assumptions: SlotAvailability<AssumptionSlot>,
        target: SemanticDigest,
    ) -> ClaimEnvelope {
        ClaimEnvelope::new(
            digest(1),
            ContractIdentities::new(
                target,
                digest(11),
                Some(digest(12)),
                Some(digest(13)),
                digest(14),
                digest(15),
                digest(16),
            ),
            ClaimKind::Point,
            Some(0.8),
            Some(Arc::from("units")),
            ReasoningView::new(
                SlotAvailability::unavailable("fixture"),
                SlotAvailability::unavailable("fixture"),
                SlotAvailability::unavailable("fixture"),
                assumptions,
            ),
            ClaimDomains::new(
                DomainStatus::Identified,
                DomainStatus::Supported,
                DomainStatus::Evaluated,
            ),
            None,
            evidence.iter().map(|item| Arc::<str>::from(item.as_str())).collect::<Vec<_>>(),
        )
    }

    fn bridge_of(
        trials: &[TrialEffectPair],
        targets: &[String],
        modifiers: &[String],
    ) -> TransportedSurrogateBridge {
        TransportedSurrogateBridge {
            surrogate: quantity("trial:biomarker", "children"),
            terminal: quantity("trial:survival", "children"),
            validation: TrialLevelValidation {
                trials: trials.to_vec(),
                min_trials: 4,
                min_r_squared: 0.9,
            },
            validated_in: BridgeScope {
                population_id: "adults".into(),
                regime_id: "do(treatment=1)".into(),
            },
            validation_evidence: "meta-analysis".into(),
            diagram: BridgeSelectionDiagram {
                source_population: "adults".into(),
                target_population: "children".into(),
                selection_targets: targets.to_vec(),
            },
            terminal_variable: "survival".into(),
            relation_modifiers: modifiers.to_vec(),
        }
    }

    fn owned(
        trials: &[TrialEffectPair],
        targets: &[String],
        modifiers: &[String],
        claim: ClaimEnvelope,
    ) -> OwnedBridge {
        OwnedBridge {
            bridge: bridge_of(trials, targets, modifiers),
            claim,
            provider_id: "bridge".to_owned(),
        }
    }

    fn bytes(input: &OwnedBridge) -> Option<[u8; 32]> {
        bridge_bytes(input).ok()
    }

    fn pair(surrogate_effect: f64, terminal_effect: f64) -> TrialEffectPair {
        TrialEffectPair { surrogate_effect, terminal_effect }
    }

    fn plain() -> ClaimEnvelope {
        claim_with(&["prior:p0".to_owned()], SlotAvailability::unavailable("fixture"), digest(10))
    }

    /// Trials are the multiset the fit sorts by its total order. Selection targets are the set
    /// `identity_hex` sorts and deduplicates. Modifiers stay in declaration order. Evidence that
    /// the derivation keeps is ordered; an alignment string is not read, and neither is the
    /// source claim's target identity or the reason an assumption slot is unavailable.
    #[test]
    fn site_surrogate_bridge_trials_are_a_multiset_and_targets_are_a_set() {
        let trials = [pair(0.0, 0.1), pair(1.0, 0.9)];
        let targets = ["age".to_owned(), "site".to_owned()];
        let modifiers = ["age".to_owned(), "dose".to_owned()];
        multiset_laws(&trials, |items| bytes(&owned(items, &targets, &modifiers, plain())));
        set_laws(&targets, |items| bytes(&owned(&trials, items, &modifiers, plain())));
        ordered_laws(&modifiers, |items| bytes(&owned(&trials, &targets, items, plain())));
        ordered_laws(&["prior:p0".to_owned(), "study:s1".to_owned()], |evidence| {
            bytes(&owned(
                &trials,
                &targets,
                &modifiers,
                claim_with(evidence, SlotAvailability::unavailable("fixture"), digest(10)),
            ))
        });
        bit_exact(|effect| bytes(&owned(&[pair(effect, 0.9)], &targets, &modifiers, plain())));
        let with_alignment = claim_with(
            &["prior:p0".to_owned(), "alignment:joint_draws:a".to_owned()],
            SlotAvailability::unavailable("fixture"),
            digest(10),
        );
        assert_eq!(
            bytes(&owned(&trials, &targets, &modifiers, plain())),
            bytes(&owned(&trials, &targets, &modifiers, with_alignment))
        );
        let other_target = claim_with(
            &["prior:p0".to_owned()],
            SlotAvailability::unavailable("fixture"),
            digest(99),
        );
        assert_eq!(
            bytes(&owned(&trials, &targets, &modifiers, plain())),
            bytes(&owned(&trials, &targets, &modifiers, other_target))
        );
        let other_reason = claim_with(
            &["prior:p0".to_owned()],
            SlotAvailability::unavailable("other"),
            digest(10),
        );
        assert_eq!(
            bytes(&owned(&trials, &targets, &modifiers, plain())),
            bytes(&owned(&trials, &targets, &modifiers, other_reason))
        );
    }

    #[test]
    fn site_unresolved_obligations_stay_in_encounter_order() {
        let obligation = |id: &str| {
            ObligationRecord::new(
                id,
                ObligationScope::Program,
                AssumptionSource::UserDeclared,
                ObligationKind::CheckNotRun,
                AssumptionStatus::Declared,
                "unresolved",
            )
        };
        let trials = [pair(0.0, 0.1), pair(1.0, 0.9)];
        let slot = |ids: &[&str]| {
            SlotAvailability::Available(AssumptionSlot::new(
                ids.iter().copied().map(obligation).collect::<Vec<_>>(),
            ))
        };
        // The bridged claim copies each unresolved record, so a repeat is another record.
        ordered_laws(&["a", "b"], |ids| {
            bytes(&owned(&trials, &[], &[], claim_with(&[], slot(ids), digest(10))))
        });
        let once = bytes(&owned(&trials, &[], &[], claim_with(&[], slot(&["a", "b"]), digest(10))));
        let repeated =
            bytes(&owned(&trials, &[], &[], claim_with(&[], slot(&["a", "b", "a"]), digest(10))));
        assert_ne!(once, repeated);
        let resolved = ObligationRecord::new(
            "done",
            ObligationScope::Program,
            AssumptionSource::UserDeclared,
            ObligationKind::EmpiricalDiagnostic,
            AssumptionStatus::Supported,
            "resolved",
        );
        let with_resolved = SlotAvailability::Available(AssumptionSlot::new(vec![
            obligation("a"),
            resolved,
            obligation("b"),
        ]));
        assert_eq!(
            once,
            bytes(&owned(&trials, &[], &[], claim_with(&[], with_resolved, digest(10))))
        );
    }

    #[test]
    fn the_bridge_compiles_to_the_surrogate_cell() {
        let ctx = ExecutionContext::for_tests(0);
        let bridge = bridge_of(&[pair(0.0, 0.1)], &[], &[]);
        let compiled = Study::compile_surrogate_bridge(&bridge, &plain(), "bridge", &ctx).unwrap();
        assert_eq!(compiled.lowered().route(), "SurrogateBridge");
        crate::planner::lowered::round_trip(compiled.lowered()).unwrap();
    }

    /// The compiled cell and the kernel derive the same bridged claim.
    #[test]
    fn execute_matches_the_kernel() {
        let mut surrogate = quantity("trial:biomarker", "children");
        surrogate.units = "mmol_per_l".into();
        let mut terminal = quantity("trial:survival", "children");
        terminal.units = "log_hazard".into();
        let bridge = TransportedSurrogateBridge {
            surrogate,
            terminal,
            validation: TrialLevelValidation {
                trials: [(0.0, 0.1), (0.5, 0.5), (1.0, 0.9), (1.5, 1.5)]
                    .into_iter()
                    .map(|(surrogate_effect, terminal_effect)| TrialEffectPair {
                        surrogate_effect,
                        terminal_effect,
                    })
                    .collect(),
                min_trials: 4,
                min_r_squared: 0.9,
            },
            validated_in: BridgeScope {
                population_id: "adults".into(),
                regime_id: "do(treatment=1)".into(),
            },
            validation_evidence: "meta-analysis:ckd-2024".into(),
            diagram: BridgeSelectionDiagram {
                source_population: "adults".into(),
                target_population: "children".into(),
                selection_targets: vec!["biomarker".into()],
            },
            terminal_variable: "survival".into(),
            relation_modifiers: vec!["age".into()],
        };
        let claim = claim_with(
            &["prior:p0".to_owned()],
            SlotAvailability::unavailable("fixture"),
            digest(10),
        );
        let mut source = claim;
        source.value = Some(0.8);
        source.outcome_units = Some(Arc::from("mmol_per_l"));
        let ctx = ExecutionContext::for_tests(0);
        let compiled = Study::compile_surrogate_bridge(&bridge, &source, "bridge", &ctx).unwrap();
        let kernel = surrogate_bridge_response_kernel(&bridge, &source, "bridge").unwrap();
        let cell = compiled.execute(&ctx).unwrap();
        assert_eq!(kernel, cell);
        assert!((cell.evaluation.value - 0.796).abs() < 1e-12);
    }
}
