//! The ordered-response observation recovery as a compiled program family.
//!
//! A checked ordered-response recovery plan (`RecoveryChain::decide`, a
//! `ChainRecoveryDecision::Recovered` plan) is evaluated on an exact nine-cell observed pattern
//! law by `antecedent_estimate::evaluate_chain_recovery`, the call `RecoveryChain::recover`
//! makes. Compiling the plan and the law binds their digest into a six-layer program under the
//! `ChainObservationRecovery` route, bound to the `recovery.chain_observation` unit. Evaluating
//! passes the support gate of that unit and then calls `evaluate_chain_recovery` with the same
//! arguments, so the recovered law is the direct path's bit for bit.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::DataClassification;
use antecedent_core::{CanonicalWriter, ExecutionContext, NotSupported};
use antecedent_estimate::{ChainPatternLaw, ChainRecoveredLaw, evaluate_chain_recovery};
use antecedent_identify::{ChainRecoveryDetail, ChainRecoveryError, ChainRecoveryPlan};

use super::{KernelProgram, KernelProgramShape, KernelRealization};
use crate::analysis::Study;
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

const CHAIN_RECOVERY: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::ChainObservationRecovery,
    plan_id: "recovery.chain_observation",
    functional: FunctionalKind::Association,
    evaluator: "ordered_response_recovery_formula",
    realization: KernelRealization::Exact,
    uncertainty: UncertaintyReading::None,
    role: EvidenceRole::Observations,
    graph_class: "binary_m_graph_response_chain",
    data: DataClassification::Tabular,
    design: "exact ordered-response recovery formula on a nine-cell observed pattern law",
};

/// A checked plan and the exact observed pattern law it is evaluated on.
#[derive(Clone, Debug)]
struct ChainRecoveryRequest {
    plan: ChainRecoveryPlan,
    observed: ChainPatternLaw,
}

/// Canonical bytes of the plan (its roles, head, tail shape, rule, premises, formula and
/// charged operations) and of every observed cell.
fn request_bytes(request: &ChainRecoveryRequest) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.kernel_program.chain_observation_recovery");
    let plan = &request.plan;
    for part in [plan.query.first, plan.query.second] {
        for id in [part.variable, part.response, part.proxy] {
            writer.word(u64::from(id.raw()));
        }
    }
    writer
        .word(plan.head as u64)
        .word(u64::from(plan.tail_depends_on_head_variable))
        .text(plan.rule_version)
        .word(plan.premises.len() as u64);
    for premise in &plan.premises {
        writer.text(premise);
    }
    writer.text(&plan.formula).word(plan.operations_consumed as u64);
    let observed = &request.observed;
    for row in observed.both() {
        writer.reals(row);
    }
    writer.reals(observed.only_first()).reals(observed.only_second()).real(observed.neither());
    writer.finish()
}

/// The support refusal as a chain-recovery refusal, carried through the kernel's
/// `route_not_supported` detail.
fn chain_refusal(refusal: &NotSupported) -> ChainRecoveryError {
    ChainRecoveryError::new(ChainRecoveryDetail::UnsupportedMechanism, refusal.message())
}

/// A compiled ordered-response recovery: the checked plan, the observed law and the program
/// that binds them.
#[derive(Clone, Debug)]
pub struct CompiledChainObservationRecovery {
    program: KernelProgram<ChainRecoveryRequest>,
}

impl Study {
    /// Compile the evaluation of a checked ordered-response recovery plan on an exact observed
    /// pattern law (the inputs of `antecedent_estimate::evaluate_chain_recovery`) under the
    /// `ChainObservationRecovery` route. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_chain_observation_recovery(
        plan: ChainRecoveryPlan,
        observed: ChainPatternLaw,
        ctx: &ExecutionContext,
    ) -> Result<CompiledChainObservationRecovery, CausalError> {
        let request = ChainRecoveryRequest { plan, observed };
        let canonical = request_bytes(&request);
        let program = KernelProgram::compile(&CHAIN_RECOVERY, request, &canonical, ctx)?;
        Ok(CompiledChainObservationRecovery { program })
    }
}

impl CompiledChainObservationRecovery {
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

    /// The checked plan the program binds.
    #[must_use]
    pub fn plan(&self) -> &ChainRecoveryPlan {
        &self.program.input().plan
    }

    /// The observed pattern law the program binds.
    #[must_use]
    pub fn observed(&self) -> &ChainPatternLaw {
        &self.program.input().observed
    }

    /// Pass the support gate, then evaluate the recovery formula as `evaluate_chain_recovery`
    /// does.
    ///
    /// # Errors
    ///
    /// The support refusal (reason `not_supported` in the message, detail
    /// `recovery_chain.unsupported_mechanism`) in a release build when the unit is not
    /// supported, and every refusal of `evaluate_chain_recovery`.
    pub fn evaluate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<ChainRecoveredLaw, ChainRecoveryError> {
        self.program.gate(ctx).map_err(|refusal| chain_refusal(&refusal))?;
        let request = self.program.input();
        evaluate_chain_recovery(&request.plan, &request.observed)
    }
}

#[cfg(test)]
mod site_tests {
    use antecedent_core::VariableId;
    use antecedent_graph::{Admg, DenseNodeId};
    use antecedent_identify::{
        ChainPartial, ChainRecoveryDecision, ChainRecoveryQuery, decide_chain_recovery,
    };

    use super::*;
    use crate::identity_laws::{ADVERSARIAL_TEXT, bit_exact, injective, ordered_laws};

    /// The checked plan of the response chain `R1 -> R2` with `X1 -> X2`.
    fn plan() -> ChainRecoveryPlan {
        let mut graph = Admg::with_variables(6);
        for (from, to) in [(0, 1), (0, 4), (2, 4), (1, 5), (3, 5), (2, 3)] {
            graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        let v = VariableId::from_raw;
        let query = ChainRecoveryQuery {
            first: ChainPartial { variable: v(0), response: v(2), proxy: v(4) },
            second: ChainPartial { variable: v(1), response: v(3), proxy: v(5) },
        };
        let ctx = ExecutionContext::for_tests(0);
        let ChainRecoveryDecision::Recovered(plan) =
            decide_chain_recovery(&graph, &query, &ctx).unwrap()
        else {
            panic!("expected a recovery plan");
        };
        *plan
    }

    /// An exact observed law whose first cell is `first`, or `None` when it is refused.
    fn law(first: f64) -> Option<ChainPatternLaw> {
        ChainPatternLaw::new([[first, 0.1], [0.1, 0.1]], [0.1, 0.1], [0.1, 0.2], 0.2).ok()
    }

    fn digest(plan: &ChainRecoveryPlan, observed: ChainPatternLaw) -> [u8; 32] {
        request_bytes(&ChainRecoveryRequest { plan: plan.clone(), observed })
    }

    /// `request_bytes`: the plan's premises are ordered (the identifier states them in order and
    /// seals the plan with them) and framed; the observed cells enter by their bits.
    #[test]
    fn site_chain_recovery_request_premises_are_ordered_and_framed() {
        let base = plan();
        let observed = law(0.0).unwrap();
        let with = |premises: &[&str]| {
            let mut plan = base.clone();
            plan.premises = premises.iter().map(|premise| (*premise).to_owned()).collect();
            Some(digest(&plan, observed))
        };
        ordered_laws(&["p1", "p2", "p3"], &with);
        injective(&ADVERSARIAL_TEXT, |parts| *parts, |&(a, b)| with(&[a, b]));
        bit_exact(|first| law(first).map(|observed| digest(&base, observed)));
        let mut flipped = base.clone();
        flipped.tail_depends_on_head_variable = !flipped.tail_depends_on_head_variable;
        assert_ne!(digest(&flipped, observed), digest(&base, observed));
    }
}
