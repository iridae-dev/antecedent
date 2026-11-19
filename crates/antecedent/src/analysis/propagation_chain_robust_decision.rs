//! Compiler-routed dominance decision over an assumption envelope.
//!
//! Each chain row fixes its sampling or posterior draw and encloses every
//! admissible interval assumption. Utilities and chance constraints are
//! evaluated by interval arithmetic. Averaging row bounds encloses expected
//! utility under every fixed admissible assumption. A winner is reported only
//! if its lower bound exceeds every possibly admissible competitor's upper
//! bound. This is conservative and carries no measured sampling coverage.

use std::sync::Arc;

use antecedent_contract::ScientificQuantity;
use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest};
use antecedent_design::decision_artifact::DecisionContractArtifact;
use antecedent_design::decision_contract::{
    DecisionContract, DecisionCriterion, UtilityExpr,
};
use antecedent_io::propagation_chain_record::PropagationChainRecordWire;
use serde::{Deserialize, Serialize};

use super::lowering::chain::lower_chain_with_decision;
use super::propagation_chain::{
    ChainReading, PropagationChain, PropagationChainPlan, compile_propagation_chain,
};
use super::propagation_chain_decision::ChainDecisionError;
use crate::planner::lowered::{CarrierDemand, DecisionProgram, LoweredProgram};
use crate::planner::{PlanStage, PlanView};

const ARTIFACT_FEATURE: &str = "chain_robust_decision_v1";
/// Maximum encoded bytes before the independent consumer decodes an artifact.
pub const MAX_CHAIN_ROBUST_DECISION_BYTES: usize = 80 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
struct Range {
    lower: f64,
    upper: f64,
}

impl Range {
    fn new(lower: f64, upper: f64) -> Result<Self, ChainDecisionError> {
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(ChainDecisionError::Coordinate("a utility interval is nonfinite or reversed"));
        }
        Ok(Self { lower, upper })
    }

    fn eval(expr: &UtilityExpr, inputs: &[Self]) -> Result<Self, ChainDecisionError> {
        let result = match expr {
            UtilityExpr::Const(value) => Self { lower: *value, upper: *value },
            UtilityExpr::Input(index) => *inputs.get(*index).ok_or(
                ChainDecisionError::Coordinate("utility input outside the declared action"),
            )?,
            UtilityExpr::Neg(a) => {
                let a = Self::eval(a, inputs)?;
                Self { lower: -a.upper, upper: -a.lower }
            }
            UtilityExpr::Add(a, b) => {
                let (a, b) = (Self::eval(a, inputs)?, Self::eval(b, inputs)?);
                Self { lower: a.lower + b.lower, upper: a.upper + b.upper }
            }
            UtilityExpr::Sub(a, b) => {
                let (a, b) = (Self::eval(a, inputs)?, Self::eval(b, inputs)?);
                Self { lower: a.lower - b.upper, upper: a.upper - b.lower }
            }
            UtilityExpr::Mul(a, b) => {
                let (a, b) = (Self::eval(a, inputs)?, Self::eval(b, inputs)?);
                let values = [
                    a.lower * b.lower, a.lower * b.upper,
                    a.upper * b.lower, a.upper * b.upper,
                ];
                Self {
                    lower: values.iter().copied().fold(f64::INFINITY, f64::min),
                    upper: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                }
            }
            UtilityExpr::Min(a, b) => {
                let (a, b) = (Self::eval(a, inputs)?, Self::eval(b, inputs)?);
                Self { lower: a.lower.min(b.lower), upper: a.upper.min(b.upper) }
            }
            UtilityExpr::Max(a, b) => {
                let (a, b) = (Self::eval(a, inputs)?, Self::eval(b, inputs)?);
                Self { lower: a.lower.max(b.lower), upper: a.upper.max(b.upper) }
            }
        };
        Self::new(result.lower, result.upper)
    }
}

/// Probability bounds for one hard constraint under all admissible assumptions.
#[derive(Clone, Debug, PartialEq)]
pub struct RobustConstraintBound {
    /// Constraint identity.
    pub id: String,
    /// Probability of rows satisfying the constraint for every assumption.
    pub probability_lower: f64,
    /// Probability of rows satisfying it for at least one assumption.
    pub probability_upper: f64,
    /// Minimum required probability.
    pub required: f64,
}

/// Conservative utility and feasibility bounds for one action.
#[derive(Clone, Debug, PartialEq)]
pub struct RobustActionBound {
    /// Action identity.
    pub id: String,
    /// Lower bound on expected utility under every fixed assumption.
    pub utility_lower: f64,
    /// Upper bound on expected utility under every fixed assumption.
    pub utility_upper: f64,
    /// Hard-constraint bounds, in contract order.
    pub constraints: Vec<RobustConstraintBound>,
    /// Every hard constraint is certified for every assumption.
    pub certainly_admissible: bool,
    /// At least one hard constraint fails under every assumption.
    pub certainly_excluded: bool,
}

/// A conservative decision conditional on the declared draw table and
/// assumption boxes, without a coverage or exact-minimax claim.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainRobustDecisionResult {
    /// The chain that supplied rowwise assumption enclosures.
    pub chain: PropagationChain,
    /// Identity of the decision contract.
    pub contract_identity: String,
    /// Bounds for every action.
    pub actions: Vec<RobustActionBound>,
    /// An action only when its lower utility exceeds every possibly admissible
    /// competitor's upper utility.
    pub dominant_action: Option<String>,
    /// Lowered six-layer program identity for this chain and contract.
    pub lowered_program_identity: SemanticDigest,
}

/// Compiler plan for a maximin decision on a draw-indexed assumption envelope.
#[derive(Clone, Debug)]
pub struct ChainRobustDecisionPlan {
    chain: PropagationChainPlan,
    contract: DecisionContract,
    lowered: Arc<LoweredProgram>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RobustArtifactWire {
    version: u32,
    required_feature: String,
    chain_record: Vec<u8>,
    chain_identity: String,
    contract: Vec<u8>,
    contract_identity: String,
    result_fingerprint: String,
    lowered_program_identity: [u8; 32],
}

/// Compile a conservative maximin decision over the chain's exact output
/// coordinate. The chain and contract occupy one lowered six-layer program.
///
/// # Errors
/// Any chain, coordinate, contract or carrier mismatch.
pub fn compile_chain_robust_decision(
    draft: PropagationChainRecordWire,
    contract: DecisionContract,
    ctx: &ExecutionContext,
) -> Result<ChainRobustDecisionPlan, ChainDecisionError> {
    let chain = compile_propagation_chain(draft, ctx)?;
    contract.validate()?;
    if chain.plan().reading != ChainReading::AssumptionEnvelope
        || !matches!(contract.criterion, DecisionCriterion::MaximinOverStructures)
    {
        return Err(ChainDecisionError::Coordinate(
            "a robust chain decision requires an assumption envelope and maximin-over-structures criterion",
        ));
    }
    let retained = chain.lowered().propagation_chain().ok_or(
        ChainDecisionError::Coordinate("the chain has no retained program"),
    )?;
    let quantity: ScientificQuantity = retained.draft().output_quantity.clone().ok_or(
        ChainDecisionError::Coordinate("the assumption envelope needs an output coordinate"),
    )?.try_into().map_err(|_| ChainDecisionError::Coordinate(
        "invalid assumption-envelope coordinate",
    ))?;
    for action in &contract.actions {
        if action.inputs.iter().any(|input| {
            input.require_same_coordinate(&quantity).is_err()
        }) {
            return Err(ChainDecisionError::Coordinate(
                "every robust action input must match the retained assumption-envelope coordinate",
            ));
        }
    }
    let chain_identity = retained.declaration_identity().to_owned();
    let decision = DecisionProgram {
        carrier: CarrierDemand::IdentifiedSet,
        utility: DecisionProgram::utility_form(None),
        actions: contract.actions.len(),
        constraints: contract.constraints.len(),
        chance_threshold: contract.constraints.iter()
            .map(|constraint| constraint.min_probability).fold(0.0_f64, f64::max),
        utility_unit: Some(Arc::from(contract.utility_units.as_str())),
        contract_identity: Some(Arc::from(contract.identity()?.as_str())),
        information_value: None,
        scenario_source: Some(Arc::from(chain_identity)),
    };
    let lowered = Arc::new(lower_chain_with_decision(
        retained.draft().clone(), chain.plan().clone(), Some(decision),
    )?);
    let chain = chain.within_lowered(Arc::clone(&lowered))?;
    Ok(ChainRobustDecisionPlan { chain, contract, lowered })
}

impl ChainRobustDecisionPlan {
    /// Every layer of the combined compiler program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }

    /// View the compiled plan without execution.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Replay the chain's rowwise assumption bounds and evaluate every action
    /// under the same compiled program identity.
    ///
    /// # Errors
    /// An invalidated output, a nonfinite utility enclosure, or cancellation.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<ChainRobustDecisionResult, ChainDecisionError> {
        let chain = self.chain.execute(ctx)?;
        let (lower, upper, stored_weights) = chain.output_envelope().ok_or(
            ChainDecisionError::Coordinate("the assumption envelope was withdrawn or is unavailable"),
        )?;
        if lower.len() != upper.len() || lower.is_empty() {
            return Err(ChainDecisionError::Coordinate("assumption-envelope row bounds differ"));
        }
        let total_weight = if let Some(bits) = stored_weights {
            if bits.len() != lower.len() {
                return Err(ChainDecisionError::Coordinate("assumption-envelope weights differ from rows"));
            }
            if bits.iter().any(|bits| {
                let weight = f64::from_bits(*bits);
                !weight.is_finite() || weight < 0.0
            }) {
                return Err(ChainDecisionError::Coordinate("invalid assumption-envelope weights"));
            }
            bits.iter().map(|bits| f64::from_bits(*bits)).sum::<f64>()
        } else {
            lower.len() as f64
        };
        if !total_weight.is_finite() || total_weight <= 0.0 {
            return Err(ChainDecisionError::Coordinate("invalid assumption-envelope weights"));
        }
        let mut actions = Vec::with_capacity(self.contract.actions.len());
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        for action in &self.contract.actions {
            let mut utility_lower = 0.0;
            let mut utility_upper = 0.0;
            let applicable = self.contract.constraints.iter().filter(|constraint| {
                constraint.applies_to.is_empty()
                    || constraint.applies_to.contains(&action.id)
            }).collect::<Vec<_>>();
            let mut probability = vec![(0.0_f64, 0.0_f64); applicable.len()];
            let mut every_row_certain = vec![true; applicable.len()];
            let mut definite_violation = vec![false; applicable.len()];
            let mut inputs = vec![Range { lower: 0.0, upper: 0.0 }; action.inputs.len()];
            for row in 0..lower.len() {
                if row % 1024 == 0 && scoped.cancellation.is_cancelled() {
                    return Err(ChainDecisionError::Coordinate("robust decision execution was cancelled"));
                }
                let box_value = Range::new(f64::from_bits(lower[row]), f64::from_bits(upper[row]))?;
                inputs.fill(box_value);
                let utility = Range::eval(&action.utility, &inputs)?;
                let weight = stored_weights.map_or(1.0, |bits| f64::from_bits(bits[row]))
                    / total_weight;
                utility_lower += weight * utility.lower;
                utility_upper += weight * utility.upper;
                for (index, constraint) in applicable.iter().enumerate() {
                    let range = Range::eval(&constraint.expr, &inputs)?;
                    if range.upper <= constraint.bound {
                        probability[index].0 += weight;
                    } else if weight > 0.0 {
                        every_row_certain[index] = false;
                    }
                    if range.lower <= constraint.bound {
                        probability[index].1 += weight;
                    } else if weight > 0.0 {
                        definite_violation[index] = true;
                    }
                }
            }
            let utility = Range::new(utility_lower, utility_upper)?;
            let constraints: Vec<RobustConstraintBound> = applicable.iter().zip(probability)
                .map(|(constraint, (lower, upper))| RobustConstraintBound {
                    id: constraint.id.clone(),
                    probability_lower: lower,
                    probability_upper: upper,
                    required: constraint.min_probability,
                }).collect();
            let certainly_admissible = constraints.iter().enumerate().all(|(index, bound)| {
                if bound.required == 1.0 {
                    every_row_certain[index]
                } else {
                    bound.probability_lower >= bound.required
                }
            });
            let certainly_excluded = constraints.iter().enumerate().any(|(index, bound)| {
                if bound.required == 1.0 {
                    definite_violation[index]
                } else {
                    bound.probability_upper < bound.required
                }
            });
            actions.push(RobustActionBound {
                id: action.id.clone(), utility_lower: utility.lower,
                utility_upper: utility.upper, constraints,
                certainly_admissible, certainly_excluded,
            });
        }
        let dominant_action = actions.iter().find(|candidate| {
            candidate.certainly_admissible && actions.iter().all(|other| {
                other.id == candidate.id || other.certainly_excluded
                    || candidate.utility_lower > other.utility_upper
            })
        }).map(|candidate| candidate.id.clone());
        Ok(ChainRobustDecisionResult {
            chain,
            contract_identity: self.contract.identity()?,
            actions,
            dominant_action,
            lowered_program_identity: self.lowered.identity(),
        })
    }

    /// Export the chain, contract and result fingerprint for independent replay.
    ///
    /// # Errors
    /// A foreign result or encoding refusal.
    pub fn export(&self, result: &ChainRobustDecisionResult, artifact_id: &str)
        -> Result<Vec<u8>, ChainDecisionError> {
        if result.lowered_program_identity != self.lowered.identity()
            || result.chain.lowered_program_identity() != self.lowered.identity()
            || result.contract_identity != self.contract.identity()?
        {
            return Err(ChainDecisionError::Coordinate("the robust result belongs to another compiler program"));
        }
        let contract = DecisionContractArtifact::new(self.contract.clone())?;
        let wire = RobustArtifactWire {
            version: 1,
            required_feature: ARTIFACT_FEATURE.into(),
            chain_record: result.chain.export()?,
            chain_identity: result.chain.identity().to_owned(),
            contract: contract.to_bytes(artifact_id)?,
            contract_identity: contract.identity().to_owned(),
            result_fingerprint: result_fingerprint(result),
            lowered_program_identity: *self.lowered.identity().as_bytes(),
        };
        let bytes = antecedent_io::to_cbor(&wire)?;
        if bytes.len() > MAX_CHAIN_ROBUST_DECISION_BYTES {
            return Err(ChainDecisionError::Coordinate("robust decision artifact exceeds its byte limit"));
        }
        Ok(bytes)
    }
}

fn result_fingerprint(result: &ChainRobustDecisionResult) -> String {
    let mut writer = CanonicalWriter::new("antecedent.chain_robust_decision_result.v1");
    writer.text(result.chain.identity())
        .bytes(result.lowered_program_identity.as_bytes())
        .text(&result.contract_identity)
        .opt_text(result.dominant_action.as_deref())
        .word(result.actions.len() as u64);
    for action in &result.actions {
        writer.text(&action.id)
            .word(action.utility_lower.to_bits())
            .word(action.utility_upper.to_bits())
            .word(u64::from(action.certainly_admissible))
            .word(u64::from(action.certainly_excluded))
            .word(action.constraints.len() as u64);
        for constraint in &action.constraints {
            writer.text(&constraint.id)
                .word(constraint.probability_lower.to_bits())
                .word(constraint.probability_upper.to_bits())
                .word(constraint.required.to_bits());
        }
    }
    writer.finish_hex()
}

/// Independently replay a robust chain decision under a retained program
/// identity. The record and contract are consumed before recompilation.
///
/// # Errors
/// A changed declaration, output enclosure, contract, program or result.
pub fn consume_chain_robust_decision(
    bytes: &[u8], expected_program: Option<SemanticDigest>, ctx: &ExecutionContext,
) -> Result<ChainRobustDecisionResult, ChainDecisionError> {
    if bytes.len() > MAX_CHAIN_ROBUST_DECISION_BYTES {
        return Err(ChainDecisionError::Coordinate("robust decision artifact exceeds its byte limit"));
    }
    let wire: RobustArtifactWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1 || wire.required_feature != ARTIFACT_FEATURE {
        return Err(ChainDecisionError::Coordinate("unsupported robust decision artifact version"));
    }
    let recorded = PropagationChainRecordWire::decode(&wire.chain_record)
        .map_err(|error| ChainDecisionError::Chain(error.to_refusal()))?;
    recorded.consume(Some(&wire.chain_identity), ctx)
        .map_err(|error| ChainDecisionError::Chain(error.to_refusal()))?;
    let contract = DecisionContractArtifact::from_bytes(&wire.contract, &wire.contract_identity)?;
    let plan = compile_chain_robust_decision(recorded, contract.contract().clone(), ctx)?;
    if wire.lowered_program_identity != *plan.lowered().identity().as_bytes()
        || expected_program.is_some_and(|expected| expected != plan.lowered().identity())
    {
        return Err(ChainDecisionError::Coordinate("robust decision program identity mismatch"));
    }
    let result = plan.execute(ctx)?;
    if result.chain.identity() != wire.chain_identity
        || result_fingerprint(&result) != wire.result_fingerprint
    {
        return Err(ChainDecisionError::Coordinate("robust decision replay differs from stored result"));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_contract::{QuantityRole, ScientificQuantity};
    use antecedent_contract::propagation_chain::{LawMeaning, StageKind, TableConstruction};
    use antecedent_design::decision_contract::{
        ActionKind, DecisionAction, HardConstraint, StructuralPolicy,
    };
    use antecedent_io::propagation_chain_record::{
        ChainOpWire, ChainSourceWire, ChainStageWire,
    };
    use antecedent_io::quantity_wire::ScientificQuantityWire;

    fn quantity() -> ScientificQuantity {
        ScientificQuantity {
            variable_id: "game:wishlists".into(),
            variable_name: "wishlists".into(),
            role: QuantityRole::Outcome,
            units: "count".into(),
            population_id: "game-a".into(),
            regime_id: "declared-traffic".into(),
            horizon: 7,
            functional_id: "mean".into(),
            conditioning: Vec::new(),
            transform_id: "identity".into(),
        }
    }

    fn draft() -> PropagationChainRecordWire {
        PropagationChain::draft(
            TableConstruction::JointDraws,
            ChainReading::AssumptionEnvelope,
            "wishlists",
            vec![
                ChainSourceWire::draws("traffic", LawMeaning::Sampling, "aligned", &[1.0, 2.0, 3.0]),
                ChainSourceWire::interval("algorithm", 0.0, 1.0),
            ],
            vec![ChainStageWire::new(
                "wishlists", StageKind::DerivedQuantity,
                &["traffic", "algorithm"], ChainOpWire::binary("sum"),
            )],
        ).with_output_quantity(ScientificQuantityWire::from(&quantity()))
    }

    fn contract() -> DecisionContract {
        DecisionContract {
            actions: vec![
                DecisionAction {
                    id: "publish".into(), kind: ActionKind::External,
                    inputs: vec![quantity()], utility: UtilityExpr::Input(0),
                },
                DecisionAction {
                    id: "wait".into(), kind: ActionKind::External,
                    inputs: vec![quantity()],
                    utility: UtilityExpr::difference(
                        UtilityExpr::Input(0), UtilityExpr::Const(10.0),
                    ),
                },
            ],
            utility_units: "score".into(),
            criterion: DecisionCriterion::MaximinOverStructures,
            constraints: vec![HardConstraint {
                id: "capacity".into(),
                expr: UtilityExpr::Input(0),
                bound: 5.0,
                min_probability: 1.0,
                units: "count".into(),
                applies_to: Vec::new(),
            }],
            target_population: "game-a".into(),
            horizon: 7,
            structural_policy: StructuralPolicy::RequireInvariantBestAction,
        }
    }

    #[test]
    fn robust_multi_action_dominance_compiles_and_replays() {
        let ctx = ExecutionContext::for_tests(19);
        let plan = compile_chain_robust_decision(draft(), contract(), &ctx).expect("compiler");
        assert_eq!(plan.lowered().evidence().decision().expect("decision").carrier(),
            CarrierDemand::IdentifiedSet);
        let result = plan.execute(&ctx).expect("robust decision");
        assert_eq!(result.dominant_action.as_deref(), Some("publish"));
        assert!(result.actions.iter().all(|action| action.certainly_admissible));
        let bytes = plan.export(&result, "robust-wishlists").expect("artifact");
        let replay = consume_chain_robust_decision(
            &bytes, Some(result.lowered_program_identity), &ctx,
        ).expect("independent replay");
        assert_eq!(replay, result);
    }

    #[test]
    fn repeated_input_in_nonlinear_utility_is_conservatively_enclosed() {
        let input = Range::new(-1.0, 2.0).expect("box");
        let square = UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(0));
        let bound = Range::eval(&square, &[input]).expect("interval arithmetic");
        assert!(bound.lower <= 0.0 && bound.upper >= 4.0);
    }
}
