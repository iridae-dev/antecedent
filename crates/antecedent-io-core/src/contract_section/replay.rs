//! Estimator-specific contract replay.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::consume::{require_nested_digest, require_payload_digest};

use antecedent_core::IdentityDomain;

use super::AnalysisResultContractWire;
use crate::contract_reason::ContractReason;
use crate::contract_wire::CheckedAipwLoweringWire;
use crate::identity::digest_wire;
use crate::query_wire::TargetPopulationWire;

mod aipw;
mod functional;
mod identity_links;
mod lowering;
mod moments;
pub(in crate::contract_section) use aipw::{
    is_aipw_contract, is_set_value, verify_checked_aipw, verify_checked_aipw_rows,
};
pub(in crate::contract_section) use identity_links::{
    verify_score_reuse_link, verify_target_weights_link,
};

pub(in crate::contract_section) use moments::{
    apply_moment_inverse, close_replay_value, invert_linear_moments, valid_linear_fit_kind,
    verify_linear_fit_moments,
};

pub(in crate::contract_section) use lowering::{
    intervention_bits, verify_checked_iv, verify_frontdoor_lowering,
};

pub(in crate::contract_section) use functional::{
    is_response_functional_effect_contract, is_scalar_functional_effect_contract,
    replay_functional_response_grid, verify_nested_counterfactual_result,
    verify_temporal_nested_point,
};

#[cfg(test)]
pub(in crate::contract_section) use functional::MAX_RESPONSE_GRID_POINTS;

/// Whether the contract advertises `estimator` or its program resolved to it.
///
/// Studies built without an explicit estimator leave `contract.estimator`
/// unset and commit the resolver's choice only in the program, so every
/// estimator-specific verification gate must consult both.
pub(in crate::contract_section) fn contract_uses_estimator(
    contract: &AnalysisResultContractWire,
    estimator: &str,
) -> bool {
    contract.estimator.as_deref() == Some(estimator)
        || contract.program.as_ref().is_some_and(|program| {
            program.commitments.resolved_estimator.as_deref() == Some(estimator)
        })
}

pub(in crate::contract_section) fn is_functional_effect_estimator_contract(
    contract: &AnalysisResultContractWire,
) -> bool {
    contract_uses_estimator(contract, "functional.effect")
}

pub(in crate::contract_section) fn is_functional_distribution_contract(
    contract: &AnalysisResultContractWire,
) -> bool {
    contract_uses_estimator(contract, "functional.distribution")
}

pub(in crate::contract_section) fn verify_checked_linear_adjustment(
    contract: &AnalysisResultContractWire,
    unresolved: &mut Vec<ContractReason>,
) {
    let lowering = contract
        .program
        .as_ref()
        .and_then(|program| program.checked_linear_adjustment_lowering.as_ref());
    // The checked linear-adjustment lowering currently covers the supplied
    // or accepted DAG ATE. Other graph classes may select the same numerical
    // estimator inside an envelope, but they do not have this lowering yet.
    let is_linear = contract.graph_class == "Dag"
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, crate::CausalQueryWire::AverageEffect { .. })
        && (contract.estimator.as_deref() == Some("linear.adjustment.ate")
            || contract.program.as_ref().is_some_and(|program| {
                program.commitments.resolved_estimator.as_deref() == Some("linear.adjustment.ate")
            }));
    if is_linear && lowering.is_none() {
        unresolved.push(ContractReason::ProgramCheckedLinearAdjustmentLowering);
        return;
    }
    let Some(lowering) = lowering else { return };
    if !verify_linear_adjustment_lowering(contract, lowering) {
        unresolved.push(ContractReason::ProgramCheckedLinearAdjustmentBinding);
    }
}

/// Whether the carried adjustment set is a sufficient backdoor set for the carried graph.
///
/// A resealed product can repeat any adjustment set, and the lowerings only compare their
/// adjustment with that product. For a static DAG this re-checks the set against the graph
/// the identification itself carries (no treatment descendant, treatment and outcome
/// d-separated once the arrows out of treatment are removed), with the identifier's own
/// criterion and no search budget. Other graph classes establish adjustment by other
/// criteria, which this check does not recompute, so they are not refused for it.
pub(in crate::contract_section) fn carried_adjustment_is_backdoor(
    contract: &AnalysisResultContractWire,
    treatment: u32,
    outcome: u32,
    adjustment: &[u32],
) -> bool {
    let Some(identification) = contract.identification.as_ref() else { return true };
    let crate::identity::GraphIdentityWire::Dag(graph) = &identification.graph else {
        return true;
    };
    let Ok(dag) = crate::dag_from_wire(graph) else { return false };
    let adjustment =
        adjustment.iter().copied().map(antecedent_core::VariableId::from_raw).collect::<Vec<_>>();
    matches!(
        antecedent_identify::backdoor::check_joint_adjustment(
            &dag,
            &[antecedent_core::VariableId::from_raw(treatment)],
            antecedent_core::VariableId::from_raw(outcome),
            &adjustment,
        ),
        Ok(Some(_))
    )
}

/// Re-derive the graph-criterion estimands of the carried identification product from the
/// carried DAG and the marginal average-effect target.
///
/// The product is a resealable payload, so replay recomputes its estimands: a
/// back-door estimand must satisfy the joint back-door criterion, an instrument the graph
/// instrument criterion, and a front-door estimand the front-door criterion. Each check
/// evaluates the carried set directly, with no candidate search. A set the criterion rejects
/// is [`ContractReason::IdentificationProductEstimands`]; a check that could not finish
/// (budget, cancellation, internal error) is the distinct
/// [`ContractReason::DependenciesIdentificationReplayBudget`], never a statement that the
/// estimand is wrong. Estimands of other methods are recomputed by
/// `verify_rederived_functionals`, not here. A back-door, instrument or front-door estimand this check cannot evaluate
/// (a graph class other than a DAG, a target other than the marginal average effect, effect
/// modifiers, or a shape outside the criterion check) is not passed: it is the distinct
/// [`ContractReason::DependenciesIdentificationEstimandCriterion`]: the estimand is not
/// replayable by this consumer. That limits nothing about the result's use and a producer may
/// still export it.
#[allow(
    clippy::needless_pass_by_value,
    reason = "the verdict is consumed once, as the identifier returned it"
)]
pub(in crate::contract_section) fn estimand_verdict(
    verdict: Result<bool, antecedent_identify::IdentificationError>,
) -> Result<(), ContractReason> {
    use antecedent_identify::IdentificationError;
    match verdict {
        Ok(true) => Ok(()),
        Ok(false)
        | Err(
            IdentificationError::InvalidQuery { .. } | IdentificationError::UnknownVariable { .. },
        ) => Err(ContractReason::IdentificationProductEstimands),
        Err(_) => Err(ContractReason::DependenciesIdentificationReplayBudget),
    }
}

/// Check every carried estimand the verifier can re-derive from the carried graph.
///
/// Graph-criterion estimands (back-door, instrument, front-door) are checked against their
/// criterion; general-ID and path-specific estimands are re-derived by running the same
/// identification on the carried graph and comparing the functional structurally. A wrong
/// estimand is reported in preference to an unresolved one.
pub(in crate::contract_section) fn verify_carried_estimands(
    contract: &AnalysisResultContractWire,
) -> Result<(), ContractReason> {
    let criterion = verify_graph_criterion_estimands(contract);
    let rederived = verify_rederived_functionals(contract);
    match (criterion, rederived) {
        (Err(ContractReason::IdentificationProductEstimands), _)
        | (_, Err(ContractReason::IdentificationProductEstimands)) => {
            Err(ContractReason::IdentificationProductEstimands)
        }
        (criterion, rederived) => criterion.and(rederived),
    }
}

/// Canonical structural identity of expression roots across arenas: equal ids mean equal
/// formulas. Variable sets are compared as sets and product factors as multisets, so an
/// arena built in another insertion order still compares equal; derivation notes are not
/// identity.
#[derive(Default)]
struct FormulaCanon {
    interned: std::collections::HashMap<String, u32>,
}

impl FormulaCanon {
    const MAX_DEPTH: usize = 4096;

    fn intern(&mut self, key: String) -> u32 {
        let next = u32::try_from(self.interned.len()).unwrap_or(u32::MAX);
        *self.interned.entry(key).or_insert(next)
    }

    fn root(&mut self, arena: &crate::ExprArenaWire, root: u32) -> Option<u32> {
        let mut memo = std::collections::HashMap::new();
        self.node(arena, &mut memo, root, 0)
    }

    fn var_set(arena: &crate::ExprArenaWire, id: u32) -> Option<String> {
        let mut vars = arena.var_sets.get(id as usize)?.clone();
        vars.sort_unstable();
        Some(format!("{vars:?}"))
    }

    fn node(
        &mut self,
        arena: &crate::ExprArenaWire,
        memo: &mut std::collections::HashMap<u32, u32>,
        id: u32,
        depth: usize,
    ) -> Option<u32> {
        use crate::ExprNodeWire as N;
        if let Some(done) = memo.get(&id) {
            return Some(*done);
        }
        if depth > Self::MAX_DEPTH {
            return None;
        }
        let key = match arena.nodes.get(id as usize)? {
            N::Distribution {
                variables,
                conditioned_on,
                intervention,
                domain,
                population,
                regime,
            } => {
                format!(
                    "D|{}|{}|{:?}|{domain}|{population}|{regime:?}",
                    Self::var_set(arena, *variables)?,
                    Self::var_set(arena, *conditioned_on)?,
                    arena.interventions.get(*intervention as usize)?,
                )
            }
            N::Kernel { body, bound, population, regime } => format!(
                "K|{}|{}|{population}|{regime:?}",
                self.node(arena, memo, *body, depth + 1)?,
                Self::var_set(arena, *bound)?,
            ),
            N::Product(list) => {
                let mut factors = Vec::new();
                for factor in arena.lists.get(*list as usize)? {
                    factors.push(self.node(arena, memo, *factor, depth + 1)?);
                }
                factors.sort_unstable();
                format!("P|{factors:?}")
            }
            N::SumOut { variables, expr } => format!(
                "S|{}|{}",
                Self::var_set(arena, *variables)?,
                self.node(arena, memo, *expr, depth + 1)?,
            ),
            N::IntegralOut { variables, expr } => format!(
                "I|{}|{}",
                Self::var_set(arena, *variables)?,
                self.node(arena, memo, *expr, depth + 1)?,
            ),
            N::Ratio { numerator, denominator } => format!(
                "R|{}|{}",
                self.node(arena, memo, *numerator, depth + 1)?,
                self.node(arena, memo, *denominator, depth + 1)?,
            ),
            N::Expectation { function, distribution } => {
                format!("E|{function}|{}", self.node(arena, memo, *distribution, depth + 1)?,)
            }
            N::Contrast { left, right, op } => format!(
                "C|{op}|{}|{}",
                self.node(arena, memo, *left, depth + 1)?,
                self.node(arena, memo, *right, depth + 1)?,
            ),
        };
        let canonical = self.intern(key);
        memo.insert(id, canonical);
        Some(canonical)
    }
}

/// Re-derive the general-ID and path-specific estimands of the carried product.
///
/// The product is a resealable payload, and its functional is bound only to the program that
/// evaluates it, so a self-consistent wrong functional (the naive conditional, say) would
/// otherwise verify. The same identification the producer ran is run on the carried graph for
/// the carried target query, and each carried estimand of that method must equal a re-derived
/// one structurally. A functional the derivation does not produce, or one it finds
/// unidentified, is [`ContractReason::IdentificationProductEstimands`]. A graph class or query
/// the derivation does not handle is the distinct
/// [`ContractReason::DependenciesIdentificationEstimandCriterion`]; a derivation that hit its
/// work cap or was cancelled is [`ContractReason::DependenciesIdentificationReplayBudget`]. Neither
/// is a statement that the estimand is wrong, and neither passes it.
fn verify_rederived_functionals(
    contract: &AnalysisResultContractWire,
) -> Result<(), ContractReason> {
    use antecedent_identify::{IdIdentifier, IdentificationError, IdentificationWorkspace};

    let Some(product) = contract.identification_product.as_ref() else {
        return Ok(());
    };
    let rederived = |method: &str| matches!(method, "general.id" | "path_specific.natural");
    if !product.estimands.iter().any(|estimand| rederived(estimand.method.as_str())) {
        return Ok(());
    }
    let unchecked = ContractReason::DependenciesIdentificationEstimandCriterion;
    let Some(identification) = contract.identification.as_ref() else {
        return Err(unchecked);
    };
    let prepared = match &identification.graph {
        crate::identity::GraphIdentityWire::Dag(graph) => {
            let Ok(dag) = crate::dag_from_wire(graph) else {
                return Err(ContractReason::IdentificationProductEstimands);
            };
            IdIdentifier::new().prepare_dag(&dag)
        }
        crate::identity::GraphIdentityWire::Admg(graph) => {
            let Ok(admg) = crate::admg_from_wire(graph) else {
                return Err(ContractReason::IdentificationProductEstimands);
            };
            IdIdentifier::new().prepare(&admg)
        }
        _ => return Err(unchecked),
    };
    let Ok(prepared) = prepared else { return Err(unchecked) };
    // Only queries whose identification is a single marginal derivation are re-derived.
    let method = match &contract.target.query {
        crate::CausalQueryWire::AverageEffect { effect_modifiers, .. }
            if effect_modifiers.is_empty() =>
        {
            "general.id"
        }
        // An unconditional interventional distribution is identified by the same ID run;
        // a conditioned one belongs to IDC and is not re-derived here.
        crate::CausalQueryWire::Distribution(distribution)
            if distribution.conditioning.is_empty() =>
        {
            "general.id"
        }
        crate::CausalQueryWire::PathSpecific(_) => "path_specific.natural",
        _ => return Err(unchecked),
    };
    let Ok(query) = crate::query_wire::causal_query_from_wire(&contract.target.query) else {
        return Err(unchecked);
    };
    let mut workspace = IdentificationWorkspace::default();
    let result = if method == "general.id" {
        IdIdentifier::new().identify(&prepared, &query, &mut workspace)
    } else {
        antecedent_identify::PathSpecificIdentifier::new().identify(
            &prepared,
            &query,
            &mut workspace,
        )
    };
    let result = match result {
        Ok(result) => result,
        Err(IdentificationError::Budget { .. } | IdentificationError::Cancelled) => {
            return Err(ContractReason::DependenciesIdentificationReplayBudget);
        }
        Err(IdentificationError::UnknownVariable { .. }) => {
            return Err(ContractReason::IdentificationProductEstimands);
        }
        Err(_) => return Err(unchecked),
    };
    let Ok(derived_arena) = crate::expr_arena_to_wire(&result.arena) else { return Err(unchecked) };
    let mut canon = FormulaCanon::default();
    let derived: Vec<u32> = result
        .estimands
        .iter()
        .filter(|estimand| &*estimand.method == method)
        .filter_map(|estimand| canon.root(&derived_arena, estimand.functional.raw()))
        .collect();
    let mut unresolved = false;
    for estimand in &product.estimands {
        let carried_method = estimand.method.as_str();
        if !rederived(carried_method) {
            continue;
        }
        if carried_method != method {
            // The target query is of another kind than this estimand's method.
            unresolved = true;
            continue;
        }
        match canon.root(&product.arena, estimand.functional) {
            Some(carried) if derived.contains(&carried) => {}
            _ => return Err(ContractReason::IdentificationProductEstimands),
        }
    }
    if unresolved { Err(unchecked) } else { Ok(()) }
}

fn verify_graph_criterion_estimands(
    contract: &AnalysisResultContractWire,
) -> Result<(), ContractReason> {
    use antecedent_core::VariableId;
    use antecedent_identify::{backdoor::check_joint_adjustment, frontdoor, iv};

    let Some(product) = contract.identification_product.as_ref() else {
        return Ok(());
    };
    // A graph-criterion estimand that this check cannot recompute is reported as not
    // replayable rather than passed as recomputed. Only methods whose criterion is a graph
    // property are examined here; other methods are recomputed by re-derivation.
    let graph_criterion = |method: &str| {
        matches!(method, "backdoor.adjustment" | "backdoor.efficient" | "iv" | "frontdoor")
    };
    let carries_graph_criterion =
        product.estimands.iter().any(|estimand| graph_criterion(estimand.method.as_str()));
    let unchecked = if carries_graph_criterion {
        Err(ContractReason::DependenciesIdentificationEstimandCriterion)
    } else {
        Ok(())
    };
    let Some(identification) = contract.identification.as_ref() else {
        return unchecked;
    };
    let crate::identity::GraphIdentityWire::Dag(graph) = &identification.graph else {
        return unchecked;
    };
    let crate::CausalQueryWire::AverageEffect { treatment, outcome, effect_modifiers, .. } =
        &contract.target.query
    else {
        return unchecked;
    };
    if !effect_modifiers.is_empty() {
        return unchecked;
    }
    let Ok(dag) = crate::dag_from_wire(graph) else {
        return Err(ContractReason::IdentificationProductEstimands);
    };
    let (treatment, outcome) = (VariableId::from_raw(*treatment), VariableId::from_raw(*outcome));
    let ids = |raw: &[u32]| raw.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>();
    let mut skipped = false;
    for estimand in &product.estimands {
        let verdict = match estimand.method.as_str() {
            "backdoor.adjustment" | "backdoor.efficient"
                if estimand.instruments.is_empty() && estimand.mediators.is_empty() =>
            {
                check_joint_adjustment(&dag, &[treatment], outcome, &ids(&estimand.adjustment_set))
                    .map(|checked| checked.is_some())
            }
            "iv" if estimand.adjustment_set.is_empty() => match ids(&estimand.instruments)[..] {
                [instrument] => iv::check_instrument(&dag, instrument, treatment, outcome),
                _ => Ok(false),
            },
            "frontdoor" => {
                frontdoor::check_frontdoor_set(&dag, treatment, outcome, &ids(&estimand.mediators))
            }
            method => {
                // A graph-criterion estimand shaped outside what the criterion check
                // evaluates is not replayable here.
                skipped |= graph_criterion(method);
                continue;
            }
        };
        estimand_verdict(verdict)?;
    }
    if skipped { unchecked } else { Ok(()) }
}

#[allow(
    clippy::too_many_lines,
    reason = "The lowering verifier compares one complete checked product across its target, design, and evidence."
)]
pub(in crate::contract_section) fn verify_linear_adjustment_lowering(
    contract: &AnalysisResultContractWire,
    lowering: &crate::CheckedLinearAdjustmentLoweringWire,
) -> bool {
    let (Some(product), Some(program), Some(snapshot), Some(binding)) = (
        contract.identification_product.as_ref(),
        contract.program.as_ref(),
        contract.data_snapshot.as_ref(),
        contract.inference_binding.as_ref(),
    ) else {
        return false;
    };
    if lowering.format != 1
        || lowering.backend != "faer"
        || lowering.population != TargetPopulationWire::AllObserved
        || lowering.complete_case_rows == 0
        || lowering.complete_case_rows > snapshot.row_count
        || lowering.design_columns.first().map(String::as_str) != Some("intercept")
        || lowering.design_columns.get(1).map(String::as_str) != Some("treatment")
        || lowering.design_columns.len() != lowering.adjustment.len() + 2
        || lowering
            .design_columns
            .iter()
            .skip(2)
            .zip(&lowering.adjustment)
            .any(|(role, id)| role != &format!("covariate:{id}"))
        || !valid_linear_fit_kind(&lowering.fit_kind)
        || !matches!(
            lowering.se_kind.as_str(),
            "homoskedastic"
                | "hc0"
                | "hc1"
                | "hc2"
                | "hc3"
                | "cluster"
                | "multiway"
                | "newey_west"
                | "panel_cluster_hac"
        )
        || matches!(lowering.se_kind.as_str(), "newey_west" | "panel_cluster_hac")
            != lowering.se_lag.is_some()
    {
        return false;
    }
    let Some(estimand) = product.estimands.iter().find(|estimand| {
        estimand.functional == lowering.functional
            && estimand.method == "backdoor.adjustment"
            && estimand.adjustment_set == lowering.adjustment
    }) else {
        return false;
    };
    if !carried_adjustment_is_backdoor(
        contract,
        lowering.treatment,
        lowering.outcome,
        &lowering.adjustment,
    ) {
        return false;
    }
    let target_matches = match &contract.target.query {
        crate::CausalQueryWire::AverageEffect {
            treatment,
            outcome,
            active,
            control,
            outcome_functional,
            target_population,
            effect_modifiers,
        } => {
            *treatment == lowering.treatment
                && *outcome == lowering.outcome
                && intervention_bits(active, *treatment) == Some(lowering.active_bits)
                && intervention_bits(control, *treatment) == Some(lowering.control_bits)
                && matches!(outcome_functional, crate::query_wire::OutcomeFunctionalWire::Mean)
                && target_population == &lowering.population
                && effect_modifiers.is_empty()
        }
        _ => false,
    };
    if !product.estimands.iter().any(|candidate| candidate.functional == lowering.functional) {
        return false;
    }
    let expression_matches = linear_adjustment_expression_matches(lowering);
    let commitments_match = program.commitments.resolved_estimator.as_deref()
        == Some("linear.adjustment.ate")
        && program.commitments.inference == "frequentist"
        && !program.commitments.prior_required
        && program.commitments.se_kind.as_deref() == Some(lowering.se_kind.as_str())
        && program.commitments.interval_method == lowering.interval_method
        && lowering.interval_method
            == if lowering.bootstrap_replicates > 0 { "bootstrap_se" } else { "analytic_se" }
        && binding.inference == "frequentist"
        && binding.bayesian.is_none()
        && binding.bootstrap_replicates == lowering.bootstrap_replicates;
    let binding_matches = match binding.estimator_spec.as_ref() {
        Some(crate::EstimatorSpecWire::Default(name)) => name == "linear.adjustment.ate",
        Some(crate::EstimatorSpecWire::LinearAdjustmentAte(config)) => {
            config.bootstrap_replicates == lowering.bootstrap_replicates
                && config.se_kind.as_deref().unwrap_or("homoskedastic") == lowering.se_kind
                && config.backend == lowering.backend
                && config.fit_kind.as_ref().is_none_or(|(family, parameter)| {
                    let expected = match lowering.fit_kind.split_once(':') {
                        Some((kind, bits)) => (kind, u64::from_str_radix(bits, 16).ok()),
                        None => (lowering.fit_kind.as_str(), None),
                    };
                    family == expected.0 && *parameter == expected.1
                })
        }
        None => true,
        _ => false,
    };
    target_matches
        && estimand.adjustment_set == lowering.adjustment
        && expression_matches
        && commitments_match
        && binding_matches
}

fn linear_adjustment_expression_matches(
    lowering: &crate::CheckedLinearAdjustmentLoweringWire,
) -> bool {
    let Ok(mut arena) = crate::expr_arena_from_wire(&lowering.arena) else {
        return false;
    };
    let treatment = antecedent_core::VariableId::from_raw(lowering.treatment);
    let outcome = antecedent_core::VariableId::from_raw(lowering.outcome);
    let adjustment = lowering
        .adjustment
        .iter()
        .copied()
        .map(antecedent_core::VariableId::from_raw)
        .collect::<Vec<_>>();
    let expected_source = arena.backdoor_ate(
        treatment,
        outcome,
        &adjustment,
        antecedent_core::Value::Float64(f64::from_bits(lowering.active_bits)),
        antecedent_core::Value::Float64(f64::from_bits(lowering.control_bits)),
    );
    let z = arena.intern_var_set(adjustment.iter().copied());
    let y = arena.intern_var_set([outcome]);
    let tz = arena.intern_var_set(std::iter::once(treatment).chain(adjustment.iter().copied()));
    let empty = arena.empty_var_set();
    let no_intervention = arena.empty_intervention_set();
    let marginal_z = arena.intern_distribution(
        z,
        empty,
        no_intervention,
        antecedent_expr::DomainRef::Observational,
    );
    let outcome_given_tz = arena.intern_distribution(
        y,
        tz,
        no_intervention,
        antecedent_expr::DomainRef::Observational,
    );
    let factors = arena.intern_list([outcome_given_tz, marginal_z]);
    let product_expr = arena.intern(antecedent_expr::ExprNode::Product(factors));
    let averaged =
        arena.intern(antecedent_expr::ExprNode::SumOut { variables: z, expr: product_expr });
    let expected_executable = arena.intern(antecedent_expr::ExprNode::Expectation {
        function: antecedent_expr::OutcomeExprId::identity(outcome),
        distribution: averaged,
    });
    let expected_arena = crate::expr_arena_to_wire(&arena).ok();
    expected_source.raw() == lowering.functional
        && expected_executable.raw() == lowering.executable
        && expected_arena.as_ref() == Some(&lowering.arena)
}

/// Whether a study-level overlap policy tag (as `inference_binding.overlap_policy`
/// spells it) trims exactly as the checked AIPW lowering does; no tag means the
/// estimator's default, untrimmed policy ran.
pub(in crate::contract_section) fn overlap_tag_trim_matches(
    tag: Option<&str>,
    lowering: &CheckedAipwLoweringWire,
) -> bool {
    let expected = format!(
        ":trim={}",
        lowering.trim_bits.map_or_else(|| "none".into(), |bits| format!("{bits:016x}"))
    );
    match tag {
        None => lowering.trim_bits.is_none(),
        Some(tag) => tag.starts_with("require_diagnostics:") && tag.ends_with(&expected),
    }
}

pub(in crate::contract_section) fn overlap_policy_tag(policy: &crate::OverlapPolicyWire) -> String {
    match policy {
        crate::OverlapPolicyWire::ExplicitOverride => "explicit_override".into(),
        crate::OverlapPolicyWire::RequireDiagnostics { clip_bits, trim_bits } => format!(
            "require_diagnostics:clip={}:trim={}",
            clip_bits.map_or_else(|| "none".into(), |bits| format!("{bits:016x}")),
            trim_bits.map_or_else(|| "none".into(), |bits| format!("{bits:016x}")),
        ),
    }
}

/// Cross-layer references: every layer must describe the same question,
/// observation contract, graph class, schema, and inference family.
pub(in crate::contract_section) fn verify_layer_links(
    contract: &AnalysisResultContractWire,
    unresolved: &mut Vec<ContractReason>,
) {
    let target = &contract.target;
    if let Some(identification) = &contract.identification {
        match digest_wire(IdentityDomain::Target, &target.question()) {
            Ok(question) if question.as_bytes() == &identification.target_question => {}
            _ => unresolved.push(ContractReason::IdentificationTargetQuestion),
        }
        if identification.observation != contract.identities.observation {
            unresolved.push(ContractReason::IdentificationObservation);
        }
        if identification.graph_class != contract.graph_class {
            unresolved.push(ContractReason::IdentificationGraphClass);
        }
        if identification.structure_source != contract.structure_source {
            unresolved.push(ContractReason::IdentificationStructureSource);
        }
        if identification
            .schema_names
            .as_ref()
            .is_some_and(|names| *names != target.schema.variable_names())
        {
            unresolved.push(ContractReason::IdentificationSchemaNames);
        }
        let depends_on: &[u32] = match target.query.target_population() {
            Some(TargetPopulationWire::CustomDistribution { depends_on, .. }) => depends_on,
            _ => &[],
        };
        if identification.population_depends_on != depends_on {
            unresolved.push(ContractReason::IdentificationPopulationDependsOn);
        }
    }
    verify_score_reuse_link(contract, unresolved);
    verify_target_weights_link(contract, unresolved);
    if let Some(observation) = &contract.observation {
        if observation.schema != target.schema {
            unresolved.push(ContractReason::ObservationSchema);
        }
    }
    if let Some(program) = &contract.program {
        let commitments = &program.commitments;
        if commitments.identifier != contract.identifier {
            unresolved.push(ContractReason::ProgramCommitmentsIdentifier);
        }
        if commitments.estimator != contract.estimator {
            unresolved.push(ContractReason::ProgramCommitmentsEstimator);
        }
        if let Some(binding) = &contract.inference_binding {
            if commitments.inference != binding.inference {
                unresolved.push(ContractReason::InferenceBindingInference);
            }
            if commitments.validation_suite != binding.validation_suite {
                unresolved.push(ContractReason::InferenceBindingValidationSuite);
            }
            if binding.bayesian.is_some() != (binding.inference == "bayesian") {
                unresolved.push(ContractReason::InferenceBindingBayesian);
            }
            if commitments.prior_required != binding.bayesian.is_some() {
                unresolved.push(ContractReason::ProgramCommitmentsPriorRequired);
            }
        }
    }
}
