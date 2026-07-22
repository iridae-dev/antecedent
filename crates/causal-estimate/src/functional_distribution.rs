//! Nonparametric estimation of identified interventional distributions via
//! discrete empirical CPT plug-in into compiled ID/IDC functionals.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::manual_flatten,
    clippy::needless_pass_by_value,
    clippy::type_complexity,
    clippy::zero_sized_map_values
)]

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use causal_core::{
    AssumptionSet, ExecutionContext, Intervention, InterventionalDistributionQuery,
    TargetPopulation, Value, VariableId,
};
use causal_data::{ColumnView, TableView, TabularData};
use causal_expr::{
    Assignment, CausalExprArena, CompiledEvaluator, DistributionProvider, DomainRef,
    EmpiricalTableProvider, EstimandMethod, EvalContext, EvalError, ExprId, ExprNode, FactorSpec,
    IdentifiedEstimand, InterventionAssignment,
};

use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::prepare::require_method;
use crate::util::{BootstrapSeResult, bootstrap_se};

/// Hard cap on discrete levels per variable (fail-closed beyond this).
const MAX_DISCRETE_LEVELS: usize = 64;

/// One outcome-level probability mass under an interventional (and optional
/// observational) conditioning assignment.
#[derive(Clone, Debug, PartialEq)]
pub struct DistributionAtom {
    /// Outcome variable assignments (aligned to query outcomes order).
    pub outcomes: Arc<[(VariableId, Value)]>,
    /// Conditioning assignments (empty when unconditional).
    pub conditioning: Arc<[(VariableId, Value)]>,
    /// Estimated probability mass.
    pub probability: f64,
}

/// Estimated interventional distribution P(Y | do(X)[, Z]).
#[derive(Clone, Debug)]
pub struct InterventionalDistributionEstimate {
    /// Probability atoms over the outcome support.
    pub atoms: Arc<[DistributionAtom]>,
    /// Interventional mean of the first numeric outcome when defined; otherwise NaN.
    pub mean: f64,
    /// Analytic SE is not defined for the discrete plug-in (multinomial delta-method out of scope).
    pub se_analytic: f64,
    /// Bootstrap SE of the interventional mean when requested.
    pub se_bootstrap: Option<f64>,
    /// Successful bootstrap replicates contributing to [`Self::se_bootstrap`].
    pub bootstrap_replicates_ok: Option<u32>,
    /// Soft-failed bootstrap replicates.
    pub bootstrap_replicates_failed: Option<u32>,
    /// Assumptions carried from identification.
    pub assumptions: AssumptionSet,
    /// Overlap policy recorded on the artifact.
    pub overlap: OverlapPolicy,
    /// Estimated retained-memory cost of fitted scratch (bytes), when known.
    pub retained_memory_bytes: Option<u64>,
}

/// Reusable scratch for [`FunctionalDistribution`] estimation.
#[derive(Clone, Debug, Default)]
pub struct FunctionalDistributionWorkspace {
    /// Scratch assignment reused across outcome atoms.
    pub assignment: Assignment,
}

impl FunctionalDistributionWorkspace {
    /// Clear reusable buffers.
    pub fn clear(&mut self) {
        self.assignment = Assignment::new();
    }
}

/// Prepared discrete functional-distribution problem.
#[derive(Clone, Debug)]
pub struct PreparedFunctionalDistribution {
    /// Identified estimand (`GeneralId` / IDC).
    pub estimand: IdentifiedEstimand,
    /// Expression arena owning the functional.
    pub arena: CausalExprArena,
    /// Compiled evaluator for the functional root.
    pub compiled: CompiledEvaluator,
    /// Empirical CPT provider built from data.
    pub provider: EmpiricalTableProvider,
    /// Outcome variables (query order).
    pub outcomes: Arc<[VariableId]>,
    /// Hard intervention bindings from the query.
    pub interventions: Arc<[InterventionAssignment]>,
    /// Observational conditioning bindings (IDC); empty when unconditional.
    pub conditioning: Arc<[InterventionAssignment]>,
    /// Assumptions from identification.
    pub assumptions: AssumptionSet,
    /// Row-aligned discrete columns for bootstrap CPT refits.
    bootstrap_columns: HashMap<VariableId, Vec<Option<Value>>>,
    /// Factor specs used to rebuild the empirical provider.
    bootstrap_factors: Vec<(Arc<[VariableId]>, Arc<[VariableId]>)>,
    /// Interventional signatures used to rebuild the empirical provider.
    bootstrap_signatures:
        Vec<(Arc<[VariableId]>, Arc<[VariableId]>, Arc<[InterventionAssignment]>, DomainRef)>,
}

/// Plug-in estimator for identified interventional distributions (discrete).
#[derive(Clone, Debug)]
pub struct FunctionalDistribution {
    /// Overlap policy (positivity is implicit in CPT support; override allowed).
    pub overlap: OverlapPolicy,
    /// Bootstrap replicates for the interventional mean SE (0 = skip).
    pub bootstrap_replicates: u32,
}

impl Default for FunctionalDistribution {
    fn default() -> Self {
        Self::new()
    }
}

impl FunctionalDistribution {
    /// Create with default overlap override (CPT positivity is data-driven).
    #[must_use]
    pub fn new() -> Self {
        Self { overlap: OverlapPolicy::ExplicitOverride, bootstrap_replicates: 0 }
    }

    /// Prepare from an identified `GeneralId` functional and tabular data.
    ///
    /// # Errors
    ///
    /// Incompatible estimand, continuous/high-cardinality columns, empty support,
    /// unsupported interventions, or expression compile failure.
    pub fn prepare(
        &self,
        data: &TabularData,
        query: &InterventionalDistributionQuery,
        estimand: &IdentifiedEstimand,
        arena: &CausalExprArena,
        assumptions: AssumptionSet,
    ) -> Result<PreparedFunctionalDistribution, EstimationError> {
        query.validate()?;
        if query.target_population != TargetPopulation::AllObserved {
            return Err(EstimationError::TargetPopulation);
        }
        require_method(
            estimand,
            &[EstimandMethod::GeneralId],
            "functional.distribution requires a general.id estimand",
        )?;

        let interventions = set_assignments(&query.interventions)?;
        let conditioning: Vec<InterventionAssignment> = query
            .conditioning
            .iter()
            .map(|&variable| {
                // Conditioning values are supplied at evaluate time per atom when
                // estimating the full conditional table; prepare stores variable ids
                // as placeholders (NaN) only when empty bindings are needed for CPT
                // domain collection. Concrete Z values come from `conditioning_values`
                // on estimate, or from evaluating the conditional density functional
                // which already conditions structurally via IDC.
                InterventionAssignment { variable, value: Value::f64(f64::NAN) }
            })
            .collect();

        let factor_specs = collect_observational_factors(arena, estimand.functional);
        let signatures = collect_factor_signatures(arena, estimand.functional);
        let mut vars_needed = HashSet::new();
        for (vars, cond) in &factor_specs {
            vars_needed.extend(vars.iter().copied());
            vars_needed.extend(cond.iter().copied());
        }
        for &y in query.outcomes.iter() {
            vars_needed.insert(y);
        }
        for a in &interventions {
            vars_needed.insert(a.variable);
        }
        for &z in query.conditioning.iter() {
            vars_needed.insert(z);
        }

        let (provider, columns) =
            build_empirical_provider(data, &vars_needed, &factor_specs, &signatures)?;
        let compiled = arena.compile(estimand.functional).map_err(eval_err)?;

        Ok(PreparedFunctionalDistribution {
            estimand: estimand.clone(),
            arena: arena.clone(),
            compiled,
            provider,
            outcomes: Arc::clone(&query.outcomes),
            interventions: Arc::from(interventions),
            conditioning: Arc::from(conditioning),
            assumptions,
            bootstrap_columns: columns,
            bootstrap_factors: factor_specs,
            bootstrap_signatures: signatures,
        })
    }

    /// Estimate the interventional distribution over the outcome support.
    ///
    /// For unconditional queries, returns P(Y=y | do(X)) for each outcome atom.
    /// For IDC queries with nonempty conditioning:
    /// - if `conditioning_values` is nonempty, binds that single Z point;
    /// - if empty, enumerates the empirical support of Z and returns atoms for each (y, z).
    ///
    /// # Errors
    ///
    /// Partial conditioning bindings, empty support, or evaluation failure.
    pub fn estimate(
        &self,
        prepared: &PreparedFunctionalDistribution,
        conditioning_values: &[(VariableId, Value)],
        workspace: &mut FunctionalDistributionWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<InterventionalDistributionEstimate, EstimationError> {
        let mut out = self.estimate_point(prepared, conditioning_values, workspace)?;
        if self.bootstrap_replicates == 0 || !out.mean.is_finite() {
            return Ok(out);
        }
        let n = prepared.bootstrap_columns.values().next().map_or(0, Vec::len);
        let boot = bootstrap_se(self.bootstrap_replicates, ctx, 0xF01D_u64, n, |idx| {
            let columns = gather_columns(&prepared.bootstrap_columns, idx);
            let provider = provider_from_columns(
                &columns,
                idx.len(),
                &prepared.bootstrap_factors,
                &prepared.bootstrap_signatures,
            )?;
            let mut prep = prepared.clone();
            prep.provider = provider;
            let mut ws = FunctionalDistributionWorkspace::default();
            match self.estimate_point(&prep, conditioning_values, &mut ws) {
                Ok(est) if est.mean.is_finite() => Ok(Some(est.mean)),
                Ok(_) | Err(_) => Ok(None),
            }
        })?;
        out.se_bootstrap = boot.se;
        out.bootstrap_replicates_ok = Some(boot.replicates_ok);
        out.bootstrap_replicates_failed = Some(boot.replicates_failed);
        Ok(out)
    }

    fn estimate_point(
        &self,
        prepared: &PreparedFunctionalDistribution,
        conditioning_values: &[(VariableId, Value)],
        workspace: &mut FunctionalDistributionWorkspace,
    ) -> Result<InterventionalDistributionEstimate, EstimationError> {
        workspace.clear();

        let needed_z: Vec<VariableId> = prepared.conditioning.iter().map(|a| a.variable).collect();
        let z_points: Vec<Vec<(VariableId, Value)>> = if needed_z.is_empty() {
            if !conditioning_values.is_empty() {
                return Err(EstimationError::unsupported(
                    "conditioning_values supplied for an unconditional distribution query",
                ));
            }
            vec![Vec::new()]
        } else if conditioning_values.is_empty() {
            let support =
                prepared.provider.support(&needed_z, &EvalContext::default()).map_err(eval_err)?;
            support
                .iter()
                .map(|row| needed_z.iter().copied().zip(row.iter().cloned()).collect::<Vec<_>>())
                .collect()
        } else {
            let provided: HashSet<VariableId> =
                conditioning_values.iter().map(|(v, _)| *v).collect();
            let needed: HashSet<VariableId> = needed_z.iter().copied().collect();
            if provided != needed {
                return Err(EstimationError::unsupported(
                    "conditioning_values must bind exactly the query conditioning set",
                ));
            }
            vec![conditioning_values.to_vec()]
        };

        let y_support = prepared
            .provider
            .support(prepared.outcomes.as_ref(), &EvalContext::default())
            .map_err(eval_err)?;
        if y_support.is_empty() {
            return Err(EstimationError::data_msg("empty outcome support"));
        }

        let mut atoms = Vec::with_capacity(y_support.len().saturating_mul(z_points.len().max(1)));
        let mut mean_acc = 0.0;
        let mut mean_ok = prepared.outcomes.len() == 1 && z_points.len() == 1;

        for z_bind in &z_points {
            for row in y_support.iter() {
                workspace.assignment = Assignment::new();
                for a in prepared.interventions.iter() {
                    workspace.assignment.set(a.variable, a.value.clone());
                }
                for (v, val) in z_bind {
                    workspace.assignment.set(*v, val.clone());
                }
                let mut outcome_pairs = Vec::with_capacity(prepared.outcomes.len());
                for (i, &y) in prepared.outcomes.iter().enumerate() {
                    let val = row.get(i).cloned().ok_or_else(|| {
                        EstimationError::data_msg("outcome support row shorter than outcomes")
                    })?;
                    workspace.assignment.set(y, val.clone());
                    outcome_pairs.push((y, val));
                }

                let p = prepared
                    .compiled
                    .evaluate_with(
                        &prepared.arena,
                        &prepared.provider,
                        &EvalContext::default(),
                        &workspace.assignment,
                    )
                    .map_err(eval_err)?;

                if mean_ok {
                    if let Some((_, val)) = outcome_pairs.first() {
                        if let Some(y) = val.as_f64() {
                            mean_acc += p * y;
                        } else {
                            mean_ok = false;
                        }
                    }
                }

                atoms.push(DistributionAtom {
                    outcomes: Arc::from(outcome_pairs),
                    conditioning: Arc::from(z_bind.clone()),
                    probability: p,
                });
            }
        }

        Ok(InterventionalDistributionEstimate {
            atoms: Arc::from(atoms),
            mean: if mean_ok { mean_acc } else { f64::NAN },
            se_analytic: f64::NAN,
            se_bootstrap: None,
            bootstrap_replicates_ok: None,
            bootstrap_replicates_failed: None,
            assumptions: prepared.assumptions.clone(),
            overlap: self.overlap,
            retained_memory_bytes: None,
        })
    }
}

/// Prepared scalar functional (ATE / path-specific NE contrast).
#[derive(Clone, Debug)]
pub struct PreparedFunctionalEffect {
    /// Identified estimand.
    pub estimand: IdentifiedEstimand,
    /// Arena owning the functional.
    pub arena: CausalExprArena,
    /// Compiled evaluator.
    pub compiled: CompiledEvaluator,
    /// Empirical CPT provider.
    pub provider: EmpiricalTableProvider,
    /// Assumptions from identification.
    pub assumptions: AssumptionSet,
    bootstrap_columns: HashMap<VariableId, Vec<Option<Value>>>,
    bootstrap_factors: Vec<(Arc<[VariableId]>, Arc<[VariableId]>)>,
    bootstrap_signatures:
        Vec<(Arc<[VariableId]>, Arc<[VariableId]>, Arc<[InterventionAssignment]>, DomainRef)>,
}

/// Discrete plug-in estimator for identified scalar functionals (contrasts).
#[derive(Clone, Debug)]
pub struct FunctionalEffect {
    /// Overlap policy.
    pub overlap: OverlapPolicy,
    /// Bootstrap replicates for the scalar SE (0 = skip).
    pub bootstrap_replicates: u32,
}

impl Default for FunctionalEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl FunctionalEffect {
    /// Create with explicit overlap override.
    #[must_use]
    pub fn new() -> Self {
        Self { overlap: OverlapPolicy::ExplicitOverride, bootstrap_replicates: 0 }
    }

    /// Prepare CPT plug-in for a path-specific / general-ID contrast functional.
    ///
    /// # Errors
    ///
    /// Incompatible estimand, continuous columns, or compile failure.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        arena: &CausalExprArena,
        assumptions: AssumptionSet,
        extra_vars: &[VariableId],
    ) -> Result<PreparedFunctionalEffect, EstimationError> {
        require_method(
            estimand,
            &[EstimandMethod::PathSpecificNatural, EstimandMethod::GeneralId],
            "functional.effect requires path_specific.natural or general.id",
        )?;
        let factor_specs = collect_observational_factors(arena, estimand.functional);
        let signatures = collect_factor_signatures(arena, estimand.functional);
        let mut vars_needed = HashSet::new();
        for (vars, cond) in &factor_specs {
            vars_needed.extend(vars.iter().copied());
            vars_needed.extend(cond.iter().copied());
        }
        vars_needed.extend(extra_vars.iter().copied());
        let (provider, columns) =
            build_empirical_provider(data, &vars_needed, &factor_specs, &signatures)?;
        let compiled = arena.compile(estimand.functional).map_err(eval_err)?;
        Ok(PreparedFunctionalEffect {
            estimand: estimand.clone(),
            arena: arena.clone(),
            compiled,
            provider,
            assumptions,
            bootstrap_columns: columns,
            bootstrap_factors: factor_specs,
            bootstrap_signatures: signatures,
        })
    }

    /// Evaluate the scalar functional.
    ///
    /// # Errors
    ///
    /// Evaluation / missing CPT entries.
    pub fn estimate(
        &self,
        prepared: &PreparedFunctionalEffect,
        _workspace: &mut FunctionalDistributionWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<crate::adjustment::EffectEstimate, EstimationError> {
        let ate = prepared
            .compiled
            .evaluate(&prepared.arena, &prepared.provider, &EvalContext::default())
            .map_err(eval_err)?;
        let boot = if self.bootstrap_replicates == 0 {
            BootstrapSeResult::skipped()
        } else {
            let n = prepared.bootstrap_columns.values().next().map_or(0, Vec::len);
            bootstrap_se(self.bootstrap_replicates, ctx, 0xF02D_u64, n, |idx| {
                let columns = gather_columns(&prepared.bootstrap_columns, idx);
                let provider = provider_from_columns(
                    &columns,
                    idx.len(),
                    &prepared.bootstrap_factors,
                    &prepared.bootstrap_signatures,
                )?;
                match prepared.compiled.evaluate(
                    &prepared.arena,
                    &provider,
                    &EvalContext::default(),
                ) {
                    Ok(v) if v.is_finite() => Ok(Some(v)),
                    _ => Ok(None),
                }
            })?
        };
        Ok(crate::adjustment::EffectEstimate {
            ate,
            // Multinomial delta-method analytic SE is out of scope for the discrete plug-in.
            se_analytic: f64::NAN,
            se_bootstrap: boot.se,
            bootstrap_replicates_ok: if self.bootstrap_replicates == 0 {
                None
            } else {
                Some(boot.replicates_ok)
            },
            bootstrap_replicates_failed: if self.bootstrap_replicates == 0 {
                None
            } else {
                Some(boot.replicates_failed)
            },
            assumptions: prepared.assumptions.clone(),
            overlap: self.overlap,
            overlap_report: None,
            retained_memory_bytes: None,
        })
    }
}

fn set_assignments(
    interventions: &[Intervention],
) -> Result<Vec<InterventionAssignment>, EstimationError> {
    let mut out = Vec::with_capacity(interventions.len());
    for iv in interventions {
        match iv {
            Intervention::Set { variable, value } => {
                if value.as_f64().is_some_and(f64::is_nan) {
                    return Err(EstimationError::unsupported(
                        "functional.distribution requires concrete Set intervention values",
                    ));
                }
                out.push(InterventionAssignment { variable: *variable, value: value.clone() });
            }
            _ => {
                return Err(EstimationError::unsupported(
                    "functional.distribution supports hard Set interventions only",
                ));
            }
        }
    }
    Ok(out)
}

fn collect_observational_factors(
    arena: &CausalExprArena,
    root: ExprId,
) -> Vec<(Arc<[VariableId]>, Arc<[VariableId]>)> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        match arena.node(id) {
            ExprNode::Distribution { variables, conditioned_on, .. } => {
                let vars: Arc<[VariableId]> = Arc::from(arena.var_set(*variables).to_vec());
                let cond: Arc<[VariableId]> = Arc::from(arena.var_set(*conditioned_on).to_vec());
                let key = (vars.clone(), cond.clone());
                if seen.insert((vars.as_ref().to_vec(), cond.as_ref().to_vec())) {
                    out.push(key);
                }
            }
            ExprNode::Product(list) => {
                for &c in arena.list(*list) {
                    stack.push(c);
                }
            }
            ExprNode::SumOut { expr, .. } | ExprNode::IntegralOut { expr, .. } => {
                stack.push(*expr);
            }
            ExprNode::Ratio { numerator, denominator } => {
                stack.push(*numerator);
                stack.push(*denominator);
            }
            ExprNode::Expectation { distribution, .. } => stack.push(*distribution),
            ExprNode::Contrast { left, right, .. } => {
                stack.push(*left);
                stack.push(*right);
            }
        }
    }
    out
}

/// Collect (variables, `conditioned_on`, intervention set, domain) for CPT duplication.
fn collect_factor_signatures(
    arena: &CausalExprArena,
    root: ExprId,
) -> Vec<(Arc<[VariableId]>, Arc<[VariableId]>, Arc<[InterventionAssignment]>, DomainRef)> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        match arena.node(id) {
            ExprNode::Distribution { variables, conditioned_on, intervention, domain } => {
                let vars: Arc<[VariableId]> = Arc::from(arena.var_set(*variables).to_vec());
                let cond: Arc<[VariableId]> = Arc::from(arena.var_set(*conditioned_on).to_vec());
                let interv: Arc<[InterventionAssignment]> =
                    Arc::from(arena.intervention_assignments(*intervention).to_vec());
                let key = (
                    vars.as_ref().to_vec(),
                    cond.as_ref().to_vec(),
                    interv.iter().map(|a| (a.variable.raw(), a.value.clone())).collect::<Vec<_>>(),
                    *domain,
                );
                if seen.insert(key) {
                    out.push((vars, cond, interv, *domain));
                }
            }
            ExprNode::Product(list) => {
                for &c in arena.list(*list) {
                    stack.push(c);
                }
            }
            ExprNode::SumOut { expr, .. } | ExprNode::IntegralOut { expr, .. } => {
                stack.push(*expr);
            }
            ExprNode::Ratio { numerator, denominator } => {
                stack.push(*numerator);
                stack.push(*denominator);
            }
            ExprNode::Expectation { distribution, .. } => stack.push(*distribution),
            ExprNode::Contrast { left, right, .. } => {
                stack.push(*left);
                stack.push(*right);
            }
        }
    }
    out
}

fn build_empirical_provider(
    data: &TabularData,
    vars_needed: &HashSet<VariableId>,
    factors: &[(Arc<[VariableId]>, Arc<[VariableId]>)],
    signatures: &[(
        Arc<[VariableId]>,
        Arc<[VariableId]>,
        Arc<[InterventionAssignment]>,
        DomainRef,
    )],
) -> Result<(EmpiricalTableProvider, HashMap<VariableId, Vec<Option<Value>>>), EstimationError> {
    let mut columns: HashMap<VariableId, Vec<Option<Value>>> = HashMap::new();
    let n = data.row_count();

    for &id in vars_needed {
        let (col, _domain) = discrete_column(data, id)?;
        if col.len() != n {
            return Err(EstimationError::data_msg("column length mismatch"));
        }
        columns.insert(id, col);
    }

    let provider = provider_from_columns(&columns, n, factors, signatures)?;
    Ok((provider, columns))
}

fn gather_columns(
    columns: &HashMap<VariableId, Vec<Option<Value>>>,
    idx: &[usize],
) -> HashMap<VariableId, Vec<Option<Value>>> {
    columns
        .iter()
        .map(|(&id, col)| {
            let gathered: Vec<Option<Value>> =
                idx.iter().map(|&i| col.get(i).cloned().flatten()).collect();
            (id, gathered)
        })
        .collect()
}

fn provider_from_columns(
    columns: &HashMap<VariableId, Vec<Option<Value>>>,
    n: usize,
    factors: &[(Arc<[VariableId]>, Arc<[VariableId]>)],
    signatures: &[(
        Arc<[VariableId]>,
        Arc<[VariableId]>,
        Arc<[InterventionAssignment]>,
        DomainRef,
    )],
) -> Result<EmpiricalTableProvider, EstimationError> {
    let mut domains: HashMap<VariableId, Vec<Value>> = HashMap::new();
    for (&id, col) in columns {
        let mut seen = HashSet::new();
        let mut domain = Vec::new();
        for cell in col {
            if let Some(val) = cell {
                if seen.insert(val.clone()) {
                    domain.push(val.clone());
                }
            }
        }
        domains.insert(id, domain);
    }

    let mut provider = EmpiricalTableProvider::new();
    for (id, domain) in &domains {
        provider.set_domain(*id, domain.iter().cloned());
    }

    // Vacuous empty factor used by some ID edge cases.
    let empty_spec = FactorSpec {
        variables: &[],
        conditioned_on: &[],
        intervention: &[],
        domain: DomainRef::Observational,
    };
    provider.insert_probability(&empty_spec, &Assignment::from_pairs([]), 1.0).map_err(eval_err)?;

    for (vars, cond) in factors {
        if vars.is_empty() && cond.is_empty() {
            continue;
        }
        // Observational CPT.
        insert_cpt(&mut provider, columns, n, vars, cond, &[], DomainRef::Observational)?;
        // Duplicate under every interventional signature with the same (vars, cond).
        for (s_vars, s_cond, interv, domain) in signatures {
            if s_vars.as_ref() != vars.as_ref() || s_cond.as_ref() != cond.as_ref() {
                continue;
            }
            if *domain == DomainRef::Observational && interv.is_empty() {
                continue;
            }
            // Intervened coordinates in `vars` are Dirac under do(.); other factors
            // reuse the observational CPT under the interventional FactorKey.
            let intervened_in_vars: Vec<_> =
                interv.iter().filter(|a| vars.iter().any(|&v| v == a.variable)).cloned().collect();
            if intervened_in_vars.is_empty() {
                insert_cpt(&mut provider, columns, n, vars, cond, interv.as_ref(), *domain)?;
            } else {
                insert_dirac_intervened(
                    &mut provider,
                    &domains,
                    vars,
                    cond,
                    interv.as_ref(),
                    *domain,
                    &intervened_in_vars,
                )?;
            }
        }
    }

    Ok(provider)
}

fn insert_dirac_intervened(
    provider: &mut EmpiricalTableProvider,
    domains: &HashMap<VariableId, Vec<Value>>,
    vars: &[VariableId],
    cond: &[VariableId],
    intervention: &[InterventionAssignment],
    domain: DomainRef,
    intervened_in_vars: &[InterventionAssignment],
) -> Result<(), EstimationError> {
    // Free vars = vars not fixed by intervention.
    let free: Vec<VariableId> = vars
        .iter()
        .copied()
        .filter(|v| !intervened_in_vars.iter().any(|a| a.variable == *v))
        .collect();
    let free_rows = cartesian_domain(domains, &free)?;
    let cond_rows = cartesian_domain(domains, cond)?;
    for free_vals in &free_rows {
        for cond_vals in &cond_rows {
            let mut assign = Assignment::new();
            for a in intervened_in_vars {
                assign.set(a.variable, a.value.clone());
            }
            for (v, val) in free.iter().copied().zip(free_vals.iter().cloned()) {
                assign.set(v, val);
            }
            for (v, val) in cond.iter().copied().zip(cond_vals.iter().cloned()) {
                assign.set(v, val);
            }
            // Probability 1: intervened vars are fixed; free vars still need a
            // density — if there are free vars, fall back is wrong. For pure
            // Dirac on all vars, mass is 1 only for the intervened assignment.
            let p = if free.is_empty() {
                1.0
            } else {
                // Should not happen for ID treatment factors; refuse.
                return Err(EstimationError::unsupported(
                    "intervened factor with free variables is unsupported in functional.effect",
                ));
            };
            let spec = FactorSpec { variables: vars, conditioned_on: cond, intervention, domain };
            provider.insert_probability(&spec, &assign, p).map_err(eval_err)?;
        }
    }
    Ok(())
}

fn cartesian_domain(
    domains: &HashMap<VariableId, Vec<Value>>,
    vars: &[VariableId],
) -> Result<Vec<Vec<Value>>, EstimationError> {
    if vars.is_empty() {
        return Ok(vec![Vec::new()]);
    }
    let mut rows: Vec<Vec<Value>> = vec![Vec::new()];
    for &v in vars {
        let domain = domains.get(&v).ok_or_else(|| EstimationError::data_msg("missing domain"))?;
        let mut next = Vec::with_capacity(rows.len() * domain.len());
        for prefix in &rows {
            for val in domain {
                let mut row = prefix.clone();
                row.push(val.clone());
                next.push(row);
            }
        }
        rows = next;
    }
    Ok(rows)
}

fn insert_cpt(
    provider: &mut EmpiricalTableProvider,
    columns: &HashMap<VariableId, Vec<Option<Value>>>,
    n: usize,
    vars: &[VariableId],
    cond: &[VariableId],
    intervention: &[InterventionAssignment],
    domain: DomainRef,
) -> Result<(), EstimationError> {
    // Count (vars, cond) joint and cond marginal among complete cases.
    let mut joint: HashMap<Vec<Value>, u64> = HashMap::new();
    let mut marg: HashMap<Vec<Value>, u64> = HashMap::new();

    for row in 0..n {
        let mut ok = true;
        let mut cond_vals = Vec::with_capacity(cond.len());
        for &v in cond {
            if let Some(val) = columns.get(&v).and_then(|c| c.get(row)).and_then(|o| o.as_ref()) {
                cond_vals.push(val.clone());
            } else {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        let mut var_vals = Vec::with_capacity(vars.len());
        for &v in vars {
            if let Some(val) = columns.get(&v).and_then(|c| c.get(row)).and_then(|o| o.as_ref()) {
                var_vals.push(val.clone());
            } else {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        *marg.entry(cond_vals.clone()).or_insert(0) += 1;
        let mut key = var_vals;
        key.extend(cond_vals);
        *joint.entry(key).or_insert(0) += 1;
    }

    if cond.is_empty() {
        let total: u64 = joint.values().sum();
        if total == 0 {
            return Err(EstimationError::data_msg("no complete cases for CPT"));
        }
        let var_rows = cartesian_domain(&domains_for_insert(columns, vars)?, vars)?;
        for var_vals in &var_rows {
            let key = var_vals.clone();
            let count = joint.get(&key).copied().unwrap_or(0);
            let assign = Assignment::from_pairs(vars.iter().copied().zip(var_vals.iter().cloned()));
            let spec = FactorSpec { variables: vars, conditioned_on: cond, intervention, domain };
            provider
                .insert_probability(&spec, &assign, count as f64 / total as f64)
                .map_err(eval_err)?;
        }
    } else {
        let var_rows = cartesian_domain(&domains_for_insert(columns, vars)?, vars)?;
        let cond_rows = cartesian_domain(&domains_for_insert(columns, cond)?, cond)?;
        for cond_vals in &cond_rows {
            let cond_count = marg.get(cond_vals).copied().unwrap_or(0);
            for var_vals in &var_rows {
                let mut key = var_vals.clone();
                key.extend(cond_vals.iter().cloned());
                let count = joint.get(&key).copied().unwrap_or(0);
                let p = if cond_count == 0 { 0.0 } else { count as f64 / cond_count as f64 };
                let assign = Assignment::from_pairs(
                    vars.iter()
                        .copied()
                        .zip(var_vals.iter().cloned())
                        .chain(cond.iter().copied().zip(cond_vals.iter().cloned())),
                );
                let spec =
                    FactorSpec { variables: vars, conditioned_on: cond, intervention, domain };
                provider.insert_probability(&spec, &assign, p).map_err(eval_err)?;
            }
        }
    }
    Ok(())
}

fn domains_for_insert(
    columns: &HashMap<VariableId, Vec<Option<Value>>>,
    vars: &[VariableId],
) -> Result<HashMap<VariableId, Vec<Value>>, EstimationError> {
    let mut domains = HashMap::new();
    for &v in vars {
        let col = columns.get(&v).ok_or_else(|| EstimationError::data_msg("missing column"))?;
        let mut seen = HashSet::new();
        let mut domain = Vec::new();
        for cell in col {
            if let Some(val) = cell {
                if seen.insert(val.clone()) {
                    domain.push(val.clone());
                }
            }
        }
        if domain.is_empty() {
            return Err(EstimationError::data_msg("empty domain in CPT insert"));
        }
        domain.sort_by(|a, b| match (a.as_f64(), b.as_f64()) {
            (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
            _ => std::cmp::Ordering::Equal,
        });
        domains.insert(v, domain);
    }
    Ok(domains)
}

fn discrete_column(
    data: &TabularData,
    id: VariableId,
) -> Result<(Vec<Option<Value>>, Vec<Value>), EstimationError> {
    let view = data.column(id).map_err(EstimationError::from)?;
    let n = view.len();
    let validity = view.validity();
    let mut values = Vec::with_capacity(n);
    let mut domain_set: HashMap<Value, ()> = HashMap::new();
    let mut domain = Vec::new();

    match view {
        ColumnView::Float64(c) => {
            for i in 0..n {
                if !validity.is_valid(i) {
                    values.push(None);
                    continue;
                }
                let v = Value::f64(c.values[i]);
                if domain_set.insert(v.clone(), ()).is_none() {
                    domain.push(v.clone());
                }
                values.push(Some(v));
            }
        }
        ColumnView::Int64(c) => {
            for i in 0..n {
                if !validity.is_valid(i) {
                    values.push(None);
                    continue;
                }
                let v = Value::Int64(c.values[i]);
                if domain_set.insert(v.clone(), ()).is_none() {
                    domain.push(v.clone());
                }
                values.push(Some(v));
            }
        }
        ColumnView::Categorical(c) => {
            for i in 0..n {
                if !validity.is_valid(i) {
                    values.push(None);
                    continue;
                }
                let v = Value::Category(c.codes[i].raw());
                if domain_set.insert(v.clone(), ()).is_none() {
                    domain.push(v.clone());
                }
                values.push(Some(v));
            }
        }
        _ => {
            return Err(EstimationError::unsupported(
                "functional.distribution supports float64 / int64 / categorical columns only",
            ));
        }
    }

    if domain.is_empty() {
        return Err(EstimationError::data_msg("empty discrete domain"));
    }
    if domain.len() > MAX_DISCRETE_LEVELS {
        return Err(EstimationError::unsupported(
            "variable exceeds discrete level cap for functional.distribution",
        ));
    }
    // Stable order by Display/hash — sort by f64/i64 when possible.
    domain.sort_by(|a, b| match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
        _ => std::cmp::Ordering::Equal,
    });
    Ok((values, domain))
}

fn eval_err(e: EvalError) -> EstimationError {
    EstimationError::data_msg(e.to_string())
}

#[cfg(test)]
mod tests {
    use causal_core::{CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType};
    use causal_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, ValidityBitmap};
    use causal_graph::{Dag, DenseNodeId};
    use causal_identify::{IdIdentifier, IdentificationStatus, IdentificationWorkspace};

    use super::*;

    fn f(x: f64) -> Value {
        Value::f64(x)
    }

    fn binary_confounding_table() -> TabularData {
        // Z, T, Y with known interventional mean E[Y|do(T=1)] = 0.7
        // Rows generated from: P(Z)=0.5, P(T|Z)=..., P(Y|T,Z) matching id_scm tables.
        // Simplified: enumerate all (Z,T,Y) with multiplicity proportional to joint.
        let mut b = CausalSchemaBuilder::new();
        for name in ["t", "y", "z"] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();

        // Joint from: P(Z=0)=P(Z=1)=0.5
        // P(T=1|Z=0)=0.4, P(T=1|Z=1)=0.7 (arbitrary; only Y|T,Z and P(Z) matter for do)
        // E[Y|T,Z] as in id_scm: (1,0)->0.8, (1,1)->0.6, (0,0)->0.3, (0,1)->0.2
        // Use 200 rows.
        let mut t_vals = Vec::new();
        let mut y_vals = Vec::new();
        let mut z_vals = Vec::new();
        let combos = [
            // (z, t, y, count) — counts encode joint
            (0.0, 0.0, 0.0, 21), // P(Y=0|T=0,Z=0)=0.7 → among T=0,Z=0
            (0.0, 0.0, 1.0, 9),  // 0.3
            (0.0, 1.0, 0.0, 4),  // P(Y=0|T=1,Z=0)=0.2
            (0.0, 1.0, 1.0, 16), // 0.8
            (1.0, 0.0, 0.0, 12), // P(Y=0|T=0,Z=1)=0.8
            (1.0, 0.0, 1.0, 3),  // 0.2
            (1.0, 1.0, 0.0, 14), // P(Y=0|T=1,Z=1)=0.4
            (1.0, 1.0, 1.0, 21), // 0.6
        ];
        // Normalize Z marginal toward 0.5 by the counts above:
        // Z=0: 21+9+4+16=50, Z=1: 12+3+14+21=50. Good.
        for (z, t, y, count) in combos {
            for _ in 0..count {
                z_vals.push(z);
                t_vals.push(t);
                y_vals.push(y);
            }
        }
        let n = t_vals.len();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t_vals),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y_vals),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(z_vals),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        TabularData::new(storage)
    }

    #[test]
    fn plug_in_matches_known_interventional_mean() {
        let mut dag = Dag::with_variables(3);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let z = DenseNodeId::from_raw(2);
        dag.insert_directed(z, t).unwrap();
        dag.insert_directed(z, y).unwrap();
        dag.insert_directed(t, y).unwrap();

        let id = IdIdentifier::new();
        let prep = id.prepare_dag(&dag).unwrap();
        let query = InterventionalDistributionQuery::new(
            VariableId::from_raw(1),
            [Intervention::set(VariableId::from_raw(0), f(1.0))],
        );
        let cq = causal_core::CausalQuery::Distribution(query.clone());
        let mut ws = IdentificationWorkspace::default();
        let id_res = id.identify(&prep, &cq, &mut ws).unwrap();
        assert_eq!(id_res.status, IdentificationStatus::NonparametricallyIdentified);

        let data = binary_confounding_table();
        let est = FunctionalDistribution::new();
        let prepared = est
            .prepare(
                &data,
                &query,
                &id_res.estimands[0],
                &id_res.arena,
                id_res.required_assumptions.clone(),
            )
            .unwrap();
        let mut ews = FunctionalDistributionWorkspace::default();
        let out = est.estimate(&prepared, &[], &mut ews, &ExecutionContext::for_tests(0)).unwrap();
        // E[Y|do(T=1)] = 0.7
        assert!((out.mean - 0.7).abs() < 0.05, "mean={} atoms={:?}", out.mean, out.atoms);
        let mass: f64 = out.atoms.iter().map(|a| a.probability).sum();
        assert!((mass - 1.0).abs() < 1e-6, "mass={mass}");
    }

    #[test]
    fn plug_in_bootstrap_se_is_finite() {
        let mut dag = Dag::with_variables(3);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let z = DenseNodeId::from_raw(2);
        dag.insert_directed(z, t).unwrap();
        dag.insert_directed(z, y).unwrap();
        dag.insert_directed(t, y).unwrap();

        let id = IdIdentifier::new();
        let prep = id.prepare_dag(&dag).unwrap();
        let query = InterventionalDistributionQuery::new(
            VariableId::from_raw(1),
            [Intervention::set(VariableId::from_raw(0), f(1.0))],
        );
        let cq = causal_core::CausalQuery::Distribution(query.clone());
        let mut ws = IdentificationWorkspace::default();
        let id_res = id.identify(&prep, &cq, &mut ws).unwrap();

        let data = binary_confounding_table();
        let est =
            FunctionalDistribution { bootstrap_replicates: 40, ..FunctionalDistribution::new() };
        let prepared = est
            .prepare(
                &data,
                &query,
                &id_res.estimands[0],
                &id_res.arena,
                id_res.required_assumptions.clone(),
            )
            .unwrap();
        let mut ews = FunctionalDistributionWorkspace::default();
        let out = est.estimate(&prepared, &[], &mut ews, &ExecutionContext::for_tests(7)).unwrap();
        let se = out.se_bootstrap.expect("bootstrap SE");
        assert!(se.is_finite() && se > 0.0, "se={se}");
    }
}
