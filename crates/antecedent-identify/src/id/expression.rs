//! Construct ID distribution factors and causal expressions.

use super::{
    Arc, Assign, BitSet, CausalExprArena, DenseNodeId, DistCtx, DomainRef, ExprId, ExprNode,
    Factor, IdentificationError, InterventionAssignment, Law, PreparedAdmg, VariableId,
};

impl DistCtx {
    /// Marginalize the current distribution over `removed` (line 2).
    pub(super) fn marginalize(&self, removed: &BitSet) -> Self {
        match self {
            // A marginal of the observational marginal is still a marginal.
            Self::Marginal => Self::Marginal,
            Self::CFactor(law) => {
                let mut sumset = law.sumset.clone();
                sumset.union_with(removed);
                Self::CFactor(Arc::new(Law { sumset, factors: law.factors.clone() }))
            }
        }
    }
}

impl Law {
    /// Conditionals `P′(vi | v_π^{(i-1)})` of this law over `v`, for `vi ∈ s`,
    /// in topological order.
    ///
    /// With an empty sumset every factor after `vi` sums to one without
    /// touching the earlier factors, so `∑_{later} P′ = ∏_{j ≤ i} f_j` and the
    /// conditional is `vi`'s own factor. Once line 2 has summed variables out
    /// of the product that telescope no longer holds node-wise, and the
    /// conditional is formed from its definition,
    /// `∑_{later} P′ / ∑_{vi} ∑_{later} P′`.
    pub(super) fn conditionals(
        &self,
        prepared: &PreparedAdmg,
        s: &BitSet,
        v: &BitSet,
    ) -> Result<Vec<Factor>, IdentificationError> {
        if !self.sumset.any() {
            let kept: Vec<Factor> =
                self.factors.iter().filter(|f| s.contains(f.var())).cloned().collect();
            if kept.len() != s.count_ones() {
                return Err(IdentificationError::InvariantViolated {
                    message: "general ID: current law does not cover the requested C-component",
                });
            }
            return Ok(kept);
        }
        let mut out = Vec::new();
        let mut later = v.clone();
        for &vi in prepared.topo() {
            if !v.contains(vi) {
                continue;
            }
            later.remove(vi);
            if s.contains(vi) {
                let mut sumset = self.sumset.clone();
                sumset.union_with(&later);
                let num = Arc::new(Self { sumset, factors: self.factors.clone() });
                out.push(Factor::Conditional { var: vi, num });
            }
        }
        Ok(out)
    }

    /// Emit `∑_{sumset} ∏ factors`.
    ///
    /// `bound` holds the variables already bound by enclosing sums of this
    /// emission. `do(·)` labels are applied only to factors whose assigned
    /// variable is *free* — a bound occurrence is the sum's dummy variable,
    /// not the do-value.
    pub(super) fn emit(
        &self,
        prepared: &PreparedAdmg,
        bound: &BitSet,
        arena: &mut CausalExprArena,
        assign: &Assign,
    ) -> Result<ExprId, IdentificationError> {
        let mut bound = bound.clone();
        bound.union_with(&self.sumset);
        let mut exprs = Vec::with_capacity(self.factors.len());
        for factor in &self.factors {
            exprs.push(factor.emit(prepared, &bound, arena, assign)?);
        }
        let product = if exprs.len() == 1 {
            exprs[0]
        } else {
            let list = arena.intern_list(exprs);
            arena.intern(ExprNode::Product(list))
        };
        if self.sumset.any() {
            let vs = intern_nodes(prepared, &self.sumset, arena)?;
            Ok(arena.intern(ExprNode::SumOut { variables: vs, expr: product }))
        } else {
            Ok(product)
        }
    }
}

impl Factor {
    pub(super) fn emit(
        &self,
        prepared: &PreparedAdmg,
        bound: &BitSet,
        arena: &mut CausalExprArena,
        assign: &Assign,
    ) -> Result<ExprId, IdentificationError> {
        match self {
            Self::Observed { var, cond } => {
                let effective_assign: Assign =
                    assign.iter().filter(|(t, _)| !bound.contains(*t)).cloned().collect();
                let var_i = prepared.dense_to_var(*var)?;
                let vars = arena.intern_var_set([var_i]);
                let cond_vars: Result<Vec<_>, _> =
                    cond.iter().map(|d| prepared.dense_to_var(d)).collect();
                let cond_vars = cond_vars?;
                let conditioned_on = arena.intern_var_set(cond_vars.clone());
                let (intervention, domain) =
                    intervention_for_factor(arena, prepared, &effective_assign, *var, &cond_vars)?;
                Ok(arena.intern_distribution(vars, conditioned_on, intervention, domain))
            }
            Self::Conditional { var, num } => {
                let numerator = num.emit(prepared, bound, arena, assign)?;
                // The denominator is literally `∑_{var}` of the numerator node,
                // so an evaluator can recognize a conditional on a null event
                // (0/0 with a bounded extension) instead of a generic 0/0. Only
                // when `var` carries a free do-label must the body be re-emitted
                // with `var` bound, since the label would otherwise pin the
                // summation variable.
                let labelled = !bound.contains(*var) && assign.iter().any(|(t, _)| t == var);
                let body = if labelled {
                    let mut inner = bound.clone();
                    inner.insert(*var);
                    num.emit(prepared, &inner, arena, assign)?
                } else {
                    numerator
                };
                let variables = arena.intern_var_set([prepared.dense_to_var(*var)?]);
                let denominator = arena.intern(ExprNode::SumOut { variables, expr: body });
                Ok(arena.intern(ExprNode::Ratio { numerator, denominator }))
            }
        }
    }
}

/// Marginal `∑_{v\y}` of the current distribution (lines 1 and degenerate).
pub(super) fn dist_marginal(
    prepared: &PreparedAdmg,
    dist: &DistCtx,
    y: &BitSet,
    v: &BitSet,
    arena: &mut CausalExprArena,
    assign: &Assign,
) -> Result<ExprId, IdentificationError> {
    match dist {
        DistCtx::Marginal => observational_marginal(prepared, y, v, arena, assign),
        DistCtx::CFactor(law) => {
            // Sum over everything in v except y, plus the carried sumset.
            let mut sumset = v.clone();
            sumset.difference_with(y);
            sumset.union_with(&law.sumset);
            let marginal = Law { sumset, factors: law.factors.clone() };
            marginal.emit(prepared, &BitSet::with_len(v.bit_len()), arena, assign)
        }
    }
}

fn intern_nodes(
    prepared: &PreparedAdmg,
    nodes: &BitSet,
    arena: &mut CausalExprArena,
) -> Result<antecedent_expr::VarSetId, IdentificationError> {
    let vars: Result<Vec<_>, _> = nodes.iter().map(|d| prepared.dense_to_var(d)).collect();
    Ok(arena.intern_var_set(vars?))
}

/// `∏ factors`: `None` for no factors, the factor itself for one, a `Product` otherwise.
fn product_of(arena: &mut CausalExprArena, factors: Vec<ExprId>) -> Option<ExprId> {
    match factors.as_slice() {
        [] => None,
        [only] => Some(*only),
        _ => {
            let list = arena.intern_list(factors);
            Some(arena.intern(ExprNode::Product(list)))
        }
    }
}

/// `∑_{nodes} body`, or `body` itself when `nodes` is empty.
pub(super) fn sum_out_if_any(
    prepared: &PreparedAdmg,
    nodes: &BitSet,
    body: ExprId,
    arena: &mut CausalExprArena,
) -> Result<ExprId, IdentificationError> {
    if nodes.any() {
        let variables = intern_nodes(prepared, nodes, arena)?;
        Ok(arena.intern(ExprNode::SumOut { variables, expr: body }))
    } else {
        Ok(body)
    }
}

fn observational_marginal(
    prepared: &PreparedAdmg,
    y: &BitSet,
    v: &BitSet,
    arena: &mut CausalExprArena,
    assign: &Assign,
) -> Result<ExprId, IdentificationError> {
    // Tian / Shpitser–Pearl: P(V) = ∏_{S ∈ C(G[V])} Q[S], Q[S] = ∏_{Vi∈S} P(Vi | V^π_<i).
    // On a DAG, C-components are singletons and this reduces to the usual Markov product
    // (conditioning on extra predecessors is redundant given pa(Vi)). On an ADMG with
    // bidirected edges, ∏ P(vi | pa(vi)) is not the observational joint.
    let comps = prepared.c_components(v);
    let factors = if comps.is_empty() {
        markov_product(prepared, v, arena, assign)?
    } else if comps.len() == 1 {
        q_component_product(prepared, &comps[0], v, arena, assign)?
    } else {
        let mut parts = Vec::with_capacity(comps.len());
        for s in &comps {
            parts.push(q_component_product(prepared, s, v, arena, assign)?);
        }
        product_of(arena, parts).ok_or(IdentificationError::InvariantViolated {
            message: "general ID: observational marginal has no C-component factors",
        })?
    };
    let mut sum_vars = v.clone();
    sum_vars.difference_with(y);
    sum_out_if_any(prepared, &sum_vars, factors, arena)
}

pub(super) fn c_component_expression(
    prepared: &PreparedAdmg,
    s: &BitSet,
    y: &BitSet,
    v: &BitSet,
    arena: &mut CausalExprArena,
    assign: &Assign,
) -> Result<ExprId, IdentificationError> {
    // ∑_{s\y} ∏_{Vi∈S} P(vi | v^{π}_{<i})
    let product = q_component_product(prepared, s, v, arena, assign)?;
    let mut sum_vars = s.clone();
    sum_vars.difference_with(y);
    sum_out_if_any(prepared, &sum_vars, product, arena)
}

fn q_component_product(
    prepared: &PreparedAdmg,
    s: &BitSet,
    v: &BitSet,
    arena: &mut CausalExprArena,
    assign: &Assign,
) -> Result<ExprId, IdentificationError> {
    let empty_i = arena.empty_intervention_set();
    let mut factors = Vec::new();
    // Variables of the nodes already passed in topological order, maintained
    // incrementally so each factor interns from a slice.
    let mut preceding: Vec<VariableId> = Vec::new();
    for &vi in prepared.topo() {
        if !v.contains(vi) {
            continue;
        }
        let var_i = prepared.dense_to_var(vi)?;
        if s.contains(vi) {
            let vars = arena.intern_var_set([var_i]);
            let conditioned_on = arena.intern_var_set(preceding.iter().copied());
            let (intervention, domain) =
                intervention_for_factor(arena, prepared, assign, vi, &preceding)?;
            factors.push(arena.intern_distribution(vars, conditioned_on, intervention, domain));
        }
        preceding.push(var_i);
    }
    if let Some(product) = product_of(arena, factors) {
        return Ok(product);
    }
    let y = intern_nodes(prepared, s, arena)?;
    let empty = arena.empty_var_set();
    Ok(arena.intern_distribution(y, empty, empty_i, DomainRef::Observational))
}

fn markov_product(
    prepared: &PreparedAdmg,
    v: &BitSet,
    arena: &mut CausalExprArena,
    assign: &Assign,
) -> Result<ExprId, IdentificationError> {
    let empty_i = arena.empty_intervention_set();
    let mut factors = Vec::new();
    for &vi in prepared.topo() {
        if !v.contains(vi) {
            continue;
        }
        let var_i = prepared.dense_to_var(vi)?;
        let vars = arena.intern_var_set([var_i]);
        let parents: Result<Vec<_>, _> = prepared
            .admg()
            .parents(vi)
            .iter()
            .copied()
            .filter(|p| v.contains(*p))
            .map(|p| prepared.dense_to_var(p))
            .collect();
        let parents = parents?;
        let conditioned_on = arena.intern_var_set(parents.clone());
        let (intervention, domain) =
            intervention_for_factor(arena, prepared, assign, vi, &parents)?;
        factors.push(arena.intern_distribution(vars, conditioned_on, intervention, domain));
    }
    if let Some(product) = product_of(arena, factors) {
        return Ok(product);
    }
    let empty = arena.empty_var_set();
    Ok(arena.intern_distribution(empty, empty, empty_i, DomainRef::Observational))
}

/// Bake `do(T=t)` into the factor that generates `T`, or into factors that condition on `T`.
fn intervention_for_factor(
    arena: &mut CausalExprArena,
    prepared: &PreparedAdmg,
    assign: &Assign,
    vi: DenseNodeId,
    conditioned_on: &[VariableId],
) -> Result<(antecedent_expr::InterventionSetId, DomainRef), IdentificationError> {
    let mut touching = Vec::new();
    for (t, val) in assign.iter() {
        let t_var = prepared.dense_to_var(*t)?;
        if *t == vi || conditioned_on.iter().any(|&v| v == t_var) {
            touching.push(InterventionAssignment { variable: t_var, value: val.clone() });
        }
    }
    if touching.is_empty() {
        return Ok((arena.empty_intervention_set(), DomainRef::Observational));
    }
    let intervention = crate::intervention_support::intern_assignments(arena, touching)?;
    Ok((intervention, DomainRef::Interventional))
}
