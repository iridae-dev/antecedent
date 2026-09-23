//! Free/bound variables, substitution, and certificate binding.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use antecedent_core::{RegimeId, VariableId};

use crate::{CausalExprArena, DomainRef, ExprError, ExprId, ExprNode, InterventionAssignment};

/// One leaf's population / regime binding.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct LeafBinding {
    /// Population key.
    pub population: Arc<str>,
    /// Catalog regime, if any.
    pub regime: Option<RegimeId>,
}

/// Everything a provider reads from one distribution leaf: which factor it is, and which
/// population and regime answer it.
///
/// Unlike [`LeafBinding`] this ties each population/regime to its own factor, so exchanging the
/// populations of two leaves changes the signature set.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct LeafSignature {
    /// Factor variables.
    pub variables: Arc<[VariableId]>,
    /// Conditioning variables.
    pub conditioned_on: Arc<[VariableId]>,
    /// Hard (or symbolic) intervention assignments.
    pub intervention: Arc<[InterventionAssignment]>,
    /// Observational vs interventional domain.
    pub domain: DomainRef,
    /// Population and regime this leaf cites.
    pub binding: LeafBinding,
}

impl CausalExprArena {
    /// Free variables of `id` (intervention-bound conditioners are not free).
    #[must_use]
    pub fn free_variables(&self, id: ExprId) -> Vec<VariableId> {
        crate::simplify::free_vars(self, id, &mut HashMap::new()).to_vec()
    }

    /// Scope-preserving substitution of variable ids.
    ///
    /// Bound sum / integral variables are not replaced; kernel parameters remain free. Capture and
    /// conflicting assignments refuse.
    ///
    /// # Errors
    ///
    /// [`ExprError::CaptureOrConflict`] or [`ExprError::MissingBinding`].
    pub fn substitute(
        &mut self,
        id: ExprId,
        replacements: &[(VariableId, VariableId)],
    ) -> Result<ExprId, ExprError> {
        let mut map = HashMap::new();
        for &(from, to) in replacements {
            if map.insert(from, to).is_some_and(|previous| previous != to) {
                return Err(ExprError::CaptureOrConflict);
            }
        }
        let free = self.free_variables(id);
        for &(from, _) in replacements {
            if !free.contains(&from) {
                return Err(ExprError::MissingBinding);
            }
        }
        let mut images = BTreeSet::new();
        for variable in &free {
            if !images.insert(*map.get(variable).unwrap_or(variable)) {
                return Err(ExprError::CaptureOrConflict);
            }
        }
        substitute_rec(self, id, &map, &BTreeSet::new())
    }

    /// Refuse when free variables disagree with the certified target.
    ///
    /// # Errors
    ///
    /// [`ExprError::FreeVariableMismatch`].
    pub fn require_free_variables(
        &self,
        id: ExprId,
        expected: &[VariableId],
    ) -> Result<(), ExprError> {
        let mut got = self.free_variables(id);
        got.sort_by_key(|v| v.raw());
        let mut exp = expected.to_vec();
        exp.sort_by_key(|v| v.raw());
        exp.dedup();
        if got == exp { Ok(()) } else { Err(ExprError::FreeVariableMismatch) }
    }

    /// Reachable distribution leaves, visiting shared subexpressions only once.
    #[must_use]
    pub fn distribution_leaves(&self, id: ExprId) -> Vec<ExprId> {
        let mut out = Vec::new();
        collect_leaves(self, id, &mut out, &mut std::collections::HashSet::new());
        out
    }

    /// Leaf population/regime bindings in stable order.
    #[must_use]
    pub fn leaf_bindings(&self, id: ExprId) -> Vec<LeafBinding> {
        let mut out: Vec<_> = self
            .distribution_leaves(id)
            .into_iter()
            .map(|id| {
                let ExprNode::Distribution { population, regime, .. } = self.node(id) else {
                    unreachable!()
                };
                LeafBinding { population: Arc::from(self.population(*population)), regime: *regime }
            })
            .collect();
        out.sort_by(|a, b| {
            (&*a.population, a.regime.map(RegimeId::raw))
                .cmp(&(&*b.population, b.regime.map(RegimeId::raw)))
        });
        out.dedup();
        out
    }

    /// Per-leaf signatures of the distinct distribution leaves, in traversal order.
    #[must_use]
    pub fn leaf_signatures(&self, id: ExprId) -> Vec<LeafSignature> {
        self.distribution_leaves(id)
            .into_iter()
            .map(|leaf| {
                let ExprNode::Distribution {
                    variables,
                    conditioned_on,
                    intervention,
                    domain,
                    population,
                    regime,
                } = self.node(leaf)
                else {
                    unreachable!()
                };
                LeafSignature {
                    variables: Arc::from(self.var_set(*variables)),
                    conditioned_on: Arc::from(self.var_set(*conditioned_on)),
                    intervention: Arc::from(self.intervention_assignments(*intervention)),
                    domain: *domain,
                    binding: LeafBinding {
                        population: Arc::from(self.population(*population)),
                        regime: *regime,
                    },
                }
            })
            .collect()
    }

    /// Bind a certificate to a lowered expression by its set of distinct
    /// `(population, regime)` pairs.
    ///
    /// This does not tie a pair to a factor, so `P^s(y|x)·P^t(x)` and `P^t(y|x)·P^s(x)` bind
    /// alike; use [`Self::bind_leaf_certificate`] when a swap between leaves must fail.
    ///
    /// # Errors
    ///
    /// [`ExprError::CertificateBindFailed`] when the pair set disagrees.
    pub fn bind_certificate(&self, id: ExprId, expected: &[LeafBinding]) -> Result<(), ExprError> {
        let got = self.leaf_bindings(id);
        let mut expected = expected.to_vec();
        expected.sort_by(|a, b| {
            (&*a.population, a.regime.map(RegimeId::raw))
                .cmp(&(&*b.population, b.regime.map(RegimeId::raw)))
        });
        expected.dedup();
        if got == expected { Ok(()) } else { Err(ExprError::CertificateBindFailed) }
    }

    /// Bind a certificate to a lowered expression leaf by leaf: every distinct leaf must appear
    /// in `expected` with the same factor identity *and* the same population and regime, and no
    /// expected signature may be left over. A population or regime swap between leaves fails.
    ///
    /// # Errors
    ///
    /// [`ExprError::CertificateBindFailed`] when the signature sets disagree.
    pub fn bind_leaf_certificate(
        &self,
        id: ExprId,
        expected: &[LeafSignature],
    ) -> Result<(), ExprError> {
        let got = self.leaf_signatures(id);
        let contains = |set: &[LeafSignature], item: &LeafSignature| set.contains(item);
        if got.iter().all(|s| contains(expected, s)) && expected.iter().all(|s| contains(&got, s)) {
            Ok(())
        } else {
            Err(ExprError::CertificateBindFailed)
        }
    }
}

fn substitute_rec(
    arena: &mut CausalExprArena,
    id: ExprId,
    map: &HashMap<VariableId, VariableId>,
    bound: &BTreeSet<VariableId>,
) -> Result<ExprId, ExprError> {
    match arena.node(id).clone() {
        ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            regime,
        } => {
            let vars = rewrite_var_set(arena, variables, map, bound)?;
            let cond = rewrite_var_set(arena, conditioned_on, map, bound)?;
            let interv = rewrite_intervention(arena, intervention, map, bound)?;
            Ok(arena.intern(ExprNode::Distribution {
                variables: vars,
                conditioned_on: cond,
                intervention: interv,
                domain,
                population,
                regime,
            }))
        }
        ExprNode::Kernel { body, bound: kernel_bound, population, regime } => {
            let new_bound = rewrite_var_set(arena, kernel_bound, map, bound)?;
            let body = substitute_rec(arena, body, map, bound)?;
            Ok(arena.intern(ExprNode::Kernel { body, bound: new_bound, population, regime }))
        }
        ExprNode::Product(list) => {
            let children: Vec<ExprId> = arena.list(list).to_vec();
            let mut out = Vec::with_capacity(children.len());
            for child in children {
                out.push(substitute_rec(arena, child, map, bound)?);
            }
            let list = arena.intern_list(out);
            Ok(arena.intern(ExprNode::Product(list)))
        }
        ExprNode::SumOut { variables, expr } | ExprNode::IntegralOut { variables, expr } => {
            let mut next_bound = bound.clone();
            next_bound.extend(arena.var_set(variables).iter().copied());
            let body = substitute_rec(arena, expr, map, &next_bound)?;
            let node = match arena.node(id) {
                ExprNode::SumOut { .. } => ExprNode::SumOut { variables, expr: body },
                _ => ExprNode::IntegralOut { variables, expr: body },
            };
            Ok(arena.intern(node))
        }
        ExprNode::Ratio { numerator, denominator } => {
            let n = substitute_rec(arena, numerator, map, bound)?;
            let d = substitute_rec(arena, denominator, map, bound)?;
            Ok(arena.intern(ExprNode::Ratio { numerator: n, denominator: d }))
        }
        ExprNode::Expectation { function, distribution } => {
            let mut next_bound = bound.clone();
            next_bound.insert(function.variable());
            let dist = substitute_rec(arena, distribution, map, &next_bound)?;
            Ok(arena.intern(ExprNode::Expectation { function, distribution: dist }))
        }
        ExprNode::Contrast { left, right, op } => {
            let l = substitute_rec(arena, left, map, bound)?;
            let r = substitute_rec(arena, right, map, bound)?;
            Ok(arena.intern(ExprNode::Contrast { left: l, right: r, op }))
        }
    }
}

fn rewrite_var_set(
    arena: &mut CausalExprArena,
    id: crate::VarSetId,
    map: &HashMap<VariableId, VariableId>,
    bound: &BTreeSet<VariableId>,
) -> Result<crate::VarSetId, ExprError> {
    let mut out = Vec::new();
    let mut changed = false;
    for &v in arena.var_set(id) {
        if bound.contains(&v) {
            out.push(v);
            continue;
        }
        if let Some(&next) = map.get(&v) {
            if bound.contains(&next) {
                return Err(ExprError::CaptureOrConflict);
            }
            changed |= next != v;
            out.push(next);
        } else {
            out.push(v);
        }
    }
    if !changed {
        return Ok(id);
    }
    Ok(arena.intern_var_set(out))
}

fn rewrite_intervention(
    arena: &mut CausalExprArena,
    id: crate::InterventionSetId,
    map: &HashMap<VariableId, VariableId>,
    bound: &BTreeSet<VariableId>,
) -> Result<crate::InterventionSetId, ExprError> {
    let assignments: Vec<_> = arena.intervention_assignments(id).to_vec();
    let mut out = Vec::new();
    let mut changed = false;
    for a in assignments {
        let variable = if bound.contains(&a.variable) {
            a.variable
        } else if let Some(&next) = map.get(&a.variable) {
            if bound.contains(&next) {
                return Err(ExprError::CaptureOrConflict);
            }
            changed |= next != a.variable;
            next
        } else {
            a.variable
        };
        if out
            .iter()
            .any(|b: &crate::InterventionAssignment| b.variable == variable && b.value != a.value)
        {
            return Err(ExprError::CaptureOrConflict);
        }
        out.push(crate::InterventionAssignment { variable, value: a.value });
    }
    if !changed {
        return Ok(id);
    }
    arena.try_intern_intervention_assignments(out)
}

fn collect_leaves(
    arena: &CausalExprArena,
    id: ExprId,
    out: &mut Vec<ExprId>,
    visited: &mut std::collections::HashSet<ExprId>,
) {
    if !visited.insert(id) {
        return;
    }
    match arena.node(id) {
        ExprNode::Distribution { .. } => out.push(id),
        ExprNode::Kernel { body, .. } => collect_leaves(arena, *body, out, visited),
        ExprNode::Product(list) => {
            for &child in arena.list(*list) {
                collect_leaves(arena, child, out, visited);
            }
        }
        ExprNode::SumOut { expr, .. } | ExprNode::IntegralOut { expr, .. } => {
            collect_leaves(arena, *expr, out, visited);
        }
        ExprNode::Ratio { numerator, denominator } => {
            collect_leaves(arena, *numerator, out, visited);
            collect_leaves(arena, *denominator, out, visited);
        }
        ExprNode::Expectation { distribution, .. } => {
            collect_leaves(arena, *distribution, out, visited)
        }
        ExprNode::Contrast { left, right, .. } => {
            collect_leaves(arena, *left, out, visited);
            collect_leaves(arena, *right, out, visited);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_parameters_are_free_and_rename_with_the_body() {
        let mut arena = CausalExprArena::new();
        let v = VariableId::from_raw;
        let y = arena.intern_var_set([v(1)]);
        let x = arena.intern_var_set([v(0)]);
        let intervention = arena.empty_intervention_set();
        let body = arena.intern_distribution(y, x, intervention, DomainRef::Observational);
        let kernel = arena.intern_kernel(body, x, "intermediate", None);
        assert_eq!(arena.free_variables(kernel), vec![v(0), v(1)]);
        let renamed = arena.substitute(kernel, &[(v(0), v(2))]).unwrap();
        assert_eq!(arena.free_variables(renamed), vec![v(1), v(2)]);
        assert_eq!(arena.leaf_bindings(kernel), arena.leaf_bindings(body));
        assert!(arena.substitute(kernel, &[(v(0), v(2)), (v(0), v(3))]).is_err());
        assert!(arena.substitute(kernel, &[(v(0), v(1))]).is_err());
        let sum = arena.intern(ExprNode::SumOut { variables: x, expr: kernel });
        assert!(arena.substitute(sum, &[(v(1), v(0))]).is_err());
    }

    #[test]
    fn leaf_certificate_detects_a_population_swap_between_leaves() {
        let mut arena = CausalExprArena::new();
        let v = VariableId::from_raw;
        let y = arena.intern_var_set([v(1)]);
        let x = arena.intern_var_set([v(0)]);
        let empty = arena.empty_var_set();
        let none = arena.empty_intervention_set();
        let regime = Some(RegimeId::from_raw(0));
        let leaf = |arena: &mut CausalExprArena, vars, given, population: &str| {
            arena
                .intern_distribution_tagged(
                    vars,
                    given,
                    none,
                    DomainRef::Observational,
                    population,
                    regime,
                    None,
                )
                .unwrap()
        };
        // Σ_x P^s(y|x) P^t(x)  versus  Σ_x P^t(y|x) P^s(x).
        let (ys, xt) = (leaf(&mut arena, y, x, "s"), leaf(&mut arena, x, empty, "t"));
        let (yt, xs) = (leaf(&mut arena, y, x, "t"), leaf(&mut arena, x, empty, "s"));
        let list = arena.intern_list([ys, xt]);
        let a = arena.intern(ExprNode::Product(list));
        let list = arena.intern_list([yt, xs]);
        let b = arena.intern(ExprNode::Product(list));
        // The pair-set binding cannot tell them apart ...
        let pairs = arena.leaf_bindings(a);
        assert_eq!(arena.bind_certificate(b, &pairs), Ok(()));
        // ... the per-leaf binding can, in both directions.
        let signatures = arena.leaf_signatures(a);
        assert_eq!(arena.bind_leaf_certificate(a, &signatures), Ok(()));
        assert_eq!(
            arena.bind_leaf_certificate(b, &signatures),
            Err(ExprError::CertificateBindFailed)
        );
        let swapped = arena.leaf_signatures(b);
        assert_eq!(arena.bind_leaf_certificate(a, &swapped), Err(ExprError::CertificateBindFailed));
    }
}
