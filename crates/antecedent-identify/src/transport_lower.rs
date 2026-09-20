//! Lower [`TransportFormula`] views into the shared expression arena.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::VariableId;
use antecedent_expr::{
    CausalExprArena, DerivationMeta, DomainRef, ExprId, ExprNode, OutcomeExprId,
};

use crate::transport::{PopulationFactor, TransportCertificate, TransportFormula};

/// Lower a transport formula into the existing hash-consed arena.
///
/// `Direct` and supplied laws become tagged [`antecedent_expr::ExprNode::Distribution`]
/// leaves. Nested conditionals (nonempty conditioning, no experiment) become
/// [`antecedent_expr::ExprNode::Kernel`] wrappers. Interventions stay symbolic
/// (`NaN` placeholders) — they are not rewritten into concrete assignments.
#[must_use]
pub fn lower_transport_formula(arena: &mut CausalExprArena, formula: &TransportFormula) -> ExprId {
    match formula {
        TransportFormula::Direct(factor) => lower_factor(arena, factor, false),
        TransportFormula::Standardize { over, source_response, target_law } => {
            let source = lower_factor(arena, source_response, false);
            let law = lower_factor(arena, target_law, false);
            let product = intern_product(arena, [source, law]);
            intern_sum_out(arena, over, product)
        }
        TransportFormula::RecursiveFactorization { sum_out, factors } => {
            let lowered: Vec<ExprId> =
                factors.iter().map(|factor| lower_factor(arena, factor, true)).collect();
            let product = intern_product(arena, lowered);
            intern_sum_out(arena, sum_out, product)
        }
    }
}

/// Lower a formula and wrap it as `E[Y | ·]` for a mean query.
#[must_use]
pub fn lower_transport_mean(
    arena: &mut CausalExprArena,
    formula: &TransportFormula,
    outcome: VariableId,
) -> ExprId {
    let distribution = lower_transport_formula(arena, formula);
    arena.intern(ExprNode::Expectation { function: OutcomeExprId::identity(outcome), distribution })
}

/// Stamp the identification certificate onto the lowered root (off the hash).
pub fn bind_transport_derivation(
    arena: &mut CausalExprArena,
    root: ExprId,
    certificate: &TransportCertificate,
    note: impl Into<Arc<str>>,
) {
    fn visit(
        arena: &mut CausalExprArena,
        id: ExprId,
        seen: &mut std::collections::HashSet<ExprId>,
    ) {
        if !seen.insert(id) {
            return;
        }
        let (rule, parents) = match arena.node(id).clone() {
            ExprNode::Distribution { .. } => ("transport.evidence", vec![]),
            ExprNode::Kernel { body, .. } => ("transport.kernel", vec![body]),
            ExprNode::Product(list) => ("transport.product", arena.list(list).to_vec()),
            ExprNode::SumOut { expr, .. } => ("transport.marginalize", vec![expr]),
            ExprNode::IntegralOut { expr, .. } => ("transport.integrate", vec![expr]),
            ExprNode::Ratio { numerator, denominator } => {
                ("transport.ratio", vec![numerator, denominator])
            }
            ExprNode::Expectation { distribution, .. } => {
                ("transport.expectation", vec![distribution])
            }
            ExprNode::Contrast { left, right, .. } => ("transport.contrast", vec![left, right]),
        };
        for &parent in &parents {
            visit(arena, parent, seen);
        }
        let evidence = arena
            .leaf_bindings(id)
            .iter()
            .filter_map(|b| b.regime)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        arena.set_derivation(
            id,
            DerivationMeta {
                rule: Arc::from(rule),
                output: Some(id),
                input: parents.first().copied(),
                parents: parents.into(),
                evidence: evidence.into(),
                ..DerivationMeta::default()
            },
        );
    }
    visit(arena, root, &mut std::collections::HashSet::new());
    let mut derivation = arena.derivation(root).cloned().unwrap_or_default();
    derivation.rule = Arc::clone(&certificate.rule);
    derivation.note = Some(note.into());
    derivation.premises = Arc::clone(&certificate.premises);
    derivation.graph_operation = Some(Arc::from(match certificate.rule.as_ref() {
        "transport.sid.direct" => "selection_ancestry",
        "transport.sid.target_g_formula" | "transport.sid.singleton_c_components" => {
            "truncated_factorization"
        }
        _ => "treatment_mutilation_and_selection_separation",
    }));
    arena.set_derivation(root, derivation);
}

fn lower_factor(
    arena: &mut CausalExprArena,
    factor: &PopulationFactor,
    wrap_intermediate: bool,
) -> ExprId {
    let variables = arena.intern_var_set(factor.variables.iter().copied());
    let conditioned_on = arena.intern_var_set(factor.conditioned_on.iter().copied());
    let intervention = arena.intern_intervention_set(factor.interventions.iter().copied());
    let domain = if factor.interventions.is_empty() {
        DomainRef::Observational
    } else {
        DomainRef::Interventional
    };
    let dist = arena
        .intern_distribution_tagged(
            variables,
            conditioned_on,
            intervention,
            domain,
            Arc::clone(&factor.population),
            factor.regime,
            None,
        )
        .expect("no regime kind to mismatch");
    let intermediate =
        wrap_intermediate && !factor.conditioned_on.is_empty() && factor.interventions.is_empty();
    if intermediate {
        arena.intern_kernel(dist, conditioned_on, Arc::clone(&factor.population), factor.regime)
    } else {
        dist
    }
}

fn intern_product(
    arena: &mut CausalExprArena,
    children: impl IntoIterator<Item = ExprId>,
) -> ExprId {
    let children: Vec<ExprId> = children.into_iter().collect();
    match children.as_slice() {
        [] => {
            let empty = arena.empty_var_set();
            let empty_i = arena.empty_intervention_set();
            arena.intern_distribution(empty, empty, empty_i, DomainRef::Observational)
        }
        [only] => *only,
        _ => {
            let list = arena.intern_list(children);
            arena.intern(ExprNode::Product(list))
        }
    }
}

fn intern_sum_out(arena: &mut CausalExprArena, over: &[VariableId], body: ExprId) -> ExprId {
    if over.is_empty() {
        return body;
    }
    let variables = arena.intern_var_set(over.iter().copied());
    arena.intern(ExprNode::SumOut { variables, expr: body })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::PopulationFactor;
    use antecedent_core::{RegimeId, Value};
    use antecedent_expr::{ExprError, ExprNode, InterventionAssignment, LeafBinding};

    fn factor(pop: &str, vars: &[u32], cond: &[u32], interv: &[u32]) -> PopulationFactor {
        PopulationFactor {
            regime: None,
            population: Arc::from(pop),
            variables: vars.iter().copied().map(VariableId::from_raw).collect(),
            conditioned_on: cond.iter().copied().map(VariableId::from_raw).collect(),
            interventions: interv.iter().copied().map(VariableId::from_raw).collect(),
        }
    }

    #[test]
    fn rename_preserves_interned_identity_of_structure() {
        let mut arena = CausalExprArena::new();
        let formula = TransportFormula::Direct(factor("trial", &[1], &[], &[0]));
        let a = lower_transport_formula(&mut arena, &formula);
        let renamed =
            arena.substitute(a, &[(VariableId::from_raw(1), VariableId::from_raw(7))]).unwrap();
        let again = arena
            .substitute(renamed, &[(VariableId::from_raw(7), VariableId::from_raw(1))])
            .unwrap();
        assert_eq!(again, a);
    }

    #[test]
    fn population_swap_is_a_different_id_and_fails_bind() {
        let mut arena = CausalExprArena::new();
        let source = lower_transport_formula(
            &mut arena,
            &TransportFormula::Direct(factor("source", &[1], &[], &[0])),
        );
        let target = lower_transport_formula(
            &mut arena,
            &TransportFormula::Direct(factor("target", &[1], &[], &[0])),
        );
        assert_ne!(source, target);
        let expected = arena.leaf_bindings(source);
        assert_eq!(arena.bind_certificate(source, &expected), Ok(()));
        assert_eq!(
            arena.bind_certificate(target, &expected),
            Err(ExprError::CertificateBindFailed)
        );
    }

    #[test]
    fn two_population_lookalikes_do_not_intern_as_one() {
        let mut arena = CausalExprArena::new();
        let a = lower_factor(&mut arena, &factor("source", &[1], &[2], &[]), false);
        let b = lower_factor(&mut arena, &factor("target", &[1], &[2], &[]), false);
        assert_ne!(a, b);
        match (arena.node(a), arena.node(b)) {
            (
                ExprNode::Distribution { population: p1, .. },
                ExprNode::Distribution { population: p2, .. },
            ) => {
                assert_ne!(p1, p2);
            }
            other => panic!("expected two distributions, got {other:?}"),
        }
    }

    #[test]
    #[allow(clippy::items_after_statements)]
    fn nested_kernel_round_trip_keeps_free_vars() {
        let mut arena = CausalExprArena::new();
        let formula = TransportFormula::RecursiveFactorization {
            sum_out: Arc::from([VariableId::from_raw(2)]),
            factors: Arc::from([
                factor("source", &[0], &[], &[]),
                factor("source", &[2], &[0], &[]),
                factor("target", &[1], &[0, 2], &[]),
            ]),
        };
        let root = lower_transport_formula(&mut arena, &formula);
        let free = arena.free_variables(root);
        assert!(free.contains(&VariableId::from_raw(0)));
        assert!(free.contains(&VariableId::from_raw(1)));
        assert!(!free.contains(&VariableId::from_raw(2)));
        let bindings = arena.leaf_bindings(root);
        assert!(bindings.iter().any(|b| b.population.as_ref() == "source"));
        assert!(bindings.iter().any(|b| b.population.as_ref() == "target"));
        let mut saw_kernel = false;
        fn walk(arena: &CausalExprArena, id: ExprId, saw: &mut bool) {
            match arena.node(id) {
                ExprNode::Kernel { body, .. } => {
                    *saw = true;
                    walk(arena, *body, saw);
                }
                ExprNode::Product(list) => {
                    for &c in arena.list(*list) {
                        walk(arena, c, saw);
                    }
                }
                ExprNode::SumOut { expr, .. } => walk(arena, *expr, saw),
                _ => {}
            }
        }
        walk(&arena, root, &mut saw_kernel);
        assert!(saw_kernel);
    }

    #[test]
    fn symbolic_and_concrete_interventions_are_distinct() {
        let mut arena = CausalExprArena::new();
        let y = arena.intern_var_set([VariableId::from_raw(1)]);
        let empty = arena.empty_var_set();
        let symbolic = arena.intern_intervention_set([VariableId::from_raw(0)]);
        let concrete = arena.intern_intervention_assignments([InterventionAssignment {
            variable: VariableId::from_raw(0),
            value: Value::f64(1.0),
        }]);
        let a = arena.intern_distribution(y, empty, symbolic, DomainRef::Interventional);
        let b = arena.intern_distribution(y, empty, concrete, DomainRef::Interventional);
        assert_ne!(a, b);
    }

    #[test]
    fn regime_mismatch_is_invalid_input() {
        let mut arena = CausalExprArena::new();
        let empty = arena.empty_var_set();
        let empty_i = arena.empty_intervention_set();
        let err = arena
            .intern_distribution_tagged(
                empty,
                empty,
                empty_i,
                DomainRef::Observational,
                "trial",
                Some(RegimeId::from_raw(1)),
                Some(DomainRef::Interventional),
            )
            .unwrap_err();
        assert_eq!(err, ExprError::RegimeDomainMismatch);
    }

    #[test]
    fn certificate_bind_rejects_regime_swap() {
        let mut arena = CausalExprArena::new();
        let y = arena.intern_var_set([VariableId::from_raw(1)]);
        let empty = arena.empty_var_set();
        let empty_i = arena.empty_intervention_set();
        let a = arena
            .intern_distribution_tagged(
                y,
                empty,
                empty_i,
                DomainRef::Observational,
                "target",
                Some(RegimeId::from_raw(1)),
                Some(DomainRef::Observational),
            )
            .unwrap();
        let b = arena
            .intern_distribution_tagged(
                y,
                empty,
                empty_i,
                DomainRef::Observational,
                "target",
                Some(RegimeId::from_raw(2)),
                Some(DomainRef::Observational),
            )
            .unwrap();
        assert_ne!(a, b);
        let expected = vec![LeafBinding {
            population: Arc::from("target"),
            regime: Some(RegimeId::from_raw(1)),
        }];
        assert_eq!(arena.bind_certificate(a, &expected), Ok(()));
        assert_eq!(arena.bind_certificate(b, &expected), Err(ExprError::CertificateBindFailed));
    }
}
