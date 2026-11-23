//! Memoized recursive branches of the ID identification algorithm.

use super::{
    Arc, BitSet, DistCtx, ExprNode, Factor, HedgeCertificate, IdOutcome, IdQuery, IdRun,
    IdentificationError, Law, PreparedAdmg, SubproblemKey, c_component_expression, dist_marginal,
    subproblem_hash, sum_out_if_any,
};

/// Run ID; returns expression for `P_x(y)` over observational factors in the run's arena.
pub(super) fn id_recurse(
    run: &mut IdRun<'_, '_>,
    query: IdQuery<'_>,
) -> Result<IdOutcome, IdentificationError> {
    Ok((*id_recurse_shared(run, query)?).clone())
}

/// [`id_recurse`] returning the memoized outcome without copying it.
fn id_recurse_shared(
    run: &mut IdRun<'_, '_>,
    query: IdQuery<'_>,
) -> Result<Arc<IdOutcome>, IdentificationError> {
    let IdQuery { y, x, v, dist, assign } = query;
    run.perf.candidates_examined = run.perf.candidates_examined.saturating_add(1);
    run.meter.enter(run.arena.len(), run.memo.len())?;
    let hash = subproblem_hash(y, x, v, &assign, dist);
    if let Some(hit) = run.memo.get(hash, y, x, v, &assign, dist) {
        let hit = Arc::clone(hit);
        run.perf.sets_returned = run.perf.sets_returned.saturating_add(1);
        run.meter.leave();
        return Ok(hit);
    }

    let key = SubproblemKey {
        y: y.clone(),
        x: x.clone(),
        v: v.clone(),
        assign: assign.clone(),
        dist: dist.clone(),
    };
    let outcome = id_body(run, IdQuery { y, x, v, dist, assign })?;
    run.meter.leave();
    run.memo.insert(hash, key, Arc::clone(&outcome));
    Ok(outcome)
}

fn id_body(
    run: &mut IdRun<'_, '_>,
    query: IdQuery<'_>,
) -> Result<Arc<IdOutcome>, IdentificationError> {
    let IdQuery { y, x, v, dist, assign } = query;
    // Line 1: x = ∅ → ∑_{v\y} of the *current* distribution
    if !x.any() {
        run.derivation.push("general.id.line1", "empty intervention; marginal of current dist");
        let expr = dist_marginal(run.prepared, dist, y, v, run.arena, &assign)?;
        return Ok(Arc::new(IdOutcome::Expr(expr)));
    }

    // Line 2: restrict to An(Y)_G; the current distribution marginalizes over
    // the removed set (a Marginal stays a Marginal; a CFactor grows its sumset).
    let an_y = run.prepared.ancestors_cached(y, v, run.ws, &mut run.ancestors);
    if !v.equal_set(&an_y) {
        let mut x2 = x.clone();
        x2.intersect_with(&an_y);
        let mut removed = v.clone();
        removed.difference_with(&an_y);
        let dist2 = dist.marginalize(&removed);
        run.derivation.push("general.id.line2", "restrict to ancestral set of Y");
        return id_recurse_shared(run, IdQuery { y, x: &x2, v: &an_y, dist: &dist2, assign });
    }

    // Line 3: W = (V\X) \ An(Y)_{G_{\bar X}} — only X changes; dist unchanged.
    let mut v_minus_x = v.clone();
    v_minus_x.difference_with(x);
    let an_bar = run.prepared.ancestors_bar_x(y, v, x, run.ws);
    let mut w = v_minus_x.clone();
    w.difference_with(&an_bar);
    if w.any() {
        let mut x2 = x.clone();
        x2.union_with(&w);
        run.derivation.push("general.id.line3", "add superfluous interventions");
        return id_recurse_shared(run, IdQuery { y, x: &x2, v, dist, assign });
    }

    // Line 4 / 5–7: C-components of G[V\X]
    let comps = run.prepared.c_components(&v_minus_x);
    if comps.is_empty() {
        // Not a published line, and unreachable: every entry point enforces
        // ∅ ≠ Y, Y ∩ X = ∅, and each line preserves Y ⊆ V \ X (line 2 keeps
        // An(Y) ⊇ Y, line 3 only adds non-ancestors of Y to X, line 4 recurses
        // on Y = S_i with X = V \ S_i, line 7 keeps Y ⊆ S ⊆ S′). An empty
        // V \ X would mean Y ⊆ X, where the answer is a point mass at x and
        // never a marginal of the current law, so nothing is emitted for it.
        return Err(IdentificationError::InvariantViolated {
            message: "general ID: outcomes are not disjoint from interventions",
        });
    }

    if comps.len() > 1 {
        run.derivation
            .push("general.id.line4", format!("C-component factorization ({} parts)", comps.len()));
        let mut factors = Vec::with_capacity(comps.len());
        for s_i in &comps {
            let mut x_i = v.clone();
            x_i.difference_with(s_i);
            let outcome = id_recurse_shared(
                run,
                IdQuery { y: s_i, x: &x_i, v, dist, assign: assign.clone() },
            )?;
            let expr = match &*outcome {
                IdOutcome::Expr(e) => Some(*e),
                IdOutcome::Fail(_) => None,
            };
            let Some(expr) = expr else {
                return Ok(outcome);
            };
            factors.push(expr);
        }
        let product = {
            let list = run.arena.intern_list(factors);
            run.arena.intern(ExprNode::Product(list))
        };
        // ∑_{v \ (y ∪ x)}
        let mut sum_vars = v.clone();
        sum_vars.difference_with(y);
        sum_vars.difference_with(x);
        let expr = sum_out_if_any(run.prepared, &sum_vars, product, run.arena)?;
        return Ok(Arc::new(IdOutcome::Expr(expr)));
    }

    // Single C-component S of G[V\X]
    let s = &comps[0];
    // Line 5: C(G) = {G} → FAIL
    if run.prepared.is_single_c_component(v) {
        run.derivation.push("general.id.line5", "hedge: G is a single C-component");
        // A certificate names real variables or is not issued: a node without
        // a variable identity fails the call rather than being given a made-up id.
        let f_vars =
            v.iter().map(|node| run.prepared.dense_to_var(node)).collect::<Result<Vec<_>, _>>()?;
        let hedge = HedgeCertificate::from_resolved_sets(v, s, &f_vars);
        return Ok(Arc::new(IdOutcome::Fail(hedge)));
    }

    id_lines_5_to_7(run, IdQuery { y, x, v, dist, assign }, s)
}

/// Lines 6–7 dispatch for the single-C-component case (line 5 handled above).
fn id_lines_5_to_7(
    run: &mut IdRun<'_, '_>,
    query: IdQuery<'_>,
    s: &BitSet,
) -> Result<Arc<IdOutcome>, IdentificationError> {
    let IdQuery { y, x, v, dist, assign } = query;
    // Districts of G (on V)
    let g_comps = run.prepared.c_components(v);
    // Line 6: S ∈ C(G) — emit the C-factor of the *current* distribution.
    if g_comps.iter().any(|c| c.equal_set(s)) {
        run.derivation.push("general.id.line6", "S is a C-component of G; factorize current dist");
        let expr = match dist {
            DistCtx::Marginal => c_component_expression(run.prepared, s, y, v, run.arena, &assign)?,
            DistCtx::CFactor(law) => {
                // ∑_{s\y} ∏_{Vi∈S} P′(vi | v_π^{(i-1)}): with an empty sumset the
                // telescope collapses each conditional to its own frozen factor;
                // after a line-2 marginalization they are ratios of marginals.
                let mut sum_vars = s.clone();
                sum_vars.difference_with(y);
                let sub = Law { sumset: sum_vars, factors: law.conditionals(run.prepared, s, v)? };
                sub.emit(run.prepared, &BitSet::with_len(v.bit_len()), run.arena, &assign)?
            }
        };
        return Ok(Arc::new(IdOutcome::Expr(expr)));
    }

    // Line 7: ∃ S' ⊃ S, S' ∈ C(G). Recurse on G_{S'} against
    // P′ = ∏_{Vi∈S′} P(vi | v_π^{(i-1)} ∩ S′, v_π^{(i-1)} \ S′), the C-factor
    // of the current distribution — its factors keep conditioning on
    // topological predecessors *outside* S' (Tian's telescope).
    if let Some(s_prime) = g_comps.iter().find(|c| s.is_subset_of(c) && !c.equal_set(s)) {
        run.derivation
            .push("general.id.line7", "recurse into containing C-component S' against Q[S']");
        let mut x2 = x.clone();
        x2.intersect_with(s_prime);
        let factors = match dist {
            DistCtx::Marginal => marginal_conditionals(run.prepared, s_prime, v),
            DistCtx::CFactor(law) => law.conditionals(run.prepared, s_prime, v)?,
        };
        let dist2 =
            DistCtx::CFactor(Arc::new(Law { sumset: BitSet::with_len(v.bit_len()), factors }));
        return id_recurse_shared(run, IdQuery { y, x: &x2, v: s_prime, dist: &dist2, assign });
    }

    Err(IdentificationError::InvariantViolated {
        message: "general ID: no line applies to the current C-component state",
    })
}

/// Chain-rule conditionals `P(vi | v_π^{(i-1)})` of the observational marginal
/// over `v`, for `vi ∈ s`, in topological order.
fn marginal_conditionals(prepared: &PreparedAdmg, s: &BitSet, v: &BitSet) -> Vec<Factor> {
    let mut factors = Vec::new();
    let mut preceding = BitSet::with_len(v.bit_len());
    for &vi in prepared.topo() {
        if !v.contains(vi) {
            continue;
        }
        if s.contains(vi) {
            factors.push(Factor::Observed { var: vi, cond: preceding.clone() });
        }
        preceding.insert(vi);
    }
    factors
}
