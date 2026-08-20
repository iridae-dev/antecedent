//! LPCMCI interleaved ancestral / non-ancestral phases (Gerhardus & Runge 2020 Alg. 1).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity
)]

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use antecedent_core::{AssumptionSet, ExecutionContext, Lag, VariableId};
use antecedent_data::{LaggedFrame, TimeSeriesData};
use antecedent_graph::{
    DenseNodeId, Endpoint, MiddleMark, NodeRef, TemporalPag, TemporalPagReview,
};
use antecedent_stats::FdrAdjustment;

use crate::engine::{DiscoveryWorkspace, PcmciEngine};
use crate::error::DiscoveryError;
use crate::evidence::pag_evidence_from_oriented;
use crate::orientation::OrientationState;
use crate::pipeline::{algorithm_record, push_diagnostic};
use crate::result::{
    DiscoveryDiagnostic, DiscoveryIteration, DiscoveryPerformanceRecord, LaggedLink, LaggedParent,
    PagDiscoveryResult, PcSepsets, ScoredLink, SepsetKey,
};
use crate::rule_scheduling::{default_lpcmci_rules, prelim_lpcmci_rules, run_lpcmci_orientation};
use crate::weakly_minimal::{make_sepset_weakly_minimal, store_weakly_minimal_sepset};

/// Map `(variable, lag)` → dense node id in a temporal PAG.
type NodeIndex = HashMap<(u32, u32), DenseNodeId>;

/// Known definite parents per contemporaneous variable (lag-0 target).
type ParentMemory = HashMap<u32, HashSet<(u32, u32)>>;

/// Build a complete LPCMCI-PAG: lagged `o→L`, contemporaneous `o–o?`.
pub fn init_complete_pag(
    variables: &[VariableId],
    max_lag: u32,
) -> Result<(TemporalPag, NodeIndex), DiscoveryError> {
    let mut pag = TemporalPag::empty();
    let mut idx = NodeIndex::new();
    for &v in variables {
        for lag in 0..=max_lag {
            let id = pag.add_lagged(v, Lag::from_raw(lag)).map_err(DiscoveryError::from)?;
            idx.insert((v.raw(), lag), id);
        }
    }
    // Contemporaneous pairs.
    for (i, &vi) in variables.iter().enumerate() {
        for &vj in &variables[i + 1..] {
            let a = idx[&(vi.raw(), 0)];
            let b = idx[&(vj.raw(), 0)];
            pag.insert_circle_circle_with_middle(a, b, MiddleMark::Unknown)
                .map_err(DiscoveryError::from)?;
        }
    }
    // Lagged: X_{t−τ} o→L Y_t for τ ≥ 1, all pairs including auto.
    for &target in variables {
        let tgt = idx[&(target.raw(), 0)];
        for &source in variables {
            for tau in 1..=max_lag {
                let src = idx[&(source.raw(), tau)];
                pag.insert_circle_arrow_with_middle(src, tgt, MiddleMark::Left)
                    .map_err(DiscoveryError::from)?;
            }
        }
    }
    Ok((pag, idx))
}

fn node_key(pag: &TemporalPag, id: DenseNodeId) -> Option<(VariableId, Lag)> {
    match pag.nodes().get(id.as_usize())? {
        NodeRef::Lagged { variable, lag } => Some((*variable, *lag)),
        _ => None,
    }
}

fn known_parents_of(
    pag: &TemporalPag,
    idx: &NodeIndex,
    target: VariableId,
) -> Vec<(VariableId, Lag)> {
    let Some(&tgt) = idx.get(&(target.raw(), 0)) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (n, at_target, at_neighbor) in pag.neighbors(tgt) {
        if matches!(at_neighbor, Endpoint::Tail) && matches!(at_target, Endpoint::Arrow) {
            if let Some(pair) = node_key(pag, n) {
                out.push(pair);
            }
        }
    }
    out.sort_unstable_by_key(|(variable, lag)| (variable.raw(), lag.raw()));
    out
}

fn known_non_ancestors(pag: &TemporalPag, idx: &NodeIndex, of: VariableId) -> HashSet<(u32, u32)> {
    let Some(&node) = idx.get(&(of.raw(), 0)) else {
        return HashSet::new();
    };
    let mut out = HashSet::new();
    for (n, at_of, at_neighbor) in pag.neighbors(node) {
        if matches!(at_of, Endpoint::Arrow) && matches!(at_neighbor, Endpoint::Arrow) {
            // bidirected: mutual non-ancestorship in MAG sense for both
            if let Some((v, l)) = node_key(pag, n) {
                out.insert((v.raw(), l.raw()));
            }
        } else if matches!(at_of, Endpoint::Arrow) && matches!(at_neighbor, Endpoint::Tail) {
            // n → of: n is ancestor, not non-ancestor
        } else if matches!(at_of, Endpoint::Tail) && matches!(at_neighbor, Endpoint::Arrow) {
            // of → n: n is non-ancestor of of
            if let Some((v, l)) = node_key(pag, n) {
                out.insert((v.raw(), l.raw()));
            }
        }
    }
    out
}

fn potential_parents(
    pag: &TemporalPag,
    idx: &NodeIndex,
    target: VariableId,
    exclude: DenseNodeId,
) -> Vec<(VariableId, Lag)> {
    let Some(&tgt) = idx.get(&(target.raw(), 0)) else {
        return Vec::new();
    };
    let non_anc = known_non_ancestors(pag, idx, target);
    let mut out = Vec::new();
    for (n, _, _) in pag.neighbors(tgt) {
        if n == exclude {
            continue;
        }
        let Some((v, l)) = node_key(pag, n) else {
            continue;
        };
        if non_anc.contains(&(v.raw(), l.raw())) {
            continue;
        }
        // Skip definite empty-middle edges that are not parents? Keep all adjacencies for search.
        out.push((v, l));
    }
    out.sort_unstable_by_key(|(variable, lag)| (variable.raw(), lag.raw()));
    out
}

fn shifted_known_parents(
    pag: &TemporalPag,
    idx: &NodeIndex,
    target: VariableId,
    target_lag: Lag,
    max_lag: u32,
) -> Vec<(VariableId, Lag)> {
    known_parents_of(pag, idx, target)
        .into_iter()
        .filter_map(|(variable, lag)| {
            let shifted = lag.raw().saturating_add(target_lag.raw());
            (shifted <= max_lag).then_some((variable, Lag::from_raw(shifted)))
        })
        .collect()
}

fn potential_parents_at(
    pag: &TemporalPag,
    idx: &NodeIndex,
    target: VariableId,
    target_lag: Lag,
    exclude: (VariableId, Lag),
    max_lag: u32,
) -> Vec<(VariableId, Lag)> {
    let Some(&target_now) = idx.get(&(target.raw(), 0)) else {
        return Vec::new();
    };
    let non_anc = known_non_ancestors(pag, idx, target);
    let auto_link = exclude.0 == target;
    let mut out = Vec::new();
    for (neighbor, _, _) in pag.neighbors(target_now) {
        let Some((variable, lag)) = node_key(pag, neighbor) else {
            continue;
        };
        if non_anc.contains(&(variable.raw(), lag.raw())) && !auto_link {
            continue;
        }
        let shifted = lag.raw().saturating_add(target_lag.raw());
        if shifted <= max_lag {
            let candidate = (variable, Lag::from_raw(shifted));
            if candidate != exclude {
                out.push(candidate);
            }
        }
    }
    out.sort_unstable_by_key(|(variable, lag)| (variable.raw(), lag.raw()));
    out.dedup();
    out
}

fn combinations<T: Copy>(items: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 {
        return vec![Vec::new()];
    }
    if k > items.len() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut indices: Vec<usize> = (0..k).collect();
    loop {
        out.push(indices.iter().map(|&index| items[index]).collect());
        let mut position = k;
        while position > 0 && indices[position - 1] == position - 1 + items.len() - k {
            position -= 1;
        }
        if position == 0 {
            break;
        }
        indices[position - 1] += 1;
        for next in position..k {
            indices[next] = indices[next - 1] + 1;
        }
    }
    out
}

fn homologous_pairs(
    idx: &NodeIndex,
    x: VariableId,
    x_lag: Lag,
    y: VariableId,
    y_lag: Lag,
    max_lag: u32,
) -> Vec<(DenseNodeId, DenseNodeId)> {
    // Stationarity: same var-pair and lag-difference, shifted.
    let dx = i64::from(x_lag.raw());
    let dy = i64::from(y_lag.raw());
    let lag_diff = dx - dy;
    let mut out = Vec::new();
    for shift in 0..=max_lag {
        let xl = dx + i64::from(shift);
        let yl = dy + i64::from(shift);
        if xl < 0 || yl < 0 || xl > i64::from(max_lag) || yl > i64::from(max_lag) {
            continue;
        }
        // Prefer pairs where one is at lag 0 (canonical LPCMCI window).
        if yl != 0 && xl != 0 {
            continue;
        }
        let _ = lag_diff;
        if let (Some(&a), Some(&b)) =
            (idx.get(&(x.raw(), xl as u32)), idx.get(&(y.raw(), yl as u32)))
        {
            out.push((a, b));
        }
    }
    // Always include the queried pair.
    if let (Some(&a), Some(&b)) =
        (idx.get(&(x.raw(), x_lag.raw())), idx.get(&(y.raw(), y_lag.raw())))
    {
        if !out.iter().any(|&(u, v)| (u == a && v == b) || (u == b && v == a)) {
            out.push((a, b));
        }
    }
    out
}

fn apply_remembered_parents(pag: &mut TemporalPag, idx: &NodeIndex, parents: &ParentMemory) {
    for (&tgt_raw, set) in parents {
        let Some(&tgt) = idx.get(&(tgt_raw, 0)) else {
            continue;
        };
        for &(src_raw, slag) in set {
            let Some(&src) = idx.get(&(src_raw, slag)) else {
                continue;
            };
            if !pag.has_edge(src, tgt) {
                continue;
            }
            let _ = pag.set_marks(src, tgt, Endpoint::Tail, Endpoint::Arrow);
            let _ = pag.set_middle(src, tgt, MiddleMark::Empty);
        }
    }
}

fn contemporaneous_mark_snapshot(
    pag: &TemporalPag,
    idx: &NodeIndex,
    variables: &[VariableId],
) -> Vec<(DenseNodeId, DenseNodeId, Endpoint, Endpoint, MiddleMark)> {
    let mut out = Vec::new();
    for (i, &vi) in variables.iter().enumerate() {
        for &vj in &variables[i + 1..] {
            let Some(&a) = idx.get(&(vi.raw(), 0)) else {
                continue;
            };
            let Some(&b) = idx.get(&(vj.raw(), 0)) else {
                continue;
            };
            if let Some(e) = pag.edge_between(a, b) {
                let (at_a, at_b) = if e.a == a { (e.at_a, e.at_b) } else { (e.at_b, e.at_a) };
                out.push((a, b, at_a, at_b, e.middle));
            }
        }
    }
    out
}

fn restore_contemporaneous_marks(
    pag: &mut TemporalPag,
    snap: &[(DenseNodeId, DenseNodeId, Endpoint, Endpoint, MiddleMark)],
) {
    for &(a, b, at_a, at_b, mid) in snap {
        if !pag.has_edge(a, b) {
            continue;
        }
        let _ = pag.set_marks(a, b, at_a, at_b);
        let _ = pag.set_middle(a, b, mid);
    }
}

/// End-of-phase: force remaining ambiguous middle marks to `!` (Both).
fn force_ambiguous_middles_to_both(pag: &mut TemporalPag) {
    let n = pag.node_count();
    for a_raw in 0..n {
        let a = DenseNodeId::from_raw(a_raw as u32);
        let nbrs: Vec<_> = pag.neighbors(a).map(|(x, _, _)| x).collect();
        for b in nbrs {
            if a.raw() > b.raw() {
                continue;
            }
            let Some(e) = pag.edge_between(a, b) else {
                continue;
            };
            if matches!(e.middle, MiddleMark::Empty | MiddleMark::Both) {
                continue;
            }
            let _ = pag.set_middle(a, b, MiddleMark::Both);
        }
    }
}

fn orient_lagged_only(
    pag: &mut TemporalPag,
    idx: &NodeIndex,
    variables: &[VariableId],
    rules: &[&dyn crate::rule_scheduling::LpcmciOrientationRule],
    state: &mut OrientationState,
) -> Result<crate::orientation::RuleDelta, DiscoveryError> {
    let snap = contemporaneous_mark_snapshot(pag, idx, variables);
    let delta = run_lpcmci_orientation(pag, rules, state).map_err(DiscoveryError::from)?;
    restore_contemporaneous_marks(pag, &snap);
    Ok(delta)
}

fn collect_parents(pag: &TemporalPag, idx: &NodeIndex, variables: &[VariableId]) -> ParentMemory {
    // Remember-only-parents: retain definite parents (tail→arrow) that still exist.
    let mut mem = ParentMemory::new();
    for &v in variables {
        let set =
            known_parents_of(pag, idx, v).into_iter().map(|(u, l)| (u.raw(), l.raw())).collect();
        mem.insert(v.raw(), set);
    }
    mem
}

/// Priority link batches matching pinned `auto_first` ancestral removal:
/// autos first, then contemporaneous, then increasing positive lag.
fn ancestral_link_batches(
    variables: &[VariableId],
    max_lag: u32,
) -> Vec<Vec<(VariableId, Lag, VariableId)>> {
    let mut batches = Vec::new();
    // Auto-lags: larger lag first, variables in order (product(N, -tau_max..0)).
    let mut autos = Vec::new();
    for &v in variables {
        for tau in (1..=max_lag).rev() {
            autos.push((v, Lag::from_raw(tau), v));
        }
    }
    batches.push(autos);
    for tau in 0..=max_lag {
        let mut batch = Vec::new();
        for &y in variables {
            for &x in variables {
                if tau == 0 && x.raw() >= y.raw() {
                    continue;
                }
                if tau > 0 && x == y {
                    continue;
                }
                batch.push((x, Lag::from_raw(tau), y));
            }
        }
        batches.push(batch);
    }
    batches
}

/// One ancestral removal phase (Algorithm S2).
fn ancestral_removal_phase(
    engine: &PcmciEngine,
    frame: &LaggedFrame,
    pag: &mut TemporalPag,
    idx: &NodeIndex,
    variables: &[VariableId],
    state: &mut OrientationState,
    sepsets_out: &mut PcSepsets,
    scored: &mut Vec<ScoredLink>,
    workspace: &mut DiscoveryWorkspace,
    ctx: &ExecutionContext,
    max_p: usize,
) -> Result<u64, DiscoveryError> {
    let max_lag = engine.constraints.temporal.max_lag.raw();
    let alpha = engine.constraints.alpha;
    let mut ci_tests = 0u64;
    let rules = prelim_lpcmci_rules();
    let batches = ancestral_link_batches(variables, max_lag);

    // While over p_pc, restart at 0 after an orientation update.
    let mut p_pc = 0usize;
    let mut guard = 0usize;
    while p_pc <= max_p && guard < 10_000 {
        guard += 1;
        let mut any_removal = false;
        let mut has_converged = true;

        for batch in &batches {
            // Defer removals until the end of the batch so co-tested links remain
            // available as conditioning candidates (`to_remove`).
            let mut to_remove: Vec<(VariableId, Lag, VariableId)> = Vec::new();

            for &(x, x_lag, y) in batch {
                let Some(&xid) = idx.get(&(x.raw(), x_lag.raw())) else {
                    continue;
                };
                let Some(&yid) = idx.get(&(y.raw(), 0)) else {
                    continue;
                };
                if !pag.has_edge(xid, yid) {
                    continue;
                }
                let mid = pag.middle_between(xid, yid).unwrap_or(MiddleMark::Empty);
                if mid.is_definite() && !(x == y && x_lag.raw() > 0) {
                    continue;
                }
                let test_y = !matches!(mid, MiddleMark::Right | MiddleMark::Both);
                // max_cond_px=0 pin: lagged links are tested from the Y side only.
                let test_x = x_lag.is_contemporaneous()
                    && !matches!(mid, MiddleMark::Left | MiddleMark::Both);

                let try_side = |engine: &PcmciEngine,
                                pag: &TemporalPag,
                                idx: &NodeIndex,
                                target: VariableId,
                                target_lag: Lag,
                                other: VariableId,
                                other_lag: Lag,
                                workspace: &mut DiscoveryWorkspace|
                 -> Result<
                    (Option<(Vec<(VariableId, Lag)>, f64, f64)>, u64),
                    DiscoveryError,
                > {
                    let mut s_def = shifted_known_parents(pag, idx, target, target_lag, max_lag);
                    s_def.extend(shifted_known_parents(pag, idx, other, other_lag, max_lag));
                    s_def.sort_unstable_by_key(|(variable, lag)| (variable.raw(), lag.raw()));
                    s_def.dedup();
                    s_def.retain(|node| {
                        *node != (target, target_lag) && *node != (other, other_lag)
                    });
                    let mut search = potential_parents_at(
                        pag,
                        idx,
                        target,
                        target_lag,
                        (other, other_lag),
                        max_lag,
                    );
                    search.retain(|p| !s_def.contains(p));
                    if search.len() < p_pc {
                        return Ok((None, 0));
                    }
                    let mut best: Option<(Vec<(VariableId, Lag)>, f64, f64)> = None;
                    let mut tests = 0u64;
                    let mut cond = Vec::new();
                    for combo in combinations(&search, p_pc) {
                        cond.clear();
                        cond.extend_from_slice(&s_def);
                        for c in &combo {
                            if !cond.contains(c) {
                                cond.push(*c);
                            }
                        }
                        cond.sort_unstable_by_key(|(variable, lag)| (variable.raw(), lag.raw()));
                        let (stat, p) = engine.ci_statistic(
                            frame, other, other_lag, target, target_lag, &cond, workspace, ctx,
                        )?;
                        tests += 1;
                        // break_once_separated: first independence after stable combo order.
                        if p > alpha {
                            best = Some((cond, stat, p));
                            break;
                        }
                    }
                    Ok((best, tests))
                };

                let mut sep_cond: Option<Vec<(VariableId, Lag)>> = None;
                let mut last_stat = 0.0;
                let mut last_p = 1.0;
                if test_y {
                    let search = potential_parents_at(
                        pag,
                        idx,
                        y,
                        Lag::CONTEMPORANEOUS,
                        (x, x_lag),
                        max_lag,
                    );
                    let s_def = known_parents_of(pag, idx, y);
                    let n_search = search.iter().filter(|p| !s_def.contains(p)).count();
                    if n_search < p_pc {
                        let _ = pag.apply_middle(xid, yid, MiddleMark::Right);
                    } else {
                        has_converged = false;
                        let (separation, tests) = try_side(
                            engine,
                            pag,
                            idx,
                            y,
                            Lag::CONTEMPORANEOUS,
                            x,
                            x_lag,
                            workspace,
                        )?;
                        ci_tests += tests;
                        if let Some((cond, stat, p)) = separation {
                            sep_cond = Some(cond);
                            last_stat = stat;
                            last_p = p;
                        }
                    }
                }
                if sep_cond.is_none() && test_x {
                    let (separation, tests) =
                        try_side(engine, pag, idx, x, x_lag, y, Lag::CONTEMPORANEOUS, workspace)?;
                    ci_tests += tests;
                    // `try_side` returns `tests == 0` exactly when its own search pool was
                    // smaller than `p_pc`, i.e. this side is exhausted. Any real test means the
                    // X side still has unexplored subsets at this depth, so the loop has not
                    // converged — without this the Y side alone decides, and an edge whose Y
                    // side exhausted first breaks out before `p_pc` ever reaches the depth at
                    // which the X side would have separated it.
                    if tests > 0 {
                        has_converged = false;
                    }
                    if let Some((cond, stat, p)) = separation {
                        sep_cond = Some(cond);
                        last_stat = stat;
                        last_p = p;
                    }
                }

                let Some(cond) = sep_cond else {
                    scored.push(ScoredLink {
                        link: LaggedLink {
                            source: x,
                            source_lag: x_lag,
                            target: y,
                            target_lag: Lag::CONTEMPORANEOUS,
                        },
                        statistic: last_stat,
                        p_value: last_p,
                        adjusted_p_value: None,
                    });
                    continue;
                };

                let ancs: Vec<_> = known_parents_of(pag, idx, x)
                    .into_iter()
                    .chain(known_parents_of(pag, idx, y))
                    .collect();
                let wm = make_sepset_weakly_minimal(
                    engine,
                    frame,
                    x,
                    x_lag,
                    y,
                    Lag::CONTEMPORANEOUS,
                    &cond,
                    &ancs,
                    workspace,
                    ctx,
                )?;
                ci_tests += 1;
                let sep_arc: Arc<[LaggedParent]> = Arc::from(wm);
                sepsets_out.insert((x, x_lag, y, Lag::CONTEMPORANEOUS), Arc::clone(&sep_arc));
                let sep_nodes: Vec<DenseNodeId> = sep_arc
                    .iter()
                    .filter_map(|&(v, l)| idx.get(&(v.raw(), l.raw())).copied())
                    .collect();
                store_weakly_minimal_sepset(state, xid, yid, Arc::from(sep_nodes));
                to_remove.push((x, x_lag, y));
            }

            for (x, x_lag, y) in to_remove {
                for (a, b) in homologous_pairs(idx, x, x_lag, y, Lag::CONTEMPORANEOUS, max_lag) {
                    let _ = pag.remove_edge(a, b);
                }
                any_removal = true;
            }
        }

        let delta = orient_lagged_only(pag, idx, variables, &rules, state)?;
        if any_removal {
            if delta.edges_changed > 0 {
                p_pc = 0;
            } else {
                p_pc += 1;
            }
        } else if has_converged {
            break;
        } else {
            p_pc += 1;
        }
    }
    // End-of-phase: force middle marks to `!`, then full orientation including contemporaneous
    // (`prelim_with_collider_rules` / `_rules_all` with `only_lagged=False`).
    force_ambiguous_middles_to_both(pag);
    let _ = run_lpcmci_orientation(pag, &default_lpcmci_rules(), state)
        .map_err(DiscoveryError::from)?;
    Ok(ci_tests)
}

/// Non-ancestral removal (Algorithm S3): CI given napds-style adjacencies of both sides.
fn non_ancestral_removal_phase(
    engine: &PcmciEngine,
    frame: &LaggedFrame,
    pag: &mut TemporalPag,
    idx: &NodeIndex,
    variables: &[VariableId],
    state: &mut OrientationState,
    sepsets_out: &mut PcSepsets,
    workspace: &mut DiscoveryWorkspace,
    ctx: &ExecutionContext,
    max_p: usize,
) -> Result<u64, DiscoveryError> {
    let max_lag = engine.constraints.temporal.max_lag.raw();
    let alpha = engine.constraints.alpha;
    let mut ci_tests = 0u64;
    let rules = prelim_lpcmci_rules();

    for p_pc in 0..=max_p {
        let mut any_removal = false;
        let phase_pag = pag.clone();
        let mut to_remove: Vec<(VariableId, Lag, VariableId)> = Vec::new();
        for &y in variables {
            for &x in variables {
                for tau in 0..=max_lag {
                    if tau == 0 && x.raw() >= y.raw() {
                        continue;
                    }
                    let x_lag = Lag::from_raw(tau);
                    let Some(&xid) = idx.get(&(x.raw(), tau)) else {
                        continue;
                    };
                    let Some(&yid) = idx.get(&(y.raw(), 0)) else {
                        continue;
                    };
                    if !phase_pag.has_edge(xid, yid) {
                        continue;
                    }
                    let mid = phase_pag.middle_between(xid, yid).unwrap_or(MiddleMark::Empty);
                    if mid.is_definite() && !(x == y && x_lag.raw() > 0) {
                        continue;
                    }
                    // Search among union of adjacencies minus known non-ancestors.
                    let mut search = potential_parents(&phase_pag, idx, y, xid);
                    if tau == 0 {
                        for p in potential_parents(&phase_pag, idx, x, yid) {
                            if !search.contains(&p) {
                                search.push(p);
                            }
                        }
                    }
                    let s_def_y = known_parents_of(&phase_pag, idx, y);
                    let s_def_x =
                        if tau == 0 { known_parents_of(&phase_pag, idx, x) } else { Vec::new() };
                    let mut cond = s_def_y.clone();
                    for p in &s_def_x {
                        if !cond.contains(p) {
                            cond.push(*p);
                        }
                    }
                    search.retain(|p| !cond.contains(p) && *p != (x, x_lag));
                    if search.len() < p_pc {
                        continue;
                    }
                    let mut best: Option<(Vec<(VariableId, Lag)>, f64, f64)> = None;
                    let mut candidate_cond = Vec::new();
                    for combo in combinations(&search, p_pc) {
                        candidate_cond.clear();
                        candidate_cond.extend_from_slice(&cond);
                        candidate_cond.extend_from_slice(&combo);
                        candidate_cond
                            .sort_unstable_by_key(|(variable, lag)| (variable.raw(), lag.raw()));
                        let (stat, p) = engine.ci_statistic(
                            frame,
                            x,
                            x_lag,
                            y,
                            Lag::CONTEMPORANEOUS,
                            &candidate_cond,
                            workspace,
                            ctx,
                        )?;
                        ci_tests += 1;
                        if p > alpha {
                            best = Some((std::mem::take(&mut candidate_cond), stat, p));
                            break;
                        }
                    }
                    let Some((cond, _stat, _p)) = best else {
                        continue;
                    };
                    let ancs: Vec<_> = s_def_x.into_iter().chain(s_def_y).collect();
                    let wm = make_sepset_weakly_minimal(
                        engine,
                        frame,
                        x,
                        x_lag,
                        y,
                        Lag::CONTEMPORANEOUS,
                        &cond,
                        &ancs,
                        workspace,
                        ctx,
                    )?;
                    let sep_arc: Arc<[LaggedParent]> = Arc::from(wm);
                    sepsets_out.insert((x, x_lag, y, Lag::CONTEMPORANEOUS), Arc::clone(&sep_arc));
                    let sep_nodes: Vec<DenseNodeId> = sep_arc
                        .iter()
                        .filter_map(|&(v, l)| idx.get(&(v.raw(), l.raw())).copied())
                        .collect();
                    store_weakly_minimal_sepset(state, xid, yid, Arc::from(sep_nodes));
                    to_remove.push((x, x_lag, y));
                }
            }
        }
        for (x, x_lag, y) in to_remove {
            for (a, b) in homologous_pairs(idx, x, x_lag, y, Lag::CONTEMPORANEOUS, max_lag) {
                let _ = pag.remove_edge(a, b);
            }
            any_removal = true;
        }
        let _ = run_lpcmci_orientation(pag, &rules, state).map_err(DiscoveryError::from)?;
        let _ = any_removal;
    }
    let _ = run_lpcmci_orientation(pag, &rules, state).map_err(DiscoveryError::from)?;
    Ok(ci_tests)
}

/// Reconcile the `scored` and `sepsets` accumulators against the final oriented PAG.
///
/// # Why this is needed
///
/// `run_lpcmci_algorithm` threads two accumulators through every phase it runs (each
/// preliminary iteration, the full ancestral phase, and the non-ancestral phase):
///
/// - `scored` is appended to only when a link's separation search came up empty, i.e. the
///   link was **retained** in that phase's PAG (see the `sep_cond.is_none()` branch above).
/// - `sepsets` is inserted into only when a separating set **was** found, immediately before
///   the corresponding edge is removed. A sepset is the recorded justification for a removal.
///
/// Both accumulators are append-only and are never rebuilt when a PAG is rebuilt. Critically,
/// each preliminary iteration calls [`init_complete_pag`] to start over from a *fresh* complete
/// graph (carrying over only remembered parents, not the prior iteration's removals). So a
/// removal made in iteration `k` — and the sepset recorded for it — can be silently undone by
/// iteration `k + 1` reintroducing that edge into its fresh PAG and failing to re-remove it (or
/// removing a *different* edge instead). Symmetrically, a link pushed to `scored` as "retained"
/// in an earlier phase can end up removed by a later phase operating on that phase's own PAG.
/// By the time the final PAG is oriented, both accumulators can describe edges that no longer
/// exist in it, or omit sepsets for edges that were, in the end, never separated.
///
/// This function is the fix: it filters both accumulators against the single source of truth,
/// the final PAG, **in opposite directions** —
///
/// - `scored` entries are kept only if their link's edge **is present** in the final PAG,
///   because `scored` records retained links; a "retained" link that isn't actually an edge
///   anymore is a stale claim.
/// - `sepsets` entries are kept only if their pair's edge **is absent** from the final PAG,
///   because a sepset records the justification for a removal; a sepset attached to an edge
///   that survived to the final PAG is exactly the stale-removal-undone case described above,
///   and reporting it as the surviving link's conditioning set would be dishonest. Dropping it
///   makes the surviving link report an empty conditioning set instead, which is honest about
///   what is actually known.
///
/// It is precisely because each preliminary iteration rebuilds a fresh complete PAG that both
/// directions of drift are reachable in practice, not just in theory.
///
/// `fci.rs:446` and `rfci.rs:423` already do the `scored` half of this for their own algorithms
/// (`scored.retain(|s| adj.contains_key(&edge_key(...)))`, filtering against live adjacency
/// after their final orientation pass) — that is the in-repo precedent for this pattern. LPCMCI
/// never had an equivalent step, and its repeated preliminary-iteration structure is exactly
/// what makes the omission observable.
///
/// # Steps
///
/// 1. Dedup `scored` by `(source, source_lag, target, target_lag)`, keeping the **last**
///    occurrence (later phases supersede earlier ones with a more current statistic/p-value),
///    while preserving the relative order of the surviving entries.
/// 2. Retain a `scored` entry only if both endpoints resolve through `idx` and
///    [`TemporalPag::has_edge`] returns `true` for the resolved pair.
/// 3. Retain a `sepsets` entry only if both endpoints resolve through `idx` **and**
///    [`TemporalPag::has_edge`] returns `false` for the resolved pair.
///
/// Endpoint resolution failure (missing from `idx`) drops the entry from either accumulator —
/// it cannot correspond to an edge in this graph.
fn reconcile_evidence_with_pag(
    pag: &TemporalPag,
    idx: &NodeIndex,
    scored: &mut Vec<ScoredLink>,
    sepsets: &mut PcSepsets,
) {
    // Step 1: dedup by link key, last occurrence wins, order-preserving.
    let mut seen: HashSet<(u32, u32, u32, u32)> = HashSet::new();
    let mut keep = vec![false; scored.len()];
    for (i, s) in scored.iter().enumerate().rev() {
        let link = s.link;
        let key =
            (link.source.raw(), link.source_lag.raw(), link.target.raw(), link.target_lag.raw());
        if seen.insert(key) {
            keep[i] = true;
        }
    }
    // Zip rather than drive `retain` from a side iterator: the lengths are structurally
    // paired here, so a mismatch cannot be silently absorbed as "drop it".
    *scored = std::mem::take(scored)
        .into_iter()
        .zip(keep)
        .filter_map(|(s, keep)| keep.then_some(s))
        .collect();

    // Step 2: scored links must be present as an edge in the final PAG.
    scored.retain(|s| {
        let link = s.link;
        if let (Some(&a), Some(&b)) = (
            idx.get(&(link.source.raw(), link.source_lag.raw())),
            idx.get(&(link.target.raw(), link.target_lag.raw())),
        ) {
            pag.has_edge(a, b)
        } else {
            false
        }
    });

    // Step 3: sepsets must be absent as an edge in the final PAG (opposite direction of step 2).
    sepsets.retain(|key: &SepsetKey, _| {
        let &(x, x_lag, y, y_lag) = key;
        if let (Some(&a), Some(&b)) =
            (idx.get(&(x.raw(), x_lag.raw())), idx.get(&(y.raw(), y_lag.raw())))
        {
            !pag.has_edge(a, b)
        } else {
            false
        }
    });
}

/// Run full LPCMCI Algorithm 1.
pub fn run_lpcmci_algorithm(
    engine: &PcmciEngine,
    data: &TimeSeriesData,
    variables: &[VariableId],
    workspace: &mut DiscoveryWorkspace,
    ctx: &ExecutionContext,
    fdr: Option<FdrAdjustment>,
    n_preliminary: u32,
) -> Result<PagDiscoveryResult, DiscoveryError> {
    let max_lag = engine.constraints.temporal.max_lag.raw();
    let alpha = engine.constraints.alpha;
    let max_cond = engine.constraints.max_cond_size;
    let frame_depth = 2 * max_lag;
    let frame = LaggedFrame::from_series(data, variables, frame_depth, &ctx.kernel_policy)
        .map_err(DiscoveryError::from)?;
    workspace.prepared_ci = None;

    let mut sepsets = PcSepsets::default();
    let mut scored = Vec::new();
    let mut state = OrientationState::default();
    let mut ci_tests = 0u64;
    let mut parents_mem = ParentMemory::new();
    let mut iterations = Vec::new();

    // Preliminary phases.
    for k in 0..n_preliminary {
        let (mut pag, idx) = init_complete_pag(variables, max_lag)?;
        apply_remembered_parents(&mut pag, &idx, &parents_mem);
        let t = ancestral_removal_phase(
            engine,
            &frame,
            &mut pag,
            &idx,
            variables,
            &mut state,
            &mut sepsets,
            &mut scored,
            workspace,
            ctx,
            max_cond,
        )?;
        ci_tests += t;
        parents_mem = collect_parents(&pag, &idx, variables);
        iterations.push(DiscoveryIteration {
            label: Arc::from(format!("lpcmci.prelim.{k}")),
            ci_tests: t,
        });
        let _ = pag;
    }

    // Full ancestral + non-ancestral.
    let (mut pag, idx) = init_complete_pag(variables, max_lag)?;
    apply_remembered_parents(&mut pag, &idx, &parents_mem);
    let t = ancestral_removal_phase(
        engine,
        &frame,
        &mut pag,
        &idx,
        variables,
        &mut state,
        &mut sepsets,
        &mut scored,
        workspace,
        ctx,
        max_cond,
    )?;
    ci_tests += t;
    iterations.push(DiscoveryIteration { label: Arc::from("lpcmci.ancestral"), ci_tests: t });

    let t = non_ancestral_removal_phase(
        engine,
        &frame,
        &mut pag,
        &idx,
        variables,
        &mut state,
        &mut sepsets,
        workspace,
        ctx,
        max_cond,
    )?;
    ci_tests += t;
    iterations.push(DiscoveryIteration { label: Arc::from("lpcmci.non_ancestral"), ci_tests: t });

    pag.clear_middle_marks();
    let rules = default_lpcmci_rules();
    let delta =
        run_lpcmci_orientation(&mut pag, &rules, &mut state).map_err(DiscoveryError::from)?;

    // The final oriented PAG is the sole authority on what survived; reconcile the
    // phase-accumulated `scored`/`sepsets` against it before packaging evidence.
    reconcile_evidence_with_pag(&pag, &idx, &mut scored, &mut sepsets);

    let _ = fdr; // alpha-based removals; FDR on residual scored links is not applied in Alg. 1.

    let algorithm = algorithm_record(
        "lpcmci",
        format!(
            "alpha={alpha},max_lag={max_lag},n_preliminary={n_preliminary},min_lag={}",
            engine.constraints.temporal.min_lag.raw()
        ),
    );
    let evidence = pag_evidence_from_oriented(pag.clone(), scored, &sepsets);
    let review = TemporalPagReview::from_pag(pag, algorithm.id.clone());
    let links_retained = evidence.links.len() as u64;
    let mut diagnostics: Vec<DiscoveryDiagnostic> = Vec::new();
    push_diagnostic(
        &mut diagnostics,
        "lpcmci.pag",
        format!(
            "oriented temporal PAG with {} nodes ({} circle edges pending), ci_tests={ci_tests}",
            evidence.graph.node_count(),
            review.pending_circles.len(),
        ),
    );
    if state.conflicts > 0 || delta.conflicts > 0 {
        push_diagnostic(
            &mut diagnostics,
            "orientation.conflicts",
            format!("{} orientation conflict(s)", state.conflicts),
        );
    }

    Ok(PagDiscoveryResult {
        evidence,
        review,
        algorithm,
        assumptions: AssumptionSet::new(),
        iterations,
        diagnostics,
        performance: DiscoveryPerformanceRecord {
            ci_tests,
            links_retained,
            targets: variables.len() as u64,
            lagged_frame_bytes: frame.values_bytes(),
            worker_threads: 1,
        },
        sepsets,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_vars() -> Vec<VariableId> {
        vec![VariableId::from_raw(0), VariableId::from_raw(1)]
    }

    fn scored_link(
        source: VariableId,
        source_lag: u32,
        target: VariableId,
        target_lag: u32,
        statistic: f64,
        p_value: f64,
    ) -> ScoredLink {
        ScoredLink {
            link: LaggedLink {
                source,
                source_lag: Lag::from_raw(source_lag),
                target,
                target_lag: Lag::from_raw(target_lag),
            },
            statistic,
            p_value,
            adjusted_p_value: None,
        }
    }

    #[test]
    fn reconcile_drops_scored_link_absent_from_final_pag() {
        // Pins: a `scored` entry for a link that was removed from the PAG (by a later
        // phase, or a fresh preliminary-iteration rebuild undoing an earlier removal)
        // must not survive reconciliation.
        let vars = two_vars();
        let (mut pag, idx) = init_complete_pag(&vars, 1).unwrap();
        let a = idx[&(vars[0].raw(), 0)];
        let b = idx[&(vars[1].raw(), 0)];
        assert!(pag.has_edge(a, b));
        let _ = pag.remove_edge(a, b);
        assert!(!pag.has_edge(a, b));

        let mut scored = vec![scored_link(vars[0], 0, vars[1], 0, 0.1, 0.9)];
        let mut sepsets = PcSepsets::default();
        reconcile_evidence_with_pag(&pag, &idx, &mut scored, &mut sepsets);
        assert!(scored.is_empty(), "removed edge must not surface as a retained link");
    }

    #[test]
    fn reconcile_keeps_scored_link_present_in_final_pag() {
        // Pins: a `scored` entry for a link that genuinely survives to the final PAG
        // keeps its statistic/p_value unchanged.
        let vars = two_vars();
        let (pag, idx) = init_complete_pag(&vars, 1).unwrap();
        // Lagged X_{t-1} o-> Y_t: never removed here, so it must still be an edge.
        let a = idx[&(vars[0].raw(), 1)];
        let b = idx[&(vars[1].raw(), 0)];
        assert!(pag.has_edge(a, b));

        let mut scored = vec![scored_link(vars[0], 1, vars[1], 0, 0.42, 0.03)];
        let mut sepsets = PcSepsets::default();
        reconcile_evidence_with_pag(&pag, &idx, &mut scored, &mut sepsets);
        assert_eq!(scored.len(), 1);
        assert!((scored[0].statistic - 0.42).abs() < 1e-12);
        assert!((scored[0].p_value - 0.03).abs() < 1e-12);
    }

    #[test]
    fn reconcile_drops_sepset_for_pair_still_edged_in_final_pag() {
        // Pins: the stale-removal-undone case — a sepset was recorded for a pair whose
        // edge was later reintroduced (e.g. a fresh preliminary-iteration PAG rebuild).
        // Attaching that stale sepset to the surviving edge would misreport its
        // conditioning set, which is exactly the bug this reconciliation step fixes.
        let vars = two_vars();
        let (pag, idx) = init_complete_pag(&vars, 1).unwrap();
        let key: SepsetKey = (vars[0], Lag::CONTEMPORANEOUS, vars[1], Lag::CONTEMPORANEOUS);
        let a = idx[&(vars[0].raw(), 0)];
        let b = idx[&(vars[1].raw(), 0)];
        assert!(pag.has_edge(a, b));

        let mut scored = Vec::new();
        let mut sepsets = PcSepsets::default();
        sepsets.insert(key, Arc::from(Vec::<LaggedParent>::new()));
        reconcile_evidence_with_pag(&pag, &idx, &mut scored, &mut sepsets);
        assert!(sepsets.is_empty(), "sepset attached to a surviving edge must be dropped");
    }

    #[test]
    fn reconcile_keeps_sepset_for_genuinely_separated_pair() {
        // Pins: a sepset must NOT regress — it is evidence for a removal that held in
        // the final PAG, so the surviving separation must keep its conditioning set.
        let vars = two_vars();
        let (mut pag, idx) = init_complete_pag(&vars, 1).unwrap();
        let key: SepsetKey = (vars[0], Lag::CONTEMPORANEOUS, vars[1], Lag::CONTEMPORANEOUS);
        let a = idx[&(vars[0].raw(), 0)];
        let b = idx[&(vars[1].raw(), 0)];
        let _ = pag.remove_edge(a, b);
        assert!(!pag.has_edge(a, b));

        let mut scored = Vec::new();
        let mut sepsets = PcSepsets::default();
        let cond: Arc<[LaggedParent]> = Arc::from(vec![(vars[0], Lag::from_raw(1))]);
        sepsets.insert(key, cond.clone());
        reconcile_evidence_with_pag(&pag, &idx, &mut scored, &mut sepsets);
        assert_eq!(sepsets.get(&key), Some(&cond));
    }

    #[test]
    fn reconcile_collapses_duplicate_scored_entries_last_wins() {
        // Pins: the same link pushed by multiple phases collapses to one entry that
        // carries the LAST (most current) statistic/p_value, since later phases
        // supersede earlier ones.
        let vars = two_vars();
        let (pag, idx) = init_complete_pag(&vars, 1).unwrap();
        let a = idx[&(vars[0].raw(), 1)];
        let b = idx[&(vars[1].raw(), 0)];
        assert!(pag.has_edge(a, b));

        let mut scored = vec![
            scored_link(vars[0], 1, vars[1], 0, 0.11, 0.10),
            scored_link(vars[0], 1, vars[1], 0, 0.22, 0.02),
        ];
        let mut sepsets = PcSepsets::default();
        reconcile_evidence_with_pag(&pag, &idx, &mut scored, &mut sepsets);
        assert_eq!(scored.len(), 1);
        assert!(
            (scored[0].p_value - 0.02).abs() < 1e-12,
            "must keep the later (second-pushed) entry's p_value"
        );
        assert!(
            (scored[0].statistic - 0.22).abs() < 1e-12,
            "must keep the later (second-pushed) entry's statistic"
        );
    }

    #[test]
    fn reconcile_drops_unresolvable_endpoints_from_both_accumulators() {
        // Pins: a link/sepset whose endpoint isn't in `idx` cannot correspond to an edge
        // in this graph and must be dropped, not kept by some permissive fallback.
        let vars = two_vars();
        let (pag, idx) = init_complete_pag(&vars, 1).unwrap();
        let ghost = VariableId::from_raw(99);

        let mut scored = vec![scored_link(ghost, 0, vars[1], 0, 0.1, 0.5)];
        let mut sepsets = PcSepsets::default();
        sepsets.insert(
            (ghost, Lag::CONTEMPORANEOUS, vars[1], Lag::CONTEMPORANEOUS),
            Arc::from(Vec::<LaggedParent>::new()),
        );
        reconcile_evidence_with_pag(&pag, &idx, &mut scored, &mut sepsets);
        assert!(scored.is_empty());
        assert!(sepsets.is_empty());
    }
}
