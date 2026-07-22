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

use causal_core::{AssumptionSet, ExecutionContext, Lag, VariableId};
use causal_data::{LaggedFrame, TimeSeriesData};
use causal_graph::{DenseNodeId, Endpoint, MiddleMark, NodeRef, TemporalPag, TemporalPagReview};
use causal_stats::FdrAdjustment;

use crate::engine::{DiscoveryWorkspace, PcmciEngine};
use crate::error::DiscoveryError;
use crate::evidence::pag_evidence_from_oriented;
use crate::orientation::OrientationState;
use crate::pipeline::{algorithm_record, push_diagnostic};
use crate::result::{
    DiscoveryDiagnostic, DiscoveryIteration, DiscoveryPerformanceRecord, LaggedLink, LaggedParent,
    PagDiscoveryResult, PcSepsets, ScoredLink,
};
use crate::rule_scheduling::{default_lpcmci_rules, run_lpcmci_orientation};
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
    for (n, at_n, at_t) in pag.neighbors(tgt) {
        if matches!(at_n, Endpoint::Tail) && matches!(at_t, Endpoint::Arrow) {
            if let Some(pair) = node_key(pag, n) {
                out.push(pair);
            }
        }
    }
    out
}

fn known_non_ancestors(pag: &TemporalPag, idx: &NodeIndex, of: VariableId) -> HashSet<(u32, u32)> {
    let Some(&node) = idx.get(&(of.raw(), 0)) else {
        return HashSet::new();
    };
    let mut out = HashSet::new();
    for (n, at_n, at_of) in pag.neighbors(node) {
        // Arrow into `of` from n with Tail at of would mean of → n; arrow at of means n is non-ancestor claim.
        if matches!(at_of, Endpoint::Arrow) && matches!(at_n, Endpoint::Arrow) {
            // bidirected: mutual non-ancestorship in MAG sense for both
            if let Some((v, l)) = node_key(pag, n) {
                out.insert((v.raw(), l.raw()));
            }
        } else if matches!(at_of, Endpoint::Arrow) && matches!(at_n, Endpoint::Tail) {
            // n → of: n is ancestor, not non-ancestor
        } else if matches!(at_of, Endpoint::Tail) && matches!(at_n, Endpoint::Arrow) {
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

fn collect_parents(pag: &TemporalPag, idx: &NodeIndex, variables: &[VariableId]) -> ParentMemory {
    let mut mem = ParentMemory::new();
    for &v in variables {
        let pa = known_parents_of(pag, idx, v);
        let set = pa.into_iter().map(|(u, l)| (u.raw(), l.raw())).collect();
        mem.insert(v.raw(), set);
    }
    mem
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
    let rules = default_lpcmci_rules();

    for p_pc in 0..=max_p {
        let mut any_removal = false;
        // Auto-lags first, then by increasing lag.
        let mut pairs: Vec<(VariableId, Lag, VariableId)> = Vec::new();
        for &y in variables {
            for &x in variables {
                for tau in 1..=max_lag {
                    if x == y {
                        pairs.push((x, Lag::from_raw(tau), y));
                    }
                }
            }
        }
        for tau in 0..=max_lag {
            for &y in variables {
                for &x in variables {
                    if tau == 0 && x.raw() >= y.raw() {
                        continue;
                    }
                    if tau > 0 && x == y {
                        continue; // already in auto list
                    }
                    pairs.push((x, Lag::from_raw(tau), y));
                }
            }
        }

        for (x, x_lag, y) in pairs {
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
            if mid.is_definite() {
                continue; // definite adjacency
            }
            // Middle-mark search restrictions.
            let test_y = !matches!(mid, MiddleMark::Right | MiddleMark::Both);
            let test_x =
                x_lag.is_contemporaneous() && !matches!(mid, MiddleMark::Left | MiddleMark::Both);

            let try_side = |engine: &PcmciEngine,
                            pag: &TemporalPag,
                            idx: &NodeIndex,
                            target: VariableId,
                            other: VariableId,
                            other_lag: Lag,
                            other_id: DenseNodeId,
                            workspace: &mut DiscoveryWorkspace|
             -> Result<
                Option<(Vec<(VariableId, Lag)>, f64, f64)>,
                DiscoveryError,
            > {
                let s_def = known_parents_of(pag, idx, target);
                let mut search = potential_parents(pag, idx, target, other_id);
                search.retain(|p| !s_def.contains(p) && *p != (other, other_lag));
                if search.len() < p_pc {
                    return Ok(None);
                }
                let combo: Vec<_> = search.into_iter().take(p_pc).collect();
                let mut cond = s_def.clone();
                for c in &combo {
                    if !cond.contains(c) {
                        cond.push(*c);
                    }
                }
                let (stat, p) = engine.ci_statistic(
                    frame,
                    other,
                    other_lag,
                    target,
                    Lag::CONTEMPORANEOUS,
                    &cond,
                    workspace,
                    ctx,
                )?;
                if p > alpha { Ok(Some((cond, stat, p))) } else { Ok(None) }
            };

            let mut sep_cond: Option<Vec<(VariableId, Lag)>> = None;
            let mut last_stat = 0.0;
            let mut last_p = 1.0;
            if test_y {
                if let Some((cond, stat, p)) =
                    try_side(engine, pag, idx, y, x, x_lag, xid, workspace)?
                {
                    ci_tests += 1;
                    sep_cond = Some(cond);
                    last_stat = stat;
                    last_p = p;
                } else {
                    ci_tests += 1;
                }
            }
            if sep_cond.is_none() && test_x {
                if let Some((cond, stat, p)) =
                    try_side(engine, pag, idx, x, y, Lag::CONTEMPORANEOUS, yid, workspace)?
                {
                    ci_tests += 1;
                    sep_cond = Some(cond);
                    last_stat = stat;
                    last_p = p;
                } else {
                    ci_tests += 1;
                }
            }

            // MMR when search set too small.
            if test_y {
                let search = potential_parents(pag, idx, y, xid);
                let s_def = known_parents_of(pag, idx, y);
                let n_search = search.iter().filter(|p| !s_def.contains(p)).count();
                if n_search < p_pc {
                    let _ = pag.apply_middle(xid, yid, MiddleMark::Right);
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

            // Refine to weakly minimal.
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

            let sep_arc: Arc<[LaggedParent]> = Arc::from(wm.clone().into_boxed_slice());
            sepsets_out.insert((x, x_lag, y, Lag::CONTEMPORANEOUS), sep_arc);

            let sep_nodes: Vec<DenseNodeId> =
                wm.iter().filter_map(|&(v, l)| idx.get(&(v.raw(), l.raw())).copied()).collect();
            store_weakly_minimal_sepset(state, xid, yid, Arc::from(sep_nodes));

            for (a, b) in homologous_pairs(idx, x, x_lag, y, Lag::CONTEMPORANEOUS, max_lag) {
                let _ = pag.remove_edge(a, b);
            }
            any_removal = true;
        }

        if any_removal {
            let _ = run_lpcmci_orientation(pag, &rules, state).map_err(DiscoveryError::from)?;
        } else if p_pc > 0 {
            // No removals at this cardinality and beyond likely stall — still try one orient pass.
            let _ = run_lpcmci_orientation(pag, &rules, state).map_err(DiscoveryError::from)?;
            break;
        }
    }
    let _ = run_lpcmci_orientation(pag, &rules, state).map_err(DiscoveryError::from)?;
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
    let rules = default_lpcmci_rules();

    for p_pc in 0..=max_p {
        let mut any_removal = false;
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
                    if !pag.has_edge(xid, yid) {
                        continue;
                    }
                    let mid = pag.middle_between(xid, yid).unwrap_or(MiddleMark::Empty);
                    if mid.is_definite() {
                        continue;
                    }
                    // Search among union of adjacencies minus known non-ancestors.
                    let mut search = potential_parents(pag, idx, y, xid);
                    if tau == 0 {
                        for p in potential_parents(pag, idx, x, yid) {
                            if !search.contains(&p) {
                                search.push(p);
                            }
                        }
                    }
                    let s_def_y = known_parents_of(pag, idx, y);
                    let s_def_x = if tau == 0 { known_parents_of(pag, idx, x) } else { Vec::new() };
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
                    let combo: Vec<_> = search.into_iter().take(p_pc).collect();
                    for c in &combo {
                        cond.push(*c);
                    }
                    let (stat, p) = engine.ci_statistic(
                        frame,
                        x,
                        x_lag,
                        y,
                        Lag::CONTEMPORANEOUS,
                        &cond,
                        workspace,
                        ctx,
                    )?;
                    ci_tests += 1;
                    let _ = stat;
                    if p <= alpha {
                        continue;
                    }
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
                    sepsets_out.insert((x, x_lag, y, Lag::CONTEMPORANEOUS), Arc::from(wm.clone()));
                    let sep_nodes: Vec<DenseNodeId> = wm
                        .iter()
                        .filter_map(|&(v, l)| idx.get(&(v.raw(), l.raw())).copied())
                        .collect();
                    store_weakly_minimal_sepset(state, xid, yid, Arc::from(sep_nodes));
                    for (a, b) in homologous_pairs(idx, x, x_lag, y, Lag::CONTEMPORANEOUS, max_lag)
                    {
                        let _ = pag.remove_edge(a, b);
                    }
                    any_removal = true;
                }
            }
        }
        if any_removal {
            let _ = run_lpcmci_orientation(pag, &rules, state).map_err(DiscoveryError::from)?;
        } else {
            break;
        }
    }
    let _ = run_lpcmci_orientation(pag, &rules, state).map_err(DiscoveryError::from)?;
    Ok(ci_tests)
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
