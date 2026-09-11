//! Population retargeting from a frozen AIPW score table.
//!
//! `θ_a = Σ w_i φ_i^a / Σ w_i` with no refit. Valid only when `w` is a
//! function of the certified adjustment set (`depends_on` is declared).
//! Treatment, intervened coordinates, and their descendants are refused.
//! Weighted overlap failure is a support refusal, not a hard error.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use antecedent_core::{Diagnostic, DiagnosticKind, DiagnosticSeverity, VariableId};
use antecedent_graph::{Admg, BitSet, Dag, DenseNodeId, GraphWorkspace, NodeRef};

use crate::crossfit_aipw::{WeightedSupport, weighted_support};
use crate::error::EstimationError;
use crate::joint_if::{JointCovariance, monotone_decreasing};
use crate::scores::{LinearContrast, ScoreSummary, ScoreTable};

/// Minimum Kish `n_eff` per arm under target weights.
pub const MIN_WEIGHTED_ARM_N_EFF: f64 = 10.0;

/// Why retargeting was refused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RetargetRefusal {
    /// `depends_on` names the treatment, an intervened coordinate, or a descendant.
    IllegalDependence,
    /// `depends_on` is not a subset of the certified adjustment set.
    OutsideAdjustmentSet,
    /// Weighted overlap / effective sample size failed.
    WeightedOverlap,
    /// Descendant closure cannot be computed without a directed graph.
    DescendantClosureUnavailable,
    /// Nonconstant weights were supplied with an empty `depends_on`.
    UndeclaredNonconstantWeights,
}

impl RetargetRefusal {
    /// Stable support-matrix / error token.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::IllegalDependence => {
                "retarget depends_on must not include the treatment, an intervened coordinate, or a descendant"
            }
            Self::OutsideAdjustmentSet => {
                "retarget weights must be a function of the certified adjustment set"
            }
            Self::WeightedOverlap => {
                "retarget refused: weighted overlap failed under the declared target weights"
            }
            Self::DescendantClosureUnavailable => {
                "retarget depends_on descendant closure requires a directed graph (DAG or ADMG); refusing rather than skipping the descendant check"
            }
            Self::UndeclaredNonconstantWeights => {
                "nonempty depends_on is required for nonconstant target weights"
            }
        }
    }
}

/// Directed-edge descendant queries. Bidirected and circle marks are ignored.
pub trait DirectedAncestry {
    /// Nodes in dense order.
    fn nodes(&self) -> &[NodeRef];
    /// Directed descendants of `nodes`, including `nodes` themselves.
    fn descendants_of(&self, nodes: &[DenseNodeId], out: &mut BitSet, ws: &mut GraphWorkspace);
}

impl DirectedAncestry for Dag {
    fn nodes(&self) -> &[NodeRef] {
        Dag::nodes(self)
    }

    fn descendants_of(&self, nodes: &[DenseNodeId], out: &mut BitSet, ws: &mut GraphWorkspace) {
        Dag::descendants_of(self, nodes, out, ws);
    }
}

impl DirectedAncestry for Admg {
    fn nodes(&self) -> &[NodeRef] {
        Admg::nodes(self)
    }

    fn descendants_of(&self, nodes: &[DenseNodeId], out: &mut BitSet, ws: &mut GraphWorkspace) {
        Admg::descendants_of(self, nodes, out, ws);
    }
}

/// Result of `plan.retarget(weights, depends_on)`.
#[derive(Clone, Debug, PartialEq)]
pub struct RetargetResult {
    /// Per-column `E_Q[μ_a(X)]` (and per-threshold `F_a` / exceedance).
    pub summary: ScoreSummary,
    /// Active − control contrast on the mean columns when both arms exist.
    pub contrast: Option<LinearContrast>,
    /// Joint IF covariance (same object as `summary.covariance`).
    pub covariance: JointCovariance,
    /// Weighted support (per-arm `n_eff`, propensity range under `w`).
    pub support: WeightedSupport,
    /// Declared weight parents.
    pub depends_on: Arc<[VariableId]>,
    /// Whether monotone rearrangement was applied to an exceedance grid.
    pub monotone_rearranged: bool,
    /// Diagnostics (rearrangement, overlap).
    pub diagnostics: Vec<Diagnostic>,
}

/// Validate `depends_on` against the frozen certificate and graph.
///
/// Empty `depends_on` is legal here; [`retarget`] refuses it when `weights`
/// are nonconstant. Nonempty `depends_on` needs a directed graph (DAG or
/// ADMG) so descendant closure can be checked along directed edges only.
///
/// # Errors
///
/// Illegal dependence or weights that are not a function of the adjustment set.
pub fn check_depends_on(
    depends_on: &[VariableId],
    table: &ScoreTable,
    graph: Option<&dyn DirectedAncestry>,
) -> Result<(), EstimationError> {
    for &v in depends_on {
        if v == table.treatment || table.intervened.iter().any(|&x| x == v) {
            return Err(EstimationError::unsupported(RetargetRefusal::IllegalDependence.as_str()));
        }
        if !table.adjustment_set.iter().any(|&z| z == v) {
            return Err(EstimationError::unsupported(
                RetargetRefusal::OutsideAdjustmentSet.as_str(),
            ));
        }
    }
    if depends_on.is_empty() {
        return Ok(());
    }
    let Some(graph) = graph else {
        return Err(EstimationError::unsupported(
            RetargetRefusal::DescendantClosureUnavailable.as_str(),
        ));
    };
    if descendant_of_intervened(depends_on, table, graph)? {
        return Err(EstimationError::unsupported(RetargetRefusal::IllegalDependence.as_str()));
    }
    Ok(())
}

fn descendant_of_intervened(
    depends_on: &[VariableId],
    table: &ScoreTable,
    graph: &dyn DirectedAncestry,
) -> Result<bool, EstimationError> {
    let mut sources = vec![table.treatment];
    sources.extend(table.intervened.iter().copied());
    let mut ws = GraphWorkspace::default();
    let mut out = BitSet::default();
    let mut dense = Vec::with_capacity(sources.len());
    for &id in &sources {
        let Some(pos) = graph.nodes().iter().position(|n| n.variable() == id) else {
            return Err(EstimationError::unsupported(
                RetargetRefusal::DescendantClosureUnavailable.as_str(),
            ));
        };
        let Ok(raw) = u32::try_from(pos) else {
            return Err(EstimationError::unsupported(
                RetargetRefusal::DescendantClosureUnavailable.as_str(),
            ));
        };
        dense.push(DenseNodeId::from_raw(raw));
    }
    graph.descendants_of(&dense, &mut out, &mut ws);
    for &v in depends_on {
        if let Some(pos) = graph.nodes().iter().position(|n| n.variable() == v) {
            if out.contains(DenseNodeId::from_raw(pos as u32)) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Weights are constant when every finite entry equals the first, up to `1e-12`.
fn weights_are_constant(weights: &[f64]) -> bool {
    let Some(&first) = weights.first() else {
        return true;
    };
    weights.iter().all(|&w| (w - first).abs() <= 1e-12)
}

/// Estimate `E_Q[μ_a(X)]` from frozen scores. Does not refit nuisances.
///
/// `overlap_failed` is `true` when weighted support fails; the caller should
/// map that to a support refusal rather than a hard error.
///
/// # Errors
///
/// Illegal `depends_on`, weight shape, or empty mass.
pub fn retarget(
    table: &ScoreTable,
    weights: &[f64],
    depends_on: &[VariableId],
    graph: Option<&dyn DirectedAncestry>,
    treatment: Option<&[f64]>,
    propensity: Option<&[f64]>,
) -> Result<(RetargetResult, bool), EstimationError> {
    if weights.len() != table.n_rows {
        return Err(EstimationError::data_msg(
            "retarget weights must align with the prepared score-table rows",
        ));
    }
    if depends_on.is_empty() && !weights_are_constant(weights) {
        return Err(EstimationError::unsupported(
            RetargetRefusal::UndeclaredNonconstantWeights.as_str(),
        ));
    }
    check_depends_on(depends_on, table, graph)?;

    let (summary, monotone_rearranged, mut diagnostics) =
        summarize_functional(table, Some(weights))?;

    let contrast = ate_contrast(table, &summary)?;
    let support = if table.observed_arm.len() == table.n_rows {
        score_weighted_support(table, weights)
    } else {
        match treatment {
            Some(t) if t.len() == table.n_rows => {
                weighted_support(t, weights, propensity, MIN_WEIGHTED_ARM_N_EFF)
            }
            _ => WeightedSupport {
                n_eff: summary.n_eff,
                n_eff_by_arm: Vec::new(),
                propensity_range: None,
                overlap_ok: false,
            },
        }
    };
    let overlap_failed = !support.overlap_ok;
    if overlap_failed {
        diagnostics.push(Diagnostic::new(
            "retarget.weighted_overlap_failed",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            RetargetRefusal::WeightedOverlap.as_str(),
        ));
    }
    Ok((
        RetargetResult {
            covariance: summary.covariance.clone(),
            summary,
            contrast,
            support,
            depends_on: Arc::from(depends_on.to_vec()),
            monotone_rearranged,
            diagnostics,
        },
        overlap_failed,
    ))
}

/// Target-local support over every observed joint cell, using raw held-out propensities.
#[must_use]
pub fn score_weighted_support(table: &ScoreTable, weights: &[f64]) -> WeightedSupport {
    let arms: std::collections::BTreeSet<_> = table.columns.iter().map(|c| c.arm).collect();
    let n_eff_by_arm: Vec<f64> = arms
        .iter()
        .map(|&arm| {
            let w: Vec<_> = weights
                .iter()
                .zip(table.observed_arm.iter())
                .filter_map(|(&w, &a)| (a == arm).then_some(w))
                .collect();
            crate::joint_if::kish_n_eff(&w)
        })
        .collect();
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for column in table.propensities.chunks(table.n_rows.max(1)) {
        for (&p, &w) in column.iter().zip(weights) {
            if w > 0.0 {
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
    }
    let range = lo.is_finite().then_some((lo, hi));
    let overlap_ok = weights.len() == table.n_rows
        && table.observed_arm.len() == table.n_rows
        && n_eff_by_arm.iter().all(|&n| n >= MIN_WEIGHTED_ARM_N_EFF)
        && !n_eff_by_arm.is_empty()
        && range.is_some_and(|(lo, hi)| lo > 1e-6 && hi < 1.0 - 1e-6);
    WeightedSupport {
        n_eff: crate::joint_if::kish_n_eff(weights),
        n_eff_by_arm,
        propensity_range: range,
        overlap_ok,
    }
}

fn ate_contrast(
    table: &ScoreTable,
    summary: &ScoreSummary,
) -> Result<Option<LinearContrast>, EstimationError> {
    let mean: Vec<usize> = table
        .columns
        .iter()
        .enumerate()
        .filter(|(_, c)| c.threshold == table.columns.first().and_then(|c| c.threshold))
        .map(|(i, _)| i)
        .collect();
    if mean.len() < 2 {
        return Ok(None);
    }
    let mut c = vec![0.0; table.n_columns()];
    c[mean[0]] = -1.0;
    c[mean[1]] = 1.0;
    Ok(Some(table.linear_contrast(summary, &c)?))
}

/// Summarize scores and rearrange an exceedance grid when present.
///
/// # Errors
///
/// Weight length mismatch or empty mass.
pub fn summarize_functional(
    table: &ScoreTable,
    weights: Option<&[f64]>,
) -> Result<(ScoreSummary, bool, Vec<Diagnostic>), EstimationError> {
    let mut summary = table.summarize(weights)?;
    let mut diagnostics = Vec::new();
    let monotone_rearranged = rearrange_exceedance(&mut summary, table, &mut diagnostics);
    Ok((summary, monotone_rearranged, diagnostics))
}

/// `F_a(c) = 1 - P(Y(a) > c)` for every threshold column, in table order.
#[must_use]
pub fn exceedance_cdf_values(summary: &ScoreSummary, table: &ScoreTable) -> Option<Arc<[f64]>> {
    let values: Vec<f64> = table
        .columns
        .iter()
        .enumerate()
        .filter(|(_, col)| col.threshold.is_some())
        .map(|(j, _)| 1.0 - summary.means[j])
        .collect();
    if values.is_empty() { None } else { Some(values.into()) }
}

fn rearrange_exceedance(
    summary: &mut ScoreSummary,
    table: &ScoreTable,
    diagnostics: &mut Vec<Diagnostic>,
) -> bool {
    let mut by_arm: std::collections::BTreeMap<u32, Vec<(usize, f64)>> =
        std::collections::BTreeMap::new();
    for (j, col) in table.columns.iter().enumerate() {
        if let Some(c) = col.threshold {
            by_arm.entry(col.arm).or_default().push((j, c));
        }
    }
    if by_arm.is_empty() {
        return false;
    }
    let mut means = summary.means.to_vec();
    let mut changed = false;
    for grid in by_arm.values_mut() {
        grid.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let raw: Vec<f64> = grid.iter().map(|(j, _)| means[*j]).collect();
        let adj = monotone_decreasing(&raw);
        for (slot, value) in grid.iter().zip(adj) {
            let bounded = value.clamp(0.0, 1.0);
            changed |= means[slot.0].total_cmp(&bounded).is_ne();
            means[slot.0] = bounded;
        }
    }
    if !changed {
        return false;
    }
    summary.means = Arc::from(means);
    diagnostics.push(Diagnostic::new(
        "estimate.exceedance.monotone_rearranged",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "exceedance means and F_a(c) use bounded isotonic projection; joint covariance and score-inference bands describe the raw AIPW scores, not the rearranged means",
    ));
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scores::ScoreColumn;

    fn table() -> ScoreTable {
        ScoreTable {
            observed_arm: Arc::from([0, 1, 0, 1]),
            propensities: Arc::from([]),
            observed_outcome: Arc::from([0.0, 1.0, 0.0, 1.0]),
            n_rows: 4,
            row_index: Arc::from([0, 1, 2, 3]),
            fold_ids: Arc::from([0, 1, 0, 1]),
            n_folds: 2,
            scores: Arc::from([0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 1.0, 3.0]),
            columns: Arc::from([
                ScoreColumn { arm: 0, threshold: None },
                ScoreColumn { arm: 1, threshold: None },
            ]),
            adjustment_set: Arc::from([VariableId::from_raw(2)]),
            nuisance_provenance: Arc::from("aipw.crossfit.v1"),
            treatment: VariableId::from_raw(0),
            intervened: Arc::from([]),
        }
    }

    #[test]
    fn refuses_treatment_in_depends_on() {
        let t = table();
        let err = check_depends_on(&[VariableId::from_raw(0)], &t, None).unwrap_err();
        assert!(err.to_string().contains("treatment"));
    }

    #[test]
    fn nonempty_depends_on_without_directed_graph_refuses() {
        let t = table();
        let err = check_depends_on(&[VariableId::from_raw(2)], &t, None).unwrap_err();
        assert!(err.to_string().contains("descendant closure requires a directed graph"));
    }

    #[test]
    fn admg_depends_on_uses_directed_descendants_only() {
        let t = table();
        let mut g = Admg::with_variables(3);
        g.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        check_depends_on(&[VariableId::from_raw(2)], &t, Some(&g)).unwrap();
    }

    #[test]
    fn nonconstant_weights_require_nonempty_depends_on() {
        let t = table();
        let w = [1.0, 2.0, 1.0, 1.0];
        let err = retarget(&t, &w, &[], None, Some(&[0.0, 1.0, 0.0, 1.0]), None).unwrap_err();
        assert!(err.to_string().contains("nonempty depends_on is required for nonconstant"));
    }

    #[test]
    fn retarget_matches_weighted_mean() {
        let t = table();
        let w = [1.0, 1.0, 1.0, 1.0];
        let (out, failed) = retarget(&t, &w, &[], None, Some(&[0.0, 1.0, 0.0, 1.0]), None).unwrap();
        // Four rows cannot meet the licensed Kish-arm floor; the means still retarget.
        let _ = failed;
        assert!((out.summary.means[1] - 2.0).abs() < 1e-12);
        assert!((out.contrast.unwrap().value - 2.0).abs() < 1e-12);
        assert!(exceedance_cdf_values(&out.summary, &t).is_none());
    }

    #[test]
    fn exceedance_cdf_is_one_minus_threshold_means() {
        let mut t = table();
        t.columns = Arc::from([
            ScoreColumn { arm: 0, threshold: Some(0.0) },
            ScoreColumn { arm: 1, threshold: Some(0.0) },
            ScoreColumn { arm: 0, threshold: Some(1.0) },
            ScoreColumn { arm: 1, threshold: Some(1.0) },
        ]);
        t.scores = Arc::from([
            0.8, 0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6, 0.6, 0.2, 0.2, 0.2, 0.2,
        ]);
        let summary = t.summarize(None).unwrap();
        let cdf = exceedance_cdf_values(&summary, &t).unwrap();
        assert_eq!(cdf.len(), 4);
        assert!((cdf[0] - 0.2).abs() < 1e-12);
        assert!((cdf[1] - 0.6).abs() < 1e-12);
        assert!((cdf[2] - 0.4).abs() < 1e-12);
        assert!((cdf[3] - 0.8).abs() < 1e-12);
    }
}
