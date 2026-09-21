//! Honest causal forest CATE (`CausalForest`).
//!
//! Trees split to maximize treatment-effect heterogeneity; honesty estimates
//! leaf CATEs as an uncentered within-leaf difference in means on a held-out
//! half of each subsample, then predicts those leaf values in-sample. This is
//! not GRF local centering. Learners never choose the adjustment set.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_arguments
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalRng, ExecutionContext, StreamDomain, TargetPopulation,
};
use antecedent_data::TabularData;
use antecedent_expr::IdentifiedEstimand;

use crate::adjustment::EffectEstimate;
use crate::dml::DmlAte;
use crate::error::EstimationError;
use crate::prepare::{require_adjustment_shaped, validate_ate_query_with_targets};
use crate::propensity::{
    PreparedPropensityProblem, default_propensity_overlap, prepare_propensity_problem_with_registry,
};

const SUBSAMPLE_FRACTION: f64 = 0.5;
const MAX_CUTS: usize = 16;

/// Native honest causal forest with a cross-fitted AIPW marginal ATE.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalForest {
    /// Number of trees.
    pub n_trees: usize,
    /// Minimum samples in a child node (both arms still required).
    pub min_leaf: usize,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Honest sample split (splitting vs estimation).
    pub honesty: bool,
}

impl Default for CausalForest {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalForest {
    /// Two hundred honest trees, depth 8, min-leaf 8.
    #[must_use]
    pub const fn new() -> Self {
        Self { n_trees: 200, min_leaf: 8, max_depth: 8, honesty: true }
    }

    /// Tree count.
    #[must_use]
    pub const fn with_n_trees(mut self, n_trees: usize) -> Self {
        self.n_trees = n_trees;
        self
    }

    /// Minimum child size.
    #[must_use]
    pub const fn with_min_leaf(mut self, min_leaf: usize) -> Self {
        self.min_leaf = min_leaf;
        self
    }

    /// Maximum depth.
    #[must_use]
    pub const fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Honesty (Wager–Athey sample split).
    #[must_use]
    pub const fn with_honesty(mut self, honesty: bool) -> Self {
        self.honesty = honesty;
        self
    }

    /// Prepare the adjustment design for honest CATE trees and the AIPW marginal score.
    ///
    /// # Errors
    ///
    /// Non-adjustment estimand or prepare failure.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedPropensityProblem, EstimationError> {
        require_adjustment_shaped(estimand, "CausalForest requires an adjustment-shaped estimand")?;
        validate_ate_query_with_targets(query)?;
        prepare_propensity_problem_with_registry(
            data,
            estimand,
            query,
            default_propensity_overlap(),
            None,
        )
    }

    /// Grow CATE trees and estimate the marginal ATE with cross-fitted AIPW.
    ///
    /// # Errors
    ///
    /// Empty design, a target other than `AllObserved`, or a grow failure.
    pub fn fit(
        &self,
        problem: &PreparedPropensityProblem,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        if !matches!(problem.target_population, TargetPopulation::AllObserved) {
            return Err(EstimationError::TargetPopulation);
        }
        if self.n_trees == 0 {
            return Err(EstimationError::data_msg("CausalForest requires at least one tree"));
        }
        if self.min_leaf == 0 || self.max_depth == 0 {
            return Err(EstimationError::data_msg(
                "CausalForest min_leaf and max_depth must be ≥ 1",
            ));
        }
        let n = problem.nrows;
        if n < self.min_leaf.saturating_mul(2) {
            return Err(EstimationError::data_msg(
                "CausalForest needs more complete-case rows than 2 × min_leaf",
            ));
        }
        let p = problem.covariates.len();
        let mut x = vec![0.0; n.saturating_mul(p)];
        for (j, col) in problem.covariates.iter().enumerate() {
            if col.len() != n {
                return Err(EstimationError::data_msg("covariate column length != nrows"));
            }
            x[j * n..(j + 1) * n].copy_from_slice(col.as_ref());
        }
        let y = problem.outcome.as_ref();
        let t = problem.treatment.as_ref();
        if !t.iter().all(|&ti| ti.abs() <= 1e-12 || (ti - 1.0).abs() <= 1e-12) {
            return Err(EstimationError::data_msg(
                "CausalForest requires binary treatment in {0, 1}",
            ));
        }
        let min_leaf = self.min_leaf;
        let max_depth = self.max_depth;
        let honesty = self.honesty;
        let trees = ctx.map_indexed(self.n_trees, |b, inner| {
            let mut rng = inner.rng.stream_for(StreamDomain::Estimate, 0xC0F0_0000_u64 ^ b as u64);
            Ok::<_, EstimationError>(grow_tree(
                &x, n, p, y, t, min_leaf, max_depth, honesty, &mut rng,
            ))
        })?;
        let mut cate = vec![0.0; n];
        let mut hits = vec![0.0; n];
        let mut se_ss = vec![0.0; n];
        let mut se_hits = vec![0.0; n];
        for tree in &trees {
            for i in 0..n {
                if let Some((tau, se)) = tree.predict_leaf(&x, n, p, i) {
                    cate[i] += tau;
                    hits[i] += 1.0;
                    if let Some(leaf_se) = se {
                        se_ss[i] += leaf_se * leaf_se;
                        se_hits[i] += 1.0;
                    }
                }
            }
        }
        let mut cate_se = vec![0.0; n];
        let mut se_complete = true;
        for i in 0..n {
            if hits[i] > 0.0 {
                cate[i] /= hits[i];
            }
            // Sqrt of the mean of honest two-sample leaf variances over trees
            // that had both arms with n≥2. Trees that only return a point
            // (single-observation arm) contribute to the CATE, not the SE.
            if se_hits[i] > 0.0 {
                cate_se[i] = cate_se_from_leaf_ses(se_ss[i], se_hits[i]);
            } else {
                se_complete = false;
            }
        }
        // Variation across fitted CATEs is heterogeneity, not sampling
        // uncertainty in the ATE. Use the orthogonal marginal score instead.
        let mut effect = DmlAte::new().with_folds(5.min(n)).fit(problem, ctx, assumptions)?;
        let portable_trees = trees
            .iter()
            .map(|tree| {
                let mut nodes = Vec::new();
                tree.portable_nodes(&mut nodes);
                nodes
            })
            .collect();
        let model = crate::FittedEffect {
            version: 1,
            features: problem.adjustment_set.iter().map(|v| v.raw()).collect(),
            intercept: false,
            predictor: antecedent_learn::PortablePredictor {
                version: 1,
                columns: p,
                provenance: antecedent_learn::LearnerProvenance {
                    spec: "causal_forest".into(),
                    implementation: "antecedent".into(),
                    version: env!("CARGO_PKG_VERSION").into(),
                },
                model: antecedent_learn::PredictionMap::Trees {
                    trees: portable_trees,
                    base: 0.0,
                    average: true,
                    logistic: false,
                    probability: false,
                },
            },
        };
        model.validate()?;
        effect.fitted_effect = Some(Arc::new(model));
        if hits.iter().any(|&count| count == 0.0) {
            return Err(EstimationError::data_msg(
                "CausalForest has no honest two-arm estimate for a row; increase tree count or sample size",
            ));
        }
        Ok(effect
            .with_cate(Some(Arc::from(cate)))
            .with_cate_se(se_complete.then_some(Arc::from(cate_se))))
    }
}

#[derive(Clone, Debug)]
enum Node {
    Leaf { cate: Option<f64>, se: Option<f64> },
    Split { feature: usize, threshold: f64, left: Box<Node>, right: Box<Node> },
}

impl Node {
    fn portable_nodes(&self, nodes: &mut Vec<antecedent_learn::PredictionNode>) -> usize {
        use antecedent_learn::PredictionNode;
        let index = nodes.len();
        nodes.push(PredictionNode::Leaf { value: None });
        nodes[index] = match self {
            Self::Leaf { cate, .. } => PredictionNode::Leaf { value: *cate },
            Self::Split { feature, threshold, left, right } => {
                let left = left.portable_nodes(nodes);
                let right = right.portable_nodes(nodes);
                PredictionNode::Split {
                    feature: *feature,
                    threshold: *threshold,
                    inclusive: true,
                    left,
                    right,
                    missing: right,
                }
            }
        };
        index
    }

    fn predict_leaf(
        &self,
        x: &[f64],
        n: usize,
        p: usize,
        row: usize,
    ) -> Option<(f64, Option<f64>)> {
        match self {
            Self::Leaf { cate, se } => cate.map(|tau| (tau, *se)),
            Self::Split { feature, threshold, left, right } => {
                if *feature >= p {
                    return None;
                }
                let v = x[*feature * n + row];
                if v <= *threshold {
                    left.predict_leaf(x, n, p, row)
                } else {
                    right.predict_leaf(x, n, p, row)
                }
            }
        }
    }
}

fn grow_tree(
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    t: &[f64],
    min_leaf: usize,
    max_depth: usize,
    honesty: bool,
    rng: &mut CausalRng,
) -> Node {
    let sample = subsample(n, SUBSAMPLE_FRACTION, min_leaf, rng);
    if sample.len() < min_leaf.saturating_mul(2) {
        return Node::Leaf { cate: leaf_tau(y, t, &sample), se: leaf_se(y, t, &sample) };
    }
    let (split_idx, est_idx) = if honesty {
        let mid = sample.len() / 2;
        (sample[..mid].to_vec(), sample[mid..].to_vec())
    } else {
        (sample.clone(), sample)
    };
    let mut root = grow_from(&split_idx, 0, x, n, p, y, t, min_leaf, max_depth, rng);
    fill_cate(&mut root, &est_idx, x, n, p, y, t, None);
    root
}

fn subsample(n: usize, frac: f64, min_leaf: usize, rng: &mut CausalRng) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..n as u32).collect();
    let want = ((n as f64) * frac).round() as usize;
    let k = want.clamp(min_leaf.saturating_mul(2).min(n), n);
    for i in 0..k {
        let rest = n - i;
        let j = i + (rng.next_u64() as usize % rest);
        idx.swap(i, j);
    }
    idx.truncate(k);
    idx
}

fn grow_from(
    rows: &[u32],
    depth: usize,
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    t: &[f64],
    min_leaf: usize,
    max_depth: usize,
    rng: &mut CausalRng,
) -> Node {
    if depth >= max_depth || rows.len() < min_leaf.saturating_mul(2) || p == 0 {
        return Node::Leaf { cate: leaf_tau(y, t, rows), se: leaf_se(y, t, rows) };
    }
    let Some((feature, threshold, left_rows, right_rows)) =
        best_split(rows, x, n, p, y, t, min_leaf, rng)
    else {
        return Node::Leaf { cate: leaf_tau(y, t, rows), se: leaf_se(y, t, rows) };
    };
    Node::Split {
        feature,
        threshold,
        left: Box::new(grow_from(&left_rows, depth + 1, x, n, p, y, t, min_leaf, max_depth, rng)),
        right: Box::new(grow_from(&right_rows, depth + 1, x, n, p, y, t, min_leaf, max_depth, rng)),
    }
}

fn best_split(
    rows: &[u32],
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    t: &[f64],
    min_leaf: usize,
    rng: &mut CausalRng,
) -> Option<(usize, f64, Vec<u32>, Vec<u32>)> {
    let feats = choose_features(p, rng);
    let mut best_score = f64::NEG_INFINITY;
    let mut best: Option<(usize, f64)> = None;
    for &feat in &feats {
        let mut order = rows.to_vec();
        order.sort_by(|&a, &b| x[feat * n + a as usize].total_cmp(&x[feat * n + b as usize]));
        let cuts = candidate_thresholds(&order, x, n, feat);
        let total = arm_sums(y, t, &order);
        let mut left_sums = [0.0; 4];
        let mut split = 0;
        for thr in cuts {
            while split < order.len() && x[feat * n + order[split] as usize] <= thr {
                let i = order[split] as usize;
                let arm = if t[i] > 0.5 { 0 } else { 2 };
                left_sums[arm] += y[i];
                left_sums[arm + 1] += 1.0;
                split += 1;
            }
            if split < min_leaf || order.len() - split < min_leaf {
                continue;
            }
            let right_sums = std::array::from_fn(|i| total[i] - left_sums[i]);
            let Some(tl) = tau_from_sums(left_sums) else { continue };
            let Some(tr) = tau_from_sums(right_sums) else { continue };
            let nl = split as f64;
            let nr = (order.len() - split) as f64;
            let nn = (nl + nr).max(1.0);
            let score = (nl * nr / nn.powi(2)) * (tl - tr).powi(2);
            if score > best_score {
                best_score = score;
                best = Some((feat, thr));
            }
        }
    }
    let (feat, thr) = best?;
    let (left, right) = partition(rows, x, n, feat, thr);
    Some((feat, thr, left, right))
}

fn choose_features(p: usize, rng: &mut CausalRng) -> Vec<usize> {
    if p == 0 {
        return Vec::new();
    }
    let mut feats: Vec<usize> = (0..p).collect();
    if p <= 4 {
        return feats;
    }
    let mtry = (p as f64).sqrt().ceil() as usize;
    for i in 0..mtry {
        let j = i + (rng.next_u64() as usize % (p - i));
        feats.swap(i, j);
    }
    feats.truncate(mtry);
    feats
}

fn candidate_thresholds(order: &[u32], x: &[f64], n: usize, feat: usize) -> Vec<f64> {
    if order.len() < 2 {
        return Vec::new();
    }
    let mut vals: Vec<f64> = order.iter().map(|&i| x[feat * n + i as usize]).collect();
    vals.dedup_by(|a, b| (*a - *b).abs() <= 1e-12);
    if vals.len() < 2 {
        return Vec::new();
    }
    let mut cuts = Vec::new();
    let gaps = vals.len() - 1;
    let stride = gaps.div_ceil(MAX_CUTS).max(1);
    let mut i = 0;
    while i < gaps {
        cuts.push(0.5 * vals[i] + 0.5 * vals[i + 1]);
        i += stride;
    }
    cuts
}

fn partition(rows: &[u32], x: &[f64], n: usize, feat: usize, thr: f64) -> (Vec<u32>, Vec<u32>) {
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &i in rows {
        if x[feat * n + i as usize] <= thr {
            left.push(i);
        } else {
            right.push(i);
        }
    }
    (left, right)
}

fn fill_cate(
    node: &mut Node,
    rows: &[u32],
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    t: &[f64],
    parent: Option<(f64, Option<f64>)>,
) {
    let honest = leaf_stat(y, t, rows).or(parent);
    match node {
        Node::Leaf { cate, se } => {
            // Empty/single-arm estimation leaves must never reuse split outcomes.
            *cate = honest.map(|h| h.0);
            *se = honest.and_then(|h| h.1);
        }
        Node::Split { feature, threshold, left, right } => {
            if *feature >= p {
                return;
            }
            let (l, r) = partition(rows, x, n, *feature, *threshold);
            fill_cate(left, &l, x, n, p, y, t, honest);
            fill_cate(right, &r, x, n, p, y, t, honest);
        }
    }
}

fn arm_sums(y: &[f64], t: &[f64], rows: &[u32]) -> [f64; 4] {
    let mut sums = [0.0; 4];
    for &i in rows {
        let i = i as usize;
        let arm = if t[i] > 0.5 { 0 } else { 2 };
        sums[arm] += y[i];
        sums[arm + 1] += 1.0;
    }
    sums
}

fn tau_from_sums([s1, n1, s0, n0]: [f64; 4]) -> Option<f64> {
    if n1 < 1.0 || n0 < 1.0 { None } else { Some(s1 / n1 - s0 / n0) }
}

fn leaf_tau(y: &[f64], t: &[f64], rows: &[u32]) -> Option<f64> {
    leaf_stat(y, t, rows).map(|s| s.0)
}

fn leaf_se(y: &[f64], t: &[f64], rows: &[u32]) -> Option<f64> {
    leaf_stat(y, t, rows).and_then(|s| s.1)
}

fn leaf_stat(y: &[f64], t: &[f64], rows: &[u32]) -> Option<(f64, Option<f64>)> {
    let [s1, n1, s0, n0] = arm_sums(y, t, rows);
    let tau = tau_from_sums([s1, n1, s0, n0])?;
    if n1 < 2.0 || n0 < 2.0 {
        return Some((tau, None));
    }
    let mut q1 = 0.0;
    let mut q0 = 0.0;
    for &i in rows {
        let i = i as usize;
        if t[i] > 0.5 {
            q1 += y[i] * y[i];
        } else {
            q0 += y[i] * y[i];
        }
    }
    let v1 = (q1 - s1 * s1 / n1) / (n1 - 1.0);
    let v0 = (q0 - s0 * s0 / n0) / (n0 - 1.0);
    let var = v1.max(0.0) / n1 + v0.max(0.0) / n0;
    Some((tau, var.is_finite().then_some(var.sqrt())))
}

/// Pointwise CATE SE from summed leaf SEs: `sqrt(mean of leaf variances)`.
///
/// `se_ss` accumulates `leaf_se²` and `se_hits` counts trees that supplied an SE.
fn cate_se_from_leaf_ses(se_ss: f64, se_hits: f64) -> f64 {
    (se_ss / se_hits).max(0.0).sqrt()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{
        AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
        SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use antecedent_expr::{ExprId, IdentifiedEstimand};
    use antecedent_kernels::standard_normal;

    use super::*;

    #[test]
    fn honest_leaf_never_reuses_split_outcomes() {
        let mut node = Node::Leaf { cate: Some(999.0), se: None };
        fill_cate(&mut node, &[0], &[], 1, 0, &[10.0], &[1.0], None);
        assert!(node.predict_leaf(&[], 1, 0, 0).is_none());
        fill_cate(&mut node, &[0], &[], 1, 0, &[10.0], &[1.0], Some((2.0, Some(0.5))));
        assert_eq!(node.predict_leaf(&[], 1, 0, 0), Some((2.0, Some(0.5))));
    }

    #[test]
    fn cate_se_equals_leaf_se_when_all_trees_agree() {
        // B identical leaf SEs of s → cate_se = s, not s/√B.
        let b = 25usize;
        let s = 0.4;
        let mut se_ss = 0.0;
        let mut se_hits = 0.0;
        for _ in 0..b {
            se_ss += s * s;
            se_hits += 1.0;
        }
        let got = cate_se_from_leaf_ses(se_ss, se_hits);
        assert!((got - s).abs() < 1e-12, "expected {s}, got {got}");
        let divided_by_sqrt_b = se_ss.sqrt() / se_hits;
        assert!(
            (divided_by_sqrt_b - s / (b as f64).sqrt()).abs() < 1e-12,
            "old formula must be s/√B so the regression stays meaningful"
        );
        assert!((got - divided_by_sqrt_b).abs() > 1e-6);
    }

    fn interaction_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand, Vec<f64>) {
        let mut rng =
            ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0x51u64);
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = standard_normal(&mut rng);
            let p = 1.0 / (1.0 + (-(-0.3 + 0.6 * zi)).exp());
            let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
            z[i] = zi;
            t[i] = ti;
            y[i] = (1.0 + zi) * ti + 0.4 * zi + standard_normal(&mut rng) * 0.4;
        }
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "z",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(z.clone()),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        (
            TabularData::new(storage),
            IdentifiedEstimand::backdoor(
                "backdoor.adjustment",
                Arc::from([VariableId::from_raw(2)]),
                ExprId::from_raw(0),
            ),
            z,
        )
    }

    #[test]
    fn interaction_recovers_monotone_cate() {
        let (data, estimand, z) = interaction_scm(500, 7);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = CausalForest::new().with_n_trees(80).with_max_depth(4).with_min_leaf(8);
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let effect = est.fit(&prep, &ExecutionContext::for_tests(2), AssumptionSet::new()).unwrap();
        let cate = effect.cate.as_ref().expect("cate");
        assert_eq!(cate.len(), z.len());
        let model = effect.fitted_effect.as_ref().expect("portable forest");
        let predicted = model
            .predict(&prep.adjustment_set, &[&z], z.len(), &ExecutionContext::for_tests(2))
            .unwrap();
        assert_eq!(predicted.as_slice(), cate.as_ref());
        assert!(effect.se_analytic > 0.0);
        assert!(effect.influence.as_ref().unwrap().iter().sum::<f64>().abs() < 1e-8);
        let mut pairs: Vec<(f64, f64)> =
            z.iter().zip(cate.iter()).map(|(&zi, &c)| (zi, c)).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let q = pairs.len() / 4;
        let lo = pairs[..q].iter().map(|p| p.1).sum::<f64>() / q as f64;
        let hi = pairs[3 * q..].iter().map(|p| p.1).sum::<f64>() / (pairs.len() - 3 * q) as f64;
        assert!(hi > lo + 0.15, "cate should rise with z: lo={lo} hi={hi}");
        let se = effect.cate_se.as_ref().expect("honest leaf SEs");
        assert_eq!(se.len(), cate.len());
        assert!(se.iter().all(|&s| s.is_finite() && s >= 0.0));
        assert!(se.iter().any(|&s| s > 0.0));
    }
}
