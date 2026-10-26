//! Honest causal forest CATE (`CausalForest`).
//!
//! Trees split to maximize treatment-effect heterogeneity; honesty estimates
//! leaf CATEs as an uncentered within-leaf difference in means on a held-out
//! half of each subsample, then predicts those leaf values in-sample. This is
//! not GRF local centering. Learners never choose the adjustment set.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalRng, ExecutionContext, StreamDomain, TargetPopulation,
};
use antecedent_data::{TableView, TabularData};
use antecedent_expr::IdentifiedEstimand;

use crate::adjustment::EffectEstimate;
use crate::cluster_dml_aipw::DeclaredUnits;
use crate::dml::DmlAte;
use crate::error::EstimationError;
use crate::prepare::{require_adjustment_shaped, validate_ate_query_with_targets};
use crate::propensity::{
    PreparedPropensityProblem, default_propensity_overlap, prepare_propensity_problem_with_registry,
};

const SUBSAMPLE_FRACTION: f64 = 0.5;
/// Floor of the candidate cut count per feature; larger nodes scale it with the node size.
const MIN_CUTS: usize = 16;

/// Native honest causal forest with a cross-fitted AIPW marginal ATE.
///
/// **Clusters.** With a declared independence unit ([`Self::units`]) rows are no longer
/// exchangeable draws, so the forest respects the unit at every stage: each tree is grown on a
/// subsample of whole clusters, honesty splits those clusters (not their rows) between the
/// placement of splits and the estimation of leaf effects, so no cluster informs both, and the
/// marginal ATE is the whole-cluster cross-fitted AIPW score with its cluster-sandwich
/// standard error. The per-row leaf dispersion stays a descriptive measure of effect
/// heterogeneity, not a sampling standard error.
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
    /// Candidate split cut points per feature at every node. `None` scales the count with the
    /// node size (at least 16, otherwise the square root of the node's rows), so large nodes are
    /// split on a finer histogram.
    pub max_cuts: Option<usize>,
    /// Declared independence unit of the rows. Without a declaration rows are the unit.
    pub units: DeclaredUnits,
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
        Self {
            n_trees: 200,
            min_leaf: 8,
            max_depth: 8,
            honesty: true,
            max_cuts: None,
            units: DeclaredUnits {
                spec: None,
                first: None,
                first_source: None,
                second: None,
                second_source: None,
            },
        }
    }

    /// Declare the independence unit: whole-cluster subsampling, honesty and cross-fitting over
    /// the unit labels set with [`Self::with_cluster_ids`] or [`Self::with_source_cluster_ids`].
    #[must_use]
    pub fn with_cluster_dml(mut self, spec: crate::cluster_dml_aipw::ClusterDml) -> Self {
        self.units.spec = Some(spec);
        self
    }

    /// Cluster label of every analysed row.
    #[must_use]
    pub fn with_cluster_ids(mut self, labels: Vec<u32>) -> Self {
        self.units.first = Some(labels);
        self
    }

    /// Cluster label of every row of the source table.
    #[must_use]
    pub fn with_source_cluster_ids(mut self, labels: Vec<u32>) -> Self {
        self.units.first_source = Some(labels);
        self
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

    /// Fix the number of candidate cut points per feature (`None` scales with node size).
    #[must_use]
    pub const fn with_max_cuts(mut self, max_cuts: Option<usize>) -> Self {
        self.max_cuts = max_cuts;
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
        self.require_row_units()?;
        self.units.check_source_rows(data.row_count())?;
        let mut problem = prepare_propensity_problem_with_registry(
            data,
            estimand,
            query,
            default_propensity_overlap(),
            None,
        )?;
        self.units.declare(
            &mut problem,
            query.outcome_functional == antecedent_core::OutcomeFunctional::Mean,
            default_propensity_overlap(),
        )?;
        Ok(problem)
    }

    /// A forest has one cluster dimension: a dyadic declaration, or labels without a declared
    /// unit, are refused.
    fn require_row_units(&self) -> Result<(), EstimationError> {
        if self.units.has_second() {
            return Err(crate::sampling_weights::unsupported_route(
                "a causal forest has one cluster dimension; a dyadic unit has no whole-cluster \
                 subsample",
            ));
        }
        let labelled = self.units.first.is_some() || self.units.first_source.is_some();
        if labelled != self.units.spec.is_some() {
            return Err(EstimationError::unsupported(
                "causal-forest clusters need both the unit labels and the cluster declaration",
            ));
        }
        Ok(())
    }

    /// The marginal-ATE estimator on the forest's nuisances and units.
    fn marginal(&self, n: usize) -> DmlAte {
        let mut dml = DmlAte::new().with_folds(5.min(n));
        dml.units = self.units.clone();
        dml
    }

    /// Cross-fitted AIPW score table of the marginal ATE [`Self::fit`] reports. That ATE is the
    /// [`DmlAte`] orthogonal AIPW score, so this is [`DmlAte::score_table`] on the same
    /// nuisances; the forest's CATE contributes no score.
    ///
    /// # Errors
    ///
    /// A target other than `AllObserved` or a failed nuisance fit.
    pub fn score_table(
        &self,
        problem: &PreparedPropensityProblem,
        ctx: &ExecutionContext,
    ) -> Result<Option<crate::scores::ScoreTable>, EstimationError> {
        if !matches!(problem.target_population, TargetPopulation::AllObserved) {
            return Err(EstimationError::TargetPopulation);
        }
        self.marginal(problem.nrows).score_table(problem, ctx)
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
        if self.min_leaf == 0 || self.max_depth == 0 || self.max_cuts == Some(0) {
            return Err(EstimationError::data_msg(
                "CausalForest min_leaf and max_depth must be ≥ 1",
            ));
        }
        let n = problem.nrows;
        // Trees index rows as `u32`; a larger table would wrap subsample indices.
        if u32::try_from(n).is_err() {
            return Err(EstimationError::data_msg("CausalForest rows exceed u32 index capacity"));
        }
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
        self.require_row_units()?;
        let cluster_rows = self
            .units
            .labels(&problem.row_index)?
            .0
            .map(|labels| crate::util::ClusterRows::new(&labels))
            .transpose()?;
        let data = ForestData {
            x: &x,
            n,
            p,
            y,
            t,
            clusters: cluster_rows.as_ref().map(crate::util::ClusterRows::members),
        };
        let shape = TreeShape {
            min_leaf: self.min_leaf,
            max_depth: self.max_depth,
            max_cuts: self.max_cuts,
        };
        let honesty = self.honesty;
        let trees = ctx.map_indexed(self.n_trees, |b, inner| {
            let mut rng = inner.rng.stream_for(
                StreamDomain::Estimate,
                antecedent_core::derive_stream(0xC0F0_0000, b as u64),
            );
            Ok::<_, EstimationError>(grow_tree(&data, shape, honesty, &mut rng))
        })?;
        let ForestCate { cate, hits, leaf_dispersion, se_complete } =
            aggregate_forest_cate(&trees, &x, n, p);
        // Variation across fitted CATEs is heterogeneity, not sampling
        // uncertainty in the ATE. Use the orthogonal marginal score instead.
        let mut effect = self.marginal(n).fit(problem, ctx, assumptions)?;
        let model = portable_forest(&trees, problem, p);
        model.validate()?;
        effect.heterogeneity.fitted_effect = Some(Arc::new(model));
        if hits.iter().any(|&count| count == 0.0) {
            return Err(EstimationError::data_msg(
                "CausalForest has no honest two-arm estimate for a row; increase tree count or sample size",
            ));
        }
        Ok(effect
            .with_cate(Some(Arc::from(cate)))
            .with_cate_leaf_dispersion(se_complete.then_some(Arc::from(leaf_dispersion))))
    }
}

/// Portable form of the grown forest: every tree's nodes, averaged at prediction.
fn portable_forest(
    trees: &[Node],
    problem: &PreparedPropensityProblem,
    columns: usize,
) -> crate::FittedEffect {
    let portable_trees = trees
        .iter()
        .map(|tree| {
            let mut nodes = Vec::new();
            tree.portable_nodes(&mut nodes);
            nodes
        })
        .collect();
    let features: Vec<u32> = problem.adjustment_set.iter().map(|v| v.raw()).collect();
    let covariate_columns: Vec<&[f64]> = problem.covariates.iter().map(|c| &c[..]).collect();
    let support = if covariate_columns.len() == features.len() {
        antecedent_learn::EffectSupport::from_rows(&covariate_columns, None)
    } else {
        None
    };
    crate::FittedEffect {
        version: 1,
        features,
        support,
        intercept: false,
        predictor: antecedent_learn::PortablePredictor {
            version: 1,
            columns,
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
    }
}

/// Per-row forest CATE, its leaf-hit counts, and its standard error.
struct ForestCate {
    cate: Vec<f64>,
    hits: Vec<f64>,
    leaf_dispersion: Vec<f64>,
    /// Every row had at least one tree with a two-sample leaf variance.
    se_complete: bool,
}

/// Average the per-tree leaf CATEs (and honest leaf variances) at every row.
fn aggregate_forest_cate(trees: &[Node], x: &[f64], n: usize, p: usize) -> ForestCate {
    let mut cate = vec![0.0; n];
    let mut hits = vec![0.0; n];
    let mut se_ss = vec![0.0; n];
    let mut se_hits = vec![0.0; n];
    for tree in trees {
        for i in 0..n {
            if let Some((tau, se)) = tree.predict_leaf(x, n, p, i) {
                cate[i] += tau;
                hits[i] += 1.0;
                if let Some(leaf_se) = se {
                    se_ss[i] += leaf_se * leaf_se;
                    se_hits[i] += 1.0;
                }
            }
        }
    }
    let mut leaf_dispersion = vec![0.0; n];
    let mut se_complete = true;
    for i in 0..n {
        if hits[i] > 0.0 {
            cate[i] /= hits[i];
        }
        // Sqrt of the mean of honest two-sample leaf variances over trees
        // that had both arms with n≥2. Trees that only return a point
        // (single-observation arm) contribute to the CATE, not the SE.
        if se_hits[i] > 0.0 {
            leaf_dispersion[i] = leaf_dispersion_from_leaf_ses(se_ss[i], se_hits[i]);
        } else {
            se_complete = false;
        }
    }
    ForestCate { cate, hits, leaf_dispersion, se_complete }
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

/// The rows a forest is grown on: column-major covariates (`n × p`) with the outcome and
/// the binary treatment.
#[derive(Clone, Copy)]
struct ForestData<'a> {
    x: &'a [f64],
    n: usize,
    p: usize,
    y: &'a [f64],
    t: &'a [f64],
    /// The rows of each declared cluster, when rows are not the independence unit.
    clusters: Option<&'a [Vec<antecedent_data::GlobalId>]>,
}

/// Stopping rules shared by every tree of the forest.
#[derive(Clone, Copy)]
struct TreeShape {
    min_leaf: usize,
    max_depth: usize,
    max_cuts: Option<usize>,
}

fn grow_tree(data: &ForestData<'_>, shape: TreeShape, honesty: bool, rng: &mut CausalRng) -> Node {
    let ForestData { n, y, t, clusters, .. } = *data;
    let TreeShape { min_leaf, .. } = shape;
    let (split_idx, est_idx) = if let Some(members) = clusters {
        cluster_subsample(members, SUBSAMPLE_FRACTION, honesty, rng)
    } else {
        let sample = subsample(n, SUBSAMPLE_FRACTION, min_leaf, rng);
        if sample.len() < min_leaf.saturating_mul(2) {
            return Node::Leaf { cate: leaf_tau(y, t, &sample), se: leaf_se(y, t, &sample) };
        }
        if honesty {
            let mid = sample.len() / 2;
            (sample[..mid].to_vec(), sample[mid..].to_vec())
        } else {
            (sample.clone(), sample)
        }
    };
    let mut root = grow_from(&split_idx, 0, data, shape, rng);
    fill_cate(&mut root, &est_idx, data, None);
    root
}

/// Rows of a subsample of whole clusters, drawn without replacement: about `frac` of the
/// clusters. With honesty the sampled clusters are halved, one half placing splits and the
/// other estimating leaf effects, so no cluster contributes to both; otherwise both sets are
/// every sampled row.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "frac is in (0, 1], so the rounded count is a non-negative integer no larger than the cluster count; a u64 draw reduced modulo rest is below rest"
)]
fn cluster_subsample(
    members: &[Vec<antecedent_data::GlobalId>],
    frac: f64,
    honesty: bool,
    rng: &mut CausalRng,
) -> (Vec<antecedent_data::GlobalId>, Vec<antecedent_data::GlobalId>) {
    let groups = members.len();
    let mut order: Vec<usize> = (0..groups).collect();
    let want = ((groups as f64) * frac).round() as usize;
    // Honesty needs a cluster on each side.
    let k = want.clamp(2.min(groups), groups);
    for i in 0..k {
        let rest = groups - i;
        let j = i + (rng.next_u64() as usize % rest);
        order.swap(i, j);
    }
    order.truncate(k);
    let rows_of = |clusters: &[usize]| -> Vec<antecedent_data::GlobalId> {
        clusters.iter().flat_map(|&g| members[g].iter().copied()).collect()
    };
    if honesty {
        let (placement, estimation) = order.split_at(k / 2);
        (rows_of(placement), rows_of(estimation))
    } else {
        let all = rows_of(&order);
        (all.clone(), all)
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "frac is in (0, 1], so the rounded count is a non-negative integer no larger than n; a u64 draw reduced modulo rest is below rest"
)]
fn subsample(
    n: usize,
    frac: f64,
    min_leaf: usize,
    rng: &mut CausalRng,
) -> Vec<antecedent_data::GlobalId> {
    let mut idx: Vec<antecedent_data::GlobalId> =
        (0..n).map(|i| antecedent_data::GlobalId::new(u64::try_from(i).expect("row id"))).collect();
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
    rows: &[antecedent_data::GlobalId],
    depth: usize,
    data: &ForestData<'_>,
    shape: TreeShape,
    rng: &mut CausalRng,
) -> Node {
    let ForestData { p, y, t, .. } = *data;
    let TreeShape { min_leaf, max_depth, .. } = shape;
    if depth >= max_depth || rows.len() < min_leaf.saturating_mul(2) || p == 0 {
        return Node::Leaf { cate: leaf_tau(y, t, rows), se: leaf_se(y, t, rows) };
    }
    let Some((feature, threshold, left_rows, right_rows)) = best_split(rows, data, shape, rng)
    else {
        return Node::Leaf { cate: leaf_tau(y, t, rows), se: leaf_se(y, t, rows) };
    };
    Node::Split {
        feature,
        threshold,
        left: Box::new(grow_from(&left_rows, depth + 1, data, shape, rng)),
        right: Box::new(grow_from(&right_rows, depth + 1, data, shape, rng)),
    }
}

fn best_split(
    rows: &[antecedent_data::GlobalId],
    data: &ForestData<'_>,
    shape: TreeShape,
    rng: &mut CausalRng,
) -> Option<(usize, f64, Vec<antecedent_data::GlobalId>, Vec<antecedent_data::GlobalId>)> {
    let ForestData { x, n, p, y, t, .. } = *data;
    let TreeShape { min_leaf, max_cuts, .. } = shape;
    let max_cuts = max_cuts.unwrap_or_else(|| MIN_CUTS.max(rows.len().isqrt()));
    let feats = choose_features(p, rng);
    let mut best_score = f64::NEG_INFINITY;
    let mut best: Option<(usize, f64)> = None;
    // Row indices and their feature values, reused across features and sorted by value only
    // (a stable sort keeps ties in node-row order, so the scan order is unchanged).
    let mut keyed: Vec<(f64, usize)> = Vec::with_capacity(rows.len());
    for &feat in &feats {
        keyed.clear();
        keyed.extend(rows.iter().map(|&r| {
            let i = usize::try_from(r.get()).expect("row id");
            (x[feat * n + i], i)
        }));
        keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
        let cuts = candidate_thresholds(&keyed, max_cuts);
        let mut total = [0.0; 4];
        for &(_, i) in &keyed {
            let arm = if t[i] > 0.5 { 0 } else { 2 };
            total[arm] += y[i];
            total[arm + 1] += 1.0;
        }
        let mut left_sums = [0.0; 4];
        let mut split = 0;
        for thr in cuts {
            while split < keyed.len() && keyed[split].0 <= thr {
                let i = keyed[split].1;
                let arm = if t[i] > 0.5 { 0 } else { 2 };
                left_sums[arm] += y[i];
                left_sums[arm + 1] += 1.0;
                split += 1;
            }
            if split < min_leaf || keyed.len() - split < min_leaf {
                continue;
            }
            let right_sums = std::array::from_fn(|i| total[i] - left_sums[i]);
            let Some(tl) = tau_from_sums(left_sums) else { continue };
            let Some(tr) = tau_from_sums(right_sums) else { continue };
            let nl = split as f64;
            let nr = (keyed.len() - split) as f64;
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

#[allow(
    clippy::cast_possible_truncation,
    reason = "the mtry count is ceil(sqrt(p)) for a feature count p, far below usize::MAX, and the u64 draw is reduced modulo the remaining feature range so truncating it on a 32-bit target still yields a valid index"
)]
#[allow(clippy::cast_sign_loss, reason = "ceil(sqrt(p)) of a feature count is non-negative")]
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

fn candidate_thresholds(sorted: &[(f64, usize)], max_cuts: usize) -> Vec<f64> {
    if sorted.len() < 2 {
        return Vec::new();
    }
    let mut vals: Vec<f64> = sorted.iter().map(|&(v, _)| v).collect();
    vals.dedup_by(|a, b| (*a - *b).abs() <= 1e-12);
    if vals.len() < 2 {
        return Vec::new();
    }
    let mut cuts = Vec::new();
    let gaps = vals.len() - 1;
    let stride = gaps.div_ceil(max_cuts).max(1);
    let mut i = 0;
    while i < gaps {
        cuts.push(0.5 * vals[i] + 0.5 * vals[i + 1]);
        i += stride;
    }
    cuts
}

fn partition(
    rows: &[antecedent_data::GlobalId],
    x: &[f64],
    n: usize,
    feat: usize,
    thr: f64,
) -> (Vec<antecedent_data::GlobalId>, Vec<antecedent_data::GlobalId>) {
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &i in rows {
        if x[feat * n + usize::try_from(i.get()).expect("row id")] <= thr {
            left.push(i);
        } else {
            right.push(i);
        }
    }
    (left, right)
}

fn fill_cate(
    node: &mut Node,
    rows: &[antecedent_data::GlobalId],
    data: &ForestData<'_>,
    parent: Option<(f64, Option<f64>)>,
) {
    let ForestData { x, n, p, y, t, .. } = *data;
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
            fill_cate(left, &l, data, honest);
            fill_cate(right, &r, data, honest);
        }
    }
}

fn arm_sums(y: &[f64], t: &[f64], rows: &[antecedent_data::GlobalId]) -> [f64; 4] {
    let mut sums = [0.0; 4];
    for &i in rows {
        let i = usize::try_from(i.get()).expect("row id");
        let arm = if t[i] > 0.5 { 0 } else { 2 };
        sums[arm] += y[i];
        sums[arm + 1] += 1.0;
    }
    sums
}

fn tau_from_sums([s1, n1, s0, n0]: [f64; 4]) -> Option<f64> {
    if n1 < 1.0 || n0 < 1.0 { None } else { Some(s1 / n1 - s0 / n0) }
}

fn leaf_tau(y: &[f64], t: &[f64], rows: &[antecedent_data::GlobalId]) -> Option<f64> {
    leaf_stat(y, t, rows).map(|s| s.0)
}

fn leaf_se(y: &[f64], t: &[f64], rows: &[antecedent_data::GlobalId]) -> Option<f64> {
    leaf_stat(y, t, rows).and_then(|s| s.1)
}

fn leaf_stat(
    y: &[f64],
    t: &[f64],
    rows: &[antecedent_data::GlobalId],
) -> Option<(f64, Option<f64>)> {
    let [s1, n1, s0, n0] = arm_sums(y, t, rows);
    let tau = tau_from_sums([s1, n1, s0, n0])?;
    if n1 < 2.0 || n0 < 2.0 {
        return Some((tau, None));
    }
    let mut q1 = 0.0;
    let mut q0 = 0.0;
    for &i in rows {
        let i = usize::try_from(i.get()).expect("row id");
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

/// Leaf-dispersion diagnostic from summed leaf SEs: `sqrt(mean of leaf variances)`.
///
/// `se_ss` accumulates `leaf_se²` and `se_hits` counts trees that supplied an SE.
fn leaf_dispersion_from_leaf_ses(se_ss: f64, se_hits: f64) -> f64 {
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
        let data = ForestData { x: &[], n: 1, p: 0, y: &[10.0], t: &[1.0], clusters: None };
        fill_cate(&mut node, &[antecedent_data::GlobalId::new(0)], &data, None);
        assert!(node.predict_leaf(&[], 1, 0, 0).is_none());
        fill_cate(&mut node, &[antecedent_data::GlobalId::new(0)], &data, Some((2.0, Some(0.5))));
        assert_eq!(node.predict_leaf(&[], 1, 0, 0), Some((2.0, Some(0.5))));
    }

    #[test]
    fn leaf_dispersion_equals_leaf_se_when_all_trees_agree() {
        // B identical leaf SEs of s → dispersion = s, not s/√B.
        let b = 25usize;
        let s = 0.4;
        let mut se_ss = 0.0;
        let mut se_hits = 0.0;
        for _ in 0..b {
            se_ss += s * s;
            se_hits += 1.0;
        }
        let got = leaf_dispersion_from_leaf_ses(se_ss, se_hits);
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
            let p = antecedent_kernels::sigmoid(-0.3 + 0.6 * zi);
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
        let cate = effect.heterogeneity.cate.as_ref().expect("cate");
        assert_eq!(cate.len(), z.len());
        let model = effect.heterogeneity.fitted_effect.as_ref().expect("portable forest");
        let predicted = model
            .predict(&prep.adjustment_set, &[&z], z.len(), &ExecutionContext::for_tests(2))
            .unwrap();
        assert_eq!(predicted.as_slice(), cate.as_ref());
        assert!(effect.se_analytic() > 0.0);
        assert!(effect.scores.influence.as_ref().unwrap().iter().sum::<f64>().abs() < 1e-8);
        let mut pairs: Vec<(f64, f64)> =
            z.iter().zip(cate.iter()).map(|(&zi, &c)| (zi, c)).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let q = pairs.len() / 4;
        let lo = pairs[..q].iter().map(|p| p.1).sum::<f64>() / q as f64;
        let hi = pairs[3 * q..].iter().map(|p| p.1).sum::<f64>() / (pairs.len() - 3 * q) as f64;
        assert!(hi > lo + 0.15, "cate should rise with z: lo={lo} hi={hi}");
        assert!(effect.heterogeneity.cate_se.is_none(), "a forest publishes no pointwise SE");
        let se =
            effect.heterogeneity.cate_leaf_dispersion.as_ref().expect("honest leaf dispersion");
        assert_eq!(se.len(), cate.len());
        assert!(se.iter().all(|&s| s.is_finite() && s >= 0.0));
        assert!(se.iter().any(|&s| s > 0.0));
    }
    #[test]
    fn honest_cluster_subsample_never_puts_a_cluster_on_both_sides() {
        let members: Vec<Vec<antecedent_data::GlobalId>> = (0..12u64)
            .map(|g| (g * 5..g * 5 + 5).map(antecedent_data::GlobalId::new).collect())
            .collect();
        let owner = |row: antecedent_data::GlobalId| row.get() / 5;
        let mut rng = ExecutionContext::for_tests(5).rng.stream_for(StreamDomain::Estimate, 1);
        for _ in 0..50 {
            let (placement, estimation) = cluster_subsample(&members, 0.5, true, &mut rng);
            let side = |rows: &[antecedent_data::GlobalId]| -> std::collections::BTreeSet<u64> {
                rows.iter().map(|r| owner(*r)).collect()
            };
            let (p, e) = (side(&placement), side(&estimation));
            assert!(p.is_disjoint(&e), "a cluster informs both splitting and estimation");
            assert_eq!(p.len() + e.len(), 6, "about half the clusters are sampled");
            // Whole clusters only.
            assert_eq!(placement.len(), 5 * p.len());
            assert_eq!(estimation.len(), 5 * e.len());
        }
        let (all_a, all_b) = cluster_subsample(&members, 0.5, false, &mut rng);
        assert_eq!(all_a, all_b);
        assert_eq!(all_a.len() % 5, 0);
    }
}
