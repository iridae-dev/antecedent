//! Graph permutation falsification from a graph and data alone.
//!
//! A causal graph implies conditional independencies in the data it describes. The local Markov
//! property says each node is independent of its non-descendants given its parents; written as
//! pairwise implications, each non-parent non-descendant `j` of a node `i` yields
//! `i ⟂ j | parents(i)`. [`falsify_graph`] tests those implications with a configured
//! conditional-independence test (any name accepted by
//! [`ci_from_name`](antecedent_stats::ci::ci_from_name), nonlinear tests included) and counts the
//! violations at the declared level.
//!
//! A violation count alone does not say whether the graph is informative: a sparse graph that
//! claims many independencies will be violated by dependent data, but so is almost any graph on
//! unrelated variables. The count is therefore compared with the counts of the same graph with its
//! nodes randomly relabelled (a node permutation), recomputed on the same data. The reported
//! p-value is `(1 + #{permutations with at most as many violations}) / (1 + #permutations)`; the
//! graph is [`GraphFalsificationOutcome::Rejected`] when that exceeds the declared significance
//! level, i.e. when it is not better than random relabellings of itself.
//!
//! The comparison is only meaningful when few relabellings are Markov equivalent to the graph
//! (same skeleton and unshielded colliders): those imply the same independencies and can never be
//! worse. When their share exceeds the declared threshold, or the graph implies nothing testable
//! (a complete graph), the outcome is [`GraphFalsificationOutcome::Uninformative`]. A budget or a
//! cancellation that stops the procedure before the minimum number of permutations is
//! [`GraphFalsificationOutcome::Incomplete`], which makes no claim in either direction.
//!
//! This is not [`ModelEvaluator`](crate::ModelEvaluator)'s shuffled-outcome likelihood baseline.
//! That diagnostic shuffles one outcome column and re-scores a *fitted* model; this procedure
//! fits nothing, relabels the graph, and compares graph-implied independencies with the data.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use antecedent_core::{
    CausalRng, ExecutionContext, NodeRef, RngFactory, SearchBudget, SearchLimits, SearchReceipt,
    SearchStop, StreamDomain, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{BitSet, Dag, DenseNodeId, GraphWorkspace};
use antecedent_stats::ci::{
    CiBatchRequest, CiPreparationPlan, CiQuery, CiWorkspace, ConditionalIndependence,
    ConfidenceMethod, PreparedCiTest, SignificanceMethod, ci_from_name, ensure_alpha_resolvable,
};

use crate::error::ModelError;

/// Most relabellings enumerated and held in memory when `n!` fits the request. A larger
/// request over a small graph is sampled instead (reported as not exhaustive), so any
/// `n_permutations` is accepted while memory stays bounded.
pub const MAX_ENUMERATED_RELABELLINGS: usize = 1 << 20;

/// Default ceiling on distinct conditional-independence tests a run may evaluate.
pub const DEFAULT_MAX_CI_TESTS: usize = 200_000;

/// Index of the permutation stream within [`StreamDomain::Resample`].
const PERMUTATION_STREAM: u64 = 0x67_72_61_70_68;

/// Settings of [`falsify_graph`].
#[derive(Clone, Debug)]
pub struct GraphFalsificationConfig {
    /// Conditional-independence test, by the name [`ci_from_name`] resolves
    /// (`parcorr`, `gsquared`, `gpdc`, `knn_dependence`, ...).
    pub ci_test: Arc<str>,
    /// Level at which one implication counts as violated (`p < alpha`).
    pub alpha: f64,
    /// Level of the permutation comparison: the graph is rejected when its p-value exceeds it.
    pub significance: f64,
    /// Node relabellings to compare against. When `n!` does not exceed this, every
    /// non-identity relabelling is used.
    pub n_permutations: usize,
    /// Fewest completed relabellings that still support a verdict; fewer is `Incomplete`.
    pub min_permutations: usize,
    /// Largest share of relabellings (the graph itself included) that may be Markov equivalent to
    /// the graph before the comparison is called uninformative. `None` uses `significance`: with a
    /// larger share the p-value cannot fall below `significance`.
    pub max_equivalent_share: Option<f64>,
    /// Ceiling on distinct conditional-independence tests (cached results are free).
    pub max_ci_tests: usize,
    /// Null replicates for tests whose p-value comes from a permutation null.
    pub ci_replicates: u32,
    /// Seed of the relabelling stream; `None` uses the execution context's master seed.
    pub seed: Option<u64>,
}

impl Default for GraphFalsificationConfig {
    fn default() -> Self {
        Self {
            ci_test: Arc::from("parcorr"),
            alpha: 0.05,
            significance: 0.05,
            n_permutations: 200,
            min_permutations: 20,
            max_equivalent_share: None,
            max_ci_tests: DEFAULT_MAX_CI_TESTS,
            ci_replicates: 199,
            seed: None,
        }
    }
}

impl GraphFalsificationConfig {
    fn validate(&self) -> Result<(), ModelError> {
        let invalid = antecedent_core::reason_code!("invalid_argument");
        let open_unit = |v: f64| v.is_finite() && v > 0.0 && v < 1.0;
        if !open_unit(self.alpha) || !open_unit(self.significance) {
            return Err(refuse(
                invalid,
                "graph_falsification.invalid_level",
                "alpha and significance must lie in (0, 1)",
            ));
        }
        if let Some(share) = self.max_equivalent_share {
            if !(share.is_finite() && share > 0.0 && share <= 1.0) {
                return Err(refuse(
                    invalid,
                    "graph_falsification.invalid_level",
                    "max_equivalent_share must lie in (0, 1]",
                ));
            }
        }
        if self.n_permutations == 0
            || self.min_permutations == 0
            || self.min_permutations > self.n_permutations
        {
            return Err(refuse(
                invalid,
                "graph_falsification.invalid_permutations",
                "n_permutations and min_permutations must be positive, with min_permutations at \
                 most n_permutations",
            ));
        }
        if self.ci_replicates == 0 {
            return Err(refuse(
                invalid,
                "graph_falsification.invalid_permutations",
                "ci_replicates must be positive",
            ));
        }
        Ok(())
    }
}

/// A typed refusal: the registered reason code, the namespaced detail, and the message.
fn refuse(code: &'static str, detail: &'static str, message: &'static str) -> ModelError {
    ModelError::Unsupported {
        message: format!("{}{code}: {detail}: {message}", antecedent_core::reason_code::PREFIX),
    }
}

/// How a graph permutation falsification ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum GraphFalsificationOutcome {
    /// The graph is not better than random relabellings of itself (p-value above the level).
    Rejected,
    /// The graph has significantly fewer violations than random relabellings. This is not
    /// evidence that the graph is correct: Markov-equivalent graphs are indistinguishable.
    NotRejected,
    /// The comparison cannot separate the graph from its relabellings.
    Uninformative,
    /// A budget or cancellation stopped the procedure before it supported a verdict.
    Incomplete,
}

/// Why a falsification is uninformative.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum GraphUninformativeReason {
    /// The graph implies no local Markov implication (every pair of nodes is adjacent or
    /// ancestrally related), so no data can contradict it.
    NoTestableImplication,
    /// Too large a share of relabellings is Markov equivalent to the graph.
    EquivalenceClassShare,
}

/// One tested local Markov implication `node ⟂ other | conditioning` of the given graph.
#[derive(Clone, Debug, PartialEq)]
pub struct MarkovImplicationTest {
    /// Node whose local Markov property is tested.
    pub node: VariableId,
    /// A non-parent non-descendant of `node`.
    pub other: VariableId,
    /// The parents of `node`.
    pub conditioning: Vec<VariableId>,
    /// Test statistic (its meaning is the CI test's).
    pub statistic: f64,
    /// p-value of the independence null.
    pub p_value: f64,
    /// Whether `p_value < alpha`.
    pub violated: bool,
}

/// Result of [`falsify_graph`].
#[derive(Clone, Debug)]
pub struct GraphFalsificationReport {
    /// Verdict.
    pub outcome: GraphFalsificationOutcome,
    /// Set when `outcome` is `Uninformative`.
    pub uninformative_reason: Option<GraphUninformativeReason>,
    /// Local Markov implications of the given graph.
    pub n_implications: usize,
    /// Of those, the number rejected at `alpha`; `None` when they were not all tested.
    pub n_violations: Option<usize>,
    /// Every tested implication of the given graph with its p-value (empty when not tested).
    pub evidence: Vec<MarkovImplicationTest>,
    /// Violation count of each completed relabelling, in the order drawn.
    pub permutation_violations: Arc<[u32]>,
    /// Permutation p-value; `None` unless a verdict was reached.
    pub p_value: Option<f64>,
    /// Relabellings requested (the number of distinct non-identity relabellings when fewer).
    pub permutations_requested: usize,
    /// Relabellings whose violation count was computed.
    pub permutations_completed: usize,
    /// Whether every non-identity relabelling was drawn (`n!` did not exceed the request).
    pub exhaustive: bool,
    /// Relabellings (excluding the graph itself) Markov equivalent to the graph.
    pub equivalent_permutations: usize,
    /// Share of the drawn relabellings, counting the graph itself, that are Markov equivalent.
    pub equivalent_share: f64,
    /// The threshold `equivalent_share` is compared with.
    pub max_equivalent_share: f64,
    /// Name of the CI test.
    pub ci_test: Arc<str>,
    /// Level at which an implication counts as violated.
    pub alpha: f64,
    /// Level of the permutation comparison.
    pub significance: f64,
    /// Seed of the relabelling stream.
    pub permutation_seed: u64,
    /// Master seed of the execution context, which drives permutation-null CI tests.
    pub execution_seed: u64,
    /// Distinct CI tests evaluated.
    pub tests_run: usize,
    /// Lookups answered from the cache of earlier tests.
    pub cache_hits: usize,
    /// Distinct tests the stopped step needed but did not run.
    pub tests_skipped: usize,
    /// Ceiling on distinct CI tests.
    pub max_ci_tests: usize,
    /// What stopped the run, when it stopped early.
    pub stop: Option<SearchStop>,
    /// Limits in force and amounts consumed, when a budget or cancellation stopped the run.
    pub search_receipt: Option<SearchReceipt>,
    /// Statements of what the verdict does not establish.
    pub limitations: Vec<Arc<str>>,
}

/// One local Markov implication in node-index form.
struct Implication {
    node: usize,
    other: usize,
    conditioning: Vec<usize>,
}

/// A conditional-independence query over data columns with `x < y` and `z` sorted.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CiKey {
    x: usize,
    y: usize,
    z: Vec<usize>,
}

/// Skeleton and unshielded colliders: the Markov equivalence signature of a DAG.
#[derive(Eq, PartialEq)]
struct EquivalenceSignature {
    skeleton: Vec<(usize, usize)>,
    colliders: Vec<(usize, usize, usize)>,
}

impl EquivalenceSignature {
    fn of(dag: &Dag) -> Self {
        let n = dag.node_count();
        let mut skeleton = Vec::new();
        for child in 0..n {
            for parent in dag.parents(DenseNodeId::from_index(child)) {
                let parent = parent.as_usize();
                skeleton.push((parent.min(child), parent.max(child)));
            }
        }
        skeleton.sort_unstable();
        skeleton.dedup();
        let mut colliders = Vec::new();
        for child in 0..n {
            let mut parents: Vec<usize> = dag
                .parents(DenseNodeId::from_index(child))
                .iter()
                .map(|p| DenseNodeId::as_usize(*p))
                .collect();
            parents.sort_unstable();
            for (i, &a) in parents.iter().enumerate() {
                for &b in &parents[i + 1..] {
                    if skeleton.binary_search(&(a, b)).is_err() {
                        colliders.push((a, child, b));
                    }
                }
            }
        }
        colliders.sort_unstable();
        Self { skeleton, colliders }
    }

    /// The signature of the same graph with node `i` renamed `perm[i]`.
    fn relabelled(&self, perm: &[usize]) -> Self {
        let mut skeleton: Vec<(usize, usize)> = self
            .skeleton
            .iter()
            .map(|&(a, b)| (perm[a].min(perm[b]), perm[a].max(perm[b])))
            .collect();
        skeleton.sort_unstable();
        let mut colliders: Vec<(usize, usize, usize)> = self
            .colliders
            .iter()
            .map(|&(a, c, b)| (perm[a].min(perm[b]), perm[c], perm[a].max(perm[b])))
            .collect();
        colliders.sort_unstable();
        Self { skeleton, colliders }
    }
}

/// Local Markov implications of `dag` as pairwise statements, in node order.
fn local_markov_implications(dag: &Dag) -> Vec<Implication> {
    let n = dag.node_count();
    let mut descendants = BitSet::with_len(n);
    let mut ws = GraphWorkspace::default();
    let mut out = Vec::new();
    for node in 0..n {
        let id = DenseNodeId::from_index(node);
        dag.descendants_of(&[id], &mut descendants, &mut ws);
        let parents = dag.parents(id);
        let mut conditioning: Vec<usize> =
            parents.iter().map(|p| DenseNodeId::as_usize(*p)).collect();
        conditioning.sort_unstable();
        for other in 0..n {
            let other_id = DenseNodeId::from_index(other);
            if other == node || parents.contains(&other_id) || descendants.contains(other_id) {
                continue;
            }
            out.push(Implication { node, other, conditioning: conditioning.clone() });
        }
    }
    out
}

fn key_under(imp: &Implication, perm: &[usize]) -> CiKey {
    let a = perm[imp.node];
    let b = perm[imp.other];
    let mut z: Vec<usize> = imp.conditioning.iter().map(|&c| perm[c]).collect();
    z.sort_unstable();
    CiKey { x: a.min(b), y: a.max(b), z }
}

/// Uniform integer in `0..bound` (multiply-shift; the bias is below `2^-64 * bound`).
fn below(rng: &mut CausalRng, bound: usize) -> usize {
    let wide = u128::from(rng.next_u64()) * bound as u128;
    usize::try_from(wide >> 64).unwrap_or(0)
}

fn shuffle<T>(items: &mut [T], rng: &mut CausalRng) {
    for i in (1..items.len()).rev() {
        items.swap(i, below(rng, i + 1));
    }
}

/// `n!` when it does not exceed `cap`.
fn factorial_within(n: usize, cap: usize) -> Option<usize> {
    let mut acc = 1_usize;
    for k in 2..=n {
        acc = acc.checked_mul(k)?;
        if acc > cap {
            return None;
        }
    }
    Some(acc)
}

/// Lexicographic successor; `false` after the last permutation.
fn next_permutation(perm: &mut [usize]) -> bool {
    let Some(pivot) = (1..perm.len()).rev().find(|&i| perm[i - 1] < perm[i]).map(|i| i - 1) else {
        return false;
    };
    let swap = (pivot + 1..perm.len()).rev().find(|&j| perm[j] > perm[pivot]).unwrap_or(pivot);
    perm.swap(pivot, swap);
    perm[pivot + 1..].reverse();
    true
}

/// The relabellings to compare against and whether they are all of the non-identity ones.
fn draw_permutations(
    n: usize,
    requested: usize,
    rng: &mut CausalRng,
    ctx: &ExecutionContext,
) -> Result<(RelabellingPlan, bool), ModelError> {
    if ctx.cancellation.is_cancelled() {
        return Err(ModelError::Cancelled);
    }
    if let Some(factorial) = factorial_within(n, requested.min(MAX_ENUMERATED_RELABELLINGS)) {
        let mut perm: Vec<usize> = (0..n).collect();
        let mut all = Vec::with_capacity(factorial.saturating_sub(1));
        while next_permutation(&mut perm) {
            if all.len() % 1024 == 0 && ctx.cancellation.is_cancelled() {
                return Err(ModelError::Cancelled);
            }
            all.push(perm.clone());
        }
        // A prefix of the lexicographic order is not a random subset.
        for i in (1..all.len()).rev() {
            if i % 1024 == 0 && ctx.cancellation.is_cancelled() {
                return Err(ModelError::Cancelled);
            }
            all.swap(i, below(rng, i + 1));
        }
        return Ok((RelabellingPlan::Listed(all), true));
    }
    Ok((RelabellingPlan::Sampled { n, count: requested, rng: rng.clone() }, false))
}

/// The relabellings of a run: an explicit list, or a counted deterministic stream.
enum RelabellingPlan {
    /// Every non-identity relabelling, in shuffled order.
    Listed(Vec<Vec<usize>>),
    /// `count` independent uniform relabellings regenerated from `rng` on each pass.
    Sampled { n: usize, count: usize, rng: CausalRng },
}

impl RelabellingPlan {
    fn len(&self) -> usize {
        match self {
            Self::Listed(all) => all.len(),
            Self::Sampled { count, .. } => *count,
        }
    }

    /// One pass over the relabellings; every pass yields the same sequence.
    fn iter(&self) -> Box<dyn Iterator<Item = Vec<usize>> + '_> {
        match self {
            Self::Listed(all) => Box::new(all.iter().cloned()),
            Self::Sampled { n, count, rng } => {
                let mut rng = rng.clone();
                let n = *n;
                Box::new((0..*count).map(move |_| {
                    let mut perm: Vec<usize> = (0..n).collect();
                    shuffle(&mut perm, &mut rng);
                    perm
                }))
            }
        }
    }
}

/// Cached CI evaluation under a shared search budget.
struct CiSession<'a> {
    test: &'a (dyn ConditionalIndependence + Send + Sync),
    prepared: PreparedCiTest,
    columns: &'a [&'a [f64]],
    significance: SignificanceMethod,
    workspace: CiWorkspace,
    cache: HashMap<CiKey, (f64, f64)>,
    budget: SearchBudget<'a>,
    ctx: &'a ExecutionContext,
    tests_run: usize,
    cache_hits: usize,
    tests_skipped: usize,
}

impl CiSession<'_> {
    /// Make every key's result available, running the missing ones as one batch.
    ///
    /// Returns the bound that stopped the batch; nothing from a stopped batch is cached.
    fn ensure(&mut self, keys: &[CiKey]) -> Result<Option<SearchStop>, ModelError> {
        let mut seen = HashSet::new();
        let mut missing: Vec<&CiKey> = Vec::new();
        for key in keys {
            if self.cache.contains_key(key) || !seen.insert(key) {
                self.cache_hits += 1;
            } else {
                missing.push(key);
            }
        }
        if missing.is_empty() {
            return Ok(None);
        }
        let remaining = self.budget.limits().operations.saturating_sub(self.budget.operations());
        if missing.len() > remaining {
            self.tests_skipped = missing.len();
            return Ok(Some(SearchStop::Operations));
        }
        for _ in &missing {
            if let Err(stop) = self.budget.charge(1, 0) {
                self.tests_skipped = missing.len();
                return Ok(Some(stop));
            }
        }
        let mut queries = Vec::with_capacity(missing.len());
        let mut z_flat = Vec::new();
        for key in &missing {
            queries.push(CiQuery { x: key.x, y: key.y, z_start: z_flat.len(), z_len: key.z.len() });
            z_flat.extend_from_slice(&key.z);
        }
        let request = CiBatchRequest {
            columns: self.columns,
            queries: &queries,
            z_flat: &z_flat,
            significance: self.significance,
            confidence: ConfidenceMethod::None,
        };
        let out = self.test.test_batch(&self.prepared, &request, &mut self.workspace, self.ctx)?;
        if out.results.len() != queries.len() {
            return Err(refuse(
                antecedent_core::reason_code!("diagnostic_not_available"),
                "graph_falsification.ci_result_count",
                "the CI test returned a different number of results than queries",
            ));
        }
        for (key, result) in missing.iter().zip(&out.results) {
            if result.p_value.is_nan() {
                return Err(refuse(
                    antecedent_core::reason_code!("diagnostic_not_available"),
                    "graph_falsification.ci_p_value_missing",
                    "a local Markov implication produced a NaN p-value",
                ));
            }
            self.cache.insert((*key).clone(), (result.statistic, result.p_value));
        }
        self.tests_run += missing.len();
        Ok(None)
    }

    fn violations(&self, keys: &[CiKey], alpha: f64) -> u32 {
        let count = keys.iter().filter(|k| self.cache.get(*k).is_some_and(|r| r.1 < alpha)).count();
        u32::try_from(count).unwrap_or(u32::MAX)
    }
}

fn limitations(config: &GraphFalsificationConfig, replicates: Option<u32>) -> Vec<Arc<str>> {
    let mut out: Vec<Arc<str>> = vec![
        Arc::from(
            "Each local Markov implication is one conditional-independence test at the declared \
             alpha; the violation count is not corrected for multiple testing and the tests are \
             dependent, so it is a comparison statistic, not a count of established violations.",
        ),
        Arc::from(
            "Failing to reject an implication is not evidence of independence; power depends on \
             the CI test (partial correlation detects linear dependence only, so use a nonlinear \
             test for nonlinear data).",
        ),
        Arc::from(
            "The permutation null relabels the graph's nodes and recomputes violations on the \
             same data. It rejects graphs that are not better than random relabellings of \
             themselves; it cannot distinguish Markov-equivalent graphs, and a graph that is not \
             rejected is not thereby shown correct.",
        ),
        Arc::from(
            "The implications assume causal sufficiency and no selection: a latent common cause \
             or a selected sample can violate the true graph's implications.",
        ),
        Arc::from(
            "Every row of every graph column must be observed and finite; rows are not dropped.",
        ),
        Arc::from(
            "This is not the shuffled-outcome likelihood diagnostic of a fitted model: no \
             mechanism is fitted and no likelihood is compared.",
        ),
    ];
    if let Some(replicates) = replicates {
        out.push(Arc::from(format!(
            "The {} test reports permutation p-values from {replicates} null replicates, so its \
             smallest attainable p-value is {:.4} and its p-values depend on the order of \
             evaluation under the execution seed.",
            config.ci_test,
            1.0 / (f64::from(replicates) + 1.0)
        )));
    }
    out
}

/// Test a causal DAG against data by comparing its violated local Markov implications with those
/// of random relabellings of its nodes. No fitted mechanisms are needed.
///
/// Nodes must be static variable nodes whose columns in `data` are numeric, observed and finite.
/// The relabelling stream is seeded by `config.seed` (or the context's master seed); CI tests with
/// permutation nulls draw from the context. Distinct CI tests are charged against
/// `config.max_ci_tests` as a [`SearchBudget`] operation and cancellation is observed at every
/// charge; results are cached across relabellings.
///
/// # Errors
///
/// Invalid configuration, a graph that is not a static DAG over at least two data columns,
/// unobserved or non-finite data, an unknown or non-frequentist CI test, or a CI-test failure.
/// Budget exhaustion and cancellation are reported as [`GraphFalsificationOutcome::Incomplete`].
#[allow(
    clippy::too_many_lines,
    reason = "one procedure: validate, draw relabellings, test the graph, then each relabelling"
)]
pub fn falsify_graph(
    dag: &Dag,
    data: &TabularData,
    config: &GraphFalsificationConfig,
    ctx: &ExecutionContext,
) -> Result<GraphFalsificationReport, ModelError> {
    config.validate()?;
    if ctx.cancellation.is_cancelled() {
        return Err(ModelError::Cancelled);
    }
    dag.validate()?;
    let n = dag.node_count();
    if n < 2 {
        return Err(refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "graph_falsification.too_few_nodes",
            "graph falsification needs a graph with at least two nodes",
        ));
    }
    if data.row_count() == 0 {
        return Err(refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "graph_falsification.empty_data",
            "graph falsification needs at least one row",
        ));
    }
    let mut variables = Vec::with_capacity(n);
    for node in dag.nodes() {
        let NodeRef::Static(variable) = node else {
            return Err(refuse(
                antecedent_core::reason_code!("route_not_supported"),
                "graph_falsification.node_not_static",
                "only static variable nodes are supported",
            ));
        };
        if variables.contains(variable) {
            return Err(refuse(
                antecedent_core::reason_code!("invalid_argument"),
                "graph_falsification.duplicate_variable",
                "distinct nodes must carry distinct variables",
            ));
        }
        variables.push(*variable);
    }
    let storage: Vec<Cow<'_, [f64]>> = variables
        .iter()
        .map(|&v| data.float64_cow(v))
        .collect::<Result<_, _>>()
        .map_err(ModelError::from)?;
    if storage.iter().any(|col| col.iter().any(|v| !v.is_finite())) {
        return Err(refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "graph_falsification.data_not_observed",
            "graph columns must be fully observed and finite; rows are not dropped",
        ));
    }
    let columns: Vec<&[f64]> = storage.iter().map(AsRef::as_ref).collect();

    let test = ci_from_name(&config.ci_test)?;
    if !test.p_value_is_frequentist() {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "graph_falsification.ci_test_not_frequentist",
            "the test's p-value is a posterior probability, not a frequentist p-value",
        ));
    }
    let significance = if test.min_attainable_p(SignificanceMethod::Analytic) == 0.0 {
        SignificanceMethod::Analytic
    } else {
        SignificanceMethod::BlockShuffle { replicates: config.ci_replicates, block_size: 1 }
    };
    ensure_alpha_resolvable(test.min_attainable_p(significance), config.alpha)?;
    let null_replicates = match significance {
        SignificanceMethod::Analytic => None,
        SignificanceMethod::BlockShuffle { replicates, .. } => Some(replicates),
    };

    let permutation_seed = config.seed.unwrap_or_else(|| ctx.rng.master_seed());
    let mut rng = RngFactory::from_seed(permutation_seed)
        .stream_for(StreamDomain::Resample, PERMUTATION_STREAM);
    let implications = local_markov_implications(dag);
    let signature = EquivalenceSignature::of(dag);
    // Reserve both the retained counts (including the final Arc conversion) and
    // any enumerated vector payload/headers before either allocation.
    let enumerated = factorial_within(n, config.n_permutations.min(MAX_ENUMERATED_RELABELLINGS));
    let count = enumerated.map_or(config.n_permutations, |v| v.saturating_sub(1));
    let bytes = count
        .checked_mul(std::mem::size_of::<u32>())
        .and_then(|v| v.checked_mul(2))
        .and_then(|v| {
            if enumerated.is_some() {
                count
                    .checked_mul(
                        n.checked_mul(std::mem::size_of::<usize>())?
                            .checked_add(std::mem::size_of::<Vec<usize>>())?,
                    )?
                    .checked_add(v)
            } else {
                Some(v)
            }
        })
        .and_then(|v| v.checked_add(n.checked_mul(std::mem::size_of::<usize>())?))
        .and_then(|v| u64::try_from(v).ok())
        .ok_or(ModelError::ResourceLimit {
            message: "graph falsification permutations",
            required_bytes: u64::MAX,
            limit_bytes: ctx.admission().available().bytes(),
        })?;
    let _lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => ModelError::Cancelled,
        _ => ModelError::ResourceLimit {
            message: "graph falsification permutations",
            required_bytes: bytes,
            limit_bytes: ctx.admission().available().bytes(),
        },
    })?;
    let (permutations, exhaustive) = draw_permutations(n, config.n_permutations, &mut rng, ctx)?;
    let mut equivalent_permutations = 0usize;
    for perm in permutations.iter() {
        if ctx.cancellation.is_cancelled() {
            return Err(ModelError::Cancelled);
        }
        if signature.relabelled(&perm) == signature {
            equivalent_permutations += 1;
        }
    }
    let equivalent_share = (1 + equivalent_permutations) as f64 / (1 + permutations.len()) as f64;
    let max_equivalent_share = config.max_equivalent_share.unwrap_or(config.significance);

    let mut report = GraphFalsificationReport {
        outcome: GraphFalsificationOutcome::Incomplete,
        uninformative_reason: None,
        n_implications: implications.len(),
        n_violations: None,
        evidence: Vec::new(),
        permutation_violations: Arc::from([]),
        p_value: None,
        permutations_requested: permutations.len(),
        permutations_completed: 0,
        exhaustive,
        equivalent_permutations,
        equivalent_share,
        max_equivalent_share,
        ci_test: Arc::clone(&config.ci_test),
        alpha: config.alpha,
        significance: config.significance,
        permutation_seed,
        execution_seed: ctx.rng.master_seed(),
        tests_run: 0,
        cache_hits: 0,
        tests_skipped: 0,
        max_ci_tests: config.max_ci_tests,
        stop: None,
        search_receipt: None,
        limitations: limitations(config, null_replicates),
    };

    if implications.is_empty() {
        report.outcome = GraphFalsificationOutcome::Uninformative;
        report.uninformative_reason = Some(GraphUninformativeReason::NoTestableImplication);
        return Ok(report);
    }
    if equivalent_share > max_equivalent_share {
        report.outcome = GraphFalsificationOutcome::Uninformative;
        report.uninformative_reason = Some(GraphUninformativeReason::EquivalenceClassShare);
        return Ok(report);
    }

    let limits = SearchLimits { operations: config.max_ci_tests, depth: 1 };
    let budget = match SearchBudget::new(limits, ctx) {
        Ok(budget) => budget,
        Err(receipt) => {
            report.stop = Some(receipt.stop);
            report.search_receipt = Some(receipt);
            return Ok(report);
        }
    };
    let prepared = test.prepare(
        &columns,
        &CiPreparationPlan { significance, confidence: ConfidenceMethod::None },
        ctx,
    )?;
    let mut session = CiSession {
        test: test.as_ref(),
        prepared,
        columns: &columns,
        significance,
        workspace: CiWorkspace::default(),
        cache: HashMap::new(),
        budget,
        ctx,
        tests_run: 0,
        cache_hits: 0,
        tests_skipped: 0,
    };

    let identity: Vec<usize> = (0..n).collect();
    let observed_keys: Vec<CiKey> = implications.iter().map(|i| key_under(i, &identity)).collect();
    if let Some(stop) = session.ensure(&observed_keys)? {
        return Ok(stopped(report, &session, stop));
    }
    let evidence: Vec<MarkovImplicationTest> = implications
        .iter()
        .zip(&observed_keys)
        .map(|(imp, key)| {
            let (statistic, p_value) = session.cache[key];
            MarkovImplicationTest {
                node: variables[imp.node],
                other: variables[imp.other],
                conditioning: imp.conditioning.iter().map(|&c| variables[c]).collect(),
                statistic,
                p_value,
                violated: p_value < config.alpha,
            }
        })
        .collect();
    let observed_violations = session.violations(&observed_keys, config.alpha);

    let mut counts = Vec::with_capacity(permutations.len());
    let mut stop = None;
    for perm in permutations.iter() {
        if ctx.cancellation.is_cancelled() {
            stop = Some(SearchStop::Cancelled);
            break;
        }
        let keys: Vec<CiKey> = implications.iter().map(|i| key_under(i, &perm)).collect();
        if let Some(reason) = session.ensure(&keys)? {
            stop = Some(reason);
            break;
        }
        counts.push(session.violations(&keys, config.alpha));
    }

    report.n_violations = Some(observed_violations as usize);
    report.evidence = evidence;
    report.permutations_completed = counts.len();
    report.tests_run = session.tests_run;
    report.cache_hits = session.cache_hits;
    report.tests_skipped = session.tests_skipped;
    if let Some(reason) = stop {
        report.stop = Some(reason);
        report.search_receipt = Some(session.budget.receipt(
            reason,
            vec![
                "graph implications".to_owned(),
                format!("permutations: {} of {}", counts.len(), permutations.len()),
            ],
            vec![format!("permutations: {}", permutations.len() - counts.len())],
        ));
    }
    if stop == Some(SearchStop::Cancelled) || counts.len() < config.min_permutations {
        report.permutation_violations = Arc::from(counts);
        return Ok(report);
    }
    let at_most = counts.iter().filter(|&&c| c <= observed_violations).count();
    let p_value = (1 + at_most) as f64 / (1 + counts.len()) as f64;
    report.p_value = Some(p_value);
    report.outcome = if p_value > config.significance {
        GraphFalsificationOutcome::Rejected
    } else {
        GraphFalsificationOutcome::NotRejected
    };
    report.permutation_violations = Arc::from(counts);
    Ok(report)
}

/// Incomplete report for a stop before the permutations began.
fn stopped(
    mut report: GraphFalsificationReport,
    session: &CiSession<'_>,
    stop: SearchStop,
) -> GraphFalsificationReport {
    report.tests_run = session.tests_run;
    report.cache_hits = session.cache_hits;
    report.tests_skipped = session.tests_skipped;
    report.stop = Some(stop);
    report.search_receipt = Some(session.budget.receipt(
        stop,
        Vec::new(),
        vec!["graph implications".to_owned(), "permutations".to_owned()],
    ));
    report
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    reason = "tests build small fixtures and compare exact counts"
)]
mod tests {
    use super::*;

    fn normal(rng: &mut CausalRng) -> f64 {
        let u1 = 1.0 - rng.next_f64();
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn table(columns: &[Vec<f64>]) -> TabularData {
        let names: Vec<String> = (0..columns.len()).map(|i| format!("x{i}")).collect();
        TabularData::from_f64_columns(
            names.iter().map(String::as_str).zip(columns.iter().map(Vec::as_slice)),
        )
        .unwrap()
    }

    fn graph(n: u32, edges: &[(u32, u32)]) -> Dag {
        let mut g = Dag::with_variables(n);
        for &(a, b) in edges {
            g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        g
    }

    /// Four strongly dependent disjoint pairs `(x0,x1) (x2,x3) (x4,x5) (x6,x7)`.
    fn linear_pairs(rows: usize, seed: u64) -> TabularData {
        let mut rng = CausalRng::from_seed(seed);
        let mut cols: Vec<Vec<f64>> = (0..8).map(|_| Vec::with_capacity(rows)).collect();
        for _ in 0..rows {
            for pair in 0..4 {
                let root = normal(&mut rng);
                cols[2 * pair].push(root);
                cols[2 * pair + 1].push(0.9 * root + 0.45 * normal(&mut rng));
            }
        }
        table(&cols)
    }

    /// Three nonlinear pairs `x(2k+1) = x(2k)^2 + noise`; Pearson correlation is zero by symmetry.
    fn quadratic_pairs(rows: usize, seed: u64) -> TabularData {
        let mut rng = CausalRng::from_seed(seed);
        let mut cols: Vec<Vec<f64>> = (0..6).map(|_| Vec::with_capacity(rows)).collect();
        for _ in 0..rows {
            for pair in 0..3 {
                let root = normal(&mut rng);
                cols[2 * pair].push(root);
                cols[2 * pair + 1].push(root * root + 0.25 * normal(&mut rng));
            }
        }
        table(&cols)
    }

    const TRUE_PAIRS: [(u32, u32); 4] = [(0, 1), (2, 3), (4, 5), (6, 7)];
    const CROSSED_PAIRS: [(u32, u32); 4] = [(0, 2), (1, 3), (4, 6), (5, 7)];

    fn config(permutations: usize) -> GraphFalsificationConfig {
        GraphFalsificationConfig {
            alpha: 0.01,
            n_permutations: permutations,
            min_permutations: 20,
            seed: Some(11),
            ..GraphFalsificationConfig::default()
        }
    }

    #[test]
    fn implications_are_non_parent_non_descendants_given_parents() {
        // 0 -> 1 -> 2, isolated 3.
        let imps = local_markov_implications(&graph(4, &[(0, 1), (1, 2)]));
        let triples: Vec<(usize, usize, Vec<usize>)> =
            imps.into_iter().map(|i| (i.node, i.other, i.conditioning)).collect();
        assert!(triples.contains(&(2, 0, vec![1])));
        assert!(triples.contains(&(0, 3, vec![])));
        assert!(triples.contains(&(3, 2, vec![])));
        assert!(!triples.iter().any(|t| t.0 == 0 && t.1 == 2), "a descendant is not tested");
        assert!(!triples.iter().any(|t| t.0 == 1 && t.1 == 0), "a parent is not tested");
    }

    #[test]
    fn markov_equivalence_signature_ignores_non_collider_orientation() {
        let chain = graph(3, &[(0, 1), (1, 2)]);
        let reversed = graph(3, &[(2, 1), (1, 0)]);
        let collider = graph(3, &[(0, 1), (2, 1)]);
        let identity = [0, 1, 2];
        assert!(
            EquivalenceSignature::of(&chain).relabelled(&identity)
                == EquivalenceSignature::of(&reversed)
        );
        assert!(
            EquivalenceSignature::of(&chain).relabelled(&identity)
                != EquivalenceSignature::of(&collider)
        );
        // Moving the middle node changes the skeleton; swapping the ends does not.
        assert!(
            EquivalenceSignature::of(&chain).relabelled(&[2, 1, 0])
                == EquivalenceSignature::of(&chain)
        );
        assert!(
            EquivalenceSignature::of(&chain).relabelled(&[1, 0, 2])
                != EquivalenceSignature::of(&chain)
        );
    }

    #[test]
    fn small_graphs_enumerate_every_non_identity_relabelling() {
        let mut rng = CausalRng::from_seed(3);
        let (perms, exhaustive) =
            draw_permutations(4, 100, &mut rng, &ExecutionContext::for_tests(1)).unwrap();
        assert!(exhaustive);
        assert_eq!(perms.len(), 23);
        let distinct: HashSet<_> = perms.iter().collect();
        assert_eq!(distinct.len(), 23);
        assert!(!distinct.contains(&vec![0, 1, 2, 3]));
        let (sampled, exhaustive) =
            draw_permutations(8, 50, &mut rng, &ExecutionContext::for_tests(1)).unwrap();
        assert!(!exhaustive);
        assert_eq!(sampled.len(), 50);
        assert_eq!(sampled.iter().collect::<Vec<_>>(), sampled.iter().collect::<Vec<_>>());
        assert_eq!(sampled.iter().count(), 50);
    }

    #[test]
    fn a_correct_graph_is_not_rejected_without_fitted_mechanisms() {
        let data = linear_pairs(2000, 5);
        let report = falsify_graph(
            &graph(8, &TRUE_PAIRS),
            &data,
            &config(300),
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert_eq!(report.outcome, GraphFalsificationOutcome::NotRejected, "{report:?}");
        assert!(report.p_value.unwrap() < 0.05);
        assert_eq!(report.n_implications, 48);
        assert!(report.n_violations.unwrap() <= 2);
        assert_eq!(report.evidence.len(), 48);
        assert_eq!(report.permutations_completed, 300);
        assert_eq!(report.permutation_violations.len(), 300);
        assert!(report.equivalent_share <= report.max_equivalent_share);
        // Seeds, budgets and limitations travel with the verdict.
        assert_eq!(report.permutation_seed, 11);
        assert_eq!(report.execution_seed, 1);
        assert_eq!(report.max_ci_tests, DEFAULT_MAX_CI_TESTS);
        assert!(report.tests_run > 0 && report.cache_hits > 0);
        assert!(report.stop.is_none() && report.search_receipt.is_none());
        assert!(report.limitations.iter().any(|l| l.contains("shuffled-outcome")));
    }

    #[test]
    fn a_graph_claiming_false_independencies_is_rejected() {
        let data = linear_pairs(2000, 5);
        let report = falsify_graph(
            &graph(8, &CROSSED_PAIRS),
            &data,
            &config(300),
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert_eq!(report.outcome, GraphFalsificationOutcome::Rejected, "{report:?}");
        assert!(report.p_value.unwrap() > 0.5);
        assert!(report.n_violations.unwrap() >= 6);
        assert!(report.evidence.iter().any(|e| e.violated && e.p_value < 0.01));
    }

    #[test]
    fn the_same_seed_reproduces_the_report() {
        let data = linear_pairs(500, 5);
        let run = || {
            falsify_graph(
                &graph(8, &TRUE_PAIRS),
                &data,
                &config(60),
                &ExecutionContext::for_tests(1),
            )
            .unwrap()
        };
        let (a, b) = (run(), run());
        assert_eq!(a.permutation_violations, b.permutation_violations);
        assert_eq!(a.p_value, b.p_value);
    }

    #[test]
    fn complete_and_tiny_graphs_are_uninformative() {
        let mut rng = CausalRng::from_seed(2);
        let cols: Vec<Vec<f64>> =
            (0..4).map(|_| (0..300).map(|_| normal(&mut rng)).collect()).collect();
        let data = table(&cols);
        let complete = graph(4, &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
        let report =
            falsify_graph(&complete, &data, &config(100), &ExecutionContext::for_tests(1)).unwrap();
        assert_eq!(report.outcome, GraphFalsificationOutcome::Uninformative);
        assert_eq!(
            report.uninformative_reason,
            Some(GraphUninformativeReason::NoTestableImplication)
        );
        assert_eq!(report.n_implications, 0);
        assert_eq!(report.tests_run, 0);

        let chain = graph(3, &[(0, 1), (1, 2)]);
        let report = falsify_graph(
            &chain,
            &table(&cols[..3]),
            &config(100),
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert_eq!(report.outcome, GraphFalsificationOutcome::Uninformative);
        assert_eq!(
            report.uninformative_reason,
            Some(GraphUninformativeReason::EquivalenceClassShare)
        );
        assert!(report.exhaustive);
        assert_eq!(report.equivalent_permutations, 1);
        assert!((report.equivalent_share - 2.0 / 6.0).abs() < 1e-12);
        assert!(report.p_value.is_none());
    }

    #[test]
    fn a_budget_that_cannot_cover_the_tests_is_incomplete() {
        let data = linear_pairs(500, 5);
        let mut cfg = config(300);
        cfg.max_ci_tests = 5;
        let report =
            falsify_graph(&graph(8, &TRUE_PAIRS), &data, &cfg, &ExecutionContext::for_tests(1))
                .unwrap();
        assert_eq!(report.outcome, GraphFalsificationOutcome::Incomplete);
        assert_eq!(report.stop, Some(SearchStop::Operations));
        assert_eq!(report.tests_run, 0);
        assert!(report.tests_skipped > 5);
        assert!(report.p_value.is_none() && report.n_violations.is_none());
        assert_eq!(report.search_receipt.as_ref().unwrap().operations_limit, 5);
    }

    #[test]
    fn a_budget_exhausted_before_the_minimum_permutations_is_incomplete() {
        let data = linear_pairs(500, 5);
        let mut cfg = config(300);
        cfg.min_permutations = 100;
        cfg.max_ci_tests = 100;
        let report =
            falsify_graph(&graph(8, &TRUE_PAIRS), &data, &cfg, &ExecutionContext::for_tests(1))
                .unwrap();
        assert_eq!(report.outcome, GraphFalsificationOutcome::Incomplete);
        assert!(report.permutations_completed < 100);
        assert!(report.tests_run <= 100);
        assert!(report.n_violations.is_some(), "the graph's own tests finished");
        assert!(report.p_value.is_none());
        assert!(report.search_receipt.is_some());
    }

    #[test]
    fn permutation_storage_is_admitted_before_enumeration_or_count_allocation() {
        let data = linear_pairs(500, 5);
        let limited = ExecutionContext::for_tests(1).with_memory_limit(Some(1024));
        assert!(matches!(
            falsify_graph(&graph(8, &TRUE_PAIRS), &data, &config(1_000_000), &limited),
            Err(ModelError::ResourceLimit { .. })
        ));
        let unlimited_count =
            GraphFalsificationConfig { n_permutations: usize::MAX, ..config(300) };
        assert!(matches!(
            falsify_graph(&graph(8, &TRUE_PAIRS), &data, &unlimited_count, &limited),
            Err(ModelError::ResourceLimit { .. })
        ));
    }

    #[test]
    fn cancellation_before_equivalence_is_refused() {
        let data = linear_pairs(500, 5);
        let ctx = ExecutionContext::for_tests(1);
        ctx.cancellation.cancel();
        assert!(matches!(
            falsify_graph(&graph(8, &TRUE_PAIRS), &data, &config(300), &ctx),
            Err(ModelError::Cancelled)
        ));
    }

    #[test]
    fn a_nonlinear_test_finds_the_violation_partial_correlation_misses() {
        let data = quadratic_pairs(200, 9);
        // Edges join variables from different pairs, so each quadratic pair is claimed independent.
        let wrong = graph(6, &[(0, 2), (1, 4), (3, 5)]);
        let correct = graph(6, &[(0, 1), (2, 3), (4, 5)]);
        let base = GraphFalsificationConfig {
            alpha: 0.02,
            n_permutations: 20,
            min_permutations: 10,
            max_equivalent_share: Some(0.5),
            ci_replicates: 99,
            seed: Some(4),
            ..GraphFalsificationConfig::default()
        };
        let ctx = ExecutionContext::for_tests(1);

        let linear = falsify_graph(&wrong, &data, &base, &ctx).unwrap();
        assert!(
            linear.n_violations.unwrap() <= 2,
            "partial correlation sees no linear dependence: {linear:?}"
        );

        let nonlinear_cfg = GraphFalsificationConfig { ci_test: Arc::from("gpdc"), ..base };
        let nonlinear = falsify_graph(&wrong, &data, &nonlinear_cfg, &ctx).unwrap();
        assert!(nonlinear.n_violations.unwrap() >= 4, "{nonlinear:?}");
        assert_eq!(nonlinear.outcome, GraphFalsificationOutcome::Rejected, "{nonlinear:?}");

        let right = falsify_graph(&correct, &data, &nonlinear_cfg, &ctx).unwrap();
        assert!(right.n_violations.unwrap() <= 2, "{right:?}");
    }

    #[test]
    fn falsification_runs_from_graph_and_data_alone() {
        // No compiled model, mechanism store or fit exists anywhere in this test.
        let data = linear_pairs(500, 5);
        let report = falsify_graph(
            &graph(8, &TRUE_PAIRS),
            &data,
            &config(40),
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert_ne!(report.outcome, GraphFalsificationOutcome::Incomplete);
        assert!(report.limitations.iter().any(|l| l.contains("no mechanism is fitted")));
    }

    #[test]
    fn the_report_carries_seeds_budgets_and_limitations() {
        let data = linear_pairs(500, 5);
        let mut cfg = config(40);
        cfg.max_ci_tests = 300;
        cfg.seed = None;
        let ctx = ExecutionContext::for_tests(77);
        let report = falsify_graph(&graph(8, &TRUE_PAIRS), &data, &cfg, &ctx).unwrap();
        // The relabelling stream defaults to the context's master seed.
        assert_eq!(report.permutation_seed, 77);
        assert_eq!(report.execution_seed, 77);
        assert_eq!(report.permutations_requested, 40);
        assert_eq!(report.max_ci_tests, 300);
        assert!(report.tests_run <= 300);
        assert!(report.limitations.len() >= 6);
        assert!(report.limitations.iter().any(|l| l.contains("multiple testing")));
        assert!(report.limitations.iter().any(|l| l.contains("partial correlation")));
        // A permutation-null test states its null replicates and attainable p-value floor.
        let nonlinear = GraphFalsificationConfig {
            ci_test: Arc::from("gpdc"),
            ci_replicates: 99,
            alpha: 0.05,
            n_permutations: 10,
            min_permutations: 5,
            max_ci_tests: 1,
            ..cfg
        };
        let small = linear_pairs(60, 5);
        let report = falsify_graph(&graph(8, &TRUE_PAIRS), &small, &nonlinear, &ctx).unwrap();
        assert!(report.limitations.iter().any(|l| l.contains("99 null replicates")));
    }

    #[test]
    fn invalid_inputs_are_refused() {
        let data = linear_pairs(100, 5);
        let ctx = ExecutionContext::for_tests(1);
        let g = graph(8, &TRUE_PAIRS);
        let bad_alpha = GraphFalsificationConfig { alpha: 0.0, ..config(10) };
        assert!(falsify_graph(&g, &data, &bad_alpha, &ctx).is_err());
        let bad_min = GraphFalsificationConfig { min_permutations: 11, ..config(10) };
        assert!(falsify_graph(&g, &data, &bad_min, &ctx).is_err());
        let unknown = GraphFalsificationConfig { ci_test: Arc::from("nope"), ..config(10) };
        assert!(falsify_graph(&g, &data, &unknown, &ctx).is_err());
        let bayes = GraphFalsificationConfig { ci_test: Arc::from("bayes_factor"), ..config(10) };
        assert!(falsify_graph(&g, &data, &bayes, &ctx).is_err());
        // A graph column the data does not carry.
        assert!(falsify_graph(&graph(9, &[]), &data, &config(10), &ctx).is_err());
    }
}
