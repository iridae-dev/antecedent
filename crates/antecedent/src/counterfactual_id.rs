//! Counterfactual identification on a bounded ADMG: the effect of treatment
//! on the treated from the observational joint (2.2B, X8).
//!
//! One cell: `P(Y_x = y | X = x')` with `x != x'`, its distribution over every
//! level of `Y` and the contrast `E[Y_x | X = x'] - E[Y | X = x']`, on an
//! explicit ADMG (or DAG) of at most six finite-discrete variables with at most
//! four levels each, from a supplied exact joint law or an empirical count
//! table. A point claim: no interval, no posterior.
//!
//! The stages are separate:
//!
//! 1. [`prepare_counterfactual_id`] decides the query once under one shared
//!    search budget ([`antecedent_identify::counterfactual_id`]) and returns
//!    the derivation, or a reason-coded refusal: `route_not_supported` with a
//!    checkable obstruction when ID* (and, for a binary treatment, the
//!    consistency complement) does not identify the query, which is not a
//!    proof of non-identifiability; `transport_budget_cancel` with a receipt;
//!    `route_not_supported` outside the contract (`path_specific_deferred` for
//!    a world that routes edges, `bounds_exceeded` over a bound);
//!    `cell_not_licensed` for a structure outside the graph contract,
//!    `estimator_inference_mismatch` for an interval request.
//! 2. [`PreparedCounterfactualId::evaluate`] evaluates the derivation on a law
//!    without re-identifying; call it again with another law of the same
//!    variables and levels to refresh.
//! 3. [`CounterfactualIdEffect::export_artifact`] writes the
//!    `counterfactual_id_admg_v1` artifact; [`consume_counterfactual_id_artifact`]
//!    re-derives it under the stored limits, requires the identical derivation
//!    and point, and recomputes the point with a separate direct-sum evaluator
//!    that shares no evaluation code with the compiled evaluator.
//!
//! What replay does not protect against: a producer that seals a wrong graph,
//! query or law on purpose (the digests bind what was sealed, not what is
//! true); and a bug shared by the identification engine and the consumer,
//! because the consumer re-derives with the same engine and the direct-sum
//! evaluator evaluates that same re-derived functional, so such a bug replays
//! identically. The direct sum checks the functional's arithmetic, not its
//! derivation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::{
    CounterfactualEventQuery, ExecutionContext, RegimeId, SearchLimits, Value, VariableId,
};
use antecedent_expr::{DiscreteAxis, ExactDiscreteLaw, ExprId, ExprNode, LawOrigin, LawTolerance};
use antecedent_graph::Admg;
use antecedent_identify::counterfactual_id::{
    COUNTERFACTUAL_ID_CONTRACT, COUNTERFACTUAL_ID_DEFAULT_LIMITS, COUNTERFACTUAL_ID_MAX_DEPTH,
    COUNTERFACTUAL_ID_MAX_OPERATIONS, COUNTERFACTUAL_ID_MEMORY_BYTES, CfSymbol, CfTerm, CfValue,
    CounterfactualFunctional, CounterfactualIdDerivation, CounterfactualIdProblem,
    CounterfactualIdRefusal, CounterfactualIdShape, decide_counterfactual_id,
    evaluate_counterfactual_functional,
};
use antecedent_io::counterfactual_id_artifact::{
    COUNTERFACTUAL_ID_ARTIFACT_FEATURE, COUNTERFACTUAL_ID_ARTIFACT_VERSION,
    COUNTERFACTUAL_ID_ESTIMAND, CounterfactualIdArtifactError, CounterfactualIdArtifactWire,
    CounterfactualIdAxisWire, CounterfactualIdLawWire, CounterfactualIdPointWire,
    CounterfactualIdQueryWire, counterfactual_id_data_digest,
};

use crate::GraphClass;
use crate::error::{CausalError, REASON_PREFIX};
use crate::support::{IntoGraphInput, StructureSource};

/// Relative tolerance of the direct-sum cross-check against the replayed point.
const DIRECT_SUM_TOLERANCE: f64 = 1e-10;

/// What the caller asks of the cell beyond the query.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CounterfactualIdOptions {
    /// Search limits of the identification (at most 100000 operations, depth 64).
    pub limits: SearchLimits,
    /// Memory cap of the search (further limited by the context's hard limit).
    pub memory_limit_bytes: u64,
    /// The caller asked for a sampling interval or Bayesian inference. Refused:
    /// the claim is point-only.
    pub interval_requested: bool,
}

impl Default for CounterfactualIdOptions {
    fn default() -> Self {
        Self {
            limits: COUNTERFACTUAL_ID_DEFAULT_LIMITS,
            memory_limit_bytes: COUNTERFACTUAL_ID_MEMORY_BYTES,
            interval_requested: false,
        }
    }
}

/// A reason-coded facade error from an identification refusal. Its text is
/// `reason=<code>: <namespaced detail>: <message>`, followed by the search
/// receipt's summary when the refusal is a budget or cancellation stop.
fn refusal_error(refusal: &CounterfactualIdRefusal) -> CausalError {
    CausalError::Compile { message: format!("{REASON_PREFIX}{refusal}") }
}

/// A decided query, ready to evaluate on any law of its variables and levels.
#[derive(Clone, Debug)]
pub struct PreparedCounterfactualId {
    problem: CounterfactualIdProblem,
    query: CounterfactualEventQuery,
    derivation: CounterfactualIdDerivation,
    names: Vec<String>,
}

/// Decide `query` on `graph` once.
///
/// `names` names the graph's variables in id order; `levels` lists each
/// variable's levels.
///
/// # Errors
///
/// A reason-coded [`CausalError`] (see the module documentation).
pub fn prepare_counterfactual_id(
    graph: impl IntoGraphInput,
    names: Vec<String>,
    levels: Vec<Vec<f64>>,
    query: &CounterfactualEventQuery,
    options: CounterfactualIdOptions,
    ctx: &ExecutionContext,
) -> Result<PreparedCounterfactualId, CausalError> {
    if options.interval_requested {
        return Err(refusal_error(&CounterfactualIdRefusal::interval_requested()));
    }
    let (structure, source) = graph.into_graph_input();
    let admg: Admg = match (source, structure.class()) {
        (StructureSource::Explicit, GraphClass::Admg) => {
            structure.as_admg().cloned().ok_or_else(|| {
                refusal_error(&CounterfactualIdRefusal::graph("the ADMG is not available"))
            })?
        }
        (StructureSource::Explicit, GraphClass::Dag) => {
            let dag = structure.as_dag().ok_or_else(|| {
                refusal_error(&CounterfactualIdRefusal::graph("the DAG is not available"))
            })?;
            antecedent_identify::dag_to_admg(dag).map_err(|e| {
                refusal_error(&CounterfactualIdRefusal::graph(format!("the DAG is invalid: {e}")))
            })?
        }
        _ => {
            return Err(refusal_error(&CounterfactualIdRefusal::graph(
                "only a supplied explicit ADMG or DAG is licensed; accepted, uncertain, \
                 equivalence-class and temporal structures are refused",
            )));
        }
    };
    let problem = CounterfactualIdProblem::from_admg(&admg, levels)
        .map_err(|refusal| refusal_error(&refusal))?;
    check_names(&names, problem.node_count()).map_err(|refusal| refusal_error(&refusal))?;
    let derivation =
        decide_counterfactual_id(&problem, query, options.limits, options.memory_limit_bytes, ctx)
            .map_err(|refusal| refusal_error(&refusal))?;
    Ok(PreparedCounterfactualId { problem, query: query.clone(), derivation, names })
}

fn check_names(names: &[String], n: usize) -> Result<(), CounterfactualIdRefusal> {
    let distinct: std::collections::BTreeSet<&String> = names.iter().collect();
    if names.len() != n || distinct.len() != n || names.iter().any(String::is_empty) {
        return Err(CounterfactualIdRefusal::invalid_query(format!(
            "{} names for a graph of {n} variables; names must be distinct and non-empty",
            names.len()
        )));
    }
    Ok(())
}

impl PreparedCounterfactualId {
    /// The derivation (identified once, reused by every evaluation).
    #[must_use]
    pub const fn derivation(&self) -> &CounterfactualIdDerivation {
        &self.derivation
    }

    /// The decided query.
    #[must_use]
    pub const fn query(&self) -> &CounterfactualEventQuery {
        &self.query
    }

    /// Evaluate the prepared derivation on `law` (no re-identification). The law
    /// must cover exactly the prepared variables and levels (axes in any order)
    /// and be a supplied exact law or an empirical count table.
    ///
    /// # Errors
    ///
    /// `invalid_argument` (`counterfactual_id.invalid_query`) for another law,
    /// origin or a zero-probability conditioning event;
    /// `counterfactual_id.positivity_violation` when a conditional the
    /// functional evaluates is on a zero-mass event; the typed cancellation.
    pub fn evaluate(
        &self,
        law: &ExactDiscreteLaw,
        ctx: &ExecutionContext,
    ) -> Result<CounterfactualIdEffect, CausalError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "counterfactual_id" });
        }
        let law_wire = law_wire(law).map_err(|refusal| refusal_error(&refusal))?;
        let point = evaluate_counterfactual_functional(&self.problem, &self.derivation, law, ctx)
            .map_err(|refusal| {
            if refusal.receipt.is_some() {
                CausalError::Cancelled { stage: "counterfactual_id" }
            } else {
                refusal_error(&refusal)
            }
        })?;
        let data_digest = counterfactual_id_data_digest(&law_wire)?;
        let distribution = point
            .numerators
            .iter()
            .map(|(level, p)| (*level, p / point.conditioning_probability))
            .collect();
        Ok(CounterfactualIdEffect {
            probability: point.probability,
            conditioning_probability: point.conditioning_probability,
            outcome_distribution: distribution,
            counterfactual_mean: point.counterfactual_mean,
            observed_mean: point.observed_mean,
            effect: point.effect,
            numerators: point.numerators,
            data_digest,
            independently_verified: false,
            prepared: self.clone(),
            law: law_wire,
        })
    }
}

/// The wire form of a law, refusing an origin the cell does not accept.
fn law_wire(law: &ExactDiscreteLaw) -> Result<CounterfactualIdLawWire, CounterfactualIdRefusal> {
    let counts = match (law.origin(), law.empirical_counts()) {
        (LawOrigin::SuppliedExact, None) => None,
        (LawOrigin::EmpiricalPlugin, Some(counts)) => Some(counts.to_vec()),
        _ => {
            return Err(CounterfactualIdRefusal::invalid_query(
                "the law must be a supplied exact law or an empirical count table",
            ));
        }
    };
    let mut axes = Vec::with_capacity(law.axes().len());
    for axis in law.axes() {
        let levels = axis
            .values
            .iter()
            .map(|v| v.as_f64().map(|x| (x + 0.0).to_bits()))
            .collect::<Option<Vec<u64>>>()
            .ok_or_else(|| CounterfactualIdRefusal::invalid_query("law levels must be numeric"))?;
        axes.push(CounterfactualIdAxisWire { variable: axis.variable.raw(), levels });
    }
    Ok(CounterfactualIdLawWire {
        axes,
        probabilities: law.probabilities().iter().map(|p| p.to_bits()).collect(),
        counts,
        origin: law.origin().as_str().into(),
        population: law.population().into(),
        snapshot: law.snapshot_identity().into(),
    })
}

/// Rebuild the law a wire stores.
fn law_from_wire(
    wire: &CounterfactualIdLawWire,
) -> Result<ExactDiscreteLaw, CounterfactualIdArtifactError> {
    let malformed =
        |e: antecedent_expr::ExactLawError| CounterfactualIdArtifactError::Malformed(e.to_string());
    let axes: Vec<DiscreteAxis> = wire
        .axes
        .iter()
        .map(|a| DiscreteAxis {
            variable: VariableId::from_raw(a.variable),
            values: a
                .levels
                .iter()
                .map(|b| Value::f64(f64::from_bits(*b)))
                .collect::<Vec<_>>()
                .into(),
        })
        .collect();
    let probabilities: Vec<f64> = wire.probabilities.iter().map(|b| f64::from_bits(*b)).collect();
    let tolerance = LawTolerance::default();
    match (wire.origin.as_str(), &wire.counts) {
        ("supplied_exact", None) => ExactDiscreteLaw::try_new(
            wire.population.as_str(),
            RegimeId::from_raw(0),
            [],
            axes,
            probabilities,
            wire.snapshot.as_str(),
            tolerance,
        )
        .map_err(malformed),
        ("empirical_plugin", Some(counts)) => ExactDiscreteLaw::try_empirical(
            wire.population.as_str(),
            RegimeId::from_raw(0),
            [],
            axes,
            probabilities,
            wire.snapshot.as_str(),
            tolerance,
        )
        .and_then(|law| law.with_empirical_counts(counts.clone()))
        .map_err(malformed),
        _ => Err(CounterfactualIdArtifactError::UnsupportedSemantics("law origin")),
    }
}

/// The point, its outcome distribution and contrast, and the derivation.
#[derive(Clone, Debug)]
pub struct CounterfactualIdEffect {
    /// `P(Y_x = y | X = x')` (0 for an exact zero).
    pub probability: f64,
    /// `P(X = x')`.
    pub conditioning_probability: f64,
    /// `(level, P(Y_x = level | X = x'))` for every level of `Y`, declared order.
    pub outcome_distribution: Vec<(f64, f64)>,
    /// `E[Y_x | X = x']`.
    pub counterfactual_mean: Option<f64>,
    /// `E[Y | X = x']`.
    pub observed_mean: Option<f64>,
    /// `E[Y_x | X = x'] - E[Y | X = x']`.
    pub effect: Option<f64>,
    /// `(level, P(Y_x = level, X = x'))`.
    pub numerators: Vec<(f64, f64)>,
    /// Digest of the law the point was computed on; compare it with your own.
    pub data_digest: String,
    /// The consumer recomputed the point with the separate direct-sum evaluator
    /// and it agreed. `false` on a freshly evaluated effect.
    pub independently_verified: bool,
    prepared: PreparedCounterfactualId,
    law: CounterfactualIdLawWire,
}

impl CounterfactualIdEffect {
    /// The derivation the point was evaluated from.
    #[must_use]
    pub const fn derivation(&self) -> &CounterfactualIdDerivation {
        &self.prepared.derivation
    }

    /// The variable names, in id order.
    #[must_use]
    pub fn names(&self) -> &[String] {
        &self.prepared.names
    }

    /// Export the replayable `counterfactual_id_admg_v1` artifact.
    ///
    /// # Errors
    ///
    /// [`CausalError::Resource`] when the artifact exceeds the format's bounds,
    /// or an encoding failure.
    pub fn export_artifact(&self) -> Result<Vec<u8>, CausalError> {
        self.artifact_wire().and_then(|wire| wire.export()).map_err(|error| match error {
            CounterfactualIdArtifactError::LimitsExceeded(what) => CausalError::Resource {
                message: format!(
                    "the counterfactual identification artifact does not carry {what}"
                ),
            },
            other => CausalError::Serialization(antecedent_io::IoError::Convert(other.to_string())),
        })
    }

    fn artifact_wire(&self) -> Result<CounterfactualIdArtifactWire, CounterfactualIdArtifactError> {
        let problem = &self.prepared.problem;
        let bits = |x: Option<f64>| x.map(f64::to_bits);
        CounterfactualIdArtifactWire {
            version: COUNTERFACTUAL_ID_ARTIFACT_VERSION,
            required_features: vec![COUNTERFACTUAL_ID_ARTIFACT_FEATURE.into()],
            variable_names: self.prepared.names.clone(),
            levels: (0..problem.node_count())
                .map(|v| problem.levels(v).into_iter().map(f64::to_bits).collect())
                .collect(),
            directed: problem.directed().to_vec(),
            bidirected: problem.bidirected().to_vec(),
            query: CounterfactualIdQueryWire::from_query(&self.prepared.query),
            estimand: COUNTERFACTUAL_ID_ESTIMAND.into(),
            contract: COUNTERFACTUAL_ID_CONTRACT.into(),
            search: self.prepared.derivation.search,
            derivation: self.prepared.derivation.canonical_text(),
            counterfactual_graph: self.prepared.derivation.counterfactual_graph.clone(),
            law: self.law.clone(),
            point: CounterfactualIdPointWire {
                probability: self.probability.to_bits(),
                conditioning_probability: self.conditioning_probability.to_bits(),
                numerators: self
                    .numerators
                    .iter()
                    .map(|(l, p)| (l.to_bits(), p.to_bits()))
                    .collect(),
                counterfactual_mean: bits(self.counterfactual_mean),
                observed_mean: bits(self.observed_mean),
                effect: bits(self.effect),
            },
            data_digest: String::new(),
            premises_digest: String::new(),
        }
        .sealed()
    }
}

/// Bounds a consumer imposes on a replay. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct CounterfactualIdConsumeLimits {
    /// Largest search limits a stored derivation may have run under.
    pub search: SearchLimits,
    /// Largest search memory cap a stored derivation may have run under.
    pub search_memory_bytes: u64,
}

impl Default for CounterfactualIdConsumeLimits {
    fn default() -> Self {
        Self {
            search: SearchLimits {
                operations: COUNTERFACTUAL_ID_MAX_OPERATIONS,
                depth: COUNTERFACTUAL_ID_MAX_DEPTH,
            },
            search_memory_bytes: COUNTERFACTUAL_ID_MEMORY_BYTES,
        }
    }
}

/// Replay a `counterfactual_id_admg_v1` artifact and accept it only when its
/// digests, derivation, search accounting and point all replay and the
/// separate direct-sum evaluation agrees.
///
/// # Errors
///
/// A [`CounterfactualIdArtifactError`] naming the failed check; its
/// [`CounterfactualIdArtifactError::reason`] is the registered pair.
pub fn consume_counterfactual_id_artifact(
    bytes: &[u8],
    limits: CounterfactualIdConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<CounterfactualIdEffect, CounterfactualIdArtifactError> {
    consume_inner(bytes, None, limits, ctx)
}

/// [`consume_counterfactual_id_artifact`] that also requires the artifact's law
/// to have the caller's data digest.
///
/// # Errors
///
/// As [`consume_counterfactual_id_artifact`], and `DataIdentityMismatch` for
/// an artifact computed on another law.
pub fn consume_counterfactual_id_artifact_for_data(
    bytes: &[u8],
    expected_data_digest: &str,
    limits: CounterfactualIdConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<CounterfactualIdEffect, CounterfactualIdArtifactError> {
    consume_inner(bytes, Some(expected_data_digest), limits, ctx)
}

fn consume_inner(
    bytes: &[u8],
    expected_data_digest: Option<&str>,
    limits: CounterfactualIdConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<CounterfactualIdEffect, CounterfactualIdArtifactError> {
    let wire = CounterfactualIdArtifactWire::decode(bytes)?;
    wire.check_bounds()?;
    // Stored limits above the consumer's refuse before any work: a limits
    // refusal the caller can retry, not a claim that the artifact is invalid.
    let exceeded = CounterfactualIdArtifactError::LimitsExceeded;
    if wire.search.operations_limit > limits.search.operations {
        return Err(exceeded("search operation limit"));
    }
    if wire.search.depth_limit > limits.search.depth {
        return Err(exceeded("search depth limit"));
    }
    let memory_cap = ctx
        .memory
        .hard_limit_bytes
        .map_or(limits.search_memory_bytes, |hard| hard.min(limits.search_memory_bytes));
    if wire.search.memory_limit_bytes > memory_cap {
        return Err(exceeded("search memory limit"));
    }
    wire.check_digests(expected_data_digest)?;
    if wire.estimand != COUNTERFACTUAL_ID_ESTIMAND {
        return Err(CounterfactualIdArtifactError::UnsupportedSemantics("estimand"));
    }
    if wire.contract != COUNTERFACTUAL_ID_CONTRACT {
        return Err(CounterfactualIdArtifactError::UnsupportedSemantics("contract"));
    }
    let levels: Vec<Vec<f64>> =
        wire.levels.iter().map(|l| l.iter().map(|b| f64::from_bits(*b)).collect()).collect();
    let problem = CounterfactualIdProblem::new(levels, &wire.directed, &wire.bidirected)
        .map_err(|r| CounterfactualIdArtifactError::Malformed(r.to_string()))?;
    if problem.directed() != wire.directed || problem.bidirected() != wire.bidirected {
        return Err(CounterfactualIdArtifactError::Malformed(
            "the stored graph is not canonical".into(),
        ));
    }
    check_names(&wire.variable_names, problem.node_count())
        .map_err(|r| CounterfactualIdArtifactError::Malformed(r.to_string()))?;
    let query = wire.query.to_query()?;
    let law = law_from_wire(&wire.law)?;
    // Re-derive under the producer's stored limits and memory cap.
    let derivation = decide_counterfactual_id(
        &problem,
        &query,
        SearchLimits { operations: wire.search.operations_limit, depth: wire.search.depth_limit },
        wire.search.memory_limit_bytes,
        ctx,
    )
    .map_err(|r| CounterfactualIdArtifactError::DerivationMismatch(r.to_string()))?;
    if derivation.canonical_text() != wire.derivation
        || derivation.search != wire.search
        || derivation.counterfactual_graph != wire.counterfactual_graph
    {
        return Err(CounterfactualIdArtifactError::DerivationMismatch(
            "the re-derived functional, search accounting or counterfactual graph differs".into(),
        ));
    }
    let prepared =
        PreparedCounterfactualId { problem, query, derivation, names: wire.variable_names.clone() };
    let mut effect = prepared.evaluate(&law, ctx).map_err(|e| {
        CounterfactualIdArtifactError::DerivationMismatch(format!("evaluation refused: {e}"))
    })?;
    let bits = |x: Option<f64>| x.map(f64::to_bits);
    let replayed = CounterfactualIdPointWire {
        probability: effect.probability.to_bits(),
        conditioning_probability: effect.conditioning_probability.to_bits(),
        numerators: effect.numerators.iter().map(|(l, p)| (l.to_bits(), p.to_bits())).collect(),
        counterfactual_mean: bits(effect.counterfactual_mean),
        observed_mean: bits(effect.observed_mean),
        effect: bits(effect.effect),
    };
    if replayed != wire.point {
        return Err(CounterfactualIdArtifactError::PointMismatch);
    }
    let direct = direct_sum_numerators(&prepared.derivation, &wire.law, &wire.levels)
        .ok_or(CounterfactualIdArtifactError::IndependentCheckMismatch)?;
    if direct.len() != effect.numerators.len()
        || direct.iter().zip(&effect.numerators).any(|(d, (_, p))| !agrees(*d, *p))
    {
        return Err(CounterfactualIdArtifactError::IndependentCheckMismatch);
    }
    effect.independently_verified = true;
    Ok(effect)
}

fn agrees(a: f64, b: f64) -> bool {
    (a - b).abs() <= DIRECT_SUM_TOLERANCE * (1.0 + a.abs().max(b.abs()))
}

/// Independent evaluation of every level's numerator: walks the functional and
/// each term's expression with its own arithmetic, reading masses straight from
/// the stored law's cells. It shares no evaluation code with the compiled
/// evaluator or its factor tables, but it evaluates the same re-derived
/// functional. `None` when the value depends on how a conditional on a
/// zero-mass event is extended, or a consistency complement is negative.
fn direct_sum_numerators(
    derivation: &CounterfactualIdDerivation,
    law: &CounterfactualIdLawWire,
    levels: &[Vec<u64>],
) -> Option<Vec<f64>> {
    if !matches!(derivation.shape, CounterfactualIdShape::EffectOnTreated { .. }) {
        return Some(Vec::new());
    }
    // A conditional on a zero-mass event is extended uniformly and by all mass on
    // each level `k` of its conditioned variables; the value must not depend on
    // the extension (it is multiplied by zero mass).
    let most = levels.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let fills: Vec<WireCells> = std::iter::once(Fill::Uniform)
        .chain((0..most).map(Fill::Level))
        .map(|fill| WireCells::new(law, levels, fill))
        .collect();
    let mut out = Vec::with_capacity(derivation.numerators.len());
    for (_, functional) in &derivation.numerators {
        let first = fills[0].functional(functional, &mut BTreeMap::new())?;
        for cells in &fills[1..] {
            if !agrees(first, cells.functional(functional, &mut BTreeMap::new())?) {
                return None;
            }
        }
        out.push(first);
    }
    Some(out)
}

/// How the direct-sum evaluator extends a conditional on a zero-mass event.
#[derive(Clone, Copy)]
enum Fill {
    /// Every configuration of the conditioned variables equally likely.
    Uniform,
    /// All mass on level `k` (clamped to the last level) of each conditioned variable.
    Level(usize),
}

/// The stored law's cells as `(variable -> level bits, probability)` rows.
struct WireCells {
    rows: Vec<(BTreeMap<u32, u64>, f64)>,
    levels: BTreeMap<u32, Vec<u64>>,
    /// How a conditional on a zero-mass event is extended.
    fill: Fill,
}

impl WireCells {
    /// `levels` is the declared level order per variable: sums and the choice
    /// of a free variable's level follow it, as the compiled evaluation does.
    fn new(law: &CounterfactualIdLawWire, levels: &[Vec<u64>], fill: Fill) -> Self {
        let mut rows = Vec::with_capacity(law.probabilities.len());
        let cards: Vec<usize> = law.axes.iter().map(|a| a.levels.len()).collect();
        let mut index = vec![0usize; cards.len()];
        for p in &law.probabilities {
            let row: BTreeMap<u32, u64> =
                law.axes.iter().zip(&index).map(|(a, &i)| (a.variable, a.levels[i])).collect();
            rows.push((row, f64::from_bits(*p)));
            for k in (0..cards.len()).rev() {
                index[k] += 1;
                if index[k] < cards[k] {
                    break;
                }
                index[k] = 0;
            }
        }
        let levels = levels
            .iter()
            .enumerate()
            .map(|(v, l)| (u32::try_from(v).unwrap_or(u32::MAX), l.clone()))
            .collect();
        Self { rows, levels, fill }
    }

    fn mass(&self, fixed: &BTreeMap<u32, u64>) -> f64 {
        self.rows
            .iter()
            .filter(|(row, _)| fixed.iter().all(|(v, bits)| row.get(v) == Some(bits)))
            .map(|(_, p)| p)
            .sum()
    }

    fn resolve(&self, value: CfValue, variable: u32, symbols: &BTreeMap<u32, u64>) -> Option<u64> {
        match value {
            CfValue::Level(bits) => Some(bits),
            CfValue::Symbol(id) => symbols.get(&id).copied(),
        }
        .filter(|bits| self.levels.get(&variable).is_some_and(|l| l.contains(bits)))
    }

    fn functional(
        &self,
        functional: &CounterfactualFunctional,
        symbols: &mut BTreeMap<u32, u64>,
    ) -> Option<f64> {
        match functional {
            CounterfactualFunctional::One => Some(1.0),
            CounterfactualFunctional::Zero => Some(0.0),
            CounterfactualFunctional::Term(term) => self.term(term, symbols),
            CounterfactualFunctional::ConsistencyComplement { interventional, observed } => {
                let interventional = self.term(interventional, symbols)?;
                let mut fixed = BTreeMap::new();
                for &(v, value) in observed {
                    fixed.insert(v, self.resolve(value, v, symbols)?);
                }
                let value = interventional - self.mass(&fixed);
                // A negative complement means a law incompatible with the graph.
                (value >= -1e-12 * (1.0 + interventional.abs())).then_some(value.max(0.0))
            }
            CounterfactualFunctional::Sum { symbols: summed, factors } => {
                self.sum(summed, factors, symbols)
            }
        }
    }

    /// `sum_{symbols} prod(factors)`: the symbols' levels are bound in declared
    /// order, innermost last.
    fn sum(
        &self,
        summed: &[CfSymbol],
        factors: &[CounterfactualFunctional],
        symbols: &mut BTreeMap<u32, u64>,
    ) -> Option<f64> {
        let Some((first, rest)) = summed.split_first() else {
            // A zero factor ends the product: a later factor on a null event is
            // not read (the compiled evaluation's convention).
            let mut product = 1.0;
            for factor in factors {
                product *= self.functional(factor, symbols)?;
                if product == 0.0 {
                    break;
                }
            }
            return Some(product);
        };
        let mut total = 0.0;
        for &bits in self.levels.get(&first.variable)? {
            symbols.insert(first.id, bits);
            total += self.sum(rest, factors, symbols)?;
        }
        symbols.remove(&first.id);
        Some(total)
    }

    fn term(&self, term: &CfTerm, symbols: &BTreeMap<u32, u64>) -> Option<f64> {
        let mut env = BTreeMap::new();
        for &(v, value) in term.intervention.iter().chain(&term.event) {
            env.insert(v, self.resolve(value, v, symbols)?);
        }
        // Variables the functional leaves free beyond the treatments and the event:
        // the first level (declared order of the stored law) where it is defined.
        let free: Vec<u32> = term
            .arena
            .free_variables(term.expression)
            .into_iter()
            .map(VariableId::raw)
            .filter(|v| !env.contains_key(v))
            .collect();
        self.first_defined(&term.arena, term.expression, &mut env, &free)
    }

    fn first_defined(
        &self,
        arena: &antecedent_expr::CausalExprArena,
        root: ExprId,
        env: &mut BTreeMap<u32, u64>,
        free: &[u32],
    ) -> Option<f64> {
        let Some((&first, rest)) = free.split_first() else {
            return self.expr(arena, root, env).filter(|x| x.is_finite());
        };
        for &bits in self.levels.get(&first)? {
            env.insert(first, bits);
            if let Some(value) = self.first_defined(arena, root, env, rest) {
                env.remove(&first);
                return Some(value);
            }
        }
        env.remove(&first);
        None
    }

    fn expr(
        &self,
        arena: &antecedent_expr::CausalExprArena,
        id: ExprId,
        env: &mut BTreeMap<u32, u64>,
    ) -> Option<f64> {
        match arena.node(id) {
            ExprNode::Distribution { variables, conditioned_on, .. } => {
                let mut cond = BTreeMap::new();
                for v in arena.var_set(*conditioned_on) {
                    cond.insert(v.raw(), *env.get(&v.raw())?);
                }
                let mut joint = cond.clone();
                for v in arena.var_set(*variables) {
                    joint.insert(v.raw(), *env.get(&v.raw())?);
                }
                let denominator = self.mass(&cond);
                if denominator > 0.0 {
                    return Some(self.mass(&joint) / denominator);
                }
                let vars = arena.var_set(*variables);
                match self.fill {
                    Fill::Level(k) => {
                        let at_k = vars.iter().all(|v| {
                            self.levels.get(&v.raw()).and_then(|l| l.get(k.min(l.len() - 1)))
                                == env.get(&v.raw())
                        });
                        Some(if at_k { 1.0 } else { 0.0 })
                    }
                    Fill::Uniform => {
                        let configurations: usize = vars
                            .iter()
                            .map(|v| self.levels.get(&v.raw()).map_or(1, Vec::len))
                            .product();
                        Some(1.0 / configurations as f64)
                    }
                }
            }
            ExprNode::Product(list) => arena
                .list(*list)
                .iter()
                .try_fold(1.0, |acc, &c| Some(acc * self.expr(arena, c, env)?)),
            ExprNode::SumOut { variables, expr } => {
                let vars: Vec<u32> = arena.var_set(*variables).iter().map(|v| v.raw()).collect();
                let saved: Vec<Option<u64>> = vars.iter().map(|v| env.get(v).copied()).collect();
                let total = self.sum_over(arena, *expr, env, &vars);
                for (v, old) in vars.iter().zip(saved) {
                    match old {
                        Some(bits) => env.insert(*v, bits),
                        None => env.remove(v),
                    };
                }
                total
            }
            ExprNode::Ratio { numerator, denominator } => {
                let d = self.expr(arena, *denominator, env)?;
                (d != 0.0).then_some(())?;
                Some(self.expr(arena, *numerator, env)? / d)
            }
            _ => None,
        }
    }

    fn sum_over(
        &self,
        arena: &antecedent_expr::CausalExprArena,
        body: ExprId,
        env: &mut BTreeMap<u32, u64>,
        vars: &[u32],
    ) -> Option<f64> {
        let Some((&first, rest)) = vars.split_first() else {
            return self.expr(arena, body, env);
        };
        let mut total = 0.0;
        for &bits in self.levels.get(&first)? {
            env.insert(first, bits);
            total += self.sum_over(arena, body, env, rest)?;
        }
        Some(total)
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::VariableId;

    use super::*;

    fn frontdoor_law(probabilities: Vec<f64>) -> ExactDiscreteLaw {
        let axes: Vec<DiscreteAxis> = (0..3)
            .map(|i| DiscreteAxis {
                variable: VariableId::from_raw(i),
                values: vec![Value::f64(0.0), Value::f64(1.0)].into(),
            })
            .collect();
        ExactDiscreteLaw::try_new(
            "target",
            RegimeId::from_raw(0),
            [],
            axes,
            probabilities,
            "unit",
            LawTolerance::default(),
        )
        .unwrap()
    }

    /// The consumer's direct-sum evaluator reads masses straight from the law's
    /// cells and walks the expressions with its own arithmetic: it reproduces
    /// the compiled evaluation and the front-door formula, and on a law where a
    /// conditional with positive weight is undefined it declines, as the
    /// compiled evaluation refuses.
    #[test]
    fn the_direct_sum_evaluation_agrees_with_the_compiled_one_and_the_formula() {
        let ctx = ExecutionContext::for_tests(1);
        let problem =
            CounterfactualIdProblem::new(vec![vec![0.0, 1.0]; 3], &[(0, 1), (1, 2)], &[(0, 2)])
                .unwrap();
        let query = CounterfactualEventQuery::effect_on_treated(
            VariableId::from_raw(0),
            1.0,
            0.0,
            VariableId::from_raw(2),
            1.0,
        )
        .unwrap();
        let derivation = decide_counterfactual_id(
            &problem,
            &query,
            COUNTERFACTUAL_ID_DEFAULT_LIMITS,
            COUNTERFACTUAL_ID_MEMORY_BYTES,
            &ctx,
        )
        .unwrap();
        let levels = vec![vec![0u64, 1.0f64.to_bits()]; 3];
        let p = vec![0.10, 0.05, 0.08, 0.17, 0.12, 0.13, 0.06, 0.29];
        let law = frontdoor_law(p.clone());
        let point = evaluate_counterfactual_functional(&problem, &derivation, &law, &ctx).unwrap();
        let direct = direct_sum_numerators(&derivation, &law_wire(&law).unwrap(), &levels).unwrap();
        assert_eq!(direct.len(), point.numerators.len());
        for (d, (_, c)) in direct.iter().zip(&point.numerators) {
            assert!(agrees(*d, *c), "{d} vs {c}");
        }
        // P(Y_x = 1, X = 0) = P(x' = 0) sum_m P(m | x = 1) P(y = 1 | m, x' = 0).
        let cell = |x: usize, m: usize, y: usize| p[x * 4 + m * 2 + y];
        let px = |x: usize| (0..4).map(|k| p[x * 4 + k]).sum::<f64>();
        let pxm = |x: usize, m: usize| cell(x, m, 0) + cell(x, m, 1);
        let formula =
            px(0) * (0..2).map(|m| pxm(1, m) / px(1) * cell(0, m, 1) / pxm(0, m)).sum::<f64>();
        assert!(agrees(direct[1], formula), "{} vs {formula}", direct[1]);
        // P(X = 0, M = 1) = 0 while P(M = 1 | X = 1) > 0: both evaluators decline.
        let gap = frontdoor_law(vec![0.2, 0.2, 0.0, 0.0, 0.1, 0.1, 0.2, 0.2]);
        assert!(direct_sum_numerators(&derivation, &law_wire(&gap).unwrap(), &levels).is_none());
        assert!(evaluate_counterfactual_functional(&problem, &derivation, &gap, &ctx).is_err());
    }
}
