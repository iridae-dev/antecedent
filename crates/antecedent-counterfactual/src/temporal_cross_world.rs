//! Fixed-population temporal counterfactual with one shared abduced history per unit.
//!
//! # Named theorem (cross-world, fixed Markovian two-slice SCM)
//!
//! The graph is one fixed, fully observed, Markovian two-slice DAG over the nodes
//! `L0 -> A0 -> L1 -> A1 -> Y` (a covariate and an action per slice, then the
//! final outcome), with no latent confounding. Every non-action node has an
//! additive-noise **linear-Gaussian** mechanism
//! `V = intercept + sum_p coef_p * parent_p + U_V`, supplied as a fitted mechanism
//! identity. Under that model each unit's exogenous history `(U_L0, U_L1, U_Y)` is
//! recovered exactly from its factual two-slice trajectory (abduction), both named
//! action histories are replayed against that **same** history (action), and the
//! final outcome of each world is computed from it (prediction). The per-unit
//! counterfactual final outcome is `Y_h(u)` and the estimand is the sample mean of
//! `Y_plus - Y_minus`. The 2.2 static edge-intervention proof does not identify
//! this quantity; drawing fresh noise per world would answer a different question
//! (an interventional, not a counterfactual, one).
//!
//! The mechanism class is chosen as linear-Gaussian additive noise (closed-form
//! abduction) rather than the static finite-SCM engine, because the engine's
//! structural-equation compiler has no lagged slice. Abduction, action and
//! prediction happen in one function; there is no entry point that abduces
//! separately from predicting.
//!
//! The Gaussian assumption is never used numerically (additive noise makes
//! abduction an exact residual); an optional per-node `noise_halfwidth` declares a
//! bounded noise support so a factual history outside it is a *refuting witness*.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::{ExecutionContext, StructuredRefusal};
use thiserror::Error;

/// Maximum number of units.
pub const MAX_UNITS: usize = 100_000;
/// Maximum number of worlds (the two named action histories).
pub const MAX_WORLDS: usize = 2;
/// Maximum (and only supported) horizon, in slices.
pub const MAX_HORIZON: usize = 2;
/// Mechanism class of this route.
pub const MECHANISM_CLASS: &str = "linear_gaussian_additive_noise";

const POLL_EVERY: usize = 1024;

/// A node of the fixed two-slice temporal graph, in topological order.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TemporalNode {
    /// Slice-0 covariate.
    Covariate0,
    /// Slice-0 action.
    Action0,
    /// Slice-1 covariate.
    Covariate1,
    /// Slice-1 action.
    Action1,
    /// Final outcome.
    Outcome,
}

impl TemporalNode {
    /// Every node in topological order.
    pub const ALL: [TemporalNode; 5] = [
        TemporalNode::Covariate0,
        TemporalNode::Action0,
        TemporalNode::Covariate1,
        TemporalNode::Action1,
        TemporalNode::Outcome,
    ];

    /// Position in topological order, which is also the index into a unit's
    /// five-value history `[L0, A0, L1, A1, Y]`.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Covariate0 => 0,
            Self::Action0 => 1,
            Self::Covariate1 => 2,
            Self::Action1 => 3,
            Self::Outcome => 4,
        }
    }

    /// Whether the node is an action (set by the history, never abduced).
    #[must_use]
    pub const fn is_action(self) -> bool {
        matches!(self, Self::Action0 | Self::Action1)
    }

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Covariate0 => "covariate_0",
            Self::Action0 => "action_0",
            Self::Covariate1 => "covariate_1",
            Self::Action1 => "action_1",
            Self::Outcome => "outcome",
        }
    }
}

macro_rules! string_id {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(String);

        impl $name {
            /// Wrap an identifier.
            #[must_use]
            pub fn new(id: impl Into<String>) -> Self {
                Self(id.into())
            }

            /// The identifier text.
            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }
    };
}

string_id!(
    /// Stable identifier of a unit (the same unit in the factual and both counterfactual worlds).
    UnitId
);
string_id!(
    /// Stable identifier of a unit's factual history.
    HistoryId
);
string_id!(
    /// Identifier of the data snapshot the factual histories were read from.
    SnapshotId
);

/// The fixed temporal DAG: its edges, horizon and a latent-confounding flag.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemporalGraph {
    /// Number of slices; only [`MAX_HORIZON`] is supported.
    pub horizon: usize,
    /// Directed edges `(parent, child)`; parents must precede children and no
    /// edge may point into an action.
    pub edges: Vec<(TemporalNode, TemporalNode)>,
    /// Whether the graph has latent confounding; `true` is refused.
    pub latent_confounding: bool,
}

/// The additive-noise linear mechanism of one non-action node.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeMechanism {
    /// The node it generates.
    pub node: TemporalNode,
    /// Intercept.
    pub intercept: f64,
    /// Parent coefficients; the parents must equal the graph's parents of `node`.
    pub parent_coefficients: Vec<(TemporalNode, f64)>,
    /// Optional half-width of the noise support; a larger abduced residual is a
    /// refuting witness.
    pub noise_halfwidth: Option<f64>,
}

/// The mechanism fit identity: one mechanism for each of `Covariate0`,
/// `Covariate1` and `Outcome`.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalMechanismFit {
    /// Stable identity of the fit (supplied or fitted elsewhere).
    pub fit_id: String,
    /// The mechanisms.
    pub mechanisms: Vec<NodeMechanism>,
}

/// One unit's factual two-slice trajectory.
#[derive(Clone, Debug, PartialEq)]
pub struct FactualUnitHistory {
    /// The unit.
    pub unit: UnitId,
    /// The unit's factual history id.
    pub history: HistoryId,
    /// Observation time of slice 0 and slice 1.
    pub times: [u32; 2],
    /// Observed `[L0, A0, L1, A1, Y]`.
    pub values: [f64; 5],
}

/// A named two-slice action history and the units it is requested for.
#[derive(Clone, Debug, PartialEq)]
pub struct NamedActionHistory {
    /// Name of the history.
    pub name: String,
    /// Time of each slice's action; must equal the factual observation times.
    pub times: [u32; 2],
    /// The actions `[A0, A1]`.
    pub actions: [f64; 2],
    /// The units this world is evaluated for; must pair with the factual units.
    pub units: Vec<UnitId>,
}

/// The temporal counterfactual request.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalCounterfactualSpec {
    /// The fixed temporal graph.
    pub graph: TemporalGraph,
    /// The mechanism fit.
    pub fit: TemporalMechanismFit,
    /// Snapshot the factual histories come from.
    pub snapshot: SnapshotId,
    /// Factual histories, one per unit.
    pub factual: Vec<FactualUnitHistory>,
    /// The added action history.
    pub plus: NamedActionHistory,
    /// The subtracted action history.
    pub minus: NamedActionHistory,
}

/// What a refusal retains about the offending unit or history.
#[derive(Clone, Debug, PartialEq)]
pub struct RefusingWitness {
    /// Offending unit.
    pub unit: Option<UnitId>,
    /// Offending factual history.
    pub history: Option<HistoryId>,
    /// World (action-history name) concerned.
    pub world: Option<String>,
    /// Node whose abduction failed.
    pub node: Option<TemporalNode>,
    /// Abduced residual.
    pub residual: Option<f64>,
    /// Declared noise half-width it violated.
    pub bound: Option<f64>,
}

impl RefusingWitness {
    const fn empty() -> Self {
        Self { unit: None, history: None, world: None, node: None, residual: None, bound: None }
    }
}

/// A structured refusal plus the retained witness.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalRefusal {
    /// Registered code, stage and `family.slot` detail.
    pub refusal: StructuredRefusal,
    /// The refusing witness, when one exists.
    pub witness: Option<RefusingWitness>,
}

/// Temporal counterfactual errors.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum TemporalCounterfactualError {
    /// A typed refusal.
    #[error("temporal counterfactual refused: {}", .0.refusal.detail)]
    Refused(Box<TemporalRefusal>),
    /// Cooperative cancellation was observed.
    #[error("cancelled during temporal counterfactual evaluation")]
    Cancelled,
}

impl TemporalCounterfactualError {
    /// The structured refusal, when the error is a refusal.
    #[must_use]
    pub fn refusal(&self) -> Option<&StructuredRefusal> {
        match self {
            Self::Refused(r) => Some(&r.refusal),
            Self::Cancelled => None,
        }
    }

    /// The retained witness, when there is one.
    #[must_use]
    pub fn witness(&self) -> Option<&RefusingWitness> {
        match self {
            Self::Refused(r) => r.witness.as_ref(),
            Self::Cancelled => None,
        }
    }
}

fn refuse(
    code: &'static str,
    detail: &str,
    offending: Option<String>,
    witness: Option<RefusingWitness>,
) -> TemporalCounterfactualError {
    TemporalCounterfactualError::Refused(Box::new(TemporalRefusal {
        refusal: StructuredRefusal {
            code,
            stage: "evaluate",
            detail: detail.to_owned(),
            offending,
            expected: None,
            supplied: None,
            capability: None,
            remedy: None,
        },
        witness,
    }))
}

fn invalid(detail: &str, offending: Option<String>) -> TemporalCounterfactualError {
    refuse(antecedent_core::reason_code!("invalid_argument"), detail, offending, None)
}

fn unsupported(
    detail: &str,
    offending: Option<String>,
    witness: Option<RefusingWitness>,
) -> TemporalCounterfactualError {
    refuse(antecedent_core::reason_code!("route_not_supported"), detail, offending, witness)
}

/// A bit-reproducible 128-bit integrity digest (two FNV-1a lanes); not cryptographic.
struct Digest {
    a: u64,
    b: u64,
}

impl Digest {
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    fn new(tag: &str) -> Self {
        let mut digest = Self { a: 0xcbf2_9ce4_8422_2325, b: 0x8422_2325_cbf2_9ce4 };
        digest.text(tag);
        digest
    }

    fn byte(&mut self, byte: u8) {
        self.a = (self.a ^ u64::from(byte)).wrapping_mul(Self::PRIME);
        self.b = (self.b ^ u64::from(byte ^ 0x5a)).wrapping_mul(Self::PRIME);
    }

    fn word(&mut self, word: u64) {
        for byte in word.to_le_bytes() {
            self.byte(byte);
        }
    }

    fn text(&mut self, text: &str) {
        self.word(text.len() as u64);
        for byte in text.bytes() {
            self.byte(byte);
        }
    }

    fn real(&mut self, value: f64) {
        self.word(value.to_bits());
    }

    fn hex(&self) -> String {
        format!("{:016x}{:016x}", self.a, self.b)
    }
}

/// One unit's exogenous-draw digest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnitDrawDigest {
    /// The unit.
    pub unit: UnitId,
    /// The unit's factual history.
    pub history: HistoryId,
    /// Digest of the unit's abduced exogenous draw `(U_L0, U_L1, U_Y)`.
    pub digest: String,
}

/// Receipt that both worlds were replayed against one shared abduced history per unit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SharedAbductionReceipt {
    /// Snapshot identity.
    pub snapshot: SnapshotId,
    /// Digest of the temporal graph.
    pub graph_digest: String,
    /// Digest of the mechanism fit identity and coefficients.
    pub fit_digest: String,
    /// Digest of the factual histories (order independent).
    pub factual_digest: String,
    /// Digest of the plus action history (name, times, actions, units).
    pub plus_digest: String,
    /// Digest of the minus action history.
    pub minus_digest: String,
    /// Per-unit exogenous-draw digests, sorted by unit id.
    pub unit_draws: Vec<UnitDrawDigest>,
    /// Digest over all unit draws.
    pub exogenous_digest: String,
    /// Number of units.
    pub n_units: usize,
    /// Number of worlds (always [`MAX_WORLDS`]).
    pub n_worlds: usize,
    /// Horizon (always [`MAX_HORIZON`]).
    pub horizon: usize,
    /// Both worlds read the same draw for every unit (always true for this route).
    pub shared_by_both_worlds: bool,
    /// Overall receipt digest.
    pub digest: String,
}

impl SharedAbductionReceipt {
    fn exogenous_of(unit_draws: &[UnitDrawDigest]) -> String {
        let mut d = Digest::new("temporal_cross_world.exogenous");
        d.word(unit_draws.len() as u64);
        for draw in unit_draws {
            d.text(draw.unit.as_str());
            d.text(draw.history.as_str());
            d.text(&draw.digest);
        }
        d.hex()
    }

    /// Recompute the overall digest and the exogenous digest from the retained
    /// components; bit-identical to [`Self::digest`] for an untampered receipt.
    #[must_use]
    pub fn recompute_digest(&self) -> String {
        let mut d = Digest::new("temporal_cross_world.receipt");
        d.text(self.snapshot.as_str());
        d.text(&self.graph_digest);
        d.text(&self.fit_digest);
        d.text(&self.factual_digest);
        d.text(&self.plus_digest);
        d.text(&self.minus_digest);
        d.text(&Self::exogenous_of(&self.unit_draws));
        d.word(self.n_units as u64);
        d.word(self.n_worlds as u64);
        d.word(self.horizon as u64);
        d.byte(u8::from(self.shared_by_both_worlds));
        d.hex()
    }

    /// Whether the retained digests recompute from the retained components.
    #[must_use]
    pub fn is_self_consistent(&self) -> bool {
        self.exogenous_digest == Self::exogenous_of(&self.unit_draws)
            && self.digest == self.recompute_digest()
    }
}

/// One unit's counterfactual outcomes in both worlds.
#[derive(Clone, Debug, PartialEq)]
pub struct UnitCounterfactual {
    /// The unit.
    pub unit: UnitId,
    /// The unit's factual history.
    pub history: HistoryId,
    /// The factual final outcome.
    pub factual_outcome: f64,
    /// Final outcome under the plus history.
    pub plus_outcome: f64,
    /// Final outcome under the minus history.
    pub minus_outcome: f64,
}

/// The temporal counterfactual answer.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalCounterfactualResult {
    /// Mechanism class ([`MECHANISM_CLASS`]).
    pub mechanism_class: &'static str,
    /// Per-unit outcomes, sorted by unit id.
    pub units: Vec<UnitCounterfactual>,
    /// Mean final outcome under the plus history.
    pub mean_plus: f64,
    /// Mean final outcome under the minus history.
    pub mean_minus: f64,
    /// Sample mean contrast `mean_plus - mean_minus`.
    pub mean_contrast: f64,
    /// The shared-abduction receipt.
    pub receipt: SharedAbductionReceipt,
}

fn poll(ctx: &ExecutionContext) -> Result<(), TemporalCounterfactualError> {
    if ctx.cancellation.is_cancelled() {
        Err(TemporalCounterfactualError::Cancelled)
    } else {
        Ok(())
    }
}

fn sorted_edges(graph: &TemporalGraph) -> Vec<(usize, usize)> {
    let mut edges: Vec<(usize, usize)> =
        graph.edges.iter().map(|(p, c)| (p.index(), c.index())).collect();
    edges.sort_unstable();
    edges
}

fn validate_graph(graph: &TemporalGraph) -> Result<(), TemporalCounterfactualError> {
    if graph.horizon > MAX_HORIZON {
        return Err(invalid(
            "temporal_counterfactual.horizon_exceeded",
            Some(graph.horizon.to_string()),
        ));
    }
    if graph.horizon != MAX_HORIZON {
        return Err(invalid("temporal_counterfactual.invalid_graph", Some("horizon".to_owned())));
    }
    if graph.latent_confounding {
        return Err(unsupported("temporal_counterfactual.latent_confounding", None, None));
    }
    let edges = sorted_edges(graph);
    for (index, (parent, child)) in edges.iter().enumerate() {
        let bad = parent >= child
            || TemporalNode::ALL[*child].is_action()
            || (index > 0 && edges[index - 1] == (*parent, *child));
        if bad {
            return Err(invalid(
                "temporal_counterfactual.invalid_graph",
                Some(format!(
                    "{} -> {}",
                    TemporalNode::ALL[*parent].name(),
                    TemporalNode::ALL[*child].name()
                )),
            ));
        }
    }
    Ok(())
}

/// Look up the mechanisms of `[Covariate0, Covariate1, Outcome]` and check them
/// against the graph.
fn compile_fit<'a>(
    graph: &TemporalGraph,
    fit: &'a TemporalMechanismFit,
) -> Result<[&'a NodeMechanism; 3], TemporalCounterfactualError> {
    let mismatch =
        |what: &str| invalid("temporal_counterfactual.fit_mismatch", Some(what.to_owned()));
    let find = |node: TemporalNode| -> Result<&'a NodeMechanism, TemporalCounterfactualError> {
        let mut hits = fit.mechanisms.iter().filter(|m| m.node == node);
        match (hits.next(), hits.next()) {
            (Some(m), None) => Ok(m),
            _ => Err(mismatch(node.name())),
        }
    };
    if fit.mechanisms.len() != 3 || fit.fit_id.trim().is_empty() {
        return Err(mismatch("mechanisms"));
    }
    let mechs = [
        find(TemporalNode::Covariate0)?,
        find(TemporalNode::Covariate1)?,
        find(TemporalNode::Outcome)?,
    ];
    let edges = sorted_edges(graph);
    for mech in mechs {
        let mut parents: Vec<usize> =
            mech.parent_coefficients.iter().map(|(p, _)| p.index()).collect();
        parents.sort_unstable();
        let graph_parents: Vec<usize> =
            edges.iter().filter(|(_, c)| *c == mech.node.index()).map(|(p, _)| *p).collect();
        let finite = mech.intercept.is_finite()
            && mech.parent_coefficients.iter().all(|(_, c)| c.is_finite())
            && mech.noise_halfwidth.is_none_or(|h| h.is_finite() && h > 0.0);
        if parents != graph_parents || !finite {
            return Err(mismatch(mech.node.name()));
        }
    }
    Ok(mechs)
}

fn node_value(mech: &NodeMechanism, values: &[f64; 5]) -> f64 {
    mech.intercept
        + mech.parent_coefficients.iter().map(|(p, c)| c * values[p.index()]).sum::<f64>()
}

fn unit_set<'a>(
    world: &str,
    units: &'a [UnitId],
) -> Result<BTreeSet<&'a UnitId>, TemporalCounterfactualError> {
    let mut set = BTreeSet::new();
    for unit in units {
        if !set.insert(unit) {
            return Err(unpaired(world, unit, None));
        }
    }
    Ok(set)
}

fn unpaired(
    world: &str,
    unit: &UnitId,
    history: Option<&HistoryId>,
) -> TemporalCounterfactualError {
    unsupported(
        "temporal_counterfactual.unpaired_histories",
        Some(unit.as_str().to_owned()),
        Some(RefusingWitness {
            unit: Some(unit.clone()),
            history: history.cloned(),
            world: Some(world.to_owned()),
            ..RefusingWitness::empty()
        }),
    )
}

fn check_pairing(
    spec: &TemporalCounterfactualSpec,
    factual: &[&FactualUnitHistory],
) -> Result<(), TemporalCounterfactualError> {
    let mut seen: BTreeSet<&UnitId> = BTreeSet::new();
    for unit in factual {
        if !seen.insert(&unit.unit) {
            return Err(unpaired("factual", &unit.unit, Some(&unit.history)));
        }
    }
    for world in [&spec.plus, &spec.minus] {
        let units = unit_set(&world.name, &world.units)?;
        if let Some(missing) = factual.iter().find(|f| !units.contains(&f.unit)) {
            return Err(unpaired(&world.name, &missing.unit, Some(&missing.history)));
        }
        if let Some(extra) = units.iter().find(|u| !seen.contains(**u)) {
            return Err(unpaired(&world.name, extra, None));
        }
    }
    for unit in factual {
        for world in [&spec.plus, &spec.minus] {
            if unit.times != world.times {
                return Err(unsupported(
                    "temporal_counterfactual.time_misaligned",
                    Some(unit.unit.as_str().to_owned()),
                    Some(RefusingWitness {
                        unit: Some(unit.unit.clone()),
                        history: Some(unit.history.clone()),
                        world: Some(world.name.clone()),
                        ..RefusingWitness::empty()
                    }),
                ));
            }
        }
    }
    Ok(())
}

fn check_inputs(
    spec: &TemporalCounterfactualSpec,
    factual: &[&FactualUnitHistory],
) -> Result<(), TemporalCounterfactualError> {
    if factual.is_empty() {
        return Err(unsupported("temporal_counterfactual.shared_history_missing", None, None));
    }
    if factual.len() > MAX_UNITS {
        return Err(invalid(
            "temporal_counterfactual.too_many_units",
            Some(factual.len().to_string()),
        ));
    }
    let names_ok = !spec.plus.name.trim().is_empty()
        && !spec.minus.name.trim().is_empty()
        && spec.plus.name != spec.minus.name;
    if !names_ok {
        return Err(invalid("temporal_counterfactual.history_name_invalid", None));
    }
    for world in [&spec.plus, &spec.minus] {
        if !world.actions.iter().all(|a| a.is_finite()) {
            return Err(invalid(
                "temporal_counterfactual.non_finite_history",
                Some(world.name.clone()),
            ));
        }
    }
    check_pairing(spec, factual)?;
    if let Some(bad) = factual.iter().find(|f| !f.values.iter().all(|v| v.is_finite())) {
        return Err(invalid(
            "temporal_counterfactual.non_finite_history",
            Some(bad.unit.as_str().to_owned()),
        ));
    }
    Ok(())
}

fn digest_graph(graph: &TemporalGraph) -> String {
    let mut d = Digest::new("temporal_cross_world.graph");
    d.word(graph.horizon as u64);
    d.byte(u8::from(graph.latent_confounding));
    for (p, c) in sorted_edges(graph) {
        d.word(p as u64);
        d.word(c as u64);
    }
    d.hex()
}

fn digest_fit(fit: &TemporalMechanismFit) -> String {
    let mut d = Digest::new("temporal_cross_world.fit");
    d.text(&fit.fit_id);
    let mut mechs: Vec<&NodeMechanism> = fit.mechanisms.iter().collect();
    mechs.sort_by_key(|m| m.node.index());
    for mech in mechs {
        d.word(mech.node.index() as u64);
        d.real(mech.intercept);
        let mut coefs = mech.parent_coefficients.clone();
        coefs.sort_by_key(|(p, _)| p.index());
        for (p, c) in coefs {
            d.word(p.index() as u64);
            d.real(c);
        }
        match mech.noise_halfwidth {
            Some(h) => {
                d.byte(1);
                d.real(h);
            }
            None => d.byte(0),
        }
    }
    d.hex()
}

fn digest_factual(factual: &[&FactualUnitHistory]) -> String {
    let mut d = Digest::new("temporal_cross_world.factual");
    d.word(factual.len() as u64);
    for unit in factual {
        d.text(unit.unit.as_str());
        d.text(unit.history.as_str());
        d.word(u64::from(unit.times[0]));
        d.word(u64::from(unit.times[1]));
        for v in unit.values {
            d.real(v);
        }
    }
    d.hex()
}

fn digest_history(history: &NamedActionHistory) -> String {
    let mut d = Digest::new("temporal_cross_world.action_history");
    d.text(&history.name);
    d.word(u64::from(history.times[0]));
    d.word(u64::from(history.times[1]));
    d.real(history.actions[0]);
    d.real(history.actions[1]);
    let mut units: Vec<&UnitId> = history.units.iter().collect();
    units.sort();
    for unit in units {
        d.text(unit.as_str());
    }
    d.hex()
}

fn digest_draw(snapshot: &SnapshotId, unit: &FactualUnitHistory, noise: &[f64; 3]) -> String {
    let mut d = Digest::new("temporal_cross_world.unit_draw");
    d.text(snapshot.as_str());
    d.text(unit.unit.as_str());
    d.text(unit.history.as_str());
    for u in noise {
        d.real(*u);
    }
    d.hex()
}

/// Abduce one unit's exogenous history `[U_L0, U_L1, U_Y]` from its factual values.
fn abduce_unit(
    mechs: &[&NodeMechanism; 3],
    unit: &FactualUnitHistory,
) -> Result<[f64; 3], TemporalCounterfactualError> {
    let mut noise = [0.0; 3];
    for (slot, mech) in mechs.iter().enumerate() {
        let residual = unit.values[mech.node.index()] - node_value(mech, &unit.values);
        let refuted =
            !residual.is_finite() || mech.noise_halfwidth.is_some_and(|h| residual.abs() > h);
        if refuted {
            return Err(unsupported(
                "temporal_counterfactual.refuting_history",
                Some(unit.unit.as_str().to_owned()),
                Some(RefusingWitness {
                    unit: Some(unit.unit.clone()),
                    history: Some(unit.history.clone()),
                    node: Some(mech.node),
                    residual: Some(residual),
                    bound: mech.noise_halfwidth,
                    ..RefusingWitness::empty()
                }),
            ));
        }
        noise[slot] = residual;
    }
    Ok(noise)
}

/// Replay one action history against an abduced exogenous history and return the
/// final outcome.
fn replay(mechs: &[&NodeMechanism; 3], noise: &[f64; 3], actions: &[f64; 2]) -> f64 {
    let mut values = [0.0; 5];
    values[0] = node_value(mechs[0], &values) + noise[0];
    values[1] = actions[0];
    values[2] = node_value(mechs[1], &values) + noise[1];
    values[3] = actions[1];
    node_value(mechs[2], &values) + noise[2]
}

/// Abduce each unit's exogenous history once, then replay both named action
/// histories against that same history.
///
/// Abduction, action and prediction are one operation; the receipt records that
/// both worlds read the same per-unit draw. Cancellation is polled before
/// validation and every 1024 units.
///
/// # Errors
///
/// `route_not_supported` with `temporal_counterfactual.unpaired_histories`
/// (factual and world units do not pair, or a unit is duplicated),
/// `temporal_counterfactual.time_misaligned`, `.shared_history_missing`,
/// `.latent_confounding` or `.refuting_history` (a factual history outside the
/// declared noise support, retaining the unit, history, node, residual and bound
/// in the witness); `invalid_argument` with `temporal_counterfactual.horizon_exceeded`,
/// `.too_many_units`, `.invalid_graph`, `.fit_mismatch`, `.history_name_invalid` or
/// `.non_finite_history`; or [`TemporalCounterfactualError::Cancelled`].
pub fn evaluate_temporal_counterfactual(
    spec: &TemporalCounterfactualSpec,
    ctx: &ExecutionContext,
) -> Result<TemporalCounterfactualResult, TemporalCounterfactualError> {
    poll(ctx)?;
    validate_graph(&spec.graph)?;
    let mechs = compile_fit(&spec.graph, &spec.fit)?;
    let mut factual: Vec<&FactualUnitHistory> = spec.factual.iter().collect();
    factual.sort_by(|a, b| a.unit.cmp(&b.unit));
    check_inputs(spec, &factual)?;

    let mut units = Vec::with_capacity(factual.len());
    let mut unit_draws = Vec::with_capacity(factual.len());
    let (mut sum_plus, mut sum_minus) = (0.0, 0.0);
    for (index, unit) in factual.iter().enumerate() {
        if index % POLL_EVERY == 0 {
            poll(ctx)?;
        }
        let noise = abduce_unit(&mechs, unit)?;
        let plus_outcome = replay(&mechs, &noise, &spec.plus.actions);
        let minus_outcome = replay(&mechs, &noise, &spec.minus.actions);
        sum_plus += plus_outcome;
        sum_minus += minus_outcome;
        unit_draws.push(UnitDrawDigest {
            unit: unit.unit.clone(),
            history: unit.history.clone(),
            digest: digest_draw(&spec.snapshot, unit, &noise),
        });
        units.push(UnitCounterfactual {
            unit: unit.unit.clone(),
            history: unit.history.clone(),
            factual_outcome: unit.values[TemporalNode::Outcome.index()],
            plus_outcome,
            minus_outcome,
        });
    }
    poll(ctx)?;
    let n = units.len() as f64;
    let mut receipt = SharedAbductionReceipt {
        snapshot: spec.snapshot.clone(),
        graph_digest: digest_graph(&spec.graph),
        fit_digest: digest_fit(&spec.fit),
        factual_digest: digest_factual(&factual),
        plus_digest: digest_history(&spec.plus),
        minus_digest: digest_history(&spec.minus),
        exogenous_digest: SharedAbductionReceipt::exogenous_of(&unit_draws),
        unit_draws,
        n_units: units.len(),
        n_worlds: MAX_WORLDS,
        horizon: MAX_HORIZON,
        shared_by_both_worlds: true,
        digest: String::new(),
    };
    receipt.digest = receipt.recompute_digest();
    Ok(TemporalCounterfactualResult {
        mechanism_class: MECHANISM_CLASS,
        units,
        mean_plus: sum_plus / n,
        mean_minus: sum_minus / n,
        mean_contrast: (sum_plus - sum_minus) / n,
        receipt,
    })
}
