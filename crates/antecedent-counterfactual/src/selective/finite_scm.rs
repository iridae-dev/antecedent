//! A fully declared static SCM over finite variables with finite exogenous
//! terms: declaration, validation, exact abduction and world evaluation.
//!
//! # Model
//!
//! Variables `V_0, .., V_{n-1}` are listed in topological order; `V_i` has
//! levels `0..levels_i`, parents with smaller indices and its own exogenous
//! term `U_{e(i)}` with a finite pmf. The mechanism is the table
//! `V_i = f_i(pa_i, U_{e(i)})`, indexed `[u * n_cfg + cfg]` where `cfg` is the
//! mixed-radix index of the parent values (first parent most significant).
//! An exogenous term may feed one variable (a Markovian model) or several
//! (a shared term, the structural form of latent confounding). Either way the
//! model is fully declared, every mechanism and every exogenous law is given,
//! so a counterfactual is a computation over the joint exogenous states and
//! carries no identification question: the shared term takes one value that
//! every mechanism reading it, in every world, sees.
//!
//! # Abduction, action, prediction (exact)
//!
//! For a unit with factual evidence `e`, the exogenous posterior is
//! `P(u | e) = P(u) 1[f(u) is consistent with e] / P(e)` over the product of
//! the exogenous supports (Pearl 2009 ch. 7; Balke and Pearl 1994). Each named
//! world evaluates the same structural equations with its interventions and
//! its cross-world routes, reading every exogenous term from the same
//! posterior state, so the worlds are coupled by construction.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::{CanonicalWriter, ExecutionContext};

use super::refusal::{CrossWorldExecutionError, invalid, poll, refuse};

/// Most endogenous variables.
pub const MAX_VARIABLES: usize = 12;
/// Most levels of a variable.
pub const MAX_LEVELS: usize = 4;
/// Most states of one exogenous term.
pub const MAX_EXOGENOUS_STATES: usize = 8;
/// Most joint exogenous states enumerated per unit.
pub const MAX_JOINT_STATES: usize = 1 << 16;
/// Most units per request.
pub const MAX_UNITS: usize = 1024;
/// Most worlds per request.
pub const MAX_WORLDS: usize = 8;
/// Most `units x joint states` of abduction work per request.
pub const MAX_ABDUCTION_WORK: usize = 4_000_000;

const PMF_TOLERANCE: f64 = 1e-9;
const POLL_EVERY: usize = 1024;

/// A finite exogenous term.
#[derive(Clone, Debug, PartialEq)]
pub struct ExogenousVariable {
    /// Name.
    pub name: String,
    /// Probability of each state.
    pub pmf: Vec<f64>,
}

/// A finite endogenous variable and its mechanism table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FiniteVariable {
    /// Name.
    pub name: String,
    /// Number of levels, `2..=MAX_LEVELS`.
    pub levels: usize,
    /// Parent variable indices (all smaller than this variable's index).
    pub parents: Vec<usize>,
    /// Index of this variable's exogenous term.
    pub exogenous: usize,
    /// Mechanism table `[u * n_cfg + cfg]`.
    pub table: Vec<u8>,
}

/// A static Markovian SCM with finite exogenous terms.
#[derive(Clone, Debug, PartialEq)]
pub struct FiniteScm {
    /// Population the mechanisms are declared for.
    pub population: String,
    /// History or regime the mechanisms are declared for.
    pub history: String,
    /// Exogenous terms.
    pub exogenous: Vec<ExogenousVariable>,
    /// Endogenous variables in topological order.
    pub variables: Vec<FiniteVariable>,
}

/// One unit's factual evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FactualUnit {
    /// Stable unit identifier (the same unit in every world).
    pub unit: String,
    /// Observed `(variable index, level)` pairs; may be empty.
    pub observed: Vec<(usize, u8)>,
}

/// A route that makes one world read a parent from another world.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorldRoute {
    /// Parent variable.
    pub parent: usize,
    /// Child variable.
    pub child: usize,
    /// World the child reads the parent from.
    pub source: usize,
}

/// A named world: hard interventions and cross-world routes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FiniteWorld {
    /// Name.
    pub name: String,
    /// `do(variable = level)` pairs.
    pub interventions: Vec<(usize, u8)>,
    /// Cross-world routes.
    pub routes: Vec<WorldRoute>,
}

/// An observed value of one variable in one world.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorldAtom {
    /// World index.
    pub world: usize,
    /// Variable index.
    pub variable: usize,
    /// Level.
    pub level: u8,
}

/// The requested quantity, averaged over the request's units.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FiniteQuantity {
    /// `P(variable = level in world | evidence)`.
    Probability(WorldAtom),
    /// `E[V_plus in world a | e] - E[V_minus in world b | e]`, level index as
    /// the numeric value.
    MeanContrast {
        /// `(world, variable)` added.
        plus: (usize, usize),
        /// `(world, variable)` subtracted.
        minus: (usize, usize),
    },
    /// The probability that every atom holds at once (a joint event over
    /// worlds).
    JointProbability(Vec<WorldAtom>),
}

/// A validated SCM with its enumeration layout.
#[derive(Debug)]
pub struct CompiledScm<'a> {
    pub(crate) scm: &'a FiniteScm,
    pub(crate) n_cfg: Vec<usize>,
    radix: Vec<usize>,
    /// Validated probability laws, normalized only when the declaration used
    /// the accepted simplex tolerance. The source declaration remains the
    /// input to `FiniteScm::digest`.
    pmfs: Vec<Vec<f64>>,
    /// Joint exogenous states.
    pub joint_states: usize,
    /// Whether an exogenous term feeds more than one variable.
    pub shared_exogenous: bool,
}

fn bounds(what: &str) -> CrossWorldExecutionError {
    refuse(
        antecedent_core::reason_code!("invalid_argument"),
        "cross_world_execution.bounds_exceeded",
        Some(what.to_owned()),
        "reduce the declaration below the documented bound",
    )
}

/// The joint exogenous enumeration is too large to run exactly: the model is
/// declared, but this route does not execute it.
fn joint_state_bound() -> CrossWorldExecutionError {
    refuse(
        antecedent_core::reason_code!("route_not_supported"),
        "cross_world_execution.joint_state_bound_exceeded",
        Some("joint_exogenous_states".to_owned()),
        "reduce the number of exogenous states or terms so their product is within the \
         documented bound; the model is declared, so the counterfactual is computable but not \
         by this bounded enumeration",
    )
}

#[allow(
    clippy::float_cmp,
    reason = "retain the declared PMF bits when its accumulated sum is exactly one"
)]
fn normalized_pmf(pmf: &[f64], sum: f64) -> Vec<f64> {
    if sum == 1.0 { pmf.to_vec() } else { pmf.iter().map(|mass| mass / sum).collect() }
}

fn unique_nonempty<'a>(
    names: impl Iterator<Item = &'a str>,
    what: &str,
) -> Result<(), CrossWorldExecutionError> {
    let mut seen = BTreeSet::new();
    for name in names {
        if name.trim().is_empty() || !seen.insert(name) {
            return Err(invalid("cross_world_execution.invalid_scm", what));
        }
    }
    Ok(())
}

impl FiniteScm {
    /// Validate the declaration and fix its enumeration layout.
    ///
    /// # Errors
    /// `invalid_argument` for a malformed declaration or a declaration bound;
    /// `route_not_supported` (`cross_world_execution.joint_state_bound_exceeded`)
    /// when the joint exogenous enumeration exceeds [`MAX_JOINT_STATES`].
    pub fn compile(&self) -> Result<CompiledScm<'_>, CrossWorldExecutionError> {
        let n = self.variables.len();
        if n > MAX_VARIABLES || self.exogenous.len() > MAX_VARIABLES {
            return Err(bounds("variables"));
        }
        if n == 0 || self.exogenous.is_empty() {
            return Err(invalid("cross_world_execution.invalid_scm", "variables"));
        }
        if self.population.trim().is_empty() || self.history.trim().is_empty() {
            return Err(invalid("cross_world_execution.invalid_scm", "population_or_history"));
        }
        unique_nonempty(self.variables.iter().map(|v| v.name.as_str()), "variable_names")?;
        unique_nonempty(self.exogenous.iter().map(|e| e.name.as_str()), "exogenous_names")?;
        let mut pmfs = Vec::with_capacity(self.exogenous.len());
        for exo in &self.exogenous {
            let sum: f64 = exo.pmf.iter().sum();
            let ok = !exo.pmf.is_empty()
                && exo.pmf.len() <= MAX_EXOGENOUS_STATES
                && exo.pmf.iter().all(|p| p.is_finite() && *p >= 0.0)
                && (sum - 1.0).abs() <= PMF_TOLERANCE;
            if !ok {
                return Err(invalid("cross_world_execution.invalid_scm", exo.name.clone()));
            }
            // Tolerated simplex slack is corrected; exact PMFs retain their bits.
            pmfs.push(normalized_pmf(&exo.pmf, sum));
        }
        let mut users = vec![0usize; self.exogenous.len()];
        let mut n_cfg = Vec::with_capacity(n);
        for (index, var) in self.variables.iter().enumerate() {
            let parents: BTreeSet<usize> = var.parents.iter().copied().collect();
            let well_formed = (MIN_LEVELS..=MAX_LEVELS).contains(&var.levels)
                && parents.len() == var.parents.len()
                && var.parents.iter().all(|p| *p < index)
                && var.exogenous < self.exogenous.len();
            if !well_formed {
                return Err(invalid("cross_world_execution.invalid_scm", var.name.clone()));
            }
            let cfg: usize =
                var.parents.iter().map(|p| self.variables[*p].levels).product::<usize>();
            let size = self.exogenous[var.exogenous].pmf.len();
            let table_ok = var.table.len() == size * cfg
                && var.table.iter().all(|l| usize::from(*l) < var.levels);
            if !table_ok {
                return Err(invalid("cross_world_execution.invalid_scm", var.name.clone()));
            }
            users[var.exogenous] += 1;
            n_cfg.push(cfg);
        }
        if let Some(unused) = users.iter().position(|u| *u == 0) {
            return Err(invalid(
                "cross_world_execution.invalid_scm",
                self.exogenous[unused].name.clone(),
            ));
        }
        let mut joint = 1usize;
        for exo in &self.exogenous {
            joint = joint
                .checked_mul(exo.pmf.len())
                .filter(|j| *j <= MAX_JOINT_STATES)
                .ok_or_else(joint_state_bound)?;
        }
        let mut radix = vec![1usize; self.exogenous.len()];
        for k in (0..self.exogenous.len().saturating_sub(1)).rev() {
            radix[k] = radix[k + 1] * self.exogenous[k + 1].pmf.len();
        }
        let shared_exogenous = users.iter().any(|u| *u > 1);
        Ok(CompiledScm { scm: self, n_cfg, radix, pmfs, joint_states: joint, shared_exogenous })
    }

    /// Canonical digest of the mechanisms: population, history, structure,
    /// tables and exogenous laws (reals by bit pattern).
    #[must_use]
    pub fn digest(&self) -> String {
        let mut w = CanonicalWriter::new("antecedent.counterfactual.selective.finite_scm.v1");
        w.text(&self.population).text(&self.history);
        w.word(self.exogenous.len() as u64);
        for exo in &self.exogenous {
            w.text(&exo.name).reals(&exo.pmf);
        }
        w.word(self.variables.len() as u64);
        for var in &self.variables {
            w.text(&var.name).word(var.levels as u64).word(var.exogenous as u64);
            w.word(var.parents.len() as u64);
            for p in &var.parents {
                w.word(*p as u64);
            }
            w.bytes(&var.table);
        }
        w.finish_hex()
    }
}

const MIN_LEVELS: usize = 2;

impl CompiledScm<'_> {
    fn exogenous_state(&self, joint: usize, exogenous: usize) -> usize {
        (joint / self.radix[exogenous]) % self.scm.exogenous[exogenous].pmf.len()
    }

    /// Decode a joint state into one state per exogenous term.
    #[must_use]
    pub fn decode(&self, joint: usize) -> Vec<usize> {
        (0..self.scm.exogenous.len()).map(|k| self.exogenous_state(joint, k)).collect()
    }

    fn mechanism(&self, variable: usize, u_state: usize, cfg: usize) -> u8 {
        self.scm.variables[variable].table[u_state * self.n_cfg[variable] + cfg]
    }

    fn mass(&self, exogenous: usize, state: usize) -> f64 {
        self.pmfs[exogenous][state]
    }
}

/// Validate the units and return them sorted by id with sorted evidence.
///
/// # Errors
/// An empty set, too many units, a duplicate or blank id, or evidence on an
/// unknown variable or level or repeated variable.
pub fn validate_units(
    scm: &FiniteScm,
    units: &[FactualUnit],
) -> Result<Vec<FactualUnit>, CrossWorldExecutionError> {
    if units.is_empty() || units.len() > MAX_UNITS {
        return Err(bounds("units"));
    }
    let mut sorted: Vec<FactualUnit> = units.to_vec();
    sorted.sort_by(|a, b| a.unit.cmp(&b.unit));
    unique_nonempty(sorted.iter().map(|u| u.unit.as_str()), "unit_ids")?;
    for unit in &mut sorted {
        unit.observed.sort_unstable();
        let mut previous = None;
        for (variable, level) in &unit.observed {
            let known =
                scm.variables.get(*variable).is_some_and(|v| usize::from(*level) < v.levels);
            if !known || previous == Some(*variable) {
                return Err(invalid("cross_world_execution.invalid_evidence", unit.unit.clone()));
            }
            previous = Some(*variable);
        }
    }
    Ok(sorted)
}

/// Validate the worlds against the model.
///
/// # Errors
/// No world, too many, a blank or repeated name, an intervention or route on an
/// unknown variable or level, a route that is not a parent edge, a route into
/// an intervened variable, or a route to a nonexistent or the same world.
pub fn validate_worlds(
    scm: &FiniteScm,
    worlds: &[FiniteWorld],
) -> Result<(), CrossWorldExecutionError> {
    if worlds.is_empty() || worlds.len() > MAX_WORLDS {
        return Err(bounds("worlds"));
    }
    unique_nonempty(worlds.iter().map(|w| w.name.as_str()), "world_names")?;
    for (index, world) in worlds.iter().enumerate() {
        let bad = |what: &str| {
            invalid("cross_world_execution.invalid_world", format!("{}:{what}", world.name))
        };
        let mut set = BTreeSet::new();
        for (variable, level) in &world.interventions {
            let known =
                scm.variables.get(*variable).is_some_and(|v| usize::from(*level) < v.levels);
            if !known || !set.insert(*variable) {
                return Err(bad("intervention"));
            }
        }
        for route in &world.routes {
            let edge =
                scm.variables.get(route.child).is_some_and(|c| c.parents.contains(&route.parent));
            if !edge
                || set.contains(&route.child)
                || route.source >= worlds.len()
                || route.source == index
            {
                return Err(bad("route"));
            }
        }
    }
    Ok(())
}

impl FiniteWorld {
    /// Canonical digest of the world's definition.
    #[must_use]
    pub fn digest(&self) -> String {
        let mut w = CanonicalWriter::new("antecedent.counterfactual.selective.world.v1");
        w.text(&self.name);
        let mut interventions = self.interventions.clone();
        interventions.sort_unstable();
        w.word(interventions.len() as u64);
        for (variable, level) in interventions {
            w.word(variable as u64).word(u64::from(level));
        }
        let mut routes: Vec<(usize, usize, usize)> =
            self.routes.iter().map(|r| (r.parent, r.child, r.source)).collect();
        routes.sort_unstable();
        w.word(routes.len() as u64);
        for (p, c, s) in routes {
            w.word(p as u64).word(c as u64).word(s as u64);
        }
        w.finish_hex()
    }
}

impl FactualUnit {
    /// Canonical digest of the unit's factual evidence (order independent).
    #[must_use]
    pub fn digest(&self) -> String {
        let mut observed = self.observed.clone();
        observed.sort_unstable();
        let mut w = CanonicalWriter::new("antecedent.counterfactual.selective.evidence.v1");
        w.text(&self.unit).word(observed.len() as u64);
        for (variable, level) in observed {
            w.word(variable as u64).word(u64::from(level));
        }
        w.finish_hex()
    }
}

impl FiniteQuantity {
    fn atoms(&self) -> Vec<(usize, usize)> {
        match self {
            Self::Probability(a) => vec![(a.world, a.variable)],
            Self::MeanContrast { plus, minus } => vec![*plus, *minus],
            Self::JointProbability(atoms) => atoms.iter().map(|a| (a.world, a.variable)).collect(),
        }
    }

    /// Check the quantity against the model and the worlds.
    ///
    /// # Errors
    /// An atom on an unknown world, variable or level, or an empty joint event.
    pub fn validate(
        &self,
        scm: &FiniteScm,
        n_worlds: usize,
    ) -> Result<(), CrossWorldExecutionError> {
        let bad = || invalid("cross_world_execution.invalid_quantity", "quantity");
        if matches!(self, Self::JointProbability(atoms) if atoms.is_empty()) {
            return Err(bad());
        }
        if self.atoms().iter().any(|(w, v)| *w >= n_worlds || *v >= scm.variables.len()) {
            return Err(bad());
        }
        let levels_ok = |a: &WorldAtom| usize::from(a.level) < scm.variables[a.variable].levels;
        let ok = match self {
            Self::Probability(a) => levels_ok(a),
            Self::JointProbability(atoms) => atoms.iter().all(levels_ok),
            Self::MeanContrast { .. } => true,
        };
        if ok { Ok(()) } else { Err(bad()) }
    }

    /// Worlds read by the quantity directly or, through routes, indirectly.
    #[must_use]
    pub fn involved_worlds(&self, worlds: &[FiniteWorld]) -> Vec<usize> {
        let mut set: BTreeSet<usize> = self.atoms().iter().map(|(w, _)| *w).collect();
        loop {
            let before = set.len();
            let sources: Vec<usize> =
                set.iter().flat_map(|w| worlds[*w].routes.iter().map(|r| r.source)).collect();
            set.extend(sources);
            if set.len() == before {
                return set.into_iter().collect();
            }
        }
    }

    /// Canonical digest of the quantity.
    #[must_use]
    pub fn digest(&self) -> String {
        let mut w = CanonicalWriter::new("antecedent.counterfactual.selective.quantity.v1");
        match self {
            Self::Probability(a) => {
                w.word(0).word(a.world as u64).word(a.variable as u64).word(u64::from(a.level));
            }
            Self::MeanContrast { plus, minus } => {
                w.word(1);
                w.word(plus.0 as u64).word(plus.1 as u64);
                w.word(minus.0 as u64).word(minus.1 as u64);
            }
            Self::JointProbability(atoms) => {
                w.word(2).word(atoms.len() as u64);
                for a in atoms {
                    w.word(a.world as u64).word(a.variable as u64).word(u64::from(a.level));
                }
            }
        }
        w.finish_hex()
    }
}

/// A unit's exogenous posterior over joint states.
#[derive(Clone, Debug, PartialEq)]
pub struct Posterior {
    /// `(joint state, normalised weight)` in ascending joint-state order.
    pub states: Vec<(u32, f64)>,
    /// `P(evidence)` under the declared mechanisms.
    pub evidence_probability: f64,
}

/// Exact abduction of one unit by enumerating the joint exogenous states.
///
/// # Errors
/// Cancellation, evidence with zero probability under the mechanisms
/// (`route_not_supported`, `cross_world_execution.evidence_outside_model_support`),
/// or a positive mass that underflows the returned `f64`
/// (`route_not_supported`, `cross_world_execution.evidence_probability_underflow`)
/// or makes a matching posterior state unrepresentable
/// (`route_not_supported`, `cross_world_execution.posterior_weight_underflow`).
#[allow(
    clippy::float_cmp,
    reason = "positive PMF products that become exactly zero are tracked separately from declared zero masses"
)]
pub fn abduct_unit(
    compiled: &CompiledScm<'_>,
    unit: &FactualUnit,
    ctx: &ExecutionContext,
) -> Result<Posterior, CrossWorldExecutionError> {
    let scm = compiled.scm;
    let n = scm.variables.len();
    let mut values = vec![0u8; n];
    let mut states: Vec<(u32, f64)> = Vec::new();
    let mut total = 0.0;
    let mut matched_underflow = false;
    for joint in 0..compiled.joint_states {
        if joint % POLL_EVERY == 0 {
            poll(ctx)?;
        }
        let mut weight = 1.0;
        let mut zero_mass = false;
        let mut underflowed = false;
        for k in 0..scm.exogenous.len() {
            let mass = compiled.mass(k, compiled.exogenous_state(joint, k));
            if mass == 0.0 {
                zero_mass = true;
                break;
            }
            weight *= mass;
            underflowed |= weight == 0.0;
        }
        if zero_mass {
            continue;
        }
        for (v, var) in scm.variables.iter().enumerate() {
            let mut cfg = 0usize;
            for p in &var.parents {
                cfg = cfg * scm.variables[*p].levels + usize::from(values[*p]);
            }
            values[v] = compiled.mechanism(v, compiled.exogenous_state(joint, var.exogenous), cfg);
        }
        if unit.observed.iter().all(|(v, l)| values[*v] == *l) {
            if underflowed {
                matched_underflow = true;
            } else {
                states.push((
                    u32::try_from(joint).map_err(|_| bounds("joint_exogenous_states"))?,
                    weight,
                ));
                total += weight;
            }
        }
    }
    if states.is_empty() && matched_underflow {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "cross_world_execution.evidence_probability_underflow",
            Some(unit.unit.clone()),
            "use an executor with probability arithmetic that can retain this positive evidence mass; this f64 route cannot represent it",
        ));
    }
    if states.is_empty() {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "cross_world_execution.evidence_outside_model_support",
            Some(unit.unit.clone()),
            "the factual evidence has probability zero under the declared mechanisms; revise \
             the mechanisms or the evidence (a refuted model is not repaired by conditioning)",
        ));
    }
    // Keep direct-product and direct-sum arithmetic for representable masses
    // so ordinary stored results retain their historical IEEE-754 bits.
    // Fail closed if a matching positive state was lost to underflow: without
    // its mass, neither its evidence probability nor its posterior weight is
    // an honest result.
    if matched_underflow {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "cross_world_execution.posterior_weight_underflow",
            Some(unit.unit.clone()),
            "use an executor with probability arithmetic that can retain this positive posterior state mass; this f64 route cannot represent it",
        ));
    }
    for state in &mut states {
        state.1 /= total;
        let weight = state.1;
        if weight == 0.0 {
            return Err(refuse(
                antecedent_core::reason_code!("route_not_supported"),
                "cross_world_execution.posterior_weight_underflow",
                Some(unit.unit.clone()),
                "rescale the declared model or use a posterior representation that can retain this positive state mass",
            ));
        }
    }
    Ok(Posterior { states, evidence_probability: total })
}

/// Per-world evaluation plan: fixed values and the world each parent is read
/// from.
#[derive(Clone, Debug)]
pub struct WorldPlan {
    fixed: Vec<Option<u8>>,
    parent_source: Vec<Vec<usize>>,
}

impl WorldPlan {
    /// Build the plan of world `index`.
    #[must_use]
    pub fn new(scm: &FiniteScm, worlds: &[FiniteWorld], index: usize) -> Self {
        let world = &worlds[index];
        let mut fixed = vec![None; scm.variables.len()];
        for (variable, level) in &world.interventions {
            fixed[*variable] = Some(*level);
        }
        let parent_source = scm
            .variables
            .iter()
            .enumerate()
            .map(|(child, var)| {
                var.parents
                    .iter()
                    .map(|parent| {
                        world
                            .routes
                            .iter()
                            .find(|r| r.parent == *parent && r.child == child)
                            .map_or(index, |r| r.source)
                    })
                    .collect()
            })
            .collect();
        Self { fixed, parent_source }
    }
}

/// Evaluate every world at one joint exogenous state into `out[world][variable]`.
pub fn evaluate_worlds(
    compiled: &CompiledScm<'_>,
    plans: &[WorldPlan],
    joint: usize,
    out: &mut [Vec<u8>],
) {
    let scm = compiled.scm;
    for (v, var) in scm.variables.iter().enumerate() {
        let u_state = compiled.exogenous_state(joint, var.exogenous);
        for (w, plan) in plans.iter().enumerate() {
            let value = if let Some(level) = plan.fixed[v] {
                level
            } else {
                let mut cfg = 0usize;
                for (slot, parent) in var.parents.iter().enumerate() {
                    let source = plan.parent_source[v][slot];
                    cfg = cfg * scm.variables[*parent].levels + usize::from(out[source][*parent]);
                }
                compiled.mechanism(v, u_state, cfg)
            };
            out[w][v] = value;
        }
    }
}

impl FiniteQuantity {
    /// The quantity's value at one evaluated state (`0.0` or `1.0` for an
    /// event, the contrast of level values for a mean contrast).
    #[must_use]
    pub fn at_state(&self, values: &[Vec<u8>]) -> f64 {
        let hit = |a: &WorldAtom| values[a.world][a.variable] == a.level;
        match self {
            Self::Probability(a) => f64::from(u8::from(hit(a))),
            Self::JointProbability(atoms) => f64::from(u8::from(atoms.iter().all(hit))),
            Self::MeanContrast { plus, minus } => {
                f64::from(values[plus.0][plus.1]) - f64::from(values[minus.0][minus.1])
            }
        }
    }
}
