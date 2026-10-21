// SPDX-License-Identifier: MIT OR Apache-2.0
//! Finite joint disturbance/selection laws for polynomial causal systems.
//! Selection is declared jointly across worlds. Root-search order and numerical
//! initialization never supply a probability or cross-world coupling. Independent
//! training units share a model-parameter candidate, not their exogenous atom.
use antecedent_core::ExecutionContext;
use antecedent_model::{
    equation_system::{Interval, IsolationPolicy, Monomial, PolynomialEquation, PolynomialSystem},
    nonlinear_equilibrium::EquilibriumError,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Enclosure of a nonnegative probability or unnormalized evidence weight.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MassBounds {
    /// Conservative lower bound.
    pub lower: f64,
    /// Conservative upper bound.
    pub upper: f64,
}
fn down(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else if x.is_infinite() {
        f64::MAX
    } else {
        f64::from_bits(x.to_bits() - 1)
    }
}
fn up(x: f64) -> f64 {
    if x.is_infinite() { x } else { f64::from_bits(x.to_bits() + 1) }
}
impl MassBounds {
    fn point(x: f64) -> Self {
        Self { lower: x, upper: x }
    }
    fn add(self, other: Self) -> Self {
        if self.upper == 0.0 {
            return other;
        }
        if other.upper == 0.0 {
            return self;
        }
        Self { lower: down(self.lower + other.lower), upper: up(self.upper + other.upper) }
    }
    fn mul(self, other: Self) -> Self {
        if self.upper == 0.0 || other.upper == 0.0 {
            return Self::point(0.0);
        }
        Self { lower: down(self.lower * other.lower), upper: up(self.upper * other.upper) }
    }
    fn probability(self) -> Self {
        Self { lower: self.lower.clamp(0.0, 1.0), upper: self.upper.clamp(0.0, 1.0) }
    }
    fn ratio(self, total: Self) -> Self {
        if self.upper == 0.0 {
            return Self::point(0.0);
        }
        Self {
            lower: if total.upper > 0.0 { down(self.lower / total.upper) } else { 0.0 },
            upper: if total.lower > 0.0 { up(self.upper / total.lower) } else { 1.0 },
        }
        .probability()
    }
}
fn invalid(message: &str) -> EquilibriumError {
    EquilibriumError::Invalid(message.into())
}
#[allow(
    clippy::float_cmp,
    reason = "the selected maximum is an exact input; only exact x/x equality licenses the outward-bound shortcut"
)]
fn normalized(weights: &[f64]) -> Result<Vec<MassBounds>, EquilibriumError> {
    if weights.is_empty() || weights.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(invalid("finite nonnegative declared weights required"));
    }
    let maximum = weights.iter().copied().fold(0.0_f64, f64::max);
    if maximum == 0.0 {
        return Err(invalid("declared weights have zero total"));
    }
    let scaled: Vec<_> = weights
        .iter()
        .map(|&x| {
            if x == 0.0 || x == maximum {
                MassBounds::point(x / maximum)
            } else {
                MassBounds { lower: down(x / maximum), upper: up(x / maximum) }
            }
        })
        .collect();
    let total = scaled.iter().copied().fold(MassBounds::point(0.0), MassBounds::add);
    Ok(scaled.into_iter().map(|x| x.ratio(total)).collect())
}

/// Replace a residual equation, with explicit handling of its original disturbance.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolynomialReplacement {
    /// Stable equation identity, preserved by the replacement.
    pub equation: String,
    /// New residual equation with the same identity.
    pub replacement: PolynomialEquation,
    /// True for a hard equation replacement that removes the old disturbance.
    pub clear_disturbance: bool,
}
/// Replace one named residual equation by an exact endogenous clamp.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolynomialClamp {
    /// Equation to replace; different equations can yield different effects.
    pub equation: String,
    /// Endogenous coordinate to fix.
    pub variable: usize,
    /// Fixed value.
    pub value: f64,
}
/// Add to a named residual equation's exogenous disturbance.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolynomialShock {
    /// Stable equation identity.
    pub equation: String,
    /// Additive structural shock.
    pub delta: f64,
}
/// Named structural world, distinct from an observation/conditioning event.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolynomialWorld {
    /// Stable prespecified world id.
    pub id: String,
    /// General residual-equation replacements.
    pub replacements: Vec<PolynomialReplacement>,
    /// Hard equation-specific clamps, removing the replaced disturbance.
    pub clamps: Vec<PolynomialClamp>,
    /// Additive equation disturbances after mechanism changes.
    pub shocks: Vec<PolynomialShock>,
}
/// One finite joint exogenous state, shared across every world of a unit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolynomialExogenousAtom {
    /// Stable atom id within its parameter candidate.
    pub id: String,
    /// Declared nonnegative relative probability; normalized within the candidate.
    pub weight: f64,
    /// Additive disturbance of each residual equation, jointly declared.
    pub disturbances: Vec<f64>,
    /// One geometric selected-root region for every named world. This is the
    /// explicit cross-world selection law, not an index into solver roots.
    pub selections: BTreeMap<String, Vec<Interval>>,
    /// Optional noisy categorical measurement kernels, world -> label -> relative
    /// probability. A missing kernel is allowed only when no label is observed.
    pub measurements: BTreeMap<String, BTreeMap<String, f64>>,
}
/// Finite parameter/model prior, with a conditional joint exogenous law.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolynomialCausalCandidate {
    /// Stable candidate/parameter identity.
    pub id: String,
    /// Named residual model and finite solution domain.
    pub model: PolynomialSystem,
    /// Declared nonnegative relative model-prior probability.
    pub prior_weight: f64,
    /// Conditional exogenous disturbance/selection law.
    pub atoms: Vec<PolynomialExogenousAtom>,
}
/// Partial state evidence with declared finite measurement resolution and/or
/// a noisy categorical measurement. Units are conditionally independent given
/// the parameter candidate and each receives its own exogenous atom.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolynomialObservation {
    /// World in which this unit was observed, including known interventions.
    pub world: String,
    /// None is unobserved. Bounds are an actual observation event, not solver tolerance.
    pub bounds: Vec<Option<Interval>>,
    /// Observed category of the explicitly supplied measurement kernel.
    pub label: Option<String>,
}
/// Complete finite polynomial abduction/action/prediction and parameter fitting program.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolynomialCausalProgram {
    /// Prespecified finite parameter candidates with declared priors.
    pub candidates: Vec<PolynomialCausalCandidate>,
    /// All observational/intervention worlds used by training and prediction.
    pub worlds: Vec<PolynomialWorld>,
    /// Independent training units, updating the shared model parameter.
    pub training: Vec<PolynomialObservation>,
    /// Factual evidence for the unit whose cross-world outcomes are requested.
    pub factual: PolynomialObservation,
    /// Root-isolation policy, never an equilibrium-selection law.
    pub policy: IsolationPolicy,
}
/// Result of applying a declared selection region to a bounded solution set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum PolynomialSelectedState {
    /// The declared region contains exactly one certified root.
    Selected {
        /// Certified enclosure of that selected state.
        enclosure: Vec<Interval>,
    },
    /// A complete root search proves that the selection region has no solution.
    Undefined {
        /// Distinguishes a system with no roots from an empty selected branch.
        reason: String,
    },
    /// Root/region work or branch uniqueness remains unresolved.
    Unresolved {
        /// Number of unresolved root boxes that intersect the selected region.
        intersecting_unresolved_boxes: usize,
        /// Number of certified roots intersecting the selected region.
        intersecting_roots: usize,
    },
}
/// One posterior exogenous/parameter atom, preserving all response states.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolynomialPosteriorAtom {
    /// Parameter candidate id.
    pub candidate: String,
    /// Exogenous state id.
    pub atom: String,
    /// Conditional mass enclosure, absent when evidence is proven impossible.
    pub posterior_mass: Option<MassBounds>,
    /// Cross-world states coupled by this very same atom.
    pub states: BTreeMap<String, PolynomialSelectedState>,
}
/// World-specific undefined/unresolved posterior mass, never discarded.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolynomialWorldMass {
    /// Named world id.
    pub world: String,
    /// Mass with a certified selected state.
    pub selected: MassBounds,
    /// Mass whose declared selection has no solution.
    pub undefined: MassBounds,
    /// Mass whose selected state remains unresolved.
    pub unresolved: MassBounds,
}
/// Finite joint posterior conditional on declared model/disturbance/selection
/// and measurement laws, with outward-rounded mass enclosures.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolynomialCausalReport {
    /// True only when the observed evidence has a positive certified probability.
    pub positive_evidence_certified: bool,
    /// True when upper evidence is zero; this does not declare non-identification.
    pub impossible_evidence: bool,
    /// Joint training/factual evidence enclosure under the declared finite prior.
    pub evidence: MassBounds,
    /// Parameter/exogenous posterior and aligned world outcomes.
    pub atoms: Vec<PolynomialPosteriorAtom>,
    /// Parameter posterior, summed over every exogenous state without dropping failures.
    pub parameter_posterior: BTreeMap<String, MassBounds>,
    /// Distinct selected/undefined/unresolved mass in every world.
    pub world_mass: Vec<PolynomialWorldMass>,
}
fn overlaps(a: &[Interval], b: &[Interval]) -> bool {
    a.iter().zip(b).all(|(a, b)| a.lower <= b.upper && b.lower <= a.upper)
}
fn contains(a: &[Interval], b: &[Interval]) -> bool {
    a.iter().zip(b).all(|(a, b)| a.lower <= b.lower && b.upper <= a.upper)
}
fn equation_index(model: &PolynomialSystem, id: &str) -> Result<usize, EquilibriumError> {
    model
        .equations
        .iter()
        .position(|e| e.id == id)
        .ok_or_else(|| invalid("unknown polynomial equation identity"))
}
fn selected_state(
    candidate: &PolynomialCausalCandidate,
    atom: &PolynomialExogenousAtom,
    world: &PolynomialWorld,
    policy: IsolationPolicy,
    ctx: &ExecutionContext,
) -> Result<PolynomialSelectedState, EquilibriumError> {
    let mut model = candidate.model.clone();
    let mut disturbance = atom.disturbances.clone();
    let mut touched = BTreeSet::new();
    for replacement in &world.replacements {
        let i = equation_index(&model, &replacement.equation)?;
        if replacement.replacement.id != replacement.equation || !touched.insert(i) {
            return Err(invalid("replacement equation identity/conflict"));
        }
        model.equations[i] = replacement.replacement.clone();
        if replacement.clear_disturbance {
            disturbance[i] = 0.0;
        }
    }
    for clamp in &world.clamps {
        let i = equation_index(&model, &clamp.equation)?;
        if !touched.insert(i) {
            return Err(invalid("conflicting equation clamp"));
        }
        model = model.replace_equation(&clamp.equation, clamp.variable, clamp.value)?;
        disturbance[i] = 0.0;
    }
    let mut shocks = BTreeSet::new();
    for shock in &world.shocks {
        let i = equation_index(&model, &shock.equation)?;
        if !shock.delta.is_finite()
            || !shocks.insert(i)
            || world.clamps.iter().any(|c| c.equation == shock.equation)
        {
            return Err(invalid("invalid/conflicting equation shock"));
        }
        disturbance[i] += shock.delta;
    }
    for (equation, &noise) in model.equations.iter_mut().zip(&disturbance) {
        if !noise.is_finite() {
            return Err(invalid("nonfinite disturbed residual"));
        }
        if noise != 0.0 {
            equation
                .terms
                .push(Monomial { coefficient: -noise, powers: vec![0; model.variables.len()] });
        }
    }
    let roots = model.isolate(policy, ctx)?;
    let region = &atom.selections[&world.id];
    let intersecting: Vec<_> = roots.roots.iter().filter(|root| overlaps(region, root)).collect();
    let unresolved = roots.unresolved.iter().filter(|root| overlaps(region, root)).count();
    if unresolved == 0 && intersecting.len() == 1 && contains(region, intersecting[0]) {
        Ok(PolynomialSelectedState::Selected { enclosure: intersecting[0].clone() })
    } else if unresolved == 0 && intersecting.is_empty() {
        Ok(PolynomialSelectedState::Undefined {
            reason: if roots.roots.is_empty() && roots.unresolved.is_empty() {
                "no_solution_in_declared_domain"
            } else {
                "no_solution_in_declared_selection_region"
            }
            .into(),
        })
    } else {
        Ok(PolynomialSelectedState::Unresolved {
            intersecting_unresolved_boxes: unresolved,
            intersecting_roots: intersecting.len(),
        })
    }
}
fn observation_likelihood(
    observation: &PolynomialObservation,
    state: &PolynomialSelectedState,
    atom: &PolynomialExogenousAtom,
) -> Result<MassBounds, EquilibriumError> {
    let empty = observation.bounds.iter().all(Option::is_none) && observation.label.is_none();
    let mut likelihood = match state {
        PolynomialSelectedState::Selected { enclosure } => {
            if observation.bounds.iter().zip(enclosure).any(|(observed, value)| {
                observed.is_some_and(|b| b.upper < value.lower || b.lower > value.upper)
            }) {
                MassBounds::point(0.0)
            } else if observation.bounds.iter().zip(enclosure).all(|(observed, value)| {
                observed.is_none_or(|b| b.lower <= value.lower && b.upper >= value.upper)
            }) {
                MassBounds::point(1.0)
            } else {
                MassBounds { lower: 0.0, upper: 1.0 }
            }
        }
        PolynomialSelectedState::Undefined { .. } => {
            if empty {
                MassBounds::point(1.0)
            } else {
                MassBounds::point(0.0)
            }
        }
        PolynomialSelectedState::Unresolved { .. } => {
            if empty {
                MassBounds::point(1.0)
            } else {
                MassBounds { lower: 0.0, upper: 1.0 }
            }
        }
    };
    if let Some(label) = &observation.label {
        let kernel = atom
            .measurements
            .get(&observation.world)
            .ok_or_else(|| invalid("no declared measurement kernel for observed label"))?;
        let weights: Vec<_> = kernel.values().copied().collect();
        let probabilities = normalized(&weights)?;
        let probability = kernel
            .keys()
            .position(|key| key == label)
            .map_or(MassBounds::point(0.0), |i| probabilities[i]);
        likelihood = likelihood.mul(probability);
    }
    Ok(likelihood.probability())
}
impl PolynomialCausalProgram {
    /// Validate identities, bounded finite laws and every observation/world binding.
    pub fn validate(&self) -> Result<(), EquilibriumError> {
        if self.candidates.is_empty()
            || self.candidates.len() > 32
            || self.worlds.is_empty()
            || self.worlds.len() > 32
            || self.training.len() > 1000
        {
            return Err(invalid("finite causal program limits"));
        }
        let n = self.candidates[0].model.variables.len();
        let mut worlds = BTreeSet::new();
        for world in &self.worlds {
            if world.id.trim().is_empty()
                || world.id.len() > 256
                || !worlds.insert(&world.id)
                || world.clamps.len() + world.replacements.len() > n
                || world.shocks.len() > n
            {
                return Err(invalid("polynomial world identities/action limits"));
            }
        }
        let mut ids = BTreeSet::new();
        let mut count = 0usize;
        for candidate in &self.candidates {
            candidate.model.validate()?;
            if candidate.model.variables != self.candidates[0].model.variables
                || candidate.id.trim().is_empty()
                || candidate.id.len() > 256
                || !ids.insert(&candidate.id)
                || candidate.atoms.is_empty()
                || candidate.atoms.len() > 256
            {
                return Err(invalid("candidate identities/variables/atom limits"));
            }
            let mut atom_ids = BTreeSet::new();
            for atom in &candidate.atoms {
                count += 1;
                if atom.id.trim().is_empty()
                    || atom.id.len() > 256
                    || !atom_ids.insert(&atom.id)
                    || atom.disturbances.len() != n
                    || atom.disturbances.iter().any(|x| !x.is_finite())
                    || atom.selections.len() != self.worlds.len()
                {
                    return Err(invalid("exogenous identity/disturbance/selection shape"));
                }
                for world in &self.worlds {
                    let region = atom
                        .selections
                        .get(&world.id)
                        .ok_or_else(|| invalid("joint selection law must name every world"))?;
                    if region.len() != n
                        || region.iter().any(|b| {
                            !b.lower.is_finite() || !b.upper.is_finite() || b.lower >= b.upper
                        })
                    {
                        return Err(invalid("finite nonempty selection regions required"));
                    }
                }
                if atom.measurements.len() > self.worlds.len() {
                    return Err(invalid("measurement kernel world limit"));
                }
                for (world, kernel) in &atom.measurements {
                    if !worlds.contains(world)
                        || kernel.is_empty()
                        || kernel.len() > 64
                        || kernel.keys().any(|k| k.is_empty() || k.len() > 256)
                    {
                        return Err(invalid("measurement kernel bindings/labels"));
                    }
                    normalized(&kernel.values().copied().collect::<Vec<_>>())?;
                }
            }
            normalized(&candidate.atoms.iter().map(|a| a.weight).collect::<Vec<_>>())?;
        }
        if count > 256 || count * self.worlds.len() > 1024 {
            return Err(invalid("joint root-execution budget"));
        }
        normalized(&self.candidates.iter().map(|c| c.prior_weight).collect::<Vec<_>>())?;
        for observation in self.training.iter().chain(std::iter::once(&self.factual)) {
            if !worlds.contains(&observation.world)
                || observation.bounds.len() != n
                || observation
                    .bounds
                    .iter()
                    .flatten()
                    .any(|b| !b.lower.is_finite() || !b.upper.is_finite() || b.lower > b.upper)
                || observation.label.as_ref().is_some_and(|v| v.is_empty() || v.len() > 256)
            {
                return Err(invalid("observation world/state/label binding"));
            }
        }
        Ok(())
    }
    /// Fit the finite parameter prior, abduct the query unit's joint exogenous
    /// state, and predict every world without replacing its selection coupling.
    /// Interval uncertainty and undefined outcomes retain their original mass.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<PolynomialCausalReport, EquilibriumError> {
        self.validate()?;
        let count = self.candidates.iter().map(|c| c.atoms.len()).sum::<usize>();
        let _retention = ctx
            .acquire_admission(
                0,
                (count * self.worlds.len() * (self.candidates[0].model.variables.len() * 128 + 512)
                    + self.training.len() * 128) as u64,
            )
            .map_err(|e| EquilibriumError::Resource(e.to_string()))?;
        let priors =
            normalized(&self.candidates.iter().map(|c| c.prior_weight).collect::<Vec<_>>())?;
        let mut output = Vec::with_capacity(count);
        let mut weights = Vec::with_capacity(count);
        for (candidate, prior) in self.candidates.iter().zip(priors) {
            let probabilities =
                normalized(&candidate.atoms.iter().map(|a| a.weight).collect::<Vec<_>>())?;
            let mut states = Vec::with_capacity(candidate.atoms.len());
            for atom in &candidate.atoms {
                if ctx.cancellation.is_cancelled() {
                    return Err(EquilibriumError::Resource("cancelled".into()));
                }
                let mut worlds = BTreeMap::new();
                for world in &self.worlds {
                    worlds.insert(
                        world.id.clone(),
                        selected_state(candidate, atom, world, self.policy, ctx)?,
                    );
                }
                states.push(worlds);
            }
            let mut training = MassBounds::point(1.0);
            for observation in &self.training {
                let mut likelihood = MassBounds::point(0.0);
                for ((atom, states), probability) in
                    candidate.atoms.iter().zip(&states).zip(&probabilities)
                {
                    likelihood = likelihood.add(probability.mul(observation_likelihood(
                        observation,
                        &states[&observation.world],
                        atom,
                    )?));
                }
                training = training.mul(likelihood.probability());
            }
            for ((atom, states), probability) in
                candidate.atoms.iter().zip(states).zip(probabilities)
            {
                let factual =
                    observation_likelihood(&self.factual, &states[&self.factual.world], atom)?;
                weights.push(prior.mul(training).mul(probability).mul(factual));
                output.push(PolynomialPosteriorAtom {
                    candidate: candidate.id.clone(),
                    atom: atom.id.clone(),
                    posterior_mass: None,
                    states,
                });
            }
        }
        let evidence =
            weights.iter().copied().fold(MassBounds::point(0.0), MassBounds::add).probability();
        let mut parameter_posterior = BTreeMap::new();
        let mut world_mass: Vec<_> = self
            .worlds
            .iter()
            .map(|world| PolynomialWorldMass {
                world: world.id.clone(),
                selected: MassBounds::point(0.0),
                undefined: MassBounds::point(0.0),
                unresolved: MassBounds::point(0.0),
            })
            .collect();
        if evidence.upper > 0.0 {
            for (atom, weight) in output.iter_mut().zip(weights) {
                let mass = weight.ratio(evidence);
                atom.posterior_mass = Some(mass);
                let entry = parameter_posterior
                    .entry(atom.candidate.clone())
                    .or_insert(MassBounds::point(0.0));
                *entry = entry.add(mass).probability();
                for world in &mut world_mass {
                    let target = match atom.states[&world.world] {
                        PolynomialSelectedState::Selected { .. } => &mut world.selected,
                        PolynomialSelectedState::Undefined { .. } => &mut world.undefined,
                        PolynomialSelectedState::Unresolved { .. } => &mut world.unresolved,
                    };
                    *target = target.add(mass).probability();
                }
            }
        }
        Ok(PolynomialCausalReport {
            positive_evidence_certified: evidence.lower > 0.0,
            impossible_evidence: evidence.upper == 0.0,
            evidence,
            atoms: output,
            parameter_posterior,
            world_mass,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn region(lower: f64, upper: f64) -> Vec<Interval> {
        vec![Interval { lower, upper }]
    }
    fn example(flipped: bool) -> PolynomialCausalProgram {
        let model = PolynomialSystem {
            variables: vec!["x".into()],
            equations: vec![PolynomialEquation {
                id: "balance".into(),
                terms: vec![
                    Monomial { coefficient: 1.0, powers: vec![3] },
                    Monomial { coefficient: -1.0, powers: vec![1] },
                ],
            }],
            domain: region(-3.0, 3.0),
        };
        let worlds = vec![
            PolynomialWorld {
                id: "factual".into(),
                replacements: vec![],
                clamps: vec![],
                shocks: vec![],
            },
            PolynomialWorld {
                id: "counterfactual".into(),
                replacements: vec![PolynomialReplacement {
                    equation: "balance".into(),
                    replacement: PolynomialEquation {
                        id: "balance".into(),
                        terms: vec![
                            Monomial { coefficient: 1.0, powers: vec![2] },
                            Monomial { coefficient: -4.0, powers: vec![0] },
                        ],
                    },
                    clear_disturbance: true,
                }],
                clamps: vec![],
                shocks: vec![],
            },
        ];
        let mut atoms = Vec::new();
        for positive in [true, false] {
            let selected_positive = positive != flipped;
            atoms.push(PolynomialExogenousAtom {
                id: if positive { "positive" } else { "negative" }.into(),
                weight: 1.0,
                disturbances: vec![0.0],
                selections: BTreeMap::from([
                    (
                        "factual".into(),
                        if positive { region(0.5, 1.5) } else { region(-1.5, -0.5) },
                    ),
                    (
                        "counterfactual".into(),
                        if selected_positive { region(1.5, 2.5) } else { region(-2.5, -1.5) },
                    ),
                ]),
                measurements: BTreeMap::from([(
                    "factual".into(),
                    BTreeMap::from([
                        ("positive_sensor".into(), if positive { 4.0 } else { 1.0 }),
                        ("negative_sensor".into(), if positive { 1.0 } else { 4.0 }),
                    ]),
                )]),
            });
        }
        PolynomialCausalProgram {
            candidates: vec![PolynomialCausalCandidate {
                id: "declared".into(),
                model,
                prior_weight: 1.0,
                atoms,
            }],
            worlds,
            training: vec![],
            factual: PolynomialObservation {
                world: "factual".into(),
                bounds: vec![None],
                label: Some("positive_sensor".into()),
            },
            policy: IsolationPolicy::default(),
        }
    }
    fn encloses(bounds: MassBounds, value: f64) {
        assert!(
            bounds.lower <= value && bounds.upper >= value,
            "{bounds:?} does not enclose {value}"
        );
    }
    #[test]
    fn declared_cross_world_selection_changes_counterfactual_but_not_marginals() {
        for flipped in [false, true] {
            let program = example(flipped);
            let result = program.execute(&ExecutionContext::for_tests(1)).unwrap();
            assert!(result.positive_evidence_certified);
            encloses(result.evidence, 0.5);
            encloses(result.atoms[0].posterior_mass.unwrap(), 0.8);
            encloses(result.atoms[1].posterior_mass.unwrap(), 0.2);
            let mut expected = 0.0;
            for atom in &result.atoms {
                let PolynomialSelectedState::Selected { enclosure } =
                    &atom.states["counterfactual"]
                else {
                    panic!("selected root expected")
                };
                let mass = atom.posterior_mass.unwrap();
                expected += 0.5
                    * (mass.lower + mass.upper)
                    * 0.5
                    * (enclosure[0].lower + enclosure[0].upper);
            }
            assert!((expected - if flipped { -1.2 } else { 1.2 }).abs() < 1e-7, "{expected}");
            // Without factual evidence, both couplings have the identical target marginal.
            let mut marginal = program;
            marginal.factual.label = None;
            let unconditioned = marginal.execute(&ExecutionContext::for_tests(2)).unwrap();
            encloses(unconditioned.atoms[0].posterior_mass.unwrap(), 0.5);
            encloses(unconditioned.atoms[1].posterior_mass.unwrap(), 0.5);
        }
    }
    #[test]
    fn finite_parameter_fitting_uses_independent_training_units_and_keeps_failure_mass() {
        let mut program = example(false);
        program.factual.label = None;
        let mut other = program.candidates[0].clone();
        other.id = "alternative".into();
        other.model.equations[0].terms[1].coefficient = -4.0;
        for atom in &mut other.atoms {
            atom.selections.insert(
                "factual".into(),
                if atom.id == "positive" { region(1.5, 2.5) } else { region(-2.5, -1.5) },
            );
        }
        for atom in &mut program.candidates[0].atoms {
            atom.measurements.get_mut("factual").unwrap().insert("positive_sensor".into(), 4.0);
            atom.measurements.get_mut("factual").unwrap().insert("negative_sensor".into(), 1.0);
        }
        for atom in &mut other.atoms {
            atom.measurements.get_mut("factual").unwrap().insert("positive_sensor".into(), 1.0);
            atom.measurements.get_mut("factual").unwrap().insert("negative_sensor".into(), 4.0);
        }
        program.candidates.push(other);
        program.training = vec![
            PolynomialObservation {
                world: "factual".into(),
                bounds: vec![None],
                label: Some("positive_sensor".into())
            };
            3
        ];
        // Intervention makes the second world unsolvable; its mass is not dropped.
        program.worlds[1].replacements[0].replacement.terms[1].coefficient = 1.0;
        let result = program.execute(&ExecutionContext::for_tests(3)).unwrap();
        encloses(result.parameter_posterior["declared"], 64.0 / 65.0);
        encloses(result.parameter_posterior["alternative"], 1.0 / 65.0);
        encloses(result.world_mass[1].undefined, 1.0);
        assert_eq!(result.world_mass[1].selected.upper.to_bits(), 0.0_f64.to_bits());
        assert!(result.atoms.iter().all(|a|matches!(&a.states["counterfactual"],PolynomialSelectedState::Undefined{reason} if reason=="no_solution_in_declared_domain")));
    }
    #[test]
    fn unresolved_selection_and_impossible_evidence_are_distinct() {
        let mut program = example(false);
        program.policy.max_boxes = 1;
        let unresolved = program.execute(&ExecutionContext::for_tests(4)).unwrap();
        assert!(!unresolved.impossible_evidence);
        assert!(!unresolved.positive_evidence_certified);
        assert!(unresolved.world_mass.iter().all(|w| w.unresolved.upper > 0.0));
        program.policy = IsolationPolicy::default();
        program.factual.label = None;
        program.factual.bounds = vec![Some(Interval { lower: 2.7, upper: 2.9 })];
        let impossible = program.execute(&ExecutionContext::for_tests(5)).unwrap();
        assert!(impossible.impossible_evidence);
        assert!(impossible.atoms.iter().all(|atom| atom.posterior_mass.is_none()));
    }
}
