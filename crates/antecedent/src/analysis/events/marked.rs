//! Finite-rank Hawkes kernel on declared mark prototypes.
//!
//! The intensity is linear in the baseline and excitation coefficients once the
//! decays and mark features are fixed. The spectral radius of the branching
//! operator is not linear in those coefficients, so the fit does not project
//! onto `ρ < 1`. It projects onto a half-space: no prototype the declaration
//! allows produces more than `reproduction_cap` expected direct children over
//! the whole future. That is sufficient for subcriticality and is generally
//! stricter than `ρ < 1`, the same role the source-column cap plays for the
//! catalog model. Declared parameters are checked with the exact prototype
//! branching matrix.

use antecedent_core::ExecutionContext;
use serde::{Deserialize, Serialize};

use super::hawkes::{self, horizon_factor};
use super::{EventError, EventWindow, MAX_EVENT_TYPES, OPERATION_BUDGET};

/// Largest prototype set whose spectral witness (`P³`) fits in half the operation budget.
pub const MAX_PROTOTYPES: usize = max_prototypes(OPERATION_BUDGET);

const fn max_prototypes(budget: usize) -> usize {
    let allowance = budget / 2;
    let mut count = 1usize;
    loop {
        let next = count + 1;
        if next > 10_000 {
            return count;
        }
        let cost = next.saturating_mul(next).saturating_mul(next);
        if cost > allowance {
            return count;
        }
        count = next;
    }
}

/// One atom of the mark space, carrying the embedding's source and target features.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarkPrototype {
    /// Nonnegative source features `v_k(x)`. Length is the kernel rank.
    pub source: Vec<f64>,
    /// Nonnegative target features `φ_q(x)`. Length is the intensity basis.
    pub target: Vec<f64>,
    /// Reference mass of this atom. The catalog model uses mass one and one-hot features.
    pub mass: f64,
}

/// One arrival identified with a declared prototype.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarkedArrival {
    /// Timestamp in the history's clock.
    pub time: f64,
    /// Index into the history's prototypes.
    pub prototype: usize,
}

/// Complete marked history. Prototypes are the quadrature, or the atoms, of the mark space.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarkedHistory {
    /// Unit of timestamp and decay.
    pub time_unit: String,
    /// Named origin for absolute time.
    pub origin: String,
    /// Stable source of the stream.
    pub source: String,
    /// Excitation is declared empty here.
    pub empty_origin: f64,
    /// Complete observation window.
    pub window: EventWindow,
    /// Arrivals between the empty origin and the window start.
    pub left: Vec<MarkedArrival>,
    /// Arrivals in the observation window.
    pub events: Vec<MarkedArrival>,
    /// Mark atoms in fixed order.
    pub prototypes: Vec<MarkPrototype>,
}

/// Finite-rank kernel. Excitation is row-major, basis by rank.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarkedParameters {
    /// Baseline coefficient of each target feature.
    pub baseline: Vec<f64>,
    /// Excitation of target feature `q` by component `k`, at `q * rank + k`.
    pub excitation: Vec<f64>,
    /// Fixed positive decays, one per component.
    pub decay: Vec<f64>,
}

/// Fixed decays and the sufficient reproduction cap used by the fit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarkedFitSpec {
    /// Declared inverse-time rates, one per component.
    pub decay: Vec<f64>,
    /// Maximum lifetime direct children of any allowed mark, strictly inside `(0, 1)`.
    pub reproduction_cap: f64,
    /// Maximum projected-gradient iterations.
    pub iterations: usize,
}

/// Diagnostics for a converged marked fit.
#[derive(Clone, Debug, Serialize)]
pub struct MarkedFit {
    /// Fitted coefficients and the decays the fit held fixed.
    pub parameters: MarkedParameters,
    /// Prototypes the coefficients were fitted against.
    pub prototypes: Vec<MarkPrototype>,
    /// Conditional log likelihood.
    pub log_likelihood: f64,
    /// Projected-gradient iterations used.
    pub iterations: usize,
    /// Stationarity residual.
    pub projected_residual: f64,
    /// Worst-prototype lifetime reproduction number.
    pub reproduction: f64,
    /// Whether the reproduction cap is active.
    pub cap_active: bool,
    /// Excitation coordinates at zero, in row-major order.
    pub zero_coefficients: Vec<usize>,
    /// Singular observed information.
    pub weak_identification: bool,
}

/// Conditional-intensity offspring of one prototype. Not an intervention effect.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MarkedOffspring {
    /// Initiating prototype.
    pub prototype: usize,
    /// Horizon in the history's time unit.
    pub horizon: f64,
    /// Direct children of each prototype on `[0, horizon)`.
    pub direct: Vec<f64>,
    /// Lifetime direct children of each prototype.
    pub lifetime_direct: Vec<f64>,
    /// All-generation descendants on `[0, horizon)`, from the filter ODE.
    pub horizon_descendants: Vec<f64>,
    /// Lifetime descendants, the column of `(I - B)^{-1} - I`.
    pub lifetime_descendants: Vec<f64>,
}

impl MarkedHistory {
    pub(super) fn validate(&self) -> Result<(), EventError> {
        self.window.validate()?;
        let valid_name = |name: &str| !name.trim().is_empty() && name.len() <= 256;
        let prototypes = self.prototypes.len();
        if !(1..=MAX_PROTOTYPES).contains(&prototypes)
            || [&self.time_unit, &self.origin, &self.source].iter().any(|name| !valid_name(name))
            || self.left.len().saturating_add(self.events.len()) > super::MAX_EVENTS
            || !self.empty_origin.is_finite()
            || self.empty_origin > self.window.start
        {
            return Err(EventError::invalid("invalid marked history"));
        }
        let rank = self.prototypes[0].source.len();
        let basis = self.prototypes[0].target.len();
        if !(1..=MAX_EVENT_TYPES).contains(&rank) || !(1..=MAX_EVENT_TYPES).contains(&basis) {
            return Err(EventError::invalid(
                "marked rank or basis exceeds the dense-feature allowance",
            ));
        }
        for prototype in &self.prototypes {
            if prototype.source.len() != rank
                || prototype.target.len() != basis
                || !prototype.mass.is_finite()
                || prototype.mass <= 0.0
                || prototype.source.iter().any(|value| !value.is_finite() || *value < 0.0)
                || prototype.target.iter().any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(EventError::invalid("invalid mark prototype"));
            }
        }
        check_arrivals(&self.left, self.empty_origin, self.window.start, prototypes)?;
        check_arrivals(&self.events, self.window.start, self.window.end, prototypes)
    }
}

#[allow(clippy::float_cmp, reason = "tied marked arrivals are exact timestamp matches")]
fn check_arrivals(
    arrivals: &[MarkedArrival],
    start: f64,
    end: f64,
    prototypes: usize,
) -> Result<(), EventError> {
    if arrivals.iter().any(|event| {
        !event.time.is_finite()
            || event.time < start
            || event.time >= end
            || event.prototype >= prototypes
    }) || arrivals.windows(2).any(|pair| pair[0].time >= pair[1].time)
    {
        return Err(EventError::invalid("out-of-window, unordered or tied marked arrivals"));
    }
    Ok(())
}

impl MarkedParameters {
    pub(super) fn validate(&self, prototypes: &[MarkPrototype]) -> Result<(), EventError> {
        let rank = self.decay.len();
        let basis = self.baseline.len();
        if prototypes.is_empty()
            || self.excitation.len() != basis * rank
            || prototypes
                .iter()
                .any(|prototype| prototype.source.len() != rank || prototype.target.len() != basis)
            || self
                .baseline
                .iter()
                .chain(&self.excitation)
                .any(|value| !value.is_finite() || *value < 0.0)
            || self.decay.iter().any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(EventError::invalid("invalid marked Hawkes coefficients"));
        }
        let matrix = branching_matrix(self, prototypes);
        spectral_radius_below_one(&matrix, prototypes.len())
    }

    /// Offspring of one prototype under this conditional intensity.
    ///
    /// # Errors
    /// The kernel is not a stable model on these prototypes, or the horizon
    /// cannot be integrated.
    pub fn excitation_offspring(
        &self,
        prototypes: &[MarkPrototype],
        prototype: usize,
        horizon: f64,
    ) -> Result<MarkedOffspring, EventError> {
        self.validate(prototypes)?;
        if prototype >= prototypes.len() || !horizon.is_finite() || horizon <= 0.0 {
            return Err(EventError::invalid(
                "marked offspring query needs a prototype and a positive finite horizon",
            ));
        }
        let count = prototypes.len();
        let direct = (0..count)
            .map(|child| direct_between(self, prototypes, prototype, child, horizon))
            .collect();
        let lifetime_direct = (0..count)
            .map(|child| direct_between(self, prototypes, prototype, child, f64::INFINITY))
            .collect();
        Ok(MarkedOffspring {
            prototype,
            horizon,
            direct,
            lifetime_direct,
            horizon_descendants: marked_filter_descendants(self, prototypes, prototype, horizon)?,
            lifetime_descendants: marked_lifetime_descendants(self, prototypes, prototype)?,
        })
    }
}

fn direct_between(
    parameters: &MarkedParameters,
    prototypes: &[MarkPrototype],
    parent: usize,
    child: usize,
    horizon: f64,
) -> f64 {
    let rank = parameters.decay.len();
    let basis = parameters.baseline.len();
    let mut kernel = 0.0;
    for feature in 0..basis {
        for component in 0..rank {
            let factor = if horizon.is_finite() {
                horizon_factor(parameters.decay[component], horizon)
            } else {
                1.0
            };
            kernel += prototypes[child].target[feature]
                * parameters.excitation[feature * rank + component]
                * prototypes[parent].source[component]
                * factor;
        }
    }
    prototypes[child].mass * kernel
}

fn feature_mass(prototypes: &[MarkPrototype]) -> Vec<f64> {
    let basis = prototypes[0].target.len();
    let mut mass = vec![0.0; basis];
    for prototype in prototypes {
        for (feature, &value) in prototype.target.iter().enumerate() {
            mass[feature] += prototype.mass * value;
        }
    }
    mass
}

fn source_bound(prototypes: &[MarkPrototype]) -> Vec<f64> {
    let rank = prototypes[0].source.len();
    let mut bound = vec![0.0_f64; rank];
    for prototype in prototypes {
        for (component, &value) in prototype.source.iter().enumerate() {
            bound[component] = bound[component].max(value);
        }
    }
    bound
}

fn branching_matrix(parameters: &MarkedParameters, prototypes: &[MarkPrototype]) -> Vec<f64> {
    let count = prototypes.len();
    let mut matrix = vec![0.0; count * count];
    for parent in 0..count {
        for child in 0..count {
            matrix[child * count + parent] =
                direct_between(parameters, prototypes, parent, child, f64::INFINITY);
        }
    }
    matrix
}

fn spectral_radius_below_one(branching: &[f64], n: usize) -> Result<(), EventError> {
    let mut system = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            system[i * n + j] = f64::from(u8::from(i == j)) - branching[i * n + j];
        }
    }
    let witness = hawkes::solve(system, vec![1.0; n], 1e-13).ok_or(EventError::Unstable)?;
    for (row, &value) in witness.iter().enumerate() {
        let image: f64 = branching[row * n..(row + 1) * n]
            .iter()
            .zip(&witness)
            .map(|(weight, coordinate)| weight * coordinate)
            .sum();
        let bound = value * (1.0 - 64.0 * f64::EPSILON);
        if !value.is_finite()
            || value <= 0.0
            || image.partial_cmp(&bound) != Some(std::cmp::Ordering::Less)
        {
            return Err(EventError::Unstable);
        }
    }
    Ok(())
}

fn reproduction(parameters: &MarkedParameters, prototypes: &[MarkPrototype]) -> f64 {
    let mass = feature_mass(prototypes);
    let bound = source_bound(prototypes);
    let rank = parameters.decay.len();
    mass.iter().enumerate().fold(0.0, |total, (feature, &weight)| {
        total
            + (0..rank).fold(0.0, |row, component| {
                row + weight * parameters.excitation[feature * rank + component] * bound[component]
            })
    })
}

struct Filters {
    time: f64,
    decay: Vec<f64>,
    values: Vec<f64>,
}

impl Filters {
    fn at(
        decay: &[f64],
        time: f64,
        prototypes: &[MarkPrototype],
        arrivals: &[MarkedArrival],
    ) -> Self {
        let mut state = Self { time, decay: decay.to_vec(), values: vec![0.0; decay.len()] };
        if arrivals.is_empty() {
            return state;
        }
        let mut scratch = vec![0.0; decay.len()];
        state.time = arrivals[0].time;
        state.jump(&prototypes[arrivals[0].prototype]);
        for event in &arrivals[1..] {
            state.advance(event.time, &mut scratch);
            state.jump(&prototypes[event.prototype]);
        }
        state.advance(time, &mut scratch);
        state
    }

    fn advance(&mut self, time: f64, integral: &mut [f64]) {
        let elapsed = time - self.time;
        for (component, value) in self.values.iter_mut().enumerate() {
            let decay = self.decay[component];
            let retained = (-decay * elapsed).exp();
            if let Some(total) = integral.get_mut(component) {
                *total += *value * (1.0 - retained) / decay;
            }
            *value *= retained;
        }
        self.time = time;
    }

    fn jump(&mut self, prototype: &MarkPrototype) {
        for (component, value) in self.values.iter_mut().enumerate() {
            *value += self.decay[component] * prototype.source[component];
        }
    }
}

struct MarkedLikelihood {
    rank: usize,
    basis: usize,
    mass: Vec<f64>,
    events: Vec<(usize, f64, Vec<f64>)>,
    filter_integral: Vec<f64>,
    span: f64,
    immigrant_integral: f64,
}

impl MarkedLikelihood {
    fn new(
        history: &MarkedHistory,
        decay: &[f64],
        immigrant: Option<&hawkes::BaselineFeature>,
        work: &mut hawkes::Work<'_>,
    ) -> Result<Self, EventError> {
        let rank = decay.len();
        let basis = history.prototypes[0].target.len();
        work.take((history.left.len() + history.events.len() + 1) * rank * basis)?;
        let mut state =
            Filters::at(decay, history.window.start, &history.prototypes, &history.left);
        let mut integral = vec![0.0; rank];
        let mut events = Vec::with_capacity(history.events.len());
        for event in &history.events {
            state.advance(event.time, &mut integral);
            let level = immigrant.map(|feature| feature.value(event.time)).unwrap_or(0.0);
            events.push((event.prototype, level, state.values.clone()));
            state.jump(&history.prototypes[event.prototype]);
        }
        state.advance(history.window.end, &mut integral);
        let immigrant_integral = immigrant
            .map(|feature| feature.integral(history.window.start, history.window.end))
            .unwrap_or(0.0);
        Ok(Self {
            rank,
            basis,
            mass: feature_mass(&history.prototypes),
            events,
            filter_integral: integral,
            span: history.window.end - history.window.start,
            immigrant_integral,
        })
    }

    fn width(&self) -> usize {
        let immigrant =
            usize::from(self.immigrant_integral != 0.0 || self.events.iter().any(|e| e.1 != 0.0));
        self.basis * (1 + immigrant + self.rank)
    }

    fn intensity(
        &self,
        prototypes: &[MarkPrototype],
        prototype: usize,
        level: f64,
        filters: &[f64],
        coefficients: &[f64],
    ) -> (f64, Vec<f64>) {
        let immigrant = self.width() > self.basis * (1 + self.rank);
        let eta_offset = if immigrant { 2 * self.basis } else { self.basis };
        let mut eta = vec![0.0; self.basis];
        for feature in 0..self.basis {
            let mut value = coefficients[feature];
            if immigrant {
                value += coefficients[self.basis + feature] * level;
            }
            for component in 0..self.rank {
                let coordinate = eta_offset + feature * self.rank + component;
                value += coefficients[coordinate] * filters[component];
            }
            eta[feature] = value;
        }
        let mark = &prototypes[prototype];
        let density: f64 =
            eta.iter().zip(&mark.target).map(|(value, feature)| value * feature).sum();
        let intensity = mark.mass * density;
        let mut gradient = vec![0.0; self.width()];
        for feature in 0..self.basis {
            let scale = mark.mass * mark.target[feature];
            gradient[feature] += scale;
            if immigrant {
                gradient[self.basis + feature] += scale * level;
            }
            for component in 0..self.rank {
                gradient[eta_offset + feature * self.rank + component] +=
                    scale * filters[component];
            }
        }
        (intensity, gradient)
    }

    fn compensator(&self, coefficients: &[f64]) -> (f64, Vec<f64>) {
        let immigrant = self.width() > self.basis * (1 + self.rank);
        let eta_offset = if immigrant { 2 * self.basis } else { self.basis };
        let mut gradient = vec![0.0; self.width()];
        let mut total = 0.0;
        for feature in 0..self.basis {
            let weight = self.mass[feature];
            gradient[feature] = weight * self.span;
            total += weight * self.span * coefficients[feature];
            if immigrant {
                gradient[self.basis + feature] = weight * self.immigrant_integral;
                total += weight * self.immigrant_integral * coefficients[self.basis + feature];
            }
            for component in 0..self.rank {
                let coordinate = eta_offset + feature * self.rank + component;
                gradient[coordinate] = weight * self.filter_integral[component];
                total += weight * self.filter_integral[component] * coefficients[coordinate];
            }
        }
        (total, gradient)
    }
}

fn loss_and_gradient(
    likelihood: &MarkedLikelihood,
    prototypes: &[MarkPrototype],
    coefficients: &[f64],
    gradient: &mut [f64],
    work: &mut hawkes::Work<'_>,
) -> Result<f64, EventError> {
    work.take((likelihood.events.len() + 1) * likelihood.width())?;
    let (mut loss, compensator_gradient) = likelihood.compensator(coefficients);
    if !gradient.is_empty() {
        gradient.copy_from_slice(&compensator_gradient);
    }
    for &(prototype, level, ref filters) in &likelihood.events {
        let (intensity, event_gradient) =
            likelihood.intensity(prototypes, prototype, level, filters, coefficients);
        if !intensity.is_finite() || intensity <= 0.0 {
            return Ok(f64::INFINITY);
        }
        loss -= intensity.ln();
        if !gradient.is_empty() {
            let weight = intensity.recip();
            for (slot, value) in gradient.iter_mut().zip(&event_gradient) {
                *slot -= value * weight;
            }
        }
    }
    Ok(loss)
}

fn project_excitation(
    mut values: Vec<f64>,
    excitation_at: usize,
    weights: &[f64],
    cap: f64,
) -> Vec<f64> {
    for value in &mut values {
        *value = value.max(0.0);
    }
    let theta = values[excitation_at..].to_vec();
    let excited: f64 = theta.iter().zip(weights).map(|(value, weight)| value * weight).sum();
    if excited <= cap {
        return values;
    }
    let mut upper = 0.0_f64;
    for (value, weight) in theta.iter().zip(weights) {
        if *weight > 0.0 {
            upper = upper.max(value / weight);
        }
    }
    let mut lower = 0.0_f64;
    for _ in 0..80 {
        let middle = 0.5 * (lower + upper);
        let mass: f64 = theta
            .iter()
            .zip(weights)
            .map(|(value, weight)| weight * (value - middle * weight).max(0.0))
            .sum();
        if mass > cap {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    for (index, (value, weight)) in theta.iter().zip(weights).enumerate() {
        values[excitation_at + index] = (value - upper * weight).max(0.0);
    }
    values
}

fn excitation_weights(prototypes: &[MarkPrototype]) -> Vec<f64> {
    let mass = feature_mass(prototypes);
    let bound = source_bound(prototypes);
    let mut weights = Vec::with_capacity(mass.len() * bound.len());
    for &weight in &mass {
        for &limit in &bound {
            weights.push(weight * limit);
        }
    }
    weights
}

impl MarkedFitSpec {
    pub(super) fn validate(&self, history: &MarkedHistory) -> Result<(), EventError> {
        let rank = history.prototypes.first().map(|prototype| prototype.source.len()).unwrap_or(0);
        if !(0.0..1.0).contains(&self.reproduction_cap)
            || self.reproduction_cap == 0.0
            || !(1..=2000).contains(&self.iterations)
            || self.decay.len() != rank
            || self.decay.iter().any(|rate| !rate.is_finite() || *rate <= 0.0)
        {
            return Err(EventError::invalid("invalid marked fit scope"));
        }
        Ok(())
    }
}

fn finish(
    history: &MarkedHistory,
    decay: &[f64],
    coefficients: &[f64],
    immigrant: bool,
    loss: f64,
    iteration: usize,
    residual: f64,
    cap: f64,
    likelihood: &MarkedLikelihood,
    work: &mut hawkes::Work<'_>,
) -> Result<MarkedFit, EventError> {
    let basis = likelihood.basis;
    let rank = likelihood.rank;
    let excitation_at = if immigrant { 2 * basis } else { basis };
    let parameters = MarkedParameters {
        baseline: coefficients[..basis].to_vec(),
        excitation: coefficients[excitation_at..excitation_at + basis * rank].to_vec(),
        decay: decay.to_vec(),
    };
    parameters.validate(&history.prototypes)?;
    let reproduction = reproduction(&parameters, &history.prototypes);
    let zero_coefficients = parameters
        .excitation
        .iter()
        .enumerate()
        .filter_map(|(index, &value)| (value <= 1e-7).then_some(index))
        .collect();
    Ok(MarkedFit {
        parameters,
        prototypes: history.prototypes.clone(),
        log_likelihood: -loss,
        iterations: iteration,
        projected_residual: residual,
        reproduction,
        cap_active: reproduction >= cap - 1e-7,
        zero_coefficients,
        weak_identification: weak_marked(likelihood, &history.prototypes, immigrant, work)?,
    })
}

fn weak_marked(
    likelihood: &MarkedLikelihood,
    prototypes: &[MarkPrototype],
    immigrant: bool,
    work: &mut hawkes::Work<'_>,
) -> Result<bool, EventError> {
    let width = likelihood.rank + 1 + usize::from(immigrant);
    work.take((likelihood.events.len() + 1) * likelihood.basis * width * width)?;
    for feature in 0..likelihood.basis {
        let mut block = vec![0.0; width * width];
        for &(prototype, level, ref filters) in &likelihood.events {
            let scale = prototypes[prototype].mass * prototypes[prototype].target[feature];
            let mut row = vec![scale];
            if immigrant {
                row.push(scale * level);
            }
            row.extend(filters.iter().map(|filter| scale * filter));
            for i in 0..width {
                for j in 0..width {
                    block[i * width + j] += row[i] * row[j];
                }
            }
        }
        let norm: Vec<_> = (0..width).map(|j| block[j * width + j].sqrt()).collect();
        if norm.iter().any(|value| !value.is_finite() || *value == 0.0) {
            return Ok(true);
        }
        for i in 0..width {
            for j in 0..width {
                block[i * width + j] /= norm[i] * norm[j];
            }
        }
        if hawkes::solve(block, vec![1.0; width], 1e-7).is_none() {
            return Ok(true);
        }
    }
    Ok(false)
}

fn fit_coefficients(
    history: &MarkedHistory,
    spec: &MarkedFitSpec,
    immigrant: Option<&hawkes::BaselineFeature>,
    work: &mut hawkes::Work<'_>,
) -> Result<MarkedFit, EventError> {
    history.validate()?;
    spec.validate(history)?;
    if let Some(feature) = immigrant {
        feature.validate(history.window)?;
    }
    if history.events.is_empty() {
        return Err(EventError::Fit("marked fit needs observed arrivals"));
    }
    let likelihood = MarkedLikelihood::new(history, &spec.decay, immigrant, work)?;
    let basis = likelihood.basis;
    let width = likelihood.width();
    let rate = history.events.len() as f64 / likelihood.span;
    let mass_norm: f64 = likelihood.mass.iter().map(|value| value * value).sum();
    if mass_norm == 0.0 {
        return Err(EventError::Fit("mark features have no reference mass"));
    }
    let mut coefficients = vec![0.0; width];
    for feature in 0..basis {
        coefficients[feature] = rate * likelihood.mass[feature] / mass_norm;
    }
    let weights = excitation_weights(&history.prototypes);
    let excitation_at = width - basis * likelihood.rank;
    let scale = history.events.len() as f64;
    let mut gradient = vec![0.0; width];
    let mut step = 1.0_f64;
    for iteration in 0..spec.iterations {
        let loss = loss_and_gradient(
            &likelihood,
            &history.prototypes,
            &coefficients,
            &mut gradient,
            work,
        )?;
        if !loss.is_finite() || gradient.iter().any(|value| !value.is_finite()) {
            return Err(EventError::Fit("nonfinite likelihood or gradient"));
        }
        for value in &mut gradient {
            *value /= scale;
        }
        let projected = project_excitation(
            coefficients.iter().zip(&gradient).map(|(value, slope)| value - slope).collect(),
            excitation_at,
            &weights,
            spec.reproduction_cap,
        );
        let residual = coefficients
            .iter()
            .zip(&projected)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max);
        if residual <= 1e-7 {
            let immigrant_on = width > basis * (1 + likelihood.rank);
            return finish(
                history,
                &spec.decay,
                &coefficients,
                immigrant_on,
                loss,
                iteration,
                residual,
                spec.reproduction_cap,
                &likelihood,
                work,
            );
        }
        step = (2.0 * step).min(1024.0);
        let mut accepted = None;
        for _ in 0..48 {
            let candidate = project_excitation(
                coefficients
                    .iter()
                    .zip(&gradient)
                    .map(|(value, slope)| value - step * slope)
                    .collect(),
                excitation_at,
                &weights,
                spec.reproduction_cap,
            );
            let descent: f64 = gradient
                .iter()
                .zip(coefficients.iter().zip(&candidate))
                .map(|(slope, (left, right))| slope * (right - left))
                .sum();
            let next =
                loss_and_gradient(&likelihood, &history.prototypes, &candidate, &mut [], work)?
                    / scale;
            if descent < 0.0 && next <= loss / scale + 1e-4 * descent {
                accepted = Some(candidate);
                break;
            }
            step *= 0.5;
        }
        coefficients = accepted.ok_or(EventError::Fit("likelihood line search stalled"))?;
    }
    Err(EventError::Fit("optimizer iteration limit"))
}

pub(super) fn fit(
    history: &MarkedHistory,
    spec: &MarkedFitSpec,
    work: &mut hawkes::Work<'_>,
) -> Result<MarkedFit, EventError> {
    fit_coefficients(history, spec, None, work)
}

pub(super) fn fit_alternative(
    history: &MarkedHistory,
    spec: &MarkedFitSpec,
    feature: &hawkes::BaselineFeature,
    work: &mut hawkes::Work<'_>,
) -> Result<MarkedFit, EventError> {
    fit_coefficients(history, spec, Some(feature), work)
}

pub(super) fn log_likelihood(
    history: &MarkedHistory,
    parameters: &MarkedParameters,
    work: &mut hawkes::Work<'_>,
) -> Result<f64, EventError> {
    history.validate()?;
    parameters.validate(&history.prototypes)?;
    let likelihood = MarkedLikelihood::new(history, &parameters.decay, None, work)?;
    let mut coefficients = parameters.baseline.clone();
    coefficients.extend_from_slice(&parameters.excitation);
    Ok(-loss_and_gradient(&likelihood, &history.prototypes, &coefficients, &mut [], work)?)
}

fn marked_lifetime_descendants(
    parameters: &MarkedParameters,
    prototypes: &[MarkPrototype],
    parent: usize,
) -> Result<Vec<f64>, EventError> {
    let count = prototypes.len();
    let matrix = branching_matrix(parameters, prototypes);
    let mut system = vec![0.0; count * count];
    for row in 0..count {
        for column in 0..count {
            system[row * count + column] =
                f64::from(u8::from(row == column)) - matrix[row * count + column];
        }
    }
    let mut rhs = vec![0.0; count];
    rhs[parent] = 1.0;
    let mut total = hawkes::solve(system, rhs, 1e-13).ok_or(EventError::Unstable)?;
    total[parent] -= 1.0;
    for value in &mut total {
        if !value.is_finite() || *value < -1e-8 {
            return Err(EventError::Numerical("nonfinite marked lifetime descendants"));
        }
        if *value < 0.0 {
            *value = 0.0;
        }
    }
    Ok(total)
}

fn marked_filter_descendants(
    parameters: &MarkedParameters,
    prototypes: &[MarkPrototype],
    parent: usize,
    horizon: f64,
) -> Result<Vec<f64>, EventError> {
    let rank = parameters.decay.len();
    let count = prototypes.len();
    let max_decay = parameters.decay.iter().copied().fold(0.0_f64, f64::max);
    let step = (1.0 / (64.0 * max_decay)).min(horizon);
    let steps = (horizon / step).ceil() as usize;
    if steps > 200_000 || !step.is_finite() {
        return Err(EventError::Numerical("offspring horizon is too long for the decay"));
    }
    let dt = horizon / steps as f64;
    let mut filters = vec![0.0; rank];
    for component in 0..rank {
        filters[component] = parameters.decay[component] * prototypes[parent].source[component];
    }
    let mut counts = vec![0.0; count];
    for _ in 0..steps {
        let (k1_f, k1_c) = marked_derivative(parameters, prototypes, &filters);
        let filters_2 = axpy(&filters, &k1_f, dt * 0.5);
        let (k2_f, k2_c) = marked_derivative(parameters, prototypes, &filters_2);
        let filters_3 = axpy(&filters, &k2_f, dt * 0.5);
        let (k3_f, k3_c) = marked_derivative(parameters, prototypes, &filters_3);
        let filters_4 = axpy(&filters, &k3_f, dt);
        let (k4_f, k4_c) = marked_derivative(parameters, prototypes, &filters_4);
        filters = rk4(&filters, &k1_f, &k2_f, &k3_f, &k4_f, dt);
        counts = rk4(&counts, &k1_c, &k2_c, &k3_c, &k4_c, dt);
    }
    if counts.iter().any(|value| !value.is_finite()) {
        return Err(EventError::Numerical("nonfinite marked horizon descendants"));
    }
    Ok(counts)
}

fn marked_derivative(
    parameters: &MarkedParameters,
    prototypes: &[MarkPrototype],
    filters: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let rank = parameters.decay.len();
    let basis = parameters.baseline.len();
    let mut eta = vec![0.0; basis];
    for feature in 0..basis {
        eta[feature] = (0..rank)
            .map(|component| parameters.excitation[feature * rank + component] * filters[component])
            .sum();
    }
    let mut intensity = vec![0.0; prototypes.len()];
    for (index, prototype) in prototypes.iter().enumerate() {
        intensity[index] = prototype.mass
            * eta
                .iter()
                .zip(&prototype.target)
                .map(|(value, feature)| value * feature)
                .sum::<f64>();
    }
    let mut slope = vec![0.0; rank];
    for component in 0..rank {
        let incoming: f64 = prototypes
            .iter()
            .zip(&intensity)
            .map(|(prototype, rate)| prototype.source[component] * rate)
            .sum();
        slope[component] = parameters.decay[component] * (incoming - filters[component]);
    }
    (slope, intensity)
}

fn axpy(state: &[f64], slope: &[f64], scale: f64) -> Vec<f64> {
    state.iter().zip(slope).map(|(value, slope)| value + scale * slope).collect()
}

fn rk4(state: &[f64], k1: &[f64], k2: &[f64], k3: &[f64], k4: &[f64], dt: f64) -> Vec<f64> {
    let scale = dt / 6.0;
    state
        .iter()
        .zip(k1)
        .zip(k2)
        .zip(k3)
        .zip(k4)
        .map(|((((value, a), b), c), d)| value + scale * (a + 2.0 * b + 2.0 * c + d))
        .collect()
}

/// Draw a marked path on `window` conditional on `prefix`.
pub(super) fn sample(
    history: &MarkedHistory,
    prefix: &[MarkedArrival],
    parameters: &MarkedParameters,
    window: EventWindow,
    stream: (
        &antecedent_core::execution::RngFactory,
        antecedent_core::execution::StreamDomain,
        u64,
    ),
    max_events: usize,
    work: &mut hawkes::Work<'_>,
) -> Result<Vec<MarkedArrival>, EventError> {
    work.take((prefix.len() + 1) * parameters.decay.len())?;
    let mut state = Filters::at(&parameters.decay, window.start, &history.prototypes, prefix);
    let mut rng = stream.0.stream_for(stream.1, stream.2);
    let mut events = Vec::new();
    let ground = |state: &Filters| -> (f64, Vec<f64>) {
        let basis = parameters.baseline.len();
        let rank = parameters.decay.len();
        let mut eta = parameters.baseline.clone();
        for feature in 0..basis {
            for component in 0..rank {
                eta[feature] +=
                    parameters.excitation[feature * rank + component] * state.values[component];
            }
        }
        let mut intensity = Vec::with_capacity(history.prototypes.len());
        for prototype in &history.prototypes {
            let density: f64 =
                eta.iter().zip(&prototype.target).map(|(value, feature)| value * feature).sum();
            intensity.push(prototype.mass * density);
        }
        (intensity.iter().sum(), intensity)
    };
    let uniform = |rng: &mut antecedent_core::execution::CausalRng| {
        (((rng.next_u64() >> 12) as f64) + 0.5) * (1.0 / (1_u64 << 52) as f64)
    };
    let (mut bound, _) = ground(&state);
    loop {
        work.take(parameters.decay.len() + history.prototypes.len())?;
        if !bound.is_finite() {
            return Err(EventError::Numerical("nonfinite marked intensity"));
        }
        if bound == 0.0 {
            break;
        }
        let proposal = state.time - uniform(&mut rng).ln() / bound;
        if proposal >= window.end {
            break;
        }
        if proposal <= state.time {
            return Err(EventError::Numerical("timestamp precision cannot resolve next event"));
        }
        state.advance(proposal, &mut vec![0.0; parameters.decay.len()]);
        let (total, intensity) = ground(&state);
        let mut draw = uniform(&mut rng) * bound;
        if draw < total {
            let mut prototype = intensity.len() - 1;
            for (index, &value) in intensity.iter().enumerate() {
                if draw < value {
                    prototype = index;
                    break;
                }
                draw -= value;
            }
            if events.len() == max_events {
                return Err(EventError::Budget);
            }
            events.push(MarkedArrival { time: proposal, prototype });
            state.jump(&history.prototypes[prototype]);
            bound = ground(&state).0;
        } else {
            bound = total;
        }
    }
    Ok(events)
}

/// Compiled marked likelihood.
pub struct PreparedMarkedLikelihood {
    history: MarkedHistory,
    parameters: MarkedParameters,
    plan: super::PreparedKernel,
    _lease: super::AdmissionLease,
}

/// Compiled marked fit.
pub struct PreparedMarkedFit {
    history: MarkedHistory,
    spec: MarkedFitSpec,
    plan: super::PreparedKernel,
    _lease: super::AdmissionLease,
}

impl crate::Study {
    /// Compile a marked conditional likelihood before evaluating it.
    ///
    /// # Errors
    /// Invalid history, unstable parameters or resource refusal.
    pub fn prepare_marked_likelihood(
        history: MarkedHistory,
        parameters: MarkedParameters,
        ctx: &ExecutionContext,
    ) -> Result<PreparedMarkedLikelihood, EventError> {
        super::super::lowering::refuse_reentry(ctx)?;
        history.validate()?;
        parameters.validate(&history.prototypes)?;
        let lease = super::admit(ctx, 0, 8_192)?;
        let source =
            super::digest("antecedent.events.marked_likelihood.v1", &(&history, &parameters))?;
        let plan = super::PreparedKernel::new(source, "marked_likelihood", super::CAPABILITY)?;
        Ok(PreparedMarkedLikelihood { history, parameters, plan, _lease: lease })
    }

    /// Compile a marked fit before estimating it.
    ///
    /// # Errors
    /// Invalid history, fit scope or resource refusal.
    pub fn prepare_marked_fit(
        history: MarkedHistory,
        spec: MarkedFitSpec,
        ctx: &ExecutionContext,
    ) -> Result<PreparedMarkedFit, EventError> {
        super::super::lowering::refuse_reentry(ctx)?;
        history.validate()?;
        spec.validate(&history)?;
        let lease = super::admit(ctx, 0, 8_192)?;
        let source = super::digest("antecedent.events.marked_fit.v1", &(&history, &spec))?;
        let plan = super::PreparedKernel::new(source, "marked_fit", super::CAPABILITY)?;
        Ok(PreparedMarkedFit { history, spec, plan, _lease: lease })
    }
}

impl PreparedMarkedLikelihood {
    /// Inspect the compiled marked likelihood.
    pub fn plan_view(&self) -> crate::planner::PlanView {
        self.plan.plan_view()
    }

    /// Evaluate the conditional log likelihood.
    ///
    /// # Errors
    /// Resource, cancellation or an unsupported release scope.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<f64, EventError> {
        let scoped = super::scoped_execution(ctx, self.plan.identity(), super::CAPABILITY)?;
        log_likelihood(&self.history, &self.parameters, &mut hawkes::Work::new(&scoped))
    }
}

impl PreparedMarkedFit {
    /// Inspect the compiled marked fit.
    pub fn plan_view(&self) -> crate::planner::PlanView {
        self.plan.plan_view()
    }

    /// Fit the finite-rank kernel under the reproduction cap.
    ///
    /// # Errors
    /// The optimizer fails, or the fitted kernel is not stable.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<MarkedFit, EventError> {
        let scoped = super::scoped_execution(ctx, self.plan.identity(), super::CAPABILITY)?;
        fit(&self.history, &self.spec, &mut hawkes::Work::new(&scoped))
    }
}

/// Compiled conditional draws from a declared marked kernel.
pub struct PreparedMarkedDraw {
    history: MarkedHistory,
    parameters: MarkedParameters,
    window: EventWindow,
    paths: usize,
    max_events: usize,
    seed: u64,
    plan: super::PreparedKernel,
    _lease: super::AdmissionLease,
}

impl crate::Study {
    /// Compile conditional marked paths before drawing them.
    ///
    /// Each path is conditioned on arrivals strictly before `window.start`.
    ///
    /// # Errors
    /// Invalid history, unstable parameters, an invalid window, or a resource refusal.
    pub fn prepare_marked_draw(
        history: MarkedHistory,
        parameters: MarkedParameters,
        window: EventWindow,
        paths: usize,
        max_events: usize,
        seed: u64,
        ctx: &ExecutionContext,
    ) -> Result<PreparedMarkedDraw, EventError> {
        super::super::lowering::refuse_reentry(ctx)?;
        history.validate()?;
        parameters.validate(&history.prototypes)?;
        window.validate()?;
        super::validate_budget(paths, max_events)?;
        let lease = super::admit(ctx, 0, 8_192)?;
        let source = super::digest(
            "antecedent.events.marked_draw.v1",
            &(&history, &parameters, window, paths, max_events, seed),
        )?;
        let plan = super::PreparedKernel::new(source, "marked_draw", super::CAPABILITY)?;
        Ok(PreparedMarkedDraw {
            history,
            parameters,
            window,
            paths,
            max_events,
            seed,
            plan,
            _lease: lease,
        })
    }
}

impl PreparedMarkedDraw {
    /// Inspect the compiled marked draw.
    pub fn plan_view(&self) -> crate::planner::PlanView {
        self.plan.plan_view()
    }

    /// Draw the compiled paths.
    ///
    /// # Errors
    /// Resource exhaustion, cancellation, or numerical failure.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<Vec<Vec<MarkedArrival>>, EventError> {
        let scoped = super::scoped_execution(ctx, self.plan.identity(), super::CAPABILITY)?;
        let mut work = hawkes::Work::new(&scoped);
        let rng = antecedent_core::execution::RngFactory::from_seed(self.seed);
        let prefix: Vec<MarkedArrival> = self
            .history
            .left
            .iter()
            .chain(&self.history.events)
            .copied()
            .filter(|event| event.time < self.window.start)
            .collect();
        (0..self.paths)
            .map(|index| {
                sample(
                    &self.history,
                    &prefix,
                    &self.parameters,
                    self.window,
                    (&rng, hawkes::FORECAST, index as u64),
                    self.max_events,
                    &mut work,
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod marked_tests {
    use super::{
        EventWindow, MarkPrototype, MarkedArrival, MarkedFitSpec, MarkedHistory, MarkedParameters,
        fit, log_likelihood, project_excitation,
    };
    use crate::Study;
    use crate::analysis::events::hawkes;
    use antecedent_core::ExecutionContext;

    fn prototype() -> MarkPrototype {
        MarkPrototype { source: vec![1.0], target: vec![1.0], mass: 1.0 }
    }

    fn history(events: Vec<MarkedArrival>, end: f64) -> MarkedHistory {
        MarkedHistory {
            time_unit: "hour".into(),
            origin: "origin".into(),
            source: "log".into(),
            empty_origin: 0.0,
            window: EventWindow { start: 0.0, end },
            left: vec![],
            events,
            prototypes: vec![prototype()],
        }
    }

    #[test]
    fn poisson_likelihood_and_offspring_match_the_univariate_hawkes_formulas() {
        let ctx = ExecutionContext::for_tests(0);
        let observed = history(
            vec![
                MarkedArrival { time: 0.5, prototype: 0 },
                MarkedArrival { time: 1.5, prototype: 0 },
            ],
            2.0,
        );
        let parameters =
            MarkedParameters { baseline: vec![2.0], excitation: vec![0.0], decay: vec![1.0] };
        let value = log_likelihood(&observed, &parameters, &mut hawkes::Work::new(&ctx)).unwrap();
        assert!((value - (2.0 * 2.0_f64.ln() - 4.0)).abs() < 1e-12);
        let excited =
            MarkedParameters { baseline: vec![0.5], excitation: vec![0.4], decay: vec![1.0] };
        let offspring = excited.excitation_offspring(&observed.prototypes, 0, 2.0).unwrap();
        let lifetime = 0.4 / 0.6;
        assert!((offspring.lifetime_descendants[0] - lifetime).abs() < 1e-9);
        assert!((offspring.direct[0] - 0.4 * (1.0 - (-2.0_f64).exp())).abs() < 1e-12);
        assert!(
            MarkedParameters { baseline: vec![0.5], excitation: vec![1.0], decay: vec![1.0] }
                .validate(&observed.prototypes)
                .is_err()
        );
    }

    #[test]
    fn reproduction_cap_projects_only_the_excitation_block() {
        let projected = project_excitation(vec![1.25, 0.8], 1, &[1.0], 0.5);
        assert!((projected[0] - 1.25).abs() < 1e-12);
        assert!((projected[1] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn regular_arrivals_fit_a_near_zero_excitation() {
        let ctx = ExecutionContext::for_tests(0);
        let events =
            (0..20).map(|i| MarkedArrival { time: i as f64 + 0.5, prototype: 0 }).collect();
        let observed = history(events, 20.0);
        let spec = MarkedFitSpec { decay: vec![1.0], reproduction_cap: 0.9, iterations: 200 };
        let fitted = fit(&observed, &spec, &mut hawkes::Work::new(&ctx)).unwrap();
        assert!(fitted.parameters.excitation[0] < 1e-6);
        assert!((fitted.parameters.baseline[0] - 1.0).abs() < 1e-4);
        assert!(!fitted.cap_active);
    }

    #[test]
    fn zero_intensity_simulation_is_empty() {
        let ctx = ExecutionContext::for_tests(0);
        let observed = history(vec![], 1.0);
        let parameters =
            MarkedParameters { baseline: vec![0.0], excitation: vec![0.0], decay: vec![1.0] };
        let drawn = Study::prepare_marked_draw(
            observed,
            parameters,
            EventWindow { start: 0.0, end: 1.0 },
            1,
            8,
            3,
            &ctx,
        )
        .unwrap()
        .execute(&ctx)
        .unwrap();
        assert!(drawn == vec![Vec::new()]);
    }
}
