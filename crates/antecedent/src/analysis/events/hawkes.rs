//! Fixed-decay linear Hawkes likelihood, fitting and conditional simulation.

use antecedent_core::ExecutionContext;
use antecedent_core::execution::{RngFactory, StreamDomain};
use serde::{Deserialize, Serialize};

use super::{Arrival, EventError, EventHistory, EventWindow, poll};

/// Rows are receiving types and columns are source types. An entry in
/// `branching` is integrated offspring mass, in row-major order.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HawkesParameters {
    /// Constant immigrant intensity per type.
    pub baseline: Vec<f64>,
    /// Nonnegative offspring masses.
    pub branching: Vec<f64>,
    /// Strictly positive inverse-time rates.
    pub decay: Vec<f64>,
}

/// Fixed-decay constrained likelihood fit.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HawkesFitSpec {
    /// Declared inverse-time rates in target/source order.
    pub decay: Vec<f64>,
    /// Maximum sum of offspring mass from one source, strictly between zero and one.
    pub column_cap: f64,
    /// Maximum projected-gradient iterations.
    pub iterations: usize,
}

/// Diagnostics for a converged fit.
#[derive(Clone, Debug, Serialize)]
pub struct HawkesFit {
    /// Fitted point parameters.
    pub parameters: HawkesParameters,
    /// Continuous observation-window conditional log likelihood.
    pub log_likelihood: f64,
    /// Number of projected-gradient iterations.
    pub iterations: usize,
    /// Projected-gradient stationarity residual.
    pub projected_residual: f64,
    /// Source columns that reached the stability cap.
    pub active_columns: Vec<usize>,
    /// Branching coordinates at zero.
    pub zero_edges: Vec<usize>,
    /// Singular or poorly conditioned observed information in the fitted design.
    pub weak_identification: bool,
}

/// Shared bound across an entire invocation, including refits and proposals.
pub(super) struct Work<'a> {
    ctx: &'a ExecutionContext,
    remaining: usize,
}

impl<'a> Work<'a> {
    pub(super) fn new(ctx: &'a ExecutionContext) -> Self {
        Self { ctx, remaining: 250_000_000 }
    }

    pub(super) fn take(&mut self, amount: usize) -> Result<(), EventError> {
        poll(self.ctx)?;
        self.remaining = self.remaining.checked_sub(amount).ok_or(EventError::Budget)?;
        Ok(())
    }
}

impl HawkesParameters {
    pub(super) fn validate(&self, n: usize) -> Result<(), EventError> {
        if self.baseline.len() != n
            || self.branching.len() != n * n
            || self.decay.len() != n * n
            || self.baseline.iter().chain(&self.branching).any(|x| !x.is_finite() || *x < 0.0)
            || self.decay.iter().any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err(EventError::invalid(
                "invalid Hawkes dimensions, baseline, branching or decay",
            ));
        }
        // A nonnegative A has rho(A) < 1 iff there is x > 0 with Ax < x.
        // Solve (I-A)x=1 then check the witness directly, with a rounding margin.
        let mut system = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                system[i * n + j] = f64::from(u8::from(i == j)) - self.branching[i * n + j];
            }
        }
        let witness = solve(system, vec![1.0; n], 1e-13).ok_or(EventError::Unstable)?;
        for (i, &x) in witness.iter().enumerate() {
            let ax: f64 = self.branching[i * n..(i + 1) * n]
                .iter()
                .zip(&witness)
                .map(|(weight, value)| weight * value)
                .sum();
            if !x.is_finite() || x <= 0.0 || !(ax < x * (1.0 - 64.0 * f64::EPSILON)) {
                return Err(EventError::Unstable);
            }
        }
        Ok(())
    }
}

fn solve(mut matrix: Vec<f64>, mut rhs: Vec<f64>, tolerance: f64) -> Option<Vec<f64>> {
    let n = rhs.len();
    for i in 0..n {
        let pivot =
            (i..n).max_by(|&a, &b| matrix[a * n + i].abs().total_cmp(&matrix[b * n + i].abs()))?;
        for j in i..n {
            matrix.swap(i * n + j, pivot * n + j);
        }
        rhs.swap(i, pivot);
        let diagonal = matrix[i * n + i];
        if !diagonal.is_finite() || diagonal.abs() <= tolerance {
            return None;
        }
        for j in i + 1..n {
            let factor = matrix[j * n + i] / diagonal;
            for k in i + 1..n {
                matrix[j * n + k] -= factor * matrix[i * n + k];
            }
            rhs[j] -= factor * rhs[i];
        }
    }
    for i in (0..n).rev() {
        let upper: f64 = (i + 1..n).map(|j| matrix[i * n + j] * rhs[j]).sum();
        rhs[i] = (rhs[i] - upper) / matrix[i * n + i];
    }
    Some(rhs)
}

struct HistoryState<'a> {
    time: f64,
    decay: &'a [f64],
    values: Vec<f64>,
    n: usize,
}

impl<'a> HistoryState<'a> {
    fn at(history: &EventHistory, decay: &'a [f64], time: f64, arrivals: &[Arrival]) -> Self {
        let n = history.catalog.len();
        let mut state = Self { time, decay, values: vec![0.0; n * n], n };
        for event in arrivals {
            for i in 0..n {
                let k = i * n + event.kind;
                let rate = decay[k];
                state.values[k] += rate * (-rate * (time - event.time)).exp();
            }
        }
        state
    }

    fn advance(&mut self, time: f64, integral: &mut [f64]) {
        let elapsed = time - self.time;
        for (k, value) in self.values.iter_mut().enumerate() {
            let exponent = -self.decay[k] * elapsed;
            if let Some(total) = integral.get_mut(k) {
                *total += *value * (-exponent.exp_m1()) / self.decay[k];
            }
            *value *= exponent.exp();
        }
        self.time = time;
    }

    fn arrive(&mut self, kind: usize) {
        for i in 0..self.n {
            let k = i * self.n + kind;
            self.values[k] += self.decay[k];
        }
    }
}

struct Likelihood {
    n: usize,
    features: Vec<(usize, Vec<f64>)>,
    compensator: Vec<f64>,
}

impl Likelihood {
    fn new(history: &EventHistory, decay: &[f64], work: &mut Work<'_>) -> Result<Self, EventError> {
        let n = history.catalog.len();
        work.take((history.left.len() + history.events.len() + 1) * n * n)?;
        let mut state = HistoryState::at(history, decay, history.window.start, &history.left);
        let mut integral = vec![0.0; n * n];
        let mut features = Vec::with_capacity(history.events.len());
        for event in &history.events {
            state.advance(event.time, &mut integral);
            let start = event.kind * n;
            features.push((event.kind, state.values[start..start + n].to_vec()));
            state.arrive(event.kind);
        }
        state.advance(history.window.end, &mut integral);
        let mut compensator = vec![history.window.end - history.window.start; n];
        compensator.extend(integral);
        Ok(Self { n, features, compensator })
    }

    fn evaluate(
        &self,
        x: &[f64],
        gradient: &mut [f64],
        work: &mut Work<'_>,
    ) -> Result<f64, EventError> {
        work.take((self.features.len() + 1) * (self.n + 1))?;
        if !gradient.is_empty() {
            gradient.copy_from_slice(&self.compensator);
        }
        let mut loss: f64 = self.compensator.iter().zip(x).map(|(a, b)| a * b).sum();
        for (kind, row) in &self.features {
            let offset = self.n + *kind * self.n;
            let intensity = x[*kind]
                + row.iter().zip(&x[offset..offset + self.n]).map(|(a, b)| a * b).sum::<f64>();
            if !intensity.is_finite() || intensity <= 0.0 {
                return Ok(f64::INFINITY);
            }
            loss -= intensity.ln();
            if !gradient.is_empty() {
                gradient[*kind] -= intensity.recip();
                for (j, value) in row.iter().enumerate() {
                    gradient[offset + j] -= value / intensity;
                }
            }
        }
        Ok(loss)
    }

    fn weak_information(&self, work: &mut Work<'_>) -> Result<bool, EventError> {
        let n = self.n;
        work.take((self.features.len() + 1) * (n + 1) * (n + 1))?;
        let width = n + 1;
        let mut covariance = vec![vec![0.0; width * width]; n];
        for (kind, features) in &self.features {
            let row: Vec<_> = std::iter::once(1.0).chain(features.iter().copied()).collect();
            for i in 0..width {
                for j in 0..width {
                    covariance[*kind][i * width + j] += row[i] * row[j];
                }
            }
        }
        for mut block in covariance {
            let norm: Vec<_> = (0..width).map(|j| block[j * width + j].sqrt()).collect();
            if norm.iter().any(|x| !x.is_finite() || *x == 0.0) {
                return Ok(true);
            }
            for i in 0..width {
                for j in 0..width {
                    block[i * width + j] /= norm[i] * norm[j];
                }
            }
            if solve(block, vec![1.0; width], 1e-7).is_none() {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

pub(super) fn log_likelihood(
    history: &EventHistory,
    parameters: &HawkesParameters,
    work: &mut Work<'_>,
) -> Result<f64, EventError> {
    let mut x = parameters.baseline.clone();
    x.extend_from_slice(&parameters.branching);
    Ok(-Likelihood::new(history, &parameters.decay, work)?.evaluate(&x, &mut [], work)?)
}

impl HawkesFitSpec {
    pub(super) fn validate(&self, n: usize) -> Result<(), EventError> {
        if !(0.0..1.0).contains(&self.column_cap)
            || self.column_cap == 0.0
            || !(1..=2000).contains(&self.iterations)
            || self.decay.len() != n * n
            || self.decay.iter().any(|rate| !rate.is_finite() || *rate <= 0.0)
        {
            return Err(EventError::invalid("invalid fixed-decay fit scope"));
        }
        Ok(())
    }
}

// Projection onto nonnegative baselines and independent capped source simplexes.
fn project(mut values: Vec<f64>, n: usize, cap: f64) -> Vec<f64> {
    for value in &mut values {
        *value = value.max(0.0);
    }
    for source in 0..n {
        let mut column: Vec<_> = (0..n).map(|target| values[n + target * n + source]).collect();
        if column.iter().sum::<f64>() <= cap {
            continue;
        }
        column.sort_unstable_by(|a, b| b.total_cmp(a));
        let mut sum = 0.0;
        let mut threshold: f64 = 0.0;
        for (i, value) in column.into_iter().enumerate() {
            sum += value;
            threshold = threshold.max((sum - cap) / (i + 1) as f64);
        }
        for target in 0..n {
            let value = &mut values[n + target * n + source];
            *value = (*value - threshold).max(0.0);
        }
    }
    values
}

pub(super) fn fit(
    history: &EventHistory,
    spec: &HawkesFitSpec,
    work: &mut Work<'_>,
) -> Result<HawkesFit, EventError> {
    let n = history.catalog.len();
    let likelihood = Likelihood::new(history, &spec.decay, work)?;
    let mut counts = vec![0_u32; n];
    for event in &history.events {
        counts[event.kind] += 1;
    }
    if counts.contains(&0) {
        return Err(EventError::Fit("every fitted type needs observed arrivals"));
    }
    let scale = history.events.len() as f64;
    let mut x: Vec<_> =
        counts.into_iter().map(|x| f64::from(x) / likelihood.compensator[0]).collect();
    x.resize(n + n * n, 0.0);
    let mut gradient = vec![0.0; x.len()];
    let mut step: f64 = 1.0;
    for iteration in 0..spec.iterations {
        let loss = likelihood.evaluate(&x, &mut gradient, work)?;
        if !loss.is_finite() || gradient.iter().any(|value| !value.is_finite()) {
            return Err(EventError::Fit("nonfinite likelihood or gradient"));
        }
        for value in &mut gradient {
            *value /= scale;
        }
        let projected =
            project(x.iter().zip(&gradient).map(|(a, b)| a - b).collect(), n, spec.column_cap);
        let residual = x.iter().zip(&projected).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        if residual <= 1e-7 {
            let branching = x.split_off(n);
            let active_columns = (0..n)
                .filter(|&j| {
                    (0..n).map(|i| branching[i * n + j]).sum::<f64>() >= spec.column_cap - 1e-7
                })
                .collect();
            let zero_edges = branching
                .iter()
                .enumerate()
                .filter_map(|(k, &x)| (x <= 1e-7).then_some(k))
                .collect();
            let parameters = HawkesParameters { baseline: x, branching, decay: spec.decay.clone() };
            parameters.validate(n)?;
            return Ok(HawkesFit {
                parameters,
                log_likelihood: -loss,
                iterations: iteration,
                projected_residual: residual,
                active_columns,
                zero_edges,
                weak_identification: likelihood.weak_information(work)?,
            });
        }
        step = (2.0 * step).min(1024.0);
        let mut accepted = None;
        for _ in 0..48 {
            let candidate = project(
                x.iter().zip(&gradient).map(|(a, b)| a - step * b).collect(),
                n,
                spec.column_cap,
            );
            let descent: f64 =
                gradient.iter().zip(x.iter().zip(&candidate)).map(|(g, (a, b))| g * (b - a)).sum();
            let next = likelihood.evaluate(&candidate, &mut [], work)? / scale;
            if descent < 0.0 && next <= loss / scale + 1e-4 * descent {
                accepted = Some(candidate);
                break;
            }
            step *= 0.5;
        }
        x = accepted.ok_or(EventError::Fit("likelihood line search stalled"))?;
    }
    Err(EventError::Fit("optimizer iteration limit"))
}

pub(super) const FORECAST: StreamDomain = StreamDomain::external(0x45564e54464f5245);
pub(super) const REFIT: StreamDomain = StreamDomain::external(0x45564e5452464954);
pub(super) const UPDATE: StreamDomain = StreamDomain::external(0x45564e5455504454);
pub(super) const STUDY: StreamDomain = StreamDomain::external(0x45564e5453545544);

pub(super) fn sample(
    history: &EventHistory,
    prefix: &[Arrival],
    parameters: &HawkesParameters,
    window: EventWindow,
    stream: (&RngFactory, StreamDomain, u64),
    max_events: usize,
    work: &mut Work<'_>,
) -> Result<Vec<Arrival>, EventError> {
    let n = history.catalog.len();
    work.take((prefix.len() + 1) * n * n)?;
    let mut state = HistoryState::at(history, &parameters.decay, window.start, prefix);
    let mut rng = stream.0.stream_for(stream.1, stream.2);
    let mut events = Vec::new();
    let mut intensity = vec![0.0; n];
    let evaluate = |state: &HistoryState<'_>, values: &mut [f64]| {
        for (i, value) in values.iter_mut().enumerate() {
            let start = i * n;
            *value = parameters.baseline[i]
                + (start..start + n)
                    .map(|k| parameters.branching[k] * state.values[k])
                    .sum::<f64>();
        }
        values.iter().sum::<f64>()
    };
    let mut bound = evaluate(&state, &mut intensity);
    // Use strictly interior 52-bit uniforms to avoid zero elapsed proposals.
    let uniform = |rng: &mut antecedent_core::execution::CausalRng| {
        (((rng.next_u64() >> 12) as f64) + 0.5) * (1.0 / (1_u64 << 52) as f64)
    };
    loop {
        work.take(n * n + 1)?;
        if !bound.is_finite() {
            return Err(EventError::Numerical("nonfinite event intensity"));
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
        state.advance(proposal, &mut []);
        let total = evaluate(&state, &mut intensity);
        let mut draw = uniform(&mut rng) * bound;
        if draw < total {
            let mut kind = intensity
                .iter()
                .rposition(|&value| value > 0.0)
                .ok_or(EventError::Numerical("empty accepted event intensity"))?;
            for (i, &value) in intensity.iter().enumerate() {
                if draw < value {
                    kind = i;
                    break;
                }
                draw -= value;
            }
            if events.len() == max_events {
                return Err(EventError::Budget);
            }
            events.push(Arrival { time: proposal, kind });
            state.arrive(kind);
            bound = evaluate(&state, &mut intensity);
        } else {
            bound = total;
        }
    }
    Ok(events)
}

pub(super) fn rescaled_gaps(
    history: &EventHistory,
    parameters: &HawkesParameters,
    work: &mut Work<'_>,
) -> Result<Vec<f64>, EventError> {
    let n = history.catalog.len();
    work.take((history.left.len() + history.events.len() + 1) * n * n)?;
    let mut state =
        HistoryState::at(history, &parameters.decay, history.window.start, &history.left);
    let mut integral = vec![0.0; n * n];
    let mut gaps = Vec::with_capacity(history.events.len());
    let baseline = parameters.baseline.iter().sum::<f64>();
    for arrival in &history.events {
        let length = arrival.time - state.time;
        integral.fill(0.0);
        state.advance(arrival.time, &mut integral);
        let gap = baseline * length
            + integral.iter().zip(&parameters.branching).map(|(a, b)| a * b).sum::<f64>();
        if !gap.is_finite() {
            return Err(EventError::Numerical("nonfinite integrated event intensity"));
        }
        gaps.push(gap);
        state.arrive(arrival.kind);
    }
    Ok(gaps)
}
