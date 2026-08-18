//! Particle-filter state cache — bootstrap/SIR for 1-D LGSSM.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CausalRng;

use crate::error::StateError;
use crate::retention::RetentionPolicy;

/// Linear-Gaussian state-space model: `x_t = a x_{t-1} + σ_proc ε`, `y_t = x_t + σ_obs η`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LgssmParams {
    /// Autoregressive coefficient.
    pub a: f64,
    /// Process noise std.
    pub process_std: f64,
    /// Observation noise std.
    pub obs_std: f64,
}

impl Default for LgssmParams {
    fn default() -> Self {
        Self { a: 0.9, process_std: 0.3, obs_std: 0.5 }
    }
}

/// Reusable per-step scratch: normalized weights and the resampling target
/// buffer. A cache only — never part of the filter's semantic state.
#[derive(Clone, Debug, Default)]
struct StepScratch {
    /// Normalized weights, refreshed once per [`ParticleFilterState::step`].
    weights: Vec<f64>,
    /// Resampling target; swapped with `particles` instead of reallocating.
    particles: Vec<f64>,
}

/// Serializable particle-filter state (no borrowed buffers / callbacks).
#[derive(Clone, Debug)]
pub struct ParticleFilterState {
    /// Particle count.
    pub n_particles: usize,
    /// Latent particle values.
    pub particles: Vec<f64>,
    /// Unnormalized log-weights.
    pub log_weights: Vec<f64>,
    /// Observations incorporated so far.
    pub n_obs: u64,
    /// Data catalog version stamp.
    pub data_version: u64,
    /// Model parameters.
    pub params: LgssmParams,
    /// Opaque [`CausalRng`] continuation state.
    pub rng_state: u64,
    /// Retention.
    pub retention: RetentionPolicy,
    /// Per-step scratch buffers (excluded from equality; see [`StepScratch`]).
    scratch: StepScratch,
}

// Manual impl so the scratch cache never participates in state equality.
impl PartialEq for ParticleFilterState {
    fn eq(&self, other: &Self) -> bool {
        self.n_particles == other.n_particles
            && self.particles == other.particles
            && self.log_weights == other.log_weights
            && self.n_obs == other.n_obs
            && self.data_version == other.data_version
            && self.params == other.params
            && self.rng_state == other.rng_state
            && self.retention == other.retention
    }
}

impl ParticleFilterState {
    /// Initialize `n_particles` from `N(0, process_std²)` using `seed`.
    ///
    /// # Errors
    ///
    /// Zero particles or non-positive noise scales.
    pub fn init(
        n_particles: usize,
        params: LgssmParams,
        data_version: u64,
        seed: u64,
    ) -> Result<Self, StateError> {
        if n_particles == 0 {
            return Err(StateError::Shape("n_particles must be > 0".into()));
        }
        if params.process_std <= 0.0 || params.obs_std <= 0.0 {
            return Err(StateError::Numerical("noise std must be positive".into()));
        }
        let mut rng = CausalRng::from_seed(seed);
        let mut particles = Vec::with_capacity(n_particles);
        for _ in 0..n_particles {
            particles.push(params.process_std * standard_normal(&mut rng));
        }
        Ok(Self {
            n_particles,
            particles,
            log_weights: vec![0.0; n_particles],
            n_obs: 0,
            data_version,
            params,
            rng_state: rng.state(),
            retention: RetentionPolicy::BoundedWindow { max_rows: n_particles as u64 },
            scratch: StepScratch::default(),
        })
    }

    /// Effective sample size `1 / Σ w²` with normalized weights.
    #[must_use]
    pub fn ess(&self) -> f64 {
        let weights = normalized_weights(&self.log_weights);
        let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
        if sum_sq <= 0.0 { 0.0 } else { 1.0 / sum_sq }
    }

    /// Weighted mean of the latent particles.
    #[must_use]
    pub fn weighted_mean(&self) -> f64 {
        let weights = normalized_weights(&self.log_weights);
        weights.iter().zip(self.particles.iter()).map(|(w, x)| w * x).sum()
    }

    /// One bootstrap-filter step: predict → update with `y` → resample if ESS low.
    ///
    /// # Errors
    ///
    /// Numerical failures.
    pub fn step(&mut self, y: f64) -> Result<(), StateError> {
        let mut rng = CausalRng::from_state(self.rng_state);
        for p in &mut self.particles {
            *p = self.params.a * *p + self.params.process_std * standard_normal(&mut rng);
        }
        let inv_var = 1.0 / (self.params.obs_std * self.params.obs_std);
        let log_norm = -0.5 * (2.0 * std::f64::consts::PI).ln() - self.params.obs_std.ln();
        for i in 0..self.n_particles {
            let err = y - self.particles[i];
            self.log_weights[i] += log_norm - 0.5 * err * err * inv_var;
        }
        self.n_obs = self.n_obs.saturating_add(1);
        // Normalize once per step; the ESS check and (when triggered) the
        // resample below share this buffer instead of re-exponentiating.
        normalized_weights_into(&self.log_weights, &mut self.scratch.weights);
        let sum_sq: f64 = self.scratch.weights.iter().map(|w| w * w).sum();
        let ess = if sum_sq <= 0.0 { 0.0 } else { 1.0 / sum_sq };
        if ess < 0.5 * self.n_particles as f64 {
            systematic_resample(self, &mut rng);
        }
        self.rng_state = rng.state();
        Ok(())
    }

    /// Run the filter over a full observation sequence from a fresh init (batch oracle).
    ///
    /// # Errors
    ///
    /// Init / step failures.
    pub fn run_batch(
        observations: &[f64],
        n_particles: usize,
        params: LgssmParams,
        data_version: u64,
        seed: u64,
    ) -> Result<Self, StateError> {
        let mut state = Self::init(n_particles, params, data_version, seed)?;
        for &y in observations {
            state.step(y)?;
        }
        Ok(state)
    }
}

fn normalized_weights(log_w: &[f64]) -> Vec<f64> {
    let mut w = Vec::new();
    normalized_weights_into(log_w, &mut w);
    w
}

/// Allocation-free core of [`normalized_weights`]: same arithmetic, writing
/// into a caller-owned buffer.
fn normalized_weights_into(log_w: &[f64], out: &mut Vec<f64>) {
    let max = log_w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    out.clear();
    if !max.is_finite() {
        let u = 1.0 / log_w.len().max(1) as f64;
        out.resize(log_w.len(), u);
        return;
    }
    out.extend(log_w.iter().map(|lw| (lw - max).exp()));
    let sum: f64 = out.iter().sum::<f64>().max(1e-300);
    for wi in out {
        *wi /= sum;
    }
}

/// Resample using the normalized weights already in `state.scratch.weights`
/// (refreshed by the caller this step); swaps the particle buffers and zeroes
/// `log_weights` in place instead of reallocating either.
fn systematic_resample(state: &mut ParticleFilterState, rng: &mut CausalRng) {
    let n = state.n_particles;
    let u0 = rng.next_f64() / n as f64;
    {
        let StepScratch { weights: w, particles: new_particles } = &mut state.scratch;
        new_particles.clear();
        new_particles.resize(n, 0.0);
        let mut cum = w[0];
        let mut i = 0usize;
        for j in 0..n {
            let target = u0 + j as f64 / n as f64;
            while target > cum && i + 1 < n {
                i += 1;
                cum += w[i];
            }
            new_particles[j] = state.particles[i];
        }
    }
    std::mem::swap(&mut state.particles, &mut state.scratch.particles);
    state.log_weights.fill(0.0);
}

fn standard_normal(rng: &mut CausalRng) -> f64 {
    let u1 = rng.next_f64().max(1e-12);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_obs(n: usize, seed: u64) -> Vec<f64> {
        let params = LgssmParams::default();
        let mut rng = CausalRng::from_seed(seed);
        let mut x = 0.0;
        let mut ys = Vec::with_capacity(n);
        for _ in 0..n {
            x = params.a * x + params.process_std * standard_normal(&mut rng);
            ys.push(x + params.obs_std * standard_normal(&mut rng));
        }
        ys
    }

    #[test]
    fn stepwise_matches_batch() {
        let ys = synth_obs(30, 7);
        let params = LgssmParams::default();
        let batch = ParticleFilterState::run_batch(&ys, 64, params, 1, 99).unwrap();
        let mut step = ParticleFilterState::init(64, params, 1, 99).unwrap();
        for &y in &ys {
            step.step(y).unwrap();
        }
        assert_eq!(step.n_obs, batch.n_obs);
        assert!((step.weighted_mean() - batch.weighted_mean()).abs() < 1e-10);
        assert!((step.ess() - batch.ess()).abs() < 1e-10);
        for i in 0..step.n_particles {
            assert!((step.particles[i] - batch.particles[i]).abs() < 1e-10);
            assert!((step.log_weights[i] - batch.log_weights[i]).abs() < 1e-10);
        }
    }
}
