//! Warmup scheduling and diagonal metric adaptation for HMC.

use antecedent_kernels::Welford;

use super::{HmcOptions, MIN_METRIC_WARMUP};

/// Slow-window schedule of metric adaptation over `n_warmup` iterations, as in Stan: a fast
/// initial buffer, then windows that double in length (each re-estimating the metric), then a
/// fast terminal buffer for the final step-size adaptation. Warmups too short for that fall
/// back to one window between proportional buffers. Returns `[start, end)` per window.
pub(super) fn metric_windows(n_warmup: usize) -> Vec<(usize, usize)> {
    if n_warmup < MIN_METRIC_WARMUP {
        return Vec::new();
    }
    let (mut init, mut term, mut size) = (75usize, 50usize, 25usize);
    if init + size + term > n_warmup {
        init = n_warmup * 15 / 100;
        term = n_warmup * 10 / 100;
        size = n_warmup - init - term;
    }
    let slow_end = n_warmup - term;
    let mut windows = Vec::new();
    let mut start = init;
    loop {
        let mut end = start + size;
        // A following doubled window that would not fit is absorbed into this one.
        if size > (slow_end - end) / 2 {
            end = slow_end;
        }
        windows.push((start, end));
        if end == slow_end {
            return windows;
        }
        start = end;
        size *= 2;
    }
}

/// Diagonal inverse-mass estimator: per-coordinate variance of the chain's own states over
/// each slow window (Welford accumulation, restarted for every window).
pub(super) struct DiagonalMetric {
    pub(super) inv_mass: Vec<f64>,
    windows: Vec<(usize, usize)>,
    cursor: usize,
    /// One unit-weight accumulator per coordinate. Every `observe` pushes each
    /// coordinate, so the masses stay equal to the old shared count.
    acc: Vec<Welford>,
}

impl DiagonalMetric {
    pub(super) fn new(dim: usize, hmc: HmcOptions) -> Self {
        let windows = if hmc.adapt_metric { metric_windows(hmc.n_warmup) } else { Vec::new() };
        Self {
            inv_mass: vec![1.0 / hmc.mass; dim],
            windows,
            cursor: 0,
            acc: vec![Welford::new(); dim],
        }
    }

    /// Feed the state after warmup transition `t`.
    pub(super) fn observe(&mut self, t: usize, q: &[f64]) {
        let Some(&(start, end)) = self.windows.get(self.cursor) else { return };
        if t < start || t >= end {
            return;
        }
        for (acc, &v) in self.acc.iter_mut().zip(q) {
            acc.push(v);
        }
    }

    /// After the last transition of a slow window, install its estimate (regularised toward a
    /// small floor, `n/(n+5) var + 1e-3 · 5/(n+5)`, as Stan does) and open the next window.
    /// Returns whether the metric changed, so the caller restarts step-size adaptation.
    pub(super) fn install_if_window_closed(&mut self, t: usize) -> bool {
        let Some(&(_, end)) = self.windows.get(self.cursor) else { return false };
        if t + 1 != end {
            return false;
        }
        self.cursor += 1;
        let n = self.acc.first().map_or(0.0, Welford::mass);
        let installed = n >= 3.0 && {
            let weight = n / (n + 5.0);
            let proposal: Vec<f64> = self
                .acc
                .iter()
                .map(|acc| weight * (acc.m2() / (n - 1.0)) + (1.0 - weight) * 1e-3)
                .collect();
            let usable = proposal.iter().all(|v| v.is_finite() && *v > 0.0);
            if usable {
                self.inv_mass = proposal.into_iter().map(|v| v.clamp(1e-12, 1e12)).collect();
            }
            usable
        };
        for acc in &mut self.acc {
            *acc = Welford::new();
        }
        installed
    }
}
