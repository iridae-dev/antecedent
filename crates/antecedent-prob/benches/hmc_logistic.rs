//! Bayesian logistic regression by HMC: the nonlinear-likelihood first answer.
//!
//! The timed section is one complete `fit_hmc_glm` call on a Bernoulli-logit design: warmup, the
//! posterior draws, and the effective-sample-size, split-R-hat and divergence checks that gate
//! publication. A fit the gate refuses ends the run, so the section never times a point estimate
//! without its diagnostics. Every repeat is retained and the median is reported; the fastest run
//! is never selected. Prints the sample times and the diagnostics of the median fit.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Instant;

use antecedent_prob::{
    BayesDesignRef, BayesFitOptions, BayesLikelihood, GaussianCoefficientPrior, HmcOptions,
    LaplaceWorkspace, PriorSet, PriorSpec, fit_hmc_glm,
};

const ROWS: usize = 3000;
const RUNS: usize = 5;

struct Noise(u64);

impl Noise {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    fn gauss(&mut self) -> f64 {
        let (u, v) = (self.next().max(1e-12), self.next());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Column-major `[1, treatment, covariate]` design and a binary outcome from a logit model.
fn logistic_data(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Noise(seed);
    let mut x = vec![1.0; 3 * n];
    let mut y = vec![0.0; n];
    for r in 0..n {
        let z = rng.gauss();
        let t = f64::from(rng.next() < sigmoid(z));
        x[n + r] = t;
        x[2 * n + r] = z;
        y[r] = f64::from(rng.next() < sigmoid(-0.5 + 1.2 * t + 0.8 * z));
    }
    (x, y)
}

fn main() {
    let (x, y) = logistic_data(ROWS, 5);
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::GaussianCoefficients(
        GaussianCoefficientPrior::shared(3, 0.0, 100.0).unwrap(),
    ));
    let design = BayesDesignRef {
        x_colmajor: &x,
        nrows: ROWS,
        ncols: 3,
        y: &y,
        weights: None,
        offsets: None,
    };
    let opts = BayesFitOptions { n_draws: 4000, seed: 5, max_iter: 50, grad_tol: 1e-8 };
    let hmc = HmcOptions::default();

    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let started = Instant::now();
        let fit = fit_hmc_glm(
            BayesLikelihood::BernoulliLogit,
            design,
            &prior,
            &opts,
            hmc,
            &mut LaplaceWorkspace::default(),
        )
        .expect("the HMC publication gate refused the logistic fit");
        samples.push((started.elapsed(), fit));
    }
    println!(
        "hmc_logistic control=nonlinear_bayes rows={ROWS} samples_ns={:?}",
        samples.iter().map(|(time, _)| time.as_nanos()).collect::<Vec<_>>()
    );
    samples.sort_by_key(|(time, _)| *time);
    let (elapsed, fit) = samples.remove(RUNS / 2);
    let d = &fit.diagnostics;
    println!(
        "hmc_logistic median {elapsed:>9.1?} map={:?} ess_bulk_min={:?} ess_tail_min={:?} \
         rhat_max={:?} postwarmup_divergences={:?} all_chains_moved={:?}",
        fit.map,
        d.ess_bulk_min,
        d.ess_tail_min,
        d.rhat_max,
        d.n_postwarmup_divergences,
        d.all_chains_moved
    );
}
