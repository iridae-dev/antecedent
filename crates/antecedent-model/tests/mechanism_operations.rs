//! A user-defined mechanism family that cannot recover its noise is refused for abduction
//! by name, while its sampling and likelihood stay available.
//!
//! Known truth: `y = 0.5 + 1.2 x^2 + 0.3 e` with `x ~ N(0, 1)` and `e ~ N(0, 1)`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "evaluation replays the literal structural equation")]

use std::sync::Arc;

use antecedent_core::CausalRng;
use antecedent_kernels::standard_normal;
use antecedent_model::{
    DynamicMechanism, MechanismSlot, MechanismWorkspace, ModelError, ParentBatch, evaluate_column,
    infer_noise_column_rng, log_prob_column, sample_noise_column,
};

const ROWS: usize = 2000;
const INTERCEPT: f64 = 0.5;
const CURVATURE: f64 = 1.2;
const SIGMA: f64 = 0.3;

/// The quadratic mechanism; `recoverable` states whether abduction is declared supported.
struct Quadratic {
    recoverable: bool,
}

impl DynamicMechanism for Quadratic {
    fn sample_noise_column(
        &self,
        n_rows: usize,
        rng: &mut CausalRng,
        output: &mut [f64],
    ) -> Result<(), ModelError> {
        for slot in &mut output[..n_rows] {
            *slot = SIGMA * standard_normal(rng);
        }
        Ok(())
    }

    fn evaluate_column(
        &self,
        parents: ParentBatch<'_>,
        noise: &[f64],
        output: &mut [f64],
        _workspace: &mut MechanismWorkspace,
    ) -> Result<(), ModelError> {
        let x = parents.column(0)?;
        for ((out, &x), &e) in output.iter_mut().zip(x).zip(noise) {
            *out = INTERCEPT + CURVATURE * x * x + e;
        }
        Ok(())
    }

    fn supports_noise_inference(&self) -> bool {
        self.recoverable
    }

    fn log_prob_column(
        &self,
        values: &[f64],
        parents: ParentBatch<'_>,
        output: &mut [f64],
    ) -> Result<(), ModelError> {
        let x = parents.column(0)?;
        for ((out, &y), &x) in output.iter_mut().zip(values).zip(x) {
            let resid = y - INTERCEPT - CURVATURE * x * x;
            *out = -0.5 * (resid / SIGMA).powi(2)
                - SIGMA.ln()
                - 0.5 * (2.0 * std::f64::consts::PI).ln();
        }
        Ok(())
    }
}

fn slot(recoverable: bool) -> MechanismSlot {
    MechanismSlot::Dynamic { id: Arc::from("y"), mechanism: Arc::new(Quadratic { recoverable }) }
}

fn parent_column(rng: &mut CausalRng) -> Vec<f64> {
    (0..ROWS).map(|_| standard_normal(rng)).collect()
}

#[test]
fn a_mechanism_without_noise_inference_is_refused_for_abduction_but_samples_and_scores() {
    let mut rng = CausalRng::from_seed(3_465_696);
    let x = parent_column(&mut rng);
    let parents = ParentBatch { n_rows: ROWS, n_parents: 1, values: &x };
    let no_inference = slot(false);

    // Sampling and evaluation reproduce the known truth draw for draw.
    let mut noise = vec![0.0; ROWS];
    sample_noise_column(&no_inference, ROWS, &mut rng, &mut noise).unwrap();
    let mut y = vec![0.0; ROWS];
    let mut workspace = MechanismWorkspace::default();
    evaluate_column(&no_inference, parents, &noise, &mut y, &mut workspace).unwrap();
    for ((&y, &x), &e) in y.iter().zip(&x).zip(&noise) {
        assert_eq!(y, INTERCEPT + CURVATURE * x * x + e);
    }

    // The likelihood is the independent Gaussian log-density of the residuals.
    let mut log_prob = vec![0.0; ROWS];
    log_prob_column(&no_inference, &y, parents, &mut log_prob).unwrap();
    for (&lp, &e) in log_prob.iter().zip(&noise) {
        let oracle =
            -0.5 * (e / SIGMA).powi(2) - SIGMA.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((lp - oracle).abs() < 1e-9, "{lp} vs {oracle}");
    }

    // Abduction is the unsupported operation, and the refusal names it and the mechanism.
    let mut recovered = vec![0.0; ROWS];
    let error =
        infer_noise_column_rng(&no_inference, &y, parents, &mut recovered, &mut rng).unwrap_err();
    let ModelError::Unsupported { message } = &error else {
        panic!("expected an unsupported-operation refusal, got {error:?}");
    };
    assert!(message.contains("mechanism operation unsupported"), "{message}");
    assert!(message.contains("noise inference"), "{message}");
    assert!(message.contains("`y`"), "{message}");
    assert!(message.contains("likelihood"), "{message}");
}

#[test]
fn the_same_family_declared_invertible_recovers_the_noise_it_sampled() {
    let mut rng = CausalRng::from_seed(3_465_697);
    let x = parent_column(&mut rng);
    let parents = ParentBatch { n_rows: ROWS, n_parents: 1, values: &x };
    let invertible = slot(true);
    let mut noise = vec![0.0; ROWS];
    sample_noise_column(&invertible, ROWS, &mut rng, &mut noise).unwrap();
    let mut y = vec![0.0; ROWS];
    evaluate_column(&invertible, parents, &noise, &mut y, &mut MechanismWorkspace::default())
        .unwrap();
    let mut recovered = vec![0.0; ROWS];
    infer_noise_column_rng(&invertible, &y, parents, &mut recovered, &mut rng).unwrap();
    for (&got, &truth) in recovered.iter().zip(&noise) {
        assert!((got - truth).abs() < 1e-12);
    }
}
