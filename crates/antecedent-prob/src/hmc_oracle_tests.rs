//! Hamiltonian Monte Carlo against closed forms on a Gaussian target.
//!
//! A unit-variance Gaussian makes the leapfrog map linear, so its exact `L`-step image is a
//! matrix power written out here independently of the integrator; and the target is known, so
//! the invariance of one full transition (fresh momentum, leapfrog, Metropolis correction) can
//! be checked against its moments and distribution function.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// Standard normal in one dimension, `log pi = -q^2 / 2`.
struct StdNormal1;

impl PosteriorTarget for StdNormal1 {
    fn dim(&self) -> usize {
        1
    }

    fn logp_and_grad(&mut self, q: &[f64], grad: &mut [f64]) -> Result<f64, ProbError> {
        grad[0] = -q[0];
        Ok(-0.5 * q[0] * q[0])
    }
}

/// For a unit-variance Gaussian the leapfrog map is linear, so `L` steps are the `L`-th power
/// of one explicit 2x2 matrix. The implementation, including its non-unit inverse mass and
/// carried-gradient first half kick, must reproduce that matrix power to rounding.
#[test]
fn leapfrog_is_the_exact_linear_map_on_a_gaussian() {
    for &(eps, steps, inv_m) in &[
        (0.1_f64, 1u32, 1.0_f64),
        (0.37, 5, 1.0),
        (0.37, 9, 0.25),
        (1.2, 7, 2.5),
        (0.05, 200, 1.0),
    ] {
        let (q0, p0) = (0.8_f64, -1.3_f64);
        // kick(eps/2) drift(eps m^-1) kick(eps/2) acting on (q, p) with grad U = q
        let (a, b) = (eps * 0.5, eps * inv_m);
        let kick = [[1.0, 0.0], [-a, 1.0]];
        let drift = [[1.0, b], [0.0, 1.0]];
        let mul = |x: [[f64; 2]; 2], y: [[f64; 2]; 2]| {
            let mut z = [[0.0; 2]; 2];
            for i in 0..2 {
                for j in 0..2 {
                    z[i][j] = x[i][0] * y[0][j] + x[i][1] * y[1][j];
                }
            }
            z
        };
        let one = mul(kick, mul(drift, kick));
        let mut total = [[1.0, 0.0], [0.0, 1.0]];
        for _ in 0..steps {
            total = mul(one, total);
        }
        let want_q = total[0][0] * q0 + total[0][1] * p0;
        let want_p = total[1][0] * q0 + total[1][1] * p0;

        let mut target = StdNormal1;
        let mut gaussian = GaussianRef(&mut target);
        let mut ws = LaplaceWorkspace::default();
        ws.prepare(1, 1, 1);
        ws.q[0] = q0;
        ws.p[0] = p0;
        let inv_mass = [inv_m];
        let path = Trajectory { step_size: eps, leapfrog_steps: steps, inv_mass: &inv_mass };
        let last = leapfrog(&mut gaussian, 1, path, &[-q0], -0.5 * q0 * q0, &mut ws)
            .unwrap()
            .expect("a Gaussian trajectory at these steps is finite");
        let scale = 1.0 + want_q.abs() + want_p.abs();
        assert!((ws.q[0] - want_q).abs() <= 1e-12 * scale, "q {} vs {want_q}", ws.q[0]);
        assert!((ws.p[0] - want_p).abs() <= 1e-12 * scale, "p {} vs {want_p}", ws.p[0]);
        assert!((last + 0.5 * want_q * want_q).abs() <= 1e-12 * scale * scale);
    }
}

/// A single transition (fresh momentum, leapfrog, Metropolis correction) leaves the target
/// invariant: started from exact `N(0, 1)` draws, the states after one step are `N(0, 1)`.
/// The step is large enough that roughly half the proposals are rejected, so a missing or
/// miscomputed correction would visibly inflate the variance.
#[test]
fn one_transition_leaves_the_gaussian_target_invariant() {
    for &(eps, steps, inv_m) in &[(1.9_f64, 3u32, 1.0_f64), (3.0, 2, 0.3), (1.8, 5, 1.0)] {
        let n = 200_000usize;
        let mut rng = CausalRng::from_seed(2024);
        let mut target = StdNormal1;
        let mut ws = LaplaceWorkspace::default();
        ws.prepare(1, 1, 1);
        let inv_mass = [inv_m];
        let (mut m1, mut m2, mut m4, mut accepts) = (0.0, 0.0, 0.0, 0usize);
        let mut below = [0usize; 3];
        let cuts = [-1.0_f64, 0.0, 1.0];
        for _ in 0..n {
            let q0 = standard_normal(&mut rng);
            let step = hmc_step_target(
                &mut target,
                &[q0],
                -0.5 * q0 * q0,
                &[-q0],
                Trajectory { step_size: eps, leapfrog_steps: steps, inv_mass: &inv_mass },
                &mut ws,
                &mut rng,
            )
            .unwrap();
            let q1 = if step.accepted { ws.q[0] } else { q0 };
            accepts += usize::from(step.accepted);
            m1 += q1;
            m2 += q1 * q1;
            m4 += q1.powi(4);
            for (c, b) in cuts.iter().zip(below.iter_mut()) {
                *b += usize::from(q1 <= *c);
            }
        }
        let nf = n as f64;
        let rate = accepts as f64 / nf;
        assert!((0.1..0.96).contains(&rate), "acceptance {rate} should exercise rejection");
        // Moments of N(0,1): 0, 1, 3 with standard errors 1/sqrt(n), sqrt(2/n), sqrt(96/n);
        // the draws within one step are dependent only through the accepted state, so six
        // standard errors is ample.
        assert!((m1 / nf).abs() < 6.0 / nf.sqrt(), "mean {}", m1 / nf);
        assert!((m2 / nf - 1.0).abs() < 6.0 * (2.0 / nf).sqrt(), "variance {}", m2 / nf);
        assert!((m4 / nf - 3.0).abs() < 6.0 * (96.0 / nf).sqrt(), "fourth moment {}", m4 / nf);
        // Distribution function at -1, 0, 1: Phi = 0.158655253931457, 0.5, 0.841344746068543.
        for (b, want) in below.iter().zip([0.158_655_253_931_457, 0.5, 0.841_344_746_068_543]) {
            let got = *b as f64 / nf;
            let se = (want * (1.0 - want) / nf).sqrt();
            assert!((got - want).abs() < 6.0 * se, "cdf {got} vs {want}");
        }
    }
}
