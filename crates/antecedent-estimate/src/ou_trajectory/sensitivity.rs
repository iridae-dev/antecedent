// SPDX-License-Identifier: MIT OR Apache-2.0
//! Analytic score and Gauss–Newton information of the exact-transition Kalman likelihood.
//!
//! Derivatives of the transition maps are Fréchet derivatives of the matrix exponential from
//! the block identity `exp([[X, E], [0, X]]) = [[e^X, L(X, E)], [0, e^X]]`, applied to the
//! mean augmentation and to the van Loan covariance block. They are carried through the
//! Kalman recursion forward in time. The stationary initial law is differentiated through
//! the steady-state equations `M m = -b` and `M P + P M^T = -L L^T`.
use super::*;
use antecedent_kernels::gauss_jordan_solve;
use antecedent_model::continuous_cyclic::OrnsteinUhlenbeck;

/// Derivative of the baseline matrices with respect to one coordinate, zero on replaced rows.
struct DModel {
    dm: Vec<f64>,
    db: Vec<f64>,
    dl: Vec<f64>,
}

fn derivative_model(coordinate: &OuMechanismParameter, mask: &[bool], n: usize) -> DModel {
    let mut out = DModel { dm: vec![0.0; n * n], db: vec![0.0; n], dl: vec![0.0; n * n] };
    match *coordinate {
        OuMechanismParameter::Drift { equation, parent } if !mask[equation] => {
            out.dm[equation * n + parent] = 1.0;
        }
        OuMechanismParameter::Forcing { equation } if !mask[equation] => out.db[equation] = 1.0,
        OuMechanismParameter::Diffusion { equation, source } if !mask[equation] => {
            out.dl[equation * n + source] = 1.0;
        }
        _ => {}
    }
    out
}

fn diffusion(factor: &[f64], n: usize) -> Vec<f64> {
    matmul(factor, &transpose(factor, n, n), n, n, n)
}

/// Upper-right `d x d` block of `exp([[x, e], [0, x]])`: the Fréchet derivative of `exp` at `x`.
fn frechet(x: &[f64], e: &[f64], d: usize) -> Result<Vec<f64>, OuTrajectoryError> {
    let w = 2 * d;
    let mut big = vec![0.0; w * w];
    for i in 0..d {
        for j in 0..d {
            big[i * w + j] = x[i * d + j];
            big[(d + i) * w + d + j] = x[i * d + j];
            big[i * w + d + j] = e[i * d + j];
        }
    }
    let exp = expm(&big, w)?;
    let mut out = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..d {
            out[i * d + j] = exp[i * w + d + j];
        }
    }
    Ok(out)
}

fn augmented(drift: &[f64], forcing: &[f64], n: usize, h: f64) -> Vec<f64> {
    let w = n + 1;
    let mut out = vec![0.0; w * w];
    for i in 0..n {
        for j in 0..n {
            out[i * w + j] = drift[i * n + j] * h;
        }
        out[i * w + n] = forcing[i] * h;
    }
    out
}

/// Van Loan matrix `[[-M, D], [0, M^T]] h`.
fn loan(drift: &[f64], diff: &[f64], n: usize, h: f64) -> Vec<f64> {
    let w = 2 * n;
    let mut out = vec![0.0; w * w];
    for i in 0..n {
        for j in 0..n {
            out[i * w + j] = -drift[i * n + j] * h;
            out[(n + i) * w + n + j] = drift[j * n + i] * h;
            out[i * w + n + j] = diff[i * n + j] * h;
        }
    }
    out
}

struct Sens {
    dphi: Vec<Vec<f64>>,
    dshift: Vec<Vec<f64>>,
    dq: Vec<Vec<f64>>,
}

fn sens_transition(
    model: &OuMechanisms,
    dmodels: &[DModel],
    h: f64,
) -> Result<Sens, OuTrajectoryError> {
    let n = model.equations.len();
    let w = 2 * n;
    let factor = &model.diffusion_factor;
    let d = diffusion(factor, n);
    let base = expm(&loan(&model.drift, &d, n, h), w)?;
    let mut f12 = vec![0.0; n * n];
    let mut f22 = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            f12[i * n + j] = base[i * w + n + j];
            f22[i * n + j] = base[(n + i) * w + n + j];
        }
    }
    let aug = augmented(&model.drift, &model.forcing, n, h);
    let mut out = Sens { dphi: vec![], dshift: vec![], dq: vec![] };
    for dm in dmodels {
        let daug = augmented(&dm.dm, &dm.db, n, h);
        let mean = frechet(&aug, &daug, n + 1)?;
        let mut dphi = vec![0.0; n * n];
        let mut dshift = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                dphi[i * n + j] = mean[i * (n + 1) + j];
            }
            dshift[i] = mean[i * (n + 1) + n];
        }
        let ld = matmul(&dm.dl, &transpose(factor, n, n), n, n, n);
        let mut dd = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                dd[i * n + j] = ld[i * n + j] + ld[j * n + i];
            }
        }
        let cov = frechet(&loan(&model.drift, &d, n, h), &loan(&dm.dm, &dd, n, h), w)?;
        let mut df12 = vec![0.0; n * n];
        let mut df22 = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                df12[i * n + j] = cov[i * w + n + j];
                df22[i * n + j] = cov[(n + i) * w + n + j];
            }
        }
        let first = matmul(&transpose(&df22, n, n), &f12, n, n, n);
        let second = matmul(&transpose(&f22, n, n), &df12, n, n, n);
        let mut dq: Vec<f64> = first.iter().zip(&second).map(|(a, b)| a + b).collect();
        symmetrize_average(&mut dq, n);
        out.dphi.push(dphi);
        out.dshift.push(dshift);
        out.dq.push(dq);
    }
    Ok(out)
}

struct SensPropagator {
    n: usize,
    starts: Vec<f64>,
    models: Vec<OuMechanisms>,
    dmodels: Vec<Vec<DModel>>,
    cache: BTreeMap<(usize, u64), (Transition, Sens)>,
}

impl SensPropagator {
    fn new(
        template: &OuMechanisms,
        schedule: &[OuScheduleSegment],
        coordinates: &[OuMechanismParameter],
    ) -> Result<Self, OuTrajectoryError> {
        let n = template.equations.len();
        let mut starts = Vec::new();
        let mut models = Vec::new();
        let mut dmodels = Vec::new();
        for segment in schedule {
            let (model, mask) = template.replaced(&segment.replacements)?;
            starts.push(segment.start);
            dmodels.push(coordinates.iter().map(|c| derivative_model(c, &mask, n)).collect());
            models.push(model);
        }
        Ok(Self { n, starts, models, dmodels, cache: BTreeMap::new() })
    }

    fn advance(
        &mut self,
        from: f64,
        to: f64,
        mean: &mut Vec<f64>,
        cov: &mut Vec<f64>,
        dmean: &mut [Vec<f64>],
        dcov: &mut [Vec<f64>],
    ) -> Result<(), OuTrajectoryError> {
        let n = self.n;
        let mut time = from;
        while time < to {
            let k = self.starts.partition_point(|s| *s <= time).saturating_sub(1);
            let end = self.starts.get(k + 1).map_or(to, |next| next.min(to));
            let h = end - time;
            let key = (k, h.to_bits());
            if !self.cache.contains_key(&key) {
                let step = transition(&self.models[k], h, None)?;
                let sens = sens_transition(&self.models[k], &self.dmodels[k], h)?;
                self.cache.insert(key, (step, sens));
            }
            let (step, sens) = &self.cache[&key];
            let phi_t = transpose(&step.phi, n, n);
            for j in 0..dmean.len() {
                let mut next = matmul(&sens.dphi[j], mean, n, n, 1);
                let carried = matmul(&step.phi, &dmean[j], n, n, 1);
                for i in 0..n {
                    next[i] += carried[i] + sens.dshift[j][i];
                }
                let a = matmul(&matmul(&sens.dphi[j], cov, n, n, n), &phi_t, n, n, n);
                let b = matmul(&matmul(&step.phi, &dcov[j], n, n, n), &phi_t, n, n, n);
                let c = matmul(
                    &matmul(&step.phi, cov, n, n, n),
                    &transpose(&sens.dphi[j], n, n),
                    n,
                    n,
                    n,
                );
                let mut next_cov: Vec<f64> =
                    (0..n * n).map(|e| a[e] + b[e] + c[e] + sens.dq[j][e]).collect();
                symmetrize_average(&mut next_cov, n);
                dmean[j] = next;
                dcov[j] = next_cov;
            }
            let moved = matmul(&step.phi, mean, n, n, 1);
            for i in 0..n {
                mean[i] = moved[i] + step.shift[i];
            }
            let left = matmul(&step.phi, cov, n, n, n);
            let mut next = matmul(&left, &phi_t, n, n, n);
            for (value, q) in next.iter_mut().zip(&step.q) {
                *value += q;
            }
            symmetrize_average(&mut next, n);
            *cov = next;
            time = end;
        }
        Ok(())
    }
}

/// Stationary law of the first segment with its derivatives in every coordinate.
fn stationary_sensitivity(
    prop: &SensPropagator,
    ctx: &ExecutionContext,
) -> Result<(Vec<f64>, Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>), OuTrajectoryError> {
    let n = prop.n;
    let model = &prop.models[0];
    let d = diffusion(&model.diffusion_factor, n);
    let process = OrnsteinUhlenbeck {
        drift: model.drift.clone(),
        center: vec![0.0; n],
        diffusion: d.clone(),
    };
    let cov = process.stationary_covariance(ctx).map_err(OuTrajectoryError::from)?;
    let mut mean: Vec<f64> = model.forcing.iter().map(|b| -b).collect();
    let mut drift = model.drift.clone();
    if !gauss_jordan_solve(&mut drift, &mut mean, n, 1) {
        return Err(numerical("the drift is singular, so there is no stationary mean"));
    }
    let (mut dmean, mut dcov) = (Vec::new(), Vec::new());
    for dm in &prop.dmodels[0] {
        let moved = matmul(&dm.dm, &mean, n, n, 1);
        let mut rhs: Vec<f64> = (0..n).map(|i| -(moved[i] + dm.db[i])).collect();
        let mut a = model.drift.clone();
        if !gauss_jordan_solve(&mut a, &mut rhs, n, 1) {
            return Err(numerical("the drift is singular, so there is no stationary mean"));
        }
        dmean.push(rhs);
        let ld = matmul(&dm.dl, &transpose(&model.diffusion_factor, n, n), n, n, n);
        let left = matmul(&dm.dm, &cov, n, n, n);
        let mut input = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                input[i * n + j] =
                    left[i * n + j] + left[j * n + i] + ld[i * n + j] + ld[j * n + i];
            }
        }
        symmetrize_average(&mut input, n);
        dcov.push(process.lyapunov_direction(&input, ctx).map_err(OuTrajectoryError::from)?);
    }
    Ok((mean, cov, dmean, dcov))
}

impl Problem<'_> {
    /// Log likelihood, analytic score and Gauss–Newton information at `theta`.
    ///
    /// The information is `sum_t dv_j' S^-1 dv_k + (1/2) tr(S^-1 dS_j S^-1 dS_k)` over the
    /// innovations `v_t` with covariance `S_t`: the observed information with the terms of
    /// expectation zero removed, positive semidefinite by construction.
    pub(super) fn score_and_information(
        &self,
        theta: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<(f64, Vec<f64>, Vec<f64>), OuTrajectoryError> {
        let p = theta.len();
        let model = bind(self.template, self.coordinates, theta);
        let (n, m) = (self.n, self.m);
        let observation = self.observation;
        let noise: Vec<f64> = observation.noise_standard_deviation.iter().map(|s| s * s).collect();
        let h_transpose = transpose(&observation.matrix, m, n);
        let log_two_pi = (2.0 * std::f64::consts::PI).ln();
        let (mut total, mut score, mut info) = (0.0, vec![0.0; p], vec![0.0; p * p]);
        for trajectory in self.trajectories {
            let mut prop = SensPropagator::new(&model, &trajectory.schedule, self.coordinates)?;
            let (mut mean, mut cov, mut dmean, mut dcov) = match &trajectory.initial {
                OuInitialLaw::Stationary => stationary_sensitivity(&prop, ctx)?,
                OuInitialLaw::Gaussian { mean, covariance } => (
                    mean.clone(),
                    covariance.clone(),
                    vec![vec![0.0; n]; p],
                    vec![vec![0.0; n * n]; p],
                ),
            };
            let mut time = trajectory.schedule[0].start;
            for point in &trajectory.observations {
                if ctx.cancellation.is_cancelled() {
                    return Err(OuTrajectoryError::Resource("cancelled".into()));
                }
                prop.advance(time, point.time, &mut mean, &mut cov, &mut dmean, &mut dcov)?;
                time = point.time;
                let hp = matmul(&observation.matrix, &cov, m, n, n);
                let mut s = matmul(&hp, &h_transpose, m, n, m);
                for i in 0..m {
                    s[i * m + i] += noise[i];
                }
                symmetrize_average(&mut s, m);
                let factor = SpdFactor::new(&s, m).ok_or_else(|| {
                    numerical("the innovation covariance is not positive definite")
                })?;
                let predicted = matmul(&observation.matrix, &mean, m, n, 1);
                let v: Vec<f64> = (0..m)
                    .map(|i| point.values[i] - predicted[i] - observation.offset_at(i))
                    .collect();
                let sinv = factor.inverse();
                let w = matmul(&sinv, &v, m, m, 1);
                total -= 0.5 * (m as f64 * log_two_pi + factor.log_det() + dot(&v, &w));
                let gain = matmul(&transpose(&hp, m, n), &sinv, n, m, m);
                // Per-coordinate derivatives of the innovation and its covariance.
                let mut dv = Vec::with_capacity(p);
                let mut ds_sinv = Vec::with_capacity(p);
                for j in 0..p {
                    let dpred = matmul(&observation.matrix, &dmean[j], m, n, 1);
                    dv.push(dpred.iter().map(|x| -x).collect::<Vec<f64>>());
                    let ds = matmul(
                        &matmul(&observation.matrix, &dcov[j], m, n, n),
                        &h_transpose,
                        m,
                        n,
                        m,
                    );
                    ds_sinv.push((ds.clone(), matmul(&sinv, &ds, m, m, m)));
                }
                for j in 0..p {
                    let (ds, a) = (&ds_sinv[j].0, &ds_sinv[j].1);
                    let trace: f64 = (0..m).map(|i| a[i * m + i]).sum();
                    let quad = dot(&w, &matmul(ds, &w, m, m, 1));
                    score[j] += -0.5 * (trace + 2.0 * dot(&w, &dv[j]) - quad);
                    let sinv_dvj = matmul(&sinv, &dv[j], m, m, 1);
                    for k in 0..=j {
                        let b = &ds_sinv[k].1;
                        let cross: f64 = (0..m)
                            .map(|a_i| {
                                (0..m).map(|b_i| a[a_i * m + b_i] * b[b_i * m + a_i]).sum::<f64>()
                            })
                            .sum();
                        let value = dot(&sinv_dvj, &dv[k]) + 0.5 * cross;
                        info[j * p + k] += value;
                        if k != j {
                            info[k * p + j] += value;
                        }
                    }
                }
                // Measurement update, differentiated in the non-Joseph form (identical at the
                // optimal gain), then the likelihood path's own state is advanced identically.
                let kv = matmul(&gain, &v, n, m, 1);
                let gain_h = matmul(&gain, &observation.matrix, n, m, n);
                let mut complement = identity(n);
                for (c, g) in complement.iter_mut().zip(&gain_h) {
                    *c -= g;
                }
                let predicted_cov = cov.clone();
                for j in 0..p {
                    let dgain = matmul(
                        &{
                            let left = matmul(&dcov[j], &h_transpose, n, n, m);
                            let right = matmul(&gain, &ds_sinv[j].0, n, m, m);
                            left.iter().zip(&right).map(|(l, r)| l - r).collect::<Vec<f64>>()
                        },
                        &sinv,
                        n,
                        m,
                        m,
                    );
                    let mut next_mean = matmul(&dgain, &v, n, m, 1);
                    let from_gain = matmul(&gain, &dv[j], n, m, 1);
                    for i in 0..n {
                        next_mean[i] += dmean[j][i] + from_gain[i];
                    }
                    let first = matmul(&complement, &dcov[j], n, n, n);
                    let second = matmul(
                        &matmul(&dgain, &observation.matrix, n, m, n),
                        &predicted_cov,
                        n,
                        n,
                        n,
                    );
                    let mut next_cov: Vec<f64> =
                        first.iter().zip(&second).map(|(a, b)| a - b).collect();
                    symmetrize_average(&mut next_cov, n);
                    dmean[j] = next_mean;
                    dcov[j] = next_cov;
                }
                for i in 0..n {
                    mean[i] += kv[i];
                }
                let left = matmul(&complement, &predicted_cov, n, n, n);
                let mut updated = matmul(&left, &transpose(&complement, n, n), n, n, n);
                for i in 0..n {
                    for j in 0..n {
                        updated[i * n + j] += (0..m)
                            .map(|l| gain[i * m + l] * noise[l] * gain[j * m + l])
                            .sum::<f64>();
                    }
                }
                symmetrize_average(&mut updated, n);
                cov = updated;
            }
        }
        if !total.is_finite() || score.iter().chain(&info).any(|v| !v.is_finite()) {
            return Err(numerical("non-finite OU trajectory score"));
        }
        Ok((total, score, info))
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
