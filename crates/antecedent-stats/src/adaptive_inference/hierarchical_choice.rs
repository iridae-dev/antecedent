//! Hierarchical posterior of repeated categorical choices: respondent-level heterogeneity in the
//! choice shares of two or three options.
//!
//! Respondent `i` makes `m_i` choices among `K` options and contributes the counts `n_i`. Their
//! own choice shares are `theta_i ~ Dirichlet(kappa pi)`: `pi` is the population mean share
//! vector, `kappa` the concentration (large `kappa` is a homogeneous population, small `kappa`
//! one whose respondents lean strongly to different options). The likelihood of the ordered
//! choices of respondent `i` given `(pi, kappa)` is the Dirichlet-multinomial sequence law
//!
//! ```text
//! p(n_i | pi, kappa) = prod_k Gamma(kappa pi_k + n_ik) / Gamma(kappa pi_k)
//!                      * Gamma(kappa) / Gamma(kappa + m_i),
//! ```
//!
//! respondents being independent given `(pi, kappa)`. The prior is `pi ~ Dirichlet(a)` with
//! every `a_k >= 1` and `ln kappa ~ Normal(mu, sigma^2)`, both declared. With two options the
//! kernel is the beta-binomial of [`super::participant_posterior`] with
//! `rho = 1 / (kappa + 1)`.
//!
//! Everything reported is deterministic quadrature of the posterior of `(pi, kappa)`, not
//! simulation. Option `k` is best where `pi_k >= pi_j` for every `j`; in the coordinates
//! `pi_k = c` and, for three options, `(pi_j, pi_l) = (1 - c)(r, 1 - r)`, that region is
//! `c >= 1/2` for any `r` and, for `1/3 <= c < 1/2`, `r` between `(1 - 2c)/(1 - c)` and
//! `c/(1 - c)`. The integrand is smooth on each piece, so Gauss-Legendre panels cut at those
//! breakpoints integrate the indicator exactly instead of smearing it. A box around the posterior
//! mass is found by repeated zooming; the reported `quadrature_error` is the largest change in
//! any reported quantity between a three-panel and a five-panel rule, and a posterior whose
//! change exceeds `1e-8` is refused rather than reported.
//!
//! The population shares and the respondent heterogeneity are statements about the sampled
//! population in the declared choice set. Stated choices are not an identified causal or
//! behavioural effect. No coverage or calibration claim is made about any interval built from the
//! posterior.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use super::{AdaptiveInferenceError as Error, composite_rule, graded_rule, panel_rule};
use crate::special::ln_gamma;

/// Most options the hierarchical family is verified for.
pub const MAX_HIERARCHICAL_OPTIONS: usize = 3;
/// Most choices one respondent may contribute.
pub const MAX_HIERARCHICAL_CHOICES: u32 = 512;
/// Most distinct respondent count vectors one posterior may hold.
pub const MAX_DISTINCT_HISTOGRAMS: usize = 4096;
/// Largest change between the coarse and the fine rule that is reported.
pub const QUADRATURE_ERROR_LIMIT: f64 = 1e-8;

const ZOOM_GRID: usize = 24;
const ZOOM_ITERATIONS: usize = 10;
const ZOOM_CUT: f64 = 55.0;
const ZOOM_SHRINK: f64 = 0.75;
/// Coarse and fine panel counts tried in turn until they agree to `QUADRATURE_ERROR_LIMIT`.
const PANEL_SCHEDULE: [(usize, usize); 3] = [(3, 5), (5, 8), (8, 12)];
const V_LIMIT: (f64, f64) = (-8.0, 13.0);
const CANCEL_STRIDE: usize = 4096;

/// The declared prior of a hierarchical choice model.
#[derive(Clone, Debug, PartialEq)]
pub struct HierarchicalChoicePrior {
    /// Dirichlet shapes of the population mean shares, each at least one.
    pub shapes: Vec<f64>,
    /// Mean of `ln kappa`.
    pub log_concentration_mean: f64,
    /// Standard deviation of `ln kappa`; positive.
    pub log_concentration_sd: f64,
}

/// What the posterior reports.
#[derive(Clone, Debug, PartialEq)]
pub struct HierarchicalSummary {
    /// Posterior mean of each population share.
    pub mean: Vec<f64>,
    /// Posterior probability that each option has the largest population share.
    pub probability_best: Vec<f64>,
    /// Posterior expected loss, in share, of selecting each option.
    pub expected_loss: Vec<f64>,
    /// Posterior mean of `ln kappa`.
    pub log_concentration_mean: f64,
    /// Posterior standard deviation of `ln kappa`.
    pub log_concentration_sd: f64,
    /// Posterior mean of the intra-respondent correlation `1 / (kappa + 1)`.
    pub correlation_mean: f64,
    /// Log marginal likelihood of the ordered choices under the declared prior.
    pub log_marginal_likelihood: f64,
    /// Largest change in any reported quantity between the coarse and the fine rule.
    pub quadrature_error: f64,
}

/// The hierarchical posterior of a set of respondents' choice counts.
#[derive(Clone, Debug)]
pub struct HierarchicalChoicePosterior {
    options: usize,
    respondents: usize,
    shapes: Vec<f64>,
    mu: f64,
    sigma: f64,
    ln_dirichlet_norm: f64,
    histograms: Vec<(Vec<u32>, f64)>,
    max_n: Vec<usize>,
    max_m: usize,
}

#[derive(Clone, Copy)]
struct Region {
    option: usize,
    c_lo: f64,
    c_hi: f64,
    constrained: bool,
}

#[derive(Clone, Copy, Debug)]
struct Cube {
    lo: [f64; 3],
    hi: [f64; 3],
}

struct Scratch {
    prefix: Vec<Vec<f64>>,
    kappa: Vec<f64>,
    pi: [f64; 3],
}

impl HierarchicalChoicePosterior {
    /// Condition the declared prior on the respondents' count vectors.
    ///
    /// # Errors
    /// Fewer than two or more than three options, a prior outside its domain, counts of the wrong
    /// length, no respondent with more than one choice (heterogeneity is not identified), or a
    /// size bound exceeded.
    pub fn new(prior: &HierarchicalChoicePrior, counts: &[Vec<u32>]) -> Result<Self, Error> {
        let k = prior.shapes.len();
        if !(2..=MAX_HIERARCHICAL_OPTIONS).contains(&k) {
            return Err(Error::Shape(
                "the hierarchical family is verified for two or three options",
            ));
        }
        if prior.shapes.iter().any(|a| !(a.is_finite() && *a >= 1.0))
            || !prior.log_concentration_mean.is_finite()
            || !(prior.log_concentration_sd.is_finite() && prior.log_concentration_sd > 0.0)
        {
            return Err(Error::InvalidParameter(
                "population shapes are at least one and ln kappa has a finite mean and a positive sd",
            ));
        }
        if counts.is_empty() || counts.iter().any(|c| c.len() != k) {
            return Err(Error::Shape("every respondent needs one count per option"));
        }
        let totals: Vec<u64> =
            counts.iter().map(|c| c.iter().map(|x| u64::from(*x)).sum()).collect();
        if totals.iter().any(|m| *m == 0 || *m > u64::from(MAX_HIERARCHICAL_CHOICES)) {
            return Err(Error::Limit("a respondent contributes 1 to 512 choices"));
        }
        if totals.iter().all(|m| *m < 2) {
            return Err(Error::PremiseFailed {
                detail: "choice_inference.heterogeneity_not_identified",
                message: "respondent-level heterogeneity is identified only by respondents who choose more than once",
            });
        }
        let mut grouped: BTreeMap<Vec<u32>, f64> = BTreeMap::new();
        for c in counts {
            *grouped.entry(c.clone()).or_insert(0.0) += 1.0;
        }
        if grouped.len() > MAX_DISTINCT_HISTOGRAMS {
            return Err(Error::Limit("more than 4096 distinct respondent count vectors"));
        }
        let histograms: Vec<(Vec<u32>, f64)> = grouped.into_iter().collect();
        let max_n = (0..k)
            .map(|j| histograms.iter().map(|(h, _)| h[j] as usize).max().unwrap_or(0))
            .collect();
        let max_m = usize::try_from(totals.iter().copied().max().unwrap_or(0))
            .map_err(|_| Error::Limit("choice count exceeds platform capacity"))?;
        let a0: f64 = prior.shapes.iter().sum();
        let ln_dirichlet_norm =
            ln_gamma(a0) - prior.shapes.iter().map(|a| ln_gamma(*a)).sum::<f64>();
        Ok(Self {
            options: k,
            respondents: counts.len(),
            shapes: prior.shapes.clone(),
            mu: prior.log_concentration_mean,
            sigma: prior.log_concentration_sd,
            ln_dirichlet_norm,
            histograms,
            max_n,
            max_m,
        })
    }

    /// Number of respondents.
    #[must_use]
    pub const fn respondents(&self) -> usize {
        self.respondents
    }

    fn scratch(&self) -> Scratch {
        Scratch {
            prefix: self.max_n.iter().map(|n| vec![0.0; n + 1]).collect(),
            kappa: vec![0.0; self.max_m + 1],
            pi: [0.0; 3],
        }
    }

    /// Log of the unnormalized posterior density of `(pi, ln kappa)`.
    fn log_target(&self, s: &mut Scratch, v: f64) -> f64 {
        let kappa = v.exp();
        for j in 0..self.options {
            let alpha = kappa * s.pi[j];
            let table = &mut s.prefix[j];
            for i in 0..self.max_n[j] {
                table[i + 1] = table[i] + (alpha + i as f64).ln();
            }
        }
        for i in 0..self.max_m {
            s.kappa[i + 1] = s.kappa[i] + (kappa + i as f64).ln();
        }
        let mut ll = 0.0;
        for (h, mult) in &self.histograms {
            let mut one = 0.0;
            let mut m = 0usize;
            for (j, &count) in h.iter().enumerate() {
                one += s.prefix[j][count as usize];
                m += count as usize;
            }
            ll += mult * (one - s.kappa[m]);
        }
        let mut lp = self.ln_dirichlet_norm;
        for j in 0..self.options {
            lp += (self.shapes[j] - 1.0) * s.pi[j].ln();
        }
        let z = (v - self.mu) / self.sigma;
        ll + lp - 0.5 * z * z - self.sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }

    /// Map region coordinates to the share vector in `scratch`; returns the log Jacobian.
    fn place(&self, region: &Region, c: f64, t: f64, s: &mut Scratch) -> f64 {
        let k = region.option;
        let others: Vec<usize> = (0..self.options).filter(|j| *j != k).collect();
        s.pi[k] = c;
        if self.options == 2 {
            s.pi[others[0]] = 1.0 - c;
            return 0.0;
        }
        let (r_lo, r_hi) = if region.constrained {
            ((1.0 - 2.0 * c) / (1.0 - c), c / (1.0 - c))
        } else {
            (0.0, 1.0)
        };
        let r = r_lo + (r_hi - r_lo) * t;
        s.pi[others[0]] = (1.0 - c) * r;
        s.pi[others[1]] = (1.0 - c) * (1.0 - r);
        ((1.0 - c) * (r_hi - r_lo)).ln()
    }

    fn regions(&self) -> Vec<Region> {
        let mut out = Vec::new();
        for option in 0..self.options {
            if self.options == 2 {
                out.push(Region { option, c_lo: 0.5, c_hi: 1.0, constrained: false });
            } else {
                out.push(Region { option, c_lo: 1.0 / 3.0, c_hi: 0.5, constrained: true });
                out.push(Region { option, c_lo: 0.5, c_hi: 1.0, constrained: false });
            }
        }
        out
    }

    /// Zoom a cube onto the posterior mass of a region: the bounding box of the coarse grid
    /// points within `ZOOM_CUT` nats of the maximum, grown by one grid step, repeated until the
    /// box stops shrinking. Returns the cube and the largest log density found.
    fn zoom(&self, region: &Region, cancelled: &dyn Fn() -> bool) -> Result<(Cube, f64), Error> {
        let active: Vec<usize> = if self.options == 2 { vec![0, 2] } else { vec![0, 1, 2] };
        let mut cube = Cube {
            lo: [region.c_lo, 0.0, V_LIMIT.0.max(self.mu - 10.0 * self.sigma)],
            hi: [region.c_hi, 1.0, V_LIMIT.1.min(self.mu + 10.0 * self.sigma)],
        };
        if self.options == 2 {
            cube.lo[1] = 0.5;
            cube.hi[1] = 0.5;
        }
        let mut scratch = self.scratch();
        let mut best = f64::NEG_INFINITY;
        for _ in 0..ZOOM_ITERATIONS {
            let steps: Vec<f64> = (0..3)
                .map(|d| {
                    if cube.hi[d] > cube.lo[d] && (d != 1 || self.options == 3) {
                        (cube.hi[d] - cube.lo[d]) / (ZOOM_GRID - 1) as f64
                    } else {
                        0.0
                    }
                })
                .collect();
            let n = |d: usize| if steps[d] > 0.0 { ZOOM_GRID } else { 1 };
            let mut found: Vec<([f64; 3], f64)> = Vec::new();
            let mut peak = f64::NEG_INFINITY;
            for a in 0..n(0) {
                if cancelled() {
                    return Err(Error::Cancelled);
                }
                let c = cube.lo[0] + steps[0] * a as f64;
                for b in 0..n(1) {
                    let t = cube.lo[1] + steps[1] * b as f64;
                    for e in 0..n(2) {
                        let v = cube.lo[2] + steps[2] * e as f64;
                        let (cc, tt) = (
                            c.clamp(region.c_lo + 1e-12, region.c_hi - 1e-12),
                            t.clamp(1e-12, 1.0 - 1e-12),
                        );
                        let jac = self.place(region, cc, tt, &mut scratch);
                        let value = self.log_target(&mut scratch, v) + jac;
                        if value.is_finite() {
                            peak = peak.max(value);
                            found.push(([c, t, v], value));
                        }
                    }
                }
            }
            if !peak.is_finite() {
                return Err(Error::Numerical("the hierarchical posterior has no finite mass"));
            }
            best = best.max(peak);
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            for (point, value) in &found {
                if *value >= peak - ZOOM_CUT {
                    for &d in &active {
                        lo[d] = lo[d].min(point[d]);
                        hi[d] = hi[d].max(point[d]);
                    }
                }
            }
            let mut shrunk = false;
            let mut next = cube;
            for &d in &active {
                if steps[d] > 0.0 {
                    next.lo[d] = (lo[d] - steps[d]).max(cube.lo[d]);
                    next.hi[d] = (hi[d] + steps[d]).min(cube.hi[d]);
                    if (next.hi[d] - next.lo[d]) < ZOOM_SHRINK * (cube.hi[d] - cube.lo[d]) {
                        shrunk = true;
                    }
                }
            }
            cube = next;
            if !shrunk {
                break;
            }
        }
        Ok((cube, best))
    }

    /// Integrate `[1, pi_1.., pi_K, v, v^2, 1/(kappa+1)] * density` over a region's cube.
    fn integrate(
        &self,
        region: &Region,
        cube: &Cube,
        panels: usize,
        reference: f64,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<f64>, Error> {
        let rule = panel_rule();
        let c_rule =
            graded_rule(&rule, cube.lo[0], cube.hi[0], panels, false, cube.hi[0] >= 1.0 - 1e-12);
        let t_rule = if self.options == 3 {
            graded_rule(
                &rule,
                cube.lo[1],
                cube.hi[1],
                panels,
                cube.lo[1] <= 1e-12,
                cube.hi[1] >= 1.0 - 1e-12,
            )
        } else {
            (vec![0.5], vec![1.0])
        };
        let v_rule = composite_rule(&rule, cube.lo[2], cube.hi[2], panels);
        let width = self.options + 4;
        let mut out = vec![0.0; width];
        let mut scratch = self.scratch();
        let mut tick = 0usize;
        for (c, wc) in c_rule.0.iter().zip(&c_rule.1) {
            for (t, wt) in t_rule.0.iter().zip(&t_rule.1) {
                let jac = self.place(region, *c, *t, &mut scratch);
                for (v, wv) in v_rule.0.iter().zip(&v_rule.1) {
                    tick += 1;
                    if tick.is_multiple_of(CANCEL_STRIDE) && cancelled() {
                        return Err(Error::Cancelled);
                    }
                    let density = (self.log_target(&mut scratch, *v) + jac - reference).exp();
                    let w = wc * wt * wv * density;
                    if !(w.is_finite()) {
                        return Err(Error::Numerical("the hierarchical posterior overflowed"));
                    }
                    out[0] += w;
                    for j in 0..self.options {
                        out[1 + j] += w * scratch.pi[j];
                    }
                    // Centred on the prior mean so the variance does not cancel.
                    let d = v - self.mu;
                    out[1 + self.options] += w * d;
                    out[2 + self.options] += w * d * d;
                    out[3 + self.options] += w / (v.exp() + 1.0);
                }
            }
        }
        Ok(out)
    }

    /// Summarize the posterior.
    ///
    /// # Errors
    /// A posterior the quadrature cannot resolve to `1e-8`, or cancellation.
    pub fn summary(&self, cancelled: &dyn Fn() -> bool) -> Result<HierarchicalSummary, Error> {
        let regions = self.regions();
        let mut zoomed = Vec::with_capacity(regions.len());
        let mut reference = f64::NEG_INFINITY;
        for region in &regions {
            let (cube, peak) = self.zoom(region, cancelled)?;
            reference = reference.max(peak);
            zoomed.push(cube);
        }
        let mut last = None;
        for panels in PANEL_SCHEDULE {
            let summary = self.attempt(&regions, &zoomed, reference, panels, cancelled)?;
            if summary.quadrature_error <= QUADRATURE_ERROR_LIMIT {
                return Ok(summary);
            }
            last = Some(summary.quadrature_error);
        }
        let _ = last;
        Err(Error::Numerical("the hierarchical posterior is not resolved by the quadrature"))
    }

    /// Integrate every region with a coarse and a fine rule and report the summary of the fine rule
    /// with the largest change between the two.
    fn attempt(
        &self,
        regions: &[Region],
        zoomed: &[Cube],
        reference: f64,
        panels: (usize, usize),
        cancelled: &dyn Fn() -> bool,
    ) -> Result<HierarchicalSummary, Error> {
        let width = self.options + 4;
        let mut totals = [vec![0.0; width], vec![0.0; width]];
        let mut per_region: Vec<[Vec<f64>; 2]> = Vec::new();
        for (region, cube) in regions.iter().zip(zoomed) {
            let coarse = self.integrate(region, cube, panels.0, reference, cancelled)?;
            let fine = self.integrate(region, cube, panels.1, reference, cancelled)?;
            for j in 0..width {
                totals[0][j] += coarse[j];
                totals[1][j] += fine[j];
            }
            per_region.push([coarse, fine]);
        }
        let k = self.options;
        let read = |pick: usize| {
            let z = totals[pick][0];
            let mut best = vec![0.0; k];
            let mut weighted = vec![0.0; k];
            for (region, parts) in regions.iter().zip(&per_region) {
                best[region.option] += parts[pick][0] / z;
                weighted[region.option] += parts[pick][1 + region.option] / z;
            }
            let mean: Vec<f64> = (0..k).map(|j| totals[pick][1 + j] / z).collect();
            let emax: f64 = weighted.iter().sum();
            let loss: Vec<f64> = mean.iter().map(|m| emax - m).collect();
            let dm = totals[pick][1 + k] / z;
            let vm = self.mu + dm;
            let vs = (totals[pick][2 + k] / z - dm * dm).max(0.0).sqrt();
            let rho = totals[pick][3 + k] / z;
            (best, mean, loss, vm, vs, rho, z)
        };
        let (cb, cm, cl, cvm, cvs, crho, _) = read(0);
        let (fb, fm, fl, fvm, fvs, frho, fz) = read(1);
        let mut error = (crho - frho).abs();
        for j in 0..k {
            error = error
                .max((cb[j] - fb[j]).abs())
                .max((cm[j] - fm[j]).abs())
                .max((cl[j] - fl[j]).abs());
        }
        error = error
            .max((cvm - fvm).abs() / (1.0 + fvm.abs().min(1.0)))
            .max((cvs - fvs).abs() / (1.0 + fvs));
        let mass: f64 = fb.iter().sum();
        if (mass - 1.0).abs() > 1e-9 {
            return Err(Error::Numerical("the option regions do not partition the posterior"));
        }
        Ok(HierarchicalSummary {
            mean: fm,
            probability_best: fb,
            expected_loss: fl,
            log_concentration_mean: fvm,
            log_concentration_sd: fvs,
            correlation_mean: frho,
            log_marginal_likelihood: fz.ln() + reference,
            quadrature_error: error,
        })
    }

    /// Equal-tailed credible interval of one option's population share.
    ///
    /// # Errors
    /// An option or level out of range, an unresolved posterior, or cancellation.
    pub fn share_interval(
        &self,
        option: usize,
        level: f64,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<(f64, f64), Error> {
        if option >= self.options || !(level > 0.0 && level < 1.0) {
            return Err(Error::InvalidParameter("an option index and a level in (0, 1)"));
        }
        let region = Region { option, c_lo: 0.0, c_hi: 1.0, constrained: false };
        let (cube, reference) = self.zoom(&region, cancelled)?;
        let rule = panel_rule();
        let c_panels = 6;
        let inner = 4;
        let c_rule = graded_rule(
            &rule,
            cube.lo[0],
            cube.hi[0],
            c_panels,
            cube.lo[0] <= 1e-12,
            cube.hi[0] >= 1.0 - 1e-12,
        );
        let t_rule = if self.options == 3 {
            graded_rule(
                &rule,
                cube.lo[1],
                cube.hi[1],
                inner,
                cube.lo[1] <= 1e-12,
                cube.hi[1] >= 1.0 - 1e-12,
            )
        } else {
            (vec![0.5], vec![1.0])
        };
        let v_rule = composite_rule(&rule, cube.lo[2], cube.hi[2], inner);
        let mut scratch = self.scratch();
        let mut density = Vec::with_capacity(c_rule.0.len());
        for c in &c_rule.0 {
            let mut f = 0.0;
            for (t, wt) in t_rule.0.iter().zip(&t_rule.1) {
                let jac = self.place(&region, *c, *t, &mut scratch);
                for (v, wv) in v_rule.0.iter().zip(&v_rule.1) {
                    f += wt * wv * (self.log_target(&mut scratch, *v) + jac - reference).exp();
                }
            }
            if cancelled() {
                return Err(Error::Cancelled);
            }
            density.push(f);
        }
        let marginal = Marginal::new(&c_rule.0, &density, cube.lo[0], cube.hi[0]);
        let total = marginal.cdf(cube.hi[0]);
        let tail = 0.5 * (1.0 - level);
        Ok((
            marginal.quantile(tail * total, cube.lo[0], cube.hi[0]),
            marginal.quantile((1.0 - tail) * total, cube.lo[0], cube.hi[0]),
        ))
    }
}

/// A density known at the nodes of composite Gauss-Legendre panels, integrated between nodes by
/// the interpolating polynomial of each panel.
struct Marginal {
    panels: Vec<(f64, f64, Vec<f64>, Vec<f64>)>,
    rule: (Vec<f64>, Vec<f64>),
}

impl Marginal {
    fn new(nodes: &[f64], density: &[f64], _lo: f64, _hi: f64) -> Self {
        let rule = panel_rule();
        let per = rule.0.len();
        // `graded_rule` panels are contiguous runs of `per` nodes with increasing abscissae.
        let panels = nodes
            .chunks(per)
            .zip(density.chunks(per))
            .map(|(x, f)| {
                let half = (x[per - 1] - x[0]) / (rule.0[per - 1] - rule.0[0]);
                let centre = x[0] - half * rule.0[0];
                (centre - half, centre + half, x.to_vec(), f.to_vec())
            })
            .collect();
        Self { panels, rule }
    }

    fn interpolate(x: &[f64], f: &[f64], t: f64) -> f64 {
        let n = x.len();
        let mut total = 0.0;
        for i in 0..n {
            let mut basis = f[i];
            for m in 0..n {
                if m != i {
                    basis *= (t - x[m]) / (x[i] - x[m]);
                }
            }
            total += basis;
        }
        total
    }

    fn cdf(&self, x: f64) -> f64 {
        let mut total = 0.0;
        for (a, b, nodes, values) in &self.panels {
            if x <= *a {
                break;
            }
            let upper = x.min(*b);
            let (centre, half) = (0.5 * (a + upper), 0.5 * (upper - a));
            for (u, w) in self.rule.0.iter().zip(&self.rule.1) {
                total += half * w * Self::interpolate(nodes, values, centre + half * u);
            }
        }
        total
    }

    fn quantile(&self, mass: f64, lo: f64, hi: f64) -> f64 {
        let (mut a, mut b) = (lo, hi);
        for _ in 0..80 {
            let mid = 0.5 * (a + b);
            if self.cdf(mid) < mass {
                a = mid;
            } else {
                b = mid;
            }
        }
        0.5 * (a + b)
    }
}
