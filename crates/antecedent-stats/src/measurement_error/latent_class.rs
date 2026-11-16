//! Repeat codings of a binary or ordinal facet as replicates: a latent-class model.
//!
//! Each unit has `K` repeat codings `c_{i1..iK}`, each in `C` ordered categories, of one latent
//! facet with `J` ordered levels (`J = 2` for a binary facet, `J = C` for an ordinal facet
//! coded on its own scale). The model is the latent-class model with **conditional
//! independence across codings given the latent level**, which is an assumption and is not
//! tested here:
//!
//! `P(c_i) = sum_j pi_j prod_k theta_{k j c_ik}`.
//!
//! Without further declarations the model is identified (up to the order of the classes) by
//! Kruskal's condition: the codings can be split into three groups whose k-ranks
//! `min(J, C^size)` sum to at least `2 J + 2` (for `J = 2` and `C >= 2` that is three
//! codings). Two codings of a binary facet identify it only under declared constraints:
//! exchangeable codings (one response law for every coding) and a declared response law of the
//! lowest class (for example a declared specificity). Anything else refuses. After the fit,
//! codings whose class-conditional laws are almost the same cannot count towards the
//! identifying number, and the fit refuses when too few separate the classes.
//!
//! Classes are ordered by their expected coding score, which removes label switching. The fit
//! is maximum likelihood by EM from several starts; the covariance is the inverse of the
//! outer product of per-unit scores in a softmax parametrization of the free probabilities,
//! carried to the probabilities by the delta method. The covariance is reported when that
//! matrix is invertible.

use antecedent_core::CausalRng;

use crate::gram::invert_square;

use super::MeasurementErrorError;

/// Most latent levels.
pub const MAX_LATENT_LEVELS: usize = 8;
/// Most repeat codings per unit.
pub const MAX_LATENT_CODINGS: usize = 64;
/// Most categories per coding.
pub const MAX_LATENT_CATEGORIES: usize = 16;

/// Declared constraints on the response laws.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LatentClassConstraints {
    /// Every coding has the same response law given the latent level.
    pub exchangeable: bool,
    /// Declared response law of the lowest latent level over the categories (for a binary
    /// facet and a coding recorded `0` when absent, the specificity and its complement).
    pub lowest_level: Option<Vec<f64>>,
}

/// Fit settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatentClassSettings {
    /// Starting points; the first is the sum-score start.
    pub starts: usize,
    /// Most EM iterations per start.
    pub max_iterations: usize,
    /// Relative log-likelihood change that stops a start.
    pub tolerance: f64,
    /// Smallest total-variation distance between adjacent latent levels' response laws for a
    /// coding to count towards identification.
    pub min_separation: f64,
    /// Seed of the random starts.
    pub seed: u64,
}

impl Default for LatentClassSettings {
    fn default() -> Self {
        Self { starts: 8, max_iterations: 2000, tolerance: 1e-10, min_separation: 0.05, seed: 1 }
    }
}

/// Repeat codings, row-major `units x coding_count`, each a category index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LatentClassData<'a> {
    /// Category index of every coding of every unit.
    pub codings: &'a [u8],
    /// Units.
    pub units: usize,
    /// Codings per unit.
    pub coding_count: usize,
    /// Categories per coding.
    pub categories: usize,
    /// Latent levels.
    pub latent_levels: usize,
}

/// Standard errors of a fit.
#[derive(Clone, Debug, PartialEq)]
pub struct LatentClassStandardErrors {
    /// Of each latent level's prevalence.
    pub prevalence: Vec<f64>,
    /// Of each response probability, laid out as the fit's `response`; zero for the declared
    /// lowest level.
    pub response: Vec<f64>,
}

/// How the fit was identified.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LatentClassIdentification {
    /// Kruskal's condition holds for the declared number of codings.
    pub kruskal: bool,
    /// Declared constraints (not Kruskal) identified a two-coding fit.
    pub constrained: bool,
    /// Codings whose adjacent latent levels differ by at least the separation.
    pub informative_codings: usize,
}

/// A fitted latent-class model.
#[derive(Clone, Debug, PartialEq)]
pub struct LatentClassFit {
    /// Latent levels `J`.
    pub latent_levels: usize,
    /// Categories `C`.
    pub categories: usize,
    /// Codings per unit `K`.
    pub coding_count: usize,
    /// Units.
    pub units: usize,
    /// Prevalence of each latent level, increasing in expected coding score.
    pub prevalence: Vec<f64>,
    /// Response laws `P(category | level)`, indexed `[slot][level][category]` flattened; `slot`
    /// is the coding, or the single shared slot when the codings are exchangeable.
    pub response: Vec<f64>,
    /// Response slots: `1` when exchangeable, else `coding_count`.
    pub response_slots: usize,
    /// Per-unit posterior over the latent levels, row-major `units x J`.
    pub posterior: Vec<f64>,
    /// Maximized log likelihood.
    pub log_likelihood: f64,
    /// EM iterations of the chosen start.
    pub iterations: usize,
    /// The chosen start stopped on the tolerance.
    pub converged: bool,
    /// Starts whose log likelihood equals the best one.
    pub starts_agreeing: usize,
    /// Starts run.
    pub starts: usize,
    /// Standard errors, when the score matrix is invertible.
    pub standard_errors: Option<LatentClassStandardErrors>,
    /// What identified the fit.
    pub identification: LatentClassIdentification,
}

impl LatentClassFit {
    /// `P(category | level)` of coding `coding`.
    #[must_use]
    pub fn response_of(&self, coding: usize, level: usize, category: usize) -> f64 {
        let slot = if self.response_slots == 1 { 0 } else { coding };
        self.response[(slot * self.latent_levels + level) * self.categories + category]
    }
}

/// Whether Kruskal's condition holds for `j` levels, `c` categories and `k` codings.
#[must_use]
pub fn kruskal_identified(j: usize, c: usize, k: usize) -> bool {
    if k < 3 || j < 2 || c < 2 {
        return false;
    }
    let rank = |size: usize| -> usize {
        let mut power = 1_usize;
        for _ in 0..size {
            power = power.saturating_mul(c);
            if power >= j {
                return j;
            }
        }
        power.min(j)
    };
    let mut best = 0;
    for a in 1..=k - 2 {
        for b in 1..=k - 1 - a {
            let rest = k - a - b;
            best = best.max(rank(a) + rank(b) + rank(rest));
        }
    }
    best >= 2 * j + 2
}

struct Layout {
    j: usize,
    c: usize,
    k: usize,
    slots: usize,
    pinned: Option<Vec<f64>>,
}

impl Layout {
    fn slot(&self, coding: usize) -> usize {
        if self.slots == 1 { 0 } else { coding }
    }
    fn at(&self, slot: usize, level: usize, category: usize) -> usize {
        (slot * self.j + level) * self.c + category
    }
    fn first_free_level(&self) -> usize {
        usize::from(self.pinned.is_some())
    }
}

struct Params {
    pi: Vec<f64>,
    theta: Vec<f64>,
}

fn m_step(data: &LatentClassData<'_>, layout: &Layout, resp: &[f64]) -> Params {
    let (n, j, c, k) = (data.units, layout.j, layout.c, layout.k);
    let mut pi = vec![0.0; j];
    for i in 0..n {
        for l in 0..j {
            pi[l] += resp[i * j + l];
        }
    }
    let total: f64 = pi.iter().sum();
    for p in &mut pi {
        *p = (*p / total).max(1e-12);
    }
    let norm: f64 = pi.iter().sum();
    for p in &mut pi {
        *p /= norm;
    }
    let mut theta = vec![0.0; layout.slots * j * c];
    for slot in 0..layout.slots {
        for level in 0..j {
            if let Some(fixed) = layout.pinned.as_ref().filter(|_| level == 0) {
                for m in 0..c {
                    theta[layout.at(slot, 0, m)] = fixed[m];
                }
                continue;
            }
            let mut counts = vec![0.0; c];
            for i in 0..n {
                let r = resp[i * j + level];
                for coding in 0..k {
                    if layout.slot(coding) == slot {
                        counts[usize::from(data.codings[i * k + coding])] += r;
                    }
                }
            }
            let sum: f64 = counts.iter().sum();
            for m in 0..c {
                theta[layout.at(slot, level, m)] =
                    if sum > 0.0 { counts[m] / sum } else { 1.0 / c as f64 };
            }
        }
    }
    Params { pi, theta }
}

/// Posterior responsibilities and the log likelihood under `params`.
fn e_step(data: &LatentClassData<'_>, layout: &Layout, params: &Params, resp: &mut [f64]) -> f64 {
    let (n, j, k) = (data.units, layout.j, layout.k);
    let log_pi: Vec<f64> = params.pi.iter().map(|p| p.max(1e-300).ln()).collect();
    let log_theta: Vec<f64> = params.theta.iter().map(|t| t.max(1e-300).ln()).collect();
    let mut loglik = 0.0;
    let mut weights = vec![0.0; j];
    for i in 0..n {
        for level in 0..j {
            let mut w = log_pi[level];
            for coding in 0..k {
                let m = usize::from(data.codings[i * k + coding]);
                w += log_theta[layout.at(layout.slot(coding), level, m)];
            }
            weights[level] = w;
        }
        let top = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = weights.iter().map(|w| (w - top).exp()).sum();
        loglik += top + sum.ln();
        for level in 0..j {
            resp[i * j + level] = (weights[level] - top).exp() / sum;
        }
    }
    loglik
}

struct Run {
    params: Params,
    resp: Vec<f64>,
    loglik: f64,
    iterations: usize,
    converged: bool,
}

fn run_em(
    data: &LatentClassData<'_>,
    layout: &Layout,
    start: Vec<f64>,
    settings: &LatentClassSettings,
) -> Run {
    let mut resp = start;
    let mut params = m_step(data, layout, &resp);
    let mut previous = f64::NEG_INFINITY;
    let mut loglik = e_step(data, layout, &params, &mut resp);
    let mut iterations = 1;
    let mut converged = false;
    while iterations < settings.max_iterations {
        if (loglik - previous).abs() <= settings.tolerance * (1.0 + loglik.abs()) {
            converged = true;
            break;
        }
        previous = loglik;
        params = m_step(data, layout, &resp);
        loglik = e_step(data, layout, &params, &mut resp);
        iterations += 1;
    }
    Run { params, resp, loglik, iterations, converged }
}

fn sum_score_start(data: &LatentClassData<'_>, layout: &Layout) -> Vec<f64> {
    let (n, j, c, k) = (data.units, layout.j, layout.c, layout.k);
    let top = (k * (c - 1)) as f64;
    let mut resp = vec![0.0; n * j];
    for i in 0..n {
        let score: f64 = (0..k).map(|coding| f64::from(data.codings[i * k + coding])).sum();
        let position = score / top * (j - 1) as f64;
        let mut total = 0.0;
        for level in 0..j {
            let d = position - level as f64;
            let w = (-(d * d) / (2.0 * 0.3 * 0.3)).exp() + 1e-6;
            resp[i * j + level] = w;
            total += w;
        }
        for level in 0..j {
            resp[i * j + level] /= total;
        }
    }
    resp
}

fn separated_codings(layout: &Layout, params: &Params, min_separation: f64) -> usize {
    (0..layout.k)
        .filter(|&coding| {
            let slot = layout.slot(coding);
            (1..layout.j).all(|level| {
                let tv: f64 = (0..layout.c)
                    .map(|m| {
                        (params.theta[layout.at(slot, level, m)]
                            - params.theta[layout.at(slot, level - 1, m)])
                        .abs()
                    })
                    .sum::<f64>()
                    / 2.0;
                tv >= min_separation
            })
        })
        .count()
}

/// Reorder the free levels so expected coding score increases; level 0 stays first when it is
/// the declared lowest level.
fn sort_levels(layout: &Layout, run: &mut Run, n: usize) {
    let (j, c) = (layout.j, layout.c);
    let first = layout.first_free_level();
    let score = |theta: &[f64], level: usize| -> f64 {
        (0..layout.slots)
            .map(|slot| (0..c).map(|m| m as f64 * theta[layout.at(slot, level, m)]).sum::<f64>())
            .sum::<f64>()
    };
    let mut order: Vec<usize> = (0..j).collect();
    order[first..].sort_by(|&a, &b| {
        score(&run.params.theta, a)
            .partial_cmp(&score(&run.params.theta, b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let old_pi = run.params.pi.clone();
    let old_theta = run.params.theta.clone();
    let old_resp = run.resp.clone();
    for (new, &old) in order.iter().enumerate() {
        run.params.pi[new] = old_pi[old];
        for slot in 0..layout.slots {
            for m in 0..c {
                run.params.theta[layout.at(slot, new, m)] = old_theta[layout.at(slot, old, m)];
            }
        }
        for i in 0..n {
            run.resp[i * j + new] = old_resp[i * j + old];
        }
    }
}

fn standard_errors(
    data: &LatentClassData<'_>,
    layout: &Layout,
    run: &Run,
) -> Option<LatentClassStandardErrors> {
    let (n, j, c, k) = (data.units, layout.j, layout.c, layout.k);
    let free_levels = j - layout.first_free_level();
    let first = layout.first_free_level();
    let base = j - 1;
    let dim = base + layout.slots * free_levels * (c - 1);
    if dim > 400 {
        // The score matrix would be too large to form; no covariance is reported.
        return None;
    }
    let theta_at = |slot: usize, level: usize, mp: usize| {
        base + (slot * free_levels + (level - first)) * (c - 1) + (mp - 1)
    };
    let mut info = vec![0.0; dim * dim];
    let mut score = vec![0.0; dim];
    for i in 0..n {
        score.iter_mut().for_each(|s| *s = 0.0);
        for l in 1..j {
            score[l - 1] = run.resp[i * j + l] - run.params.pi[l];
        }
        for coding in 0..k {
            let slot = layout.slot(coding);
            let category = usize::from(data.codings[i * k + coding]);
            for level in first..j {
                let r = run.resp[i * j + level];
                for mp in 1..c {
                    let hit = if category == mp { 1.0 } else { 0.0 };
                    score[theta_at(slot, level, mp)] +=
                        r * (hit - run.params.theta[layout.at(slot, level, mp)]);
                }
            }
        }
        for a in 0..dim {
            for b in 0..dim {
                info[a * dim + b] += score[a] * score[b];
            }
        }
    }
    let cov = invert_square(&info, dim)?;
    if cov.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let quad = |grad: &[(usize, f64)]| -> f64 {
        let mut total = 0.0;
        for &(a, ga) in grad {
            for &(b, gb) in grad {
                total += ga * gb * cov[a * dim + b];
            }
        }
        total.max(0.0).sqrt()
    };
    let pi = &run.params.pi;
    let prevalence: Vec<f64> = (0..j)
        .map(|level| {
            let grad: Vec<(usize, f64)> = (1..j)
                .map(|l| {
                    let delta = if level == l { 1.0 } else { 0.0 };
                    (l - 1, pi[level] * (delta - pi[l]))
                })
                .collect();
            quad(&grad)
        })
        .collect();
    let mut response = vec![0.0; layout.slots * j * c];
    for slot in 0..layout.slots {
        for level in first..j {
            for m in 0..c {
                let grad: Vec<(usize, f64)> = (1..c)
                    .map(|mp| {
                        let delta = if m == mp { 1.0 } else { 0.0 };
                        let t = &run.params.theta;
                        (
                            theta_at(slot, level, mp),
                            t[layout.at(slot, level, m)] * (delta - t[layout.at(slot, level, mp)]),
                        )
                    })
                    .collect();
                response[layout.at(slot, level, m)] = quad(&grad);
            }
        }
    }
    Some(LatentClassStandardErrors { prevalence, response })
}

/// Fit the latent-class model to repeat codings.
///
/// # Errors
/// A shape or range violation; too few codings for the declared levels and constraints
/// (three for two levels, more for more levels, or two with exchangeable codings and a
/// declared lowest-level law); a fit whose codings do not separate the levels refuse.
#[allow(
    clippy::too_many_lines,
    reason = "one flat sequence of identification checks, the multi-start fit and the assembly of its report"
)]
pub fn fit_latent_class(
    data: &LatentClassData<'_>,
    constraints: &LatentClassConstraints,
    settings: &LatentClassSettings,
) -> Result<LatentClassFit, MeasurementErrorError> {
    let (n, j, c, k) = (data.units, data.latent_levels, data.categories, data.coding_count);
    if !(2..=MAX_LATENT_LEVELS).contains(&j)
        || !(2..=MAX_LATENT_CATEGORIES).contains(&c)
        || !(2..=MAX_LATENT_CODINGS).contains(&k)
    {
        return Err(MeasurementErrorError::Shape(
            "latent levels, categories or codings are outside the supported bounds",
        ));
    }
    if n.checked_mul(k) != Some(data.codings.len()) || n < 10 * j {
        return Err(MeasurementErrorError::Shape("codings do not form units x coding_count"));
    }
    if data.codings.iter().any(|&m| usize::from(m) >= c) {
        return Err(MeasurementErrorError::InvalidParameter(
            "a coding is outside the declared categories",
        ));
    }
    if settings.starts == 0
        || settings.max_iterations == 0
        || !(settings.tolerance.is_finite() && settings.tolerance > 0.0)
        || !(settings.min_separation.is_finite() && (0.0..1.0).contains(&settings.min_separation))
    {
        return Err(MeasurementErrorError::InvalidParameter("invalid latent-class settings"));
    }
    if let Some(fixed) = &constraints.lowest_level
        && (fixed.len() != c
            || fixed.iter().any(|v| !(v.is_finite() && *v >= 0.0))
            || (fixed.iter().sum::<f64>() - 1.0).abs() > 1e-9)
    {
        return Err(MeasurementErrorError::InvalidParameter(
            "the declared lowest-level law must be a probability vector over the categories",
        ));
    }
    let kruskal = kruskal_identified(j, c, k);
    let constrained = !kruskal
        && k == 2
        && j == 2
        && constraints.exchangeable
        && constraints.lowest_level.is_some();
    if !(kruskal || constrained) {
        return Err(MeasurementErrorError::InsufficientReplicates(
            "the repeat codings do not identify the latent classes: three or more conditionally independent codings are needed, or two with exchangeable codings and a declared law of the lowest level",
        ));
    }
    let layout = Layout {
        j,
        c,
        k,
        slots: if constraints.exchangeable { 1 } else { k },
        pinned: constraints.lowest_level.clone(),
    };
    let mut rng = CausalRng::from_seed(settings.seed);
    let base = sum_score_start(data, &layout);
    let mut runs = Vec::with_capacity(settings.starts);
    for start in 0..settings.starts {
        let resp = if start == 0 {
            base.clone()
        } else {
            let mut r = base.clone();
            for i in 0..n {
                let mut total = 0.0;
                for level in 0..j {
                    let v = 0.5 * r[i * j + level] + 0.5 * (rng.next_f64() + 1e-3);
                    r[i * j + level] = v;
                    total += v;
                }
                for level in 0..j {
                    r[i * j + level] /= total;
                }
            }
            r
        };
        runs.push(run_em(data, &layout, resp, settings));
    }
    let best_index = runs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.loglik.partial_cmp(&b.1.loglik).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i);
    let best_loglik = runs[best_index].loglik;
    let agreeing = runs
        .iter()
        .filter(|r| (r.loglik - best_loglik).abs() <= 1e-6 * (1.0 + best_loglik.abs()))
        .count();
    let starts = runs.len();
    let mut run = runs.swap_remove(best_index);
    sort_levels(&layout, &mut run, n);
    let informative = separated_codings(&layout, &run.params, settings.min_separation);
    let needed = if kruskal { 3 } else { 2 };
    if informative < needed {
        return Err(MeasurementErrorError::InsufficientReplicates(
            "fewer codings than identification needs separate the latent classes; their response laws are nearly equal across classes",
        ));
    }
    let standard_errors = standard_errors(data, &layout, &run);
    Ok(LatentClassFit {
        latent_levels: j,
        categories: c,
        coding_count: k,
        units: n,
        prevalence: run.params.pi.clone(),
        response: run.params.theta.clone(),
        response_slots: layout.slots,
        posterior: run.resp.clone(),
        log_likelihood: run.loglik,
        iterations: run.iterations,
        converged: run.converged,
        starts_agreeing: agreeing,
        starts,
        standard_errors,
        identification: LatentClassIdentification {
            kruskal,
            constrained,
            informative_codings: informative,
        },
    })
}
