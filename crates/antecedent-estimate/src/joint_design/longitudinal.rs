//! Nonparametric longitudinal g-computation retaining complete covariate histories.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{JointDesignEstimate, JointUncertainty};
use std::collections::{BTreeMap, BTreeSet};

/// One independent episode; covariate `t` is measured before joint assignment `t`.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LongitudinalEpisode {
    /// Original independent unit identity.
    pub id: String,
    /// All covariate states, including treatment-affected states at later times.
    pub covariates: Vec<usize>,
    /// Complete multichannel arm at each time, not independent marginal treatments.
    pub assignments: Vec<usize>,
    /// Terminal outcome.
    pub outcome: f64,
}
/// Design premises and explicitly requested joint intervention schedules.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LongitudinalJointDesign {
    /// Stable world key and joint-arm sequence.
    pub worlds: Vec<(String, Vec<usize>)>,
    /// All observed histories suffice for treatment assignment exchangeability at each time.
    pub sequential_exchangeability: String,
    /// Treatment versions and measured variables have a consistent intervention interpretation.
    pub consistency: String,
    /// Independent episode sampling and target population declaration.
    pub sampling_population: String,
}
fn matches_prefix(e: &LongitudinalEpisode, l: &[usize], a: &[usize]) -> bool {
    e.covariates.starts_with(l) && e.assignments.starts_with(a)
}
fn recurse(
    episodes: &[&LongitudinalEpisode],
    schedule: &[usize],
    history: &mut Vec<usize>,
    probability: f64,
) -> Result<f64, String> {
    let t = history.len() - 1;
    let prefix = &schedule[..=t];
    let cell =
        episodes.iter().copied().filter(|e| matches_prefix(e, history, prefix)).collect::<Vec<_>>();
    if cell.is_empty() {
        return Err(format!(
            "sequential positivity failure for covariate history {history:?} and joint assignments {prefix:?}"
        ));
    }
    if t + 1 == schedule.len() {
        return Ok(probability * cell.iter().map(|e| e.outcome).sum::<f64>() / cell.len() as f64);
    }
    let mut next = BTreeMap::new();
    for e in &cell {
        *next.entry(e.covariates[t + 1]).or_insert(0usize) += 1;
    }
    let mut result = 0.0;
    for (state, count) in next {
        history.push(state);
        result +=
            recurse(episodes, schedule, history, probability * count as f64 / cell.len() as f64)?;
        history.pop();
    }
    Ok(result)
}
fn means(
    design: &LongitudinalJointDesign,
    episodes: &[&LongitudinalEpisode],
) -> Result<Vec<f64>, String> {
    let mut initial = BTreeMap::new();
    for e in episodes {
        *initial.entry(e.covariates[0]).or_insert(0usize) += 1;
    }
    design
        .worlds
        .iter()
        .map(|(_, schedule)| {
            let mut value = 0.0;
            for (&state, &count) in &initial {
                value += recurse(
                    episodes,
                    schedule,
                    &mut vec![state],
                    count as f64 / episodes.len() as f64,
                )?;
            }
            Ok(value)
        })
        .collect()
}
/// Execute the longitudinal g-formula over full observed histories.
/// Later covariates are integrated with their treatment-specific transition law;
/// fixing their observed values across regimes would target a different functional.
/// Joint uncertainty deletes entire episodes and repeats all transition/outcome fits.
///
/// # Errors
/// Missing scientific premises, invalid episodes, unsupported history/arm cells,
/// or a delete-episode realization losing positivity. Failures never silently
/// become zero weights, a shorter history or marginal-arm support.
pub fn longitudinal_joint_means(
    design: &LongitudinalJointDesign,
    episodes: &[LongitudinalEpisode],
) -> Result<JointDesignEstimate, String> {
    let horizon = design.worlds.first().map_or(0, |(_, s)| s.len());
    if horizon == 0
        || horizon > 32
        || design.worlds.len() > 256
        || episodes.len() > 4096
        || episodes.len() < 3
        || [&design.sequential_exchangeability, &design.consistency, &design.sampling_population]
            .iter()
            .any(|s| s.trim().is_empty())
        || design.worlds.iter().any(|(key, s)| key.is_empty() || s.len() != horizon)
        || design.worlds.iter().map(|(key, _)| key).collect::<BTreeSet<_>>().len()
            != design.worlds.len()
    {
        return Err("longitudinal joint design requires complete schedules and explicit exchangeability/consistency/population premises".into());
    }
    let mut ids = BTreeSet::new();
    for e in episodes {
        if e.id.is_empty()
            || !ids.insert(&e.id)
            || e.covariates.len() != horizon
            || e.assignments.len() != horizon
            || !e.outcome.is_finite()
        {
            return Err("invalid longitudinal episode or duplicate independent unit".into());
        }
    }
    let refs = episodes.iter().collect::<Vec<_>>();
    let point = means(design, &refs)?;
    let n = episodes.len();
    let k = point.len();
    let mut leave = Vec::with_capacity(n);
    for omitted in 0..n {
        let sample = refs
            .iter()
            .enumerate()
            .filter_map(|(i, e)| (i != omitted).then_some(*e))
            .collect::<Vec<_>>();
        leave.push(
            means(design, &sample)
                .map_err(|e| format!("whole-episode uncertainty realization {omitted}: {e}"))?,
        );
    }
    let center =
        (0..k).map(|j| leave.iter().map(|m| m[j]).sum::<f64>() / n as f64).collect::<Vec<_>>();
    let mut covariance = vec![0.0; k * k];
    for realization in leave {
        for j in 0..k {
            for l in 0..k {
                covariance[j * k + l] += (n - 1) as f64 / n as f64
                    * (realization[j] - center[j])
                    * (realization[l] - center[l]);
            }
        }
    }
    Ok(JointDesignEstimate {
        worlds: design.worlds.iter().map(|(key, _)| key.clone()).collect(),
        means: point,
        covariance,
        sampling_unit: "independent longitudinal episode".into(),
        uncertainty: JointUncertainty::DeleteEpisodeJackknife,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gformula_retains_treatment_affected_history_and_rejects_missing_joint_support() {
        let mut episodes = Vec::new();
        // P(L1=1|A0=0)=1/4, P(L1=1|A0=1)=3/4. Y=2A0+3A1+4L1+A0*A1.
        for a in 0..2 {
            for l in 0..2 {
                for b in 0..2 {
                    for repeat in 0..if a == l { 6 } else { 2 } {
                        episodes.push(LongitudinalEpisode {
                            id: format!("{a}/{l}/{b}/{repeat}"),
                            covariates: vec![0, l],
                            assignments: vec![a, b],
                            outcome: (2 * a + 3 * b + 4 * l + a * b) as f64,
                        });
                    }
                }
            }
        }
        let design = LongitudinalJointDesign {
            worlds: vec![
                ("00".into(), vec![0, 0]),
                ("10".into(), vec![1, 0]),
                ("01".into(), vec![0, 1]),
                ("11".into(), vec![1, 1]),
            ],
            sequential_exchangeability: "A_t randomized conditional on entire measured history"
                .into(),
            consistency: "joint arm versions fixed".into(),
            sampling_population: "independent episodes in source population".into(),
        };
        let fit = longitudinal_joint_means(&design, &episodes).unwrap();
        assert_eq!(fit.means, vec![1.0, 5.0, 4.0, 9.0]);
        assert_eq!(fit.contrast(&[1.0, -1.0, -1.0, 1.0]).unwrap().0, 1.0);
        let mut invalid = design.clone();
        invalid.sequential_exchangeability.clear();
        assert!(longitudinal_joint_means(&invalid, &episodes).is_err());
        episodes.retain(|e| e.assignments != [1, 1]);
        assert!(longitudinal_joint_means(&design, &episodes).is_err());
    }
}

/// Proper finite-state conditional likelihood and independent symmetric Dirichlet priors.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LongitudinalDirichletPrior {
    /// Complete covariate state space at each time, including possible unobserved states.
    pub covariate_states: Vec<Vec<usize>>,
    /// Complete terminal outcome support; no empirical-Bayes support selection is hidden.
    pub outcome_values: Vec<f64>,
    /// Positive prior mass per possible category in each conditional table.
    pub concentration: f64,
    /// Aligned full-mechanism draws.
    pub draws: usize,
    /// Deterministic seed.
    pub seed: u64,
}
type HistoryKey = (Vec<usize>, Vec<usize>);
fn dirichlet_probabilities(
    counts: &[usize],
    alpha: f64,
    rng: &mut antecedent_core::CausalRng,
) -> Result<Vec<f64>, String> {
    let weights = counts
        .iter()
        .map(|n| antecedent_kernels::sample_gamma(*n as f64 + alpha, 1.0, rng))
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err("Dirichlet conditional mechanism draw overflow or underflow".into());
    }
    Ok(weights.iter().map(|v| v / total).collect())
}
struct PosteriorMechanism<'a> {
    episodes: &'a [LongitudinalEpisode],
    prior: &'a LongitudinalDirichletPrior,
    transitions: BTreeMap<HistoryKey, Vec<f64>>,
    outcomes: BTreeMap<HistoryKey, f64>,
    rng: &'a mut antecedent_core::CausalRng,
}
impl PosteriorMechanism<'_> {
    fn evaluate(
        &mut self,
        schedule: &[usize],
        history: &mut Vec<usize>,
        probability: f64,
    ) -> Result<f64, String> {
        let t = history.len() - 1;
        let key = (history.clone(), schedule[..=t].to_vec());
        let cell = self
            .episodes
            .iter()
            .filter(|e| matches_prefix(e, history, &schedule[..=t]))
            .collect::<Vec<_>>();
        if cell.is_empty() {
            return Err(format!(
                "posterior mechanism retains no empirical joint-history assignment support for {key:?}"
            ));
        }
        if t + 1 == schedule.len() {
            let mean = if let Some(value) = self.outcomes.get(&key) {
                *value
            } else {
                let counts = self
                    .prior
                    .outcome_values
                    .iter()
                    .map(|y| cell.iter().filter(|e| e.outcome == *y).count())
                    .collect::<Vec<_>>();
                let p = dirichlet_probabilities(&counts, self.prior.concentration, self.rng)?;
                let value =
                    p.iter().zip(&self.prior.outcome_values).map(|(p, y)| p * y).sum::<f64>();
                self.outcomes.insert(key, value);
                value
            };
            return Ok(probability * mean);
        }
        let probabilities = if let Some(value) = self.transitions.get(&key) {
            value.clone()
        } else {
            let counts = self.prior.covariate_states[t + 1]
                .iter()
                .map(|state| cell.iter().filter(|e| e.covariates[t + 1] == *state).count())
                .collect::<Vec<_>>();
            let value = dirichlet_probabilities(&counts, self.prior.concentration, self.rng)?;
            self.transitions.insert(key, value.clone());
            value
        };
        let mut value = 0.0;
        for (&state, p) in self.prior.covariate_states[t + 1].iter().zip(probabilities) {
            history.push(state);
            value += self.evaluate(schedule, history, probability * p)?;
            history.pop();
        }
        Ok(value)
    }
}
/// Bayesian longitudinal g-computation from full-history categorical mechanisms.
/// Each draw samples initial population, every treatment-specific covariate transition
/// and every terminal outcome mechanism once, then evaluates all requested regimes
/// against that same draw. Priors are proper and fixed over caller-declared supports.
/// Identification premises are exactly those retained by the frequentist g-formula.
///
/// # Errors
/// Invalid finite-state prior, observed outcomes outside its declared support,
/// absent reachable joint-history assignment support or numerical draw failure.
pub fn longitudinal_joint_posterior(
    design: &LongitudinalJointDesign,
    episodes: &[LongitudinalEpisode],
    prior: &LongitudinalDirichletPrior,
) -> Result<super::JointDesignPosterior, String> {
    // Identical scientific checker and complete-procedure positivity checks for both modes.
    let _identified = longitudinal_joint_means(design, episodes)?;
    let horizon = episodes[0].covariates.len();
    if prior.covariate_states.len() != horizon
        || prior.covariate_states.iter().any(|states| {
            states.is_empty() || states.iter().collect::<BTreeSet<_>>().len() != states.len()
        })
        || prior.outcome_values.is_empty()
        || prior.outcome_values.iter().any(|y| !y.is_finite())
        || prior
            .outcome_values
            .iter()
            .enumerate()
            .any(|(i, y)| prior.outcome_values[..i].contains(y))
        || !prior.concentration.is_finite()
        || prior.concentration <= 0.0
        || prior.draws < 2
        || prior.draws > 1_000_000
    {
        return Err("longitudinal posterior requires complete distinct finite supports and a proper Dirichlet prior".into());
    }
    if prior
        .covariate_states
        .iter()
        .try_fold(1usize, |total, states| total.checked_mul(states.len()))
        .is_none_or(|total| total > 65536)
    {
        return Err(
            "declared full-history state space exceeds bounded finite-table implementation".into(),
        );
    }
    for e in episodes {
        if !prior.outcome_values.contains(&e.outcome)
            || e.covariates.iter().enumerate().any(|(t, l)| !prior.covariate_states[t].contains(l))
        {
            return Err(
                "observed episode lies outside declared categorical mechanism support".into()
            );
        }
    }
    let mut rng = antecedent_core::CausalRng::from_seed(prior.seed);
    let mut draws = Vec::with_capacity(prior.draws);
    for _ in 0..prior.draws {
        let counts = prior.covariate_states[0]
            .iter()
            .map(|state| episodes.iter().filter(|e| e.covariates[0] == *state).count())
            .collect::<Vec<_>>();
        let initial = dirichlet_probabilities(&counts, prior.concentration, &mut rng)?;
        let mut mechanism = PosteriorMechanism {
            episodes,
            prior,
            transitions: BTreeMap::new(),
            outcomes: BTreeMap::new(),
            rng: &mut rng,
        };
        let mut values = Vec::with_capacity(design.worlds.len());
        for (_, schedule) in &design.worlds {
            let mut value = 0.0;
            for (&state, &p) in prior.covariate_states[0].iter().zip(&initial) {
                value += mechanism.evaluate(schedule, &mut vec![state], p)?;
            }
            values.push(value);
        }
        draws.push(values);
    }
    super::JointDesignPosterior::from_draws(
        design.worlds.iter().map(|(key, _)| key.clone()).collect(),
        draws,
        "independent longitudinal episode".into(),
        format!(
            "full-history finite categorical transition and terminal-outcome likelihood; independent symmetric Dirichlet({}) priors on explicitly supplied fixed supports",
            prior.concentration
        ),
    )
}

#[cfg(test)]
mod posterior_tests {
    use super::*;
    #[test]
    fn finite_history_likelihood_posterior_matches_beta_expectations() {
        let design = LongitudinalJointDesign {
            worlds: vec![("base".into(), vec![0]), ("joint".into(), vec![1])],
            sequential_exchangeability: "randomized joint assignment conditional on baseline"
                .into(),
            consistency: "fixed treatment versions".into(),
            sampling_population: "independent source episodes".into(),
        };
        let episodes = (0..40)
            .map(|i| LongitudinalEpisode {
                id: i.to_string(),
                covariates: vec![0],
                assignments: vec![i % 2],
                outcome: if i % 2 == 0 {
                    if i < 10 { 1.0 } else { 0.0 }
                } else {
                    if i < 30 { 1.0 } else { 0.0 }
                },
            })
            .collect::<Vec<_>>();
        let prior = LongitudinalDirichletPrior {
            covariate_states: vec![vec![0]],
            outcome_values: vec![0.0, 1.0],
            concentration: 1.0,
            draws: 20000,
            seed: 18,
        };
        let posterior = longitudinal_joint_posterior(&design, &episodes, &prior).unwrap();
        for j in 0..2 {
            let ones = episodes.iter().filter(|e| e.assignments[0] == j && e.outcome == 1.0).count()
                as f64;
            let a = ones + 1.0;
            let b = 21.0 - ones;
            let mean = a / (a + b);
            let variance = a * b / ((a + b).powi(2) * (a + b + 1.0));
            assert!(
                (posterior.estimate.means[j] - mean).abs()
                    < 6.0 * (variance / prior.draws as f64).sqrt()
            );
            assert!((posterior.estimate.covariance[j * 2 + j] - variance).abs() < 0.001);
        }
        let mut unsupported = prior.clone();
        unsupported.covariate_states[0].push(1);
        assert!(longitudinal_joint_posterior(&design, &episodes, &unsupported).is_err());
    }
}
