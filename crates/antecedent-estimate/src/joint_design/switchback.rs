//! Finite-carryover joint-history design and a distinct Gaussian sequence likelihood.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{JointDesignEstimate, JointDesignPosterior, JointUncertainty};
use crate::switchback::{HistoryAssignment, switchback_history_means};
use std::collections::BTreeSet;
/// Explicit history support and causal restrictions of a switchback experiment.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JointSwitchbackDesign {
    /// Complete joint histories, each with one arm at every carryover position.
    pub worlds: Vec<(String, Vec<usize>)>,
    /// Maximum carryover order; each retained history must contain exactly order+1 arms.
    pub carryover: usize,
    /// Explicit no-anticipation restriction.
    pub no_anticipation: String,
    /// Scientific justification of finite carryover or washout at the declared order.
    pub finite_carryover: String,
    /// No between-sequence interference and independent assignment units.
    pub sequence_isolation: String,
    /// Averaging population and time domain, including exclusion of initial unobserved history.
    pub population: String,
}
/// Gaussian conditional joint-history response model and proper mean priors.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SwitchbackGaussianPrior {
    /// Prior mean of each history's constant conditional response.
    pub mean: Vec<f64>,
    /// Proper independent prior variances.
    pub variance: Vec<f64>,
    /// Known marginal Gaussian residual variance of each independent sequence.
    pub sequence_variance: Vec<f64>,
    /// Known AR(1) residual coefficient within each sequence, absolute value below one.
    pub sequence_rho: Vec<f64>,
    /// Aligned posterior draws.
    pub draws: usize,
    /// Deterministic posterior stream.
    pub seed: u64,
}
fn checked_histories(design: &JointSwitchbackDesign) -> Result<Vec<Vec<usize>>, String> {
    if design.carryover > 32
        || design.worlds.len() > 256
        || design.worlds.is_empty()
        || design.worlds.iter().any(|(key, h)| key.is_empty() || h.len() != design.carryover + 1)
        || design.worlds.iter().map(|(key, _)| key).collect::<BTreeSet<_>>().len()
            != design.worlds.len()
        || [
            &design.no_anticipation,
            &design.finite_carryover,
            &design.sequence_isolation,
            &design.population,
        ]
        .iter()
        .any(|s| s.trim().is_empty())
    {
        return Err("switchback requires complete finite-carryover histories and explicit no-anticipation/isolation/population premises".into());
    }
    Ok(design.worlds.iter().map(|(_, history)| history.clone()).collect())
}
/// Design-based joint-history means using actual whole-schedule inclusion probabilities.
/// The original switchback owner excludes the initial carryover periods and uses
/// sequence-level covariance; marginal arm probabilities cannot replace history support.
/// # Errors
/// Missing causal premises, malformed assignment law, missing joint histories or invalid outcomes.
pub fn switchback_joint_means(
    design: &JointSwitchbackDesign,
    sequences: &[HistoryAssignment],
) -> Result<JointDesignEstimate, String> {
    if sequences.len() > 65536 {
        return Err("switchback independent-sequence resource bound exceeded".into());
    }
    let histories = checked_histories(design)?;
    let fit = switchback_history_means(sequences, &histories).map_err(str::to_string)?;
    Ok(JointDesignEstimate {
        worlds: design.worlds.iter().map(|(key, _)| key.clone()).collect(),
        means: fit.means,
        covariance: fit.covariance,
        sampling_unit: "independently randomized complete sequence".into(),
        uncertainty: JointUncertainty::IndependentUnitSandwich,
    })
}
/// Bayesian joint-history means with proper Gaussian mean priors and known AR(1)
/// within-sequence residual likelihood. Scientific identification is shared with
/// the design-based owner; stationarity of conditional history means and the known
/// residual covariance are additional statistical assumptions retained by this law.
/// # Errors
/// Invalid design/likelihood/prior, unmodeled realized history or numerical factorization failure.
pub fn switchback_joint_posterior(
    design: &JointSwitchbackDesign,
    sequences: &[HistoryAssignment],
    prior: &SwitchbackGaussianPrior,
) -> Result<JointDesignPosterior, String> {
    let _identified = switchback_joint_means(design, sequences)?;
    let histories = checked_histories(design)?;
    let k = histories.len();
    if prior.mean.len() != k
        || prior.variance.len() != k
        || prior.sequence_variance.len() != sequences.len()
        || prior.sequence_rho.len() != sequences.len()
        || prior.draws < 2
        || prior.draws > 100_000
        || prior.mean.iter().any(|v| !v.is_finite())
        || prior
            .variance
            .iter()
            .chain(&prior.sequence_variance)
            .any(|v| !v.is_finite() || *v <= 0.0)
        || prior.sequence_rho.iter().any(|v| !v.is_finite() || v.abs() >= 1.0)
    {
        return Err("switchback Gaussian posterior requires proper priors and known nonsingular sequence covariance".into());
    }
    let mut precision = vec![0.0; k * k];
    let mut rhs = vec![0.0; k];
    for j in 0..k {
        precision[j * k + j] = 1.0 / prior.variance[j];
        rhs[j] = prior.mean[j] / prior.variance[j];
    }
    for (index, sequence) in sequences.iter().enumerate() {
        let schedule = &sequence.schedules[sequence.realized];
        let first = design.carryover;
        let m = schedule.len() - first;
        let slots = (first..schedule.len())
            .map(|t| {
                histories
                    .iter()
                    .position(|history| schedule[t - first..=t] == history[..])
                    .ok_or_else(|| {
                        "realized eligible history has no declared Gaussian response coordinate"
                            .to_string()
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let rho = prior.sequence_rho[index];
        let variance = prior.sequence_variance[index];
        if m == 1 {
            let j = slots[0];
            precision[j * k + j] += 1.0 / variance;
            rhs[j] += sequence.outcomes[first] / variance;
            continue;
        }
        let scale = 1.0 / (variance * (1.0 - rho * rho));
        for t in 0..m {
            let j = slots[t];
            let diagonal = if t == 0 || t + 1 == m { scale } else { (1.0 + rho * rho) * scale };
            precision[j * k + j] += diagonal;
            rhs[j] += diagonal * sequence.outcomes[first + t];
            if t + 1 < m {
                let l = slots[t + 1];
                let off = -rho * scale;
                precision[j * k + l] += off;
                precision[l * k + j] += off;
                rhs[j] += off * sequence.outcomes[first + t + 1];
                rhs[l] += off * sequence.outcomes[first + t];
            }
        }
    }
    let chol = antecedent_stats::cholesky_spd(&precision, k)
        .ok_or("joint-history posterior precision is not positive definite")?;
    let mean = antecedent_stats::chol_solve(&chol, k, &rhs)
        .ok_or("joint-history posterior mean solve failed")?;
    let mut covariance = vec![0.0; k * k];
    for j in 0..k {
        let mut basis = vec![0.0; k];
        basis[j] = 1.0;
        let column = antecedent_stats::chol_solve(&chol, k, &basis)
            .ok_or("joint-history posterior covariance solve failed")?;
        for l in 0..k {
            covariance[l * k + j] = column[l];
        }
    }
    let raw = antecedent_prob::sample_gaussian_mvn(
        &mean,
        &covariance,
        prior.draws,
        prior.seed,
        &mut antecedent_prob::LaplaceWorkspace::default(),
    )
    .map_err(|e| e.to_string())?
    .to_vec();
    // The shared MVN sampler stores coordinate-major columns. Restore draw-major
    // rows explicitly so one row remains one complete joint mechanism realization.
    let draws =
        (0..prior.draws).map(|d| (0..k).map(|j| raw[j * prior.draws + d]).collect()).collect();
    JointDesignPosterior::from_draws(design.worlds.iter().map(|(key,_)|key.clone()).collect(),draws,"independently randomized complete sequence".into(),"Gaussian stationary joint-history conditional means with declared known sequence AR(1) residual covariance and proper independent Gaussian mean priors; finite-carryover no-anticipation design retained".into())
}

/// Exhaustive finite-assignment inference for the sharp null of no history effects.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct JointSwitchbackRandomizationTest {
    /// Actual retained joint-history contrast statistic.
    pub observed: f64,
    /// Lower/upper weighted tail probability; equal when the finite distribution is exhausted.
    pub p_value: [f64; 2],
    /// Fully evaluated whole-program assignment combinations.
    pub assignments_evaluated: usize,
    /// Whether the exact finite assignment distribution was exhausted.
    pub exhaustive: bool,
    /// Explicit sharp null; this is not inference for a weak average-effect null.
    pub null: String,
}
/// Exact whole-schedule randomization inference for a prespecified joint-history
/// contrast under the sharp null that all potential history responses equal the
/// observed outcomes. Arbitrary within-sequence dependence is retained. At a case
/// budget the unknown assignment probability mass yields a valid partial tail
/// interval; no asymptotic or exhaustive claim replaces an unfinished enumeration.
/// # Errors
/// Invalid design/contrast/distribution, case budget, cancellation or arithmetic failure.
pub fn switchback_joint_sharp_null_test(
    design: &JointSwitchbackDesign,
    sequences: &[HistoryAssignment],
    contrast: &[f64],
    max_cases: usize,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<JointSwitchbackRandomizationTest, String> {
    let histories = checked_histories(design)?;
    let _observed_support = switchback_joint_means(design, sequences)?;
    let k = histories.len();
    if contrast.len() != k
        || contrast.iter().any(|c| !c.is_finite())
        || contrast.iter().all(|c| *c == 0.0)
        || max_cases == 0
        || max_cases > 10_000_000
    {
        return Err("switchback randomization inference requires a finite nonzero joint contrast and bounded positive case budget".into());
    }
    let total = sequences.iter().map(|s| s.outcomes.len() - design.carryover).sum::<usize>() as f64;
    let mut schedule_statistics = Vec::with_capacity(sequences.len());
    for sequence in sequences {
        let mut values = vec![0.0; sequence.schedules.len()];
        for t in design.carryover..sequence.outcomes.len() {
            for (j, history) in histories.iter().enumerate() {
                let probability = sequence
                    .schedules
                    .iter()
                    .zip(&sequence.probabilities)
                    .filter(|(s, _)| s[t - design.carryover..=t] == history[..])
                    .map(|(_, p)| p)
                    .sum::<f64>();
                if probability <= 0.0 {
                    return Err(
                        "history assignment positivity failure in randomization statistic".into()
                    );
                }
                for (index, schedule) in sequence.schedules.iter().enumerate() {
                    if schedule[t - design.carryover..=t] == history[..] {
                        values[index] += contrast[j] * sequence.outcomes[t] / probability / total;
                    }
                }
            }
        }
        schedule_statistics.push(values);
    }
    let observed =
        sequences.iter().zip(&schedule_statistics).map(|(s, v)| v[s.realized]).sum::<f64>();
    let mut indices = vec![0usize; sequences.len()];
    let mut evaluated = 0usize;
    let mut visited = 0.0;
    let mut tail = 0.0;
    let mut exhaustive = false;
    loop {
        if ctx.cancellation.is_cancelled() {
            return Err("switchback exact randomization inference cancelled".into());
        }
        let probability = sequences
            .iter()
            .zip(&indices)
            .map(|(s, index)| s.probabilities[*index] / s.probabilities.iter().sum::<f64>())
            .product::<f64>();
        let statistic = schedule_statistics
            .iter()
            .zip(&indices)
            .map(|(values, index)| values[*index])
            .sum::<f64>();
        if !probability.is_finite() || !statistic.is_finite() {
            return Err("randomization probability/statistic overflow".into());
        }
        visited += probability;
        if statistic.abs() >= observed.abs() {
            tail += probability;
        }
        evaluated += 1;
        let mut carry = true;
        for j in (0..indices.len()).rev() {
            indices[j] += 1;
            if indices[j] < sequences[j].schedules.len() {
                carry = false;
                break;
            }
            indices[j] = 0;
        }
        if carry {
            exhaustive = true;
            break;
        }
        if evaluated >= max_cases {
            break;
        }
    }
    let remaining = if exhaustive { 0.0 } else { (1.0 - visited).max(0.0) };
    // A deterministic rounding envelope is retained even for exhaustive evaluation.
    let probability_operations =
        sequences.iter().map(|s| s.probabilities.len() as f64 + 2.0).sum::<f64>();
    let rounding = (evaluated as f64) * (probability_operations + 8.0) * f64::EPSILON;
    Ok(JointSwitchbackRandomizationTest{observed,p_value:[(tail-rounding).max(0.0),(tail+remaining+rounding).min(1.0)],assignments_evaluated:evaluated,exhaustive,null:"sharp null: every eligible potential response is invariant to the complete finite-carryover joint history".into()})
}

#[cfg(test)]
mod exact_tests {
    use super::*;
    #[test]
    fn whole_schedule_sharp_null_probability_and_partial_mass_are_independent_of_asymptotics() {
        let design = JointSwitchbackDesign {
            worlds: vec![("base".into(), vec![0]), ("joint".into(), vec![1])],
            carryover: 0,
            no_anticipation: "no future effects".into(),
            finite_carryover: "no carryover".into(),
            sequence_isolation: "independent clusters".into(),
            population: "three eligible sequence outcomes".into(),
        };
        let sequences = [(1.0, 0), (2.0, 1), (4.0, 1)]
            .into_iter()
            .map(|(y, realized)| HistoryAssignment {
                schedules: vec![vec![0], vec![1]],
                probabilities: vec![0.5; 2],
                realized,
                outcomes: vec![y],
            })
            .collect::<Vec<_>>();
        let exact = switchback_joint_sharp_null_test(
            &design,
            &sequences,
            &[-1.0, 1.0],
            8,
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert!(exact.exhaustive);
        assert_eq!(exact.assignments_evaluated, 8);
        assert!(exact.p_value[0] <= 0.5 && exact.p_value[1] >= 0.5);
        assert!(exact.p_value[1] - exact.p_value[0] < 1e-12);
        let partial = switchback_joint_sharp_null_test(
            &design,
            &sequences,
            &[-1.0, 1.0],
            3,
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert!(!partial.exhaustive);
        assert!(partial.p_value[0] <= 0.5 && partial.p_value[1] >= 0.5);
    }
}
