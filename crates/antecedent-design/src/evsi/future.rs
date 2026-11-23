use super::IntegrationMethod;
use crate::preposterior::expected_max_affine;
use antecedent_core::is_blank;
use std::collections::BTreeSet;
use thiserror::Error;

/// Why a future-study program refuses.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum FutureStudyError {
    /// Prior mean not finite, or prior variance not finite and strictly positive.
    #[error("a normal state belief needs a finite mean and a finite variance greater than zero")]
    InvalidBelief,
    /// No observations, or a noise variance that is not finite and strictly positive.
    #[error("future data needs at least one observation and finite noise variance > 0")]
    InvalidFutureData,
    /// The observed sample mean is not finite.
    #[error("the future sample mean is not finite")]
    InvalidObservation,
    /// The terminal action set is empty.
    #[error("a future study needs at least one terminal action")]
    EmptyActions,
    /// An action id is empty or only whitespace.
    #[error("an action id is blank")]
    BlankAction,
    /// Two actions share an id.
    #[error("action ids must be unique")]
    DuplicateAction,
    /// Coefficients are missing, mis-aligned, or not finite.
    #[error("an affine utility needs one finite intercept and slope per action")]
    InvalidUtility,
    /// A declared cost is negative or not finite. An absent cost is not this error.
    #[error("a declared study cost must be finite and nonnegative")]
    InvalidCost,
    /// A conjugate quantity that should be finite was not.
    #[error("the conjugate update or its preposterior value was not finite")]
    NonFiniteValue,
}

/// Normal belief `θ ~ N(mean, variance)` over a scalar decision state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalStateBelief {
    mean: f64,
    variance: f64,
}

impl NormalStateBelief {
    /// Finite `mean` and finite `variance` strictly greater than zero.
    ///
    /// # Errors
    ///
    /// [`FutureStudyError::InvalidBelief`] otherwise.
    pub const fn try_new(mean: f64, variance: f64) -> Result<Self, FutureStudyError> {
        if !mean.is_finite() || !finite_positive(variance) {
            return Err(FutureStudyError::InvalidBelief);
        }
        Ok(Self { mean, variance })
    }

    /// Prior mean `μ₀`.
    #[must_use]
    pub const fn mean(self) -> f64 {
        self.mean
    }

    /// Prior variance `τ²`.
    #[must_use]
    pub const fn variance(self) -> f64 {
        self.variance
    }
}

/// `n` i.i.d. observations `N(θ, noise_variance)`. The sufficient statistic is their mean.
///
/// `noise_variance` is the variance of one observation, not the variance of the mean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalMeanFutureData {
    sample_size: u64,
    noise_variance: f64,
}

impl NormalMeanFutureData {
    /// `sample_size ≥ 1` and finite `noise_variance` strictly greater than zero.
    ///
    /// # Errors
    ///
    /// [`FutureStudyError::InvalidFutureData`] otherwise.
    pub const fn try_new(sample_size: u64, noise_variance: f64) -> Result<Self, FutureStudyError> {
        if sample_size == 0 || !finite_positive(noise_variance) {
            return Err(FutureStudyError::InvalidFutureData);
        }
        Ok(Self { sample_size, noise_variance })
    }

    /// Number of i.i.d. observations.
    #[must_use]
    pub const fn sample_size(self) -> u64 {
        self.sample_size
    }

    /// Variance of one observation.
    #[must_use]
    pub const fn noise_variance(self) -> f64 {
        self.noise_variance
    }
}

/// Conjugate update for a normal mean observed with known variance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NormalMeanKnownVarianceUpdate;

/// Posterior `N(mean, variance)` after [`NormalMeanKnownVarianceUpdate`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalPosterior {
    /// Posterior mean.
    pub mean: f64,
    /// Posterior variance.
    pub variance: f64,
}

impl NormalMeanKnownVarianceUpdate {
    /// Posterior after observing `sample_mean` from `data`.
    ///
    /// Precision adds: `1/τ² + n/σ²`. The posterior mean is the precision-weighted
    /// average of the prior mean and the sample mean. The posterior variance does
    /// not depend on the sample mean.
    ///
    /// # Errors
    ///
    /// [`FutureStudyError::InvalidObservation`] when `sample_mean` is not finite.
    /// [`FutureStudyError::NonFiniteValue`] when the updated mean or variance is not
    /// a finite positive-variance normal.
    pub const fn update(
        self,
        belief: NormalStateBelief,
        data: NormalMeanFutureData,
        sample_mean: f64,
    ) -> Result<NormalPosterior, FutureStudyError> {
        let Self = self;
        if !sample_mean.is_finite() {
            return Err(FutureStudyError::InvalidObservation);
        }
        let prior_precision = 1.0 / belief.variance();
        let data_precision = data.sample_size() as f64 / data.noise_variance();
        let posterior_precision = prior_precision + data_precision;
        let mean =
            (prior_precision * belief.mean() + data_precision * sample_mean) / posterior_precision;
        let variance = 1.0 / posterior_precision;
        if !mean.is_finite() || !finite_positive(variance) {
            return Err(FutureStudyError::NonFiniteValue);
        }
        Ok(NormalPosterior { mean, variance })
    }
}

/// One terminal action. The set is the same before and after the study.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StudyAction {
    id: String,
}

impl StudyAction {
    /// An id that is not blank.
    ///
    /// # Errors
    ///
    /// [`FutureStudyError::BlankAction`] when `id` is empty or only whitespace.
    pub fn try_new(id: &str) -> Result<Self, FutureStudyError> {
        if is_blank(id) {
            return Err(FutureStudyError::BlankAction);
        }
        Ok(Self { id: id.to_owned() })
    }

    /// Action id.
    #[must_use]
    pub fn id(&self) -> &str {
        self.id.as_str()
    }
}

/// Utility of each action, affine in the scalar state: `intercept + slope · θ`.
///
/// Coefficients are aligned with [`FutureStudyProgram`]'s action list.
#[derive(Clone, Debug, PartialEq)]
pub struct AffineStateUtility {
    intercept: Vec<f64>,
    slope: Vec<f64>,
}

impl AffineStateUtility {
    /// One finite intercept and slope per action.
    ///
    /// # Errors
    ///
    /// [`FutureStudyError::InvalidUtility`] when the vectors differ in length, are
    /// empty, or contain a non-finite coefficient.
    pub fn try_new(intercept: Vec<f64>, slope: Vec<f64>) -> Result<Self, FutureStudyError> {
        if intercept.is_empty() || intercept.len() != slope.len() {
            return Err(FutureStudyError::InvalidUtility);
        }
        if intercept.iter().chain(&slope).any(|coef| !coef.is_finite()) {
            return Err(FutureStudyError::InvalidUtility);
        }
        Ok(Self { intercept, slope })
    }
}

/// Future study: belief, data generator, update, actions, utility, and an optional cost.
///
/// Gross EVSI is the expected increase in expected utility from seeing the study
/// before acting. It does not read [`Self::study_cost`]. Net value subtracts that
/// cost only when a cost is declared. `None` is not a zero cost.
#[derive(Clone, Debug, PartialEq)]
pub struct FutureStudyProgram {
    belief: NormalStateBelief,
    future_data: NormalMeanFutureData,
    update: NormalMeanKnownVarianceUpdate,
    actions: Vec<StudyAction>,
    utility: AffineStateUtility,
    study_cost: Option<f64>,
}

/// One draw of the update-and-decision loop: the posterior, then its Bayes action.
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use = "the posterior and the action it selects are one step of the decision loop"]
pub struct UpdatedDecision {
    /// Index into the program's action list. Ties keep the earliest action.
    pub action_index: usize,
    /// Posterior mean of the state.
    pub posterior_mean: f64,
    /// Posterior variance of the state.
    pub posterior_variance: f64,
    /// Expected utility of `action_index` under the posterior.
    pub expected_utility: f64,
}

/// Gross EVSI, and net value only when the program declared a study cost.
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use = "gross EVSI is separate from an optional study cost"]
pub struct FutureStudyValue {
    /// Expected utility of the Bayes action under the current belief.
    pub prior_expected_utility: f64,
    /// `E[max_a E[U(a, θ) | study]]`, the preposterior value of acting after the study.
    pub preposterior_expected_utility: f64,
    /// `preposterior_expected_utility − prior_expected_utility`. Does not subtract cost.
    pub gross_evsi: f64,
    /// The program's declared cost, or `None` when it has no cost.
    pub study_cost: Option<f64>,
    /// `gross_evsi − study_cost` when a cost is declared, and `None` when it is not.
    pub net_value: Option<f64>,
    /// How the preposterior expectation was obtained.
    pub integration: IntegrationMethod,
}

impl FutureStudyProgram {
    /// Assemble a program. `study_cost` is utility units already; `None` declares no cost.
    ///
    /// # Errors
    ///
    /// An empty action list, a blank or duplicate id, a utility that does not match
    /// the actions, or a declared cost that is negative or not finite.
    pub fn try_new(
        belief: NormalStateBelief,
        future_data: NormalMeanFutureData,
        update: NormalMeanKnownVarianceUpdate,
        actions: Vec<StudyAction>,
        utility: AffineStateUtility,
        study_cost: Option<f64>,
    ) -> Result<Self, FutureStudyError> {
        if actions.is_empty() {
            return Err(FutureStudyError::EmptyActions);
        }
        if utility.intercept.len() != actions.len() {
            return Err(FutureStudyError::InvalidUtility);
        }
        let mut seen = BTreeSet::new();
        for action in &actions {
            if !seen.insert(action.id.as_str()) {
                return Err(FutureStudyError::DuplicateAction);
            }
        }
        if !accepts_cost(study_cost) {
            return Err(FutureStudyError::InvalidCost);
        }
        Ok(Self { belief, future_data, update, actions, utility, study_cost })
    }

    /// Declared study cost. `None` means the program has no cost.
    #[must_use = "an absent cost is not a zero cost"]
    pub const fn study_cost(&self) -> Option<f64> {
        self.study_cost
    }

    /// Terminal actions, in the order the utility coefficients use.
    #[must_use]
    pub fn actions(&self) -> &[StudyAction] {
        self.actions.as_slice()
    }

    /// Update on `sample_mean`, then pick the Bayes action under that posterior.
    ///
    /// Ties keep the earliest action. The terminal set is the program's action list.
    ///
    /// # Errors
    ///
    /// [`FutureStudyError::InvalidObservation`] or [`FutureStudyError::NonFiniteValue`].
    pub fn updated_decision(&self, sample_mean: f64) -> Result<UpdatedDecision, FutureStudyError> {
        let posterior = self.update.update(self.belief, self.future_data, sample_mean)?;
        let (action_index, expected_utility) =
            bayes_expected_utility(&self.utility, posterior.mean)?;
        Ok(UpdatedDecision {
            action_index,
            posterior_mean: posterior.mean,
            posterior_variance: posterior.variance,
            expected_utility,
        })
    }

    /// Gross EVSI. The study cost is not read and is not subtracted.
    ///
    /// For this conjugate model the posterior mean is preposteriorly `N(μ₀, s²)` with
    /// `s² = τ² − (1/τ² + n/σ²)⁻¹`, and
    ///
    /// ```text
    /// EVSI = E[max_a (α_a + β_a μ)] − max_a (α_a + β_a μ₀)
    /// ```
    ///
    /// with `μ ~ N(μ₀, s²)`. That expectation is the normal linear-loss integral of the
    /// upper envelope. It is not a Monte Carlo estimate.
    ///
    /// # Errors
    ///
    /// [`FutureStudyError::NonFiniteValue`] when the integral is not finite.
    pub fn gross_evsi(&self) -> Result<f64, FutureStudyError> {
        Ok(self.score()?.gross_evsi)
    }

    /// Gross EVSI, and net value only when [`Self::study_cost`] is `Some`.
    ///
    /// # Errors
    ///
    /// [`FutureStudyError::NonFiniteValue`] when the integral is not finite.
    /// [`FutureStudyError::InvalidCost`] when a declared cost is not finite and nonnegative.
    pub fn evaluate_future_study(&self) -> Result<FutureStudyValue, FutureStudyError> {
        let scored = self.score()?;
        Ok(FutureStudyValue {
            prior_expected_utility: scored.prior_expected_utility,
            preposterior_expected_utility: scored.preposterior_expected_utility,
            gross_evsi: scored.gross_evsi,
            study_cost: self.study_cost,
            net_value: net_of(scored.gross_evsi, self.study_cost)?,
            integration: IntegrationMethod::Exact,
        })
    }
}

struct PreposteriorScore {
    prior_expected_utility: f64,
    preposterior_expected_utility: f64,
    gross_evsi: f64,
}

const fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

const fn finite_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

const fn accepts_cost(study_cost: Option<f64>) -> bool {
    match study_cost {
        None => true,
        Some(cost) => finite_nonnegative(cost),
    }
}

const fn net_of(gross_evsi: f64, study_cost: Option<f64>) -> Result<Option<f64>, FutureStudyError> {
    match study_cost {
        None => Ok(None),
        Some(cost) if finite_nonnegative(cost) => Ok(Some(gross_evsi - cost)),
        Some(_) => Err(FutureStudyError::InvalidCost),
    }
}

/// Standard deviation of the preposterior mean: `Var(E[θ | ȳ]) = τ²·nτ²/(nτ² + σ²)`.
///
/// Subtracting the posterior variance from `τ²` cancels catastrophically when
/// `nτ²/σ²` is small, so the closed form is used.
fn preposterior_spread(
    belief: NormalStateBelief,
    data: NormalMeanFutureData,
) -> Result<f64, FutureStudyError> {
    let prior_variance = belief.variance();
    let information = data.sample_size() as f64 * prior_variance;
    let spread_sq = prior_variance * information / (information + data.noise_variance());
    if !spread_sq.is_finite() {
        return Err(FutureStudyError::NonFiniteValue);
    }
    Ok(spread_sq.sqrt())
}

fn bayes_expected_utility(
    utility: &AffineStateUtility,
    mean: f64,
) -> Result<(usize, f64), FutureStudyError> {
    let mut chosen: Option<(usize, f64)> = None;
    for (index, (&intercept, &slope)) in utility.intercept.iter().zip(&utility.slope).enumerate() {
        let value = intercept + slope * mean;
        if !value.is_finite() {
            return Err(FutureStudyError::NonFiniteValue);
        }
        if chosen.is_none_or(|(_, best)| value > best) {
            chosen = Some((index, value));
        }
    }
    chosen.ok_or(FutureStudyError::EmptyActions)
}

/// Gap-to-Bayes lines `(Δc_a, Δβ_a)`: the action's expected utility at the prior mean
/// minus the Bayes action's, and its slope minus the Bayes action's. The Bayes action's
/// own line is `(0, 0)`, so the preposterior gain is computed without subtracting two
/// large absolute utilities.
fn gap_lines(utility: &AffineStateUtility, mean: f64, bayes: usize) -> Vec<(f64, f64)> {
    let bayes_value = utility.intercept[bayes] + utility.slope[bayes] * mean;
    let bayes_slope = utility.slope[bayes];
    utility
        .intercept
        .iter()
        .zip(&utility.slope)
        .map(|(&intercept, &slope)| (intercept + slope * mean - bayes_value, slope - bayes_slope))
        .collect()
}

impl FutureStudyProgram {
    fn score(&self) -> Result<PreposteriorScore, FutureStudyError> {
        let spread = preposterior_spread(self.belief, self.future_data)?;
        let mean = self.belief.mean();
        let (bayes, prior_expected_utility) = bayes_expected_utility(&self.utility, mean)?;
        let lines = gap_lines(&self.utility, mean, bayes);
        let gross_evsi = expected_max_affine(&lines, spread);
        let preposterior_expected_utility = prior_expected_utility + gross_evsi;
        if !prior_expected_utility.is_finite()
            || !preposterior_expected_utility.is_finite()
            || !gross_evsi.is_finite()
        {
            return Err(FutureStudyError::NonFiniteValue);
        }
        Ok(PreposteriorScore { prior_expected_utility, preposterior_expected_utility, gross_evsi })
    }
}
