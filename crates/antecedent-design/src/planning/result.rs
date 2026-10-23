//! What a plan reports, and how a plan feeds design ranking.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{SemanticDigest, VariableId};

use crate::candidate::{CandidateDesign, DesignCost, ExperimentPlan, SamplingPlan};
use crate::evsi::{FutureStudyError, NormalMeanFutureData};

/// Familywise characteristics of a many-to-one comparison.
#[derive(Clone, Debug, PartialEq)]
pub struct FamilyCharacteristics {
    /// Probability that at least one comparison rejects.
    pub any: f64,
    /// Probability that every comparison rejects.
    pub all: f64,
    /// Probability that a given comparison rejects.
    pub per_comparison: f64,
    /// Critical values of the procedure, largest first.
    pub critical: Vec<f64>,
    /// Correlation of two comparison statistics through the shared control.
    pub correlation: f64,
}

/// Operating characteristics of a group-sequential boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct SequentialCharacteristics {
    /// Critical value on the z statistic at each look.
    pub critical: Vec<f64>,
    /// Enrolled units at each look.
    pub look_units: Vec<u64>,
    /// Probability of stopping for efficacy first at each look, at the effect.
    pub stop_probability_alternative: Vec<f64>,
    /// Probability of stopping for efficacy first at each look, at no effect.
    pub stop_probability_null: Vec<f64>,
    /// Expected enrolled units at the effect.
    pub expected_units_alternative: f64,
    /// Expected enrolled units at no effect.
    pub expected_units_null: f64,
    /// Units a fixed design with the z test needs for the same power, when the
    /// plan solved for a size.
    pub fixed_design_units: Option<u64>,
}

/// Analytic operating characteristics at the planned size.
#[derive(Clone, Debug, PartialEq)]
pub struct AnalyticCharacteristics {
    /// The criterion: power, assurance, posterior standard deviation or
    /// probability of a correct decision.
    pub value: f64,
    /// Variance of the planned contrast estimator (per comparison for a
    /// multi-arm family, at the final look for a sequential design).
    pub estimator_variance: f64,
    /// Noncentrality of the test statistic (the F noncentrality for an omnibus
    /// test, the drift at full information for a sequential design).
    pub noncentrality: Option<f64>,
    /// Degrees of freedom of the reference law; `None` is the normal law.
    pub reference_df: Option<f64>,
    /// Critical value of a fixed-size test.
    pub critical_value: Option<f64>,
    /// Variance inflation of the design over individual randomization.
    pub design_effect: Option<f64>,
    /// Frequentist power at the effect, when the criterion is Bayesian and an
    /// effect is defined, or the criterion itself.
    pub power: Option<f64>,
    /// Familywise characteristics.
    pub family: Option<FamilyCharacteristics>,
    /// Sequential characteristics.
    pub sequential: Option<SequentialCharacteristics>,
    /// How the value was computed.
    pub method: &'static str,
}

/// Familywise simulation summary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FamilySimulation {
    /// Fraction of replicates with at least one rejection.
    pub any: f64,
    /// Fraction of replicates rejecting every comparison.
    pub all: f64,
    /// Fraction of comparisons rejected.
    pub per_comparison: f64,
}

/// Sequential simulation summary.
#[derive(Clone, Debug, PartialEq)]
pub struct SequentialSimulation {
    /// Fraction of replicates stopping first at each look.
    pub stop_frequency: Vec<f64>,
    /// Mean enrolled units.
    pub expected_units: f64,
    /// Monte Carlo standard error of the mean enrolled units.
    pub expected_units_se: f64,
}

/// Simulated operating characteristics: the declared data-generating process,
/// the real analysis estimator and the planned decision, replicated.
#[derive(Clone, Debug, PartialEq)]
pub struct SimulatedCharacteristics {
    /// Simulated criterion.
    pub value: f64,
    /// Monte Carlo standard error of [`Self::value`].
    pub monte_carlo_se: f64,
    /// Replicates run.
    pub replicates: u32,
    /// Seed of the keyed streams.
    pub seed: u64,
    /// Replicates whose estimator returned no estimate (counted as no rejection).
    pub failed_replicates: u32,
    /// Mean of the estimates.
    pub mean_estimate: f64,
    /// Standard deviation of the estimates.
    pub estimate_sd: f64,
    /// Familywise frequencies.
    pub family: Option<FamilySimulation>,
    /// Sequential frequencies.
    pub sequential: Option<SequentialSimulation>,
    /// `(simulated − analytic) / Monte Carlo SE`.
    pub agreement_z: Option<f64>,
}

/// A plan: the size, the effect and the operating characteristics, with the
/// declared request it answers.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanningResult {
    /// Identity of the planning request.
    pub request_identity: SemanticDigest,
    /// Planned design.
    pub design: &'static str,
    /// Analysis the characteristics are for.
    pub analysis: &'static str,
    /// Criterion.
    pub criterion: &'static str,
    /// Goal the plan solved.
    pub goal: &'static str,
    /// Randomization unit the size counts.
    pub unit: &'static str,
    /// Planned or given randomization units.
    pub units: u64,
    /// Enrolled units in each arm or cell (control first), in randomization units.
    pub arm_units: Vec<u64>,
    /// Enrolled individuals (unit-periods for a switchback design).
    pub enrolled_individuals: f64,
    /// Individuals whose outcome is analysed after attrition.
    pub analysed_individuals: f64,
    /// Assumed effect, or the minimum detectable effect the plan solved.
    pub effect: Option<f64>,
    /// Target of the criterion.
    pub target: f64,
    /// Whether the analytic criterion meets its target at the planned size.
    pub target_met: bool,
    /// Analytic characteristics.
    pub analytic: AnalyticCharacteristics,
    /// Simulated characteristics, when a simulation was requested and run.
    pub simulation: Option<SimulatedCharacteristics>,
    /// The declared assumptions, as `name=value` labels.
    pub assumptions: Vec<String>,
    /// Calibration status of the characteristics.
    pub calibration: &'static str,
}

impl PlanningResult {
    /// Enrolled sample size a ranking reads: individuals, rounded up.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a finite, nonnegative count of individuals"
    )]
    pub fn sample_size(&self) -> u64 {
        self.enrolled_individuals.ceil().max(0.0) as u64
    }

    /// An intervention candidate whose sample budget is this plan's sample
    /// size and whose cost is `cost_per_individual` per enrolled individual.
    #[must_use]
    pub fn experiment_candidate(
        &self,
        targets: Arc<[VariableId]>,
        cost_per_individual: f64,
        tag: u64,
    ) -> CandidateDesign {
        let sample_budget = self.sample_size();
        CandidateDesign::Intervene(ExperimentPlan {
            targets,
            cost: DesignCost { amount: cost_per_individual * sample_budget as f64, sample_budget },
            tag,
        })
    }

    /// A sampling candidate that adds this plan's sample size to an existing
    /// design.
    #[must_use]
    pub fn sampling_candidate(&self, cost_per_individual: f64, tag: u64) -> CandidateDesign {
        let sample_budget = self.sample_size();
        CandidateDesign::IncreaseSamplingRate(SamplingPlan {
            additional_samples: sample_budget,
            cost: DesignCost { amount: cost_per_individual * sample_budget as f64, sample_budget },
            tag,
        })
    }

    /// The planned contrast as future data of a value-of-information
    /// evaluation: the analysed individuals and the per-individual noise
    /// variance whose mean has the planned estimator variance.
    ///
    /// # Errors
    ///
    /// [`FutureStudyError::InvalidFutureData`] when the plan analyses no
    /// individual or its variance is not finite and positive.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a finite, nonnegative count of individuals"
    )]
    pub fn future_data(&self) -> Result<NormalMeanFutureData, FutureStudyError> {
        let sample_size = self.analysed_individuals.round().max(0.0) as u64;
        NormalMeanFutureData::try_new(
            sample_size,
            self.analytic.estimator_variance * sample_size as f64,
        )
    }
}
