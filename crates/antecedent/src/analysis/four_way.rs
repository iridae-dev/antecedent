//! Same-target four-way slice.
//!
//! Static frequentist, static Bayesian, horizon-one frequentist, and horizon-one
//! Bayesian realizations of one active-minus-control contrast. The static
//! frequentist cell executes the retained linear/Gaussian route. The other three
//! cells execute the existing study path on that same contrast. Frequentist
//! coverage and posterior quality are separate records. A utility-only change
//! on the static cell reuses the causal fit.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(not(test), allow(dead_code))]

use antecedent_contract::{DecisionRequest, EvidencePort, EvidencePortError};
use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, Conditioning, ContrastForm,
    ExecutionContext, GraphFamily, HistoryWindow, InferenceSelection, InitialLaw, IntervalMethod,
    Intervention, InterventionWorld, Lag, ModelAssumptions, ObservationUnit, OutcomeFunctional,
    OutcomeQuantity, QuantityUnits, ScientificTarget, ScientificWorldId, TargetForm,
    TemporalEffectQuery, TemporalPolicy, TimeFrame, Value, WorldAction, WorldBinding,
    WorldCollection,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_graph::ensure_lagged;
use antecedent_stats::fit_counts::count_least_squares_solves;

use crate::{
    BayesianConfig, Dag, EffectEstimate, InferenceMode, PublishedScalarUncertainty, RefuteSuite,
    Study, StudyResult, TemporalDag,
};

use super::prepared::recompile::{LifecycleInput, LifecycleProduct};

const LINE: [(f64, f64); 8] = [
    (0.0, 1.0),
    (1.0, 3.0),
    (0.0, 1.0),
    (1.0, 3.0),
    (0.0, 1.0),
    (1.0, 3.0),
    (0.0, 1.0),
    (1.0, 3.0),
];

const CONTROL_WORLD: u32 = 0;
const ACTIVE_WORLD: u32 = 1;
const POSTERIOR_DRAWS: usize = 1024;
const POSTERIOR_PRIOR_SCALE: f64 = 1.0e6;
const FREQUENTIST_LEVEL: f64 = 0.95;
/// Distance from the exact line effect inside which a zero-variance estimate
/// counts as the degenerate interval at that effect.
const DEGENERATE_TOLERANCE: f64 = 1.0e-12;

/// One cell of the static/horizon-one × frequentist/Bayesian slice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FourWayCell {
    /// Static ordinary-least-squares contrast.
    StaticFrequentist,
    /// Static conjugate posterior for the same contrast.
    StaticBayesian,
    /// Horizon-one frequentist pulse for the same contrast.
    HorizonOneFrequentist,
    /// Horizon-one conjugate posterior for the same contrast.
    HorizonOneBayesian,
}

impl FourWayCell {
    /// Declaration order is the slice order.
    pub(crate) const ALL: [Self; 4] = [
        Self::StaticFrequentist,
        Self::StaticBayesian,
        Self::HorizonOneFrequentist,
        Self::HorizonOneBayesian,
    ];

    /// Index into [`Self::ALL`].
    #[must_use]
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::StaticFrequentist => 0,
            Self::StaticBayesian => 1,
            Self::HorizonOneFrequentist => 2,
            Self::HorizonOneBayesian => 3,
        }
    }

    /// Whether this cell is a posterior realization.
    #[must_use]
    pub(crate) const fn is_bayesian(self) -> bool {
        matches!(self, Self::StaticBayesian | Self::HorizonOneBayesian)
    }

    /// Whether this cell is a sampling realization.
    #[must_use]
    pub(crate) const fn is_frequentist(self) -> bool {
        matches!(self, Self::StaticFrequentist | Self::HorizonOneFrequentist)
    }

    /// Stable name.
    #[must_use]
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::StaticFrequentist => "static_frequentist",
            Self::StaticBayesian => "static_bayesian",
            Self::HorizonOneFrequentist => "horizon_one_frequentist",
            Self::HorizonOneBayesian => "horizon_one_bayesian",
        }
    }
}

/// Carrier a decision still needs, or that a cell could not publish.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MissingCarrier {
    /// Parameter posterior. A confidence interval is not this carrier.
    ParameterPosterior,
    /// Predictive or interventional law. A mean is not this carrier.
    OutcomeLaw,
}

impl MissingCarrier {
    /// Stable name: `parameter_posterior` or `outcome_law`.
    #[must_use]
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::ParameterPosterior => "parameter_posterior",
            Self::OutcomeLaw => "outcome_law",
        }
    }
}

/// What the decision asked the cell to score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum DecisionDemand {
    /// Affine utility of the contrast, scored from a parameter posterior.
    AffinePosterior { slope: f64, intercept: f64 },
    /// Affine utility of the contrast, scored from the mean.
    AffineMean { slope: f64, intercept: f64 },
    /// Nonlinear utility. A mean cannot carry it.
    Nonlinear,
    /// Threshold or other outcome risk. A mean cannot carry it.
    Risk,
}

/// Affine score and the carrier that produced it.
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use = "an affine score names the carrier that produced it"]
pub(crate) struct ScoredDecision {
    /// `slope * location + intercept`.
    pub(crate) value: f64,
    /// Posterior or mean. Never a confidence interval.
    pub(crate) carrier: EvidencePort,
}

/// Location of one fitted cell.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FittedCell {
    cell: FourWayCell,
    effect: f64,
    standard_error: f64,
    interval_method: IntervalMethod,
    posterior_mean: Option<f64>,
    posterior_lower: Option<f64>,
    posterior_upper: Option<f64>,
}

impl FittedCell {
    /// Cell this fit belongs to.
    #[must_use]
    pub(crate) const fn cell(&self) -> FourWayCell {
        self.cell
    }

    /// Published scalar contrast.
    #[must_use]
    pub(crate) const fn effect(&self) -> f64 {
        self.effect
    }

    /// Analytic standard error. NaN when the cell withholds it.
    #[must_use]
    pub(crate) const fn standard_error(&self) -> f64 {
        self.standard_error
    }

    /// How the cell's interval was formed.
    #[must_use]
    pub(crate) const fn interval_method(&self) -> IntervalMethod {
        self.interval_method
    }

    /// Posterior mean of the contrast, when this cell published one.
    #[must_use]
    pub(crate) const fn posterior_mean(&self) -> Option<f64> {
        self.posterior_mean
    }

    /// Score `demand` from this cell.
    ///
    /// An affine posterior request uses the posterior mean. An affine mean
    /// request uses the published contrast and does not require a posterior.
    /// A nonlinear utility or an outcome risk is refused and names
    /// [`MissingCarrier::OutcomeLaw`]. A posterior request on a cell that has
    /// no posterior names [`MissingCarrier::ParameterPosterior`].
    pub(crate) fn decide(&self, demand: DecisionDemand) -> Result<ScoredDecision, MissingCarrier> {
        match demand {
            DecisionDemand::AffinePosterior { slope, intercept } => {
                let Some(mean) = self.posterior_mean else {
                    return Err(MissingCarrier::ParameterPosterior);
                };
                Ok(score(slope, mean, intercept, EvidencePort::ParameterPosterior))
            }
            DecisionDemand::AffineMean { slope, intercept } => {
                Ok(score(slope, self.effect, intercept, EvidencePort::Mean))
            }
            DecisionDemand::Nonlinear | DecisionDemand::Risk => Err(MissingCarrier::OutcomeLaw),
        }
    }
}

/// Static contrast and its horizon-one specialization.
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use = "compare both functionals with the analytic truth"]
pub(crate) struct FunctionalMatch {
    /// Active-minus-control form shared by the static and horizon-one targets.
    pub(crate) contrast: ContrastForm,
    /// Horizon of the static target. A static frame has none.
    pub(crate) static_horizon: Option<u32>,
    /// Horizon of the temporal target, in steps.
    pub(crate) horizon_steps: u32,
    /// Frequentist static contrast.
    pub(crate) static_effect: f64,
    /// Frequentist horizon-one contrast.
    pub(crate) horizon_effect: f64,
    /// Effect of the exact data line: mean treated outcome minus mean control
    /// outcome in the fixture, computed from its rows and not from any fit.
    pub(crate) analytic_truth: f64,
    /// Scientific identities differ because the time frames differ.
    pub(crate) identities_differ: bool,
    /// A frequentist or Bayesian label does not change the horizon-one identity.
    pub(crate) inference_label_preserves_identity: bool,
}

/// Sampling coverage for one frequentist cell. This is not a posterior probability.
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use = "coverage is a frequentist record, not a posterior probability"]
pub(crate) struct FrequentistCoverage {
    /// Frequentist cell.
    pub(crate) cell: FourWayCell,
    /// Nominal level of a published sampling interval.
    pub(crate) nominal_level: f64,
    /// A positive sampling standard error was published.
    pub(crate) interval_published: bool,
    /// The sampling interval contains the analytic truth.
    ///
    /// A zero analytic standard error is the degenerate interval at the fitted
    /// point. That point covers the truth when it is the truth. A withheld
    /// interval does not claim coverage.
    pub(crate) covers_analytic_truth: bool,
}

/// Posterior mean and equal-tailed quantiles. This is not a coverage rate.
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use = "posterior quality is not a coverage rate"]
pub(crate) struct PosteriorQuality {
    /// Bayesian cell.
    pub(crate) cell: FourWayCell,
    /// Posterior mean of the contrast.
    pub(crate) mean: f64,
    /// Absolute error of `mean` against the analytic truth.
    pub(crate) absolute_error: f64,
    /// Equal-tailed 2.5% quantile.
    pub(crate) lower: f64,
    /// Equal-tailed 97.5% quantile.
    pub(crate) upper: f64,
}

/// Result of [`FourWaySlice::utility_only_change`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use = "a utility change reports whether the causal fit was reused"]
pub(crate) struct UtilityReplay {
    /// Contrast returned by the reused fit.
    pub(crate) effect: f64,
    /// Least-squares solves during the replayed execute.
    pub(crate) least_squares_solves: u64,
    /// Causal fits recorded on the route after the replay.
    pub(crate) causal_fit_count: u64,
    /// The causal fit was still valid after the utility input.
    pub(crate) fit_valid: bool,
    /// The decision was invalid after the utility input and before execute.
    pub(crate) decision_valid_before_execute: bool,
    /// Execute restored the decision from the retained fit.
    pub(crate) decision_valid_after_execute: bool,
}

/// The four realizations, the shared functional, and the static utility replay.
#[must_use = "read the four cells before discarding the slice"]
pub(crate) struct FourWaySlice {
    cells: [FittedCell; 4],
    functional: FunctionalMatch,
    coverage: [FrequentistCoverage; 2],
    posterior: [PosteriorQuality; 2],
    static_fits: u64,
    utility_only: Box<dyn UtilityOnly>,
}

impl FourWaySlice {
    /// Fitted cell.
    #[must_use]
    pub(crate) const fn cell(&self, cell: FourWayCell) -> &FittedCell {
        &self.cells[cell.index()]
    }

    /// Static and horizon-one functionals, and the analytic truth.
    #[must_use = "the match is the comparison of the two functionals"]
    pub(crate) const fn functional_match(&self) -> FunctionalMatch {
        self.functional
    }

    /// Coverage of the two frequentist cells, in static then horizon-one order.
    #[must_use = "frequentist coverage is not posterior quality"]
    pub(crate) const fn frequentist_coverage(&self) -> [FrequentistCoverage; 2] {
        self.coverage
    }

    /// Posterior quality of the two Bayesian cells, in static then horizon-one order.
    #[must_use = "posterior quality is not a confidence interval"]
    pub(crate) const fn posterior_quality(&self) -> [PosteriorQuality; 2] {
        self.posterior
    }

    /// Causal fits the static frequentist route has run.
    #[must_use]
    pub(crate) const fn static_frequentist_fits(&self) -> u64 {
        self.static_fits
    }

    /// Apply a utility change on the static frequentist cell and execute again.
    ///
    /// The route reuses the causal fit. [`UtilityReplay::least_squares_solves`]
    /// counts solves inside that execute.
    pub(crate) fn utility_only_change(&self) -> Result<UtilityReplay, crate::CausalError> {
        self.utility_only.replay()
    }
}

/// Offer a confidence interval as a parameter posterior.
///
/// The offer is refused. The interval is unchanged, and the request stays a
/// parameter posterior.
pub(crate) fn confidence_interval_as_posterior() -> Result<EvidencePort, EvidencePortError> {
    EvidencePort::ConfidenceRegion.try_coerce(DecisionRequest::ParameterPosterior)
}

/// Fit the four cells on the perfect-line contrast.
///
/// # Errors
///
/// A study or the lowered route refuses, or a Bayesian cell publishes no
/// effect posterior.
pub(crate) fn same_target_slice(
    ctx: &ExecutionContext,
) -> Result<FourWaySlice, crate::CausalError> {
    let static_study = linear_study(FourWayCell::StaticFrequentist)?;
    let route = static_study.lower_linear_gaussian_intervention(ctx)?;
    let compiled = static_study.compile(ctx)?;
    if compiled.record != route.physical().record
        || compiled.logical.record != route.physical().logical.record
    {
        return Err(compile("compile did not return the lowered linear/Gaussian physical plan"));
    }
    let analytic_truth = line_effect();
    let claim = route.execute(ctx)?;
    let static_target = route.scientific().clone();
    let static_fits = route.causal_fit_count();
    let static_cell = frequentist_cell(FourWayCell::StaticFrequentist, &claim.estimate)?;
    let replay_ctx = ctx.clone();
    let utility_only = Box::new(CapturedUtility {
        replay_fn: move || {
            route.apply_input(LifecycleInput::Utility);
            let fit_valid = route.product_is_valid(LifecycleProduct::CausalFit);
            let decision_before = route.product_is_valid(LifecycleProduct::Decision);
            let (executed, solves) = count_least_squares_solves(|| route.execute(&replay_ctx));
            let replayed = executed?;
            Ok(UtilityReplay {
                effect: replayed.estimate.ate,
                least_squares_solves: solves,
                causal_fit_count: route.causal_fit_count(),
                fit_valid,
                decision_valid_before_execute: decision_before,
                decision_valid_after_execute: route.product_is_valid(LifecycleProduct::Decision),
            })
        },
    });

    let horizon_result = linear_study(FourWayCell::HorizonOneFrequentist)?.run(ctx)?;
    let horizon_estimate = effect_of(&horizon_result)?;
    let horizon_cell = frequentist_cell(FourWayCell::HorizonOneFrequentist, horizon_estimate)?;
    let static_bayes = bayesian_cell(FourWayCell::StaticBayesian, ctx, analytic_truth)?;
    let horizon_bayes = bayesian_cell(FourWayCell::HorizonOneBayesian, ctx, analytic_truth)?;
    let functional =
        functional_match(&static_target, horizon_cell.effect, static_cell.effect, analytic_truth)?;
    let coverage = [
        coverage_of(&static_cell, &claim.estimate, analytic_truth),
        coverage_of(&horizon_cell, horizon_estimate, analytic_truth),
    ];
    let posterior = [static_bayes.quality, horizon_bayes.quality];
    Ok(FourWaySlice {
        cells: [static_cell, static_bayes.cell, horizon_cell, horizon_bayes.cell],
        functional,
        coverage,
        posterior,
        static_fits,
        utility_only,
    })
}

struct BayesianFit {
    cell: FittedCell,
    quality: PosteriorQuality,
}

trait UtilityOnly {
    fn replay(&self) -> Result<UtilityReplay, crate::CausalError>;
}

struct CapturedUtility<F> {
    replay_fn: F,
}

impl<F> UtilityOnly for CapturedUtility<F>
where
    F: Fn() -> Result<UtilityReplay, crate::CausalError> + 'static,
{
    fn replay(&self) -> Result<UtilityReplay, crate::CausalError> {
        (self.replay_fn)()
    }
}

const fn score(slope: f64, location: f64, intercept: f64, carrier: EvidencePort) -> ScoredDecision {
    ScoredDecision { value: intercept + slope * location, carrier }
}

fn frequentist_cell(
    cell: FourWayCell,
    estimate: &EffectEstimate,
) -> Result<FittedCell, crate::CausalError> {
    if !cell.is_frequentist() {
        return Err(compile("frequentist cell expected"));
    }
    Ok(FittedCell {
        cell,
        effect: estimate.ate,
        standard_error: estimate.se_analytic(),
        interval_method: PublishedScalarUncertainty::select(estimate).method,
        posterior_mean: None,
        posterior_lower: None,
        posterior_upper: None,
    })
}

fn bayesian_cell(
    cell: FourWayCell,
    ctx: &ExecutionContext,
    truth: f64,
) -> Result<BayesianFit, crate::CausalError> {
    if !cell.is_bayesian() {
        return Err(compile("a Bayesian cell was requested for a frequentist realization"));
    }
    let result = linear_study(cell)?.run(ctx)?;
    bayesian_fit(cell, &result, truth)
}

fn bayesian_fit(
    cell: FourWayCell,
    result: &StudyResult,
    truth: f64,
) -> Result<BayesianFit, crate::CausalError> {
    let estimate = effect_of(result)?;
    let posterior = result
        .posterior
        .as_ref()
        .ok_or_else(|| compile("Bayesian cell published no parameter posterior"))?;
    let column = posterior
        .effect_column()
        .ok_or_else(|| compile("parameter posterior has no effect column"))?;
    let mean = *posterior
        .summaries
        .mean
        .get(column)
        .ok_or_else(|| compile("parameter posterior mean is missing the effect column"))?;
    let lower = *posterior.summaries.q025.get(column).ok_or_else(|| {
        compile("parameter posterior lower quantile is missing the effect column")
    })?;
    let upper = *posterior.summaries.q975.get(column).ok_or_else(|| {
        compile("parameter posterior upper quantile is missing the effect column")
    })?;
    let fitted = FittedCell {
        cell,
        effect: estimate.ate,
        standard_error: estimate.se_analytic(),
        interval_method: result.primary_interval_binding(true).method,
        posterior_mean: Some(mean),
        posterior_lower: Some(lower),
        posterior_upper: Some(upper),
    };
    let quality =
        PosteriorQuality { cell, mean, absolute_error: (mean - truth).abs(), lower, upper };
    Ok(BayesianFit { cell: fitted, quality })
}

fn coverage_of(cell: &FittedCell, estimate: &EffectEstimate, truth: f64) -> FrequentistCoverage {
    let published = PublishedScalarUncertainty::select(estimate);
    let interval_published =
        published.standard_error.is_some_and(|error| error.is_finite() && error > 0.0);
    let covers_analytic_truth = if let Some((lower, upper)) = published.interval(estimate) {
        lower <= truth && truth <= upper
    } else {
        (estimate.ate - truth).abs() <= DEGENERATE_TOLERANCE && is_zero(estimate.se_analytic())
    };
    FrequentistCoverage {
        cell: cell.cell,
        nominal_level: FREQUENTIST_LEVEL,
        interval_published,
        covers_analytic_truth,
    }
}

fn functional_match(
    static_target: &ScientificTarget,
    horizon_effect: f64,
    static_effect: f64,
    analytic_truth: f64,
) -> Result<FunctionalMatch, crate::CausalError> {
    let static_contrast = contrast_of(static_target)?;
    let horizon_target = horizon_target(outcome_variable(static_contrast)?)?;
    let horizon_contrast = contrast_of(&horizon_target)?;
    if static_contrast.form() != horizon_contrast.form() {
        return Err(compile("static and horizon-one contrast forms differ"));
    }
    if outcome_variable(static_contrast)? != outcome_variable(horizon_contrast)? {
        return Err(compile("static and horizon-one outcomes differ"));
    }
    let contrast = static_contrast.form();
    let static_horizon = frame_of(static_contrast)?.horizon_steps();
    let horizon_steps = frame_of(horizon_contrast)?.horizon_steps().unwrap_or(0);
    let horizon_identity = horizon_target.identity();
    let labeled = horizon_target.clone().with_inference(InferenceSelection::Frequentist);
    let relabeled = horizon_target.with_inference(InferenceSelection::Bayesian);
    Ok(FunctionalMatch {
        contrast,
        static_horizon,
        horizon_steps,
        static_effect,
        horizon_effect,
        analytic_truth,
        identities_differ: static_target.identity() != horizon_identity,
        inference_label_preserves_identity: labeled == relabeled
            && labeled.identity() == relabeled.identity(),
    })
}

fn contrast_of(
    target: &ScientificTarget,
) -> Result<&antecedent_core::InterventionContrast, crate::CausalError> {
    match target.form() {
        TargetForm::Contrast(contrast) => Ok(contrast),
        TargetForm::World(_)
        | TargetForm::Worlds(_)
        | TargetForm::DoseTime(_)
        | TargetForm::Mediation(_)
        | _ => Err(compile("four-way target is not a contrast")),
    }
}

fn frame_of(
    contrast: &antecedent_core::InterventionContrast,
) -> Result<&TimeFrame, crate::CausalError> {
    contrast
        .worlds()
        .worlds()
        .first()
        .map(antecedent_core::InterventionWorld::time)
        .ok_or_else(|| compile("contrast has no worlds"))
}

fn outcome_variable(
    contrast: &antecedent_core::InterventionContrast,
) -> Result<antecedent_core::VariableId, crate::CausalError> {
    let outcome = contrast
        .worlds()
        .worlds()
        .first()
        .map(|world| world.binding().outcome())
        .ok_or_else(|| compile("contrast has no outcome"))?;
    if outcome.functional() != &OutcomeFunctional::Mean {
        return Err(compile("four-way outcome functional is not a mean"));
    }
    Ok(outcome.outcome())
}

fn horizon_target(
    outcome: antecedent_core::VariableId,
) -> Result<ScientificTarget, crate::CausalError> {
    let treatment = antecedent_core::VariableId::from_raw(0);
    let binding = WorldBinding::new(
        antecedent_core::TargetPopulation::AllObserved,
        ObservationUnit::new("row").map_err(|err| compile(err.to_string()))?,
        OutcomeQuantity::mean(outcome, QuantityUnits::dimensionless())
            .map_err(|err| compile(err.to_string()))?,
        Conditioning::unconditional(),
    )
    .map_err(|err| compile(err.to_string()))?;
    let time = TimeFrame::temporal(
        1,
        HistoryWindow::planner_default(),
        InitialLaw::contemporaneous_observed(),
    )
    .map_err(|err| compile(err.to_string()))?;
    let policy = TemporalPolicy::pulse(-1);
    let control = InterventionWorld::new(
        ScientificWorldId::from_raw(CONTROL_WORLD),
        WorldAction::schedule(Intervention::set(treatment, Value::f64(0.0)), policy.clone())
            .map_err(|err| compile(err.to_string()))?,
        true,
        binding.clone(),
        time.clone(),
    )
    .map_err(|err| compile(err.to_string()))?;
    let active = InterventionWorld::new(
        ScientificWorldId::from_raw(ACTIVE_WORLD),
        WorldAction::schedule(Intervention::set(treatment, Value::f64(1.0)), policy)
            .map_err(|err| compile(err.to_string()))?,
        false,
        binding,
        time,
    )
    .map_err(|err| compile(err.to_string()))?;
    let worlds = WorldCollection::new([control, active]).map_err(|err| compile(err.to_string()))?;
    let contrast = antecedent_core::InterventionContrast::difference(
        worlds,
        ScientificWorldId::from_raw(ACTIVE_WORLD),
        ScientificWorldId::from_raw(CONTROL_WORLD),
        ScientificWorldId::from_raw(CONTROL_WORLD),
    )
    .map_err(|err| compile(err.to_string()))?;
    Ok(ScientificTarget::new(
        TargetForm::Contrast(contrast),
        ModelAssumptions::new(GraphFamily::TemporalDag, AssumptionSet::new()),
        InferenceSelection::Frequentist,
    ))
}

fn linear_study(cell: FourWayCell) -> Result<Study, crate::CausalError> {
    let builder = match cell {
        FourWayCell::StaticFrequentist | FourWayCell::StaticBayesian => {
            let (treatment, outcome) = static_columns();
            static_builder(&treatment, &outcome)?
        }
        FourWayCell::HorizonOneFrequentist | FourWayCell::HorizonOneBayesian => horizon_builder()?,
    };
    finish_study(builder, cell.is_bayesian())
}

fn finish_study(
    mut builder: crate::StudyBuilder,
    bayesian: bool,
) -> Result<Study, crate::CausalError> {
    if bayesian {
        builder = builder.inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(POSTERIOR_DRAWS).prior_scale(POSTERIOR_PRIOR_SCALE),
        ));
    }
    builder.refute(RefuteSuite::None).bootstrap_replicates(0).build()
}

fn static_builder(
    treatment: &[f64],
    outcome: &[f64],
) -> Result<crate::StudyBuilder, crate::CausalError> {
    let schema = CausalSchemaBuilder::new()
        .continuous("t")
        .treatment()
        .continuous("y")
        .outcome()
        .build()
        .map_err(|err| compile(err.to_string()))?;
    let data = TabularData::try_from_schema_f64(schema.clone(), [("t", treatment), ("y", outcome)])
        .map_err(|err| compile(err.to_string()))?;
    let graph =
        Dag::from_named_edges(&schema, &[("t", "y")]).map_err(|err| compile(err.to_string()))?;
    let query = AverageEffectQuery::binary_ate(
        schema.id_of("t").map_err(|err| compile(err.to_string()))?,
        schema.id_of("y").map_err(|err| compile(err.to_string()))?,
    );
    Ok(Study::tabular(data).graph(graph).query(query))
}

fn horizon_builder() -> Result<crate::StudyBuilder, crate::CausalError> {
    let (treatment, outcome) = series_columns();
    let series = TimeSeriesData::from_f64_columns(
        [("t", treatment.as_slice()), ("y", outcome.as_slice())],
        1,
    )
    .map_err(|err| compile(err.to_string()))?;
    let treatment_id = antecedent_core::VariableId::from_raw(0);
    let outcome_id = antecedent_core::VariableId::from_raw(1);
    let mut graph = TemporalDag::empty();
    let lagged = ensure_lagged(&mut graph, treatment_id, Lag::from_raw(1))
        .map_err(|err| compile(err.to_string()))?;
    let now = ensure_lagged(&mut graph, outcome_id, Lag::CONTEMPORANEOUS)
        .map_err(|err| compile(err.to_string()))?;
    graph.insert_directed(lagged, now).map_err(|err| compile(err.to_string()))?;
    let query = TemporalEffectQuery::pulse(treatment_id, outcome_id, 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);
    Ok(Study::series(series).graph(graph).temporal_query(query))
}

/// Mean treated outcome minus mean control outcome over the fixture rows.
fn line_effect() -> f64 {
    let (mut treated, mut control) = ((0.0, 0.0), (0.0, 0.0));
    for &(arm, outcome) in &LINE {
        let bucket = if arm > 0.5 { &mut treated } else { &mut control };
        bucket.0 += outcome;
        bucket.1 += 1.0;
    }
    treated.0 / treated.1 - control.0 / control.1
}

fn static_columns() -> ([f64; 8], [f64; 8]) {
    let mut treatment = [0.0; 8];
    let mut outcome = [0.0; 8];
    for (index, pair) in LINE.iter().enumerate() {
        treatment[index] = pair.0;
        outcome[index] = pair.1;
    }
    (treatment, outcome)
}

fn series_columns() -> ([f64; 9], [f64; 9]) {
    let mut treatment = [0.0; 9];
    let mut outcome = [0.0; 9];
    for (index, pair) in LINE.iter().enumerate() {
        treatment[index] = pair.0;
        outcome[index + 1] = pair.1;
    }
    (treatment, outcome)
}

fn effect_of(result: &StudyResult) -> Result<&EffectEstimate, crate::CausalError> {
    result.estimate.as_effect().ok_or_else(|| compile("four-way cell published no effect estimate"))
}

fn compile(message: impl Into<String>) -> crate::CausalError {
    crate::CausalError::Compile { message: message.into() }
}

const fn is_zero(value: f64) -> bool {
    value.abs().to_bits() == 0.0_f64.to_bits()
}

#[cfg(test)]
mod tests {
    use antecedent_contract::{DecisionRequest, EvidencePort};
    use antecedent_core::{ContrastForm, ExecutionContext, IntervalMethod, ScientificWorldId};
    use antecedent_stats::fit_counts::count_least_squares_solves;

    use super::{
        DecisionDemand, FourWayCell, MissingCarrier, PublishedScalarUncertainty, bayesian_fit,
        confidence_interval_as_posterior, effect_of, finish_study, frequentist_cell,
        same_target_slice, static_builder,
    };

    #[test]
    fn static_and_horizon_one_functionals_match_the_analytic_truth() {
        let ctx = ExecutionContext::for_tests(1);
        let slice = same_target_slice(&ctx).unwrap();
        let matched = slice.functional_match();
        assert_eq!(
            matched.contrast,
            ContrastForm::Difference {
                plus: ScientificWorldId::from_raw(1),
                minus: ScientificWorldId::from_raw(0),
            }
        );
        assert_eq!(matched.static_horizon, None);
        assert_eq!(matched.horizon_steps, 1);
        assert!(matched.identities_differ);
        assert!(matched.inference_label_preserves_identity);
        // The fixture is the exact line y = 1 + 2 t, so the effect is 2.
        assert!((matched.analytic_truth - 2.0).abs() <= 1e-12);
        assert!(
            (matched.static_effect - 2.0).abs() <= 1e-12,
            "static effect {}",
            matched.static_effect
        );
        assert!(
            (matched.horizon_effect - 2.0).abs() <= 1e-8,
            "horizon-one effect {}",
            matched.horizon_effect
        );
        assert!((slice.cell(FourWayCell::StaticFrequentist).effect() - 2.0).abs() <= 1e-12);
        for cell in FourWayCell::ALL {
            assert_eq!(slice.cell(cell).cell(), cell);
            assert!(slice.cell(cell).effect().is_finite(), "{}", cell.as_str());
        }
    }

    #[test]
    fn utility_only_change_performs_zero_causal_refits() {
        let ctx = ExecutionContext::for_tests(1);
        let slice = same_target_slice(&ctx).unwrap();
        let fits = slice.static_frequentist_fits();
        assert_eq!(fits, 1);
        let (replay, solves) = count_least_squares_solves(|| slice.utility_only_change().unwrap());
        assert_eq!(solves, 0);
        assert_eq!(replay.least_squares_solves, 0);
        assert_eq!(replay.causal_fit_count, fits);
        assert!(replay.fit_valid);
        assert!(!replay.decision_valid_before_execute);
        assert!(replay.decision_valid_after_execute);
        assert!((replay.effect - 2.0).abs() <= 1e-12, "replayed effect {}", replay.effect);
        assert_eq!(slice.static_frequentist_fits(), fits);
    }

    #[test]
    fn exact_line_records_stay_apart_between_frequentist_and_posterior_cells() {
        let ctx = ExecutionContext::for_tests(1);
        let slice = same_target_slice(&ctx).unwrap();
        let coverage = slice.frequentist_coverage();
        assert_eq!(
            [coverage[0].cell, coverage[1].cell],
            [FourWayCell::StaticFrequentist, FourWayCell::HorizonOneFrequentist]
        );
        assert!(coverage.iter().all(|item| item.cell.is_frequentist()));
        // The exact line has zero residual variance: the interval degenerates to
        // the fitted effect. That is a point mass at the truth, not evidence
        // about sampling coverage (the noisy fixture below tests coverage).
        let static_error = slice.cell(FourWayCell::StaticFrequentist).standard_error();
        assert!(coverage[0].covers_analytic_truth, "static se {static_error}");
        assert_eq!(coverage[0].nominal_level.to_bits(), 0.95_f64.to_bits());
        assert!(!coverage[1].interval_published);

        let quality = slice.posterior_quality();
        assert_eq!(
            [quality[0].cell, quality[1].cell],
            [FourWayCell::StaticBayesian, FourWayCell::HorizonOneBayesian]
        );
        for item in quality {
            assert!(item.cell.is_bayesian());
            assert!(
                item.absolute_error < 0.05,
                "{} error {}",
                item.cell.as_str(),
                item.absolute_error
            );
            assert!(item.lower <= item.mean && item.mean <= item.upper);
            assert_eq!(slice.cell(item.cell).interval_method(), IntervalMethod::PosteriorQuantile);
            assert!(slice.cell(item.cell).posterior_mean().is_some());
        }
        assert!(slice.cell(FourWayCell::StaticFrequentist).posterior_mean().is_none());
        assert!(slice.cell(FourWayCell::HorizonOneFrequentist).posterior_mean().is_none());
    }

    /// Twelve rows with true effect 2 and intercept 1 plus fixed noise. The
    /// reference is the closed-form two-sample least-squares fit, written out
    /// from the raw rows.
    #[test]
    fn noisy_data_matches_the_closed_form_fit_and_keeps_interpretations_apart() {
        let control = [1.2, 0.7, 1.5, 0.9, 1.1, 0.6];
        let treated = [3.3, 2.6, 3.1, 2.8, 3.4, 2.7];
        let mut arm = Vec::new();
        let mut outcome = Vec::new();
        for i in 0..6 {
            arm.extend([0.0, 1.0]);
            outcome.extend([control[i], treated[i]]);
        }
        let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
        let (m0, m1) = (mean(&control), mean(&treated));
        let rss: f64 = control.iter().map(|v| (v - m0).powi(2)).sum::<f64>()
            + treated.iter().map(|v| (v - m1).powi(2)).sum::<f64>();
        let ols_effect = m1 - m0;
        let ols_se = (rss / 10.0 * (1.0 / 6.0 + 1.0 / 6.0)).sqrt();
        // t quantile with 10 degrees of freedom.
        let t975 = 2.228_138_851_986_274;
        assert!((ols_effect - 1.983_333_333_333_333).abs() < 1e-12);

        let ctx = ExecutionContext::for_tests(1);
        let frequentist = finish_study(static_builder(&arm, &outcome).unwrap(), false)
            .unwrap()
            .run(&ctx)
            .unwrap();
        let estimate = effect_of(&frequentist).unwrap();
        assert!((estimate.ate - ols_effect).abs() < 1e-10, "{} vs {ols_effect}", estimate.ate);
        let published = PublishedScalarUncertainty::select(estimate);
        let se = published.standard_error.expect("a noisy fit publishes a standard error");
        assert!(se > 0.5 * ols_se && se < 1.5 * ols_se, "se {se} vs closed form {ols_se}");
        let (lower, upper) = published.interval(estimate).expect("a published interval");
        assert!(lower <= 2.0 && 2.0 <= upper, "interval [{lower}, {upper}] must cover 2");
        // The closed-form interval covers the true effect too.
        assert!((ols_effect - 2.0).abs() < t975 * ols_se);

        let bayesian =
            finish_study(static_builder(&arm, &outcome).unwrap(), true).unwrap().run(&ctx).unwrap();
        let fit = bayesian_fit(FourWayCell::StaticBayesian, &bayesian, 2.0).unwrap();
        // 1024 draws under a flat prior: the mean is the least-squares effect up
        // to Monte Carlo error, and the 95% band is a t band of that width.
        assert!(
            (fit.quality.mean - ols_effect).abs() < 0.25 * ols_se,
            "posterior mean {} vs {ols_effect}",
            fit.quality.mean
        );
        let width = fit.quality.upper - fit.quality.lower;
        assert!(
            width > 0.8 * 2.0 * 1.96 * ols_se && width < 1.5 * 2.0 * 1.96 * ols_se,
            "posterior band width {width} vs closed form {}",
            2.0 * t975 * ols_se
        );
        assert!(fit.quality.lower <= 2.0 && 2.0 <= fit.quality.upper);

        // Different interpretation: a sampling interval versus posterior quantiles.
        assert_eq!(fit.cell.interval_method(), IntervalMethod::PosteriorQuantile);
        assert_ne!(
            PublishedScalarUncertainty::select(estimate).method,
            IntervalMethod::PosteriorQuantile
        );
        assert!(fit.cell.posterior_mean().is_some());
        let frequentist_cell = frequentist_cell(FourWayCell::StaticFrequentist, estimate).unwrap();
        assert!(frequentist_cell.posterior_mean().is_none());
        assert_ne!(frequentist_cell.interval_method(), fit.cell.interval_method());
    }

    #[test]
    fn confidence_interval_offered_as_posterior_is_rejected_and_the_decision_proceeds() {
        let refused = confidence_interval_as_posterior().unwrap_err();
        assert_eq!(refused.source_port, EvidencePort::ConfidenceRegion);
        assert_eq!(refused.request, DecisionRequest::ParameterPosterior);
        assert_eq!(refused.legal_ports, &[EvidencePort::ParameterPosterior]);
        assert!(refused.message.contains("not a parameter posterior"));

        let ctx = ExecutionContext::for_tests(1);
        let slice = same_target_slice(&ctx).unwrap();
        let affine_posterior = DecisionDemand::AffinePosterior { slope: 2.0, intercept: -0.5 };
        let affine_mean = DecisionDemand::AffineMean { slope: 2.0, intercept: -0.5 };
        for cell in [FourWayCell::StaticBayesian, FourWayCell::HorizonOneBayesian] {
            let fitted = slice.cell(cell);
            let scored = fitted.decide(affine_posterior).unwrap();
            let mean = fitted.posterior_mean().unwrap();
            assert_eq!(scored.carrier, EvidencePort::ParameterPosterior);
            assert_eq!(scored.value.to_bits(), 2.0_f64.mul_add(mean, -0.5).to_bits());
            let weaker = fitted.decide(affine_mean).unwrap();
            assert_eq!(weaker.carrier, EvidencePort::Mean);
            assert_eq!(weaker.value.to_bits(), 2.0_f64.mul_add(fitted.effect(), -0.5).to_bits());
        }
        for cell in [FourWayCell::StaticFrequentist, FourWayCell::HorizonOneFrequentist] {
            let fitted = slice.cell(cell);
            assert_eq!(
                fitted.decide(affine_posterior).unwrap_err(),
                MissingCarrier::ParameterPosterior
            );
            let scored = fitted.decide(affine_mean).unwrap();
            assert_eq!(scored.carrier, EvidencePort::Mean);
            assert_eq!(scored.value.to_bits(), 2.0_f64.mul_add(fitted.effect(), -0.5).to_bits());
            assert_eq!(
                fitted.decide(DecisionDemand::Nonlinear).unwrap_err(),
                MissingCarrier::OutcomeLaw
            );
            assert_eq!(
                fitted.decide(DecisionDemand::Risk).unwrap_err(),
                MissingCarrier::OutcomeLaw
            );
            assert_eq!(MissingCarrier::OutcomeLaw.as_str(), "outcome_law");
        }
        let bayesian = slice.cell(FourWayCell::StaticBayesian);
        assert_eq!(
            bayesian.decide(DecisionDemand::Nonlinear).unwrap_err(),
            MissingCarrier::OutcomeLaw
        );
        assert_eq!(bayesian.decide(DecisionDemand::Risk).unwrap_err(), MissingCarrier::OutcomeLaw);
    }
}
