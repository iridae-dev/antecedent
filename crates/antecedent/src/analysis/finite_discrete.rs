//! Finite-discrete four-way slice.
//!
//! Static frequentist, static Bayesian, horizon-one frequentist, and horizon-one
//! Bayesian realizations of one adjustment functional. The expression is
//! `Σ_z P(z) P(y | t, z)` planned by [`ExactEvaluationPlan`]. Horizon one is
//! that same expression cited at another regime. Frequentist cells carry the
//! empirical plug-in. Bayesian cells carry a two-draw posterior mixture.
//! A utility or action change reads the stored laws and does not plan again.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the parent inserts this module; the study facade does not call the slice yet"
    )
)]

use std::sync::Arc;

use antecedent_contract::{DecisionRequest, EvidencePort, EvidencePortError};
use antecedent_core::execution::{AdmissionError, AdmissionLease};
use antecedent_core::{
    EncodedWidth, ExecutionContext, RegimeId, RouteBudget, RouteMemoryDeclaration, Value,
    VariableId,
};
use antecedent_expr::{
    Assignment, CausalExprArena, DiscreteAxis, DomainRef, ExactDiscreteLaw, ExactEvaluationLimits,
    ExactEvaluationPlan, ExactEvaluationSpec, ExactTransportData, ExprId, ExprNode,
    FactorRequirement, FunctionalProgram, InterventionAssignment, LawOrigin, LawTolerance,
    ProgramLimits, ProgramSchema, ProgramVariable,
};
use antecedent_io::IoError;
use antecedent_io::exact_law_wire::ExactLawWire;
use serde::{Deserialize, Serialize};

const TREATMENT: u32 = 0;
const CONFOUNDER: u32 = 1;
const OUTCOME: u32 = 2;
const POPULATION: &str = "study";
/// Joint `P(t, z, y)`, last axis fastest. Counts are the frequency table.
const FREQUENCY_COUNTS: [u64; 8] = [27, 9, 3, 1, 6, 6, 3, 9];
/// First posterior draw, denominator 16. Interventional contrast is 1/4.
const DRAW_LOW: [u64; 8] = [3, 1, 3, 1, 2, 2, 2, 2];
/// Second posterior draw, denominator 16. Interventional contrast is 1/2.
const DRAW_HIGH: [u64; 8] = [3, 1, 3, 1, 1, 3, 1, 3];
const DRAW_WEIGHTS: [f64; 2] = [0.25, 0.75];
const OPERATIONS: usize = 10_000;
const DEPTH: usize = 32;
/// Cap on materialized support rows. The joint table has eight cells.
const SUPPORT_ROWS: usize = 8;
const STATIC_REGIME: u32 = 0;
const HORIZON_REGIME: u32 = 1;
/// Largest count-table total whose counts convert to `f64` exactly (`2^53`).
const MAX_EXACT_COUNT: u64 = 1 << 53;
/// Tail mass of the equal-tailed posterior interval.
const POSTERIOR_TAIL: f64 = 0.025;
/// Largest departure of the posterior weights from one.
const WEIGHT_TOLERANCE: f64 = 1e-9;

/// One posterior draw of the joint, as a count table, and its weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PosteriorDraw {
    counts: [u64; 8],
    weight: f64,
}

/// A frequency table over `(treatment, confounder, outcome)` and the posterior draws of
/// the same joint. The frequentist cells plan the table; the Bayesian cells plan each
/// draw and mix the results by weight.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DiscreteInputs {
    counts: [u64; 8],
    draws: Vec<PosteriorDraw>,
}

impl DiscreteInputs {
    /// Inputs from a frequency table and weighted posterior draws.
    ///
    /// # Errors
    ///
    /// A table has no mass or a total beyond exact `f64` range, there is no draw, or a weight is
    /// negative, not finite, or the weights do not sum to one.
    pub(crate) fn new(
        counts: [u64; 8],
        draws: impl IntoIterator<Item = ([u64; 8], f64)>,
    ) -> Result<Self, crate::CausalError> {
        let draws: Vec<PosteriorDraw> =
            draws.into_iter().map(|(counts, weight)| PosteriorDraw { counts, weight }).collect();
        if draws.is_empty() {
            return Err(compile("a finite-discrete posterior needs at least one draw"));
        }
        for table in std::iter::once(&counts).chain(draws.iter().map(|draw| &draw.counts)) {
            let total = table_total(table)?;
            if total == 0 {
                return Err(compile("a finite-discrete table has no mass"));
            }
            if total > MAX_EXACT_COUNT {
                return Err(compile("a finite-discrete table total is not exactly representable"));
            }
        }
        if draws.iter().any(|draw| !(draw.weight.is_finite() && draw.weight >= 0.0)) {
            return Err(compile("a posterior draw weight is negative or not finite"));
        }
        let mass: f64 = draws.iter().map(|draw| draw.weight).sum();
        if (mass - 1.0).abs() > WEIGHT_TOLERANCE {
            return Err(compile("the posterior draw weights do not sum to one"));
        }
        Ok(Self { counts, draws })
    }

    /// The reference table: a 64-row frequency table and two posterior draws.
    pub(crate) fn reference() -> Self {
        Self {
            counts: FREQUENCY_COUNTS,
            draws: vec![
                PosteriorDraw { counts: DRAW_LOW, weight: DRAW_WEIGHTS[0] },
                PosteriorDraw { counts: DRAW_HIGH, weight: DRAW_WEIGHTS[1] },
            ],
        }
    }
}

fn table_total(counts: &[u64; 8]) -> Result<u64, crate::CausalError> {
    counts
        .iter()
        .try_fold(0_u64, |total, count| total.checked_add(*count))
        .ok_or_else(|| compile("a finite-discrete table total overflows"))
}

/// One cell of the static/horizon-one × frequentist/Bayesian slice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DiscreteCell {
    /// Static empirical plug-in of the adjustment.
    StaticFrequentist,
    /// Static posterior mixture of the adjustment.
    StaticBayesian,
    /// Horizon-one empirical plug-in of the same adjustment.
    HorizonOneFrequentist,
    /// Horizon-one posterior mixture of the same adjustment.
    HorizonOneBayesian,
}

impl DiscreteCell {
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

    /// Whether this cell cites the horizon-one regime.
    #[must_use]
    pub(crate) const fn is_horizon(self) -> bool {
        matches!(self, Self::HorizonOneFrequentist | Self::HorizonOneBayesian)
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

/// Carrier a decision still needs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MissingCarrier {
    /// Parameter posterior. A confidence interval is not this carrier.
    ParameterPosterior,
    /// Predictive or interventional law. A mean cannot carry it.
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
    /// Affine utility of the contrast, scored from the published mean.
    AffineMean { slope: f64, intercept: f64 },
    /// Nonlinear utility. A mean cannot carry it.
    Nonlinear,
    /// Threshold or other outcome risk. A mean cannot carry it.
    Risk,
}

/// Which stored world an action reads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DiscreteAction {
    /// Control arm, treatment level 0.
    Control,
    /// Active arm, treatment level 1.
    Active,
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

/// Location of one planned cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FittedCell {
    cell: DiscreteCell,
    effect: f64,
    active_mean: f64,
    control_mean: f64,
    origin: LawOrigin,
    port: EvidencePort,
    regime: RegimeId,
    posterior_mean: Option<f64>,
    posterior_lower: Option<f64>,
    posterior_upper: Option<f64>,
    sample_size: Option<u64>,
    outcome_levels: [i64; 2],
    active_probabilities: [f64; 2],
}

impl FittedCell {
    /// Cell this fit belongs to.
    #[must_use]
    pub(crate) const fn cell(&self) -> DiscreteCell {
        self.cell
    }

    /// Published scalar contrast: active mean minus control mean.
    #[must_use]
    pub(crate) const fn effect(&self) -> f64 {
        self.effect
    }

    /// Interventional mean of the outcome at treatment 1.
    #[must_use]
    pub(crate) const fn active_mean(&self) -> f64 {
        self.active_mean
    }

    /// Interventional mean of the outcome at treatment 0.
    #[must_use]
    pub(crate) const fn control_mean(&self) -> f64 {
        self.control_mean
    }

    /// Finite-discrete law origin. Gaussian density is not an origin.
    #[must_use]
    pub(crate) const fn origin(&self) -> LawOrigin {
        self.origin
    }

    /// Uncertainty object. A sampling law and a parameter posterior differ.
    #[must_use]
    pub(crate) const fn port(&self) -> EvidencePort {
        self.port
    }

    /// Regime the expression cites.
    #[must_use]
    pub(crate) const fn regime(&self) -> RegimeId {
        self.regime
    }

    /// Posterior mean of the contrast, when this cell published one.
    #[must_use]
    pub(crate) const fn posterior_mean(&self) -> Option<f64> {
        self.posterior_mean
    }

    /// Equal-tailed lower posterior quantile.
    #[must_use]
    pub(crate) const fn posterior_lower(&self) -> Option<f64> {
        self.posterior_lower
    }

    /// Equal-tailed upper posterior quantile.
    #[must_use]
    pub(crate) const fn posterior_upper(&self) -> Option<f64> {
        self.posterior_upper
    }

    /// Empirical sample size, when the cell is a frequency plug-in.
    #[must_use]
    pub(crate) const fn sample_size(&self) -> Option<u64> {
        self.sample_size
    }

    /// Outcome levels in atom order.
    #[allow(
        dead_code,
        reason = "the masses are stored with the fit; the facade does not read them yet"
    )]
    #[must_use]
    pub(crate) const fn outcome_levels(&self) -> [i64; 2] {
        self.outcome_levels
    }

    /// Active-arm outcome masses in [`Self::outcome_levels`] order.
    #[allow(
        dead_code,
        reason = "the masses are stored with the fit; the facade does not read them yet"
    )]
    #[must_use]
    pub(crate) const fn active_probabilities(&self) -> [f64; 2] {
        self.active_probabilities
    }

    /// Score `demand` from this cell.
    ///
    /// An affine posterior request uses the posterior mean. An affine mean
    /// request uses the published contrast. A nonlinear utility or an outcome
    /// risk is refused and names [`MissingCarrier::OutcomeLaw`]. A posterior
    /// request on a cell that has no posterior names
    /// [`MissingCarrier::ParameterPosterior`].
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

/// Static adjustment and its horizon-one citation.
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use = "compare both horizons with the hand discrete expectation"]
pub(crate) struct FunctionalMatch {
    /// Horizon of the static target. A static frame has none.
    pub(crate) static_horizon: Option<u32>,
    /// Horizon of the second target, in steps.
    pub(crate) horizon_steps: u32,
    /// Frequentist static contrast.
    pub(crate) static_effect: f64,
    /// Frequentist horizon-one contrast.
    pub(crate) horizon_effect: f64,
    /// Node kinds of the two horizons agree.
    pub(crate) same_shape: bool,
    /// Regime citations differ, so the planned identities differ.
    pub(crate) identities_differ: bool,
}

/// Budgets the exact plan actually used, and the atoms those budgets admitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use = "the bound is what stops enumeration"]
pub(crate) struct EnumerationBound {
    /// Shared operation budget for preflight and evaluation.
    pub(crate) operations: usize,
    /// Maximum expression depth.
    pub(crate) depth: usize,
    /// Maximum support rows materialized for one coordinate set.
    pub(crate) support_rows: usize,
    /// Outcome atoms the admitted plan returned.
    pub(crate) outcome_atoms: usize,
}

/// Result of [`FiniteDiscreteSlice::utility_change`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use = "a utility change reports the carrier and that planning was not repeated"]
pub(crate) struct UtilityReplay {
    /// Affine score.
    pub(crate) value: f64,
    /// Posterior or mean.
    pub(crate) carrier: EvidencePort,
    /// Exact plans compiled while building the slice.
    pub(crate) plan_compilations: u64,
}

/// Result of [`FiniteDiscreteSlice::action_change`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use = "an action change reports the stored world mean"]
pub(crate) struct ActionReplay {
    /// Stored interventional mean of the chosen arm.
    pub(crate) mean: f64,
    /// Exact plans compiled while building the slice.
    pub(crate) plan_compilations: u64,
}

/// Exported frequentist cell: the joint law and the planned outcome atoms.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[must_use = "a loaded export is the complete frequentist result"]
pub(crate) struct ExportedDiscreteResult {
    /// Law origin name.
    pub(crate) origin: String,
    /// Planned contrast.
    pub(crate) effect: f64,
    /// `None` is the static horizon.
    pub(crate) horizon_steps: Option<u32>,
    /// Frequency-table size.
    pub(crate) sample_size: Option<u64>,
    /// Absent on a frequentist export.
    pub(crate) posterior_mean: Option<f64>,
    /// Outcome levels in atom order.
    pub(crate) outcome_levels: Vec<i64>,
    /// Active-arm masses, one per outcome level.
    pub(crate) outcome_probabilities: Vec<f64>,
    /// Complete joint, including every cell of the eight-point table.
    pub(crate) law: ExactLawWire,
}

/// The four realizations, the shared functional, and the budgets.
#[must_use = "read the four cells before discarding the slice"]
pub(crate) struct FiniteDiscreteSlice {
    cells: [FittedCell; 4],
    functional: FunctionalMatch,
    static_program: FunctionalProgram,
    horizon_program: FunctionalProgram,
    static_kinds: Vec<String>,
    horizon_kinds: Vec<String>,
    frequency: ExactDiscreteLaw,
    plan_compilations: u64,
    enumeration: EnumerationBound,
}

impl FiniteDiscreteSlice {
    /// Fitted cell.
    #[must_use]
    pub(crate) fn cell(&self, cell: DiscreteCell) -> &FittedCell {
        let fitted = &self.cells[cell.index()];
        debug_assert_eq!(fitted.cell(), cell, "a slice stores each fit at its cell's index");
        fitted
    }

    /// Static and horizon-one citations of the adjustment.
    #[must_use = "the match is the comparison of the two horizons"]
    pub(crate) const fn functional(&self) -> FunctionalMatch {
        self.functional
    }

    /// Factors the checked program will request at this horizon.
    #[must_use = "the factors are what the exact plan binds"]
    pub(crate) fn factors(&self, cell: DiscreteCell) -> &[FactorRequirement] {
        if cell.is_horizon() {
            self.horizon_program.factor_requirements()
        } else {
            self.static_program.factor_requirements()
        }
    }

    /// Expression node kinds for this cell's horizon.
    #[must_use]
    pub(crate) fn expression_kinds(&self, cell: DiscreteCell) -> &[String] {
        if cell.is_horizon() { &self.horizon_kinds } else { &self.static_kinds }
    }

    /// Exact plans compiled while building the slice.
    #[must_use]
    pub(crate) const fn plan_compilations(&self) -> u64 {
        self.plan_compilations
    }

    /// Operation, depth, and support budgets, with the admitted atom count.
    #[must_use = "enumeration is only valid inside this bound"]
    pub(crate) const fn enumeration(&self) -> EnumerationBound {
        self.enumeration
    }

    /// Score `demand` from a cell. Does not plan again.
    pub(crate) fn utility_change(
        &self,
        cell: DiscreteCell,
        demand: DecisionDemand,
    ) -> Result<UtilityReplay, MissingCarrier> {
        let scored = self.cell(cell).decide(demand)?;
        Ok(UtilityReplay {
            value: scored.value,
            carrier: scored.carrier,
            plan_compilations: self.plan_compilations,
        })
    }

    /// Read a stored arm. Does not plan again.
    #[must_use = "the chosen arm is read from the stored plan"]
    pub(crate) fn action_change(&self, cell: DiscreteCell, action: DiscreteAction) -> ActionReplay {
        let fitted = self.cell(cell);
        let mean = match action {
            DiscreteAction::Control => fitted.control_mean,
            DiscreteAction::Active => fitted.active_mean,
        };
        ActionReplay { mean, plan_compilations: self.plan_compilations }
    }

    /// Serialize the static frequentist joint and its planned outcome atoms.
    ///
    /// # Errors
    ///
    /// JSON encoding failed.
    pub(crate) fn export_static_frequentist(&self) -> Result<Vec<u8>, crate::CausalError> {
        let fitted = self.cell(DiscreteCell::StaticFrequentist);
        let exported = ExportedDiscreteResult {
            origin: fitted.origin.as_str().to_owned(),
            effect: fitted.effect,
            horizon_steps: None,
            sample_size: fitted.sample_size,
            posterior_mean: None,
            outcome_levels: fitted.outcome_levels.to_vec(),
            outcome_probabilities: fitted.active_probabilities.to_vec(),
            law: ExactLawWire::from_law(&self.frequency),
        };
        serde_json::to_vec(&exported)
            .map_err(|err| crate::CausalError::Serialization(IoError::Convert(err.to_string())))
    }
}

/// Accept a finite-discrete law origin.
///
/// `gaussian` and `linear_gaussian` are refused. Empty and unknown names are
/// refused. The four [`LawOrigin`] names are the non-Gaussian evidence types
/// this slice recognizes.
///
/// # Errors
///
/// `name` is not a finite-discrete origin.
pub(crate) fn finite_discrete_origin(name: &str) -> Result<LawOrigin, &'static str> {
    if name == "gaussian" || name == "linear_gaussian" {
        return Err("gaussian evidence is not a finite-discrete law");
    }
    if name.is_empty() {
        return Err("unknown evidence is not a finite-discrete law");
    }
    LawOrigin::parse(name).ok_or("unknown evidence is not a finite-discrete law")
}

/// Offer a confidence interval as a parameter posterior.
///
/// The offer is refused. The interval is unchanged, and the request stays a
/// parameter posterior.
pub(crate) fn confidence_interval_as_posterior() -> Result<EvidencePort, EvidencePortError> {
    EvidencePort::ConfidenceRegion.try_coerce(DecisionRequest::ParameterPosterior)
}

/// Parse an export produced by [`FiniteDiscreteSlice::export_static_frequentist`].
///
/// # Errors
///
/// `bytes` is not that export.
pub(crate) fn load_exported(bytes: &[u8]) -> Result<ExportedDiscreteResult, crate::CausalError> {
    serde_json::from_slice(bytes)
        .map_err(|err| crate::CausalError::Serialization(IoError::Convert(err.to_string())))
}

/// Plan the loaded joint and return the active-minus-control contrast.
///
/// # Errors
///
/// The export, the law, or the exact plan is refused, or the route budget
/// or cancellation stops evaluation.
pub(crate) fn effect_of_exported_law(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<f64, crate::CausalError> {
    let law = load_exported(bytes)?.law.to_law()?;
    let regime = law.regime();
    let (arena, root) = adjustment(regime);
    let data = transport([law], SUPPORT_ROWS)?;
    let mut ignored = 0_u64;
    let means = contrast(&arena, root, &data, limits(), ctx, &mut ignored)?;
    Ok(means.active - means.control)
}

/// Atom count of the active arm, or the budget that refused enumeration.
///
/// # Errors
///
/// The exact plan refuses the budgets or the frequency law, or the route
/// budget or cancellation stops evaluation.
pub(crate) fn enumerated_atoms(
    limits: ExactEvaluationLimits,
    support_rows: usize,
    ctx: &ExecutionContext,
) -> Result<usize, crate::CausalError> {
    let (arena, root) = adjustment(RegimeId::from_raw(STATIC_REGIME));
    let data = transport(
        [frequency_law(RegimeId::from_raw(STATIC_REGIME), &FREQUENCY_COUNTS, ctx)?],
        support_rows,
    )?;
    let mut ignored = 0_u64;
    let law = plan_world(&arena, root, &data, 1, limits, ctx, &mut ignored)?;
    Ok(law.probabilities.len())
}

/// Plan the four cells on the finite-discrete adjustment.
///
/// # Errors
///
/// A law or the exact plan refuses, or the route budget or cancellation
/// stops evaluation.
pub(crate) fn finite_discrete_slice(
    ctx: &ExecutionContext,
) -> Result<FiniteDiscreteSlice, crate::CausalError> {
    finite_discrete_slice_over(&DiscreteInputs::reference(), ctx)
}

/// Plan the four cells on the finite-discrete adjustment over `inputs`.
///
/// # Errors
///
/// A law or the exact plan refuses, the table cannot be admitted to the pool, or
/// cancellation stops evaluation between plans.
pub(crate) fn finite_discrete_slice_over(
    inputs: &DiscreteInputs,
    ctx: &ExecutionContext,
) -> Result<FiniteDiscreteSlice, crate::CausalError> {
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    let mut plan_compilations = 0_u64;
    let static_fit =
        fit_horizon(&HorizonRequest::static_frame(), inputs, ctx, &mut plan_compilations)?;
    let horizon_fit =
        fit_horizon(&HorizonRequest::horizon_frame(), inputs, ctx, &mut plan_compilations)?;
    let static_kinds = expression_kinds(&static_fit.program)?;
    let horizon_kinds = expression_kinds(&horizon_fit.program)?;
    let frequentist = static_fit.frequentist;
    let horizon = horizon_fit.frequentist;
    Ok(FiniteDiscreteSlice {
        cells: [
            static_fit.frequentist,
            static_fit.bayesian,
            horizon_fit.frequentist,
            horizon_fit.bayesian,
        ],
        functional: FunctionalMatch {
            static_horizon: None,
            horizon_steps: 1,
            static_effect: frequentist.effect,
            horizon_effect: horizon.effect,
            same_shape: static_kinds == horizon_kinds,
            identities_differ: frequentist.regime != horizon.regime,
        },
        static_program: static_fit.program,
        horizon_program: horizon_fit.program,
        static_kinds,
        horizon_kinds,
        frequency: static_fit.frequency,
        plan_compilations,
        enumeration: EnumerationBound {
            operations: OPERATIONS,
            depth: DEPTH,
            support_rows: SUPPORT_ROWS,
            outcome_atoms: frequentist.outcome_levels.len(),
        },
    })
}

/// The adjustment contrast of a supplied finite-discrete table, as the four cells report it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FiniteDiscreteContrast {
    /// Rows of the frequency table.
    pub sample_size: u64,
    /// Empirical plug-in contrast: active mean minus control mean.
    pub plug_in_effect: f64,
    /// Posterior mean of the contrast over the supplied draws.
    pub posterior_mean: f64,
    /// Equal-tailed lower posterior quantile.
    pub posterior_lower: f64,
    /// Equal-tailed upper posterior quantile.
    pub posterior_upper: f64,
    /// Exact plans compiled for the four cells.
    pub plan_compilations: u64,
}

/// Plan the finite-discrete adjustment `sum_z P(z) P(y | t, z)` over a caller's frequency
/// table `counts` (index `4t + 2z + y`) and weighted posterior draws of the same joint.
///
/// The laws are built directly from the counts, each exact
/// plan observes `ctx`'s cancellation, and the plug-in and posterior cells come from one
/// exact plan of the expression.
///
/// # Errors
///
/// The table has no mass, its total is beyond exact `f64` range, or the weights do not sum to one; the exact plan
/// refuses; the pool cannot hold the support enumeration; or `ctx` is cancelled.
pub fn finite_discrete_contrast(
    counts: [u64; 8],
    draws: impl IntoIterator<Item = ([u64; 8], f64)>,
    ctx: &ExecutionContext,
) -> Result<FiniteDiscreteContrast, crate::CausalError> {
    let inputs = DiscreteInputs::new(counts, draws)?;
    let slice = finite_discrete_slice_over(&inputs, ctx)?;
    let frequentist = slice.cell(DiscreteCell::StaticFrequentist);
    let posterior = slice.cell(DiscreteCell::StaticBayesian);
    let bounds =
        (posterior.posterior_lower(), posterior.posterior_upper(), posterior.posterior_mean());
    let (Some(lower), Some(upper), Some(mean)) = bounds else {
        return Err(compile("the Bayesian cell published no posterior"));
    };
    Ok(FiniteDiscreteContrast {
        sample_size: frequentist.sample_size().unwrap_or(0),
        plug_in_effect: frequentist.effect(),
        posterior_mean: mean,
        posterior_lower: lower,
        posterior_upper: upper,
        plan_compilations: slice.plan_compilations(),
    })
}

struct HorizonRequest {
    frequentist: DiscreteCell,
    bayesian: DiscreteCell,
    regime: RegimeId,
}

impl HorizonRequest {
    const fn static_frame() -> Self {
        Self {
            frequentist: DiscreteCell::StaticFrequentist,
            bayesian: DiscreteCell::StaticBayesian,
            regime: RegimeId::from_raw(STATIC_REGIME),
        }
    }

    const fn horizon_frame() -> Self {
        Self {
            frequentist: DiscreteCell::HorizonOneFrequentist,
            bayesian: DiscreteCell::HorizonOneBayesian,
            regime: RegimeId::from_raw(HORIZON_REGIME),
        }
    }
}

struct HorizonFit {
    frequentist: FittedCell,
    bayesian: FittedCell,
    program: FunctionalProgram,
    frequency: ExactDiscreteLaw,
}

struct WorldMeans {
    active: f64,
    control: f64,
    active_probabilities: [f64; 2],
    outcome_levels: [i64; 2],
}

struct AtomLaw {
    mean: f64,
    probabilities: [f64; 2],
    levels: [i64; 2],
}

fn fit_horizon(
    request: &HorizonRequest,
    inputs: &DiscreteInputs,
    ctx: &ExecutionContext,
    compilations: &mut u64,
) -> Result<HorizonFit, crate::CausalError> {
    let (arena, root) = adjustment(request.regime);
    let frequency = frequency_law(request.regime, &inputs.counts, ctx)?;
    let frequentist_means = contrast(
        &arena,
        root,
        &transport([frequency.clone()], SUPPORT_ROWS)?,
        limits(),
        ctx,
        compilations,
    )?;
    let mut draws = Vec::with_capacity(inputs.draws.len());
    for (index, draw) in inputs.draws.iter().enumerate() {
        let law = posterior_law(request.regime, &draw.counts, &format!("draw-{index}"), ctx)?;
        draws.push(contrast(
            &arena,
            root,
            &transport([law], SUPPORT_ROWS)?,
            limits(),
            ctx,
            compilations,
        )?);
    }
    let weights: Vec<f64> = inputs.draws.iter().map(|draw| draw.weight).collect();
    let program = program(arena, root)?;
    let sample_size = table_total(&inputs.counts)?;
    Ok(HorizonFit {
        frequentist: frequentist_cell(
            request.frequentist,
            request.regime,
            &frequentist_means,
            sample_size,
        ),
        bayesian: bayesian_cell(request.bayesian, request.regime, &draws, &weights),
        program,
        frequency,
    })
}

fn frequentist_cell(
    cell: DiscreteCell,
    regime: RegimeId,
    means: &WorldMeans,
    sample_size: u64,
) -> FittedCell {
    FittedCell {
        cell,
        effect: means.active - means.control,
        active_mean: means.active,
        control_mean: means.control,
        origin: LawOrigin::EmpiricalPlugin,
        port: EvidencePort::SamplingLaw,
        regime,
        posterior_mean: None,
        posterior_lower: None,
        posterior_upper: None,
        sample_size: Some(sample_size),
        outcome_levels: means.outcome_levels,
        active_probabilities: means.active_probabilities,
    }
}

fn bayesian_cell(
    cell: DiscreteCell,
    regime: RegimeId,
    draws: &[WorldMeans],
    weights: &[f64],
) -> FittedCell {
    let mix = |pick: &dyn Fn(&WorldMeans) -> f64| -> f64 {
        weights.iter().zip(draws).map(|(weight, means)| weight * pick(means)).sum()
    };
    let active = mix(&|means| means.active);
    let control = mix(&|means| means.control);
    let contrasts: Vec<f64> = draws.iter().map(|means| means.active - means.control).collect();
    let posterior_mean: f64 =
        weights.iter().zip(&contrasts).map(|(weight, value)| weight * value).sum();
    FittedCell {
        cell,
        effect: active - control,
        active_mean: active,
        control_mean: control,
        origin: LawOrigin::BayesianPosterior,
        port: EvidencePort::ParameterPosterior,
        regime,
        posterior_mean: Some(posterior_mean),
        posterior_lower: Some(weighted_quantile(&contrasts, weights, POSTERIOR_TAIL)),
        posterior_upper: Some(weighted_quantile(&contrasts, weights, 1.0 - POSTERIOR_TAIL)),
        sample_size: None,
        outcome_levels: draws[0].outcome_levels,
        active_probabilities: [
            mix(&|means| means.active_probabilities[0]),
            mix(&|means| means.active_probabilities[1]),
        ],
    }
}

/// Smallest value whose cumulative weight reaches `p`: the left inverse of the weighted
/// distribution function, so a two-point law with `p` in its tails returns the extremes.
fn weighted_quantile(values: &[f64], weights: &[f64], p: f64) -> f64 {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
    let mut cumulative = 0.0;
    for index in &order {
        cumulative += weights[*index];
        if cumulative >= p - WEIGHT_TOLERANCE {
            return values[*index];
        }
    }
    values[order[order.len() - 1]]
}

fn contrast(
    arena: &CausalExprArena,
    root: ExprId,
    data: &ExactTransportData,
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
    compilations: &mut u64,
) -> Result<WorldMeans, crate::CausalError> {
    let active = plan_world(arena, root, data, 1, limits, ctx, compilations)?;
    let control = plan_world(arena, root, data, 0, limits, ctx, compilations)?;
    if active.levels != control.levels {
        return Err(compile("active and control outcome domains differ"));
    }
    Ok(WorldMeans {
        active: active.mean,
        control: control.mean,
        active_probabilities: active.probabilities,
        outcome_levels: active.levels,
    })
}

fn plan_world(
    arena: &CausalExprArena,
    root: ExprId,
    data: &ExactTransportData,
    level: i64,
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
    compilations: &mut u64,
) -> Result<AtomLaw, crate::CausalError> {
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    let _lease = admit_discrete(ctx)?;
    let plan = ExactEvaluationPlan::compile(
        ExactEvaluationSpec {
            arena,
            root,
            outcomes: Arc::from([variable(OUTCOME)]),
            request: Assignment::from_pairs([(variable(TREATMENT), Value::Int64(level))]),
            limits,
            tolerance: LawTolerance::default(),
        },
        data.clone(),
        ctx,
    )
    .map_err(crate::CausalError::from)?;
    *compilations += 1;
    let distribution = plan.evaluate(ctx).map_err(crate::CausalError::from)?;
    if distribution.probabilities.len() != 2 || distribution.atoms.len() != 2 {
        return Err(compile("outcome enumeration was not the binary domain"));
    }
    let mean = distribution.mean(variable(OUTCOME)).map_err(crate::CausalError::from)?;
    let levels = [outcome_level(&distribution.atoms[0])?, outcome_level(&distribution.atoms[1])?];
    let probabilities = [distribution.probabilities[0], distribution.probabilities[1]];
    Ok(AtomLaw { mean, probabilities, levels })
}

fn outcome_level(atom: &Arc<[Value]>) -> Result<i64, crate::CausalError> {
    match atom.as_ref() {
        [Value::Int64(level)] => Ok(*level),
        _ => Err(compile("outcome atom is not one discrete level")),
    }
}

fn adjustment(regime: RegimeId) -> (CausalExprArena, ExprId) {
    let mut arena = CausalExprArena::new();
    let population = arena.intern_population(POPULATION);
    let empty = arena.empty_var_set();
    let none = arena.intern_intervention_set([]);
    let confounder = arena.intern_var_set([variable(CONFOUNDER)]);
    let outcome = arena.intern_var_set([variable(OUTCOME)]);
    let given = arena.intern_var_set([variable(TREATMENT), variable(CONFOUNDER)]);
    let marginal = distribution(&mut arena, confounder, empty, population, regime, none);
    let conditional = distribution(&mut arena, outcome, given, population, regime, none);
    let factors = arena.intern_list([marginal, conditional]);
    let product = arena.intern(ExprNode::Product(factors));
    let root = arena.intern(ExprNode::SumOut { variables: confounder, expr: product });
    (arena, root)
}

fn distribution(
    arena: &mut CausalExprArena,
    variables: antecedent_expr::VarSetId,
    conditioned_on: antecedent_expr::VarSetId,
    population: antecedent_expr::PopulationKeyId,
    regime: RegimeId,
    intervention: antecedent_expr::InterventionSetId,
) -> ExprId {
    arena.intern(ExprNode::Distribution {
        variables,
        conditioned_on,
        intervention,
        domain: DomainRef::Observational,
        population,
        regime: Some(regime),
    })
}

fn program(arena: CausalExprArena, root: ExprId) -> Result<FunctionalProgram, crate::CausalError> {
    let schema = ProgramSchema::new([
        (variable(TREATMENT), ProgramVariable { name: Arc::from("treatment") }),
        (variable(CONFOUNDER), ProgramVariable { name: Arc::from("confounder") }),
        (variable(OUTCOME), ProgramVariable { name: Arc::from("outcome") }),
    ]);
    FunctionalProgram::new(arena, schema, root, root, ProgramLimits::default())
        .map_err(crate::CausalError::from)
}

fn expression_kinds(program: &FunctionalProgram) -> Result<Vec<String>, crate::CausalError> {
    let arena = program.arena();
    let mut kinds = Vec::with_capacity(arena.len());
    for index in 0..arena.len() {
        let raw = u32::try_from(index).map_err(|_| compile("expression index overflowed"))?;
        let kind = match arena.node(ExprId::from_raw(raw)) {
            ExprNode::Distribution { .. } => "distribution",
            ExprNode::Product(_) => "product",
            ExprNode::SumOut { .. } => "sum_out",
            ExprNode::Kernel { .. } => "kernel",
            ExprNode::IntegralOut { .. } => "integral_out",
            ExprNode::Ratio { .. } => "ratio",
            ExprNode::Expectation { .. } => "expectation",
            ExprNode::Contrast { .. } => "contrast",
        };
        kinds.push(kind.to_owned());
    }
    Ok(kinds)
}

fn frequency_law(
    regime: RegimeId,
    counts: &[u64; 8],
    ctx: &ExecutionContext,
) -> Result<ExactDiscreteLaw, crate::CausalError> {
    let probabilities = probability_table(counts, ctx)?;
    Ok(ExactDiscreteLaw::try_empirical(
        POPULATION,
        regime,
        no_intervention(),
        axes(),
        probabilities,
        "frequency",
        LawTolerance::default(),
    )?
    .with_empirical_counts(counts.to_vec())?)
}

fn posterior_law(
    regime: RegimeId,
    counts: &[u64; 8],
    snapshot: &str,
    ctx: &ExecutionContext,
) -> Result<ExactDiscreteLaw, crate::CausalError> {
    let probabilities = probability_table(counts, ctx)?;
    Ok(ExactDiscreteLaw::try_bayesian_posterior(
        POPULATION,
        regime,
        no_intervention(),
        axes(),
        probabilities,
        snapshot,
        LawTolerance::default(),
    )?)
}

fn admit_discrete(ctx: &ExecutionContext) -> Result<AdmissionLease, crate::CausalError> {
    admit_rows(ctx, SUPPORT_ROWS as u64)
}

/// Lease the memory of a table of `rows` rows from the shared pool, or refuse it with the
/// route budget's bytes before anything is built.
fn admit_rows(ctx: &ExecutionContext, rows: u64) -> Result<AdmissionLease, crate::CausalError> {
    let admitted = RouteMemoryDeclaration::new()
        .rows(rows)
        .encoded_column(EncodedWidth::int64())
        .encoded_column(EncodedWidth::int64())
        .encoded_column(EncodedWidth::int64())
        .masks(1)
        .design_expansion(0)
        .learner_fold_copies(0, 0)
        .solver_scratch(64)
        .retained_scores_draws(0)
        .output(8)
        .workers(1)
        .admit(RouteBudget::new(execution_byte_budget(ctx)))
        .map_err(|error| crate::CausalError::Resource { message: error.to_string() })?;
    ctx.acquire_admission(1, admitted.total()).map_err(admission_error)
}

fn execution_byte_budget(ctx: &ExecutionContext) -> u64 {
    let pool = ctx.admission().byte_budget();
    match ctx.memory_limit_bytes() {
        Some(limit) => limit.min(pool),
        None => pool,
    }
}

fn admission_error(error: AdmissionError) -> crate::CausalError {
    match error {
        AdmissionError::Cancelled => cancelled(),
        other => crate::CausalError::Resource { message: other.to_string() },
    }
}

fn cancelled() -> crate::CausalError {
    crate::CausalError::Cancelled { stage: super::stage::STAGE_ESTIMATE_POINT }
}

/// Cell probabilities `count / total` of a count table, read directly from the counts.
///
/// The law is the normalised joint itself, so the table is never expanded into rows. The
/// total is bounded by [`MAX_EXACT_COUNT`], which keeps every count and the total exactly
/// representable as an `f64`.
fn probability_table(
    counts: &[u64; 8],
    ctx: &ExecutionContext,
) -> Result<Vec<f64>, crate::CausalError> {
    if ctx.cancellation.is_cancelled() {
        return Err(cancelled());
    }
    let total = table_total(counts)?;
    if total == 0 {
        return Err(compile("a finite-discrete table has no mass"));
    }
    if total > MAX_EXACT_COUNT {
        return Err(compile("a finite-discrete table total is not exactly representable"));
    }
    let denominator = total as f64;
    Ok(counts.iter().map(|count| *count as f64 / denominator).collect())
}

fn axes() -> [DiscreteAxis; 3] {
    [TREATMENT, CONFOUNDER, OUTCOME].map(|raw| DiscreteAxis {
        variable: variable(raw),
        values: Arc::from([Value::Int64(0), Value::Int64(1)]),
    })
}

fn transport(
    laws: impl Into<Arc<[ExactDiscreteLaw]>>,
    support_rows: usize,
) -> Result<ExactTransportData, crate::CausalError> {
    ExactTransportData::try_new(laws, support_rows).map_err(crate::CausalError::from)
}

fn no_intervention() -> Arc<[InterventionAssignment]> {
    Arc::from([])
}

const fn variable(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn limits() -> ExactEvaluationLimits {
    ExactEvaluationLimits { operations: OPERATIONS, depth: DEPTH }
}

fn score(slope: f64, location: f64, intercept: f64, carrier: EvidencePort) -> ScoredDecision {
    ScoredDecision { value: slope.mul_add(location, intercept), carrier }
}

fn compile(message: impl Into<String>) -> crate::CausalError {
    crate::CausalError::Compile { message: message.into() }
}

#[cfg(test)]
mod tests {
    use antecedent_contract::{DecisionRequest, EvidencePort};
    use antecedent_core::{ExecutionContext, RegimeId};
    use antecedent_expr::{DomainRef, ExactEvaluationLimits, LawOrigin};

    use super::{
        DecisionDemand, DiscreteAction, DiscreteCell, DiscreteInputs, MAX_EXACT_COUNT,
        MissingCarrier, confidence_interval_as_posterior, effect_of_exported_law, enumerated_atoms,
        finite_discrete_origin, finite_discrete_slice, finite_discrete_slice_over, load_exported,
    };

    fn hand_frequency_contrast() -> (f64, f64, f64) {
        // P(Z=0) = 3/4, P(Z=1) = 1/4.
        // P(Y=1 | T=1, Z=0) = 1/2, P(Y=1 | T=1, Z=1) = 3/4.
        // P(Y=1 | T=0, Z=0) = 1/4, P(Y=1 | T=0, Z=1) = 1/4.
        let active = 0.5 * 0.75 + 0.75 * 0.25;
        let control = 0.25 * 0.75 + 0.25 * 0.25;
        (active, control, active - control)
    }

    fn hand_posterior_contrast() -> (f64, f64, f64) {
        // Draw weight 1/4: both arms ignore Z, so the means are 1/2 and 1/4.
        // Draw weight 3/4: active mean 3/4, control mean 1/4.
        let low = 0.5 - 0.25;
        let high = 0.75 - 0.25;
        let mean = 0.25 * low + 0.75 * high;
        (low, high, mean)
    }

    #[test]
    fn static_and_horizon_one_match_the_hand_discrete_expectation() {
        let ctx = ExecutionContext::for_tests(1);
        let slice = finite_discrete_slice(&ctx).unwrap();
        let (active, control, contrast) = hand_frequency_contrast();
        let matched = slice.functional();
        assert_eq!(matched.static_horizon, None);
        assert_eq!(matched.horizon_steps, 1);
        assert!(matched.same_shape);
        assert!(matched.identities_differ);
        assert_eq!(
            slice.expression_kinds(DiscreteCell::StaticFrequentist),
            slice.expression_kinds(DiscreteCell::HorizonOneBayesian)
        );
        assert_eq!(
            slice.expression_kinds(DiscreteCell::StaticFrequentist),
            ["distribution", "distribution", "product", "sum_out"].as_slice()
        );
        // The raw association is a different number: 15/24 - 10/40.
        let observational = 15.0 / 24.0 - 10.0 / 40.0;
        assert!((observational - 0.375_f64).abs() < 1e-12);
        for cell in [DiscreteCell::StaticFrequentist, DiscreteCell::HorizonOneFrequentist] {
            let fitted = slice.cell(cell);
            assert!((fitted.effect() - contrast).abs() < 1e-12, "{}", cell.as_str());
            assert!((fitted.active_mean() - active).abs() < 1e-12);
            assert!((fitted.control_mean() - control).abs() < 1e-12);
            assert!((fitted.effect() - observational).abs() > 1e-3);
        }
        assert!((matched.static_effect - contrast).abs() < 1e-12);
        assert!((matched.horizon_effect - contrast).abs() < 1e-12);
        assert_ne!(
            slice.cell(DiscreteCell::StaticFrequentist).regime(),
            slice.cell(DiscreteCell::HorizonOneFrequentist).regime()
        );
        assert_eq!(
            slice.cell(DiscreteCell::StaticFrequentist).regime(),
            slice.cell(DiscreteCell::StaticBayesian).regime()
        );
    }

    #[test]
    fn frequentist_sampling_and_bayesian_posterior_are_different_objects() {
        let ctx = ExecutionContext::for_tests(1);
        let slice = finite_discrete_slice(&ctx).unwrap();
        let (_, _, plug_in) = hand_frequency_contrast();
        let (low, high, posterior) = hand_posterior_contrast();
        assert!((plug_in - posterior).abs() > 1e-3);
        for cell in [DiscreteCell::StaticFrequentist, DiscreteCell::HorizonOneFrequentist] {
            let fitted = slice.cell(cell);
            assert_eq!(fitted.origin(), LawOrigin::EmpiricalPlugin);
            assert_eq!(fitted.port(), EvidencePort::SamplingLaw);
            assert_eq!(
                fitted.port().uncertainty_axis(),
                Some(antecedent_contract::UncertaintyAxis::Sampling)
            );
            assert_eq!(fitted.sample_size(), Some(27 + 9 + 3 + 1 + 6 + 6 + 3 + 9));
            assert!(fitted.posterior_mean().is_none());
            assert!((fitted.effect() - plug_in).abs() < 1e-12);
        }
        for cell in [DiscreteCell::StaticBayesian, DiscreteCell::HorizonOneBayesian] {
            let fitted = slice.cell(cell);
            assert_eq!(fitted.origin(), LawOrigin::BayesianPosterior);
            assert_eq!(fitted.port(), EvidencePort::ParameterPosterior);
            assert_eq!(
                fitted.port().uncertainty_axis(),
                Some(antecedent_contract::UncertaintyAxis::Parameter)
            );
            assert!(fitted.sample_size().is_none());
            assert!((fitted.posterior_mean().unwrap() - posterior).abs() < 1e-12);
            assert!((fitted.posterior_lower().unwrap() - low).abs() < 1e-12);
            assert!((fitted.posterior_upper().unwrap() - high).abs() < 1e-12);
            assert!((fitted.effect() - posterior).abs() < 1e-12);
        }
        assert_eq!(finite_discrete_origin("empirical_plugin").unwrap(), LawOrigin::EmpiricalPlugin);
        assert_eq!(
            finite_discrete_origin("bayesian_posterior").unwrap(),
            LawOrigin::BayesianPosterior
        );
        assert_eq!(finite_discrete_origin("learned_plugin").unwrap(), LawOrigin::LearnedPlugin);
        assert_eq!(finite_discrete_origin("supplied_exact").unwrap(), LawOrigin::SuppliedExact);
        assert!(finite_discrete_origin("gaussian").is_err());
        assert!(finite_discrete_origin("linear_gaussian").is_err());
        assert!(finite_discrete_origin("").is_err());
    }

    #[test]
    fn utility_and_action_changes_reuse_the_compiled_plan() {
        let ctx = ExecutionContext::for_tests(1);
        let slice = finite_discrete_slice(&ctx).unwrap();
        let before = slice.plan_compilations();
        assert!(before > 0);
        let (_, _, contrast) = hand_frequency_contrast();
        let (active, control, _) = hand_frequency_contrast();
        let score = |location: f64| 2.0_f64.mul_add(location, -0.5);
        let (utility, counts) = antecedent_expr::execution_counts::count_static_work(|| {
            slice
                .utility_change(
                    DiscreteCell::StaticFrequentist,
                    DecisionDemand::AffineMean { slope: 2.0, intercept: -0.5 },
                )
                .unwrap()
        });
        assert_eq!(counts.program_compilations, 0);
        assert_eq!(counts.provider_bindings, 0);
        assert_eq!(counts.integrations, 0);
        assert_eq!(counts.provider_calls, 0);
        assert_eq!(utility.plan_compilations, before);
        assert_eq!(utility.carrier, EvidencePort::Mean);
        assert!((utility.value - score(contrast)).abs() < 1e-12);
        let (action, action_counts) = antecedent_expr::execution_counts::count_static_work(|| {
            slice.action_change(DiscreteCell::StaticFrequentist, DiscreteAction::Control)
        });
        assert_eq!(action_counts.program_compilations, 0);
        assert_eq!(action_counts.integrations, 0);
        assert_eq!(action.plan_compilations, before);
        assert!((action.mean - control).abs() < 1e-12);
        let active_action =
            slice.action_change(DiscreteCell::HorizonOneFrequentist, DiscreteAction::Active);
        assert_eq!(active_action.plan_compilations, before);
        assert!((active_action.mean - active).abs() < 1e-12);
        assert_eq!(slice.plan_compilations(), before);
    }

    #[test]
    fn confidence_interval_offered_as_posterior_is_rejected_and_risk_needs_an_outcome_law() {
        let refused = confidence_interval_as_posterior().unwrap_err();
        assert_eq!(refused.source_port, EvidencePort::ConfidenceRegion);
        assert_eq!(refused.request, DecisionRequest::ParameterPosterior);
        assert_eq!(refused.legal_ports, &[EvidencePort::ParameterPosterior]);
        assert!(refused.message.contains("not a parameter posterior"));

        let ctx = ExecutionContext::for_tests(1);
        let slice = finite_discrete_slice(&ctx).unwrap();
        let posterior = DecisionDemand::AffinePosterior { slope: 2.0, intercept: -0.5 };
        let mean = DecisionDemand::AffineMean { slope: 2.0, intercept: -0.5 };
        for cell in [DiscreteCell::StaticBayesian, DiscreteCell::HorizonOneBayesian] {
            let fitted = slice.cell(cell);
            let scored = fitted.decide(posterior).unwrap();
            assert_eq!(scored.carrier, EvidencePort::ParameterPosterior);
            let location = fitted.posterior_mean().unwrap();
            assert!((scored.value - 2.0_f64.mul_add(location, -0.5)).abs() < 1e-12);
        }
        for cell in [DiscreteCell::StaticFrequentist, DiscreteCell::HorizonOneFrequentist] {
            let fitted = slice.cell(cell);
            assert_eq!(fitted.decide(posterior).unwrap_err(), MissingCarrier::ParameterPosterior);
            let scored = fitted.decide(mean).unwrap();
            assert_eq!(scored.carrier, EvidencePort::Mean);
        }
        for cell in DiscreteCell::ALL {
            let fitted = slice.cell(cell);
            assert_eq!(
                fitted.decide(DecisionDemand::Nonlinear).unwrap_err(),
                MissingCarrier::OutcomeLaw
            );
            assert_eq!(
                fitted.decide(DecisionDemand::Risk).unwrap_err(),
                MissingCarrier::OutcomeLaw
            );
        }
        assert_eq!(MissingCarrier::OutcomeLaw.as_str(), "outcome_law");
        let sampling = slice.cell(DiscreteCell::StaticFrequentist).port();
        assert!(sampling.try_coerce(DecisionRequest::ParameterPosterior).is_err());
    }

    #[test]
    fn enumeration_is_bounded_and_the_plan_names_its_factors() {
        let ctx = ExecutionContext::for_tests(1);
        let slice = finite_discrete_slice(&ctx).unwrap();
        let bound = slice.enumeration();
        assert_eq!(bound.operations, 10_000);
        assert_eq!(bound.depth, 32);
        assert_eq!(bound.support_rows, 8);
        assert_eq!(bound.outcome_atoms, 2);
        let domain =
            enumerated_atoms(ExactEvaluationLimits { operations: 10_000, depth: 32 }, 8, &ctx)
                .unwrap();
        assert_eq!(domain, 2);
        let domain_budget =
            enumerated_atoms(ExactEvaluationLimits { operations: 1, depth: 32 }, 8, &ctx)
                .unwrap_err();
        assert!(domain_budget.to_string().contains("exact domain budget exceeded"));
        let depth_budget =
            enumerated_atoms(ExactEvaluationLimits { operations: 10_000, depth: 1 }, 8, &ctx)
                .unwrap_err();
        assert!(depth_budget.to_string().contains("exact recursion budget exceeded"));
        let support_budget =
            enumerated_atoms(ExactEvaluationLimits { operations: 10_000, depth: 32 }, 1, &ctx)
                .unwrap_err();
        assert!(support_budget.to_string().contains("exact support budget exceeded"));

        for cell in [DiscreteCell::StaticFrequentist, DiscreteCell::HorizonOneFrequentist] {
            let factors = slice.factors(cell);
            assert_eq!(factors.len(), 2);
            assert!(factors.iter().all(|factor| factor.domain == DomainRef::Observational));
            assert!(factors.iter().all(|factor| factor.population.as_ref() == "study"));
            assert!(factors.iter().all(|factor| factor.intervention.is_empty()));
            let marginal = factors.iter().find(|factor| factor.conditioned_on.is_empty()).unwrap();
            let conditional =
                factors.iter().find(|factor| !factor.conditioned_on.is_empty()).unwrap();
            assert_eq!(marginal.variables.as_ref(), &[antecedent_core::VariableId::from_raw(1)]);
            assert_eq!(conditional.variables.as_ref(), &[antecedent_core::VariableId::from_raw(2)]);
            assert_eq!(
                conditional.conditioned_on.as_ref(),
                &[
                    antecedent_core::VariableId::from_raw(0),
                    antecedent_core::VariableId::from_raw(1)
                ]
            );
        }
        assert_eq!(
            slice.factors(DiscreteCell::StaticFrequentist)[0].regime,
            Some(RegimeId::from_raw(0))
        );
        assert_eq!(
            slice.factors(DiscreteCell::HorizonOneFrequentist)[0].regime,
            Some(RegimeId::from_raw(1))
        );
    }

    #[test]
    fn loaded_law_keeps_every_cell_and_the_hand_expectation() {
        let ctx = ExecutionContext::for_tests(1);
        let slice = finite_discrete_slice(&ctx).unwrap();
        let bytes = slice.export_static_frequentist().unwrap();
        let loaded = load_exported(&bytes).unwrap();
        let (_, _, contrast) = hand_frequency_contrast();
        let (active, _, _) = hand_frequency_contrast();
        assert_eq!(loaded.origin, "empirical_plugin");
        assert_eq!(loaded.horizon_steps, None);
        assert!(loaded.posterior_mean.is_none());
        assert_eq!(loaded.sample_size, Some(64));
        assert!((loaded.effect - contrast).abs() < 1e-12);
        assert_eq!(loaded.outcome_levels, vec![0, 1]);
        // P(Y=0 | do(T=1)) = 1 - 9/16, P(Y=1 | do(T=1)) = 9/16.
        assert!((loaded.outcome_probabilities[0] - (1.0 - active)).abs() < 1e-12);
        assert!((loaded.outcome_probabilities[1] - active).abs() < 1e-12);
        assert!((loaded.outcome_probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let expected = [
            27.0 / 64.0,
            9.0 / 64.0,
            3.0 / 64.0,
            1.0 / 64.0,
            6.0 / 64.0,
            6.0 / 64.0,
            3.0 / 64.0,
            9.0 / 64.0,
        ];
        assert_eq!(loaded.law.probabilities.len(), expected.len());
        assert_eq!(loaded.law.axes.len(), 3);
        assert!(loaded.law.axes.iter().all(|(_, levels)| levels.len() == 2));
        assert!(loaded.law.interventions.is_empty());
        assert_eq!(loaded.law.origin, "empirical_plugin");
        for (got, want) in loaded.law.probabilities.iter().zip(expected) {
            assert!((got - want).abs() < 1e-12);
        }
        assert!((loaded.law.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let counts = loaded.law.empirical_counts.as_ref().unwrap();
        assert_eq!(counts.len(), 8);
        assert_eq!(counts.iter().sum::<u64>(), 64);
        let (reloaded, work) = antecedent_expr::execution_counts::count_static_work(|| {
            effect_of_exported_law(&bytes, &ctx).unwrap()
        });
        assert!(work.program_compilations > 0);
        assert!(work.integrations > 0);
        assert!((reloaded - contrast).abs() < 1e-12);
    }

    /// Active-minus-control contrast of the adjustment read off a count table by hand:
    /// `sum_z P(z) P(y = 1 | t, z)` with index `4t + 2z + y`.
    fn hand_contrast(counts: &[u64; 8]) -> (f64, f64) {
        let total = counts.iter().sum::<u64>() as f64;
        let mean = |t: usize| -> f64 {
            (0..2)
                .map(|z| {
                    let z_mass = (0..2)
                        .flat_map(|tt| (0..2).map(move |y| (tt, y)))
                        .map(|(tt, y)| counts[tt * 4 + z * 2 + y])
                        .sum::<u64>() as f64
                        / total;
                    let cell = |y: usize| counts[t * 4 + z * 2 + y] as f64;
                    z_mass * cell(1) / (cell(0) + cell(1))
                })
                .sum()
        };
        (mean(1), mean(0))
    }

    const SUPPLIED: [u64; 8] = [40, 10, 12, 18, 9, 21, 15, 25];
    const DRAW_A: [u64; 8] = [30, 12, 12, 16, 10, 20, 14, 26];
    const DRAW_B: [u64; 8] = [28, 14, 16, 12, 12, 18, 10, 30];
    const DRAW_C: [u64; 8] = [35, 9, 14, 14, 8, 22, 18, 20];

    #[test]
    fn a_supplied_table_and_its_posterior_draws_plan_to_the_hand_expectation() {
        let ctx = ExecutionContext::for_tests(1);
        let weights = [0.2, 0.5, 0.3];
        let inputs = DiscreteInputs::new(
            SUPPLIED,
            [(DRAW_A, weights[0]), (DRAW_B, weights[1]), (DRAW_C, weights[2])],
        )
        .unwrap();
        let slice = finite_discrete_slice_over(&inputs, &ctx).unwrap();

        let (active, control) = hand_contrast(&SUPPLIED);
        for cell in [DiscreteCell::StaticFrequentist, DiscreteCell::HorizonOneFrequentist] {
            let fitted = slice.cell(cell);
            assert!((fitted.active_mean() - active).abs() < 1e-12, "{}", cell.as_str());
            assert!((fitted.control_mean() - control).abs() < 1e-12);
            assert!((fitted.effect() - (active - control)).abs() < 1e-12);
            assert_eq!(fitted.sample_size(), Some(SUPPLIED.iter().sum::<u64>()));
        }
        let contrasts: Vec<f64> = [DRAW_A, DRAW_B, DRAW_C]
            .iter()
            .map(|counts| {
                let (a, c) = hand_contrast(counts);
                a - c
            })
            .collect();
        let mean: f64 = weights.iter().zip(&contrasts).map(|(w, c)| w * c).sum();
        let lowest = contrasts.iter().copied().fold(f64::INFINITY, f64::min);
        let highest = contrasts.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for cell in [DiscreteCell::StaticBayesian, DiscreteCell::HorizonOneBayesian] {
            let fitted = slice.cell(cell);
            assert!((fitted.posterior_mean().unwrap() - mean).abs() < 1e-12, "{}", cell.as_str());
            assert!((fitted.posterior_lower().unwrap() - lowest).abs() < 1e-12);
            assert!((fitted.posterior_upper().unwrap() - highest).abs() < 1e-12);
        }
        // The reference table is untouched by a supplied one.
        let reference = finite_discrete_slice(&ctx).unwrap();
        assert!(
            (reference.cell(DiscreteCell::StaticFrequentist).effect()
                - slice.cell(DiscreteCell::StaticFrequentist).effect())
            .abs()
                > 1e-6
        );
    }

    #[test]
    fn the_facade_entry_reports_the_plug_in_and_the_posterior_of_a_supplied_table() {
        use super::finite_discrete_contrast;
        let ctx = ExecutionContext::for_tests(1);
        let report =
            finite_discrete_contrast(SUPPLIED, [(DRAW_A, 0.5), (DRAW_B, 0.5)], &ctx).unwrap();
        let (active, control) = hand_contrast(&SUPPLIED);
        assert!((report.plug_in_effect - (active - control)).abs() < 1e-12);
        assert_eq!(report.sample_size, SUPPLIED.iter().sum::<u64>());
        let contrasts: Vec<f64> = [DRAW_A, DRAW_B]
            .iter()
            .map(|counts| {
                let (a, c) = hand_contrast(counts);
                a - c
            })
            .collect();
        assert!((report.posterior_mean - 0.5 * (contrasts[0] + contrasts[1])).abs() < 1e-12);
        assert!(report.posterior_lower <= report.posterior_mean);
        assert!(report.posterior_mean <= report.posterior_upper);
        assert!(report.plan_compilations > 0);
        ctx.cancellation.cancel();
        assert!(finite_discrete_contrast(SUPPLIED, [(DRAW_A, 1.0)], &ctx).is_err());
    }

    #[test]
    fn the_supplied_table_survives_an_independent_consumer() {
        let ctx = ExecutionContext::for_tests(1);
        let inputs = DiscreteInputs::new(SUPPLIED, [(DRAW_A, 1.0)]).unwrap();
        let slice = finite_discrete_slice_over(&inputs, &ctx).unwrap();
        let bytes = slice.export_static_frequentist().unwrap();
        let loaded = load_exported(&bytes).unwrap();
        assert_eq!(loaded.sample_size, Some(SUPPLIED.iter().sum::<u64>()));
        assert_eq!(loaded.law.empirical_counts.as_deref(), Some(SUPPLIED.as_slice()));
        let (active, control) = hand_contrast(&SUPPLIED);
        let reloaded = effect_of_exported_law(&bytes, &ctx).unwrap();
        assert!((reloaded - (active - control)).abs() < 1e-12);
    }

    #[test]
    fn inputs_without_mass_draws_or_normalized_weights_are_refused() {
        let refused = |inputs: Result<DiscreteInputs, crate::CausalError>| inputs.is_err();
        assert!(refused(DiscreteInputs::new([0; 8], [(DRAW_A, 1.0)])));
        assert!(refused(DiscreteInputs::new(SUPPLIED, [(DRAW_A, 0.9)])));
        assert!(refused(DiscreteInputs::new(SUPPLIED, [(DRAW_A, 1.5), (DRAW_B, -0.5)])));
        assert!(refused(DiscreteInputs::new(SUPPLIED, [(DRAW_A, f64::NAN)])));
        assert!(refused(DiscreteInputs::new(SUPPLIED, std::iter::empty())));
        assert!(refused(DiscreteInputs::new(SUPPLIED, [([0; 8], 1.0)])));
        let mut huge = SUPPLIED;
        huge[0] = MAX_EXACT_COUNT;
        assert!(refused(DiscreteInputs::new(huge, [(DRAW_A, 1.0)])));
        assert!(refused(DiscreteInputs::new([u64::MAX, 1, 0, 0, 0, 0, 0, 0], [(DRAW_A, 1.0)])));
    }

    #[test]
    fn a_table_beyond_any_row_expansion_plans_from_its_counts() {
        const SCALE: u64 = 10_000_000;
        let ctx = ExecutionContext::for_tests(1);
        let scaled = SUPPLIED.map(|count| count * SCALE);
        let draw = DRAW_A.map(|count| count * SCALE);
        let inputs = DiscreteInputs::new(scaled, [(draw, 1.0)]).unwrap();
        let slice = finite_discrete_slice_over(&inputs, &ctx).unwrap();
        let (active, control) = hand_contrast(&SUPPLIED);
        let fitted = slice.cell(DiscreteCell::StaticFrequentist);
        assert!((fitted.effect() - (active - control)).abs() < 1e-12);
        assert_eq!(fitted.sample_size(), Some(SUPPLIED.iter().sum::<u64>() * SCALE));
    }

    #[test]
    fn cancellation_stops_planning_before_and_between_plans() {
        let inputs = DiscreteInputs::new(SUPPLIED, [(DRAW_A, 0.5), (DRAW_B, 0.5)]).unwrap();

        let cancelled = ExecutionContext::for_tests(1);
        cancelled.cancellation.cancel();
        let before = finite_discrete_slice_over(&inputs, &cancelled).err().unwrap();
        assert!(matches!(before, crate::CausalError::Cancelled { .. }));

        let mut late = ExecutionContext::for_tests(1);
        late.cancellation = antecedent_core::CancellationToken::cancel_after_checks(6);
        let between = finite_discrete_slice_over(&inputs, &late).err().unwrap();
        assert!(matches!(between, crate::CausalError::Cancelled { .. }), "{between:?}");
    }

    #[test]
    fn a_pool_that_cannot_hold_the_support_enumeration_is_refused_before_planning() {
        use antecedent_core::execution::AdmissionPool;
        let inputs = DiscreteInputs::new(SUPPLIED, [(DRAW_A, 1.0)]).unwrap();
        let pool = AdmissionPool::fixed(2, 64);
        let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
        let refused = finite_discrete_slice_over(&inputs, &ctx).err().unwrap();
        assert!(matches!(refused, crate::CausalError::Resource { .. }), "{refused:?}");
        assert_eq!(pool.available().bytes(), 64, "a refused admission holds nothing");

        let enough = AdmissionPool::fixed(2, 1 << 30);
        let ctx = ExecutionContext::for_tests(1).with_admission(enough.clone());
        drop(finite_discrete_slice_over(&inputs, &ctx).unwrap());
        assert_eq!(enough.available().bytes(), 1 << 30, "every lease is released");
    }
}
