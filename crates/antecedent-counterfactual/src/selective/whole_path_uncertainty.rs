//! Whole-path uncertainty of the finite Markovian cross-world row.
//!
//! The path is abduction, action, per-world prediction and summary, with one
//! exogenous posterior state shared by every world of a unit. This module
//! derives which uncertainty sources that path has, which of them it
//! propagates and how, and which it only records as a premise. It computes the
//! propagated quantities from the executed row; it never turns them into an
//! interval or a posterior.
//!
//! # Sources
//!
//! * **Structural.** The mechanisms, the exogenous laws, Markovianity, the
//!   history and the population are declared, not estimated. Their adequacy is
//!   a premise of the row, not a quantity this row can propagate; no
//!   mechanism-parameter uncertainty exists because nothing is fitted
//!   (`mechanism_fits` is zero). A fitted or sampled mechanism is a different
//!   row with its own posterior input and is closed here.
//! * **Posterior.** The exogenous posterior of a unit given its factual
//!   evidence is enumerated exactly, so the abduction has no estimation error.
//!   What remains is the *predictive* variance of the unit's counterfactual
//!   quantity under that posterior, `v_i = sum_s w_s (g_s - q_i)^2`, where `g_s`
//!   is the quantity at posterior state `s`, evaluated with every world
//!   reading the same state. For a contrast or a joint event across worlds this
//!   is an expectation under the aligned (shared-exogenous) joint law; it is
//!   not the sum of the worlds' marginal variances.
//! * **Sampling.** The units are a set of cases. The mean of the unit values
//!   is the estimand of the *fixed* units, with no sampling variance, or the
//!   estimate of the mean over a population from which the units are an
//!   independent sample, with sampling variance `s^2 / n` of the unit values
//!   `q_i`. Which one holds is a declaration ([`UnitSampling`]); an undeclared
//!   one is refused. Units are independent only when their evidence is
//!   distinct, which the joint-law rule checks.
//! * **Numerical.** Exact enumeration of the joint exogenous states: no draws
//!   are taken (`shared_draws` is zero). The observed defect of each posterior's
//!   weight sum is reported; floating-point rounding of the enumeration is not
//!   bounded here.
//!
//! The two spreads are for two estimands and are never added:
//! `realized_mean_predictive_variance` is `sum v_i / n^2`, the variance of the
//! mean of the realized counterfactual outcomes of these units about the mean of
//! their unit probabilities (units independent given their evidence);
//! `unit_mean_sampling_variance` is `s^2 / n`, the variance of the mean unit
//! probability as an estimate of the population mean.
//!
//! # Re-derivation
//!
//! The derivation re-abduces and re-predicts every unit from the request and
//! requires every unit value and the mean to equal the executed result bit for
//! bit. A result computed from other mechanisms, evidence, worlds or quantity,
//! even at an unchanged shape, is refused rather than annotated. The function is
//! pure: it retains and replaces nothing.
//!
//! Every spread carries the label `uncalibrated_spread`: there is no measured
//! coverage record for the abduction-action-prediction path, so an interval or
//! posterior reading is closed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{
    InputAlignment, InputRole, PropagationDependence, PropagationInput, PropagationOperation,
    PropagationRequest, RequestedReading, check_propagation,
};
use antecedent_core::{CanonicalWriter, ExecutionContext};

use super::finite_executor::{ExogenousInference, FiniteExecutionRequest, FiniteExecutionResult};
use super::finite_scm::{
    CompensatedSum, MAX_ABDUCTION_WORK, MAX_EVALUATION_WORK, WorldPlan, abduct_unit,
    evaluate_worlds, validate_units, validate_worlds,
};
use super::refusal::{CrossWorldExecutionError, from_external, invalid, poll, refuse};
use super::stages::WorkCounters;

/// Version of the derivation identity.
pub const WHOLE_PATH_UNCERTAINTY_VERSION: u32 = 1;

/// How the units of a request relate to the population the claim is about.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnitSampling {
    /// Nothing declared. Refused: the estimand and its sampling variance are
    /// not determined.
    Undeclared,
    /// The units are the whole set the claim is about; the mean is the
    /// estimand and has no sampling variance.
    FixedUnits,
    /// The units are an independent sample from a population and the claim is
    /// about the population mean of the unit probabilities.
    IndependentSample,
}

impl UnitSampling {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Undeclared => "undeclared",
            Self::FixedUnits => "fixed_units",
            Self::IndependentSample => "independent_sample",
        }
    }
}

/// A source of uncertainty on the abduction-action-prediction path.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum UncertaintySource {
    /// Mechanisms, exogenous laws, structure, history, population.
    Structural,
    /// The exogenous posterior given the factual evidence.
    Posterior,
    /// The units as a sample of a population.
    Sampling,
    /// Enumeration arithmetic.
    Numerical,
}

impl UncertaintySource {
    /// Every source in a fixed order.
    pub const ALL: [UncertaintySource; 4] =
        [Self::Structural, Self::Posterior, Self::Sampling, Self::Numerical];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Structural => "structural",
            Self::Posterior => "posterior",
            Self::Sampling => "sampling",
            Self::Numerical => "numerical",
        }
    }
}

/// What the path does with one source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SourceTreatment {
    /// A declared premise: not propagated, not estimable by this row.
    Premise,
    /// Exact by enumeration: no estimation error. The observed defect of the
    /// enumerated weights is recorded.
    Exact {
        /// Largest `|sum of posterior weights - 1|` over the units.
        observed_weight_defect: f64,
    },
    /// Propagated into a variance.
    Propagated {
        /// The variance contribution (see the module documentation for which
        /// estimand it belongs to).
        variance: f64,
    },
    /// Not applicable under the declaration.
    NotApplicable(&'static str),
    /// Required but not determined by the data supplied.
    Unavailable(&'static str),
}

/// One source and what the path does with it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceAccount {
    /// The source.
    pub source: UncertaintySource,
    /// Its treatment.
    pub treatment: SourceTreatment,
}

/// One unit's value and predictive variance.
#[derive(Clone, Debug, PartialEq)]
pub struct UnitSpread {
    /// The unit.
    pub unit: String,
    /// `q_i`, the unit's counterfactual quantity.
    pub value: f64,
    /// `v_i`, the predictive variance of the quantity under the unit's
    /// exogenous posterior with shared exogenous states.
    pub predictive_variance: f64,
}

/// The derived whole-path uncertainty of an executed request.
#[derive(Clone, Debug, PartialEq)]
pub struct WholePathUncertainty {
    /// Always `uncalibrated_spread`.
    pub label: &'static str,
    /// The declared unit sampling.
    pub unit_sampling: UnitSampling,
    /// Mean of the unit values (identical to the executed result).
    pub value: f64,
    /// Per-unit values and predictive variances, in unit-id order.
    pub units: Vec<UnitSpread>,
    /// `sum v_i / n^2`.
    pub realized_mean_predictive_variance: f64,
    /// `s^2 / n` of the unit values, present only for an independent sample of
    /// at least two units.
    pub unit_mean_sampling_variance: Option<f64>,
    /// Every source and its treatment, in [`UncertaintySource::ALL`] order.
    pub sources: Vec<SourceAccount>,
    /// The joint-law routes the rule licensed for the spread (within-unit,
    /// then between-unit when applicable).
    pub routes: Vec<&'static str>,
    /// Work observed in this derivation.
    pub work: WorkCounters,
    /// Identity over the executed receipt, the declaration and every number.
    pub identity: String,
}

impl WholePathUncertainty {
    /// The account of one source.
    #[must_use]
    pub fn source(&self, source: UncertaintySource) -> Option<&SourceAccount> {
        self.sources.iter().find(|account| account.source == source)
    }

    /// Standard error of the realized-outcome mean about the mean of the unit
    /// probabilities.
    #[must_use]
    pub fn realized_mean_standard_error(&self) -> f64 {
        self.realized_mean_predictive_variance.sqrt()
    }

    /// Standard error of the mean unit probability as a population-mean
    /// estimate, when determined.
    #[must_use]
    pub fn unit_mean_standard_error(&self) -> Option<f64> {
        self.unit_mean_sampling_variance.map(f64::sqrt)
    }
}

fn rule_failure(request: &PropagationRequest) -> Result<&'static str, CrossWorldExecutionError> {
    check_propagation(request).map(|plan| plan.route.as_str()).map_err(from_external)
}

/// Derive the whole-path uncertainty of an executed finite-row request.
///
/// `result` must be the result of executing `request`; the derivation
/// re-abduces and re-predicts every unit and refuses a result that differs.
///
/// # Errors
/// A typed refusal ([`CrossWorldExecutionError::Refused`]): undeclared unit
/// sampling, a sampled or transported request, a coupling the joint-law rule
/// does not license for the within-unit spread, units with shared evidence
/// declared independent, invalid declarations, a result that does not match
/// the request, or an exceeded bound; or cancellation.
#[allow(clippy::too_many_lines, reason = "one sequential derivation of the four named sources")]
pub fn derive_whole_path_uncertainty(
    request: &FiniteExecutionRequest,
    result: &FiniteExecutionResult,
    unit_sampling: UnitSampling,
    ctx: &ExecutionContext,
) -> Result<WholePathUncertainty, CrossWorldExecutionError> {
    poll(ctx)?;
    if unit_sampling == UnitSampling::Undeclared {
        return Err(refuse(
            antecedent_core::reason_code!("sampling_dependence_unknown"),
            "cross_world_uncertainty.unit_sampling_undeclared",
            Some("unit_sampling".to_owned()),
            "declare whether the units are the whole set the claim is about (fixed units) or an \
             independent sample of a population; the estimand and its sampling variance depend on it",
        ));
    }
    if request.exogenous == ExogenousInference::PosteriorSampling {
        return Err(refuse(
            antecedent_core::reason_code!("cell_not_licensed"),
            "cross_world_uncertainty.sampled_path_not_licensed",
            Some("posterior_sampling".to_owned()),
            "a sampled exogenous posterior adds Monte Carlo and mixing error whose coverage for the \
             whole path is not measured; use exact enumeration, whose posterior has no estimation \
             error",
        ));
    }
    if request.target_population.as_ref().is_some_and(|t| t != &request.scm.population) {
        return Err(refuse(
            antecedent_core::reason_code!("cell_not_licensed"),
            "cross_world_uncertainty.transported_path_closed",
            request.target_population.clone(),
            "the uncertainty of a transported cross-world quantity composes the transport \
             identification with a target-population abduction, for which no derivation exists; \
             derive the uncertainty in the population the mechanisms were declared for",
        ));
    }
    if request.evidence_history != request.scm.history {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "cross_world_uncertainty.incompatible_history",
            Some(request.evidence_history.clone()),
            "declare the mechanisms for the history the evidence was gathered under",
        ));
    }

    let compiled = request.scm.compile()?;
    let units = validate_units(&request.scm, &request.units)?;
    validate_worlds(&request.scm, &request.worlds)?;
    request.quantity.validate(&request.scm, request.worlds.len())?;
    if units.len().saturating_mul(compiled.joint_states) > MAX_ABDUCTION_WORK {
        return Err(invalid("cross_world_uncertainty.bounds_exceeded", "abduction_work"));
    }
    if compiled
        .joint_states
        .saturating_mul(request.scm.variables.len())
        .saturating_mul(request.worlds.len())
        > MAX_EVALUATION_WORK
    {
        return Err(invalid("cross_world_uncertainty.bounds_exceeded", "evaluation_work"));
    }
    if result.units.len() != units.len() {
        return Err(mismatch("unit_count"));
    }

    // Rule checks run before any enumeration. The within-unit spread of a
    // quantity over several worlds is an expectation under the one joint law
    // of the worlds, so it needs the shared exogenous state declared.
    let involved = request.quantity.involved_worlds(&request.worlds);
    let evidence_id = result.receipt.abduction_digest.clone();
    let mut routes = Vec::with_capacity(2);
    let within = if involved.len() >= 2 {
        let (alignment, dependence) = match request.coupling {
            super::finite_executor::NoiseCoupling::SharedAbducedExogenous => (
                InputAlignment::Aligned { alignment_id: format!("shared_exogenous:{evidence_id}") },
                PropagationDependence::AlignedJointLaw,
            ),
            super::finite_executor::NoiseCoupling::IndependentWorldNoise => {
                (InputAlignment::IndependentMarginals, PropagationDependence::DeclaredIndependent)
            }
            super::finite_executor::NoiseCoupling::Undeclared => {
                (InputAlignment::IndependentMarginals, PropagationDependence::Unknown)
            }
        };
        PropagationRequest {
            operation: PropagationOperation::NonlinearFunctional,
            inputs: involved
                .iter()
                .map(|w| PropagationInput {
                    id: request.worlds[*w].name.clone(),
                    evidence_id: evidence_id.clone(),
                    role: InputRole::Estimate,
                    alignment: alignment.clone(),
                })
                .collect(),
            dependence,
            reading: RequestedReading::Spread,
        }
    } else {
        // One world: the unit's quantity is a function of the one aligned
        // posterior, read by a single input.
        PropagationRequest {
            operation: PropagationOperation::LinearTransform,
            inputs: vec![PropagationInput {
                id: "world".to_owned(),
                evidence_id: evidence_id.clone(),
                role: InputRole::Estimate,
                alignment: InputAlignment::Aligned {
                    alignment_id: format!("shared_exogenous:{evidence_id}"),
                },
            }],
            dependence: PropagationDependence::AlignedJointLaw,
            reading: RequestedReading::Spread,
        }
    };
    routes.push(rule_failure(&within)?);
    if unit_sampling == UnitSampling::IndependentSample && units.len() >= 2 {
        let between = PropagationRequest {
            operation: PropagationOperation::WeightedSum,
            inputs: units
                .iter()
                .map(|u| PropagationInput {
                    id: u.unit.clone(),
                    evidence_id: u.digest(),
                    role: InputRole::Estimate,
                    alignment: InputAlignment::MomentsOnly,
                })
                .collect(),
            dependence: PropagationDependence::DeclaredIndependent,
            reading: RequestedReading::Spread,
        };
        routes.push(rule_failure(&between)?);
    }

    // Re-abduce and re-predict; the executed values must be reproduced exactly.
    let plans: Vec<WorldPlan> = (0..request.worlds.len())
        .map(|w| WorldPlan::new(&request.scm, &request.worlds, w))
        .collect();
    let n_vars = request.scm.variables.len();
    let mut values = vec![vec![0u8; n_vars]; request.worlds.len()];
    let mut work = WorkCounters::default();
    let mut unit_spreads = Vec::with_capacity(units.len());
    let mut total = CompensatedSum::default();
    let mut weight_defect = 0.0_f64;
    for (unit, executed) in units.iter().zip(&result.units) {
        poll(ctx)?;
        if executed.unit != unit.unit {
            return Err(mismatch("unit_identity"));
        }
        let posterior = abduct_unit(&compiled, unit, ctx)?;
        work.abductions += 1;
        work.abduction_states_enumerated += compiled.joint_states as u64;
        let mut at_state = Vec::with_capacity(posterior.states.len());
        let mut expectation = CompensatedSum::default();
        let mut weight_sum = CompensatedSum::default();
        for (state, weight) in &posterior.states {
            evaluate_worlds(&compiled, &plans, *state as usize, &mut values);
            let g = request.quantity.at_state(&values);
            at_state.push((*weight, g));
            expectation.add(weight * g);
            weight_sum.add(*weight);
        }
        let value = expectation.total();
        let weight_sum = weight_sum.total();
        work.world_predictions += request.worlds.len() as u64;
        work.posterior_states_predicted += (posterior.states.len() * request.worlds.len()) as u64;
        if value.to_bits() != executed.value.to_bits() {
            return Err(mismatch("unit_value"));
        }
        // Two-pass predictive variance about the unit's own mean.
        let predictive_variance: f64 =
            at_state.iter().map(|(w, g)| w * (g - value) * (g - value)).sum();
        weight_defect = weight_defect.max((weight_sum - 1.0).abs());
        total.add(value);
        unit_spreads.push(UnitSpread { unit: unit.unit.clone(), value, predictive_variance });
    }
    let n = unit_spreads.len() as f64;
    let mean = total.total() / n;
    if mean.to_bits() != result.value.to_bits() {
        return Err(mismatch("mean"));
    }

    let realized: f64 = unit_spreads.iter().map(|u| u.predictive_variance).sum::<f64>() / (n * n);
    let sampling_variance =
        if unit_sampling == UnitSampling::IndependentSample && unit_spreads.len() >= 2 {
            let ss: f64 = unit_spreads.iter().map(|u| (u.value - mean) * (u.value - mean)).sum();
            Some(ss / (n - 1.0) / n)
        } else {
            None
        };
    work.summaries += 1;

    let sampling = match (unit_sampling, sampling_variance) {
        (UnitSampling::IndependentSample, Some(variance)) => {
            SourceTreatment::Propagated { variance }
        }
        (UnitSampling::IndependentSample, None) => {
            SourceTreatment::Unavailable("one unit has no between-unit variance")
        }
        _ => SourceTreatment::NotApplicable("the units are the whole set the claim is about"),
    };
    let sources = vec![
        SourceAccount {
            source: UncertaintySource::Structural,
            treatment: SourceTreatment::Premise,
        },
        SourceAccount {
            source: UncertaintySource::Posterior,
            treatment: SourceTreatment::Propagated { variance: realized },
        },
        SourceAccount { source: UncertaintySource::Sampling, treatment: sampling },
        SourceAccount {
            source: UncertaintySource::Numerical,
            treatment: SourceTreatment::Exact { observed_weight_defect: weight_defect },
        },
    ];

    let mut w = CanonicalWriter::new("antecedent.counterfactual.selective.whole_path.v1");
    w.word(u64::from(WHOLE_PATH_UNCERTAINTY_VERSION));
    w.text(&result.receipt.digest).text(unit_sampling.name());
    w.word(unit_spreads.len() as u64);
    for u in &unit_spreads {
        w.text(&u.unit).real(u.value).real(u.predictive_variance);
    }
    w.real(realized);
    match sampling_variance {
        Some(v) => w.word(1).real(v),
        None => w.word(0),
    };
    for route in &routes {
        w.text(route);
    }
    let identity = w.finish_hex();

    Ok(WholePathUncertainty {
        label: antecedent_contract::UNCALIBRATED_SPREAD_LABEL,
        unit_sampling,
        value: mean,
        units: unit_spreads,
        realized_mean_predictive_variance: realized,
        unit_mean_sampling_variance: sampling_variance,
        sources,
        routes,
        work,
        identity,
    })
}

fn mismatch(what: &str) -> CrossWorldExecutionError {
    refuse(
        antecedent_core::reason_code!("invalid_argument"),
        "cross_world_uncertainty.result_mismatch",
        Some(what.to_owned()),
        "supply the result of executing this very request; the derivation re-abduces every unit and \
         refuses a result computed from other mechanisms, evidence, worlds or quantity",
    )
}
