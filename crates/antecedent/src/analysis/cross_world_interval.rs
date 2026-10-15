//! The measured interval of a whole-path cross-world quantity.
//!
//! Two intervals exist, one for each declared [`UnitSampling`], and each is a
//! statement conditional on the declared mechanisms being the truth (model
//! adequacy is a premise of the row, never a check):
//!
//! * **Independent sample of a population:** the normal interval of the mean
//!   unit probability for the population mean of the counterfactual
//!   probability, with the sampling variance `s^2 / n` of the unit values.
//! * **Fixed units:** the prediction interval of the mean of the realized
//!   counterfactual outcomes of these units about the mean of their unit
//!   probabilities, with the predictive variance `sum v_i / n^2`.
//!
//! The calibration key is derived from the executed request and the declared
//! unit sampling, and the harness binds the same key. A request outside the
//! measured model class (more than three variables, a variable with more than
//! four levels, a sampled posterior, a transported population) has no record and
//! is closed. A sampled posterior and a posterior statement stay closed: with
//! declared mechanisms the exogenous posterior is exact, so there is no
//! parameter uncertainty a posterior could describe.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CalibrationBasis, ExecutionContext};
use antecedent_io::calibration::{
    CalibrationBasisWire, CalibrationKeyWire, CalibrationScopeWire, calibration_slot,
};

use super::cross_world_execution::{
    CrossWorldExecution, CrossWorldExecutionError, CrossWorldExecutionRefusalWire,
    ExogenousInference, FiniteExecutionRequest, FiniteQuantity, UnitSampling, WholePathUncertainty,
};

/// Query axis of the cross-world records.
pub const CROSS_WORLD_QUERY: &str = "Counterfactual";
/// Estimator label of the whole abduction-action-prediction path.
pub const CROSS_WORLD_ESTIMATOR: &str = "cross_world.finite_scm.whole_path";
/// Interval construction.
pub const CROSS_WORLD_CONSTRUCTION: &str = "normal_approximation";
/// Nominal level.
pub const CROSS_WORLD_LEVEL: f64 = 0.95;
/// Most variables and levels the design measured.
const MEASURED_VARIABLES: usize = 3;
const MEASURED_LEVELS: usize = 4;

fn refused(detail: &str, offending: &str, remedy: &str) -> CrossWorldExecutionRefusalWire {
    CrossWorldExecutionRefusalWire {
        code: antecedent_core::reason_code!("cell_not_licensed").to_owned(),
        stage: "uncertainty".to_owned(),
        detail: detail.to_owned(),
        offending: Some(offending.to_owned()),
        remedy: Some(remedy.to_owned()),
        missing: vec!["whole_path_calibration_record".to_owned()],
    }
}

fn from_error(error: &CrossWorldExecutionError) -> CrossWorldExecutionRefusalWire {
    match error.refusal() {
        Some(r) => CrossWorldExecutionRefusalWire {
            code: r.code.to_owned(),
            stage: "uncertainty".to_owned(),
            detail: r.detail.clone(),
            offending: r.offending.clone(),
            remedy: r.remedy.map(str::to_owned),
            missing: error.missing().to_vec(),
        },
        None => CrossWorldExecutionRefusalWire {
            code: antecedent_core::reason_code!("cancelled_no_claim").to_owned(),
            stage: "uncertainty".to_owned(),
            detail: "cross_world_execution.cancelled".to_owned(),
            offending: None,
            remedy: None,
            missing: Vec::new(),
        },
    }
}

fn quantity_label(quantity: &FiniteQuantity) -> &'static str {
    match quantity {
        FiniteQuantity::Probability(_) => "counterfactual_probability",
        FiniteQuantity::MeanContrast { .. } => "counterfactual_mean_contrast",
        FiniteQuantity::JointProbability(_) => "counterfactual_joint_probability",
    }
}

/// The calibration key of a whole-path interval.
///
/// # Errors
/// A request outside the measured model class: a sampled posterior, a
/// transported population, more than three variables or more than four levels.
#[allow(clippy::result_large_err, reason = "the refusal wire is the shared execution refusal type")]
pub fn whole_path_interval_basis(
    request: &FiniteExecutionRequest,
    unit_sampling: UnitSampling,
) -> Result<CalibrationBasis, CrossWorldExecutionRefusalWire> {
    let outside = |what: &str| {
        refused(
            "cross_world_execution.interval_route_closed",
            what,
            "only exact enumeration of a declared finite Markovian SCM with at most three variables of two to four levels, in the population the mechanisms were declared for, has a measured whole-path coverage record; use the point value",
        )
    };
    if request.exogenous != ExogenousInference::ExactEnumeration {
        return Err(outside("exogenous"));
    }
    if request.target_population.as_ref().is_some_and(|t| t != &request.scm.population) {
        return Err(outside("target_population"));
    }
    if request.scm.variables.len() > MEASURED_VARIABLES
        || request.scm.variables.iter().any(|v| v.levels > MEASURED_LEVELS)
    {
        return Err(outside("scm"));
    }
    let a = |text: &str| Arc::<str>::from(text);
    Ok(CalibrationBasis::new(
        [
            a(CROSS_WORLD_QUERY),
            a("Dag"),
            a("fixed"),
            a("tabular"),
            a("Frequentist"),
            a(CROSS_WORLD_ESTIMATOR),
            a(CROSS_WORLD_CONSTRUCTION),
            a(""),
            a(unit_sampling.name()),
            a(""),
            a(quantity_label(&request.quantity)),
        ],
        CROSS_WORLD_LEVEL,
        a("point"),
        request.units.len() as u64,
        None,
        None,
        0.0,
    ))
}

/// The point, the interval and the scalar name a derived account reports.
///
/// `None` when the declared sampling has no determined variance (a single
/// unit under independent sampling) or the level is invalid.
#[must_use]
pub fn whole_path_interval(
    uncertainty: &WholePathUncertainty,
    level: f64,
) -> Option<(f64, (f64, f64), &'static str)> {
    if !(level > 0.0 && level < 1.0) {
        return None;
    }
    #[allow(
        clippy::match_wildcard_for_single_variants,
        reason = "an undeclared sampling carries no interval"
    )]
    let (variance, scalar) = match uncertainty.unit_sampling {
        UnitSampling::FixedUnits => {
            (uncertainty.realized_mean_predictive_variance, "realized_counterfactual_mean")
        }
        UnitSampling::IndependentSample => {
            (uncertainty.unit_mean_sampling_variance?, "population_mean_counterfactual_probability")
        }
        _ => return None,
    };
    if !(variance.is_finite() && variance >= 0.0 && uncertainty.value.is_finite()) {
        return None;
    }
    let z = antecedent_stats::special::normal_ppf(0.5 + level / 2.0);
    let half = z * variance.sqrt();
    Some((uncertainty.value, (uncertainty.value - half, uncertainty.value + half), scalar))
}

/// A whole-path interval bound to its governing coverage record.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossWorldInterval {
    /// The reported scalar: `population_mean_counterfactual_probability` or
    /// `realized_counterfactual_mean`.
    pub scalar: &'static str,
    /// The point value.
    pub point: f64,
    /// Lower end.
    pub lower: f64,
    /// Upper end.
    pub upper: f64,
    /// Nominal level.
    pub level: f64,
    /// The declared unit sampling.
    pub unit_sampling: &'static str,
    /// Identity of the governing coverage record.
    pub record_id: String,
    /// Always `measured_interval_under_the_declared_model`.
    pub label: &'static str,
}

fn resolve(basis: &CalibrationBasis) -> (Option<String>, bool, Option<String>) {
    let wire = CalibrationBasisWire {
        key: CalibrationKeyWire {
            query: basis.query.to_string(),
            graph_class: basis.graph_class.to_string(),
            structure: basis.structure.to_string(),
            modality: basis.modality.to_string(),
            inference: basis.inference.to_string(),
            estimator: basis.estimator.to_string(),
            interval_method: basis.interval_method.to_string(),
            se_kind: basis.se_kind.to_string(),
            dependence: basis.dependence.to_string(),
            posterior: basis.posterior.to_string(),
            functional: basis.functional.to_string(),
            level: basis.level,
            identification: basis.identification.to_string(),
        },
        scope: CalibrationScopeWire {
            row_count: basis.row_count,
            replicates_ok: basis.replicates_ok,
            posterior_draws: basis.posterior_draws,
            unidentified_mass: basis.unidentified_mass,
        },
    };
    let slot = calibration_slot(&wire);
    (slot.record_id, slot.status == "calibrated", slot.reason)
}

impl CrossWorldExecution {
    /// The interval of the executed quantity under a declared unit sampling,
    /// licensed only by a currently attesting whole-path coverage record.
    ///
    /// The unqualified [`Self::interval`] stays closed; this reading names the
    /// declared unit sampling and is a statement conditional on the declared
    /// mechanisms. A posterior reading and a sampled abduction stay closed.
    ///
    /// # Errors
    /// Undeclared unit sampling or any refusal of
    /// [`Self::whole_path_uncertainty`]; a request outside the measured model
    /// class; a single unit under independent sampling; or no currently
    /// attesting record (`cell_not_licensed`,
    /// `cross_world_execution.interval_route_closed`).
    #[allow(
        clippy::result_large_err,
        reason = "the refusal wire is the shared execution refusal type"
    )]
    pub fn measured_interval(
        &self,
        unit_sampling: UnitSampling,
        ctx: &ExecutionContext,
    ) -> Result<CrossWorldInterval, CrossWorldExecutionRefusalWire> {
        let basis = whole_path_interval_basis(self.artifact().request(), unit_sampling)?;
        let account =
            self.whole_path_uncertainty(unit_sampling, ctx).map_err(|e| from_error(&e))?;
        let Some((point, (lower, upper), scalar)) =
            whole_path_interval(account.uncertainty(), CROSS_WORLD_LEVEL)
        else {
            return Err(refused(
                "cross_world_execution.interval_route_closed",
                "unit_sampling",
                "the declared sampling determines no variance for these units (an independent sample needs at least two units); use the point value",
            ));
        };
        match resolve(&basis) {
            (Some(record_id), true, _) => Ok(CrossWorldInterval {
                scalar,
                point,
                lower,
                upper,
                level: CROSS_WORLD_LEVEL,
                unit_sampling: unit_sampling.name(),
                record_id,
                label: "measured_interval_under_the_declared_model",
            }),
            (_, _, reason) => Err(refused(
                "cross_world_execution.interval_route_closed",
                reason.as_deref().unwrap_or("no_record"),
                "no currently attesting whole-path coverage record governs this model class, quantity, unit sampling and unit count; use the point value",
            )),
        }
    }
}
