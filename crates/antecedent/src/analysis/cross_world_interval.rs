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
//! unit sampling, and the harness binds the same key. The interval is always
//! reported; its calibration slot says whether a current coverage record measured
//! that model class (`calibrated`), or the request lies outside what was measured
//! (more variables or levels than the measured designs, a shared exogenous term:
//! `outside_measured_scope` / `not_measured`). A transported population is a
//! scientific refusal: no theorem composes the transport identification with a
//! target-population abduction. A posterior statement stays refused: with declared
//! mechanisms the exogenous posterior is exact, so there is no parameter
//! uncertainty a posterior could describe.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CalibrationBasis, ExecutionContext};
use antecedent_io::calibration::{
    CalibrationBasisWire, CalibrationInference, CalibrationStructure, calibration_slot,
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
/// Most variables the whole-path coverage records measured.
///
/// This is the measured model class of the records `cov.counterfactual.*.cross_world_interval_*`
/// (`parity/coverage_records.toml`). The calibration key carries no variable-count axis, so a
/// larger model is reported with the status `outside_measured_scope`, never withheld.
const MEASURED_VARIABLES: usize = 3;
/// Most levels per variable the whole-path coverage records measured (see
/// [`MEASURED_VARIABLES`]).
const MEASURED_LEVELS: usize = 4;

super::string_enum! {
    /// What closed the whole-path interval route, named on the refusal as its `offending` part.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum RouteClosure, lookup = from_name {
        /// The abduction is sampled, not exact enumeration.
        Exogenous => "exogenous",
        /// The target population is not the population the mechanisms were declared for.
        TargetPopulation => "target_population",
        /// More variables or levels than the coverage records measured.
        Scm => "scm",
        /// An exogenous term is shared by several variables.
        SharedExogenous => "shared_exogenous",
        /// The declared unit sampling determines no variance for these units.
        UnitSampling => "unit_sampling",
    }
}

super::string_enum! {
    /// The scalar a whole-path interval is a statement about.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum IntervalScalar, lookup = from_name {
        /// Mean of the realized counterfactual outcomes of the fixed units.
        RealizedCounterfactualMean => "realized_counterfactual_mean",
        /// Population mean of the counterfactual probability.
        PopulationMeanCounterfactualProbability => "population_mean_counterfactual_probability",
    }
}

fn refused(
    code: &str,
    detail: &str,
    offending: &str,
    remedy: &str,
) -> CrossWorldExecutionRefusalWire {
    CrossWorldExecutionRefusalWire {
        code: code.to_owned(),
        stage: "uncertainty".to_owned(),
        detail: detail.to_owned(),
        offending: Some(offending.to_owned()),
        remedy: Some(remedy.to_owned()),
        missing: Vec::new(),
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
            detail: antecedent_core::reason_details::cross_world_execution::CANCELLED.to_owned(),
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
/// A scientific refusal for a request with no whole-path derivation: a sampled
/// posterior (exact enumeration is available, so sampling the abduction only adds
/// Monte Carlo error to an exactly computable quantity) or a transported
/// population (no theorem composes the transport identification with a
/// target-population abduction). A model larger than the measured designs is not
/// refused: its slot reports the scope.
#[allow(clippy::result_large_err, reason = "the refusal wire is the shared execution refusal type")]
pub fn whole_path_interval_basis(
    request: &FiniteExecutionRequest,
    unit_sampling: UnitSampling,
) -> Result<CalibrationBasis, CrossWorldExecutionRefusalWire> {
    if request.exogenous != ExogenousInference::ExactEnumeration {
        return Err(refused(
            antecedent_core::reason_code!("estimator_inference_mismatch"),
            antecedent_core::reason_details::cross_world_execution::NO_INTERVAL_METHOD,
            RouteClosure::Exogenous.as_str(),
            "with declared mechanisms the exogenous posterior is enumerated exactly; a sampled abduction adds Monte Carlo error to an exactly computable quantity, so request exact enumeration",
        ));
    }
    if request.target_population.as_ref().is_some_and(|t| t != &request.scm.population) {
        return Err(refused(
            antecedent_core::reason_code!("transport_not_certified"),
            antecedent_core::reason_details::cross_world_execution::NO_INTERVAL_METHOD,
            RouteClosure::TargetPopulation.as_str(),
            "no theorem composes the transport identification with a target-population abduction; derive the interval in the population the mechanisms were declared for",
        ));
    }
    // The measured designs drew one exogenous term per variable, at most
    // `MEASURED_VARIABLES` variables of at most `MEASURED_LEVELS` levels. A larger model
    // or a shared term is outside that scope: it gets the slot's scope status, and the
    // key carries the model-class label so no record is matched across classes.
    let measured_class = request.scm.variables.len() <= MEASURED_VARIABLES
        && request.scm.variables.iter().all(|v| v.levels <= MEASURED_LEVELS)
        && {
            let mut users = vec![0usize; request.scm.exogenous.len()];
            for variable in &request.scm.variables {
                if let Some(count) = users.get_mut(variable.exogenous) {
                    *count += 1;
                }
            }
            users.iter().all(|count| *count <= 1)
        };
    let a = |text: &str| Arc::<str>::from(text);
    Ok(CalibrationBasis::new(
        [
            a(CROSS_WORLD_QUERY),
            a("Dag"),
            a(CalibrationStructure::Fixed.as_str()),
            a("tabular"),
            a(CalibrationInference::Frequentist.as_str()),
            a(CROSS_WORLD_ESTIMATOR),
            a(CROSS_WORLD_CONSTRUCTION),
            a(""),
            a(unit_sampling.name()),
            a(""),
            a(&if measured_class {
                quantity_label(&request.quantity).to_owned()
            } else {
                format!("{}.not_measured_model_class", quantity_label(&request.quantity))
            }),
        ],
        CROSS_WORLD_LEVEL,
        a("point"),
        request.units.len() as u64,
        None,
        None,
        0.0,
    )
    .with_validation("not_applicable"))
}

/// The slot of a whole-path interval: what the coverage records give its basis, nothing set
/// by the route itself.
fn whole_path_interval_slot(
    basis: &CalibrationBasis,
) -> antecedent_io::contract_section::CalibrationSlotWire {
    calibration_slot(&CalibrationBasisWire::from(basis))
}

/// The point, the interval and the scalar name a derived account reports.
///
/// `None` when the declared sampling has no determined variance (a single
/// unit under independent sampling) or the level is invalid.
#[must_use]
pub fn whole_path_interval(
    uncertainty: &WholePathUncertainty,
    level: f64,
) -> Option<(f64, (f64, f64), IntervalScalar)> {
    if !(level > 0.0 && level < 1.0) {
        return None;
    }
    #[allow(
        clippy::match_wildcard_for_single_variants,
        reason = "an undeclared sampling carries no interval"
    )]
    let (variance, scalar) = match uncertainty.unit_sampling {
        UnitSampling::FixedUnits => (
            uncertainty.realized_mean_predictive_variance,
            IntervalScalar::RealizedCounterfactualMean,
        ),
        UnitSampling::IndependentSample => (
            uncertainty.unit_mean_sampling_variance?,
            IntervalScalar::PopulationMeanCounterfactualProbability,
        ),
        _ => return None,
    };
    if !(variance.is_finite() && variance >= 0.0 && uncertainty.value.is_finite()) {
        return None;
    }
    // An independent sample estimates the unit variance from the units themselves, so the
    // standardized mean follows a t law on `n - 1` degrees of freedom; the fixed-units
    // predictive variance is a declared model quantity and keeps the normal quantile.
    let quantile = match uncertainty.unit_sampling {
        UnitSampling::IndependentSample if uncertainty.units.len() >= 2 => {
            #[allow(clippy::cast_precision_loss, reason = "unit counts are small")]
            let degrees = (uncertainty.units.len() - 1) as f64;
            antecedent_stats::special::student_t_ppf(0.5 + level / 2.0, degrees)
        }
        _ => antecedent_stats::special::normal_ppf(0.5 + level / 2.0),
    };
    let half = quantile * variance.sqrt();
    Some((uncertainty.value, (uncertainty.value - half, uncertainty.value + half), scalar))
}

/// A whole-path interval bound to its governing coverage record.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossWorldInterval {
    /// The reported scalar.
    pub scalar: IntervalScalar,
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
    /// Identity of the governing coverage record, when one matched.
    pub record_id: Option<String>,
    /// Calibration of this interval: `calibrated` inside a current record's measured scope,
    /// otherwise the status and reason that name why not.
    pub calibration: antecedent_io::contract_section::CalibrationSlotWire,
    /// Always `measured_interval_under_the_declared_model`.
    pub label: &'static str,
}

impl CrossWorldExecution {
    /// The interval of the executed quantity under a declared unit sampling, with
    /// the calibration slot of the whole-path coverage records.
    ///
    /// The unqualified [`Self::interval`] stays closed; this reading names the
    /// declared unit sampling and is a statement conditional on the declared
    /// mechanisms. A posterior reading and a sampled abduction stay refused.
    ///
    /// # Errors
    /// Undeclared unit sampling or any refusal of
    /// [`Self::whole_path_uncertainty`]; a sampled or transported request; or a
    /// single unit under independent sampling.
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
                antecedent_core::reason_code!("uncertainty_representation_insufficient"),
                antecedent_core::reason_details::cross_world_execution::NO_INTERVAL_METHOD,
                RouteClosure::UnitSampling.as_str(),
                "the declared sampling determines no variance for these units (an independent sample needs at least two units); use the point value",
            ));
        };
        let calibration = whole_path_interval_slot(&basis);
        Ok(CrossWorldInterval {
            scalar,
            point,
            lower,
            upper,
            level: CROSS_WORLD_LEVEL,
            unit_sampling: unit_sampling.name(),
            record_id: calibration.record_id.clone(),
            calibration,
            label: "measured_interval_under_the_declared_model",
        })
    }
}

#[cfg(test)]
mod standing_tests {
    use super::super::cross_world_execution::{
        ExogenousVariable, FactualUnit, FiniteScm, FiniteVariable, FiniteWorld, NoiseCoupling,
        WorldAtom,
    };
    use super::*;
    use antecedent_contract::RequestedReading;
    use antecedent_io::calibration::rederive_calibration;
    use antecedent_io::contract_section::CalibrationStatus;

    fn variable(
        name: &str,
        levels: usize,
        parents: Vec<usize>,
        exogenous: usize,
    ) -> FiniteVariable {
        let table = match parents.len() {
            0 => (0..u8::try_from(levels).expect("a few levels")).collect(),
            _ => vec![0, 0, 0, 1, 1, 0, 1, 1],
        };
        FiniteVariable { name: name.to_owned(), levels, parents, exogenous, table }
    }

    fn request(
        variables: Vec<FiniteVariable>,
        exogenous: ExogenousInference,
    ) -> FiniteExecutionRequest {
        let scm = FiniteScm {
            population: "pop-a".to_owned(),
            history: "regime-1".to_owned(),
            exogenous: vec![
                ExogenousVariable { name: "u_x".to_owned(), pmf: vec![0.6, 0.4] },
                ExogenousVariable { name: "u_y".to_owned(), pmf: vec![0.2, 0.5, 0.1, 0.2] },
            ],
            variables,
        };
        let world = |name: &str, level: u8| FiniteWorld {
            name: name.to_owned(),
            interventions: vec![(0, level)],
            routes: vec![],
        };
        FiniteExecutionRequest {
            evidence_history: scm.history.clone(),
            scm,
            units: vec![
                FactualUnit { unit: "case".to_owned(), observed: vec![(0, 1), (1, 1)] },
                FactualUnit { unit: "open".to_owned(), observed: vec![] },
            ],
            worlds: vec![world("x0", 0), world("x1", 1)],
            quantity: FiniteQuantity::Probability(WorldAtom { world: 0, variable: 1, level: 0 }),
            coupling: NoiseCoupling::SharedAbducedExogenous,
            exogenous,
            target_population: None,
            reading: RequestedReading::Point,
        }
    }

    fn measured() -> Vec<FiniteVariable> {
        vec![variable("x", 2, vec![], 0), variable("y", 2, vec![0], 1)]
    }

    #[test]
    fn site_whole_path_slot_is_keyed_by_the_route_and_set_only_by_records() {
        for sampling in [UnitSampling::IndependentSample, UnitSampling::FixedUnits] {
            let basis = whole_path_interval_basis(
                &request(measured(), ExogenousInference::ExactEnumeration),
                sampling,
            )
            .expect("a whole-path basis");
            let wire = CalibrationBasisWire::from(&basis);
            assert_eq!(wire.key.query, CROSS_WORLD_QUERY);
            assert_eq!(wire.key.estimator, CROSS_WORLD_ESTIMATOR);
            assert_eq!(wire.key.interval_method, CROSS_WORLD_CONSTRUCTION);
            assert_eq!(wire.key.dependence, sampling.name());
            assert_eq!(wire.key.functional, "counterfactual_probability");
            assert_eq!(wire.scope.row_count, 2);
            let slot = whole_path_interval_slot(&basis);
            assert_eq!(slot.basis.as_ref(), Some(&wire));
            let again = rederive_calibration(&slot);
            assert_eq!(again.status, slot.status);
            assert_eq!(again.record_id, slot.record_id);
            if slot.status == CalibrationStatus::Calibrated {
                assert!(slot.record_id.is_some());
            }
        }
    }

    #[test]
    fn site_an_unmeasured_model_class_never_borrows_a_record() {
        // A shared exogenous term is outside the measured class: its key names the class, so
        // no record of the measured class calibrates it.
        let shared = vec![variable("x", 2, vec![], 0), variable("y", 2, vec![0], 0)];
        let basis = whole_path_interval_basis(
            &request(shared, ExogenousInference::ExactEnumeration),
            UnitSampling::IndependentSample,
        )
        .expect("a basis outside the measured class");
        let wire = CalibrationBasisWire::from(&basis);
        assert_eq!(wire.key.functional, "counterfactual_probability.not_measured_model_class");
        assert_ne!(whole_path_interval_slot(&basis).status, CalibrationStatus::Calibrated);
        // A sampled abduction has no whole-path interval at all.
        assert!(
            whole_path_interval_basis(
                &request(measured(), ExogenousInference::PosteriorSampling),
                UnitSampling::IndependentSample,
            )
            .is_err()
        );
    }
}
