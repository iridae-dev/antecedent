//! Intervals read from joint-response results, each with the calibration key it is measured
//! under.
//!
//! A reading derives its key from the result alone: the procedure, the inference mode, the
//! interval construction, the number of independent units behind it and its resampling or
//! posterior size. The key is resolved against the shipped coverage records, so a reading
//! at another level, on another construction or outside a record's measured range reports
//! that it is not calibrated. The coverage harness binds the same key.
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Query axis of every joint-response coverage record.
pub const JOINT_RESPONSE_QUERY: &str = "JointResponse";

/// Fewest units per complete program at which the equal-weight randomized intervals and the
/// reference-decomposition regions were measured.
pub(super) const EQUAL_WEIGHT_UNITS_PER_PROGRAM: usize = 32;
/// Fewest units per complete program at which the unequal-weight randomized interval was
/// measured.
pub(super) const UNEQUAL_WEIGHT_UNITS_PER_PROGRAM: usize = 4;
/// Fewest units per complete program at which the updated posterior interval was measured.
const UPDATE_UNITS_PER_PROGRAM: usize = 16;

/// An interval or simultaneous region with its calibration key and that key's standing.
#[derive(Clone, Debug, PartialEq)]
pub struct JointResponseReading<T> {
    /// Lower and upper bounds.
    pub bounds: T,
    /// Key the reading is measured under.
    pub basis: antecedent_core::CalibrationBasis,
    /// Standing of the key against the shipped coverage records.
    pub calibration: antecedent_io::contract_section::CalibrationSlotWire,
}

/// Construction axes of one joint-response reading.
pub(super) struct JointResponseAxes<'a> {
    pub(super) procedure: &'a str,
    pub(super) inference: &'a str,
    pub(super) estimator: &'a str,
    pub(super) interval_method: &'a str,
    pub(super) support: &'a str,
    pub(super) posterior: &'a str,
    pub(super) functional: &'a str,
}

/// Calibration key of a joint-response reading over `rows` independent units.
pub(super) fn joint_response_basis(
    axes: &JointResponseAxes<'_>,
    level: f64,
    rows: usize,
    replicates: Option<usize>,
    draws: Option<usize>,
) -> antecedent_core::CalibrationBasis {
    let text = |value: &str| std::sync::Arc::<str>::from(value);
    let count = |value: Option<usize>| value.map(|n| u32::try_from(n).unwrap_or(u32::MAX));
    antecedent_core::CalibrationBasis::new(
        [
            text(JOINT_RESPONSE_QUERY),
            text("NotApplicable"),
            text("design"),
            text(axes.procedure),
            text(axes.inference),
            text(axes.estimator),
            text(axes.interval_method),
            text(""),
            text(axes.support),
            text(axes.posterior),
            text(axes.functional),
        ],
        level,
        text("point"),
        rows as u64,
        count(replicates),
        count(draws),
        0.0,
    )
    .with_validation("not_applicable")
}

fn joint_response_reading<T>(
    bounds: T,
    basis: antecedent_core::CalibrationBasis,
) -> JointResponseReading<T> {
    let calibration = crate::analysis::joint_law_composed_basis::slot(&basis);
    JointResponseReading { bounds, basis, calibration }
}

/// The request cannot be read: a level outside (0, 1) or mismatched coefficients.
fn request_refusal(reason: &str) -> super::JointPanelError {
    super::JointPanelError::Compile(format!(
        "{} [{}]: {reason}",
        antecedent_core::reason_details::joint_response::READING_LEVEL_INVALID,
        antecedent_core::reason_code!("invalid_argument"),
    ))
}

/// The result retains nothing finite to build the interval from.
fn reading_refusal(reason: &str) -> super::JointPanelError {
    super::JointPanelError::Compile(format!(
        "{} [{}]: {reason}",
        antecedent_core::reason_details::joint_response::READING_LAW_UNAVAILABLE,
        antecedent_core::reason_code!("uncertainty_representation_insufficient"),
    ))
}

fn checked_level(level: f64) -> Result<(), super::JointPanelError> {
    if level.is_finite() && level > 0.0 && level < 1.0 {
        Ok(())
    } else {
        Err(request_refusal("an interval needs a level strictly inside (0, 1)"))
    }
}

/// Normal interval of a held-out value and its standard error.
fn heldout_normal_interval(
    heldout: &antecedent_design::program_search::selection::procedure::HeldoutProgramValue,
    level: f64,
) -> Result<[f64; 2], super::JointPanelError> {
    checked_level(level)?;
    let radius = antecedent_stats::normal_ppf(0.5 + 0.5 * level) * heldout.standard_error;
    if !(heldout.value.is_finite() && radius.is_finite() && radius >= 0.0) {
        return Err(reading_refusal("the held-out value or its standard error is not finite"));
    }
    Ok([heldout.value - radius, heldout.value + radius])
}

/// Equal-tailed bounds of sorted draws; each tail must hold at least one draw.
fn equal_tailed_bounds(
    mut values: Vec<f64>,
    level: f64,
) -> Result<[f64; 2], super::JointPanelError> {
    let n = values.len();
    let tail = 0.5 * (1.0 - level) * n as f64;
    let cut = (1..=n).take_while(|k| *k as f64 <= tail).count();
    if cut == 0 || values.iter().any(|value| !value.is_finite()) {
        return Err(reading_refusal(
            "too few finite retained draws to resolve the requested level",
        ));
    }
    values.sort_by(f64::total_cmp);
    Ok([values[cut], values[n - 1 - cut]])
}

impl super::selection::RandomizedSelectionResult {
    /// Normal interval at `level` for the value of the selected program, from its held-out
    /// estimate and standard error. The key counts the smaller of the training and held-out
    /// samples.
    /// # Errors
    /// A level outside (0, 1) or a non-finite held-out estimate.
    pub fn selected_value_reading(
        &self,
        level: f64,
    ) -> Result<JointResponseReading<[f64; 2]>, super::JointPanelError> {
        let procedure = self.procedure();
        let bounds = heldout_normal_interval(&procedure.observed.heldout, level)?;
        let rows = procedure.partition.training.len().min(procedure.partition.validation.len());
        let basis = joint_response_basis(
            &JointResponseAxes {
                procedure: "heldout_program_selection",
                inference: "Frequentist",
                estimator: "joint_design.whole_program_ht",
                interval_method: "heldout_normal_approximation",
                support: "independent_heldout_units",
                posterior: "",
                functional: "selected_program_value",
            },
            level,
            rows,
            None,
            None,
        );
        Ok(joint_response_reading(bounds, basis))
    }
}

impl super::selection::instrumented::InstrumentedSelectionResult {
    /// Normal interval at `level` for the structural value of the selected dose program, from
    /// its held-out estimate and standard error. The key counts the smaller of the training and
    /// held-out samples.
    /// # Errors
    /// A level outside (0, 1) or a non-finite held-out estimate.
    pub fn selected_value_reading(
        &self,
        level: f64,
    ) -> Result<JointResponseReading<[f64; 2]>, super::JointPanelError> {
        let procedure = &self.procedure;
        let bounds = heldout_normal_interval(&procedure.observed.heldout, level)?;
        let rows = procedure.partition.training.len().min(procedure.partition.validation.len());
        let basis = joint_response_basis(
            &JointResponseAxes {
                procedure: "heldout_program_selection",
                inference: "Frequentist",
                estimator: "joint_design.structural_feature_iv",
                interval_method: "heldout_normal_approximation",
                support: "independent_heldout_units",
                posterior: "",
                functional: "selected_structural_increment",
            },
            level,
            rows,
            None,
            None,
        );
        Ok(joint_response_reading(bounds, basis))
    }
}

/// Units and complete programs of a whole-program randomized design with equal weights.
fn equal_weight_randomized_support(
    result: &super::designs::JointDesignResult,
) -> Option<(usize, usize)> {
    let antecedent_estimate::joint_design::compiled::JointDesignRecipe::Randomized {
        design,
        units,
    } = &result.prepared().proof().recipe
    else {
        return None;
    };
    let equal = units
        .windows(2)
        .all(|pair| pair[0].population_weight.to_bits() == pair[1].population_weight.to_bits());
    equal.then_some((units.len(), design.worlds.len()))
}

impl super::designs::JointDesignResult {
    /// Simultaneous region at joint `level` for the four Shapley contributions of a population
    /// reference decomposition computed from this result.
    ///
    /// Each contribution takes a Bonferroni share of the error: a normal interval of the
    /// propagated covariance in frequentist mode, the equal-tailed interval of the aligned
    /// posterior draws in Bayesian mode. The region is conservative by the union bound.
    /// # Errors
    /// A level outside (0, 1), a decomposition without the covariance or draws its mode
    /// reads, or too few draws to resolve the level.
    pub fn population_reference_region(
        &self,
        decomposition: &super::PopulationReferenceDecomposition,
        level: f64,
    ) -> Result<JointResponseReading<[[f64; 2]; 4]>, super::JointPanelError> {
        checked_level(level)?;
        let component_level = 1.0 - (1.0 - level) / 4.0;
        let bayesian = matches!(
            self.law(),
            antecedent_estimate::joint_design::compiled::JointDesignLaw::Bayesian(_)
        );
        let mut bounds = [[0.0_f64; 2]; 4];
        let draws = decomposition.decomposition.draws.iter().flatten().count();
        if bayesian {
            for (component, bound) in bounds.iter_mut().enumerate() {
                let values = decomposition
                    .decomposition
                    .draws
                    .iter()
                    .flatten()
                    .map(|draw| draw.shapley[component])
                    .collect::<Vec<_>>();
                *bound = equal_tailed_bounds(values, component_level)?;
            }
        } else {
            let covariance = decomposition.covariance.as_ref().ok_or_else(|| {
                reading_refusal("the decomposition carries no propagated joint covariance")
            })?;
            let quantile = antecedent_stats::normal_ppf(0.5 + 0.5 * component_level);
            for (component, bound) in bounds.iter_mut().enumerate() {
                let variance = covariance.shapley[component][component];
                let point = decomposition.decomposition.point.shapley[component];
                let radius = quantile * variance.max(0.0).sqrt();
                if !(point.is_finite() && radius.is_finite()) {
                    return Err(reading_refusal(
                        "a Shapley contribution or its variance is not finite",
                    ));
                }
                *bound = [point - radius, point + radius];
            }
        }
        let (rows, support) = match equal_weight_randomized_support(self) {
            Some((units, programs)) if units >= EQUAL_WEIGHT_UNITS_PER_PROGRAM * programs => {
                (units, "equal_weights_32_units_per_program")
            }
            Some((units, _)) => (units, "equal_weights_sparse_programs"),
            None => (0, "other_source_design"),
        };
        let basis = joint_response_basis(
            &JointResponseAxes {
                procedure: "population_reference_decomposition",
                inference: if bayesian { "Bayesian" } else { "Frequentist" },
                estimator: "joint_design.whole_program_ht",
                interval_method: if bayesian {
                    "bonferroni_equal_tailed_posterior"
                } else {
                    "bonferroni_normal_approximation"
                },
                support,
                posterior: if bayesian { "randomized_gaussian" } else { "" },
                functional: "shapley_contributions_simultaneous",
            },
            level,
            rows,
            None,
            bayesian.then_some(draws),
        );
        Ok(joint_response_reading(bounds, basis))
    }

    /// Equal-tailed interval at `level` for a contrast of the target worlds under an
    /// experiment-conditioned law computed from this result's posterior.
    ///
    /// The interval reads the weighted draws of the updated law. Its key counts the units of
    /// this source design and the retained posterior draws.
    /// # Errors
    /// A level outside (0, 1), coefficients that do not match the law's worlds, an unweighted
    /// or non-finite law, too little weight in a tail to resolve the level, or evidence the
    /// ledger has withdrawn or changed.
    pub fn updated_contrast_reading(
        &self,
        updated: &super::JointLawExperimentResult,
        ledger: &antecedent_contract::evidence_ledger::EvidenceLedger,
        coefficients: &[f64],
        level: f64,
    ) -> Result<JointResponseReading<[f64; 2]>, super::JointPanelError> {
        checked_level(level)?;
        let law = updated.numerical_law_source(ledger)?;
        let width = coefficients.len();
        if width == 0
            || coefficients.iter().any(|c| !c.is_finite())
            || law.draws().len() % width != 0
        {
            return Err(request_refusal("contrast coefficients do not match the updated law"));
        }
        let values = law
            .draws()
            .chunks_exact(width)
            .map(|row| row.iter().zip(coefficients).map(|(v, c)| v * c).sum::<f64>())
            .collect::<Vec<_>>();
        let weights = law
            .metadata()
            .weights
            .as_ref()
            .filter(|weights| weights.len() == values.len())
            .ok_or_else(|| reading_refusal("the updated law carries no aligned weights"))?;
        let total = weights.iter().sum::<f64>();
        if !(total.is_finite() && total > 0.0) || values.iter().any(|v| !v.is_finite()) {
            return Err(reading_refusal("the updated law is not finite"));
        }
        let mut ordered = values.iter().copied().zip(weights.iter().copied()).collect::<Vec<_>>();
        ordered.sort_by(|a, b| a.0.total_cmp(&b.0));
        // Smallest value whose cumulative weight share reaches `p`.
        let quantile = |p: f64| {
            let mut mass = 0.0;
            for (value, weight) in &ordered {
                mass += weight / total;
                if mass >= p {
                    return *value;
                }
            }
            ordered[ordered.len() - 1].0
        };
        let bounds = [quantile(0.5 * (1.0 - level)), quantile(0.5 + 0.5 * level)];
        let bayesian = matches!(
            self.law(),
            antecedent_estimate::joint_design::compiled::JointDesignLaw::Bayesian(_)
        );
        let (rows, support) = match equal_weight_randomized_support(self) {
            Some((units, programs)) if bayesian && units >= UPDATE_UNITS_PER_PROGRAM * programs => {
                (units, "equal_weights_16_units_per_program")
            }
            Some((units, _)) => (units, "equal_weights_sparse_programs"),
            None => (0, "other_source_design"),
        };
        let basis = joint_response_basis(
            &JointResponseAxes {
                procedure: "experiment_update",
                inference: "Bayesian",
                estimator: "joint_design.whole_program_ht",
                interval_method: "weighted_equal_tailed_posterior",
                support,
                posterior: "randomized_gaussian_modular_importance",
                functional: "updated_world_contrast",
            },
            level,
            rows,
            None,
            Some(values.len()),
        );
        Ok(joint_response_reading(bounds, basis))
    }
}
