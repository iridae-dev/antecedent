//! Continuous selection evaluates the chosen joint structural-feature contrast
//! using an independently fitted validation sample, rather than another RNG seed.
// SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest, ValueExpr, VariableId};
use antecedent_design::program_search::{
    ProgramSearchError,
    certified::{
        CertificateOptions, CertificateStop, CertifiedControl, CertifiedRestriction,
        ObjectiveInterval, PreparedCertifiedSearch,
    },
    selection::{
        ObservationalSpan,
        procedure::{
            HeldoutProgramValue, ProgramSelectionProcedure, RankedProgram, SelectionDataBlock,
            WholeProcedureSelection, execute_heldout_selection,
        },
    },
};
use antecedent_estimate::joint_design::instrumented::{
    JointInstrumentedDesign, JointInstrumentedModelFit, JointInstrumentedUnit,
    JointStructuralFeatureBinding, fit_instrumented_joint_polynomial,
};
use serde::{Deserialize, Serialize};

/// One raw-dose coordinate of the complete joint intervention.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstrumentedDoseControl {
    /// Inclusive finite raw-dose lower bound.
    pub lower: f64,
    /// Inclusive finite raw-dose upper bound.
    pub upper: f64,
    /// Nonnegative operational resource per dose unit.
    pub unit_cost: f64,
    /// Linear utility cost per dose unit, separate from feasibility.
    pub utility_cost: f64,
}
/// Candidate-domain generation, repeated inside every complete procedure.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum InstrumentedDomainGeneration {
    /// The complete prespecified bounded dose domain.
    Prespecified,
    /// Intersect prespecified bounds with the training sample's observed raw-dose
    /// range plus the declared margin. Validation doses never enter this step.
    TrainingRange {
        /// Nonnegative finite margin in raw-dose units.
        margin: f64,
    },
}
/// Complete retained continuous selection procedure and original data.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstrumentedSelectionRequest {
    /// Genuine homogeneous structural moment model and explicit population.
    pub design: JointInstrumentedDesign,
    /// Actual raw-dose/structural-feature binding verified against each observation.
    pub features: JointStructuralFeatureBinding,
    /// Original independent training units, including instruments and outcomes.
    pub training: Vec<JointInstrumentedUnit>,
    /// Independently reserved original evaluation units from the same moment population.
    pub validation: Vec<JointInstrumentedUnit>,
    /// Explicit common independent-unit sampling frame.
    pub independent_sampling_frame: String,
    /// One bounded control per raw dose, in the feature binding's raw-axis order.
    pub controls: Vec<InstrumentedDoseControl>,
    /// All extra hard linear restrictions in control order.
    pub restrictions: Vec<CertifiedRestriction>,
    /// Optional inclusive operational resource budget.
    pub budget: Option<f64>,
    /// Prespecified raw-dose reference program. The target is improvement relative
    /// to this program, so original-population exogenous means cancel exactly.
    pub reference: Vec<f64>,
    /// Initial program, required to remain feasible after candidate generation.
    pub initial: Vec<f64>,
    /// Finite utility multiplier on the actual structural outcome contrast.
    pub benefit_multiplier: f64,
    /// Utility scale shared by all stages.
    pub utility_units: String,
    /// Prespecified or genuinely training-dependent generation.
    pub generation: InstrumentedDomainGeneration,
    /// Deterministic training search budget and tolerance.
    pub certificate: CertificateOptions,
}
/// Actual continuous selected action, including the training search's real gap.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstrumentedSelectedAction {
    /// Complete raw-dose program.
    pub controls: Vec<f64>,
    /// Bound for the actual fitted training utility, not selected-value uncertainty.
    pub training_global_upper: f64,
    /// Certified training objective gap, retained even at resource exhaustion.
    pub training_gap: f64,
    /// Actual bounded-search stopping reason.
    pub training_stop: CertificateStop,
}
/// An executed original-observation split and complete resampling receipt.
#[derive(Debug)]
pub struct InstrumentedSelectionResult {
    /// Selected action, held-out estimate and every bootstrap slot/failure.
    pub procedure: WholeProcedureSelection<InstrumentedSelectedAction>,
    /// Full data, feature, moment, population and algorithm identity.
    pub identity: SemanticDigest,
    request: InstrumentedSelectionRequest,
    seed: u64,
    repetitions: usize,
    tie_tolerance: f64,
}
fn invalid(e: impl std::fmt::Display) -> ProgramSearchError {
    ProgramSearchError::Invalid(e.to_string())
}
fn sum(mut values: Vec<ValueExpr>) -> ValueExpr {
    if values.is_empty() {
        return ValueExpr::constant(0.0);
    }
    while values.len() > 1 {
        let mut next = Vec::with_capacity(values.len().div_ceil(2));
        let mut iter = values.into_iter();
        while let Some(a) = iter.next() {
            next.push(if let Some(b) = iter.next() { a.add(b) } else { a });
        }
        values = next;
    }
    values.pop().unwrap_or_else(|| ValueExpr::constant(0.0))
}
fn feature(powers: &[u32], doses: &[f64]) -> f64 {
    powers
        .iter()
        .zip(doses)
        .map(|(power, value)| value.powi(i32::try_from(*power).unwrap_or(9)))
        .product()
}
impl InstrumentedSelectionRequest {
    fn validate(&self) -> Result<(), ProgramSearchError> {
        let d = self.features.raw_feature_indices.len();
        if d == 0
            || d > 16
            || self.controls.len() != d
            || self.reference.len() != d
            || self.initial.len() != d
            || self.reference.iter().chain(&self.initial).any(|x| !x.is_finite())
            || self.independent_sampling_frame.trim().is_empty()
            || self.utility_units.trim().is_empty()
            || !self.benefit_multiplier.is_finite()
            || self.training.is_empty()
            || self.validation.is_empty()
            || self.controls.iter().any(|c| {
                !c.lower.is_finite()
                    || !c.upper.is_finite()
                    || c.lower < 0.0
                    || c.upper < c.lower
                    || !c.unit_cost.is_finite()
                    || c.unit_cost < 0.0
                    || !c.utility_cost.is_finite()
            })
            || self.budget.is_some_and(|b| !b.is_finite() || b < 0.0)
            || self.features.powers.iter().any(|p| p.len() != d)
            || matches!(self.generation,InstrumentedDomainGeneration::TrainingRange { margin } if !margin.is_finite()||margin<0.0)
        {
            return Err(invalid("invalid complete structural-dose selection request"));
        }
        let mut training = std::collections::BTreeSet::new();
        let mut validation = std::collections::BTreeSet::new();
        if self.training.iter().any(|u| u.id.is_empty() || !training.insert(&u.id))
            || self
                .validation
                .iter()
                .any(|u| u.id.is_empty() || training.contains(&u.id) || !validation.insert(&u.id))
        {
            return Err(invalid(
                "structural selection requires distinct original training and held-out units",
            ));
        }
        Ok(())
    }
    /// Execute native joint-feature IV fitting, compiler-derived continuous search,
    /// and independent observational evaluation of the chosen structural contrast.
    /// Every resample repeats the complete same method-owner procedure. Its
    /// delete-unit uncertainty is asymptotic under the explicit homogeneous moment
    /// model, not a posterior, weak-IV guarantee or release calibration license.
    /// # Errors
    /// Invalid split/target, failed rank or feature premises, infeasible domains,
    /// unsupported certificate expressions, cancellation or resources.
    pub fn execute(
        &self,
        repetitions: usize,
        tie_tolerance: f64,
        ctx: &ExecutionContext,
    ) -> Result<InstrumentedSelectionResult, ProgramSearchError> {
        if ctx.cancellation.is_cancelled() {
            return Err(ProgramSearchError::Cancelled);
        }
        self.validate()?;
        let bytes = self
            .training
            .iter()
            .chain(&self.validation)
            .try_fold(0u64, |total, u| {
                let values = u
                    .endogenous
                    .len()
                    .checked_add(u.instruments.len())?
                    .checked_add(u.exogenous.len())?;
                let row = u64::try_from(values)
                    .ok()?
                    .checked_mul(64)?
                    .checked_add(u64::try_from(u.id.len()).ok()?.checked_mul(8)?)?
                    .checked_add(4096)?;
                total.checked_add(row)
            })
            .ok_or_else(|| invalid("structural selection admission overflow"))?;
        let _data = ctx
            .acquire_admission(0, bytes)
            .map_err(|e| ProgramSearchError::Resource(e.to_string()))?;
        fn blocks(
            units: &[JointInstrumentedUnit],
        ) -> Vec<SelectionDataBlock<'_, JointInstrumentedUnit>> {
            units
                .iter()
                .map(|u| SelectionDataBlock {
                    span: ObservationalSpan { unit: u.id.clone(), first: 0, last: 0 },
                    observations: std::borrow::Cow::Borrowed(u),
                })
                .collect()
        }
        let procedure = execute_heldout_selection(
            self,
            &blocks(&self.training),
            &blocks(&self.validation),
            0,
            repetitions,
            tie_tolerance,
            ctx,
        )?;
        let encoded = antecedent_io::convert::to_cbor(self).map_err(super::codec)?;
        let identity = SemanticDigest::from_bytes(CanonicalWriter::digest_bytes(
            "antecedent.structural_dose_selection.v1",
            &encoded,
        ));
        Ok(InstrumentedSelectionResult {
            procedure,
            identity,
            request: self.clone(),
            seed: ctx.rng.master_seed(),
            repetitions,
            tie_tolerance,
        })
    }
    fn fit_view(
        &self,
        data: &[&JointInstrumentedUnit],
        ctx: &ExecutionContext,
    ) -> Result<JointInstrumentedModelFit, ProgramSearchError> {
        let copies = data
            .iter()
            .enumerate()
            .map(|(slot, u)| {
                let mut copy = (*u).clone();
                copy.id = format!("{}:bootstrap_occurrence:{slot}", u.id);
                copy
            })
            .collect::<Vec<_>>();
        fit_instrumented_joint_polynomial(&self.design, &copies, self.features.clone(), ctx)
            .map_err(invalid)
    }
    fn value(
        &self,
        fit: &JointInstrumentedModelFit,
        doses: &[f64],
    ) -> Result<(f64, f64), ProgramSearchError> {
        let p = self.features.powers.len();
        let k = fit.coefficients.len();
        let delta = self
            .features
            .powers
            .iter()
            .map(|powers| {
                self.benefit_multiplier
                    * (feature(powers, doses) - feature(powers, &self.reference))
            })
            .collect::<Vec<_>>();
        let cost = self.controls.iter().zip(doses).map(|(c, d)| c.utility_cost * d).sum::<f64>();
        let value = delta.iter().zip(&fit.coefficients).map(|(d, b)| d * b).sum::<f64>() - cost;
        let variance = (0..p)
            .flat_map(|j| (0..p).map(move |l| (j, l)))
            .map(|(j, l)| delta[j] * delta[l] * fit.covariance[j * k + l])
            .sum::<f64>();
        if !value.is_finite() || !variance.is_finite() || variance < -1e-10 {
            return Err(invalid("selected structural contrast arithmetic failed"));
        }
        Ok((value, variance.max(0.0)))
    }
}
impl ProgramSelectionProcedure<JointInstrumentedUnit> for InstrumentedSelectionRequest {
    type Candidates = Vec<ObjectiveInterval>;
    type Fit = JointInstrumentedModelFit;
    type Action = InstrumentedSelectedAction;
    fn generate(
        &self,
        data: &[&JointInstrumentedUnit],
        _: usize,
        _: &ExecutionContext,
    ) -> Result<Self::Candidates, ProgramSearchError> {
        self.controls
            .iter()
            .enumerate()
            .map(|(axis, c)| {
                let mut bounds = ObjectiveInterval { lower: c.lower, upper: c.upper };
                if let InstrumentedDomainGeneration::TrainingRange { margin } = self.generation {
                    let index = self.features.raw_feature_indices[axis];
                    let values = data
                        .iter()
                        .map(|u| {
                            u.endogenous
                                .get(index)
                                .copied()
                                .ok_or_else(|| invalid("missing raw dose feature"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    bounds.lower = bounds
                        .lower
                        .max(values.iter().copied().fold(f64::INFINITY, f64::min) - margin);
                    bounds.upper = bounds
                        .upper
                        .min(values.iter().copied().fold(f64::NEG_INFINITY, f64::max) + margin);
                }
                if bounds.lower > bounds.upper {
                    return Err(invalid("training-generated raw-dose domain is empty"));
                }
                Ok(bounds)
            })
            .collect()
    }
    fn fit(
        &self,
        data: &[&JointInstrumentedUnit],
        _: &Self::Candidates,
        _: usize,
        ctx: &ExecutionContext,
    ) -> Result<Self::Fit, ProgramSearchError> {
        self.fit_view(data, ctx)
    }
    fn feasible(
        &self,
        _: &Self::Fit,
        bounds: Self::Candidates,
        _: &ExecutionContext,
    ) -> Result<Self::Candidates, ProgramSearchError> {
        Ok(bounds)
    }
    fn search(
        &self,
        fit: &Self::Fit,
        bounds: Self::Candidates,
        _: usize,
        ctx: &ExecutionContext,
    ) -> Result<Vec<RankedProgram<Self::Action>>, ProgramSearchError> {
        let variables = (0..bounds.len())
            .map(|i| {
                u32::try_from(i)
                    .map(VariableId::from_raw)
                    .map_err(|e| ProgramSearchError::upstream("raw-dose feature index", e))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut terms = Vec::new();
        for (powers, beta) in self.features.powers.iter().zip(&fit.coefficients) {
            let mut term = ValueExpr::constant(1.0);
            for (v, power) in variables.iter().zip(powers) {
                for _ in 0..*power {
                    term = term.mul(ValueExpr::var(*v));
                }
            }
            terms.push(
                ValueExpr::constant(self.benefit_multiplier * beta)
                    .mul(term.sub(ValueExpr::constant(feature(powers, &self.reference)))),
            );
        }
        for (variable, c) in variables.iter().zip(&self.controls) {
            terms.push(ValueExpr::constant(-c.utility_cost).mul(ValueExpr::var(*variable)));
        }
        let controls = variables
            .iter()
            .zip(bounds)
            .map(|(v, bounds)| CertifiedControl { variable: *v, bounds })
            .collect();
        let mut restrictions = self.restrictions.clone();
        if let Some(budget) = self.budget {
            restrictions.push(CertifiedRestriction {
                coefficients: self.controls.iter().map(|c| c.unit_cost).collect(),
                bound: budget,
            });
        }
        let plan = PreparedCertifiedSearch::compile(sum(terms), controls, restrictions)?;
        let c = plan.execute(self.initial.clone(), self.certificate, ctx)?;
        let value = self.value(fit, &c.controls)?.0;
        let points = c
            .controls
            .iter()
            .map(|d| ObjectiveInterval { lower: *d, upper: *d })
            .collect::<Vec<_>>();
        let interval = plan.bounds(&points)?;
        if value < interval.lower || value > interval.upper {
            return Err(invalid(
                "selected structural utility disagrees with its compiler-derived objective",
            ));
        }
        let bytes = antecedent_io::convert::to_cbor(&c.controls).map_err(super::codec)?;
        let id = SemanticDigest::from_bytes(CanonicalWriter::digest_bytes(
            "antecedent.selected_raw_doses.v1",
            &bytes,
        ))
        .to_hex();
        Ok(vec![RankedProgram {
            id,
            training_value: value,
            action: InstrumentedSelectedAction {
                controls: c.controls,
                training_global_upper: c.global_upper,
                training_gap: c.gap,
                training_stop: c.stop,
            },
        }])
    }
    fn evaluate(
        &self,
        data: &[&JointInstrumentedUnit],
        action: &Self::Action,
        _: usize,
        ctx: &ExecutionContext,
    ) -> Result<HeldoutProgramValue, ProgramSearchError> {
        let fit = self.fit_view(data, ctx)?;
        let (value, variance) = self.value(&fit, &action.controls)?;
        Ok(HeldoutProgramValue {
            target: format!(
                "{}:{}:{}:structural_increment",
                self.design.population, self.independent_sampling_frame, self.utility_units
            ),
            value,
            standard_error: variance.sqrt(),
        })
    }
}
impl InstrumentedSelectionResult {
    /// Selected controls, actual training bound, held-out contrast and full resampling evidence.
    #[must_use]
    pub fn report(&self) -> serde_json::Value {
        let p = &self.procedure;
        serde_json::json!({"identity":self.identity.to_hex(),"selected":p.observed.action,
            "training_value":p.observed.training_value,"heldout":p.observed.heldout,
            "partition":p.partition,"repetitions":p.repetitions,"mean_optimism":p.mean_optimism,
            "changed_selection":p.changed_selection,"calibration":"unmeasured",
            "uncertainty":"independent_heldout_full_joint_iv_delete_unit_contrast_covariance",
            "target":"homogeneous_structural_moment_model_population_increment"})
    }
    /// Replayable original observations, method specification, algorithm and computed receipt.
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, ProgramSearchError> {
        antecedent_io::convert::to_cbor(&Document {
            version: 1,
            request: self.request.clone(),
            seed: self.seed,
            repetitions: self.repetitions,
            tie_tolerance: self.tie_tolerance,
            report: self.report(),
        })
        .map_err(super::codec)
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    version: u32,
    request: InstrumentedSelectionRequest,
    seed: u64,
    repetitions: usize,
    tie_tolerance: f64,
    report: serde_json::Value,
}
/// Repeat native fitting, domain generation, certified search and held-out evaluation.
/// # Errors
/// Changed current request, forged report, malformed artifact or execution refusal.
pub fn replay_instrumented_selection(
    bytes: &[u8],
    expected: Option<&InstrumentedSelectionRequest>,
    ctx: &ExecutionContext,
) -> Result<InstrumentedSelectionResult, ProgramSearchError> {
    if ctx.cancellation.is_cancelled() {
        return Err(ProgramSearchError::Cancelled);
    }
    if bytes.len() > 32 * 1024 * 1024 {
        return Err(invalid("structural selection artifact size limit"));
    }
    let _input = ctx
        .acquire_admission(0, (bytes.len() as u64).saturating_mul(16))
        .map_err(|e| ProgramSearchError::Resource(e.to_string()))?;
    let d: Document = antecedent_io::convert::from_cbor(bytes).map_err(super::codec)?;
    if d.version != 1 {
        return Err(invalid("unsupported structural selection artifact version"));
    }
    if let Some(expected) = expected {
        if antecedent_io::convert::to_cbor(expected).map_err(super::codec)?
            != antecedent_io::convert::to_cbor(&d.request).map_err(super::codec)?
        {
            return Err(invalid("current structural selection data/moments/domain changed"));
        }
    }
    let mut execution = ctx.clone();
    execution.rng = antecedent_core::execution::RngFactory::from_seed(d.seed);
    let result = d.request.execute(d.repetitions, d.tie_tolerance, &execution)?;
    if result.report() != d.report {
        return Err(invalid("structural selection replay mismatch"));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(seed: u64, noise: f64) -> InstrumentedSelectionRequest {
        let mut rng = antecedent_core::execution::CausalRng::from_seed(seed);
        let mut units = |stage: &str| {
            (0..64)
                .map(|i| {
                    let z = 2.0 * rng.next_f64() - 1.0;
                    let disturbance = noise * (2.0 * rng.next_f64() - 1.0);
                    let dose = 1.5 + z + 0.1 * disturbance;
                    JointInstrumentedUnit {
                        id: format!("{stage}:{i}"),
                        endogenous: vec![dose, dose * dose],
                        instruments: vec![z, z * z],
                        exogenous: vec![1.0],
                        outcome: 4.0 * dose - dose * dose + disturbance,
                    }
                })
                .collect()
        };
        let training = units("training");
        let validation = units("heldout");
        InstrumentedSelectionRequest {
            design: JointInstrumentedDesign {
                contrasts: vec![("dose_increment".into(), vec![1.0, 1.0])],
                structural_model: "Y = beta1*dose + beta2*dose^2 + intercept + error".into(),
                moment_restrictions:
                    "independent uniform Z; E[Z error]=E[Z^2 error]=0; homogeneous coefficients"
                        .into(),
                population: "independent units from the declared common moment population".into(),
            },
            features: JointStructuralFeatureBinding {
                raw_feature_indices: vec![0],
                powers: vec![vec![1], vec![2]],
            },
            training,
            validation,
            independent_sampling_frame:
                "original independently sampled units reserved before fitting".into(),
            controls: vec![InstrumentedDoseControl {
                lower: 0.0,
                upper: 3.0,
                unit_cost: 1.0,
                utility_cost: 0.0,
            }],
            restrictions: vec![],
            budget: Some(1.75),
            reference: vec![0.0],
            initial: vec![1.0],
            benefit_multiplier: 1.0,
            utility_units: "outcome units".into(),
            generation: InstrumentedDomainGeneration::Prespecified,
            certificate: CertificateOptions { absolute_gap: 0.002, max_splits: 256 },
        }
    }

    #[test]
    fn native_continuous_selection_fits_joint_features_obeys_budget_and_replays() {
        let request = fixture(17, 0.0);
        let ctx = ExecutionContext::for_tests(31);
        let result = request.execute(3, 0.01, &ctx).unwrap();
        let selected = &result.procedure.observed;
        assert!((selected.action.controls[0] - 1.75).abs() < 0.002);
        assert!((selected.heldout.value - (7.0 - 1.75 * 1.75)).abs() < 0.005);
        assert!(selected.action.training_gap <= 0.002);
        assert_eq!(result.procedure.repetitions.len(), 3);
        let bytes = result.export().unwrap();
        assert_eq!(
            replay_instrumented_selection(&bytes, Some(&request), &ctx).unwrap().report(),
            result.report()
        );
        let mut changed = request.clone();
        changed.budget = Some(1.0);
        assert!(replay_instrumented_selection(&bytes, Some(&changed), &ctx).is_err());
        changed = request.clone();
        changed.validation[0].id = changed.training[0].id.clone();
        assert!(changed.execute(0, 0.0, &ctx).is_err());
        changed = request.clone();
        changed.training[0].endogenous[1] += 1.0;
        assert!(changed.execute(0, 0.0, &ctx).is_err());
    }

    #[test]
    fn continuous_selection_repeats_training_dependent_domain_generation() {
        let mut request = fixture(42, 0.3);
        request.generation = InstrumentedDomainGeneration::TrainingRange { margin: 0.0 };
        request.budget = None;
        let result = request.execute(4, 0.01, &ExecutionContext::for_tests(44)).unwrap();
        let selected = &result.procedure.observed;
        assert!((selected.action.controls[0] - 2.0).abs() < 0.2);
        assert!(selected.heldout.standard_error > 0.0);
        assert!(result.procedure.repetitions.iter().all(|r| r.result.is_some()));
        assert!(
            result.procedure.partition.training.iter().all(|s| s.unit.starts_with("training:"))
        );
        assert!(
            result.procedure.partition.validation.iter().all(|s| s.unit.starts_with("heldout:"))
        );
    }

    fn weak_decision_simulation(seed: u64, repetitions: usize) -> (bool, f64, f64) {
        let mut request = fixture(seed, 0.3);
        for unit in request.training.iter_mut().chain(&mut request.validation) {
            let dose = unit.endogenous[0];
            unit.outcome += -3.95 * dose + 0.99 * dose * dose;
        }
        request.budget = None;
        request.generation = InstrumentedDomainGeneration::TrainingRange { margin: 0.1 };
        request.certificate.absolute_gap = 0.001;
        let result =
            request.execute(repetitions, 0.01, &ExecutionContext::for_tests(seed)).unwrap();
        let selected = &result.procedure.observed;
        let dose = selected.action.controls[0];
        let truth = 0.05 * dose - 0.01 * dose * dose;
        let error = selected.heldout.value - truth;
        (
            error.abs() <= 1.959963984540054 * selected.heldout.standard_error,
            selected.training_value - truth,
            error,
        )
    }

    #[test]
    fn native_continuous_selected_values_use_independent_observations_across_datasets() {
        let outcomes = (0..400).map(|seed| weak_decision_simulation(seed, 0)).collect::<Vec<_>>();
        let covered = outcomes.iter().filter(|r| r.0).count();
        assert!(covered >= 350, "heldout selected structural coverage {covered}/400");
        let optimism = outcomes.iter().map(|r| r.1).sum::<f64>() / 400.0;
        let heldout_bias = outcomes.iter().map(|r| r.2).sum::<f64>() / 400.0;
        assert!(optimism > 0.002, "training optimism={optimism}");
        assert!(heldout_bias.abs() < 0.015, "heldout bias={heldout_bias}");
    }

    #[test]
    #[ignore = "Q08/E14: measure only on a stable candidate; implementation simulations grant no license"]
    fn whole_procedure_continuous_structural_selection_calibration_l95() {
        let n = std::env::var("ANTECEDENT_CALIBRATION_NSIM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(400)
            .max(400);
        let outcomes =
            (0..n).map(|seed| weak_decision_simulation(seed as u64, 10)).collect::<Vec<_>>();
        let covered = outcomes.iter().filter(|r| r.0).count();
        let rate = covered as f64 / n as f64;
        let mcse = (0.95 * 0.05 / n as f64).sqrt();
        println!(
            "continuous_structural_selection n={n} coverage={rate} optimism={} heldout_bias={}",
            outcomes.iter().map(|r| r.1).sum::<f64>() / n as f64,
            outcomes.iter().map(|r| r.2).sum::<f64>() / n as f64
        );
        assert!((rate - 0.95).abs() <= 3.0 * mcse + 0.005);
    }
}
