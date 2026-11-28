//! Source-target trial-level bridge discrepancy under a declared transport premise.
//!
//! Independent validation trials in both scopes fit the same surrogate-to-endpoint
//! line. Simultaneous Gaussian Wald bands describe its target-minus-source
//! difference on a prespecified surrogate grid. The bands and an equivalence
//! screen are model-conditional diagnostics, not proof of transportability or
//! an earned error-control claim. A bridge decision must retain its premise.

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_contract::DataClassification;
use antecedent_contract::bridge::{
    BridgeContract, BridgeMapping, BridgeRefusal, BridgeScope, TransportabilityPremise,
    TrialEffectPair, TrialLevelFit, TrialLevelValidation, fit_trial_level_linear,
};
use antecedent_core::{
    ByteEstimates, CalibrationBasis, CanonicalWriter, ExecutionContext, PhysicalEffect,
    PhysicalOperator, ScoreFrequentist, ScoreModel, SemanticDigest,
};
use antecedent_io::IoError;
use antecedent_kernels::norm_inv;
use serde::{Deserialize, Serialize};

use super::program_handle::{CompiledProgramHandle, gate_program_when};

use crate::planner::PlanView;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramCell, ProgramIdentityScope,
    ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};

const FEATURE: &str = "bridge_transport_premise_assessment_v1";
const MAX_ARTIFACT_BYTES: usize = 8 * 1024 * 1024;

/// Whether `execute` passes the support gate of the unit the route is bound to
/// (`parity/compiled_units.toml`). It stays off until that unit's record has earned its support,
/// so no valid use refuses before then; turning it on makes a release build refuse the route
/// while the unit is unsupported and a development build record what it would refuse.
const GATE_ON_UNIT_SUPPORT: bool = false;

/// Independent source and target validation trials of a proposed bridge
/// invariance premise. Grid points must lie in both observed surrogate ranges.
#[derive(Clone, Debug)]
pub struct BridgeTransportPremiseAssessmentRequest {
    /// Named source-to-target population and regime premise.
    pub premise: TransportabilityPremise,
    /// Source trial pairs used by the proposed bridge mapping.
    pub source: TrialLevelValidation,
    /// Independent target-scope validation trial pairs.
    pub target: TrialLevelValidation,
    /// Source trial identities aligned with source pairs.
    pub source_trial_ids: Vec<String>,
    /// Target trial identities aligned with target pairs.
    pub target_trial_ids: Vec<String>,
    /// Strictly increasing surrogate effects where the relation is assessed.
    pub surrogate_grid: Vec<f64>,
    /// Prespecified positive tolerance for target-minus-source terminal effects.
    pub equivalence_margin: f64,
}

/// One source-target discrepancy and its unmeasured simultaneous band.
#[derive(Clone, Debug, PartialEq)]
pub struct BridgeTransportDiscrepancy {
    /// Surrogate effect at which both fitted mappings are compared.
    pub surrogate_effect: f64,
    /// Target minus source predicted terminal effect.
    pub difference: f64,
    /// Plug-in standard error under independent Gaussian trial residuals.
    pub standard_error: f64,
    /// Lower simultaneous normal-Wald endpoint.
    pub lower: f64,
    /// Upper simultaneous normal-Wald endpoint.
    pub upper: f64,
}

/// Both fitted trial relations and the unearned equivalence diagnostic.
#[derive(Clone, Debug, PartialEq)]
pub struct BridgeTransportPremiseAssessmentResult {
    /// Source-scope trial regression.
    pub source_fit: TrialLevelFit,
    /// Target-scope trial regression.
    pub target_fit: TrialLevelFit,
    /// One diagnostic per prespecified surrogate grid point.
    pub grid: Vec<BridgeTransportDiscrepancy>,
    /// Every unmeasured simultaneous band lies within the declared margin.
    /// This is a screen, not a proof that the premise is true.
    pub equivalence_screen_passed: bool,
    /// No coverage or equivalence error control has been earned.
    pub calibration: &'static str,
    /// Complete compiler program identity.
    pub lowered_program_identity: SemanticDigest,
}

/// Source-only simultaneous discrepancy-band candidate and its exact design
/// coordinate. No coverage or equivalence error control is claimed.
#[derive(Clone, Debug)]
pub struct BridgeTransportPremiseCalibrationCandidate {
    /// Full validation result on the prespecified overlap grid.
    pub result: BridgeTransportPremiseAssessmentResult,
    /// Prospective whole-route calibration coordinate.
    pub basis: CalibrationBasis,
}

/// Refusal of the trial design, fit, compiler or replay.
#[derive(Debug, thiserror::Error)]
pub enum BridgeTransportPremiseAssessmentError {
    /// Invalid declaration or changed independent replay.
    #[error("bridge transport premise assessment: {0}")]
    Invalid(String),
    /// Artifact encoding or decoding failed.
    #[error("bridge transport premise artifact: {0}")]
    Io(#[from] IoError),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TrialValidationWire {
    trials: Vec<[u64; 2]>,
    min_trials: u32,
    min_r_squared: u64,
}

impl TrialValidationWire {
    fn from_domain(value: &TrialLevelValidation) -> Self {
        Self {
            trials: value
                .trials
                .iter()
                .map(|pair| [pair.surrogate_effect.to_bits(), pair.terminal_effect.to_bits()])
                .collect(),
            min_trials: value.min_trials,
            min_r_squared: value.min_r_squared.to_bits(),
        }
    }

    fn to_domain(&self) -> TrialLevelValidation {
        TrialLevelValidation {
            trials: self
                .trials
                .iter()
                .map(|bits| TrialEffectPair {
                    surrogate_effect: f64::from_bits(bits[0]),
                    terminal_effect: f64::from_bits(bits[1]),
                })
                .collect(),
            min_trials: self.min_trials,
            min_r_squared: f64::from_bits(self.min_r_squared),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AssessmentInputWire {
    premise_id: String,
    premise_statement: String,
    established_population: String,
    established_regime: String,
    target_population: String,
    target_regime: String,
    premise_evidence: Option<String>,
    source: TrialValidationWire,
    target: TrialValidationWire,
    source_trial_ids: Vec<String>,
    target_trial_ids: Vec<String>,
    surrogate_grid: Vec<u64>,
    equivalence_margin: u64,
}

impl AssessmentInputWire {
    fn from_domain(value: &BridgeTransportPremiseAssessmentRequest) -> Self {
        Self {
            premise_id: value.premise.id.clone(),
            premise_statement: value.premise.statement.clone(),
            established_population: value.premise.established_in.population_id.clone(),
            established_regime: value.premise.established_in.regime_id.clone(),
            target_population: value.premise.applies_to.population_id.clone(),
            target_regime: value.premise.applies_to.regime_id.clone(),
            premise_evidence: value.premise.evidence.clone(),
            source: TrialValidationWire::from_domain(&value.source),
            target: TrialValidationWire::from_domain(&value.target),
            source_trial_ids: value.source_trial_ids.clone(),
            target_trial_ids: value.target_trial_ids.clone(),
            surrogate_grid: value.surrogate_grid.iter().map(|x| x.to_bits()).collect(),
            equivalence_margin: value.equivalence_margin.to_bits(),
        }
    }

    fn to_domain(&self) -> BridgeTransportPremiseAssessmentRequest {
        BridgeTransportPremiseAssessmentRequest {
            premise: TransportabilityPremise {
                id: self.premise_id.clone(),
                statement: self.premise_statement.clone(),
                established_in: BridgeScope {
                    population_id: self.established_population.clone(),
                    regime_id: self.established_regime.clone(),
                },
                applies_to: BridgeScope {
                    population_id: self.target_population.clone(),
                    regime_id: self.target_regime.clone(),
                },
                evidence: self.premise_evidence.clone(),
            },
            source: self.source.to_domain(),
            target: self.target.to_domain(),
            source_trial_ids: self.source_trial_ids.clone(),
            target_trial_ids: self.target_trial_ids.clone(),
            surrogate_grid: self.surrogate_grid.iter().copied().map(f64::from_bits).collect(),
            equivalence_margin: f64::from_bits(self.equivalence_margin),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AssessmentArtifactWire {
    version: u32,
    required_feature: String,
    input: AssessmentInputWire,
    program_identity: [u8; 32],
    result_fingerprint: [u8; 32],
}

/// Source and target trial fits retained behind one six-layer compiler program.
#[derive(Clone, Debug)]
pub struct CompiledBridgeTransportPremiseAssessment {
    request: BridgeTransportPremiseAssessmentRequest,
    input: AssessmentInputWire,
    lowered: Arc<LoweredProgram>,
}

fn invalid(message: impl Into<String>) -> BridgeTransportPremiseAssessmentError {
    BridgeTransportPremiseAssessmentError::Invalid(message.into())
}

/// Refusal of a trial-level line fit, naming its reason code, detail and offending coordinate.
fn fit_refused(refusal: &BridgeRefusal) -> BridgeTransportPremiseAssessmentError {
    let mut message = format!("{}: {}", refusal.code, refusal.detail);
    if let Some(offending) = &refusal.offending {
        message.push_str(": ");
        message.push_str(offending);
    }
    invalid(message)
}

fn validate(
    request: &BridgeTransportPremiseAssessmentRequest,
) -> Result<(), BridgeTransportPremiseAssessmentError> {
    let premise = &request.premise;
    let mut ids = BTreeSet::new();
    if premise.id.trim().is_empty()
        || premise.statement.trim().is_empty()
        || premise.established_in.population_id.trim().is_empty()
        || premise.established_in.regime_id.trim().is_empty()
        || premise.applies_to.population_id.trim().is_empty()
        || premise.applies_to.regime_id.trim().is_empty()
        || premise.established_in == premise.applies_to
        || premise.evidence.as_ref().is_some_and(|evidence| evidence.trim().is_empty())
        || request.source_trial_ids.len() != request.source.trials.len()
        || request.target_trial_ids.len() != request.target.trials.len()
        || request
            .source_trial_ids
            .iter()
            .chain(&request.target_trial_ids)
            .any(|id| id.trim().is_empty() || !ids.insert(id.as_str()))
        || request.surrogate_grid.is_empty()
        || request.surrogate_grid.len() > 64
        || !request.equivalence_margin.is_finite()
        || request.equivalence_margin <= 0.0
    {
        return Err(invalid("premise, independent trial identities, grid or margin is invalid"));
    }
    if request.surrogate_grid.iter().enumerate().any(|(index, &value)| {
        !value.is_finite() || (index > 0 && value <= request.surrogate_grid[index - 1])
    }) {
        return Err(invalid("surrogate grid must be finite and strictly increasing"));
    }
    let valid_trials = |validation: &TrialLevelValidation| {
        validation.min_trials >= 3
            && validation.trials.len() >= validation.min_trials as usize
            && validation.min_r_squared.is_finite()
            && (0.0..=1.0).contains(&validation.min_r_squared)
            && validation.trials.iter().all(|trial| {
                trial.surrogate_effect.is_finite() && trial.terminal_effect.is_finite()
            })
    };
    if !valid_trials(&request.source) || !valid_trials(&request.target) {
        return Err(invalid("source or target validation trials are malformed"));
    }
    let range = |validation: &TrialLevelValidation| {
        validation.trials.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), trial| {
            (lo.min(trial.surrogate_effect), hi.max(trial.surrogate_effect))
        })
    };
    let (source_lo, source_hi) = range(&request.source);
    let (target_lo, target_hi) = range(&request.target);
    if request
        .surrogate_grid
        .iter()
        .any(|&x| x < source_lo || x > source_hi || x < target_lo || x > target_hi)
    {
        return Err(invalid(
            "assessment grid leaves the overlap of source and target trial effects",
        ));
    }
    Ok(())
}

/// Compile the entire comparison before computing an equivalence screen.
/// This is a diagnostic about a declared bridge relation, not an automatic
/// discharge of the transportability premise in a downstream decision.
///
/// # Errors
/// Invalid independent trials, fit, support overlap, margin or lowering.
pub fn compile_bridge_transport_premise_assessment(
    request: BridgeTransportPremiseAssessmentRequest,
) -> Result<CompiledBridgeTransportPremiseAssessment, BridgeTransportPremiseAssessmentError> {
    validate(&request)?;
    let input = AssessmentInputWire::from_domain(&request);
    let input_bytes = antecedent_io::to_cbor(&input)?;
    if input_bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(invalid("bridge premise input exceeds its byte bound"));
    }
    let source_digest = CanonicalWriter::digest_bytes(
        "antecedent.bridge_transport_premise_assessment.source.v1",
        &input_bytes,
    );
    let mut writer =
        CanonicalWriter::new("antecedent.bridge_transport_premise_assessment.question.v1");
    writer.bytes(&source_digest);
    let question = writer.finish();
    let design = DesignShape::quoted(
        (request.source.trials.len() + request.target.trials.len()) as u64,
        2,
        true,
    );
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::MechanismChange,
            outcome: None,
            query: question,
        },
        plan_id: Arc::from("bridge_transport_premise_assessment.v1"),
        data: DataClassification::Tabular,
        graph_class: "source_target_trial_mapping",
        query_variables: Arc::from([]),
        source_digest: Some(source_digest),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: question },
    ));
    let score = ScoreModel::new("independent_trial_mapping_difference")
        .map_err(|_| invalid("invalid mapping difference score identity"))?;
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::Frequentist(ScoreFrequentist::new(score)),
        models: Arc::from([]),
        folds: None,
        resampling: None,
        posterior: None,
        design,
        operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([
            EvidenceSourceRef { role: EvidenceRole::Observations, digest: Some(source_digest) },
            EvidenceSourceRef { role: EvidenceRole::TransportSource, digest: Some(question) },
        ]),
        composition: Composition::Chained { stages: 2 },
        decision: None,
        planning: None,
        chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .estimates(ByteEstimates::absent().with_output(input_bytes.len() as u64))
                .build()
                .map_err(|_| invalid("invalid bridge validation read operator"))?,
            PhysicalOperator::declare(PhysicalEffect::Fit)
                .build()
                .map_err(|_| invalid("invalid bridge validation fit operator"))?,
        ]),
        design,
    };
    let lowered = Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::BridgeTransportPremiseAssessment,
            scientific,
            identification,
            statistical,
            evidence,
            physical,
            claim: ClaimDemand { uncertainty: UncertaintyReading::Sampling, kind: None },
        }
        .finish(),
    );
    Ok(CompiledBridgeTransportPremiseAssessment { request, input, lowered })
}

fn predicted_variance(
    validation: &TrialLevelValidation,
    fit: TrialLevelFit,
    x: f64,
) -> Result<f64, BridgeTransportPremiseAssessmentError> {
    let n = validation.trials.len();
    let nf = n as f64;
    let mean_x = validation.trials.iter().map(|trial| trial.surrogate_effect).sum::<f64>() / nf;
    let sxx = validation
        .trials
        .iter()
        .map(|trial| (trial.surrogate_effect - mean_x).powi(2))
        .sum::<f64>();
    if n < 3 || !sxx.is_finite() || sxx <= 0.0 {
        return Err(invalid("trial regression has no finite surrogate spread"));
    }
    let rss = validation
        .trials
        .iter()
        .map(|trial| {
            (trial.terminal_effect - fit.intercept - fit.slope * trial.surrogate_effect).powi(2)
        })
        .sum::<f64>();
    let variance = rss / (nf - 2.0) * (1.0 / nf + (x - mean_x).powi(2) / sxx);
    if !variance.is_finite() || variance < 0.0 {
        return Err(invalid("trial regression prediction variance is invalid"));
    }
    Ok(variance)
}

fn fingerprint(result: &BridgeTransportPremiseAssessmentResult) -> [u8; 32] {
    let mut writer =
        CanonicalWriter::new("antecedent.bridge_transport_premise_assessment.result.v1");
    writer
        .word(result.source_fit.intercept.to_bits())
        .word(result.source_fit.slope.to_bits())
        .word(result.target_fit.intercept.to_bits())
        .word(result.target_fit.slope.to_bits())
        .word(u64::from(result.equivalence_screen_passed))
        .text(result.calibration)
        .bytes(result.lowered_program_identity.as_bytes());
    for point in &result.grid {
        writer
            .word(point.surrogate_effect.to_bits())
            .word(point.difference.to_bits())
            .word(point.standard_error.to_bits())
            .word(point.lower.to_bits())
            .word(point.upper.to_bits());
    }
    writer.finish()
}

impl CompiledProgramHandle for CompiledBridgeTransportPremiseAssessment {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledBridgeTransportPremiseAssessment {
    /// Six-layer program for the two trial fits and prespecified grid.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the retained program without fitting diagnostics.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        CompiledProgramHandle::plan_view(self)
    }

    /// Whether the assessment matches a trial-level bridge's own source
    /// mapping and exact transport premise. A matched assessment is evidence
    /// about that premise, not automatic proof or calibration of the bridge.
    #[must_use]
    pub fn matches_bridge(&self, contract: &BridgeContract) -> bool {
        contract.transport.as_ref() == Some(&self.request.premise)
            && matches!(&contract.mapping,
                BridgeMapping::TrialLevelLinear(source) if source == &self.request.source)
            && contract.evidence_scope == self.request.premise.established_in
    }

    /// Fit both trial relations, form Bonferroni-split normal-Wald discrepancy
    /// bands and screen them against the declared equivalence margin.
    ///
    /// # Errors
    /// Nonfinite regression variance or contrast.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<BridgeTransportPremiseAssessmentResult, BridgeTransportPremiseAssessmentError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        gate_program_when(GATE_ON_UNIT_SUPPORT, &scoped, self).map_err(|refusal| {
            BridgeTransportPremiseAssessmentError::Invalid(refusal.to_string())
        })?;
        if scoped.cancellation.is_cancelled() {
            return Err(invalid("bridge premise assessment was cancelled"));
        }
        let source_fit = fit_trial_level_linear(&self.request.source)
            .map_err(|refusal| fit_refused(&refusal))?;
        let target_fit = fit_trial_level_linear(&self.request.target)
            .map_err(|refusal| fit_refused(&refusal))?;
        let count = self.request.surrogate_grid.len() as f64;
        let z = norm_inv(1.0 - 0.05 / (2.0 * count));
        let mut grid = Vec::with_capacity(self.request.surrogate_grid.len());
        for &x in &self.request.surrogate_grid {
            let source = source_fit.intercept + source_fit.slope * x;
            let target = target_fit.intercept + target_fit.slope * x;
            let difference = target - source;
            let variance = predicted_variance(&self.request.source, source_fit, x)?
                + predicted_variance(&self.request.target, target_fit, x)?;
            let standard_error = variance.sqrt();
            let lower = difference - z * standard_error;
            let upper = difference + z * standard_error;
            if !lower.is_finite() || !upper.is_finite() {
                return Err(invalid("bridge discrepancy band is nonfinite"));
            }
            grid.push(BridgeTransportDiscrepancy {
                surrogate_effect: x,
                difference,
                standard_error,
                lower,
                upper,
            });
        }
        let equivalence_screen_passed = grid.iter().all(|point| {
            point.lower > -self.request.equivalence_margin
                && point.upper < self.request.equivalence_margin
        });
        Ok(BridgeTransportPremiseAssessmentResult {
            source_fit,
            target_fit,
            grid,
            equivalence_screen_passed,
            calibration: self.unit_calibration(),
            lowered_program_identity: self.lowered.identity(),
        })
    }

    /// Digest of the assessment's construction: the declared premise, the fit
    /// thresholds, the equivalence margin and the surrogate evaluation grid.
    /// The trials themselves, their identities and their count are data.
    pub(crate) fn calibration_construction(&self) -> SemanticDigest {
        let request = &self.request;
        let mut writer = CanonicalWriter::new(
            "antecedent.bridge_transport_premise_assessment.calibration_construction.v1",
        );
        writer
            .text(&request.premise.id)
            .text(&request.premise.statement)
            .text(&request.premise.established_in.population_id)
            .text(&request.premise.established_in.regime_id)
            .text(&request.premise.applies_to.population_id)
            .text(&request.premise.applies_to.regime_id)
            .text(request.premise.evidence.as_deref().unwrap_or(""))
            .word(u64::from(request.source.min_trials))
            .word(request.source.min_r_squared.to_bits())
            .word(u64::from(request.target.min_trials))
            .word(request.target.min_r_squared.to_bits())
            .word(request.equivalence_margin.to_bits());
        writer.word(request.surrogate_grid.len() as u64);
        for &x in &request.surrogate_grid {
            writer.word(x.to_bits());
        }
        SemanticDigest::from_bytes(writer.finish())
    }

    /// Candidate for whole-grid simultaneous coverage calibration. The
    /// construction binds the premise, fit thresholds, margin and surrogate
    /// grid; the trials and their count are the record's scope.
    ///
    /// # Errors
    /// Trial fitting or discrepancy construction refused.
    pub fn calibration_candidate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<BridgeTransportPremiseCalibrationCandidate, BridgeTransportPremiseAssessmentError>
    {
        let result = self.execute(ctx)?;
        let request = &self.request;
        let functional = format!(
            "trial_mapping_discrepancy_grid:construction={}",
            self.calibration_construction().to_hex()
        );
        let basis = CalibrationBasis::new(
            [
                "MechanismChange",
                "TrialLevelBridge",
                "fixed",
                "two_scope_trials",
                "Frequentist",
                FEATURE,
                "bonferroni_normal_wald_95_grid",
                "ols_residual",
                "independent_source_target_trials",
                "none",
                functional.as_str(),
            ]
            .map(Arc::from),
            0.95,
            Arc::from("model_conditional"),
            (request.source.trials.len() + request.target.trials.len()) as u64,
            None,
            None,
            0.0,
        )
        .with_validation("none");
        Ok(BridgeTransportPremiseCalibrationCandidate { result, basis })
    }

    /// Recompute before sealing the independent validation evidence.
    ///
    /// # Errors
    /// Changed result or oversized artifact.
    pub fn export(
        &self,
        result: &BridgeTransportPremiseAssessmentResult,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, BridgeTransportPremiseAssessmentError> {
        let live = self.execute(ctx)?;
        if fingerprint(result) != fingerprint(&live) {
            return Err(invalid("bridge premise assessment changed on replay"));
        }
        let bytes = antecedent_io::to_cbor(&AssessmentArtifactWire {
            version: 1,
            required_feature: FEATURE.into(),
            input: self.input.clone(),
            program_identity: *self.lowered.identity().as_bytes(),
            result_fingerprint: fingerprint(&live),
        })?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(invalid("bridge premise artifact exceeds its byte bound"));
        }
        Ok(bytes)
    }
}

/// Recompile and refit both validation trial sets from a portable artifact,
/// comparing the result to an independently retained compiler identity.
///
/// # Errors
/// Foreign feature, changed premise, trial data, fit or result.
pub fn consume_bridge_transport_premise_assessment(
    bytes: &[u8],
    expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<BridgeTransportPremiseAssessmentResult, BridgeTransportPremiseAssessmentError> {
    Ok(recompile_bridge_transport_premise_assessment(bytes, expected_program, ctx)?.1)
}

pub(crate) fn recompile_bridge_transport_premise_assessment(
    bytes: &[u8],
    expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<
    (CompiledBridgeTransportPremiseAssessment, BridgeTransportPremiseAssessmentResult),
    BridgeTransportPremiseAssessmentError,
> {
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err(invalid("bridge premise artifact exceeds its byte bound"));
    }
    let wire: AssessmentArtifactWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1
        || wire.required_feature != FEATURE
        || wire.program_identity != *expected_program.as_bytes()
    {
        return Err(invalid("bridge premise feature or program identity differs"));
    }
    let plan = compile_bridge_transport_premise_assessment(wire.input.to_domain())?;
    if plan.lowered.identity() != expected_program {
        return Err(invalid("recompiled bridge premise program differs"));
    }
    let live = plan.execute(ctx)?;
    if fingerprint(&live) != wire.result_fingerprint {
        return Err(invalid("bridge premise assessment differs on replay"));
    }
    Ok((plan, live))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_target_trials_replay_the_bridge_discrepancy() {
        let trials = |target: bool| TrialLevelValidation {
            trials: (0..20)
                .map(|i| {
                    let x = -1.0 + 2.0 * f64::from(i) / 19.0;
                    let shift = if target { 0.05 } else { 0.0 };
                    TrialEffectPair {
                        surrogate_effect: x,
                        terminal_effect: 0.3
                            + 2.0 * x
                            + shift
                            + (if i % 2 == 0 { 0.01 } else { -0.01 }),
                    }
                })
                .collect(),
            min_trials: 3,
            min_r_squared: 0.0,
        };
        let request = BridgeTransportPremiseAssessmentRequest {
            premise: TransportabilityPremise {
                id: "surrogate-line-invariant".into(),
                statement: "the trial-level surrogate relation transfers".into(),
                established_in: BridgeScope {
                    population_id: "source".into(),
                    regime_id: "trial".into(),
                },
                applies_to: BridgeScope {
                    population_id: "target".into(),
                    regime_id: "trial".into(),
                },
                evidence: Some("independent-target-validation".into()),
            },
            source: trials(false),
            target: trials(true),
            source_trial_ids: (0..20).map(|i| format!("source-trial-{i}")).collect(),
            target_trial_ids: (0..20).map(|i| format!("target-trial-{i}")).collect(),
            surrogate_grid: vec![-0.8, 0.0, 0.8],
            equivalence_margin: 0.2,
        };
        let ctx = ExecutionContext::for_tests(73);
        let plan = compile_bridge_transport_premise_assessment(request).unwrap();
        assert_eq!(plan.plan_view().route().as_str(), "BridgeTransportPremiseAssessment");
        let result = plan.execute(&ctx).unwrap();
        assert!(result.equivalence_screen_passed);
        assert_eq!(result.calibration, "not_measured");
        let bytes = plan.export(&result, &ctx).unwrap();
        let replay = consume_bridge_transport_premise_assessment(
            &bytes,
            result.lowered_program_identity,
            &ctx,
        )
        .unwrap();
        assert_eq!(replay, result);
    }

    /// `(trial id, surrogate effect, terminal effect)` of twenty trials on one line.
    fn trial_rows(prefix: &str, shift: f64) -> Vec<(String, f64, f64)> {
        (0..20)
            .map(|i| {
                let x = -1.0 + 2.0 * f64::from(i) / 19.0;
                let noise = if i % 2 == 0 { 0.01 } else { -0.01 };
                (format!("{prefix}-trial-{i}"), x, 0.3 + 2.0 * x + shift + noise)
            })
            .collect()
    }

    fn assessment_of(
        source: &[(String, f64, f64)],
        target: &[(String, f64, f64)],
        grid: &[f64],
    ) -> BridgeTransportPremiseAssessmentRequest {
        let validation = |trials: &[(String, f64, f64)]| TrialLevelValidation {
            trials: trials
                .iter()
                .map(|(_, surrogate_effect, terminal_effect)| TrialEffectPair {
                    surrogate_effect: *surrogate_effect,
                    terminal_effect: *terminal_effect,
                })
                .collect(),
            min_trials: 3,
            min_r_squared: 0.0,
        };
        BridgeTransportPremiseAssessmentRequest {
            premise: TransportabilityPremise {
                id: "surrogate-line-invariant".into(),
                statement: "the trial-level surrogate relation transfers".into(),
                established_in: BridgeScope {
                    population_id: "source".into(),
                    regime_id: "trial".into(),
                },
                applies_to: BridgeScope {
                    population_id: "target".into(),
                    regime_id: "trial".into(),
                },
                evidence: Some("independent-target-validation".into()),
            },
            source: validation(source),
            target: validation(target),
            source_trial_ids: source.iter().map(|(id, ..)| id.clone()).collect(),
            target_trial_ids: target.iter().map(|(id, ..)| id.clone()).collect(),
            surrogate_grid: grid.to_vec(),
            equivalence_margin: 0.2,
        }
    }

    fn question_of(request: BridgeTransportPremiseAssessmentRequest) -> Option<[u8; 32]> {
        let plan = compile_bridge_transport_premise_assessment(request).ok()?;
        match plan.lowered().scientific().target() {
            ProblemTarget::QueryFunctional { query, .. } => Some(*query),
            _ => None,
        }
    }

    #[test]
    fn site_bridge_transport_premise_assessment_question_orders_trials_and_binds_parts() {
        let grid = [-0.8, 0.0, 0.8];
        let target = trial_rows("target", 0.05);
        // Each trial is aligned with its identity and the least-squares sums accumulate in
        // declared order, so the bit-exact fit reads the trials in that order.
        crate::identity_laws::ordered_laws(
            &trial_rows("source", 0.0),
            |source: &[(String, f64, f64)]| question_of(assessment_of(source, &target, &grid)),
        );
        let source = trial_rows("source", 0.0);
        // The grid is a set held in its canonical form: an unsorted grid is refused.
        assert!(question_of(assessment_of(&source, &target, &[0.8, 0.0, -0.8])).is_none());
        let mut margin = assessment_of(&source, &target, &grid);
        margin.equivalence_margin = 0.25;
        let mut premise = assessment_of(&source, &target, &grid);
        premise.premise.id = "another-premise".into();
        let mut renamed = target.clone();
        renamed[0].0 = "renamed-trial".into();
        crate::identity_laws::all_distinct(&[
            question_of(assessment_of(&source, &target, &grid)),
            question_of(assessment_of(&source, &trial_rows("target", 0.06), &grid)),
            question_of(assessment_of(&source, &target, &[-0.8, 0.0, 0.7])),
            question_of(assessment_of(&source, &renamed, &grid)),
            question_of(margin),
            question_of(premise),
        ]);
    }

    #[test]
    fn site_bridge_transport_premise_assessment_result_fingerprint_reads_every_field() {
        let plan = compile_bridge_transport_premise_assessment(assessment_of(
            &trial_rows("source", 0.0),
            &trial_rows("target", 0.05),
            &[-0.8, 0.0, 0.8],
        ))
        .expect("a valid assessment");
        let base = plan.execute(&ExecutionContext::for_tests(73)).expect("both trial fits");
        let mut variants = vec![base.clone()];
        let mut changed = base.clone();
        changed.source_fit.intercept += 1.0;
        variants.push(changed);
        let mut changed = base.clone();
        changed.target_fit.slope += 1.0;
        variants.push(changed);
        let mut changed = base.clone();
        changed.equivalence_screen_passed = !changed.equivalence_screen_passed;
        variants.push(changed);
        let mut changed = base.clone();
        changed.calibration = "measured";
        variants.push(changed);
        let mut changed = base.clone();
        changed.lowered_program_identity = antecedent_core::SemanticDigest::from_bytes([1; 32]);
        variants.push(changed);
        // Grid points are reported in grid order.
        let mut changed = base.clone();
        changed.grid.swap(0, 2);
        variants.push(changed);
        let mut changed = base.clone();
        changed.grid[1].upper += 1.0;
        variants.push(changed);
        crate::identity_laws::all_distinct(&variants.iter().map(fingerprint).collect::<Vec<_>>());
        crate::identity_laws::bit_exact(|difference| {
            let mut changed = base.clone();
            changed.grid[0].difference = difference;
            Some(fingerprint(&changed))
        });
    }
}

/// Joint intercept and slope draws under the normal linear trial model and
/// reference prior `p(intercept, slope, variance) ∝ 1 / variance`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TargetLinePosteriorDraw {
    pub intercept: f64,
    pub slope: f64,
}

impl CompiledBridgeTransportPremiseAssessment {
    /// Draw from the joint target-line posterior, retaining a common variance
    /// scale for both coefficients. The target design is conditioned on.
    /// Number of independently declared target validation trials.
    pub(crate) fn target_trial_count(&self) -> usize {
        self.request.target.trials.len()
    }

    pub(crate) fn target_line_posterior(
        &self,
        count: usize,
        seed: u64,
        ctx: &ExecutionContext,
    ) -> Result<Vec<TargetLinePosteriorDraw>, BridgeTransportPremiseAssessmentError> {
        let trials = &self.request.target.trials;
        if count == 0 || count > 1_000_000 || trials.len() < 3 {
            return Err(invalid("target posterior draw count or trial count invalid"));
        }
        let fit = fit_trial_level_linear(&self.request.target)
            .map_err(|refusal| fit_refused(&refusal))?;
        let n = trials.len() as f64;
        let xbar = trials.iter().map(|p| p.surrogate_effect).sum::<f64>() / n;
        let sxx = trials.iter().map(|p| (p.surrogate_effect - xbar).powi(2)).sum::<f64>();
        let rss = trials
            .iter()
            .map(|p| (p.terminal_effect - fit.intercept - fit.slope * p.surrogate_effect).powi(2))
            .sum::<f64>();
        if !sxx.is_finite() || sxx <= 0.0 || !rss.is_finite() || rss <= 0.0 {
            return Err(invalid("target reference-prior line posterior is improper or singular"));
        }
        let df = n - 2.0;
        let variance = rss / df;
        let mut rng = antecedent_core::CausalRng::from_seed(seed);
        let mut result = Vec::with_capacity(count);
        for _ in 0..count {
            if ctx.cancellation.is_cancelled() {
                return Err(invalid("target posterior sampling cancelled"));
            }
            let chi2 = antecedent_kernels::sample_gamma(df / 2.0, 0.5, &mut rng);
            let scale = (df / chi2).sqrt();
            let slope = fit.slope
                + scale * (variance / sxx).sqrt() * antecedent_kernels::standard_normal(&mut rng);
            let intercept = fit.intercept - (slope - fit.slope) * xbar
                + scale * (variance / n).sqrt() * antecedent_kernels::standard_normal(&mut rng);
            if !slope.is_finite() || !intercept.is_finite() {
                return Err(invalid("target posterior draw is nonfinite"));
            }
            result.push(TargetLinePosteriorDraw { intercept, slope });
        }
        Ok(result)
    }
}
