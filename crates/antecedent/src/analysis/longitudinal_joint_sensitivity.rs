//! Simultaneous sensitivity to two treatment selections and final-outcome MNAR.
//!
//! For each schedule, sequential selection bias is the mean potential outcome among
//! off-schedule subjects minus the mean among on-schedule subjects, conditional on their
//! currently observed history and adherence to the earlier schedule. Its value is the
//! negative of the declared stage departure for schedule 11 and the positive departure
//! for schedule 00. The MNAR shift is missing minus recorded conditional mean, with
//! the same sign convention. No departure concerns a marginal propensity-odds ratio.
//!
//! Under consistency, independent subjects, sequential positivity and the declared
//! mean restrictions, complete the final outcome means first, then mix over the
//! second treatment's natural assignment at each history and the intermediate
//! distribution under the first assigned treatment. Finally mix over the first
//! treatment's natural assignment within baseline history. Each step is a
//! conditional total expectation and gives an affine function of the departures.
// SPDX-License-Identifier: MIT OR Apache-2.0

mod kernel;

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{
    ByteEstimates, CalibrationBasis, CanonicalWriter, ExecutionContext, PhysicalEffect,
    PhysicalOperator, SemanticDigest,
};
use antecedent_io::calibration::{calibration_slots, CalibrationBasisWire};
use antecedent_io::contract_section::CalibrationSlotWire;
use serde::{Deserialize, Serialize};

pub use kernel::{
    LongitudinalSensitivityChoice, LongitudinalSensitivityCorner, LongitudinalSensitivityDecision,
    LongitudinalSensitivityPoint, LongitudinalSensitivityReport,
};

use crate::analysis::supplied_terminal::{admit_compile, compiled_terminal};
use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramCell, ProgramIdentityScope,
    ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{
    Realization, ResampleUnit, ResamplingMethod, ResamplingPlan, StatisticalProgram,
};

const MAX_WIRE_BYTES: usize = 16 * 1024 * 1024;
const VERSION: u16 = 1;
const ARTIFACT_KIND: &str = "longitudinal_joint_sensitivity";
const SECTION: &str = "longitudinal_joint_sensitivity.result";
const LAYOUT: &str = "foreign longitudinal joint sensitivity layout";
const MODEL: &str = "longitudinal.sequential_mean_selection.final_pattern_mixture";
const CLAIM_METHOD: &str = "paired_subject_bootstrap_supremum";
const COVERAGE_SURFACE: &str = "balanced_binary_bernoulli_v1:full_box_simultaneous_band";
const COVERAGE_EXTREMA: &str = "balanced_binary_bernoulli_v1:joint_decision_extrema";

/// One complete subject history. Only the final outcome may be unrecorded.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LongitudinalSubject {
    /// Subject identity, unique within this independent-subject sampling design.
    pub subject: u64,
    /// Binary baseline history before the first treatment.
    pub baseline: u8,
    /// First and second treatment, respectively.
    pub treatment: [u8; 2],
    /// Binary history after the first treatment and before the second.
    pub intermediate: u8,
    /// Recorded numeric outcome, or `None` for a missing outcome.
    pub outcome: Option<f64>,
}

/// Source-fixed design for a measured coverage record. This is a caller assumption.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LongitudinalSensitivityCoverageDesign {
    /// Independent fair-coin histories/assignments, observation probability 0.75
    /// and recorded Bernoulli mean `0.30 + 0.20*A0 + 0.04*L1 + 0.02*W`.
    BalancedBinaryBernoulliV1,
}

/// Sensitivity box, outcome semantics and sampling design for schedule 11 versus 00.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LongitudinalSensitivitySpec {
    /// Study population named by this target.
    pub population: String,
    /// Final scalar outcome name.
    pub outcome: String,
    /// Unit of the outcome, departures and optional decision threshold.
    pub unit: String,
    /// Two closed ranges for the additive stage-one and stage-two selection contrasts.
    pub assignment_limits: [[f64; 2]; 2],
    /// Closed range for the pattern-mixture shift.
    pub mnar_limits: [f64; 2],
    /// The outcome range when scientifically bounded; None for real-valued outcomes.
    pub outcome_support: Option<[f64; 2]>,
    /// Number of whole-subject refits, under one shared bootstrap index per replicate.
    pub replicates: u32,
    /// Nominal simultaneous coverage target, strictly between zero and one.
    pub level: f64,
    /// An optional minimum advantage of schedule 11 over 00.
    pub decision_threshold: Option<f64>,
    /// An optional source-fixed law to match a coverage record.
    #[serde(default)]
    pub coverage_design: Option<LongitudinalSensitivityCoverageDesign>,
}

/// Longitudinal joint-sensitivity failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum JointSensitivityError {
    /// Missing naming, invalid rows or unsupported finite bounds.
    #[error("invalid longitudinal sensitivity declaration: {0}")]
    Invalid(&'static str),
    /// At least one required history lacks treatment overlap or an observed outcome.
    #[error(
        "each history needs both treatments and every on-schedule cell needs a recorded outcome"
    )]
    Positivity,
    /// A resample cannot independently refit a required treatment or outcome cell.
    #[error("whole-subject bootstrap replicate {0} lacks a required history or observed outcome")]
    RefitPositivity(u32),
    /// A corner's implied potential-outcome mean violates the declared outcome support.
    #[error("a departure corner implies a mean outside the declared outcome support")]
    IncoherentBox,
    /// Some numerical results were unrepresentable.
    #[error("a longitudinal sensitivity calculation is non-finite")]
    Numeric,
    /// The caller cancelled the computation.
    #[error("longitudinal joint sensitivity was cancelled")]
    Cancelled,
    /// Caller resource admission refused the operation.
    #[error("longitudinal joint sensitivity lacks resources: {0}")]
    Resource(String),
    /// The compiler or support gate refused.
    #[error(transparent)]
    Compile(#[from] CausalError),
    /// The portable carrier was ill formed.
    #[error(transparent)]
    Wire(#[from] antecedent_io::IoError),
    /// Expected identity, configuration, or retained result differs from live replay.
    #[error("longitudinal sensitivity replay differs from the retained inputs or result")]
    Identity,
}

impl LongitudinalSensitivitySpec {
    fn check(&self, rows: &[LongitudinalSubject]) -> Result<(), JointSensitivityError> {
        let named = |name: &str| !name.trim().is_empty() && name.len() <= 200;
        if ![&self.population, &self.outcome, &self.unit].iter().all(|name| named(name))
            || self.replicates < 199
            || !(0.0 < self.level && self.level < 1.0)
            || self.decision_threshold.is_some_and(|value| !value.is_finite())
            || rows.len() < 16
        {
            return Err(JointSensitivityError::Invalid(
                "target names, sampling design, or cohort size",
            ));
        }
        let ordered = |[lo, hi]: [f64; 2]| lo.is_finite() && hi.is_finite() && lo <= hi;
        let through_zero = |range: [f64; 2]| ordered(range) && range[0] <= 0.0 && range[1] >= 0.0;
        if !self.assignment_limits.into_iter().all(through_zero)
            || !through_zero(self.mnar_limits)
            || self.outcome_support.is_some_and(|range| !ordered(range))
        {
            return Err(JointSensitivityError::Invalid(
                "ordered finite departure ranges must include zero and outcome support must be ordered and finite",
            ));
        }
        let measured_design = self.coverage_design.is_some();
        if measured_design
            && (self.unit != "proportion"
                || self.level.to_bits() != 0.95_f64.to_bits()
                || self.replicates != 199
                || self.assignment_limits != [[0.0, 0.055], [0.0, 0.055]]
                || self.mnar_limits != [0.0, 0.10]
                || self.outcome_support != Some([0.0, 1.0])
                || self.decision_threshold != Some(0.075))
        {
            return Err(JointSensitivityError::Invalid(
                "the declared coverage design fixes the outcome unit and support, departure box, decision threshold, and 199 refits at nominal 0.95",
            ));
        }
        let mut subjects = BTreeSet::new();
        for row in rows {
            if !subjects.insert(row.subject)
                || row.baseline > 1
                || row.intermediate > 1
                || row.treatment.iter().any(|&a| a > 1)
                || row.outcome.is_some_and(|value| {
                    !value.is_finite()
                        || (measured_design && ![0.0, 1.0].contains(&value))
                        || self.outcome_support.is_some_and(|[lo, hi]| value < lo || value > hi)
                })
            {
                return Err(JointSensitivityError::Invalid(
                    "subject IDs must be unique, histories and treatments binary, and recorded outcomes finite and in support",
                ));
            }
        }
        Ok(())
    }
}

compiled_terminal! {
    /// Compiled two-decision joint selection and missingness sensitivity under one bootstrap.
    #[derive(Clone, Debug)]
    CompiledLongitudinalJointSensitivity {
        spec: LongitudinalSensitivitySpec,
        rows: Arc<[LongitudinalSubject]>,
    }
}

fn admit_joint_work(
    ctx: &ExecutionContext,
    rows: &[LongitudinalSubject],
    replicates: u32,
) -> Result<antecedent_core::execution::AdmissionLease, JointSensitivityError> {
    let rows = u64::try_from(rows.len()).map_err(|_| {
        JointSensitivityError::Resource("longitudinal joint sensitivity quote overflowed".into())
    })?;
    let bytes = rows
        .checked_mul(48)
        .and_then(|body| {
            u64::from(replicates).checked_mul(16).and_then(|draws| body.checked_add(draws))
        })
        .ok_or_else(|| {
            JointSensitivityError::Resource(
                "longitudinal joint sensitivity quote overflowed".into(),
            )
        })?;
    ctx.acquire_serial_numerical_admission(bytes).map_err(|error| {
        if ctx.cancellation.is_cancelled() {
            JointSensitivityError::Cancelled
        } else {
            JointSensitivityError::Resource(error.to_string())
        }
    })
}

/// Compile schedule 11 minus schedule 00 with paired subject refits and a named joint region.
///
/// # Errors
/// Malformed declarations, compiler re-entry, resource refusal, or program lowering failure.
pub fn compile_longitudinal_joint_sensitivity(
    spec: LongitudinalSensitivitySpec,
    rows: impl Into<Arc<[LongitudinalSubject]>>,
    ctx: &ExecutionContext,
) -> Result<CompiledLongitudinalJointSensitivity, JointSensitivityError> {
    admit_compile(ctx)?;
    let rows = rows.into();
    spec.check(&rows)?;
    let _lease = admit_joint_work(ctx, &rows, spec.replicates)?;
    let mut declaration = CanonicalWriter::new("antecedent.longitudinal.joint_sensitivity.v1");
    declaration.text(MODEL).text(CLAIM_METHOD).bytes(&antecedent_io::to_cbor(&spec)?);
    let mut source = CanonicalWriter::new("antecedent.longitudinal.joint_sensitivity.subjects.v1");
    source.word(rows.len() as u64);
    for row in rows.iter() {
        source
            .word(row.subject)
            .word(u64::from(row.baseline))
            .word(u64::from(row.treatment[0]))
            .word(u64::from(row.intermediate))
            .word(u64::from(row.treatment[1]))
            .opt_word(row.outcome.map(f64::to_bits));
    }
    let source = source.finish();
    declaration.bytes(&source);
    let question = declaration.finish();
    let lowered = lower(&spec, question, source, rows.len() as u64)?;
    Ok(CompiledLongitudinalJointSensitivity { lowered, spec, rows })
}

fn lower(
    spec: &LongitudinalSensitivitySpec,
    question: [u8; 32],
    source: [u8; 32],
    rows: u64,
) -> Result<Arc<LoweredProgram>, JointSensitivityError> {
    let design = DesignShape::quoted(rows, 5, false);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::IdentifiedSet,
            outcome: None,
            query: question,
        },
        plan_id: Arc::from(MODEL),
        data: DataClassification::Tabular,
        graph_class: "two_binary_decisions_selection_bias",
        query_variables: Arc::from([]),
        source_digest: Some(source),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: question },
    ));
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::ExactPoint { evaluator: Arc::from(CLAIM_METHOD) },
        models: Arc::from([]),
        folds: None,
        resampling: Some(ResamplingPlan {
            method: ResamplingMethod::Bootstrap,
            replicates: spec.replicates,
            unit: ResampleUnit::Subject,
        }),
        posterior: None,
        design,
        operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef {
            role: EvidenceRole::Observations,
            digest: Some(source),
        }]),
        composition: Composition::Chained { stages: 3 },
        decision: None,
        planning: None,
        chain: None,
    };
    let operator = |effect| {
        PhysicalOperator::declare(effect)
            .estimates(ByteEstimates::absent().with_output(rows * 48))
            .build()
            .map_err(|_| {
                JointSensitivityError::Invalid("the physical operator could not be lowered")
            })
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            operator(PhysicalEffect::Read)?,
            operator(PhysicalEffect::Condition)?,
        ]),
        design,
    };
    Ok(Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::LongitudinalJointSensitivity,
            scientific,
            identification,
            statistical,
            evidence,
            physical,
            claim: ClaimDemand {
                uncertainty: UncertaintyReading::Sampling,
                kind: Some(ClaimKind::Bounds),
            },
        }
        .finish(),
    ))
}

/// Replayable result of the two-decision sensitivity procedure.
#[derive(Clone, Debug)]
pub struct LongitudinalSensitivityArtifact {
    wire: Portable,
    identity: SemanticDigest,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Portable {
    version: u16,
    spec: LongitudinalSensitivitySpec,
    rows: Arc<[LongitudinalSubject]>,
    report: LongitudinalSensitivityReport,
    program: String,
}

impl CompiledLongitudinalJointSensitivity {
    /// Refit both mechanisms on the same complete subjects and score the simultaneous range.
    ///
    /// # Errors
    /// Failure of positivity or model coherence, resource refusal, or cancellation.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<LongitudinalSensitivityArtifact, JointSensitivityError> {
        let scoped = self.enter(ctx)?;
        let _lease = admit_joint_work(&scoped, &self.rows, self.spec.replicates)?;
        let report = kernel::execute(&self.spec, &self.rows, &scoped)?;
        let wire = Portable {
            version: VERSION,
            spec: self.spec.clone(),
            rows: Arc::clone(&self.rows),
            report,
            program: self.identity().to_hex(),
        };
        assemble(wire)
    }
}

fn assemble(wire: Portable) -> Result<LongitudinalSensitivityArtifact, JointSensitivityError> {
    let body = antecedent_io::to_cbor(&wire)?;
    if body.len() > MAX_WIRE_BYTES {
        return Err(JointSensitivityError::Resource("portable artifact exceeds byte bound".into()));
    }
    let mut hash = CanonicalWriter::new("antecedent.longitudinal.joint_sensitivity.artifact.v1");
    hash.bytes(&body);
    let identity = SemanticDigest::from_bytes(hash.finish());
    Ok(LongitudinalSensitivityArtifact { identity, wire })
}

impl LongitudinalSensitivityArtifact {
    /// Model and study declaration.
    #[must_use]
    pub const fn spec(&self) -> &LongitudinalSensitivitySpec {
        &self.wire.spec
    }

    /// Assumption range, simultaneous sampling band, and decision.
    #[must_use]
    pub const fn report(&self) -> &LongitudinalSensitivityReport {
        &self.wire.report
    }

    /// Runtime match keys for the simultaneous box and its two jointly covered decision extrema.
    ///
    /// Keys issue only for the source-fixed coverage declaration. They describe the
    /// estimator and its coverage targets independently of the measured result.
    #[must_use]
    pub fn calibration_bases(&self) -> Vec<CalibrationBasis> {
        if self.wire.spec.coverage_design.is_none() {
            return Vec::new();
        }
        [COVERAGE_SURFACE, COVERAGE_EXTREMA]
            .map(|functional| {
                CalibrationBasis::new(
                    [
                        "LongitudinalRegime",
                        "None",
                        "design",
                        "tabular",
                        "Frequentist",
                        MODEL,
                        "simultaneous_band",
                        "",
                        "shared_subject_refits",
                        "",
                        functional,
                    ]
                    .map(Arc::from),
                    self.wire.report.nominal_level,
                    Arc::from("partial"),
                    self.wire.rows.len() as u64,
                    Some(self.wire.report.refits),
                    None,
                    0.0,
                )
                .with_validation("not_applicable")
            })
            .into()
    }

    /// Scoped calibration, resolved from the current shipped coverage registry.
    #[must_use]
    pub fn calibration(&self) -> CalibrationSlotWire {
        let bases = self.calibration_bases();
        if bases.is_empty() {
            CalibrationSlotWire::not_measured("coverage_design_not_declared")
        } else {
            calibration_slots(&bases.iter().map(CalibrationBasisWire::from).collect::<Vec<_>>())
        }
    }

    /// Coverage evidence status of the simultaneous box.
    #[must_use]
    pub fn calibration_status(&self) -> &'static str {
        self.calibration().status.as_str()
    }

    /// Identity of the sealed report and the data needed for live replay.
    #[must_use]
    pub const fn identity(&self) -> SemanticDigest {
        self.identity
    }

    /// The compiled program identity used to produce this report.
    #[must_use]
    pub fn program_identity(&self) -> &str {
        &self.wire.program
    }

    /// Encode the report and its source data as a re-executable artifact.
    ///
    /// # Errors
    /// Serialization failure or byte bound exceeded.
    pub fn to_bytes(&self) -> Result<Vec<u8>, JointSensitivityError> {
        Ok(antecedent_io::artifact_codec::encode_sectioned_io(
            &antecedent_io::ArtifactContainer {
                kind: ARTIFACT_KIND,
                note: MODEL,
                max_bytes: MAX_WIRE_BYTES,
                artifact_id: "longitudinal-joint-sensitivity",
                library_version: env!("CARGO_PKG_VERSION"),
            },
            SECTION,
            &self.wire,
            &[],
            LAYOUT,
        )?)
    }
}

/// Independently recompute the portable report under caller-retained identities.
///
/// # Errors
/// Carrier is invalid or too large, any identity differs, or the model refuses live replay.
pub fn consume_longitudinal_joint_sensitivity(
    bytes: &[u8],
    expected: SemanticDigest,
    expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<LongitudinalSensitivityArtifact, JointSensitivityError> {
    consume_artifact(bytes, Some((expected, expected_program)), ctx)
}

pub(crate) fn recognizes(bytes: &[u8]) -> bool {
    bytes.len() <= MAX_WIRE_BYTES
        && antecedent_io::artifact_codec::container_kind(bytes).as_deref() == Some(ARTIFACT_KIND)
}

pub(crate) fn consume_artifact(
    bytes: &[u8],
    expected: Option<(SemanticDigest, SemanticDigest)>,
    ctx: &ExecutionContext,
) -> Result<LongitudinalSensitivityArtifact, JointSensitivityError> {
    if bytes.len() > MAX_WIRE_BYTES {
        return Err(JointSensitivityError::Resource("portable artifact exceeds byte bound".into()));
    }
    // A different context may choose other numerical controls, but never a new bootstrap seed.
    let _lease = ctx.acquire_admission(0, bytes.len() as u64 * 2).map_err(|error| {
        if ctx.cancellation.is_cancelled() {
            JointSensitivityError::Cancelled
        } else {
            JointSensitivityError::Resource(error.to_string())
        }
    })?;
    let (wire, []): (Portable, [Vec<u8>; 0]) =
        antecedent_io::artifact_codec::decode_sectioned_io_versioned(
            ARTIFACT_KIND,
            MAX_WIRE_BYTES,
            SECTION,
            &[],
            LAYOUT,
            VERSION,
            bytes,
        )?;
    let spec = wire.spec.clone();
    let rows = Arc::clone(&wire.rows);
    let seed = wire.report.seed;
    let retained = assemble(wire)?;
    if expected.is_some_and(|(identity, program)| {
        retained.identity != identity || retained.wire.program != program.to_hex()
    }) {
        return Err(JointSensitivityError::Identity);
    }
    drop(_lease);
    let mut replay = ctx.clone();
    replay.rng = antecedent_core::execution::RngFactory::from_seed(seed);
    let plan = compile_longitudinal_joint_sensitivity(spec, rows, &replay)?;
    if plan.identity().to_hex() != retained.wire.program {
        return Err(JointSensitivityError::Identity);
    }
    let recomputed = plan.execute(&replay)?;
    if recomputed.identity != retained.identity || recomputed.wire.report != retained.wire.report {
        return Err(JointSensitivityError::Identity);
    }
    Ok(recomputed)
}
