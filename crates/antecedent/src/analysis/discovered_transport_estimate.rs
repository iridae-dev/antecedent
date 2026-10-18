//! Estimating a transported response across the members of a discovered class.
//!
//! [`super::discovered_transport`] says which members of a class the structural rules certify.
//! This module turns that into numbers, honestly. For every member of the class it runs the
//! catalog-aware sID derivation on the member's graph, binds the formula it derives to the supplied
//! source and target evidence, and evaluates the response at every grid level with the retained
//! transport grid (one durable artifact per member). The members are then summarised, never
//! mixed:
//!
//! * **Point.** Every member is identified and they agree: they derive the same expression, or
//!   their estimates are equal. The answer is one number per level.
//! * **Structural range.** Every member is identified but they differ. The answer is the
//!   interval from the smallest to the largest member estimate per level, labelled
//!   `structural_range`: a statement about which values the class allows, with no probability
//!   over members and no interval around any of them.
//! * **Refusal.** Some member is not identified (the refusal names it and its reason), some
//!   member cannot be estimated from the evidence, or the enumeration of the class did not
//!   finish (a class-wide statement would be unearned).
//!
//! # Evidence
//!
//! Source rows come from an experiment on the treatment: each row's treatment value is the
//! experimental assignment, so the rows with one value are a sample of the experimental world
//! `do(treatment = value)`. Target rows are an observational sample of the target population.
//! Catalog-aware identification may use only those two supplied regimes. A requested
//! intervention on another measured column is not an experiment, and no source observational
//! regime is inferred from a randomized trial. Other experiment catalogs need an explicit grid.
//! Every column is a non-negative integer code; the domain of a variable is `0..=max` over both
//! tables. Rows must be complete: nothing is dropped silently.
//!
//! # Intervals
//!
//! A point answer carries a pointwise percentile-bootstrap interval for the contrast only when
//! replicates are requested and the rows are declared independent, or a block length declares
//! them an ordered series (a circular block bootstrap of each table). The interval is conditional
//! on the discovered class (discovery is not repeated in a replicate) and carries no coverage
//! claim until a complete-procedure calibration record is bound to it; its calibration status
//! is `not_bound_to_this_execution`. A structural range has no interval.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![expect(
    clippy::result_large_err,
    reason = "the public error shape is kept as released; boxing the refusal would break callers that match on it"
)]

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use antecedent_core::{
    CanonicalWriter, DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog,
    EvidenceKind, EvidenceRegime, ExecutionContext, RegimeBinding, RegimeId, RegimeKind,
    SamplingDesign, TargetSampling, TransportQuery, Value, VariableCoordinate, VariableDomain,
    VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::{EmpiricalTableOptions, RegimeSample, StatisticalTransportInput};
use antecedent_expr::{Assignment, ExactEvaluationLimits, InterventionAssignment};
use antecedent_graph::{Admg, SelectionDiagram};
use antecedent_identify::discovered_transport::ancestral_subgraph;
use antecedent_identify::{
    CatalogTransportResult, ClassicalTransportQuery, ClassicalTransportResult, SidLimits,
    identify_catalog_transport_standardization_first, identify_classical_transport,
};
use antecedent_io::IoError;
use serde::{Deserialize, Serialize};

use super::discovered_transport::{
    DISCOVERED_TRANSPORT_REASON, DiscoveredTransportError, DiscoveredTransportRefusal,
    DiscoveredTransportReport, EnvelopeStatus, UnresolvedPair,
};
use super::{
    PreparedStudy, StudyBuilder, TransportGridData, TransportGridPoint, TransportGridQuery,
    TransportGridResult, TransportGridState,
};

/// Most levels one variable may take. Larger domains are refused: the empirical joint of the
/// factors a formula reads grows with the product of their domains.
pub const MAX_LEVELS_PER_VARIABLE: u32 = 32;

const ESTIMATE_FORMAT: &str = "antecedent.discovered_transport_estimate.v1";
const IDENTITY_TAG: &str = "antecedent.discovered_transport.estimate.v1";

/// What an interval request is refused with when the replicates are not requested.
const NO_INTERVAL_REQUESTED: &str = "bootstrap_not_requested";
/// Why a structural range has no interval.
const RANGE_HAS_NO_INTERVAL: &str = "structural_range_has_no_interval";

/// Integer-coded columns, aligned by row.
pub(crate) type Columns = BTreeMap<VariableId, Vec<Option<f64>>>;

/// Source and target evidence for an estimate across a discovered class.
#[derive(Clone, Copy, Debug)]
pub struct TransportEvidence<'a> {
    /// Rows of the source population. The treatment column holds the experimental assignment.
    pub source: &'a TabularData,
    /// Rows of the target population, observational.
    pub target: &'a TabularData,
    /// How the target rows relate to the target population. A convenience sample is refused.
    pub target_sampling: TargetSampling,
    /// Whether the rows of each table are independent draws. Only then can an interval be
    /// requested.
    pub rows_independent: bool,
    /// Bootstrap replicates for a pointwise interval on a point answer; zero asks for none.
    pub bootstrap_replicates: u32,
    /// Rows per block of a circular block bootstrap, for tables whose rows are an ordered
    /// series. `None` resamples rows independently, which needs `rows_independent`. A temporal
    /// estimate chooses a length from its window when this is `None`.
    pub block_length: Option<u32>,
    /// Evaluation limits of the retained grid.
    pub limits: ExactEvaluationLimits,
    /// Search limits of the sID derivation run on each member.
    pub sid_limits: SidLimits,
}

impl<'a> TransportEvidence<'a> {
    /// Independent rows from a representative target sample, no interval, default limits.
    #[must_use]
    pub fn new(source: &'a TabularData, target: &'a TabularData) -> Self {
        Self {
            source,
            target,
            target_sampling: TargetSampling::RepresentativeSample,
            rows_independent: true,
            bootstrap_replicates: 0,
            block_length: None,
            limits: ExactEvaluationLimits::default(),
            sid_limits: SidLimits::default(),
        }
    }
}

/// How the members' answers were summarised.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnswerKind {
    /// Every member identified and the members agree.
    Point,
    /// Every member identified and the members differ: the range the class allows.
    StructuralRange,
}

impl AnswerKind {
    /// Stable label: `point` or `structural_range`.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Point => "point",
            Self::StructuralRange => "structural_range",
        }
    }
}

/// The answer at one treatment level: the smallest and the largest member estimate of the mean
/// outcome (equal for a point).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelAnswer {
    /// Treatment level.
    pub level: f64,
    /// Smallest member estimate.
    pub low: f64,
    /// Largest member estimate.
    pub high: f64,
}

/// A pointwise percentile interval on the contrast of a point answer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContrastInterval {
    /// Lower edge.
    pub low: f64,
    /// Upper edge.
    pub high: f64,
    /// Nominal coverage the edges were cut for.
    pub coverage_target: f64,
    /// Always `not_bound_to_this_execution`: no complete-procedure calibration record is bound.
    pub calibration: &'static str,
}

/// The summary of the members' estimates.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportAnswer {
    /// Point or structural range.
    pub kind: AnswerKind,
    /// The answer at each grid level, in grid order.
    pub levels: Vec<LevelAnswer>,
    /// The contrast from the first to the last level (`last - first`), as the smallest and the
    /// largest member contrast. `None` for a single level.
    pub contrast: Option<(f64, f64)>,
    /// Whether every member derived the same expression.
    pub expressions_agree: bool,
    /// The interval on the contrast of a point answer, when requested and licensed.
    pub interval: Option<ContrastInterval>,
    /// Why there is no interval, when there is none.
    pub interval_reason: Option<&'static str>,
}

/// One member's outcome.
#[derive(Clone, Debug)]
pub enum MemberOutcome {
    /// Identified and estimated.
    Estimated(MemberEstimate),
    /// The complete sID derivation did not identify the effect in this member.
    Unidentified {
        /// `not_certified` (no derivation found) or `proven_non_transportable` (a checked
        /// obstruction).
        verdict: &'static str,
    },
    /// Identified, but the evidence cannot supply a factor the formula needs.
    Unestimable {
        /// What was missing.
        detail: String,
    },
}

/// One identified member's estimate at every level.
#[derive(Clone, Debug)]
pub struct MemberEstimate {
    /// The bound expression the member derived.
    pub expression: String,
    /// The estimated mean outcome at each grid level.
    pub means: Vec<f64>,
    /// `last - first` mean, when there are two levels or more.
    pub contrast: Option<f64>,
    /// The pointwise interval on the contrast, when requested and licensed.
    pub contrast_interval: Option<(f64, f64)>,
    /// The retained transport grid of this member as a durable artifact.
    pub artifact: Vec<u8>,
    /// The outcome node of the member's graph, for replay.
    outcome: VariableId,
}

/// An estimate across a discovered class.
#[derive(Clone, Debug)]
pub struct DiscoveredTransportEstimate {
    levels: Vec<f64>,
    members: Vec<MemberOutcome>,
    answer: TransportAnswer,
    report_identity: [u8; 32],
    selection_targets: Vec<VariableId>,
    identity: [u8; 32],
}

impl DiscoveredTransportEstimate {
    /// The treatment levels, in grid order.
    #[must_use]
    pub fn levels(&self) -> &[f64] {
        &self.levels
    }

    /// Every member's outcome, in the order of the report's envelope.
    #[must_use]
    pub fn members(&self) -> &[MemberOutcome] {
        &self.members
    }

    /// The summary of the members' estimates.
    #[must_use]
    pub const fn answer(&self) -> &TransportAnswer {
        &self.answer
    }

    /// Identity of the structural report the estimate was made over.
    #[must_use]
    pub const fn report_identity(&self) -> &[u8; 32] {
        &self.report_identity
    }

    /// Identity of the whole estimate: the report, the levels, every member's expression, means
    /// and artifact, and the answer.
    #[must_use]
    pub const fn identity(&self) -> &[u8; 32] {
        &self.identity
    }

    /// Export the estimate as a durable artifact: the answer, every member's expression and
    /// means, and every member's retained transport grid. [`verify_discovered_transport_estimate`]
    /// replays it.
    ///
    /// # Errors
    ///
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        let members = self
            .members
            .iter()
            .enumerate()
            .filter_map(|(index, member)| match member {
                MemberOutcome::Estimated(estimate) => Some(MemberWire {
                    index: u32::try_from(index).unwrap_or(u32::MAX),
                    expression: estimate.expression.clone(),
                    outcome: estimate.outcome.raw(),
                    means: estimate.means.clone(),
                    artifact: estimate.artifact.clone(),
                }),
                _ => None,
            })
            .collect();
        antecedent_io::to_cbor(&EstimateWire {
            format: ESTIMATE_FORMAT.into(),
            report_identity: self.report_identity.to_vec(),
            selection_targets: self.selection_targets.iter().map(|v| v.raw()).collect(),
            levels: self.levels.clone(),
            members,
            kind: self.answer.kind.label().into(),
            identity: self.identity.to_vec(),
        })
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MemberWire {
    index: u32,
    expression: String,
    outcome: u32,
    means: Vec<f64>,
    artifact: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EstimateWire {
    format: String,
    report_identity: Vec<u8>,
    selection_targets: Vec<u32>,
    levels: Vec<f64>,
    members: Vec<MemberWire>,
    kind: String,
    identity: Vec<u8>,
}

/// What replaying an exported estimate established.
#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedEstimate {
    /// The answer recomputed from the consumed member artifacts.
    pub answer: TransportAnswer,
    /// Identity of the structural report the estimate was made over.
    pub report_identity: [u8; 32],
    /// Members that were consumed, by their index in the report's envelope.
    pub members: Vec<usize>,
}

/// Replay an exported estimate: consume every member's transport grid with its embedded proof
/// and tables, recompute the mean at every level, check the stored means, expression-free
/// summary and identity, and return the recomputed answer.
///
/// # Errors
///
/// A malformed artifact, a member grid that does not consume, a mean or an answer that does not
/// replay, or an identity that does not match.
pub fn verify_discovered_transport_estimate(
    bytes: &[u8],
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
) -> Result<VerifiedEstimate, IoError> {
    let mismatch = |what: &str| IoError::Convert(format!("discovered transport estimate: {what}"));
    let wire: EstimateWire = antecedent_io::from_cbor(bytes)?;
    if wire.format != ESTIMATE_FORMAT {
        return Err(mismatch("unsupported format"));
    }
    let report_identity: [u8; 32] =
        wire.report_identity.as_slice().try_into().map_err(|_| mismatch("report identity"))?;
    let stored: [u8; 32] =
        wire.identity.as_slice().try_into().map_err(|_| mismatch("estimate identity"))?;
    let mut outcomes = Vec::with_capacity(wire.members.len());
    for member in &wire.members {
        let (_, result) =
            PreparedStudy::<TransportGridState>::consume(&member.artifact, limits, ctx)?;
        let outcome = VariableId::from_raw(member.outcome);
        let means = result
            .points()
            .iter()
            .map(|point| {
                point
                    .distribution()
                    .ok_or_else(|| mismatch("a member grid has an unavailable point"))?
                    .mean(outcome)
                    .map_err(|e| mismatch(&e.to_string()))
            })
            .collect::<Result<Vec<f64>, _>>()?;
        if means != member.means {
            return Err(mismatch("a member mean does not replay"));
        }
        outcomes.push(MemberOutcome::Estimated(MemberEstimate {
            expression: member.expression.clone(),
            contrast: contrast_of(&means),
            means,
            contrast_interval: None,
            artifact: Vec::new(),
            outcome,
        }));
    }
    let answer = summarise(&wire.levels, &outcomes, None)
        .map_err(|_| mismatch("the members do not summarise"))?;
    if answer.kind.label() != wire.kind {
        return Err(mismatch("the answer kind does not replay"));
    }
    let targets: Vec<VariableId> =
        wire.selection_targets.iter().copied().map(VariableId::from_raw).collect();
    let rebuilt = identity_of(&report_identity, &wire.levels, &targets, &wire, &answer);
    if rebuilt != stored {
        return Err(mismatch("the identity does not replay"));
    }
    Ok(VerifiedEstimate {
        answer,
        report_identity,
        members: wire.members.iter().map(|m| m.index as usize).collect(),
    })
}

fn contrast_of(means: &[f64]) -> Option<f64> {
    (means.len() >= 2).then(|| means[means.len() - 1] - means[0])
}

fn identity_of(
    report: &[u8; 32],
    levels: &[f64],
    targets: &[VariableId],
    wire: &EstimateWire,
    answer: &TransportAnswer,
) -> [u8; 32] {
    let mut writer = CanonicalWriter::new(IDENTITY_TAG);
    writer.bytes(report).word(levels.len() as u64);
    for level in levels {
        writer.word(level.to_bits());
    }
    writer.word(targets.len() as u64);
    for target in targets {
        writer.word(u64::from(target.raw()));
    }
    writer.word(wire.members.len() as u64);
    for member in &wire.members {
        writer.word(u64::from(member.index)).text(&member.expression);
        writer.word(u64::from(member.outcome)).word(member.means.len() as u64);
        for mean in &member.means {
            writer.word(mean.to_bits());
        }
        writer.bytes(&member.artifact_digest());
    }
    writer.text(answer.kind.label());
    for level in &answer.levels {
        writer.word(level.low.to_bits()).word(level.high.to_bits());
    }
    writer.finish()
}

impl MemberWire {
    fn artifact_digest(&self) -> [u8; 32] {
        let mut writer = CanonicalWriter::new(IDENTITY_TAG);
        writer.bytes(&self.artifact);
        writer.finish()
    }
}

/// Relative tolerance under which two member estimates are the same number.
const AGREEMENT: f64 = 1e-12;

fn same_number(a: f64, b: f64) -> bool {
    (a - b).abs() <= AGREEMENT * (1.0 + a.abs().max(b.abs()))
}

/// Summarise the members' outcomes. `Err` carries the members that block a class-wide answer.
fn summarise(
    levels: &[f64],
    members: &[MemberOutcome],
    interval: Option<(f64, f64, f64)>,
) -> Result<TransportAnswer, Vec<(usize, String)>> {
    let mut blocked = Vec::new();
    let mut estimates = Vec::new();
    for (index, member) in members.iter().enumerate() {
        match member {
            MemberOutcome::Estimated(estimate) => estimates.push(estimate),
            MemberOutcome::Unidentified { verdict } => blocked.push((index, (*verdict).to_owned())),
            MemberOutcome::Unestimable { detail } => blocked.push((index, detail.clone())),
        }
    }
    if !blocked.is_empty() {
        return Err(blocked);
    }
    let per_level = |level: usize| {
        let values: Vec<f64> = estimates.iter().map(|e| e.means[level]).collect();
        let low = values.iter().copied().fold(f64::INFINITY, f64::min);
        let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (low, high)
    };
    let answers: Vec<LevelAnswer> = levels
        .iter()
        .enumerate()
        .map(|(i, level)| {
            let (low, high) = per_level(i);
            LevelAnswer { level: *level, low, high }
        })
        .collect();
    let expressions_agree = estimates.windows(2).all(|w| w[0].expression == w[1].expression);
    let numerically_equal = answers.iter().all(|a| same_number(a.low, a.high));
    let kind = if expressions_agree || numerically_equal {
        AnswerKind::Point
    } else {
        AnswerKind::StructuralRange
    };
    let contrasts: Vec<f64> = estimates.iter().filter_map(|e| e.contrast).collect();
    let contrast = (!contrasts.is_empty()).then(|| {
        (
            contrasts.iter().copied().fold(f64::INFINITY, f64::min),
            contrasts.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    });
    let (interval, interval_reason) = match (kind, interval, expressions_agree) {
        (AnswerKind::StructuralRange, _, _) => (None, Some(RANGE_HAS_NO_INTERVAL)),
        (AnswerKind::Point, Some((low, high, coverage_target)), true) => (
            Some(ContrastInterval {
                low,
                high,
                coverage_target,
                calibration: "not_bound_to_this_execution",
            }),
            None,
        ),
        (AnswerKind::Point, _, _) => (None, Some(NO_INTERVAL_REQUESTED)),
    };
    Ok(TransportAnswer {
        kind,
        levels: answers,
        contrast,
        expressions_agree,
        interval,
        interval_reason,
    })
}

// -- evidence ---------------------------------------------------------------------------------

/// One population's rows as integer-coded columns.
pub(crate) fn columns_of(
    data: &TabularData,
    variables: &[VariableId],
) -> Result<Columns, DiscoveredTransportError> {
    let sample = RegimeSample::from_tabular(
        "evidence",
        RegimeId::from_raw(0),
        "evidence",
        Vec::<InterventionAssignment>::new(),
        data,
        variables,
    )
    .map_err(|error| invalid_evidence(format!("a column cannot be read: {error}")))?;
    Ok(sample.columns)
}

fn invalid_evidence(message: String) -> DiscoveredTransportError {
    DiscoveredTransportError::Refused(DiscoveredTransportRefusal {
        code: antecedent_core::reason_code!("invalid_argument"),
        detail: "discovered_transport.evidence_invalid",
        unresolved: Vec::new(),
        exhausted_search_bounds: 0,
        unknown_selection_targets: Vec::new(),
        member_reasons: Vec::new(),
        remedy: message,
    })
}

/// This two-table adapter supplies one treatment trial, not an arbitrary experiment catalog.
pub(crate) fn validate_evidence_query(
    query: &TransportQuery,
) -> Result<(), DiscoveredTransportError> {
    let Some((treatment, _)) = query.response.functional.primary_pair() else {
        return Err(invalid_evidence("the query names no treatment".to_owned()));
    };
    if query.source_experiments.as_ref() != [treatment] || query.catalog.is_some() {
        return Err(invalid_evidence(
            "the two-table evidence route requires exactly the declared treatment experiment; \
             use an explicit transport grid for other experimental regimes or a supplied catalog"
                .to_owned(),
        ));
    }
    Ok(())
}

/// Source and target columns the members are estimated from.
pub(crate) struct EvidenceColumns {
    pub(crate) source: Columns,
    pub(crate) target: Columns,
}

/// What one member's estimation needs besides the evidence.
pub(crate) struct MemberProblem {
    pub(crate) graph: Admg,
    pub(crate) selection: Arc<[VariableId]>,
    pub(crate) treatment: VariableId,
    pub(crate) outcome: VariableId,
}

/// The estimation settings every member shares.
pub(crate) struct Settings<'a> {
    pub(crate) levels: &'a [f64],
    pub(crate) source: &'a Arc<str>,
    pub(crate) target: &'a Arc<str>,
    pub(crate) evidence: &'a TransportEvidence<'a>,
}

fn domain_cardinalities(
    columns: &EvidenceColumns,
    variables: &BTreeSet<VariableId>,
) -> Result<BTreeMap<VariableId, u32>, DiscoveredTransportError> {
    let mut out = BTreeMap::new();
    for variable in variables {
        let mut max = 0.0_f64;
        for (name, table) in [("source", &columns.source), ("target", &columns.target)] {
            let column = table.get(variable).ok_or_else(|| {
                invalid_evidence(format!(
                    "the {name} table has no column for variable v{}",
                    variable.raw()
                ))
            })?;
            for value in column {
                let value = value.ok_or_else(|| {
                    invalid_evidence(format!(
                        "variable v{} has a missing value in the {name} table; this route needs \
                         complete rows",
                        variable.raw()
                    ))
                })?;
                if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
                    return Err(invalid_evidence(format!(
                        "variable v{} has a value that is not a non-negative integer code in \
                         the {name} table",
                        variable.raw()
                    )));
                }
                max = max.max(value);
            }
        }
        // Domains are at most `MAX_LEVELS_PER_VARIABLE`, which fits a `u32` exactly.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let levels = max as u32 + 1;
        if levels > MAX_LEVELS_PER_VARIABLE {
            return Err(invalid_evidence(format!(
                "variable v{} takes {levels} levels; at most {MAX_LEVELS_PER_VARIABLE} are \
                 supported",
                variable.raw()
            )));
        }
        out.insert(*variable, levels.max(2));
    }
    Ok(out)
}

fn coordinate(variable: VariableId, cardinality: u32) -> VariableCoordinate {
    VariableCoordinate {
        variable,
        domain: if cardinality == 2 {
            VariableDomain::Binary
        } else {
            VariableDomain::Categorical { cardinality }
        },
        unit: None,
    }
}

fn snapshot_identity(
    population: &str,
    regime: RegimeId,
    variables: &[VariableId],
    columns: &Columns,
) -> String {
    let mut writer = CanonicalWriter::new("antecedent.discovered_transport.snapshot.v1");
    writer.text(population).word(u64::from(regime.raw())).word(variables.len() as u64);
    for variable in variables {
        writer.word(u64::from(variable.raw()));
        for value in &columns[variable] {
            match value {
                Some(value) => writer.word(1).word(value.to_bits()),
                None => writer.word(0),
            };
        }
    }
    writer.finish_hex()
}

/// A regime the derived formula needs evidence for.
struct RegimePlan {
    id: RegimeId,
    population: Arc<str>,
    interventions: Vec<VariableId>,
    measured: Vec<VariableId>,
}

/// Actual supplied regimes: the target observational sample and the source treatment trial.
/// A formula's requests never establish the existence of additional experiments.
fn plan_regimes(
    treatment: VariableId,
    columns: &EvidenceColumns,
    settings: &Settings<'_>,
) -> Vec<RegimePlan> {
    vec![
        RegimePlan {
            id: RegimeId::from_raw(0),
            population: Arc::clone(settings.target),
            interventions: Vec::new(),
            measured: columns.target.keys().copied().collect(),
        },
        RegimePlan {
            id: RegimeId::from_raw(1),
            population: Arc::clone(settings.source),
            interventions: vec![treatment],
            measured: columns.source.keys().copied().collect(),
        },
    ]
}

fn catalog_of(
    plans: &[RegimePlan],
    cardinalities: &BTreeMap<VariableId, u32>,
    diagram: &SelectionDiagram,
    settings: &Settings<'_>,
    columns: &EvidenceColumns,
) -> Result<EvidenceCatalog, DiscoveredTransportError> {
    let invalid = |error: antecedent_core::QueryError| invalid_evidence(error.to_string());
    let populations: BTreeSet<&Arc<str>> = plans.iter().map(|p| &p.population).collect();
    let mut environments = Vec::new();
    for population in populations {
        let variables: BTreeSet<VariableId> = plans
            .iter()
            .filter(|p| &p.population == population)
            .flat_map(|p| p.measured.iter().copied())
            .collect();
        let selection: Vec<VariableId> = if population == settings.source {
            diagram.selection_targets().to_vec()
        } else {
            Vec::new()
        };
        environments.push(
            Environment::try_new(
                Arc::clone(population),
                variables.iter().map(|v| coordinate(*v, cardinalities[v])).collect::<Vec<_>>(),
                selection,
            )
            .map_err(invalid)?,
        );
    }
    let mut regimes = Vec::new();
    let mut bindings = Vec::new();
    let (sampling, dependence) = if settings.evidence.rows_independent {
        (SamplingDesign::Independent, DependenceGroup::IndependentStudies)
    } else {
        (SamplingDesign::Unknown, DependenceGroup::UnknownDependence)
    };
    for plan in plans {
        let kind = if plan.interventions.is_empty() {
            RegimeKind::Observational
        } else {
            RegimeKind::Experimental
        };
        regimes.push(
            EvidenceRegime::try_new(
                plan.id,
                kind,
                EvidenceKind::Available,
                plan.interventions.clone(),
                Vec::<antecedent_core::InterventionAssignment>::new(),
                plan.measured.clone(),
                Arc::clone(&plan.population),
                DistributionAvailability::Joint,
            )
            .map_err(invalid)?,
        );
        let table =
            if &plan.population == settings.source { &columns.source } else { &columns.target };
        bindings.push(RegimeBinding {
            dataset_identity: None,
            regime: plan.id,
            snapshot_identity: Arc::from(snapshot_identity(
                &plan.population,
                plan.id,
                &plan.measured,
                table,
            )),
            schema_names: Arc::from([]),
            sampling,
            weights: None,
            dependence,
        });
    }
    EvidenceCatalog::try_new(
        environments,
        regimes,
        bindings,
        Some(settings.evidence.target_sampling),
    )
    .map_err(invalid)
}

/// One sample per experimental world of each regime, from the population's rows.
fn samples_of(
    plans: &[RegimePlan],
    catalog: &EvidenceCatalog,
    settings: &Settings<'_>,
    columns: &EvidenceColumns,
) -> Result<Vec<RegimeSample>, DiscoveredTransportError> {
    let mut samples = Vec::new();
    for plan in plans {
        let table =
            if &plan.population == settings.source { &columns.source } else { &columns.target };
        let snapshot = catalog
            .bindings
            .iter()
            .find(|b| b.regime == plan.id)
            .map(|b| Arc::clone(&b.snapshot_identity))
            .ok_or_else(|| invalid_evidence("a regime has no binding".to_owned()))?;
        let axes: Vec<VariableId> =
            plan.measured.iter().copied().filter(|v| !plan.interventions.contains(v)).collect();
        let rows = table.values().next().map_or(0, Vec::len);
        // Rows of each experimental world, keyed by the assigned levels (one empty key for an
        // observational regime).
        let mut worlds: BTreeMap<Vec<u64>, Vec<usize>> = BTreeMap::new();
        for row in 0..rows {
            let key = plan
                .interventions
                .iter()
                .map(|v| table[v][row].map_or(0, f64::to_bits))
                .collect::<Vec<_>>();
            worlds.entry(key).or_default().push(row);
        }
        for (key, rows) in worlds {
            let interventions: Vec<InterventionAssignment> = plan
                .interventions
                .iter()
                .zip(&key)
                .map(|(v, bits)| {
                    InterventionAssignment::concrete(*v, Value::f64(f64::from_bits(*bits)))
                })
                .collect();
            samples.push(RegimeSample {
                population: Arc::clone(&plan.population),
                regime: plan.id,
                snapshot_identity: Arc::clone(&snapshot),
                interventions: interventions.into(),
                columns: axes
                    .iter()
                    .map(|v| (*v, rows.iter().map(|row| table[v][*row]).collect()))
                    .collect(),
            });
        }
    }
    Ok(samples)
}

fn estimation_failure(error: &impl std::fmt::Display) -> DiscoveredTransportError {
    DiscoveredTransportError::Estimation(error.to_string())
}

/// Whether an artifact-boundary error is the evidence not covering the formula (a missing
/// provider or a support failure) rather than a failure of the computation.
fn is_evidence_gap(error: &IoError) -> bool {
    matches!(error.reason_code(), Some("transport_missing_provider" | "transport_support_failure"))
}

/// A member whose effect the complete sID derived, ready to bind to evidence.
struct Identified {
    diagram: SelectionDiagram,
    // Identification is conditional on the fixed graph and declared available regimes.
    // Resampled rows change empirical laws and snapshot bindings, not this causal proof.
    chosen_derivation: RefCell<Option<antecedent_identify::ClassicalTransportDerivation>>,
}

/// Run the complete sID on one member, or say why it did not identify.
fn identify_member(
    problem: &MemberProblem,
    settings: &Settings<'_>,
    ctx: &ExecutionContext,
) -> Result<Result<Identified, MemberOutcome>, DiscoveredTransportError> {
    let invalid = |error: antecedent_graph::GraphError| estimation_failure(&error);
    let diagram = SelectionDiagram::try_new(problem.graph.clone(), problem.selection.to_vec())
        .map_err(invalid)?;
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([problem.outcome]),
        treatments: Arc::from([problem.treatment]),
        source: Arc::clone(settings.source),
        target: Arc::clone(settings.target),
    };
    Ok(
        match identify_classical_transport(&diagram, &query, settings.evidence.sid_limits, ctx)
            .map_err(|e| estimation_failure(&e))?
        {
            ClassicalTransportResult::Identified(_) => {
                Ok(Identified { diagram, chosen_derivation: RefCell::new(None) })
            }
            ClassicalTransportResult::NotCertified => {
                Err(MemberOutcome::Unidentified { verdict: "not_certified" })
            }
            ClassicalTransportResult::ProvenNonTransportable(_) => {
                Err(MemberOutcome::Unidentified { verdict: "proven_non_transportable" })
            }
        },
    )
}

/// Identify and estimate one member.
pub(crate) fn estimate_member(
    problem: &MemberProblem,
    columns: &EvidenceColumns,
    settings: &Settings<'_>,
    ctx: &ExecutionContext,
) -> Result<MemberOutcome, DiscoveredTransportError> {
    let identified = match identify_member(problem, settings, ctx)? {
        Ok(identified) => identified,
        Err(outcome) => return Ok(outcome),
    };
    let replicates = if settings.evidence.block_length.is_some() {
        0
    } else {
        settings.evidence.bootstrap_replicates
    };
    evaluate_member(&identified, problem, columns, settings, replicates, true, ctx)
}

/// Bind an identified member's formula to the evidence and evaluate it at every level.
///
/// `replicates` is the iid row bootstrap of the retained grid; `export` keeps the durable grid
/// artifact.
fn evaluate_member(
    identified: &Identified,
    problem: &MemberProblem,
    columns: &EvidenceColumns,
    settings: &Settings<'_>,
    replicates: u32,
    export: bool,
    ctx: &ExecutionContext,
) -> Result<MemberOutcome, DiscoveredTransportError> {
    let Identified { diagram, chosen_derivation } = identified;
    let plans = plan_regimes(problem.treatment, columns, settings);
    let variables: BTreeSet<VariableId> =
        plans.iter().flat_map(|p| p.measured.iter().copied()).collect();
    let cardinalities = domain_cardinalities(columns, &variables)?;
    let catalog = catalog_of(&plans, &cardinalities, diagram, settings, columns)?;
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([problem.outcome]),
        treatments: Arc::from([problem.treatment]),
        source: Arc::clone(settings.source),
        target: Arc::clone(settings.target),
    };
    let cached = chosen_derivation.borrow();
    let selected = if let Some(derivation) = cached.as_ref() {
        match derivation.bind_catalog(&catalog) {
            Ok(functional) => CatalogTransportResult::Identified(Box::new(functional)),
            Err(error) => CatalogTransportResult::MissingEvidence {
                searched: Arc::from([Arc::from("fixed_causal_derivation_rebound")]),
                obligations: Arc::from([Arc::from(error.to_string())]),
            },
        }
    } else {
        identify_catalog_transport_standardization_first(
            diagram,
            &query,
            &catalog,
            settings.evidence.sid_limits,
            ctx,
        )
        .map_err(|error| estimation_failure(&error))?
    };
    drop(cached);
    let functional = match selected {
        CatalogTransportResult::Identified(functional) => *functional,
        CatalogTransportResult::MissingEvidence { obligations, .. }
        | CatalogTransportResult::NotCertified { obligations, .. } => {
            return Ok(MemberOutcome::Unestimable {
                detail: format!(
                    "no formula binds to the supplied treatment experiment and target \
                     observational sample: {obligations:?}"
                ),
            });
        }
    };
    if chosen_derivation.borrow().is_none() {
        *chosen_derivation.borrow_mut() = Some(functional.derivation().clone());
    }
    let samples = samples_of(&plans, &catalog, settings, columns)?;
    let treatment_cardinality = cardinalities.get(&problem.treatment).copied().unwrap_or(2);
    let at = settings
        .levels
        .iter()
        .map(|level| {
            if level.fract() == 0.0 && *level >= 0.0 && *level < f64::from(treatment_cardinality) {
                // A level is a small non-negative integer code, checked just above.
                #[allow(clippy::cast_possible_truncation)]
                Ok(Assignment::from_pairs([(problem.treatment, Value::Int64(*level as i64))]))
            } else {
                Err(invalid_evidence(format!(
                    "treatment level {level} is not a code of the treatment's domain \
                     0..{treatment_cardinality}"
                )))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expression = functional.arena().pretty(functional.root());
    let options = EmpiricalTableOptions {
        bootstrap_replicates: replicates,
        ..EmpiricalTableOptions::default()
    };
    let grid = StudyBuilder::transport_grid(
        TransportGridQuery {
            diagram: diagram.clone(),
            functional,
            at,
            limits: settings.evidence.limits,
        },
        TransportGridData::Statistical(
            StatisticalTransportInput { supplied: Vec::new(), samples },
            options,
        ),
        ctx,
    );
    let result = match grid.and_then(|study| study.estimate(ctx)) {
        Ok(result) => result,
        Err(error) if is_evidence_gap(&error) => {
            return Ok(MemberOutcome::Unestimable { detail: error.to_string() });
        }
        Err(error) => return Err(estimation_failure(&error)),
    };
    member_estimate(&result, problem.outcome, expression, export)
}

fn member_estimate(
    result: &TransportGridResult,
    outcome: VariableId,
    expression: String,
    export: bool,
) -> Result<MemberOutcome, DiscoveredTransportError> {
    let mut means = Vec::with_capacity(result.points().len());
    for (index, point) in result.points().iter().enumerate() {
        match point {
            TransportGridPoint::Unavailable(failure) => {
                return Ok(MemberOutcome::Unestimable {
                    detail: format!("grid level {index} is unavailable: {}", failure.detail),
                });
            }
            other => {
                let distribution =
                    other.distribution().ok_or_else(|| estimation_failure(&"missing law"))?;
                means.push(distribution.mean(outcome).map_err(|e| estimation_failure(&e))?);
            }
        }
    }
    let last = means.len().saturating_sub(1);
    let contrast_interval = if means.len() >= 2 {
        result
            .contrast(last, 0, outcome)
            .ok()
            .and_then(|c| c.interval.map(|(low, high)| (low, high, c.coverage_target)))
    } else {
        None
    };
    let artifact =
        if export { result.export().map_err(|e| estimation_failure(&e))? } else { Vec::new() };
    Ok(MemberOutcome::Estimated(MemberEstimate {
        expression,
        contrast: contrast_of(&means),
        means,
        contrast_interval: contrast_interval.map(|(low, high, _)| (low, high)),
        artifact,
        outcome,
    }))
}

/// Rows of every column at the drawn time indices.
fn gathered(table: &Columns, times: &[u32]) -> Columns {
    table
        .iter()
        .map(|(variable, column)| {
            (*variable, times.iter().map(|t| column[*t as usize]).collect::<Vec<_>>())
        })
        .collect()
}

/// A pointwise percentile interval on the `last - first` contrast of a member from a circular
/// block bootstrap of the time axis.
///
/// Each replicate draws blocks of `length` consecutive time indices once per population and
/// takes every column, and so every arm of an experiment, at those indices: contiguous in time,
/// and arms that share a time axis are resampled jointly. The formula is re-evaluated on the
/// resampled evidence; a replicate that cannot be evaluated is counted as failed and the
/// interval is withheld unless the replicate floor is met.
fn blocked_contrast_interval(
    identified: &Identified,
    problem: &MemberProblem,
    columns: &EvidenceColumns,
    settings: &Settings<'_>,
    length: u32,
    ctx: &ExecutionContext,
) -> Result<Option<(f64, f64, f64)>, DiscoveredTransportError> {
    // Freeze the checked empirical functional on the original evidence before any draw.
    // Only its catalog bindings and empirical laws are rebuilt on resampled rows.
    if !matches!(
        evaluate_member(identified, problem, columns, settings, 0, false, ctx)?,
        MemberOutcome::Estimated(_)
    ) {
        return Ok(None);
    }
    let requested = settings.evidence.bootstrap_replicates;
    let rows = |table: &Columns| table.values().next().map_or(0, Vec::len);
    let plan = antecedent_data::ResamplingPlan::CircularBlock { length: length as usize };
    let draw = |population: u64, replicate: u32, n: usize| -> Option<Vec<u32>> {
        let mut rng = ctx.rng.stream_for(
            antecedent_core::StreamDomain::Transport,
            (0x626c_6f63_u64 << 32) | (population << 24) | u64::from(replicate),
        );
        let mut times = Vec::new();
        antecedent_data::fill_resample_indexes(plan, n, &mut rng, &mut times).ok()?;
        Some(times)
    };
    let mut contrasts = Vec::new();
    for replicate in 0..requested {
        if ctx.cancellation.is_cancelled() {
            return Err(DiscoveredTransportError::Estimation("cancelled".to_owned()));
        }
        let (Some(source_times), Some(target_times)) =
            (draw(0, replicate, rows(&columns.source)), draw(1, replicate, rows(&columns.target)))
        else {
            continue;
        };
        let resampled = EvidenceColumns {
            source: gathered(&columns.source, &source_times),
            target: gathered(&columns.target, &target_times),
        };
        if let MemberOutcome::Estimated(estimate) =
            evaluate_member(identified, problem, &resampled, settings, 0, false, ctx)?
        {
            if let Some(contrast) = estimate.contrast.filter(|c| c.is_finite()) {
                contrasts.push(contrast);
            }
        }
    }
    let ok = u32::try_from(contrasts.len()).unwrap_or(u32::MAX);
    let failed = requested.saturating_sub(ok);
    if super::transport_common::PERCENTILE_FLOOR.decide(requested, ok, failed).is_err() {
        return Ok(None);
    }
    let level = EmpiricalTableOptions::default().coverage_level;
    Ok(antecedent_estimate::percentile_interval(&contrasts, level)
        .map(|(low, high)| (low, high, level)))
}

/// What the structural report contributes to an estimate.
pub(crate) struct ClassFacts<'a> {
    pub(crate) report_identity: [u8; 32],
    pub(crate) selection_targets: &'a [VariableId],
    pub(crate) unresolved: &'a [UnresolvedPair],
    pub(crate) status: EnvelopeStatus,
}

/// Run every member and summarise, or refuse.
#[allow(
    clippy::too_many_lines,
    reason = "keep class-wide agreement, deferred shared inference and complete refusal accounting in one pipeline"
)]
pub(crate) fn estimate_class(
    class: &ClassFacts<'_>,
    problems: &[MemberProblem],
    columns: &EvidenceColumns,
    settings: &Settings<'_>,
    ctx: &ExecutionContext,
) -> Result<DiscoveredTransportEstimate, DiscoveredTransportError> {
    let ClassFacts { report_identity, selection_targets, unresolved, status } = *class;
    let refuse = |detail: &'static str, member_reasons: Vec<(usize, String)>, remedy: String| {
        DiscoveredTransportError::Refused(DiscoveredTransportRefusal {
            code: DISCOVERED_TRANSPORT_REASON,
            detail,
            unresolved: unresolved.to_vec(),
            exhausted_search_bounds: 0,
            unknown_selection_targets: Vec::new(),
            member_reasons,
            remedy,
        })
    };
    if status == EnvelopeStatus::Partial {
        return Err(refuse(
            "discovered_transport.class_enumeration_partial",
            Vec::new(),
            "the enumeration of the class did not finish, so no class-wide answer is made"
                .to_owned(),
        ));
    }
    // Establish the class-wide point answer before paying for sampling inference. A class
    // whose members disagree or cannot all be estimated cannot publish this interval, and
    // identical expressions need only one bootstrap of their shared empirical functional.
    let point_evidence = TransportEvidence { bootstrap_replicates: 0, ..*settings.evidence };
    let point_settings = Settings {
        levels: settings.levels,
        source: settings.source,
        target: settings.target,
        evidence: &point_evidence,
    };
    let mut members = Vec::with_capacity(problems.len());
    for problem in problems {
        members.push(estimate_member(problem, columns, &point_settings, ctx)?);
    }
    let common_expression = members.first().and_then(|first| match first {
        MemberOutcome::Estimated(estimate) => Some(estimate.expression.as_str()),
        _ => None,
    });
    let agree = !members.is_empty()
        && common_expression.is_some()
        && members.iter().all(|member| match member {
            MemberOutcome::Estimated(estimate) => {
                Some(estimate.expression.as_str()) == common_expression
            }
            _ => false,
        });
    let mut interval = None;
    if settings.evidence.block_length.is_none()
        && settings.evidence.bootstrap_replicates > 0
        && agree
    {
        let bootstrapped = estimate_member(&problems[0], columns, settings, ctx)?;
        if let MemberOutcome::Estimated(estimate) = &bootstrapped {
            interval = estimate.contrast_interval;
        }
        members[0] = bootstrapped;
        for member in &mut members[1..] {
            if let MemberOutcome::Estimated(estimate) = member {
                estimate.contrast_interval = interval;
            }
        }
    }
    if let Some(length) = settings.evidence.block_length {
        // Ordered rows: the interval is the block bootstrap of the time axis, for a point only.
        interval = None;
        let estimated: Vec<(usize, &MemberEstimate)> = members
            .iter()
            .enumerate()
            .filter_map(|(i, m)| match m {
                MemberOutcome::Estimated(e) => Some((i, e)),
                _ => None,
            })
            .collect();
        let agree = estimated.len() == members.len()
            && estimated.windows(2).all(|w| w[0].1.expression == w[1].1.expression);
        if settings.evidence.bootstrap_replicates > 0 && agree {
            if let Some((index, _)) = estimated.first() {
                if let Ok(identified) = identify_member(&problems[*index], settings, ctx)? {
                    interval = blocked_contrast_interval(
                        &identified,
                        &problems[*index],
                        columns,
                        settings,
                        length,
                        ctx,
                    )?
                    .map(|(low, high, _)| (low, high));
                }
            }
        }
    }
    let coverage = EmpiricalTableOptions::default().coverage_level;
    let answer = match summarise(
        settings.levels,
        &members,
        interval.map(|(low, high)| (low, high, coverage)),
    ) {
        Ok(answer) => answer,
        Err(blocked) => {
            let unidentified =
                members.iter().any(|m| matches!(m, MemberOutcome::Unidentified { .. }));
            let (detail, remedy) = if unidentified {
                (
                    "discovered_transport.class_not_identified",
                    "the effect is not identified in every member of the class; the members it \
                     is not identified in are named, and no point or range is offered",
                )
            } else {
                (
                    "discovered_transport.member_not_estimable",
                    "a member is identified but the evidence cannot supply a factor its formula \
                     needs",
                )
            };
            return Err(refuse(detail, blocked, remedy.to_owned()));
        }
    };
    let mut estimate = DiscoveredTransportEstimate {
        levels: settings.levels.to_vec(),
        members,
        answer,
        report_identity,
        selection_targets: selection_targets.to_vec(),
        identity: [0; 32],
    };
    let wire = estimate_wire_for_identity(&estimate);
    estimate.identity = identity_of(
        &estimate.report_identity,
        &estimate.levels,
        &estimate.selection_targets,
        &wire,
        &estimate.answer,
    );
    Ok(estimate)
}

fn estimate_wire_for_identity(estimate: &DiscoveredTransportEstimate) -> EstimateWire {
    EstimateWire {
        format: ESTIMATE_FORMAT.into(),
        report_identity: Vec::new(),
        selection_targets: Vec::new(),
        levels: Vec::new(),
        members: estimate
            .members
            .iter()
            .enumerate()
            .filter_map(|(index, member)| match member {
                MemberOutcome::Estimated(e) => Some(MemberWire {
                    index: u32::try_from(index).unwrap_or(u32::MAX),
                    expression: e.expression.clone(),
                    outcome: e.outcome.raw(),
                    means: e.means.clone(),
                    artifact: e.artifact.clone(),
                }),
                _ => None,
            })
            .collect(),
        kind: String::new(),
        identity: Vec::new(),
    }
}

// -- the static entry -------------------------------------------------------------------------

/// Estimate `query` across the class `report` was made over, from source and target tables.
///
/// The treatment grid of `query` gives the levels; the outcome and treatment are its mean-curve
/// variables. Columns are read by variable id, so the tables carry one column per variable of
/// the structure, in the order the discovery read them.
///
/// # Errors
///
/// [`DiscoveredTransportError::Refused`] when some member is not identified
/// (`discovered_transport.class_not_identified`), some member cannot be estimated from the
/// evidence (`discovered_transport.member_not_estimable`), the enumeration did not finish
/// (`discovered_transport.class_enumeration_partial`), or the tables are not usable
/// (`discovered_transport.evidence_invalid`); [`DiscoveredTransportError::Estimation`] for a
/// failure of the computation, a resource bound or a cancellation.
pub fn estimate_discovered_transport(
    report: &DiscoveredTransportReport,
    query: &TransportQuery,
    evidence: &TransportEvidence<'_>,
    ctx: &ExecutionContext,
) -> Result<DiscoveredTransportEstimate, DiscoveredTransportError> {
    estimate_selected_discovered_transport(report, query, evidence, None, None, ctx)
}

pub(crate) fn estimate_selected_discovered_transport(
    report: &DiscoveredTransportReport,
    query: &TransportQuery,
    evidence: &TransportEvidence<'_>,
    selected: Option<&[usize]>,
    refined_identity: Option<[u8; 32]>,
    ctx: &ExecutionContext,
) -> Result<DiscoveredTransportEstimate, DiscoveredTransportError> {
    validate_evidence_query(query)?;
    let envelope = report.envelope();
    let antecedent_core::ResponseFunctional::MeanCurve { outcome, treatment } =
        &query.response.functional
    else {
        return Err(invalid_evidence(
            "an estimate asks for the mean response curve of one outcome under one treatment"
                .to_owned(),
        ));
    };
    let antecedent_core::GridSpec::Values(levels) = &treatment.grid else {
        return Err(invalid_evidence("the treatment grid must list its levels".to_owned()));
    };
    let variables = envelope.variables();
    let columns = EvidenceColumns {
        source: columns_of(evidence.source, variables)?,
        target: columns_of(evidence.target, variables)?,
    };
    let mut problems = Vec::with_capacity(envelope.members().len());
    for index in 0..envelope.members().len() {
        if selected.is_some_and(|members| !members.contains(&index)) {
            continue;
        }
        let graph = envelope
            .member_graph(index)
            .ok_or_else(|| estimation_failure(&"a member graph cannot be rebuilt"))?;
        let graph = ancestral_subgraph(&graph, &[treatment.variable, *outcome])
            .map_err(|e| estimation_failure(&e))?;
        let selection: Arc<[VariableId]> = report
            .selection_targets()
            .iter()
            .copied()
            .filter(|v| graph.dense_of_static(*v).is_some())
            .collect();
        problems.push(MemberProblem {
            graph,
            selection,
            treatment: treatment.variable,
            outcome: *outcome,
        });
    }
    let settings = Settings {
        levels,
        source: &query.source_population,
        target: &query.target_population,
        evidence,
    };
    let class = ClassFacts {
        report_identity: refined_identity.unwrap_or(*report.identity()),
        selection_targets: report.selection_targets(),
        unresolved: envelope.unresolved(),
        status: envelope.status(),
    };
    estimate_class(&class, &problems, &columns, &settings, ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_graph::DenseNodeId;

    #[test]
    fn a_treatment_trial_standardizes_before_an_unsupported_target_joint() {
        let mut source_rows = [Vec::new(), Vec::new(), Vec::new()];
        for x in 0..2 {
            for z in 0..2 {
                for draw in 0..10 {
                    source_rows[0].push(f64::from(x));
                    source_rows[1].push(f64::from(draw < 1 + 4 * x + 3 * z));
                    source_rows[2].push(f64::from(z));
                }
            }
        }
        let target_rows =
            [vec![0.0; 10], vec![0.0; 10], vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]];
        let table = |rows: &[Vec<f64>; 3]| {
            TabularData::from_f64_columns([
                ("x", rows[0].as_slice()),
                ("y", rows[1].as_slice()),
                ("z", rows[2].as_slice()),
            ])
            .unwrap()
        };
        let source = table(&source_rows);
        let target = table(&target_rows);
        let evidence = TransportEvidence::new(&source, &target);
        let population = Arc::from("source");
        let target_population = Arc::from("target");
        let settings = Settings {
            levels: &[0.0, 1.0],
            source: &population,
            target: &target_population,
            evidence: &evidence,
        };
        let ids = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
        let mut graph = Admg::with_variables(3);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
        // Target rows contain no X=1 observations. A target-first conditional is structurally
        // identified but cannot estimate that cell; the available treatment trial can.
        let problem = MemberProblem {
            graph,
            selection: Arc::from([ids[2]]),
            treatment: ids[0],
            outcome: ids[1],
        };
        let columns = EvidenceColumns {
            source: columns_of(&source, &ids).unwrap(),
            target: columns_of(&target, &ids).unwrap(),
        };
        let MemberOutcome::Estimated(estimate) =
            estimate_member(&problem, &columns, &settings, &ExecutionContext::for_tests(91))
                .unwrap()
        else {
            panic!(
                "the available treatment trial and target covariate marginal identify the response"
            )
        };
        // Independent standardization oracle: .2*(.1+.4*x) + .8*(.4+.4*x).
        assert!(same_number(estimate.means[0], 0.34));
        assert!(same_number(estimate.means[1], 0.74));
        assert!(!estimate.expression.contains("V2:="));
        assert!(estimate.expression.contains("V0:="));
        let ctx = ExecutionContext::for_tests(91);
        let cached = identify_member(&problem, &settings, &ctx).unwrap().unwrap();
        assert!(matches!(
            evaluate_member(&cached, &problem, &columns, &settings, 0, false, &ctx).unwrap(),
            MemberOutcome::Estimated(_)
        ));
        for offset in [0, 1] {
            // A non-permutation resample changes source empirical probabilities; no fitted
            // laws or row snapshots may be reused merely because the causal graph is fixed.
            let source_indices: Vec<u32> = (0..40).map(|i| (2 * i + offset) % 40).collect();
            let target_indices: Vec<u32> = (0..10).map(|i| (2 * i + offset) % 10).collect();
            let resampled = EvidenceColumns {
                source: gathered(&columns.source, &source_indices),
                target: gathered(&columns.target, &target_indices),
            };
            let MemberOutcome::Estimated(rebound) =
                evaluate_member(&cached, &problem, &resampled, &settings, 0, false, &ctx).unwrap()
            else {
                panic!("rebound sample")
            };
            let MemberOutcome::Estimated(fresh) =
                estimate_member(&problem, &resampled, &settings, &ctx).unwrap()
            else {
                panic!("fresh sample")
            };
            assert_eq!(rebound.means, fresh.means);
            assert_ne!(rebound.means, estimate.means);
        }
    }

    #[test]
    fn supplied_regimes_do_not_promote_observed_covariates_to_experiments() {
        let x = [0.0, 0.0, 1.0, 1.0];
        let z = [0.0, 1.0, 0.0, 1.0];
        let source =
            TabularData::from_f64_columns([("x", x.as_slice()), ("z", z.as_slice())]).unwrap();
        let evidence = TransportEvidence::new(&source, &source);
        let population = Arc::from("source");
        let target = Arc::from("target");
        let settings = Settings {
            levels: &[0.0, 1.0],
            source: &population,
            target: &target,
            evidence: &evidence,
        };
        let treatment = VariableId::from_raw(0);
        let covariate = VariableId::from_raw(1);
        let columns = EvidenceColumns {
            source: columns_of(&source, &[treatment, covariate]).unwrap(),
            target: columns_of(&source, &[treatment, covariate]).unwrap(),
        };
        let plans = plan_regimes(treatment, &columns, &settings);
        let graph = Admg::with_variables(2);
        let diagram = SelectionDiagram::try_new(graph, Vec::new()).unwrap();
        let cardinalities =
            domain_cardinalities(&columns, &BTreeSet::from([treatment, covariate])).unwrap();
        let catalog = catalog_of(&plans, &cardinalities, &diagram, &settings, &columns).unwrap();
        assert_eq!(catalog.regimes.len(), 2);
        let source_regime = catalog.regimes.iter().find(|r| r.population == population).unwrap();
        assert_eq!(source_regime.kind, RegimeKind::Experimental);
        assert_eq!(source_regime.interventions.as_ref(), [treatment]);
        let target_regime = catalog.regimes.iter().find(|r| r.population == target).unwrap();
        assert_eq!(target_regime.kind, RegimeKind::Observational);
        assert!(target_regime.interventions.is_empty());
        let samples = samples_of(&plans, &catalog, &settings, &columns).unwrap();
        for sample in samples.iter().filter(|s| s.population == population) {
            assert_eq!(sample.interventions.len(), 1);
            assert_eq!(sample.interventions[0].variable, treatment);
            assert!(sample.columns.contains_key(&covariate));
        }
    }

    #[test]
    fn a_common_class_bootstrap_matches_one_member_and_keeps_every_proof() {
        let x: Vec<f64> = (0..160).map(|i| f64::from(i % 8 >= 4)).collect();
        let y: Vec<f64> = (0..160).map(|i| f64::from(i % 8 == 0 || i % 8 >= 5)).collect();
        let source =
            TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
        let mut evidence = TransportEvidence::new(&source, &source);
        evidence.bootstrap_replicates = 200;
        let population = Arc::from("source");
        let target = Arc::from("target");
        let settings = Settings {
            levels: &[0.0, 1.0],
            source: &population,
            target: &target,
            evidence: &evidence,
        };
        let mut graph = Admg::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let problem = || MemberProblem {
            graph: graph.clone(),
            selection: Arc::from([]),
            treatment: VariableId::from_raw(0),
            outcome: VariableId::from_raw(1),
        };
        let variables = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let columns = EvidenceColumns {
            source: columns_of(&source, &variables).unwrap(),
            target: columns_of(&source, &variables).unwrap(),
        };
        let ctx = ExecutionContext::for_tests(913);
        let MemberOutcome::Estimated(single) =
            estimate_member(&problem(), &columns, &settings, &ctx).unwrap()
        else {
            panic!("the single member must be estimable");
        };
        let class = ClassFacts {
            report_identity: [0; 32],
            selection_targets: &[],
            unresolved: &[],
            status: EnvelopeStatus::IdentifiedInEveryMember,
        };
        let estimate =
            estimate_class(&class, &[problem(), problem()], &columns, &settings, &ctx).unwrap();
        let interval = estimate.answer().interval.expect("the common functional has an interval");
        assert_eq!(Some((interval.low, interval.high)), single.contrast_interval);
        assert_eq!(estimate.members.len(), 2);
        for member in &estimate.members {
            let MemberOutcome::Estimated(member) = member else {
                panic!("every member must estimate")
            };
            assert_eq!(member.means, single.means);
            assert!(!member.artifact.is_empty());
        }
        let bytes = estimate.export().unwrap();
        let replayed = verify_discovered_transport_estimate(&bytes, evidence.limits, &ctx).unwrap();
        assert_eq!(replayed.answer.levels, estimate.answer().levels);
    }
}
