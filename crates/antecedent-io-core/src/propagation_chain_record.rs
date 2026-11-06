//! The retained propagation record of a chain of transformations, and the
//! independent consumer that re-derives it.
//!
//! Format version 1, feature `propagation_chain_v1`, carried as the optional
//! `analysis_result.propagation` section of a composite artifact or as a
//! standalone CBOR payload. The record binds, in one sealed identity:
//!
//! * the **sources**: how each carries its uncertainty (a point, an interval,
//!   or one column of an aligned draw table), what its draws are a law of, the
//!   alignment identity of the table, and the rows, clusters, models, priors,
//!   disturbances and periods it uses (the dependence contract);
//! * the **stages**: for each, what it does, the operator and its parameters
//!   as IEEE bit patterns, the nodes it reads, and the factors it brings itself;
//! * the future-study draws of any preposterior update;
//! * the **changes**, appended in order: a source withdrawn, or the rows of a
//!   source re-weighted because an assumption about it changed;
//! * the stored **results** of every stage and the label of the answer.
//!
//! The identity covers everything except the results, through a
//! [`antecedent_core::CanonicalWriter`] stream under a versioned tag. Numeric tags of the kinds,
//! classes, representations and operators live next to their enums in
//! exhaustive matches and are append-only.
//!
//! # What a consumer does
//!
//! A consumer trusts none of it. It recomputes the identity, runs the chain
//! rule on the stored declaration (`check_propagation_chain`) and requires the
//! stored label to be the one the rule yields, replays every stage with the
//! chain arithmetic of `antecedent-stats` and accepts only results that equal
//! the stored bits. A stage that depends on a withdrawn source is stored as
//! invalidated, with the withdrawn sources named, and carries no number. A
//! stage that depends on a re-weighted source is summarized under the
//! row weights its sources' changes imply, and every other stage is summarized
//! exactly as it was before the change.
//!
//! What this protects against: corruption, edits that are not resealed, and
//! resealed edits that change an operator, a parameter, a declared factor, a
//! draw or the label. It does not protect against a producer that seals wrong
//! but coherent draws, or a bug shared by producer and consumer; the oracles
//! are in the test suites.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod bounds;
mod identity;
mod wire;

use bounds::check_bounds;
pub use identity::propagation_chain_identity;
pub use wire::{
    ChainChangeWire, ChainCoefficientWire, ChainExprTermWire, ChainFactorWire, ChainOpWire,
    ChainSourceWire, ChainStageResultWire, ChainStageWire, ChainUnaryWire,
    ConsumedPropagationChain, PropagationChainRecordWire,
};

use std::collections::{BTreeMap, BTreeSet};

use antecedent_contract::dependence::SharedFactorKind;
use antecedent_contract::propagation_chain::{
    ChainDecl, ChainPlan, ChainReading, ChainRefusal, ChainSourceDecl, ChainStageDecl,
    DROP_DUPLICATED_SOURCE, FIX_CHAIN_DECLARATION, FactorDecl, LawMeaning, OperatorClass,
    RAISE_BUDGET_OR_THIN_DRAWS, RAISE_DRAWS_OR_WIDEN_STUDY, Representation,
    SUPPLY_DRAWS_FOR_INTERVAL, StageKind, TableConstruction, check_propagation_chain,
};
#[cfg(test)]
use antecedent_core::CanonicalWriter;
use antecedent_core::ExecutionContext;
use antecedent_stats::propagation_chain::{
    ChainError, Coefficient, Combine, Enclosure, Evaluated, ExprTerm, Preposterior, StageOp,
    StageSpec, UnaryFn, decide, evaluate_columns, evaluate_enclosures, preposterior, summarize,
    weights_from_log,
};

use crate::IoError;
use crate::container::{CompressPolicy, EncodedArtifact, pack_section_shared};
use crate::joint_law_propagation_artifact::verdict_code;

/// The section id of the retained record on a composite artifact.
pub const PROPAGATION_CHAIN_SECTION: &str = "analysis_result.propagation";
/// The record format this reader writes and accepts.
pub const PROPAGATION_CHAIN_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const PROPAGATION_CHAIN_FEATURE: &str = "propagation_chain_v1";
/// Domain tag of the record identity.
pub const PROPAGATION_CHAIN_TAG: &str = "antecedent.propagation_chain.v1";
/// Most bytes an encoded record may occupy; larger input is refused before any
/// CBOR is parsed.
pub const MAX_PROPAGATION_CHAIN_BYTES: usize = 64 * 1024 * 1024;
/// Most changes one record may append.
pub const MAX_PROPAGATION_CHAIN_CHANGES: usize = 256;
/// Longest identity or name in the record, in bytes.
pub const MAX_PROPAGATION_CHAIN_TEXT_BYTES: usize = 4096;
/// Most named values one stage result may hold.
pub const MAX_STAGE_VALUES: usize = 64;
/// Most cells (`rows × (sources + stages)`) a record may evaluate. Point sources are
/// replicated down every row of the table, so the cells, not the wire bytes, set the memory
/// an execution needs; the check runs before any column is built.
pub const MAX_PROPAGATION_CHAIN_CELLS: usize = 1 << 26;

/// Intake bounds for a propagation chain record. The defaults are
/// [`MAX_PROPAGATION_CHAIN_BYTES`], [`MAX_PROPAGATION_CHAIN_CHANGES`],
/// [`MAX_PROPAGATION_CHAIN_TEXT_BYTES`] and [`MAX_STAGE_VALUES`]; a consumer with a larger
/// budget passes its own to [`PropagationChainRecordWire::decode_within`] and
/// [`PropagationChainRecordWire::consume_within`]. The chain rule's node, row and term bounds
/// belong to the rule itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PropagationChainLimits {
    /// Most encoded bytes.
    pub max_bytes: usize,
    /// Most changes one record may append.
    pub max_changes: usize,
    /// Longest identity or name, in bytes.
    pub max_text_bytes: usize,
    /// Most named values one stage result may hold.
    pub max_stage_values: usize,
}

impl Default for PropagationChainLimits {
    fn default() -> Self {
        Self {
            max_bytes: MAX_PROPAGATION_CHAIN_BYTES,
            max_changes: MAX_PROPAGATION_CHAIN_CHANGES,
            max_text_bytes: MAX_PROPAGATION_CHAIN_TEXT_BYTES,
            max_stage_values: MAX_STAGE_VALUES,
        }
    }
}

/// A stage that depends on no changed source keeps its stored numbers.
pub const STATUS_CURRENT: &str = "current";
/// A stage summarized under the row weights its sources' changes imply.
pub const STATUS_REWEIGHTED: &str = "reweighted";
/// A stage that depends on a withdrawn source and carries no number.
pub const STATUS_INVALIDATED: &str = "invalidated";

/// Why a propagation record was refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum PropagationChainError {
    /// The bytes do not decode as this format.
    #[error("propagation record does not decode: {0}")]
    Decode(String),
    /// The feature marker is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// Another format version, refused before the payload is interpreted.
    #[error("unsupported record version {version}")]
    UnsupportedVersion {
        /// The stored version.
        version: u32,
    },
    /// A stored collection exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The identity does not match the stored content, or the caller's
    /// expected identity.
    #[error("identity mismatch")]
    IdentityMismatch,
    /// The record could not be encoded.
    #[error("propagation record does not encode: {0}")]
    Encode(String),
    /// A stored field is not a well formed declaration.
    #[error("malformed declaration: {0}")]
    Malformed(String),
    /// The chain is refused: the refusal carries a registered code, a
    /// namespaced detail and the next attempt.
    #[error("propagation chain refused: {0}")]
    Refused(ChainRefusal),
    /// The replayed results differ from the stored ones.
    #[error("results do not replay")]
    ResultMismatch,
    /// Cancellation was observed at a work boundary.
    #[error("propagation chain cancelled")]
    Cancelled,
}

type Error = PropagationChainError;

impl From<IoError> for PropagationChainError {
    fn from(error: IoError) -> Self {
        Self::Decode(error.to_string())
    }
}

impl From<ChainRefusal> for PropagationChainError {
    fn from(refusal: ChainRefusal) -> Self {
        Self::Refused(refusal)
    }
}

impl PropagationChainError {
    /// The error as a typed refusal. A refused chain keeps its own refusal; a
    /// malformed or edited record is `invalid_argument` with the attempt to
    /// supply an intact record.
    #[must_use]
    pub fn to_refusal(&self) -> ChainRefusal {
        let invalid = antecedent_core::reason_code!("invalid_argument");
        match self {
            Self::Refused(refusal) => refusal.clone(),
            Self::Cancelled => ChainRefusal::new(
                antecedent_core::reason_code!("cancelled_no_claim"),
                "propagation_chain.cancelled",
                "cancellation is not a verdict on the chain",
                None,
                FIX_CHAIN_DECLARATION,
            ),
            Self::IdentityMismatch | Self::ResultMismatch => ChainRefusal::new(
                invalid,
                "propagation_chain.record_refused",
                self.to_string(),
                None,
                FIX_CHAIN_DECLARATION,
            ),
            other => ChainRefusal::new(
                invalid,
                "propagation_chain.record_invalid",
                other.to_string(),
                None,
                FIX_CHAIN_DECLARATION,
            ),
        }
    }
}

fn refused(
    code: &'static str,
    detail: &'static str,
    message: impl Into<String>,
    stage: Option<&str>,
    next: antecedent_core::NextAttempt,
) -> Error {
    Error::Refused(ChainRefusal::new(code, detail, message, stage, next))
}

fn malformed(message: impl Into<String>) -> Error {
    Error::Malformed(message.into())
}

fn map_chain_error(error: ChainError) -> Error {
    let invalid = antecedent_core::reason_code!("invalid_argument");
    match error {
        ChainError::Cancelled => Error::Cancelled,
        ChainError::SingularPool => refused(
            antecedent_core::reason_code!("evidence_double_counted"),
            "propagation_chain.pool_inputs_duplicated",
            "the pooled inputs are perfectly correlated, so they carry the same evidence and no weights exist",
            None,
            DROP_DUPLICATED_SOURCE,
        ),
        ChainError::LowEffectiveSize { effective, minimum } => refused(
            antecedent_core::reason_code!("posterior_draws_below_supported"),
            "propagation_chain.update_effective_size",
            format!(
                "the updated draws have effective size {effective:.3}, below the minimum {minimum}"
            ),
            None,
            RAISE_DRAWS_OR_WIDEN_STUDY,
        ),
        ChainError::Budget(what) => refused(
            invalid,
            "propagation_chain.budget_exceeded",
            format!("the chain exceeds the declared budget of {what}"),
            None,
            RAISE_BUDGET_OR_THIN_DRAWS,
        ),
        ChainError::Denominator(what) => refused(
            invalid,
            "propagation_chain.denominator_not_bounded",
            what,
            None,
            FIX_CHAIN_DECLARATION,
        ),
        ChainError::EnclosureUnbounded(what) => refused(
            antecedent_core::reason_code!("route_not_supported"),
            "propagation_chain.enclosure_unbounded",
            format!("the enclosure is unbounded: {what}"),
            None,
            FIX_CHAIN_DECLARATION,
        ),
        ChainError::Domain(what) => refused(
            invalid,
            "propagation_chain.stage_left_domain",
            what,
            None,
            FIX_CHAIN_DECLARATION,
        ),
        ChainError::PoolCovariance(inner) => refused(
            invalid,
            "propagation_chain.pool_refused",
            inner.to_string(),
            None,
            FIX_CHAIN_DECLARATION,
        ),
        other => Error::Malformed(other.to_string()),
    }
}

fn parse<T>(found: Option<T>, what: &str, name: &str) -> Result<T, Error> {
    found.ok_or_else(|| malformed(format!("unknown {what} `{name}`")))
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn floats(values: &[u64]) -> Vec<f64> {
    values.iter().map(|b| f64::from_bits(*b)).collect()
}

/// Bit patterns of a float slice, for building record wires.
#[must_use]
pub fn chain_float_bits(values: &[f64]) -> Vec<u64> {
    bits(values)
}

fn factor_decl(factor: &ChainFactorWire) -> Result<FactorDecl, Error> {
    Ok(FactorDecl {
        kind: parse(SharedFactorKind::from_name(&factor.kind), "factor kind", &factor.kind)?,
        id: factor.id.clone(),
    })
}

fn factor_decls(factors: &[ChainFactorWire]) -> Result<Vec<FactorDecl>, Error> {
    factors.iter().map(factor_decl).collect()
}

fn coefficient(wire: &ChainCoefficientWire) -> Result<Coefficient, Error> {
    Ok(match wire {
        ChainCoefficientWire::Fixed { value } => Coefficient::Fixed(f64::from_bits(*value)),
        ChainCoefficientWire::Input { position } => Coefficient::Input(
            usize::try_from(*position)
                .map_err(|_| malformed("a coefficient position is too large"))?,
        ),
    })
}

fn unary(wire: &ChainUnaryWire) -> UnaryFn {
    let f = f64::from_bits;
    match wire {
        ChainUnaryWire::Exp => UnaryFn::Exp,
        ChainUnaryWire::Log => UnaryFn::Log,
        ChainUnaryWire::Square => UnaryFn::Square,
        ChainUnaryWire::Reciprocal => UnaryFn::Reciprocal,
        ChainUnaryWire::Logistic => UnaryFn::Logistic,
        ChainUnaryWire::ExponentialUtility { risk_aversion } => {
            UnaryFn::ExponentialUtility { risk_aversion: f(*risk_aversion) }
        }
        ChainUnaryWire::PiecewiseLinear { kink, below_slope, above_slope } => {
            UnaryFn::PiecewiseLinear {
                kink: f(*kink),
                below_slope: f(*below_slope),
                above_slope: f(*above_slope),
            }
        }
        ChainUnaryWire::Threshold { cutoff, below, above } => {
            UnaryFn::Threshold { cutoff: f(*cutoff), below: f(*below), above: f(*above) }
        }
    }
}

fn expr_term(wire: &ChainExprTermWire) -> Result<ExprTerm, Error> {
    let f = f64::from_bits;
    Ok(match wire {
        ChainExprTermWire::Input { position } => ExprTerm::Input(
            usize::try_from(*position)
                .map_err(|_| malformed("an expression input position is too large"))?,
        ),
        ChainExprTermWire::Constant { value } => ExprTerm::Constant(f(*value)),
        ChainExprTermWire::Add => ExprTerm::Add,
        ChainExprTermWire::Subtract => ExprTerm::Subtract,
        ChainExprTermWire::Multiply => ExprTerm::Multiply,
        ChainExprTermWire::Divide { min_magnitude } => {
            ExprTerm::Divide { min_magnitude: f(*min_magnitude) }
        }
        ChainExprTermWire::RequireRange { low, high } => {
            ExprTerm::RequireRange { low: f(*low), high: f(*high) }
        }
    })
}

fn stage_op(wire: &ChainOpWire) -> Result<StageOp, Error> {
    Ok(match wire {
        ChainOpWire::Affine { scale, shift } => {
            StageOp::Affine { scale: coefficient(scale)?, shift: coefficient(shift)? }
        }
        ChainOpWire::Reweight { weights } => StageOp::Reweight { weights: floats(weights) },
        ChainOpWire::Pool { dependent } => StageOp::Pool { dependent: *dependent },
        ChainOpWire::Binary { combine } => {
            StageOp::Binary(parse(Combine::from_name(combine), "combination", combine)?)
        }
        ChainOpWire::Unary { function } => StageOp::Unary(unary(function)),
        ChainOpWire::Expression { terms } => {
            StageOp::Expression(terms.iter().map(expr_term).collect::<Result<Vec<_>, _>>()?)
        }
        ChainOpWire::Preposterior { noise_sd, sample_size, reference_utility, noise } => {
            StageOp::Preposterior(Preposterior {
                noise_sd: f64::from_bits(*noise_sd),
                sample_size: f64::from_bits(*sample_size),
                reference_utility: f64::from_bits(*reference_utility),
                noise: floats(noise),
            })
        }
    })
}

fn operator_class(wire: &ChainOpWire) -> OperatorClass {
    match wire {
        ChainOpWire::Affine { .. } => OperatorClass::Affine,
        ChainOpWire::Reweight { .. } => OperatorClass::Reweight,
        ChainOpWire::Pool { dependent: true } => OperatorClass::DependentPool,
        ChainOpWire::Pool { dependent: false } => OperatorClass::IndependentPool,
        ChainOpWire::Binary { .. } | ChainOpWire::Unary { .. } | ChainOpWire::Expression { .. } => {
            OperatorClass::Nonlinear
        }
        ChainOpWire::Preposterior { .. } => OperatorClass::Update,
    }
}

fn word_of(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}

// ------------------------------------------------------------------ identity

// ----------------------------------------------------------------- execution

struct Executed {
    plan: ChainPlan,
    results: Vec<ChainStageResultWire>,
}

struct ChangeState {
    withdrawn: BTreeSet<String>,
    log_weights: BTreeMap<String, Vec<f64>>,
}

fn push_float(values: &mut Vec<(String, u64)>, name: &str, value: f64) {
    values.push((name.to_owned(), value.to_bits()));
}

fn push_count(values: &mut Vec<(String, u64)>, name: &str, value: u64) {
    values.push((name.to_owned(), value));
}

impl PropagationChainRecordWire {
    /// A draft record: the declaration with no changes, results or identity.
    /// [`Self::seal`] checks it, computes the results and the identity.
    #[must_use]
    pub fn draft(
        construction: TableConstruction,
        reading: ChainReading,
        output: &str,
        sources: Vec<ChainSourceWire>,
        stages: Vec<ChainStageWire>,
    ) -> Self {
        Self {
            version: PROPAGATION_CHAIN_VERSION,
            required_features: vec![PROPAGATION_CHAIN_FEATURE.to_owned()],
            construction: construction.as_str().to_owned(),
            reading: reading.as_str().to_owned(),
            output: output.to_owned(),
            sources,
            stages,
            changes: Vec::new(),
            label: String::new(),
            results: Vec::new(),
            identity: String::new(),
        }
    }

    fn declaration(&self) -> Result<ChainDecl, Error> {
        let mut sources = Vec::with_capacity(self.sources.len());
        for source in &self.sources {
            sources.push(ChainSourceDecl {
                id: source.id.clone(),
                representation: parse(
                    Representation::from_name(&source.representation),
                    "representation",
                    &source.representation,
                )?,
                meaning: match &source.meaning {
                    Some(name) => Some(parse(LawMeaning::from_name(name), "law meaning", name)?),
                    None => None,
                },
                alignment: source.alignment.clone(),
                complete_disclosure: source.complete_disclosure,
                factors: factor_decls(&source.factors)?,
            });
        }
        let mut stages = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            stages.push(ChainStageDecl {
                id: stage.id.clone(),
                kind: parse(StageKind::from_name(&stage.kind), "stage kind", &stage.kind)?,
                class: operator_class(&stage.op),
                inputs: stage.inputs.clone(),
                own_factors: factor_decls(&stage.own_factors)?,
            });
        }
        Ok(ChainDecl {
            construction: parse(
                TableConstruction::from_name(&self.construction),
                "construction",
                &self.construction,
            )?,
            sources,
            stages,
            output: self.output.clone(),
            reading: parse(ChainReading::from_name(&self.reading), "reading", &self.reading)?,
        })
    }

    /// Run the chain rule on the stored declaration.
    ///
    /// # Errors
    /// A malformed declaration or a refusal of the rule, with its next attempt.
    pub fn plan(&self) -> Result<ChainPlan, Error> {
        self.plan_within(&PropagationChainLimits::default())
    }

    /// [`Self::plan`] under the caller's bounds instead of the defaults.
    ///
    /// # Errors
    /// As [`Self::plan`]; a bound of `limits` exceeded is `LimitsExceeded`.
    pub fn plan_within(&self, limits: &PropagationChainLimits) -> Result<ChainPlan, Error> {
        check_bounds(self, limits)?;
        Ok(check_propagation_chain(&self.declaration()?)?)
    }

    fn change_state(&self, rows: Option<usize>) -> Result<ChangeState, Error> {
        let mut state = ChangeState { withdrawn: BTreeSet::new(), log_weights: BTreeMap::new() };
        for change in &self.changes {
            match change {
                ChainChangeWire::Withdraw { source } => {
                    self.known_source(source)?;
                    if !state.withdrawn.insert(source.clone()) {
                        return Err(malformed(format!("source `{source}` is withdrawn twice")));
                    }
                }
                ChainChangeWire::Reweight { source, log_weights } => {
                    let declared = self.known_source(source)?;
                    if declared.representation != Representation::Draws.as_str() {
                        return Err(refused(
                            antecedent_core::reason_code!(
                                "uncertainty_representation_insufficient"
                            ),
                            "propagation_chain.reweight_needs_draws",
                            format!(
                                "source `{source}` is not draws, so its rows cannot be re-weighted"
                            ),
                            Some(source),
                            SUPPLY_DRAWS_FOR_INTERVAL,
                        ));
                    }
                    if state.withdrawn.contains(source) {
                        return Err(malformed(format!("source `{source}` is withdrawn")));
                    }
                    let log = floats(log_weights);
                    if Some(log.len()) != rows || log.iter().any(|v| !v.is_finite()) {
                        return Err(malformed(
                            "a re-weighting has one finite log weight per row of the draw table",
                        ));
                    }
                    let slot = state
                        .log_weights
                        .entry(source.clone())
                        .or_insert_with(|| vec![0.0; log.len()]);
                    for (total, extra) in slot.iter_mut().zip(&log) {
                        *total += extra;
                    }
                }
            }
        }
        Ok(state)
    }

    fn known_source(&self, id: &str) -> Result<&ChainSourceWire, Error> {
        self.sources
            .iter()
            .find(|source| source.id == id)
            .ok_or_else(|| malformed(format!("a change names `{id}`, which is not a source")))
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one linear pass: build the table, evaluate, summarize each stage"
    )]
    fn execute(
        &self,
        ctx: &ExecutionContext,
        limits: &PropagationChainLimits,
    ) -> Result<Executed, Error> {
        check_bounds(self, limits)?;
        let plan = self.plan_within(limits)?;
        let positions: BTreeMap<&str, usize> = self
            .sources
            .iter()
            .map(|s| s.id.as_str())
            .chain(self.stages.iter().map(|s| s.id.as_str()))
            .enumerate()
            .map(|(index, id)| (id, index))
            .collect();
        let n_sources = self.sources.len();
        let mut specs = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            let inputs = stage
                .inputs
                .iter()
                .map(|id| {
                    positions.get(id.as_str()).copied().ok_or_else(|| {
                        malformed(format!("stage `{}` reads unknown `{id}`", stage.id))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            specs.push(StageSpec { inputs, op: stage_op(&stage.op)? });
        }
        let enclosure_mode =
            self.sources.iter().any(|s| s.representation == Representation::Interval.as_str());
        let rows = self
            .sources
            .iter()
            .find(|s| s.representation == Representation::Draws.as_str())
            .map(|s| s.values.len());
        for source in &self.sources {
            let expected = match source.representation.as_str() {
                "draws" => rows.unwrap_or(0),
                "point" => 1,
                _ => 2,
            };
            if source.values.len() != expected || expected == 0 {
                return Err(malformed(format!(
                    "source `{}` carries {} values, not the {} its representation needs",
                    source.id,
                    source.values.len(),
                    expected
                )));
            }
        }
        let state = self.change_state(rows)?;
        let stage_status = |stage: &str| -> (&'static str, Vec<String>) {
            let sources = plan.stage(stage).map(|p| p.sources.as_slice()).unwrap_or_default();
            let hit: Vec<String> =
                sources.iter().filter(|s| state.withdrawn.contains(*s)).cloned().collect();
            if !hit.is_empty() {
                return (STATUS_INVALIDATED, hit);
            }
            if sources.iter().any(|s| state.log_weights.contains_key(s)) {
                return (STATUS_REWEIGHTED, Vec::new());
            }
            (STATUS_CURRENT, Vec::new())
        };
        let mut results = Vec::with_capacity(self.stages.len());
        if enclosure_mode {
            let boxes = self
                .sources
                .iter()
                .map(|source| {
                    let values = floats(&source.values);
                    if values.len() == 2 {
                        Enclosure::new(values[0], values[1])
                    } else {
                        Enclosure::new(values[0], values[0])
                    }
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(map_chain_error)?;
            let evaluated = evaluate_enclosures(&boxes, &specs).map_err(map_chain_error)?;
            for (stage, enclosure) in self.stages.iter().zip(&evaluated) {
                let (status, invalidated_by) = stage_status(&stage.id);
                let mut values = Vec::new();
                if status != STATUS_INVALIDATED {
                    push_float(&mut values, "lower", enclosure.lower);
                    push_float(&mut values, "upper", enclosure.upper);
                }
                results.push(ChainStageResultWire {
                    stage: stage.id.clone(),
                    status: status.to_owned(),
                    invalidated_by,
                    values,
                });
            }
            return Ok(Executed { plan, results });
        }
        let table_rows = rows.unwrap_or(1);
        if table_rows
            .checked_mul(n_sources.saturating_add(self.stages.len()))
            .is_none_or(|cells| cells > MAX_PROPAGATION_CHAIN_CELLS)
        {
            return Err(map_chain_error(ChainError::Budget("table cells")));
        }
        let columns: Vec<Vec<f64>> = self
            .sources
            .iter()
            .map(|source| {
                if source.representation == Representation::Point.as_str() {
                    vec![f64::from_bits(source.values[0]); table_rows]
                } else {
                    floats(&source.values)
                }
            })
            .collect();
        let evaluated =
            evaluate_columns(&columns, &specs, &ctx.cancellation).map_err(map_chain_error)?;
        for (position, stage) in self.stages.iter().enumerate() {
            let (status, invalidated_by) = stage_status(&stage.id);
            let mut values = Vec::new();
            if status != STATUS_INVALIDATED {
                let sources = plan.stage(&stage.id).map(|p| p.sources.clone()).unwrap_or_default();
                let weights = stage_weights(&state, &sources)?;
                Self::summarize_stage(
                    position,
                    stage,
                    &specs[position],
                    (&columns, &evaluated, n_sources),
                    weights.as_deref(),
                    ctx,
                    &mut values,
                )?;
            }
            results.push(ChainStageResultWire {
                stage: stage.id.clone(),
                status: status.to_owned(),
                invalidated_by,
                values,
            });
        }
        Ok(Executed { plan, results })
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the stage, its spec, the evaluated table, its weights and the output slot are separate facts"
    )]
    fn summarize_stage(
        position: usize,
        stage: &ChainStageWire,
        spec: &StageSpec,
        table: (&[Vec<f64>], &Evaluated, usize),
        weights: Option<&[f64]>,
        ctx: &ExecutionContext,
        values: &mut Vec<(String, u64)>,
    ) -> Result<(), Error> {
        let (sources, evaluated, n_sources) = table;
        if let StageOp::Preposterior(study) = &spec.op {
            let (Some(parameter), Some(utility)) = (
                node_column(sources, evaluated, n_sources, spec.inputs[0]),
                node_column(sources, evaluated, n_sources, spec.inputs[1]),
            ) else {
                return Err(malformed("an update reads two columns"));
            };
            let outcome = preposterior(parameter, utility, weights, study, &ctx.cancellation)
                .map_err(map_chain_error)?;
            push_float(values, "current_expected_utility", outcome.current_expected_utility);
            push_float(values, "expected_value_with_study", outcome.expected_value_with_study);
            push_float(values, "value_of_information", outcome.value_of_information);
            push_float(
                values,
                "mean_updated_expected_utility",
                outcome.mean_updated_expected_utility,
            );
            push_float(
                values,
                "probability_decision_changes",
                outcome.probability_decision_changes,
            );
            push_float(values, "mean_effective_size", outcome.mean_effective_size);
            push_float(values, "min_effective_size", outcome.min_effective_size);
            push_float(values, "reference_utility", study.reference_utility);
            return Ok(());
        }
        let Some(column) = evaluated.columns[position].as_deref() else {
            return Err(malformed("a stage has no column"));
        };
        let summary = summarize(column, weights).map_err(map_chain_error)?;
        push_float(values, "mean", summary.mean);
        push_float(values, "variance", summary.variance);
        push_float(values, "effective_size", summary.effective_size);
        push_float(values, "mean_standard_error", summary.mean_standard_error);
        push_count(values, "n_rows", u64::try_from(summary.n).unwrap_or(u64::MAX));
        if let Some(pool) = evaluated.pools[position].as_ref() {
            for (index, weight) in pool.weights.iter().enumerate() {
                push_float(values, &format!("weight_{index}"), *weight);
            }
            push_float(values, "independent_variance", pool.independent_variance);
            push_float(values, "joint_variance", pool.joint_variance);
        }
        if let Some(reference) = stage.decision_reference {
            let decision = decide(&summary, f64::from_bits(reference));
            push_float(values, "reference_utility", decision.reference_utility);
            push_float(values, "utility_gap", decision.gap);
            push_float(values, "decision_band", decision.band);
            push_count(values, "verdict_code", verdict_code(decision.verdict));
        }
        Ok(())
    }

    /// Check the declaration, compute every stage's result and seal the
    /// identity. The draft's changes, if any, are applied.
    ///
    /// # Errors
    /// A malformed declaration, a refusal of the chain rule (each carries its
    /// registered code and next attempt), a refusal of the arithmetic,
    /// cancellation or an encoding failure.
    pub fn seal(draft: Self, ctx: &ExecutionContext) -> Result<Self, Error> {
        Self::seal_within(draft, ctx, &PropagationChainLimits::default())
    }

    /// [`Self::seal`] under the caller's bounds instead of the defaults.
    ///
    /// # Errors
    /// As [`Self::seal`]; a bound of `limits` exceeded is `LimitsExceeded`.
    pub fn seal_within(
        mut draft: Self,
        ctx: &ExecutionContext,
        limits: &PropagationChainLimits,
    ) -> Result<Self, Error> {
        draft.version = PROPAGATION_CHAIN_VERSION;
        draft.required_features = vec![PROPAGATION_CHAIN_FEATURE.to_owned()];
        let executed = draft.execute(ctx, limits)?;
        executed.plan.label.clone_into(&mut draft.label);
        draft.results = executed.results;
        draft.identity = propagation_chain_identity(&draft)?;
        check_bounds(&draft, limits)?;
        Ok(draft)
    }

    /// Append `change` and return the resealed record. This record is not
    /// modified. Stages that depend on no changed source keep their stored
    /// numbers bit for bit.
    ///
    /// # Errors
    /// A change naming no source, a withdrawn source, a re-weighting whose
    /// length differs from the draw table, a re-weighting of a source that is
    /// not draws, or any refusal of [`Self::seal`].
    pub fn with_change(
        &self,
        change: ChainChangeWire,
        ctx: &ExecutionContext,
    ) -> Result<Self, Error> {
        self.with_change_within(change, ctx, &PropagationChainLimits::default())
    }

    /// [`Self::with_change`] under the caller's bounds instead of the defaults.
    ///
    /// # Errors
    /// As [`Self::with_change`]; more changes than `limits.max_changes` is `LimitsExceeded`.
    pub fn with_change_within(
        &self,
        change: ChainChangeWire,
        ctx: &ExecutionContext,
        limits: &PropagationChainLimits,
    ) -> Result<Self, Error> {
        let mut next = self.clone();
        next.changes.push(change);
        Self::seal_within(next, ctx, limits)
    }

    /// The stages a change would invalidate or re-weight: the plan's
    /// dependents of `source`.
    ///
    /// # Errors
    /// Any refusal of the chain rule.
    pub fn dependents_of(&self, source: &str) -> Result<Vec<String>, Error> {
        Ok(self.plan()?.dependents_of(source).into_iter().map(str::to_owned).collect())
    }

    /// Encode as CBOR.
    ///
    /// # Errors
    /// A bound exceeded or an encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, Error> {
        self.export_within(&PropagationChainLimits::default())
    }

    /// [`Self::export`] under the caller's bounds instead of the defaults.
    ///
    /// # Errors
    /// As [`Self::export`].
    pub fn export_within(&self, limits: &PropagationChainLimits) -> Result<Vec<u8>, Error> {
        check_bounds(self, limits)?;
        crate::to_cbor(self).map_err(|error| Error::Encode(error.to_string()))
    }

    /// Decode, refusing any other version or feature before the payload is
    /// interpreted.
    ///
    /// # Errors
    /// A decoding failure, another version or a foreign feature.
    pub fn decode(bytes: &[u8]) -> Result<Self, Error> {
        Self::decode_within(bytes, &PropagationChainLimits::default())
    }

    /// [`Self::decode`] under the caller's bounds instead of the defaults.
    ///
    /// # Errors
    /// As [`Self::decode`]; a bound of `limits` exceeded is `LimitsExceeded`.
    pub fn decode_within(bytes: &[u8], limits: &PropagationChainLimits) -> Result<Self, Error> {
        if bytes.len() > limits.max_bytes {
            return Err(Error::LimitsExceeded("record bytes"));
        }
        let wire: Self = crate::decode_with_version(bytes, |version: u32| {
            if version == PROPAGATION_CHAIN_VERSION {
                Ok(())
            } else {
                Err(Error::UnsupportedVersion { version })
            }
        })?;
        if wire.required_features != [PROPAGATION_CHAIN_FEATURE] {
            return Err(Error::UnsupportedSemantics("required features"));
        }
        check_bounds(&wire, limits)?;
        Ok(wire)
    }

    /// Verify and replay.
    ///
    /// `expected_identity`, when given, is the identity the caller holds
    /// independently of the record.
    ///
    /// # Errors
    /// Any of the refusals of [`PropagationChainError`].
    pub fn consume(
        &self,
        expected_identity: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<ConsumedPropagationChain, Error> {
        self.consume_within(expected_identity, ctx, &PropagationChainLimits::default())
    }

    /// [`Self::consume`] under the caller's bounds instead of the defaults.
    ///
    /// # Errors
    /// As [`Self::consume`]; a bound of `limits` exceeded is `LimitsExceeded`.
    pub fn consume_within(
        &self,
        expected_identity: Option<&str>,
        ctx: &ExecutionContext,
        limits: &PropagationChainLimits,
    ) -> Result<ConsumedPropagationChain, Error> {
        if self.version != PROPAGATION_CHAIN_VERSION {
            return Err(Error::UnsupportedVersion { version: self.version });
        }
        if self.required_features != [PROPAGATION_CHAIN_FEATURE] {
            return Err(Error::UnsupportedSemantics("required features"));
        }
        check_bounds(self, limits)?;
        let recomputed = propagation_chain_identity(self)?;
        if recomputed != self.identity {
            return Err(Error::IdentityMismatch);
        }
        if let Some(expected) = expected_identity {
            if expected != recomputed {
                return Err(Error::IdentityMismatch);
            }
        }
        // The rule decides again from the stored declaration, and the arithmetic
        // runs again from the stored draws: a resealed edit that changes an
        // operator, a declared factor, a draw or the label no longer matches.
        let executed = self.execute(ctx, limits)?;
        if executed.plan.label != self.label {
            return Err(Error::IdentityMismatch);
        }
        if executed.results != self.results {
            return Err(Error::ResultMismatch);
        }
        Ok(ConsumedPropagationChain {
            identity: recomputed,
            label: self.label.clone(),
            output: self.output.clone(),
            results: executed.results,
            plan: executed.plan,
        })
    }

    /// Carry the record as the optional propagation section of a composite
    /// artifact. The section is not required: a reader that does not know it
    /// reads the artifact as before.
    ///
    /// # Errors
    /// An encoding failure.
    pub fn attach_to(&self, artifact: &mut EncodedArtifact) -> Result<(), Error> {
        let bytes = self.export()?;
        let (mut descriptor, section) = pack_section_shared(
            PROPAGATION_CHAIN_SECTION,
            crate::CONTENT_CBOR,
            bytes.into(),
            CompressPolicy::Auto,
        );
        descriptor.required = false;
        artifact.manifest.sections.retain(|d| d.id != PROPAGATION_CHAIN_SECTION);
        artifact.sections.retain(|s| s.id != PROPAGATION_CHAIN_SECTION);
        artifact.manifest.sections.push(descriptor);
        artifact.sections.push(section);
        Ok(())
    }

    /// The record carried by an artifact, if it has the section.
    ///
    /// # Errors
    /// A present section that does not decode.
    pub fn from_artifact(artifact: &EncodedArtifact) -> Result<Option<Self>, Error> {
        let Some(section) =
            artifact.sections.iter().find(|section| section.id == PROPAGATION_CHAIN_SECTION)
        else {
            return Ok(None);
        };
        Self::decode(&section.data).map(Some)
    }
}

fn node_column<'a>(
    sources: &'a [Vec<f64>],
    evaluated: &'a Evaluated,
    n_sources: usize,
    index: usize,
) -> Option<&'a [f64]> {
    if index < n_sources {
        sources.get(index).map(Vec::as_slice)
    } else {
        evaluated.columns.get(index - n_sources).and_then(|column| column.as_deref())
    }
}

/// Row weights a stage is summarized under: the sum of the log weights of the
/// re-weighted sources it depends on, normalized. `None` when it depends on
/// none.
fn stage_weights(state: &ChangeState, sources: &[String]) -> Result<Option<Vec<f64>>, Error> {
    let mut total: Option<Vec<f64>> = None;
    for source in sources {
        let Some(log) = state.log_weights.get(source) else {
            continue;
        };
        match &mut total {
            None => total = Some(log.clone()),
            Some(sum) => {
                for (slot, extra) in sum.iter_mut().zip(log) {
                    *slot += extra;
                }
            }
        }
    }
    match total {
        None => Ok(None),
        Some(log) => Ok(Some(weights_from_log(&log).map_err(map_chain_error)?)),
    }
}

#[cfg(test)]
#[path = "propagation_chain_record_tests.rs"]
mod tests;
