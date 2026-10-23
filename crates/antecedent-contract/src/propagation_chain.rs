// SPDX-License-Identifier: MIT OR Apache-2.0

//! The rule for a chain of transformations: which uncertainty each stage can
//! carry, and nothing computed until the chain is licensed.
//!
//! A chain starts from declared sources and runs stages: a measurement
//! mapping, a transport, a pool, a derived quantity, a nonlinear decision
//! utility, a preposterior update. [`check_propagation_chain`] is the one place
//! that decides whether the chain may run, and it builds on the rules the
//! single-step operators already have rather than restating them:
//!
//! * **Dependence.** Every source declares the rows, clusters, models, priors,
//!   disturbances and periods it uses ([`DependenceSource`]). A stage's
//!   evidence is the union of its inputs' evidence and the factors it brings
//!   itself (a calibration sample behind a measurement mapping, a selection
//!   model behind a transport). Every source also carries an implicit row
//!   factor of its own: two stages derived from one source share it, so a
//!   source with no declared factors is still shared by everything built on
//!   it. An independent pool of inputs whose evidence overlaps would count the
//!   shared evidence twice and is refused with `evidence_double_counted`; a
//!   pool whose overlap is undeclared is refused with `evidence_overlap_unknown`.
//!   Declaring the pool dependent weights it by the joint covariance of the
//!   aligned draws instead.
//! * **Representation.** A nonlinear stage needs the joint law of its inputs,
//!   carried by draws that share one alignment identity. Draws of unrelated
//!   sources are never paired by index (`joint_law_required`). Intervals and
//!   points are carried through nonlinear stages as an enclosure, which holds
//!   under every coupling and is not a law. A point cannot give a spread, an
//!   interval cannot give a posterior, a sampling law cannot give a posterior,
//!   an enclosure cannot be pooled, and an interval reading of a composed
//!   quantity needs a measured coverage record. Each is a typed refusal with a
//!   registered code and a [`NextAttempt`] that names the representation that
//!   would carry the uncertainty.
//! * **Updates.** A preposterior update re-evaluates a decision after a future
//!   study. It needs posterior draws of the parameter, and nothing may consume
//!   it.
//!
//! The plan ([`ChainPlan`]) names, for every stage, the sources it depends on,
//! so that a changed assumption about a source reaches exactly its dependents
//! ([`ChainPlan::dependents_of`]).
//!
//! This module computes nothing numerical. The arithmetic is
//! `antecedent_stats::propagation_chain`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use antecedent_core::{NextAttempt, impl_refused, named_variants};

use crate::dependence::{
    DependenceContract, DependenceSource, DependenceStatus, SharedFactorKind, SharedFactorSets,
};
use crate::evidence_ledger::MAX_TEXT_BYTES;

/// Most sources plus stages one chain may name.
pub const MAX_CHAIN_NODES: usize = 128;
/// Label of a quantity propagated as posterior draws.
pub const POSTERIOR_PROPAGATION_LABEL: &str = "posterior_propagation";
/// Label of a chain read as an enclosure of declared intervals.
pub const PROPAGATION_ENCLOSURE_LABEL: &str = "propagation_enclosure";
/// Label of a chain value computed under the declared dependence.
pub const PROPAGATION_POINT_LABEL: &str = crate::joint_law_rule::POINT_ONLY_LABEL;
/// Label of a chain spread computed under the declared dependence, with no
/// coverage statement.
pub const PROPAGATION_SPREAD_LABEL: &str = crate::joint_law_rule::UNCALIBRATED_SPREAD_LABEL;

const INVALID_CODE: &str = antecedent_core::reason_code!("invalid_argument");
const NOT_SUPPORTED_CODE: &str = antecedent_core::reason_code!("route_not_supported");
const JOINT_LAW_CODE: &str = antecedent_core::reason_code!("joint_law_required");
const DOUBLE_COUNTED_CODE: &str = antecedent_core::reason_code!("evidence_double_counted");
const OVERLAP_UNKNOWN_CODE: &str = antecedent_core::reason_code!("evidence_overlap_unknown");
const REPRESENTATION_CODE: &str = antecedent_core::reason_code!("representation_mapping_required");
const MEANING_CODE: &str = antecedent_core::reason_code!("distribution_meaning_mismatch");
const NOT_MEASURED_CODE: &str = antecedent_core::reason_code!("interval_level_not_measured");

/// The attempt a refused chain names when the sources are not columns of one
/// aligned joint draw table.
pub const SUPPLY_ALIGNED_DRAWS: NextAttempt = NextAttempt::new(
    "supply_aligned_draws",
    "supply the draws of every source as columns of one aligned joint draw table that shares one alignment identity",
);

/// The attempt a refused chain names when independent sources were given as
/// separate draw sets.
pub const BUILD_INDEPENDENT_PRODUCT_TABLE: NextAttempt = NextAttempt::new(
    "build_independent_product_table",
    "declare the sources independent and build their product table, so no row of one source is paired with a row of another by index",
);

/// The attempt a refused independent pool names: pool by the joint covariance.
pub const POOL_WITH_JOINT_COVARIANCE: NextAttempt = NextAttempt::new(
    "pool_with_joint_covariance",
    "declare the pool dependent so its weights come from the joint covariance of the aligned draws",
);

/// The attempt a refused chain names when a source left its overlap undeclared.
pub const DECLARE_SHARED_FACTORS: NextAttempt = NextAttempt::new(
    "declare_shared_factors",
    "declare every row, cluster, model, prior and disturbance each source uses, then check the chain again",
);

/// The attempt a refused independent product table names when sources share a
/// factor.
pub const SUPPLY_JOINT_DRAWS: NextAttempt = NextAttempt::new(
    "supply_joint_draws_for_shared_factors",
    "supply joint draws that carry the shared factors instead of an independent product table",
);

/// The attempt a refused posterior reading or update names.
pub const SUPPLY_POSTERIOR_DRAWS: NextAttempt = NextAttempt::new(
    "supply_posterior_draws",
    "supply the posterior draws of the source quantity under a declared prior; an interval, a point or a sampling law is not a posterior",
);

/// The attempt a refused spread or pool of a point names.
pub const SUPPLY_DRAWS_FOR_FIXED_INPUT: NextAttempt = NextAttempt::new(
    "supply_draws_for_fixed_input",
    "supply draws that carry the uncertainty of the point input, or read the chain as a point",
);

/// The attempt a refused mix or pool of an interval names.
pub const SUPPLY_DRAWS_FOR_INTERVAL: NextAttempt = NextAttempt::new(
    "supply_draws_for_interval_input",
    "supply draws of the interval input under a declared law, or carry every input as an interval and read the enclosure",
);

/// The attempt a refused law reading of an enclosure names.
pub const USE_ENCLOSURE_READING: NextAttempt = NextAttempt::new(
    "read_the_enclosure",
    "read the chain as an enclosure of the declared intervals, which holds under every coupling",
);

/// The attempt a refused interval reading of a composed quantity names.
pub const MEASURE_COMPOSED_COVERAGE: NextAttempt = NextAttempt::new(
    "measure_composed_coverage",
    "measure coverage for the composed coordinate and request the interval through the composed-interval check; read the spread meanwhile",
);

/// The attempt a refused declaration names.
pub const FIX_CHAIN_DECLARATION: NextAttempt = NextAttempt::new(
    "fix_chain_declaration",
    "correct the declaration the refusal names, then check the chain again",
);

/// The attempt a refused law-meaning mix names.
pub const MATCH_LAW_MEANING: NextAttempt = NextAttempt::new(
    "match_law_meaning",
    "give every input the same meaning, sampling or posterior; a sampling law does not become a posterior by being combined with one",
);

/// The attempt a stage invalidated by a withdrawn source names.
pub const RERUN_WITHOUT_WITHDRAWN_SOURCE: NextAttempt = NextAttempt::new(
    "rerun_without_withdrawn_source",
    "supply a replacement for the withdrawn source and re-run the stages that depend on it",
);

/// The attempt a refused pool of duplicated evidence names.
pub const DROP_DUPLICATED_SOURCE: NextAttempt = NextAttempt::new(
    "drop_duplicated_source",
    "drop all but one of the inputs that carry the same evidence, or pool their common source once",
);

/// The attempt a refused update with too few updated draws names.
pub const RAISE_DRAWS_OR_WIDEN_STUDY: NextAttempt = NextAttempt::new(
    "raise_draws_or_widen_study",
    "use more posterior draws, or a study whose likelihood is wider relative to the posterior",
);

/// The attempt a refused budget names.
pub const RAISE_BUDGET_OR_THIN_DRAWS: NextAttempt = NextAttempt::new(
    "raise_budget_or_thin_draws",
    "thin the draw table to the declared row budget, or raise the budget",
);

/// Every attempt this module names.
pub const NEXT_ATTEMPTS: [NextAttempt; 16] = [
    SUPPLY_ALIGNED_DRAWS,
    BUILD_INDEPENDENT_PRODUCT_TABLE,
    POOL_WITH_JOINT_COVARIANCE,
    DECLARE_SHARED_FACTORS,
    SUPPLY_JOINT_DRAWS,
    SUPPLY_POSTERIOR_DRAWS,
    SUPPLY_DRAWS_FOR_FIXED_INPUT,
    SUPPLY_DRAWS_FOR_INTERVAL,
    USE_ENCLOSURE_READING,
    MEASURE_COMPOSED_COVERAGE,
    FIX_CHAIN_DECLARATION,
    MATCH_LAW_MEANING,
    RERUN_WITHOUT_WITHDRAWN_SOURCE,
    DROP_DUPLICATED_SOURCE,
    RAISE_DRAWS_OR_WIDEN_STUDY,
    RAISE_BUDGET_OR_THIN_DRAWS,
];

// ---------------------------------------------------------------------- enums

/// What a stage does to the quantity it reads.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum StageKind {
    /// Map a measured quantity onto the construct, with an uncertain mapping.
    MeasurementMapping,
    /// Carry a quantity into the target population.
    Transport,
    /// Pool several estimates of one quantity.
    Pooling,
    /// A quantity derived from others.
    DerivedQuantity,
    /// A decision utility of a quantity.
    DecisionUtility,
    /// Update the posterior by a future study and re-evaluate the decision.
    PreposteriorUpdate,
}

named_variants! {
    StageKind {
        MeasurementMapping => "measurement_mapping",
        Transport => "transport",
        Pooling => "pooling",
        DerivedQuantity => "derived_quantity",
        DecisionUtility => "decision_utility",
        PreposteriorUpdate => "preposterior_update",
    }
}

impl StageKind {
    /// Canonical tag. Append-only: a new kind takes the next number and an
    /// existing number never changes meaning.
    #[must_use]
    pub const fn tag(self) -> u64 {
        match self {
            Self::MeasurementMapping => 1,
            Self::Transport => 2,
            Self::Pooling => 3,
            Self::DerivedQuantity => 4,
            Self::DecisionUtility => 5,
            Self::PreposteriorUpdate => 6,
        }
    }
}

/// The class of the operator a stage applies.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum OperatorClass {
    /// An affine map whose coefficients may be uncertain inputs.
    Affine,
    /// A fixed-weight combination of the inputs.
    Reweight,
    /// A pool weighted by the variances alone: it assumes independent inputs.
    IndependentPool,
    /// A pool weighted by the joint covariance of the aligned draws.
    DependentPool,
    /// A nonlinear transformation or combination.
    Nonlinear,
    /// A preposterior update.
    Update,
}

named_variants! {
    OperatorClass {
        Affine => "affine",
        Reweight => "reweight",
        IndependentPool => "independent_pool",
        DependentPool => "dependent_pool",
        Nonlinear => "nonlinear",
        Update => "update",
    }
}

impl OperatorClass {
    /// Canonical tag. Append-only.
    #[must_use]
    pub const fn tag(self) -> u64 {
        match self {
            Self::Affine => 1,
            Self::Reweight => 2,
            Self::IndependentPool => 3,
            Self::DependentPool => 4,
            Self::Nonlinear => 5,
            Self::Update => 6,
        }
    }

    const fn is_pool(self) -> bool {
        matches!(self, Self::IndependentPool | Self::DependentPool)
    }
}

/// How a source carries its uncertainty.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum Representation {
    /// A value fixed by declaration. It carries no uncertainty.
    Point,
    /// A declared interval. It is not a law.
    Interval,
    /// Draws, one column of an aligned joint draw table.
    Draws,
}

named_variants! {
    Representation {
        Point => "point",
        Interval => "interval",
        Draws => "draws",
    }
}

impl Representation {
    /// Canonical tag. Append-only.
    #[must_use]
    pub const fn tag(self) -> u64 {
        match self {
            Self::Point => 1,
            Self::Interval => 2,
            Self::Draws => 3,
        }
    }
}

/// What the draws of a source are a law of.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum LawMeaning {
    /// A sampling or bootstrap distribution of an estimator.
    Sampling,
    /// A posterior under a declared prior.
    Posterior,
}

named_variants! {
    LawMeaning {
        Sampling => "sampling",
        Posterior => "posterior",
    }
}

impl LawMeaning {
    /// Canonical tag. Append-only.
    #[must_use]
    pub const fn tag(self) -> u64 {
        match self {
            Self::Sampling => 1,
            Self::Posterior => 2,
        }
    }
}

/// How the aligned draw table of the sources was made.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum TableConstruction {
    /// Joint draws of the sources: the table carries whatever dependence the
    /// sources have.
    JointDraws,
    /// The product of independent source draws. It carries no dependence, so
    /// it is licensed only for sources that share no factor.
    IndependentProduct,
}

named_variants! {
    TableConstruction {
        JointDraws => "joint_draws",
        IndependentProduct => "independent_product",
    }
}

impl TableConstruction {
    /// Canonical tag. Append-only.
    #[must_use]
    pub const fn tag(self) -> u64 {
        match self {
            Self::JointDraws => 1,
            Self::IndependentProduct => 2,
        }
    }
}

/// What the chain's output is read as.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum ChainReading {
    /// The value under the declared dependence.
    Point,
    /// The value with its spread, labelled uncalibrated.
    Spread,
    /// Posterior draws of the output quantity.
    Posterior,
    /// The enclosure of the declared intervals.
    Enclosure,
    /// An interval of the composed quantity. Closed without a measured
    /// coverage record at the composed coordinate.
    Interval,
}

named_variants! {
    ChainReading {
        Point => "point",
        Spread => "spread",
        Posterior => "posterior",
        Enclosure => "enclosure",
        Interval => "interval",
    }
}

impl ChainReading {
    /// Canonical tag. Append-only.
    #[must_use]
    pub const fn tag(self) -> u64 {
        match self {
            Self::Point => 1,
            Self::Spread => 2,
            Self::Posterior => 3,
            Self::Enclosure => 4,
            Self::Interval => 5,
        }
    }

    /// The label a result read this way carries.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Point => PROPAGATION_POINT_LABEL,
            Self::Spread | Self::Interval => PROPAGATION_SPREAD_LABEL,
            Self::Posterior => POSTERIOR_PROPAGATION_LABEL,
            Self::Enclosure => PROPAGATION_ENCLOSURE_LABEL,
        }
    }
}

/// What a node of the chain holds after its stage.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum NodeRepresentation {
    /// A fixed value.
    Point,
    /// An enclosure.
    Enclosure,
    /// A column of draws.
    Draws,
}

named_variants! {
    NodeRepresentation {
        Point => "point",
        Enclosure => "enclosure",
        Draws => "draws",
    }
}

// ---------------------------------------------------------------- declaration

/// One shared factor a source or stage uses.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FactorDecl {
    /// The kind of factor.
    pub kind: SharedFactorKind,
    /// The factor's identity.
    pub id: String,
}

/// One source of the chain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChainSourceDecl {
    /// The source's identity.
    pub id: String,
    /// How the source carries its uncertainty.
    pub representation: Representation,
    /// What the draws are a law of. Present exactly for draws.
    pub meaning: Option<LawMeaning>,
    /// The alignment identity of the draw table the source is a column of.
    /// Present exactly for draws.
    pub alignment: Option<String>,
    /// Whether the factor list is complete. An incomplete list leaves the
    /// overlap with other sources unknown.
    pub complete_disclosure: bool,
    /// The factors the source uses.
    pub factors: Vec<FactorDecl>,
}

/// One stage of the chain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChainStageDecl {
    /// The stage's identity.
    pub id: String,
    /// What the stage does.
    pub kind: StageKind,
    /// The operator class.
    pub class: OperatorClass,
    /// Identities of the sources and earlier stages it reads, in operator order.
    pub inputs: Vec<String>,
    /// Factors the stage brings itself.
    pub own_factors: Vec<FactorDecl>,
}

/// A declared chain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChainDecl {
    /// How the draw table was made.
    pub construction: TableConstruction,
    /// The sources.
    pub sources: Vec<ChainSourceDecl>,
    /// The stages, each reading only sources and earlier stages.
    pub stages: Vec<ChainStageDecl>,
    /// The stage whose output is the chain's answer.
    pub output: String,
    /// What the answer is read as.
    pub reading: ChainReading,
}

// -------------------------------------------------------------------- refusal

/// A typed refusal of a chain: a registered code, a namespaced detail, what
/// was refused and the attempt that would carry the uncertainty.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChainRefusal {
    code: &'static str,
    detail: &'static str,
    message: String,
    stage: Option<String>,
    next: NextAttempt,
}

impl ChainRefusal {
    /// A refusal from its parts.
    #[must_use]
    pub fn new(
        code: &'static str,
        detail: &'static str,
        message: impl Into<String>,
        stage: Option<&str>,
        next: NextAttempt,
    ) -> Self {
        Self { code, detail, message: message.into(), stage: stage.map(str::to_owned), next }
    }

    /// Registered reason code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        self.code
    }

    /// Namespaced detail, `propagation_chain.<rule>`.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        self.detail
    }

    /// What was refused.
    #[must_use]
    pub fn message(&self) -> &str {
        self.message.as_str()
    }

    /// The stage or source the refusal is about, when it is about one.
    #[must_use]
    pub fn stage(&self) -> Option<&str> {
        self.stage.as_deref()
    }

    /// The attempt that would carry the uncertainty.
    #[must_use = "the next attempt is what the caller should try"]
    pub const fn next_attempt(&self) -> NextAttempt {
        self.next
    }

    /// The message with the next attempt appended.
    #[must_use]
    pub fn message_with_next_attempt(&self) -> String {
        self.next.append_to(&self.message)
    }
}

impl fmt::Display for ChainRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.detail, self.message_with_next_attempt())
    }
}

impl std::error::Error for ChainRefusal {}

impl_refused!(ChainRefusal);

fn refuse(
    code: &'static str,
    detail: &'static str,
    message: impl Into<String>,
    stage: Option<&str>,
    next: NextAttempt,
) -> ChainRefusal {
    ChainRefusal::new(code, detail, message, stage, next)
}

fn malformed(message: impl Into<String>, stage: Option<&str>) -> ChainRefusal {
    refuse(INVALID_CODE, "propagation_chain.malformed", message, stage, FIX_CHAIN_DECLARATION)
}

// ----------------------------------------------------------------------- plan

/// The dependence of a stage that combines several inputs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageDependence {
    /// Independent, dependent or unknown.
    pub status: DependenceStatus,
    /// The shared factors, as `kind:id`, in order.
    pub shared: Vec<String>,
}

/// What the rule decided for one stage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StagePlan {
    /// The stage's identity.
    pub id: String,
    /// What the stage does.
    pub kind: StageKind,
    /// The operator class.
    pub class: OperatorClass,
    /// What the stage's output holds.
    pub representation: NodeRepresentation,
    /// What the draws are a law of, for draws.
    pub meaning: Option<LawMeaning>,
    /// The sources the stage depends on, directly or through earlier stages,
    /// in ascending order.
    pub sources: Vec<String>,
    /// The dependence of the stage's inputs, for a stage with two or more.
    pub dependence: Option<StageDependence>,
}

/// A chain the rule licensed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChainPlan {
    /// One plan per stage, in declaration order.
    pub stages: Vec<StagePlan>,
    /// The stage whose output is the answer.
    pub output: String,
    /// What the answer is read as.
    pub reading: ChainReading,
    /// The label a result carries.
    pub label: &'static str,
    /// The alignment identity of the draw table, when there are draws.
    pub alignment: Option<String>,
    /// Point sources the answer depends on although the chain carries draws:
    /// their uncertainty is not carried.
    pub fixed_sources: Vec<String>,
}

impl ChainPlan {
    /// The plan of a stage.
    #[must_use]
    pub fn stage(&self, id: &str) -> Option<&StagePlan> {
        self.stages.iter().find(|stage| stage.id == id)
    }

    /// Stages that depend on `source`, directly or through earlier stages, in
    /// declaration order. A change to the source reaches these and no others.
    #[must_use]
    pub fn dependents_of(&self, source: &str) -> Vec<&str> {
        self.stages
            .iter()
            .filter(|stage| stage.sources.iter().any(|s| s == source))
            .map(|stage| stage.id.as_str())
            .collect()
    }
}

// ---------------------------------------------------------------------- check

#[derive(Clone, Copy)]
struct NodeFacts {
    representation: NodeRepresentation,
    meaning: Option<LawMeaning>,
}

type FactorSet = BTreeSet<(SharedFactorKind, String)>;

fn blank(text: &str) -> bool {
    antecedent_core::is_blank(text.trim())
}

fn check_text(text: &str, what: &str) -> Result<(), ChainRefusal> {
    if blank(text) {
        return Err(malformed(format!("{what} is blank"), None));
    }
    if text.len() > MAX_TEXT_BYTES {
        return Err(malformed(format!("{what} is longer than {MAX_TEXT_BYTES} bytes"), None));
    }
    Ok(())
}

fn factor_set(factors: &[FactorDecl], owner: &str) -> Result<FactorSet, ChainRefusal> {
    let mut out = FactorSet::new();
    for factor in factors {
        if blank(&factor.id) {
            return Err(malformed("a factor id is blank", Some(owner)));
        }
        out.insert((factor.kind, factor.id.clone()));
    }
    Ok(out)
}

fn dependence_source(
    id: &str,
    factors: &FactorSet,
    complete: bool,
) -> Result<DependenceSource, ChainRefusal> {
    let mut sets = SharedFactorSets::empty();
    for (kind, factor) in factors {
        sets.try_insert(*kind, factor).map_err(|error| malformed(error.to_string(), Some(id)))?;
    }
    let source = if complete {
        DependenceSource::try_declared(id, sets)
    } else {
        DependenceSource::try_with_unknown_overlap(id, sets)
    };
    source.map_err(|error| malformed(error.to_string(), Some(id)))
}

fn implicit_factor(source: &str) -> (SharedFactorKind, String) {
    (SharedFactorKind::Row, format!("chain_source:{source}"))
}

fn class_fits(kind: StageKind, class: OperatorClass) -> bool {
    match kind {
        StageKind::Pooling => class.is_pool(),
        StageKind::PreposteriorUpdate => class == OperatorClass::Update,
        StageKind::MeasurementMapping | StageKind::DecisionUtility => {
            matches!(class, OperatorClass::Affine | OperatorClass::Nonlinear)
        }
        StageKind::Transport | StageKind::DerivedQuantity => {
            matches!(
                class,
                OperatorClass::Affine | OperatorClass::Reweight | OperatorClass::Nonlinear
            )
        }
    }
}

fn factor_label(kind: SharedFactorKind, id: &str) -> String {
    format!("{}:{id}", kind.as_str())
}

/// Check a declared chain.
///
/// # Errors
///
/// The first refusal, in this order: a malformed declaration; an operator that
/// does not fit its stage kind; draws that are not aligned; a table built as an
/// independent product over sources that share a factor; an input
/// representation the stage cannot carry; mixed law meanings; an independent
/// pool of overlapping or undeclared evidence; a reading the output
/// representation cannot give.
#[allow(
    clippy::too_many_lines,
    reason = "one linear pass over the checks, in the order the refusals are documented"
)]
pub fn check_propagation_chain(decl: &ChainDecl) -> Result<ChainPlan, ChainRefusal> {
    // ---- structure
    if decl.sources.is_empty() {
        return Err(malformed("a chain names at least one source", None));
    }
    if decl.stages.is_empty() {
        return Err(malformed("a chain names at least one stage", None));
    }
    if decl.sources.len() + decl.stages.len() > MAX_CHAIN_NODES {
        return Err(refuse(
            INVALID_CODE,
            "propagation_chain.too_many_nodes",
            format!("a chain names at most {MAX_CHAIN_NODES} sources and stages"),
            None,
            RAISE_BUDGET_OR_THIN_DRAWS,
        ));
    }
    check_text(&decl.output, "the output stage id")?;
    let mut facts: BTreeMap<&str, NodeFacts> = BTreeMap::new();
    let mut factors: BTreeMap<String, FactorSet> = BTreeMap::new();
    let mut disclosure: BTreeMap<String, bool> = BTreeMap::new();
    let mut ancestors: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for source in &decl.sources {
        check_text(&source.id, "a source id")?;
        if facts.contains_key(source.id.as_str()) {
            return Err(malformed("a source id repeats a node id", Some(&source.id)));
        }
        let draws = source.representation == Representation::Draws;
        if draws != source.meaning.is_some() {
            return Err(malformed(
                "draws declare what they are a law of, and nothing else does",
                Some(&source.id),
            ));
        }
        match &source.alignment {
            Some(alignment) if draws => check_text(alignment, "an alignment id")?,
            None if !draws => {}
            _ => {
                return Err(malformed(
                    "draws declare their alignment identity, and nothing else does",
                    Some(&source.id),
                ));
            }
        }
        let mut own = factor_set(&source.factors, &source.id)?;
        own.insert(implicit_factor(&source.id));
        facts.insert(
            source.id.as_str(),
            NodeFacts {
                representation: match source.representation {
                    Representation::Point => NodeRepresentation::Point,
                    Representation::Interval => NodeRepresentation::Enclosure,
                    Representation::Draws => NodeRepresentation::Draws,
                },
                meaning: source.meaning,
            },
        );
        factors.insert(source.id.clone(), own);
        disclosure.insert(source.id.clone(), source.complete_disclosure);
        ancestors.insert(source.id.clone(), BTreeSet::from([source.id.clone()]));
    }
    // ---- alignment of the draws
    let alignments: BTreeSet<&str> =
        decl.sources.iter().filter_map(|s| s.alignment.as_deref()).collect();
    let draw_sources: Vec<&ChainSourceDecl> =
        decl.sources.iter().filter(|s| s.representation == Representation::Draws).collect();
    if alignments.len() > 1 {
        let assessed = assess_sources(&draw_sources, &factors, &disclosure)?;
        let next = if assessed.is_independent() {
            BUILD_INDEPENDENT_PRODUCT_TABLE
        } else {
            SUPPLY_ALIGNED_DRAWS
        };
        return Err(refuse(
            JOINT_LAW_CODE,
            "propagation_chain.unaligned_draws",
            format!(
                "the draws of {} carry {} different alignment identities; rows of unrelated draws are never paired by index",
                draw_sources.iter().map(|s| s.id.as_str()).collect::<Vec<_>>().join(", "),
                alignments.len()
            ),
            None,
            next,
        ));
    }
    let intervals = decl.sources.iter().any(|s| s.representation == Representation::Interval);
    if intervals && !draw_sources.is_empty() {
        return Err(refuse(
            REPRESENTATION_CODE,
            "propagation_chain.mixed_draws_and_interval",
            "a chain carries draws or intervals, not both: an interval has no law to pair with draws",
            None,
            SUPPLY_DRAWS_FOR_INTERVAL,
        ));
    }
    if decl.construction == TableConstruction::IndependentProduct && draw_sources.len() > 1 {
        let assessed = assess_sources(&draw_sources, &factors, &disclosure)?;
        if assessed.is_dependent() {
            let shared: Vec<String> = assessed
                .shared_factors()
                .iter()
                .map(|f| factor_label(f.kind(), f.id().as_str()))
                .collect();
            return Err(refuse(
                DOUBLE_COUNTED_CODE,
                "propagation_chain.shared_factor_in_independent_product",
                format!(
                    "the sources share {}, but an independent product table carries no dependence",
                    shared.join(", ")
                ),
                None,
                SUPPLY_JOINT_DRAWS,
            ));
        }
        if assessed.is_unknown() {
            return Err(refuse(
                OVERLAP_UNKNOWN_CODE,
                "propagation_chain.overlap_unknown_in_independent_product",
                "a source left its overlap undeclared, so an independent product table is not licensed",
                None,
                DECLARE_SHARED_FACTORS,
            ));
        }
    }
    // ---- stages
    let mut plans: Vec<StagePlan> = Vec::with_capacity(decl.stages.len());
    for stage in &decl.stages {
        check_text(&stage.id, "a stage id")?;
        if facts.contains_key(stage.id.as_str()) {
            return Err(malformed("a stage id repeats a node id", Some(&stage.id)));
        }
        if stage.inputs.is_empty() {
            return Err(malformed("a stage reads at least one input", Some(&stage.id)));
        }
        if !class_fits(stage.kind, stage.class) {
            return Err(refuse(
                NOT_SUPPORTED_CODE,
                "propagation_chain.operator_not_licensed_for_stage",
                format!(
                    "a {} stage does not apply a {} operator",
                    stage.kind.as_str(),
                    stage.class.as_str()
                ),
                Some(&stage.id),
                FIX_CHAIN_DECLARATION,
            ));
        }
        if stage.class.is_pool() && stage.inputs.len() < 2 {
            return Err(malformed("a pool reads at least two inputs", Some(&stage.id)));
        }
        if stage.class == OperatorClass::Update && stage.inputs.len() != 2 {
            return Err(malformed(
                "an update reads the parameter and the utility it re-evaluates",
                Some(&stage.id),
            ));
        }
        let mut input_facts: Vec<NodeFacts> = Vec::with_capacity(stage.inputs.len());
        let mut union_factors = factor_set(&stage.own_factors, &stage.id)?;
        let mut union_complete = true;
        let mut stage_sources: BTreeSet<String> = BTreeSet::new();
        for input in &stage.inputs {
            let Some(found) = facts.get(input.as_str()) else {
                return Err(malformed(
                    "a stage reads only sources and earlier stages",
                    Some(&stage.id),
                ));
            };
            if plans.iter().any(|p| p.id == *input && p.class == OperatorClass::Update) {
                return Err(malformed("no stage may read a preposterior update", Some(&stage.id)));
            }
            input_facts.push(*found);
            if let Some(set) = factors.get(input) {
                union_factors.extend(set.iter().cloned());
            }
            union_complete &= disclosure.get(input).copied().unwrap_or(true);
            if let Some(set) = ancestors.get(input) {
                stage_sources.extend(set.iter().cloned());
            }
        }
        // ---- representation
        let has_draws = input_facts.iter().any(|f| f.representation == NodeRepresentation::Draws);
        let has_enclosure =
            input_facts.iter().any(|f| f.representation == NodeRepresentation::Enclosure);
        let has_point = input_facts.iter().any(|f| f.representation == NodeRepresentation::Point);
        if stage.class == OperatorClass::Update {
            if has_enclosure {
                return Err(refuse(
                    REPRESENTATION_CODE,
                    "propagation_chain.interval_is_not_a_posterior",
                    "an update re-evaluates a decision under the posterior of a parameter, and an interval is not a posterior",
                    Some(&stage.id),
                    SUPPLY_POSTERIOR_DRAWS,
                ));
            }
            if has_point {
                return Err(refuse(
                    REPRESENTATION_CODE,
                    "propagation_chain.point_has_no_posterior",
                    "an update re-evaluates a decision under the posterior of a parameter, and a fixed value has none",
                    Some(&stage.id),
                    SUPPLY_POSTERIOR_DRAWS,
                ));
            }
            if input_facts.iter().any(|f| f.meaning != Some(LawMeaning::Posterior)) {
                return Err(refuse(
                    REPRESENTATION_CODE,
                    "propagation_chain.sampling_law_is_not_a_posterior",
                    "an update needs posterior draws of the parameter; a sampling law is not updated by a likelihood",
                    Some(&stage.id),
                    SUPPLY_POSTERIOR_DRAWS,
                ));
            }
        } else if stage.class.is_pool() {
            if has_enclosure {
                return Err(refuse(
                    REPRESENTATION_CODE,
                    "propagation_chain.interval_cannot_be_pooled",
                    "a pool weights its inputs by their joint covariance, and an interval has none",
                    Some(&stage.id),
                    SUPPLY_DRAWS_FOR_INTERVAL,
                ));
            }
            if has_point {
                return Err(refuse(
                    REPRESENTATION_CODE,
                    "propagation_chain.point_cannot_be_pooled",
                    "a pool weights its inputs by their variances, and a fixed value has none",
                    Some(&stage.id),
                    SUPPLY_DRAWS_FOR_FIXED_INPUT,
                ));
            }
        } else if has_draws && has_enclosure {
            return Err(refuse(
                REPRESENTATION_CODE,
                "propagation_chain.mixed_draws_and_interval",
                "a stage reads draws and an interval together: the interval has no law to pair with the draws",
                Some(&stage.id),
                SUPPLY_DRAWS_FOR_INTERVAL,
            ));
        }
        let representation = if has_draws {
            NodeRepresentation::Draws
        } else if has_enclosure {
            NodeRepresentation::Enclosure
        } else {
            NodeRepresentation::Point
        };
        // ---- meaning
        let meanings: BTreeSet<LawMeaning> = input_facts.iter().filter_map(|f| f.meaning).collect();
        if meanings.len() > 1 {
            return Err(refuse(
                MEANING_CODE,
                "propagation_chain.law_meaning_mismatch",
                "a stage reads a sampling law together with a posterior",
                Some(&stage.id),
                MATCH_LAW_MEANING,
            ));
        }
        let meaning = meanings.into_iter().next();
        // ---- dependence of a stage that combines several inputs
        let dependence = if stage.inputs.len() >= 2 {
            let assessed = assess_inputs(&stage.inputs, &factors, &disclosure)?;
            if stage.class == OperatorClass::IndependentPool {
                if assessed.is_dependent() {
                    let shared: Vec<String> = assessed
                        .shared_factors()
                        .iter()
                        .map(|f| factor_label(f.kind(), f.id().as_str()))
                        .collect();
                    return Err(refuse(
                        DOUBLE_COUNTED_CODE,
                        "propagation_chain.pool_shares_evidence",
                        format!("an independent pool would count {} twice", shared.join(", ")),
                        Some(&stage.id),
                        POOL_WITH_JOINT_COVARIANCE,
                    ));
                }
                if assessed.is_unknown() {
                    return Err(refuse(
                        OVERLAP_UNKNOWN_CODE,
                        "propagation_chain.pool_overlap_unknown",
                        "an independent pool needs the overlap of its inputs declared",
                        Some(&stage.id),
                        DECLARE_SHARED_FACTORS,
                    ));
                }
            }
            Some(StageDependence {
                status: if assessed.is_dependent() {
                    DependenceStatus::Dependent
                } else if assessed.is_unknown() {
                    DependenceStatus::Unknown
                } else {
                    DependenceStatus::Independent
                },
                shared: assessed
                    .shared_factors()
                    .iter()
                    .map(|f| factor_label(f.kind(), f.id().as_str()))
                    .collect(),
            })
        } else {
            None
        };
        let stage_complete = union_complete;
        facts.insert(stage.id.as_str(), NodeFacts { representation, meaning });
        factors.insert(stage.id.clone(), union_factors);
        disclosure.insert(stage.id.clone(), stage_complete);
        ancestors.insert(stage.id.clone(), stage_sources.clone());
        plans.push(StagePlan {
            id: stage.id.clone(),
            kind: stage.kind,
            class: stage.class,
            representation,
            meaning,
            sources: stage_sources.into_iter().collect(),
            dependence,
        });
    }
    // ---- the answer
    let Some(output) = plans.iter().find(|plan| plan.id == decl.output) else {
        return Err(malformed("the output names a stage of the chain", Some(&decl.output)));
    };
    check_reading(decl.reading, output)?;
    let fixed_sources = if output.representation == NodeRepresentation::Draws {
        output
            .sources
            .iter()
            .filter(|id| {
                facts.get(id.as_str()).map(|f| f.representation) == Some(NodeRepresentation::Point)
            })
            .cloned()
            .collect()
    } else {
        Vec::new()
    };
    Ok(ChainPlan {
        stages: plans,
        output: decl.output.clone(),
        reading: decl.reading,
        label: decl.reading.label(),
        alignment: alignments.into_iter().next().map(str::to_owned),
        fixed_sources,
    })
}

fn check_reading(reading: ChainReading, output: &StagePlan) -> Result<(), ChainRefusal> {
    let stage = Some(output.id.as_str());
    if reading == ChainReading::Interval {
        return Err(refuse(
            NOT_MEASURED_CODE,
            "propagation_chain.interval_not_measured",
            "an interval of a composed quantity needs a measured coverage record at its composed coordinate, and a component interval does not cover the composition",
            stage,
            MEASURE_COMPOSED_COVERAGE,
        ));
    }
    match output.representation {
        NodeRepresentation::Point => {
            if matches!(reading, ChainReading::Spread | ChainReading::Posterior) {
                return Err(refuse(
                    REPRESENTATION_CODE,
                    "propagation_chain.point_has_no_spread",
                    "every input is a fixed value, so the chain has a value and no spread or posterior",
                    stage,
                    SUPPLY_DRAWS_FOR_FIXED_INPUT,
                ));
            }
        }
        NodeRepresentation::Enclosure => match reading {
            ChainReading::Enclosure => {}
            ChainReading::Posterior => {
                return Err(refuse(
                    REPRESENTATION_CODE,
                    "propagation_chain.interval_is_not_a_posterior",
                    "the chain carries intervals, and an interval is not a posterior",
                    stage,
                    SUPPLY_POSTERIOR_DRAWS,
                ));
            }
            _ => {
                return Err(refuse(
                    REPRESENTATION_CODE,
                    "propagation_chain.enclosure_is_not_a_law",
                    "the chain carries intervals, which hold under every coupling and give neither a value nor a spread",
                    stage,
                    USE_ENCLOSURE_READING,
                ));
            }
        },
        NodeRepresentation::Draws => match reading {
            ChainReading::Point | ChainReading::Spread => {}
            ChainReading::Posterior => {
                if output.meaning != Some(LawMeaning::Posterior) {
                    return Err(refuse(
                        REPRESENTATION_CODE,
                        "propagation_chain.sampling_law_is_not_a_posterior",
                        "the draws are a sampling law, and a sampling law is not a posterior",
                        stage,
                        SUPPLY_POSTERIOR_DRAWS,
                    ));
                }
                if output.class == OperatorClass::Update {
                    return Err(malformed(
                        "an update reports a value of information, not posterior draws",
                        stage,
                    ));
                }
            }
            ChainReading::Enclosure | ChainReading::Interval => {
                return Err(refuse(
                    REPRESENTATION_CODE,
                    "propagation_chain.draws_are_not_an_enclosure",
                    "the chain carries draws, and the range of draws is not the image of a declared box",
                    stage,
                    FIX_CHAIN_DECLARATION,
                ));
            }
        },
    }
    Ok(())
}

fn assess_sources(
    sources: &[&ChainSourceDecl],
    factors: &BTreeMap<String, FactorSet>,
    disclosure: &BTreeMap<String, bool>,
) -> Result<DependenceContract, ChainRefusal> {
    let ids: Vec<String> = sources.iter().map(|s| s.id.clone()).collect();
    assess_inputs(&ids, factors, disclosure)
}

fn assess_inputs(
    inputs: &[String],
    factors: &BTreeMap<String, FactorSet>,
    disclosure: &BTreeMap<String, bool>,
) -> Result<DependenceContract, ChainRefusal> {
    let mut sources = Vec::with_capacity(inputs.len());
    for input in inputs {
        let set = factors.get(input).cloned().unwrap_or_default();
        let complete = disclosure.get(input).copied().unwrap_or(true);
        sources.push(dependence_source(input, &set, complete)?);
    }
    DependenceContract::assess(&sources).map_err(|error| {
        refuse(
            INVALID_CODE,
            "propagation_chain.too_many_inputs",
            error.to_string(),
            None,
            FIX_CHAIN_DECLARATION,
        )
    })
}

#[cfg(test)]
#[path = "propagation_chain_tests.rs"]
mod tests;
