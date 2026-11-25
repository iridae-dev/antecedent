//! Layer four of the lowered program: evidence composition and decision demand.
//!
//! Which sources the program reads and how they combine, and, when compile starts
//! from a decision or a value-of-information request, what that decision needs the
//! statistical program to publish.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::DecisionRequest;
use antecedent_core::{CanonicalWriter, SemanticDigest, VariableId};

use super::planning::canon::CanonPut;

#[path = "child_codec.rs"]
pub(crate) mod child_codec;

const EVIDENCE_PROGRAM_TAG: &str = "antecedent.lowered_program.evidence";

/// What a source contributes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum EvidenceRole {
    /// Observational rows of the study.
    Observations,
    /// Rows of a randomized experiment.
    Trial,
    /// A prior carried by an artifact.
    PriorArtifact,
    /// A source population that a transport reads.
    TransportSource,
    /// The compiled search and review a discovered structure came from.
    DiscoveredStructure,
    /// A source of a propagation chain: a column of draws, an interval or a point.
    ChainSource,
    /// Explicitly joint states of a compiled robust scenario terminal.
    JointScenarios,
}

impl EvidenceRole {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Observations => "observations",
            Self::Trial => "trial",
            Self::PriorArtifact => "prior_artifact",
            Self::TransportSource => "transport_source",
            Self::DiscoveredStructure => "discovered_structure",
            Self::ChainSource => "chain_source",
            Self::JointScenarios => "joint_scenarios",
        }
    }
}

/// One source the program reads.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct EvidenceSourceRef {
    pub(crate) role: EvidenceRole,
    pub(crate) digest: Option<[u8; 32]>,
}

impl EvidenceSourceRef {
    /// What the source contributes.
    #[must_use]
    pub const fn role(&self) -> EvidenceRole {
        self.role
    }

    /// Content digest of the source, when the lowering bound one.
    #[must_use]
    pub const fn digest(&self) -> Option<[u8; 32]> {
        self.digest
    }
}

/// How the sources combine into one answer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Composition {
    /// One source, one structure.
    Single,
    /// A weighted mixture over graph-posterior atoms.
    AtomMixture {
        /// Number of atoms.
        atoms: usize,
    },
    /// A finite set of joint scenarios; membership does not imply mixing weights.
    ScenarioSet {
        /// Number of declared scenarios.
        scenarios: usize,
    },
    /// A source population transported to a target.
    Transported,
    /// Several sources pooled under a declared dependence.
    Pooled {
        /// Number of pooled sources.
        sources: usize,
    },
    /// Stages run in sequence, each fed by the stages before it.
    Chained {
        /// Number of stages.
        stages: usize,
    },
}

/// Evidence a decision needs: a published statistical answer or an explicit joint scenario set.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CarrierDemand {
    /// An affine utility of the mean; a point estimate carries it.
    Mean,
    /// Expected loss over a parameter; only a posterior carries it.
    ParameterPosterior,
    /// Risk or a threshold of an individual outcome; only an outcome law carries it.
    OutcomeLaw,
    /// A declared predictive law, without an interventional interpretation.
    PredictiveLaw,
    /// A declared joint identified-set/scenario carrier for robust evaluation.
    IdentifiedSet,
    /// Aligned per-candidate utilities for diagnostic decision reliability.
    DecisionDraws,
}

impl CarrierDemand {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::ParameterPosterior => "parameter_posterior",
            Self::OutcomeLaw => "outcome_law",
            Self::PredictiveLaw => "predictive_law",
            Self::IdentifiedSet => "identified_set",
            Self::DecisionDraws => "decision_draws",
        }
    }

    /// The evidence-port request this demand puts to the statistical program.
    ///
    /// A mean is not an uncertainty request, so it has none. A parameter
    /// posterior stays a parameter-posterior request; an outcome law is a
    /// request for individual outcome risk.
    #[must_use]
    pub const fn evidence_request(self) -> Option<DecisionRequest> {
        match self {
            Self::Mean | Self::IdentifiedSet | Self::DecisionDraws => None,
            Self::ParameterPosterior => Some(DecisionRequest::ParameterPosterior),
            Self::OutcomeLaw | Self::PredictiveLaw => Some(DecisionRequest::IndividualOutcomeRisk),
        }
    }
}

/// Form of the utility the decision program was compiled for.
#[derive(Clone, Debug, PartialEq)]
pub enum UtilityForm {
    /// `U(a, θ) = α_a + β_a·θ` in a scalar state: one `(α_a, β_a)` per action.
    Affine(Arc<[(f64, f64)]>),
    /// The utility declares no affine form. Its identity is the decision
    /// contract's; a mean cannot carry it.
    Opaque,
    /// A value over several outcome attributes whose trade-off parameters range over a
    /// declared or learned set. Its identity is the bound preference contract's; the
    /// program names every scientific quantity the actions read.
    PreferenceValue {
        /// Canonical identities of the quantities the actions read, in first-read order.
        quantities: Arc<[Arc<str>]>,
        /// Whether the means of those quantities answer the decision at every admissible
        /// parameter value: the criterion is an expectation and every part of the value
        /// is affine in them.
        means_suffice: bool,
        /// What the plan states about the decision: each quantity's coordinates, the source
        /// representation, the trade-off parameters with their admissible ranges and, for
        /// separately estimated outcomes joined into one law, the declared joint law.
        facts: Arc<PreferenceValueFacts>,
    },
}

/// One scientific quantity a preference value reads, as the plan states it.
#[derive(Clone, Debug, PartialEq)]
pub struct PreferenceQuantityFact {
    pub(crate) identity: Arc<str>,
    pub(crate) variable: Option<VariableId>,
    pub(crate) variable_id: Arc<str>,
    pub(crate) regime: Arc<str>,
    pub(crate) units: Arc<str>,
    pub(crate) population: Arc<str>,
    pub(crate) horizon: u32,
    pub(crate) functional: Arc<str>,
    pub(crate) transform: Arc<str>,
    pub(crate) source: usize,
}

impl PreferenceQuantityFact {
    /// Canonical identity of the coordinate.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// The schema variable, when the source is a study whose schema names it.
    #[must_use]
    pub const fn variable(&self) -> Option<VariableId> {
        self.variable
    }

    /// The coordinate's variable identity.
    #[must_use]
    pub fn variable_id(&self) -> &str {
        &self.variable_id
    }

    /// The regime the coordinate is read under.
    #[must_use]
    pub fn regime(&self) -> &str {
        &self.regime
    }

    /// Units of the coordinate.
    #[must_use]
    pub fn units(&self) -> &str {
        &self.units
    }

    /// Population of the coordinate.
    #[must_use]
    pub fn population(&self) -> &str {
        &self.population
    }

    /// Horizon of the coordinate.
    #[must_use]
    pub const fn horizon(&self) -> u32 {
        self.horizon
    }

    /// Functional of the coordinate.
    #[must_use]
    pub fn functional(&self) -> &str {
        &self.functional
    }

    /// Scale of the coordinate.
    #[must_use]
    pub fn transform(&self) -> &str {
        &self.transform
    }

    /// Index of the study or design that targets the coordinate.
    #[must_use]
    pub const fn source(&self) -> usize {
        self.source
    }
}

/// One trade-off parameter of a preference value, with its admissible range.
#[derive(Clone, Debug, PartialEq)]
pub struct PreferenceParameterFact {
    pub(crate) id: Arc<str>,
    pub(crate) units: Arc<str>,
    pub(crate) lower: f64,
    pub(crate) upper: f64,
    pub(crate) assumption: Arc<str>,
}

impl PreferenceParameterFact {
    /// Parameter identity.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Units of the parameter.
    #[must_use]
    pub fn units(&self) -> &str {
        &self.units
    }

    /// Lower end of the declared range.
    #[must_use]
    pub const fn lower(&self) -> f64 {
        self.lower
    }

    /// Upper end of the declared range.
    #[must_use]
    pub const fn upper(&self) -> f64 {
        self.upper
    }

    /// The assumption the range stands on.
    #[must_use]
    pub fn assumption(&self) -> &str {
        &self.assumption
    }
}

/// One column of a study's law the joint law reads: the chain source and stage it becomes.
#[derive(Clone, Debug, PartialEq)]
pub struct PreferenceChainInputFact {
    pub(crate) source: Arc<str>,
    pub(crate) stage: Arc<str>,
    pub(crate) study: usize,
    pub(crate) quantity: Arc<str>,
    pub(crate) factor: Arc<str>,
    pub(crate) factors: Arc<[Arc<str>]>,
    pub(crate) basis: Arc<str>,
    pub(crate) namespace: Option<Arc<str>>,
}

impl PreferenceChainInputFact {
    /// Identity of the chain source.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Identity of the stage whose output is the joined column.
    #[must_use]
    pub fn stage(&self) -> &str {
        &self.stage
    }

    /// Index of the study whose law supplies the column.
    #[must_use]
    pub const fn study(&self) -> usize {
        self.study
    }

    /// Canonical identity of the coordinate the column carries.
    #[must_use]
    pub fn quantity(&self) -> &str {
        &self.quantity
    }

    /// The first row factor the source declares: for a study whose overlap is judged by table
    /// content, the content identity of the table its rows came from (the source table of a
    /// column projection, else its own table).
    #[must_use]
    pub fn factor(&self) -> &str {
        &self.factor
    }

    /// Every row factor the source declares. A study with a declared unit key declares one per
    /// study it shares units with (named by the shared units), or one naming its own units
    /// when it shares none.
    #[must_use]
    pub fn factors(&self) -> &[Arc<str>] {
        &self.factors
    }

    /// How the study's overlap with the others was judged: `table_content` (the rows of the
    /// table it was taken from) or `unit_key` (its declared unit identifiers).
    #[must_use]
    pub fn basis(&self) -> &str {
        &self.basis
    }

    /// The identifier space of the study's unit key, when overlap was judged by unit key.
    #[must_use]
    pub fn namespace(&self) -> Option<&str> {
        self.namespace.as_deref()
    }
}

/// The joint law separately estimated outcomes are joined into, declared at compile: the
/// propagation chain's inputs, the declared dependence and the joint outputs it produces.
#[derive(Clone, Debug, PartialEq)]
pub struct PreferenceJointLawFact {
    pub(crate) inputs: Arc<[PreferenceChainInputFact]>,
    pub(crate) pairs: Arc<[(usize, usize, Arc<str>)]>,
    pub(crate) dependence: Arc<str>,
    pub(crate) cross_blocks: Arc<[(usize, usize)]>,
    pub(crate) draws: usize,
    pub(crate) seed: u64,
    pub(crate) reading: Arc<str>,
    pub(crate) label: Arc<str>,
    pub(crate) declaration: Arc<str>,
    pub(crate) program: SemanticDigest,
}

impl PreferenceJointLawFact {
    /// Every column the chain reads, in output order.
    #[must_use]
    pub fn inputs(&self) -> &[PreferenceChainInputFact] {
        &self.inputs
    }

    /// `(first, second, basis)` of every pair of studies, in pair order: how the evidence
    /// ledger judged whether the pair shares observations. `shared_rows` and `shared_units`
    /// for a pair that shares rows of one source table or units of one identifier space (it
    /// has a declared cross-study block); `disjoint_rows` and `disjoint_units` for a pair the
    /// ledger found sharing none; `declared_distinct` for studies of different tables without
    /// unit keys, which the ledger cannot compare and which are distinct on the declared
    /// dependence alone. Stated, not part of the declaration's digest.
    #[must_use]
    pub fn pairs(&self) -> &[(usize, usize, Arc<str>)] {
        &self.pairs
    }

    /// The declared dependence between the studies (`distinct_evidence` or
    /// `shared_evidence`).
    #[must_use]
    pub fn dependence(&self) -> &str {
        &self.dependence
    }

    /// The study pairs whose cross covariance is declared.
    #[must_use]
    pub fn cross_blocks(&self) -> &[(usize, usize)] {
        &self.cross_blocks
    }

    /// Aligned rows of the joined law.
    #[must_use]
    pub const fn draws(&self) -> usize {
        self.draws
    }

    /// Seed of the family table.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// What the chain's output is read as.
    #[must_use]
    pub fn reading(&self) -> &str {
        &self.reading
    }

    /// The label the chain rule gives the result.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Digest of the checked chain declaration and the declared dependence.
    #[must_use]
    pub fn declaration(&self) -> &str {
        &self.declaration
    }

    /// Identity of the chain's lowered program, compiled before anything is fitted: the
    /// program whose identity the execution's receipt carries for the chain.
    #[must_use]
    pub const fn program(&self) -> SemanticDigest {
        self.program
    }
}

/// The resampling law a compiled design draws for a decision that needs a joint law of its
/// arm means, declared at compile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreferenceDesignLawFact {
    pub(crate) kind: Arc<str>,
    pub(crate) unit: Arc<str>,
    pub(crate) replicates: usize,
    pub(crate) seed: u64,
    pub(crate) sampling: Option<Arc<str>>,
}

impl PreferenceDesignLawFact {
    /// The law: `static_regime_refits` (aligned whole-subject refits of a static-regime
    /// design) or `independent_unit_bootstrap` (refits over the independent units of a
    /// randomized design).
    #[must_use]
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// The unit resampled: `subject`, `cluster`, `pair` or `sequence`.
    #[must_use]
    pub fn unit(&self) -> &str {
        &self.unit
    }

    /// Complete refits.
    #[must_use]
    pub const fn replicates(&self) -> usize {
        self.replicates
    }

    /// Resampling seed.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Assumption identity of the declared premise that the units were sampled independently
    /// from a larger population; required for an independent-unit bootstrap.
    #[must_use]
    pub fn sampling(&self) -> Option<&str> {
        self.sampling.as_deref()
    }
}

/// What a plan states about a decision across preferences.
#[derive(Clone, Debug, PartialEq)]
pub struct PreferenceValueFacts {
    pub(crate) quantities: Arc<[PreferenceQuantityFact]>,
    pub(crate) representation: Arc<str>,
    pub(crate) accepted: Arc<[Arc<str>]>,
    pub(crate) parameters: Arc<[PreferenceParameterFact]>,
    pub(crate) joint_law: Option<PreferenceJointLawFact>,
    pub(crate) design_law: Option<PreferenceDesignLawFact>,
}

impl PreferenceValueFacts {
    /// Every quantity the actions read, in first-read order.
    #[must_use]
    pub fn quantities(&self) -> &[PreferenceQuantityFact] {
        &self.quantities
    }

    /// The representation the route reads (`mean` or `joint_draws`).
    #[must_use]
    pub fn representation(&self) -> &str {
        &self.representation
    }

    /// The representations the contract accepts.
    #[must_use]
    pub fn accepted(&self) -> &[Arc<str>] {
        &self.accepted
    }

    /// Every trade-off parameter with its admissible range.
    #[must_use]
    pub fn parameters(&self) -> &[PreferenceParameterFact] {
        &self.parameters
    }

    /// The declared joint law of separately estimated outcomes, when the decision reads one.
    #[must_use]
    pub const fn joint_law(&self) -> Option<&PreferenceJointLawFact> {
        self.joint_law.as_ref()
    }

    /// The resampling law a compiled design draws, when the decision reads one.
    #[must_use]
    pub const fn design_law(&self) -> Option<&PreferenceDesignLawFact> {
        self.design_law.as_ref()
    }
}

impl UtilityForm {
    /// Whether the utility is affine in a scalar state.
    #[must_use]
    pub const fn is_affine(&self) -> bool {
        matches!(self, Self::Affine(_))
    }

    /// Whether a mean carries the utility: an affine utility of a scalar state, or a
    /// preference value whose expectation the means of its quantities answer.
    #[must_use]
    pub const fn means_suffice(&self) -> bool {
        matches!(self, Self::Affine(_) | Self::PreferenceValue { means_suffice: true, .. })
    }

    /// Per-action `(intercept, slope)` pairs of an affine utility.
    #[must_use]
    pub fn coefficients(&self) -> Option<&[(f64, f64)]> {
        match self {
            Self::Affine(coefficients) => Some(&coefficients[..]),
            Self::Opaque | Self::PreferenceValue { .. } => None,
        }
    }

    /// Canonical identities of the scientific quantities a preference value reads; empty
    /// for any other utility.
    #[must_use]
    pub fn quantities(&self) -> &[Arc<str>] {
        match self {
            Self::PreferenceValue { quantities, .. } => &quantities[..],
            Self::Affine(_) | Self::Opaque => &[],
        }
    }

    /// What the plan states about a preference value; `None` for any other utility.
    #[must_use]
    pub fn preference_facts(&self) -> Option<&PreferenceValueFacts> {
        match self {
            Self::PreferenceValue { facts, .. } => Some(facts),
            Self::Affine(_) | Self::Opaque => None,
        }
    }

    fn write<S: CanonPut>(&self, writer: &mut S) {
        match self {
            Self::Affine(coefficients) => {
                writer.text("affine").word(coefficients.len() as u64);
                for &(intercept, slope) in &**coefficients {
                    writer.real(intercept).real(slope);
                }
            }
            Self::Opaque => {
                writer.text("opaque");
            }
            Self::PreferenceValue { quantities, means_suffice, facts } => {
                writer
                    .text("preference_value")
                    .word(u64::from(*means_suffice))
                    .texts(&quantities[..]);
                // The coordinates, representation and parameter ranges are bound by the
                // quantity identities and the bound contract identity; a declared joint law
                // is not, so it is written after a presence word.
                match &facts.joint_law {
                    Some(joint) => {
                        writer
                            .word(1)
                            .text("joint_law")
                            .text(&joint.declaration)
                            .bytes(joint.program.as_bytes());
                    }
                    None => {
                        writer.word(0);
                    }
                }
                // The resampling law of a design changes the law the decision reads, and the
                // design program's identity excludes the inference executed on it, so it is
                // written after a presence word.
                match &facts.design_law {
                    Some(law) => {
                        writer
                            .word(1)
                            .text("design_law")
                            .text(&law.kind)
                            .text(&law.unit)
                            .word(law.replicates as u64)
                            .word(law.seed)
                            .opt_text(law.sampling.as_deref());
                    }
                    None => {
                        writer.word(0);
                    }
                }
            }
        }
    }
}

/// Monte Carlo schedule of a preposterior ranking.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RankSchedule {
    pub(crate) min_batches: u32,
    pub(crate) max_batches: u32,
    pub(crate) batch_size: u32,
    pub(crate) rank_uncertainty_threshold: f64,
}

impl RankSchedule {
    /// Minimum batches before an early stop is considered.
    #[must_use]
    pub const fn min_batches(&self) -> u32 {
        self.min_batches
    }

    /// Maximum batches.
    #[must_use]
    pub const fn max_batches(&self) -> u32 {
        self.max_batches
    }

    /// Replicates per batch.
    #[must_use]
    pub const fn batch_size(&self) -> u32 {
        self.batch_size
    }

    /// Adjusted half-width at which adjacent ranks count as separated.
    #[must_use]
    pub const fn rank_uncertainty_threshold(&self) -> f64 {
        self.rank_uncertainty_threshold
    }

    /// Most draws one candidate can be scored on: the ranker runs at least
    /// `max(min_batches, 1)` batches and never fewer than it was asked for.
    #[must_use]
    pub fn draws_per_candidate(&self) -> u64 {
        let floor = self.min_batches.max(1);
        u64::from(self.max_batches.max(floor)) * u64::from(self.batch_size)
    }
}

/// Positive linear map from study cost to the decision's utility unit.
#[derive(Clone, Debug, PartialEq)]
pub struct CostMapping {
    pub(crate) cost_unit: Arc<str>,
    pub(crate) utility_unit: Arc<str>,
    pub(crate) utility_per_cost: f64,
}

impl CostMapping {
    /// Unit study costs are declared in.
    #[must_use]
    pub fn cost_unit(&self) -> &str {
        &self.cost_unit
    }

    /// Utility unit the cost maps into.
    #[must_use]
    pub fn utility_unit(&self) -> &str {
        &self.utility_unit
    }

    /// Utility per unit cost.
    #[must_use]
    pub const fn utility_per_cost(&self) -> f64 {
        self.utility_per_cost
    }
}

/// The Monte Carlo task dimension a preposterior evaluation schedules.
///
/// Every evaluated candidate is scored on every draw, and every draw carries one
/// common-random-number seed shared across candidates. The store therefore holds
/// `candidates × draws` scores and `draws` seeds. The quote is checked arithmetic;
/// a store whose size does not fit in `u64` has no quote.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct MonteCarloTasks {
    pub(crate) candidates: u64,
    pub(crate) draws: u64,
    pub(crate) tasks: u64,
    pub(crate) score_bytes: u64,
    pub(crate) seed_bytes: u64,
    pub(crate) total_bytes: u64,
}

impl MonteCarloTasks {
    /// Bytes of one candidate's score on one draw.
    pub const SCORE_BYTES: u64 = std::mem::size_of::<f64>() as u64;
    /// Bytes of one draw's common-random-number seed.
    pub const SEED_BYTES: u64 = std::mem::size_of::<u64>() as u64;

    /// Quote the store for `candidates` scored on `draws` draws each, or `None`
    /// when a product or the sum overflows `u64`.
    #[must_use]
    pub const fn quote(candidates: u64, draws: u64) -> Option<Self> {
        let Some(tasks) = candidates.checked_mul(draws) else {
            return None;
        };
        let Some(score_bytes) = tasks.checked_mul(Self::SCORE_BYTES) else {
            return None;
        };
        let Some(seed_bytes) = draws.checked_mul(Self::SEED_BYTES) else {
            return None;
        };
        let Some(total_bytes) = score_bytes.checked_add(seed_bytes) else {
            return None;
        };
        Some(Self { candidates, draws, tasks, score_bytes, seed_bytes, total_bytes })
    }

    /// Candidates evaluated.
    #[must_use]
    pub const fn candidates(&self) -> u64 {
        self.candidates
    }

    /// Draws each candidate can be scored on.
    #[must_use]
    pub const fn draws(&self) -> u64 {
        self.draws
    }

    /// `candidates × draws` scoring tasks.
    #[must_use]
    pub const fn tasks(&self) -> u64 {
        self.tasks
    }

    /// Bytes of the score store.
    #[must_use]
    pub const fn score_bytes(&self) -> u64 {
        self.score_bytes
    }

    /// Bytes of the per-draw seeds.
    #[must_use]
    pub const fn seed_bytes(&self) -> u64 {
        self.seed_bytes
    }

    /// Bytes the evaluation must hold at its cap.
    #[must_use]
    pub const fn total_bytes(&self) -> u64 {
        self.total_bytes
    }
}

/// Preposterior value-of-information request the decision carries.
///
/// The Monte Carlo task dimension ([`Self::monte_carlo`]) is physical accounting
/// derived from the candidates and the schedule, which the identity already
/// covers; it is not written to the identity again.
#[derive(Clone, Debug, PartialEq)]
pub struct EvsiProgram {
    pub(crate) candidates: Arc<[Arc<str>]>,
    pub(crate) prior_observation_ids: Arc<[Arc<str>]>,
    pub(crate) rng_seed: u64,
    pub(crate) mc_error_tolerance: f64,
    pub(crate) require_net_value: bool,
    pub(crate) tie_tolerance: f64,
    pub(crate) max_candidates: u64,
    pub(crate) rank: RankSchedule,
    pub(crate) cost_map: Option<CostMapping>,
    pub(crate) monte_carlo: MonteCarloTasks,
}

impl EvsiProgram {
    /// The ranking schedule of a request.
    pub(crate) const fn schedule(
        min_batches: u32,
        max_batches: u32,
        batch_size: u32,
        rank_uncertainty_threshold: f64,
    ) -> RankSchedule {
        RankSchedule { min_batches, max_batches, batch_size, rank_uncertainty_threshold }
    }

    /// The Monte Carlo store of `candidates` scored on `draws` draws each; see
    /// [`MonteCarloTasks::quote`].
    pub(crate) const fn quote_tasks(candidates: u64, draws: u64) -> Option<MonteCarloTasks> {
        MonteCarloTasks::quote(candidates, draws)
    }

    /// A declared cost-to-utility map.
    pub(crate) fn cost_mapping(
        cost_unit: &str,
        utility_unit: &str,
        utility_per_cost: f64,
    ) -> CostMapping {
        CostMapping {
            cost_unit: Arc::from(cost_unit),
            utility_unit: Arc::from(utility_unit),
            utility_per_cost,
        }
    }

    /// Semantic ids of the evaluated candidate designs, in canonical order.
    #[must_use]
    pub fn candidates(&self) -> &[Arc<str>] {
        &self.candidates
    }

    /// Observations already summarized by the prior.
    #[must_use]
    pub fn prior_observation_ids(&self) -> &[Arc<str>] {
        &self.prior_observation_ids
    }

    /// Seed of the Monte Carlo evaluation.
    #[must_use]
    pub const fn rng_seed(&self) -> u64 {
        self.rng_seed
    }

    /// Standard error at or below which an estimate counts as converged.
    #[must_use]
    pub const fn mc_error_tolerance(&self) -> f64 {
        self.mc_error_tolerance
    }

    /// Whether a missing cost-to-utility map refuses instead of reporting value and cost apart.
    #[must_use]
    pub const fn require_net_value(&self) -> bool {
        self.require_net_value
    }

    /// Absolute gap at or below which two candidates are tied.
    #[must_use]
    pub const fn tie_tolerance(&self) -> f64 {
        self.tie_tolerance
    }

    /// Search bound on candidates evaluated.
    #[must_use]
    pub const fn max_candidates(&self) -> u64 {
        self.max_candidates
    }

    /// Monte Carlo schedule of the ranking.
    #[must_use]
    pub const fn rank(&self) -> RankSchedule {
        self.rank
    }

    /// Cost-to-utility map, when one is declared.
    #[must_use]
    pub const fn cost_map(&self) -> Option<&CostMapping> {
        self.cost_map.as_ref()
    }

    /// Candidate × draw task dimension and its quoted bytes.
    #[must_use]
    pub const fn monte_carlo(&self) -> MonteCarloTasks {
        self.monte_carlo
    }

    fn write<S: CanonPut>(&self, writer: &mut S) {
        writer
            .texts(&self.candidates[..])
            .texts(&self.prior_observation_ids[..])
            .word(self.rng_seed)
            .real(self.mc_error_tolerance)
            .word(u64::from(self.require_net_value))
            .real(self.tie_tolerance)
            .word(self.max_candidates)
            .word(u64::from(self.rank.min_batches))
            .word(u64::from(self.rank.max_batches))
            .word(u64::from(self.rank.batch_size))
            .real(self.rank.rank_uncertainty_threshold);
        match &self.cost_map {
            Some(map) => {
                writer
                    .word(1)
                    .text(&map.cost_unit)
                    .text(&map.utility_unit)
                    .real(map.utility_per_cost);
            }
            None => {
                writer.word(0);
            }
        }
    }
}

/// The decision compile started from.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionProgram {
    pub(crate) carrier: CarrierDemand,
    pub(crate) utility: UtilityForm,
    pub(crate) actions: usize,
    pub(crate) constraints: usize,
    pub(crate) chance_threshold: f64,
    pub(crate) utility_unit: Option<Arc<str>>,
    pub(crate) contract_identity: Option<Arc<str>>,
    pub(crate) information_value: Option<EvsiProgram>,
    pub(crate) scenario_source: Option<Arc<str>>,
}

impl DecisionProgram {
    /// The utility form of declared per-action `(intercept, slope)` pairs, or an
    /// opaque utility when there are none.
    pub(crate) fn utility_form(affine: Option<Vec<(f64, f64)>>) -> UtilityForm {
        affine.map_or(UtilityForm::Opaque, |pairs| UtilityForm::Affine(Arc::from(pairs)))
    }

    /// The utility form of a preference value reading `quantities` (canonical identities),
    /// with what the plan states about it.
    pub(crate) fn preference_value_form(
        quantities: Vec<Arc<str>>,
        means_suffice: bool,
        facts: PreferenceValueFacts,
    ) -> UtilityForm {
        UtilityForm::PreferenceValue {
            quantities: Arc::from(quantities),
            means_suffice,
            facts: Arc::new(facts),
        }
    }

    /// Evidence the decision needs.
    #[must_use]
    pub const fn carrier(&self) -> CarrierDemand {
        self.carrier
    }

    /// Form of the utility.
    #[must_use]
    pub const fn utility(&self) -> &UtilityForm {
        &self.utility
    }

    /// Number of candidate actions.
    #[must_use]
    pub const fn actions(&self) -> usize {
        self.actions
    }

    /// Number of constraints on admissible actions.
    #[must_use]
    pub const fn constraints(&self) -> usize {
        self.constraints
    }

    /// Minimum chance-constraint satisfaction.
    #[must_use]
    pub const fn chance_threshold(&self) -> f64 {
        self.chance_threshold
    }

    /// Unit of the utility, when declared.
    #[must_use]
    pub fn utility_unit(&self) -> Option<&str> {
        self.utility_unit.as_deref()
    }

    /// Identity of the decision contract, when declared.
    #[must_use]
    pub fn contract_identity(&self) -> Option<&str> {
        self.contract_identity.as_deref()
    }

    /// Value-of-information request, when compile started from one.
    #[must_use]
    pub const fn information_value(&self) -> Option<&EvsiProgram> {
        self.information_value.as_ref()
    }

    /// Exact explicit joint scenario carrier, when this is a robust terminal.
    #[must_use]
    pub fn scenario_source_identity(&self) -> Option<&str> {
        self.scenario_source.as_deref()
    }

    fn write<S: CanonPut>(&self, writer: &mut S) {
        writer.text(self.carrier.as_str());
        self.utility.write(writer);
        writer
            .word(self.actions as u64)
            .word(self.constraints as u64)
            .real(self.chance_threshold)
            .opt_text(self.utility_unit.as_deref())
            .opt_text(self.contract_identity.as_deref());
        // Absent for existing programs: preserve their identities byte for byte.
        if let Some(source) = &self.scenario_source {
            writer.text("joint_scenarios_v1").text(source);
        }
        match &self.information_value {
            Some(evsi) => {
                writer.word(1);
                evsi.write(writer);
            }
            None => {
                writer.word(0);
            }
        }
    }
}

/// Layer four: evidence composition and the decision demand.
#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceDecisionProgram {
    pub(crate) sources: Arc<[EvidenceSourceRef]>,
    pub(crate) composition: Composition,
    pub(crate) decision: Option<DecisionProgram>,
    pub(crate) planning: Option<super::planning::PlanningProgram>,
    pub(crate) chain: Option<super::chain::ChainProgram>,
}

impl EvidenceDecisionProgram {
    /// Sources the program reads.
    #[must_use]
    pub fn sources(&self) -> &[EvidenceSourceRef] {
        &self.sources
    }

    /// How the sources combine.
    #[must_use]
    pub const fn composition(&self) -> Composition {
        self.composition
    }

    /// The decision compile started from, when it started from one.
    #[must_use]
    pub const fn decision(&self) -> Option<&DecisionProgram> {
        self.decision.as_ref()
    }

    /// The planning request compile started from, when it started from one.
    #[must_use]
    pub const fn planning(&self) -> Option<&super::planning::PlanningProgram> {
        self.planning.as_ref()
    }

    /// The propagation chain compile started from, when it started from one.
    #[must_use]
    pub const fn chain(&self) -> Option<&super::chain::ChainProgram> {
        self.chain.as_ref()
    }

    /// Identity of this layer.
    pub(crate) fn identity(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new(EVIDENCE_PROGRAM_TAG);
        self.encode_identity(&mut writer);
        SemanticDigest::from_bytes(writer.finish())
    }

    /// The frames [`Self::identity`] hashes. A child codec reads them back.
    fn encode_identity<S: CanonPut>(&self, writer: &mut S) {
        writer.word(self.sources.len() as u64);
        for source in &*self.sources {
            writer.text(source.role.as_str());
            match &source.digest {
                Some(digest) => writer.word(1).bytes(digest),
                None => writer.word(0),
            };
        }
        match self.composition {
            Composition::Single => writer.text("single"),
            Composition::AtomMixture { atoms } => writer.text("atom_mixture").word(atoms as u64),
            Composition::ScenarioSet { scenarios } => {
                writer.text("scenario_set").word(scenarios as u64)
            }
            Composition::Transported => writer.text("transported"),
            Composition::Pooled { sources } => writer.text("pooled").word(sources as u64),
            Composition::Chained { stages } => writer.text("chained").word(stages as u64),
        };
        match &self.decision {
            Some(decision) => {
                writer.word(1);
                decision.write(writer);
            }
            None => {
                writer.word(0);
            }
        }
        // Written only when present, so a program without a planning request
        // keeps the identity it had before planning goals existed.
        if let Some(planning) = &self.planning {
            planning.write(writer);
        }
        // Likewise for a propagation chain.
        if let Some(chain) = &self.chain {
            chain.write(writer);
        }
    }
}

#[cfg(test)]
mod site_tests {
    use super::*;
    use crate::identity_laws::{
        ADVERSARIAL_TEXT, all_distinct, bit_exact, injective, ordered_laws,
    };

    fn decision() -> DecisionProgram {
        DecisionProgram {
            carrier: CarrierDemand::Mean,
            utility: DecisionProgram::utility_form(None),
            actions: 2,
            constraints: 0,
            chance_threshold: 0.95,
            utility_unit: None,
            contract_identity: None,
            information_value: None,
            scenario_source: None,
        }
    }

    fn evsi(candidates: &[&str], prior: &[&str]) -> EvsiProgram {
        EvsiProgram {
            candidates: candidates.iter().map(|&id| Arc::<str>::from(id)).collect(),
            prior_observation_ids: prior.iter().map(|&id| Arc::<str>::from(id)).collect(),
            rng_seed: 3,
            mc_error_tolerance: 0.01,
            require_net_value: false,
            tie_tolerance: 0.0,
            max_candidates: 8,
            rank: EvsiProgram::schedule(1, 4, 16, 0.0),
            cost_map: None,
            monte_carlo: EvsiProgram::quote_tasks(2, 64).expect("a small store"),
        }
    }

    fn program(
        sources: Vec<EvidenceSourceRef>,
        decision: Option<DecisionProgram>,
    ) -> SemanticDigest {
        EvidenceDecisionProgram {
            sources: Arc::from(sources),
            composition: Composition::Single,
            decision,
            planning: None,
            chain: None,
        }
        .identity()
    }

    fn source(role: EvidenceRole, digest: Option<u8>) -> EvidenceSourceRef {
        EvidenceSourceRef { role, digest: digest.map(|byte| [byte; 32]) }
    }

    fn composed(composition: Composition) -> SemanticDigest {
        EvidenceDecisionProgram {
            sources: Arc::from(Vec::new()),
            composition,
            decision: None,
            planning: None,
            chain: None,
        }
        .identity()
    }

    #[test]
    fn site_decision_evidence_program_orders_sources_and_tags_its_parts() {
        // Sources are written in the role order the lowering declares them.
        ordered_laws(
            &[
                source(EvidenceRole::Observations, Some(1)),
                source(EvidenceRole::Trial, Some(2)),
                source(EvidenceRole::PriorArtifact, None),
            ],
            |sources: &[EvidenceSourceRef]| Some(program(sources.to_vec(), None)),
        );
        all_distinct(&[
            program(Vec::new(), None),
            program(vec![source(EvidenceRole::Observations, None)], None),
            program(vec![source(EvidenceRole::Observations, Some(0))], None),
            program(vec![source(EvidenceRole::Trial, None)], None),
            program(Vec::new(), Some(decision())),
            composed(Composition::Transported),
            composed(Composition::AtomMixture { atoms: 2 }),
            composed(Composition::AtomMixture { atoms: 3 }),
            composed(Composition::ScenarioSet { scenarios: 2 }),
            composed(Composition::Pooled { sources: 2 }),
            composed(Composition::Chained { stages: 2 }),
        ]);
        assert_eq!(program(Vec::new(), None), composed(Composition::Single));
    }

    fn with(edit: fn(&mut DecisionProgram)) -> SemanticDigest {
        let mut changed = decision();
        edit(&mut changed);
        program(Vec::new(), Some(changed))
    }

    #[test]
    fn site_decision_decision_program_frames_its_optional_fields() {
        all_distinct(&[
            program(Vec::new(), Some(decision())),
            with(|d| d.utility_unit = Some(Arc::from("u"))),
            with(|d| d.contract_identity = Some(Arc::from("u"))),
            with(|d| d.scenario_source = Some(Arc::from("u"))),
            with(|d| d.information_value = Some(evsi(&["a"], &[]))),
            with(|d| d.carrier = CarrierDemand::OutcomeLaw),
            with(|d| d.actions = 3),
            with(|d| d.constraints = 1),
        ]);
        bit_exact(|threshold| {
            let mut changed = decision();
            changed.chance_threshold = threshold;
            Some(program(Vec::new(), Some(changed)))
        });
    }

    fn evsi_digest(evsi: &EvsiProgram) -> [u8; 32] {
        let mut writer = CanonicalWriter::new("site");
        evsi.write(&mut writer);
        writer.finish()
    }

    #[test]
    fn site_decision_evsi_frames_candidates_prior_ids_and_reals() {
        // Candidate and prior-observation lists are count-prefixed and each id framed.
        injective(
            &ADVERSARIAL_TEXT,
            |parts| *parts,
            |(candidate, prior)| Some(evsi_digest(&evsi(&[*candidate], &[*prior]))),
        );
        all_distinct(&[
            evsi_digest(&evsi(&["a", "b"], &[])),
            evsi_digest(&evsi(&["a"], &["b"])),
            evsi_digest(&evsi(&[], &["a", "b"])),
            evsi_digest(&evsi(&["ab"], &[])),
            evsi_digest(&evsi(&["a"], &["ab"])),
            evsi_digest(&evsi(&["a"], &["a", "b"])),
        ]);
        bit_exact(|tolerance| {
            let mut program = evsi(&["a"], &[]);
            program.mc_error_tolerance = tolerance;
            Some(evsi_digest(&program))
        });
        bit_exact(|tie| {
            let mut program = evsi(&["a"], &[]);
            program.tie_tolerance = tie;
            Some(evsi_digest(&program))
        });
    }
}
