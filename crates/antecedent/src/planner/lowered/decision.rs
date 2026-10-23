//! Layer four of the lowered program: evidence composition and decision demand.
//!
//! Which sources the program reads and how they combine, and, when compile starts
//! from a decision or a value-of-information request, what that decision needs the
//! statistical program to publish.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::DecisionRequest;
use antecedent_core::{CanonicalWriter, SemanticDigest};

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
    /// A source population transported to a target.
    Transported,
    /// Several sources pooled under a declared dependence.
    Pooled {
        /// Number of pooled sources.
        sources: usize,
    },
}

/// The evidence a decision needs the statistical program to publish.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CarrierDemand {
    /// An affine utility of the mean; a point estimate carries it.
    Mean,
    /// Expected loss over a parameter; only a posterior carries it.
    ParameterPosterior,
    /// Risk or a threshold of an individual outcome; only an outcome law carries it.
    OutcomeLaw,
}

impl CarrierDemand {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::ParameterPosterior => "parameter_posterior",
            Self::OutcomeLaw => "outcome_law",
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
            Self::Mean => None,
            Self::ParameterPosterior => Some(DecisionRequest::ParameterPosterior),
            Self::OutcomeLaw => Some(DecisionRequest::IndividualOutcomeRisk),
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
}

impl UtilityForm {
    /// Whether the utility is affine in a scalar state.
    #[must_use]
    pub const fn is_affine(&self) -> bool {
        matches!(self, Self::Affine(_))
    }

    /// Per-action `(intercept, slope)` pairs of an affine utility.
    #[must_use]
    pub fn coefficients(&self) -> Option<&[(f64, f64)]> {
        match self {
            Self::Affine(coefficients) => Some(&coefficients[..]),
            Self::Opaque => None,
        }
    }

    fn write(&self, writer: &mut CanonicalWriter) {
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

    fn write(&self, writer: &mut CanonicalWriter) {
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
}

impl DecisionProgram {
    /// The utility form of declared per-action `(intercept, slope)` pairs, or an
    /// opaque utility when there are none.
    pub(crate) fn utility_form(affine: Option<Vec<(f64, f64)>>) -> UtilityForm {
        affine.map_or(UtilityForm::Opaque, |pairs| UtilityForm::Affine(Arc::from(pairs)))
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

    fn write(&self, writer: &mut CanonicalWriter) {
        writer.text(self.carrier.as_str());
        self.utility.write(writer);
        writer
            .word(self.actions as u64)
            .word(self.constraints as u64)
            .real(self.chance_threshold)
            .opt_text(self.utility_unit.as_deref())
            .opt_text(self.contract_identity.as_deref());
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

    /// Identity of this layer.
    pub(crate) fn identity(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new(EVIDENCE_PROGRAM_TAG);
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
            Composition::Transported => writer.text("transported"),
            Composition::Pooled { sources } => writer.text("pooled").word(sources as u64),
        };
        match &self.decision {
            Some(decision) => {
                writer.word(1);
                decision.write(&mut writer);
            }
            None => {
                writer.word(0);
            }
        }
        // Written only when present, so a program without a planning request
        // keeps the identity it had before planning goals existed.
        if let Some(planning) = &self.planning {
            planning.write(&mut writer);
        }
        SemanticDigest::from_bytes(writer.finish())
    }
}
