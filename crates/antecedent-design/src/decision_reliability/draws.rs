//! The decision-draws carrier: per-draw utilities of a finite candidate set, the stream
//! each candidate's draws came from, and the declarations the reliability claims rest on.
//!
//! The carrier is estimator-free. Producers that will feed it: the cohort-decision route
//! (fitted effect model to cohort-mean decision) supplies per-draw cohort utilities with
//! `producer = "cohort_decision"`; the sequential-evaluation route supplies per-draw
//! policy values with `producer = "sequential_evaluation"`; a design-ranking or
//! decision-evaluation run that kept its simulated utilities may supply them with its
//! own producer label. Each producer declares the dependence; nothing is inferred.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::{CanonicalWriter, is_blank};
use serde::{Deserialize, Serialize};

use super::stream::{DrawSegment, StreamId};
use super::{MAX_CANDIDATES, MAX_CELLS, MIN_DRAWS, ReliabilityError};

/// What law the draws are realizations of, and so which paradigm a ranking statement
/// carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceParadigm {
    /// Posterior or posterior-predictive draws: statements are posterior probabilities
    /// under the supplied posterior.
    Posterior,
    /// Replicates of a repeated-sampling simulation: statements are sampling
    /// frequencies under the simulated design.
    SamplingDistribution,
    /// The law is known exactly and the draws are Monte Carlo noise only: there is no
    /// source uncertainty to report.
    KnownLaw,
}

impl SourceParadigm {
    /// Stable lowercase label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Posterior => "posterior",
            Self::SamplingDistribution => "sampling_distribution",
            Self::KnownLaw => "known_law",
        }
    }
}

/// How the candidates' draws depend on each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateCoupling {
    /// Draw `i` of every candidate is the same simulated state (common random numbers
    /// or one posterior sample). Candidates are paired by draw index.
    SharedSimulation,
    /// The source law declares the candidates' utilities independent, and each
    /// candidate was drawn on its own stream; index pairing forms a product-law sample.
    DeclaredIndependent,
    /// Not declared. Refused wherever candidates are compared.
    Unknown,
}

impl CandidateCoupling {
    /// Stable lowercase label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SharedSimulation => "shared_simulation",
            Self::DeclaredIndependent => "declared_independent",
            Self::Unknown => "unknown",
        }
    }
}

/// Serial dependence of the draws within one candidate's stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WithinStreamDependence {
    /// Draws are serially independent.
    Independent,
    /// Draws form a dependent stream (Markov chain); standard errors use batch means.
    Autocorrelated,
    /// Not declared; every estimate refuses.
    Unknown,
}

impl WithinStreamDependence {
    /// Stable lowercase label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Independent => "independent",
            Self::Autocorrelated => "autocorrelated",
            Self::Unknown => "unknown",
        }
    }
}

/// How the finite candidate set came to exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateOrigin {
    /// The set was fixed before any utility was evaluated.
    Enumerated,
    /// The set was produced by a search that looked at evaluated utilities.
    AdaptivelyGenerated,
}

impl CandidateOrigin {
    /// Stable lowercase label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Enumerated => "enumerated",
            Self::AdaptivelyGenerated => "adaptively_generated",
        }
    }
}

/// The criterion by which one decision was chosen among the evaluated candidates.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionRule {
    /// Largest expected utility.
    MaximizeExpectedUtility,
    /// Largest probability of exceeding a utility threshold.
    MaximizeThresholdProbability,
    /// Largest utility quantile.
    MaximizeQuantile,
    /// Largest lower-tail conditional value at risk.
    MaximizeCvar,
    /// Any other declared rule, by label.
    Declared(String),
}

impl SelectionRule {
    /// Stable label of the rule family (the calibration coordinate).
    #[must_use]
    pub fn kind(&self) -> String {
        match self {
            Self::MaximizeExpectedUtility => "maximize_expected_utility".into(),
            Self::MaximizeThresholdProbability => "maximize_threshold_probability".into(),
            Self::MaximizeQuantile => "maximize_quantile".into(),
            Self::MaximizeCvar => "maximize_cvar".into(),
            Self::Declared(label) => format!("declared:{label}"),
        }
    }
}

/// The identity of an adaptive candidate search: which generator produced the
/// candidates, under what budget, how many it evaluated, and how the decision was chosen.
///
/// Without it, a ranking or optimality statement over the retained candidates would
/// silently condition on a search the claim does not describe.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateSearchRecord {
    /// Generator identity (algorithm, version and configuration label).
    pub generator: String,
    /// Most candidates the search was allowed to evaluate.
    pub budget: u64,
    /// Candidates the search actually evaluated, retained or not.
    pub n_evaluated: u64,
    /// The rule that chose the reported decision.
    pub selection_rule: SelectionRule,
    /// Stream the search itself drew from, when it was stochastic.
    pub generator_stream: Option<StreamId>,
}

impl CandidateSearchRecord {
    /// Typed length-delimited identity of the record.
    #[must_use]
    pub fn identity(&self) -> String {
        let mut writer = CanonicalWriter::new("decision_reliability.search_record.v1");
        writer
            .text(&self.generator)
            .word(self.budget)
            .word(self.n_evaluated)
            .text(&self.selection_rule.kind())
            .opt_text(self.generator_stream.as_ref().map(StreamId::digest_hex).as_deref());
        writer.finish_hex()
    }

    /// Whether the budget was fully used: the search stopped because it ran out, not
    /// because it converged, so the evaluated set is budget-limited.
    #[must_use]
    pub const fn budget_exhausted(&self) -> bool {
        self.n_evaluated >= self.budget
    }

    fn validate(&self, retained: usize) -> Result<(), ReliabilityError> {
        if is_blank(&self.generator) {
            return Err(ReliabilityError::SearchRecordInconsistent {
                message: "blank generator identity".into(),
            });
        }
        if self.n_evaluated < retained as u64 {
            return Err(ReliabilityError::SearchRecordInconsistent {
                message: format!(
                    "{} candidates retained but only {} evaluated",
                    retained, self.n_evaluated
                ),
            });
        }
        if self.budget == 0 || self.n_evaluated > self.budget {
            return Err(ReliabilityError::SearchRecordInconsistent {
                message: format!(
                    "{} evaluated against a budget of {}",
                    self.n_evaluated, self.budget
                ),
            });
        }
        Ok(())
    }
}

/// One candidate's per-draw utilities and the draws they were computed on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateDraws {
    /// Semantic candidate identity (never a position).
    pub id: String,
    /// Utility of this candidate at each draw.
    pub utilities: Vec<f64>,
    /// Stream segment the draws came from.
    pub segment: DrawSegment,
}

/// Utilities of a finite candidate set with stream identities and declared dependence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionDraws {
    /// Utility unit shared by every candidate.
    pub utility_unit: String,
    /// Producer label (`cohort_decision`, `sequential_evaluation`, or an external one).
    pub producer: String,
    /// Which law the draws realize.
    pub source: SourceParadigm,
    /// How the candidates' draws depend on each other.
    pub coupling: CandidateCoupling,
    /// Serial dependence within each stream.
    pub within_stream: WithinStreamDependence,
    /// Optional importance weights, one per draw index, shared by every candidate.
    pub weights: Option<Vec<f64>>,
    /// Earlier evaluations of these candidates on these same draws (each one looked at
    /// the draws). Declared by the producer; a stream already used this way is not a
    /// clean validation stream.
    pub repeated_evaluations: u64,
    /// How the candidate set came to exist.
    pub origin: CandidateOrigin,
    /// The search record, required when the set was adaptively generated.
    pub search_record: Option<CandidateSearchRecord>,
    /// The candidates.
    pub candidates: Vec<CandidateDraws>,
}

impl DecisionDraws {
    /// Number of candidates.
    #[must_use]
    pub fn n_candidates(&self) -> usize {
        self.candidates.len()
    }

    /// Number of draws per candidate (zero for an empty carrier).
    #[must_use]
    pub fn n_draws(&self) -> usize {
        self.candidates.first().map_or(0, |c| c.utilities.len())
    }

    /// Indices into [`Self::candidates`] in canonical (semantic id) order, so every
    /// result is invariant to the order the producer listed the candidates.
    #[must_use]
    pub fn canonical_order(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.candidates.len()).collect();
        order.sort_by(|&a, &b| self.candidates[a].id.cmp(&self.candidates[b].id));
        order
    }

    /// The candidate with identity `id`.
    #[must_use]
    pub fn candidate(&self, id: &str) -> Option<&CandidateDraws> {
        self.candidates.iter().find(|c| c.id == id)
    }

    /// Number of candidates the search considered: the search record's count when one
    /// exists, otherwise the retained set.
    #[must_use]
    pub fn n_considered(&self) -> u64 {
        let retained = self.candidates.len() as u64;
        self.search_record.as_ref().map_or(retained, |r| r.n_evaluated.max(retained))
    }

    /// Digest of the candidate set identity: ids in canonical order, origin and the
    /// search record. Does not cover the draws.
    #[must_use]
    pub fn candidate_set_digest(&self) -> String {
        let mut writer = CanonicalWriter::new("decision_reliability.candidate_set.v1");
        writer.text(self.origin.as_str());
        let record = self.search_record.as_ref().map(CandidateSearchRecord::identity);
        writer.opt_text(record.as_deref());
        let ids: Vec<&str> =
            self.canonical_order().iter().map(|&i| self.candidates[i].id.as_str()).collect();
        writer.texts(&ids);
        writer.finish_hex()
    }

    /// Digest of everything the carrier asserts: declarations, stream segments and
    /// every utility bit pattern, in canonical candidate order.
    #[must_use]
    pub fn digest_hex(&self) -> String {
        let mut writer = CanonicalWriter::new("decision_reliability.draws.v1");
        writer
            .text(&self.utility_unit)
            .text(&self.producer)
            .text(self.source.as_str())
            .text(self.coupling.as_str())
            .text(self.within_stream.as_str())
            .word(self.repeated_evaluations)
            .text(&self.candidate_set_digest());
        match &self.weights {
            Some(weights) => writer.word(1).reals(weights),
            None => writer.word(0),
        };
        for &i in &self.canonical_order() {
            let candidate = &self.candidates[i];
            writer
                .text(&candidate.id)
                .text(&candidate.segment.digest_hex())
                .reals(&candidate.utilities);
        }
        writer.finish_hex()
    }

    /// Check shape, finiteness and every dependence declaration.
    ///
    /// # Errors
    /// Structural defects, undeclared dependence, a coupling that the stream
    /// identities contradict, or an adaptive set without its search record.
    pub fn validate(&self) -> Result<(), ReliabilityError> {
        let invalid = ReliabilityError::invalid;
        if self.candidates.is_empty() || self.candidates.len() > MAX_CANDIDATES {
            return Err(invalid("candidate_count", "between 1 and 65536 candidates are required"));
        }
        if is_blank(&self.utility_unit) || is_blank(&self.producer) {
            return Err(invalid("blank_label", "utility unit and producer must be named"));
        }
        let n = self.n_draws();
        let cells = n.checked_mul(self.candidates.len());
        if n < MIN_DRAWS || cells.is_none_or(|c| c > MAX_CELLS) {
            return Err(invalid("draw_count", "at least 4 draws and at most 2^26 cells"));
        }
        let mut ids = std::collections::BTreeSet::new();
        for candidate in &self.candidates {
            if is_blank(&candidate.id) || !ids.insert(candidate.id.as_str()) {
                return Err(invalid("candidate_id", "candidate ids must be non-blank and unique"));
            }
            if candidate.utilities.len() != n || candidate.segment.len != n as u64 {
                return Err(invalid("shape", "every candidate needs one utility per draw"));
            }
            if candidate.utilities.iter().any(|u| !u.is_finite()) {
                return Err(invalid("non_finite_utility", "utilities must be finite"));
            }
        }
        self.validate_weights(n)?;
        self.validate_dependence()?;
        self.validate_search()
    }

    fn validate_weights(&self, n: usize) -> Result<(), ReliabilityError> {
        let Some(weights) = &self.weights else {
            return Ok(());
        };
        if weights.len() != n || weights.iter().any(|w| !w.is_finite() || *w <= 0.0) {
            return Err(ReliabilityError::invalid(
                "weights",
                "weights must be finite and positive, one per draw",
            ));
        }
        if self.coupling == CandidateCoupling::DeclaredIndependent && self.candidates.len() > 1 {
            return Err(ReliabilityError::RouteNotSupported {
                message: "importance weights on independently drawn candidates have no shared \
                          state to weight"
                    .into(),
            });
        }
        if self.within_stream == WithinStreamDependence::Autocorrelated {
            return Err(ReliabilityError::RouteNotSupported {
                message: "importance weights on an autocorrelated stream: the variance of a \
                          weighted dependent stream is not estimated"
                    .into(),
            });
        }
        Ok(())
    }

    fn validate_dependence(&self) -> Result<(), ReliabilityError> {
        if self.within_stream == WithinStreamDependence::Unknown {
            return Err(ReliabilityError::DependenceUnknown { what: "within_stream_serial" });
        }
        if self.candidates.len() < 2 {
            return Ok(());
        }
        match self.coupling {
            CandidateCoupling::Unknown => {
                Err(ReliabilityError::DependenceUnknown { what: "candidate_coupling" })
            }
            CandidateCoupling::SharedSimulation => {
                let first = &self.candidates[0].segment;
                if self.candidates.iter().all(|c| c.segment.same_draws(first)) {
                    Ok(())
                } else {
                    Err(ReliabilityError::JointLawRequired {
                        reason: "shared_simulation_declared_but_segments_differ",
                    })
                }
            }
            CandidateCoupling::DeclaredIndependent => {
                let mut by_stream: BTreeMap<&StreamId, Vec<(u64, u64)>> = BTreeMap::new();
                for candidate in &self.candidates {
                    by_stream
                        .entry(&candidate.segment.stream)
                        .or_default()
                        .push((candidate.segment.start, candidate.segment.end()));
                }
                for ranges in by_stream.values_mut() {
                    ranges.sort_unstable();
                    if ranges.windows(2).any(|w| w[0].1 > w[1].0) {
                        return Err(ReliabilityError::JointLawRequired {
                            reason: "declared_independent_but_draws_overlap",
                        });
                    }
                }
                Ok(())
            }
        }
    }

    fn validate_search(&self) -> Result<(), ReliabilityError> {
        match (&self.origin, &self.search_record) {
            (CandidateOrigin::AdaptivelyGenerated, None) => {
                Err(ReliabilityError::SearchRecordRequired { candidates: self.candidates.len() })
            }
            (_, Some(record)) => record.validate(self.candidates.len()),
            (CandidateOrigin::Enumerated, None) => Ok(()),
        }
    }
}
