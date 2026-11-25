//! Project the winning ranked design from the candidate id and sample size the ranking stored.
//!
//! not_scientific: a copy of the rank-0 row. The ranking units `design.ranking_artifact` and
//! `design.evsi` remain the units that order candidates. This projection does not call
//! `design.consume` and creates no support or coverage unit.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

use crate::candidate::CandidateDesign;
use crate::choice_evsi::StudyRank;
use crate::design_ranking_artifact::DesignRankingArtifactWire;
use crate::ranking::RankingRecord;
use crate::result::ScoredDesignRanking;

/// One candidate id, rank and sample size already stored on a ranking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoredDesignRow<'a> {
    /// Candidate id the ranking stored.
    pub candidate_id: &'a str,
    /// Rank the ranking stored. Zero is the winner.
    pub rank: u64,
    /// Sample size the ranking stored for this candidate.
    pub n: u64,
}

/// The declaration copied from the ranking's winning row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WinningDesignDeclaration {
    /// Candidate id of the stored rank-0 row.
    pub candidate_id: String,
    /// Sample size of the stored rank-0 row.
    pub n: u64,
}

/// Why a stored ranking has no single winning design to copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WinningDesignError {
    /// The ranking stored no candidates.
    Empty,
    /// No stored row has rank 0.
    NoWinner,
    /// More than one stored row has rank 0.
    Ambiguous,
    /// The rank-0 row's candidate id is empty.
    BlankId,
    /// Probabilistic entries and structural entries are both present.
    Mixed,
}

impl fmt::Display for WinningDesignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Empty => "the ranking stored no candidates",
            Self::NoWinner => "the ranking stored no rank-0 candidate",
            Self::Ambiguous => "the ranking stored more than one rank-0 candidate",
            Self::BlankId => "the ranking stored a blank candidate id",
            Self::Mixed => "the ranking stored both probabilistic and structural entries",
        })
    }
}

impl std::error::Error for WinningDesignError {}

/// Copy the rank-0 candidate id and sample size. Rows are not reordered.
pub fn project_stored_winner<'a>(
    rows: impl IntoIterator<Item = StoredDesignRow<'a>>,
) -> Result<WinningDesignDeclaration, WinningDesignError> {
    let mut seen = false;
    let mut winner: Option<WinningDesignDeclaration> = None;
    for row in rows {
        seen = true;
        if row.rank != 0 {
            continue;
        }
        if winner.is_some() {
            return Err(WinningDesignError::Ambiguous);
        }
        if row.candidate_id.is_empty() {
            return Err(WinningDesignError::BlankId);
        }
        winner = Some(WinningDesignDeclaration {
            candidate_id: row.candidate_id.to_owned(),
            n: row.n,
        });
    }
    if !seen {
        return Err(WinningDesignError::Empty);
    }
    winner.ok_or(WinningDesignError::NoWinner)
}

/// Read the winner from a [`RankingRecord`]: `semantic_id` and `sample_size` on a probabilistic
/// row, or `semantic_id` and `cost.sample_budget` on a structural row. Rank 0 is the winner.
pub fn project_ranking_record(
    record: &RankingRecord,
) -> Result<WinningDesignDeclaration, WinningDesignError> {
    match (record.entries.is_empty(), record.structural.is_empty()) {
        (true, true) => Err(WinningDesignError::Empty),
        (false, false) => Err(WinningDesignError::Mixed),
        (false, true) => project_stored_winner(record.entries.iter().map(|row| StoredDesignRow {
            candidate_id: row.semantic_id.as_str(),
            rank: row.rank as u64,
            n: row.sample_size,
        })),
        (true, false) => project_stored_winner(record.structural.iter().map(|row| StoredDesignRow {
            candidate_id: row.semantic_id.as_str(),
            rank: row.rank as u64,
            n: row.cost.sample_budget,
        })),
    }
}

/// Read the winner from the sealed wire's stored `semantic_id`, `sample_size` and `rank`.
///
/// The signal request's candidate id is a different field and is not read.
pub fn project_ranking_wire(
    wire: &DesignRankingArtifactWire,
) -> Result<WinningDesignDeclaration, WinningDesignError> {
    project_stored_winner(wire.candidates.iter().map(|row| StoredDesignRow {
        candidate_id: row.semantic_id.as_str(),
        rank: row.rank,
        n: row.sample_size,
    }))
}

/// Read the winner a choice-study ranking stored first.
///
/// [`crate::choice_evsi`] stores the ordered ranking with the winner at index 0. This copies
/// that candidate's label and `observations` and does not sort by net value.
pub fn project_choice_study_ranking(
    ranking: &[StudyRank],
) -> Result<WinningDesignDeclaration, WinningDesignError> {
    let Some(winner) = ranking.first() else {
        return Err(WinningDesignError::Empty);
    };
    if winner.candidate.label.is_empty() {
        return Err(WinningDesignError::BlankId);
    }
    Ok(WinningDesignDeclaration {
        candidate_id: winner.candidate.label.clone(),
        n: u64::from(winner.candidate.observations),
    })
}

/// Sample size stored on a scored candidate design.
///
/// A sampling plan stores `additional_samples`, an environment plan stores `additional_rows`,
/// and a measurement or intervention stores `sample_budget`.
#[must_use]
pub fn stored_sample_size(design: &CandidateDesign) -> u64 {
    match design {
        CandidateDesign::IncreaseSamplingRate(plan) => plan.additional_samples,
        CandidateDesign::ObserveEnvironment(plan) => plan.additional_rows,
        CandidateDesign::Measure(plan) => plan.cost.sample_budget,
        CandidateDesign::Intervene(plan) => plan.cost.sample_budget,
    }
}

/// Read the winner from a [`ScoredDesignRanking`].
///
/// The ranker stores the candidate index, not a semantic id, so the declaration's id is that
/// index in decimal. The sample size is [`stored_sample_size`] of the rank-0 design. The slice
/// is not sorted by score.
pub fn project_scored_design(
    ranking: &ScoredDesignRanking,
) -> Result<WinningDesignDeclaration, WinningDesignError> {
    if ranking.ranked.is_empty() {
        return Err(WinningDesignError::Empty);
    }
    let mut winner = None;
    for row in ranking.ranked.iter() {
        if row.rank != 0 {
            continue;
        }
        if winner.is_some() {
            return Err(WinningDesignError::Ambiguous);
        }
        winner = Some(row);
    }
    let Some(winner) = winner else {
        return Err(WinningDesignError::NoWinner);
    };
    Ok(WinningDesignDeclaration {
        candidate_id: winner.candidate_index.to_string(),
        n: stored_sample_size(&winner.candidate),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{EnvironmentId, MonteCarloBudget, MonteCarloError, VariableId};

    use super::{
        project_choice_study_ranking, project_ranking_record, project_scored_design,
        project_stored_winner, stored_sample_size, StoredDesignRow, WinningDesignDeclaration,
        WinningDesignError,
    };
    use crate::candidate::{
        CandidateDesign, DesignCost, EnvironmentPlan, ExperimentPlan, MeasurementPlan, SamplingPlan,
    };
    use crate::choice_evsi::{EvsiMethod, StudyCandidate, StudyRank};
    use crate::evsi::{IntegrationMethod, SearchReceipt};
    use crate::ranking::{DesignRankingBasis, EvsiRankRow, RankingRecord, StructuralEntry};
    use crate::result::{RankedCandidate, ScoreEvaluation, ScoredDesignRanking};
    use crate::signal::{SignalTrustLabel, SignalUpdateMode};
    use crate::study_planner::StudyCost;

    const KEPT: &str = "kept";
    const BIGGER: &str = "bigger";
    const SMALLER: &str = "smaller";
    const KEPT_N: u64 = 40;
    const BIGGER_N: u64 = 500;
    const SMALLER_N: u64 = 10;

    fn declaration() -> WinningDesignDeclaration {
        WinningDesignDeclaration { candidate_id: KEPT.to_owned(), n: KEPT_N }
    }

    fn fixture() -> [StoredDesignRow<'static>; 3] {
        [
            StoredDesignRow { candidate_id: BIGGER, rank: 1, n: BIGGER_N },
            StoredDesignRow { candidate_id: KEPT, rank: 0, n: KEPT_N },
            StoredDesignRow { candidate_id: SMALLER, rank: 2, n: SMALLER_N },
        ]
    }

    #[test]
    fn rank_zero_sample_size_is_copied_when_neighbours_are_larger_and_smaller() {
        let copied = project_stored_winner(fixture()).expect("rank 0 is stored");
        assert_eq!(copied, declaration());
    }

    #[test]
    fn refusals_name_the_stored_ranking() {
        let none: [StoredDesignRow<'_>; 0] = [];
        assert_eq!(project_stored_winner(none).unwrap_err(), WinningDesignError::Empty);
        assert_eq!(
            project_stored_winner([StoredDesignRow { candidate_id: BIGGER, rank: 1, n: BIGGER_N }])
                .unwrap_err(),
            WinningDesignError::NoWinner
        );
        assert_eq!(
            project_stored_winner([
                StoredDesignRow { candidate_id: KEPT, rank: 0, n: KEPT_N },
                StoredDesignRow { candidate_id: BIGGER, rank: 0, n: BIGGER_N },
            ])
            .unwrap_err(),
            WinningDesignError::Ambiguous
        );
        assert_eq!(
            project_stored_winner([StoredDesignRow { candidate_id: "", rank: 0, n: KEPT_N }])
                .unwrap_err(),
            WinningDesignError::BlankId
        );
        assert_eq!(
            WinningDesignError::Empty.to_string(),
            "the ranking stored no candidates"
        );
    }

    fn search() -> SearchReceipt {
        SearchReceipt { supplied: 3, evaluated: 3, truncated: false, unevaluated_ids: Vec::new() }
    }

    fn evsi_row(semantic_id: &str, rank: usize, sample_size: u64) -> EvsiRankRow {
        EvsiRankRow {
            semantic_id: semantic_id.to_owned(),
            rank,
            evsi: if rank == 0 { 0.1 } else { 9.0 },
            mc_stderr: 0.0,
            replicates: 0,
            integration: IntegrationMethod::Exact,
            evpi: 0.0,
            sample_size,
            study_cost_amount: 0.0,
            study_cost_unit: "utility".to_owned(),
            net_value: None,
            rank_uncertain: false,
            signal_identity: "signal".to_owned(),
            request_fingerprint: "request".to_owned(),
            update_mode: SignalUpdateMode::NativeUpdate,
            provider_trust: SignalTrustLabel::NativeSupported,
        }
    }

    fn record(entries: Vec<EvsiRankRow>, structural: Vec<StructuralEntry>) -> RankingRecord {
        RankingRecord {
            basis: if structural.is_empty() {
                DesignRankingBasis::Evsi
            } else {
                DesignRankingBasis::StructuralSufficiencyCost
            },
            decision_contract_identity: None,
            utility_unit: None,
            cost_mapping: None,
            rng_seed: None,
            source_digests: Vec::new(),
            entries,
            structural,
            search: search(),
        }
    }

    #[test]
    fn ranking_record_reads_stored_sample_size_not_a_new_order() {
        let record = record(
            vec![
                evsi_row(BIGGER, 1, BIGGER_N),
                evsi_row(KEPT, 0, KEPT_N),
                evsi_row(SMALLER, 2, SMALLER_N),
            ],
            Vec::new(),
        );
        assert_eq!(project_ranking_record(&record).expect("winner"), declaration());
    }

    #[test]
    fn structural_record_reads_sample_budget_of_rank_zero() {
        let structural = vec![
            StructuralEntry {
                semantic_id: SMALLER.to_owned(),
                rank: 2,
                verified_sufficient: true,
                cost: StudyCost { units: 1, sample_budget: SMALLER_N },
            },
            StructuralEntry {
                semantic_id: KEPT.to_owned(),
                rank: 0,
                verified_sufficient: true,
                cost: StudyCost { units: 9, sample_budget: KEPT_N },
            },
            StructuralEntry {
                semantic_id: BIGGER.to_owned(),
                rank: 1,
                verified_sufficient: true,
                cost: StudyCost { units: 1, sample_budget: BIGGER_N },
            },
        ];
        let record = record(Vec::new(), structural);
        assert_eq!(project_ranking_record(&record).expect("winner"), declaration());
    }

    #[test]
    fn mixed_record_is_refused() {
        let record = record(
            vec![evsi_row(KEPT, 0, KEPT_N)],
            vec![StructuralEntry {
                semantic_id: BIGGER.to_owned(),
                rank: 0,
                verified_sufficient: true,
                cost: StudyCost { units: 1, sample_budget: BIGGER_N },
            }],
        );
        assert_eq!(project_ranking_record(&record).unwrap_err(), WinningDesignError::Mixed);
    }

    fn study(label: &str, observations: u32, net_value: f64) -> StudyRank {
        StudyRank {
            candidate: StudyCandidate { label: label.to_owned(), observations, cost: 0.0 },
            evsi: net_value,
            standard_error: 0.0,
            method: EvsiMethod::Exact { compositions: 1 },
            net_value,
            rank_uncertain: false,
            calibration: "not_measured",
        }
    }

    #[test]
    fn choice_study_reads_the_stored_first_candidate() {
        let ranking = [
            study(KEPT, u32::try_from(KEPT_N).expect("n"), 0.1),
            study(BIGGER, u32::try_from(BIGGER_N).expect("n"), 9.0),
        ];
        assert_eq!(project_choice_study_ranking(&ranking).expect("winner"), declaration());
    }

    fn sampling(n: u64) -> CandidateDesign {
        CandidateDesign::IncreaseSamplingRate(SamplingPlan {
            additional_samples: n,
            cost: DesignCost { amount: 1.0, sample_budget: 1 },
            tag: 0,
        })
    }

    fn ranked(
        index: usize,
        rank: usize,
        score: f64,
        candidate: CandidateDesign,
    ) -> RankedCandidate {
        RankedCandidate {
            candidate_index: index,
            candidate,
            score,
            monte_carlo: MonteCarloError::default(),
            rank,
            rank_uncertain: false,
            implemented_functional: Arc::<str>::from("stored"),
            evaluation: ScoreEvaluation::Exact,
        }
    }

    #[test]
    fn scored_design_reads_rank_zero_index_and_stored_n() {
        let ranking = ScoredDesignRanking {
            ranked: Arc::from([
                ranked(3, 1, 9.0, sampling(BIGGER_N)),
                ranked(7, 0, 0.1, sampling(KEPT_N)),
            ]),
            violations: Arc::from([]),
            budget: MonteCarloBudget::default(),
            early_stopped: false,
        };
        assert_eq!(
            project_scored_design(&ranking).expect("winner"),
            WinningDesignDeclaration { candidate_id: "7".to_owned(), n: KEPT_N }
        );
    }

    #[test]
    fn stored_sample_size_follows_the_plan() {
        assert_eq!(stored_sample_size(&sampling(KEPT_N)), KEPT_N);
        assert_eq!(
            stored_sample_size(&CandidateDesign::ObserveEnvironment(EnvironmentPlan {
                environment: EnvironmentId::from_raw(1),
                additional_rows: KEPT_N,
                cost: DesignCost { amount: 1.0, sample_budget: SMALLER_N },
                tag: 0,
            })),
            KEPT_N
        );
        assert_eq!(
            stored_sample_size(&CandidateDesign::Measure(MeasurementPlan {
                variables: Arc::from([VariableId::from_raw(0)]),
                cost: DesignCost { amount: 3.0, sample_budget: KEPT_N },
                tag: 0,
            })),
            KEPT_N
        );
        assert_eq!(
            stored_sample_size(&CandidateDesign::Intervene(ExperimentPlan {
                targets: Arc::from([VariableId::from_raw(0)]),
                cost: DesignCost { amount: 3.0, sample_budget: KEPT_N },
                tag: 0,
            })),
            KEPT_N
        );
    }
}
