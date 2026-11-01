//! Decision value of further choice studies: the entry through which hosts reach the EVSI ranking
//! of [`antecedent_design::choice_evsi`]. The posterior it values comes from the choice inference
//! ([`super::choice_inference::ChoiceInferenceSession::rank_studies`]).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_design::choice_evsi::{
    ChoiceEvsiError, EvsiMethod, MAX_EXACT_COMPOSITIONS, StudyCandidate, StudyRank,
    rank_hierarchical_studies, rank_studies, rank_studies_with_context,
};
