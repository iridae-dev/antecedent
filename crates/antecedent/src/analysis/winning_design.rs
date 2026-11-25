//! Facade for the read-only winning-design projection.
//!
//! not_scientific: copies the candidate id and sample size already stored at rank 0.
//! It does not rank, does not call `design.consume`, and creates no support or coverage unit.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_design::winning_design::{
    StoredDesignRow, WinningDesignDeclaration, WinningDesignError,
};

/// Copy the rank-0 candidate id and sample size from rows the ranking already stored.
pub fn project_stored_winner<'a>(
    rows: impl IntoIterator<Item = StoredDesignRow<'a>>,
) -> Result<WinningDesignDeclaration, WinningDesignError> {
    antecedent_design::winning_design::project_stored_winner(rows)
}
