//! Public support-matrix lookup.
//!
//! Axes, n/a predicates, licensed cells, refusal-reason rules, and the legacy
//! allowlist source are generated from `parity/support_*.toml`. Runtime states
//! are three: licensed
//! (`parity/support_licensed.toml`), n/a (`parity/support_n_a.toml`, typed
//! impossibility, [`SupportRefusal::NotApplicable`]), or not licensed
//! ([`SupportRefusal::Refused`]). `parity/support_closed.toml` is the reason
//! table for refused cells, not a fourth state; its filename is retained for
//! compatibility. `allowed_unlicensed` is also retained as a wire value for
//! older artifacts and clients, but `parity/support_allowlist.toml` must have
//! zero active entries (a parity gate checks it). Any refused cell without a
//! reason uses the shared default-refusal message.
//!
//! Geometric [`classify`] keys on the five-tuple cell. [`classify_estimator`]
//! additionally requires the concrete [`EstimatorId`](crate::EstimatorId) to appear on that cell's
//! evidence list (`LicensedCell::estimators`), so matching / IV / RD / forest /
//! two-stage front-door cannot inherit a license whose calibration never ran
//! them.
//!
//! # Module boundary
//!
//! The tree is built so the matrix can later move to its own crate:
//!
//! - `cell`, `matrix`, `matrix_data` and `axis` depend only on lower-level
//!   facade types ([`crate::GraphClass`], [`crate::AcceptedGraph`],
//!   [`crate::CausalError`], [`crate::EstimatorId`]) and classify plain axis
//!   strings. None of them names `analysis::*`.
//! - `bridge` is the narrow interface used by `analysis`: it maps a
//!   [`CausalQuery`](antecedent_core::CausalQuery), an
//!   [`InferenceMode`](crate::InferenceMode) and a
//!   [`RefuteSuite`](crate::RefuteSuite) onto axis strings and holds the
//!   declared-off-axis allow-list. It is the only file here that depends on
//!   `analysis`, and it stays in the facade on extraction.
//!
//! What remains before it can be a crate of its own: `GraphClass`,
//! `AcceptedGraph` and `EstimatorId` must live below it (they are facade
//! types today), and `CausalError::Support` embeds [`SupportRefusal`], so the
//! refusal type has to move with the matrix while `CausalError` stays above it
//! and converts. The `parity/support_*.toml` generator and
//! `data/support_matrix.json` move with `matrix_data`.

mod axis;
mod bridge;
mod cell;
mod matrix;
mod matrix_data;

#[cfg(test)]
mod graphless_tests;
#[cfg(test)]
mod tests;

pub use antecedent_io::graphless_support::{
    GraphlessAssignmentSupport, GraphlessSupportKey, GraphlessSupportStatus, classify_graphless,
};

pub use axis::query_axis_name;
pub(crate) use axis::{effective_graph_class, matrix_graph_class};
pub use bridge::{
    is_declared_off_axis, refuse_undeclared_off_axis, support_cell, support_cell_named,
};
pub use cell::{
    CellStatus, IntoGraphInput, StructureSource, SupportCell, SupportGraphClass, SupportInference,
    SupportQuery, SupportRefusal, SupportStructure, SupportValidation,
};
pub(crate) use matrix::executes_unlicensed;
pub use matrix::{
    cell_coordinate, classify, classify_estimator, licensed_estimators, licensed_neighbors,
    licensed_route_estimator, licensed_support_cells, refuse_if_not_applicable, refused_message,
    support_cell_from_coordinate,
};

/// The execute-side graphless license handshake: return
/// [`CellStatus::Licensed`] when `key`/`observed` classify as licensed, and
/// `None` otherwise. Every checked-execution operation shares this so the
/// per-site scaffold is only the key derivation and support-field population.
#[must_use]
pub(crate) fn license_if_graphless(
    key: GraphlessSupportKey<'_>,
    observed: GraphlessAssignmentSupport,
) -> Option<CellStatus> {
    matches!(classify_graphless(key, observed), GraphlessSupportStatus::Licensed { .. })
        .then_some(CellStatus::Licensed)
}
