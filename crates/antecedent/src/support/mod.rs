//! Public support-matrix lookup.
//!
//! Axes, n/a predicates, supported cells, and refusal-reason rules are generated
//! from `parity/support_*.toml`. Runtime states are three: supported
//! (`parity/support_cells.toml`), n/a (`parity/support_n_a.toml`, typed
//! impossibility, [`SupportRefusal::NotApplicable`]), or not supported
//! ([`SupportRefusal::NotSupported`]). `parity/support_refusal_reasons.toml` is the reason
//! table for refused cells, not a fourth state; its filename is retained for
//! compatibility. Any refused cell without a reason uses the shared
//! default-refusal message.
//!
//! Geometric [`classify`] keys on the five-tuple cell. [`classify_estimator`]
//! additionally requires the concrete [`EstimatorId`](crate::EstimatorId) to appear on that cell's
//! evidence list (`SupportedCell::estimators`), so matching / IV / RD / forest /
//! two-stage front-door cannot inherit support whose calibration never ran
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
mod gate;
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
pub(crate) use bridge::validation_axis;
pub use bridge::{
    is_declared_off_axis, refuse_undeclared_off_axis, support_cell, support_cell_named,
    support_cell_named_for_data,
};
pub use cell::{
    CellStatus, IntoGraphInput, StructureSource, SupportCell, SupportGraphClass, SupportInference,
    SupportQuery, SupportRefusal, SupportStructure, SupportValidation,
};
pub(crate) use matrix::retarget_permitted;
pub use matrix::{
    cell_coordinate, classify, classify_estimator, refuse_if_not_applicable,
    refuse_if_not_applicable_in, refused_message, support_cell_from_coordinate, supported_cells,
    supported_estimators, supported_neighbors, supported_route_estimator,
};

/// The execute-side graphless support handshake: return
/// [`CellStatus::Supported`] when `key`/`observed` classify as supported, and
/// `None` otherwise. Every checked-execution operation shares this so the
/// per-site scaffold is only the key derivation and support-field population.
#[must_use]
pub(crate) fn support_if_graphless(
    key: GraphlessSupportKey<'_>,
    observed: GraphlessAssignmentSupport,
) -> Option<CellStatus> {
    matches!(classify_graphless(key, observed), GraphlessSupportStatus::Supported { .. })
        .then_some(CellStatus::Supported)
}
