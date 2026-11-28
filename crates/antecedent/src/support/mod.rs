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
mod calibration;
mod cell;
mod gate;
mod matrix;
mod matrix_data;
mod trace;
#[allow(dead_code, reason = "each compiled route is bound to its unit as it adopts the decision")]
pub(crate) mod unit;
pub(crate) mod units_generated;

pub(crate) use calibration::capability_calibration;
pub use calibration::{NOT_MEASURED, unit_calibration};
pub(crate) use trace::executed as trace_executed;
pub(crate) use trace::interval_reported as trace_interval;
#[cfg(feature = "evidence-trace")]
pub(crate) use trace::output_produced as trace_output;
pub(crate) use trace::route_ran as trace_route;
pub use trace::{TRACE_ENV, binary_name, caller_thread, declare_unit};
pub use unit::{UnitId, UnitRef};

#[cfg(test)]
mod graphless_tests;
#[cfg(test)]
mod tests;

pub use antecedent_io::graphless_support::{
    GraphlessAssignmentSupport, GraphlessSupportKey, GraphlessSupportStatus, classify_graphless,
    graphless_missing_evidence,
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
pub(crate) use gate::capability_supported;
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
    let supported =
        matches!(classify_graphless(key, observed), GraphlessSupportStatus::Supported { .. });
    trace::executed(
        &graphless_unit(key),
        if supported { &CellStatus::Supported } else { &CellStatus::NotSupported },
    );
    supported.then_some(CellStatus::Supported)
}

/// [`support_if_graphless`] answering with the supported unit, for a result to stamp.
#[must_use]
pub(crate) fn graphless_supported_unit(
    key: GraphlessSupportKey<'_>,
    observed: GraphlessAssignmentSupport,
) -> Option<UnitRef> {
    support_if_graphless(key, observed).map(|_| graphless_unit(key))
}

/// [`antecedent_io::graphless_support::graphless_admits_resampled`], recorded in the evidence
/// trace as a decision about the row `key` names.
#[must_use]
pub(crate) fn admits_resampled(
    key: GraphlessSupportKey<'_>,
    observed: GraphlessAssignmentSupport,
    resampled: antecedent_io::graphless_support::GraphlessResampleSupport,
) -> bool {
    let admitted =
        antecedent_io::graphless_support::graphless_admits_resampled(key, observed, resampled);
    trace::executed(
        &graphless_unit(key),
        if admitted { &CellStatus::Supported } else { &CellStatus::NotSupported },
    );
    admitted
}

/// [`admits_resampled`] answering with the admitting row's unit, for a result to stamp.
#[must_use]
pub(crate) fn admitted_resampled_unit(
    key: GraphlessSupportKey<'_>,
    observed: GraphlessAssignmentSupport,
    resampled: antecedent_io::graphless_support::GraphlessResampleSupport,
) -> Option<UnitRef> {
    admits_resampled(key, observed, resampled).then(|| graphless_unit(key))
}

/// The graphless row a rollout section was decided under, recorded in the evidence trace as a
/// decision about that row; the row's unit when it admits the section.
#[must_use]
pub(crate) fn rollout_row_unit(
    decision: Option<antecedent_io::analysis_result_artifact::rollout_support::RolloutRowDecision>,
) -> Option<UnitRef> {
    let decision = decision?;
    let unit = graphless_unit(GraphlessSupportKey {
        family: decision.family,
        design: decision.design,
        method: &decision.method,
        inference_claim: decision.inference_claim,
    });
    trace::executed(
        &unit,
        if decision.admitted { &CellStatus::Supported } else { &CellStatus::NotSupported },
    );
    decision.admitted.then_some(unit)
}

/// The unit a graphless support key names.
#[must_use]
pub(crate) fn graphless_unit(key: GraphlessSupportKey<'_>) -> UnitRef {
    UnitRef::Graphless {
        family: key.family.to_owned(),
        design: key.design.to_owned(),
        method: key.method.to_owned(),
        inference_claim: key.inference_claim.to_owned(),
    }
}
