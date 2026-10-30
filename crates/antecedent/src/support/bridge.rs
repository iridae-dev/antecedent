//! The adapter between the matrix and the analysis layer.
//!
//! This is the only file of the `support` tree that names `analysis::*`
//! ([`RefuteSuite`], the graphless-design predicate) or the inference mode.
//! Everything else in the tree classifies plain axis strings, so extracting the
//! matrix into its own crate moves every sibling file and leaves this one in
//! the facade.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CausalQuery, CausalQueryKind};

use crate::accepted::GraphClass;
use crate::analysis::RefuteSuite;
use crate::error::CausalError;
use crate::inference::InferenceMode;

use super::axis::{naming_class, query_axis};
use super::cell::{
    StructureSource, SupportCell, SupportGraphClass, SupportInference, SupportQuery,
    SupportRefusal, SupportValidation,
};

/// Query kinds that intentionally sit off the matrix axes: the mechanism- and unit-change
/// studies, which have no geometric cell to license and run only against a supplied static DAG;
/// structure discovery, attribution and simulation on a fitted model, which execute through the
/// compiler and stay unlicensed until a coverage record licenses them; and the graphless designs,
/// which are licensed by their own design key rather than a graph class.
///
/// This is the closed allow-list for "no cell": every other query kind must classify onto the
/// axis (for some graph class), so a variant added later without a matrix row is refused at build
/// rather than silently defaulting to allowed.
#[must_use]
pub fn is_declared_off_axis(query: &CausalQuery) -> bool {
    matches!(
        query.kind(),
        CausalQueryKind::MechanismChange
            | CausalQueryKind::UnitChange
            | CausalQueryKind::Discover
            | CausalQueryKind::Attribute
            | CausalQueryKind::Simulate
    ) || crate::analysis::query_license::is_graphless_design(query)
}

/// Refuse a query the matrix does not know and that is not a declared off-axis kind.
///
/// "Default = refused" holds on the axis because [`classify`] falls through to
/// [`CellStatus::Refused`]; this closes the same default for a query kind with no axis name at
/// all. A `Response` whose temporal/static attachment disagrees with the graph class has an axis
/// name for some class and is left to the compiler's typed refusal.
///
/// # Errors
///
/// [`CausalError::Support`] ([`SupportRefusal::Refused`]) for an unlisted off-axis kind.
pub fn refuse_undeclared_off_axis(query: &CausalQuery) -> Result<(), CausalError> {
    if is_declared_off_axis(query)
        || query_axis(query, GraphClass::Dag).is_some()
        || query_axis(query, GraphClass::TemporalDag).is_some()
    {
        return Ok(());
    }
    Err(CausalError::Support { id: SupportRefusal::Refused, message: UNDECLARED_OFF_AXIS })
}

const UNDECLARED_OFF_AXIS: &str = "this query kind has no support-matrix cell and is not a \
     declared off-axis kind; it is refused until a cell licenses it.";

fn inference_axis(mode: &InferenceMode) -> SupportInference {
    match mode {
        InferenceMode::Frequentist => SupportInference::Frequentist,
        InferenceMode::Bayesian(_) => SupportInference::Bayesian,
    }
}

pub(crate) fn validation_axis(suite: RefuteSuite) -> SupportValidation {
    match suite {
        RefuteSuite::None => SupportValidation::None,
        RefuteSuite::Cheap => SupportValidation::Cheap,
        RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => SupportValidation::Full,
    }
}

/// Build a [`SupportCell`] when `query` is on the public axis.
#[must_use]
pub fn support_cell(
    query: &CausalQuery,
    graph_class: GraphClass,
    structure: StructureSource,
    inference: &InferenceMode,
    refute: RefuteSuite,
) -> Option<SupportCell> {
    support_cell_named(query, graph_class.as_str(), structure, inference, refute)
}

/// Like [`support_cell`], but `graph_class` may be a classification-only axis
/// (`CoDetermined`, `Unknown`) that is not a [`GraphClass`] variant.
#[must_use]
pub fn support_cell_named(
    query: &CausalQuery,
    graph_class: &'static str,
    structure: StructureSource,
    inference: &InferenceMode,
    refute: RefuteSuite,
) -> Option<SupportCell> {
    support_cell_named_for_data(query, graph_class, structure, inference, refute, false)
}

/// Like [`support_cell_named`], told whether the study reads a time series.
///
/// A mediation on a `TemporalAdmg` is two different queries: on tabular data it reads the
/// contemporaneous projection of the graph (`MediationEffect`), on a series it reads the
/// unrolled window of the stationary template (`TemporalMediationEffect`). The class alone
/// cannot tell them apart, so the data modality decides.
#[must_use]
pub fn support_cell_named_for_data(
    query: &CausalQuery,
    graph_class: &'static str,
    structure: StructureSource,
    inference: &InferenceMode,
    refute: RefuteSuite,
    series: bool,
) -> Option<SupportCell> {
    let mut cell = support_cell_unadjusted(query, graph_class, structure, inference, refute)?;
    if series
        && matches!(query, CausalQuery::Mediation(_))
        && cell.query == SupportQuery::MediationEffect
        && graph_class == GraphClass::TemporalAdmg.as_str()
    {
        cell.query = SupportQuery::TemporalMediationEffect;
    }
    Some(cell)
}

fn support_cell_unadjusted(
    query: &CausalQuery,
    graph_class: &'static str,
    structure: StructureSource,
    inference: &InferenceMode,
    refute: RefuteSuite,
) -> Option<SupportCell> {
    Some(SupportCell {
        query: query_axis(query, naming_class(graph_class)?)?,
        graph_class: SupportGraphClass::parse(graph_class)?,
        structure: structure.into(),
        inference: inference_axis(inference),
        validation: validation_axis(refute),
    })
}
