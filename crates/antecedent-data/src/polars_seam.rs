//! Prototype Rust Polars seam behind the internal source and design contracts.
//!
//! The optional `polars` feature is off by default. Arrow stays the default
//! feature. This repository does not pin a Polars crate version, so this module
//! does not depend on Polars and does not resolve one. Nothing here opens a
//! Parquet file or builds a Polars plan.
//!
//! The three jobs are a wide-Parquet adjusted effect, panel and
//! multi-environment temporal preparation that borrows lag windows, and a
//! compatible multi-query plan for the existing prepared and batch API. Each
//! job adapts into library-owned [`ColumnView`] values and, at the numeric
//! barrier, a [`DesignLayout`]. Polars is not a backend of the internal
//! contract (ADR 0026).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::sync::Arc;

use antecedent_core::VariableId;

use crate::column::ColumnView;
use crate::design_layout::{DesignLayout, LayoutError};
use crate::error::DataError;
use crate::global_index::GlobalId;
use crate::missing::complete_rows;
use crate::panel_window::{BorrowError, CalendarSpan, PanelWindow, PopulationUnit, UnitPartition};
use crate::source::{OwnedSource, SourceModality};
use crate::storage::OwnedColumnarStorage;
use crate::table::TableView;

/// The `polars` feature is not part of the default feature set.
///
/// Enabling the feature compiles this module. It does not select a crate.
pub const POLARS_FEATURE_ON_BY_DEFAULT: bool = false;

/// No `polars` crate version in this repository was linked.
pub const POLARS_CRATE_RESOLVED: bool = false;

/// Where an adjusted-effect interval is computed.
pub const ADJUSTED_EFFECT_KERNEL_BARRIER: &str =
    "existing adjusted-effect estimator, after column views and the design layout";

/// Multi-query execution stays on this API. This seam does not call it.
pub const EXISTING_PREPARED_BATCH_API: &str =
    "antecedent::analysis::BatchStudy::prepare and antecedent::analysis::PreparedStudy";

/// Refusal for a backend that keeps null distinct from an observed `NaN`.
pub const NULL_DISTINCT_REFUSAL: &str =
    "null distinct from NaN is not a column view; compile invalid-iff-NaN before the design";

/// Why this seam does not call Polars.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolarsResolution {
    /// This repository does not name a Polars crate version.
    Unresolved,
}

impl PolarsResolution {
    /// The resolution available in this repository.
    #[must_use]
    pub const fn current() -> Self {
        Self::Unresolved
    }
}

/// How incoming nulls meet the `f64` column rule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IncomingNullPolicy {
    /// Null becomes the invalid-iff-`NaN` rule before a kernel sees the column.
    CompileToInvalidIffNan,
    /// Null and an observed `NaN` would stay distinct. The seam refuses this.
    NullDistinctFromNan,
}

/// Who owns the interval for an adjusted effect.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UncertaintyPlace {
    /// This seam does not estimate. The existing estimator owns the interval.
    ExistingEstimator,
}

/// A source, design, or lag refusal from one of the three jobs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SeamError {
    /// The source, the mask, or the null policy refused the job.
    Data(DataError),
    /// A selected column could not be placed in a [`DesignLayout`].
    Layout(LayoutError),
}

impl fmt::Display for SeamError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Data(error) => write!(formatter, "{error}"),
            Self::Layout(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for SeamError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Data(error) => Some(error),
            Self::Layout(error) => Some(error),
        }
    }
}

impl From<DataError> for SeamError {
    fn from(error: DataError) -> Self {
        Self::Data(error)
    }
}

impl From<LayoutError> for SeamError {
    fn from(error: LayoutError) -> Self {
        Self::Layout(error)
    }
}

/// Columns, mask, and null policy for one wide adjusted effect.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdjustedEffectRequest {
    treatment: Arc<str>,
    outcome: Arc<str>,
    adjustment: Vec<Arc<str>>,
    mask: Vec<bool>,
    nulls: IncomingNullPolicy,
}

impl AdjustedEffectRequest {
    /// Request `treatment`, `outcome`, and at least one adjustment column.
    ///
    /// `mask[i]` is the caller's inclusion bit for storage row `i`. Length is
    /// checked against the source when the job is prepared. The mask is
    /// intersected with the analysis mask and with validity of every required
    /// column.
    ///
    /// # Errors
    ///
    /// An empty name, a treatment that is the outcome, an empty adjustment
    /// list, or a repeated adjustment name.
    #[must_use = "an invalid adjusted-effect request is not a column selection"]
    pub fn try_new(
        treatment: &str,
        outcome: &str,
        adjustment: &[&str],
        mask: &[bool],
        nulls: IncomingNullPolicy,
    ) -> Result<Self, SeamError> {
        let treatment = require_name(treatment, "treatment")?;
        let outcome = require_name(outcome, "outcome")?;
        if treatment.as_ref() == outcome.as_ref() {
            return Err(invalid("treatment and outcome are the same column").into());
        }
        if adjustment.is_empty() {
            return Err(invalid("an adjusted effect needs an adjustment column").into());
        }
        let mut stored = Vec::with_capacity(adjustment.len());
        for name in adjustment {
            stored.push(require_name(name, "adjustment")?);
        }
        if duplicate_name(&stored) {
            return Err(invalid("adjustment lists the same column twice").into());
        }
        Ok(Self { treatment, outcome, adjustment: stored, mask: mask.to_vec(), nulls })
    }

    /// Caller's row inclusion bits.
    #[must_use]
    pub fn mask(&self) -> &[bool] {
        &self.mask
    }

    /// Null policy the job compiles before the design.
    #[must_use]
    pub const fn nulls(&self) -> IncomingNullPolicy {
        self.nulls
    }
}

/// Wide-Parquet adjusted effect, adapted into column views and one design.
///
/// `path` is a label. This job does not open it. The treatment and outcome
/// views are the source columns. The design is the adjustment columns at the
/// rows that survive the mask, the analysis mask, and validity. Its ownership
/// receipt is the copy cost. The parent digest is the full storage digest,
/// which already binds columns, the analysis mask, and weights. The required
/// columns are a projection and do not reuse that digest.
#[derive(Clone, Debug)]
#[must_use = "the effect plan is the kernel barrier for the existing estimator"]
pub struct WideParquetAdjustedEffect<'a> {
    path: Arc<str>,
    treatment: ColumnView<'a>,
    outcome: ColumnView<'a>,
    adjustment: Vec<ColumnView<'a>>,
    design: DesignLayout<'a>,
    parent_digest: [u8; 32],
    projected: bool,
}

impl<'a> WideParquetAdjustedEffect<'a> {
    /// Adapt `storage` into the adjusted-effect barrier.
    ///
    /// The source snapshot is tabular. Its validity rule is invalid-iff-`NaN`.
    /// [`IncomingNullPolicy::NullDistinctFromNan`] is refused before a design
    /// buffer exists.
    ///
    /// # Errors
    ///
    /// An empty path, a distinct-null policy, an unknown column, a mask whose
    /// length is not the row count, an empty selected sample, or a non-numeric
    /// adjustment column.
    #[must_use = "a refused wide-parquet effect is not a design"]
    pub fn prepare(
        path: &str,
        storage: &'a OwnedColumnarStorage,
        request: &AdjustedEffectRequest,
    ) -> Result<Self, SeamError> {
        compiled_nulls(request.nulls)?;
        let path = parquet_path(path)?;
        let treatment_id = variable_id(storage, &request.treatment)?;
        let outcome_id = variable_id(storage, &request.outcome)?;
        let mut adjustment_ids = Vec::with_capacity(request.adjustment.len());
        for name in &request.adjustment {
            adjustment_ids.push(variable_id(storage, name)?);
        }
        let mut ids = Vec::new();
        extend_unique(&mut ids, treatment_id);
        extend_unique(&mut ids, outcome_id);
        for id in adjustment_ids.iter().copied() {
            extend_unique(&mut ids, id);
        }
        let (parent_digest, projected) = source_projection(storage, &ids)?;
        require_projection(projected)?;
        let rows = kept_rows(storage, &ids, &request.mask)?;
        let adjustment = views(storage, &adjustment_ids)?;
        let design = DesignLayout::from_views(&adjustment, &rows)?;
        Ok(Self {
            path,
            treatment: storage.column(treatment_id)?,
            outcome: storage.column(outcome_id)?,
            adjustment,
            design,
            parent_digest,
            projected,
        })
    }

    /// File label. The job did not open it.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Crate resolution. This repository leaves Polars unresolved.
    #[must_use]
    pub const fn resolution(&self) -> PolarsResolution {
        PolarsResolution::current()
    }

    /// Source treatment column, including rows the design drops.
    #[must_use]
    pub const fn treatment(&self) -> ColumnView<'a> {
        self.treatment
    }

    /// Source outcome column, including rows the design drops.
    #[must_use]
    pub const fn outcome(&self) -> ColumnView<'a> {
        self.outcome
    }

    /// Adjustment columns, in request order.
    #[must_use]
    pub fn adjustment(&self) -> &[ColumnView<'a>] {
        &self.adjustment
    }

    /// Adjustment columns at the selected rows. The receipt is the copy cost.
    #[must_use]
    pub const fn design(&self) -> &DesignLayout<'a> {
        &self.design
    }

    /// Full-source digest. A projection does not replace it.
    #[must_use]
    pub const fn parent_digest(&self) -> [u8; 32] {
        self.parent_digest
    }

    /// The required columns are a projection, not the full-source identity.
    #[must_use]
    pub const fn is_projected(&self) -> bool {
        self.projected
    }

    /// The interval stays with the existing estimator.
    #[must_use]
    pub const fn uncertainty(&self) -> UncertaintyPlace {
        UncertaintyPlace::ExistingEstimator
    }

    /// Named barrier where this job stops.
    #[must_use]
    pub const fn kernel_barrier(&self) -> &'static str {
        ADJUSTED_EFFECT_KERNEL_BARRIER
    }
}

/// One lag read. A borrowed index names a row of the caller's column.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LagRead {
    /// Local row of the same unit, inside the split and the span.
    Borrowed(usize),
    /// The lag would leave the unit, the split, or the calendar span.
    Refused(BorrowError),
}

/// Target row and the lag read for that row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LagStep {
    target: usize,
    read: LagRead,
}

impl LagStep {
    /// Local row the lag was read from.
    #[must_use]
    pub const fn target(&self) -> usize {
        self.target
    }

    /// Borrowed source row, or the boundary the lag would cross.
    #[must_use]
    pub const fn read(&self) -> &LagRead {
        &self.read
    }
}

/// Panel or multi-environment lag preparation over a borrowed column.
///
/// [`PanelWindow::borrow_lag`] supplies every index. The lagged series is not
/// an owned shift. A chunk cut is not a new boundary.
#[derive(Clone, Debug)]
#[must_use = "the lag plan borrows the unit window instead of shifting a copy"]
pub struct BorrowedLagPreparation<'a> {
    population: PopulationUnit,
    window: PanelWindow,
    column: &'a [f64],
    lag: u32,
    steps: Vec<LagStep>,
}

impl<'a> BorrowedLagPreparation<'a> {
    /// Borrow `lag` for every row of `split` inside `span`.
    ///
    /// `column` is the unit's column in local row order. Its length is the
    /// unit length, not the window length. `population` is the partition's
    /// unit, either a panel unit or one environment.
    ///
    /// # Errors
    ///
    /// An unknown split, a span that meets no row of that split, or a column
    /// whose length is not the unit length.
    #[must_use = "a refused lag preparation is not a borrowed window"]
    pub fn prepare(
        partition: &UnitPartition,
        split: &str,
        span: CalendarSpan,
        column: &'a [f64],
        lag: u32,
    ) -> Result<Self, SeamError> {
        if column.len() != partition.row_count() {
            return Err(DataError::LengthMismatch {
                expected: partition.row_count(),
                actual: column.len(),
                context: "polars seam lag column",
            }
            .into());
        }
        let window = partition.window(split, span)?;
        let mut steps = Vec::new();
        for target in window.local_rows() {
            let read = match window.borrow_lag(target, lag) {
                Ok(source) => LagRead::Borrowed(source),
                Err(error) => LagRead::Refused(error),
            };
            steps.push(LagStep { target, read });
        }
        Ok(Self { population: partition.unit(), window, column, lag, steps })
    }

    /// Panel unit or environment the window belongs to.
    #[must_use]
    pub const fn population(&self) -> PopulationUnit {
        self.population
    }

    /// Window whose `borrow_lag` produced [`Self::steps`].
    #[must_use]
    pub const fn window(&self) -> &PanelWindow {
        &self.window
    }

    /// Caller's column. Lagged values are read from this slice.
    #[must_use]
    pub const fn column(&self) -> &'a [f64] {
        self.column
    }

    /// Lag, in earlier local rows of the same unit.
    #[must_use]
    pub const fn lag(&self) -> u32 {
        self.lag
    }

    /// One step per window row, in unit order.
    #[must_use]
    pub fn steps(&self) -> &[LagStep] {
        &self.steps
    }

    /// Value at the borrowed source for `target`.
    ///
    /// `None` when `target` is outside the window or the lag was refused.
    #[must_use]
    pub fn borrowed_value(&self, target: usize) -> Option<f64> {
        let step = self.steps.iter().find(|step| step.target == target)?;
        match step.read {
            LagRead::Borrowed(source) => self.column.get(source).copied(),
            LagRead::Refused(_) => None,
        }
    }
}

/// One query in a shared batch: a label, a treatment, and an outcome.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompatibleQuery {
    label: Arc<str>,
    treatment: Arc<str>,
    outcome: Arc<str>,
}

impl CompatibleQuery {
    /// Name the query and the two columns it reads besides the shared design.
    ///
    /// # Errors
    ///
    /// An empty label, treatment, or outcome, or a treatment that is the outcome.
    #[must_use = "an invalid query is not part of the shared batch"]
    pub fn try_new(label: &str, treatment: &str, outcome: &str) -> Result<Self, SeamError> {
        let label = require_name(label, "query")?;
        let treatment = require_name(treatment, "treatment")?;
        let outcome = require_name(outcome, "outcome")?;
        if treatment.as_ref() == outcome.as_ref() {
            return Err(invalid("treatment and outcome are the same column").into());
        }
        Ok(Self { label, treatment, outcome })
    }

    /// Caller-stable query label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }
}

/// Shared adjustment design and the queries that reuse it.
#[derive(Clone, Debug)]
pub struct SharedBatchRequest {
    adjustment: Vec<Arc<str>>,
    mask: Vec<bool>,
    nulls: IncomingNullPolicy,
    queries: Vec<CompatibleQuery>,
}

impl SharedBatchRequest {
    /// One adjustment list, one mask, and at least one query.
    ///
    /// # Errors
    ///
    /// An empty adjustment list, a repeated adjustment name, an empty query
    /// list, or two queries with the same label.
    #[must_use = "an invalid shared batch is not a prepared design"]
    pub fn try_new(
        adjustment: &[&str],
        mask: &[bool],
        nulls: IncomingNullPolicy,
        queries: Vec<CompatibleQuery>,
    ) -> Result<Self, SeamError> {
        if adjustment.is_empty() {
            return Err(invalid("a shared design needs an adjustment column").into());
        }
        if queries.is_empty() {
            return Err(invalid("a shared batch needs a query").into());
        }
        let mut stored = Vec::with_capacity(adjustment.len());
        for name in adjustment {
            stored.push(require_name(name, "adjustment")?);
        }
        if duplicate_name(&stored) {
            return Err(invalid("adjustment lists the same column twice").into());
        }
        let labels: Vec<Arc<str>> = queries.iter().map(|query| Arc::clone(&query.label)).collect();
        if duplicate_name(&labels) {
            return Err(invalid("two queries share a label").into());
        }
        Ok(Self { adjustment: stored, mask: mask.to_vec(), nulls, queries })
    }
}

/// One query bound to source column views.
#[derive(Clone, Debug)]
pub struct BoundQuery<'a> {
    label: Arc<str>,
    treatment: ColumnView<'a>,
    outcome: ColumnView<'a>,
}

impl<'a> BoundQuery<'a> {
    /// Query label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Source treatment column.
    #[must_use]
    pub const fn treatment(&self) -> ColumnView<'a> {
        self.treatment
    }

    /// Source outcome column.
    #[must_use]
    pub const fn outcome(&self) -> ColumnView<'a> {
        self.outcome
    }
}

/// Compatible multi-query plan: one design, N borrowed query columns.
///
/// The design is the adjustment columns at the shared selected rows. It is
/// built once. Each query borrows its own treatment and outcome from that same
/// storage. Execution stays on [`EXISTING_PREPARED_BATCH_API`]. This job does
/// not estimate and does not start a Polars engine.
#[derive(Clone, Debug)]
#[must_use = "the plan is the shared design the prepared batch API already consumes"]
pub struct CompatibleMultiQuery<'a> {
    queries: Vec<BoundQuery<'a>>,
    design: DesignLayout<'a>,
    parent_digest: [u8; 32],
    projected: bool,
}

impl<'a> CompatibleMultiQuery<'a> {
    /// Prepare one shared design for every query on `storage`.
    ///
    /// Selected rows are the mask intersected with the analysis mask and with
    /// validity of the adjustment columns and every query's treatment and
    /// outcome. The snapshot is one full-source digest. The column list is a
    /// projection of that source.
    ///
    /// # Errors
    ///
    /// A distinct-null policy, an unknown column, a mask length mismatch, an
    /// empty selected sample, or a non-numeric adjustment column.
    #[must_use = "a refused multi-query plan is not the prepared batch design"]
    pub fn prepare(
        storage: &'a OwnedColumnarStorage,
        request: &SharedBatchRequest,
    ) -> Result<Self, SeamError> {
        compiled_nulls(request.nulls)?;
        let mut ids = Vec::new();
        let mut adjustment_ids = Vec::with_capacity(request.adjustment.len());
        for name in &request.adjustment {
            let id = variable_id(storage, name)?;
            adjustment_ids.push(id);
            extend_unique(&mut ids, id);
        }
        let mut query_ids = Vec::with_capacity(request.queries.len());
        for query in &request.queries {
            let treatment = variable_id(storage, &query.treatment)?;
            let outcome = variable_id(storage, &query.outcome)?;
            extend_unique(&mut ids, treatment);
            extend_unique(&mut ids, outcome);
            query_ids.push((treatment, outcome));
        }
        let (parent_digest, projected) = source_projection(storage, &ids)?;
        require_projection(projected)?;
        let rows = kept_rows(storage, &ids, &request.mask)?;
        let adjustment = views(storage, &adjustment_ids)?;
        let design = DesignLayout::from_views(&adjustment, &rows)?;
        let mut queries = Vec::with_capacity(query_ids.len());
        for (query, (treatment, outcome)) in request.queries.iter().zip(query_ids) {
            queries.push(BoundQuery {
                label: Arc::clone(&query.label),
                treatment: storage.column(treatment)?,
                outcome: storage.column(outcome)?,
            });
        }
        Ok(Self { queries, design, parent_digest, projected })
    }

    /// API that executes this plan. This seam does not call it.
    #[must_use]
    pub const fn execution_api(&self) -> &'static str {
        EXISTING_PREPARED_BATCH_API
    }

    /// Crate resolution. This repository leaves Polars unresolved.
    #[must_use]
    pub const fn resolution(&self) -> PolarsResolution {
        PolarsResolution::current()
    }

    /// Queries in request order. They share [`Self::design`].
    #[must_use]
    pub fn queries(&self) -> &[BoundQuery<'a>] {
        &self.queries
    }

    /// Adjustment design built once for the whole family.
    #[must_use]
    pub const fn design(&self) -> &DesignLayout<'a> {
        &self.design
    }

    /// Full-source digest shared by every query.
    #[must_use]
    pub const fn parent_digest(&self) -> [u8; 32] {
        self.parent_digest
    }

    /// The selected columns are a projection, not the full-source identity.
    #[must_use]
    pub const fn is_projected(&self) -> bool {
        self.projected
    }
}

fn invalid(message: impl Into<String>) -> DataError {
    DataError::InvalidArgument { message: message.into() }
}

fn require_name(name: &str, role: &str) -> Result<Arc<str>, DataError> {
    if name.is_empty() {
        return Err(invalid(format!("{role} name is empty")));
    }
    Ok(Arc::from(name))
}

fn duplicate_name(names: &[Arc<str>]) -> bool {
    names.iter().enumerate().any(|(index, name)| {
        names.iter().take(index).any(|earlier| earlier.as_ref() == name.as_ref())
    })
}

fn compiled_nulls(nulls: IncomingNullPolicy) -> Result<(), DataError> {
    match nulls {
        IncomingNullPolicy::CompileToInvalidIffNan => Ok(()),
        IncomingNullPolicy::NullDistinctFromNan => Err(invalid(NULL_DISTINCT_REFUSAL)),
    }
}

fn parquet_path(path: &str) -> Result<Arc<str>, DataError> {
    if path.is_empty() {
        return Err(invalid("wide parquet path is empty; the seam does not open a file"));
    }
    Ok(Arc::from(path))
}

fn variable_id(storage: &OwnedColumnarStorage, name: &str) -> Result<VariableId, DataError> {
    storage.schema().id_of(name).map_err(DataError::from)
}

fn extend_unique(ids: &mut Vec<VariableId>, id: VariableId) {
    if !ids.contains(&id) {
        ids.push(id);
    }
}

fn source_projection(
    storage: &OwnedColumnarStorage,
    ids: &[VariableId],
) -> Result<([u8; 32], bool), DataError> {
    let source = OwnedSource::from_storage(storage.clone(), SourceModality::Tabular);
    let Some(parent) = source.contract().full_digest() else {
        return Err(invalid("wide and batch sources keep a full content digest"));
    };
    let dense: Vec<u32> = ids.iter().copied().map(VariableId::raw).collect();
    let projection = source.project(&dense)?;
    Ok((parent, projection.full_digest().is_none()))
}

fn require_projection(projected: bool) -> Result<(), DataError> {
    if projected {
        return Ok(());
    }
    Err(invalid("a required-column selection must not reuse the full-source digest"))
}

fn views<'a>(
    storage: &'a OwnedColumnarStorage,
    ids: &[VariableId],
) -> Result<Vec<ColumnView<'a>>, DataError> {
    ids.iter().copied().map(|id| storage.column(id)).collect()
}

fn kept_rows(
    storage: &OwnedColumnarStorage,
    variables: &[VariableId],
    mask: &[bool],
) -> Result<Vec<GlobalId>, DataError> {
    if mask.len() != storage.row_count() {
        return Err(DataError::LengthMismatch {
            expected: storage.row_count(),
            actual: mask.len(),
            context: "polars seam mask",
        });
    }
    let mut keep = complete_rows(storage, storage.analysis_mask(), variables)?;
    for (row, included) in mask.iter().copied().enumerate() {
        if let Some(slot) = keep.get_mut(row) {
            *slot = *slot && included;
        }
    }
    if !keep.iter().any(|included| *included) {
        return Err(DataError::EmptySelection { context: "polars seam" });
    }
    keep.iter()
        .enumerate()
        .filter(|&(_, &included)| included)
        .map(|(row, _)| global_id(row))
        .collect()
}

fn global_id(row: usize) -> Result<GlobalId, DataError> {
    u64::try_from(row)
        .map(GlobalId::new)
        .map_err(|_| invalid(format!("row {row} does not fit in a global id")))
}

#[cfg(test)]
mod tests {
    use crate::dataset::TabularData;
    use crate::design_layout::OwnershipCopy;
    use crate::panel_window::{DeclaredSplit, LONGER_SPAN_OF_SAME_UNIT, WindowBoundary};
    use crate::table::TableView;

    use super::*;

    fn table(columns: &[(&str, &[f64])]) -> TabularData {
        TabularData::from_f64_columns(columns.iter().copied()).expect("table")
    }

    fn effect_request(mask: &[bool], nulls: IncomingNullPolicy) -> AdjustedEffectRequest {
        AdjustedEffectRequest::try_new("t", "y", &["z"], mask, nulls).expect("request")
    }

    fn float_ptr(view: ColumnView<'_>) -> *const f64 {
        match view {
            ColumnView::Float64(column) => column.values.as_slice().as_ptr(),
            ColumnView::Int64(_)
            | ColumnView::Boolean(_)
            | ColumnView::Categorical(_)
            | ColumnView::Timestamp(_)
            | ColumnView::FixedVector(_) => panic!("expected a float64 column view"),
        }
    }

    #[test]
    fn polars_crate_is_unresolved_and_the_feature_is_off_by_default() {
        let manifest = include_str!("../Cargo.toml");
        assert!(manifest.contains("default = [\"arrow\"]"), "{manifest}");
        assert!(manifest.contains("\npolars = []\n"), "{manifest}");
        assert!(!manifest.contains("polars = {"), "{manifest}");
        assert_eq!(PolarsResolution::current(), PolarsResolution::Unresolved);
    }

    #[test]
    fn wide_parquet_adjusted_effect_borrows_a_contiguous_design() {
        let data =
            table(&[("z", &[1.0, 2.0, 3.0]), ("t", &[0.0, 1.0, 0.0]), ("y", &[1.0, 1.0, 2.0])]);
        let request =
            effect_request(&[true, true, true], IncomingNullPolicy::CompileToInvalidIffNan);
        let effect = WideParquetAdjustedEffect::prepare("wide.parquet", data.storage(), &request)
            .expect("contiguous effect");
        let z = data.storage().float64_slice(VariableId::from_raw(0)).expect("z");
        assert_eq!(effect.path(), "wide.parquet");
        assert_eq!(effect.resolution(), PolarsResolution::Unresolved);
        assert_eq!(effect.uncertainty(), UncertaintyPlace::ExistingEstimator);
        assert_eq!(effect.kernel_barrier(), ADJUSTED_EFFECT_KERNEL_BARRIER);
        assert!(effect.is_projected());
        assert_eq!(effect.parent_digest(), data.storage().content_digest());
        assert_eq!(effect.treatment().len(), 3);
        assert_eq!(effect.outcome().len(), 3);
        assert_eq!(effect.adjustment().len(), 1);
        assert!(effect.design().receipt().is_empty());
        assert_eq!(effect.design().n_rows(), 3);
        let design_z = effect.design().column(0).expect("z design");
        assert!(std::ptr::eq(design_z.as_ptr(), z.as_ptr()));
    }

    #[test]
    fn wide_parquet_adjusted_effect_records_a_gather_when_a_row_is_out() {
        let masked =
            table(&[("z", &[1.0, 2.0, 3.0]), ("t", &[0.0, 1.0, 0.0]), ("y", &[4.0, 5.0, 6.0])]);
        let request =
            effect_request(&[true, false, true], IncomingNullPolicy::CompileToInvalidIffNan);
        let effect = WideParquetAdjustedEffect::prepare("wide.parquet", masked.storage(), &request)
            .expect("masked effect");
        let gathered = effect.design().column(0).expect("gathered z");
        let copies = effect.design().receipt().copies();
        assert_eq!(effect.treatment().len(), 3);
        assert_eq!(effect.design().n_rows(), 2);
        assert_eq!(copies.len(), 1);
        assert_eq!(copies[0].name(), OwnershipCopy::GATHER);
        assert_eq!(copies[0].column(), 0);
        let z = masked.storage().float64_slice(VariableId::from_raw(0)).expect("z");
        assert!(!std::ptr::eq(gathered.as_ptr(), z.as_ptr()));
        assert!((gathered[0] - 1.0).abs() < f64::EPSILON);
        assert!((gathered[1] - 3.0).abs() < f64::EPSILON);

        let missing = table(&[
            ("z", &[1.0, 2.0, 3.0]),
            ("t", &[0.0, 1.0, 0.0]),
            ("y", &[4.0, f64::NAN, 6.0]),
        ]);
        let all = effect_request(&[true, true, true], IncomingNullPolicy::CompileToInvalidIffNan);
        let dropped = WideParquetAdjustedEffect::prepare("wide.parquet", missing.storage(), &all)
            .expect("invalid row drops out of the design");
        assert_eq!(dropped.outcome().len(), 3);
        assert_eq!(dropped.design().n_rows(), 2);
        assert_eq!(dropped.design().receipt().copies()[0].name(), OwnershipCopy::GATHER);
    }

    #[test]
    fn wide_parquet_adjusted_effect_refuses_null_distinct_from_nan() {
        let data = table(&[("z", &[1.0]), ("t", &[0.0]), ("y", &[1.0])]);
        let request = effect_request(&[true], IncomingNullPolicy::NullDistinctFromNan);
        let error = WideParquetAdjustedEffect::prepare("wide.parquet", data.storage(), &request)
            .expect_err("distinct nulls are not a column view");
        match error {
            SeamError::Data(DataError::InvalidArgument { message }) => {
                assert_eq!(message, NULL_DISTINCT_REFUSAL);
            }
            other => panic!("expected the null-policy refusal, got {other}"),
        }
    }

    fn assert_borrowed_lag(unit: PopulationUnit) {
        let column = [10.0, 20.0, 30.0, 40.0];
        let split = DeclaredSplit::try_new("estimation", 0..4).expect("split");
        let calendar = vec![0_i64, 1, 2, 3];
        let partition = UnitPartition::try_new(unit, calendar, 1, [split]).expect("unit");
        let span = CalendarSpan::try_new(0, 4).expect("span");
        let prep = BorrowedLagPreparation::prepare(&partition, "estimation", span, &column, 1)
            .expect("lag window");
        assert_eq!(prep.population(), unit);
        assert_eq!(prep.lag(), 1);
        assert!(std::ptr::eq(prep.column().as_ptr(), column.as_ptr()));
        assert_eq!(prep.column().len(), column.len());
        match prep.steps()[0].read() {
            LagRead::Refused(error) => {
                let refusal = error.refusal().expect("unit boundary");
                assert_eq!(refusal.boundary(), WindowBoundary::Unit);
                assert_eq!(refusal.next_attempt().name(), LONGER_SPAN_OF_SAME_UNIT);
            }
            LagRead::Borrowed(_) => panic!("lag 1 at the first row leaves the unit"),
        }
        let across = prep.window().borrow_lag(3, 1).expect("row 3 stays in the span");
        match prep.steps().iter().find(|step| step.target() == 3).expect("row 3").read() {
            LagRead::Borrowed(source) => assert_eq!(*source, across),
            LagRead::Refused(_) => panic!("row 3 borrows inside the span"),
        }
        assert_eq!(across, 2);
        let value = prep.borrowed_value(3).expect("row 3 has a borrowed value");
        assert!((value - column[2]).abs() < f64::EPSILON);
        let (left, right) = prep.window().chunk_at(2).expect("cut inside the window");
        assert_eq!(left.local_rows(), 0..2);
        assert_eq!(right.local_rows(), 2..4);
        assert_eq!(prep.window().borrow_lag(3, 1).expect("a chunk cut is not a boundary"), across);
    }

    #[test]
    fn panel_temporal_preparation_borrows_lag_windows() {
        assert_borrowed_lag(PopulationUnit::panel(7));
    }

    #[test]
    fn multi_environment_temporal_preparation_borrows_lag_windows() {
        assert_borrowed_lag(PopulationUnit::environment(4));
    }

    #[test]
    fn compatible_multi_query_shares_one_prepared_design() {
        let data = table(&[
            ("z", &[1.0, 2.0, 3.0]),
            ("t", &[0.0, 1.0, 0.0]),
            ("y0", &[1.0, 1.0, 2.0]),
            ("y1", &[3.0, 4.0, 5.0]),
        ]);
        let queries = vec![
            CompatibleQuery::try_new("ate-y0", "t", "y0").expect("y0"),
            CompatibleQuery::try_new("ate-y1", "t", "y1").expect("y1"),
        ];
        let request = SharedBatchRequest::try_new(
            &["z"],
            &[true, true, true],
            IncomingNullPolicy::CompileToInvalidIffNan,
            queries,
        )
        .expect("batch request");
        let plan = CompatibleMultiQuery::prepare(data.storage(), &request).expect("shared design");
        let z = data.storage().float64_slice(VariableId::from_raw(0)).expect("z");
        assert_eq!(plan.resolution(), PolarsResolution::Unresolved);
        assert_eq!(plan.execution_api(), EXISTING_PREPARED_BATCH_API);
        assert!(plan.is_projected());
        assert_eq!(plan.parent_digest(), data.storage().content_digest());
        assert!(plan.design().receipt().is_empty());
        assert_eq!(plan.design().n_columns(), 1);
        assert_eq!(plan.design().n_rows(), 3);
        let design_z = plan.design().column(0).expect("shared z");
        assert!(std::ptr::eq(design_z.as_ptr(), z.as_ptr()));
        assert_eq!(plan.queries().len(), 2);
        assert_eq!(plan.queries()[0].label(), "ate-y0");
        assert_eq!(plan.queries()[1].label(), "ate-y1");
        let shared_treatment = float_ptr(plan.queries()[0].treatment());
        assert!(std::ptr::eq(shared_treatment, float_ptr(plan.queries()[1].treatment())));
        let y0 = float_ptr(plan.queries()[0].outcome());
        assert!(!std::ptr::eq(y0, float_ptr(plan.queries()[1].outcome())));
    }
}
