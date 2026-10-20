//! Explicit materialization and the preparation plan estimators consume.
//!
//! The accepted path is direct Arrow. A lazy backend is admitted only when this
//! crate has adopted one. Column order is the caller's order. Schema, category,
//! and missing-value conversions run in that order, then resource admission,
//! then delivery. A streaming scan is not a bounded end-to-end route: the bound
//! is proved only when the estimator is given a finite table and no longer sees
//! the scan.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ValueType, VariableId};

#[cfg(feature = "arrow")]
use antecedent_core::{Diagnostic, DiagnosticKind, DiagnosticSeverity};

use crate::backend_boundary::{LazyBackendRefusal, PreparationBackend};
use crate::categorical::{CategoryConversion, Contrast, ExpandedContrast, expand_contrast};
use crate::column::OwnedColumn;
use crate::dataset::TabularData;
use crate::error::DataError;
use crate::missing::MissingConversion;
use crate::sample_policy::MissingPolicy;
use crate::table::TableView;

/// Why a buffer was materialized.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg(feature = "arrow")]
pub(crate) enum MaterializationReason {
    /// Arrow / foreign buffer could not be borrowed zero-copy.
    ForeignBufferIncompatible,
    /// A validity bitmap was built from Arrow null slots beside a zero-copy value borrow.
    ValidityBitmap,
    /// The downstream estimator cannot stream, so chunks were copied into one table.
    EstimatorCannotStream,
}

/// Record a materialization as an execution diagnostic.
#[must_use]
#[cfg(feature = "arrow")]
pub(crate) fn materialization_diagnostic(reason: MaterializationReason, bytes: u64) -> Diagnostic {
    let code = match reason {
        MaterializationReason::ForeignBufferIncompatible => "exec.materialize.foreign",
        MaterializationReason::ValidityBitmap => "exec.materialize.validity",
        MaterializationReason::EstimatorCannotStream => "exec.materialize.estimator",
    };
    let mut d = Diagnostic::new(
        code,
        DiagnosticKind::Execution,
        DiagnosticSeverity::Info,
        format!("materialized {bytes} bytes ({reason:?})"),
    );
    d.fields = std::sync::Arc::from([(
        std::sync::Arc::<str>::from("bytes"),
        std::sync::Arc::<str>::from(bytes.to_string()),
    )]);
    d
}

/// Steps of the accepted preparation path, in the only order they run.
pub const PREPARATION_ORDER: [PreparationStep; 5] = [
    PreparationStep::Schema,
    PreparationStep::Category,
    PreparationStep::MissingValue,
    PreparationStep::Admission,
    PreparationStep::Delivery,
];

/// One step of [`PREPARATION_ORDER`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PreparationStep {
    /// Bind each requested column's schema type.
    Schema,
    /// Expand categorical columns through an explicit contrast.
    Category,
    /// Apply the declared missing-value policy.
    MissingValue,
    /// Admit the byte estimate against the budget.
    Admission,
    /// Hand the estimator a finite table or a scan.
    Delivery,
}

impl PreparationStep {
    /// Stable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Schema => "schema",
            Self::Category => "category",
            Self::MissingValue => "missing_value",
            Self::Admission => "admission",
            Self::Delivery => "delivery",
        }
    }
}

/// Whether the estimator can read a scan.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum EstimatorStreamNeed {
    /// The estimator reads one finite table.
    CannotStream,
    /// The estimator can scan chunks and does not retain them.
    CanStream,
}

/// Byte budget for one preparation.
///
/// Admission compares a declared estimate with this limit. It does not allocate.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PreparationBudget {
    bytes: u64,
}

impl PreparationBudget {
    /// Budget of `bytes`.
    #[must_use]
    pub const fn new(bytes: u64) -> Self {
        Self { bytes }
    }

    /// Budget in bytes.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.bytes
    }
}

/// Controls for one preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PreparationControls {
    /// Byte budget. `None` admits the resident table's own payload.
    pub budget: Option<PreparationBudget>,
    /// Direct Arrow, or a lazy backend when one has been adopted.
    pub backend: PreparationBackend,
    /// What the downstream estimator can read.
    pub stream: EstimatorStreamNeed,
}

impl PreparationControls {
    /// Direct Arrow, a finite table, and a budget equal to that table's payload.
    #[must_use]
    pub const fn finite_resident() -> Self {
        Self {
            budget: None,
            backend: PreparationBackend::DirectArrow,
            stream: EstimatorStreamNeed::CannotStream,
        }
    }
}

/// One requested column after schema conversion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchemaColumn {
    /// Schema name.
    pub name: Arc<str>,
    /// Dense id.
    pub id: VariableId,
    /// Value-type name.
    pub value_kind: &'static str,
}

/// Schema conversion in the caller's column order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchemaConversion {
    /// Columns in request order. This is not sorted back into schema order.
    pub columns: Arc<[SchemaColumn]>,
}

/// How many bytes were admitted, and against which budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum AdmissionRecord {
    /// A finite copy or a resident table of `bytes`, inside `budget`.
    Materialized {
        /// Admitted payload.
        bytes: u64,
        /// Budget the payload was compared with.
        budget: u64,
    },
    /// Scratch for a scan, inside `budget`.
    Streaming {
        /// Admitted scratch.
        scratch_bytes: u64,
        /// Budget the scratch was compared with.
        budget: u64,
    },
}

/// What the estimator was given.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DeliveryRecord {
    /// One finite table of `rows` rows.
    Finite {
        /// Row count of that table.
        rows: u64,
    },
    /// A scan. Not an end-to-end bound.
    Scan,
}

/// End-to-end bound, or an explicit statement that the scan is not one.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RouteBound {
    /// The estimator reads this many rows and does not see a scan.
    Proved {
        /// Finite row count.
        rows: u64,
    },
    /// The estimator was handed a scan. This is not a bound.
    NotAnEndToEndRoute,
}

impl RouteBound {
    /// Whether this value is a proved end-to-end bound.
    #[must_use]
    pub const fn is_bounded_end_to_end_route(self) -> bool {
        matches!(self, Self::Proved { .. })
    }
}

/// Where a receipt came from.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ReceiptSource {
    /// [`prepare_table`] or a scan delivery.
    PreparedPath,
    /// Finite rows concatenated from problems that were already materialized.
    ComposedFinite,
}

/// Plan receipt: path, order, conversions, admission, and delivery.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparationReceipt {
    path: &'static str,
    source: ReceiptSource,
    schema: SchemaConversion,
    categories: Arc<[CategoryConversion]>,
    missing: Option<MissingConversion>,
    admission: AdmissionRecord,
    delivery: DeliveryRecord,
    route_bound: RouteBound,
    digest: Arc<str>,
}

impl PreparationReceipt {
    /// Receipt for rows that already sit in one finite buffer.
    ///
    /// The row count is the bound. This is not a scan.
    ///
    /// # Errors
    ///
    /// `rows` does not fit in `u64`.
    pub fn composed_finite(rows: usize) -> Result<Self, DataError> {
        let rows = fit_u64(rows)?;
        Self::finish(
            ReceiptSource::ComposedFinite,
            SchemaConversion { columns: Arc::from([]) },
            Arc::from([]),
            None,
            AdmissionRecord::Materialized { bytes: 0, budget: 0 },
            DeliveryRecord::Finite { rows },
            RouteBound::Proved { rows },
        )
    }

    /// Preparation path name.
    #[must_use]
    pub const fn path(&self) -> &'static str {
        self.path
    }

    /// How this receipt was built.
    #[must_use]
    pub const fn source(&self) -> ReceiptSource {
        self.source
    }

    /// Schema conversion in request order.
    #[must_use]
    pub const fn schema(&self) -> &SchemaConversion {
        &self.schema
    }

    /// Category conversions. Empty when no requested column is categorical.
    #[must_use]
    pub fn categories(&self) -> &[CategoryConversion] {
        &self.categories
    }

    /// Missing-value conversion, when the path ran one.
    pub const fn missing(&self) -> Option<&MissingConversion> {
        self.missing.as_ref()
    }

    /// Admitted bytes.
    #[must_use]
    pub const fn admission(&self) -> AdmissionRecord {
        self.admission
    }

    /// What was delivered.
    #[must_use]
    pub const fn delivery(&self) -> DeliveryRecord {
        self.delivery
    }

    /// End-to-end bound.
    #[must_use]
    pub const fn route_bound(&self) -> RouteBound {
        self.route_bound
    }

    /// Whether [`Self::route_bound`] is a proved finite row count.
    #[must_use]
    pub const fn is_bounded_end_to_end_route(&self) -> bool {
        self.route_bound.is_bounded_end_to_end_route()
    }

    /// Whether the estimator was given a scan rather than a finite table.
    #[must_use]
    pub const fn estimator_sees_unbounded_stream(&self) -> bool {
        matches!(self.delivery, DeliveryRecord::Scan)
    }

    /// Proved row count. `None` when the delivery is a scan.
    pub const fn proved_rows(&self) -> Option<u64> {
        match self.route_bound {
            RouteBound::Proved { rows } => Some(rows),
            RouteBound::NotAnEndToEndRoute => None,
        }
    }

    /// Lower-case hex digest of the plan. It does not include a timing.
    #[must_use]
    pub fn digest(&self) -> &str {
        &self.digest
    }

    /// Steps, always [`PREPARATION_ORDER`].
    #[must_use]
    pub const fn steps(&self) -> &'static [PreparationStep] {
        &PREPARATION_ORDER
    }

    /// Refuse a receipt that calls a scan a bounded route.
    ///
    /// # Errors
    ///
    /// The route is marked bounded and the estimator still sees a scan.
    pub fn prove_bound_hides_stream(&self) -> Result<(), DataError> {
        if self.is_bounded_end_to_end_route() && self.estimator_sees_unbounded_stream() {
            return Err(DataError::InvalidArgument {
                message: "a streaming scan was described as a bounded end-to-end route while the \
                          estimator still sees the scan"
                    .into(),
            });
        }
        Ok(())
    }

    fn finish(
        source: ReceiptSource,
        schema: SchemaConversion,
        categories: Arc<[CategoryConversion]>,
        missing: Option<MissingConversion>,
        admission: AdmissionRecord,
        delivery: DeliveryRecord,
        route_bound: RouteBound,
    ) -> Result<Self, DataError> {
        let receipt = Self {
            path: PreparationBackend::accepted().as_str(),
            source,
            schema,
            categories,
            missing,
            admission,
            delivery,
            route_bound,
            digest: Arc::from(""),
        };
        let digest = Arc::from(receipt_digest(&receipt));
        let receipt = Self { digest, ..receipt };
        receipt.prove_bound_hides_stream()?;
        Ok(receipt)
    }
}

/// Owned products of preparing a finite table.
#[derive(Clone, Debug)]
pub struct PreparationProduct {
    /// Contrast columns for categorical variables. Float columns are absent.
    pub expanded: Vec<ExpandedContrast>,
    /// Plan receipt.
    pub receipt: PreparationReceipt,
}

/// A scan handed to an estimator that can stream.
///
/// [`Self::end_to_end_bound`] is always `None`. Chunk lengths describe the
/// prefix that was seen. They are not a bound on a later chunk.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamingScan {
    chunk_rows: Arc<[u64]>,
    receipt: PreparationReceipt,
}

impl StreamingScan {
    /// Chunk lengths of the prefix. Their sum is not an end-to-end bound.
    #[must_use]
    pub fn chunk_rows(&self) -> &[u64] {
        &self.chunk_rows
    }

    /// End-to-end row bound.
    ///
    /// This is `None`. A scan stays unbounded until the estimator is given a
    /// finite table instead.
    pub const fn end_to_end_bound(&self) -> Option<u64> {
        None
    }

    /// The estimator received this scan.
    #[must_use]
    pub const fn estimator_sees_stream(&self) -> bool {
        true
    }

    /// Plan receipt. Its route is not an end-to-end bound.
    #[must_use]
    pub fn receipt(&self) -> &PreparationReceipt {
        &self.receipt
    }
}

/// Quote for a stream that is not an Arrow frame.
///
/// `materialize_bytes` and `scratch_bytes` are declared estimates. Absence is
/// not zero.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamQuote {
    /// Scratch a streaming estimator would retain.
    pub scratch_bytes: Option<u64>,
    /// Bytes a materialization would copy.
    pub materialize_bytes: Option<u64>,
    /// Per-chunk row counts of the prefix.
    pub chunk_rows: Vec<u64>,
}

/// Result of delivering a [`StreamQuote`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StreamDelivery {
    /// The estimator can stream. The scan is not a bounded route.
    Scan(StreamingScan),
    /// The estimator cannot stream and the byte quote was admitted.
    ///
    /// The quote has no resident rows. The Arrow frame is what materializes.
    MaterializeRequired {
        /// Admitted copy size.
        bytes: u64,
    },
}

/// Prepare `table` for an estimator that cannot stream.
///
/// Order is `order`, not schema order. A categorical column in `order` needs a
/// matching entry in `categories`.
///
/// # Errors
///
/// The backend is not admitted, the estimator asked to stream a finite table,
/// a column or contrast is missing, the missing-value policy fails, or the
/// payload exceeds the budget.
pub fn prepare_table(
    table: &TabularData,
    order: &[VariableId],
    missing: MissingPolicy,
    categories: &[(VariableId, Contrast)],
    controls: PreparationControls,
) -> Result<PreparationProduct, DataError> {
    admit_backend(controls.backend)?;
    if matches!(controls.stream, EstimatorStreamNeed::CanStream) {
        return Err(DataError::InvalidArgument {
            message: "a finite table is not opened as an unbounded scan; stream a chunked frame \
                      or read the table"
                .into(),
        });
    }
    let schema = convert_schema(table, order)?;
    let expanded = convert_categories(table, order, categories)?;
    let missing = convert_missing(table, order, missing)?;
    let bytes = payload_bytes(table)?;
    let budget = controls.budget.map_or(bytes, PreparationBudget::bytes);
    admit_bytes(bytes, budget, "materialization")?;
    let rows = fit_u64(table.row_count())?;
    let categories = expanded.iter().map(|column| column.conversion.clone()).collect::<Vec<_>>();
    let receipt = PreparationReceipt::finish(
        ReceiptSource::PreparedPath,
        schema,
        Arc::from(categories),
        Some(missing),
        AdmissionRecord::Materialized { bytes, budget },
        DeliveryRecord::Finite { rows },
        RouteBound::Proved { rows },
    )?;
    Ok(PreparationProduct { expanded, receipt })
}

/// Deliver `quote` without inventing a table.
///
/// An estimator that cannot stream either gets [`StreamDelivery::MaterializeRequired`]
/// when the copy fits, or a refusal when the estimate is absent or over budget.
/// An estimator that can stream gets a scan whose end-to-end bound is absent.
///
/// # Errors
///
/// The backend is not admitted, a required estimate is absent, or the estimate
/// exceeds `budget`.
pub fn deliver_stream(
    quote: &StreamQuote,
    budget: PreparationBudget,
    controls: PreparationControls,
) -> Result<StreamDelivery, DataError> {
    admit_backend(controls.backend)?;
    match controls.stream {
        EstimatorStreamNeed::CanStream => {
            let scratch = required_estimate(quote.scratch_bytes, "streaming scratch")?;
            admit_bytes(scratch, budget.bytes(), "streaming scratch")?;
            Ok(StreamDelivery::Scan(scan_from_quote(quote, scratch, budget.bytes())?))
        }
        EstimatorStreamNeed::CannotStream => {
            let bytes = required_estimate(quote.materialize_bytes, "materialization")?;
            admit_bytes(bytes, budget.bytes(), "materialization")?;
            Ok(StreamDelivery::MaterializeRequired { bytes })
        }
    }
}

/// Build a scan receipt. `missing.population_bound` must be false.
///
/// # Errors
///
/// `missing` claims a population bound, or the receipt would call the scan bounded.
pub fn finish_scan(
    schema: SchemaConversion,
    categories: &[CategoryConversion],
    missing: MissingConversion,
    chunk_rows: &[u64],
    scratch: u64,
    budget: u64,
) -> Result<StreamingScan, DataError> {
    if missing.population_bound {
        return Err(DataError::InvalidArgument {
            message: "a streaming scan must not record a population bound".into(),
        });
    }
    let receipt = PreparationReceipt::finish(
        ReceiptSource::PreparedPath,
        schema,
        Arc::from(categories),
        Some(missing),
        AdmissionRecord::Streaming { scratch_bytes: scratch, budget },
        DeliveryRecord::Scan,
        RouteBound::NotAnEndToEndRoute,
    )?;
    Ok(StreamingScan { chunk_rows: Arc::from(chunk_rows), receipt })
}

impl AdmissionRecord {
    /// Materialized payload, when delivery copied or kept a finite table.
    pub const fn materialized_bytes(self) -> Option<u64> {
        match self {
            Self::Materialized { bytes, .. } => Some(bytes),
            Self::Streaming { .. } => None,
        }
    }
}

fn scan_from_quote(
    quote: &StreamQuote,
    scratch: u64,
    budget: u64,
) -> Result<StreamingScan, DataError> {
    let receipt = PreparationReceipt::finish(
        ReceiptSource::PreparedPath,
        SchemaConversion { columns: Arc::from([]) },
        Arc::from([]),
        None,
        AdmissionRecord::Streaming { scratch_bytes: scratch, budget },
        DeliveryRecord::Scan,
        RouteBound::NotAnEndToEndRoute,
    )?;
    Ok(StreamingScan { chunk_rows: Arc::from(quote.chunk_rows.as_slice()), receipt })
}

fn admit_backend(backend: PreparationBackend) -> Result<(), DataError> {
    backend.admit().map(|_| ()).map_err(|refusal: LazyBackendRefusal| {
        DataError::InvalidArgument { message: refusal.to_string() }
    })
}

fn admit_bytes(estimated: u64, budget: u64, what: &str) -> Result<(), DataError> {
    if estimated > budget {
        return Err(DataError::InvalidArgument {
            message: format!(
                "preparation refused: {what} of {estimated} bytes exceeds budget {budget}"
            ),
        });
    }
    Ok(())
}

fn required_estimate(estimate: Option<u64>, what: &str) -> Result<u64, DataError> {
    estimate.ok_or_else(|| DataError::InvalidArgument {
        message: format!(
            "preparation refused: {what} estimate is absent; an absent estimate is not zero"
        ),
    })
}

fn convert_schema(
    table: &TabularData,
    order: &[VariableId],
) -> Result<SchemaConversion, DataError> {
    let mut seen = Vec::with_capacity(order.len());
    let mut columns = Vec::with_capacity(order.len());
    for &id in order {
        if seen.contains(&id) {
            continue;
        }
        seen.push(id);
        let variable = table.schema().get(id)?;
        let physical = table
            .storage()
            .columns()
            .get(id.as_usize())
            .ok_or(DataError::UnknownVariable { id })?;
        if !physical_matches(&variable.value_type, physical) {
            return Err(DataError::TypeMismatch {
                id,
                expected: "a column whose storage matches the schema value type",
            });
        }
        columns.push(SchemaColumn {
            name: Arc::clone(&variable.name),
            id,
            value_kind: value_kind(&variable.value_type),
        });
    }
    Ok(SchemaConversion { columns: Arc::from(columns) })
}

fn convert_categories(
    table: &TabularData,
    order: &[VariableId],
    categories: &[(VariableId, Contrast)],
) -> Result<Vec<ExpandedContrast>, DataError> {
    let mut expanded = Vec::new();
    for &id in order {
        let column = table
            .storage()
            .columns()
            .get(id.as_usize())
            .ok_or(DataError::UnknownVariable { id })?;
        let OwnedColumn::Categorical(categorical) = column else {
            continue;
        };
        let Some((_, contrast)) = categories.iter().find(|(variable, _)| *variable == id) else {
            return Err(DataError::InvalidArgument {
                message: format!(
                    "category conversion requires an explicit contrast for variable {id}"
                ),
            });
        };
        expanded.push(expand_contrast(categorical, contrast)?);
    }
    for (id, _) in categories {
        if !order.contains(id) {
            return Err(DataError::InvalidArgument {
                message: format!("category contrast for variable {id} is not in the column order"),
            });
        }
        let column = table
            .storage()
            .columns()
            .get(id.as_usize())
            .ok_or(DataError::UnknownVariable { id: *id })?;
        if !matches!(column, OwnedColumn::Categorical(_)) {
            return Err(DataError::TypeMismatch { id: *id, expected: "categorical" });
        }
    }
    Ok(expanded)
}

fn convert_missing(
    table: &TabularData,
    order: &[VariableId],
    policy: MissingPolicy,
) -> Result<MissingConversion, DataError> {
    MissingConversion::convert(table, table.storage().analysis_mask(), order, policy)
}

fn physical_matches(value_type: &ValueType, column: &OwnedColumn) -> bool {
    match (value_type, column) {
        (ValueType::Continuous | ValueType::Count | ValueType::Binary, OwnedColumn::Float64(_))
        | (ValueType::Count | ValueType::Binary, OwnedColumn::Int64(_))
        | (ValueType::Binary, OwnedColumn::Boolean(_))
        | (ValueType::Count, OwnedColumn::Timestamp(_))
        | (ValueType::Categorical | ValueType::Ordinal, OwnedColumn::Categorical(_))
        | (ValueType::Vector { .. }, OwnedColumn::FixedVector(_)) => true,
        _ => false,
    }
}

fn value_kind(value_type: &ValueType) -> &'static str {
    match value_type {
        ValueType::Continuous => "continuous",
        ValueType::Count => "count",
        ValueType::Binary => "binary",
        ValueType::Categorical => "categorical",
        ValueType::Ordinal => "ordinal",
        ValueType::Unspecified => "unspecified",
        ValueType::Vector { .. } => "vector",
    }
}

fn payload_bytes(table: &TabularData) -> Result<u64, DataError> {
    let mut total = 0u64;
    for column in table.storage().columns() {
        let (n_values, width) = match column {
            OwnedColumn::Float64(column) => (column.len(), core::mem::size_of::<f64>()),
            OwnedColumn::Int64(column) => (column.len(), core::mem::size_of::<i64>()),
            OwnedColumn::Timestamp(column) => (column.len(), core::mem::size_of::<i64>()),
            OwnedColumn::Boolean(column) => (column.len(), 1usize),
            OwnedColumn::Categorical(column) => {
                (column.len(), core::mem::size_of::<crate::categorical::CategoryCode>())
            }
            OwnedColumn::FixedVector(column) => (column.values.len(), core::mem::size_of::<f64>()),
        };
        let values = checked_usize(n_values, width)?;
        let validity = column.as_view().validity().len().div_ceil(8);
        let column_bytes = values.checked_add(validity).ok_or_else(payload_overflow)?;
        total = total.checked_add(fit_u64(column_bytes)?).ok_or_else(payload_overflow)?;
    }
    Ok(total)
}

fn checked_usize(count: usize, width: usize) -> Result<usize, DataError> {
    count.checked_mul(width).ok_or_else(payload_overflow)
}

fn payload_overflow() -> DataError {
    DataError::InvalidArgument { message: "column payload size overflows".into() }
}

fn fit_u64(value: usize) -> Result<u64, DataError> {
    u64::try_from(value).map_err(|_| DataError::InvalidArgument {
        message: "value does not fit u64".into(),
    })
}

fn receipt_digest(receipt: &PreparationReceipt) -> String {
    let mut hasher = blake3::Hasher::new_derive_key("antecedent.data.preparation.v1");
    hasher.update(receipt.path.as_bytes());
    for step in PREPARATION_ORDER {
        hasher.update(step.as_str().as_bytes());
    }
    for column in receipt.schema.columns.iter() {
        hasher.update(column.name.as_bytes());
        hasher.update(column.value_kind.as_bytes());
    }
    for category in receipt.categories.iter() {
        hasher.update(category.coding.as_bytes());
        hasher.update(&category.n_columns.to_le_bytes());
    }
    if let Some(missing) = &receipt.missing {
        hasher.update(missing.policy.name().as_bytes());
        hasher.update(&[u8::from(missing.population_bound)]);
    }
    match receipt.admission {
        AdmissionRecord::Materialized { bytes, budget } => {
            hasher.update(&bytes.to_le_bytes());
            hasher.update(&budget.to_le_bytes());
        }
        AdmissionRecord::Streaming { scratch_bytes, budget } => {
            hasher.update(&scratch_bytes.to_le_bytes());
            hasher.update(&budget.to_le_bytes());
        }
    }
    match receipt.delivery {
        DeliveryRecord::Finite { rows } => {
            hasher.update(&rows.to_le_bytes());
        }
        DeliveryRecord::Scan => {
            hasher.update(b"scan");
        }
    }
    hasher.finalize().to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend_boundary::PreparationBackend;
    use crate::sample_policy::MissingPolicy;

    fn two_columns() -> TabularData {
        TabularData::from_f64_columns([
            ("x", &[1.0, f64::NAN][..]),
            ("y", &[3.0, 4.0][..]),
        ])
        .unwrap()
    }

    #[test]
    fn order_is_the_request_and_missing_rows_are_proved() {
        let table = two_columns();
        let order = [VariableId::from_raw(1), VariableId::from_raw(0)];
        let product = prepare_table(
            &table,
            &order,
            MissingPolicy::CompleteCase,
            &[],
            PreparationControls::finite_resident(),
        )
        .unwrap();
        let names: Vec<_> =
            product.receipt.schema().columns.iter().map(|column| column.name.as_ref()).collect();
        assert_eq!(names, ["y", "x"]);
        assert_eq!(
            product.receipt.steps().iter().map(|step| step.as_str()).collect::<Vec<_>>(),
            ["schema", "category", "missing_value", "admission", "delivery"]
        );
        assert_eq!(product.receipt.path(), "direct_arrow");
        assert_eq!(product.receipt.proved_rows(), Some(2));
        assert!(!product.receipt.estimator_sees_unbounded_stream());
        assert!(product.receipt.is_bounded_end_to_end_route());
        let missing = product.receipt.missing().unwrap();
        assert!(missing.population_bound);
        assert_eq!(missing.proved_rows_kept(), Some(1));
        assert_eq!(product.receipt.digest().len(), 64);
    }

    #[test]
    fn a_budget_below_the_payload_refuses_before_a_bound_is_claimed() {
        let table = two_columns();
        let mut controls = PreparationControls::finite_resident();
        controls.budget = Some(PreparationBudget::new(0));
        let refused = prepare_table(
            &table,
            &[VariableId::from_raw(0)],
            MissingPolicy::CompleteCase,
            &[],
            controls,
        );
        let message = refused.expect_err("zero budget").to_string();
        assert!(message.contains("exceeds budget 0"), "{message}");
    }

    #[test]
    fn lazy_backend_is_refused_because_none_is_adopted() {
        let table = two_columns();
        let mut controls = PreparationControls::finite_resident();
        controls.backend = PreparationBackend::Lazy;
        let message = prepare_table(&table, &[], MissingPolicy::CompleteCase, &[], controls)
            .expect_err("lazy")
            .to_string();
        assert!(message.contains("lazy backend is not adopted"), "{message}");
    }

    #[test]
    fn a_scan_is_not_a_bounded_route_while_the_estimator_sees_it() {
        let quote = StreamQuote {
            scratch_bytes: Some(16),
            materialize_bytes: Some(32),
            chunk_rows: vec![2, 5],
        };
        let mut controls = PreparationControls::finite_resident();
        controls.stream = EstimatorStreamNeed::CanStream;
        controls.budget = Some(PreparationBudget::new(16));
        let StreamDelivery::Scan(scan) =
            deliver_stream(&quote, PreparationBudget::new(16), controls).unwrap()
        else {
            panic!("expected a scan");
        };
        if scan.end_to_end_bound().is_some() || scan.receipt().is_bounded_end_to_end_route() {
            assert!(
                !scan.estimator_sees_stream(),
                "bound claimed while the estimator still sees an unbounded stream"
            );
        }
        assert!(scan.estimator_sees_stream());
        assert!(scan.end_to_end_bound().is_none());
        assert!(!scan.receipt().is_bounded_end_to_end_route());
        assert_eq!(scan.chunk_rows(), &[2, 5]);

        controls.stream = EstimatorStreamNeed::CannotStream;
        let required =
            deliver_stream(&quote, PreparationBudget::new(32), controls).unwrap();
        assert!(matches!(required, StreamDelivery::MaterializeRequired { bytes: 32 }));
        let absent = StreamQuote { materialize_bytes: None, ..quote };
        let message = deliver_stream(&absent, PreparationBudget::new(32), controls)
            .expect_err("absent estimate")
            .to_string();
        assert!(message.contains("estimate is absent"), "{message}");
    }
}
