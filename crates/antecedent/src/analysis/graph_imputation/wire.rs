//! Original-source wire for the graph-imputation composite.
//!
//! The artifact is the shared contract's canonical source: the original table and declaration,
//! the compiler's lowered program identity and, for every supported graph, the child operation
//! the compiler retained. Decoding never grants authority. Consumption hands the bytes to the
//! native replay in the I/O layer, which re-derives each child operation from its graph and
//! refuses unless it equals the compiler's, then executes under the carried program identity and
//! compares a separately held expected program, source and result identity.
#[allow(
    clippy::wildcard_imports,
    reason = "the wire module shares the private compiled candidate types"
)]
use super::*;
use antecedent_data::OwnedColumn;

const MAX_NAME_BYTES: usize = antecedent_core::graph_imputation_contract::MAX_NAME_BYTES;

/// An independently held expected question, source and numerical result identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphImputationExpectation {
    /// Expected compiled program identity.
    pub program: [u8; 32],
    /// Expected original source identity.
    pub source: [u8; 32],
    /// Expected complete execution identity.
    pub receipt: [u8; 32],
}

/// The serialized source and checked result held with its admission lease.
pub struct GraphImputationArtifact {
    bytes: Arc<[u8]>,
    expected: GraphImputationExpectation,
    _lease: antecedent_core::execution::AdmissionLease,
}
impl GraphImputationArtifact {
    /// Encoded native source and retained child operations.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// Identity to retain independently of encoded bytes.
    pub fn expectation(&self) -> GraphImputationExpectation {
        self.expected
    }
}

pub(super) type RawColumns = Vec<(Arc<str>, Vec<f64>)>;

pub(super) fn raw_columns(source: &TabularData) -> Result<RawColumns, CausalError> {
    if source.storage().analysis_mask().is_some() || source.storage().weights().is_some() {
        return Err(invalid("source mask or weights cannot be replayed by this bounded cell"));
    }
    let mut columns = Vec::new();
    for variable in source.schema().variables() {
        let Some(OwnedColumn::Float64(column)) =
            source.storage().columns().get(variable.id.raw() as usize)
        else {
            return Err(invalid("source contains a non-float or non-dense column"));
        };
        if column.id != variable.id
            || variable.name.is_empty()
            || variable.name.len() > MAX_NAME_BYTES
        {
            return Err(invalid("source variable identity is invalid or too long"));
        }
        columns.push((Arc::clone(&variable.name), column.values.as_slice().to_vec()));
    }
    Ok(columns)
}

/// Require the canonical named-float schema and bit-identical source, including missing-cell
/// validity and NaN payloads, before any artifact can be admitted for this bounded cell.
pub(super) fn canonical_source(
    source: &TabularData,
    ctx: &ExecutionContext,
) -> Result<RawColumns, CausalError> {
    let columns = raw_columns(source)?;
    let rebuilt = TabularData::from_f64_columns(
        columns.iter().map(|(name, values)| (Arc::clone(name), values.as_slice())),
    )?;
    let digest = |table: &TabularData| {
        table.storage().content_digest_with_context(ctx).map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "graph imputation source replay" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })
    };
    if rebuilt.schema() != source.schema() || digest(&rebuilt)? != digest(source)? {
        return Err(invalid(
            "source schema or missing-cell representation cannot be replayed exactly",
        ));
    }
    Ok(columns)
}

impl CompiledGraphImputation {
    /// Export an already executed point only from this exact retained source and program.
    ///
    /// # Errors
    /// Refuses changed identities and resource limits.
    pub fn export(
        &self,
        point: &GraphImputationPoint,
        ctx: &ExecutionContext,
    ) -> Result<GraphImputationArtifact, CausalError> {
        if point.program != *self.lowered.identity().as_bytes()
            || point.source != self.source_digest
        {
            return Err(invalid("point belongs to another retained program or source"));
        }
        let quote = (self.source_bytes.len() as u64)
            .checked_mul(2)
            .and_then(|n| n.checked_add(8192))
            .ok_or_else(|| invalid("artifact size overflow"))?;
        let lease = ctx.acquire_admission(0, quote).map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "graph imputation export" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })?;
        let expected = GraphImputationExpectation {
            program: point.program,
            source: point.source,
            receipt: point.receipt,
        };
        Ok(GraphImputationArtifact {
            bytes: Arc::clone(&self.source_bytes),
            expected,
            _lease: lease,
        })
    }
}

/// Replay an artifact's original source and the compiler's retained child operations. A
/// caller's expected identity must be retained separately; no deserialized output is accepted
/// as proof.
///
/// # Errors
/// Refuses invalid or changed inputs, operations that differ from their graphs, replay
/// mismatches, and resource or execution failures.
pub fn consume(
    bytes: &[u8],
    expected: GraphImputationExpectation,
    ctx: &ExecutionContext,
) -> Result<GraphImputationPoint, GraphImputationError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: "graph imputation consume" }.into());
    }
    let point = GraphImputationPoint::replay(bytes, ctx)?;
    if (point.source, point.program, point.receipt)
        != (expected.source, expected.program, expected.receipt)
    {
        return Err(invalid("artifact does not match separately retained identity").into());
    }
    Ok(point)
}
