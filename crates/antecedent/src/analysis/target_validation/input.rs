//! Identity and equality of the executed refutation input.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, DataInput, ExecutionContext, PreparedTargetData, StudyResult, TableView,
    validate_runtime_refutation_proof,
};

// Compare caller-supplied observations against the immutable input actually
// retained by this execution. Content scanning is chunked, cancellable and
// admitted; schema strings are compared in bounded chunks.
fn same_bytes(a: &[u8], b: &[u8], ctx: &ExecutionContext) -> Result<bool, CausalError> {
    if a.len() != b.len() {
        return Ok(false);
    }
    for (a, b) in a.chunks(4096).zip(b.chunks(4096)) {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "refutation input identity" });
        }
        if a != b {
            return Ok(false);
        }
    }
    Ok(true)
}

fn same_schema(
    a: &antecedent_core::CausalSchema,
    b: &antecedent_core::CausalSchema,
    ctx: &ExecutionContext,
) -> Result<bool, CausalError> {
    if a.len() != b.len() {
        return Ok(false);
    }
    for (x, y) in a.variables().iter().zip(b.variables()) {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "refutation input identity" });
        }
        if x.id != y.id
            || x.value_type != y.value_type
            || x.role_hints != y.role_hints
            || x.category_domain != y.category_domain
            || x.measurement.noisy != y.measurement.noisy
            || !same_bytes(x.name.as_bytes(), y.name.as_bytes(), ctx)?
        {
            return Ok(false);
        }
        for (left, right) in
            [(&x.unit, &y.unit), (&x.measurement.description, &y.measurement.description)]
        {
            match (left, right) {
                (None, None) => {}
                (Some(left), Some(right))
                    if same_bytes(left.as_bytes(), right.as_bytes(), ctx)? => {}
                _ => return Ok(false),
            }
        }
    }
    Ok(true)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExecutedInputKind {
    Tabular,
    Temporal,
    Event,
    Panel,
    MultiEnv,
}

#[derive(Clone, Debug)]
pub(crate) struct ExecutedInputPart {
    schema: antecedent_core::CausalSchema,
    digest: [u8; 32],
    time: Option<antecedent_data::TimeIndex>,
    unit_id: Option<u32>,
}

#[derive(Debug)]
struct ExecutedInputParts {
    parts: Vec<ExecutedInputPart>,
    _lease: antecedent_core::execution::AdmissionLease,
}

#[derive(Clone, Debug)]
enum ExecutedInputStorage {
    Single(ExecutedInputPart),
    Segmented(Arc<ExecutedInputParts>),
}

#[derive(Clone, Debug)]
pub(crate) struct ExecutedInputIdentity {
    kind: ExecutedInputKind,
    storage: ExecutedInputStorage,
}
impl ExecutedInputIdentity {
    fn parts(&self) -> &[ExecutedInputPart] {
        match &self.storage {
            ExecutedInputStorage::Single(part) => std::slice::from_ref(part),
            ExecutedInputStorage::Segmented(parts) => &parts.parts,
        }
    }
}

fn input_digest(
    storage: &antecedent_data::OwnedColumnarStorage,
    ctx: &ExecutionContext,
) -> Result<[u8; 32], CausalError> {
    storage.content_digest_with_context(ctx).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => {
            CausalError::Cancelled { stage: "refutation input identity" }
        }
        other => CausalError::Resource { message: other.to_string() },
    })
}

#[allow(
    clippy::too_many_lines,
    reason = "the target-specific branches preserve a single checked operation"
)]
pub(crate) fn executed_input_identity(
    data: &DataInput,
    ctx: &ExecutionContext,
) -> Result<ExecutedInputIdentity, CausalError> {
    // A run cancelled before it was finalised (a bootstrap stopped part-way) still returns the
    // partial result it reached, and that result is bound to its input. Only a cancellation
    // arriving while the input is being identified stops the identification.
    let finalised;
    let ctx = if ctx.cancellation.is_cancelled() {
        let mut fresh = ctx.clone();
        fresh.cancellation = antecedent_core::CancellationToken::new();
        finalised = fresh;
        &finalised
    } else {
        ctx
    };
    let part = |storage: &antecedent_data::OwnedColumnarStorage,
                time: Option<antecedent_data::TimeIndex>,
                unit_id: Option<u32>|
     -> Result<ExecutedInputPart, CausalError> {
        Ok(ExecutedInputPart {
            schema: storage.schema().clone(),
            digest: input_digest(storage, ctx)?,
            time,
            unit_id,
        })
    };
    let (kind, storage) = match data {
        DataInput::Tabular(table) => (
            ExecutedInputKind::Tabular,
            ExecutedInputStorage::Single(part(table.storage(), None, None)?),
        ),
        DataInput::Temporal(series) => (
            ExecutedInputKind::Temporal,
            ExecutedInputStorage::Single(part(
                series.storage(),
                Some(series.time_index().clone()),
                None,
            )?),
        ),
        DataInput::Event(series) => (
            ExecutedInputKind::Event,
            ExecutedInputStorage::Single(part(
                series.storage(),
                Some(series.time_index().clone()),
                None,
            )?),
        ),
        DataInput::Panel(panel) => {
            let count = panel.units().len();
            let bytes = u64::try_from(count)
                .ok()
                .and_then(|n| n.checked_mul(256))
                .and_then(|n| n.checked_add(512))
                .ok_or_else(|| CausalError::Resource {
                    message: "input identity size overflow".into(),
                })?;
            let lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
                antecedent_core::execution::AdmissionError::Cancelled => {
                    CausalError::Cancelled { stage: "refutation input identity" }
                }
                other => CausalError::Resource { message: other.to_string() },
            })?;
            let mut parts = Vec::new();
            parts
                .try_reserve_exact(count)
                .map_err(|error| CausalError::Resource { message: error.to_string() })?;
            for unit in panel.units() {
                parts.push(part(
                    unit.series.storage(),
                    Some(unit.series.time_index().clone()),
                    Some(unit.unit_id),
                )?);
            }
            (
                ExecutedInputKind::Panel,
                ExecutedInputStorage::Segmented(Arc::new(ExecutedInputParts {
                    parts,
                    _lease: lease,
                })),
            )
        }
        DataInput::MultiEnv(multi) => {
            let count = multi.environments().len();
            let bytes = u64::try_from(count)
                .ok()
                .and_then(|n| n.checked_mul(256))
                .and_then(|n| n.checked_add(512))
                .ok_or_else(|| CausalError::Resource {
                    message: "input identity size overflow".into(),
                })?;
            let lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
                antecedent_core::execution::AdmissionError::Cancelled => {
                    CausalError::Cancelled { stage: "refutation input identity" }
                }
                other => CausalError::Resource { message: other.to_string() },
            })?;
            let mut parts = Vec::new();
            parts
                .try_reserve_exact(count)
                .map_err(|error| CausalError::Resource { message: error.to_string() })?;
            for series in multi.environments() {
                parts.push(part(series.storage(), Some(series.time_index().clone()), None)?);
            }
            (
                ExecutedInputKind::MultiEnv,
                ExecutedInputStorage::Segmented(Arc::new(ExecutedInputParts {
                    parts,
                    _lease: lease,
                })),
            )
        }
    };
    Ok(ExecutedInputIdentity { kind, storage })
}

fn same_part(
    part: &ExecutedInputPart,
    storage: &antecedent_data::OwnedColumnarStorage,
    time: Option<&antecedent_data::TimeIndex>,
    unit_id: Option<u32>,
    ctx: &ExecutionContext,
) -> Result<bool, CausalError> {
    Ok(part.unit_id == unit_id
        && part.time.as_ref() == time
        && same_schema(&part.schema, storage.schema(), ctx)?
        && part.digest == input_digest(storage, ctx)?)
}

fn same_executed_input(
    original: &ExecutedInputIdentity,
    supplied: PreparedTargetData<'_>,
    ctx: &ExecutionContext,
) -> Result<bool, CausalError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: "refutation input identity" });
    }
    match (original.kind, supplied) {
        (ExecutedInputKind::Tabular, PreparedTargetData::Tabular(table)) => {
            Ok(original.parts().len() == 1
                && same_part(&original.parts()[0], table.storage(), None, None, ctx)?)
        }
        (
            ExecutedInputKind::Temporal | ExecutedInputKind::Event,
            PreparedTargetData::Series(series),
        ) => Ok(original.parts().len() == 1
            && same_part(
                &original.parts()[0],
                series.storage(),
                Some(series.time_index()),
                None,
                ctx,
            )?),
        (ExecutedInputKind::Panel, PreparedTargetData::Panel(panel)) => {
            if original.parts().len() != panel.units().len() {
                return Ok(false);
            }
            for (part, unit) in original.parts().iter().zip(panel.units()) {
                if !same_part(
                    part,
                    unit.series.storage(),
                    Some(unit.series.time_index()),
                    Some(unit.unit_id),
                    ctx,
                )? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        (ExecutedInputKind::MultiEnv, PreparedTargetData::MultiEnv(multi)) => {
            if original.parts().len() != multi.environments().len() {
                return Ok(false);
            }
            for (part, series) in original.parts().iter().zip(multi.environments()) {
                if !same_part(part, series.storage(), Some(series.time_index()), None, ctx)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// Borrow-scoped capability issued only after comparing actual execution input.
pub(in crate::analysis) struct CheckedRefutationInput<'a> {
    pub(super) owner: Arc<()>,
    _result: std::marker::PhantomData<&'a StudyResult>,
}

pub(in crate::analysis) fn checked_refutation_input<'a>(
    result: &'a StudyResult,
    data: PreparedTargetData<'_>,
    ctx: &ExecutionContext,
) -> Result<CheckedRefutationInput<'a>, CausalError> {
    validate_runtime_refutation_proof(result)?;
    let owner = match &result.authority {
        crate::result::RuntimeRefutationAuthority::Sealed { owner, .. } => Arc::clone(owner),
        _ => {
            return Err(CausalError::Conflict {
                what: "result",
                detail: "target refutation requires an unmodified engine-produced result",
            });
        }
    };
    let actual = result.executed_input.as_ref().ok_or(CausalError::Conflict {
        what: "data",
        detail: "the executed observations were not retained for target refutation",
    })?;
    if !same_executed_input(actual, data, ctx)? {
        return Err(CausalError::Conflict {
            what: "data",
            detail: "the refit observations differ from those that produced the result",
        });
    }
    Ok(CheckedRefutationInput { owner, _result: std::marker::PhantomData })
}
