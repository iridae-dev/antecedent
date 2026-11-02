//! Bounded original-source wire for the draft graph-imputation composite.
//!
//! Decoding never grants authority: consumption reconstructs the original native source,
//! recompiles every graph atom and re-executes the composite, comparing a separately held
//! expected program/source/result identity. No calibrated interval is issued here.
#[allow(
    clippy::wildcard_imports,
    reason = "the wire module shares the private compiled candidate types"
)]
use super::*;
use antecedent_data::OwnedColumn;

const MAGIC: &[u8] = b"ANTECEDENT-GRAPH-IMPUTATION\x02";
const MAX_WIRE_BYTES: usize = 512 * 1024;
const MAX_NAME_BYTES: usize = 256;

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
    /// Encoded native source and receipt.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// Identity to retain independently of encoded bytes.
    pub fn expectation(&self) -> GraphImputationExpectation {
        self.expected
    }
}

type RawColumns = Vec<(Arc<str>, Vec<f64>)>;

fn raw_columns(source: &TabularData) -> Result<RawColumns, CausalError> {
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
) -> Result<(), CausalError> {
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
    Ok(())
}

fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

impl CompiledGraphImputation {
    /// Export an already executed point only from this exact retained source and program.
    ///
    /// # Errors
    /// Refuses changed identities, unsupported source representations, and resource limits.
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
        let quote = (self.source.row_count() as u64)
            .checked_mul(self.source.schema().variables().len() as u64)
            .and_then(|n| n.checked_mul(32))
            .and_then(|n| n.checked_add(8192))
            .ok_or_else(|| invalid("artifact size overflow"))?;
        let lease = ctx.acquire_admission(0, quote).map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "graph imputation export" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })?;
        canonical_source(&self.source, ctx)?;
        let columns = raw_columns(&self.source)?;
        let capacity =
            usize::try_from(quote).map_err(|_| invalid("artifact size exceeds platform"))?;
        let mut bytes = Vec::with_capacity(capacity);
        bytes.extend_from_slice(MAGIC);
        bytes.push(u8::try_from(columns.len()).map_err(|_| invalid("too many columns"))?);
        put_u32(
            &mut bytes,
            u32::try_from(self.source.row_count()).map_err(|_| invalid("too many rows"))?,
        );
        put_u32(&mut bytes, self.spec.treatment.raw());
        put_u32(&mut bytes, self.spec.outcome.raw());
        match self.spec.support {
            GraphImputationSupport::TreatmentRootDirectEffect => bytes.push(1),
        }
        put_u32(&mut bytes, self.spec.imputation.imputations);
        put_u32(&mut bytes, self.spec.imputation.iterations);
        put_u64(&mut bytes, self.spec.imputation.seed);
        put_u32(&mut bytes, self.spec.posterior_draws);
        put_u64(&mut bytes, self.spec.prior_scale.to_bits());
        put_u64(&mut bytes, self.spec.execution_seed);
        bytes.extend_from_slice(&self.source_digest);
        bytes.extend_from_slice(self.lowered.identity().as_bytes());
        bytes.extend_from_slice(&point.receipt);
        put_u64(&mut bytes, point.mean.to_bits());
        put_u64(&mut bytes, point.diagnostic_sd.to_bits());
        for ((name, values), &domain) in columns.iter().zip(self.spec.domains.iter()) {
            bytes.push(domain);
            put_u32(
                &mut bytes,
                u32::try_from(name.len()).map_err(|_| invalid("column name too long"))?,
            );
            bytes.extend_from_slice(name.as_bytes());
            for value in values {
                put_u64(&mut bytes, value.to_bits());
            }
        }
        if bytes.len() > MAX_WIRE_BYTES {
            return Err(invalid("artifact exceeds bounded size"));
        }
        let expected = GraphImputationExpectation {
            program: point.program,
            source: point.source,
            receipt: point.receipt,
        };
        Ok(GraphImputationArtifact { bytes: Arc::from(bytes), expected, _lease: lease })
    }
}

#[cfg(test)]
struct Reader<'a> {
    remaining: &'a [u8],
}
#[cfg(test)]
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], CausalError> {
        if n > self.remaining.len() {
            return Err(invalid("truncated artifact"));
        }
        let (prefix, rest) = self.remaining.split_at(n);
        self.remaining = rest;
        Ok(prefix)
    }
    fn byte(&mut self) -> Result<u8, CausalError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, CausalError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().map_err(|_| invalid("u32"))?))
    }
    fn u64(&mut self) -> Result<u64, CausalError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().map_err(|_| invalid("u64"))?))
    }
    fn digest(&mut self) -> Result<[u8; 32], CausalError> {
        self.take(32)?.try_into().map_err(|_| invalid("digest"))
    }
}

/// Consume only by reconstructing source and recompiling/replaying all native work. A caller's
/// expected identity must be retained separately; no deserialized output is accepted as proof.
///
/// # Errors
/// Refuses invalid or changed inputs, replay mismatches, and resource or execution failures.
pub fn consume(
    _bytes: &[u8],
    _expected: GraphImputationExpectation,
    ctx: &ExecutionContext,
) -> Result<GraphImputationPoint, GraphImputationError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: "graph imputation consume" }.into());
    }
    Err(GraphImputationError::PointUnavailable)
}

// No public artifact becomes authority before the source-bound candidate is licensed.
#[cfg(test)]
#[allow(
    clippy::too_many_lines,
    reason = "artifact reconstruction validates the complete source and identities"
)]
pub(super) fn consume_candidate(
    bytes: &[u8],
    expected: GraphImputationExpectation,
    ctx: &ExecutionContext,
) -> Result<GraphImputationPoint, GraphImputationError> {
    if bytes.len() > MAX_WIRE_BYTES || !bytes.starts_with(MAGIC) {
        return Err(invalid("invalid or oversized graph-imputation artifact").into());
    }
    let quote = (bytes.len() as u64)
        .checked_mul(8)
        .and_then(|v| v.checked_add(4096))
        .ok_or_else(|| invalid("artifact admission overflow"))?;
    let _lease = ctx.acquire_admission(0, quote).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => {
            CausalError::Cancelled { stage: "graph imputation consume" }
        }
        other => CausalError::Resource { message: other.to_string() },
    })?;
    let mut reader = Reader { remaining: &bytes[MAGIC.len()..] };
    let columns = usize::from(reader.byte()?);
    let rows = reader.u32()? as usize;
    if !(2..=3).contains(&columns) || !(4..=10_000).contains(&rows) {
        return Err(invalid("artifact shape is outside the bounded cell").into());
    }
    let treatment = VariableId::from_raw(reader.u32()?);
    let outcome = VariableId::from_raw(reader.u32()?);
    let support = match reader.byte()? {
        1 => GraphImputationSupport::TreatmentRootDirectEffect,
        _ => return Err(invalid("unknown graph support").into()),
    };
    let imputations = reader.u32()?;
    let iterations = reader.u32()?;
    let imputation_seed = reader.u64()?;
    let posterior_draws = reader.u32()?;
    let prior_scale = f64::from_bits(reader.u64()?);
    let execution_seed = reader.u64()?;
    if !(2..=64).contains(&imputations)
        || !(1..=64).contains(&iterations)
        || !(100..=4096).contains(&posterior_draws)
        || !prior_scale.is_finite()
        || prior_scale <= 0.0
    {
        return Err(invalid("artifact configuration is outside the bounded cell").into());
    }
    let stored_source = reader.digest()?;
    let stored_program = reader.digest()?;
    let stored_receipt = reader.digest()?;
    let stored_mean = reader.u64()?;
    let stored_sd = reader.u64()?;
    if (stored_source, stored_program, stored_receipt)
        != (expected.source, expected.program, expected.receipt)
    {
        return Err(invalid("artifact does not match separately retained identity").into());
    }
    let mut names = Vec::with_capacity(columns);
    let mut values = Vec::with_capacity(columns);
    let mut domains = Vec::with_capacity(columns);
    for _ in 0..columns {
        let domain = reader.byte()?;
        let name_len = reader.u32()? as usize;
        if domain > 1 || name_len == 0 || name_len > MAX_NAME_BYTES {
            return Err(invalid("artifact variable domain or name is invalid").into());
        }
        let name = std::str::from_utf8(reader.take(name_len)?)
            .map_err(|_| invalid("artifact variable name is not UTF-8"))?
            .to_owned();
        // Refuse impossible advertised shape before allocating its values.
        if rows.checked_mul(8).is_none_or(|n| n > reader.remaining.len()) {
            return Err(invalid("artifact column is truncated").into());
        }
        let mut column = Vec::with_capacity(rows);
        for _ in 0..rows {
            column.push(f64::from_bits(reader.u64()?));
        }
        domains.push(domain);
        names.push(name);
        values.push(column);
    }
    if !reader.remaining.is_empty() {
        return Err(invalid("artifact has trailing bytes").into());
    }
    let source = TabularData::from_f64_columns(
        names.iter().zip(&values).map(|(name, column)| (name.as_str(), column.as_slice())),
    )
    .map_err(CausalError::from)?;
    let spec = GraphImputationSpec {
        treatment,
        outcome,
        support,
        imputation: ImputationSpec::new()
            .with_imputations(imputations)
            .with_iterations(iterations)
            .with_seed(imputation_seed),
        domains: Arc::from(domains),
        posterior_draws,
        prior_scale,
        execution_seed,
    };
    let plan = CompiledGraphImputation::compile(source, spec, ctx)?;
    if plan.source_digest != expected.source
        || *plan.lowered.identity().as_bytes() != expected.program
    {
        return Err(invalid("reconstructed source or compiled program changed").into());
    }
    let result = plan.execute_candidate(ctx)?;
    if result.receipt != expected.receipt
        || result.mean.to_bits() != stored_mean
        || result.diagnostic_sd.to_bits() != stored_sd
    {
        return Err(invalid("independent graph-imputation replay differs").into());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reader_rejects_truncation_and_preserves_bits() {
        let bits = 0x7ff8_0000_0000_0042u64;
        let encoded = bits.to_le_bytes();
        let mut reader = Reader { remaining: &encoded };
        assert_eq!(reader.u64().unwrap(), bits);
        assert!(reader.byte().is_err());
        for n in 0..8 {
            assert!(Reader { remaining: &encoded[..n] }.u64().is_err());
        }
    }

    #[test]
    fn reader_never_accepts_incomplete_digest() {
        for n in 0..32 {
            assert!(Reader { remaining: &[0; 32][..n] }.digest().is_err());
        }
    }
}
