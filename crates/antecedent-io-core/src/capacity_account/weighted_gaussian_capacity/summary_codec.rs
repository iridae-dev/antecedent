use super::{
    CapacityControl, CeilingCharges, FitDiagnostics, GaussianCrossProducts, LeverageSummary,
    OutputRetention, ResidualCheck, SPILL_BACKENDS, SpillFault, WeightedCapacityError,
    WeightedCapacityResult, decode_f64, spill_payload_bytes, streamed_choice, zeros,
};
use crate::byte_cursor::ByteCursor;

fn control_code(control: CapacityControl) -> u8 {
    match control {
        CapacityControl::Glm => 0,
        CapacityControl::GaussianBayes => 1,
    }
}

fn put_f64s(out: &mut Vec<u8>, values: &[f64]) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// Canonical little-endian payload: control, `ncols`, the matrices and vectors,
/// the scalar summaries, the identity, the fit diagnostics, the charges, and
/// the counters, in that order.
pub(super) fn encode_summary(
    result: &WeightedCapacityResult,
    out: &mut Vec<u8>,
) -> Result<(), WeightedCapacityError> {
    let p = result.coefficients.len();
    let gram = p.checked_mul(p).ok_or(WeightedCapacityError::Overflow)?;
    let shaped = p > 0
        && result.robust_covariance.len() == gram
        && result.cross_products.xtwx.len() == gram
        && result.cross_products.xtwy.len() == p;
    if !shaped {
        return Err(WeightedCapacityError::Spill(SpillFault::PayloadLayout));
    }
    let backend = SPILL_BACKENDS
        .iter()
        .position(|name| *name == result.fit.backend)
        .and_then(|index| u8::try_from(index).ok())
        .ok_or(WeightedCapacityError::Spill(SpillFault::UnknownBackend))?;
    let ncols = u32::try_from(p).map_err(|_| WeightedCapacityError::Overflow)?;
    let rank = u64::try_from(result.fit.rank).map_err(|_| WeightedCapacityError::Overflow)?;
    out.push(control_code(result.control));
    out.extend_from_slice(&ncols.to_le_bytes());
    put_f64s(out, &result.coefficients);
    put_f64s(out, &result.robust_covariance);
    put_f64s(out, &result.cross_products.xtwx);
    put_f64s(out, &result.cross_products.xtwy);
    put_f64s(out, &[result.cross_products.n_eff, result.cross_products.ytwy]);
    put_f64s(out, &[result.leverage.max, result.leverage.sum]);
    put_u64(out, result.leverage.saturated_rows);
    put_f64s(out, &[result.residual.weighted_rss, result.residual.max_abs_residual]);
    put_u64(out, result.residual.positive_weight_rows);
    out.extend_from_slice(&result.content_identity);
    put_u64(out, rank);
    out.push(u8::from(result.fit.rcond.is_some()));
    put_f64s(out, &[result.fit.rcond.unwrap_or(0.0)]);
    out.push(backend);
    out.extend_from_slice(&result.fit.grow_count.to_le_bytes());
    let charges = &result.charges;
    for word in [
        charges.input_ownership_bytes,
        charges.content_identity_bytes,
        charges.decoder_bytes,
        charges.spill_bytes,
        charges.output_bytes,
        charges.total_bytes,
        result.batches,
        result.source_passes,
    ] {
        put_u64(out, word);
    }
    Ok(())
}

/// Cursor over a spill payload. Every read that runs past the end is a layout fault.
type PayloadReader<'a> = ByteCursor<'a, WeightedCapacityError>;

fn payload_fault() -> WeightedCapacityError {
    WeightedCapacityError::Spill(SpillFault::PayloadLayout)
}

/// The next `len` little-endian `f64` values of `reader`.
fn read_f64s(
    reader: &mut PayloadReader<'_>,
    len: usize,
    what: &'static str,
) -> Result<Vec<f64>, WeightedCapacityError> {
    let raw = reader.take(len.checked_mul(8).ok_or(WeightedCapacityError::Overflow)?)?;
    let mut values = zeros(len, what)?;
    for (slot, chunk) in values.iter_mut().zip(raw.chunks_exact(8)) {
        *slot = decode_f64(chunk)?;
    }
    Ok(values)
}

pub(super) fn decode_summary(
    payload: &[u8],
) -> Result<WeightedCapacityResult, WeightedCapacityError> {
    let layout = || WeightedCapacityError::Spill(SpillFault::PayloadLayout);
    let mut reader = PayloadReader::new(payload, payload_fault);
    let control = match reader.u8()? {
        0 => CapacityControl::Glm,
        1 => CapacityControl::GaussianBayes,
        _ => return Err(layout()),
    };
    let p = usize::try_from(reader.u32()?).map_err(|_| WeightedCapacityError::Overflow)?;
    if p == 0 || u64::try_from(payload.len()).ok() != Some(spill_payload_bytes(p)?) {
        return Err(layout());
    }
    let gram = p * p;
    let coefficients = read_f64s(&mut reader, p, "spilled coefficients")?;
    let robust_covariance = read_f64s(&mut reader, gram, "spilled covariance")?;
    let xtwx = read_f64s(&mut reader, gram, "spilled X'WX")?;
    let xtwy = read_f64s(&mut reader, p, "spilled X'Wy")?;
    let (n_eff, ytwy) = (reader.f64()?, reader.f64()?);
    let leverage =
        LeverageSummary { max: reader.f64()?, sum: reader.f64()?, saturated_rows: reader.u64()? };
    let residual = ResidualCheck {
        weighted_rss: reader.f64()?,
        max_abs_residual: reader.f64()?,
        positive_weight_rows: reader.u64()?,
    };
    let content_identity = reader.array::<32>()?;
    let rank = usize::try_from(reader.u64()?).map_err(|_| WeightedCapacityError::Overflow)?;
    let has_rcond = match reader.u8()? {
        0 => false,
        1 => true,
        _ => return Err(layout()),
    };
    let rcond = reader.f64()?;
    let backend = SPILL_BACKENDS
        .get(usize::from(reader.u8()?))
        .copied()
        .ok_or(WeightedCapacityError::Spill(SpillFault::UnknownBackend))?;
    let grow_count = reader.u32()?;
    let charges = CeilingCharges {
        input_ownership_bytes: reader.u64()?,
        content_identity_bytes: reader.u64()?,
        decoder_bytes: reader.u64()?,
        spill_bytes: reader.u64()?,
        output_bytes: reader.u64()?,
        total_bytes: reader.u64()?,
    };
    let batches = reader.u64()?;
    let source_passes = reader.u64()?;
    reader.finish(layout())?;
    Ok(WeightedCapacityResult {
        choice: Box::new(streamed_choice(charges.total_bytes)?),
        control,
        retention: OutputRetention::DiskBacked,
        coefficients,
        robust_covariance,
        leverage,
        residual,
        cross_products: GaussianCrossProducts { n_eff, xtwx, xtwy, ytwy },
        content_identity,
        fit: FitDiagnostics::new(rank, has_rcond.then_some(rcond), backend, grow_count),
        charges,
        batches,
        source_passes,
    })
}
