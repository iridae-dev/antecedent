//! Exact funded JSON encoding of borrowed native read models.
//!
//! Source ownership is supplied by the caller. The quote pass writes to a
//! counter, then one output buffer is admitted before allocation. It does not
//! create a JSON value tree or estimate capacity from an encoded-size multiplier.
use crate::IoError;
use antecedent_core::ExecutionContext;
use antecedent_core::execution::{AdmissionError, AdmissionLease};
use serde::Serialize;
use std::io::{self, Write};

/// Native encoded JSON whose actual output capacity remains admitted.
///
/// Share this owner with `Arc` when necessary; do not detach its buffer from
/// the lease. The source objects need not remain alive after serialization.
pub struct AdmittedJson {
    text: String,
    _output: AdmissionLease,
    _allocator_extra: Option<AdmissionLease>,
}
impl AdmittedJson {
    /// Borrow the encoded payload while its admission owner remains alive.
    #[must_use]
    pub fn as_str(&self) -> &str { &self.text }
    /// Borrow the exact encoded bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] { self.text.as_bytes() }
    /// Actual funded output capacity, without a guessed encoded-size multiplier.
    #[must_use]
    pub fn retained_bytes(&self) -> usize { self.text.capacity() }
}

fn cancelled(ctx: &ExecutionContext) -> Result<(), IoError> {
    if ctx.cancellation.is_cancelled() { Err(IoError::StreamCancelled) } else { Ok(()) }
}
fn admit(ctx: &ExecutionContext, bytes: usize) -> Result<AdmissionLease, IoError> {
    let bytes = u64::try_from(bytes).map_err(|_| IoError::StreamBudgetRefused {
        message: "native JSON admission exceeds u64 bytes".into(),
    })?;
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        AdmissionError::Cancelled => IoError::StreamCancelled,
        other => IoError::StreamBudgetRefused { message: other.to_string() },
    })
}
fn serialization(error: serde_json::Error, ctx: &ExecutionContext) -> IoError {
    if ctx.cancellation.is_requested() { IoError::StreamCancelled }
    else { IoError::Convert(format!("native JSON serialization failed: {error}")) }
}
struct Counter<'a> { ctx: &'a ExecutionContext, length: usize, overflow: bool }
impl Write for Counter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.ctx.cancellation.is_cancelled() { return Err(io::Error::other("native JSON serialization cancelled")); }
        let Some(length) = self.length.checked_add(bytes.len()) else {
            self.overflow = true;
            return Err(io::Error::other("native JSON byte count overflow"));
        };
        self.length = length;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}
struct Output<'a> { ctx: &'a ExecutionContext, bytes: &'a mut Vec<u8>, quoted: usize }
impl Write for Output<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.ctx.cancellation.is_cancelled() { return Err(io::Error::other("native JSON serialization cancelled")); }
        if bytes.len() > self.quoted.saturating_sub(self.bytes.len()) {
            return Err(io::Error::other("native JSON changed after its exact quote"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

/// Encode a stable borrowed native serializer under exact output admission.
///
/// Serializers must be pure read models: the counting and encoding passes must
/// describe the same retained data and must not execute scientific procedures.
/// This funds the new output, not pre-existing source storage or allocations
/// hidden by an arbitrary caller-provided `Serialize` implementation.
///
/// # Errors
/// Cancellation, insufficient admission, allocator failure, serialization failure
/// or a source serializer that changes its output length between the two passes.
pub fn serialize_admitted<T: Serialize + ?Sized>(
    source: &T, ctx: &ExecutionContext,
) -> Result<AdmittedJson, IoError> {
    cancelled(ctx)?;
    let mut quote = Counter { ctx, length: 0, overflow: false };
    if let Err(error) = serde_json::to_writer(&mut quote, source) {
        if quote.overflow { return Err(IoError::StreamBudgetRefused {
            message: "native JSON byte count exceeds usize".into(),
        }); }
        return Err(serialization(error, ctx));
    }
    cancelled(ctx)?;
    let output = admit(ctx, quote.length)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(quote.length).map_err(|error| IoError::StreamBudgetRefused {
        message: format!("cannot allocate {} admitted JSON bytes: {error}", quote.length),
    })?;
    let extra = bytes.capacity().checked_sub(quote.length).ok_or_else(|| {
        IoError::Convert("native JSON allocator returned insufficient capacity".into())
    })?;
    let allocator_extra = if extra == 0 { None } else { Some(admit(ctx, extra)?) };
    cancelled(ctx)?;
    serde_json::to_writer(&mut Output { ctx, bytes: &mut bytes, quoted: quote.length }, source)
        .map_err(|error| serialization(error, ctx))?;
    if bytes.len() != quote.length {
        return Err(IoError::Convert("native JSON changed after its exact quote".into()));
    }
    cancelled(ctx)?;
    let text = String::from_utf8(bytes).map_err(|error| IoError::Convert(error.to_string()))?;
    cancelled(ctx)?;
    Ok(AdmittedJson { text, _output: output, _allocator_extra: allocator_extra })
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::execution::AdmissionPool;
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    #[derive(Serialize)]
    struct Payload<'a> { text: &'a str, rows: &'a [u32] }

    #[test]
    fn exact_json_buffer_is_funded_before_encoding_and_until_last_owner_drops() {
        let payload = Payload { text: "original\\\"λ", rows: &[9,2,0] };
        let expected = serde_json::to_vec(&payload).unwrap();
        let budget = u64::try_from(expected.len()).unwrap();
        let pool = AdmissionPool::fixed(1, budget);
        let ctx = ExecutionContext::for_tests(17).with_admission(pool.clone());
        let owner = Arc::new(serialize_admitted(&payload, &ctx).unwrap());
        assert_eq!(owner.as_bytes(), expected);
        assert_eq!(u64::try_from(owner.retained_bytes()).unwrap(), budget);
        assert_eq!(pool.available().bytes(), 0);
        let retained = Arc::clone(&owner);
        drop(owner);
        ctx.cancellation.cancel();
        assert_eq!(pool.available().bytes(), 0, "cancelled live buffers remain funded");
        assert_eq!(retained.as_str(), std::str::from_utf8(&expected).unwrap());
        drop(retained);
        assert_eq!(pool.available().bytes(), budget);
        let short = AdmissionPool::fixed(1, budget - 1);
        let ctx = ExecutionContext::for_tests(17).with_admission(short.clone());
        assert!(matches!(serialize_admitted(&payload, &ctx), Err(IoError::StreamBudgetRefused { .. })));
        assert_eq!(short.available().bytes(), budget - 1);
    }

    #[test]
    fn cancellation_during_funded_output_releases_the_exact_lease() {
        struct CancelOnEncoding<'a> { ctx: &'a ExecutionContext, calls: AtomicUsize }
        impl Serialize for CancelOnEncoding<'_> {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                if self.calls.fetch_add(1,Ordering::SeqCst) == 1 {
                    assert_eq!(self.ctx.admission.available().bytes(), 0);
                    self.ctx.cancellation.cancel();
                }
                serializer.serialize_str("proof")
            }
        }
        let pool = AdmissionPool::fixed(1, 7);
        let ctx = ExecutionContext::for_tests(17).with_admission(pool.clone());
        let payload = CancelOnEncoding { ctx: &ctx, calls: AtomicUsize::new(0) };
        assert!(matches!(serialize_admitted(&payload,&ctx), Err(IoError::StreamCancelled)));
        assert_eq!(payload.calls.load(Ordering::SeqCst),2);
        assert_eq!(pool.available().bytes(),7);
    }
}
