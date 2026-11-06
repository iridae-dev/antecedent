//! Common admission and encoding helpers for frozen panel artifacts.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_core::execution::AdmissionLease;

use crate::joint_panel_artifact::PanelArtifactError;

pub(crate) fn invalid(error: impl std::fmt::Display) -> PanelArtifactError {
    PanelArtifactError::Invalid(error.to_string())
}
pub(crate) fn cancelled(ctx: &ExecutionContext) -> Result<(), PanelArtifactError> {
    if ctx.cancellation.is_cancelled() { Err(PanelArtifactError::Cancelled) } else { Ok(()) }
}
pub(crate) fn admit(
    ctx: &ExecutionContext,
    bytes: u64,
) -> Result<AdmissionLease, PanelArtifactError> {
    let limit = ctx.memory_limit_bytes();
    if limit.is_some_and(|limit| bytes > limit) {
        return Err(PanelArtifactError::Resource("configured memory limit".into()));
    }
    ctx.acquire_admission(0, bytes).map_err(|error| PanelArtifactError::Resource(error.to_string()))
}
pub(crate) struct Counter(pub(crate) usize);
impl std::io::Write for Counter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len())
            .ok_or_else(|| std::io::Error::other("byte count overflow"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::execution::AdmissionPool;

    #[test]
    fn panel_admission_holds_and_releases_its_quote() {
        let pool = AdmissionPool::fixed(1, 32);
        let ctx = ExecutionContext::for_tests(0).with_admission(pool.clone());
        let lease = admit(&ctx, 24).unwrap();
        assert_eq!(pool.available().bytes(), 8);
        assert!(matches!(admit(&ctx, 9), Err(PanelArtifactError::Resource(_))));
        drop(lease);
        assert_eq!(pool.available().bytes(), 32);
        ctx.cancellation.cancel();
        assert!(matches!(cancelled(&ctx), Err(PanelArtifactError::Cancelled)));
    }
}
