//! Shared PCMCI-family builder helpers (SOLID/DRY).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Fluent builders shared by PCMCI / PCMCI+ / LPCMCI / J-PCMCI+.
///
/// Expects `self.engine: PcmciEngine` and `self.fdr: Option<FdrAdjustment>`.
macro_rules! pcmci_family_builders {
    () => {
        /// Configure discovery constraints.
        #[must_use]
        pub fn with_constraints(mut self, constraints: crate::constraints::DiscoveryConstraints) -> Self {
            self.engine.constraints = constraints;
            self
        }

        /// Enable / disable BH FDR.
        #[must_use]
        pub fn with_fdr(mut self, fdr: bool) -> Self {
            self.fdr = fdr.then(causal_stats::FdrAdjustment::bh);
            self
        }

        /// Full FDR / FWER configuration.
        #[must_use]
        pub fn with_fdr_adjustment(
            mut self,
            fdr: Option<causal_stats::FdrAdjustment>,
        ) -> Self {
            self.fdr = fdr;
            self
        }

        /// Replace the CI test on the shared engine.
        #[must_use]
        pub fn with_ci(
            mut self,
            ci: std::sync::Arc<dyn causal_stats::ConditionalIndependence + Send + Sync>,
        ) -> Self {
            self.engine = self.engine.with_ci(ci);
            self
        }

        /// Borrow the shared PCMCI engine (constraints / CI).
        #[must_use]
        pub fn engine(&self) -> &crate::engine::PcmciEngine {
            &self.engine
        }
    };
}

pub(crate) use pcmci_family_builders;
