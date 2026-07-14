//! Sharp regression-discontinuity identification (DESIGN.md §2.1).
//!
//! Records design assumptions explicitly; does not search a causal graph.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, CausalQuery, VariableId,
};
use causal_expr::{CausalExprArena, IdentifiedEstimand};

use crate::error::IdentificationError;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentificationStatus,
};

/// Configuration for sharp RD identification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SharpRdConfig {
    /// Running variable.
    pub running_variable: VariableId,
    /// Cutoff threshold.
    pub cutoff: f64,
    /// Bandwidth around the cutoff.
    pub bandwidth: f64,
}

/// Identifier for sharp regression discontinuity designs.
///
/// Produces a nonparametrically identified estimand under explicit RD assumptions
/// (continuity of potential outcomes at the cutoff; deterministic treatment assignment).
#[derive(Clone, Debug)]
pub struct SharpRdIdentifier {
    /// Design configuration.
    pub config: SharpRdConfig,
}

impl SharpRdIdentifier {
    /// Construct.
    #[must_use]
    pub const fn new(config: SharpRdConfig) -> Self {
        Self { config }
    }

    /// Identify the average effect under sharp RD assumptions.
    ///
    /// # Errors
    ///
    /// Query is not an average-effect query, or design config is invalid.
    pub fn identify(
        &self,
        query: CausalQuery,
    ) -> Result<IdentificationResult, IdentificationError> {
        let CausalQuery::AverageEffect(ate) = &query else {
            return Err(IdentificationError::UnsupportedQuery {
                message: "sharp RD identifier requires an average-effect query",
            });
        };
        let ate = ate.clone();
        if !self.config.bandwidth.is_finite() || self.config.bandwidth <= 0.0 {
            return Err(IdentificationError::UnsupportedQuery {
                message: "sharp RD bandwidth must be finite and positive",
            });
        }
        if !self.config.cutoff.is_finite() {
            return Err(IdentificationError::UnsupportedQuery {
                message: "sharp RD cutoff must be finite",
            });
        }

        // Build the functional in a real arena (same convention as the
        // backdoor/frontdoor/IV identifiers): the sharp-RD estimand is the
        // do-contrast at the cutoff, expressed as a backdoor ATE with an
        // empty adjustment set.
        let (active, control) = match (&ate.active, &ate.control) {
            (
                causal_core::Intervention::Set { value: active, .. },
                causal_core::Intervention::Set { value: control, .. },
            ) => (active.clone(), control.clone()),
            _ => {
                return Err(IdentificationError::UnsupportedQuery {
                    message: "sharp RD ATE requires Set interventions",
                });
            }
        };
        let mut arena = CausalExprArena::new();
        let functional = arena.backdoor_ate(ate.treatment, ate.outcome, &[], active, control);
        let estimand = IdentifiedEstimand::backdoor("rd.sharp", Arc::from([]), functional);

        let mut assumptions = AssumptionSet::new();
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("rd.continuity"),
                description: Arc::from(
                    "potential-outcome means continuous in the running variable at the cutoff",
                ),
            },
            source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from("rd.sharp") },
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("rd.sharp_assignment"),
                description: Arc::from(
                    "treatment is a deterministic function of the running variable at the cutoff",
                ),
            },
            source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from("rd.sharp") },
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });

        let mut derivation = DerivationTrace::default();
        derivation.push(
            "rd.sharp",
            format!(
                "sharp RD at cutoff={} bandwidth={} on running variable {:?}",
                self.config.cutoff, self.config.bandwidth, self.config.running_variable
            ),
        );

        Ok(IdentificationResult {
            status: IdentificationStatus::NonparametricallyIdentified,
            query,
            estimands: vec![estimand],
            arena,
            derivation,
            required_assumptions: assumptions,
            diagnostics: Vec::new(),
            performance: IdentificationPerformanceRecord {
                candidates_examined: 1,
                sets_returned: 1,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use causal_core::{AverageEffectQuery, VariableId};

    use super::*;

    #[test]
    fn identifies_with_declared_assumptions() {
        let id = SharpRdIdentifier::new(SharpRdConfig {
            running_variable: VariableId::from_raw(2),
            cutoff: 0.0,
            bandwidth: 1.0,
        });
        let q = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let result = id.identify(q).unwrap();
        assert_eq!(result.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(result.estimands.len(), 1);
        assert_eq!(result.required_assumptions.len(), 2);
        // The functional must resolve to a real node in the result's arena.
        assert!(!result.arena.is_empty());
        let _ = result.arena.node(result.estimands[0].functional);
    }
}
