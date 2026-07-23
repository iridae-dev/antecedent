//! Fluent builder for change attribution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    AllocationMethod, AttributionComponents, ChangeAttributionQuery, ExecutionContext,
    PopulationSelector, ShapleyConfig, VariableId,
};
use antecedent_data::TabularData;
use antecedent_model::CompiledCausalModel;

use crate::distribution_change::{
    DifferenceMeasure, DistributionChangeOptions, distribution_change,
};
use crate::error::AttributionError;
use crate::result::ChangeAttributionResult;
use crate::robust::{RobustChangeOptions, distribution_change_robust};
use crate::structure_change::{StructureChangeOptions, structure_change};

/// Builder for [`ChangeAttribution`].
#[derive(Clone, Debug)]
pub struct ChangeAttribution {
    outcome: Option<VariableId>,
    baseline: Option<PopulationSelector>,
    comparison: Option<PopulationSelector>,
    components: AttributionComponents,
    allocation: AllocationMethod,
    robust: bool,
    measure: DifferenceMeasure,
    n_samples: usize,
    seed: u64,
}

impl Default for ChangeAttribution {
    fn default() -> Self {
        Self::new()
    }
}

impl ChangeAttribution {
    /// Start a change-attribution builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            outcome: None,
            baseline: None,
            comparison: None,
            components: AttributionComponents::Mechanisms,
            allocation: AllocationMethod::Shapley {
                approximation: ShapleyConfig::monte_carlo(2_000),
            },
            robust: false,
            measure: DifferenceMeasure::MeanDiff,
            n_samples: 2_000,
            seed: 0,
        }
    }

    /// Set outcome variable.
    #[must_use]
    pub const fn outcome(mut self, outcome: VariableId) -> Self {
        self.outcome = Some(outcome);
        self
    }

    /// Set baseline population.
    #[must_use]
    pub fn baseline(mut self, baseline: PopulationSelector) -> Self {
        self.baseline = Some(baseline);
        self
    }

    /// Set comparison population.
    #[must_use]
    pub fn comparison(mut self, comparison: PopulationSelector) -> Self {
        self.comparison = Some(comparison);
        self
    }

    /// Set attribution components.
    #[must_use]
    pub const fn components(mut self, components: AttributionComponents) -> Self {
        self.components = components;
        self
    }

    /// Set allocation method.
    #[must_use]
    pub fn allocation(mut self, allocation: AllocationMethod) -> Self {
        self.allocation = allocation;
        self
    }

    /// Use the robust (regression) estimator.
    #[must_use]
    pub const fn robust(mut self, robust: bool) -> Self {
        self.robust = robust;
        self
    }

    /// Difference measure / sample budget for the standard estimator.
    #[must_use]
    pub const fn with_sampling(
        mut self,
        measure: DifferenceMeasure,
        n_samples: usize,
        seed: u64,
    ) -> Self {
        self.measure = measure;
        self.n_samples = n_samples;
        self.seed = seed;
        self
    }

    /// Run against a compiled graph model and tabular data.
    ///
    /// Mechanism / robust paths only. For structure attribution use
    /// [`Self::run_structure`] with dual graphs.
    ///
    /// # Errors
    ///
    /// Missing fields, `AttributionComponents::Structure`, or attribution failures.
    pub fn run(
        self,
        model: &CompiledCausalModel,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<ChangeAttributionResult, AttributionError> {
        if matches!(self.components, AttributionComponents::Structure | AttributionComponents::All)
        {
            return Err(AttributionError::unsupported(
                "ChangeAttribution::run does not support Structure/All; use run_structure for \
                 Structure, or InputsAndMechanisms for joint input+mechanism change",
            ));
        }
        let robust = self.robust;
        let opts = DistributionChangeOptions {
            measure: self.measure,
            n_samples: self.n_samples,
            seed: self.seed,
        };
        let query = self.into_query()?;
        if robust {
            distribution_change_robust(model, data, &query, &RobustChangeOptions::default(), ctx)
        } else {
            distribution_change(model, data, &query, &opts, ctx)
        }
    }

    /// Run structure-change attribution over baseline vs comparison DAGs.
    ///
    /// # Errors
    ///
    /// Missing fields, non-Structure components, or attribution failures.
    pub fn run_structure(
        self,
        baseline_model: &CompiledCausalModel,
        comparison_model: &CompiledCausalModel,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<ChangeAttributionResult, AttributionError> {
        if !matches!(self.components, AttributionComponents::Structure) {
            return Err(AttributionError::unsupported(
                "ChangeAttribution::run_structure requires AttributionComponents::Structure",
            ));
        }
        if self.robust {
            return Err(AttributionError::unsupported(
                "robust estimator is not defined for structure-change attribution",
            ));
        }
        let opts = StructureChangeOptions {
            measure: self.measure,
            n_samples: self.n_samples,
            seed: self.seed,
        };
        let query = self.into_query()?;
        structure_change(baseline_model, comparison_model, data, &query, &opts, ctx)
    }

    fn into_query(self) -> Result<ChangeAttributionQuery, AttributionError> {
        let outcome = self
            .outcome
            .ok_or(AttributionError::invalid_input("ChangeAttribution missing outcome"))?;
        let baseline = self
            .baseline
            .ok_or(AttributionError::invalid_input("ChangeAttribution missing baseline"))?;
        let comparison = self
            .comparison
            .ok_or(AttributionError::invalid_input("ChangeAttribution missing comparison"))?;
        Ok(ChangeAttributionQuery {
            outcome,
            baseline,
            comparison,
            components: self.components,
            allocation: self.allocation,
            max_components: 64,
        })
    }
}
