//! The unlicensed Bayesian randomized-dose grid posterior and its portable artifact.
//!
//! The dose-grid row has no `Study` query (its frequentist row,
//! [`crate::analysis::dose_grid_functional`], is a standalone family too), so its Bayesian
//! counterpart is evaluated and sealed here on the same coordinate: the level, derivative or
//! contrast of a randomized-dose response curve, as the joint posterior over the grid under a
//! declared polynomial coefficient prior with the covariance across grid points retained. An
//! observational dose is refused (`dose_grid.graph_not_certified`). The survival and fixed
//! group-dose policy counterparts are `Study` routes (`bayesian.piecewise_survival`,
//! `bayesian.continuous_dose_policy`) and the smoothed dose transport posterior is
//! [`crate::PreparedSmoothedDose::posterior`].
//!
//! The row is explicitly unlicensed and its calibration is `unmeasured`; the result says so and
//! no support-matrix coordinate is claimed. [`BayesianPosteriorRow::evaluate`] runs and seals the
//! row; [`BayesianPosteriorRow::consume`] recomputes the posterior from the exported table and
//! refuses any resealed change of premise, table or value.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_io::bayesian_posterior_row_artifact::{
    DOSE_GRID_POSTERIOR_ESTIMATOR, DoseGridPosteriorWire, POSTERIOR_ROW_ARTIFACT_FEATURE,
    POSTERIOR_ROW_ARTIFACT_VERSION, POSTERIOR_ROW_CALIBRATION, POSTERIOR_ROW_UNCERTAINTY,
    PosteriorRowArtifactError, PosteriorRowArtifactWire, PosteriorRowConsumeLimits,
    PosteriorRowRequestWire, PosteriorRowResultWire, ResidualPriorWire,
};

use crate::error::CausalError;

/// A sealed or consumed Bayesian posterior row.
#[derive(Clone, Debug)]
pub struct BayesianPosteriorRow {
    artifact: PosteriorRowArtifactWire,
    _program: std::sync::Arc<antecedent_io::bayesian_posterior_row_artifact::CompiledPosteriorRow>,
    native_execution: Option<antecedent_core::execution_facts::ExecutionFacts>,
}

impl BayesianPosteriorRow {
    /// Run the posterior of a declared request and seal it with its digests.
    ///
    /// # Errors
    /// A typed refusal whose detail is one of `dose_grid.graph_not_certified`,
    /// `dose_grid.unsupported_dose`, `dose_grid.invalid_request` or `dose_grid.refused`; see
    /// [`PosteriorRowArtifactError::refusal`].
    pub fn evaluate(request: PosteriorRowRequestWire) -> Result<Self, PosteriorRowArtifactError> {
        Self::evaluate_with_context(
            request,
            &antecedent_core::ExecutionContext::production_default(0),
        )
    }

    /// Convenience execution through the common prepared compiler contract.
    ///
    /// # Errors
    /// Invalid premises, unsupported dose, cancellation, or resource refusal.
    pub fn evaluate_with_context(
        request: PosteriorRowRequestWire,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, PosteriorRowArtifactError> {
        crate::Study::prepare_posterior_row(request, ctx)?.estimate(ctx)
    }

    /// Independently consume with the same compiler/engine and caller execution policy.
    ///
    /// # Errors
    /// Invalid artifact data or version, replay disagreement, cancellation, or resource refusal.
    pub fn consume_with_context(
        bytes: &[u8],
        limits: PosteriorRowConsumeLimits,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, PosteriorRowArtifactError> {
        let observed = antecedent_core::execution_attempt::observe_with_context(ctx, || {
            PosteriorRowArtifactWire::consume_prepared_with_context(bytes, limits, ctx)
        });
        let (artifact, program) = observed.result?;
        Ok(Self {
            artifact,
            _program: std::sync::Arc::new(program),
            native_execution: observed.facts,
        })
    }

    /// Native observations from this execution; no additional scientific license.
    #[must_use]
    pub fn execution_facts(&self) -> Option<&antecedent_core::execution_facts::ExecutionFacts> {
        self.native_execution.as_ref()
    }

    /// Consume an exported artifact by recomputing the posterior from its embedded table.
    ///
    /// # Errors
    /// Corruption, an unknown version, a changed digest or premise, or a stored result that
    /// does not replay bit for bit.
    pub fn consume(
        bytes: &[u8],
        limits: PosteriorRowConsumeLimits,
    ) -> Result<Self, PosteriorRowArtifactError> {
        Self::consume_with_context(
            bytes,
            limits,
            &antecedent_core::ExecutionContext::production_default(0),
        )
    }

    /// Export the artifact.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, PosteriorRowArtifactError> {
        self.artifact
            .export()
            .map_err(|error| PosteriorRowArtifactError::Undecodable(error.to_string()))
    }

    /// The estimator id of this row, for example `bayesian.dose_grid`.
    #[must_use]
    pub fn estimator(&self) -> &str {
        &self.artifact.result.estimator
    }

    /// The stored request (premises and compact table).
    #[must_use]
    pub const fn request(&self) -> &PosteriorRowRequestWire {
        &self.artifact.request
    }

    /// The retained posterior: summaries, joint covariance and named diagnostics.
    #[must_use]
    pub const fn result(&self) -> &PosteriorRowResultWire {
        &self.artifact.result
    }

    /// `(premises digest, data digest)`.
    #[must_use]
    pub fn digests(&self) -> (&str, &str) {
        (&self.artifact.premises_digest, &self.artifact.data_digest)
    }

    /// The underlying artifact.
    #[must_use]
    pub const fn artifact(&self) -> &PosteriorRowArtifactWire {
        &self.artifact
    }
}

impl From<PosteriorRowArtifactError> for CausalError {
    fn from(error: PosteriorRowArtifactError) -> Self {
        let (code, detail) = error.refusal();
        CausalError::Compile {
            message: format!("{}{code}: {detail}: {error}", crate::error::REASON_PREFIX),
        }
    }
}

impl PartialEq for BayesianPosteriorRow {
    fn eq(&self, other: &Self) -> bool {
        self.artifact == other.artifact
    }
}

/// Frozen checked dose-grid posterior specialization of the common prepared handle.
#[derive(Clone, Debug)]
pub struct PosteriorRowPreparedState {
    program: std::sync::Arc<antecedent_io::bayesian_posterior_row_artifact::CompiledPosteriorRow>,
}
impl crate::Study {
    /// Compile/inspect the randomized-dose counterpart without fitting or drawing.
    ///
    /// # Errors
    /// Invalid premises, unsupported dose, cancellation, or resource refusal.
    pub fn prepare_posterior_row(
        request: PosteriorRowRequestWire,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<crate::PreparedStudy<PosteriorRowPreparedState>, PosteriorRowArtifactError> {
        let program =
            antecedent_io::bayesian_posterior_row_artifact::CompiledPosteriorRow::compile(
                request, ctx,
            )?;
        Ok(crate::PreparedStudy {
            state: PosteriorRowPreparedState { program: std::sync::Arc::new(program) },
        })
    }
}
impl crate::PreparedStudy<PosteriorRowPreparedState> {
    /// Frozen premise and data identities; inspection executes no statistical work.
    #[must_use]
    pub fn inspect(&self) -> (&str, &str) {
        self.state.program.identities()
    }
    /// Execute the retained checked statistical operation with caller policy.
    ///
    /// # Errors
    /// Numerical failure, cancellation, or resource refusal.
    pub fn estimate(
        &self,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<BayesianPosteriorRow, PosteriorRowArtifactError> {
        let observed = antecedent_core::execution_attempt::observe_with_context(ctx, || {
            self.state.program.artifact(ctx)
        });
        Ok(BayesianPosteriorRow {
            artifact: observed.result?,
            _program: std::sync::Arc::clone(&self.state.program),
            native_execution: observed.facts,
        })
    }
    /// Change evaluation coordinates while preserving the fitted data/prior posterior.
    ///
    /// # Errors
    /// Invalid grid or functional, cancellation, or retained-resource refusal.
    pub fn retarget(
        &self,
        grid: Vec<f64>,
        functional: antecedent_io::dose_grid_functional_artifact::DoseFunctionalWire,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, PosteriorRowArtifactError> {
        Ok(Self {
            state: PosteriorRowPreparedState {
                program: std::sync::Arc::new(self.state.program.retarget(grid, functional, ctx)?),
            },
        })
    }
}

#[cfg(test)]
mod prepared_tests {
    use super::*;
    use antecedent_core::{
        ExecutionContext,
        execution_attempt::{Operation, observe_execution},
    };
    use antecedent_io::dose_grid_functional_artifact::DoseFunctionalWire;
    fn request() -> PosteriorRowRequestWire {
        PosteriorRowRequestWire::DoseGrid(DoseGridPosteriorWire {
            design: "randomized_dose".into(),
            functional: DoseFunctionalWire { kind: "level".into(), from: None, to: None },
            grid: vec![0.25, 0.75],
            support: (0.0, 1.0),
            coefficient_scales: vec![2.0, 1.0],
            residual: ResidualPriorWire::KnownVariance(0.5),
            draws: 200,
            seed: 3,
            dose: vec![0.0, 0.25, 0.5, 0.75, 1.0],
            outcome: vec![0.0, 0.4, 1.1, 1.4, 2.0],
        })
    }
    #[test]
    fn public_preparation_inspects_without_fitting_and_retargets_the_frozen_posterior() {
        let ctx = ExecutionContext::for_tests(3);
        let request = request();
        let compiled =
            observe_execution(|| crate::Study::prepare_posterior_row(request.clone(), &ctx));
        assert_eq!(compiled.report.counts(Operation::PosteriorFit).attempted, 0);
        let prepared = compiled.result.unwrap();
        let original_identity = prepared.inspect().0.to_owned();
        drop(request);
        let first = prepared.estimate(&ctx).unwrap();
        assert_eq!(first.execution_facts().unwrap().observed_model_fits(), Some(1));
        let next = prepared
            .retarget(
                vec![0.5],
                DoseFunctionalWire { kind: "derivative".into(), from: None, to: None },
                &ctx,
            )
            .unwrap();
        assert_ne!(next.inspect().0, original_identity);
        let observed = observe_execution(|| next.estimate(&ctx));
        assert_eq!(observed.report.counts(Operation::PosteriorFit).attempted, 0);
        let result = observed.result.unwrap();
        assert_eq!(result.execution_facts().unwrap().observed_model_fits(), Some(0));
        assert_eq!(result.result().calibration, "unmeasured");
        assert!(!result.result().licensed);
        let bytes = result.export().unwrap();
        drop(prepared);
        drop(next);
        let replay = BayesianPosteriorRow::consume_with_context(
            &bytes,
            PosteriorRowConsumeLimits::default(),
            &ctx,
        )
        .unwrap();
        assert_eq!(replay, result);
    }
    #[test]
    fn preparation_does_not_apply_artifact_consumer_row_limits() {
        let mut request = request();
        let PosteriorRowRequestWire::DoseGrid(wire) = &mut request;
        wire.dose = (0..100_001_u32).map(|i| f64::from(i) / 100_000.0).collect();
        wire.outcome = wire.dose.iter().map(|dose| 2.0 * dose).collect();
        let ctx = ExecutionContext::for_tests(3);
        let compilation = observe_execution(|| crate::Study::prepare_posterior_row(request, &ctx));
        assert!(compilation.result.is_ok());
        assert_eq!(compilation.report.counts(Operation::PosteriorFit).attempted, 0);
    }

    #[test]
    fn changed_inputs_refit_and_execution_policy_refuses_before_fitting() {
        let ctx = ExecutionContext::for_tests(3);
        let original = request();
        let prepared = crate::Study::prepare_posterior_row(original.clone(), &ctx).unwrap();
        prepared.estimate(&ctx).unwrap();
        for prior in [false, true] {
            let mut changed = original.clone();
            let PosteriorRowRequestWire::DoseGrid(wire) = &mut changed;
            if prior {
                wire.coefficient_scales[0] = 3.0;
            } else {
                wire.outcome[0] = 0.1;
            }
            let new = crate::Study::prepare_posterior_row(changed, &ctx).unwrap();
            let execution = observe_execution(|| new.estimate(&ctx));
            assert!(execution.result.is_ok());
            assert_eq!(execution.report.counts(Operation::PosteriorFit).completed, 1);
        }
        let mut limited = ctx.clone();
        limited.memory.hard_limit_bytes = Some(1);
        let refused = observe_execution(|| prepared.estimate(&limited));
        assert!(refused.result.is_err());
        assert_eq!(refused.report.counts(Operation::PosteriorFit).attempted, 0);
        ctx.cancellation.cancel();
        assert!(prepared.estimate(&ctx).is_err());
        let mut unsupported = original;
        let PosteriorRowRequestWire::DoseGrid(wire) = &mut unsupported;
        wire.design = "observational".into();
        assert!(
            crate::Study::prepare_posterior_row(unsupported, &ExecutionContext::for_tests(3))
                .is_err()
        );
    }
}
