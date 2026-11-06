//! Population-level counterfactual resampling and its admission lease.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{Case, CausalError, CounterfactualProcedure, ExecutionContext, Table, case_values};
use antecedent_estimate::RowResampling;

/// Stream key of the frequentist replicates.
const FREQUENTIST_STREAM: u64 = 0x0CF1_C1A5;
/// Stream key of the Bayesian-bootstrap draws.
const BAYESIAN_STREAM: u64 = 0x0CF1_BA7E;

// Admission covers the nested matrix headers and values, plus the peak row
// weights and block-weight vector. The lease travels with the matrix.
pub(super) fn admit_counterfactual_resamples(
    ctx: &ExecutionContext,
    draws: usize,
    width: usize,
    rows: usize,
) -> Result<antecedent_core::execution::AdmissionLease, CausalError> {
    let bytes = draws
        .checked_mul(width)
        .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
        .and_then(|n| {
            draws
                .checked_mul(std::mem::size_of::<Vec<f64>>())
                .and_then(|headers| n.checked_add(headers))
        })
        .and_then(|n| {
            rows.checked_mul(2 * std::mem::size_of::<f64>())
                .and_then(|scratch| n.checked_add(scratch))
        })
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| CausalError::Resource {
            message: "counterfactual resample size overflow".into(),
        })?;
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => {
            CausalError::Cancelled { stage: super::super::super::stage::STAGE_UNCERTAINTY }
        }
        other => {
            CausalError::Resource { message: format!("counterfactual resample admission: {other}") }
        }
    })
}

// Fields drop in declaration order: release the matrix before its admission.
pub(super) struct AdmittedCounterfactualMatrix {
    pub(super) matrix: Vec<Vec<f64>>,
    _lease: antecedent_core::execution::AdmissionLease,
}

/// What the resampling of the identified functional produced.
pub(super) enum Resampled {
    /// Point only.
    None,
    /// Bootstrap replicates: one row of case values per successful replicate.
    Replicates { held: AdmittedCounterfactualMatrix, failed: u32, requested: u32 },
    /// Bayesian-bootstrap draws: one row of case values per draw.
    Draws { held: AdmittedCounterfactualMatrix },
}

impl super::super::Study {
    /// Resample the identified functionals: a row bootstrap or a Dirichlet-weighted bootstrap.
    pub(super) fn resample_counterfactual(
        &self,
        table: &Table,
        cases: &[Case],
        ctx: &ExecutionContext,
    ) -> Result<Resampled, CausalError> {
        let rows = table.cells.len();
        let (resampling, _) = self.functional_row_resampling()?;
        let block = match resampling {
            RowResampling::Rows => None,
            RowResampling::SeriesBlocks | RowResampling::WindowBlocks { .. } => {
                Some(antecedent_data::circular_block_length(1, rows))
            }
        };
        let cancelled =
            || CausalError::Cancelled { stage: super::super::super::stage::STAGE_UNCERTAINTY };
        match CounterfactualProcedure::for_inference(&self.settings.inference)? {
            CounterfactualProcedure::DirichletRowWeights { draws } => {
                let width = cases
                    .iter()
                    .try_fold(0usize, |n, case| n.checked_add(case.prepared.len()))
                    .ok_or_else(|| CausalError::Resource {
                        message: "counterfactual resample width overflow".into(),
                    })?;
                let lease = admit_counterfactual_resamples(ctx, draws, width, rows)?;
                let mut rng =
                    ctx.rng.stream_for(antecedent_core::StreamDomain::Attribution, BAYESIAN_STREAM);
                let mut matrix = Vec::with_capacity(draws);
                for _ in 0..draws {
                    if ctx.cancellation.is_cancelled() {
                        return Err(cancelled());
                    }
                    let weights = match block {
                        None => {
                            antecedent_kernels::bayesian_bootstrap_weights(&mut rng, rows, false)
                        }
                        Some(length) => {
                            super::super::temporal_attribution::block_dirichlet_row_weights(
                                rows, length, &mut rng,
                            )
                        }
                    };
                    matrix.push(case_values(cases, &table.weighted_law(&weights)?, ctx)?);
                }
                Ok(Resampled::Draws {
                    held: AdmittedCounterfactualMatrix { matrix, _lease: lease },
                })
            }
            _ if self.settings.bootstrap_replicates == 0 => Ok(Resampled::None),
            _ => {
                if block.is_none() {
                    super::super::gcm_bootstrap::require_subject_resampling(
                        self.declared.sampling_design,
                    )?;
                }
                let requested = self.settings.bootstrap_replicates;
                let width = cases
                    .iter()
                    .try_fold(0usize, |n, case| n.checked_add(case.prepared.len()))
                    .ok_or_else(|| CausalError::Resource {
                        message: "counterfactual resample width overflow".into(),
                    })?;
                let draws = usize::try_from(requested).map_err(|_| CausalError::Resource {
                    message: "counterfactual replicate count overflow".into(),
                })?;
                let lease = admit_counterfactual_resamples(ctx, draws, width, rows)?;
                let mut matrix = Vec::with_capacity(draws);
                let mut failed = 0_u32;
                for replicate in 0..requested {
                    if ctx.cancellation.is_cancelled() {
                        return Err(cancelled());
                    }
                    let mut rng = super::super::gcm_bootstrap::replicate_stream(
                        ctx,
                        FREQUENTIST_STREAM,
                        replicate,
                    );
                    let counts = match block {
                        None => super::super::gcm_bootstrap::bootstrap_counts(&mut rng, rows),
                        Some(length) => super::super::temporal_attribution::block_bootstrap_counts(
                            rows, length, &mut rng,
                        ),
                    };
                    let values = table
                        .empirical_law(table.counts(&counts))
                        .and_then(|law| case_values(cases, &law, ctx));
                    match values {
                        Ok(values) => matrix.push(values),
                        Err(_) if ctx.cancellation.is_cancelled() => return Err(cancelled()),
                        Err(_) => failed += 1,
                    }
                }
                Ok(Resampled::Replicates {
                    held: AdmittedCounterfactualMatrix { matrix, _lease: lease },
                    failed,
                    requested,
                })
            }
        }
    }
}
