//! Discrete CI with a permutation reference when a chi-square reference is unsupported.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;

use super::{
    CiBatchRequest, CiBatchResult, CiWorkspace, ConditionalIndependenceTest, GSquared,
    PreparedCiTest, SignificanceMethod, SymbolicCmi,
};
use crate::StatsError;

/// G-squared on sufficiently populated contingency tables, with an explicit within-stratum
/// permutation fallback for sparse tables. A lack of permutation support still refuses the
/// query; it is never interpreted as independence. Existing `GSquared` behavior is unchanged.
///
/// The fallback reports the symbolic conditional mutual information statistic and its
/// permutation p-value. Consumers making CI decisions must use `p_value`, not assume one
/// statistic scale across references. Declare this provider and its budget in a discovery
/// contract rather than labeling it plain G-squared.
#[derive(Clone, Debug)]
pub struct SparseDiscreteCi {
    permutations: u32,
}

impl SparseDiscreteCi {
    /// Set the fallback null budget. An explicitly requested block-shuffle plan is honored
    /// instead, including its block length and unsupported-conditional-block refusals.
    ///
    /// # Errors
    /// A zero permutation budget.
    pub fn new(permutations: u32) -> Result<Self, StatsError> {
        if permutations == 0 {
            return Err(StatsError::Shape { message: "sparse discrete CI needs permutations" });
        }
        Ok(Self { permutations })
    }
}

impl ConditionalIndependenceTest for SparseDiscreteCi {
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        match GSquared::new().test_batch(prepared, request, workspace, ctx) {
            Err(StatsError::Unsupported { message })
                if message.starts_with("G² refuses sparse") =>
            {
                let mut permutation = prepared.clone();
                if matches!(permutation.plan.significance, SignificanceMethod::Analytic) {
                    permutation.plan.significance = SignificanceMethod::BlockShuffle {
                        replicates: self.permutations,
                        block_size: 1,
                    };
                }
                SymbolicCmi::new().test_batch(&permutation, request, workspace, ctx)
            }
            result => result,
        }
    }

    fn min_attainable_p(&self, significance: SignificanceMethod) -> f64 {
        let count = match significance {
            SignificanceMethod::Analytic => self.permutations,
            SignificanceMethod::BlockShuffle { replicates, .. } => replicates.max(1),
        };
        1.0 / (f64::from(count) + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ci::{CiPreparationPlan, CiQuery, ConfidenceMethod};

    fn compare(n: usize, significance: SignificanceMethod) {
        let x: Vec<f64> = (0..n).map(|i| f64::from(i % 2 == 0)).collect();
        let y = x.clone();
        let columns = [&x[..], &y[..]];
        let query = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let request = CiBatchRequest {
            columns: &columns,
            queries: &query,
            z_flat: &[],
            significance,
            confidence: ConfidenceMethod::None,
        };
        let ctx = ExecutionContext::for_tests(71);
        let provider = SparseDiscreteCi::new(199).unwrap();
        let prepared = provider
            .prepare(
                &columns,
                &CiPreparationPlan { significance, confidence: ConfidenceMethod::None },
                &ctx,
            )
            .unwrap();
        let actual =
            provider.test_batch(&prepared, &request, &mut CiWorkspace::default(), &ctx).unwrap();
        let expected = if n >= 20 {
            GSquared::new()
                .test_batch(&prepared, &request, &mut CiWorkspace::default(), &ctx)
                .unwrap()
        } else {
            let mut permutation = prepared;
            if matches!(significance, SignificanceMethod::Analytic) {
                permutation.plan.significance =
                    SignificanceMethod::BlockShuffle { replicates: 199, block_size: 1 };
            }
            SymbolicCmi::new()
                .test_batch(&permutation, &request, &mut CiWorkspace::default(), &ctx)
                .unwrap()
        };
        assert_eq!(actual.results[0].p_value.to_bits(), expected.results[0].p_value.to_bits());
        assert_eq!(actual.results[0].statistic.to_bits(), expected.results[0].statistic.to_bits());
    }

    #[test]
    fn populated_tables_keep_the_original_analytic_numbers() {
        compare(40, SignificanceMethod::Analytic);
    }

    #[test]
    fn sparse_tables_use_the_declared_conditional_permutation_null() {
        compare(8, SignificanceMethod::Analytic);
        compare(8, SignificanceMethod::BlockShuffle { replicates: 19, block_size: 1 });
    }

    #[test]
    fn the_null_budget_and_attainable_resolution_are_explicit() {
        assert!(SparseDiscreteCi::new(0).is_err());
        let provider = SparseDiscreteCi::new(199).unwrap();
        assert_eq!(
            provider.min_attainable_p(SignificanceMethod::Analytic).to_bits(),
            0.005_f64.to_bits()
        );
        assert_eq!(
            provider
                .min_attainable_p(SignificanceMethod::BlockShuffle {
                    replicates: 19,
                    block_size: 1
                })
                .to_bits(),
            0.05_f64.to_bits()
        );
    }
}
