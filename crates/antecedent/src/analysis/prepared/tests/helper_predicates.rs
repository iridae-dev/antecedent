//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::analysis::prepared::query_reads_score_table;
use antecedent_core::{AverageEffectQuery, CausalQuery, OutcomeFunctional, VariableId};

/// A mean click keeps the estimator's own value, so refitting the cross-fit score table
/// for it would be discarded work; only exceedance, grid and quantile clicks read it.
#[test]
fn only_non_mean_functionals_read_the_frozen_score_table() {
    let base = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    assert!(!query_reads_score_table(&CausalQuery::AverageEffect(base.clone())));
    for functional in [
        OutcomeFunctional::exceedance(0.5),
        OutcomeFunctional::exceedance_grid(vec![0.0, 1.0]),
        OutcomeFunctional::quantile(0.5),
    ] {
        let query = base.clone().with_outcome_functional(functional);
        assert!(query_reads_score_table(&CausalQuery::AverageEffect(query)));
    }
}
