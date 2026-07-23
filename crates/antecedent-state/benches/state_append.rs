//! Criterion smoke: append + invalidate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, clippy::cast_precision_loss)]

use std::sync::Arc;

use antecedent_core::{AverageEffectQuery, CacheBudget, CausalQuery, VariableId};
use antecedent_state::{CausalState, DataBatchRef, LinearOlsSuffStats, StateEvent};
use criterion::{Criterion, criterion_group, criterion_main};

fn bench_append(c: &mut Criterion) {
    c.bench_function("state_append_invalidate_ols", |b| {
        b.iter(|| {
            let mut state = CausalState::new(CacheBudget::new(1 << 20));
            let q = state.queries.register(CausalQuery::AverageEffect(
                AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)),
            ));
            let _ = state.refresh_results(&[(q, 1, 32)]);
            state.suff_stats.ols.insert(Arc::from("ols"), LinearOlsSuffStats::new(2));
            for i in 0..64u64 {
                state
                    .apply(StateEvent::AppendData(DataBatchRef {
                        id: Arc::from(format!("b{i}")),
                        nrows: 8,
                        bytes: 64,
                    }))
                    .expect("apply");
                let key: Arc<str> = Arc::from("ols");
                let ols = state.suff_stats.ols.get_mut(&key).unwrap();
                ols.append_row(&[1.0, i as f64], 2.0 * i as f64).unwrap();
            }
            state
        });
    });
}

criterion_group!(benches, bench_append);
criterion_main!(benches);
