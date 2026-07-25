#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::manual_map,
    clippy::match_wildcard_for_single_variants,
    clippy::doc_markdown,
    clippy::map_unwrap_or
)]
//! `CausalState` as the primary online append path (ADR 0016).
//!
//! 1. Append (or replace) data — state versions and marks registered queries stale.
//! 2. Call `refresh_results` under the `CacheBudget` — never auto-reruns on append.
//! 3. UI code must key on `version` / stale queries so it never mixes an old
//!    identification summary with a new estimate.
//!
//! Run: `cargo run -p antecedent --example causal_state_workflow`

use std::sync::Arc;

use antecedent::prelude::*;
use antecedent::state::{
    DataBatchRef, LinearOlsSuffStats, StateEvent, apply_state_event, new_antecedent_state,
};
use antecedent_core::{AverageEffectQuery, CacheBudget, CausalQuery};

fn main() -> Result<(), CausalError> {
    let mut rng_state = 1u64;
    let mut next_f64 = || {
        rng_state = rng_state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (rng_state >> 11) as f64 / ((1u64 << 53) as f64)
    };
    let mut gauss = || {
        let u1 = next_f64().max(1e-12);
        let u2 = next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    // Bound retained result bytes; over-budget refresh refuses instead of silent drop.
    let mut state = new_antecedent_state(CacheBudget::new(1 << 20));

    let n = 40usize;
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let ti = gauss();
        t.push(ti);
        y.push(0.5 * ti + gauss() * 0.1);
    }

    let ver = apply_state_event(
        &mut state,
        StateEvent::AppendData(DataBatchRef {
            id: Arc::from("b0"),
            nrows: n as u64,
            bytes: (n * 16) as u64,
        }),
    )?;
    println!("version after append={}", ver.raw());

    // Register a query; refresh stores a versioned fingerprint (does not run estimators).
    let qid = state.queries.register(CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
    )));
    state.refresh_results(&[(qid, 1, 8)])?;
    println!(
        "stale_queries={} batches={}",
        state.stale_queries().len(),
        state.data_catalog.batches.len()
    );

    // Incremental OLS: append rows, then compare shape to a full design.
    let key: Arc<str> = Arc::from("m1");
    state.suff_stats.ols.insert(Arc::clone(&key), LinearOlsSuffStats::new(2));
    for (ti, yi) in t.iter().zip(y.iter()) {
        state.suff_stats.ols.get_mut(&key).unwrap().append_row(&[1.0, *ti], *yi)?;
    }
    let ols = &state.suff_stats.ols[&key];
    println!("ols n={} ncols={}", ols.n, ols.ncols);

    // Replace data → registered query becomes stale until explicit refresh.
    let new_ver = state.data_catalog.version.next();
    apply_state_event(&mut state, StateEvent::ReplaceData(new_ver))?;
    println!("after replace stale={}", state.stale_queries().len());
    state.refresh_results(&[(qid, 1, 8)])?;
    println!("after refresh stale={} version={}", state.stale_queries().len(), state.version.raw());
    Ok(())
}
