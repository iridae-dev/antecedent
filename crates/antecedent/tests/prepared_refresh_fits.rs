//! A refresh derives each cross-fitted nuisance once, and the handle it leaves
//! is the handle a fresh prepare on the same rows builds.
//!
//! The refresh click and the frozen score table of the refreshed study fit the
//! same propensity and outcome models. They are served from one nuisance
//! scope, so the second request is a reuse instead of a refit, and the scope
//! keys every fit on all of its inputs, so sharing cannot change a number.

use antecedent::{RefuteSuite, Study, StudyBuilder};
use antecedent_core::{ExecutionContext, OutcomeFunctional};
use antecedent_data::TabularData;
use antecedent_estimate::{AipwAte, CrossfitCacheStats, CrossfitNuisanceCache};

#[path = "fixtures_common/mod.rs"]
mod common;
use common::fixtures::confounded_scm;

fn builder(data: &TabularData, functional: Option<OutcomeFunctional>) -> StudyBuilder {
    let (_, dag, query) = confounded_scm(8, 1);
    let query = match functional {
        Some(functional) => query.with_outcome_functional(functional),
        None => query,
    };
    Study::tabular(data.clone())
        .graph(dag)
        .query(query)
        .estimator(AipwAte::new().with_bootstrap_replicates(0))
        .refute(RefuteSuite::None)
}

/// Refresh a handle prepared on one table to another inside a counting scope,
/// then compare it with a handle prepared directly on the new rows.
fn refresh_counts_and_matches_fresh(functional: Option<OutcomeFunctional>) -> CrossfitCacheStats {
    let ctx = ExecutionContext::for_tests(73);
    let (first, _, _) = confounded_scm(400, 5);
    let (second, _, _) = confounded_scm(640, 9);

    let mut prepared = builder(&first, functional.clone()).build().unwrap().prepare(&ctx).unwrap();
    let cache = CrossfitNuisanceCache::new();
    let (refreshed, _) = cache.scope(|| rebind_estimate(&mut prepared, second.clone(), &ctx));
    let refreshed = refreshed.unwrap();
    let stats = cache.stats();

    // Neither of these runs inside a scope, so every fit is a fresh one.
    let fresh = builder(&second, functional).build().unwrap().prepare(&ctx).unwrap();
    let expected = fresh.estimate_retained(&ctx).unwrap();
    let (a, b) =
        (refreshed.estimate().as_effect().unwrap(), expected.estimate().as_effect().unwrap());
    assert_eq!(a.ate.to_bits(), b.ate.to_bits());
    assert_eq!(a.se_analytic().to_bits(), b.se_analytic().to_bits());
    assert_eq!(a.scores.table, b.scores.table);
    assert_eq!(refreshed.executed_contract(), expected.executed_contract());
    assert!(prepared.score_table().is_some(), "an AllObserved iid AIPW plan freezes its scores");
    assert_eq!(prepared.score_table(), fresh.score_table(), "frozen scores are the fresh build");
    stats
}

#[test]
fn a_mean_refresh_fits_the_nuisances_once() {
    let stats = refresh_counts_and_matches_fresh(None);
    // The click and the frozen table each request the propensity and the
    // outcome regressions; one fit serves both.
    assert_eq!(stats.propensity_fits, 1, "{stats:?}");
    assert!(stats.propensity_reuses >= 1, "{stats:?}");
    assert_eq!(stats.outcome_fits, 1, "{stats:?}");
    assert!(stats.outcome_reuses >= 1, "{stats:?}");
}

#[test]
fn a_functional_refresh_fits_each_nuisance_once() {
    let stats = refresh_counts_and_matches_fresh(Some(OutcomeFunctional::exceedance(0.5)));
    assert!(stats.propensity_reuses >= 1, "{stats:?}");
    assert_eq!(stats.propensity_fits, 1, "{stats:?}");
    assert!(stats.outcome_reuses >= 1, "{stats:?}");
}

/// Bind `data` on a new branch, materialize and estimate it, then adopt the branch as the handle.
/// A failure leaves the handle unchanged, as the immutable lifecycle does.
fn rebind_estimate(
    prepared: &mut antecedent::PreparedStudy,
    data: antecedent_data::TabularData,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent::StudyResult, antecedent::CausalError> {
    let branch = prepared.rebind_data(data, ctx)?.materialize(ctx)?;
    let result = branch.estimate_retained(ctx)?;
    *prepared = branch;
    Ok(result)
}
