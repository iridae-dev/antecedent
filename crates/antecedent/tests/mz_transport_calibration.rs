//! Coverage of the multi-source (`TR^mz`) joint percentile bootstrap against
//! the known truth of R-443 Figure 1(c,d).
//!
//! The target, source `a` and source `b` structural models of the shared
//! fixture enumerate every supplied law and the target's interventional truth
//! `P*(Y = 1 | do(X = 1))`. Each replicate draws fresh count tables from those
//! laws and scores the internal estimator the public route withholds
//! (`cell_not_licensed`) until these records exist:
//! [`antecedent_estimate::mz_transport_bootstrap_interval`].
//!
//! * `multi_source_mz_independent_studies`: every regime is its own declared
//!   study with its own `n` rows; each table is resampled on its own stream.
//! * `multi_source_mz_shared_units`: source `b`'s trial publishes its
//!   `(X, Z2, Y)` joint and the `(X, Z2)` margin of the same `n` rows under one
//!   forwarded dataset; the bootstrap resamples the joint once per replicate and
//!   projects the margin from it.
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;
#[path = "../../antecedent-estimate/tests/common/mod.rs"]
mod transport_fixture;

use antecedent_core::{EvidenceCatalog, ExecutionContext, RegimeId, Value};
use antecedent_estimate::mz_transport_bootstrap_interval;
use antecedent_expr::{Assignment, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::{
    BoundMzTransportFunctional, MZ_TRANSPORT_DEFAULT_LIMITS, MzTransportDecision,
    bind_mz_transport_catalog, decide_mz_transport,
};
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, map_replicates,
    n_sim, stream_seed, unit_uniform,
};
use transport_fixture::mz_fixture::{
    B_TRIAL_REGIME, SharedTable, X, Y, evidence, graph, query, sources, target_scm,
    with_shared_b_trial, with_studies,
};
use transport_fixture::z_scm::vid;

const INTERVAL: &str = "percentile_bootstrap";
/// The joint percentile bootstrap's existing replicate floor.
const REPLICATES: u32 = 199;
/// The independent-study interval showed 0.941 coverage in 8,000 datasets at
/// 1,600 rows/law with 199 bootstrap draws. Its 2.5% tail quantile has only
/// about five draws there, so measure the same estimator with a more stable
/// tail before licensing the independent-study coordinate.
const INDEPENDENT_REPLICATES: u32 = 999;

fn construction(dependence: &str) -> Construction {
    Construction {
        query: "ClassicalTransport".into(),
        graph_class: "Admg".into(),
        structure: "fixed".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: "transport.mz_joint_bootstrap".into(),
        interval_method: INTERVAL.into(),
        se_kind: "percentile".into(),
        dependence: dependence.into(),
        posterior: String::new(),
        functional: "target_interventional_mean".into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
    }
}

fn request() -> Assignment {
    Assignment::from_pairs([(vid(X), Value::Bool(true))])
}

fn truth() -> f64 {
    target_scm().risk(&[(X, 1)], Y)
}

/// Decide once on the known catalog; every replicate reuses the frozen proof.
fn bound(catalog: &EvidenceCatalog) -> BoundMzTransportFunctional {
    let ctx = ExecutionContext::for_tests(1);
    let MzTransportDecision::Identified { derivation, .. } = decide_mz_transport(
        &graph(),
        &query(sources()),
        catalog,
        MZ_TRANSPORT_DEFAULT_LIMITS,
        &ctx,
    )
    .unwrap() else {
        panic!("R-443 Figure 1(c,d) identifies from the complementary catalog");
    };
    bind_mz_transport_catalog(&graph(), &derivation, catalog).unwrap()
}

/// `n` iid rows from `probabilities`, as cell counts.
fn draw_counts(n: usize, seed: u64, probabilities: &[f64]) -> Vec<u64> {
    let mut counts = vec![0u64; probabilities.len()];
    for row in 0..n {
        let mut threshold = unit_uniform(stream_seed(seed, row as u64));
        let mut chosen = probabilities.len() - 1;
        for (index, probability) in probabilities.iter().enumerate() {
            if threshold < *probability {
                chosen = index;
                break;
            }
            threshold -= probability;
        }
        counts[chosen] += 1;
    }
    counts
}

#[expect(clippy::cast_precision_loss, reason = "replicate counts fit f64 exactly")]
fn counted(law: &ExactDiscreteLaw, counts: Vec<u64>) -> ExactDiscreteLaw {
    let total = counts.iter().sum::<u64>() as f64;
    ExactDiscreteLaw::try_empirical(
        law.population(),
        law.regime(),
        law.interventions().to_vec(),
        law.axes().to_vec(),
        counts.iter().map(|c| *c as f64 / total).collect::<Vec<_>>(),
        law.snapshot_identity(),
        law.tolerance(),
    )
    .unwrap()
    .with_empirical_counts(counts)
    .unwrap()
}

/// Every regime its own study: `n` fresh rows per supplied law.
fn mz_figure_1_independent_studies(
    exact: &ExactTransportData,
    n: usize,
    seed: u64,
) -> ExactTransportData {
    let laws = exact
        .laws()
        .iter()
        .enumerate()
        .map(|(k, law)| {
            counted(law, draw_counts(n, stream_seed(seed, k as u64), law.probabilities()))
        })
        .collect::<Vec<_>>();
    ExactTransportData::try_new(laws, exact.max_support_rows()).unwrap()
}

/// As [`mz_figure_1_independent_studies`], but source `b`'s trial also publishes
/// the `(X, Z2)` margin of its own `n` rows under the same forwarded dataset.
/// `shared` carries the law shapes; `exact` the known laws rows are drawn from.
fn mz_figure_1_shared_units(
    exact: &ExactTransportData,
    shared: &ExactTransportData,
    n: usize,
    seed: u64,
) -> ExactTransportData {
    let known = |regime| exact.laws().iter().find(|law| law.regime() == regime);
    let joint = known(RegimeId::from_raw(B_TRIAL_REGIME)).unwrap();
    let rows = |regime: RegimeId, probabilities: &[f64]| {
        draw_counts(n, stream_seed(seed, u64::from(regime.raw())), probabilities)
    };
    let joint_counts = rows(joint.regime(), joint.probabilities());
    let laws = shared
        .laws()
        .iter()
        .map(|law| match known(law.regime()) {
            Some(source) if source.regime() == joint.regime() => counted(law, joint_counts.clone()),
            Some(source) => counted(law, rows(source.regime(), source.probabilities())),
            // Joint axes are (X, Z2, Y), last fastest: the margin sums Y out.
            None => counted(law, joint_counts.chunks(2).map(|pair| pair.iter().sum()).collect()),
        })
        .collect::<Vec<_>>();
    ExactTransportData::try_new(laws, shared.max_support_rows()).unwrap()
}

/// Score the do(X=1) mean interval of the internal joint bootstrap per replicate.
fn measure(
    tally: &mut CoverageTally,
    functional: &BoundMzTransportFunctional,
    n: usize,
    dependence: &str,
    bootstrap_replicates: u32,
    data: impl Fn(u64) -> ExactTransportData + Sync,
) {
    let truth = truth();
    let rows = map_replicates(n_sim(), |rep| {
        let data = data(rep);
        let ctx = ExecutionContext::for_tests(rep);
        mz_transport_bootstrap_interval(
            functional,
            &data,
            &[request()],
            ExactEvaluationLimits::default(),
            bootstrap_replicates,
            REPORTED_LEVEL,
            &ctx,
        )
        .ok()
        .and_then(Result::ok)
        .map(|intervals| {
            let band = intervals.requests[0]
                .mean_intervals
                .iter()
                .find(|(variable, _, _)| *variable == vid(Y))
                .map(|(_, lo, hi)| (*lo, *hi));
            (intervals.replicates_ok, band)
        })
    });
    for row in rows {
        match row {
            Some((replicates_ok, band)) => {
                tally.bind(
                    &construction(dependence),
                    ScopeFacts {
                        row_count: n as u64,
                        replicates_ok: Some(replicates_ok),
                        posterior_draws: None,
                        unidentified_mass: 0.0,
                    },
                );
                tally.record(band, truth);
            }
            None => tally.skip(),
        }
    }
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn multi_source_mz_independent_studies() {
    let (catalog, exact) = evidence();
    let catalog = with_studies(&catalog);
    let functional = bound(&catalog);
    // The 200-row independent-study design covered 0.930 in 2,000 repetitions,
    // below the 0.940 precision floor. Begin this interval's measured scope at
    // 400 rows per supplied law; the shared-unit design has its own grid below.
    let n = grid_n(800);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "multi_source_mz_independent_studies",
            dgp: "mz_figure_1_independent_studies",
            interval: INTERVAL,
        },
        REPORTED_LEVEL,
    );
    measure(&mut tally, &functional, n, "iid", INDEPENDENT_REPLICATES, |rep| {
        mz_figure_1_independent_studies(&exact, n, rep.wrapping_mul(1_000_003))
    });
    tally.assert();
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn multi_source_mz_shared_units() {
    let (catalog, exact) = evidence();
    // The declared catalog and law shapes; every replicate redraws the counts.
    let seedling = mz_figure_1_independent_studies(&exact, 64, 0);
    let (catalog, shared) = with_shared_b_trial(&catalog, &seedling, SharedTable::Margin);
    let functional = bound(&catalog);
    let n = grid_n(400);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "multi_source_mz_shared_units",
            dgp: "mz_figure_1_shared_units",
            interval: INTERVAL,
        },
        REPORTED_LEVEL,
    );
    measure(&mut tally, &functional, n, "shared_units", REPLICATES, |rep| {
        mz_figure_1_shared_units(&exact, &shared, n, rep.wrapping_mul(1_000_033))
    });
    tally.assert();
}
