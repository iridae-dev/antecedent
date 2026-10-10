//! Frozen repeated-row Verma-SCM design for the closed expected-Fisher/delta candidate.
//! Measurement is ignored by ordinary tests; no public interval is activated here.
//! IID multinomial rows are generated independently, all three actual intervals are
//! scored separately, and every refusal counts as a miss under the shared 1% cap.
#![allow(clippy::cast_precision_loss, reason = "bounded simulated integer frequency tables")]
#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;
#[path = "common/candidate_calibration.rs"]
mod candidate;
use antecedent_core::ExecutionContext;
use antecedent_estimate::nested_markov_binary::{
    AdmgDeclaration, FitOptions, NestedMarkovInput, Regime, RegimeCounts,
};
use antecedent_estimate::nested_markov_uncertainty::{
    NestedFisherFunctional, nested_markov_fisher_internal,
};
use calibration::{CoverageTally, RecordKey, grid_n, grid_seed, map_replicates, n_sim};

fn sampled_verma(n: usize, seed: u64, latent: bool) -> NestedMarkovInput {
    let mut rng = candidate::Generator::new(seed);
    let mut cells = vec![0.; 16];
    for _ in 0..n {
        let x1 = rng.binary(0.5);
        let u = rng.binary(if latent { 0.3 } else { 0.0 });
        // Hidden U confounds X2 and X4; X1 provides variation in X2 without a direct Y edge.
        let p2 = if latent { 0.2 + 0.25 * x1 as f64 + 0.35 * u as f64 } else { 0.5 };
        let x2 = rng.binary(p2);
        let m = rng.binary(0.25 + 0.5 * x2 as f64);
        let py =
            if latent { 0.15 + 0.35 * m as f64 + 0.3 * u as f64 } else { 0.2 + 0.4 * m as f64 };
        let y = rng.binary(py);
        cells[8 * x1 + 4 * x2 + 2 * m + y] += 1.;
    }
    NestedMarkovInput {
        graph: AdmgDeclaration::selected(),
        regimes: vec![RegimeCounts { regime: Regime::Observational, levels: vec![2; 4], cells }],
    }
}

fn measure(latent: bool, test: &'static str, expected_ids: [&str; 3]) {
    let functionals = [
        NestedFisherFunctional::Mean0,
        NestedFisherFunctional::Mean1,
        NestedFisherFunctional::Contrast,
    ];
    // Independent structural sum: E[Y|do(x2)] integrates M and hidden U directly.
    let truth = if latent { [0.3275, 0.5025, 0.175] } else { [0.3, 0.5, 0.2] };
    let mut tallies = functionals.map(|f| {
        CoverageTally::for_record(
            RecordKey {
                test,
                dgp: if latent { "iid_latent_verma_rows" } else { "iid_markov_submodel_rows" },
                interval: "analytic_se",
            },
            0.95,
        )
        .labelled(format!("{f:?}"))
    });
    let n = grid_n(2000);
    let results = map_replicates(n_sim(), |rep| {
        let seed = grid_seed(0x43fa_1900 + rep);
        let input = sampled_verma(n, seed, latent);
        nested_markov_fisher_internal(
            &input,
            &FitOptions::default(),
            0.95,
            &ExecutionContext::for_tests(seed),
        )
    });
    for result in results {
        match result {
            Ok(result) => {
                for (i, tally) in tallies.iter_mut().enumerate() {
                    candidate::bind(tally, &result.calibration_basis(functionals[i]));
                    let [lo, hi] = result.interval_candidates[i];
                    tally.record(Some((lo, hi)), truth[i]);
                }
            }
            Err(_) => {
                for tally in &mut tallies {
                    tally.skip();
                }
            }
        }
    }
    for (tally, id) in tallies.into_iter().zip(expected_ids) {
        assert_eq!(tally.record_id().as_deref(), Some(id));
        tally.assert();
    }
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn nested_fisher_iid_latent_verma_l95() {
    measure(
        true,
        "nested_fisher_iid_latent_verma_l95",
        [
            "cov.intervention_response.admg.frequentist.analytic_se.l95.nested_fisher_iid_latent_verma_l95.mean0",
            "cov.intervention_response.admg.frequentist.analytic_se.l95.nested_fisher_iid_latent_verma_l95.mean1",
            "cov.average_effect.admg.frequentist.analytic_se.l95.nested_fisher_iid_latent_verma_l95.contrast",
        ],
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn nested_fisher_iid_markov_submodel_l95() {
    measure(
        false,
        "nested_fisher_iid_markov_submodel_l95",
        [
            "cov.intervention_response.admg.frequentist.analytic_se.l95.nested_fisher_iid_markov_submodel_l95.mean0",
            "cov.intervention_response.admg.frequentist.analytic_se.l95.nested_fisher_iid_markov_submodel_l95.mean1",
            "cov.average_effect.admg.frequentist.analytic_se.l95.nested_fisher_iid_markov_submodel_l95.contrast",
        ],
    );
}
