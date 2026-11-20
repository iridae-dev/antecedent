//! Independent observed-data references and six-variable development studies.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn from_columns(
    predictors: &[Vec<f64>],
    y: &[Option<f64>],
    variables: &[VariableId],
    prior: &GraphPrior,
    seed: u64,
    draws: usize,
    ctx: &ExecutionContext,
) -> Result<crate::observed_marginals::ObservedMarginals, crate::DiscoveryError> {
    let mut cols = predictors.to_vec();
    cols.push(y.iter().map(|o| o.unwrap_or(f64::NAN)).collect());
    let table = antecedent_data::TabularData::from_f64_columns(
        cols.iter().enumerate().map(|(j, v)| (format!("v{j}"), v.as_slice())),
    )?;
    crate::observed_marginals::signed_uniform_observed_marginals(
        &table, variables, prior, seed, draws, ctx,
    )
}

#[test]
fn observed_score_matches_direct_missing_cell_oracle() {
    let x = vec![0.8, -0.3, 1.25];
    let y = vec![Some(-1.4), Some(0.1), None];
    let v = variable_ids(2);
    let prior = GraphPrior::default();
    let ctx = ExecutionContext::for_tests(1);
    let out = from_columns(&[x], &y, &v, &prior, 94, 8192, &ctx).unwrap();
    // Independent integration of the missing value and coefficient in the joint density.
    assert!((out.edge_marginals[1] - 0.382_026_14).abs() < 0.02);
    assert!((out.edge_marginals[2] - 0.281_894_05).abs() < 0.02);
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn observed_score_six_variable_per_edge_reliability() {
    let n = RELIABILITY_VARIABLES;
    validate_observed("per_edge_observed", &per_edge_prior(n, MAX_PARENTS), 0x7158_0000);
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn observed_score_six_variable_soft_tier_reliability() {
    let n = RELIABILITY_VARIABLES;
    validate_observed(
        "per_edge_and_soft_tiers_observed",
        &per_edge_and_soft_tier_prior(n, MAX_PARENTS),
        0x7159_0000,
    );
}

fn validate_observed(design: &str, prior: &GraphPrior, seed: u64) {
    let n = RELIABILITY_VARIABLES;
    let vars = variable_ids(n);
    let mut panels = Vec::new();
    let mut refused = 0;
    for rep in 0..calibration_replicates(RELIABILITY_REPLICATES) {
        let mut rng = CausalRng::from_seed(seed + rep);
        let truth = draw_dag_from_prior(prior, n, &mut rng);
        let mut cols = rows_from_dag(&truth, n, N_ROWS, &mut rng);
        let y: Vec<_> = cols
            .pop()
            .unwrap()
            .into_iter()
            .map(|value| (rng.next_f64() >= POOLED_MISSING_SHARE).then_some(value))
            .collect();
        let ctx = ExecutionContext::for_tests(3 + rep);
        let out = match from_columns(&cols, &y, &vars, prior, 0x0eda_0000 + rep, 16384, &ctx) {
            Ok(out) => out,
            Err(why) => {
                eprintln!("observed-score {design} dataset {rep} refused: {why}");
                refused += 1;
                continue;
            }
        };
        let mut cells = [(0., 0., 0.); N_BINS];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let p = out.edge_marginals[i * n + j];
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "the finite directed probability indexes one of ten bins"
                )]
                let k = ((p * N_BINS as f64) as usize).min(N_BINS - 1);
                cells[k].0 += 1.;
                cells[k].1 += p;
                cells[k].2 += f64::from(u8::from(truth.has_edge(n, i, j)));
            }
        }
        panels.push(cells);
        if (rep + 1) % 25 == 0 {
            eprintln!("observed-score {design} finished {} datasets", rep + 1);
        }
    }
    let sums = pool(&panels);
    let mut passed = refused == 0;
    for (k, &(cells, p, y)) in sums.iter().enumerate() {
        if cells < RELIABILITY_MIN_CELLS {
            continue;
        }
        let spread =
            panels.iter().map(|panel| (panel[k].2 - panel[k].1).powi(2)).sum::<f64>().sqrt();
        let residual = (y - p).abs();
        passed &= residual <= RELIABILITY_MAX_Z * spread;
        eprintln!(
            "observed-score {design} bin={k} cells={cells} predicted={} observed={} z={}",
            p / cells,
            y / cells,
            (y - p) / spread
        );
    }
    eprintln!("observed-score {design} panels={} refusals={refused}", panels.len());
    print_reliability_report(
        design,
        GraphScoreFamily::GaussianSignedUniform,
        &panels,
        &sums,
        passed,
    );
    assert!(passed);
}

#[test]
fn six_variable_embedding_matches_direct_missing_cell_oracle() {
    use crate::constraints::contemp_link;
    let v = variable_ids(6);
    let mut prior = GraphPrior::default();
    prior.constraints.max_parents = Some(1);
    let mut forbidden = Vec::new();
    for i in 0..6 {
        for j in 0..6 {
            if i != j && (i, j) != (0, 5) && (i, j) != (5, 0) {
                forbidden.push(contemp_link(v[i], v[j]));
            }
        }
    }
    prior.constraints.forbidden = forbidden.into();
    let mut x = vec![vec![0.; 3]; 5];
    x[0] = vec![0.8, -0.3, 1.25];
    let y = vec![Some(-1.4), Some(0.1), None];
    let r = from_columns(&x, &y, &v, &prior, 831, 4096, &ExecutionContext::for_tests(1)).unwrap();
    assert!((r.edge_marginals[5] - 0.382_026_14).abs() < 0.02);
    assert!((r.edge_marginals[30] - 0.281_894_05).abs() < 0.02);
    assert!(r.edge_marginals.iter().enumerate().all(|(i, p)| i == 5 || i == 30 || *p == 0.0));
}

#[test]
fn observed_score_matches_six_variable_forest_quadrature() {
    // Independent reference enumerates the 16,807 forests, integrates both
    // missing values and every coefficient, and normalizes the graph weights.
    let expected = [
        0.0,
        0.075_652_663_659_192_67,
        0.038_803_991_169_517,
        0.107_567_965_454_038_53,
        0.422_605_860_526_116_6,
        0.065_746_069_064_484_34,
        0.033_913_535_787_448_185,
        0.0,
        0.053_692_795_904_412_64,
        0.075_935_059_144_153_7,
        0.083_063_185_821_671_3,
        0.051_700_474_966_808_8,
        0.023_821_426_244_945_88,
        0.067_729_845_945_423_89,
        0.0,
        0.151_816_523_229_946,
        0.018_333_613_117_960_165,
        0.126_103_696_597_181_04,
        0.133_523_183_798_591_65,
        0.200_357_785_936_652_68,
        0.375_546_874_057_307_47,
        0.0,
        0.175_578_548_534_075_95,
        0.262_441_054_411_939_05,
        0.496_815_913_029_330_6,
        0.197_734_375_495_227_5,
        0.030_935_506_717_242_454,
        0.144_659_787_988_759_8,
        0.0,
        0.061_743_597_041_062_26,
        0.032_232_251_854_892_2,
        0.056_059_583_700_290_024,
        0.114_780_817_492_565_11,
        0.114_942_690_765_355_61,
        0.032_380_925_307_235_11,
        0.0,
    ];
    let cols: Vec<Vec<f64>> = (0..6)
        .map(|col| {
            (0..9).map(|row| f64::from((row * 7 + col * 3 + row * col) % 17 - 8) / 5.).collect()
        })
        .collect();
    let y: Vec<_> = cols[5].iter().enumerate().map(|(r, &v)| (r < 7).then_some(v)).collect();
    let mut prior = GraphPrior::default();
    prior.constraints.max_parents = Some(1);
    for seed in [95913, 99537, 104_991] {
        let out = from_columns(
            &cols[..5],
            &y,
            &variable_ids(6),
            &prior,
            seed,
            16384,
            &ExecutionContext::for_tests(seed),
        )
        .unwrap();
        let worst =
            out.edge_marginals.iter().zip(expected).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
        eprintln!(
            "six-variable enumerated forest reference, seed={seed}, draws={} worst edge error={worst}, log normalizer error={}",
            out.draws_used,
            out.log_evidence + 65.590_653_142_520_3
        );
        assert!(worst < 0.035);
        assert!((out.log_evidence + 65.590_653_142_520_3).abs() < 0.25);
    }
}
