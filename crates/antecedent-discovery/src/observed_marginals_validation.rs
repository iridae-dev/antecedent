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
    assert!((out.edge_marginals[1] - 0.38202614).abs() < 0.02);
    assert!((out.edge_marginals[2] - 0.28189405).abs() < 0.02);
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
    assert!((r.edge_marginals[5] - 0.38202614).abs() < 0.02);
    assert!((r.edge_marginals[30] - 0.28189405).abs() < 0.02);
    assert!(r.edge_marginals.iter().enumerate().all(|(i, p)| i == 5 || i == 30 || *p == 0.0));
}

#[test]
fn observed_score_matches_six_variable_forest_quadrature() {
    // Independent reference enumerates the 16,807 forests, integrates both
    // missing values and every coefficient, and normalizes the graph weights.
    let expected = [
        0.0,
        0.07565266365919267,
        0.038803991169517,
        0.10756796545403853,
        0.4226058605261166,
        0.06574606906448434,
        0.033913535787448185,
        0.0,
        0.05369279590441264,
        0.0759350591441537,
        0.0830631858216713,
        0.0517004749668088,
        0.02382142624494588,
        0.06772984594542389,
        0.0,
        0.151816523229946,
        0.018333613117960165,
        0.12610369659718104,
        0.13352318379859165,
        0.20035778593665268,
        0.37554687405730747,
        0.0,
        0.17557854853407595,
        0.26244105441193905,
        0.4968159130293306,
        0.1977343754952275,
        0.030935506717242454,
        0.1446597879887598,
        0.0,
        0.06174359704106226,
        0.0322322518548922,
        0.056059583700290024,
        0.11478081749256511,
        0.11494269076535561,
        0.03238092530723511,
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
    for seed in [95913, 99537, 104991] {
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
            out.log_evidence + 65.5906531425203
        );
        assert!(worst < 0.035);
        assert!((out.log_evidence + 65.5906531425203).abs() < 0.25);
    }
}
