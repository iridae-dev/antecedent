//! Survey-weighted discovery and downstream estimation on a row split of 3,000 participants and
//! ten variables.
//!
//! The study is a sparse connected linear-Gaussian model (`common::sparse_scm`) with design
//! weights of coefficient of variation 0.8 that depend on no outcome. Oracles: the effect is the
//! weighted least-squares coefficient of the treatment given the treatment's true parents on
//! the analysis rows, with its HC1 sandwich standard error, both solved here from the normal
//! equations; the weighted population's effect is the structural total effect, computed from
//! the coefficients alone; the effective sample size is the Kish size of the rows used.

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "carried weights are compared bit for bit; bounded simulation sizes and small indices"
)]

use std::sync::Arc;

use antecedent::analysis::sample_split_analysis::{
    AnalysisEffectRequest, SplitAnalysisEffect, SplitRule, discover_on_split, split_rows,
};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{OwnedColumnarStorage, TableView, TabularData, ValidityBitmap};
use antecedent_discovery::selective::{CiTestSpec, StaticAlgorithm, StaticDiscoveryContract};
use antecedent_io::measured_inference::split_parent_set_effect::treatment_parent_set;

#[path = "common/sparse_scm.rs"]
mod sparse_scm;

use sparse_scm::{Scm, Stream, Topology, invert};

const PARTICIPANTS: usize = 3_000;
const VARIABLES: usize = 10;
const WEIGHT_CV: f64 = 0.8;
const SPLIT_SEED: u64 = 31;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(SPLIT_SEED)
}

fn v(index: usize) -> VariableId {
    VariableId::from_raw(u32::try_from(index).expect("small index"))
}

/// Survey expansion weights `exp(s z - s^2 / 2)`, `s^2 = ln(1 + cv^2)`: mean one, coefficient
/// of variation `cv`, and a function of the design stream alone.
fn survey_weights(rows: usize, cv: f64, seed: u64) -> Vec<f64> {
    let sigma2 = (1.0 + cv * cv).ln();
    let mut stream = Stream::new(seed);
    (0..rows).map(|_| (sigma2.sqrt() * stream.normal() - sigma2 / 2.0).exp()).collect()
}

fn kish(weights: &[f64]) -> f64 {
    let total: f64 = weights.iter().sum();
    total * total / weights.iter().map(|w| w * w).sum::<f64>()
}

struct Study {
    scm: Scm,
    columns: Vec<Vec<f64>>,
}

fn study(seed: u64) -> Study {
    let scm = Scm::new(Topology::HubConfoundedMultiPath, VARIABLES, seed);
    assert!(scm.margin() >= sparse_scm::MIN_MARGIN, "margin {}", scm.margin());
    let columns = scm.simulate(PARTICIPANTS, seed ^ 0xDA7A);
    Study { scm, columns }
}

/// The study's table carrying `weights` as observation weights and, when given, the analysis
/// mask `keep` (`keep[i]` keeps row `i`).
fn weighted_table(study: &Study, weights: Vec<f64>, keep: Option<&[bool]>) -> TabularData {
    let plain = TabularData::from_f64_columns(
        study.scm.names.iter().map(String::as_str).zip(study.columns.iter().map(Vec::as_slice)),
    )
    .unwrap();
    let storage = plain.storage();
    let mask = keep.map(|keep| {
        let mut bytes = vec![0u8; keep.len().div_ceil(8)];
        for (i, kept) in keep.iter().enumerate() {
            if *kept {
                bytes[i / 8] |= 1 << (i % 8);
            }
        }
        ValidityBitmap::from_bytes(bytes, keep.len()).unwrap()
    });
    TabularData::new(
        OwnedColumnarStorage::try_new(
            storage.schema().clone(),
            storage.columns().to_vec(),
            mask,
            Some(Arc::from(weights)),
        )
        .unwrap(),
    )
}

fn contract(variables: usize) -> StaticDiscoveryContract {
    let mut contract = StaticDiscoveryContract::new(
        StaticAlgorithm::Pc,
        (0..variables).map(v).collect(),
        CiTestSpec::named("weighted_parcorr"),
        SPLIT_SEED,
    );
    contract.fdr = None;
    contract.constraints.alpha = 0.01;
    contract.constraints.max_cond_size = 4;
    contract
}

fn rule() -> SplitRule {
    SplitRule::SeededPermutation { seed: SPLIT_SEED, discovery_fraction: 0.5 }
}

/// Weighted least squares of `y` on an intercept, `x` and `adjustment` over the rows of
/// positive weight, weights normalized to mean one over those rows: the coefficient of `x` and
/// its HC1 sandwich standard error `(X'WX)^-1 [sum (w e)^2 x x'] (X'WX)^-1 n / (n - k)`.
fn weighted_ols(x: &[f64], y: &[f64], adjustment: &[Vec<f64>], weights: &[f64]) -> (f64, f64) {
    let rows: Vec<usize> = (0..x.len()).filter(|&i| weights[i] > 0.0).collect();
    let n = rows.len() as f64;
    let scale = n / rows.iter().map(|&i| weights[i]).sum::<f64>();
    let k = 2 + adjustment.len();
    let design = |i: usize| -> Vec<f64> {
        let mut row = vec![1.0, x[i]];
        row.extend(adjustment.iter().map(|column| column[i]));
        row
    };
    let mut gram = vec![vec![0.0; k]; k];
    let mut moment = vec![0.0; k];
    for &i in &rows {
        let (w, r) = (weights[i] * scale, design(i));
        for a in 0..k {
            moment[a] += w * r[a] * y[i];
            for b in 0..k {
                gram[a][b] += w * r[a] * r[b];
            }
        }
    }
    let bread = invert(&gram);
    let beta: Vec<f64> = (0..k).map(|a| (0..k).map(|b| bread[a][b] * moment[b]).sum()).collect();
    let mut meat = vec![vec![0.0; k]; k];
    for &i in &rows {
        let r = design(i);
        let fitted: f64 = r.iter().zip(&beta).map(|(a, b)| a * b).sum();
        let scaled = weights[i] * scale * (y[i] - fitted);
        for a in 0..k {
            for b in 0..k {
                meat[a][b] += scaled * scaled * r[a] * r[b];
            }
        }
    }
    let mut variance = 0.0;
    for a in 0..k {
        for b in 0..k {
            variance += bread[1][a] * meat[a][b] * bread[b][1];
        }
    }
    (beta[1], (variance * n / (n - k as f64)).sqrt())
}

/// The analysis rows' columns of the study, in the part's row order.
fn analysis_columns(analysis: &TabularData) -> Vec<Vec<f64>> {
    (0..VARIABLES).map(|i| analysis.float64_values(v(i)).unwrap()).collect()
}

fn request(parents: &[usize], scm: &Scm) -> AnalysisEffectRequest {
    AnalysisEffectRequest {
        treatment: v(scm.treatment),
        outcome: v(scm.outcome),
        adjustment: parents.iter().map(|&p| v(p)).collect(),
        levels: (0.0, 1.0),
        clusters: None,
    }
}

#[test]
fn weighted_discovery_and_downstream_estimation_equal_the_weighted_population_oracle() {
    let study = study(1_264_134);
    let weights = survey_weights(PARTICIPANTS, WEIGHT_CV, 77);
    let cv = {
        let mean = weights.iter().sum::<f64>() / PARTICIPANTS as f64;
        (weights.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / PARTICIPANTS as f64).sqrt()
            / mean
    };
    assert!((cv - WEIGHT_CV).abs() < 0.1, "design weights have coefficient of variation {cv}");
    let table = weighted_table(&study, weights.clone(), None);
    let split = discover_on_split(&contract(VARIABLES), &table, rule(), &ctx()).unwrap();

    // Weighted discovery on the discovery part finds the treatment's parents.
    let neighbourhood = treatment_parent_set(
        &split.outcome().structure,
        &study.scm.names,
        u32::try_from(study.scm.treatment).unwrap(),
        u32::try_from(study.scm.outcome).unwrap(),
    )
    .unwrap();
    let parents: Vec<usize> = neighbourhood.parents.iter().map(|&p| p as usize).collect();
    let mut truth_parents = study.scm.treatment_parents();
    truth_parents.sort_unstable();
    assert_eq!(parents, truth_parents);

    // The downstream estimate is the weighted-population oracle on the analysis rows.
    let effect: SplitAnalysisEffect =
        split.estimate_analysis_effect(&request(&parents, &study.scm), &ctx()).unwrap();
    let analysis = split.analysis_rows();
    let columns = analysis_columns(analysis);
    let analysis_weights = split.analysis_weights().unwrap().to_vec();
    let ids = split.analysis_row_ids();
    assert!(ids.iter().zip(&analysis_weights).all(|(&row, &w)| weights[row] == w));
    let adjustment: Vec<Vec<f64>> = parents.iter().map(|&p| columns[p].clone()).collect();
    let (slope, se) = weighted_ols(
        &columns[study.scm.treatment],
        &columns[study.scm.outcome],
        &adjustment,
        &analysis_weights,
    );
    assert!((effect.effect - slope).abs() < 1e-9, "{} vs {slope}", effect.effect);
    assert!((effect.standard_error - se).abs() < 1e-9, "{} vs {se}", effect.standard_error);

    // The weighted population's effect is the structural total effect (the weights depend on
    // no outcome), and the estimate covers it within four standard errors.
    let truth = study.scm.total_effect();
    assert!(
        (effect.effect - truth).abs() < 4.0 * effect.standard_error,
        "effect {} (se {}) vs truth {truth}",
        effect.effect,
        effect.standard_error
    );

    // Target population, normalization and effective sample size.
    assert!(SplitAnalysisEffect::TARGET_POPULATION.contains("weighted population"));
    assert!(SplitAnalysisEffect::TARGET_POPULATION.contains("selected graph"));
    let record = effect.sampling_weights.expect("a weighted analysis reports its record");
    assert_eq!(record.zero_weight_rows, 0);
    assert_eq!(record.rows_used, analysis.row_count() as u64);
    let size = kish(&analysis_weights);
    assert!((record.effective_sample_size - size).abs() < 1e-9 * size);
    assert!(
        size < 0.75 * analysis.row_count() as f64,
        "unequal weights carry less information than rows: {size} of {}",
        analysis.row_count()
    );
    assert!(
        (record.weight_sum - analysis_weights.iter().sum::<f64>()).abs() < 1e-9 * record.weight_sum
    );
    assert_eq!(effect.rows, analysis.row_count() as u64);

    // The interval is the sandwich standard error at the Student quantile of the effective
    // sample size less the regression's coefficients, and its coverage is labelled unmeasured.
    let coefficients = parents.len() + 2;
    assert_eq!(effect.coefficients, coefficients as u64);
    let interval = effect.interval(0.95).unwrap();
    assert!((interval.degrees_of_freedom - (size - coefficients as f64)).abs() < 1e-9 * size);
    let centre = f64::midpoint(interval.lower, interval.upper);
    assert!((centre - effect.effect).abs() < 1e-12);
    let half = (interval.upper - interval.lower) / 2.0;
    assert!(half > 1.96 * effect.standard_error && half < 2.1 * effect.standard_error);
    assert_eq!(interval.calibration, "unmeasured");
    assert!(effect.interval(1.0).is_err() && effect.interval(0.0).is_err());

    // Normalization: the estimate, its standard error and the effective size do not move when
    // every weight is rescaled.
    let scaled = weighted_table(&study, weights.iter().map(|w| w * 250.0).collect(), None);
    let rescaled = discover_on_split(&contract(VARIABLES), &scaled, rule(), &ctx()).unwrap();
    let again = rescaled.estimate_analysis_effect(&request(&parents, &study.scm), &ctx()).unwrap();
    assert!((again.effect - effect.effect).abs() < 1e-9);
    assert!((again.standard_error - effect.standard_error).abs() < 1e-9);
    let again_record = again.sampling_weights.unwrap();
    assert!(
        (again_record.effective_sample_size - record.effective_sample_size).abs() < 1e-9 * size
    );
    assert!((again_record.weight_sum - 250.0 * record.weight_sum).abs() < 1e-6 * record.weight_sum);
}

#[test]
fn zero_weight_rows_are_excluded_and_counted_and_unequal_weights_are_carried() {
    let study = study(9_983_411);
    let mut weights = survey_weights(PARTICIPANTS, WEIGHT_CV, 78);
    // 150 participants represent no one: their design weight is zero.
    let zeros: Vec<usize> = (0..PARTICIPANTS).step_by(20).collect();
    assert_eq!(zeros.len(), 150);
    for &row in &zeros {
        weights[row] = 0.0;
    }
    // 300 participants are outside the analysis mask.
    let mut keep = vec![true; PARTICIPANTS];
    for row in (7..PARTICIPANTS).step_by(10).take(300) {
        keep[row] = false;
    }
    assert_eq!(keep.iter().filter(|kept| !**kept).count(), 300);
    let table = weighted_table(&study, weights.clone(), Some(&keep));
    let split = discover_on_split(&contract(VARIABLES), &table, rule(), &ctx()).unwrap();
    assert_eq!(split.receipt().masked_out_rows.len(), 300);

    // Unequal weights are carried by gather, unnormalized, with the zeros in place.
    let ids = split.analysis_row_ids();
    let analysis_weights = split.analysis_weights().unwrap().to_vec();
    assert!(ids.iter().zip(&analysis_weights).all(|(&row, &w)| weights[row] == w));
    let positive: Vec<f64> = analysis_weights.iter().copied().filter(|w| *w > 0.0).collect();
    let zero_rows = analysis_weights.len() - positive.len();
    assert!(zero_rows > 0 && zero_rows < 150, "the zeros are split between the parts");
    let (low, high) = positive.iter().fold((f64::MAX, 0.0_f64), |a, &w| (a.0.min(w), a.1.max(w)));
    assert!(high / low > 5.0, "weights are unequal: {low} to {high}");
    let receipt = split.receipt();
    let recorded = receipt.weights.as_ref().expect("weights are recorded in the receipt");
    assert!(
        (recorded.analysis_effective_sample_size - kish(&positive)).abs() < 1e-9 * kish(&positive)
    );
    assert!(receipt.incomplete_rows.is_empty());

    // Zero-weight rows are excluded from the weighted estimate and counted. The adjustment set
    // is the treatment's true parents: this test is about the weights, not about discovery.
    let parents = study.scm.treatment_parents();
    let effect = split.estimate_analysis_effect(&request(&parents, &study.scm), &ctx()).unwrap();
    let record = effect.sampling_weights.expect("a weighted analysis reports its record");
    assert_eq!(record.zero_weight_rows, zero_rows as u64);
    assert_eq!(record.rows_used, positive.len() as u64);
    assert_eq!(
        record.rows_used + record.zero_weight_rows,
        split.analysis_rows().row_count() as u64
    );
    assert!((record.effective_sample_size - kish(&positive)).abs() < 1e-9 * kish(&positive));
    // The estimate is that of the rows of positive weight alone.
    let columns = analysis_columns(split.analysis_rows());
    let adjustment: Vec<Vec<f64>> = parents.iter().map(|&p| columns[p].clone()).collect();
    let (slope, se) = weighted_ols(
        &columns[study.scm.treatment],
        &columns[study.scm.outcome],
        &adjustment,
        &analysis_weights,
    );
    assert!((effect.effect - slope).abs() < 1e-9, "{} vs {slope}", effect.effect);
    assert!((effect.standard_error - se).abs() < 1e-9, "{} vs {se}", effect.standard_error);

    // A part whose rows all have weight zero has no population to describe: refused.
    let mut sparse_weights = vec![0.0; PARTICIPANTS];
    for (row, weight) in sparse_weights.iter_mut().enumerate().take(100) {
        *weight = 1.0 + (row % 3) as f64;
    }
    let only_discovery = weighted_table(&study, sparse_weights, None);
    let all: Vec<VariableId> = (0..VARIABLES).map(v).collect();
    let refusal = split_rows(
        &only_discovery,
        &all,
        SplitRule::Explicit { discovery_rows: (0..100).collect() },
    )
    .unwrap_err();
    assert_eq!(refusal.detail(), Some("selective_discovery.split_part_weight_invalid"));
}
