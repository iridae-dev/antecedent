//! Clamp-experiment constraint assembly against systems whose direct effects are known.
//!
//! `oracle_data/clamp_experiment_reference.py` (frozen in the JSON beside it) fixes a linear
//! cyclic system `x = B x + c`, draws the clamped columns of each regime (independently or
//! correlated) and computes the passive columns exactly in rational arithmetic with no
//! disturbance, so every experimental effect equals `(I - B_UU)^-1 B_UJ` and every row the
//! product assembles must be satisfied by `B`. The cases add a response offset of 1e9, columns
//! scaled over twelve decades (with `B` rescaled consistently) and nearly collinear clamps.

use antecedent_core::{Value, VariableId};
use antecedent_identify::experiment_moments::{
    RegimeSample, UnitColumn, assemble_experiment_constraints,
};
use antecedent_identify::experiment_regime::{
    DependencePremise, ExperimentRegime, PopulationId, RegimeContract, RegimeContractDecision,
    RegimeIntervention, SourceIdentity, UnitMask, validate_regime_contract,
};
use serde_json::Value as Json;

fn unit(raw: usize) -> VariableId {
    VariableId::from_raw(u32::try_from(raw).unwrap())
}

#[test]
fn every_assembled_row_is_satisfied_by_the_true_direct_effects() {
    let cases: Json =
        serde_json::from_str(include_str!("oracle_data/clamp_experiment_reference.json")).unwrap();
    for case in cases["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let n = case["units"].as_u64().unwrap() as usize;
        let direct: Vec<Vec<f64>> = case["direct"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
            .collect();
        let regimes_json = case["regimes"].as_array().unwrap();
        let regimes: Vec<ExperimentRegime> = regimes_json
            .iter()
            .enumerate()
            .map(|(k, r)| ExperimentRegime {
                targets: r["clamped"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|u| RegimeIntervention {
                        unit: unit(u.as_u64().unwrap() as usize),
                        value: Value::Int64(0),
                    })
                    .collect(),
                units: (0..n).map(unit).collect(),
                population: PopulationId::from_raw(1),
                premise: DependencePremise::Independence,
                source: SourceIdentity::from_raw(k as u64 + 1),
            })
            .collect();
        let RegimeContractDecision::Accepted(contract) =
            validate_regime_contract(&RegimeContract {
                regimes,
                mask: UnitMask { dropped: vec![] },
            })
        else {
            panic!("{name}: the regime contract was refused");
        };
        let data: Vec<Vec<Vec<f64>>> = regimes_json
            .iter()
            .map(|r| {
                r["columns"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|col| {
                        col.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
                    })
                    .collect()
            })
            .collect();
        let columns: Vec<Vec<UnitColumn<'_>>> = data
            .iter()
            .map(|cols| {
                cols.iter()
                    .enumerate()
                    .map(|(u, values)| UnitColumn { unit: unit(u), values })
                    .collect()
            })
            .collect();
        let samples: Vec<RegimeSample<'_>> = columns
            .iter()
            .enumerate()
            .map(|(k, cols)| RegimeSample {
                columns: cols,
                source: SourceIdentity::from_raw(k as u64 + 1),
            })
            .collect();
        let outcome = assemble_experiment_constraints(&contract, &samples, u64::MAX);
        if let Some(accepted) = case["refusal"].as_array() {
            let refusal = format!("{:?}", outcome.expect_err(name));
            assert!(
                accepted.iter().any(|a| refusal.starts_with(a.as_str().unwrap())),
                "{name}: {refusal}"
            );
            continue;
        }
        let assembled = outcome.unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(assembled.blocks.len(), n, "{name}: every target is covered");
        assert!(assembled.uncovered.is_empty(), "{name}");
        let tolerance = case["direct_tolerance"].as_f64().unwrap();
        let exact_tolerance = case["exact_tolerance"].as_f64().unwrap();
        let (mut worst_direct, mut worst_exact): (f64, f64) = (0.0, 0.0);
        let mut matched = 0;
        for block in &assembled.blocks {
            let target = block.target.raw() as usize;
            let width = block.sources.len();
            for (k, (row, rhs)) in block.rows.iter().zip(&block.rhs).enumerate() {
                let coefficients = &block.coefficients[k * width..(k + 1) * width];
                // The row is satisfied by the true direct effects.
                let mut total = 0.0;
                let mut magnitude = rhs.abs();
                for (coefficient, source) in coefficients.iter().zip(&block.sources) {
                    let term = coefficient * direct[target][source.raw() as usize];
                    total += term;
                    magnitude += term.abs();
                }
                let error = (total - rhs).abs() / magnitude.max(f64::MIN_POSITIVE);
                worst_direct = worst_direct.max(error);
                assert!(
                    error <= tolerance,
                    "{name}: target {target}, regime {}, probe {:?}: {total:.17e} vs {rhs:.17e}",
                    row.regime,
                    row.probe
                );
                // The row equals the exact computation on the binary64 data.
                let want = case["rows"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|w| {
                        w["regime"].as_u64().unwrap() as usize == row.regime
                            && w["probe"].as_u64().unwrap() == u64::from(row.probe.raw())
                            && w["target"].as_u64().unwrap() as usize == target
                    })
                    .unwrap_or_else(|| panic!("{name}: unexpected row"));
                matched += 1;
                let scale: f64 = want["coefficients"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap().abs())
                    .fold(want["rhs"].as_f64().unwrap().abs(), f64::max);
                for (got, w) in coefficients.iter().zip(want["coefficients"].as_array().unwrap()) {
                    let e = (got - w.as_f64().unwrap()).abs() / scale;
                    worst_exact = worst_exact.max(e);
                    assert!(e <= exact_tolerance, "{name}: coefficient {got:e} vs {w}");
                }
                let e = (rhs - want["rhs"].as_f64().unwrap()).abs() / scale;
                worst_exact = worst_exact.max(e);
                assert!(e <= exact_tolerance, "{name}: rhs {rhs:e} vs {}", want["rhs"]);
            }
        }
        assert_eq!(matched, case["rows"].as_array().unwrap().len(), "{name}: row count");
        eprintln!(
            "{name}: worst error {worst_direct:.2e} (true effects), {worst_exact:.2e} (exact on data)"
        );
    }
}
