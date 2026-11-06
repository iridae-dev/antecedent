//! Owed fixed-target interval measurements, separate from source-population coverage.
//! No calibration is earned by registering these tests.
mod common;
use antecedent::{RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::{AnalyticSeKind, GlmAdjustmentAte};
use antecedent_graph::{Dag, DenseNodeId};
use common::calibration::{CoverageTally, RecordKey, grid_n, map_replicates, n_sim};

fn measure(name: &'static str, se: AnalyticSeKind) {
    let per_cell = grid_n(120);
    let rows = map_replicates(n_sim(), |seed| {
        let ctx = ExecutionContext::for_tests(seed);
        let mut rng = ctx.rng.stream(0);
        let (mut t, mut z, mut y, mut w) = (vec![], vec![], vec![], vec![]);
        for (a, b, p) in [(0.0, 0.0, 0.25), (1.0, 0.0, 0.5), (0.0, 1.0, 0.4), (1.0, 1.0, 2.0 / 3.0)]
        {
            for _ in 0..per_cell {
                t.push(a);
                z.push(b);
                y.push(f64::from(rng.next_f64() < p));
                w.push(if b > 0.5 { 3.0 } else { 1.0 });
            }
        }
        let data = TabularData::from_f64_columns(vec![
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("z", z.as_slice()),
        ])
        .unwrap();
        let mut graph = Dag::with_variables(3);
        for (a, b) in [(2, 0), (2, 1), (0, 1)] {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let prepared = Study::tabular(data)
            .graph(graph)
            .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
            .estimator(GlmAdjustmentAte::new().with_bootstrap_replicates(0).with_se_kind(se))
            .refute(RefuteSuite::None)
            .build()
            .ok()?
            .prepare(&ctx)
            .ok()?;
        prepared.estimate_retained(&ctx).ok()?;
        let result = prepared.retarget(&w, &[VariableId::from_raw(2)], &ctx).ok()?;
        let contract = prepared.contract_for_result(&result).ok()?;
        let metadata = common::calibration_bind::constructions(&contract, &result);
        assert!(metadata.iter().all(|(c, _, _)| c.functional == "row_weights.mean"));
        let effect = result.estimate().as_effect()?;
        let interval = antecedent::PublishedScalarUncertainty::select(effect).interval(effect);
        Some((metadata, interval))
    });
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: name,
            dgp: "fixed_four_cell_logit_target_1_to_3",
            interval: "analytic_se",
        },
        0.95,
    );
    for row in rows {
        match row {
            Some((metadata, interval)) => {
                common::calibration_bind::bind_reported(&mut tally, &metadata);
                tally.record(interval, 21.0 / 80.0);
            }
            None => tally.skip(),
        }
    }
    tally.assert();
}
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn prepared_adjusted_fixed_target_fisher_interval() {
    measure("prepared_adjusted_fixed_target_fisher_interval", AnalyticSeKind::Homoskedastic);
}
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn prepared_adjusted_fixed_target_hc1_interval() {
    measure("prepared_adjusted_fixed_target_hc1_interval", AnalyticSeKind::Hc1);
}
