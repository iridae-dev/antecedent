//! Complete-procedure coverage of the discovered-PAG transport contrast interval.
//!
//! Every replicate runs the whole procedure on fresh data: FCI on the source rows, the
//! enumeration of the discovered PAG's members, the identification of the effect in every
//! member, the estimate over the source and target rows, and the percentile interval on the
//! contrast of a point answer. A replicate whose procedure refuses (the class is not identified
//! in every member, or the answer is a structural range) is scored as a non-report, so the
//! coverage is of the whole procedure including its discovery and the rate at which it reports.
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build). The interval stays
//! unbound to any record until this design is measured.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent::analysis::discovered_transport::{
    DiscoveredTransportError, discover_and_transport,
};
use antecedent::analysis::discovered_transport_estimate::{AnswerKind, TransportEvidence};
use antecedent::analysis::discovered_transport_temporal::{
    TemporalHorizon, estimate_discovered_temporal_transport,
    transport_discovered_temporal_structure,
};
use antecedent::analysis::discovered_transport_trial::{
    TrialCompatibilityOptions, TrialRandomization,
};
use antecedent_core::{
    ContinuousDomain, ExecutionContext, GridSpec, Lag, ResponseFunctional, ResponseQuery,
    TransportQuery, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_discovery::selective::{
    CiTestSpec, StaticAlgorithm, StaticDiscoveryContract, TemporalAlgorithm,
    TemporalDiscoveryContract, TemporalSelectiveDiscovery,
};
use antecedent_stats::ci::{
    CiBatchRequest, CiWorkspace, ConditionalIndependenceTest, GSquared, SignificanceMethod,
    SymbolicCmi,
};
use antecedent_stats::{FdrAdjustment, MultipleTestingMethod, SparseDiscreteCi};
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, map_replicates,
    n_sim, smoke, stream_seed, unit_uniform, validation_seed,
};

const INTERVAL: &str = "percentile_bootstrap_or_logical_support";
const REPLICATES: u32 = 200;

fn discrete_ci() -> CiTestSpec {
    CiTestSpec::declared(
        "gsquared_analytic_sparse_fallback_symbolic_cmi_199_within_stratum_permutations",
        Arc::new(SparseDiscreteCi::new(199).unwrap()),
    )
}

fn static_discrete_ci() -> CiTestSpec {
    CiTestSpec::declared(
        "gsquared_analytic_sparse_fallback_symbolic_cmi_1023_within_stratum_permutations",
        Arc::new(SparseDiscreteCi::new(1023).unwrap()),
    )
}

#[test]
fn sparse_discrete_ci_uses_the_declared_permutation_null() {
    use antecedent_stats::SparseDiscreteCi;
    use antecedent_stats::ci::{CiPreparationPlan, CiQuery, ConfidenceMethod};
    let x = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let y = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    let columns = [&x[..], &y[..]];
    let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
    let request = CiBatchRequest {
        columns: &columns,
        queries: &queries,
        z_flat: &[],
        significance: SignificanceMethod::Analytic,
        confidence: ConfidenceMethod::None,
    };
    let ctx = ExecutionContext::for_tests(71);
    let provider = SparseDiscreteCi::new(199).unwrap();
    let prepared = provider.prepare(&columns, &CiPreparationPlan::default(), &ctx).unwrap();
    assert!(
        GSquared::new().test_batch(&prepared, &request, &mut CiWorkspace::default(), &ctx).is_err()
    );
    let actual =
        provider.test_batch(&prepared, &request, &mut CiWorkspace::default(), &ctx).unwrap();
    let mut permutation = prepared;
    permutation.plan.significance =
        SignificanceMethod::BlockShuffle { replicates: 199, block_size: 1 };
    let expected = SymbolicCmi::new()
        .test_batch(&permutation, &request, &mut CiWorkspace::default(), &ctx)
        .unwrap();
    assert_eq!(actual.results[0].statistic.to_bits(), expected.results[0].statistic.to_bits());
    assert_eq!(actual.results[0].p_value.to_bits(), expected.results[0].p_value.to_bits());
}

/// Why a replicate reported no interval, as a short stable label.
fn refusal_label(error: &DiscoveredTransportError) -> String {
    if std::env::var_os("ANTECEDENT_TRANSPORT_TRACE_REFUSAL").is_some() {
        eprintln!("transport refusal detail: {error:?}");
    }
    match error {
        DiscoveredTransportError::Refused(refusal) => refusal.detail.to_owned(),
        other => other.to_string().chars().take(96).collect(),
    }
}

/// Why a discovery run failed, as a short label.
fn discovery_label(error: &impl std::fmt::Debug) -> String {
    format!("discovery: {error:?}").chars().take(96).collect()
}

/// Counts of the reasons replicates reported nothing, printed so a coverage shortfall that comes
/// from the rate of reporting is told apart from one that comes from the intervals.
#[derive(Default)]
struct NonReports {
    reasons: BTreeMap<String, u32>,
    widths: Vec<f64>,
    total: usize,
}

impl NonReports {
    /// Score one replicate's outcome, keeping the reason it reported nothing.
    fn interval(&mut self, outcome: Result<(f64, f64), String>) -> Option<(f64, f64)> {
        self.total += 1;
        match outcome {
            Ok(interval)
                if interval.0.is_finite() && interval.1.is_finite() && interval.0 <= interval.1 =>
            {
                self.widths.push(interval.1 - interval.0);
                Some(interval)
            }
            Ok(_) => {
                *self.reasons.entry("invalid_interval".into()).or_default() += 1;
                None
            }
            Err(reason) => {
                *self.reasons.entry(reason).or_default() += 1;
                None
            }
        }
    }

    fn print(&self, test: &str, effect: f64, manifest_point: bool) {
        eprintln!("{test}: replicates that reported nothing, by reason: {:?}", self.reasons);
        let mut widths = self.widths.clone();
        widths.sort_by(f64::total_cmp);
        let median = widths.get(widths.len() / 2).copied().unwrap_or(f64::NAN);
        #[allow(clippy::cast_precision_loss, reason = "bounded simulation replicate counts")]
        let reporting_rate = widths.len() as f64 / self.total as f64;
        let width_over_effect = median / effect.abs();
        eprintln!(
            "{test}: reporting_rate={reporting_rate:.6} median_width={median:.6} \
             median_width_over_effect={width_over_effect:.6} manifest_point={manifest_point}"
        );
        if manifest_point && !smoke() {
            assert!(reporting_rate >= 0.9, "frozen manifest reporting rate must be at least .9");
            assert!(
                width_over_effect.is_finite() && width_over_effect <= 1.5,
                "frozen manifest median width must be at most 1.5 times the true effect"
            );
        }
    }
}

/// The interval of a point answer, or why the replicate reported none.
fn point_interval(
    estimate: &antecedent::analysis::discovered_transport_estimate::DiscoveredTransportEstimate,
) -> Result<(f64, f64), String> {
    let answer = estimate.answer();
    if answer.kind != AnswerKind::Point {
        return Err("structural_range".to_owned());
    }
    answer
        .interval
        .map(|i| (i.low, i.high))
        .ok_or_else(|| format!("no_interval:{:?}", answer.interval_reason))
}

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn bern(seed: u64, p: f64) -> f64 {
    f64::from(unit_uniform(seed) < p)
}

/// `P(y = 1 | x, z, L)` of the structural model `x -> y <- z <- {c, d}`, `y -> r`, `z -> q`,
/// with a latent `L -> w` and `L -> y`.
fn p_y(x: f64, z: f64, l: f64) -> f64 {
    0.1 + 0.3 * x + 0.2 * z + 0.2 * l + 0.15 * x * z
}

/// `P(z = 1 | c, d)`: the mechanism that differs between the populations.
fn p_z(c: f64, d: f64, target: bool) -> f64 {
    if target { 0.85 - 0.3 * c - 0.3 * d } else { 0.15 + 0.3 * c + 0.3 * d }
}

fn study(rows: usize, target: bool, seed: u64) -> TabularData {
    let mut columns: Vec<Vec<f64>> = (0..8).map(|_| Vec::with_capacity(rows)).collect();
    for i in 0..rows as u64 {
        let draw = |k: u64| stream_seed(seed, i * 16 + k);
        let (c, d, l, x) =
            (bern(draw(0), 0.5), bern(draw(1), 0.5), bern(draw(2), 0.5), bern(draw(3), 0.5));
        let z = bern(draw(4), p_z(c, d, target));
        let w = bern(draw(5), 0.2 + 0.6 * l);
        let y = bern(draw(6), p_y(x, z, l));
        let q = bern(draw(7), 0.1 + 0.7 * z);
        let r = bern(draw(8), 0.2 + 0.6 * y);
        for (column, value) in columns.iter_mut().zip([x, y, z, c, d, w, q, r]) {
            column.push(value);
        }
    }
    let names: Vec<String> = (0..8).map(|i| format!("v{i}")).collect();
    TabularData::from_f64_columns(
        names.iter().map(String::as_str).zip(columns.iter().map(Vec::as_slice)),
    )
    .unwrap()
}

/// `E*[y | do(x = a)]` by enumerating the target population's structural model.
fn truth(a: f64) -> f64 {
    let mut total = 0.0;
    for c in [0.0, 1.0] {
        for d in [0.0, 1.0] {
            for l in [0.0, 1.0] {
                for z in [0.0, 1.0] {
                    let pz = p_z(c, d, true);
                    total += 0.125 * if z > 0.5 { pz } else { 1.0 - pz } * p_y(a, z, l);
                }
            }
        }
    }
    total
}

fn construction() -> Construction {
    Construction {
        query: "DiscoveredTransport".into(),
        graph_class: "Pag".into(),
        structure: "discovered".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: "transport.empirical_table_plugin".into(),
        interval_method: INTERVAL.into(),
        se_kind: "percentile_or_logical_support".into(),
        dependence: "iid".into(),
        posterior: String::new(),
        functional: "target_interventional_contrast.fci_cutoff_0_05_sparse_1023.trial_compatibility_fisher_exact_alpha_0_01.declared_selection_joint_pretreatment_standardization.bounded_outcome_confidence_set"
            .into(),
        identification: "discovered_class".into(),
        reported_level: REPORTED_LEVEL,
    }
}

fn query() -> TransportQuery {
    TransportQuery::new(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: v(1),
            treatment: ContinuousDomain::new(v(0), GridSpec::Values(Arc::from([0.0, 1.0]))),
        }),
        "trial",
        "clinic",
        [v(0)],
    )
}

/// Actual production confidence set and its separately retained point-identification status.
struct PagConfidenceInterval {
    bounds: (f64, f64),
    identified_point: bool,
    logical_reason: Option<String>,
}

/// The whole static procedure on one replicate, including scientifically unresolved points.
fn pag_interval(
    contract: &StaticDiscoveryContract,
    source: &TabularData,
    target: &TabularData,
    ctx: &ExecutionContext,
) -> Result<PagConfidenceInterval, String> {
    let run = discover_and_transport(contract, source, &[v(2)], &query(), 256, ctx)
        .map_err(|error| discovery_label(&error))?;
    let report = run.transport.map_err(|error| refusal_label(&error))?;
    let mut evidence = TransportEvidence::new(source, target);
    evidence.bootstrap_replicates = REPLICATES;
    evidence.limits.operations = 2_000_000_000;
    let set = antecedent::analysis::discovered_transport_trial::estimate_trial_compatible_transport_confidence_set(
        &report,
        &query(),
        &evidence,
        TrialCompatibilityOptions {
            alpha: 0.01,
            randomization: TrialRandomization::CompletelyRandomized,
        },
        antecedent::analysis::discovered_transport_trial::TransportOutcomeSupport {
            lower: 0.0,
            upper: 1.0,
        },
        ctx,
    )
    .map_err(|error| refusal_label(&error))?;
    if let Some(estimate) = set
        .retained_estimate()
        .filter(|_| std::env::var_os("ANTECEDENT_TRANSPORT_TRACE_TRIAL").is_some())
    {
        eprintln!(
            "trial compatibility: p={}, retained={:?}, members={:?}",
            estimate.compatibility().p_value,
            estimate.compatibility().retained_members(),
            estimate.estimate().members().iter().map(|m| match m {
                antecedent::analysis::discovered_transport_estimate::MemberOutcome::Estimated(e) => format!("{} => {:?}",e.expression,e.means),
                other => format!("{other:?}"),
            }).collect::<Vec<_>>()
        );
    }
    Ok(PagConfidenceInterval {
        bounds: set.bounds(),
        identified_point: set.kind()
            == antecedent::analysis::discovered_transport_trial::TrialConfidenceSetKind::IdentifiedSamplingInterval,
        logical_reason: set.logical_reason().map(|reason| format!("{reason:?}")),
    })
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn discovered_pag_transport_contrast_complete_procedure_coverage() {
    let n = grid_n(4000);
    let effect = truth(1.0) - truth(0.0);
    let mut non_reports = NonReports::default();
    let mut identified_points = 0_u32;
    let mut logical_bounds = BTreeMap::<String, u32>::new();
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "discovered_pag_transport_contrast_complete_procedure_coverage",
            dgp: "latent_confounded_selection_on_z",
            interval: INTERVAL,
        },
        REPORTED_LEVEL,
    )
    .labelled(
        "fci_cutoff_0_05_sparse_1023_trial_compatibility_fisher_exact_alpha_0_01_declared_selection_joint_pretreatment_standardization_bounded_outcome_confidence_set",
    );
    for (rep, outcome) in map_replicates(n_sim(), |rep| {
        let seed = rep.wrapping_mul(1_000_003);
        let source = study(n, false, seed);
        let target = study(n, true, seed ^ 0x5eed);
        let algorithm_seed = validation_seed(rep);
        let ctx = ExecutionContext::for_tests(algorithm_seed);
        let mut contract = StaticDiscoveryContract::new(
            StaticAlgorithm::Fci,
            (0..8).map(VariableId::from_raw).collect(),
            static_discrete_ci(),
            algorithm_seed,
        );
        // FCI's ordinary nominal independence-test cutoff preserves power for indirect
        // relations. Dividing it by the number of pairs instead encourages false
        // independences and collider orientations. Whole-procedure coverage is measured
        // separately; this cutoff makes no graph-wide error-control claim.
        contract.constraints.alpha = 0.05;
        let interval = pag_interval(&contract, &source, &target, &ctx);
        (rep, interval)
    }) {
        let _ = rep;
        let outcome = outcome.map(|set| {
            if set.identified_point {
                identified_points += 1;
            } else if let Some(reason) = set.logical_reason {
                *logical_bounds.entry(reason).or_default() += 1;
            }
            set.bounds
        });
        let outcome = non_reports.interval(outcome);
        tally.bind(
            &construction(),
            ScopeFacts {
                row_count: n as u64,
                replicates_ok: Some(REPLICATES),
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
        );
        tally.record(outcome, effect);
    }
    // Logical support bounds are confidence sets, never identified-point reports.
    // Include their full widths in the original usefulness metric.
    non_reports.widths.sort_by(f64::total_cmp);
    let median = non_reports.widths.get(non_reports.widths.len() / 2).copied().unwrap_or(f64::NAN);
    let reporting_rate = f64::from(identified_points) / f64::from(tally.attempts());
    let width_over_effect = median / effect.abs();
    eprintln!(
        "discovered_pag_transport_contrast_complete_procedure_coverage: \
         identified_points={identified_points} logical_bounds={logical_bounds:?} \
         errors={:?} reporting_rate={reporting_rate:.6} median_width={median:.6} \
         median_width_over_effect={width_over_effect:.6}",
        non_reports.reasons,
    );
    if !smoke() {
        assert!(reporting_rate >= 0.9, "frozen identified-point reporting floor is .9");
        assert!(
            width_over_effect.is_finite() && width_over_effect <= 1.5,
            "frozen median confidence-set width is at most 1.5 times the true effect",
        );
    }
    tally.assert();
}

fn p_y_lagged(x: f64, z: f64) -> f64 {
    0.1 + 0.4 * x + 0.3 * z + 0.1 * x * z
}

/// A binary series: `z` autoregressive with a population-specific mechanism, `x` randomized,
/// `y_t` caused by `x_{t-1}` and `z_{t-1}`.
fn series(steps: usize, target: bool, seed: u64) -> (TimeSeriesData, TabularData) {
    let (mut xs, mut ys, mut zs) = (vec![0.0], vec![0.0], vec![0.0]);
    for t in 1..steps {
        let draw = |k: u64| stream_seed(seed, t as u64 * 4 + k);
        let pz = if target { 0.6 - 0.4 * zs[t - 1] } else { 0.2 + 0.5 * zs[t - 1] };
        zs.push(bern(draw(0), pz));
        xs.push(bern(draw(1), 0.5));
        ys.push(bern(draw(2), p_y_lagged(xs[t - 1], zs[t - 1])));
    }
    let columns = [("v0", xs.as_slice()), ("v1", ys.as_slice()), ("v2", zs.as_slice())];
    (
        TimeSeriesData::from_f64_columns(columns, 1_000_000_000).unwrap(),
        TabularData::from_f64_columns(columns).unwrap(),
    )
}

/// The whole temporal procedure on one replicate: the class PCMCI+ returns on the source series,
/// the temporal class at `horizon`, identification in every member, the windowed estimate and
/// its block-bootstrap interval; or why the replicate reported none.
fn temporal_interval(
    contract: &TemporalDiscoveryContract,
    source: &TimeSeriesData,
    source_table: &TabularData,
    target_table: &TabularData,
    horizon: TemporalHorizon,
    ctx: &ExecutionContext,
) -> Result<(f64, f64), String> {
    let mut adapter = TemporalSelectiveDiscovery::new();
    adapter.run(contract, source, ctx).map_err(|error| discovery_label(&error))?;
    let outcome = adapter.outcome().ok_or_else(|| "discovery: no outcome".to_owned())?;
    let report = transport_discovered_temporal_structure(outcome, horizon, &[v(2)], &query(), 256)
        .map_err(|error| refusal_label(&error))?;
    let mut evidence = TransportEvidence::new(source_table, target_table);
    evidence.bootstrap_replicates = REPLICATES;
    evidence.limits.operations = 2_000_000_000;
    let estimate = estimate_discovered_temporal_transport(&report, &query(), &evidence, ctx)
        .map_err(|error| refusal_label(&error))?;
    point_interval(&estimate)
}

fn temporal_construction() -> Construction {
    Construction {
        graph_class: "TemporalCpdag".into(),
        dependence: "block".into(),
        interval_method: "block_percentile_bootstrap".into(),
        functional: "target_interventional_contrast.pcmci_plus_full_allowed_edge_holm_including_contemporaneous.v1".into(),
        ..construction()
    }
}

/// Complete-procedure coverage of the temporal contrast interval: PCMCI+ on the source series,
/// the temporal class at horizon one, identification in every member, the estimate over the
/// windowed tables and the circular block bootstrap interval, in every replicate.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn discovered_temporal_transport_contrast_complete_procedure_coverage() {
    let steps = grid_n(4000);
    let pi = 0.6 / 1.4;
    let truth = |a: f64| (1.0 - pi) * p_y_lagged(a, 0.0) + pi * p_y_lagged(a, 1.0);
    let effect = truth(1.0) - truth(0.0);
    let mut non_reports = NonReports::default();
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "discovered_temporal_transport_contrast_complete_procedure_coverage",
            dgp: "lagged_selection_on_autoregressive_z",
            interval: "block_percentile_bootstrap",
        },
        REPORTED_LEVEL,
    )
    .labelled("full_allowed_edge_holm_including_contemporaneous_v1");
    for outcome in map_replicates(n_sim(), |rep| {
        let seed = rep.wrapping_mul(1_000_003);
        let (source, source_table) = series(steps, false, seed);
        let (_, target_table) = series(steps, true, seed ^ 0x5eed);
        let algorithm_seed = validation_seed(rep);
        let ctx = ExecutionContext::for_tests(algorithm_seed);
        let mut contract = TemporalDiscoveryContract::new(
            TemporalAlgorithm::PcmciPlus,
            (0..3).map(VariableId::from_raw).collect(),
            discrete_ci(),
            algorithm_seed,
        );
        contract.fdr = Some(FdrAdjustment {
            method: MultipleTestingMethod::Holm,
            exclude_contemporaneous: false,
        });
        temporal_interval(
            &contract,
            &source,
            &source_table,
            &target_table,
            TemporalHorizon { horizon_steps: 1, history: None },
            &ctx,
        )
    }) {
        let outcome = non_reports.interval(outcome);
        tally.bind(
            &temporal_construction(),
            ScopeFacts {
                row_count: steps as u64,
                replicates_ok: Some(REPLICATES),
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
        );
        tally.record(outcome, effect);
    }
    non_reports.print(
        "discovered_temporal_transport_contrast_complete_procedure_coverage",
        effect,
        false,
    );
    tally.assert();
}

/// Six binary variables with two lags. `x` (variable 0) is randomized; `y` (1) is caused by
/// `x_{t-2}` and `z_{t-2}`; `z` (2) is autoregressive with a population-specific mechanism and is
/// the selected cause; `w` (3) is a nuisance autoregressive at lag two; `q` (4) is caused by
/// `z_{t-1}`; `r` (5) is caused by `y_{t-1}`. The template's largest lag is two.
fn series_two_lags(steps: usize, target: bool, seed: u64) -> (TimeSeriesData, TabularData) {
    let mut columns: Vec<Vec<f64>> = (0..6).map(|_| vec![0.0; 2]).collect();
    for t in 2..steps {
        let draw = |k: u64| stream_seed(seed, t as u64 * 8 + k);
        let (x, y, z, w) = (0, 1, 2, 3);
        let pz = if target { 0.6 - 0.4 * columns[z][t - 1] } else { 0.2 + 0.5 * columns[z][t - 1] };
        let values = [
            bern(draw(1), 0.5),
            bern(draw(2), p_y_lagged(columns[x][t - 2], columns[z][t - 2])),
            bern(draw(0), pz),
            bern(draw(3), 0.2 + 0.5 * columns[w][t - 2]),
            bern(draw(4), 0.1 + 0.7 * columns[z][t - 1]),
            bern(draw(5), 0.2 + 0.6 * columns[y][t - 1]),
        ];
        for (column, value) in columns.iter_mut().zip(values) {
            column.push(value);
        }
    }
    let named: Vec<(String, &[f64])> =
        columns.iter().enumerate().map(|(i, c)| (format!("v{i}"), c.as_slice())).collect();
    let pairs = || named.iter().map(|(name, column)| (name.as_str(), *column));
    (
        TimeSeriesData::from_f64_columns(pairs(), 1_000_000_000).unwrap(),
        TabularData::from_f64_columns(pairs()).unwrap(),
    )
}

/// Complete-procedure coverage of the temporal contrast interval on six variables with two
/// lags: PCMCI+ (maximum lag two) on the source series, the temporal class at horizon two,
/// identification in every member, the estimate over the windowed tables and the circular block
/// bootstrap interval, in every replicate.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn discovered_temporal_transport_two_lag_six_variable_complete_procedure_coverage() {
    let steps = grid_n(4000);
    let pi = 0.6 / 1.4;
    let truth = |a: f64| (1.0 - pi) * p_y_lagged(a, 0.0) + pi * p_y_lagged(a, 1.0);
    let effect = truth(1.0) - truth(0.0);
    let mut non_reports = NonReports::default();
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "discovered_temporal_transport_two_lag_six_variable_complete_procedure_coverage",
            dgp: "lag_two_selection_on_autoregressive_z_six_variables",
            interval: "block_percentile_bootstrap",
        },
        REPORTED_LEVEL,
    )
    .labelled("full_allowed_edge_holm_including_contemporaneous_v1");
    for outcome in map_replicates(n_sim(), |rep| {
        let seed = rep.wrapping_mul(1_000_003);
        let (source, source_table) = series_two_lags(steps, false, seed);
        let (_, target_table) = series_two_lags(steps, true, seed ^ 0x5eed);
        let algorithm_seed = validation_seed(rep);
        let ctx = ExecutionContext::for_tests(algorithm_seed);
        let mut contract = TemporalDiscoveryContract::new(
            TemporalAlgorithm::PcmciPlus,
            (0..6).map(VariableId::from_raw).collect(),
            discrete_ci(),
            algorithm_seed,
        );
        contract.constraints.temporal.max_lag = Lag::from_raw(2);
        contract.fdr = Some(FdrAdjustment {
            method: MultipleTestingMethod::Holm,
            exclude_contemporaneous: false,
        });
        temporal_interval(
            &contract,
            &source,
            &source_table,
            &target_table,
            TemporalHorizon { horizon_steps: 2, history: None },
            &ctx,
        )
    }) {
        let outcome = non_reports.interval(outcome);
        tally.bind(
            &temporal_construction(),
            ScopeFacts {
                row_count: steps as u64,
                replicates_ok: Some(REPLICATES),
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
        );
        tally.record(outcome, effect);
    }
    non_reports.print(
        "discovered_temporal_transport_two_lag_six_variable_complete_procedure_coverage",
        effect,
        steps == 4000,
    );
    tally.assert();
}
