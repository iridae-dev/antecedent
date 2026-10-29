//! Period refresh of one autoregressive mechanism, least squares and conjugate posterior on the
//! same rows. Reference values are numpy fits of the rows a fresh fit would use: design
//! `[1, v_{t-1}, v_{t-2}]`, prior `m₀ = 0`, `Λ₀ = diag(0.5, 0.4, 0.3)`, `InvGamma(3, 2)`.

use std::collections::BTreeSet;

use antecedent::period_refresh::{
    PeriodRefresh, PeriodRefreshOutcome, PeriodRefreshSpec, SummaryOnlyFit,
};
use antecedent_estimate::temporal_refresh::{
    MechanismStanding, ObservationCadence, ObservationPeriod, TemporalWindowIdentity, WindowProduct,
};
use antecedent_state::ConjugateOlsPrior;

const S: [f64; 18] =
    [0.3, 0.9, 0.4, 1.1, 0.8, 1.5, 1.0, 1.6, 1.2, 1.9, 1.1, 1.7, 1.5, 2.0, 1.4, 2.2, 1.6, 2.1];

fn set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

fn window(start: i64, end: i64, snapshot: &str, units: &[&str]) -> TemporalWindowIdentity {
    TemporalWindowIdentity {
        graph_id: "two_slice_graph".into(),
        horizon: 2,
        lag_alignment: vec![("v".into(), 1)],
        intervention_history: vec!["a1=1".into(), "a2=0".into()],
        selection_targets: set(&["s_t1"]),
        regimes: set(&["observational"]),
        period: ObservationPeriod { start, end },
        unit_ids: set(units),
        snapshot_id: snapshot.into(),
        proof_id: "proof_1".into(),
    }
}

fn units() -> [&'static str; 2] {
    ["u1", "u2"]
}

fn prior() -> ConjugateOlsPrior {
    ConjugateOlsPrior::new(
        vec![0.0; 3],
        vec![0.5, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.3],
        3.0,
        2.0,
    )
    .unwrap()
}

fn spec(standing: MechanismStanding, initial_law: &'static str) -> PeriodRefreshSpec<'static> {
    PeriodRefreshSpec {
        series_id: "v",
        cadence: ObservationCadence::new(1).unwrap(),
        initial_law,
        standing,
        lags: 2,
    }
}

fn cadence() -> ObservationCadence {
    ObservationCadence::new(1).unwrap()
}

fn start(standing: MechanismStanding) -> PeriodRefresh {
    let first = window(0, 6, "snap_1", &units());
    PeriodRefresh::start(&spec(standing, "law_1"), &first, &prior(), &S[0..6]).unwrap()
}

fn close(actual: f64, expected: f64, label: &str) {
    assert!((actual - expected).abs() <= 1e-9, "{label}: {actual:.17e} vs {expected:.17e}");
}

fn assert_fit(
    refresh: &PeriodRefresh,
    n: u64,
    beta: [f64; 3],
    sigma2: f64,
    posterior_mean: [f64; 3],
    shape: f64,
    rate: f64,
) {
    let fit = refresh.fit().snapshot();
    assert_eq!(fit.n, n);
    for (got, want) in fit.beta.as_ref().unwrap().iter().zip(beta) {
        close(*got, want, "beta");
    }
    close(fit.residual_variance.unwrap(), sigma2, "residual variance");
    let summary = refresh.summary().unwrap();
    assert_eq!(summary.n, n);
    for (got, want) in summary.mean.iter().zip(posterior_mean) {
        close(*got, want, "posterior mean");
    }
    close(summary.shape, shape, "shape");
    close(summary.rate, rate, "rate");
    refresh.check_against_fresh().expect("cached results equal a fresh reconstruction");
}

#[test]
fn the_first_period_equals_the_reference_fit() {
    let refresh = start(MechanismStanding::DeclaredInvariant);
    assert_fit(
        &refresh,
        4,
        [-0.391_764_705_882_354_8, 0.505_882_352_941_178_2, 1.388_235_294_117_647_5],
        0.016_000_000_000_000_004,
        [0.227_294_251_819_161_9, 0.161_648_216_787_117_94, 0.837_000_582_627_775_3],
        5.0,
        2.176_386_592_123_368_5,
    );
}

#[test]
fn a_declared_invariant_mechanism_extends_across_periods_like_one_fit() {
    let mut refresh = start(MechanismStanding::DeclaredInvariant);
    let second = window(6, 12, "snap_2", &units());
    let outcome = refresh.refresh(&second, cadence(), "law_1", &S[6..12]).unwrap();
    assert_eq!(outcome, PeriodRefreshOutcome::Extended { contiguous: true });
    assert_fit(
        &refresh,
        10,
        [0.475_785_721_036_215_36, -0.120_816_750_847_937_65, 0.834_722_936_858_797_1],
        0.035_272_832_826_045_5,
        [0.351_587_156_825_020_3, 0.004_469_720_612_162_134, 0.799_713_370_682_002_2],
        8.0,
        2.265_297_812_709_253,
    );
    assert_eq!(refresh.fit().recomputes(), 0, "an append is not a rebuild");
    let third = window(12, 18, "snap_3", &units());
    refresh.refresh(&third, cadence(), "law_1", &S[12..18]).unwrap();
    assert_fit(
        &refresh,
        16,
        [0.401_256_185_763_230_36, -0.050_715_938_039_882_38, 0.863_620_391_789_404_9],
        0.034_823_352_396_483_49,
        [0.327_808_087_109_398_9, 0.022_038_223_784_948_335, 0.835_365_507_854_59],
        11.0,
        2.367_227_811_489_82,
    );
}

#[test]
fn an_undeclared_mechanism_restarts_on_the_new_period_under_the_stated_prior() {
    let mut refresh = start(MechanismStanding::Undeclared);
    let second = window(6, 12, "snap_2", &units());
    let outcome = refresh.refresh(&second, cadence(), "law_1", &S[6..12]).unwrap();
    let PeriodRefreshOutcome::Restarted { invalidated } = outcome else {
        panic!("an undeclared mechanism must not carry its fit across periods");
    };
    assert!(invalidated.contains(&WindowProduct::Model));
    assert!(invalidated.contains(&WindowProduct::Design));
    assert!(invalidated.contains(&WindowProduct::Evidence));
    assert!(!invalidated.contains(&WindowProduct::Identification));
    assert_fit(
        &refresh,
        4,
        [2.467_974_683_544_098_5, -0.829_113_924_050_557_5, 0.146_835_443_038_041_72],
        0.063_835_443_037_974_59,
        [0.249_983_037_430_736_55, 0.084_247_427_343_661_53, 0.752_007_237_362_885_9],
        5.0,
        2.188_748_727_807_305,
    );
    assert_eq!(refresh.fit().ledger().rows, 4, "the old period's rows are gone");
}

#[test]
fn a_gap_is_not_filled_and_only_the_new_periods_own_rows_enter() {
    let mut refresh = start(MechanismStanding::DeclaredInvariant);
    let later = window(8, 14, "snap_later", &units());
    let outcome = refresh.refresh(&later, cadence(), "law_1", &S[8..14]).unwrap();
    assert_eq!(outcome, PeriodRefreshOutcome::Extended { contiguous: false });
    assert_fit(
        &refresh,
        8,
        [0.367_051_388_559_222_6, -0.006_186_209_352_005_272, 0.839_737_123_190_124_4],
        0.066_312_277_474_483_73,
        [0.297_058_773_345_710_7, 0.095_566_831_945_017_68, 0.776_357_233_288_173_9],
        7.0,
        2.290_711_811_657_754_3,
    );
}

#[test]
fn population_drift_restarts_even_when_the_mechanism_was_declared_invariant() {
    let mut refresh = start(MechanismStanding::DeclaredInvariant);
    let drifted = window(12, 18, "snap_drift", &["u1", "u3"]);
    let outcome = refresh.refresh(&drifted, cadence(), "law_1", &S[12..18]).unwrap();
    let PeriodRefreshOutcome::Restarted { invalidated } = outcome else {
        panic!("population drift invalidates the model");
    };
    assert!(invalidated.contains(&WindowProduct::Model));
    assert_fit(
        &refresh,
        4,
        [1.863_005_780_346_729_5, -0.494_219_653_179_161_9, 0.479_768_786_127_189_7],
        0.066_589_595_375_722_58,
        [0.178_415_482_886_833_15, 0.108_244_351_225_093_81, 0.805_319_858_899_799_3],
        5.0,
        2.163_647_869_196_301,
    );
}

#[test]
fn a_changed_initial_law_or_cadence_restarts_a_declared_invariant_mechanism() {
    for (law, tick) in [("law_2", 1_u64), ("law_1", 2)] {
        let mut refresh = start(MechanismStanding::DeclaredInvariant);
        let second = window(6, 12, "snap_2", &units());
        let tick = ObservationCadence::new(tick).unwrap();
        let outcome = refresh.refresh(&second, tick, law, &S[6..12]).unwrap();
        let PeriodRefreshOutcome::Restarted { invalidated } = outcome else {
            panic!("a changed {law} / {tick:?} reads the model");
        };
        assert!(invalidated.contains(&WindowProduct::Model));
        assert_eq!(refresh.fit().snapshot().n, 4);
        refresh.check_against_fresh().unwrap();
    }
}

const OVERLAP: &str = "period_refresh.overlap: the period overlaps the summarized one or reuses its snapshot, so the same observations would enter twice; supply a later period of a new snapshot";

fn invalid_event<T>(result: Result<T, antecedent_state::StateError>) -> String {
    match result {
        Err(antecedent_state::StateError::InvalidEvent(message)) => message,
        Err(other) => panic!("expected an invalid-event refusal, got {other:?}"),
        Ok(_) => panic!("expected a refusal"),
    }
}

#[test]
fn an_overlapping_period_or_reused_snapshot_is_refused_and_changes_nothing() {
    let mut refresh = start(MechanismStanding::DeclaredInvariant);
    let before = refresh.fit().snapshot();
    let posterior = refresh.summary().unwrap();
    let overlap = window(4, 10, "snap_overlap", &units());
    assert_eq!(
        invalid_event(refresh.refresh(&overlap, cadence(), "law_1", &S[4..10])),
        OVERLAP
    );
    let reused = window(6, 12, "snap_1", &units());
    assert_eq!(invalid_event(refresh.refresh(&reused, cadence(), "law_1", &S[6..12])), OVERLAP);
    let short = window(6, 12, "snap_2", &units());
    assert_eq!(
        invalid_event(refresh.refresh(&short, cadence(), "law_1", &S[6..11])),
        "one value per time index of the period"
    );
    assert_eq!(refresh.fit().snapshot(), before);
    assert_eq!(refresh.summary().unwrap(), posterior);
}

#[test]
fn a_stated_prior_of_the_wrong_width_is_refused() {
    let narrow = ConjugateOlsPrior::isotropic(2, 1.0, 3.0, 2.0).unwrap();
    let first = window(0, 6, "snap_1", &units());
    let spec = spec(MechanismStanding::DeclaredInvariant, "law_1");
    assert_eq!(
        invalid_event(PeriodRefresh::start(&spec, &first, &narrow, &S[0..6])),
        "a lag order of at least one and a prior of width lags + 1"
    );
}

#[test]
fn the_summary_only_export_serves_the_least_squares_fit_and_is_not_the_ledger() {
    let refresh = start(MechanismStanding::DeclaredInvariant);
    let fit = refresh.fit().snapshot();
    let summary_bytes = refresh.export_summary_only().unwrap();
    let ledger_bytes = refresh.export_ledger().unwrap();
    // The producer still retains its rows.
    assert!(!refresh.fit().rows().is_empty());
    let consumer = SummaryOnlyFit::consume(&summary_bytes).unwrap();
    assert_eq!(consumer.n(), fit.n);
    assert_eq!(consumer.beta(), fit.beta.as_deref());
    assert_eq!(consumer.residual_variance(), fit.residual_variance);
    assert_eq!(consumer.source_digest(), fit.source_digest);
    // Neither form is read as the other.
    let message = invalid_event(SummaryOnlyFit::consume(&ledger_bytes));
    assert!(
        message.starts_with("cbor error: Custom(\"unknown field `series_id`, expected one of"),
        "{message}"
    );
    match antecedent::consume_period_refresh_artifact(&summary_bytes) {
        Err(antecedent_io::error::IoError::Refused { code, message }) => {
            assert_eq!(code, "invalid_argument");
            assert!(message.starts_with("period_refresh.summary_only: this is the summary-only form"));
        }
        other => panic!("the summary-only form is not the full ledger, got {:?}", other.err()),
    }
}
