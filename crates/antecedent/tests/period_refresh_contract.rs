//! Period refresh of one autoregressive mechanism, least squares and conjugate posterior on the
//! same rows. Reference values are numpy fits of the rows a fresh fit would use: design
//! `[1, v_{t-1}, v_{t-2}]`, prior `m₀ = 0`, `Λ₀ = diag(0.5, 0.4, 0.3)`, `InvGamma(3, 2)`.

use std::collections::BTreeSet;

use antecedent::period_refresh::{PeriodRefresh, PeriodRefreshOutcome, PeriodRefreshSpec};
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
        [-0.3917647058823548, 0.5058823529411782, 1.3882352941176475],
        0.016000000000000004,
        [0.2272942518191619, 0.16164821678711794, 0.8370005826277753],
        5.0,
        2.1763865921233685,
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
        [0.47578572103621536, -0.12081675084793765, 0.8347229368587971],
        0.0352728328260455,
        [0.3515871568250203, 0.004469720612162134, 0.7997133706820022],
        8.0,
        2.265297812709253,
    );
    assert_eq!(refresh.fit().recomputes(), 0, "an append is not a rebuild");
    let third = window(12, 18, "snap_3", &units());
    refresh.refresh(&third, cadence(), "law_1", &S[12..18]).unwrap();
    assert_fit(
        &refresh,
        16,
        [0.40125618576323036, -0.05071593803988238, 0.8636203917894049],
        0.03482335239648349,
        [0.3278080871093989, 0.022038223784948335, 0.83536550785459],
        11.0,
        2.36722781148982,
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
        [2.4679746835440985, -0.8291139240505575, 0.14683544303804172],
        0.06383544303797459,
        [0.24998303743073655, 0.08424742734366153, 0.7520072373628859],
        5.0,
        2.188748727807305,
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
        [0.3670513885592226, -0.006186209352005272, 0.8397371231901244],
        0.06631227747448373,
        [0.2970587733457107, 0.09556683194501768, 0.7763572332881739],
        7.0,
        2.2907118116577543,
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
        [1.8630057803467295, -0.4942196531791619, 0.4797687861271897],
        0.06658959537572258,
        [0.17841548288683315, 0.10824435122509381, 0.8053198588997993],
        5.0,
        2.163647869196301,
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

#[test]
fn an_overlapping_period_or_reused_snapshot_is_refused_and_changes_nothing() {
    let mut refresh = start(MechanismStanding::DeclaredInvariant);
    let before = refresh.fit().snapshot();
    let posterior = refresh.summary().unwrap();
    let overlap = window(4, 10, "snap_overlap", &units());
    assert!(refresh.refresh(&overlap, cadence(), "law_1", &S[4..10]).is_err());
    let reused = window(6, 12, "snap_1", &units());
    assert!(refresh.refresh(&reused, cadence(), "law_1", &S[6..12]).is_err());
    let short = window(6, 12, "snap_2", &units());
    assert!(refresh.refresh(&short, cadence(), "law_1", &S[6..11]).is_err());
    assert_eq!(refresh.fit().snapshot(), before);
    assert_eq!(refresh.summary().unwrap(), posterior);
}

#[test]
fn a_stated_prior_of_the_wrong_width_is_refused() {
    let narrow = ConjugateOlsPrior::isotropic(2, 1.0, 3.0, 2.0).unwrap();
    let first = window(0, 6, "snap_1", &units());
    let spec = spec(MechanismStanding::DeclaredInvariant, "law_1");
    assert!(PeriodRefresh::start(&spec, &first, &narrow, &S[0..6]).is_err());
}
