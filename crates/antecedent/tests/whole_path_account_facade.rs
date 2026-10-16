//! The whole-path uncertainty account of a cross-world execution through the facade:
//! derived from the sealed request and result under a declared unit sampling, bound to
//! the execution it stands on, and re-derived (never trusted) by a fresh consumer.
//!
//! The response-type model: `X = U_X` with `P(U_X = 1) = 0.4`; `Y = f(X, U_Y)` with `U_Y`
//! over four response types (never, complier, defier, always) with masses
//! `q = [0.2, 0.5, 0.1, 0.2]`. For `P(Y_{x0} = 0 | evidence)` over three units:
//! `case (X = 1, Y = 1)`: `5/7`, variance `10/49`; `clean (X = 0, Y = 0)`: `1`, variance `0`;
//! `open`: `0.7`, variance `0.21`. The mean is `169/210`; the realized-outcome variance is
//! `2029/44100`; the sampling variance of the mean is `2526/264600`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::cross_world_execution::{
    CrossWorldExecution, ExogenousInference, ExogenousVariable, FactualUnit,
    FiniteExecutionRequest, FiniteQuantity, FiniteScm, FiniteVariable, FiniteWorld, NoiseCoupling,
    SourceTreatment, UncertaintySource, UnitSampling, WorldAtom,
};
use antecedent_contract::RequestedReading;
use antecedent_core::ExecutionContext;

const TOL: f64 = 1e-12;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(11)
}

fn scm() -> FiniteScm {
    FiniteScm {
        population: "pop-a".to_owned(),
        history: "regime-1".to_owned(),
        exogenous: vec![
            ExogenousVariable { name: "u_x".to_owned(), pmf: vec![0.6, 0.4] },
            ExogenousVariable { name: "u_y".to_owned(), pmf: vec![0.2, 0.5, 0.1, 0.2] },
        ],
        variables: vec![
            FiniteVariable {
                name: "x".to_owned(),
                levels: 2,
                parents: vec![],
                exogenous: 0,
                table: vec![0, 1],
            },
            FiniteVariable {
                name: "y".to_owned(),
                levels: 2,
                parents: vec![0],
                exogenous: 1,
                table: vec![0, 0, 0, 1, 1, 0, 1, 1],
            },
        ],
    }
}

fn unit(id: &str, observed: &[(usize, u8)]) -> FactualUnit {
    FactualUnit { unit: id.to_owned(), observed: observed.to_vec() }
}

fn do_x(name: &str, level: u8) -> FiniteWorld {
    FiniteWorld { name: name.to_owned(), interventions: vec![(0, level)], routes: vec![] }
}

fn request(units: Vec<FactualUnit>) -> FiniteExecutionRequest {
    let scm = scm();
    FiniteExecutionRequest {
        evidence_history: scm.history.clone(),
        scm,
        units,
        worlds: vec![do_x("x0", 0), do_x("x1", 1)],
        quantity: FiniteQuantity::Probability(WorldAtom { world: 0, variable: 1, level: 0 }),
        coupling: NoiseCoupling::SharedAbducedExogenous,
        exogenous: ExogenousInference::ExactEnumeration,
        target_population: None,
        reading: RequestedReading::Point,
    }
}

fn three_units() -> FiniteExecutionRequest {
    request(vec![
        unit("case", &[(0, 1), (1, 1)]),
        unit("clean", &[(0, 0), (1, 0)]),
        unit("open", &[]),
    ])
}

#[test]
fn the_account_is_the_enumerated_spread_bound_to_its_execution() {
    let sealed = CrossWorldExecution::seal(&three_units(), &ctx()).unwrap();
    let account = sealed.whole_path_uncertainty(UnitSampling::IndependentSample, &ctx()).unwrap();
    let u = account.uncertainty();
    assert_eq!(u.label, "uncalibrated_spread");
    assert!((u.value - 169.0 / 210.0).abs() < TOL);
    assert_eq!(u.value.to_bits(), sealed.result().value.to_bits());
    assert!((u.realized_mean_predictive_variance - 2029.0 / 44100.0).abs() < TOL);
    assert!((u.unit_mean_sampling_variance.unwrap() - 2526.0 / 264_600.0).abs() < TOL);
    assert_eq!(
        u.source(UncertaintySource::Structural).unwrap().treatment,
        SourceTreatment::Premise
    );
    assert_eq!(account.execution_receipt_digest(), sealed.identity().receipt_digest);

    let inspection = account.inspect();
    assert_eq!(inspection.label, "uncalibrated_spread");
    assert_eq!(inspection.unit_sampling, "independent_sample");
    assert!((inspection.realized_mean_standard_error - (2029.0_f64 / 44100.0).sqrt()).abs() < TOL);
    assert!(inspection.unit_mean_standard_error.is_some());
    assert_eq!(inspection.sources.len(), 4);
    assert_eq!(inspection.sources[0].0, "structural");

    // The reading stays closed: a spread is a labelled diagnostic, never an interval.
    assert!(sealed.interval().is_err());
    assert_eq!(sealed.posterior().unwrap_err().code, "cell_not_licensed");
}

#[test]
fn a_fresh_consumer_derives_the_same_account_from_the_artifact() {
    let sealed = CrossWorldExecution::seal(&three_units(), &ctx()).unwrap();
    let account = sealed.whole_path_uncertainty(UnitSampling::FixedUnits, &ctx()).unwrap();
    assert_eq!(account.uncertainty().unit_mean_sampling_variance, None);

    let bytes = sealed.to_bytes("whole-path").unwrap();
    let consumed = CrossWorldExecution::consume(&bytes, Some(sealed.identity()), &ctx()).unwrap();
    let rederived = consumed.whole_path_uncertainty(UnitSampling::FixedUnits, &ctx()).unwrap();
    assert_eq!(rederived, account);
    account.verify_against(&consumed, &ctx()).unwrap();

    // The declared unit sampling is part of the account: another declaration is another
    // account, not a relabelled one.
    let sampled = consumed.whole_path_uncertainty(UnitSampling::IndependentSample, &ctx()).unwrap();
    assert_ne!(sampled.uncertainty().identity, account.uncertainty().identity);
}

#[test]
fn an_account_of_other_units_is_refused_rather_than_annotated() {
    let sealed = CrossWorldExecution::seal(&three_units(), &ctx()).unwrap();
    let account = sealed.whole_path_uncertainty(UnitSampling::IndependentSample, &ctx()).unwrap();
    // New factual evidence is new units and a new execution: the earlier account does
    // not describe it.
    let other = CrossWorldExecution::seal(
        &request(vec![unit("case", &[(0, 1), (1, 1)]), unit("clean", &[(0, 0), (1, 0)])]),
        &ctx(),
    )
    .unwrap();
    let error = account.verify_against(&other, &ctx()).unwrap_err();
    let refusal = error.refusal().unwrap();
    assert_eq!(refusal.detail, "cross_world_uncertainty.account_mismatch");
    assert_eq!(refusal.code, "invalid_argument");
    assert!(refusal.remedy.is_some());
}

#[test]
fn undeclared_unit_sampling_is_refused_with_its_remedy() {
    let sealed = CrossWorldExecution::seal(&three_units(), &ctx()).unwrap();
    let error = sealed.whole_path_uncertainty(UnitSampling::Undeclared, &ctx()).unwrap_err();
    let refusal = error.refusal().unwrap();
    assert_eq!(refusal.code, "sampling_dependence_unknown");
    assert_eq!(refusal.detail, "cross_world_uncertainty.unit_sampling_undeclared");
    assert!(refusal.remedy.is_some());
}
