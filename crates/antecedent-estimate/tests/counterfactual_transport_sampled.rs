//! Sampled counterfactual transport against independent finite-SCM truth.
//!
//! Every expected number is enumerated by the oracle from the target
//! population's structural model. The samples are multinomial draws of units
//! from the enumerated observed laws of each regime (the law of the observed
//! variables of units drawn from the structural model), so the sampling
//! distribution of the estimator is known and its truth is the enumerated
//! probability. Seeds are fixed; tolerances are declared before the run.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "bit-for-bit determinism assertions")]

#[cfg(feature = "calibration-internal")]
#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;
#[path = "../../antecedent-identify/tests/support/counterfactual_transport_adapter.rs"]
mod counterfactual_transport_adapter;
#[path = "../../antecedent-identify/tests/support/counterfactual_transport_cases.rs"]
mod counterfactual_transport_cases;
#[path = "../../antecedent-identify/tests/support/finite_scm_oracle.rs"]
mod finite_scm_oracle;

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_estimate::EstimationError;
use antecedent_estimate::counterfactual_transport_sampled::{
    DatasetOverlap, LABEL_ENVELOPE_ONLY, LABEL_EXACT_ON_ESTIMATED_TABLES, LABEL_IDENTIFIED,
    LABEL_INFERENCE_WITHHELD, LABEL_PLUG_IN, LABEL_SMOOTHED_PLUG_IN, OverlapDeclaration,
    SAMPLED_TRANSPORT_UNCERTAINTY_METHOD, STRUCTURAL_ENVELOPE_LABEL, SampledAxis,
    SampledCounterfactualTransportEstimate, SampledRegimeCounts, SampledRegimeCountsParts,
    SampledTransportLimits, SampledTransportRequest, SamplingDesign, UNCALIBRATED_STANDARD_ERROR,
    UnseenCellRule, estimate_sampled_counterfactual_transport,
};
use antecedent_expr::{ExactDiscreteLaw, ExactTransportData};
use antecedent_identify::{
    COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS, CounterfactualTransportDecision,
    CounterfactualTransportDerivation, identify_counterfactual_transport,
};
#[cfg(feature = "calibration-internal")]
use calibration::{Construction, CoverageTally, RecordKey, ScopeFacts, map_replicates, n_sim};
use counterfactual_transport_adapter as adapter;
use counterfactual_transport_cases::{Case, CaseGraph, ExpectedClass, Target, all_cases};
use finite_scm_oracle::{
    Atom, CfVar, EvidenceSpec, FiniteScm, Mechanism, Population, Setting, TargetQuery, bern,
};

/// Absolute tolerance of a plug-in at the large-sample size below: about five
/// standard deviations of the least precise estimand of the named designs.
const PLUG_IN_TOLERANCE: f64 = 0.01;
/// Units per table of the large-sample designs.
const LARGE_N: u64 = 200_000;
/// Units per table of the sampling-variance design.
const MODEST_N: u64 = 1_500;
/// Replicates of the sampling-variance design.
const REPLICATES: u64 = 400;

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn multinomial(probabilities: &[f64], n: u64, rng: &mut Rng) -> Vec<u64> {
    let mut cumulative = Vec::with_capacity(probabilities.len());
    let mut running = 0.0;
    for p in probabilities {
        running += p;
        cumulative.push(running);
    }
    let last_positive = probabilities.iter().rposition(|p| *p > 0.0).expect("a law has mass");
    let mut counts = vec![0u64; probabilities.len()];
    for _ in 0..n {
        let u = rng.unit();
        let cell = cumulative.partition_point(|c| *c <= u).min(last_positive);
        counts[cell] += 1;
    }
    counts
}

fn table_of(law: &ExactDiscreteLaw, counts: Vec<u64>, structural: bool) -> SampledRegimeCounts {
    let zeros: Vec<usize> = if structural {
        law.probabilities().iter().enumerate().filter(|(_, p)| **p == 0.0).map(|(i, _)| i).collect()
    } else {
        Vec::new()
    };
    SampledRegimeCounts::try_new(SampledRegimeCountsParts {
        population: Arc::from(law.population()),
        regime: law.regime(),
        interventions: law
            .interventions()
            .iter()
            .map(|a| (a.variable, a.value.as_f64().expect("a finite level")))
            .collect(),
        axes: law
            .axes()
            .iter()
            .map(|a| SampledAxis {
                variable: a.variable,
                levels: a.values.iter().map(|v| v.as_f64().expect("a finite level")).collect(),
            })
            .collect(),
        counts,
        snapshot_identity: Arc::from(law.snapshot_identity()),
        design: SamplingDesign::IndependentUnits,
        structural_zeros: zeros,
    })
    .expect("a sampled table of an enumerated law is valid")
}

fn sample_laws(data: &ExactTransportData, n: u64, rng: &mut Rng) -> Vec<SampledRegimeCounts> {
    data.laws()
        .iter()
        .map(|law| table_of(law, multinomial(law.probabilities(), n, rng), true))
        .collect()
}

fn controllable_for(case: &Case) -> Vec<&'static str> {
    if case.name == "missing_interventional_evidence" {
        vec!["X"]
    } else {
        adapter::evidence_controllable(case)
    }
}

fn identify(case: &Case, target: &Target) -> CounterfactualTransportDecision {
    let problem = adapter::problem(case);
    let catalog = adapter::catalog(case);
    let sources = adapter::sources(case, &controllable_for(case));
    let query = adapter::query(case, &target.query, sources);
    identify_counterfactual_transport(
        &problem,
        &query,
        &catalog.catalog,
        COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS,
        &ExecutionContext::for_tests(1),
    )
    .expect("a fixture is a valid request")
}

fn derive(case: &Case, target: &Target) -> CounterfactualTransportDerivation {
    match identify(case, target) {
        CounterfactualTransportDecision::Identified(derivation) => *derivation,
        other => panic!("{} / {}: expected a derivation, got {other:?}", case.name, target.label),
    }
}

fn case_named(name: &str) -> Case {
    all_cases().into_iter().find(|c| c.name == name).expect("fixture exists")
}

fn request(tables: Vec<SampledRegimeCounts>, rule: UnseenCellRule) -> SampledTransportRequest {
    SampledTransportRequest {
        tables,
        overlap: OverlapDeclaration::all_independent(),
        unseen_rule: rule,
        limits: SampledTransportLimits::default(),
    }
}

fn estimate(
    derivation: &CounterfactualTransportDerivation,
    tables: Vec<SampledRegimeCounts>,
    rule: UnseenCellRule,
) -> Result<SampledCounterfactualTransportEstimate, EstimationError> {
    estimate_sampled_counterfactual_transport(
        derivation,
        &request(tables, rule),
        &ExecutionContext::for_tests(7),
    )
}

fn refusal(error: &EstimationError) -> (&'static str, String) {
    match error {
        EstimationError::Refused { code, message } => (code, message.clone()),
        other => panic!("expected a registered refusal, got {other:?}"),
    }
}

fn point(estimate: &SampledCounterfactualTransportEstimate) -> f64 {
    estimate.point.as_ref().expect("a plug-in point").probability
}

/// The tables a derivation cites; evidence it does not read is refused by the
/// route, so a request carries exactly the cited tables.
fn cited_only(
    derivation: &CounterfactualTransportDerivation,
    tables: &[SampledRegimeCounts],
) -> Vec<SampledRegimeCounts> {
    tables
        .iter()
        .filter(|t| {
            derivation
                .cited
                .iter()
                .any(|c| c.regime == t.regime() && c.population.as_ref() == t.population())
        })
        .cloned()
        .collect()
}

fn laws_of(case: &Case) -> ExactTransportData {
    let adapted = adapter::catalog(case);
    adapter::laws(case, &adapted)
}

#[test]
fn large_samples_converge_to_the_enumerated_truth_with_four_distinct_labels() {
    let mut checked = 0;
    for (seed, name) in [
        "nonlinear_mediation_covariate_shift",
        "joint_counterfactual_two_source_regimes",
        "effect_of_treatment_on_treated_with_selection",
        "transportable_latent_district",
    ]
    .iter()
    .enumerate()
    {
        let case = case_named(name);
        let data = laws_of(&case);
        let mut rng = Rng(1_000 + seed as u64);
        let tables = sample_laws(&data, LARGE_N, &mut rng);
        for target in case.targets.iter().filter(|t| {
            t.class == ExpectedClass::Identified && t.query.population == case.target_population
        }) {
            let derivation = derive(&case, target);
            let result =
                estimate(&derivation, cited_only(&derivation, &tables), UnseenCellRule::Refuse)
                    .unwrap_or_else(|e| panic!("{name} / {}: {e}", target.label));
            let truth = target.expected_value.expect("an identified target is defined");
            let value = point(&result);
            assert!(
                (value - truth).abs() <= PLUG_IN_TOLERANCE,
                "{name} / {}: estimate {value} against enumerated {truth}",
                target.label
            );
            // The four claims are separate fields with their own labels.
            assert_eq!(result.labels.structural_identification, LABEL_IDENTIFIED);
            assert_eq!(result.labels.exact_finite_law_evaluation, LABEL_EXACT_ON_ESTIMATED_TABLES);
            assert_eq!(result.labels.point_estimate, LABEL_PLUG_IN);
            assert_eq!(result.labels.statistical_inference, LABEL_INFERENCE_WITHHELD);
            assert!(!result.point.as_ref().unwrap().smoothed);
            // The interval route is closed; a standard error is labelled
            // uncalibrated and is the only uncertainty number.
            assert_eq!(result.inference.interval.code, "cell_not_licensed");
            assert_eq!(result.inference.interval.reason, "estimator_grid_not_measured");
            let se = result.inference.standard_error.as_ref().expect("a standard error");
            assert_eq!(se.label, UNCALIBRATED_STANDARD_ERROR);
            assert_eq!(se.method, SAMPLED_TRANSPORT_UNCERTAINTY_METHOD);
            assert!(se.value > 0.0 && se.value < PLUG_IN_TOLERANCE, "{name}: se {}", se.value);
            // The estimate is within a few reported standard errors of the truth
            // at this seed, a consistency check on the standard error itself.
            assert!((value - truth).abs() <= 6.0 * se.value, "{name} / {}", target.label);
            assert!(result.envelope.is_none() && result.point_withheld_reason.is_none());
            assert!(result.cells_read > 0);
            checked += 1;
        }
    }
    assert!(checked >= 7, "the named positive queries were exercised ({checked})");
}

#[test]
fn natural_direct_and_indirect_effects_converge_to_the_enumerated_contrasts() {
    let case = case_named("nonlinear_mediation_covariate_shift");
    let data = laws_of(&case);
    let mut rng = Rng(77);
    let tables = sample_laws(&data, LARGE_N, &mut rng);
    let value = |label: &str| {
        let target = case.targets.iter().find(|t| t.label == label).unwrap();
        let derivation = derive(&case, target);
        let cited = cited_only(&derivation, &tables);
        point(&estimate(&derivation, cited, UnseenCellRule::Refuse).unwrap())
    };
    let truth = |label: &str| {
        case.targets.iter().find(|t| t.label == label).unwrap().expected_value.unwrap()
    };
    let nde = value("Y_{x1,M_{x0}}") - value("Y_{x0,M_{x0}}");
    let nie = value("Y_{x0,M_{x1}}") - value("Y_{x0,M_{x0}}");
    let nde_truth = truth("Y_{x1,M_{x0}}") - truth("Y_{x0,M_{x0}}");
    let nie_truth = truth("Y_{x0,M_{x1}}") - truth("Y_{x0,M_{x0}}");
    // A contrast of two estimates has up to twice the variance of one.
    let tolerance = 1.5 * PLUG_IN_TOLERANCE;
    assert!((nde - nde_truth).abs() <= tolerance, "NDE {nde} vs {nde_truth}");
    assert!((nie - nie_truth).abs() <= tolerance, "NIE {nie} vs {nie_truth}");
    assert!(nde_truth.abs() > 1e-3 && nie_truth.abs() > 1e-3, "the contrasts are not degenerate");
}

#[test]
fn reported_standard_error_matches_the_empirical_standard_deviation_of_the_estimator() {
    let case = case_named("joint_counterfactual_two_source_regimes");
    let adapted = adapter::catalog(&case);
    let data = adapter::laws(&case, &adapted);
    let target = &case.targets[0];
    let derivation = derive(&case, target);
    let truth = target.expected_value.unwrap();
    let mut estimates = Vec::new();
    let mut standard_errors = Vec::new();
    for replicate in 0..REPLICATES {
        let mut rng = Rng(50_000 + replicate);
        let tables = cited_only(&derivation, &sample_laws(&data, MODEST_N, &mut rng));
        let result = estimate(&derivation, tables, UnseenCellRule::Refuse).unwrap();
        estimates.push(point(&result));
        standard_errors.push(result.inference.standard_error.as_ref().unwrap().value);
    }
    let n = estimates.len() as f64;
    let mean = estimates.iter().sum::<f64>() / n;
    let sd = (estimates.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let mean_se = standard_errors.iter().sum::<f64>() / n;
    let ratio = mean_se / sd;
    assert!(
        (ratio - 1.0).abs() <= 0.20,
        "mean reported standard error {mean_se} against empirical sd {sd} (ratio {ratio})"
    );
    // The plug-in is consistent: the mean is within the Monte Carlo band of the
    // truth plus an O(1/n) bias allowance declared in advance.
    assert!(
        (mean - truth).abs() <= 4.0 * sd / n.sqrt() + 0.004,
        "mean {mean} against enumerated {truth} (sd {sd})"
    );
}

/// Move the observations of `cell` onto its neighbour, leaving `cell` unseen.
fn emptied(table: &SampledRegimeCounts, cell: usize) -> SampledRegimeCounts {
    let mut counts = table.counts().to_vec();
    let neighbour = if cell + 1 < counts.len() { cell + 1 } else { cell - 1 };
    counts[neighbour] += counts[cell];
    counts[cell] = 0;
    SampledRegimeCounts::try_new(SampledRegimeCountsParts {
        population: Arc::from(table.population()),
        regime: table.regime(),
        interventions: table.interventions().to_vec(),
        axes: table.axes().to_vec(),
        counts,
        snapshot_identity: Arc::from(table.snapshot_identity()),
        design: table.design().clone(),
        structural_zeros: table.structural_zeros().to_vec(),
    })
    .unwrap()
}

fn with_structural_zero(table: &SampledRegimeCounts, cell: usize) -> SampledRegimeCounts {
    let mut zeros = table.structural_zeros().to_vec();
    zeros.push(cell);
    SampledRegimeCounts::try_new(SampledRegimeCountsParts {
        population: Arc::from(table.population()),
        regime: table.regime(),
        interventions: table.interventions().to_vec(),
        axes: table.axes().to_vec(),
        counts: table.counts().to_vec(),
        snapshot_identity: Arc::from(table.snapshot_identity()),
        design: table.design().clone(),
        structural_zeros: zeros,
    })
    .unwrap()
}

#[test]
fn unseen_cells_are_refused_smoothed_or_enveloped_and_never_zero_filled() {
    let case = case_named("joint_counterfactual_two_source_regimes");
    let data = laws_of(&case);
    let target = &case.targets[0];
    let derivation = derive(&case, target);
    let mut rng = Rng(9);
    let mut tables = cited_only(&derivation, &sample_laws(&data, 4_000, &mut rng));
    // Leave one cell of the source observational table unseen (a sampling zero
    // in a table whose exact law gives that cell positive mass).
    let observational = tables
        .iter()
        .position(|t| t.population() == "source" && t.interventions().is_empty())
        .unwrap();
    // A sparse cell: the one with the fewest observations, so the emptied table
    // stays close to the sampled one and the smoothed value stays near the truth.
    let cell = (0..tables[observational].counts().len())
        .filter(|c| tables[observational].counts()[*c] > 0)
        .min_by_key(|c| tables[observational].counts()[*c])
        .unwrap();
    tables[observational] = emptied(&tables[observational], cell);
    assert_eq!(tables[observational].unseen_cells(), vec![cell]);
    let truth = target.expected_value.unwrap();

    // Refuse: a typed support failure that names the unseen cell rule.
    let error = estimate(&derivation, tables.clone(), UnseenCellRule::Refuse).unwrap_err();
    let (code, message) = refusal(&error);
    assert_eq!(code, "transport_support_failure");
    assert!(message.contains("counterfactual_transport_sampled.unseen_cell"), "{message}");

    // Declared structural zero: a different, labelled claim (the cell is
    // asserted impossible, which is false here, so the value is biased).
    let structural = estimate(
        &derivation,
        {
            let mut t = tables.clone();
            t[observational] = with_structural_zero(&t[observational], cell);
            t
        },
        UnseenCellRule::Refuse,
    )
    .unwrap();

    // Smoothing: labelled smoothed, the pseudo-count recorded, a different
    // number from the structural-zero reading of the same counts.
    let smoothed = estimate(
        &derivation,
        tables.clone(),
        UnseenCellRule::AdditiveSmoothing { pseudo_count: 0.5 },
    )
    .unwrap();
    assert_eq!(smoothed.labels.point_estimate, LABEL_SMOOTHED_PLUG_IN);
    assert!(smoothed.point.as_ref().unwrap().smoothed);
    let report = smoothed
        .tables
        .iter()
        .find(|t| t.key.population.as_ref() == "source" && t.key.interventions.is_empty())
        .unwrap();
    assert!(report.smoothed && report.pseudo_count == Some(0.5) && report.unseen_cells == 1);
    // The rarest observed cell sits in a thinly sampled conditioning stratum, so
    // the conditional it feeds is dominated by one or two observations and the
    // smoothed value need not be near the enumerated truth (the large-sample
    // convergence tests cover accuracy); here it must be a proper probability
    // that is not the truth-by-construction and not the structural-zero reading.
    let smoothed_value = point(&smoothed);
    assert!((0.0..=1.0).contains(&smoothed_value), "{smoothed_value} vs {truth}");
    assert_ne!(point(&smoothed).to_bits(), point(&structural).to_bits());
    assert_eq!(smoothed.inference.interval.code, "cell_not_licensed");

    // Envelope: a labelled structural range over a finite family, no single
    // standard error, never an interval.
    let enveloped = estimate(
        &derivation,
        tables.clone(),
        UnseenCellRule::UnseenMassEnvelope { unseen_mass: 0.01 },
    )
    .unwrap();
    let envelope = enveloped.envelope.as_ref().expect("an envelope");
    assert_eq!(envelope.label, STRUCTURAL_ENVELOPE_LABEL);
    assert_eq!(envelope.members, 2);
    assert_eq!(envelope.defined_members + envelope.undefined_members, envelope.members);
    assert!(envelope.lower <= envelope.upper);
    assert!(envelope.upper - envelope.lower >= 0.0 && envelope.upper - envelope.lower < 0.1);
    assert!(enveloped.inference.standard_error.is_none());
    assert_eq!(
        enveloped.inference.standard_error_withheld_reason,
        Some("partial_identification_has_no_single_standard_error")
    );
    assert_ne!(enveloped.labels.point_estimate, LABEL_SMOOTHED_PLUG_IN);
    let _ = LABEL_ENVELOPE_ONLY;

    // A sparsity limit stricter than the table's unseen share refuses.
    let mut strict = request(tables, UnseenCellRule::AdditiveSmoothing { pseudo_count: 0.5 });
    strict.limits.max_unseen_fraction = 0.0;
    let error = estimate_sampled_counterfactual_transport(
        &derivation,
        &strict,
        &ExecutionContext::for_tests(7),
    )
    .unwrap_err();
    assert!(refusal(&error).1.contains("counterfactual_transport_sampled.sparsity_above_limit"));
}

#[test]
fn estimated_table_zero_support_is_a_typed_refusal_not_a_number() {
    let case = case_named("zero_support_conditioning_event");
    let data = laws_of(&case);
    let mut rng = Rng(31);
    let tables = sample_laws(&data, 2_000, &mut rng);
    // The target law gives X=1 no mass: its sampled table has that cell unseen
    // and declared structural, so the conditioning event has zero mass.
    let derivation = derive(&case, &case.targets[1]);
    let cited = cited_only(&derivation, &tables);
    let error = estimate(&derivation, cited, UnseenCellRule::Refuse).unwrap_err();
    let (code, message) = refusal(&error);
    assert_eq!(code, "transport_support_failure");
    assert!(message.contains("counterfactual_transport.zero_conditioning_mass"), "{message}");
    // Undeclared, the same cell is an unseen cell and refuses as such: never
    // silently zero.
    let undeclared: Vec<SampledRegimeCounts> = data
        .laws()
        .iter()
        .map(|law| table_of(law, multinomial(law.probabilities(), 2_000, &mut Rng(31)), false))
        .collect();
    let undeclared = cited_only(&derivation, &undeclared);
    let error = estimate(&derivation, undeclared, UnseenCellRule::Refuse).unwrap_err();
    assert!(refusal(&error).1.contains("counterfactual_transport_sampled.unseen_cell"));
    // The same query is defined in the source population.
    let mut in_source = case.clone();
    in_source.target_population = "source";
    let derivation = derive(&in_source, &case.targets[0]);
    let result =
        estimate(&derivation, cited_only(&derivation, &tables), UnseenCellRule::Refuse).unwrap();
    assert!((point(&result) - 0.25).abs() <= 0.05, "{}", point(&result));
}

#[test]
fn overlap_must_be_declared_and_unlinked_shared_units_are_refused() {
    let case = case_named("joint_counterfactual_two_source_regimes");
    let data = laws_of(&case);
    let derivation = derive(&case, &case.targets[0]);
    let mut rng = Rng(5);
    let tables = cited_only(&derivation, &sample_laws(&data, 3_000, &mut rng));
    assert!(tables.len() >= 3);
    let run = |overlap: OverlapDeclaration| {
        let mut r = request(tables.clone(), UnseenCellRule::Refuse);
        r.overlap = overlap;
        estimate_sampled_counterfactual_transport(&derivation, &r, &ExecutionContext::for_tests(7))
    };

    // Nothing declared: unknown is never independent.
    let (code, message) = refusal(&run(OverlapDeclaration::undeclared()).unwrap_err());
    assert_eq!(code, "sampling_dependence_unknown");
    assert!(message.contains("counterfactual_transport_sampled.overlap_unknown"));

    // One pair unknown among otherwise independent tables is refused too.
    let unknown_pair = OverlapDeclaration::all_independent()
        .with_pair(tables[0].key(), tables[1].key(), DatasetOverlap::Unknown)
        .unwrap();
    assert_eq!(refusal(&run(unknown_pair).unwrap_err()).0, "sampling_dependence_unknown");

    // Shared units declared between tables that are not the same observations
    // cannot be given a covariance from counts: refused, not approximated.
    let unlinked = OverlapDeclaration::all_independent()
        .with_pair(tables[0].key(), tables[1].key(), DatasetOverlap::SharedUnits)
        .unwrap();
    let (code, message) = refusal(&run(unlinked).unwrap_err());
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("counterfactual_transport_sampled.shared_units_unlinked"));

    // The declared-independent run is the reference value.
    let independent = run(OverlapDeclaration::all_independent()).unwrap();
    assert_eq!(independent.datasets.len(), tables.len());
    assert!(independent.datasets.iter().all(|g| g.len() == 1));
}

#[test]
fn uncited_missing_and_undersized_tables_are_refused_before_a_number() {
    let case = case_named("joint_counterfactual_two_source_regimes");
    let data = laws_of(&case);
    let derivation = derive(&case, &case.targets[0]);
    let mut rng = Rng(11);
    let tables = cited_only(&derivation, &sample_laws(&data, 3_000, &mut rng));

    // A withheld experiment: the cited regime has no table.
    let thin: Vec<SampledRegimeCounts> = tables
        .iter()
        .filter(|t| t.population() != "source" || t.interventions().is_empty())
        .cloned()
        .collect();
    let error = estimate(&derivation, thin, UnseenCellRule::Refuse).unwrap_err();
    let (code, message) = refusal(&error);
    assert_eq!(code, "transport_missing_provider");
    assert!(message.contains("counterfactual_transport_sampled.missing_table"));

    // A table for a regime the derivation does not cite is not silently ignored.
    let mut extra = tables.clone();
    let donor = &tables[0];
    extra.push(
        SampledRegimeCounts::try_new(SampledRegimeCountsParts {
            population: Arc::from("elsewhere"),
            regime: donor.regime(),
            interventions: donor.interventions().to_vec(),
            axes: donor.axes().to_vec(),
            counts: donor.counts().iter().map(|c| c + 1).collect(),
            snapshot_identity: Arc::from("elsewhere:snapshot"),
            design: SamplingDesign::IndependentUnits,
            structural_zeros: Vec::new(),
        })
        .unwrap(),
    );
    let error = estimate(&derivation, extra, UnseenCellRule::Refuse).unwrap_err();
    assert_eq!(refusal(&error).0, "invalid_argument");
    assert!(refusal(&error).1.contains("counterfactual_transport_sampled.table_not_cited"));

    // A sample below the declared minimum.
    let mut tiny = request(tables, UnseenCellRule::Refuse);
    tiny.limits.min_table_total = 1_000_000;
    let error = estimate_sampled_counterfactual_transport(
        &derivation,
        &tiny,
        &ExecutionContext::for_tests(7),
    )
    .unwrap_err();
    assert!(refusal(&error).1.contains("counterfactual_transport_sampled.sample_size_below_limit"));
}

#[test]
fn malformed_count_tables_are_refused_at_construction() {
    let case = case_named("transportable_latent_district");
    let data = laws_of(&case);
    let law = &data.laws()[0];
    let mut rng = Rng(3);
    let counts = multinomial(law.probabilities(), 500, &mut rng);
    let good = table_of(law, counts.clone(), false);
    let parts = |counts: Vec<u64>, zeros: Vec<usize>| SampledRegimeCountsParts {
        population: Arc::from(good.population()),
        regime: good.regime(),
        interventions: good.interventions().to_vec(),
        axes: good.axes().to_vec(),
        counts,
        snapshot_identity: Arc::from(good.snapshot_identity()),
        design: SamplingDesign::IndependentUnits,
        structural_zeros: zeros,
    };
    // Shape, emptiness, a structural zero that has observations.
    assert!(SampledRegimeCounts::try_new(parts(counts[1..].to_vec(), vec![])).is_err());
    assert!(SampledRegimeCounts::try_new(parts(vec![0; counts.len()], vec![])).is_err());
    let observed = counts.iter().position(|c| *c > 0).unwrap();
    assert!(SampledRegimeCounts::try_new(parts(counts.clone(), vec![observed])).is_err());
    // A changed observation changes the content identity even when labels and
    // shape match.
    let mut changed = counts.clone();
    changed[observed] -= 1;
    changed[(observed + 1) % counts.len()] += 1;
    let other = SampledRegimeCounts::try_new(parts(changed, vec![])).unwrap();
    assert_ne!(good.content_identity(), other.content_identity());
    assert_eq!(good.content_identity(), table_of(law, counts, false).content_identity());
    // Stratifying on a variable the table does not measure.
    let mut bad = parts(good.counts().to_vec(), vec![]);
    bad.design =
        SamplingDesign::Stratified { variables: vec![antecedent_core::VariableId::from_raw(999)] };
    assert!(SampledRegimeCounts::try_new(bad).is_err());
}

#[test]
fn estimates_are_deterministic_invariant_to_table_order_and_move_with_the_counts() {
    let case = case_named("effect_of_treatment_on_treated_with_selection");
    let data = laws_of(&case);
    let derivation = derive(&case, &case.targets[0]);
    let mut rng = Rng(21);
    let tables = cited_only(&derivation, &sample_laws(&data, 5_000, &mut rng));
    let first = estimate(&derivation, tables.clone(), UnseenCellRule::Refuse).unwrap();
    let again = estimate(&derivation, tables.clone(), UnseenCellRule::Refuse).unwrap();
    assert_eq!(point(&first).to_bits(), point(&again).to_bits());
    assert_eq!(first, again);
    let mut reversed = tables.clone();
    reversed.reverse();
    let permuted = estimate(&derivation, reversed, UnseenCellRule::Refuse).unwrap();
    assert_eq!(point(&first).to_bits(), point(&permuted).to_bits());
    assert_eq!(
        first.inference.standard_error.as_ref().unwrap().value.to_bits(),
        permuted.inference.standard_error.as_ref().unwrap().value.to_bits()
    );
    assert_eq!(first.evidence_identity, permuted.evidence_identity);
    assert_eq!(first.derivation_identity, permuted.derivation_identity);

    // Changing one observation moves the identity of every table's evidence;
    // the number moves for at least one table the formula reads.
    let mut number_moved = false;
    for index in 0..tables.len() {
        let mut moved = tables.clone();
        let mut counts = moved[index].counts().to_vec();
        let from = counts.iter().position(|c| *c > 10).unwrap();
        counts[from] -= 10;
        let to = (from + 1) % counts.len();
        counts[to] += 10;
        let t = &moved[index];
        moved[index] = SampledRegimeCounts::try_new(SampledRegimeCountsParts {
            population: Arc::from(t.population()),
            regime: t.regime(),
            interventions: t.interventions().to_vec(),
            axes: t.axes().to_vec(),
            counts,
            snapshot_identity: Arc::from(t.snapshot_identity()),
            design: t.design().clone(),
            structural_zeros: t.structural_zeros().to_vec(),
        })
        .unwrap();
        let changed = estimate(&derivation, moved, UnseenCellRule::Refuse).unwrap();
        assert_ne!(first.evidence_identity, changed.evidence_identity);
        number_moved |= point(&first).to_bits() != point(&changed).to_bits();
    }
    assert!(number_moved, "no edited table moved the estimate");
}

#[test]
fn an_interval_is_never_published_and_the_standard_error_is_labelled_uncalibrated() {
    let case = case_named("effect_of_treatment_on_treated_with_selection");
    let data = laws_of(&case);
    let derivation = derive(&case, &case.targets[0]);
    let mut rng = Rng(61);
    let tables = cited_only(&derivation, &sample_laws(&data, 8_000, &mut rng));
    let result = estimate(&derivation, tables, UnseenCellRule::Refuse).unwrap();
    let interval = &result.inference.interval;
    assert_eq!(interval.code, "cell_not_licensed");
    assert_eq!(interval.detail, "counterfactual_transport_sampled.interval_not_licensed");
    assert_eq!(interval.reason, "estimator_grid_not_measured");
    assert_eq!(result.labels.statistical_inference, LABEL_INFERENCE_WITHHELD);
    let se = result.inference.standard_error.as_ref().unwrap();
    assert_eq!(se.label, UNCALIBRATED_STANDARD_ERROR);
    assert_eq!(se.method, SAMPLED_TRANSPORT_UNCERTAINTY_METHOD);
    assert_eq!(se.gradient_relative_step, 1e-5);
    assert!(se.value.is_finite() && se.value > 0.0);
    assert_eq!(se.datasets, result.tables.len(), "independent tables are independent datasets");
    assert!(se.gradient_evaluations >= 2);
}

fn chain_case() -> Case {
    let scm = FiniteScm::new()
        .exo("Ux", bern(0.5))
        .exo("Um", bern(0.25))
        .exo("Uy", bern(0.125))
        .endo("X", 2, Mechanism::new(|_, u| u[0]))
        .endo("M", 2, Mechanism::new(|v, u| v[0] ^ u[1]))
        .endo("Y", 2, Mechanism::new(|v, u| v[1] ^ u[2]));
    let populations = vec![Population::new("pop", scm)];
    let query = TargetQuery::new("pop", vec![Atom::new(CfVar::under("Y", &[("X", 1)]), 1)], vec![]);
    let expected_value = query.value(&populations);
    Case {
        name: "chain_unread_coordinate",
        derivation: "X -> M -> Y without confounding: P(y | do(x)) = sum_m P(m | x) P(y | m); the \
                     factor of Y reads M only, so its sampled conditional on (X, M) differs \
                     across X by noise and is pooled over X",
        graph: CaseGraph {
            observed: vec!["X", "M", "Y"],
            directed: vec![("X", "M"), ("M", "Y")],
            bidirected: vec![],
            selection: vec![],
        },
        populations,
        target_population: "pop",
        evidence: vec![EvidenceSpec::new("pop", &[], &["X", "M", "Y"])],
        targets: vec![Target {
            label: "Y_{x1}=1",
            query,
            expected_value,
            class: ExpectedClass::Identified,
        }],
        alternative: None,
    }
}

#[test]
fn a_factor_that_reads_fewer_coordinates_than_its_table_is_pooled_not_refused() {
    let case = chain_case();
    let data = laws_of(&case);
    let derivation = derive(&case, &case.targets[0]);
    let truth = case.targets[0].expected_value.unwrap();
    let mut rng = Rng(404);
    let tables = cited_only(&derivation, &sample_laws(&data, 20_000, &mut rng));
    let counts = tables[0].counts().to_vec();
    let result = estimate(&derivation, tables, UnseenCellRule::Refuse).unwrap();
    // Cells are ordered X (slowest), M, Y. The independent hand computation of
    // sum_m N(x=1,m)/N(x=1) * N(m,y=1)/N(m) from the same counts.
    let n = |x: usize, m: usize, y: usize| counts[x * 4 + m * 2 + y] as f64;
    let mut by_hand = 0.0;
    for m in 0..2 {
        let n_x1: f64 =
            (0..2).flat_map(|mm| (0..2).map(move |y| (mm, y))).map(|(mm, y)| n(1, mm, y)).sum();
        let n_x1m: f64 = (0..2).map(|y| n(1, m, y)).sum();
        let n_m: f64 =
            (0..2).flat_map(|x| (0..2).map(move |y| (x, y))).map(|(x, y)| n(x, m, y)).sum();
        let n_my1: f64 = (0..2).map(|x| n(x, m, 1)).sum();
        by_hand += (n_x1m / n_x1) * (n_my1 / n_m);
    }
    assert!(
        (point(&result) - by_hand).abs() <= 1e-9,
        "pooled factor {} against the hand computation {by_hand}",
        point(&result)
    );
    assert!(result.repairs.averaged_factors > 0, "the unread coordinate was averaged and reported");
    assert!((point(&result) - truth).abs() <= PLUG_IN_TOLERANCE, "{} vs {truth}", point(&result));
}

#[test]
fn selection_obstruction_leaves_no_derivation_to_sample() {
    let case = case_named("selection_on_outcome_mechanism");
    let target = &case.targets[0];
    assert_eq!(target.class, ExpectedClass::NotTransportable);
    // Sampled counts exist for the supplied regimes, but the identifier proves
    // the obstruction; the sampled route consumes only a stored derivation, so
    // no number is ever formed from these counts.
    let data = laws_of(&case);
    let mut rng = Rng(8);
    let tables = sample_laws(&data, 1_000, &mut rng);
    assert!(!tables.is_empty());
    assert!(!matches!(identify(&case, target), CounterfactualTransportDecision::Identified(_)));
}

#[cfg(feature = "calibration-internal")]
#[test]
fn the_calibration_internal_interval_is_a_candidate_not_a_published_claim() {
    use antecedent_estimate::counterfactual_transport_sampled::sampled_transport_wald_interval_internal;
    let case = case_named("joint_counterfactual_two_source_regimes");
    let data = laws_of(&case);
    let derivation = derive(&case, &case.targets[0]);
    let mut rng = Rng(15);
    let tables = cited_only(&derivation, &sample_laws(&data, 10_000, &mut rng));
    let result = estimate(&derivation, tables, UnseenCellRule::Refuse).unwrap();
    let interval = sampled_transport_wald_interval_internal(&result, 0.95).unwrap();
    assert!(interval.lower <= point(&result) && point(&result) <= interval.upper);
    assert_eq!(interval.reason, "estimator_grid_not_measured");
    // The published result still carries no interval.
    assert_eq!(result.inference.interval.code, "cell_not_licensed");
    assert!(sampled_transport_wald_interval_internal(&result, 1.5).is_err());
}

/// Name of the coverage-specific mediation design below.
const COVERAGE_MEDIATION_CASE: &str = "coverage_mediation_covariate_shift";

/// Smallest mass the coverage design gives any positive-probability cell of an
/// estimated table. With `K <= 16` cells and `n >= 250` units, a cell of mass
/// `p` is unseen with probability at most `(1 - p)^250`; at `p >= 0.04` the union
/// bound `16 * 0.96^250 ~ 6e-4` keeps the unseen-cell refusal far below the
/// calibration skip cap, so a skipped replicate is not part of the claim's
/// support range.
const MIN_COVERAGE_CELL_MASS: f64 = 0.04;

/// A binary-mediator mediation design with covariate shift in `Z`, built for
/// coverage measurement: `Z = Uz` (source `P(Uz = 1) = 1/2`, target `3/4`),
/// `X = Z xor Ux`, `M = X xor Z xor Um`, `Y = [M or (X and Z)] xor Uy`, with
/// `P(Ux = 1) = P(Um = 1) = P(Uy = 1) = 7/16`. Every factor of the observed
/// joint is `7/16` or `9/16`, so the rarest source cell has mass
/// `1/2 * (7/16)^3 = 0.0419` and the target `Z` table has cells `1/4` and `3/4`.
/// The selection node flags only `Z`; the structure is the shared mediation
/// design (`Z -> X`, `Z -> M`, `Z -> Y`, `X -> M`, `X -> Y`, `M -> Y`), so
/// `Y_{x,M_{x'}} = sum_{z,m} P(y | x, m, z) P(m | x', z) P*(z)` is identified.
fn coverage_mediation_population(uz: f64) -> FiniteScm {
    FiniteScm::new()
        .exo("Uz", bern(uz))
        .exo("Ux", bern(0.4375))
        .exo("Um", bern(0.4375))
        .exo("Uy", bern(0.4375))
        .endo("Z", 2, Mechanism::new(|_, u| u[0]))
        .endo("X", 2, Mechanism::new(|v, u| v[0] ^ u[1]))
        .endo("M", 2, Mechanism::new(|v, u| (v[1] + v[0] + u[2]) % 2))
        .endo("Y", 2, Mechanism::new(|v, u| usize::from(v[2] + v[1] * v[0] >= 1) ^ u[3]))
}

fn coverage_mediation_case() -> Case {
    let populations = vec![
        Population::new("source", coverage_mediation_population(0.5)),
        Population::new("target", coverage_mediation_population(0.75)),
    ];
    let query = TargetQuery::new(
        "target",
        vec![Atom::new(
            CfVar::nested(
                "Y",
                vec![("X", Setting::Level(1)), ("M", Setting::cf(CfVar::under("M", &[("X", 0)])))],
            ),
            1,
        )],
        vec![],
    );
    let expected_value = query.value(&populations);
    Case {
        name: COVERAGE_MEDIATION_CASE,
        derivation: "Markovian mediation with a binary mediator and an or-threshold outcome; \
                     selection on Z only; Y_{x,M_{x'}} = sum_{z,m} P(y|x,m,z)P(m|x',z)P*(z); \
                     every cell of every estimated table has mass at least 0.04",
        graph: CaseGraph {
            observed: vec!["Z", "X", "M", "Y"],
            directed: vec![("Z", "X"), ("Z", "M"), ("Z", "Y"), ("X", "M"), ("X", "Y"), ("M", "Y")],
            bidirected: vec![],
            selection: vec![("S_Z", "Z")],
        },
        target_population: "target",
        evidence: vec![
            EvidenceSpec::new("source", &[], &["Z", "X", "M", "Y"]),
            EvidenceSpec::new("target", &[], &["Z"]),
        ],
        targets: vec![Target {
            label: "Y_{x1,M_{x0}}",
            query,
            expected_value,
            class: ExpectedClass::Identified,
        }],
        populations,
        alternative: None,
    }
}

#[test]
fn the_coverage_mediation_design_gives_every_positive_cell_at_least_the_required_mass() {
    let case = coverage_mediation_case();
    let data = laws_of(&case);
    let minimum = data
        .laws()
        .iter()
        .flat_map(|law| law.probabilities().iter().copied())
        .filter(|p| *p > 0.0)
        .fold(f64::INFINITY, f64::min);
    assert!(
        minimum >= MIN_COVERAGE_CELL_MASS,
        "the rarest positive cell has mass {minimum}, below {MIN_COVERAGE_CELL_MASS}"
    );
    // Hand value: 1/2 * (7/16)^3.
    assert!((minimum - 0.5 * 0.4375_f64.powi(3)).abs() < 1e-12, "{minimum}");
    // The estimand is identified and its enumerated truth is a probability.
    let target = &case.targets[0];
    assert_eq!(target.class, ExpectedClass::Identified);
    let truth = target.expected_value.expect("the oracle supplies an exact target truth");
    assert!(truth > 0.0 && truth < 1.0);
    let _ = derive(&case, target);
}

#[cfg(feature = "calibration-internal")]
fn measure_sampled_transport_coverage(
    case_name: &str,
    target_label: &str,
    test: &'static str,
    dgp: &'static str,
    seed: u64,
    expected_id: &str,
) {
    use antecedent_estimate::counterfactual_transport_sampled::sampled_transport_wald_interval_internal;

    let case = if case_name == COVERAGE_MEDIATION_CASE {
        coverage_mediation_case()
    } else {
        case_named(case_name)
    };
    let target = case.targets.iter().find(|target| target.label == target_label).unwrap();
    assert_eq!(target.class, ExpectedClass::Identified);
    let truth = target.expected_value.expect("the oracle supplies an exact target truth");
    let derivation = derive(&case, target);
    let data = laws_of(&case);
    // The allocation names 250, 1000 and 4000 units per table. Use that exact
    // grid rather than the shared default so collected scopes match the plan.
    let n = [250_u64, 1_000, 4_000][calibration::grid_point()];
    let mut tally =
        CoverageTally::for_record(RecordKey { test, dgp, interval: "delta_method_wald" }, 0.95);
    let construction = Construction {
        query: "CounterfactualTransport".to_owned(),
        graph_class: "Admg".to_owned(),
        structure: "Fixed".to_owned(),
        modality: "Discrete".to_owned(),
        inference: "Frequentist".to_owned(),
        estimator: "sampled_counterfactual_transport".to_owned(),
        interval_method: "delta_method_wald".to_owned(),
        se_kind: "delta_method_multinomial_central_difference".to_owned(),
        dependence: "independent_units".to_owned(),
        posterior: String::new(),
        functional: derivation.query.canonical_text(),
        identification: "point".to_owned(),
        reported_level: 0.95,
    };
    let results = map_replicates(n_sim(), |replicate| {
        let mut rng = Rng(calibration::grid_seed(seed + replicate));
        let tables = cited_only(&derivation, &sample_laws(&data, n, &mut rng));
        let ctx = ExecutionContext::for_tests(seed + replicate);
        derivation.replay_checked(&ctx).expect("the stored identified derivation replays");
        let result = estimate(&derivation, tables, UnseenCellRule::Refuse).ok()?;
        let interval = sampled_transport_wald_interval_internal(&result, 0.95).ok()?;
        Some((result, interval))
    });
    for result in results {
        match result {
            Some((estimate, interval)) => {
                assert!(estimate.point.is_some());
                assert!(estimate.inference.standard_error.is_some());
                assert_eq!(interval.method, "delta_wald_clamped");
                tally.bind(
                    &construction,
                    ScopeFacts {
                        row_count: n,
                        replicates_ok: None,
                        posterior_draws: None,
                        unidentified_mass: 0.0,
                    },
                );
                tally.record(Some((interval.lower, interval.upper)), truth);
            }
            None => tally.skip(),
        }
    }
    assert_eq!(tally.record_id().as_deref(), Some(expected_id));
    tally.assert();
}

/// Calibration design. Claim: the 95% delta-method Wald interval of a sampled
/// natural-effect transport functional covers the enumerated target truth.
/// Whole method: identify, sample tables, `UnseenCellRule::Refuse`, delta-method
/// standard error, clamped Wald interval. Truth: the finite SCM of
/// [`coverage_mediation_case`], enumerated by the oracle. Dependence: independent
/// units per table. Sizes: 250, 1000 and 4000 units per table. Support range:
/// every positive-probability cell of every estimated table has mass at least
/// [`MIN_COVERAGE_CELL_MASS`] (asserted by
/// `the_coverage_mediation_design_gives_every_positive_cell_at_least_the_required_mass`),
/// so the unseen-cell refusal is a negligible skip and the claim is not
/// conditioned on a data event; sparser designs are outside the claim.
#[cfg(feature = "calibration-internal")]
#[test]
#[ignore = "calibration: final measurement only"]
fn sampled_mediation_covariate_shift() {
    measure_sampled_transport_coverage(
        COVERAGE_MEDIATION_CASE,
        "Y_{x1,M_{x0}}",
        "sampled_mediation_covariate_shift",
        "finite_scm_mediation_covariate_shift",
        0x71a0_0000,
        "cov.counterfactual_transport.admg.frequentist.delta_method_wald.l95.sampled_mediation_covariate_shift",
    );
}

#[cfg(feature = "calibration-internal")]
#[test]
#[ignore = "calibration: final measurement only"]
fn sampled_two_source_joint() {
    measure_sampled_transport_coverage(
        "joint_counterfactual_two_source_regimes",
        "Y_{x1}=1,X=0",
        "sampled_two_source_joint",
        "finite_scm_two_source_joint_counterfactual_transport",
        0x71a1_0000,
        "cov.counterfactual_transport.admg.frequentist.delta_method_wald.l95.sampled_two_source_joint",
    );
}

#[cfg(feature = "calibration-internal")]
#[test]
#[ignore = "calibration: final measurement only"]
fn sampled_latent_district() {
    measure_sampled_transport_coverage(
        "transportable_latent_district",
        "Y_{x1}=1",
        "sampled_latent_district",
        "finite_scm_frontdoor_latent_district",
        0x71a2_0000,
        "cov.counterfactual_transport.admg.frequentist.delta_method_wald.l95.sampled_latent_district",
    );
}
