//! Coverage of the complete sampled target path contrast. Each replicate
//! independently samples every cited source and target regime table from the
//! finite-SCM oracle, recompiles both edge-intervention worlds and
//! differentiates their difference on the shared count stack.

mod common;

#[path = "../../antecedent-identify/tests/support/counterfactual_transport_adapter.rs"]
mod counterfactual_transport_adapter;
#[path = "../../antecedent-identify/tests/support/counterfactual_transport_cases.rs"]
mod counterfactual_transport_cases;
#[path = "../../antecedent-identify/tests/support/finite_scm_oracle.rs"]
mod finite_scm_oracle;

use std::sync::Arc;

use antecedent::CounterfactualTransportOptions;
use antecedent::analysis::temporal_path_transport::{
    TemporalPathSpec, compile_sampled_temporal_path_transport,
};
use antecedent_core::{CalibrationBasis, ExecutionContext};
use antecedent_estimate::counterfactual_transport_sampled::{
    OverlapDeclaration, SampledAxis, SampledRegimeCounts, SampledRegimeCountsParts,
    SampledTransportLimits, SampledTransportRequest, SamplingDesign, UnseenCellRule,
};
use antecedent_expr::ExactDiscreteLaw;
use antecedent_identify::{
    COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS, CounterfactualTransportDecision,
    CounterfactualTransportDerivation, identify_counterfactual_transport,
};
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, grid_seed,
    map_replicates, n_sim,
};
use counterfactual_transport_adapter as adapter;
use counterfactual_transport_cases::{Case, Target, all_cases};

const TEST: &str = "sampled_temporal_path_transport_l95";
const DGP: &str = "finite_scm_mediation_covariate_shift_independent_regime_samples";
const INTERVAL: &str = "paired_delta_wald_clamped_95";

struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn multinomial(law: &ExactDiscreteLaw, n: u64, rng: &mut Rng) -> Vec<u64> {
    let mut cumulative = Vec::with_capacity(law.probabilities().len());
    let mut running = 0.0;
    for probability in law.probabilities() {
        running += probability;
        cumulative.push(running);
    }
    let last = law.probabilities().iter().rposition(|p| *p > 0.0).unwrap();
    let mut counts = vec![0u64; law.probabilities().len()];
    for _ in 0..n {
        // One uniform per unit: the binary search calls its predicate several times, and a
        // fresh draw per call would sample a distorted law that starves the late cells.
        let u = rng.unit();
        let cell = cumulative.partition_point(|c| *c <= u).min(last);
        counts[cell] += 1;
    }
    counts
}

fn table_of(law: &ExactDiscreteLaw, n: u64, rng: &mut Rng) -> SampledRegimeCounts {
    SampledRegimeCounts::try_new(SampledRegimeCountsParts {
        population: Arc::from(law.population()),
        regime: law.regime(),
        interventions: law
            .interventions()
            .iter()
            .map(|a| (a.variable, a.value.as_f64().unwrap()))
            .collect(),
        axes: law
            .axes()
            .iter()
            .map(|axis| SampledAxis {
                variable: axis.variable,
                levels: axis.values.iter().map(|value| value.as_f64().unwrap()).collect(),
            })
            .collect(),
        counts: multinomial(law, n, rng),
        snapshot_identity: Arc::from(law.snapshot_identity()),
        design: SamplingDesign::IndependentUnits,
        structural_zeros: law
            .probabilities()
            .iter()
            .enumerate()
            .filter_map(|(cell, p)| (*p == 0.0).then_some(cell))
            .collect(),
    })
    .unwrap()
}

fn target<'a>(case: &'a Case, label: &str) -> &'a Target {
    case.targets.iter().find(|target| target.label == label).unwrap()
}

fn derive(case: &Case, target: &Target) -> CounterfactualTransportDerivation {
    let problem = adapter::problem(case);
    let adapted = adapter::catalog(case);
    let sources = adapter::sources(case, &adapter::evidence_controllable(case));
    let query = adapter::query(case, &target.query, sources);
    match identify_counterfactual_transport(
        &problem,
        &query,
        &adapted.catalog,
        COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS,
        &ExecutionContext::for_tests(1),
    )
    .unwrap()
    {
        CounterfactualTransportDecision::Identified(value) => *value,
        other => panic!("path world is not identified: {other:?}"),
    }
}

fn replicate(seed: u64, n: usize) -> Option<((f64, f64), f64, CalibrationBasis)> {
    let case =
        all_cases().into_iter().find(|case| case.name == "nonlinear_mediation_covariate_shift")?;
    let active = target(&case, "Y_{x1,M_{x0}}");
    let baseline = target(&case, "Y_{x0,M_{x0}}");
    let truth = active.expected_value? - baseline.expected_value?;
    let active_derivation = derive(&case, active);
    let baseline_derivation = derive(&case, baseline);
    let adapted = adapter::catalog(&case);
    let data = adapter::laws(&case, &adapted);
    let mut rng = Rng(grid_seed(seed));
    let cited = |law: &ExactDiscreteLaw| {
        active_derivation.cited.iter().chain(&baseline_derivation.cited).any(|regime| {
            regime.regime == law.regime() && regime.population.as_ref() == law.population()
        })
    };
    let tables = data
        .laws()
        .iter()
        .filter(|law| cited(law))
        .map(|law| table_of(law, n as u64, &mut rng))
        .collect();
    let mut limits = SampledTransportLimits::default();
    limits.evaluation.operations = 40_000_000;
    let request = SampledTransportRequest {
        tables,
        overlap: OverlapDeclaration::all_independent(),
        unseen_rule: UnseenCellRule::Refuse,
        limits,
    };
    let id = |name: &str| adapter::variable_id(&case, name).raw();
    let spec = TemporalPathSpec {
        treatment: id("X"),
        outcome: id("Y"),
        control_bits: 0.0_f64.to_bits(),
        active_bits: 1.0_f64.to_bits(),
        outcome_level_bits: 1.0_f64.to_bits(),
        times: [("Z", 0), ("X", 1), ("M", 2), ("Y", 3)]
            .into_iter()
            .map(|(name, time)| (id(name), time))
            .collect(),
        active_edges: vec![(id("X"), id("Y"))],
    };
    let ctx = ExecutionContext::for_tests(grid_seed(seed ^ 0x7011_2001));
    let plan = compile_sampled_temporal_path_transport(
        adapter::problem(&case),
        Arc::from("target"),
        Arc::from(adapter::sources(&case, &adapter::evidence_controllable(&case))),
        adapted.catalog,
        request,
        spec,
        CounterfactualTransportOptions {
            variable_names: case.graph.observed.iter().map(|name| (*name).to_owned()).collect(),
            ..Default::default()
        },
        &ctx,
    )
    .ok()?;
    let candidate = plan.calibration_candidate(&ctx).ok()?;
    assert_eq!(candidate.result.estimate.calibration, "not_measured");
    assert_eq!(candidate.basis.interval_method.as_ref(), INTERVAL);
    Some((candidate.interval, truth, candidate.basis))
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn sampled_temporal_path_transport_l95() {
    let n = grid_n(40_000);
    let mut tally = CoverageTally::for_record(
        RecordKey { test: TEST, dgp: DGP, interval: INTERVAL },
        REPORTED_LEVEL,
    );
    for result in map_replicates(n_sim(), |rep| replicate(0x7011_7001 + rep, n)) {
        if let Some((interval, truth, basis)) = result {
            let construction = Construction {
                query: basis.query.to_string(),
                graph_class: basis.graph_class.to_string(),
                structure: basis.structure.to_string(),
                modality: basis.modality.to_string(),
                inference: basis.inference.to_string(),
                estimator: basis.estimator.to_string(),
                interval_method: basis.interval_method.to_string(),
                se_kind: basis.se_kind.to_string(),
                dependence: basis.dependence.to_string(),
                posterior: basis.posterior.to_string(),
                functional: basis.functional.to_string(),
                identification: basis.identification.to_string(),
                reported_level: basis.level,
                unit: None,
            };
            tally.bind_validation(
                &construction,
                ScopeFacts {
                    row_count: basis.row_count,
                    replicates_ok: basis.replicates_ok,
                    posterior_draws: basis.posterior_draws,
                    unidentified_mass: basis.unidentified_mass,
                },
                &basis.validation,
            );
            tally.record(Some(interval), truth);
        } else {
            tally.skip();
        }
    }
    tally.assert();
}
