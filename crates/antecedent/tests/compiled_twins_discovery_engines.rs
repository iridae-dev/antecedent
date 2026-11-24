//! Twin equality of the discovery-engine side doors with their compiled programs, the identity
//! of those programs, and independent checks through them.
//!
//! * Discover-then-transport (`DiscoveredClassTransport`): the compiled program runs the
//!   compiled `DiscoverStatic` search and the transport kernels on its class, so the outcome,
//!   the structural envelope, the estimate, the trial-compatible estimate and the bounded-outcome
//!   confidence set (and its replay) equal `discover_and_transport` followed by the direct
//!   kernel, bit for bit, on the fixtures of the facade and trial tests.
//! * Per-population discovery (`PerPopulationStructureSearch`): the compiled program runs the
//!   per-population driver on the declared row sets; its report equals `discover_per_population`.
//! * Independent checks: the chain class holds exactly the three DAGs a d-separation oracle
//!   fixes, with the verdict the structural identifier fixes for each; a PAG class estimate
//!   covers the transported response enumerated from the target population's structural model;
//!   per-population structures of a known chain and a known collider differ on exactly the
//!   hand-derived marks.
//!
//! Seeds: the selective adapter refuses a context whose master seed is not the contract's, so the
//! direct run and the program share one context, and the bootstrap of a confidence set is seeded
//! by that context's master seed in both.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(deprecated, reason = "the twins call the direct entries the compiled programs replace")]
#![allow(
    clippy::float_cmp,
    clippy::too_many_lines,
    clippy::unwrap_used,
    reason = "integration test with deterministic fixtures and bitwise comparisons"
)]

use std::sync::Arc;

use antecedent::Study;
use antecedent::analysis::compiled_discovered_transport::{
    DiscoveredClassEvidence, DiscoveredClassStage, DiscoveredClassStageResult,
    DiscoveredClassTransportRequest, DiscoveredClassTransportRun,
};
use antecedent::analysis::compiled_per_population_discovery::{
    PerPopulationSearchError, PerPopulationSearchRequest,
};
use antecedent::analysis::discovered_transport::{
    DiscoveredTransportError, DiscoveredTransportRun, EnvelopeStatus, MemberVerdict,
    discover_and_transport, formula_kind,
};
use antecedent::analysis::discovered_transport_estimate::{
    DiscoveredTransportEstimate, TransportAnswer, TransportEvidence, estimate_discovered_transport,
};
use antecedent::analysis::discovered_transport_trial::{
    TransportOutcomeSupport, TrialCompatibilityOptions, TrialRandomization,
    estimate_trial_compatible_discovered_transport,
    estimate_trial_compatible_transport_confidence_set,
    verify_trial_compatible_transport_confidence_set_against,
};
use antecedent::analysis::per_population_discovery::{PopulationRows, discover_per_population};
use antecedent_core::{
    ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    TransportQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::selective::{CiTestSpec, Mark, StaticAlgorithm, StaticDiscoveryContract};
use antecedent_stats::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependenceTest,
    PreparedCiTest, StatsError,
};

const SEED: u64 = 4_417;

fn on_large_stack(name: &str, body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name(name.into())
        .stack_size(32 * 1024 * 1024)
        .spawn(body)
        .unwrap()
        .join()
        .unwrap();
}

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn vars(n: u32) -> Vec<VariableId> {
    (0..n).map(VariableId::from_raw).collect()
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(SEED)
}

// -- d-separation oracle (independent of the library's graphs) -----------------------------------

fn d_separated(parents: &[Vec<usize>], x: usize, y: usize, z: &[usize]) -> bool {
    let n = parents.len();
    let mut ancestral = vec![false; n];
    let mut stack: Vec<usize> = vec![x, y];
    stack.extend_from_slice(z);
    for &s in &stack {
        ancestral[s] = true;
    }
    while let Some(node) = stack.pop() {
        for &p in &parents[node] {
            if !ancestral[p] {
                ancestral[p] = true;
                stack.push(p);
            }
        }
    }
    let mut adj = vec![vec![false; n]; n];
    for node in 0..n {
        if !ancestral[node] {
            continue;
        }
        for &p in &parents[node] {
            adj[node][p] = true;
            adj[p][node] = true;
            for &q in &parents[node] {
                if p != q {
                    adj[p][q] = true;
                    adj[q][p] = true;
                }
            }
        }
    }
    let mut seen = vec![false; n];
    let mut frontier = vec![x];
    seen[x] = true;
    while let Some(node) = frontier.pop() {
        if node == y {
            return false;
        }
        for u in 0..n {
            if adj[node][u] && ancestral[u] && !seen[u] && !z.contains(&u) {
                seen[u] = true;
                frontier.push(u);
            }
        }
    }
    true
}

struct DsepOracle {
    parents: Vec<Vec<usize>>,
}

impl ConditionalIndependenceTest for DsepOracle {
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        _workspace: &mut CiWorkspace,
        _ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        prepared.ensure_compatible(request)?;
        let results = request
            .queries
            .iter()
            .map(|q| {
                let z = &request.z_flat[q.z_start..q.z_start + q.z_len];
                let separated = d_separated(&self.parents, q.x, q.y, z);
                CiResult {
                    statistic: if separated { 0.0 } else { 1.0 },
                    p_value: if separated { 1.0 } else { 0.0 },
                    df: 0.0,
                    ci: None,
                }
            })
            .collect();
        Ok(CiBatchResult { results })
    }
}

struct Lcg(u64);

impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (f64::from(u32::try_from(self.0 >> 40).unwrap()) + 0.5) / f64::from(1u32 << 24)
    }

    fn normal(&mut self) -> f64 {
        let (u, w) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * w).cos()
    }

    fn bernoulli(&mut self, p: f64) -> f64 {
        f64::from(u8::from(self.uniform() < p))
    }
}

/// `n_vars` named columns of independent noise: the oracle ignores the values.
fn noise(n_vars: usize, rows: usize, seed: u64) -> TabularData {
    let mut rng = Lcg(seed);
    let columns: Vec<(String, Vec<f64>)> =
        (0..n_vars).map(|i| (format!("v{i}"), (0..rows).map(|_| rng.normal()).collect())).collect();
    TabularData::from_f64_columns(columns.iter().map(|(n, c)| (n.as_str(), c.as_slice()))).unwrap()
}

fn oracle_contract(
    algorithm: StaticAlgorithm,
    label: &str,
    parents: Vec<Vec<usize>>,
    observed: u32,
) -> StaticDiscoveryContract {
    let mut contract = StaticDiscoveryContract::new(
        algorithm,
        vars(observed),
        CiTestSpec::declared(label, Arc::new(DsepOracle { parents })),
        SEED,
    );
    contract.fdr = None;
    contract
}

fn chain3() -> StaticDiscoveryContract {
    oracle_contract(StaticAlgorithm::Pc, "dsep:chain3", vec![vec![], vec![0], vec![1]], 3)
}

fn collider3() -> StaticDiscoveryContract {
    oracle_contract(StaticAlgorithm::Pc, "dsep:collider3", vec![vec![], vec![], vec![0, 1]], 3)
}

fn chain4() -> StaticDiscoveryContract {
    oracle_contract(StaticAlgorithm::Pc, "dsep:chain4", vec![vec![], vec![0], vec![1], vec![2]], 4)
}

/// `E[outcome | do(v0)]` from an experiment on `v0`, trial to clinic.
fn query(outcome: u32) -> TransportQuery {
    TransportQuery::new(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: v(outcome),
            treatment: ContinuousDomain::new(v(0), GridSpec::Values(Arc::from([0.0, 1.0]))),
        }),
        "trial",
        "clinic",
        [v(0)],
    )
}

fn compiled_run(request: DiscoveredClassTransportRequest) -> DiscoveredClassTransportRun {
    let compiled = Study::compile_discovered_class_transport(request, &ctx()).unwrap();
    compiled.execute(&ctx()).unwrap()
}

fn envelope_request(
    contract: StaticDiscoveryContract,
    data: TabularData,
    targets: Vec<VariableId>,
    outcome: u32,
    bound: usize,
) -> DiscoveredClassTransportRequest {
    DiscoveredClassTransportRequest::new(
        contract,
        data,
        targets,
        query(outcome),
        DiscoveredClassStage::Envelope,
    )
    .with_member_bound(bound)
}

/// The run's discovery and transport request equal the direct entry's.
fn assert_same_discovery(compiled: &DiscoveredClassTransportRun, direct: &DiscoveredTransportRun) {
    assert_eq!(compiled.outcome.result_identity, direct.outcome.result_identity);
    assert_eq!(compiled.outcome.structure_identity, direct.outcome.structure_identity);
    assert_eq!(
        format!("{:?}", compiled.outcome.structure),
        format!("{:?}", direct.outcome.structure)
    );
    assert_eq!(format!("{:?}", compiled.discovery), format!("{:?}", direct.discovery));
    assert_eq!(compiled.transport, direct.transport);
    if let (Ok(compiled), Ok(direct)) = (&compiled.transport, &direct.transport) {
        assert_eq!(compiled.identity(), direct.identity());
    }
}

fn answer_bits(answer: &TransportAnswer) -> Vec<u64> {
    let mut bits = Vec::new();
    for level in &answer.levels {
        bits.extend([level.level.to_bits(), level.low.to_bits(), level.high.to_bits()]);
    }
    if let Some((low, high)) = answer.contrast {
        bits.extend([low.to_bits(), high.to_bits()]);
    }
    if let Some(interval) = answer.interval {
        bits.extend([
            interval.low.to_bits(),
            interval.high.to_bits(),
            interval.coverage_target.to_bits(),
        ]);
    }
    bits
}

fn assert_same_estimate(
    compiled: &DiscoveredTransportEstimate,
    direct: &DiscoveredTransportEstimate,
) {
    assert_eq!(answer_bits(compiled.answer()), answer_bits(direct.answer()));
    assert_eq!(compiled.answer(), direct.answer());
    assert_eq!(compiled.identity(), direct.identity());
    assert_eq!(compiled.report_identity(), direct.report_identity());
    let levels = |e: &DiscoveredTransportEstimate| -> Vec<u64> {
        e.levels().iter().map(|x| x.to_bits()).collect()
    };
    assert_eq!(levels(compiled), levels(direct));
    assert_eq!(compiled.export().unwrap(), direct.export().unwrap());
    if let Some(interval) = compiled.answer().interval {
        assert_eq!(interval.calibration, direct.answer().interval.unwrap().calibration);
    }
}

mod envelope {
    use super::*;

    #[test]
    fn the_compiled_envelope_is_the_direct_envelope_on_cpdag_and_pag_classes() {
        on_large_stack("twin-envelope", || {
            let cases: Vec<(StaticDiscoveryContract, TabularData, Vec<VariableId>, u32, usize)> = vec![
                (chain3(), noise(3, 60, 1), vec![v(1)], 2, 64),
                (collider3(), noise(3, 60, 2), vec![v(1)], 2, 64),
                (chain4(), noise(4, 60, 8), vec![v(1)], 3, 2),
                (
                    oracle_contract(
                        StaticAlgorithm::Fci,
                        "dsep:chain3:fci",
                        vec![vec![], vec![0], vec![1]],
                        3,
                    ),
                    noise(3, 60, 9),
                    vec![v(1)],
                    2,
                    64,
                ),
                // A selection target outside the structure: the same typed refusal.
                (chain3(), noise(3, 60, 1), vec![v(7)], 2, 64),
                // A zero member bound: the same typed refusal.
                (chain3(), noise(3, 60, 1), vec![v(1)], 2, 0),
            ];
            for (contract, data, targets, outcome, bound) in cases {
                let direct = discover_and_transport(
                    &contract,
                    &data,
                    &targets,
                    &query(outcome),
                    bound,
                    &ctx(),
                )
                .unwrap();
                let compiled =
                    compiled_run(envelope_request(contract, data, targets, outcome, bound));
                assert_same_discovery(&compiled, &direct);
                assert!(compiled.stage.is_none());
                assert!(compiled.discovery_program_identity.is_some());
            }
        });
    }

    /// Independent check: the d-separation oracle makes PC return the chain's CPDAG, whose
    /// class is exactly the three DAGs without a collider; the structural identifier fixes the
    /// verdict of each.
    #[test]
    fn the_compiled_chain_class_holds_the_three_oracle_members_with_their_fixed_verdicts() {
        on_large_stack("truth-envelope", || {
            let run = compiled_run(envelope_request(chain3(), noise(3, 60, 1), vec![v(1)], 2, 64));
            let report = run.transport.unwrap();
            let envelope = report.envelope();
            assert_eq!(report.kind(), "structural_envelope");
            assert!(envelope.enumeration_complete());
            assert_eq!(envelope.members().len(), 3);
            assert_eq!(envelope.status(), EnvelopeStatus::IdentifiedInEveryMember);
            assert!(!envelope.formulas_agree());
            let verdict = |edges: &[(u32, u32)]| -> (String, &'static str) {
                let want: Vec<(VariableId, VariableId)> =
                    edges.iter().map(|&(p, c)| (v(p), v(c))).collect();
                let member = envelope
                    .members()
                    .iter()
                    .find(|m| m.directed_edges == want)
                    .unwrap_or_else(|| panic!("member {edges:?} missing"));
                match &member.verdict {
                    MemberVerdict::Identified { rule, formula } => {
                        (rule.clone(), formula_kind(formula))
                    }
                    other => panic!("{edges:?}: {other:?}"),
                }
            };
            assert_eq!(verdict(&[(0, 1), (1, 2)]).1, "recursive_factorization");
            verdict(&[(1, 0), (1, 2)]);
            assert_eq!(verdict(&[(1, 0), (2, 1)]), ("transport.sid.direct".to_owned(), "direct"));
        });
    }
}

// -- estimates across a PAG class -----------------------------------------------------------------

/// `x -> y <- z <- {c, d}`, `y -> r`, `z -> q`, and a latent `L` with `L -> w` and `L -> y`.
/// Variables: x0 y1 z2 c3 d4 w5 q6 r7 (latent 8).
fn latent_parents() -> Vec<Vec<usize>> {
    vec![vec![], vec![0, 2, 8], vec![3, 4], vec![], vec![], vec![8], vec![2], vec![1], vec![]]
}

fn p_y(x: f64, z: f64, l: f64) -> f64 {
    0.1 + 0.3 * x + 0.2 * z + 0.2 * l + 0.15 * x * z
}

fn p_z(c: f64, d: f64, target: bool) -> f64 {
    if target { 0.85 - 0.3 * c - 0.3 * d } else { 0.15 + 0.3 * c + 0.3 * d }
}

fn study_table(rows: usize, target: bool, seed: u64) -> TabularData {
    let mut rng = Lcg(seed);
    let mut columns = vec![Vec::with_capacity(rows); 8];
    for _ in 0..rows {
        let (c, d, l, x) =
            (rng.bernoulli(0.5), rng.bernoulli(0.5), rng.bernoulli(0.5), rng.bernoulli(0.5));
        let z = rng.bernoulli(p_z(c, d, target));
        let w = rng.bernoulli(0.2 + 0.6 * l);
        let y = rng.bernoulli(p_y(x, z, l));
        let q = rng.bernoulli(0.1 + 0.7 * z);
        let r = rng.bernoulli(0.2 + 0.6 * y);
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

/// `E*[y | do(x = a)]` enumerated from the target population's structural model.
fn target_truth(a: f64) -> f64 {
    let mut total = 0.0;
    for c in [0.0, 1.0] {
        for d in [0.0, 1.0] {
            for l in [0.0, 1.0] {
                for z in [0.0, 1.0] {
                    let pz = p_z(c, d, true);
                    let weight = 0.125 * if z > 0.5 { pz } else { 1.0 - pz };
                    total += weight * p_y(a, z, l);
                }
            }
        }
    }
    total
}

fn latent_fci() -> StaticDiscoveryContract {
    oracle_contract(StaticAlgorithm::Fci, "dsep:latent:fci", latent_parents(), 8)
}

fn effect_query() -> TransportQuery {
    query(1)
}

mod estimate {
    use super::*;

    fn estimate_of(run: DiscoveredClassTransportRun) -> DiscoveredTransportEstimate {
        match run.stage.unwrap().unwrap() {
            DiscoveredClassStageResult::Estimate(estimate) => *estimate,
            other => panic!("expected an estimate, got {other:?}"),
        }
    }

    #[test]
    fn the_compiled_pag_class_estimate_is_the_direct_estimate_and_covers_the_enumerated_truth() {
        on_large_stack("twin-estimate", || {
            let source = study_table(30_000, false, 11);
            let target = study_table(30_000, true, 12);
            let direct_run = discover_and_transport(
                &latent_fci(),
                &source,
                &[v(2)],
                &effect_query(),
                256,
                &ctx(),
            )
            .unwrap();
            let report = direct_run.transport.as_ref().unwrap();
            let direct = estimate_discovered_transport(
                report,
                &effect_query(),
                &TransportEvidence::new(&source, &target),
                &ctx(),
            )
            .unwrap();

            let request = DiscoveredClassTransportRequest::new(
                latent_fci(),
                source.clone(),
                vec![v(2)],
                effect_query(),
                DiscoveredClassStage::Estimate {
                    evidence: DiscoveredClassEvidence::new(source.clone(), target.clone()),
                },
            );
            let run = compiled_run(request);
            assert_same_discovery(&run, &direct_run);
            let compiled = estimate_of(run);
            assert_same_estimate(&compiled, &direct);

            // Independent check: the transported response enumerated from the target model.
            let tolerance = 0.014;
            let answer = compiled.answer();
            for (level, truth) in answer.levels.iter().zip([target_truth(0.0), target_truth(1.0)]) {
                assert!(
                    level.low - tolerance <= truth && truth <= level.high + tolerance,
                    "{level:?} vs {truth}"
                );
            }
            let (low, high) = answer.contrast.unwrap();
            let truth = target_truth(1.0) - target_truth(0.0);
            assert!(
                low - tolerance <= truth && truth <= high + tolerance,
                "{low} {high} vs {truth}"
            );
        });
    }
}

// -- trial-compatible refinement and the bounded-outcome confidence set ------------------------------

mod trial {
    use super::*;

    const TRIAL_SEED: u64 = 71;

    fn trial_ctx() -> ExecutionContext {
        ExecutionContext::for_tests(TRIAL_SEED)
    }

    /// The fixture of the trial module's confidence-set tests: a completely randomized binary
    /// trial on 200 rows, FCI with the analytic G-squared test.
    fn fixture() -> (StaticDiscoveryContract, TabularData, TransportQuery) {
        let x: Vec<f64> = (0..200).map(|i| f64::from(i >= 100)).collect();
        let y: Vec<f64> = (0..200).map(|i| f64::from(i < 20 || (100..180).contains(&i))).collect();
        let source =
            TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
        let treatment = v(0);
        let query = TransportQuery::new(
            ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: v(1),
                treatment: ContinuousDomain::new(
                    treatment,
                    GridSpec::Values(Arc::from([0.0, 1.0])),
                ),
            }),
            "trial",
            "target",
            [treatment],
        );
        let contract = StaticDiscoveryContract::new(
            StaticAlgorithm::Fci,
            vec![treatment, v(1)],
            CiTestSpec::declared(
                "gsquared_analytic",
                Arc::new(antecedent_stats::ci::GSquared::new()),
            ),
            TRIAL_SEED,
        );
        (contract, source, query)
    }

    fn options() -> TrialCompatibilityOptions {
        TrialCompatibilityOptions {
            alpha: 0.01,
            randomization: TrialRandomization::CompletelyRandomized,
        }
    }

    fn support() -> TransportOutcomeSupport {
        TransportOutcomeSupport { lower: 0.0, upper: 1.0 }
    }

    fn evidence(source: &TabularData) -> DiscoveredClassEvidence {
        let mut evidence = DiscoveredClassEvidence::new(source.clone(), source.clone());
        evidence.bootstrap_replicates = 64;
        evidence
    }

    fn run(stage: DiscoveredClassStage) -> DiscoveredClassTransportRun {
        let (contract, source, query) = fixture();
        let request =
            DiscoveredClassTransportRequest::new(contract, source, Vec::new(), query, stage)
                .with_member_bound(32);
        Study::compile_discovered_class_transport(request, &trial_ctx())
            .unwrap()
            .execute(&trial_ctx())
            .unwrap()
    }

    #[test]
    fn the_compiled_trial_estimate_and_confidence_set_are_the_direct_ones_and_replay() {
        on_large_stack("twin-trial", || {
            let (contract, source, query) = fixture();
            let direct_run =
                discover_and_transport(&contract, &source, &[], &query, 32, &trial_ctx()).unwrap();
            let report = direct_run.transport.as_ref().unwrap();
            let mut direct_evidence = TransportEvidence::new(&source, &source);
            direct_evidence.bootstrap_replicates = 64;

            // The trial-compatible estimate.
            let direct = estimate_trial_compatible_discovered_transport(
                report,
                &query,
                &direct_evidence,
                options(),
                &trial_ctx(),
            )
            .unwrap();
            let compiled_run = run(DiscoveredClassStage::TrialCompatibleEstimate {
                evidence: evidence(&source),
                options: options(),
            });
            assert_same_discovery(&compiled_run, &direct_run);
            let compiled = match compiled_run.stage.unwrap().unwrap() {
                DiscoveredClassStageResult::TrialCompatibleEstimate(estimate) => *estimate,
                other => panic!("expected a trial-compatible estimate, got {other:?}"),
            };
            assert_same_estimate(compiled.estimate(), direct.estimate());
            assert_eq!(compiled.compatibility(), direct.compatibility());
            assert_eq!(compiled.export().unwrap(), direct.export().unwrap());

            // The bounded-outcome confidence set, with its calibration annotation.
            let direct_set = estimate_trial_compatible_transport_confidence_set(
                report,
                &query,
                &direct_evidence,
                options(),
                support(),
                &trial_ctx(),
            )
            .unwrap();
            let set_of = |run: DiscoveredClassTransportRun| match run.stage.unwrap().unwrap() {
                DiscoveredClassStageResult::TrialConfidenceSet(set) => *set,
                other => panic!("expected a confidence set, got {other:?}"),
            };
            let compiled_set = set_of(run(DiscoveredClassStage::TrialConfidenceSet {
                evidence: evidence(&source),
                options: options(),
                support: support(),
            }));
            let bits = |(low, high): (f64, f64)| [low.to_bits(), high.to_bits()];
            assert_eq!(bits(compiled_set.bounds()), bits(direct_set.bounds()));
            assert_eq!(compiled_set.kind(), direct_set.kind());
            assert_eq!(compiled_set.method(), direct_set.method());
            assert_eq!(compiled_set.calibration(), direct_set.calibration());
            assert_eq!(compiled_set.execution_seed(), direct_set.execution_seed());
            assert_eq!(compiled_set.input_identity(), direct_set.input_identity());
            assert_eq!(compiled_set.identity().unwrap(), direct_set.identity().unwrap());
            let receipt = direct_set.export().unwrap();
            assert_eq!(compiled_set.export().unwrap(), receipt);

            // The replay of the receipt against the same inputs.
            let direct_replay = verify_trial_compatible_transport_confidence_set_against(
                &receipt,
                report,
                &query,
                &direct_evidence,
                options(),
                support(),
                &trial_ctx(),
            )
            .unwrap();
            let compiled_replay = set_of(run(DiscoveredClassStage::TrialConfidenceSetReplay {
                receipt: receipt.clone(),
                evidence: evidence(&source),
                options: options(),
                support: support(),
            }));
            assert_eq!(bits(compiled_replay.bounds()), bits(direct_replay.bounds()));
            assert_eq!(compiled_replay.identity().unwrap(), direct_replay.identity().unwrap());

            // A receipt that does not replay is the same typed refusal.
            let mut forged = receipt;
            let last = forged.len() - 1;
            forged[last] ^= 1;
            let direct_refusal = verify_trial_compatible_transport_confidence_set_against(
                &forged,
                report,
                &query,
                &direct_evidence,
                options(),
                support(),
                &trial_ctx(),
            )
            .map_err(antecedent::analysis::discovered_transport_estimate::discovered_transport_io_failure)
            .unwrap_err();
            let compiled_refusal = run(DiscoveredClassStage::TrialConfidenceSetReplay {
                receipt: forged,
                evidence: evidence(&source),
                options: options(),
                support: support(),
            })
            .stage
            .unwrap()
            .unwrap_err();
            assert_eq!(compiled_refusal, direct_refusal);
        });
    }
}

// -- program identity -----------------------------------------------------------------------------

mod identity {
    use super::*;

    fn identity_of(request: DiscoveredClassTransportRequest) -> String {
        Study::compile_discovered_class_transport(request, &ctx())
            .unwrap()
            .lowered()
            .identity()
            .to_hex()
    }

    #[test]
    fn the_discovered_transport_identity_binds_the_search_the_rows_the_query_and_the_stage() {
        on_large_stack("identity-envelope", || {
            let base = || envelope_request(chain3(), noise(3, 60, 1), vec![v(1), v(2)], 2, 64);
            let same = identity_of(base());
            assert_eq!(same, identity_of(base()), "the same request is the same program");
            // Selection targets are a set: order and repetition do not change the program.
            assert_eq!(
                same,
                identity_of(envelope_request(
                    chain3(),
                    noise(3, 60, 1),
                    vec![v(2), v(1), v(2)],
                    2,
                    64
                ))
            );
            let changed = [
                envelope_request(chain3(), noise(3, 60, 1), vec![v(1)], 2, 64),
                envelope_request(chain3(), noise(3, 60, 1), vec![v(1), v(2)], 2, 63),
                envelope_request(chain3(), noise(3, 60, 1), vec![v(1), v(2)], 1, 64),
                envelope_request(chain3(), noise(3, 60, 3), vec![v(1), v(2)], 2, 64),
                envelope_request(collider3(), noise(3, 60, 1), vec![v(1), v(2)], 2, 64),
            ];
            for request in changed {
                assert_ne!(same, identity_of(request));
            }
            // The stage and every setting of its evidence enter the program.
            let rows = noise(3, 60, 1);
            let estimate = |replicates: u32| {
                let mut evidence = DiscoveredClassEvidence::new(rows.clone(), rows.clone());
                evidence.bootstrap_replicates = replicates;
                DiscoveredClassTransportRequest::new(
                    chain3(),
                    rows.clone(),
                    vec![v(1)],
                    query(2),
                    DiscoveredClassStage::Estimate { evidence },
                )
            };
            assert_ne!(identity_of(estimate(0)), identity_of(estimate(10)));
            assert_ne!(
                identity_of(estimate(0)),
                identity_of(envelope_request(chain3(), rows.clone(), vec![v(1)], 2, 256))
            );
        });
    }

    #[test]
    fn the_per_population_identity_is_the_sorted_populations_their_rows_and_the_search() {
        let (north, south) = (super::per_population::chain(1), super::per_population::collider(2));
        let identity = |populations: Vec<(&str, &TabularData)>,
                        contract: StaticDiscoveryContract| {
            let request = PerPopulationSearchRequest::new(
                contract,
                populations.into_iter().map(|(l, d)| (l.to_owned(), d.clone())).collect(),
            );
            Study::compile_per_population_discovery(request, &ctx())
                .unwrap()
                .lowered()
                .identity()
                .to_hex()
        };
        let contract = super::per_population::contract;
        let same = identity(vec![("north", &north), ("south", &south)], contract());
        assert_eq!(same, identity(vec![("south", &south), ("north", &north)], contract()));
        assert_ne!(same, identity(vec![("north", &south), ("south", &north)], contract()));
        assert_ne!(same, identity(vec![("north", &north), ("east", &south)], contract()));
        let other_rows = super::per_population::collider(3);
        assert_ne!(same, identity(vec![("north", &north), ("south", &other_rows)], contract()));
        let mut stricter = contract();
        stricter.constraints.alpha = 0.0001;
        assert_ne!(same, identity(vec![("north", &north), ("south", &south)], stricter));
    }
}

// -- per-population discovery -----------------------------------------------------------------------

mod per_population {
    use super::*;

    const ROWS: usize = 4_000;
    const POPULATION_SEED: u64 = 31_337;

    fn frame(x: &[f64], y: &[f64], z: &[f64]) -> TabularData {
        TabularData::from_f64_columns([("x", x), ("y", y), ("z", z)]).unwrap()
    }

    /// `x -> y -> z`.
    pub(super) fn chain(seed: u64) -> TabularData {
        let mut rng = Lcg(seed);
        let (mut x, mut y, mut z) = (vec![], vec![], vec![]);
        for _ in 0..ROWS {
            let a = rng.normal();
            let b = 0.8 * a + 0.6 * rng.normal();
            let c = 0.8 * b + 0.6 * rng.normal();
            x.push(a);
            y.push(b);
            z.push(c);
        }
        frame(&x, &y, &z)
    }

    /// `x -> y <- z`.
    pub(super) fn collider(seed: u64) -> TabularData {
        let mut rng = Lcg(seed);
        let (mut x, mut y, mut z) = (vec![], vec![], vec![]);
        for _ in 0..ROWS {
            let a = rng.normal();
            let c = rng.normal();
            let b = 0.8 * a + 0.8 * c + 0.5 * rng.normal();
            x.push(a);
            y.push(b);
            z.push(c);
        }
        frame(&x, &y, &z)
    }

    pub(super) fn contract() -> StaticDiscoveryContract {
        let mut contract = StaticDiscoveryContract::new(
            StaticAlgorithm::Pc,
            vars(3),
            CiTestSpec::named("parcorr"),
            POPULATION_SEED,
        );
        contract.fdr = None;
        contract.constraints.alpha = 0.001;
        contract
    }

    fn population_ctx() -> ExecutionContext {
        ExecutionContext::for_tests(POPULATION_SEED)
    }

    fn request(populations: &[(&str, &TabularData)]) -> PerPopulationSearchRequest {
        PerPopulationSearchRequest::new(
            contract(),
            populations.iter().map(|(l, d)| ((*l).to_owned(), (*d).clone())).collect(),
        )
    }

    #[test]
    fn the_compiled_per_population_report_is_the_direct_report() {
        let (north, south) = (chain(1), collider(2));
        let direct = discover_per_population(
            &contract(),
            &[
                PopulationRows { population: "south", data: &south },
                PopulationRows { population: "north", data: &north },
            ],
            &population_ctx(),
        )
        .unwrap();
        let compiled = Study::compile_per_population_discovery(
            request(&[("south", &south), ("north", &north)]),
            &population_ctx(),
        )
        .unwrap();
        let run = compiled.execute(&population_ctx()).unwrap();
        assert_eq!(run.lowered_program_identity(), compiled.lowered().identity());
        let (ours, theirs) = (run.report().unwrap(), direct.report().unwrap());
        assert_eq!(ours.identity, theirs.identity);
        assert_eq!(ours.edges, theirs.edges);
        assert_eq!(ours.populations.len(), theirs.populations.len());
        for (a, b) in ours.populations.iter().zip(&theirs.populations) {
            assert_eq!(a.population, b.population);
            assert_eq!(a.data_identity, b.data_identity);
            assert_eq!(a.structure_identity, b.structure_identity);
            assert_eq!(a.result_identity, b.result_identity);
            assert_eq!(a.receipts_identity, b.receipts_identity);
            assert_eq!(a.retained_statistics, b.retained_statistics);
            assert_eq!(a.search_incomplete, b.search_incomplete);
            assert_eq!(a.unresolved_edges, b.unresolved_edges);
            assert_eq!(format!("{:?}", a.report), format!("{:?}", b.report));
        }
        for label in ["north", "south"] {
            assert_eq!(
                run.driver().adapter(label).unwrap().retained_statistics(),
                direct.adapter(label).unwrap().retained_statistics()
            );
        }
    }

    /// Independent check: the chain's class keeps both edges undirected and the collider
    /// compels both, so both edges are present in both populations with different marks.
    #[test]
    fn the_compiled_per_population_structures_differ_on_exactly_the_hand_derived_marks() {
        let (north, south) = (chain(1), collider(2));
        let run = Study::compile_per_population_discovery(
            request(&[("north", &north), ("south", &south)]),
            &population_ctx(),
        )
        .unwrap()
        .execute(&population_ctx())
        .unwrap();
        let report = run.report().unwrap();
        let labels: Vec<&str> = report.populations.iter().map(|p| p.population.as_str()).collect();
        assert_eq!(labels, vec!["north", "south"]);
        assert_eq!(report.populations[0].unresolved_edges, 2);
        assert_eq!(report.populations[1].unresolved_edges, 0);
        assert_eq!(report.edges.len(), 2);
        assert_eq!(report.differing_edges().len(), 2);
        assert_eq!(report.edges[0].marks[1].1, Some((Mark::Tail, Mark::Arrow)));
        assert_eq!(report.edges[1].marks[1].1, Some((Mark::Arrow, Mark::Tail)));
    }

    #[test]
    fn a_pooled_request_is_the_drivers_typed_refusal_through_the_program() {
        let north = chain(1);
        let pooled = request(&[("north", &north), ("south", &north)]);
        let direct = discover_per_population(
            &contract(),
            &[
                PopulationRows { population: "north", data: &north },
                PopulationRows { population: "south", data: &north },
            ],
            &population_ctx(),
        )
        .unwrap_err();
        let compiled = Study::compile_per_population_discovery(pooled, &population_ctx())
            .unwrap()
            .execute(&population_ctx())
            .unwrap_err();
        let PerPopulationSearchError::Search(compiled) = compiled else {
            panic!("expected the driver's refusal, got {compiled}");
        };
        assert_eq!(compiled.code(), direct.code());
        assert_eq!(compiled.to_string(), direct.to_string());
        assert_eq!(
            compiled.to_string().split(':').next(),
            Some(antecedent_core::reason_details::selective_discovery::POOLED_POPULATION_DATA)
        );
    }

    #[test]
    fn a_compiled_program_refuses_reentry() {
        let (north, south) = (chain(1), collider(2));
        let compiled = Study::compile_per_population_discovery(
            request(&[("north", &north), ("south", &south)]),
            &population_ctx(),
        )
        .unwrap();
        let inside = population_ctx().enter_lowered_dispatch(compiled.lowered().identity());
        assert!(matches!(compiled.execute(&inside), Err(PerPopulationSearchError::Program(_))));
    }
}

/// The direct refusal type is reached through the program unchanged.
#[test]
fn a_transport_refusal_keeps_the_discovery() {
    on_large_stack("refusal-keeps-discovery", || {
        let run = compiled_run(envelope_request(chain3(), noise(3, 60, 1), vec![v(9)], 2, 64));
        assert!(matches!(run.transport, Err(DiscoveredTransportError::Refused(_))));
        assert!(run.outcome.structure.edges().len() == 2);
    });
}
