//! Target-native validation of a compiled result, through the compiler.
//!
//! A result's refutation target is read from the result itself and bound to the identity of
//! the lowered program that produced it:
//!
//! * [`result_refutation_target`] names the quantity the result reports — its
//!   [`TargetFamily`], its quadrant (the lowered program's data classification and uncertainty
//!   reading) and one label per coordinate — with the lowered-program identity as the target's
//!   defining source, so a refutation computed under another program is another target.
//! * [`LoweredProgram::target_refutation_plan`] attaches the target-native refuter catalogue
//!   ([`TargetRefutationPlan::for_target`]) to the lowered program.
//! * [`refute_prepared_target`] executes that plan through the prepared handle: every refit
//!   perturbs the data and re-executes the handle's compiled program in full, then reads the
//!   whole target vector back from the refitted result. Refuters the generic data perturbation
//!   cannot construct for a route are returned as typed refusals naming the next attempt.
//! * [`certify_result_refutations`] binds target-native reports to the result's claim and
//!   refuses any report on another target; the claim itself refuses a refutation its own
//!   program did not produce when its target is not the scalar average effect
//!   (`refuse_scalar_refutations_for_result`, enforced by [`StudyResult::claim`] and
//!   [`certify_result_refutations`]). Route
//!   reports are stamped with the producing lowered program where an execution returns its
//!   result, so a route's own reports always pass.
//!
//! Every report is a diagnostic: thresholds are conventions, and calibration of the
//! refuters is not measured here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::DataClassification;
use antecedent_core::{
    CausalQuery, CausalRng, ContinuousDomain, DerivativeScale, DerivativeWeighting,
    ExecutionContext, GridSpec, Intervention, Lag, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseUncertainty, ResponseValue, Value, VariableId,
};
use antecedent_data::{OwnedColumn, PanelData, PanelUnit, TableView, TabularData, TimeSeriesData};
use antecedent_graph::{DSeparationWorkspace, Dag, DenseNodeId, TemporalDag, ensure_lagged};
use antecedent_kernels::{sample_gamma, shuffle, unbiased_index};
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};
use antecedent_validate::RefutationReport;
use antecedent_validate::target_native::{
    DerivativeCheckPoint, DerivativeConsistency, LocalLinearity, PosteriorPredictiveTarget,
    PriorSensitivityTarget, certify_target_refutations, finite_difference_consistency,
    local_linearity, posterior_predictive_target_check, prior_sensitivity_target_check,
    refuse_scalar_reports_for_target, run_target_plan, target_refuter_stream,
};
pub use antecedent_validate::target_native::{
    JointStatistic, JointStatisticKind, RefutationTarget, TargetCertificate,
    TargetCertificationError, TargetFamily, TargetPerturbation, TargetQuadrant, TargetRefit,
    TargetRefitError, TargetRefutationConfig, TargetRefutationPlan, TargetRefutationReport,
    TargetRefutationRun, TargetRefuterKind, TargetRefuterRefusal, TargetSuite,
};
use antecedent_validate::{ValidationError, mediation_sensitivity_fit};

use super::builder::DataInput;
use super::prepared::PreparedStudy;
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::planner::lowered::{LoweredProgram, UncertaintyReading};
use crate::result::StudyResult;

/// Float columns to replace, each with its new values.
type ColumnReplacements = Vec<(VariableId, Arc<[f64]>)>;

/// Stamp every refutation `result` carries with the lowered program that produced it.
///
/// Called where an execution of a compiled program returns its result (dispatch, the prepared
/// click, the prepared contract binding and the prepared second-click refute): the reports in
/// the result at that point were produced by that program for its own target, whatever
/// suite produced them. A report already stamped keeps its producer.
pub(crate) fn stamp_route_refutations(result: &mut StudyResult) {
    if !matches!(&result.authority, crate::result::RuntimeRefutationAuthority::Fresh) {
        return;
    }
    if let Some(program) = result.lowered_identity {
        let program = *program.as_bytes();
        for report in &mut result.refutations {
            report.stamp_producer(program);
        }
        // Repeated binding is common, but must never bless edits to a previously
        // issued result. Only a checked follow-up operation can issue anew.
        let _ = seal_runtime_refutations(result, crate::result::RuntimeRefutationOrigin::Route);
    }
}

impl LoweredProgram {
    /// Quadrant of this program: temporal when the scientific problem reads series, panel or
    /// event data, or multiple environments under a temporal structure; Bayesian when the
    /// claim reads a posterior.
    #[must_use]
    pub fn refutation_quadrant(&self) -> TargetQuadrant {
        TargetQuadrant {
            temporal: temporal_refutation_data(
                self.scientific().data(),
                Some(self.scientific().graph_class()),
            ),
            bayesian: self.claim().uncertainty() == UncertaintyReading::Posterior,
        }
    }

    /// The refutation target of `family` with `labels`, defined by this program's identity.
    ///
    /// # Errors
    ///
    /// Labels the target refuses (none, or several for a scalar average effect).
    pub fn refutation_target(
        &self,
        family: TargetFamily,
        labels: Vec<Arc<str>>,
    ) -> Result<RefutationTarget, CausalError> {
        let identity = self.semantic_identity().ok_or(CausalError::Unsupported {
            message: "a runtime callback declaration cannot identify an executable refutation target",
        })?;
        RefutationTarget::new(family, self.refutation_quadrant(), labels, *identity.as_bytes())
            .map_err(CausalError::from)
    }

    /// The target-native refuter plan of this program for `family` and `labels`.
    ///
    /// # Errors
    ///
    /// As [`Self::refutation_target`].
    pub fn target_refutation_plan(
        &self,
        family: TargetFamily,
        labels: Vec<Arc<str>>,
        suite: TargetSuite,
    ) -> Result<TargetRefutationPlan, CausalError> {
        Ok(TargetRefutationPlan::for_target(self.refutation_target(family, labels)?, suite))
    }
}

/// Internal stable selector for one actual point-producing temporal class member.
/// Enumeration position is deliberately not part of the identity.
#[derive(Clone, Debug)]
struct TemporalResponseMemberSelector {
    outer_atom: u64,
    horizon_index: usize,
    horizon: u32,
    completion_digest: [u8; 32],
}

mod contributors;
#[cfg(test)]
use contributors::compare_static_class_witnesses;
use contributors::{
    check_static_response_contributors, check_temporal_response_contributors,
    exact_admg_contributors, same_admg_posterior_source, same_static_class_witnesses,
    temporal_response_member_values,
};

mod target;

pub(crate) use target::canonical_law_atoms;
use target::temporal_refutation_data;
pub use target::{
    result_quadrant, result_refutation_target, result_target_family, result_target_values,
};

mod certification;

pub use certification::certify_result_refutations;
pub(super) use certification::{
    descriptive_route_report_source, seal_runtime_refutations, validate_runtime_refutation_proof,
};
pub(crate) use certification::{refuse_scalar_refutations_for_result, reset_pooled_refutations};

/// Declared controls and settings of a prepared target refutation.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetRefitOptions {
    /// A variable known not to cause the outcome, swapped in for the treatment. When `None`,
    /// the graph supplies one: a float variable outside the adjustment set that is not a
    /// descendant of the treatment and is d-separated from the outcome given the adjustment
    /// set (static graphs), or none of whose nodes reaches the outcome (temporal graphs).
    pub negative_control_exposure: Option<VariableId>,
    /// A variable the treatment is known not to cause, swapped in for the outcome. When
    /// `None`, the graph supplies one on a static graph (not a descendant of the treatment,
    /// d-separated from it given the adjustment set); on series and panels the default is the
    /// lagged placebo: the outcome moved back past the longest lag of the graph, which no
    /// later treatment can affect.
    pub negative_control_outcome: Option<VariableId>,
    /// The single mediator of a mediation target, for the sequential-ignorability range.
    pub mediator: Option<VariableId>,
    /// Smallest absolute break-even mediator-outcome error correlation that passes.
    pub mediation_rho_bar: f64,
    /// Strength with which the simulated common cause enters the outcome (and a non-binary
    /// treatment). Zero simulates an independent cause; either way the cause is declared in
    /// the graph as a parent of treatment and outcome and the program adjusts for it.
    pub common_cause_strength: f64,
    /// Alternative isotropic prior scales, as multiples of the study's prior scale.
    pub prior_scale_factors: Vec<f64>,
    /// Largest acceptable shift of the target posterior, in posterior standard deviations.
    pub prior_shift_tolerance: f64,
    /// Step of the derivative diagnostics; `None` takes a quarter of the treatment's sample
    /// standard deviation.
    pub derivative_step: Option<f64>,
    /// Relative tolerance of the finite-difference consistency check.
    pub derivative_tolerance: f64,
    /// Largest acceptable relative nonlinearity over the step.
    pub linearity_tolerance: f64,
    /// Draw the series bootstrap, pre-trend, anticipation and lagged-placebo refits as
    /// explicit gathered circular-block series and cut windows, as a route that reads no row
    /// weights requires, instead of as row multiplicities. A route that reads row weights
    /// keeps the multiplicities unless this is set.
    pub gathered_resamples: bool,
    /// The declared division of a class's mass among its DAG members, which a common cause
    /// declared in every atom of a class posterior needs. `None` refuses such a posterior.
    pub within_class: Option<WithinClassDistribution>,
}

/// How the mass of a Markov equivalence class divides among its DAG members, an assumption the
/// posterior does not state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WithinClassDistribution {
    /// Every DAG member of the class has the same share.
    UniformOverMembers,
}

/// One report on the whole target vector from the per-coordinate derivative checks: the
/// coordinates (horizons, or the outcomes and directions of a vector derivative) are the
/// evaluation points of a single check, in target order.
fn merge_derivative_checks(
    kind: TargetRefuterKind,
    coordinates: Vec<DerivativeCoordinate>,
    target: &RefutationTarget,
) -> Result<TargetRefutationReport, TargetRefitError> {
    if kind == TargetRefuterKind::FiniteDifferenceConsistency {
        let mut merged: Option<DerivativeConsistency> = None;
        for coordinate in coordinates {
            let DerivativeCoordinate::FiniteDifference(check) = coordinate else {
                continue;
            };
            match &mut merged {
                None => merged = Some(check),
                Some(all) => {
                    all.points.extend(check.points);
                    all.reported.extend(check.reported);
                    all.finite_difference.extend(check.finite_difference);
                    all.truncation_error.extend(check.truncation_error);
                    all.standardized.extend(check.standardized);
                }
            }
        }
        let check = merged.ok_or_else(|| unsupported("no value", "a target"))?;
        Ok(check.to_report(target)?)
    } else {
        let mut merged: Option<LocalLinearity> = None;
        for coordinate in coordinates {
            let DerivativeCoordinate::Linearity(check) = coordinate else {
                continue;
            };
            match &mut merged {
                None => merged = Some(check),
                Some(all) => {
                    all.points.extend(check.points);
                    all.slope.extend(check.slope);
                    all.curvature.extend(check.curvature);
                    all.nonlinearity.extend(check.nonlinearity);
                }
            }
        }
        let check = merged.ok_or_else(|| unsupported("no value", "a target"))?;
        Ok(check.to_report(target)?)
    }
}

/// A lagged-template mask repacked from `n` to `m` nodes: the bit of the edge `from -> to` at
/// lag `lag + 1` moves from `lag * n * n + from * n + to` to `lag * m * m + from * m + to`.
fn repack_lag_mask(
    old: &antecedent_discovery::LagMask,
    n: usize,
    m: usize,
    max_lag: usize,
) -> antecedent_discovery::LagMask {
    let mut packed = antecedent_discovery::LagMask::EMPTY;
    for bit in old.set_bits().map(|bit| bit as usize).filter(|&bit| bit < max_lag * n * n) {
        let (lag, from, to) = (bit / (n * n), (bit % (n * n)) / n, bit % n);
        packed.set_bit(u32::try_from(lag * m * m + from * m + to).unwrap_or(u32::MAX), true);
    }
    packed
}

#[cfg(test)]
mod repack_tests {
    use antecedent_validate::target_native::{TargetFamily, TargetQuadrant};

    use super::*;

    fn test_join_rows(map: &[usize], span: usize, block: Option<usize>) -> JoinFreeRows {
        let ctx = ExecutionContext::for_tests(0);
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        join_free_rows(map, span, block, &mut budget, &ctx).unwrap()
    }

    #[test]
    fn block_boundary_cannot_bridge_even_successive_source_ticks() {
        let ticks = [3, 4, 5, 6, 7, 8];
        assert_eq!(*test_join_rows(&ticks, 2, Some(3)), [false, true, true, false, true, true]);
        assert_eq!(*test_join_rows(&ticks, 3, Some(3)), [false, false, true, false, false, true]);
        assert_eq!(*test_join_rows(&ticks, 2, None), [false, true, true, true, true, true]);
    }

    /// A gathered circular-block series, with a series whose every value names its own period:
    /// the lag pair a design row reads is `(source of row - k, source of row)`, and a pair is
    /// an observed one exactly when its sources are `k` periods apart. Built here from the
    /// block definition, not from the product's alignment test.
    #[test]
    fn a_gathered_block_bootstrap_hides_every_design_row_whose_lag_window_crosses_a_join() {
        let n = 60;
        for (seed, span) in [(1_u64, 2_usize), (2, 3), (3, 2), (4, 4)] {
            let mut rng = antecedent_core::RngFactory::from_seed(seed)
                .stream_for(antecedent_core::StreamDomain::Test, 9);
            let length = antecedent_data::circular_block_length(span, n);
            let mut map = vec![0_usize; n];
            antecedent_data::fill_circular_block_indexes(n, length, &mut rng, &mut map).unwrap();
            let free = test_join_rows(&map, span, Some(length));
            let mut joins = 0;
            for row in 0..n {
                // Brute force from the definition: the window of `row` is the original's when
                // each earlier member is the source period exactly its lag before (the wrap of
                // a block past the last period joins the last period to the first, so it is a
                // join); the first rows have no earlier member to cross a join with.
                let observed = row + 1 >= span
                    && row % length + 1 >= span
                    && (1..span).all(|k| map[row] >= k && map[row - k] == map[row] - k);
                assert_eq!(free[row], observed, "row {row}, span {span}");
                joins += usize::from(!free[row]);
            }
            assert!(joins > 0, "the construction must contain joins to be a test");
            // The mask carries it: the rows an attribution fits are the join-free ones.
            let storage = TimeSeriesData::from_f64_columns(
                [("x", vec![0.0; n].as_slice()), ("y", vec![0.0; n].as_slice())],
                1,
            )
            .unwrap();
            let mask = join_free_mask(
                storage.storage(),
                &map,
                span,
                Some(length),
                &mut antecedent_core::TemporalBudget::new(
                    ExecutionContext::for_tests(0).temporal_materialization_limits(),
                ),
                &ExecutionContext::for_tests(0),
            )
            .unwrap();
            for row in 0..n {
                assert_eq!(mask.is_valid(row), free[row]);
            }
        }
    }

    /// A contiguous sub-window has no join: its source map is consecutive, so every design row
    /// reads the lag window the original reads (judged by the same brute-force definition as
    /// the block bootstrap above), for any window length and span.
    #[test]
    fn a_contiguous_subsample_window_has_no_crossing_design_row() {
        let mut rng = antecedent_core::RngFactory::from_seed(3)
            .stream_for(antecedent_core::StreamDomain::Test, 4);
        for (rows, span) in [(12_usize, 2_usize), (30, 3), (7, 4), (100, 2)] {
            for _ in 0..50 {
                let (start, width) = subsample_bounds(rows, span, &mut rng);
                assert!(width >= (span + 2).min(rows) && start + width <= rows);
                let map: Vec<usize> = (start..start + width).collect();
                let free = test_join_rows(&map, span, None);
                for row in 0..width {
                    let observed = row + 1 >= span
                        && (1..span).all(|k| map[row] >= k && map[row - k] == map[row] - k);
                    assert_eq!(observed, free[row], "rows {rows}, span {span}, row {row}");
                }
            }
        }
    }

    /// The multiplicity form of the block bootstrap (read by the routes that take row weights)
    /// puts every unit of weight on a source design row, which keeps its own lag window, so a
    /// join cannot appear: the weights are exactly the counts of the source rows the gathered
    /// form draws, with the same stream.
    #[test]
    fn block_multiplicities_are_the_counts_of_source_design_rows() {
        for (n, span) in [(40_usize, 2_usize), (57, 3)] {
            let draw = || {
                antecedent_core::RngFactory::from_seed(8)
                    .stream_for(antecedent_core::StreamDomain::Test, 1)
            };
            let weights = block_weights(n, span, &mut draw()).unwrap();
            let length = antecedent_data::circular_block_length(span, n);
            let mut map = vec![0_usize; n];
            antecedent_data::fill_circular_block_indexes(n, length, &mut draw(), &mut map).unwrap();
            let mut counts = vec![0.0_f64; n];
            for &source in &map {
                counts[source] += 1.0;
            }
            assert_eq!(&weights[..], &counts[..]);
            assert!((weights.iter().sum::<f64>() - n as f64).abs() < 1e-12);
        }
    }

    #[test]
    fn multi_environment_refutation_quadrant_preserves_the_retained_structure_kind() {
        let data = DataClassification::MultiEnvironment;
        for class in [
            crate::GraphClass::TemporalDag,
            crate::GraphClass::TemporalCpdag,
            crate::GraphClass::TemporalPag,
            crate::GraphClass::TemporalAdmg,
        ] {
            assert!(temporal_refutation_data(data, Some(class.as_str())));
        }
        for class in [
            crate::GraphClass::Dag,
            crate::GraphClass::Admg,
            crate::GraphClass::Cpdag,
            crate::GraphClass::Pag,
        ] {
            assert!(!temporal_refutation_data(data, Some(class.as_str())));
        }
        assert!(!temporal_refutation_data(data, Some("cyclic")));
        assert!(!temporal_refutation_data(data, None));
        assert!(temporal_refutation_data(DataClassification::Temporal, None));
    }

    #[test]
    fn retained_multi_environment_pulse_and_its_lowered_plan_are_both_temporal() {
        let environments = (0..3)
            .map(|environment| {
                let x: Vec<_> = (0..160)
                    .map(|row| (f64::from(row) * 1.37 + f64::from(environment)).sin())
                    .collect();
                let y: Vec<_> = (0..160)
                    .map(|row| {
                        let past = if row == 0 { 0.0 } else { x[row - 1] };
                        let time = f64::from(u32::try_from(row).unwrap());
                        1.0 + 2.0 * past + 0.1 * (time * 2.17).cos()
                    })
                    .collect();
                TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let data = antecedent_data::MultiEnvironmentData::try_new(environments).unwrap();
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        let mut graph = TemporalDag::empty();
        let x1 = ensure_lagged(&mut graph, treatment, Lag::from_raw(1)).unwrap();
        let y0 = ensure_lagged(&mut graph, outcome, Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(x1, y0).unwrap();
        let query = antecedent_core::TemporalEffectQuery::pulse(treatment, outcome, 1.0)
            .with_policy(antecedent_core::TemporalPolicy::pulse(-1))
            .with_horizon_steps(1);
        let ctx = ExecutionContext::for_tests(38);
        let prepared = crate::Study::series_multi(data.clone())
            .graph(graph)
            .temporal_query(query)
            .refute(crate::RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap();
        let plan = prepared.plan().lowered().unwrap();
        assert_eq!(plan.refutation_quadrant(), TargetQuadrant::TEMPORAL_FREQUENTIST);
        let result = prepared.estimate_multi_env(&data, &ctx).unwrap();
        assert_eq!(result_quadrant(&result), plan.refutation_quadrant());
        let target = result_refutation_target(&result).unwrap();
        assert_eq!(target.source(), result.lowered_program_identity().unwrap().as_bytes());
        assert_eq!(target.quadrant(), TargetQuadrant::TEMPORAL_FREQUENTIST);
    }

    /// A target with one coordinate per horizon.
    fn horizon_target(horizons: usize) -> RefutationTarget {
        RefutationTarget::new(
            TargetFamily::ResponseDerivative,
            TargetQuadrant::TEMPORAL_FREQUENTIST,
            (1..=horizons).map(|h| format!("horizon[{h}]")),
            [7; 32],
        )
        .unwrap()
    }

    /// One coordinate per horizon: a linear response with that horizon's slope.
    fn horizon_checks(
        kind: TargetRefuterKind,
        slopes: &[f64],
        reported: &[f64],
    ) -> Vec<DerivativeCoordinate> {
        slopes
            .iter()
            .zip(reported)
            .map(|(&slope, &value)| {
                if kind == TargetRefuterKind::FiniteDifferenceConsistency {
                    DerivativeCoordinate::FiniteDifference(
                        finite_difference_consistency(
                            |offset| Ok(1.0 + slope * offset),
                            &[DerivativeCheckPoint { x: 0.0, reported: value }],
                            0.5,
                            0.05,
                        )
                        .unwrap(),
                    )
                } else {
                    DerivativeCoordinate::Linearity(
                        local_linearity(|offset| Ok(1.0 + slope * offset), &[0.0], 0.5, 0.25, 1e-9)
                            .unwrap(),
                    )
                }
            })
            .collect()
    }

    #[test]
    fn horizons_with_distinct_slopes_merge_into_one_report_on_the_horizon_target() {
        let target = horizon_target(2);
        let kind = TargetRefuterKind::FiniteDifferenceConsistency;
        let checks = horizon_checks(kind, &[2.0, 3.0], &[2.0, 3.0]);
        let sound = merge_derivative_checks(kind, checks, &target).unwrap();
        assert!(sound.passed, "{:?}", sound.failure);
        assert!((sound.refuted[0] - 2.0).abs() < 1e-9 && (sound.refuted[1] - 3.0).abs() < 1e-9);
        assert_eq!(&*sound.original, &[2.0, 3.0]);
        // One wrong horizon fails the whole vector.
        let checks = horizon_checks(kind, &[2.0, 3.0], &[2.0, 3.6]);
        assert!(!merge_derivative_checks(kind, checks, &target).unwrap().passed);
        // A report with another number of coordinates than the target is refused.
        let checks = horizon_checks(kind, &[2.0], &[2.0]);
        assert!(merge_derivative_checks(kind, checks, &target).is_err());
        // Local linearity merges the same way: a linear response is linear at every horizon.
        let kind = TargetRefuterKind::LocalLinearity;
        let checks = horizon_checks(kind, &[2.0, 3.0], &[2.0, 3.0]);
        let linear = merge_derivative_checks(kind, checks, &target).unwrap();
        assert!(linear.passed);
        assert_eq!(linear.original.len(), 2);
    }

    #[test]
    fn a_class_within_the_limit_keeps_every_member_and_a_larger_one_a_stable_subsample() {
        let member = |i: usize| {
            antecedent_discovery::adjacency_mask::set_edge(
                antecedent_discovery::AdjacencyMask::EMPTY,
                16,
                0,
                1 + i % 15,
                true,
            )
        };
        let mut all = ClassSample::new(8);
        for i in 0..8 {
            all.offer(i, member(i));
        }
        assert_eq!(all.seen, 8);
        assert_eq!(all.into_members(), (0..8).map(member).collect::<Vec<_>>());
        let sample = |limit: usize| {
            let mut kept = ClassSample::new(limit);
            for i in 0..1000 {
                kept.offer(i, member(i));
            }
            kept
        };
        let kept = sample(16);
        assert_eq!((kept.seen, kept.members.len()), (1000, 16));
        // The subsample is the 16 positions of smallest rank, whatever the order they arrive in.
        let mut expected: Vec<usize> = (0..1000).collect();
        expected.sort_by_key(|&i| (ClassSample::rank(i), i));
        expected.truncate(16);
        expected.sort_unstable();
        let positions: Vec<usize> = kept.members.iter().map(|&(_, p, _)| p).collect();
        let mut positions_sorted = positions;
        positions_sorted.sort_unstable();
        assert_eq!(positions_sorted, expected);
    }

    #[test]
    fn a_lagged_edge_keeps_its_lag_source_and_target_when_a_node_is_added() {
        // Two nodes, two lags: bit (lag, from, to) = lag * 4 + from * 2 + to.
        let old = antecedent_discovery::LagMask::from_word((1_u64 << 1) | (1_u64 << (4 + 2)));
        // lag 1: 0 -> 1; lag 2: 1 -> 0
        let packed = repack_lag_mask(&old, 2, 3, 2);
        assert_eq!(packed, (1_u64 << 1) | (1_u64 << (9 + 3)));
        assert!(repack_lag_mask(&antecedent_discovery::LagMask::EMPTY, 2, 3, 2).is_empty());
        // A template wider than a machine word repacks too.
        let wide = antecedent_discovery::LagMask::EMPTY.with_bit(7 * 49 + 3 * 7 + 5, true);
        let moved = repack_lag_mask(&wide, 7, 8, 8);
        assert!(moved.bit(7 * 64 + 3 * 8 + 5) && moved.count_ones() == 1);
    }
}

/// Interventional means one derivative coordinate reads: the evaluation point and one step each way.
const DERIVATIVE_POINTS_PER_COORDINATE: usize = 3;

/// DAG members of one class the common-cause refuter retains when the context declares no memory
/// limit; a larger class is reduced to a deterministic subsample of this size.
const DEFAULT_WITHIN_CLASS_MEMBERS: usize = 512;

/// Bytes budgeted per enumerated DAG member of an `n`-node class: its edge set, the augmented
/// copy with the common cause, and the bookkeeping of both.
const fn within_class_member_bytes(n: usize) -> u64 {
    16 * (n as u64 * n as u64 + 64)
}

/// Most DAG members of one `n`-node class the common-cause refuter retains: as many as the
/// context's memory limit holds, or [`DEFAULT_WITHIN_CLASS_MEMBERS`] when it declares none. A
/// class with more members is subsampled to this many (see [`ClassSample`]).
fn within_class_member_cap(ctx: &ExecutionContext, n: usize) -> usize {
    ctx.memory_limit_bytes().map_or(DEFAULT_WITHIN_CLASS_MEMBERS, |limit| {
        usize::try_from(limit / within_class_member_bytes(n)).unwrap_or(usize::MAX).max(1)
    })
}

/// The members of one equivalence class a bounded retention keeps, with the number enumerated.
///
/// Each member is ranked by a fixed hash of its position in the enumeration, and the `limit`
/// members of smallest rank are kept. The enumeration order is deterministic, so the sample is
/// too, and because the hash does not depend on the member's content or on `limit`, the kept
/// members are a uniform draw without replacement from the class. A class within `limit` keeps
/// every member, in enumeration order.
struct ClassSample {
    limit: usize,
    /// Members kept, as (rank, position in the enumeration, member).
    members: Vec<(u64, usize, antecedent_discovery::AdjacencyMask)>,
    /// Members offered, kept or not.
    seen: usize,
}

impl ClassSample {
    fn new(limit: usize) -> Self {
        Self { limit: limit.max(1), members: Vec::new(), seen: 0 }
    }

    /// The splitmix64 finalizer of the position: a fixed, well mixed rank.
    fn rank(position: usize) -> u64 {
        let mut z = (position as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn offer(&mut self, position: usize, member: antecedent_discovery::AdjacencyMask) {
        self.seen += 1;
        let rank = Self::rank(position);
        if self.members.len() < self.limit {
            self.members.push((rank, position, member));
            return;
        }
        // A position never repeats, so the ranks of two members differ unless the hash
        // collides; the position breaks such a tie.
        let Some((worst, _)) = self
            .members
            .iter()
            .enumerate()
            .max_by_key(|(_, (rank, position, _))| (*rank, *position))
        else {
            return;
        };
        let (worst_rank, worst_position, _) = self.members[worst];
        if (rank, position) < (worst_rank, worst_position) {
            self.members[worst] = (rank, position, member);
        }
    }

    /// The kept members in enumeration order.
    fn into_members(mut self) -> Vec<antecedent_discovery::AdjacencyMask> {
        self.members.sort_unstable_by_key(|&(_, position, _)| position);
        self.members.into_iter().map(|(_, _, member)| member).collect()
    }
}

impl Default for TargetRefitOptions {
    fn default() -> Self {
        Self {
            negative_control_exposure: None,
            negative_control_outcome: None,
            mediator: None,
            mediation_rho_bar: 0.1,
            common_cause_strength: 0.0,
            prior_scale_factors: vec![0.25, 4.0],
            prior_shift_tolerance: 0.5,
            derivative_step: None,
            derivative_tolerance: 0.1,
            linearity_tolerance: 0.25,
            gathered_resamples: false,
            within_class: None,
        }
    }
}

/// Data a prepared target refutation perturbs.
#[derive(Clone, Copy, Debug)]
pub enum PreparedTargetData<'a> {
    /// Independent rows.
    Tabular(&'a TabularData),
    /// One regularly sampled series.
    Series(&'a TimeSeriesData),
    /// A panel of unit series.
    Panel(&'a PanelData),
    /// Independent environments, each one series of a shared schema.
    MultiEnv(&'a antecedent_data::MultiEnvironmentData),
}

/// Plan, reports, refusals and certificate of one prepared target refutation.
/// Readable target-native outputs. Mutable access invalidates checked authority.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetRefutationOutcomeView {
    /// The plan attached to the result's lowered program.
    pub plan: TargetRefutationPlan,
    /// Reports and refusals.
    pub run: TargetRefutationRun,
    /// The reports, bound to the result's target.
    pub certificate: TargetCertificate,
}

/// A checked prepared-operation outcome. Its private owner is not deserializable.
#[derive(Clone, Debug)]
pub struct TargetRefutationOutcome {
    view: TargetRefutationOutcomeView,
    owner: Option<Arc<()>>,
}
impl std::ops::Deref for TargetRefutationOutcome {
    type Target = TargetRefutationOutcomeView;
    fn deref(&self) -> &Self::Target {
        &self.view
    }
}
impl std::ops::DerefMut for TargetRefutationOutcome {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.owner = None;
        &mut self.view
    }
}

/// Run the target-native suite of `result` through `prepared`.
///
/// `data` must be the data `result` was computed on (the retained data of the handle). Every
/// refit re-executes a compiled program in full: the handle's own program on perturbed data,
/// or, where the refuter changes the scientific problem (a simulated common cause declared in
/// the graph, an alternative prior, the fitted response curve of a derivative), a study
/// compiled and prepared from the handle's study with that one change. A prepared handle whose
/// study requests a scalar refutation suite pays for it on every refit of its own program:
/// prepare with no scalar suite for target refutation.
///
/// # Errors
///
/// The result's target has no point value, the data modality does not match the handle,
/// cancellation, an admission refusal, a failed refit, or (impossible by construction) a
/// report on another target.
pub fn refute_prepared_target(
    prepared: &PreparedStudy,
    result: &StudyResult,
    data: PreparedTargetData<'_>,
    suite: TargetSuite,
    config: &TargetRefutationConfig,
    options: &TargetRefitOptions,
    ctx: &ExecutionContext,
) -> Result<TargetRefutationOutcome, CausalError> {
    let checked = checked_refutation_input(result, data, ctx)?;
    let owner = Arc::clone(&checked.owner);
    let target = result_refutation_target(result)?;
    let (_, _, original) = result_target_values(result)?;
    // An attribution has no treatment; its stability refuters never read one, so the
    // outcome stands in for the unused field.
    let treatment = match result.treatment {
        Some(treatment) => treatment,
        None if matches!(
            target.family(),
            TargetFamily::Attribution | TargetFamily::InterventionalLaw
        ) =>
        {
            result.outcome
        }
        None => {
            return Err(CausalError::Unsupported {
                message: "target refutation needs a result with a treatment",
            });
        }
    };
    if let Some(lowered) = prepared.plan().lowered() {
        if lowered.semantic_identity().is_none()
            || lowered.semantic_identity() != result.lowered_program_identity()
        {
            return Err(CausalError::Conflict {
                what: "program",
                detail: "the result was not produced by this prepared handle's lowered program",
            });
        }
    }
    let plan = TargetRefutationPlan::for_target(target.clone(), suite);
    let refit = PreparedRefit {
        prepared,
        result,
        data,
        target: target.clone(),
        member: None,
        temporal_budget: prepared.study().temporal_validation_budget.clone().unwrap_or_else(|| {
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits())
        }),
        original,
        treatment,
        outcome: result.outcome,
        options: options.clone(),
        replicates: config.replicates,
        alpha: config.alpha,
        curve: std::cell::OnceCell::new(),
    };
    let run = run_target_plan(&plan, &refit, config, ctx)?;
    let certificate = certify_target_refutations(&target, &run.reports)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(TargetRefutationOutcome {
        view: TargetRefutationOutcomeView { plan, run, certificate },
        owner: Some(owner),
    })
}

// Run a scoped member's actual level target through full-program refits. The
// selector comes from the point-time witness retained by this same operation.
#[allow(
    clippy::too_many_arguments,
    reason = "the checked refit binds its member, data and execution context"
)]
pub(super) fn refute_temporal_response_member(
    prepared: &PreparedStudy,
    result: &StudyResult,
    data: PreparedTargetData<'_>,
    outer_atom: u64,
    witness: &crate::result::TemporalResponseWitness,
    suite: TargetSuite,
    config: &TargetRefutationConfig,
    checked: &CheckedRefutationInput<'_>,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<(antecedent_validate::RefutationScope, TargetRefutationRun), CausalError> {
    if !matches!(&result.authority, crate::result::RuntimeRefutationAuthority::Sealed { owner, .. } if Arc::ptr_eq(owner, &checked.owner))
    {
        return Err(CausalError::Conflict {
            what: "result",
            detail: "the checked refutation input belongs to another result",
        });
    }
    let source = result.lowered_program_identity().ok_or(CausalError::Unsupported {
        message: "scoped temporal refutation requires the producing lowered program",
    })?;
    let lowered = prepared.plan().lowered().ok_or(CausalError::Unsupported {
        message: "scoped temporal refutation requires a lowered prepared program",
    })?;
    if lowered.semantic_identity() != Some(source) {
        return Err(CausalError::Conflict {
            what: "program",
            detail: "the result and scoped refit have different lowered programs",
        });
    }
    let selector = TemporalResponseMemberSelector {
        outer_atom,
        horizon_index: witness.horizon_index,
        horizon: witness.horizon,
        completion_digest: witness.completion_digest,
    };
    let original = temporal_response_member_values(result, &selector, budget, ctx)?;
    let prefix = format!(
        "scope:v1:temporal_class_completion:{}:{}:{}:{}",
        outer_atom,
        witness.horizon_index,
        witness.horizon,
        antecedent_core::identity::to_hex(&witness.completion_digest)
    );
    let labels = (0..original.len()).map(|i| format!("{prefix}|coordinate[{i}]"));
    let target = RefutationTarget::new(
        TargetFamily::WorldValued,
        result_quadrant(result),
        labels,
        *source.as_bytes(),
    )
    .map_err(CausalError::from)?;
    let treatment = result.treatment.ok_or(CausalError::Unsupported {
        message: "scoped temporal response requires a treatment",
    })?;
    let plan = TargetRefutationPlan::for_target(target.clone(), suite);
    let refit = PreparedRefit {
        prepared,
        result,
        data,
        target: target.clone(),
        member: Some(selector),
        temporal_budget: budget.clone(),
        original,
        treatment,
        outcome: result.outcome,
        options: TargetRefitOptions::default(),
        replicates: config.replicates,
        alpha: config.alpha,
        curve: std::cell::OnceCell::new(),
    };
    let run = run_target_plan(&plan, &refit, config, ctx)?;
    // Checked target certification also rejects any report naming another target.
    certify_target_refutations(&target, &run.reports)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    let scope = antecedent_validate::RefutationScope::TemporalClassCompletion {
        outer_atom,
        horizon_index: witness.horizon_index,
        horizon: witness.horizon,
        completion_digest: witness.completion_digest,
        target,
    };
    Ok((scope, run))
}

impl PreparedStudy {
    /// Refute the supplied result on this preparation's exact retained observations.
    ///
    /// The caller must supply the result produced by this handle. The existing target
    /// certification checks bind the result to the lowered program; no data is re-ingested.
    ///
    /// # Errors
    /// The same target, modality, execution and admission errors as [`refute_prepared_target`].
    pub fn refute_retained_target(
        &self,
        result: &StudyResult,
        suite: TargetSuite,
        config: &TargetRefutationConfig,
        options: &TargetRefitOptions,
        ctx: &ExecutionContext,
    ) -> Result<TargetRefutationOutcome, CausalError> {
        let data = match &self.study().data {
            DataInput::Tabular(data) => PreparedTargetData::Tabular(data),
            DataInput::Temporal(data) | DataInput::Event(data) => PreparedTargetData::Series(data),
            DataInput::Panel(data) => PreparedTargetData::Panel(data),
            DataInput::MultiEnv(data) => PreparedTargetData::MultiEnv(data),
        };
        refute_prepared_target(self, result, data, suite, config, options, ctx)
    }
}

mod input;
mod rows;

use rows::{
    JoinFreeRows, block_weights, join_free_mask, join_free_rows, mask_from, series_with, shifted,
    subsample_bounds, times, truncated_series, window_weights,
};

pub(super) use input::{CheckedRefutationInput, checked_refutation_input};
pub(crate) use input::{ExecutedInputIdentity, executed_input_identity};

/// Data of one refit, owned.
enum Modal {
    Tabular(TabularData),
    Series(TimeSeriesData),
    Panel(PanelData),
    MultiEnv(antecedent_data::MultiEnvironmentData),
}

/// One coordinate's derivative check, before the coordinates are merged into one report.
enum DerivativeCoordinate {
    FiniteDifference(DerivativeConsistency),
    Linearity(LocalLinearity),
}

struct PreparedRefit<'a> {
    prepared: &'a PreparedStudy,
    result: &'a StudyResult,
    data: PreparedTargetData<'a>,
    target: RefutationTarget,
    member: Option<TemporalResponseMemberSelector>,
    temporal_budget: antecedent_core::TemporalBudget,
    original: Vec<f64>,
    treatment: VariableId,
    outcome: VariableId,
    options: TargetRefitOptions,
    replicates: u32,
    alpha: f64,
    /// Fitted response curve of a derivative, `(point, value per horizon)`, computed once.
    curve: std::cell::OnceCell<Vec<(f64, Vec<f64>)>>,
}

/// [`antecedent_kernels::design_gram_cholesky`] with a rank-deficient design reported as the
/// refit refusal `rank_deficient`.
fn gram_cholesky_factor(
    design: &[f64],
    n: usize,
    p: usize,
    rank_deficient: &'static str,
) -> Result<Vec<f64>, TargetRefitError> {
    antecedent_kernels::design_gram_cholesky(design, n, p)
        .ok_or_else(|| unsupported(rank_deficient, "a full-rank adjustment"))
}

fn unsupported(reason: impl Into<Arc<str>>, next_attempt: &str) -> TargetRefitError {
    TargetRefitError::Unsupported { reason: reason.into(), next_attempt: Arc::from(next_attempt) }
}

/// A refuter that cannot speak to this target (a degenerate replicate law, too few draws) is
/// recorded as not run with its reason and next attempt; any other failure stays an error.
fn refuter_outcome(error: ValidationError) -> TargetRefitError {
    match error {
        ValidationError::NotApplicable { message } => unsupported(
            message,
            "raise the replicate count, or check that the perturbation moves the target",
        ),
        other => TargetRefitError::Failed(other),
    }
}

/// Copy of an explicit coefficient prior with every Gaussian coefficient standard deviation
/// multiplied by `factor`; means, residual-variance and restriction specs are kept.
fn scale_prior_deviations(
    prior: &antecedent_prob::PriorSet,
    factor: f64,
) -> Result<antecedent_prob::PriorSet, TargetRefitError> {
    let mut scaled = prior.clone();
    let mut found = false;
    for spec in &mut scaled.specs {
        if let antecedent_prob::PriorSpec::GaussianCoefficients(coefficients) = spec {
            coefficients.variance =
                coefficients.variance.iter().map(|variance| variance * factor * factor).collect();
            found = true;
        }
    }
    if found {
        Ok(scaled)
    } else {
        Err(unsupported(
            "the explicit prior has no Gaussian coefficient block to scale",
            "declare a Gaussian coefficient prior, or leave the isotropic prior scale",
        ))
    }
}

fn failed(error: &CausalError) -> TargetRefitError {
    // A perturbed table that leaves a cell the functional reads empty is a table the
    // refuter cannot refit, not a failed refit: the refuter is refused with its reason.
    let message = error.to_string();
    if message.contains("missing table cell") || message.contains("missing probability table entry")
    {
        return unsupported(
            format!("a perturbed table leaves a required probability cell empty: {message}"),
            "supply more rows per cell, or a table whose resamples keep every cell the \
             functional reads populated",
        );
    }
    TargetRefitError::Failed(match error {
        CausalError::Cancelled { .. } => ValidationError::Cancelled,
        CausalError::Resource { message } => ValidationError::Resource { message: message.clone() },
        _ => ValidationError::estimation_msg(error.to_string()),
    })
}

fn operational_refit_error(error: &TargetRefitError) -> bool {
    matches!(
        error,
        TargetRefitError::Failed(ValidationError::Cancelled | ValidationError::Resource { .. })
    )
}

fn data_failed(error: antecedent_data::DataError) -> TargetRefitError {
    TargetRefitError::Failed(ValidationError::from(error))
}

fn float_column(storage: &antecedent_data::OwnedColumnarStorage, id: VariableId) -> Option<&[f64]> {
    match storage.columns().get(id.as_usize()) {
        Some(OwnedColumn::Float64(column)) => Some(&column.values[..]),
        _ => None,
    }
}

/// Longest lag of a temporal graph (zero for a static one).
fn max_graph_lag(study: &crate::analysis::execute::Study) -> usize {
    study.graph.as_temporal_dag().map_or(0, |graph| {
        graph
            .nodes()
            .iter()
            .filter_map(|node| match node {
                antecedent_core::NodeRef::Lagged { lag, .. } => Some(lag.raw() as usize),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    })
}

/// Dense node of a static variable in a DAG.
fn static_node(dag: &Dag, variable: VariableId) -> Option<DenseNodeId> {
    dag.nodes()
        .iter()
        .position(|node| *node == antecedent_core::NodeRef::Static(variable))
        .and_then(|index| u32::try_from(index).ok().map(DenseNodeId::from_raw))
}

/// Nodes of a variable in a temporal DAG.
fn lagged_nodes(graph: &TemporalDag, variable: VariableId) -> Vec<(DenseNodeId, usize)> {
    graph
        .nodes()
        .iter()
        .enumerate()
        .filter_map(|(index, node)| match node {
            antecedent_core::NodeRef::Lagged { variable: v, lag } if *v == variable => {
                u32::try_from(index)
                    .ok()
                    .map(|raw| (DenseNodeId::from_raw(raw), lag.raw() as usize))
            }
            _ => None,
        })
        .collect()
}

mod common_cause;
mod controls;
mod derivative;
mod panel;
mod predictive;
mod refit;
mod series;
mod tabular;

impl PreparedRefit<'_> {
    fn study(&self) -> &crate::analysis::execute::Study {
        self.prepared.study()
    }

    fn storage(&self) -> &antecedent_data::OwnedColumnarStorage {
        match self.data {
            PreparedTargetData::Tabular(data) => data.storage(),
            PreparedTargetData::Series(data) => data.storage(),
            PreparedTargetData::Panel(data) => data.units()[0].series.storage(),
            PreparedTargetData::MultiEnv(data) => data.environments()[0].storage(),
        }
    }

    fn is_float(&self, id: VariableId) -> bool {
        float_column(self.storage(), id).is_some()
    }

    fn permuted(values: &[f64], rng: &mut CausalRng, circular: bool) -> Arc<[f64]> {
        let n = values.len();
        if circular && n > 3 {
            // A circular shift by between a quarter and three quarters of the series keeps the
            // series' own dependence and breaks its alignment with the other columns.
            let offset = n / 4 + unbiased_index(rng, (n / 2).max(1));
            (0..n).map(|i| values[(i + offset) % n]).collect()
        } else {
            let mut out = values.to_vec();
            shuffle(rng, &mut out);
            out.into()
        }
    }

    fn column_of(
        storage: &antecedent_data::OwnedColumnarStorage,
        id: VariableId,
    ) -> Result<&[f64], TargetRefitError> {
        float_column(storage, id).ok_or_else(|| {
            unsupported(
                "the perturbed column is not a float64 column",
                "encode the treatment, outcome and controls as float64 columns",
            )
        })
    }

    fn modal_data(&self) -> Modal {
        match self.data {
            PreparedTargetData::Tabular(d) => Modal::Tabular(d.clone()),
            PreparedTargetData::Series(d) => Modal::Series(d.clone()),
            PreparedTargetData::Panel(d) => Modal::Panel(d.clone()),
            PreparedTargetData::MultiEnv(d) => Modal::MultiEnv((*d).clone()),
        }
    }

    /// Posterior draws of the whole target (`draws × coordinates`, row-major).
    fn target_draws(&self, result: &StudyResult) -> Option<Vec<f64>> {
        let k = self.target.dimension();
        if let Some(response) = result.response() {
            let draws = match &response.uncertainty {
                ResponseUncertainty::Scalar { draws: Some(draws), .. }
                | ResponseUncertainty::PointwiseBand { draws: Some(draws), .. } => Some(draws),
                _ => None,
            };
            if let Some(draws) = draws.filter(|d| d.n_coordinates() == k && d.n_draws > 0) {
                let mut rows = Vec::with_capacity(draws.n_draws * k);
                for s in 0..draws.n_draws {
                    for j in 0..k {
                        rows.push(draws.column(j)?[s]);
                    }
                }
                return Some(rows);
            }
        }
        if k == 1 {
            let posterior = result.posterior.as_ref()?;
            // Posterior columns may put nuisance coefficients before the effect.
            // Read the declared effect, never the first column by position.
            let effect = posterior.effect_column()?;
            return posterior.draws.column(effect).ok().map(<[f64]>::to_vec);
        }
        None
    }

    fn mediation_check(&self) -> Result<TargetRefutationReport, TargetRefitError> {
        let Some(mediator) = self.options.mediator else {
            return Err(unsupported(
                "no mediator is declared for the sequential-ignorability range",
                "declare the single mediator in TargetRefitOptions",
            ));
        };
        let PreparedTargetData::Tabular(data) = self.data else {
            return Err(unsupported(
                "the closed-form sequential-ignorability range is for independent rows",
                "on series, the temporal mediation suite's placebo-mediator and contiguous-window \
                 checks",
            ));
        };
        let storage = data.storage();
        let mut ids = vec![self.treatment, mediator, self.outcome];
        ids.extend(self.adjustment());
        let columns: Vec<&[f64]> =
            ids.iter().map(|&id| Self::column_of(storage, id)).collect::<Result<_, _>>()?;
        let n = data.row_count();
        let mask = storage.analysis_mask();
        let complete: Vec<usize> = (0..n)
            .filter(|&i| {
                mask.is_none_or(|m| m.is_valid(i)) && columns.iter().all(|c| c[i].is_finite())
            })
            .collect();
        let take = |c: &[f64]| complete.iter().map(|&i| c[i]).collect::<Vec<f64>>();
        let covariates: Vec<Vec<f64>> = columns[3..].iter().map(|c| take(c)).collect();
        let summary = mediation_sensitivity_fit(
            &take(columns[0]),
            &take(columns[1]),
            &take(columns[2]),
            &covariates,
        )?;
        let break_even = summary.rho_tilde.abs();
        let passed = break_even >= self.options.mediation_rho_bar;
        let mut refuted = self.original.clone();
        if let Some(last) = refuted.last_mut() {
            *last = summary.acme(0.0)?;
        }
        Ok(TargetRefutationReport {
            target: self.target.clone(),
            kind: TargetRefuterKind::SequentialIgnorability,
            original: self.original.clone().into(),
            refuted: refuted.into(),
            statistic: JointStatistic {
                kind: JointStatisticKind::BreakEven,
                value: break_even,
                df: 1,
                p_value: None,
            },
            passed,
            informative: true,
            standardised_bias: Arc::from([]),
            standardised_bias_norm: None,
            replicates: 0,
            failure: (!passed).then(|| {
                Arc::from(format!(
                    "a mediator-outcome error correlation of {:.3} explains the indirect effect \
                     away (bar {})",
                    summary.rho_tilde, self.options.mediation_rho_bar
                ))
            }),
        })
    }
}

/// Lease `count × width` cells for a native check's buffers.
fn admit_target_cells(
    ctx: &ExecutionContext,
    count: usize,
    width: usize,
) -> Result<antecedent_core::execution::AdmissionLease, TargetRefitError> {
    let bytes = count
        .checked_mul(width)
        .and_then(|cells| cells.checked_mul(std::mem::size_of::<f64>()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| {
            TargetRefitError::Failed(ValidationError::estimation_msg("check buffer size overflow"))
        })?;
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => {
            TargetRefitError::Failed(ValidationError::Cancelled)
        }
        other => TargetRefitError::Failed(ValidationError::estimation_msg(format!(
            "admission pool cannot fund the check buffers: {other}"
        ))),
    })
}

impl TargetRefit for PreparedRefit<'_> {
    fn target(&self) -> &RefutationTarget {
        &self.target
    }

    fn original(&self) -> &[f64] {
        &self.original
    }

    fn refit(
        &self,
        perturbation: TargetPerturbation,
        rng: &mut CausalRng,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        if self.result.temporal_posterior_witnesses.is_some()
            && matches!(perturbation, TargetPerturbation::RandomCommonCause { .. })
        {
            return Err(unsupported(
                "the added common cause changes completion identity",
                "a scoped projection that maps the augmented graph to the original completion",
            ));
        }
        match self.data {
            PreparedTargetData::Tabular(data) => self.tabular_refit(data, perturbation, rng, ctx),
            PreparedTargetData::Series(data) => self.series_refit(data, perturbation, rng, ctx),
            PreparedTargetData::Panel(data) => self.panel_refit(data, perturbation, rng, ctx),
            PreparedTargetData::MultiEnv(data) => {
                self.multi_env_refit(data, perturbation, rng, ctx)
            }
        }
    }

    fn native_check(
        &self,
        kind: TargetRefuterKind,
        ctx: &ExecutionContext,
    ) -> Result<TargetRefutationReport, TargetRefitError> {
        match kind {
            TargetRefuterKind::SequentialIgnorability => self.mediation_check(),
            TargetRefuterKind::FiniteDifferenceConsistency | TargetRefuterKind::LocalLinearity => {
                self.derivative_check(kind, ctx)
            }
            TargetRefuterKind::PriorSensitivity => self.prior_sensitivity(ctx),
            TargetRefuterKind::PosteriorPredictive => self.posterior_predictive(ctx),
            _ => Err(unsupported("no native check for this refuter", "a replicate refuter")),
        }
    }
}

/// Sample standard deviation of the finite values (zero for fewer than two).
fn sample_sd(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.len() < 2 {
        return 0.0;
    }
    let n = finite.len() as f64;
    let mean = finite.iter().sum::<f64>() / n;
    (finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}

#[cfg(test)]
mod panel_shift_tests {
    use super::*;

    #[test]
    fn shifted_panel_windows_keep_alignment_and_refuse_empty_units() {
        let make_unit = |id, values: &[f64], other: &[f64]| PanelUnit {
            unit_id: id,
            series: TimeSeriesData::from_f64_columns([("t", values), ("y", other)], 1).unwrap(),
        };
        let units = vec![
            make_unit(7, &[1., 2., 3., 4.], &[10., 20., 30., 40.]),
            make_unit(9, &[5., 6., 7.], &[50., 60., 70.]),
        ];
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        let forward = PreparedRefit::shifted_panel_units(&units, treatment, 1).unwrap();
        assert_eq!(forward.iter().map(|u| u.unit_id).collect::<Vec<_>>(), [7, 9]);
        assert_eq!(float_column(forward[0].series.storage(), treatment).unwrap(), [2., 3., 4.]);
        assert_eq!(float_column(forward[0].series.storage(), outcome).unwrap(), [10., 20., 30.]);
        assert_eq!(float_column(forward[1].series.storage(), treatment).unwrap(), [6., 7.]);
        let backward = PreparedRefit::shifted_panel_units(&units, outcome, -1).unwrap();
        assert_eq!(float_column(backward[0].series.storage(), treatment).unwrap(), [2., 3., 4.]);
        assert_eq!(float_column(backward[0].series.storage(), outcome).unwrap(), [10., 20., 30.]);
        assert_eq!(float_column(backward[1].series.storage(), outcome).unwrap(), [50., 60.]);
        assert!(PreparedRefit::shifted_panel_units(&units, outcome, -3).is_err());
        assert!(PreparedRefit::shifted_panel_units(&units, treatment, 3).is_err());
    }
}

#[cfg(test)]
mod temporal_contributor_order_tests {
    use super::*;
    use crate::{RefuteSuite, Study};
    use antecedent_core::{
        CausalQuery, Intervention, ResponseFunctional, ResponseQuery, TemporalPolicy,
        TemporalResponseSpec, Value, VariableId,
    };
    use antecedent_data::TimeSeriesData;
    use antecedent_discovery::adjacency_mask::set_edge;
    use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
    use antecedent_prob::InferenceDiagnostics;
    use std::sync::Arc;

    fn result(ctx: &ExecutionContext) -> StudyResult {
        let n = 300;
        let (mut t, mut y, mut z) = (vec![0.; n], vec![0.; n], vec![0.; n]);
        for i in 0..n {
            let s = i as f64;
            let r = (s * 0.29).sin();
            z[i] = 0.4 * r + (s * 0.13).cos() + 0.3 * (s * 1.7).sin();
            t[i] = 0.3 + 0.5 * r + 0.2 * z[i] + 0.4 * (s * 2.3).cos();
            if i > 0 {
                y[i] = 1. + 2. * t[i - 1] + 0.6 * z[i - 1] + 0.1 * (s * 0.77).sin();
            }
        }
        let data = TimeSeriesData::from_f64_columns(
            [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
            1,
        )
        .unwrap();
        let adj = set_edge(0, 3, 2, 0, true).to_word().unwrap();
        let posterior = GraphPosterior::from_atoms(
            3,
            vec![0.5, 0.5],
            vec![adj, 0],
            vec![0.; 9],
            vec![0.; 9],
            SamplingSummary {
                ess: 1.,
                diagnostics: InferenceDiagnostics::analytic("contributor_order"),
                rejected_invalid: 0,
            },
        )
        .unwrap()
        .with_atom_kind(GraphPosteriorAtomKind::Cpdag)
        .with_lagged_marginals(1, vec![0.; 9])
        .unwrap()
        .with_lag_masks(vec![(1 << 1) | (1 << 7), 1 << 1])
        .unwrap();
        let query = CausalQuery::Response(
            ResponseQuery::new(ResponseFunctional::InterventionResponse {
                outcome: VariableId::from_raw(1),
                interventions: Arc::from([Intervention::set(
                    VariableId::from_raw(0),
                    Value::f64(1.),
                )]),
            })
            .with_temporal(
                TemporalResponseSpec::new([1], TemporalPolicy::pulse(-1), Some(1)).unwrap(),
            ),
        );
        Study::series(data)
            .graph_posterior(posterior)
            .query(query)
            .refute(RefuteSuite::Cheap)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(ctx)
            .unwrap()
    }

    #[allow(
        clippy::used_underscore_binding,
        reason = "the copied member must retain its graph admission leases"
    )]
    fn copied_member(
        m: &crate::result::TemporalPosteriorWitnessMember,
    ) -> crate::result::TemporalPosteriorWitnessMember {
        crate::result::TemporalPosteriorWitnessMember {
            outer_atom: m.outer_atom,
            outer_weight: m.outer_weight,
            witnesses: Arc::clone(&m.witnesses),
            structural: m.structural.clone(),
            _resources: m._resources.clone(),
        }
    }

    #[test]
    fn contributor_matching_is_keyed_not_enumeration_order() {
        let ctx = ExecutionContext::for_tests(1);
        let original = result(&ctx);
        let source = original.temporal_posterior_witnesses.as_ref().unwrap();
        assert!(source.members.len() >= 2);
        let mut reordered = original.clone();
        let mut members: Vec<_> = source.members.iter().map(copied_member).collect();
        members.reverse();
        for member in &mut members {
            let mut entries = member.witnesses.entries.clone();
            entries.reverse();
            member.witnesses = Arc::new(crate::result::TemporalResponseWitnessSet {
                entries,
                _leases: Vec::new(),
            });
        }
        reordered.temporal_posterior_witnesses =
            Some(Arc::new(crate::result::TemporalPosteriorWitnessSet {
                members,
                _lease: ctx.acquire_admission(0, 0).unwrap(),
            }));
        let compare = |other: &StudyResult| {
            let mut budget =
                antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
            check_temporal_response_contributors(&original, other, &mut budget, true, &ctx)
        };
        assert!(compare(&reordered).is_ok());
        let mut missing = original.clone();
        missing.temporal_posterior_witnesses =
            Some(Arc::new(crate::result::TemporalPosteriorWitnessSet {
                members: source.members.iter().skip(1).map(copied_member).collect(),
                _lease: ctx.acquire_admission(0, 0).unwrap(),
            }));
        assert!(compare(&missing).is_err());
        let mut changed = original.clone();
        let mut members: Vec<_> = source.members.iter().map(copied_member).collect();
        let mut entries = members[0].witnesses.entries.clone();
        assert!(!entries.is_empty());
        entries[0].completion_digest[0] ^= 1;
        members[0].witnesses =
            Arc::new(crate::result::TemporalResponseWitnessSet { entries, _leases: Vec::new() });
        changed.temporal_posterior_witnesses =
            Some(Arc::new(crate::result::TemporalPosteriorWitnessSet {
                members,
                _lease: ctx.acquire_admission(0, 0).unwrap(),
            }));
        assert!(compare(&changed).is_err());
    }
}

#[cfg(test)]
mod static_contributor_tests {
    use super::*;
    use crate::result::{
        StructuralAggregationPolicy, StructuralResponseAtom, StructuralResponseMixture,
        StructuralWeightBasis,
    };
    use antecedent_core::IdentificationStatus;

    fn mixture() -> StructuralResponseMixture {
        let atom = |graph_key, weight, value| StructuralResponseAtom {
            graph_key,
            weight,
            status: IdentificationStatus::NonparametricallyIdentified,
            value: Some(value),
            posterior: None,
            response: None,
        };
        StructuralResponseMixture {
            weight_basis: StructuralWeightBasis::PosteriorProbability,
            atoms: vec![
                atom(7, 0.4, ResponseValue::Scalar(1.0)),
                atom(9, 0.6, ResponseValue::Vector(Arc::from([2.0, 3.0]))),
            ],
            identified_mass: 1.0,
            unidentified_mass: 0.0,
            unevaluable_mass: 0.0,
            subsampled_out_mass: 0.0,
            identified_set: None,
            identified_set_interval: None,
            conditional_on_identified: Some(ResponseValue::Scalar(2.0)),
            full_mass_scope: true,
            truncated_atoms: 0,
            aggregation_policy: Some(StructuralAggregationPolicy::SameEstimandWeightedMean),
        }
    }

    #[test]
    fn wide_admg_sources_require_the_exact_immutable_posterior_allocation() {
        use antecedent_discovery::{
            AdjacencyMask, GraphPosterior, GraphPosteriorAtomKind, SamplingSummary,
        };
        let ctx = ExecutionContext::for_tests(1);
        let posterior = GraphPosterior::from_atoms(
            9,
            vec![0.4, 0.6],
            vec![AdjacencyMask::from_words(&[7, 1]), AdjacencyMask::from_words(&[9, 1])],
            vec![0.0; 81],
            vec![0.0; 81],
            SamplingSummary {
                ess: 1.0,
                diagnostics: antecedent_prob::InferenceDiagnostics::analytic("wide_admg_source"),
                rejected_invalid: 0,
            },
        )
        .unwrap()
        .with_atom_kind(GraphPosteriorAtomKind::Admg);
        let witness = |source: &Arc<[AdjacencyMask]>| crate::result::StaticPosteriorSourceWitness {
            source: Arc::downgrade(source),
            _lease: ctx.acquire_admission(0, 1024).unwrap(),
        };
        let first = witness(posterior.atoms());
        let second = witness(posterior.atoms());
        assert!(same_admg_posterior_source(&posterior, Some(&first), Some(&second)));
        let foreign: Arc<[AdjacencyMask]> = Arc::from(posterior.atoms().to_vec());
        let foreign_witness = witness(&foreign);
        assert!(!same_admg_posterior_source(&posterior, Some(&first), Some(&foreign_witness)));
        assert!(!same_admg_posterior_source(&posterior, Some(&first), None));
        let mut original = mixture();
        for (atom, key) in original.atoms.iter_mut().zip(posterior.graph_keys().iter()) {
            atom.graph_key = *key;
        }
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        assert!(
            exact_admg_contributors(&posterior, &original, &original, true, &mut budget, &ctx)
                .unwrap()
        );
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        assert!(
            !exact_admg_contributors(&posterior, &original, &original, false, &mut budget, &ctx)
                .unwrap()
        );
    }

    #[test]
    fn unbound_small_admg_refuses_oversized_mask_storage() {
        use antecedent_discovery::{
            AdjacencyMask, GraphPosterior, GraphPosteriorAtomKind, SamplingSummary,
        };
        let ctx = ExecutionContext::for_tests(1);
        let posterior = GraphPosterior::from_atoms(
            3,
            vec![0.4, 0.6],
            vec![AdjacencyMask::from_words(&[0, 0, 0, 0, 1]), AdjacencyMask::from_word(9)],
            vec![0.0; 9],
            vec![0.0; 9],
            SamplingSummary {
                ess: 1.0,
                diagnostics: antecedent_prob::InferenceDiagnostics::analytic("oversized_admg_mask"),
                rejected_invalid: 0,
            },
        )
        .unwrap()
        .with_atom_kind(GraphPosteriorAtomKind::Admg);
        let mut original = mixture();
        for (atom, key) in original.atoms.iter_mut().zip(posterior.graph_keys().iter()) {
            atom.graph_key = *key;
        }
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        assert!(
            !exact_admg_contributors(&posterior, &original, &original, false, &mut budget, &ctx)
                .unwrap()
        );
    }

    #[test]
    fn admg_contributors_bind_every_ordered_word_weight_and_disposition() {
        use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
        let ctx = ExecutionContext::for_tests(1);
        let posterior = GraphPosterior::new(
            3,
            vec![0.4, 0.6],
            vec![7, 9],
            vec![0.0; 9],
            vec![0.0; 9],
            SamplingSummary {
                ess: 1.0,
                diagnostics: antecedent_prob::InferenceDiagnostics::analytic(
                    "exact_admg_contributors",
                ),
                rejected_invalid: 0,
            },
        )
        .unwrap()
        .with_atom_kind(GraphPosteriorAtomKind::Admg);
        let mut original = mixture();
        original.atoms[1].status = IdentificationStatus::NotIdentified;
        let check = |candidate: &StructuralResponseMixture| {
            let mut budget =
                antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
            exact_admg_contributors(&posterior, &original, candidate, false, &mut budget, &ctx)
                .unwrap()
        };
        assert!(check(&original));
        let mut changed = original.clone();
        changed.atoms[1].graph_key = 10;
        assert!(!check(&changed));
        changed = original.clone();
        changed.atoms[1].weight = 0.5;
        assert!(!check(&changed));
        changed = original.clone();
        changed.atoms[1].status = IdentificationStatus::NonparametricallyIdentified;
        assert!(!check(&changed));
        changed = original.clone();
        changed.atoms.reverse();
        assert!(!check(&changed));
    }

    #[test]
    fn source_bound_ordered_samples_allow_repeats_but_preserve_each_sample() {
        let ctx = ExecutionContext::for_tests(1);
        let mut original = mixture();
        original.atoms[1].graph_key = original.atoms[0].graph_key;
        let check = |candidate: &StructuralResponseMixture, ordered| {
            let mut budget =
                antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
            check_static_response_contributors(&original, candidate, ordered, &mut budget, &ctx)
        };
        assert!(check(&original, true).is_ok());
        assert!(check(&original, false).is_err());
        let mut changed = original.clone();
        changed.atoms.swap(0, 1);
        assert!(check(&changed, true).is_err());
        changed = original.clone();
        changed.atoms[1].status = IdentificationStatus::NotIdentified;
        assert!(check(&changed, true).is_err());
        changed = original.clone();
        changed.atoms[1].value = None;
        assert!(check(&changed, true).is_err());
    }

    #[test]
    fn static_contributors_require_unique_stable_identities_mass_and_coordinates() {
        let ctx = ExecutionContext::for_tests(1);
        let original = mixture();
        let check = |candidate: &StructuralResponseMixture| {
            let mut budget =
                antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
            check_static_response_contributors(&original, candidate, false, &mut budget, &ctx)
        };
        let mut reordered = original.clone();
        reordered.atoms.reverse();
        reordered.atoms[0].value = Some(ResponseValue::Vector(Arc::from([4.0, 5.0])));
        assert!(check(&reordered).is_ok());
        let mut missing = original.clone();
        missing.atoms.pop();
        assert!(check(&missing).is_err());
        let mut changed_key = original.clone();
        changed_key.atoms[0].graph_key = 8;
        assert!(check(&changed_key).is_err());
        let mut ambiguous = original.clone();
        ambiguous.atoms[1].graph_key = ambiguous.atoms[0].graph_key;
        assert!(check(&ambiguous).is_err());
        let mut changed_weight = original.clone();
        changed_weight.atoms[0].weight = 0.5;
        assert!(check(&changed_weight).is_err());
        let mut changed_mass = original.clone();
        changed_mass.unidentified_mass = 0.1;
        assert!(check(&changed_mass).is_err());
        let mut changed_coordinates = original.clone();
        changed_coordinates.atoms[1].value = Some(ResponseValue::Vector(Arc::from([4.0])));
        assert!(check(&changed_coordinates).is_err());
        let mut changed_status = original.clone();
        changed_status.atoms[0].status = IdentificationStatus::NotIdentified;
        assert!(check(&changed_status).is_err());
    }
    #[test]
    fn class_witness_refuses_changed_source_completion_and_mass() {
        let ctx = ExecutionContext::for_tests(1);
        let cache: Arc<dyn std::fmt::Debug + Send + Sync + std::panic::RefUnwindSafe> =
            Arc::new(());
        let atoms: Arc<[antecedent_discovery::AdjacencyMask]> = Arc::from([1_u64.into()]);
        let marks: Arc<[antecedent_discovery::AdjacencyMask]> = Arc::from([0_u64.into()]);
        let make = || crate::result::StaticClassWitnessSet {
            atom_kind: antecedent_discovery::GraphPosteriorAtomKind::Pag,
            n_vars: 3,
            identification_cache: Arc::downgrade(&cache),
            source_atoms: Arc::downgrade(&atoms),
            source_marks: Some(Arc::downgrade(&marks)),
            samples: vec![(1, 1.0_f64.to_bits())],
            atoms: vec![crate::result::StaticClassAtomWitness {
                graph_key: 1,
                outer_weight: 1.0_f64.to_bits(),
                completions: crate::result::StaticFittedCompletions {
                    entries: vec![(
                        [1; 32],
                        1.0_f64.to_bits(),
                        IdentificationStatus::NonparametricallyIdentified,
                    )],
                    _lease: ctx.acquire_admission(0, 256).unwrap(),
                },
            }],
            _lease: ctx.acquire_admission(0, 256).unwrap(),
        };
        let original = make();
        let mut candidate = make();
        let compare = |candidate: &crate::result::StaticClassWitnessSet| {
            let mut budget =
                antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
            compare_static_class_witnesses(&original, candidate, &mut budget, &ctx).unwrap()
        };
        assert!(compare(&candidate));
        candidate.atoms[0].completions.entries[0].0 = [2; 32];
        assert!(!compare(&candidate));
        candidate.atoms[0].completions.entries[0].0 = [1; 32];
        candidate.atoms[0].outer_weight = 0.5_f64.to_bits();
        assert!(!compare(&candidate));
        candidate.atoms[0].outer_weight = 1.0_f64.to_bits();
        let foreign: Arc<[antecedent_discovery::AdjacencyMask]> = Arc::from([1_u64.into()]);
        candidate.source_atoms = Arc::downgrade(&foreign);
        assert!(!compare(&candidate));
    }
}
