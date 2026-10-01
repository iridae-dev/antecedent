//! 2.2B X3: joint mechanism deviations of the registered surrogate z formula.
//!
//! Every range here is checked against an independent computation from the
//! generating model (brute-force product-polytope vertex enumeration, random
//! interior points, an enumerated latent SCM), never against the evaluator's
//! own intermediate values.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::needless_pass_by_value,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    reason = "test fixtures: counts are rounded non-negative cell masses, and bit-exact float equality is the property under test"
)]

use std::sync::Arc;

use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceRegime, ExecutionContext, InterventionAssignment, MemoryBudget, ProgressSink,
    RegimeBinding, RegimeId, RegimeKind, SamplingDesign, SearchStop, Value, VariableCoordinate,
    VariableDomain, VariableId,
};
use antecedent_expr::{DiscreteAxis, ExactDiscreteLaw, ExactTransportData, LawTolerance};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    BoundZTransportFunctional, SidLimits, ZTransportQuery, ZTransportResult,
    bind_z_transport_catalog, identify_z_transport,
};
use antecedent_validate::{
    JointDeviationSpec, JointFactor, JointFactorBound, JointMechanismSensitivityResult,
    JointSensitivityError, JointSensitivityLimits, TippingStatus,
    z_transport_joint_mechanism_sensitivity, z_transport_mechanism_sensitivity,
};

const W: VariableId = VariableId::from_raw(0);
const Z: VariableId = VariableId::from_raw(1);
const X: VariableId = VariableId::from_raw(2);
const Y: VariableId = VariableId::from_raw(3);

/// A source model of the cited joint `P(W) P(X | W) P(Y | W, X)` under `do(Z = 0)`.
#[derive(Clone, Debug)]
struct Model {
    pw: Vec<f64>,
    px1: Vec<f64>,
    /// `py[w][x][y]`.
    py: Vec<[Vec<f64>; 2]>,
    y_values: Vec<f64>,
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    fn simplex(&mut self, k: usize) -> Vec<f64> {
        let raw = (0..k).map(|_| 0.05 + self.next()).collect::<Vec<_>>();
        let total = raw.iter().sum::<f64>();
        raw.iter().map(|v| v / total).collect()
    }

    fn model(&mut self, levels: usize, categories: usize) -> Model {
        Model {
            pw: self.simplex(levels),
            px1: (0..levels).map(|_| 0.1 + 0.8 * self.next()).collect(),
            py: (0..levels).map(|_| [self.simplex(categories), self.simplex(categories)]).collect(),
            y_values: (0..categories).map(|_| -2.0 + 4.0 * self.next()).collect(),
        }
    }
}

impl Model {
    fn probabilities(&self) -> Vec<f64> {
        let mut out = Vec::new();
        for (w, pw) in self.pw.iter().enumerate() {
            for x in 0..2 {
                let px = if x == 1 { self.px1[w] } else { 1.0 - self.px1[w] };
                for py in &self.py[w][x] {
                    out.push(pw * px * py);
                }
            }
        }
        out
    }

    fn mean(&self, w: usize, x: usize, kernel: &[f64]) -> f64 {
        let _ = (w, x);
        kernel.iter().zip(&self.y_values).map(|(p, y)| p * y).sum()
    }

    fn delta(&self, w: usize) -> f64 {
        self.mean(w, 1, &self.py[w][1]) - self.mean(w, 0, &self.py[w][0])
    }

    /// Target response under explicit contaminated factors.
    fn response(&self, qw: &[f64], qy: &[[Vec<f64>; 2]]) -> f64 {
        (0..self.pw.len())
            .map(|w| qw[w] * (self.mean(w, 1, &qy[w][1]) - self.mean(w, 0, &qy[w][0])))
            .sum()
    }

    fn mix(p: &[f64], r: &[f64], eps: f64) -> Vec<f64> {
        p.iter().zip(r).map(|(p, r)| (1.0 - eps) * p + eps * r).collect()
    }

    /// Brute-force extrema over every product-polytope vertex.
    fn vertex_extrema(&self, kernel: f64, parent: f64) -> (f64, f64) {
        let levels = self.pw.len();
        let k = self.y_values.len();
        let vertex =
            |i: usize, n: usize| (0..n).map(|j| f64::from(u8::from(i == j))).collect::<Vec<_>>();
        let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
        for e_y in [0.0, kernel] {
            for e_w in [0.0, parent] {
                for rw in 0..levels {
                    let qw = Self::mix(&self.pw, &vertex(rw, levels), e_w);
                    let combos = k.pow(u32::try_from(2 * levels).unwrap());
                    for mut code in 0..combos {
                        let qy = (0..levels)
                            .map(|w| {
                                let a = code % k;
                                code /= k;
                                let b = code % k;
                                code /= k;
                                [
                                    Self::mix(&self.py[w][0], &vertex(a, k), e_y),
                                    Self::mix(&self.py[w][1], &vertex(b, k), e_y),
                                ]
                            })
                            .collect::<Vec<_>>();
                        let value = self.response(&qw, &qy);
                        low = low.min(value);
                        high = high.max(value);
                    }
                }
            }
        }
        (low, high)
    }

    /// Closed-form `(L, U)` computed from the model, independent of the crate.
    fn closed_form(&self, kernel: f64, parent: f64) -> (f64, f64) {
        let high_y = self.y_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let low_y = self.y_values.iter().copied().fold(f64::INFINITY, f64::min);
        let spread = high_y - low_y;
        let up = (0..self.pw.len())
            .map(|w| (1.0 - kernel) * self.delta(w) + kernel * spread)
            .collect::<Vec<_>>();
        let down = (0..self.pw.len())
            .map(|w| (1.0 - kernel) * self.delta(w) - kernel * spread)
            .collect::<Vec<_>>();
        let avg = |d: &[f64]| self.pw.iter().zip(d).map(|(p, d)| p * d).sum::<f64>();
        let max = up.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min = down.iter().copied().fold(f64::INFINITY, f64::min);
        ((1.0 - parent) * avg(&down) + parent * min, (1.0 - parent) * avg(&up) + parent * max)
    }
}

fn diagram() -> SelectionDiagram {
    let mut graph = Admg::with_variables(4);
    for (from, to) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    for (a, b) in [(0, 3), (1, 3), (1, 2)] {
        graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap()
}

fn query() -> ZTransportQuery {
    ZTransportQuery {
        outcomes: Arc::from([Y]),
        treatments: Arc::from([X]),
        controllable: Arc::from([Z]),
        experiment_assignment: Arc::from([InterventionAssignment {
            variable: Z,
            value: Value::Bool(false),
        }]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    }
}

fn catalog(snapshot: &str) -> EvidenceCatalog {
    let regime = EvidenceRegime::try_new(
        RegimeId::from_raw(0),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [Z],
        [InterventionAssignment { variable: Z, value: Value::Bool(false) }],
        [W, X, Y],
        "source",
        DistributionAvailability::Joint,
    )
    .unwrap();
    EvidenceCatalog::try_new(
        [Environment::try_new(
            "source",
            [W, Z, X, Y]
                .into_iter()
                .map(|variable| VariableCoordinate {
                    variable,
                    domain: VariableDomain::Unspecified,
                    unit: None,
                })
                .collect::<Vec<_>>(),
            [],
        )
        .unwrap()],
        [regime],
        [RegimeBinding {
            dataset_identity: None,
            regime: RegimeId::from_raw(0),
            snapshot_identity: Arc::from(snapshot),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        }],
        None,
    )
    .unwrap()
}

/// The source catalog plus a target observational joint over `Z, X, Y`.
fn with_target(source: EvidenceCatalog) -> EvidenceCatalog {
    let coordinates = |population: &str| {
        Environment::try_new(
            population,
            [W, Z, X, Y]
                .into_iter()
                .map(|variable| VariableCoordinate {
                    variable,
                    domain: VariableDomain::Unspecified,
                    unit: None,
                })
                .collect::<Vec<_>>(),
            [],
        )
        .unwrap()
    };
    let target = EvidenceRegime::try_new(
        RegimeId::from_raw(1),
        RegimeKind::Observational,
        EvidenceKind::Available,
        [],
        [],
        [W, Z, X, Y],
        "target",
        DistributionAvailability::Joint,
    )
    .unwrap();
    let mut regimes = source.regimes.to_vec();
    regimes.push(target);
    let mut bindings = source.bindings.to_vec();
    bindings.push(RegimeBinding {
        dataset_identity: None,
        regime: RegimeId::from_raw(1),
        snapshot_identity: Arc::from("target-0"),
        schema_names: Arc::from([]),
        sampling: SamplingDesign::Independent,
        weights: None,
        dependence: DependenceGroup::IndependentStudies,
    });
    EvidenceCatalog::try_new(
        [coordinates("source"), coordinates("target")],
        regimes,
        bindings,
        None,
    )
    .unwrap()
}

fn functional() -> BoundZTransportFunctional {
    let ZTransportResult::Identified(proof) = identify_z_transport(
        &diagram(),
        &query(),
        SidLimits::default(),
        &ExecutionContext::for_tests(0),
    )
    .unwrap() else {
        panic!("the registered surrogate formula identifies");
    };
    bind_z_transport_catalog(&diagram(), &query(), &proof, &catalog("snapshot-0")).unwrap()
}

fn data(model: &Model, snapshot: &str) -> ExactTransportData {
    let levels = |n: usize| -> Arc<[Value]> {
        (0..n).map(|i| Value::Int64(i64::try_from(i).unwrap())).collect::<Vec<_>>().into()
    };
    let law = ExactDiscreteLaw::try_new(
        "source",
        RegimeId::from_raw(0),
        [antecedent_expr::InterventionAssignment::concrete(Z, Value::Bool(false))],
        [
            DiscreteAxis { variable: W, values: levels(model.pw.len()) },
            DiscreteAxis {
                variable: X,
                values: Arc::from([Value::Bool(false), Value::Bool(true)]),
            },
            DiscreteAxis {
                variable: Y,
                values: model
                    .y_values
                    .iter()
                    .map(|y| Value::Float64(*y))
                    .collect::<Vec<_>>()
                    .into(),
            },
        ],
        model.probabilities(),
        snapshot,
        LawTolerance::default(),
    )
    .unwrap();
    ExactTransportData::try_new([law], 1 << 20).unwrap()
}

fn spec(kernel: Option<f64>, parent: Option<f64>) -> JointDeviationSpec {
    let mut factors = Vec::new();
    if let Some(max_fraction) = kernel {
        factors.push(JointFactorBound { factor: JointFactor::OutcomeKernel, max_fraction });
    }
    if let Some(max_fraction) = parent {
        factors.push(JointFactorBound { factor: JointFactor::SharedParentMarginal, max_fraction });
    }
    JointDeviationSpec::new(factors)
}

fn run(
    model: &Model,
    spec: &JointDeviationSpec,
) -> Result<JointMechanismSensitivityResult, JointSensitivityError> {
    run_ctx(model, spec, &ExecutionContext::for_tests(1))
}

fn run_ctx(
    model: &Model,
    spec: &JointDeviationSpec,
    ctx: &ExecutionContext,
) -> Result<JointMechanismSensitivityResult, JointSensitivityError> {
    z_transport_joint_mechanism_sensitivity(
        &diagram(),
        &functional(),
        &data(model, "snapshot-0"),
        spec,
        ctx,
    )
}

fn fixed_model() -> Model {
    Model {
        pw: vec![0.5, 0.3, 0.2],
        px1: vec![0.4, 0.6, 0.5],
        py: vec![
            [vec![0.6, 0.3, 0.1], vec![0.2, 0.3, 0.5]],
            [vec![0.5, 0.25, 0.25], vec![0.4, 0.2, 0.4]],
            [vec![0.3, 0.3, 0.4], vec![0.1, 0.2, 0.7]],
        ],
        y_values: vec![0.0, 1.0, 3.0],
    }
}

fn assert_close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol, "{what}: {a} vs {b}");
}

#[test]
fn exact_range_equals_brute_force_vertex_enumeration() {
    let mut rng = Rng(11);
    let mut checked = 0;
    for case in 0..40 {
        let levels = 1 + case % 3;
        let categories = 2 + case % 2;
        let model = rng.model(levels, categories);
        let (kernel, parent) = (rng.next(), rng.next());
        let result = run(&model, &spec(Some(kernel), Some(parent))).unwrap();
        let (low, high) = model.vertex_extrema(kernel, parent);
        assert_close(result.range.minimum, low, 1e-12, "minimum vs vertex enumeration");
        assert_close(result.range.maximum, high, 1e-12, "maximum vs vertex enumeration");
        // No random point inside the box, at any smaller fraction, leaves the range.
        for _ in 0..200 {
            let (e_y, e_w) = (kernel * rng.next(), parent * rng.next());
            let qw = Model::mix(&model.pw, &rng.simplex(levels), e_w);
            let qy = (0..levels)
                .map(|w| {
                    [
                        Model::mix(&model.py[w][0], &rng.simplex(categories), e_y),
                        Model::mix(&model.py[w][1], &rng.simplex(categories), e_y),
                    ]
                })
                .collect::<Vec<_>>();
            let value = model.response(&qw, &qy);
            assert!(result.range.minimum - 1e-12 <= value && value <= result.range.maximum + 1e-12);
            checked += 1;
        }
        // The parent vertex witness attains the maximum.
        let level = result.range.maximizing_parent_level.unwrap();
        assert!(level < levels);
    }
    assert_eq!(checked, 40 * 200);
}

#[test]
fn range_contains_enumerated_latent_scm_truth_under_a_known_joint_deviation() {
    // Target SCM: latent U_W ~ Bernoulli(d_W) switches W to a replacement draw
    // R_W; latent U_Y ~ Bernoulli(d_Y) switches Y to R_Y(. | w, x). The target
    // effect is enumerated over (U_W, U_Y, W, Y) directly.
    let model = fixed_model();
    let (e_y, e_w) = (0.2, 0.3);
    let enumerate = |d_y: f64, d_w: f64, rw: &[f64], ry: &dyn Fn(usize, usize) -> Vec<f64>| {
        let mut effect = 0.0;
        for (u_w, p_uw) in [(false, 1.0 - d_w), (true, d_w)] {
            for (u_y, p_uy) in [(false, 1.0 - d_y), (true, d_y)] {
                for w in 0..model.pw.len() {
                    let p_w = if u_w { rw[w] } else { model.pw[w] };
                    for x in 0..2 {
                        let sign = if x == 1 { 1.0 } else { -1.0 };
                        let kernel = if u_y { ry(w, x) } else { model.py[w][x].clone() };
                        for (p_y, y) in kernel.iter().zip(&model.y_values) {
                            effect += sign * p_uw * p_uy * p_w * p_y * y;
                        }
                    }
                }
            }
        }
        effect
    };
    let result = run(&model, &spec(Some(e_y), Some(e_w))).unwrap();
    // An interior deviation.
    let interior = enumerate(0.1, 0.2, &[0.2, 0.2, 0.6], &|w, x| {
        if (w + x) % 2 == 0 { vec![0.1, 0.1, 0.8] } else { vec![0.7, 0.2, 0.1] }
    });
    assert!(result.range.minimum < interior && interior < result.range.maximum);
    // The extremal deviation: parent vertex at the witness, kernel vertices
    // y_max on the active arm and y_min on the control arm, at the box corner.
    let witness = result.range.maximizing_parent_level.unwrap();
    let mut rw = vec![0.0; 3];
    rw[witness] = 1.0;
    let extremal = enumerate(e_y, e_w, &rw, &|_, x| {
        if x == 1 { vec![0.0, 0.0, 1.0] } else { vec![1.0, 0.0, 0.0] }
    });
    assert_close(extremal, result.range.maximum, 1e-12, "extremal SCM attains U");
    let baseline = enumerate(0.0, 0.0, &rw, &|_, _| vec![1.0, 0.0, 0.0]);
    assert_close(baseline, result.baseline, 1e-12, "unperturbed SCM is the baseline");
}

#[test]
fn zero_deviation_reproduces_the_baseline_exactly() {
    let model = fixed_model();
    for factors in [spec(Some(0.0), Some(0.0)), spec(Some(0.0), None), spec(None, Some(0.0))] {
        let result = run(&model, &factors).unwrap();
        assert_eq!(result.range.minimum.to_bits(), result.baseline.to_bits());
        assert_eq!(result.range.maximum.to_bits(), result.baseline.to_bits());
    }
    let one_factor = z_transport_mechanism_sensitivity(
        &diagram(),
        &functional(),
        &data(&model, "snapshot-0"),
        0.0,
        None,
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    let result = run(&model, &spec(Some(0.0), Some(0.0))).unwrap();
    assert_eq!(result.baseline.to_bits(), one_factor.response.baseline.to_bits());
}

#[test]
fn a_single_kernel_factor_is_bit_equal_to_the_2_1_route() {
    let mut rng = Rng(5);
    for case in 0..30 {
        let model = rng.model(1 + case % 4, 2 + case % 3);
        let fraction = rng.next();
        let one_factor = z_transport_mechanism_sensitivity(
            &diagram(),
            &functional(),
            &data(&model, "snapshot-0"),
            fraction,
            None,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        let threshold = one_factor.response.baseline
            + 0.5 * (one_factor.response.maximum - one_factor.response.baseline);
        let with_threshold = z_transport_mechanism_sensitivity(
            &diagram(),
            &functional(),
            &data(&model, "snapshot-0"),
            fraction,
            Some(threshold),
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        let joint = run(&model, &spec(Some(fraction), None).with_threshold(threshold)).unwrap();
        assert_eq!(joint.baseline.to_bits(), one_factor.response.baseline.to_bits());
        assert_eq!(joint.range.minimum.to_bits(), one_factor.response.minimum.to_bits());
        assert_eq!(joint.range.maximum.to_bits(), one_factor.response.maximum.to_bits());
        assert_eq!(
            joint.range.minimizing_outcome_by_stratum,
            one_factor.response.receipt.minimizing_outcome_by_stratum
        );
        assert_eq!(
            joint.range.maximizing_outcome_by_stratum,
            one_factor.response.receipt.maximizing_outcome_by_stratum
        );
        let axis = joint.axis_tipping[0];
        assert_eq!(axis.factor, JointFactor::OutcomeKernel);
        assert_eq!(
            axis.analytic.map(f64::to_bits),
            with_threshold.response.tipping_fraction.map(f64::to_bits)
        );
        assert_eq!(joint.provider_snapshot, one_factor.provider_snapshot);
        assert_eq!(joint.query_binding, one_factor.query_binding);
    }
}

#[test]
fn nested_boxes_give_nested_ranges() {
    let mut rng = Rng(29);
    let mut comparisons = 0;
    for case in 0..60 {
        let model = rng.model(1 + case % 4, 2 + case % 3);
        let (kernel, parent) = (rng.next(), rng.next());
        let outer = run(&model, &spec(Some(kernel), Some(parent))).unwrap();
        for _ in 0..8 {
            let (k, p) = (kernel * rng.next(), parent * rng.next());
            let inner = run(&model, &spec(Some(k), Some(p))).unwrap();
            assert!(outer.range.minimum <= inner.range.minimum, "nested minimum");
            assert!(inner.range.maximum <= outer.range.maximum, "nested maximum");
            assert!(inner.range.minimum <= inner.baseline && inner.baseline <= inner.range.maximum);
            comparisons += 1;
        }
    }
    assert_eq!(comparisons, 480);
}

#[test]
fn axis_tipping_points_match_the_2_1_analytic_values() {
    let model = fixed_model();
    let (kernel, parent) = (0.6, 0.7);
    let baseline = run(&model, &spec(Some(0.0), None)).unwrap().baseline;
    // Thresholds each axis reaches on its own, so both analytic values exist.
    let (kernel_low, kernel_high) = model.closed_form(kernel, 0.0);
    let (parent_low, parent_high) = model.closed_form(0.0, parent);
    let up = 0.5 * (kernel_high.min(parent_high) - baseline);
    let down = 0.5 * (baseline - kernel_low.max(parent_low));
    assert!(up > 0.0 && down > 0.0);
    for threshold in [baseline + up, baseline - down] {
        let mut declared = spec(Some(kernel), Some(parent)).with_threshold(threshold);
        declared.tolerance = 1e-12;
        let result = run(&model, &declared).unwrap();
        let one_factor = z_transport_mechanism_sensitivity(
            &diagram(),
            &functional(),
            &data(&model, "snapshot-0"),
            kernel,
            Some(threshold),
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        let [kernel_axis, parent_axis] = [result.axis_tipping[0], result.axis_tipping[1]];
        // Kernel axis: the 2.1 one-factor analytic value, inside its bracket.
        assert_eq!(kernel_axis.analytic, one_factor.response.tipping_fraction);
        let t = kernel_axis.analytic.expect("reached inside the kernel box");
        let bracket = kernel_axis.bracket.unwrap();
        assert_eq!(kernel_axis.status, TippingStatus::Bracketed);
        assert!(bracket.lower - 1e-12 <= t && t <= bracket.upper + 1e-12);
        assert!(bracket.upper - bracket.lower <= 1e-12);
        // Parent axis: psi(0, e) = b + e (extreme Delta - b), solved independently.
        let deltas = (0..3).map(|w| model.delta(w)).collect::<Vec<_>>();
        let target = if threshold > baseline {
            deltas.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        } else {
            deltas.iter().copied().fold(f64::INFINITY, f64::min)
        };
        let independent = (threshold - baseline) / (target - baseline);
        let analytic = parent_axis.analytic.expect("reached inside the parent box");
        assert_close(analytic, independent, 1e-12, "parent-axis tipping");
        let bracket = parent_axis.bracket.unwrap();
        assert!(bracket.lower - 1e-12 <= analytic && analytic <= bracket.upper + 1e-12);
        // Every frontier bracket is certified against the independent closed form.
        for point in &result.frontier {
            match point.status {
                TippingStatus::Bracketed => {
                    let b = point.bracket.unwrap();
                    let reached = |k: f64| {
                        let (l, u) = model.closed_form(k, point.parent_fraction);
                        if threshold > baseline {
                            u >= threshold - 1e-12
                        } else {
                            l <= threshold + 1e-12
                        }
                    };
                    assert!(reached(b.upper));
                    assert!(!reached(b.lower - 1e-9) || b.lower == 0.0);
                    assert!(b.upper - b.lower <= 1e-12);
                }
                TippingStatus::ReachedAtOrigin => {
                    let (l, u) = model.closed_form(0.0, point.parent_fraction);
                    assert!(if threshold > baseline {
                        u >= threshold - 1e-12
                    } else {
                        l <= threshold + 1e-12
                    });
                }
                other => panic!("unexpected frontier status {other:?}"),
            }
        }
        // More parent deviation never needs more kernel deviation.
        let uppers =
            result.frontier.iter().map(|p| p.bracket.map_or(0.0, |b| b.upper)).collect::<Vec<_>>();
        assert!(uppers.windows(2).all(|pair| pair[1] <= pair[0] + 1e-12));
        assert_eq!(result.frontier.len(), 17);
    }
}

#[test]
fn a_tied_argmax_parent_level_keeps_the_range_and_frontier_exact() {
    let mut model = fixed_model();
    // Levels 1 and 2 share one kernel, so Delta ties at the argmax and argmin.
    model.py[2] = model.py[1].clone();
    model.px1[2] = model.px1[1];
    for (kernel, parent) in [(0.25, 0.4), (0.0, 0.9), (0.8, 0.0)] {
        let result = run(&model, &spec(Some(kernel), Some(parent))).unwrap();
        let (low, high) = model.vertex_extrema(kernel, parent);
        assert_close(result.range.minimum, low, 1e-12, "tied minimum");
        assert_close(result.range.maximum, high, 1e-12, "tied maximum");
    }
    let baseline = run(&model, &spec(Some(0.0), None)).unwrap().baseline;
    let threshold = baseline + 0.2;
    let result = run(&model, &spec(Some(0.5), Some(0.5)).with_threshold(threshold)).unwrap();
    for point in result.frontier.iter().filter(|p| p.status == TippingStatus::Bracketed) {
        let b = point.bracket.unwrap();
        assert!(model.closed_form(b.upper, point.parent_fraction).1 >= threshold - 1e-12);
        assert!(model.closed_form(b.lower, point.parent_fraction).1 <= threshold + 1e-12);
    }
    assert!(result.frontier.iter().any(|p| p.status == TippingStatus::Bracketed));
    // A threshold the response attains exactly at the first bisection midpoint
    // (kernel fraction 0.25 of the box's 0.5, on the parent-fraction-0 line):
    // "reaches" is `>=` (upward) / `<=` (downward), so that midpoint is the
    // bracket's upper end and the lower end strictly misses the threshold, both
    // on the evaluator's own floating-point response and on the independent
    // closed form.
    let at_middle = run(&model, &spec(Some(0.25), None)).unwrap().range;
    for (threshold, upward) in [(at_middle.maximum, true), (at_middle.minimum, false)] {
        let result = run(&model, &spec(Some(0.5), Some(0.5)).with_threshold(threshold)).unwrap();
        let line = result.frontier[0];
        assert_eq!(line.parent_fraction, 0.0);
        assert_eq!(line.status, TippingStatus::Bracketed);
        let b = line.bracket.unwrap();
        assert_eq!(b.upper, 0.25, "the attained midpoint is the upper end (upward {upward})");
        assert!(b.lower < b.upper && b.upper - b.lower <= 1e-9);
        let evaluated = run(&model, &spec(Some(b.lower), None)).unwrap().range;
        let (independent_low, independent_high) = model.closed_form(b.lower, 0.0);
        if upward {
            assert!(evaluated.maximum < threshold, "U(lower) must miss the threshold");
            assert!(independent_high < threshold);
        } else {
            assert!(evaluated.minimum > threshold, "L(lower) must miss the threshold");
            assert!(independent_low > threshold);
        }
    }
}

#[test]
fn the_parent_witness_is_absent_unless_the_parent_is_perturbed() {
    let model = fixed_model();
    // No parent factor, or a parent factor bounded at zero: no parent vertex.
    for declared in [spec(Some(0.2), None), spec(Some(0.2), Some(0.0))] {
        let range = run(&model, &declared).unwrap().range;
        assert_eq!(range.maximizing_parent_level, None);
        assert_eq!(range.minimizing_parent_level, None);
        assert_eq!(range, run(&model, &spec(Some(0.2), None)).unwrap().range);
    }
    // A positive parent bound names the vertex: argmax / argmin of D+ / D-.
    let range = run(&model, &spec(Some(0.2), Some(0.3))).unwrap().range;
    let spread = 3.0;
    let up = (0..3).map(|w| 0.8 * model.delta(w) + 0.2 * spread).collect::<Vec<_>>();
    let down = (0..3).map(|w| 0.8 * model.delta(w) - 0.2 * spread).collect::<Vec<_>>();
    let argmax = (0..3).max_by(|a, b| up[*a].total_cmp(&up[*b])).unwrap();
    let argmin = (0..3).min_by(|a, b| down[*a].total_cmp(&down[*b])).unwrap();
    assert_eq!(range.maximizing_parent_level, Some(argmax));
    assert_eq!(range.minimizing_parent_level, Some(argmin));
}

fn refusal(
    result: Result<JointMechanismSensitivityResult, JointSensitivityError>,
) -> (&'static str, &'static str) {
    let error = result.expect_err("expected a refusal");
    (error.reason_code(), error.detail())
}

#[test]
fn out_of_scope_factors_and_budget_coupling_refuse_by_detail() {
    let model = fixed_model();
    let mut coupled = spec(Some(0.1), Some(0.1));
    coupled.total_budget = Some(0.15);
    assert_eq!(
        refusal(run(&model, &coupled)),
        ("route_not_supported", "joint_sensitivity.budget_coupling")
    );
    for (factor, detail) in [
        (JointFactor::FixedGraphParentMechanism, "joint_sensitivity.fixed_graph_parent"),
        (JointFactor::FixedGraphConditionalMechanism, "joint_sensitivity.fixed_graph_conditional"),
        (JointFactor::SourceTargetDiscrepancy, "joint_sensitivity.source_target_discrepancy"),
    ] {
        let mut declared = spec(Some(0.1), None);
        declared.factors.push(JointFactorBound { factor, max_fraction: 0.1 });
        assert_eq!(refusal(run(&model, &declared)), ("route_not_supported", detail));
    }
    let none = JointDeviationSpec::new(Vec::new());
    assert_eq!(
        refusal(run(&model, &none)),
        ("route_not_supported", "joint_sensitivity.factor_count")
    );
    // The cap is two: both factors of the formula are admitted, and a third
    // declaration of any kind (a duplicate or an out-of-scope factor) refuses
    // by count before its kind is examined.
    run(&model, &spec(Some(0.1), Some(0.1))).unwrap();
    for third in [
        JointFactor::OutcomeKernel,
        JointFactor::SharedParentMarginal,
        JointFactor::TreatmentMechanism,
        JointFactor::FixedGraphParentMechanism,
        JointFactor::FixedGraphConditionalMechanism,
        JointFactor::SourceTargetDiscrepancy,
    ] {
        let mut three = spec(Some(0.1), Some(0.1));
        three.factors.push(JointFactorBound { factor: third, max_fraction: 0.1 });
        assert_eq!(
            refusal(run(&model, &three)),
            ("route_not_supported", "joint_sensitivity.factor_count"),
            "third factor {third:?}"
        );
    }
}

#[test]
fn invalid_fractions_thresholds_and_duplicates_refuse() {
    let model = fixed_model();
    let mut duplicate = spec(Some(0.1), None);
    duplicate
        .factors
        .push(JointFactorBound { factor: JointFactor::OutcomeKernel, max_fraction: 0.2 });
    assert_eq!(
        refusal(run(&model, &duplicate)),
        ("invalid_argument", "joint_sensitivity.duplicate_factor")
    );
    let mut treatment = spec(Some(0.1), None);
    treatment
        .factors
        .push(JointFactorBound { factor: JointFactor::TreatmentMechanism, max_fraction: 0.1 });
    assert_eq!(
        refusal(run(&model, &treatment)),
        ("invalid_argument", "joint_sensitivity.treatment_factor")
    );
    for bad in [1.0 + 1e-12, -1e-12, f64::NAN, f64::INFINITY] {
        assert_eq!(
            refusal(run(&model, &spec(Some(0.1), Some(bad)))),
            ("invalid_argument", "joint_sensitivity.invalid_fraction")
        );
    }
    for good in [0.0, 1.0] {
        run(&model, &spec(Some(good), Some(good))).unwrap();
    }
    for bad in [f64::NAN, f64::NEG_INFINITY] {
        assert_eq!(
            refusal(run(&model, &spec(Some(0.1), None).with_threshold(bad))),
            ("invalid_argument", "joint_sensitivity.invalid_threshold")
        );
    }
    for (tolerance, ok) in
        [(1e-12, true), (1e-2, true), (9e-13, false), (0.011, false), (f64::NAN, false)]
    {
        let mut declared = spec(Some(0.1), None).with_threshold(0.0);
        declared.tolerance = tolerance;
        match run(&model, &declared) {
            Ok(_) => assert!(ok, "tolerance {tolerance} admitted"),
            Err(error) => {
                assert!(!ok, "tolerance {tolerance} refused");
                assert_eq!(
                    (error.reason_code(), error.detail()),
                    ("invalid_argument", "joint_sensitivity.invalid_tolerance")
                );
            }
        }
    }
}

#[test]
fn declared_bounds_admit_the_cap_and_refuse_one_more() {
    let model = fixed_model();
    let bounded = |mutate: &dyn Fn(&mut JointDeviationSpec)| {
        let mut declared = spec(Some(0.1), Some(0.1)).with_threshold(0.0);
        mutate(&mut declared);
        run(&model, &declared)
    };
    let exceeded = ("route_not_supported", "joint_sensitivity.bounds_exceeded");
    assert_eq!(bounded(&|s| s.frontier_points = 33).unwrap().frontier.len(), 33);
    assert_eq!(refusal(bounded(&|s| s.frontier_points = 34)), exceeded);
    assert_eq!(refusal(bounded(&|s| s.frontier_points = 0)), exceeded);
    bounded(&|s| s.limits.operations = 100_000).unwrap();
    assert_eq!(refusal(bounded(&|s| s.limits.operations = 100_001)), exceeded);
    bounded(&|s| s.limits.depth = 64).unwrap();
    assert_eq!(refusal(bounded(&|s| s.limits.depth = 65)), exceeded);
    // The declared memory cap has a hard ceiling (512 MiB); a larger request is
    // refused, not silently raised.
    bounded(&|s| s.limits.memory_bytes = 512 << 20).unwrap();
    assert_eq!(refusal(bounded(&|s| s.limits.memory_bytes = (512 << 20) + 1)), exceeded);
    assert_eq!(refusal(bounded(&|s| s.limits.memory_bytes = u64::MAX)), exceeded);
    // 64 parent levels and 32 outcome categories are admitted; one more refuses.
    let mut rng = Rng(3);
    run(&rng.model(64, 2), &spec(Some(0.1), Some(0.1))).unwrap();
    assert_eq!(refusal(run(&rng.model(65, 2), &spec(Some(0.1), Some(0.1)))), exceeded);
    run(&rng.model(1, 32), &spec(Some(0.1), Some(0.1))).unwrap();
    assert_eq!(refusal(run(&rng.model(1, 33), &spec(Some(0.1), Some(0.1)))), exceeded);
}

#[test]
fn law_bounds_refuse_before_the_charged_read_of_the_law() {
    let mut rng = Rng(5);
    // The read of the law is charged one operation per parent level: a
    // one-operation budget stops a 64-level law on its second level ...
    let mut one = spec(Some(0.1), Some(0.1));
    one.limits.operations = 1;
    match run(&rng.model(64, 2), &one) {
        Err(JointSensitivityError::Budget(receipt)) => {
            assert_eq!(receipt.stop, SearchStop::Operations);
            assert_eq!(receipt.operations_consumed, Some(1));
            assert_eq!(receipt.unevaluated, ["range"]);
        }
        other => panic!("expected the read to be charged, got {other:?}"),
    }
    // ... while a 65-level (or 33-category) law refuses on its shape, before any
    // cell is read or charged.
    for model in [rng.model(65, 2), rng.model(2, 33)] {
        assert_eq!(
            refusal(run(&model, &one)),
            ("route_not_supported", "joint_sensitivity.bounds_exceeded")
        );
    }
    // Cancellation is observed when the budget is created, before the shape is
    // read, so a cancelled context is a budget stop even for an oversized law.
    let ctx = ExecutionContext::for_tests(1);
    ctx.cancellation.cancel();
    assert_eq!(
        refusal(run_ctx(&rng.model(65, 2), &spec(Some(0.1), None), &ctx)).1,
        "joint_sensitivity.budget"
    );
}

#[test]
fn a_non_surrogate_formula_or_wrong_provider_refuses() {
    let model = fixed_model();
    // A provider whose snapshot the catalog binding does not name.
    let wrong = z_transport_joint_mechanism_sensitivity(
        &diagram(),
        &functional(),
        &data(&model, "snapshot-other"),
        &spec(Some(0.1), None),
        &ExecutionContext::for_tests(1),
    );
    assert_eq!(
        refusal(wrong),
        ("transport_missing_provider", "joint_sensitivity.provider_mismatch")
    );
    // A checked z formula that is not the surrogate factorization: Z -> X -> Y
    // with no shared parent has no confounder factor.
    let mut graph = Admg::with_variables(4);
    for (from, to) in [(1, 2), (2, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let other = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
    let ZTransportResult::Identified(proof) = identify_z_transport(
        &other,
        &query(),
        SidLimits::default(),
        &ExecutionContext::for_tests(0),
    )
    .unwrap() else {
        panic!("a direct z formula identifies");
    };
    let functional =
        bind_z_transport_catalog(&other, &query(), &proof, &with_target(catalog("snapshot-0")))
            .unwrap();
    let result = z_transport_joint_mechanism_sensitivity(
        &other,
        &functional,
        &data(&model, "snapshot-0"),
        &spec(Some(0.1), None),
        &ExecutionContext::for_tests(1),
    );
    assert_eq!(
        refusal(result),
        ("transport_not_certified", "joint_sensitivity.formula_incompatible")
    );
    // The surrogate proof replayed against another diagram fails its input identity.
    let result = z_transport_joint_mechanism_sensitivity(
        &other,
        &self::functional(),
        &data(&model, "snapshot-0"),
        &spec(Some(0.1), None),
        &ExecutionContext::for_tests(1),
    );
    assert_eq!(
        refusal(result),
        ("transport_not_certified", "joint_sensitivity.formula_incompatible")
    );
}

#[test]
fn every_joint_sensitivity_detail_pairs_with_its_recorded_reason_code() {
    let model = fixed_model();
    let recorded = [
        ("route_not_supported", "joint_sensitivity.factor_count"),
        ("route_not_supported", "joint_sensitivity.budget_coupling"),
        ("route_not_supported", "joint_sensitivity.fixed_graph_parent"),
        ("route_not_supported", "joint_sensitivity.fixed_graph_conditional"),
        ("route_not_supported", "joint_sensitivity.source_target_discrepancy"),
        ("route_not_supported", "joint_sensitivity.bounds_exceeded"),
        ("route_not_supported", "joint_sensitivity.unsupported_domain"),
        ("invalid_argument", "joint_sensitivity.duplicate_factor"),
        ("invalid_argument", "joint_sensitivity.invalid_fraction"),
        ("invalid_argument", "joint_sensitivity.treatment_factor"),
        ("invalid_argument", "joint_sensitivity.invalid_threshold"),
        ("invalid_argument", "joint_sensitivity.invalid_tolerance"),
        ("transport_not_certified", "joint_sensitivity.formula_incompatible"),
        ("transport_missing_provider", "joint_sensitivity.provider_mismatch"),
        ("transport_support_failure", "joint_sensitivity.incomplete_kernel"),
        ("transport_budget_cancel", "joint_sensitivity.budget"),
    ];
    let mut observed = Vec::new();
    let mut push = |result| observed.push(refusal(result));
    let with = |mutate: &dyn Fn(&mut JointDeviationSpec)| {
        let mut declared = spec(Some(0.1), Some(0.1)).with_threshold(0.0);
        mutate(&mut declared);
        run(&model, &declared)
    };
    push(with(&|s| s.factors.clear()));
    push(with(&|s| s.total_budget = Some(0.1)));
    for factor in [
        JointFactor::FixedGraphParentMechanism,
        JointFactor::FixedGraphConditionalMechanism,
        JointFactor::SourceTargetDiscrepancy,
    ] {
        push(with(&|s| s.factors[1].factor = factor));
    }
    push(with(&|s| s.frontier_points = 99));
    // A non-numeric outcome level.
    let mut labelled = data(&model, "snapshot-0").laws()[0].clone();
    let mut axes = labelled.axes().to_vec();
    axes[2].values =
        Arc::from([Value::Label(Arc::from("a")), Value::Float64(1.0), Value::Float64(3.0)]);
    labelled = ExactDiscreteLaw::try_new(
        "source",
        RegimeId::from_raw(0),
        labelled.interventions().to_vec(),
        axes,
        labelled.probabilities().to_vec(),
        "snapshot-0",
        LawTolerance::default(),
    )
    .unwrap();
    push(z_transport_joint_mechanism_sensitivity(
        &diagram(),
        &functional(),
        &ExactTransportData::try_new([labelled], 1 << 20).unwrap(),
        &spec(Some(0.1), None),
        &ExecutionContext::for_tests(1),
    ));
    push(with(&|s| s.factors[1].factor = JointFactor::OutcomeKernel));
    push(with(&|s| s.factors[0].max_fraction = 2.0));
    push(with(&|s| s.factors[1].factor = JointFactor::TreatmentMechanism));
    push(with(&|s| s.decision_threshold = Some(f64::NAN)));
    push(with(&|s| s.tolerance = 0.0));
    let mut graph = Admg::with_variables(4);
    for (from, to) in [(1, 2), (2, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph.insert_bidirected(DenseNodeId::from_raw(1), DenseNodeId::from_raw(3)).unwrap();
    let other = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
    push(z_transport_joint_mechanism_sensitivity(
        &other,
        &functional(),
        &data(&model, "snapshot-0"),
        &spec(Some(0.1), None),
        &ExecutionContext::for_tests(1),
    ));
    push(z_transport_joint_mechanism_sensitivity(
        &diagram(),
        &functional(),
        &data(&model, "snapshot-other"),
        &spec(Some(0.1), None),
        &ExecutionContext::for_tests(1),
    ));
    // A (w, x) stratum without mass.
    let mut empty = model.clone();
    empty.px1[1] = 0.0;
    push(run(&empty, &spec(Some(0.1), None)));
    push(with(&|s| s.limits.operations = 0));
    assert_eq!(observed, recorded);
}

#[test]
fn an_operation_budget_leaves_the_frontier_unresolved_and_the_range_exact() {
    let model = fixed_model();
    let baseline = run(&model, &spec(Some(0.0), None)).unwrap().baseline;
    let full = run(&model, &spec(Some(0.5), Some(0.5)).with_threshold(baseline + 0.4)).unwrap();
    assert_eq!(full.unresolved_detail, None);
    assert_eq!(full.receipt.stop, None);
    // The read of the law charges one operation per parent level (3); the
    // closed-form range is not charged; the first frontier line needs more
    // than two further operations.
    let mut tight = spec(Some(0.5), Some(0.5)).with_threshold(baseline + 0.4);
    tight.limits.operations = 5;
    let stopped = run(&model, &tight).unwrap();
    assert_eq!(stopped.range, full.range, "the range is exact whatever the frontier budget");
    assert_eq!(stopped.receipt.stop, Some(SearchStop::Operations));
    assert_eq!(stopped.receipt.operations_consumed, 5);
    assert_eq!(stopped.unresolved_detail, Some("joint_sensitivity.budget"));
    assert_eq!(stopped.receipt.explored, ["range"]);
    assert_eq!(stopped.receipt.unevaluated.len(), 18);
    assert!(stopped.frontier.iter().all(|p| p.status == TippingStatus::Unevaluated));
    assert!(stopped.axis_tipping.iter().all(|a| a.status == TippingStatus::Unevaluated));
    // Fewer operations than the read of the law needs refuse with the receipt.
    tight.limits.operations = 2;
    match run(&model, &tight) {
        Err(JointSensitivityError::Budget(receipt)) => {
            assert_eq!(receipt.stop, SearchStop::Operations);
            assert_eq!(receipt.operations_consumed, Some(2));
            assert_eq!(receipt.unevaluated, ["range"]);
        }
        other => panic!("expected a budget refusal, got {other:?}"),
    }
}

#[test]
fn depth_and_memory_stops_are_receipts_checked_on_every_charge() {
    let model = fixed_model();
    let baseline = run(&model, &spec(Some(0.0), None)).unwrap().baseline;
    let mut deep = spec(Some(0.5), Some(0.5)).with_threshold(baseline + 0.4);
    deep.tolerance = 1e-12;
    deep.limits.depth = 3;
    let stopped = run(&model, &deep).unwrap();
    assert_eq!(stopped.receipt.stop, Some(SearchStop::Depth));
    assert_eq!(stopped.receipt.depth_reached, 4);
    assert_eq!(stopped.receipt.limits.depth, 3);
    assert_eq!(stopped.unresolved_detail, Some("joint_sensitivity.budget"));
    // Memory: the range fits, the cumulative frontier state does not.
    let full = run(&model, &spec(Some(0.5), Some(0.5)).with_threshold(baseline + 0.4)).unwrap();
    let mut wider = spec(Some(0.5), Some(0.5)).with_threshold(baseline + 0.4);
    wider.frontier_points = 18;
    let line =
        run(&model, &wider).unwrap().receipt.live_state_bytes - full.receipt.live_state_bytes;
    assert!(line > 0, "every resolved line adds live state");
    let range_bytes = full.receipt.live_state_bytes - 18 * line;
    let mut ctx = ExecutionContext::for_tests(1);
    ctx.memory =
        MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: Some(range_bytes + 2 * line) };
    let stopped =
        run_ctx(&model, &spec(Some(0.5), Some(0.5)).with_threshold(baseline + 0.4), &ctx).unwrap();
    assert_eq!(stopped.receipt.stop, Some(SearchStop::Memory));
    assert_eq!(stopped.receipt.memory_limit_bytes, range_bytes + 2 * line);
    assert_eq!(stopped.receipt.explored.len(), 3, "range and two frontier lines fit");
    assert_eq!(stopped.range, full.range);
    // A cap below the range state refuses before the range, with the receipt.
    ctx.memory.hard_limit_bytes = Some(8);
    match run_ctx(&model, &spec(Some(0.5), Some(0.5)), &ctx) {
        Err(JointSensitivityError::Budget(receipt)) => {
            assert_eq!(receipt.stop, SearchStop::Memory);
            assert_eq!(receipt.memory_limit_bytes, Some(8));
        }
        other => panic!("expected a memory refusal, got {other:?}"),
    }
    // The declared cap is the budget's own cap when the context has none.
    let mut capped = spec(Some(0.5), Some(0.5)).with_threshold(baseline + 0.4);
    capped.limits =
        JointSensitivityLimits { memory_bytes: range_bytes + 2 * line, ..capped.limits };
    let stopped = run(&model, &capped).unwrap();
    assert_eq!(stopped.receipt.stop, Some(SearchStop::Memory));
}

struct CancelAfterFirstLine(antecedent_core::CancellationToken);

impl ProgressSink for CancelAfterFirstLine {
    fn report(&self, _fraction: f64, stage: &str) {
        if stage == "joint sensitivity frontier" {
            self.0.cancel();
        }
    }
}

#[test]
fn cancellation_stops_the_bracketing_loop_with_a_receipt() {
    let model = fixed_model();
    let baseline = run(&model, &spec(Some(0.0), None)).unwrap().baseline;
    let declared = spec(Some(0.5), Some(0.5)).with_threshold(baseline + 0.4);
    let mut ctx = ExecutionContext::for_tests(1);
    ctx.progress = Some(Arc::new(CancelAfterFirstLine(ctx.cancellation.clone())));
    let stopped = run_ctx(&model, &declared, &ctx).unwrap();
    assert_eq!(stopped.receipt.stop, Some(SearchStop::Cancelled));
    assert_eq!(stopped.receipt.explored.len(), 2, "range and the first frontier line");
    assert_eq!(stopped.receipt.unevaluated.len(), 17);
    assert_eq!(stopped.frontier[0].status, TippingStatus::Bracketed);
    assert_eq!(stopped.frontier[1].status, TippingStatus::Unevaluated);
    assert_eq!(stopped.unresolved_detail, Some("joint_sensitivity.budget"));
    // Cancelled before entry: a refusal with its receipt.
    let ctx = ExecutionContext::for_tests(1);
    ctx.cancellation.cancel();
    match run_ctx(&model, &declared, &ctx) {
        Err(JointSensitivityError::Budget(receipt)) => {
            assert_eq!(receipt.stop, SearchStop::Cancelled);
            assert_eq!(receipt.operations_consumed, None);
        }
        other => panic!("expected a cancellation refusal, got {other:?}"),
    }
}

#[test]
fn one_budget_is_shared_by_the_range_and_every_frontier_line() {
    let model = fixed_model();
    let baseline = run(&model, &spec(Some(0.0), None)).unwrap().baseline;
    let declared = spec(Some(0.5), Some(0.5)).with_threshold(baseline + 0.4);
    let full = run(&model, &declared).unwrap();
    let total = full.receipt.operations_consumed;
    // Each line alone needs far less than the total, yet the total minus one
    // operation leaves the last stage unevaluated: the budget does not reset.
    let mut short = declared.clone();
    short.limits.operations = total - 1;
    let stopped = run(&model, &short).unwrap();
    assert_eq!(stopped.receipt.stop, Some(SearchStop::Operations));
    assert_eq!(stopped.receipt.operations_consumed, total - 1);
    assert_eq!(stopped.receipt.unevaluated, ["axis[shared_parent_marginal]"]);
    let mut exact = declared;
    exact.limits.operations = total;
    assert_eq!(run(&model, &exact).unwrap().receipt.stop, None);
    let per_line = full.receipt.operations_consumed / 19;
    assert!(per_line < total - 1);
}

#[test]
fn factor_and_stratum_order_do_not_change_the_result() {
    let model = fixed_model();
    let baseline = run(&model, &spec(Some(0.0), None)).unwrap().baseline;
    let forward = run(&model, &spec(Some(0.3), Some(0.4)).with_threshold(baseline + 0.2)).unwrap();
    let mut reversed = spec(Some(0.3), Some(0.4)).with_threshold(baseline + 0.2);
    reversed.factors.reverse();
    let backward = run(&model, &reversed).unwrap();
    assert_eq!(forward, backward);
    assert_eq!(forward.factors[0].factor, JointFactor::OutcomeKernel);
    // Relabelling the parent levels permutes the witnesses, never the range.
    let mut permuted = model.clone();
    permuted.pw.rotate_left(1);
    permuted.px1.rotate_left(1);
    permuted.py.rotate_left(1);
    let moved = run(&permuted, &spec(Some(0.3), Some(0.4))).unwrap();
    assert_close(moved.range.minimum, forward.range.minimum, 1e-12, "permuted minimum");
    assert_close(moved.range.maximum, forward.range.maximum, 1e-12, "permuted maximum");
    assert_eq!(
        (moved.range.maximizing_parent_level.unwrap() + 1) % 3,
        forward.range.maximizing_parent_level.unwrap()
    );
}

/// The model's law with every cell scaled by `scale`, accepted under `tolerance`.
fn scaled_data(model: &Model, scale: f64, tolerance: LawTolerance) -> ExactTransportData {
    let exact = data(model, "snapshot-0");
    let law = &exact.laws()[0];
    let law = ExactDiscreteLaw::try_new(
        law.population(),
        law.regime(),
        law.interventions().to_vec(),
        law.axes().to_vec(),
        law.probabilities().iter().map(|p| p * scale).collect::<Vec<_>>(),
        law.snapshot_identity(),
        tolerance,
    )
    .unwrap();
    ExactTransportData::try_new([law], 1 << 20).unwrap()
}

#[test]
fn factor_normalization_is_enforced_and_every_stage_reads_one_scale() {
    let model = fixed_model();
    let run_on = |data: &ExactTransportData, declared: &JointDeviationSpec| {
        z_transport_joint_mechanism_sensitivity(
            &diagram(),
            &functional(),
            data,
            declared,
            &ExecutionContext::for_tests(1),
        )
    };
    // A law the caller's own loose tolerance accepts, whose shared parent
    // marginal is not a distribution (mass 0.8): the contamination class is
    // undefined on it, so the route refuses instead of mixing scales.
    let loose = LawTolerance { absolute: 0.0, relative: 0.25 };
    for scale in [0.8, 1.2, 1.0 + 2e-10] {
        let unnormalized = scaled_data(&model, scale, loose);
        for declared in [spec(Some(0.2), Some(0.3)), spec(Some(0.2), None), spec(None, Some(0.3))] {
            assert_eq!(
                refusal(run_on(&unnormalized, &declared)),
                ("transport_support_failure", "joint_sensitivity.incomplete_kernel"),
                "parent marginal of mass {scale}"
            );
        }
    }
    // A law within the unit-mass tolerance (1e-10) is evaluated as read: the
    // range, the frontier brackets and both axis values share its one scale,
    // so each analytic axis value lies inside its certified bracket.
    let near = scaled_data(&model, 1.0 + 9e-11, LawTolerance::default());
    let baseline = run_on(&near, &spec(Some(0.0), None)).unwrap().baseline;
    for threshold in [baseline + 0.15, baseline - 0.15] {
        let mut declared = spec(Some(0.6), Some(0.7)).with_threshold(threshold);
        declared.tolerance = 1e-12;
        let result = run_on(&near, &declared).unwrap();
        for axis in &result.axis_tipping {
            let analytic = axis.analytic.expect("reached on each axis");
            let bracket = axis.bracket.expect("resolved");
            assert_eq!(axis.status, TippingStatus::Bracketed, "{:?}", axis.factor);
            assert!(
                bracket.lower - 1e-12 <= analytic && analytic <= bracket.upper + 1e-12,
                "{:?}: analytic {analytic} outside [{}, {}]",
                axis.factor,
                bracket.lower,
                bracket.upper
            );
        }
    }
}

#[test]
fn the_range_is_an_assumption_range_and_the_interval_is_withheld() {
    let result = run(&fixed_model(), &spec(Some(0.2), Some(0.2))).unwrap();
    assert_eq!(result.inference_claim, "assumption_range");
    assert!(result.interpretation.contains("not a confidence interval"));
    assert_eq!(result.uncertainty.reason_code, "cell_not_licensed");
    assert_eq!(result.uncertainty.detail, "joint_sensitivity.interval_withheld");
    assert_eq!(result.uncertainty.method, "conservative_endpoint_percentile_bootstrap");
    assert!(result.uncertainty.coverage_target.starts_with("one_sided"));
}

/// Counted source table: `n` rows allocated to the model's cells.
fn counted(model: &Model, n: u64) -> ExactTransportData {
    let exact = data(model, "snapshot-0");
    let law = &exact.laws()[0];
    let counts =
        law.probabilities().iter().map(|p| (p * n as f64).round() as u64).collect::<Vec<_>>();
    let total = counts.iter().sum::<u64>() as f64;
    let law = ExactDiscreteLaw::try_empirical(
        law.population(),
        law.regime(),
        law.interventions().to_vec(),
        law.axes().to_vec(),
        counts.iter().map(|c| *c as f64 / total).collect::<Vec<_>>(),
        law.snapshot_identity(),
        law.tolerance(),
    )
    .unwrap()
    .with_empirical_counts(counts)
    .unwrap();
    ExactTransportData::try_new([law], 1 << 20).unwrap()
}

#[test]
fn the_endpoint_bootstrap_runs_behind_calibration_internal_and_dominates_the_pointwise_interval() {
    use antecedent_validate::joint_sensitivity_bootstrap_interval_internal as endpoint_bootstrap;
    let model = fixed_model();
    let data = counted(&model, 600);
    let ctx = ExecutionContext::for_tests(17);
    let interval = |declared: &JointDeviationSpec, replicates: u32| {
        endpoint_bootstrap(&diagram(), &functional(), &data, declared, replicates, 0.9, &ctx)
    };
    // Zero box: exactly the ordinary percentile bootstrap of the point.
    let zero = interval(&spec(Some(0.0), Some(0.0)), 60).unwrap().unwrap();
    assert_eq!(zero.lower.to_bits(), zero.pointwise_at_zero.0.to_bits());
    assert_eq!(zero.upper.to_bits(), zero.pointwise_at_zero.1.to_bits());
    assert!(zero.lower < zero.upper);
    assert_eq!(zero.replicates_ok + zero.replicates_failed, 60);
    assert_eq!(zero.method, "conservative_endpoint_percentile_bootstrap");
    assert!(zero.coverage_target.starts_with("one_sided"));
    // Positive box: the endpoint interval contains the pointwise interval of
    // psi(0) (delta0 = 0 lies in the box) and is strictly wider.
    let wide = interval(&spec(Some(0.1), Some(0.1)), 60).unwrap().unwrap();
    assert_eq!(wide.pointwise_at_zero, zero.pointwise_at_zero, "same draws, same stream");
    assert!(wide.lower < zero.lower && zero.upper < wide.upper);
    // Deterministic under its seed.
    assert_eq!(interval(&spec(Some(0.1), Some(0.1)), 60).unwrap().unwrap(), wide);
    // Replicates and level are bounded; exact laws withhold the interval.
    for (replicates, detail) in
        [(1, "joint_sensitivity.bounds_exceeded"), (2001, "joint_sensitivity.bounds_exceeded")]
    {
        let error = interval(&spec(Some(0.1), None), replicates).unwrap_err();
        assert_eq!((error.reason_code(), error.detail()), ("route_not_supported", detail));
    }
    for admitted in [2, 2000] {
        let at_cap = interval(&spec(Some(0.1), None), admitted).unwrap().unwrap();
        assert_eq!(at_cap.replicates_requested, admitted);
        assert_eq!(at_cap.replicates_ok + at_cap.replicates_failed, admitted);
    }
    let exact = endpoint_bootstrap(
        &diagram(),
        &functional(),
        &self::data(&model, "snapshot-0"),
        &spec(Some(0.1), None),
        10,
        0.9,
        &ctx,
    )
    .unwrap();
    assert_eq!(exact, Err("exact_supplied_law_no_sampling_uncertainty"));
}
