//! Exact sensitivity ranges for finite discrete outcome mechanisms.
//!
//! The declared source kernel is contaminated with a single replacement
//! distribution per parent stratum. The fixed-graph route can apply this
//! across both response arms or restrict it to one treatment-level slice while
//! holding the other arm fixed. The reported range is an assumption range, not
//! a sampling interval or a sharp causal bound without the declared graph and
//! kernel assumptions.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use antecedent_core::{EvidenceKind, RegimeId, VariableId};
use antecedent_expr::ExactTransportData;
use antecedent_expr::ExprNode;
use antecedent_graph::{DenseNodeId, NodeRef, SelectionDiagram};
use antecedent_identify::{
    BoundTransportFunctional, BoundZTransportFunctional, ClassicalTransportQuery, SidLimits,
    verify_classical_transport,
};

/// Inputs for discrete outcome-kernel contamination on a fixed graph.
///
/// `source_kernel` is a row-major matrix indexed by parent stratum then outcome.
/// `stratum_contrast_weights` defines the response contrast (for example, the
/// difference in target stratum probabilities under two interventions). A
/// single contaminated kernel is used for every intervention arm.
#[derive(Clone, Debug)]
pub struct DiscreteKernelSensitivity {
    /// Source probabilities `P(Y=y | parents=s)`, row-major by stratum.
    pub source_kernel: Vec<Vec<f64>>,
    /// Outcome utility/response for each ordered outcome category.
    pub outcome_values: Vec<f64>,
    /// Contrast coefficient for each parent stratum.
    pub stratum_contrast_weights: Vec<f64>,
    /// Largest contamination fraction, in `[0, 1]`.
    pub max_fraction: f64,
    /// Optional decision threshold for the response contrast.
    pub decision_threshold: Option<f64>,
}

/// Exact optimization witness for the two extremal response values.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscreteKernelOptimizationReceipt {
    /// Outcome category selected for the minimizing replacement kernel in each stratum.
    pub minimizing_outcome_by_stratum: Vec<usize>,
    /// Outcome category selected for the maximizing replacement kernel in each stratum.
    pub maximizing_outcome_by_stratum: Vec<usize>,
    /// Declared contamination domain `[0, max_fraction]`.
    pub fraction_domain: [f64; 2],
    /// Short description of the exact simplex-vertex optimization.
    pub method: &'static str,
}

/// Assumption range and baseline for one-factor discrete kernel contamination.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscreteKernelSensitivityResult {
    /// Response contrast under the unmodified source kernel.
    pub baseline: f64,
    /// Minimum response over the declared contamination domain.
    pub minimum: f64,
    /// Maximum response over the declared contamination domain.
    pub maximum: f64,
    /// Smallest contamination fraction for which the interval contains the threshold.
    pub tipping_fraction: Option<f64>,
    /// These extrema are assumption ranges, not sampling intervals.
    pub interval_interpretation: &'static str,
    /// Exact optimization witness.
    pub receipt: DiscreteKernelOptimizationReceipt,
}

/// Invalid finite-kernel sensitivity inputs.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum DiscreteKernelSensitivityError {
    /// No strata or categories were supplied, or dimensions disagree.
    #[error("kernel, outcome, and stratum dimensions must agree and be nonempty")]
    InvalidDimensions,
    /// Probabilities or response values are nonfinite, or a kernel row is not a distribution.
    #[error("source kernel rows must be finite probability distributions")]
    InvalidKernel,
    /// The contamination fraction is outside `[0, 1]`.
    #[error("maximum contamination fraction must be in [0, 1]")]
    InvalidFraction,
    /// A contrast weight or decision threshold is nonfinite.
    #[error("contrast weights, outcome values, and threshold must be finite")]
    InvalidContrast,
}

/// One source outcome-kernel row indexed by the outcome's ordered graph parents.
#[derive(Clone, Debug)]
pub struct SourceOutcomeKernelRow {
    /// Categorical index for each parent, in ascending variable order.
    pub parent_levels: Vec<usize>,
    /// Outcome probabilities in the declared category order.
    pub outcome_probabilities: Vec<f64>,
}

/// Source-population mass for each non-treatment outcome-parent stratum.
#[derive(Clone, Debug)]
pub struct SourceParentLawRow {
    /// Category index for each outcome parent other than treatment.
    pub parent_levels: Vec<usize>,
    /// Source probability mass of this stratum.
    pub probability: f64,
}

/// One target parent-law row under a treatment intervention.
#[derive(Clone, Debug)]
pub struct TargetParentLawRow {
    /// Treatment category index.
    pub treatment_level: usize,
    /// Category index for each outcome parent other than treatment.
    pub parent_levels: Vec<usize>,
    /// Target mass of this parent stratum under the intervention.
    pub probability: f64,
}

/// Fixed-graph one-factor mechanism sensitivity contract.
#[derive(Clone, Debug)]
pub struct FixedGraphMechanismSensitivitySpec {
    /// Numeric outcome value corresponding to each probability column.
    pub outcome_values: Vec<f64>,
    /// Cardinalities of the outcome parents, in ascending variable order.
    pub parent_cardinalities: Vec<usize>,
    /// Control and active treatment category indices.
    pub treatment_levels: [usize; 2],
    /// Maximum outcome-kernel contamination fraction.
    pub max_fraction: f64,
    /// Optional response threshold for tipping analysis.
    pub decision_threshold: Option<f64>,
    /// Available source regime and snapshot for the kernel values.
    pub source_kernel_regime: RegimeId,
    /// Snapshot identity for the source kernel values.
    pub source_kernel_snapshot: String,
    /// Available target regime and snapshot for the parent-law values.
    pub target_parent_regime: RegimeId,
    /// Snapshot identity for the target parent-law values.
    pub target_parent_snapshot: String,
    /// Complete source response kernel over all outcome-parent strata.
    pub source_kernel: Vec<SourceOutcomeKernelRow>,
    /// Source distribution of the non-treatment outcome parents.
    pub source_parent_law: Vec<SourceParentLawRow>,
    /// Target parent distributions for both treatment intervention worlds.
    pub target_parent_law: Vec<TargetParentLawRow>,
}

/// Auditable fixed-graph mechanism sensitivity result.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedGraphMechanismSensitivityResult {
    /// Human-readable target estimand.
    pub estimand: String,
    /// Outcome and directed parents from the checked graph.
    pub outcome: VariableId,
    /// Directed outcome parents in their required stratum order.
    pub parents: Vec<VariableId>,
    /// Source and target population identities from the checked query.
    pub source_population: String,
    /// Target population identity from the checked query.
    pub target_population: String,
    /// Bound source kernel regime and snapshot.
    pub source_kernel_binding: (RegimeId, String),
    /// Bound target parent regime and snapshot.
    pub target_parent_binding: (RegimeId, String),
    /// Assumptions and support restrictions defining the result's scope.
    pub assumptions: Vec<String>,
    /// Baseline, exact assumption range, tipping point, and optimization witness.
    pub response: DiscreteKernelSensitivityResult,
}

/// Auditable one-factor sensitivity result for changing an isolated root
/// mechanism among the non-treatment parents of the outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedGraphRootMechanismSensitivityResult {
    /// Human-readable target estimand and perturbed factor.
    pub estimand: String,
    /// Outcome whose target response is evaluated.
    pub outcome: VariableId,
    /// Named root variable whose target marginal mechanism is perturbed.
    pub mechanism: VariableId,
    /// Source and target population identities from the checked query.
    pub source_population: String,
    /// Target population identity from the checked query.
    pub target_population: String,
    /// Snapshot binding used for the source outcome kernel.
    pub source_kernel_binding: (RegimeId, String),
    /// Snapshot binding used for the target parent law.
    pub target_parent_binding: (RegimeId, String),
    /// Response at the unmodified source root mechanism.
    pub baseline: f64,
    /// Minimum response over the declared contamination domain.
    pub minimum: f64,
    /// Maximum response over the declared contamination domain.
    pub maximum: f64,
    /// Smallest contamination fraction at which the range reaches the threshold.
    pub tipping_fraction: Option<f64>,
    /// Declared contamination fraction interval.
    pub fraction_domain: [f64; 2],
    /// Root-mechanism level selected at each extremum.
    pub minimizing_mechanism_level: usize,
    /// Root-mechanism level selected for the maximum response.
    pub maximizing_mechanism_level: usize,
    /// Assumptions required for the named factor perturbation to remain coherent.
    pub assumptions: Vec<String>,
}

/// Auditable one-factor sensitivity result for a checked categorical
/// conditional mechanism among the outcome's non-treatment parents.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedGraphConditionalMechanismSensitivityResult {
    /// Human-readable target estimand and perturbed factor.
    pub estimand: String,
    /// Outcome whose target response is evaluated.
    pub outcome: VariableId,
    /// Named conditional mechanism changed in the target population.
    pub mechanism: VariableId,
    /// Source and target population identities from the checked query.
    pub source_population: String,
    /// Target population identity from the checked query.
    pub target_population: String,
    /// Snapshot binding used for the source outcome kernel.
    pub source_kernel_binding: (RegimeId, String),
    /// Snapshot binding used for the target parent law.
    pub target_parent_binding: (RegimeId, String),
    /// Response at the unmodified source mechanism.
    pub baseline: f64,
    /// Minimum response over the declared contamination domain.
    pub minimum: f64,
    /// Maximum response over the declared contamination domain.
    pub maximum: f64,
    /// Smallest contamination fraction at which the range reaches the threshold.
    pub tipping_fraction: Option<f64>,
    /// Declared contamination fraction interval.
    pub fraction_domain: [f64; 2],
    /// Mechanism category selected at each parent stratum for the extrema.
    pub minimizing_mechanism_level_by_parent_stratum: Vec<usize>,
    /// Mechanism category selected at each parent stratum for the maximum.
    pub maximizing_mechanism_level_by_parent_stratum: Vec<usize>,
    /// Assumptions and factorization checks defining the result's scope.
    pub assumptions: Vec<String>,
}

#[path = "mechanism_sensitivity_z.rs"]
mod z;
pub(crate) use z::{ScanStop, ZSurrogateKernels, z_surrogate_kernels_metered};
pub use z::{
    ZTransportMechanismSensitivityResult, ZTransportSensitivityError,
    z_transport_mechanism_sensitivity,
};

#[path = "mechanism_sensitivity_fixed.rs"]
mod fixed;
pub use fixed::{
    FixedGraphSensitivityError, fixed_graph_conditional_mechanism_sensitivity,
    fixed_graph_mechanism_sensitivity, fixed_graph_root_mechanism_sensitivity,
    fixed_graph_treatment_level_sensitivity,
};

/// The fraction-independent part of a [`DiscreteKernelSensitivity`]
/// evaluation: the baseline and the two extremal slopes with their vertex
/// witnesses, read once from the kernel so that the response can be evaluated
/// at any fraction (a bisection step of the joint route) without re-reading
/// or copying the kernel. [`DiscreteKernelSensitivity::evaluate`] is exactly
/// [`Self::at`] on these slopes at the declared fraction.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DiscreteKernelSlopes {
    baseline: f64,
    low_slope: f64,
    high_slope: f64,
    low_witness: Vec<usize>,
    high_witness: Vec<usize>,
}

fn check_fraction(max_fraction: f64) -> Result<(), DiscreteKernelSensitivityError> {
    if !max_fraction.is_finite() || !(0.0..=1.0).contains(&max_fraction) {
        return Err(DiscreteKernelSensitivityError::InvalidFraction);
    }
    Ok(())
}

impl DiscreteKernelSlopes {
    /// Response under the unmodified source kernel.
    pub(crate) const fn baseline(&self) -> f64 {
        self.baseline
    }

    /// Exact `(minimum, maximum)` at `max_fraction` (assumed in `[0, 1]`),
    /// the endpoints of [`Self::at`] without its witnesses or allocation.
    pub(crate) fn extrema(&self, max_fraction: f64) -> (f64, f64) {
        let end_low = self.baseline + max_fraction * self.low_slope;
        let end_high = self.baseline + max_fraction * self.high_slope;
        (self.baseline.min(end_low), self.baseline.max(end_high))
    }

    /// The exact range at `max_fraction` with its witnesses and, given a
    /// threshold, the analytic tipping fraction.
    pub(crate) fn at(
        &self,
        max_fraction: f64,
        decision_threshold: Option<f64>,
    ) -> Result<DiscreteKernelSensitivityResult, DiscreteKernelSensitivityError> {
        check_fraction(max_fraction)?;
        if decision_threshold.is_some_and(|x| !x.is_finite()) {
            return Err(DiscreteKernelSensitivityError::InvalidContrast);
        }
        let baseline = self.baseline;
        let (minimum, maximum) = self.extrema(max_fraction);
        let tipping_fraction = decision_threshold.and_then(|threshold| {
            if !(minimum..=maximum).contains(&threshold) {
                return None;
            }
            if exact_threshold_matches_baseline(threshold, baseline) {
                return Some(0.0);
            }
            let slope = if threshold < baseline { self.low_slope } else { self.high_slope };
            crate::tipping::slope_fraction(baseline, threshold, slope).and_then(|fraction| {
                crate::tipping::fraction_in_closed_domain(fraction, max_fraction)
            })
        });

        Ok(DiscreteKernelSensitivityResult {
            baseline,
            minimum,
            maximum,
            tipping_fraction,
            interval_interpretation: "assumption range under declared discrete outcome-kernel contamination; not a sampling interval",
            receipt: DiscreteKernelOptimizationReceipt {
                minimizing_outcome_by_stratum: self.low_witness.clone(),
                maximizing_outcome_by_stratum: self.high_witness.clone(),
                fraction_domain: [0.0, max_fraction],
                method: "linear objective over product of outcome simplexes; exact simplex-vertex extrema",
            },
        })
    }
}

impl DiscreteKernelSensitivity {
    /// Evaluate exact extrema over all replacement distributions in each stratum.
    ///
    /// Since the response is linear in each `R(.|s)`, an extremum occurs at a
    /// simplex vertex. The method therefore checks outcome categories exactly;
    /// it does not discretize the contamination fraction.
    pub fn evaluate(
        &self,
    ) -> Result<DiscreteKernelSensitivityResult, DiscreteKernelSensitivityError> {
        self.check_dimensions()?;
        check_fraction(self.max_fraction)?;
        self.slopes()?.at(self.max_fraction, self.decision_threshold)
    }

    fn check_dimensions(&self) -> Result<(), DiscreteKernelSensitivityError> {
        let strata = self.source_kernel.len();
        let categories = self.outcome_values.len();
        if strata == 0
            || categories == 0
            || self.stratum_contrast_weights.len() != strata
            || self.source_kernel.iter().any(|row| row.len() != categories)
        {
            return Err(DiscreteKernelSensitivityError::InvalidDimensions);
        }
        Ok(())
    }

    /// Validate the kernel and contrast and read the baseline, the extremal
    /// slopes and their vertex witnesses once; `max_fraction` is checked by
    /// [`DiscreteKernelSlopes::at`].
    pub(crate) fn slopes(&self) -> Result<DiscreteKernelSlopes, DiscreteKernelSensitivityError> {
        self.check_dimensions()?;
        let strata = self.source_kernel.len();
        if self.outcome_values.iter().any(|x| !x.is_finite())
            || self.stratum_contrast_weights.iter().any(|x| !x.is_finite())
            || self.decision_threshold.is_some_and(|x| !x.is_finite())
        {
            return Err(DiscreteKernelSensitivityError::InvalidContrast);
        }
        for row in &self.source_kernel {
            if row.iter().any(|p| !p.is_finite() || *p < 0.0)
                || (row.iter().sum::<f64>() - 1.0).abs() > 1e-10
            {
                return Err(DiscreteKernelSensitivityError::InvalidKernel);
            }
        }

        let baseline = self
            .source_kernel
            .iter()
            .zip(&self.stratum_contrast_weights)
            .map(|(row, weight)| {
                weight * row.iter().zip(&self.outcome_values).map(|(p, y)| p * y).sum::<f64>()
            })
            .sum::<f64>();

        // The same R is used in both contrast arms. `stratum_contrast_weights`
        // already contains their signed target-mass difference, so choosing
        // one outcome per stratum is the exact shared-kernel optimization.
        let mut low_slope = 0.0;
        let mut high_slope = 0.0;
        let mut low_witness = Vec::with_capacity(strata);
        let mut high_witness = Vec::with_capacity(strata);
        for (row, weight) in self.source_kernel.iter().zip(&self.stratum_contrast_weights) {
            let source_mean = row.iter().zip(&self.outcome_values).map(|(p, y)| p * y).sum::<f64>();
            let (low_i, low_y) = self
                .outcome_values
                .iter()
                .copied()
                .enumerate()
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .expect("nonempty categories");
            let (high_i, high_y) = self
                .outcome_values
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .expect("nonempty categories");
            if *weight >= 0.0 {
                low_slope += weight * (low_y - source_mean);
                high_slope += weight * (high_y - source_mean);
                low_witness.push(low_i);
                high_witness.push(high_i);
            } else {
                low_slope += weight * (high_y - source_mean);
                high_slope += weight * (low_y - source_mean);
                low_witness.push(high_i);
                high_witness.push(low_i);
            }
        }
        Ok(DiscreteKernelSlopes { baseline, low_slope, high_slope, low_witness, high_witness })
    }
}

// A zero tipping fraction is exact only when the declared threshold is exactly
// the computed baseline; applying a tolerance would change the reported quantity.
#[allow(
    clippy::float_cmp,
    reason = "a threshold equal to the baseline is an exact user-supplied coincidence"
)]
fn exact_threshold_matches_baseline(threshold: f64, baseline: f64) -> bool {
    threshold == baseline
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
        EvidenceRegime, RegimeBinding, RegimeKind, SamplingDesign, TargetSampling,
        VariableCoordinate, VariableDomain,
    };
    use antecedent_graph::{Admg, DenseNodeId};
    use antecedent_identify::{
        CatalogTransportResult, SidLimits, ZTransportQuery, bind_z_transport_catalog,
        identify_catalog_transport, identify_z_transport,
    };
    use std::sync::Arc;

    fn fixture(max_fraction: f64, threshold: Option<f64>) -> DiscreteKernelSensitivity {
        // Two arms have stratum masses (.8,.2) and (.2,.8), producing
        // signed contrast weights (.6,-.6). The shared contaminant is optimized
        // once per stratum, not independently per arm.
        DiscreteKernelSensitivity {
            source_kernel: vec![vec![0.75, 0.25], vec![0.25, 0.75]],
            outcome_values: vec![0.0, 1.0],
            stratum_contrast_weights: vec![0.6, -0.6],
            max_fraction,
            decision_threshold: threshold,
        }
    }

    // The zero-contamination contract requires the extrema to equal baseline
    // exactly; a tolerance could hide an unintended perturbation.
    fn assert_exact_float_eq(actual: f64, expected: f64) {
        assert_eq!(actual, expected);
    }

    #[test]
    fn zero_fraction_returns_baseline_and_threshold_tips_exactly() {
        let result = fixture(0.0, Some(0.3)).evaluate().unwrap();
        assert!((result.baseline + 0.3).abs() < 1e-12);
        assert_exact_float_eq(result.minimum, result.baseline);
        assert_exact_float_eq(result.maximum, result.baseline);
        assert_eq!(result.tipping_fraction, None);
        assert!(result.interval_interpretation.contains("not a sampling interval"));
    }

    #[test]
    fn range_is_nested_and_uses_shared_kernel_witnesses() {
        let narrow = fixture(0.2, Some(-0.4)).evaluate().unwrap();
        let broad = fixture(0.6, Some(-0.4)).evaluate().unwrap();
        assert!(broad.minimum <= narrow.minimum);
        assert!(broad.maximum >= narrow.maximum);
        assert_eq!(broad.receipt.minimizing_outcome_by_stratum, vec![0, 1]);
        assert_eq!(broad.receipt.maximizing_outcome_by_stratum, vec![1, 0]);
        assert!(broad.tipping_fraction.is_some_and(|d| d <= 0.6));
    }

    #[test]
    // Keep construction, provider binding, and changed-input checks together
    // because they jointly verify this single staged z-route contract.
    #[allow(
        clippy::too_many_lines,
        reason = "one fixture exercised end to end; later assertions read locals built above them"
    )]
    fn checked_z_formula_sensitivity_binds_its_provider_and_delta_domain() {
        use antecedent_expr::{
            DiscreteAxis, ExactDiscreteLaw, InterventionAssignment, LawTolerance,
        };
        let (w, z, x, y) = (
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            VariableId::from_raw(2),
            VariableId::from_raw(3),
        );
        let mut graph = Admg::with_variables(4);
        for (from, to) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
            graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        for (a, b) in [(0, 3), (1, 3), (1, 2)] {
            graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
        let query = ZTransportQuery {
            outcomes: Arc::from([y]),
            treatments: Arc::from([x]),
            controllable: Arc::from([z]),
            experiment_assignment: Arc::from([antecedent_core::InterventionAssignment {
                variable: z,
                value: antecedent_core::Value::Bool(false),
            }]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        let antecedent_identify::ZTransportResult::Identified(proof) = identify_z_transport(
            &diagram,
            &query,
            SidLimits::default(),
            &antecedent_core::ExecutionContext::for_tests(0),
        )
        .unwrap() else {
            panic!("expected checked z formula")
        };
        let variables = [w, z, x, y];
        let env = antecedent_core::Environment::try_new(
            "source",
            variables.map(|variable| VariableCoordinate {
                variable,
                domain: VariableDomain::Binary,
                unit: None,
            }),
            Arc::<[VariableId]>::from([]),
        )
        .unwrap();
        let measured: Arc<[VariableId]> = Arc::from(variables);
        let regimes = [false, true].map(|level| {
            antecedent_core::EvidenceRegime::try_new(
                RegimeId::from_raw(u32::from(level)),
                RegimeKind::Experimental,
                EvidenceKind::Available,
                [z],
                [antecedent_core::InterventionAssignment {
                    variable: z,
                    value: antecedent_core::Value::Bool(level),
                }],
                Arc::clone(&measured),
                "source",
                antecedent_core::DistributionAvailability::Joint,
            )
            .unwrap()
        });
        let bindings = [false, true].map(|level| RegimeBinding {
            dataset_identity: None,
            regime: RegimeId::from_raw(u32::from(level)),
            snapshot_identity: Arc::from(if level { "z-true" } else { "source-snapshot" }),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        });
        let catalog = EvidenceCatalog::try_new([env], regimes, bindings, None).unwrap();
        let functional = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
        let mut probabilities = Vec::with_capacity(8);
        for row in 0..8 {
            let xl = (row / 2) % 2;
            let yl = row % 2;
            probabilities.push(0.25 * if yl == xl { 0.7 } else { 0.3 });
        }
        let law = ExactDiscreteLaw::try_new(
            "source",
            RegimeId::from_raw(0),
            [InterventionAssignment::concrete(z, antecedent_core::Value::Bool(false))],
            [w, x, y].map(|variable| DiscreteAxis {
                variable,
                values: Arc::from([
                    antecedent_core::Value::Bool(false),
                    antecedent_core::Value::Bool(true),
                ]),
            }),
            probabilities,
            "source-snapshot",
            LawTolerance::default(),
        )
        .unwrap();
        let data = ExactTransportData::try_new([law], 32).unwrap();
        let baseline = z_transport_mechanism_sensitivity(
            &diagram,
            &functional,
            &data,
            0.0,
            None,
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert!((baseline.response.baseline - 0.4).abs() < 1e-12);
        assert_exact_float_eq(baseline.response.minimum, baseline.response.baseline);
        assert_exact_float_eq(baseline.response.maximum, baseline.response.baseline);
        assert_eq!(baseline.provider_snapshot, "source-snapshot");
        let tipping = z_transport_mechanism_sensitivity(
            &diagram,
            &functional,
            &data,
            0.0,
            Some(baseline.response.baseline),
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert_eq!(tipping.response.tipping_fraction, Some(0.0));
    }

    #[test]
    fn rejects_non_distributions_and_invalid_fraction() {
        let mut invalid = fixture(0.2, None);
        invalid.source_kernel[0] = vec![0.2, 0.2];
        assert_eq!(invalid.evaluate().unwrap_err(), DiscreteKernelSensitivityError::InvalidKernel);
        let invalid = fixture(1.2, None);
        assert_eq!(
            invalid.evaluate().unwrap_err(),
            DiscreteKernelSensitivityError::InvalidFraction
        );
    }

    fn checked_transport_fixture()
    -> (SelectionDiagram, ClassicalTransportQuery, antecedent_identify::BoundTransportFunctional)
    {
        let u = VariableId::from_raw(0);
        let x = VariableId::from_raw(1);
        let y = VariableId::from_raw(2);
        let coordinate =
            |variable| VariableCoordinate { variable, domain: VariableDomain::Binary, unit: None };
        let source =
            Environment::try_new("source", [coordinate(u), coordinate(x), coordinate(y)], [x])
                .unwrap();
        let target = Environment::try_new("target", [coordinate(u)], []).unwrap();
        let source_regime = EvidenceRegime::try_new(
            RegimeId::from_raw(1),
            RegimeKind::Experimental,
            EvidenceKind::Available,
            [x],
            [],
            [u, y],
            "source",
            DistributionAvailability::Joint,
        )
        .unwrap();
        let target_regime = EvidenceRegime::try_new(
            RegimeId::from_raw(2),
            RegimeKind::Observational,
            EvidenceKind::Available,
            [],
            [],
            [u],
            "target",
            DistributionAvailability::Joint,
        )
        .unwrap();
        let binding = |regime, snapshot: &str| RegimeBinding {
            dataset_identity: None,
            regime,
            snapshot_identity: Arc::from(snapshot),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::UnknownDependence,
        };
        let catalog = EvidenceCatalog::try_new(
            [source, target],
            [source_regime, target_regime],
            [
                binding(RegimeId::from_raw(1), "source-snapshot"),
                binding(RegimeId::from_raw(2), "target-snapshot"),
            ],
            Some(TargetSampling::RepresentativeSample),
        )
        .unwrap();
        let mut graph = Admg::with_variables(3);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let diagram = SelectionDiagram::try_new(graph, [x]).unwrap();
        let query = ClassicalTransportQuery {
            outcomes: Arc::from([y]),
            treatments: Arc::from([x]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        let ctx = antecedent_core::ExecutionContext::for_tests(1);
        let CatalogTransportResult::Identified(functional) =
            identify_catalog_transport(&diagram, &query, &catalog, SidLimits::default(), &ctx)
                .unwrap()
        else {
            panic!("source intervention should identify the fixed-DAG response");
        };
        (diagram, query, *functional)
    }

    fn checked_transport_fixture_with_dependent_outcome_parent()
    -> (SelectionDiagram, ClassicalTransportQuery, antecedent_identify::BoundTransportFunctional)
    {
        let u = VariableId::from_raw(0);
        let x = VariableId::from_raw(1);
        let v = VariableId::from_raw(2);
        let y = VariableId::from_raw(3);
        let coordinate =
            |variable| VariableCoordinate { variable, domain: VariableDomain::Binary, unit: None };
        let source = Environment::try_new(
            "source",
            [coordinate(u), coordinate(x), coordinate(v), coordinate(y)],
            [x],
        )
        .unwrap();
        let target = Environment::try_new("target", [coordinate(u), coordinate(v)], []).unwrap();
        let source_regime = EvidenceRegime::try_new(
            RegimeId::from_raw(1),
            RegimeKind::Experimental,
            EvidenceKind::Available,
            [x],
            [],
            [u, v, y],
            "source",
            DistributionAvailability::Joint,
        )
        .unwrap();
        let target_regime = EvidenceRegime::try_new(
            RegimeId::from_raw(2),
            RegimeKind::Observational,
            EvidenceKind::Available,
            [],
            [],
            [u, v],
            "target",
            DistributionAvailability::Joint,
        )
        .unwrap();
        let binding = |regime, snapshot: &str| RegimeBinding {
            dataset_identity: None,
            regime,
            snapshot_identity: Arc::from(snapshot),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::UnknownDependence,
        };
        let catalog = EvidenceCatalog::try_new(
            [source, target],
            [source_regime, target_regime],
            [
                binding(RegimeId::from_raw(1), "source-snapshot"),
                binding(RegimeId::from_raw(2), "target-snapshot"),
            ],
            Some(TargetSampling::RepresentativeSample),
        )
        .unwrap();
        let mut graph = Admg::with_variables(4);
        for (from, to) in [(0, 1), (0, 2), (2, 3), (1, 3), (0, 3)] {
            graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        let diagram = SelectionDiagram::try_new(graph, [x]).unwrap();
        let query = ClassicalTransportQuery {
            outcomes: Arc::from([y]),
            treatments: Arc::from([x]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        let ctx = antecedent_core::ExecutionContext::for_tests(1);
        let CatalogTransportResult::Identified(functional) =
            identify_catalog_transport(&diagram, &query, &catalog, SidLimits::default(), &ctx)
                .unwrap()
        else {
            panic!("source intervention should identify the fixed-DAG response");
        };
        (diagram, query, *functional)
    }

    #[test]
    #[allow(clippy::too_many_lines, reason = "one cohesive known-truth stratum-weight assertion")]
    fn fixed_graph_route_derives_stratum_weights_and_matches_known_truth() {
        let (diagram, query, functional) = checked_transport_fixture();
        let mut spec = FixedGraphMechanismSensitivitySpec {
            outcome_values: vec![0.0, 1.0],
            parent_cardinalities: vec![2, 2],
            treatment_levels: [0, 1],
            max_fraction: 0.2,
            decision_threshold: Some(0.3),
            source_kernel_regime: RegimeId::from_raw(1),
            source_kernel_snapshot: "source-snapshot".into(),
            target_parent_regime: RegimeId::from_raw(2),
            target_parent_snapshot: "target-snapshot".into(),
            source_kernel: vec![
                SourceOutcomeKernelRow {
                    parent_levels: vec![0, 0],
                    outcome_probabilities: vec![0.8, 0.2],
                },
                SourceOutcomeKernelRow {
                    parent_levels: vec![0, 1],
                    outcome_probabilities: vec![0.2, 0.8],
                },
                SourceOutcomeKernelRow {
                    parent_levels: vec![1, 0],
                    outcome_probabilities: vec![0.7, 0.3],
                },
                SourceOutcomeKernelRow {
                    parent_levels: vec![1, 1],
                    outcome_probabilities: vec![0.3, 0.7],
                },
            ],
            source_parent_law: vec![
                SourceParentLawRow { parent_levels: vec![0], probability: 0.75 },
                SourceParentLawRow { parent_levels: vec![1], probability: 0.25 },
            ],
            target_parent_law: vec![
                TargetParentLawRow {
                    treatment_level: 0,
                    parent_levels: vec![0],
                    probability: 0.75,
                },
                TargetParentLawRow {
                    treatment_level: 0,
                    parent_levels: vec![1],
                    probability: 0.25,
                },
                TargetParentLawRow {
                    treatment_level: 1,
                    parent_levels: vec![0],
                    probability: 0.75,
                },
                TargetParentLawRow {
                    treatment_level: 1,
                    parent_levels: vec![1],
                    probability: 0.25,
                },
            ],
        };
        let result = fixed_graph_mechanism_sensitivity(
            &diagram,
            &query,
            &functional,
            &spec,
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert_eq!(result.parents, [VariableId::from_raw(0), VariableId::from_raw(1)]);
        assert!((result.response.baseline - 0.55).abs() < 1e-12);
        assert!((result.response.minimum - 0.24).abs() < 1e-12);
        assert!((result.response.maximum - 0.64).abs() < 1e-12);
        assert!((result.response.tipping_fraction.unwrap() - (0.25 / 1.55)).abs() < 1e-12);

        // A one-factor intervention on the active treatment slice leaves the
        // control mechanism exactly fixed while preserving the full contrast.
        spec.decision_threshold = Some(0.5);
        let active_only = fixed_graph_treatment_level_sensitivity(
            &diagram,
            &query,
            &functional,
            &spec,
            1,
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert!((active_only.response.baseline - 0.55).abs() < 1e-12);
        assert!((active_only.response.minimum - 0.395).abs() < 1e-12);
        assert!((active_only.response.maximum - 0.595).abs() < 1e-12);
        assert!((active_only.response.tipping_fraction.unwrap() - (0.05 / 0.775)).abs() < 1e-12);

        // The root U mechanism is available as a distinct one-factor family:
        // replace its target marginal while preserving the Y|X,U kernel.
        spec.decision_threshold = Some(0.53);
        let root_mechanism = fixed_graph_root_mechanism_sensitivity(
            &diagram,
            &query,
            &functional,
            &spec,
            VariableId::from_raw(0),
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert!((root_mechanism.baseline - 0.55).abs() < 1e-12);
        assert!((root_mechanism.minimum - 0.52).abs() < 1e-12);
        assert!((root_mechanism.maximum - 0.56).abs() < 1e-12);
        assert!((root_mechanism.tipping_fraction.unwrap() - (0.02 / 0.15)).abs() < 1e-12);
        assert_eq!(root_mechanism.minimizing_mechanism_level, 1);
        assert_eq!(root_mechanism.maximizing_mechanism_level, 0);
        let mut zero_fraction_spec = spec.clone();
        zero_fraction_spec.max_fraction = 0.0;
        let root_zero = fixed_graph_root_mechanism_sensitivity(
            &diagram,
            &query,
            &functional,
            &zero_fraction_spec,
            VariableId::from_raw(0),
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert!((root_zero.minimum - root_zero.baseline).abs() < 1e-12);
        assert!((root_zero.maximum - root_zero.baseline).abs() < 1e-12);
        assert_eq!(
            fixed_graph_root_mechanism_sensitivity(
                &diagram,
                &query,
                &functional,
                &spec,
                VariableId::from_raw(1),
                &antecedent_core::ExecutionContext::for_tests(1),
            )
            .unwrap_err(),
            FixedGraphSensitivityError::InvalidQuery
        );

        spec.max_fraction = 0.0;
        let zero = fixed_graph_treatment_level_sensitivity(
            &diagram,
            &query,
            &functional,
            &spec,
            1,
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert!((zero.response.minimum - result.response.baseline).abs() < 1e-12);
        assert!((zero.response.maximum - result.response.baseline).abs() < 1e-12);

        assert_eq!(
            fixed_graph_treatment_level_sensitivity(
                &diagram,
                &query,
                &functional,
                &spec,
                2,
                &antecedent_core::ExecutionContext::for_tests(1),
            )
            .unwrap_err(),
            FixedGraphSensitivityError::InvalidQuery
        );
    }

    #[test]
    fn fixed_graph_route_refuses_changed_proof_and_incomplete_parent_law() {
        let (diagram, query, functional) = checked_transport_fixture();
        let mut spec = FixedGraphMechanismSensitivitySpec {
            outcome_values: vec![0.0, 1.0],
            parent_cardinalities: vec![2, 2],
            treatment_levels: [0, 1],
            max_fraction: 0.2,
            decision_threshold: None,
            source_kernel_regime: RegimeId::from_raw(1),
            source_kernel_snapshot: "source-snapshot".into(),
            target_parent_regime: RegimeId::from_raw(2),
            target_parent_snapshot: "target-snapshot".into(),
            source_kernel: vec![
                SourceOutcomeKernelRow {
                    parent_levels: vec![0, 0],
                    outcome_probabilities: vec![0.8, 0.2],
                },
                SourceOutcomeKernelRow {
                    parent_levels: vec![0, 1],
                    outcome_probabilities: vec![0.2, 0.8],
                },
                SourceOutcomeKernelRow {
                    parent_levels: vec![1, 0],
                    outcome_probabilities: vec![0.7, 0.3],
                },
                SourceOutcomeKernelRow {
                    parent_levels: vec![1, 1],
                    outcome_probabilities: vec![0.3, 0.7],
                },
            ],
            source_parent_law: vec![
                SourceParentLawRow { parent_levels: vec![0], probability: 0.75 },
                SourceParentLawRow { parent_levels: vec![1], probability: 0.25 },
            ],
            target_parent_law: vec![
                TargetParentLawRow {
                    treatment_level: 0,
                    parent_levels: vec![0],
                    probability: 0.75,
                },
                TargetParentLawRow {
                    treatment_level: 0,
                    parent_levels: vec![1],
                    probability: 0.25,
                },
                TargetParentLawRow {
                    treatment_level: 1,
                    parent_levels: vec![0],
                    probability: 0.75,
                },
                TargetParentLawRow {
                    treatment_level: 1,
                    parent_levels: vec![1],
                    probability: 0.25,
                },
            ],
        };
        spec.source_kernel_snapshot = "stale-snapshot".into();
        assert_eq!(
            fixed_graph_mechanism_sensitivity(
                &diagram,
                &query,
                &functional,
                &spec,
                &antecedent_core::ExecutionContext::for_tests(1),
            )
            .unwrap_err(),
            FixedGraphSensitivityError::SnapshotMismatch
        );
        spec.source_kernel_snapshot = "source-snapshot".into();
        spec.target_parent_law.clear();
        assert!(matches!(
            fixed_graph_mechanism_sensitivity(
                &diagram,
                &query,
                &functional,
                &spec,
                &antecedent_core::ExecutionContext::for_tests(1),
            ),
            Err(FixedGraphSensitivityError::InvalidParentLaw)
        ));
    }

    #[test]
    fn root_mechanism_route_refuses_dependency_on_another_outcome_parent() {
        let (diagram, query, functional) =
            checked_transport_fixture_with_dependent_outcome_parent();
        let mut source_kernel = Vec::new();
        for u in 0..2 {
            for x in 0..2 {
                for v in 0..2 {
                    let probability_one = if x == 1 { 0.7 } else { 0.3 };
                    source_kernel.push(SourceOutcomeKernelRow {
                        parent_levels: vec![u, x, v],
                        outcome_probabilities: vec![1.0 - probability_one, probability_one],
                    });
                }
            }
        }
        let source_parent_law = (0..2)
            .flat_map(|u| {
                (0..2).map(move |v| SourceParentLawRow {
                    parent_levels: vec![u, v],
                    probability: 0.25,
                })
            })
            .collect();
        let target_parent_law = (0..2)
            .flat_map(|x| {
                (0..2).flat_map(move |u| {
                    (0..2).map(move |v| TargetParentLawRow {
                        treatment_level: x,
                        parent_levels: vec![u, v],
                        probability: 0.25,
                    })
                })
            })
            .collect();
        let spec = FixedGraphMechanismSensitivitySpec {
            outcome_values: vec![0.0, 1.0],
            parent_cardinalities: vec![2, 2, 2],
            treatment_levels: [0, 1],
            max_fraction: 0.2,
            decision_threshold: None,
            source_kernel_regime: RegimeId::from_raw(1),
            source_kernel_snapshot: "source-snapshot".into(),
            target_parent_regime: RegimeId::from_raw(2),
            target_parent_snapshot: "target-snapshot".into(),
            source_kernel,
            source_parent_law,
            target_parent_law,
        };
        assert_eq!(
            fixed_graph_root_mechanism_sensitivity(
                &diagram,
                &query,
                &functional,
                &spec,
                VariableId::from_raw(0),
                &antecedent_core::ExecutionContext::for_tests(1),
            )
            .unwrap_err(),
            FixedGraphSensitivityError::UnsupportedGraph
        );
    }

    #[test]
    fn conditional_mechanism_route_propagates_a_nonroot_factor_exactly() {
        let (diagram, query, functional) =
            checked_transport_fixture_with_dependent_outcome_parent();
        let mut source_kernel = Vec::new();
        for u in 0..2 {
            for x in 0..2 {
                for v in 0..2 {
                    let probability_one = match (u, x, v) {
                        (0, 0, 0) => 0.1,
                        (0, 0, 1) => 0.2,
                        (0, 1, 0) => 0.3,
                        (0, 1, 1) => 0.6,
                        (1, 0, 0) => 0.15,
                        (1, 0, 1) => 0.25,
                        (1, 1, 0) => 0.4,
                        (1, 1, 1) => 0.7,
                        _ => unreachable!(),
                    };
                    source_kernel.push(SourceOutcomeKernelRow {
                        parent_levels: vec![u, x, v],
                        outcome_probabilities: vec![1.0 - probability_one, probability_one],
                    });
                }
            }
        }
        let source_parent_law = vec![
            SourceParentLawRow { parent_levels: vec![0, 0], probability: 0.6 },
            SourceParentLawRow { parent_levels: vec![0, 1], probability: 0.15 },
            SourceParentLawRow { parent_levels: vec![1, 0], probability: 0.05 },
            SourceParentLawRow { parent_levels: vec![1, 1], probability: 0.2 },
        ];
        let target_parent_law = (0..2)
            .flat_map(|x| {
                source_parent_law.iter().map(move |row| TargetParentLawRow {
                    treatment_level: x,
                    parent_levels: row.parent_levels.clone(),
                    probability: row.probability,
                })
            })
            .collect();
        let spec = FixedGraphMechanismSensitivitySpec {
            outcome_values: vec![0.0, 1.0],
            parent_cardinalities: vec![2, 2, 2],
            treatment_levels: [0, 1],
            max_fraction: 0.2,
            decision_threshold: Some(0.27),
            source_kernel_regime: RegimeId::from_raw(1),
            source_kernel_snapshot: "source-snapshot".into(),
            target_parent_regime: RegimeId::from_raw(2),
            target_parent_snapshot: "target-snapshot".into(),
            source_kernel,
            source_parent_law,
            target_parent_law,
        };
        let result = fixed_graph_conditional_mechanism_sensitivity(
            &diagram,
            &query,
            &functional,
            &spec,
            VariableId::from_raw(2),
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert!((result.baseline - 0.2825).abs() < 1e-12);
        assert!((result.minimum - 0.2685).abs() < 1e-12);
        assert!((result.maximum - 0.3085).abs() < 1e-12);
        assert!((result.tipping_fraction.unwrap() - (0.0125 / 0.07)).abs() < 1e-12);
        assert_eq!(result.minimizing_mechanism_level_by_parent_stratum, [0, 0]);
        assert_eq!(result.maximizing_mechanism_level_by_parent_stratum, [1, 1]);
    }

    // --- Independent-path reconstruction of the "synthetic" tipping claim. ---
    //
    // The tests above pin the reported tipping fraction and exact range to
    // closed-form arithmetic constants (`0.25 / 1.55`, `0.0125 / 0.07`, ...).
    // That checks the route against its own linear-slope algebra but never
    // against an independently contaminated target effect. The reconstructions
    // below build a genuine contaminated mechanism `K' = (1 - delta) * K +
    // delta * R` for a known replacement `R`, recompute the TRUE contaminated
    // response by pushing `K'` through the honest functional (never through the
    // route's own min/max/tipping formula), and check that (a) `delta = 0`
    // reproduces the baseline, (b) the recomputed effect stays inside the
    // reported `[minimum, maximum]` for every `delta` in the declared domain,
    // and (c) the recomputed effect crosses the decision threshold at exactly
    // the `delta` the route reports as its tipping fraction.

    /// Independent honest functional: the signed stratum contrast of the mean
    /// response under an explicit outcome kernel. This deliberately does NOT use
    /// `low_slope`/`high_slope`; it evaluates the supplied kernel directly.
    fn contrast_response(kernel: &[Vec<f64>], weights: &[f64], outcome_values: &[f64]) -> f64 {
        kernel
            .iter()
            .zip(weights)
            .map(|(row, weight)| {
                weight * row.iter().zip(outcome_values).map(|(p, y)| p * y).sum::<f64>()
            })
            .sum()
    }

    /// `K' = (1 - delta) * K + delta * R`, row by row.
    fn contaminate(kernel: &[Vec<f64>], replacement: &[Vec<f64>], delta: f64) -> Vec<Vec<f64>> {
        kernel
            .iter()
            .zip(replacement)
            .map(|(row, r)| row.iter().zip(r).map(|(p, q)| (1.0 - delta) * p + delta * q).collect())
            .collect()
    }

    #[test]
    fn outcome_kernel_tipping_matches_a_reconstructed_contaminated_target() {
        let base = fixture(0.6, Some(-0.4));
        let reported = base.evaluate().unwrap();
        let tipping = reported.tipping_fraction.expect("threshold lies inside the reported range");
        let threshold = -0.4;

        // Known contaminant: the exact minimizing simplex vertex the route
        // reports, i.e. a point mass on the witnessed outcome per stratum. This
        // is the worst-case R that the reported (below-baseline) tipping assumes.
        let categories = base.outcome_values.len();
        let replacement: Vec<Vec<f64>> = reported
            .receipt
            .minimizing_outcome_by_stratum
            .iter()
            .map(|&winner| (0..categories).map(|c| f64::from(u8::from(c == winner))).collect())
            .collect();

        // (a) delta = 0 reproduces the sensitivity baseline exactly.
        let at_zero = contaminate(&base.source_kernel, &replacement, 0.0);
        let effect_at_zero =
            contrast_response(&at_zero, &base.stratum_contrast_weights, &base.outcome_values);
        assert!((effect_at_zero - reported.baseline).abs() < 1e-12);

        // (b) every delta in the declared domain lands inside the reported range.
        //     The honest hand functional is additionally cross-checked against
        //     the crate's own evaluation path applied to the contaminated kernel
        //     (evaluate baseline at zero contamination), which must agree.
        let grid = [0.0, 0.1, 0.2, 0.3, 1.0 / 3.0, 0.4, 0.5, 0.6];
        for &delta in &grid {
            let contaminated = contaminate(&base.source_kernel, &replacement, delta);
            let recomputed = contrast_response(
                &contaminated,
                &base.stratum_contrast_weights,
                &base.outcome_values,
            );
            let via_eval = DiscreteKernelSensitivity {
                source_kernel: contaminated,
                outcome_values: base.outcome_values.clone(),
                stratum_contrast_weights: base.stratum_contrast_weights.clone(),
                max_fraction: 0.0,
                decision_threshold: None,
            }
            .evaluate()
            .unwrap()
            .baseline;
            assert!((recomputed - via_eval).abs() < 1e-12);
            assert!(
                recomputed >= reported.minimum - 1e-12 && recomputed <= reported.maximum + 1e-12,
                "delta {delta}: recomputed {recomputed} escaped [{}, {}]",
                reported.minimum,
                reported.maximum
            );
        }

        // (c) the recomputed true effect crosses the threshold at exactly the
        //     reported tipping fraction: at that delta the effect equals the
        //     threshold, and it is on opposite sides just inside and outside it.
        let at_tipping = contaminate(&base.source_kernel, &replacement, tipping);
        let effect_at_tipping =
            contrast_response(&at_tipping, &base.stratum_contrast_weights, &base.outcome_values);
        assert!((effect_at_tipping - threshold).abs() < 1e-12);

        let before = contaminate(&base.source_kernel, &replacement, tipping - 1e-3);
        let after = contaminate(&base.source_kernel, &replacement, tipping + 1e-3);
        assert!(
            contrast_response(&before, &base.stratum_contrast_weights, &base.outcome_values)
                > threshold
        );
        assert!(
            contrast_response(&after, &base.stratum_contrast_weights, &base.outcome_values)
                < threshold
        );
    }

    #[test]
    fn conditional_mechanism_tipping_matches_a_reconstructed_contaminated_target() {
        let (diagram, query, functional) =
            checked_transport_fixture_with_dependent_outcome_parent();

        // Same checked source kernel as the arithmetic-constant test above.
        let mut source_kernel = Vec::new();
        for u in 0..2 {
            for x in 0..2 {
                for v in 0..2 {
                    let probability_one = match (u, x, v) {
                        (0, 0, 0) => 0.1,
                        (0, 0, 1) => 0.2,
                        (0, 1, 0) => 0.3,
                        (0, 1, 1) => 0.6,
                        (1, 0, 0) => 0.15,
                        (1, 0, 1) => 0.25,
                        (1, 1, 0) => 0.4,
                        (1, 1, 1) => 0.7,
                        _ => unreachable!(),
                    };
                    source_kernel.push(SourceOutcomeKernelRow {
                        parent_levels: vec![u, x, v],
                        outcome_probabilities: vec![1.0 - probability_one, probability_one],
                    });
                }
            }
        }

        // P(U) and the *uncontaminated* V|U mechanism the joint factorizes into.
        let p_u = [0.75, 0.25];
        let v_given_u = [[0.8, 0.2], [0.2, 0.8]];
        let max_fraction = 0.2;
        let threshold = 0.27;

        // Build the target/source joint parent law for a contaminated V|U
        // mechanism K'(V | U=u) = (1 - delta) * P(V | u) + delta * R_u, with R_u a
        // known point mass on V = 0 for both U contexts (the minimizing vertex
        // the route reports). The joint stays a valid DAG factorization, so
        // feeding it back through the route recomputes the honest response.
        let build_spec = |delta: f64| {
            let mut source_parent_law = Vec::new();
            for u in 0..2usize {
                for v in 0..2usize {
                    let r = f64::from(u8::from(v == 0));
                    let contaminated_v = (1.0 - delta) * v_given_u[u][v] + delta * r;
                    source_parent_law.push(SourceParentLawRow {
                        parent_levels: vec![u, v],
                        probability: p_u[u] * contaminated_v,
                    });
                }
            }
            let target_parent_law = (0..2)
                .flat_map(|x| {
                    source_parent_law.iter().map(move |row| TargetParentLawRow {
                        treatment_level: x,
                        parent_levels: row.parent_levels.clone(),
                        probability: row.probability,
                    })
                })
                .collect();
            FixedGraphMechanismSensitivitySpec {
                outcome_values: vec![0.0, 1.0],
                parent_cardinalities: vec![2, 2, 2],
                treatment_levels: [0, 1],
                max_fraction,
                decision_threshold: Some(threshold),
                source_kernel_regime: RegimeId::from_raw(1),
                source_kernel_snapshot: "source-snapshot".into(),
                target_parent_regime: RegimeId::from_raw(2),
                target_parent_snapshot: "target-snapshot".into(),
                source_kernel: source_kernel.clone(),
                source_parent_law,
                target_parent_law,
            }
        };

        // The route under test: reported range/tipping for the uncontaminated
        // (delta = 0) mechanism over the declared [0, max_fraction] domain.
        let reported = fixed_graph_conditional_mechanism_sensitivity(
            &diagram,
            &query,
            &functional,
            &build_spec(0.0),
            VariableId::from_raw(2),
            &antecedent_core::ExecutionContext::for_tests(1),
        )
        .unwrap();
        let tipping = reported.tipping_fraction.expect("threshold lies inside the reported range");

        // Independent recomputation of the true contaminated response: evaluate a
        // spec whose parent law already carries the contaminated K'; the route's
        // baseline is the honest functional of that joint and does not depend on
        // the reported min/max/tipping algebra.
        let true_effect = |delta: f64| {
            fixed_graph_conditional_mechanism_sensitivity(
                &diagram,
                &query,
                &functional,
                &build_spec(delta),
                VariableId::from_raw(2),
                &antecedent_core::ExecutionContext::for_tests(1),
            )
            .unwrap()
            .baseline
        };

        // (a) delta = 0 reproduces the reported baseline.
        assert!((true_effect(0.0) - reported.baseline).abs() < 1e-12);

        // (b) every delta in the declared domain lands inside the reported range.
        let grid = [0.0, 0.05, 0.1, tipping, 0.15, 0.2];
        for &delta in &grid {
            let effect = true_effect(delta);
            assert!(
                effect >= reported.minimum - 1e-12 && effect <= reported.maximum + 1e-12,
                "delta {delta}: reconstructed {effect} escaped [{}, {}]",
                reported.minimum,
                reported.maximum
            );
        }

        // (c) the reconstructed true effect crosses the threshold at exactly the
        //     reported tipping fraction.
        assert!((true_effect(tipping) - threshold).abs() < 1e-12);
        assert!(true_effect(tipping - 1e-3) > threshold);
        assert!(true_effect(tipping + 1e-3) < threshold);
    }
}
