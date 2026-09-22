//! Finite categorical frequency plug-in for transport joints.
//!
//! One fitted joint per `(population, regime, intervention world)`. Leaves from
//! the same sample project this joint; they are never fitted as independent studies.

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::{
    DependenceGroup, EvidenceCatalog, RegimeId, SamplingDesign, TargetSampling, Value,
    VariableDomain, VariableId,
};
use antecedent_data::{ColumnView, TableView, TabularData};
use antecedent_expr::{
    DiscreteAxis, ExactDiscreteLaw, ExactLawError, ExactTransportData, InterventionAssignment,
    LawTolerance,
};
use antecedent_identify::BoundTransportFunctional;

use crate::error::EstimationError;

/// Licensed default estimator id.
pub const EMPIRICAL_TABLE_PLUGIN: &str = "transport.empirical_table_plugin";

/// Named but unlicensed smoothing choice.
pub const EMPIRICAL_TABLE_DIRICHLET: &str = "transport.empirical_table_dirichlet";

/// Explicit statistical-provider choice. Defaults never hide empty cells.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EmpiricalTableEstimator {
    /// Frequencies from fully observed samples, no pseudocounts.
    Plugin,
    /// Named Dirichlet / pseudocount smoother. Not licensed in T6.1.
    Dirichlet,
    /// Model-based finite categorical joint; independently uncalibrated.
    Learned(crate::LearnerSpec),
}

impl EmpiricalTableEstimator {
    /// Stable provider identity including every learned hyperparameter.
    #[must_use]
    pub fn identity(self) -> String {
        match self {
            Self::Learned(spec) => format!("{}:{}", self.as_str(), spec.identity()),
            _ => self.as_str().into(),
        }
    }

    /// Stable estimator id.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Plugin => EMPIRICAL_TABLE_PLUGIN,
            Self::Dirichlet => EMPIRICAL_TABLE_DIRICHLET,
            Self::Learned(_) => "transport.learned_categorical_plugin",
        }
    }
}

/// Inference settings for the empirical-table path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmpiricalTableOptions {
    /// Estimator. Plugin and learned categorical are licensed; Dirichlet is not.
    pub estimator: EmpiricalTableEstimator,
    /// Outer bootstrap replicates. Zero withholds the interval.
    pub bootstrap_replicates: u32,
    /// Nominal coverage of pointwise percentile intervals.
    pub coverage_level: f64,
    /// Refuse rather than densify a larger Cartesian product.
    pub max_joint_cells: usize,
}

impl Default for EmpiricalTableOptions {
    fn default() -> Self {
        Self {
            estimator: EmpiricalTableEstimator::Plugin,
            bootstrap_replicates: 199,
            coverage_level: 0.95,
            max_joint_cells: 1_000_000,
        }
    }
}

/// Identity of one independent sampled world, including its concrete values.
pub type SampleKey = (Arc<str>, Arc<str>, RegimeId, Vec<(u32, u64)>);

/// Canonical world key; numeric signed zero has one representation.
#[must_use]
pub fn sample_key(sample: &RegimeSample) -> SampleKey {
    let mut world: Vec<_> = sample
        .interventions
        .iter()
        .map(|a| {
            let value = a.value.as_f64().unwrap_or(f64::NAN);
            (a.variable.raw(), if value == 0.0 { 0 } else { value.to_bits() })
        })
        .collect();
    world.sort_unstable();
    (sample.population.clone(), sample.snapshot_identity.clone(), sample.regime, world)
}

/// Resampling key with explicit forwarded-dataset aliases resolved.
#[must_use]
pub fn bound_sample_key(catalog: &EvidenceCatalog, sample: &RegimeSample) -> SampleKey {
    let mut key = sample_key(sample);
    if let Some(identity) = catalog
        .bindings
        .iter()
        .find(|b| b.regime == sample.regime)
        .and_then(|b| b.dataset_identity.as_ref())
    {
        key.0 = Arc::from(format!("shared-dataset:{identity}"));
        key.1 = Arc::from("");
        key.2 = RegimeId::from_raw(0);
    }
    key
}
/// Check that explicitly forwarded datasets agree before fitting or resampling.
/// # Errors
/// An alias names different rows or an incompatible sampling/dependence contract.
pub fn validate_dataset_aliases(
    catalog: &EvidenceCatalog,
    samples: &[RegimeSample],
) -> Result<(), EstimationError> {
    let mut seen = BTreeMap::<SampleKey, &RegimeSample>::new();
    for sample in samples {
        let key = bound_sample_key(catalog, sample);
        if let Some(previous) = seen.insert(key, sample) {
            let binding =
                |sample: &RegimeSample| catalog.bindings.iter().find(|b| b.regime == sample.regime);
            let compatible = match (binding(previous), binding(sample)) {
                (Some(a), Some(b)) => {
                    a.sampling == b.sampling
                        && a.dependence == b.dependence
                        && a.weights == b.weights
                }
                _ => false,
            };
            if previous.columns != sample.columns || !compatible {
                return Err(EstimationError::data_msg("conflicting forwarded dataset aliases"));
            }
        }
    }
    Ok(())
}

/// Validate inference settings before allocating a table or running a bootstrap.
/// # Errors
/// Nonfinite/invalid coverage or an empty resource bound.
pub fn validate_options(options: &EmpiricalTableOptions) -> Result<(), EstimationError> {
    if !options.coverage_level.is_finite()
        || options.coverage_level <= 0.0
        || options.coverage_level >= 1.0
    {
        return Err(EstimationError::data_msg(
            "coverage_level must be finite and strictly between zero and one",
        ));
    }
    if options.max_joint_cells == 0 {
        return Err(EstimationError::data_msg("max_joint_cells must be positive"));
    }
    Ok(())
}

/// One estimated `(population, regime, world)` sample.
#[derive(Clone, Debug)]
pub struct RegimeSample {
    /// Population identity.
    pub population: Arc<str>,
    /// Catalog regime.
    pub regime: RegimeId,
    /// Snapshot identity shared by every world of this regime.
    pub snapshot_identity: Arc<str>,
    /// Concrete intervention world this sample was drawn under.
    pub interventions: Arc<[InterventionAssignment]>,
    /// Columns aligned by row; missing entries require an explicit missingness model.
    pub columns: BTreeMap<VariableId, Vec<Option<f64>>>,
}

impl RegimeSample {
    /// Number of physical rows.
    #[must_use]
    pub fn n(&self) -> usize {
        self.columns.values().next().map_or(0, Vec::len)
    }

    /// Extract named discrete columns from a tabular snapshot.
    ///
    /// # Errors
    /// Missing columns, length mismatch, or empty input.
    pub fn from_tabular(
        population: impl Into<Arc<str>>,
        regime: RegimeId,
        snapshot_identity: impl Into<Arc<str>>,
        interventions: impl Into<Arc<[InterventionAssignment]>>,
        data: &TabularData,
        variables: &[VariableId],
    ) -> Result<Self, EstimationError> {
        let mut columns = BTreeMap::new();
        let n = data.row_count();
        for &id in variables {
            columns.insert(id, discrete_column(data, id, n)?);
        }
        if columns.is_empty() {
            return Err(EstimationError::data_msg("empirical sample has no measured columns"));
        }
        Ok(Self {
            population: population.into(),
            regime,
            snapshot_identity: snapshot_identity.into(),
            interventions: interventions.into(),
            columns,
        })
    }
}

/// Mixed supplied laws and estimated samples for one statistical prepare.
#[derive(Clone, Debug, Default)]
pub struct StatisticalTransportInput {
    /// Known supplied laws. Cloned unchanged across bootstrap replicates.
    pub supplied: Vec<ExactDiscreteLaw>,
    /// Estimated regimes. One joint is fitted per sample.
    pub samples: Vec<RegimeSample>,
}

/// Fit one empirical joint on catalog domains, optionally on resampled rows.
///
/// # Errors
/// Empty or missing observations, values outside the catalog domain, cardinality overflow,
/// or an unlicensed smoother.
pub fn fit_empirical_joint(
    sample: &RegimeSample,
    axes: &[DiscreteAxis],
    options: &EmpiricalTableOptions,
    rows: Option<&[u32]>,
) -> Result<ExactDiscreteLaw, EstimationError> {
    if options.estimator != EmpiricalTableEstimator::Plugin {
        return Err(EstimationError::Refused {
            code: antecedent_core::reason_code!("transport_unsupported_evaluator"),
            message: format!(
                "{} is not a licensed transport provider; use {}",
                options.estimator.as_str(),
                EMPIRICAL_TABLE_PLUGIN
            ),
        });
    }
    validate_options(options)?;
    let n = sample.n();
    if n == 0 || n > u32::MAX as usize || sample.columns.values().any(|column| column.len() != n) {
        return Err(EstimationError::data_msg(
            "empirical columns must have equal, nonzero representable lengths",
        ));
    }
    let size = cartesian_size(axes, options.max_joint_cells)?;
    if size == 0 {
        return Err(law_err(sample, "invalid_law_axis"));
    }
    let indexed = axes
        .iter()
        .map(|axis| {
            let column = sample
                .columns
                .get(&axis.variable)
                .ok_or_else(|| EstimationError::data_msg("sample is missing a catalog axis"))?;
            let levels = axis
                .values
                .iter()
                .enumerate()
                .map(|(i, value)| {
                    let value = value.as_f64().filter(|v| v.is_finite()).ok_or_else(|| {
                        EstimationError::data_msg("empirical axes require finite numeric codes")
                    })?;
                    Ok((if value == 0.0 { 0 } else { value.to_bits() }, i))
                })
                .collect::<Result<std::collections::HashMap<_, _>, EstimationError>>()?;
            if levels.len() != axis.values.len() {
                return Err(EstimationError::data_msg("duplicate empirical domain levels"));
            }
            Ok(IndexedAxis { column, levels })
        })
        .collect::<Result<Vec<_>, EstimationError>>()?;
    let mut counts = vec![0.0; size];
    let mut complete = 0.0;
    let n = sample.n();
    let take: Box<dyn Iterator<Item = usize>> = match rows {
        Some(index) => Box::new(index.iter().map(|i| *i as usize)),
        None => Box::new(0..n),
    };
    for row in take {
        if row >= n {
            return Err(EstimationError::data_msg("empirical resample index out of range"));
        }
        let cell = cell_index(&indexed, row)?;
        counts[cell] += 1.0;
        complete += 1.0;
    }
    if complete == 0.0 {
        return Err(EstimationError::EmptyEmpiricalSample {
            message: law_error(sample, "empty_empirical_sample").to_string(),
        });
    }
    for count in &mut counts {
        *count /= complete;
    }
    ExactDiscreteLaw::try_empirical(
        sample.population.clone(),
        sample.regime,
        sample.interventions.clone(),
        axes.to_vec(),
        counts,
        sample.snapshot_identity.clone(),
        LawTolerance::default(),
    )
    .map_err(|e| EstimationError::data_msg(e.to_string()))
}

fn fit_statistical_joint(
    sample: &RegimeSample,
    axes: &[DiscreteAxis],
    options: &EmpiricalTableOptions,
    rows: Option<&[u32]>,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<ExactDiscreteLaw, EstimationError> {
    let EmpiricalTableEstimator::Learned(learner) = options.estimator else {
        return fit_empirical_joint(sample, axes, options, rows);
    };
    let empirical = fit_empirical_joint(
        sample,
        axes,
        &EmpiricalTableOptions { estimator: EmpiricalTableEstimator::Plugin, ..*options },
        rows,
    )?;
    let indexes: Vec<usize> =
        rows.map_or_else(|| (0..sample.n()).collect(), |r| r.iter().map(|v| *v as usize).collect());
    let columns = axes
        .iter()
        .map(|axis| {
            indexes
                .iter()
                .map(|row| {
                    let value = sample.columns[&axis.variable][*row]
                        .ok_or_else(|| EstimationError::data_msg("missing categorical value"))?;
                    axis.values.iter().position(|v| v.as_f64() == Some(value)).ok_or_else(|| {
                        EstimationError::data_msg("categorical value outside declared domain")
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    if axes.is_empty() {
        return Ok(empirical);
    }
    let model = antecedent_learn::FiniteJoint::fit(
        &columns,
        &axes.iter().map(|a| a.values.len()).collect::<Vec<_>>(),
        learner,
        options.max_joint_cells,
        ctx,
    )
    .map_err(crate::learn_nuisance::learn_err)?;
    let probabilities = model.probabilities(ctx).map_err(crate::learn_nuisance::learn_err)?;
    ExactDiscreteLaw::try_new(
        sample.population.clone(),
        sample.regime,
        sample.interventions.clone(),
        axes.to_vec(),
        probabilities,
        sample.snapshot_identity.clone(),
        LawTolerance::default(),
    )
    .and_then(|law| law.with_empirical_counts(model.counts().to_vec()))
    .map_err(|e| EstimationError::data_msg(e.to_string()))
}

/// Assemble supplied + fitted joints after checking provider coverage.
///
/// # Errors
/// Missing providers, convenience target samples, unlicensed smoothers, or domain errors.
pub fn assemble_statistical_laws(
    input: &StatisticalTransportInput,
    functional: &BoundTransportFunctional,
    options: &EmpiricalTableOptions,
    row_indexes: &BTreeMap<SampleKey, Vec<u32>>,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<ExactTransportData, EstimationError> {
    assemble_laws(input, functional, options, row_indexes, true, ctx)
}
fn assemble_laws(
    input: &StatisticalTransportInput,
    functional: &BoundTransportFunctional,
    options: &EmpiricalTableOptions,
    row_indexes: &BTreeMap<SampleKey, Vec<u32>>,
    require_coverage: bool,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<ExactTransportData, EstimationError> {
    if options.estimator == EmpiricalTableEstimator::Dirichlet {
        return Err(EstimationError::Refused {
            code: antecedent_core::reason_code!("transport_unsupported_evaluator"),
            message: format!("{} is not licensed", options.estimator.as_str()),
        });
    }
    validate_options(options)?;
    let catalog = functional.catalog();
    refuse_convenience_target(catalog, input, &functional.derivation().query().target)?;
    if input.supplied.iter().any(|law| law.origin() != antecedent_expr::LawOrigin::SuppliedExact) {
        return Err(EstimationError::data_msg(
            "empirical laws require their sample provenance; cannot supply them as known exact laws",
        ));
    }
    validate_dataset_aliases(catalog, &input.samples)?;
    let mut fitted_joints = BTreeMap::<SampleKey, ExactDiscreteLaw>::new();
    let mut laws = input.supplied.clone();
    for sample in &input.samples {
        if catalog.bindings.iter().any(|b| b.regime == sample.regime && b.weights.is_some()) {
            return Err(EstimationError::data_msg(
                "weighted sampling is not the unweighted empirical-table estimator",
            ));
        }
        if laws.iter().any(|law| {
            law.population() == sample.population.as_ref()
                && law.regime() == sample.regime
                && same_world(law.interventions(), &sample.interventions)
        }) {
            return Err(EstimationError::data_msg(
                "a regime cannot bind both a supplied law and an estimated sample",
            ));
        }
        let axes = catalog_axes_bounded(catalog, sample, options.max_joint_cells)?;
        let regime =
            catalog.regimes.iter().find(|r| r.id == sample.regime).expect("validated regime");
        let mut interventions: Vec<_> = sample.interventions.iter().map(|a| a.variable).collect();
        interventions.sort_unstable();
        let mut expected = regime.interventions.to_vec();
        expected.sort_unstable();
        if interventions != expected
            || sample.interventions.iter().any(|a| !a.value.as_f64().is_some_and(f64::is_finite))
        {
            return Err(EstimationError::data_msg(
                "sample intervention world disagrees with its regime",
            ));
        }
        let key = bound_sample_key(catalog, sample);
        let fitted = if let Some(joint) = fitted_joints.get(&key) {
            if joint.axes() != axes {
                return Err(EstimationError::data_msg(
                    "forwarded aliases have different measured domains",
                ));
            }
            let alias = ExactDiscreteLaw::try_empirical(
                sample.population.clone(),
                sample.regime,
                sample.interventions.clone(),
                axes,
                joint.probabilities().to_vec(),
                sample.snapshot_identity.clone(),
                joint.tolerance(),
            )
            .map_err(|e| EstimationError::data_msg(e.to_string()))?;
            if let Some(counts) = joint.empirical_counts() {
                alias
                    .with_empirical_counts(counts.to_vec())
                    .map_err(|e| EstimationError::data_msg(e.to_string()))?
            } else {
                alias
            }
        } else {
            let joint = fit_statistical_joint(
                sample,
                &axes,
                options,
                row_indexes.get(&key).map(Vec::as_slice),
                ctx,
            )?;
            fitted_joints.insert(key, joint.clone());
            joint
        };
        laws.push(fitted);
    }
    if require_coverage {
        require_leaf_providers(functional, &laws, input)?;
    }
    ExactTransportData::try_new(laws, options.max_joint_cells.max(1))
        .map_err(|e| EstimationError::data_msg(e.to_string()))
}

/// Point-estimate assembly (no resampling).
///
/// # Errors
/// Same as [`assemble_statistical_laws`].
pub fn assemble_point_laws(
    input: &StatisticalTransportInput,
    functional: &BoundTransportFunctional,
    options: &EmpiricalTableOptions,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<ExactTransportData, EstimationError> {
    assemble_statistical_laws(input, functional, options, &BTreeMap::new(), ctx)
}

/// Assemble available grid providers, leaving missing factors to located per-point preflight.
/// # Errors
/// Invalid supplied providers or sampling contracts; missing providers are retained by the grid.
pub fn assemble_grid_point_laws(
    input: &StatisticalTransportInput,
    functional: &BoundTransportFunctional,
    options: &EmpiricalTableOptions,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<ExactTransportData, EstimationError> {
    assemble_laws(input, functional, options, &BTreeMap::new(), false, ctx)
}

/// Catalog-declared finite axes for one sample's measured non-intervention coordinates.
///
/// # Errors
/// Missing environment, continuous/unspecified domains, or cardinality overflow.
pub fn catalog_axes(
    catalog: &EvidenceCatalog,
    sample: &RegimeSample,
) -> Result<Vec<DiscreteAxis>, EstimationError> {
    catalog_axes_bounded(catalog, sample, 1_000_000)
}

fn catalog_axes_bounded(
    catalog: &EvidenceCatalog,
    sample: &RegimeSample,
    max_cells: usize,
) -> Result<Vec<DiscreteAxis>, EstimationError> {
    let regime = catalog
        .regimes
        .iter()
        .find(|r| r.id == sample.regime && r.population.as_ref() == sample.population.as_ref())
        .ok_or_else(|| EstimationError::data_msg("empirical sample names an unknown regime"))?;
    if !regime.evidence_kind.can_satisfy_factor() {
        return Err(EstimationError::data_msg(
            "proposed or manipulable regimes cannot be estimated",
        ));
    }
    let environment = catalog
        .environments
        .iter()
        .find(|env| env.identity.as_ref() == sample.population.as_ref())
        .ok_or_else(|| {
            EstimationError::data_msg("empirical sample names an unknown environment")
        })?;
    for assignment in sample.interventions.iter() {
        let cardinality = environment
            .variables
            .iter()
            .find(|coordinate| coordinate.variable == assignment.variable)
            .map_or(0, |coordinate| match coordinate.domain {
                VariableDomain::Binary => 2,
                VariableDomain::Categorical { cardinality } => cardinality,
                _ => 0,
            });
        if !assignment.value.as_f64().is_some_and(|value| {
            value.is_finite()
                && value.fract() == 0.0
                && value >= 0.0
                && value < f64::from(cardinality)
        }) {
            return Err(EstimationError::data_msg(
                "intervention value is outside its declared finite domain",
            ));
        }
    }
    let mut axes = Vec::new();
    let mut cells = 1usize;
    for variable in regime.measured.iter() {
        if sample.interventions.iter().any(|a| a.variable == *variable) {
            continue;
        }
        let coordinate =
            environment.variables.iter().find(|c| c.variable == *variable).ok_or_else(|| {
                EstimationError::data_msg("measured variable lacks an environment coordinate")
            })?;
        let cardinality = match coordinate.domain {
            VariableDomain::Binary => 2,
            VariableDomain::Categorical { cardinality } => cardinality as usize,
            _ => 0,
        };
        if cardinality == 0 || cells.checked_mul(cardinality).is_none_or(|n| n > max_cells) {
            return Err(EstimationError::data_msg(
                "sparse or high-dimensional empirical table; declared finite domains required",
            ));
        }
        cells *= cardinality; // Checked against overflow and the configured bound above.
        axes.push(DiscreteAxis { variable: *variable, values: domain_levels(&coordinate.domain)? });
    }
    if axes.is_empty() {
        return Err(EstimationError::data_msg("empirical joint has no remaining measured axes"));
    }
    Ok(axes)
}

/// Whether every estimated binding is IID independent studies.
#[must_use]
pub fn licensed_iid_dependence(catalog: &EvidenceCatalog, samples: &[RegimeSample]) -> bool {
    licensed_iid_regimes(catalog, samples.iter().map(|sample| sample.regime))
}

/// [`licensed_iid_dependence`] over bare regime ids, for callers that hold a sample
/// summary rather than the samples. A catalog binds one dataset per regime, so the
/// regime's binding is unique.
#[must_use]
pub fn licensed_iid_regimes(
    catalog: &EvidenceCatalog,
    regimes: impl IntoIterator<Item = RegimeId>,
) -> bool {
    regimes.into_iter().all(|regime| {
        let Some(binding) = catalog.bindings.iter().find(|b| b.regime == regime) else {
            return false;
        };
        binding.sampling == SamplingDesign::Independent
            && binding.dependence == DependenceGroup::IndependentStudies
            && binding.weights.is_none()
    })
}

/// Stable reason when a licensed interval cannot be claimed.
#[must_use]
pub fn dependence_refusal(
    catalog: &EvidenceCatalog,
    samples: &[RegimeSample],
) -> Option<&'static str> {
    if samples.is_empty() {
        return Some("no_estimated_regime");
    }
    if licensed_iid_dependence(catalog, samples) {
        None
    } else {
        Some("transport.unsupported_dependence")
    }
}

fn refuse_convenience_target(
    catalog: &EvidenceCatalog,
    input: &StatisticalTransportInput,
    target: &str,
) -> Result<(), EstimationError> {
    if !matches!(catalog.target_sampling, Some(TargetSampling::ConvenienceSample)) {
        return Ok(());
    }
    if input.samples.iter().any(|sample| sample.population.as_ref() == target) {
        return Err(EstimationError::data_msg(
            "convenience samples cannot stand in for the target population law",
        ));
    }
    Ok(())
}

fn require_leaf_providers(
    functional: &BoundTransportFunctional,
    laws: &[ExactDiscreteLaw],
    input: &StatisticalTransportInput,
) -> Result<(), EstimationError> {
    use antecedent_expr::ExprNode;
    let arena = functional.arena();
    let mut pending = vec![functional.root()];
    let mut seen = std::collections::BTreeSet::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id.raw()) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Distribution { population, regime, .. } => {
                let population = arena.population(*population);
                let Some(regime) = *regime else {
                    return Err(EstimationError::Refused {
                        code: antecedent_core::reason_code!("transport_missing_provider"),
                        message: "certified leaf is missing a catalog regime binding".into(),
                    });
                };
                let covered =
                    laws.iter().any(|law| law.population() == population && law.regime() == regime)
                        || input.samples.iter().any(|sample| {
                            sample.population.as_ref() == population && sample.regime == regime
                        });
                if !covered {
                    return Err(EstimationError::Refused {
                        code: antecedent_core::reason_code!("transport_missing_provider"),
                        message: format!(
                            "available regime {regime:?} in {population} has neither a supplied law nor a bound sample"
                        ),
                    });
                }
            }
            ExprNode::Kernel { body, .. } => pending.push(*body),
            ExprNode::Product(list) => pending.extend(arena.list(*list)),
            ExprNode::SumOut { expr, .. } | ExprNode::IntegralOut { expr, .. } => {
                pending.push(*expr)
            }
            ExprNode::Ratio { numerator, denominator } => {
                pending.extend([*numerator, *denominator]);
            }
            ExprNode::Expectation { distribution, .. } => pending.push(*distribution),
            ExprNode::Contrast { left, right, .. } => pending.extend([*left, *right]),
        }
    }
    Ok(())
}

fn domain_levels(domain: &VariableDomain) -> Result<Arc<[Value]>, EstimationError> {
    match *domain {
        VariableDomain::Binary => Ok(Arc::from([Value::Int64(0), Value::Int64(1)])),
        VariableDomain::Categorical { cardinality } => {
            if cardinality == 0 {
                return Err(EstimationError::data_msg("categorical cardinality must be positive"));
            }
            Ok((0..i64::from(cardinality)).map(Value::Int64).collect())
        }
        VariableDomain::Unspecified | VariableDomain::Continuous | VariableDomain::Count => {
            Err(EstimationError::data_msg(
                "empirical tables require a declared finite binary or categorical domain",
            ))
        }
    }
}

fn cartesian_size(axes: &[DiscreteAxis], max_cells: usize) -> Result<usize, EstimationError> {
    let mut size = 1usize;
    for axis in axes {
        size =
            size.checked_mul(axis.values.len()).filter(|n| *n <= max_cells).ok_or_else(|| {
                EstimationError::data_msg("sparse or high-dimensional empirical table")
            })?;
    }
    Ok(size)
}

struct IndexedAxis<'a> {
    column: &'a [Option<f64>],
    levels: std::collections::HashMap<u64, usize>,
}

fn cell_index(axes: &[IndexedAxis<'_>], row: usize) -> Result<usize, EstimationError> {
    let mut index = 0usize;
    for axis in axes {
        let value = axis.column[row].ok_or_else(|| EstimationError::data_msg("missing sample value: complete observations or an explicit missingness model are required"))?;
        let key = if value == 0.0 { 0 } else { value.to_bits() };
        let level = axis.levels.get(&key).ok_or_else(|| {
            EstimationError::data_msg("sample value lies outside the catalog domain")
        })?;
        // The full Cartesian size was checked before allocation.
        index = index * axis.levels.len() + level;
    }
    Ok(index)
}

/// One column as `f64` cells (`None` = missing), read through the one discrete reader
/// (`TabularData::discrete_column`) that the functional-distribution estimator also uses:
/// invalid cells and rows outside the analysis mask are missing.
fn discrete_column(
    data: &TabularData,
    id: VariableId,
    n: usize,
) -> Result<Vec<Option<f64>>, EstimationError> {
    let view = data.column(id).map_err(EstimationError::from)?;
    if view.len() != n {
        return Err(EstimationError::data_msg("column length mismatch"));
    }
    if !matches!(view, ColumnView::Float64(_) | ColumnView::Int64(_)) {
        return Err(EstimationError::data_msg("empirical tables require numeric columns"));
    }
    let column = data.discrete_column(id).map_err(EstimationError::from)?;
    // Integer cells are finite-domain codes: they must be representable as `u32`.
    let level_values = column
        .levels
        .iter()
        .map(|level| match level {
            Value::Int64(v) => u32::try_from(*v).map(f64::from).map_err(|_| {
                EstimationError::data_msg("categorical code is outside the finite domain")
            }),
            other => other.as_f64().ok_or_else(|| {
                EstimationError::data_msg("empirical tables require numeric columns")
            }),
        })
        .collect::<Result<Vec<f64>, _>>()?;
    Ok(column
        .codes
        .iter()
        .map(|&code| {
            (code != antecedent_data::DiscreteColumn::MISSING).then(|| level_values[code as usize])
        })
        .collect())
}

fn same_world(left: &[InterventionAssignment], right: &[InterventionAssignment]) -> bool {
    left.len() == right.len()
        && left.iter().all(|a| {
            right.iter().any(|b| a.variable == b.variable && a.value.as_f64() == b.value.as_f64())
        })
}

fn law_error(sample: &RegimeSample, kind: &'static str) -> ExactLawError {
    ExactLawError {
        kind,
        population: sample.population.clone(),
        regime: Some(sample.regime),
        variables: Arc::from([]),
        conditioning: Arc::from([]),
        interventions: sample.interventions.clone(),
    }
}

fn law_err(sample: &RegimeSample, kind: &'static str) -> EstimationError {
    EstimationError::data_msg(law_error(sample, kind).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        DistributionAvailability, Environment, EvidenceKind, EvidenceRegime, RegimeKind,
        VariableCoordinate,
    };

    fn v(i: u32) -> VariableId {
        VariableId::from_raw(i)
    }

    fn binary_env(identity: &str) -> Environment {
        Environment::try_new(
            identity,
            [
                VariableCoordinate { variable: v(0), domain: VariableDomain::Binary, unit: None },
                VariableCoordinate { variable: v(1), domain: VariableDomain::Binary, unit: None },
            ],
            [],
        )
        .unwrap()
    }

    fn sample(p00: usize, p01: usize, p10: usize, p11: usize) -> RegimeSample {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for (xv, yv, n) in [(0.0, 0.0, p00), (0.0, 1.0, p01), (1.0, 0.0, p10), (1.0, 1.0, p11)] {
            x.extend(std::iter::repeat_n(Some(xv), n));
            y.extend(std::iter::repeat_n(Some(yv), n));
        }
        RegimeSample {
            population: Arc::from("target"),
            regime: RegimeId::from_raw(0),
            snapshot_identity: Arc::from("s"),
            interventions: Arc::from([]),
            columns: BTreeMap::from([(v(0), x), (v(1), y)]),
        }
    }

    #[test]
    fn rows_outside_the_analysis_mask_are_not_in_the_sample() {
        // x = 0,0,1,1 and y = 0,1,0,1; rows 1 and 2 are masked out.
        let data = TabularData::from_f64_columns([
            ("x", &[0.0, 0.0, 1.0, 1.0][..]),
            ("y", &[0.0, 1.0, 0.0, 1.0][..]),
        ])
        .unwrap();
        let mask = antecedent_data::ValidityBitmap::from_bytes(vec![0b1001u8], 4).unwrap();
        let masked = data.with_analysis_mask(mask).unwrap();
        let sample = RegimeSample::from_tabular(
            "target",
            RegimeId::from_raw(0),
            "s",
            Vec::<InterventionAssignment>::new(),
            &masked,
            &[v(0), v(1)],
        )
        .unwrap();
        assert_eq!(sample.columns[&v(0)], vec![Some(0.0), None, None, Some(1.0)]);
        assert_eq!(sample.columns[&v(1)], vec![Some(0.0), None, None, Some(1.0)]);
    }

    #[test]
    fn plugin_normalizes_fully_observed_frequencies() {
        let table = fit_empirical_joint(
            &sample(1, 1, 1, 1),
            &[
                DiscreteAxis {
                    variable: v(0),
                    values: Arc::from([Value::Int64(0), Value::Int64(1)]),
                },
                DiscreteAxis {
                    variable: v(1),
                    values: Arc::from([Value::Int64(0), Value::Int64(1)]),
                },
            ],
            &EmpiricalTableOptions::default(),
            None,
        )
        .unwrap();
        assert_eq!(table.origin(), antecedent_expr::LawOrigin::EmpiricalPlugin);
        assert!(table.probabilities().iter().all(|p| (*p - 0.25).abs() < 1e-15));
    }

    #[test]
    fn incomplete_rows_require_a_missingness_contract() {
        let mut s = sample(2, 0, 0, 0);
        s.columns.get_mut(&v(1)).unwrap()[1] = None;
        let error = fit_empirical_joint(
            &s,
            &[
                DiscreteAxis {
                    variable: v(0),
                    values: Arc::from([Value::Int64(0), Value::Int64(1)]),
                },
                DiscreteAxis {
                    variable: v(1),
                    values: Arc::from([Value::Int64(0), Value::Int64(1)]),
                },
            ],
            &EmpiricalTableOptions::default(),
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("missing sample value"));
    }

    #[test]
    fn unequal_native_columns_are_not_implicit_missing_values() {
        let mut input = sample(1, 1, 1, 1);
        input.columns.get_mut(&v(1)).unwrap().pop();
        let axes = [0, 1].map(|i| DiscreteAxis {
            variable: v(i),
            values: Arc::from([Value::Int64(0), Value::Int64(1)]),
        });
        let error = fit_empirical_joint(&input, &axes, &EmpiricalTableOptions::default(), None)
            .unwrap_err();
        assert!(error.to_string().contains("equal, nonzero"));
    }

    #[test]
    fn an_empty_resample_is_a_typed_error_not_a_message_match() {
        let axes = [DiscreteAxis {
            variable: v(0),
            values: Arc::from([Value::Int64(0), Value::Int64(1)]),
        }];
        let error = fit_empirical_joint(
            &sample(2, 1, 1, 2),
            &axes,
            &EmpiricalTableOptions::default(),
            Some(&[]),
        )
        .unwrap_err();
        assert!(matches!(error, EstimationError::EmptyEmpiricalSample { .. }), "{error}");
        // Any other data error stays an ordinary error.
        let other = fit_empirical_joint(
            &sample(2, 1, 1, 2),
            &axes,
            &EmpiricalTableOptions::default(),
            Some(&[u32::MAX]),
        )
        .unwrap_err();
        assert!(!matches!(other, EstimationError::EmptyEmpiricalSample { .. }));
    }

    #[test]
    fn dirichlet_is_an_unlicensed_evaluator() {
        let err = fit_empirical_joint(
            &sample(1, 0, 0, 0),
            &[DiscreteAxis {
                variable: v(0),
                values: Arc::from([Value::Int64(0), Value::Int64(1)]),
            }],
            &EmpiricalTableOptions {
                estimator: EmpiricalTableEstimator::Dirichlet,
                ..EmpiricalTableOptions::default()
            },
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("transport_unsupported_evaluator"));
    }

    #[test]
    fn catalog_axes_use_declared_binary_domains() {
        let catalog = EvidenceCatalog::try_new(
            [binary_env("target")],
            [EvidenceRegime::try_new(
                RegimeId::from_raw(0),
                RegimeKind::Observational,
                EvidenceKind::Available,
                [],
                [],
                [v(0), v(1)],
                "target",
                DistributionAvailability::Joint,
            )
            .unwrap()],
            [],
            None,
        )
        .unwrap();
        let axes = catalog_axes(&catalog, &sample(1, 0, 0, 0)).unwrap();
        assert_eq!(axes.len(), 2);
        assert_eq!(axes[0].values.len(), 2);
    }
}
