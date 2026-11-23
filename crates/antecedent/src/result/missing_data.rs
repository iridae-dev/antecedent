/// How an execution treated missing values: the policy, the assumption that policy needs,
/// and the row accounting.
///
/// Present on every result whose route reads rows of a table, including results that dropped
/// nothing, so a reader never has to infer that no row was removed.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct MissingDataReport {
    /// Rows total, kept and dropped, per variable and per reason, with the dropped-row digest.
    pub accounting: antecedent_data::RowAccounting,
    /// Whether the caller declared the policy; `false` when the route's own default applied.
    pub declared: bool,
    /// Names of the covariates of the observation model, for responder weighting.
    pub covariates: std::sync::Arc<[std::sync::Arc<str>]>,
    /// The missingness assumption the policy needs, as published in the missing-data guide.
    pub assumption: std::sync::Arc<str>,
    /// Whether the policy differs from the route's default. Only then does a claim over the
    /// result carry the policy, its assumption and the dropped-row digest, so default claims
    /// keep their identity.
    pub non_default: bool,
    /// The impute-fit-pool record, for multiple imputation.
    pub imputation: Option<ImputationReport>,
}

/// How a multiple-imputation result was formed: the declared specification, what was
/// imputed, every completed table's estimate and the pooled variance components.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ImputationReport {
    /// The declared specification (model, imputations, sweeps, seed, assumption).
    pub spec: antecedent_data::ImputationSpec,
    /// Unit the parameter draws resampled (`row` or `group`).
    pub unit: std::sync::Arc<str>,
    /// Units resampled.
    pub units: usize,
    /// Cells imputed per variable of the imputation model.
    pub imputed_cells: std::sync::Arc<[(std::sync::Arc<str>, usize)]>,
    /// Estimator fitted on every completed table.
    pub estimator: std::sync::Arc<str>,
    /// `rubin` (frequentist), `stacked_posterior` (Bayesian) or `stacked_graph_posterior` (Bayesian
    /// mixture over declared graphs, whose weights stay inside every table's draws).
    pub pooling: std::sync::Arc<str>,
    /// Each completed table's point estimate, in imputation order.
    pub points: std::sync::Arc<[f64]>,
    /// Within-imputation variance.
    pub within: f64,
    /// Between-imputation variance.
    pub between: f64,
    /// Total variance (Rubin's `T`, or the stacked posterior variance).
    pub total: f64,
    /// Degrees of freedom of the Student-t reference (Barnard–Rubin when the estimator has a
    /// finite complete-data degrees of freedom); `None` for a normal reference or a posterior.
    pub df: Option<f64>,
    /// Complete-data degrees of freedom used by the Barnard–Rubin adjustment.
    pub df_complete: Option<f64>,
    /// Fraction of missing information, for Rubin pooling.
    pub fraction_missing_information: Option<f64>,
    /// Relative increase in variance due to nonresponse, for Rubin pooling.
    pub relative_increase: Option<f64>,
    /// Digest (hex) of the model and design assumptions the pooled estimate rests on: the
    /// imputation model, tables, sweeps, seed, missingness assumption, resampling unit, declared
    /// sampling design and every variable's kind and imputed-cell count. Two reports with
    /// different digests are different analyses.
    pub assumptions_digest: std::sync::Arc<str>,
    /// Canonical declared model-column domains.
    pub imputation_domains:
        std::sync::Arc<[(std::sync::Arc<str>, antecedent_data::ImputationDomain)]>,
    /// Binary model variables and their single observed class under empirical-bootstrap support.
    pub binary_boundary_variables: std::sync::Arc<[(std::sync::Arc<str>, u8)]>,
    /// Neighbouring time steps on each side the imputation model conditioned on, for a series;
    /// zero for exchangeable rows.
    pub lags: usize,
    /// The joint pooling of the answer's coordinates, for a route whose answer has several
    /// (the cells of a conditional effect, a response curve, the levels of a distribution, a
    /// mediation decomposition, the arms of a randomized design, the horizons of a series).
    pub vector: Option<ImputationVectorReport>,
}

/// The Wald test of a pooled answer's coordinates against a stated null (Li, Raghunathan and
/// Rubin's `D1`), referred to an `F` law.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ImputationWald {
    /// The hypothesised value of every coordinate.
    pub null: std::sync::Arc<[f64]>,
    /// The pooled statistic.
    pub statistic: f64,
    /// Numerator degrees of freedom, the number of coordinates.
    pub df_numerator: f64,
    /// Denominator degrees of freedom; infinite is the chi-square reference.
    pub df_denominator: f64,
}

/// A quantity of the answer that has no published within-imputation variance, pooled by its
/// mean over the completed tables with the between-imputation variance beside it.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ImputationExtra {
    /// What the quantity is.
    pub label: std::sync::Arc<str>,
    /// Its value on each completed table, in imputation order.
    pub points: std::sync::Arc<[f64]>,
    /// The mean over the completed tables.
    pub point: f64,
    /// Between-imputation variance of the value: a lower bound on its pooled variance.
    pub between: f64,
}

/// The joint pooling of an answer with several coordinates.
///
/// Matrices are row-major `p x p` over the coordinates in `labels` order. Under Rubin pooling
/// `total = within + (1 + 1/m) between`; under a stacked posterior `total` is the covariance of
/// the equal-weight mixture and `within` carries each coordinate's mean posterior variance on
/// its diagonal.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ImputationVectorReport {
    /// Name of each coordinate.
    pub labels: std::sync::Arc<[std::sync::Arc<str>]>,
    /// How the within-imputation covariance was formed: `joint_covariance` (a retained joint
    /// influence covariance), `marginal_variances` (each coordinate's published variance on the
    /// diagonal, no cross-covariance), or `posterior_draws` (a stacked posterior).
    pub covariance_basis: std::sync::Arc<str>,
    /// Every completed table's coordinates, in imputation order.
    pub points: std::sync::Arc<[std::sync::Arc<[f64]>]>,
    /// Pooled coordinates.
    pub point: std::sync::Arc<[f64]>,
    /// Within-imputation covariance.
    pub within: std::sync::Arc<[f64]>,
    /// Between-imputation covariance.
    pub between: std::sync::Arc<[f64]>,
    /// Total covariance.
    pub total: std::sync::Arc<[f64]>,
    /// Degrees of freedom of each coordinate's Student-t reference (Rubin pooling); `None` for a
    /// normal reference or a posterior.
    pub coordinate_df: std::sync::Arc<[Option<f64>]>,
    /// Fraction of missing information of each coordinate (Rubin pooling).
    pub coordinate_fraction_missing_information: std::sync::Arc<[Option<f64>]>,
    /// Average relative increase in variance due to nonresponse (Rubin pooling).
    pub average_relative_increase: Option<f64>,
    /// The joint Wald test that every coordinate is zero, when the within-imputation covariance
    /// is invertible (Rubin pooling).
    pub wald: Option<ImputationWald>,
    /// Quantities of the answer pooled by their mean alone.
    pub extras: std::sync::Arc<[ImputationExtra]>,
}

/// Bitwise equality of two number slices, without building either side's bit vector.
fn same_bits(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| a.to_bits() == b.to_bits())
}

/// Bitwise equality of two optional numbers.
fn same_option_bits(left: Option<f64>, right: Option<f64>) -> bool {
    left.map(f64::to_bits) == right.map(f64::to_bits)
}

/// Bitwise equality of two slices of optional numbers.
fn same_option_slice_bits(left: &[Option<f64>], right: &[Option<f64>]) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| same_option_bits(*a, *b))
}

// The `PartialEq` impls below destructure both sides without `..`, so a field added to a
// report is a compile error here until the comparison names it.

impl PartialEq for ImputationWald {
    fn eq(&self, other: &Self) -> bool {
        let Self { null, statistic, df_numerator, df_denominator } = self;
        let Self {
            null: other_null,
            statistic: other_statistic,
            df_numerator: other_df_numerator,
            df_denominator: other_df_denominator,
        } = other;
        same_bits(null, other_null)
            && same_bits(
                &[*statistic, *df_numerator, *df_denominator],
                &[*other_statistic, *other_df_numerator, *other_df_denominator],
            )
    }
}

impl Eq for ImputationWald {}

impl PartialEq for ImputationExtra {
    fn eq(&self, other: &Self) -> bool {
        let Self { label, points, point, between } = self;
        let Self {
            label: other_label,
            points: other_points,
            point: other_point,
            between: other_between,
        } = other;
        label == other_label
            && same_bits(points, other_points)
            && same_bits(&[*point, *between], &[*other_point, *other_between])
    }
}

impl Eq for ImputationExtra {}

impl PartialEq for ImputationVectorReport {
    fn eq(&self, other: &Self) -> bool {
        let Self {
            labels,
            covariance_basis,
            points,
            point,
            within,
            between,
            total,
            coordinate_df,
            coordinate_fraction_missing_information,
            average_relative_increase,
            wald,
            extras,
        } = self;
        let Self {
            labels: other_labels,
            covariance_basis: other_covariance_basis,
            points: other_points,
            point: other_point,
            within: other_within,
            between: other_between,
            total: other_total,
            coordinate_df: other_coordinate_df,
            coordinate_fraction_missing_information: other_fraction,
            average_relative_increase: other_average_relative_increase,
            wald: other_wald,
            extras: other_extras,
        } = other;
        labels == other_labels
            && covariance_basis == other_covariance_basis
            && points.len() == other_points.len()
            && points.iter().zip(other_points.iter()).all(|(a, b)| same_bits(a, b))
            && same_bits(point, other_point)
            && same_bits(within, other_within)
            && same_bits(between, other_between)
            && same_bits(total, other_total)
            && same_option_slice_bits(coordinate_df, other_coordinate_df)
            && same_option_slice_bits(coordinate_fraction_missing_information, other_fraction)
            && same_option_bits(*average_relative_increase, *other_average_relative_increase)
            && wald == other_wald
            && extras == other_extras
    }
}

impl Eq for ImputationVectorReport {}

/// Bitwise equality of the numbers, so the report is `Eq` like the accounting it travels
/// with: two reports are equal when they would serialize identically.
impl PartialEq for ImputationReport {
    fn eq(&self, other: &Self) -> bool {
        let Self {
            spec,
            unit,
            units,
            imputed_cells,
            estimator,
            pooling,
            points,
            within,
            between,
            total,
            df,
            df_complete,
            fraction_missing_information,
            relative_increase,
            assumptions_digest,
            imputation_domains,
            binary_boundary_variables,
            lags,
            vector,
        } = self;
        let Self {
            spec: other_spec,
            unit: other_unit,
            units: other_units,
            imputed_cells: other_imputed_cells,
            estimator: other_estimator,
            pooling: other_pooling,
            points: other_points,
            within: other_within,
            between: other_between,
            total: other_total,
            df: other_df,
            df_complete: other_df_complete,
            fraction_missing_information: other_fraction,
            relative_increase: other_relative_increase,
            assumptions_digest: other_assumptions_digest,
            imputation_domains: other_imputation_domains,
            binary_boundary_variables: other_binary_boundary_variables,
            lags: other_lags,
            vector: other_vector,
        } = other;
        spec == other_spec
            && unit == other_unit
            && units == other_units
            && imputed_cells == other_imputed_cells
            && estimator == other_estimator
            && pooling == other_pooling
            && same_bits(points, other_points)
            && same_bits(
                &[*within, *between, *total],
                &[*other_within, *other_between, *other_total],
            )
            && same_option_bits(*df, *other_df)
            && same_option_bits(*df_complete, *other_df_complete)
            && same_option_bits(*fraction_missing_information, *other_fraction)
            && same_option_bits(*relative_increase, *other_relative_increase)
            && assumptions_digest == other_assumptions_digest
            && imputation_domains == other_imputation_domains
            && binary_boundary_variables == other_binary_boundary_variables
            && lags == other_lags
            && vector == other_vector
    }
}

impl Eq for ImputationReport {}

impl MissingDataReport {
    /// The policy the accounting was produced under.
    #[must_use]
    pub fn policy(&self) -> antecedent_data::MissingPolicy {
        self.accounting.policy
    }
}
