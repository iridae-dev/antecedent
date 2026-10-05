//! Cluster-aware cross-fitted AIPW (2.2 E4): the independence unit is a declared cluster.
//!
//! Rows of one cluster are never split across cross-fit folds, so the nuisances scoring a
//! cluster are fit on other clusters only, and the dependence enters the *score* level: the
//! scores of a cluster are summed before they are squared. This is not an IID cross-fit
//! followed by a cluster standard error (`AnalyticSeKind::Cluster` on `AipwAte`): there, a
//! cluster's rows train the nuisances that score its own rows, which the cluster sandwich
//! cannot undo.
//!
//! The route publishes the cross-fitted point estimate, the score table and the
//! cluster-sandwich standard error of the scores (`se_analytic`, also the named
//! [`ClusterDmlReceipt`]; the derivation and its conditions are in
//! `docs/guides/clustered-dml.md`), with the few-cluster Student-t reference
//! (`EffectEstimate::se_reference_df`). A retarget of the table uses the cluster-summed joint
//! covariance ([`ClusterDml::influence_covariance`]). Its coverage is measured by the
//! calibration harness, not asserted here. A cluster count below the declared minimum is a
//! typed refusal.
//!
//! A *dyadic* (two-way) unit is the second supported declaration: rows carry two endpoint
//! labels in separate namespaces, folds own whole connected components of the endpoint graph
//! (so no endpoint is shared across folds), and the receipt is the two-way
//! Cameron-Gelbach-Miller variance of the cluster-summed scores. What it does not cover
//! (an entity appearing in both endpoint roles, a giant component, too few components per
//! fold) is a typed `dyadic_dependence_not_licensed` / `too_few_clusters` refusal.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use antecedent_core::TargetPopulation;

use crate::error::{EstimationError, RefusalFields};
use crate::joint_if::JointCovariance;
use crate::overlap::OverlapPolicy;
use crate::propensity::{refuse, trim_of};
use crate::scores::ScoreTable;
use crate::se::AnalyticSeKind;

/// Smallest cluster count a declaration may require. Below it the sandwich over cluster sums
/// has too few terms to mean anything, and a fold's training set would hold under eight
/// clusters at the default five folds.
pub const MIN_CLUSTERS_FLOOR: usize = 10;

/// Declared minimum cluster count when a caller does not choose one.
pub const DEFAULT_MIN_CLUSTERS: usize = 20;

/// Smallest number of connected components per fold a dyadic declaration may require.
pub const MIN_COMPONENTS_PER_FOLD_FLOOR: usize = 2;

/// Declared minimum components per fold when a dyadic caller does not choose one.
pub const DEFAULT_MIN_COMPONENTS_PER_FOLD: usize = 4;

/// Provenance tag a cluster-DML score table carries; a table bearing it publishes no interval.
pub(crate) const CLUSTER_DML_PROVENANCE_TAG: &str = ";cluster_dml=";

/// What is independent across the rows of the data.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum IndependenceUnit {
    /// Disjoint clusters (households, sites, patients): whole clusters are independent.
    Cluster,
    /// Rows are dyads: two endpoint labels per row (in separate namespaces) whose endpoints
    /// repeat across rows, so rows are dependent through either endpoint. Folds own whole
    /// connected components of the endpoint graph; the variance is the two-way
    /// Cameron-Gelbach-Miller construction.
    Dyad,
}

impl IndependenceUnit {
    /// Stable wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cluster => "cluster",
            Self::Dyad => "dyad",
        }
    }
}

/// Declared cluster-DML configuration of an `AipwAte`: the independence unit and the smallest
/// cluster count the route accepts. The labels are the estimator's `cluster_ids` (the first
/// endpoint for a dyadic unit) and, for a dyadic unit, `cluster_ids_second`, all aligned to
/// complete-case rows.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ClusterDml {
    unit: IndependenceUnit,
    min_clusters: usize,
    /// Dyadic only (zero for a cluster unit): the fewest connected components each fold must own.
    min_components_per_fold: usize,
}

impl ClusterDml {
    /// Cluster independence with a declared minimum cluster count.
    ///
    /// # Errors
    ///
    /// `invalid_argument` when `min_clusters` is below [`MIN_CLUSTERS_FLOOR`].
    pub fn new(min_clusters: usize) -> Result<Self, EstimationError> {
        check_min_clusters(min_clusters)?;
        Ok(Self { unit: IndependenceUnit::Cluster, min_clusters, min_components_per_fold: 0 })
    }

    /// A dyadic (two-way) independence declaration: at least `min_clusters` distinct labels
    /// at each endpoint and at least `min_components_per_fold` connected components of the
    /// endpoint graph per fold.
    ///
    /// # Errors
    ///
    /// `invalid_argument` when `min_clusters` is below [`MIN_CLUSTERS_FLOOR`] or
    /// `min_components_per_fold` is below [`MIN_COMPONENTS_PER_FOLD_FLOOR`].
    pub fn dyadic(
        min_clusters: usize,
        min_components_per_fold: usize,
    ) -> Result<Self, EstimationError> {
        check_min_clusters(min_clusters)?;
        if min_components_per_fold < MIN_COMPONENTS_PER_FOLD_FLOOR {
            return Err(refuse(
                antecedent_core::reason_code!("invalid_argument"),
                "cluster_dml.invalid_min_components",
                &format!(
                    "the declared minimum components per fold {min_components_per_fold} is \
                     below the floor {MIN_COMPONENTS_PER_FOLD_FLOOR}"
                ),
            ));
        }
        Ok(Self { unit: IndependenceUnit::Dyad, min_clusters, min_components_per_fold })
    }

    /// The declared independence unit.
    #[must_use]
    pub const fn independence_unit(&self) -> IndependenceUnit {
        self.unit
    }

    /// The declared minimum cluster count.
    #[must_use]
    pub const fn min_clusters(&self) -> usize {
        self.min_clusters
    }

    /// The declared minimum components per fold (zero for a cluster unit).
    #[must_use]
    pub const fn min_components_per_fold(&self) -> usize {
        self.min_components_per_fold
    }

    /// Stable key of the declaration, part of the estimator-spec identity. The cluster form
    /// is unchanged from before the dyadic unit existed.
    #[must_use]
    pub fn canonical_key(&self) -> String {
        match self.unit {
            IndependenceUnit::Cluster => {
                format!("cluster_dml.aipw.v1;unit=cluster;min_clusters={}", self.min_clusters)
            }
            IndependenceUnit::Dyad => format!(
                "cluster_dml.aipw.v1;unit=dyad;min_clusters={};min_components_per_fold={}",
                self.min_clusters, self.min_components_per_fold
            ),
        }
    }

    fn require_cluster_count(&self, found: usize, dimension: &str) -> Result<(), EstimationError> {
        if found < self.min_clusters {
            return Err(refuse_counted(
                antecedent_core::reason_code!("too_few_clusters"),
                "cluster_dml.too_few_clusters",
                &format!(
                    "{found} {dimension} are below the declared minimum {}; a sandwich over \
                     cluster sums needs many independent clusters and no interval or \
                     standard error is formed",
                    self.min_clusters
                ),
                &CountFacts {
                    reason: "too_few_clusters",
                    found,
                    minimum: self.min_clusters,
                    remedy: "supply more independent clusters (or coarser-grained units)",
                },
            ));
        }
        Ok(())
    }

    /// Check the declared labels against the prepared rows and the declared minimum, and
    /// return the fold unit of every row: the cluster label itself, or for a dyadic unit the
    /// connected component of the endpoint graph (named by its smallest first-endpoint
    /// label) so that no endpoint's rows are spread across folds.
    ///
    /// # Errors
    ///
    /// `required_option_missing` without labels, a length mismatch, `too_few_clusters` below
    /// the declared minimum (labels per endpoint, or components per fold), and for a dyadic
    /// unit `dyadic_dependence_not_licensed` when an endpoint label appears in both roles
    /// or one connected component holds more than one fold's share of the rows.
    pub fn declare_units(
        &self,
        first: Option<&[u32]>,
        second: Option<&[u32]>,
        nrows: usize,
    ) -> Result<Arc<[u32]>, EstimationError> {
        let Some(ids) = first else {
            return Err(refuse(
                antecedent_core::reason_code!("required_option_missing"),
                "cluster_dml.cluster_ids_missing",
                "cluster-DML AIPW needs the cluster label of every complete-case row \
                 (estimator cluster_ids)",
            ));
        };
        if ids.len() != nrows {
            return Err(EstimationError::data_msg(format!(
                "cluster_ids length {} != nrows {nrows}",
                ids.len()
            )));
        }
        match (self.unit, second) {
            (IndependenceUnit::Cluster, Some(_)) => Err(refuse(
                antecedent_core::reason_code!("invalid_argument"),
                "cluster_dml.second_ids_unexpected",
                "a cluster unit takes one label per row; second-endpoint labels belong to a \
                 dyadic declaration",
            )),
            (IndependenceUnit::Cluster, None) => {
                self.require_cluster_count(distinct_count(ids), "clusters")?;
                Ok(Arc::from(ids))
            }
            (IndependenceUnit::Dyad, None) => Err(refuse(
                antecedent_core::reason_code!("required_option_missing"),
                "cluster_dml.second_ids_missing",
                "a dyadic cluster-DML declaration needs the second-endpoint label of every \
                 complete-case row (estimator cluster_ids_second)",
            )),
            (IndependenceUnit::Dyad, Some(second)) => {
                if second.len() != nrows {
                    return Err(EstimationError::data_msg(format!(
                        "cluster_ids_second length {} != nrows {nrows}",
                        second.len()
                    )));
                }
                self.require_cluster_count(distinct_count(ids), "first-endpoint clusters")?;
                self.require_cluster_count(distinct_count(second), "second-endpoint clusters")?;
                let units = dyad_units(ids, second)?;
                self.require_balanced_components(&units)?;
                Ok(Arc::from(units))
            }
        }
    }

    /// Components must be small enough and numerous enough for whole-component folds.
    fn require_balanced_components(&self, units: &[u32]) -> Result<(), EstimationError> {
        let folds = crate::crossfit_aipw::DEFAULT_AIPW_FOLDS;
        let mut sizes: BTreeMap<u32, usize> = BTreeMap::new();
        for &unit in units {
            *sizes.entry(unit).or_default() += 1;
        }
        let largest = sizes.values().copied().max().unwrap_or(0);
        if largest * folds > units.len() {
            return Err(refuse_counted(
                antecedent_core::reason_code!("dyadic_dependence_not_licensed"),
                "cluster_dml.dyadic_giant_component",
                &format!(
                    "the largest connected component of the endpoint graph holds {largest} of \
                     {} rows, more than one of the {folds} folds' share; whole-component folds \
                     cannot be balanced and the held-out component would not be independent of \
                     most of the training data, so no cross-fit is run",
                    units.len()
                ),
                &CountFacts {
                    reason: "dyadic_giant_component",
                    found: sizes.len(),
                    minimum: 0,
                    remedy: "split or drop the hub endpoints that connect most rows, or declare \
                             a cluster unit instead of a dyadic one",
                },
            ));
        }
        let per_fold = sizes.len() / folds;
        if per_fold < self.min_components_per_fold {
            return Err(refuse_counted(
                antecedent_core::reason_code!("too_few_clusters"),
                "cluster_dml.too_few_components",
                &format!(
                    "{} connected components give the smallest of {folds} folds {per_fold}, \
                     below the declared minimum {} components per fold",
                    sizes.len(),
                    self.min_components_per_fold
                ),
                &CountFacts {
                    reason: "too_few_components_per_fold",
                    found: sizes.len(),
                    minimum: self.min_components_per_fold * folds,
                    remedy: "supply more independent components (a sparser endpoint graph)",
                },
            ));
        }
        Ok(())
    }

    /// Few-cluster reference degrees of freedom of the declared labels: `G - 1` for a cluster
    /// unit, `min(G_a, G_b) - 1` for a dyadic one (the receipt's `reference_df`).
    #[must_use]
    pub fn reference_df(&self, first: &[u32], second: Option<&[u32]>) -> usize {
        let found = distinct_count(first);
        second.map_or(found, |s| distinct_count(s).min(found)).saturating_sub(1)
    }

    /// Joint covariance of cluster-summed influence columns.
    ///
    /// Each column `psi_k` is a claim's per-row influence scaled so that its own
    /// cluster-sandwich variance is the claim's variance (`psi_kr = n a_kr (phi_kr - theta_k)`
    /// with `a_kr = w_kr / sum w`; uniform weights give the receipt's scores). The variance of
    /// one column is the receipt's: `G/(G-1) n^-2 sum_g S_g^2` (cluster unit) or the
    /// Cameron-Gelbach-Miller `V_a + V_b - V_ab` (dyadic). The cross-covariance of two columns
    /// is the polarization `(V(psi_k + psi_l) - V(psi_k - psi_l)) / 4` of that quadratic form,
    /// which is exactly the bilinear cluster-summed form.
    ///
    /// # Errors
    ///
    /// A label/column length mismatch, labels that do not match the declared unit, too few
    /// clusters, or a materially negative two-way variance.
    pub fn influence_covariance(
        &self,
        columns: &[&[f64]],
        first: &[u32],
        second: Option<&[u32]>,
    ) -> Result<JointCovariance, EstimationError> {
        let n = first.len();
        if columns.iter().any(|c| c.len() != n) || second.is_some_and(|s| s.len() != n) {
            return Err(EstimationError::data_msg(
                "cluster labels must align with the influence columns",
            ));
        }
        match (self.unit, second) {
            (IndependenceUnit::Cluster, None) => {
                self.require_cluster_count(distinct_count(first), "clusters")?;
            }
            (IndependenceUnit::Dyad, Some(second)) => {
                self.require_cluster_count(distinct_count(first), "first-endpoint clusters")?;
                self.require_cluster_count(distinct_count(second), "second-endpoint clusters")?;
                // Keep this public covariance route inside the same dyadic design class as
                // receipt(): a label in both endpoint roles breaks the two-way model.
                dyad_units(first, second)?;
            }
            _ => {
                return Err(refuse(
                    antecedent_core::reason_code!("invalid_argument"),
                    "cluster_dml.second_ids_unexpected",
                    "the labels do not match the declared independence unit",
                ));
            }
        }
        let variance = |psi: &[f64]| -> Result<f64, EstimationError> {
            let se = match second {
                None => crate::se::cluster_influence_se(psi, first)?,
                Some(second) => {
                    crate::se::multiway_influence_se(psi, &[first.to_vec(), second.to_vec()])?
                }
            };
            Ok(se * se)
        };
        let dim = columns.len();
        let mut values = vec![0.0; dim * dim];
        for j in 0..dim {
            values[j * dim + j] = variance(columns[j])?;
            for i in 0..j {
                let sum: Vec<f64> = columns[i].iter().zip(columns[j]).map(|(a, b)| a + b).collect();
                let diff: Vec<f64> =
                    columns[i].iter().zip(columns[j]).map(|(a, b)| a - b).collect();
                let cov = (variance(&sum)? - variance(&diff)?) / 4.0;
                values[j * dim + i] = cov;
                values[i * dim + j] = cov;
            }
        }
        Ok(JointCovariance { dim, values: Arc::from(values) })
    }

    /// The standard error of a cluster-DML score table with its receipt: the cluster
    /// sandwich for a cluster unit, the two-way Cameron-Gelbach-Miller variance for a dyadic
    /// one (`second` carries the second-endpoint labels; it must be `None` for a cluster
    /// unit).
    ///
    /// `clusters` are the (first-endpoint) labels of the table's rows. The table must hold
    /// the mean functional's two arm columns and every fold unit's rows (a cluster, or a
    /// connected component of the endpoint graph) must lie in one fold; a table whose units
    /// were split across folds is not cluster-DML evidence and is refused.
    ///
    /// # Errors
    ///
    /// A label/row mismatch, too few clusters, a unit split across folds, a table that is
    /// not a single mean contrast, `dyadic_dependence_not_licensed` for an endpoint label in
    /// both roles, or (dyadic) a two-way variance that is materially negative, which is
    /// reported as an error and never truncated to zero.
    pub fn receipt(
        &self,
        table: &ScoreTable,
        clusters: &[u32],
        second: Option<&[u32]>,
    ) -> Result<ClusterDmlReceipt, EstimationError> {
        if clusters.len() != table.n_rows || table.fold_ids.len() != table.n_rows {
            return Err(EstimationError::data_msg(
                "cluster labels and fold ids must align with the score table rows",
            ));
        }
        if table.columns.len() != 2 {
            return Err(EstimationError::unsupported(
                "the cluster-DML receipt covers the mean functional's two arm columns",
            ));
        }
        let found = distinct_count(clusters);
        let psi_of = || -> Result<Vec<f64>, EstimationError> {
            Ok(table.column(0)?.iter().zip(table.column(1)?).map(|(&a, &b)| b - a).collect())
        };
        let (units, n_second, se) = match (self.unit, second) {
            (IndependenceUnit::Cluster, None) => {
                self.require_cluster_count(found, "clusters")?;
                require_whole_clusters_per_fold(clusters, &table.fold_ids)?;
                (clusters.to_vec(), None, crate::se::cluster_influence_se(&psi_of()?, clusters)?)
            }
            (IndependenceUnit::Cluster, Some(_)) => {
                return Err(refuse(
                    antecedent_core::reason_code!("invalid_argument"),
                    "cluster_dml.second_ids_unexpected",
                    "a cluster unit takes one label per row",
                ));
            }
            (IndependenceUnit::Dyad, None) => {
                return Err(refuse(
                    antecedent_core::reason_code!("required_option_missing"),
                    "cluster_dml.second_ids_missing",
                    "a dyadic receipt needs the second-endpoint label of every row",
                ));
            }
            (IndependenceUnit::Dyad, Some(second)) => {
                if second.len() != table.n_rows {
                    return Err(EstimationError::data_msg(
                        "second-endpoint labels must align with the score table rows",
                    ));
                }
                let found_second = distinct_count(second);
                self.require_cluster_count(found, "first-endpoint clusters")?;
                self.require_cluster_count(found_second, "second-endpoint clusters")?;
                let units = dyad_units(clusters, second)?;
                require_whole_clusters_per_fold(&units, &table.fold_ids)?;
                let se = crate::se::multiway_influence_se(
                    &psi_of()?,
                    &[clusters.to_vec(), second.to_vec()],
                )?;
                (units, Some(found_second), se)
            }
        };
        let reference_df = n_second.map_or(found, |g| g.min(found)) - 1;
        Ok(ClusterDmlReceipt {
            independence_unit: self.unit,
            n_clusters: found,
            n_clusters_second: n_second,
            n_components: n_second.map(|_| distinct_count(&units)),
            reference_df,
            min_clusters: self.min_clusters,
            n_rows: table.n_rows,
            folds: table.n_folds as usize,
            unit_digest: unit_digest(&units),
            cluster_sandwich_se: se,
        })
    }
}

/// Declared unit, counts and the cluster-sandwich standard error of the cross-fitted scores.
///
/// For a cluster unit `cluster_sandwich_se` is `sqrt(G/(G-1) * sum_g S_g^2) / n` with `S_g`
/// the cluster sum of the centered contrast scores. For a dyadic unit it is
/// `sqrt(V_a + V_b - V_ab) / n` with each term the same sandwich over the first-endpoint,
/// second-endpoint and (first, second) cell sums (Cameron-Gelbach-Miller). Either is a
/// variance receipt of the stored scores under the declared independence unit, not an
/// interval and not a calibrated claim.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterDmlReceipt {
    /// The declared independence unit.
    pub independence_unit: IndependenceUnit,
    /// Distinct clusters (first-endpoint labels for a dyadic unit).
    pub n_clusters: usize,
    /// Distinct second-endpoint labels; `None` for a cluster unit.
    pub n_clusters_second: Option<usize>,
    /// Connected components of the endpoint graph that own the folds; `None` for a cluster
    /// unit.
    pub n_components: Option<usize>,
    /// The few-cluster reference convention `min(G_a, G_b) - 1` (`G - 1` for a cluster
    /// unit): the degrees of freedom a t reference would use. The facade's published interval
    /// uses this as its Student-t reference (`EffectEstimate::se_reference_df`).
    pub reference_df: usize,
    /// The declared minimum the count satisfied.
    pub min_clusters: usize,
    /// Complete-case rows.
    pub n_rows: usize,
    /// Cross-fit folds.
    pub folds: usize,
    /// Fingerprint of the sorted distinct fold-unit labels (row-order invariant).
    pub unit_digest: String,
    /// Cluster-sandwich standard error of the mean contrast.
    pub cluster_sandwich_se: f64,
}

/// Seeded whole-cluster fold plan: every distinct cluster is dealt to one fold, so a fold
/// depends on the seed and the cluster label, never on row order or on how many rows a
/// cluster has. Clusters are not arm-stratified (arms vary inside a cluster).
///
/// # Errors
///
/// Fewer than two folds, or fewer distinct clusters than folds.
pub fn cluster_fold_plan(
    clusters: &[u32],
    folds: usize,
    seed: u64,
) -> Result<Vec<u32>, EstimationError> {
    crate::learn_nuisance::crossfit_fold_plan(&vec![0; clusters.len()], clusters, folds, seed)
}

/// The route is licensed for the untrimmed `AllObserved` mean ATE only.
pub(crate) fn require_scope(
    mean_functional: bool,
    population: &TargetPopulation,
    overlap: OverlapPolicy,
) -> Result<(), EstimationError> {
    if !mean_functional
        || !matches!(population, TargetPopulation::AllObserved)
        || trim_of(overlap).is_some()
    {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "cluster_dml.scope",
            "cluster-DML AIPW is licensed only for the untrimmed AllObserved mean ATE: other \
             functionals and populations, and trimmed fits, refit full-sample nuisances that \
             have no whole-cluster cross-fit",
        ));
    }
    Ok(())
}

/// The stop of a cancelled whole-cluster fit: no estimate or score table is reported and the
/// stop is not a verdict on the data.
pub(crate) fn cancelled() -> EstimationError {
    refuse(
        antecedent_core::reason_code!("cancelled_no_claim"),
        "cluster_dml.cancelled",
        "the cluster-DML cross-fit was cancelled before every fold was fit; no estimate is \
         reported and the stop is not a verdict on the data",
    )
}

/// A penalized propensity keeps points only and its remainder is not shown negligible, so it
/// is not combined with the cluster claim.
pub(crate) fn require_unpenalized(penalized: bool) -> Result<(), EstimationError> {
    if penalized {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "cluster_dml.penalized_propensity",
            "a penalized propensity publishes a point only and is not combined with the \
             cluster-DML declaration",
        ));
    }
    Ok(())
}

/// The counts behind a cluster-count refusal.
struct CountFacts {
    reason: &'static str,
    /// Clusters (or components) found.
    found: usize,
    /// Fewest the declaration needs; 0 when the refusal is not a shortfall.
    minimum: usize,
    remedy: &'static str,
}

/// [`refuse`] that also records the counts as structured fields (stage, reason, cluster count,
/// the minimum when the refusal is a shortfall, remedy). The message is `refuse`'s.
fn refuse_counted(
    code: &'static str,
    detail: &str,
    message: &str,
    facts: &CountFacts,
) -> EstimationError {
    EstimationError::refused_with_fields(
        code,
        format!("{detail}: {message}"),
        RefusalFields {
            stage: Some("cluster_dml".to_string()),
            reason: Some(facts.reason.to_string()),
            cluster_count: Some(u64::try_from(facts.found).unwrap_or(u64::MAX)),
            cluster_minimum: (facts.minimum > 0)
                .then(|| u64::try_from(facts.minimum).unwrap_or(u64::MAX)),
            remedy: Some(facts.remedy.to_string()),
            ..RefusalFields::default()
        },
    )
}

/// The route's standard error is its own cluster sandwich: a bootstrap (row resampling would
/// break clusters) or another analytic SE kind is refused instead of ignored.
pub(crate) fn refuse_interval_request(
    bootstrap_replicates: u32,
    se_kind: AnalyticSeKind,
) -> Result<(), EstimationError> {
    if bootstrap_replicates != 0 || !matches!(se_kind, AnalyticSeKind::Homoskedastic) {
        return Err(refuse(
            antecedent_core::reason_code!("cluster_interval_not_licensed"),
            "cluster_dml.interval_withheld",
            "cluster-DML AIPW reports its own cluster-sandwich standard error (a row \
             bootstrap would split clusters and another se_kind is ambiguous); set \
             bootstrap_replicates to 0 and keep the default se_kind to receive the point \
             estimate, the score table and the cluster-sandwich standard error",
        ));
    }
    Ok(())
}

/// Provenance suffix a cluster-DML score table carries.
pub(crate) fn provenance_suffix(units: &[u32], unit: IndependenceUnit) -> String {
    let counted = match unit {
        IndependenceUnit::Cluster => "clusters",
        IndependenceUnit::Dyad => "components",
    };
    format!(
        "{CLUSTER_DML_PROVENANCE_TAG}unit={};{counted}={};digest={}",
        unit.as_str(),
        distinct_count(units),
        unit_digest(units)
    )
}

/// Whether a score table's provenance marks whole-cluster cross-fitting.
pub(crate) fn provenance_marks_cluster_units(provenance: &str) -> bool {
    provenance.contains(CLUSTER_DML_PROVENANCE_TAG)
}

fn check_min_clusters(min_clusters: usize) -> Result<(), EstimationError> {
    if min_clusters < MIN_CLUSTERS_FLOOR {
        return Err(refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "cluster_dml.invalid_min_clusters",
            &format!(
                "the declared minimum cluster count {min_clusters} is below the floor \
                 {MIN_CLUSTERS_FLOOR}"
            ),
        ));
    }
    Ok(())
}

/// Connected component of the endpoint graph for every row, named by the smallest endpoint
/// label of the component. Both endpoint labels live in one namespace and every row joins its
/// two endpoints (union-find), so rows that share an endpoint, directly or through a chain of
/// rows, get one component; the name does not depend on row order.
///
/// `first` and `second` must have equal length (callers validate it).
#[must_use]
pub fn endpoint_components(first: &[u32], second: &[u32]) -> Vec<u32> {
    let mut labels: Vec<u32> = first.iter().chain(second).copied().collect();
    labels.sort_unstable();
    labels.dedup();
    let slot: BTreeMap<u32, usize> =
        labels.iter().enumerate().map(|(i, &label)| (label, i)).collect();
    let mut parent: Vec<usize> = (0..labels.len()).collect();
    for (a, b) in first.iter().zip(second) {
        let (ra, rb) = (find_root(&mut parent, slot[a]), find_root(&mut parent, slot[b]));
        // The smaller slot (labels are sorted) stays the root, so a component's root is its
        // smallest label whatever the order the rows arrive in.
        if ra != rb {
            parent[ra.max(rb)] = ra.min(rb);
        }
    }
    first.iter().map(|a| labels[find_root(&mut parent, slot[a])]).collect()
}

fn find_root(parent: &mut [usize], mut node: usize) -> usize {
    while parent[node] != node {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    node
}

/// Fold unit of every dyad: its connected component, named by the smallest first-endpoint
/// label in it (so that components that are exactly the first-endpoint clusters carry the
/// cluster labels themselves). Endpoint labels must not appear in both roles: the two-way
/// construction treats the first and second labels as two dimensions, and an entity that is
/// a first endpoint of some rows and a second endpoint of others would be dependent across
/// the two dimensions in a way the construction does not model.
fn dyad_units(first: &[u32], second: &[u32]) -> Result<Vec<u32>, EstimationError> {
    let first_labels: BTreeSet<u32> = first.iter().copied().collect();
    if let Some(shared) = second.iter().find(|label| first_labels.contains(label)) {
        return Err(refuse(
            antecedent_core::reason_code!("dyadic_dependence_not_licensed"),
            "cluster_dml.dyadic_shared_namespace",
            &format!(
                "label {shared} is both a first and a second endpoint: an entity in both roles \
                 is dependent across the two dimensions, which the two-way construction does \
                 not model; give the two endpoint roles separate label sets"
            ),
        ));
    }
    let component = endpoint_components(first, second);
    let mut name: BTreeMap<u32, u32> = BTreeMap::new();
    for (&c, &a) in component.iter().zip(first) {
        name.entry(c).and_modify(|smallest| *smallest = (*smallest).min(a)).or_insert(a);
    }
    Ok(component.iter().map(|c| name[c]).collect())
}

/// Typed refusal of a cluster or dependence declaration on a flexible-learner estimator
/// (`dml`, `dr.learner`, `causal.forest`), whose configurations carry no dependence option.
///
/// The refusal is `route_not_supported` (`cluster_dml.flexible_learner_closed`).
#[must_use]
pub fn flexible_learner_dependence_refusal(estimator_id: &str, option: &str) -> EstimationError {
    refuse(
        antecedent_core::reason_code!("route_not_supported"),
        "cluster_dml.flexible_learner_closed",
        &format!(
            "estimator {estimator_id:?} does not support the dependence option {option:?}: \
             its cross-fitting and inference are not justified under clustered or dyadic \
             dependence, so the declaration is refused rather than ignored; use \
             Aipw(bootstrap=0, cluster_dml=ClusterDml(...)) for the cluster-aware point"
        ),
    )
}

fn distinct_count(ids: &[u32]) -> usize {
    ids.iter().collect::<BTreeSet<_>>().len()
}

/// FNV-1a over the sorted distinct labels: a fingerprint, not a cryptographic digest (the
/// estimator identity digests the full label vector).
fn unit_digest(ids: &[u32]) -> String {
    let distinct: BTreeSet<u32> = ids.iter().copied().collect();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for id in distinct {
        for byte in id.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

fn require_whole_clusters_per_fold(
    clusters: &[u32],
    fold_ids: &[u32],
) -> Result<(), EstimationError> {
    let mut fold_of: BTreeMap<u32, u32> = BTreeMap::new();
    for (&cluster, &fold) in clusters.iter().zip(fold_ids) {
        let first = *fold_of.entry(cluster).or_insert(fold);
        if first != fold {
            return Err(EstimationError::data_msg(format!(
                "cluster {cluster} is split across cross-fit folds {first} and {fold}: its rows \
                 are not independent of the nuisances that score them"
            )));
        }
    }
    Ok(())
}
