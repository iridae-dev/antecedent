//! Retained two-period panel `DiD` execution.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, CausalExprArena, CausalQuery, DerivationTrace, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EstimatorId, ExecutionContext, GraphClass,
    IdentificationPerformanceRecord, IdentificationResult, IdentifiedEstimand,
    IdentifiedExecuteFinish, IdentifierId, InferenceMode, Instant, PhysicalExecutionPlan,
    RandomizedTrialOperation, RandomizedTrialQuery, RefuteSuite, Strategy, Study, StudyResult,
    TableView, TabularData, elapsed_ns, finish_identified_execute_with_context,
};
use crate::analysis::missing_data::{self, MissingRoute};
use antecedent_data::{MissingDeclaration, MissingPolicy};
use std::collections::{BTreeMap, BTreeSet};

mod bayesian;
mod weighted;

pub(super) use bayesian::bayesian_panel_did_plan_ids;
use super::BayesianConfig;

/// Why `design` cannot execute `policy`, if it cannot.
fn design_refusal(
    design: antecedent_core::DidSamplingDesign,
    policy: MissingPolicy,
) -> Option<&'static str> {
    use antecedent_core::DidSamplingDesign;
    match (policy, design) {
        (MissingPolicy::ErrorOnMissing, _) => None,
        (_, DidSamplingDesign::AugmentedPanel) => Some(
            "the augmented panel's supplied nuisance predictions are row-aligned columns and cannot be restricted",
        ),
        (MissingPolicy::BalancedPanel, DidSamplingDesign::RepeatedCrossSection) => {
            Some("a repeated cross-section has no panel to balance; each row is its own subject")
        }
        _ => None,
    }
}

fn panel_policy_refusal(policy: MissingPolicy, reason: &str) -> CausalError {
    CausalError::Estimate(antecedent_estimate::missing_policy::policy_not_applicable(
        policy,
        MissingRoute::Panel.label(),
        reason,
        MissingRoute::Panel.supported(),
    ))
}

/// Missing-data handling frozen with a panel operation.
#[derive(Clone, Debug, Default)]
pub(crate) struct PanelMissing {
    declaration: Option<MissingDeclaration>,
    /// The declared policy removed rows. No coverage record measures a restricted panel, so
    /// no interval is published.
    restricted: bool,
    /// Each contrast uses the units observed in its own periods (unbalanced panel).
    available_case: bool,
    /// The study's Bayesian configuration: the retained operation reports a posterior, not a
    /// sampling interval.
    bayesian: Option<BayesianConfig>,
}

pub(crate) type CheckedPanelDidOperation =
    RandomizedTrialOperation<antecedent_core::PanelDidQuery, PanelMissing>;

impl RandomizedTrialQuery for antecedent_core::PanelDidQuery {
    const OPERATION: &'static str = "CheckedPanelDidOperation";
    const REFRESH_MISMATCH: &'static str =
        "panel DiD refresh requires the prepared schema and row order";
}

/// Per-subject cohort, cluster and outcome by period of a staggered `DiD` panel.
type StaggeredSubjects<'a> = BTreeMap<&'a str, (i64, &'a str, BTreeMap<i64, f64>)>;

/// Effect, cluster-score SE and interval of a balanced two-period panel `DiD`.
struct BalancedDidFit {
    effect: f64,
    se: f64,
    groups: usize,
    interval_supported: bool,
    interval_95: Option<[f64; 2]>,
    cluster_counts: [usize; 2],
}

/// Difference of group-mean change scores with the cluster-score sandwich SE.
fn balanced_did_fit(
    changes: Vec<(bool, &str, f64)>,
    sums: [f64; 2],
    counts: [usize; 2],
    interval_allowed: bool,
) -> Result<BalancedDidFit, CausalError> {
    let means = [sums[0] / counts[0] as f64, sums[1] / counts[1] as f64];
    let effect = means[1] - means[0];
    let mut scores: BTreeMap<&str, f64> = BTreeMap::new();
    let mut clusters: [BTreeSet<&str>; 2] = Default::default();
    for (treated, cluster, change) in changes {
        let g = usize::from(treated);
        clusters[g].insert(cluster);
        *scores.entry(cluster).or_default() +=
            (if treated { 1.0 } else { -1.0 }) * (change - means[g]) / counts[g] as f64;
    }
    if clusters.iter().any(|set| set.len() < 2) {
        return Err(CausalError::Unsupported {
            message: "cluster SE requires at least two clusters in each group",
        });
    }
    if !clusters[0].is_disjoint(&clusters[1]) {
        return Err(CausalError::Unsupported {
            message: "panel DiD requires clusters nested within treatment groups for cluster inference",
        });
    }
    let g = scores.len();
    let variance = g as f64 / (g - 1) as f64 * scores.values().map(|v| v * v).sum::<f64>();
    let se = variance.sqrt();
    let interval_supported = interval_allowed
        && clusters.iter().all(|set| set.len() >= 30)
        && se.is_finite()
        && se > 0.0;
    let interval_95 = interval_supported.then(|| {
        let radius = antecedent_stats::normal_ppf(0.975) * se;
        [effect - radius, effect + radius]
    });
    Ok(BalancedDidFit {
        effect,
        se,
        groups: g,
        interval_supported,
        interval_95,
        cluster_counts: [clusters[0].len(), clusters[1].len()],
    })
}

/// Descriptive joint pre-trend statistic of an event study, or why it is unavailable.
fn pretrend_diagnostic(
    effects: &[antecedent_estimate::staggered_event_study::EventTimeEffect],
) -> Diagnostic {
    match antecedent_estimate::staggered_event_study::pretrend_falsification_statistic(effects) {
        Ok(statistic) => Diagnostic::new(
            "diagnostic.quasi.event_study.pretrend_joint_max_cluster_z",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "descriptive joint pre-period maximum |effect/cluster SE|={:.8} across {} non-reference cohort-specific leads (minimum {} clusters); no calibrated p-value or cutoff; a small value cannot establish parallel untreated trends",
                statistic.max_absolute_cluster_studentized_lead,
                statistic.leads,
                statistic.min_clusters,
            ),
        ),
        Err(reason) => Diagnostic::new(
            "diagnostic.quasi.event_study.pretrend_joint_unavailable",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "joint pre-period cluster statistic unavailable: {reason}; parallel untreated trends remains a declared assumption"
            ),
        ),
    }
}

/// Event-time intervals with the cluster counts the support gate reads.
struct EventTimeIntervals {
    intervals: Vec<Option<[f64; 2]>>,
    control_cluster_count: usize,
    treated_clusters_per_cohort: BTreeMap<i64, usize>,
}

impl CheckedPanelDidOperation {
    pub(crate) fn checked(
        study: &Study,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
    ) -> Result<Self, CausalError> {
        let CausalQuery::PanelDid(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "panel DiD operation requires PanelDidQuery".into(),
            });
        };
        query.validate()?;
        if study.graph.class() != GraphClass::RandomizedTrial
            || study.structure_source != crate::support::StructureSource::RandomizedTrial
            || study.refute != RefuteSuite::None
            || study.bootstrap_replicates != 0
            || !study.custom_validators.is_empty()
            || study.graph_posterior.is_some()
            || study.tiered.is_some()
            || query.treated.len() != data.row_count()
        {
            return Err(CausalError::Unsupported {
                message: "panel DiD currently accepts only the graphless two-period panel query with point estimate and cluster standard error, or its Bayesian posterior",
            });
        }
        let bayesian = match &study.inference {
            InferenceMode::Frequentist => None,
            InferenceMode::Bayesian(config) => {
                super::bayesian_quasi_common::require_built_in_model(config, "Bayesian panel DiD")?;
                Some(config.clone())
            }
        };
        data.schema().get(query.outcome).map_err(CausalError::rejected)?;
        if let Some(nuisance) = &query.augmented {
            for variable in
                [nuisance.outcome_pre, nuisance.propensity, nuisance.untreated_change_prediction]
            {
                data.schema().get(variable).map_err(CausalError::rejected)?;
            }
        }
        let (plan_id, identifier, estimator) = if bayesian.is_some() {
            bayesian_panel_did_plan_ids(query.design)
        } else {
            panel_did_plan_ids(query.design)
        };
        if physical.logical.query != study.query
            || physical.logical.record.plan_id.as_ref() != plan_id
            || !physical.binding().names_wire(identifier, estimator)
        {
            return Err(CausalError::Compile {
                message: "panel DiD design differs from its compiled plan".into(),
            });
        }
        let (identification, estimand) = panel_did_identification(query);
        if let Some(declaration) = &study.missing {
            if let Some(reason) = design_refusal(query.design, declaration.policy) {
                return Err(panel_policy_refusal(declaration.policy, reason));
            }
        }
        let extra = PanelMissing {
            declaration: study.missing.clone(),
            bayesian,
            ..PanelMissing::default()
        };
        Ok(Self::freeze(study, query, data, physical, identification, estimand, extra))
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.begin_execute(data, ctx)?;
        let antecedent_data::ColumnView::Float64(column) =
            data.column(self.query.outcome).map_err(CausalError::from)?
        else {
            return Err(CausalError::Unsupported {
                message: "panel DiD outcome must be continuous",
            });
        };
        let y = column.values.as_slice();
        let observed: Vec<bool> = (0..y.len()).map(|i| column.validity.is_valid(i)).collect();
        if observed.iter().all(|seen| *seen) {
            let mut result = self.execute_rows(data, y, ctx)?;
            let keep = vec![true; y.len()];
            result.missing_data = Some(self.missing_report(data, &observed, &keep)?);
            return Ok(result);
        }
        let (keep, available_case) = self.restriction(data, &observed)?;
        let query = self.query.restrict_rows(&keep).ok_or(CausalError::Unsupported {
            message: "panel DiD cannot restrict rows of an augmented panel",
        })?;
        let restricted_y: Vec<f64> =
            y.iter().zip(&keep).filter(|(_, kept)| **kept).map(|(value, _)| *value).collect();
        if restricted_y.is_empty() {
            return Err(CausalError::Unsupported {
                message: "the declared missing-data policy leaves no panel rows",
            });
        }
        let restricted = Self {
            query,
            extra: PanelMissing { restricted: true, available_case, ..self.extra.clone() },
            ..self.clone()
        };
        let mut result = restricted.execute_rows(data, &restricted_y, ctx)?;
        result.missing_data = Some(self.missing_report(data, &observed, &keep)?);
        Ok(result)
    }

    /// The rows the declared policy keeps and whether each contrast then uses the units
    /// observed in its own periods; the refusal naming the first missing cell when no policy
    /// that handles missing outcomes is declared.
    fn restriction(
        &self,
        data: &TabularData,
        observed: &[bool],
    ) -> Result<(Vec<bool>, bool), CausalError> {
        use antecedent_core::DidSamplingDesign;
        let supported = MissingRoute::Panel.supported();
        let Some(declaration) = self
            .extra
            .declaration
            .as_ref()
            .filter(|declaration| declaration.policy != MissingPolicy::ErrorOnMissing)
        else {
            let cell = antecedent_data::first_missing_cell(data, &[self.query.outcome])?.ok_or(
                CausalError::Unsupported {
                    message: "panel DiD outcome validity changed during execution",
                },
            )?;
            return Err(CausalError::Estimate(
                antecedent_estimate::missing_policy::policy_required(
                    &cell,
                    MissingRoute::Panel.label(),
                    &supported[1..],
                ),
            ));
        };
        if let Some(reason) = design_refusal(self.query.design, declaration.policy) {
            return Err(panel_policy_refusal(declaration.policy, reason));
        }
        let staggered = matches!(
            self.query.design,
            DidSamplingDesign::StaggeredGroupTime | DidSamplingDesign::StaggeredEventStudy
        );
        match (declaration.policy, self.query.design) {
            (MissingPolicy::AvailableCase, DidSamplingDesign::RepeatedCrossSection) => {
                Ok((observed.to_vec(), false))
            }
            (MissingPolicy::AvailableCase, _) if staggered => Ok((observed.to_vec(), true)),
            (MissingPolicy::BalancedPanel | MissingPolicy::AvailableCase, _) => {
                let mut complete: BTreeMap<&str, bool> = BTreeMap::new();
                for (subject, seen) in self.query.subjects.iter().zip(observed) {
                    let entry = complete.entry(subject.as_ref()).or_insert(true);
                    *entry = *entry && *seen;
                }
                Ok((
                    self.query.subjects.iter().map(|subject| complete[subject.as_ref()]).collect(),
                    false,
                ))
            }
            _ => Err(panel_policy_refusal(declaration.policy, "the route has no estimator for it")),
        }
    }

    /// Policy, assumption and accounting: rows and units, with the treated and comparison
    /// groups.
    fn missing_report(
        &self,
        data: &TabularData,
        observed: &[bool],
        keep: &[bool],
    ) -> Result<crate::result::MissingDataReport, CausalError> {
        let (policy, declared) =
            missing_data::effective_policy(MissingRoute::Panel, self.extra.declaration.as_ref());
        let accounting = antecedent_data::RowAccounting::from_selection(
            data,
            data.storage().analysis_mask(),
            policy,
            &[self.query.outcome],
            keep,
        )?;
        let labels: Vec<Arc<str>> = self
            .query
            .treated
            .iter()
            .map(|treated| Arc::<str>::from(if *treated { "treated" } else { "comparison" }))
            .collect();
        let mut units: BTreeMap<&str, bool> = BTreeMap::new();
        for (subject, kept) in self.query.subjects.iter().zip(keep) {
            let entry = units.entry(subject.as_ref()).or_insert(false);
            *entry = *entry || *kept;
        }
        let dropped_units = units.values().filter(|kept| !**kept).count();
        let accounting =
            accounting.with_groups(&labels, keep, observed)?.with_units(units.len(), dropped_units);
        Ok(missing_data::report(MissingRoute::Panel, declared, Arc::from([]), accounting))
    }

    fn execute_rows(
        &self,
        data: &TabularData,
        y: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if self.extra.bayesian.is_some() {
            return self.execute_bayesian(data, y, ctx);
        }
        if data.storage().weights().is_some()
            && self.query.design != antecedent_core::DidSamplingDesign::BalancedPanel
        {
            return Err(CausalError::Estimate(
                antecedent_estimate::sampling_weights::unsupported_route(
                    "sampling weights are read by the balanced two-period panel DiD; staggered, \
                     repeated cross-section and augmented designs have no sampling-weighted \
                     contrast",
                ),
            ));
        }
        if self.query.design == antecedent_core::DidSamplingDesign::AugmentedPanel {
            return self.execute_augmented_panel(data, y);
        }
        if self.query.design == antecedent_core::DidSamplingDesign::StaggeredEventStudy {
            return self.execute_staggered_event_study(data, y);
        }
        if self.query.design == antecedent_core::DidSamplingDesign::StaggeredGroupTime {
            return self.execute_staggered_group_time(data, y);
        }
        if self.query.design == antecedent_core::DidSamplingDesign::RepeatedCrossSection {
            return self.execute_repeated_cross_section(data, y);
        }
        if let Some(weights) = data.storage().weights() {
            return self.execute_weighted_balanced(data, y, weights);
        }
        let (changes, sums, counts) = self.balanced_subject_changes(y)?;
        let BalancedDidFit {
            effect,
            se,
            groups: g,
            interval_supported,
            interval_95,
            cluster_counts,
        } = balanced_did_fit(changes, sums, counts, !self.extra.restricted)?;
        let identification = self.identification.clone();
        let assumptions = identification.required_assumptions.clone();
        let estimate = EffectEstimate::new(
            effect,
            if interval_supported { se } else { f64::NAN },
            assumptions,
            antecedent_estimate::OverlapPolicy::ExplicitOverride,
        )
        .with_se_kind(antecedent_estimate::AnalyticSeKind::Cluster);
        let started = Instant::now();
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish::new(&self.physical, identification, self.estimand.clone(), estimate, Strategy::new(IdentifierId::RandomizedDesign, EstimatorId::RandomizedHt), self.query.outcome, self.query.outcome).extra_diagnostics(vec![
                    Diagnostic::new(
                        "estimate.quasi.panel_did.cluster_se",
                        DiagnosticKind::Scientific,
                        DiagnosticSeverity::Info,
                        "two-period subject change-score DiD; cluster score sandwich SE; pointwise normal interval requires 30 independent clusters per arm",
                    ),
                    Diagnostic::new(
                        "identification.quasi.parallel_trends_untestable_two_periods",
                        DiagnosticKind::Scientific,
                        DiagnosticSeverity::Info,
                        "parallel untreated trends is a declared identifying assumption; two periods do not permit a pre-trend check",
                    ),
                ]).wall_time_ns(elapsed_ns(started)),
        );
        result.payload = crate::result::ResultPayload::PanelDid(crate::PanelDidEstimate {
            effect,
            standard_error: se,
            interval_95,
            treated_subjects: counts[1],
            comparison_subjects: counts[0],
            clusters: g,
            uncertainty: Arc::from(if interval_supported {
                "cluster_robust_normal_interval_independent_clusters"
            } else {
                "cluster_robust_standard_error_no_interval"
            }),
            event_time_effects: Arc::from([]),
            event_time_intervals_95: Arc::from([]),
            augmented: None,
        });
        if crate::support::license_if_graphless(
            panel_did_graphless_key(antecedent_core::DidSamplingDesign::BalancedPanel),
            crate::support::GraphlessAssignmentSupport {
                assignment_unit: "cluster",
                treated: cluster_counts[1],
                control: cluster_counts[0],
                interval_95_published: interval_95.is_some(),
                reported_intervals: usize::from(interval_95.is_some()),
                ..Default::default()
            },
        )
        .is_some()
        {
            result.support_status = Some(crate::support::CellStatus::Licensed);
        }
        result.treatment = None;
        result.rebind_interval(false);
        Ok(result)
    }

    /// Pre/post change score per subject, with the group change sums and subject counts.
    #[allow(
        clippy::type_complexity,
        reason = "the change tuples and the two group arrays are read only by `balanced_did_fit`"
    )]
    fn balanced_subject_changes(
        &self,
        y: &[f64],
    ) -> Result<(Vec<(bool, &str, f64)>, [f64; 2], [usize; 2]), CausalError> {
        #[allow(
            clippy::type_complexity,
            reason = "one-off per-subject accumulator tuple confined to this local map"
        )]
        let mut subjects: BTreeMap<
            &str,
            (Option<f64>, Option<f64>, Option<bool>, Option<&str>),
        > = BTreeMap::new();
        #[allow(
            clippy::needless_range_loop,
            reason = "index addresses multiple row-aligned query slices (y, subjects, treated)"
        )]
        for i in 0..y.len() {
            if !y[i].is_finite() {
                return Err(CausalError::Unsupported {
                    message: "panel DiD requires complete finite outcomes",
                });
            }
            let subject = self.query.subjects[i].as_ref();
            let entry = subjects.entry(subject).or_insert((None, None, None, None));
            if entry.2.is_some_and(|v| v != self.query.treated[i]) {
                return Err(CausalError::Compile {
                    message: "treatment must be stable within subject".into(),
                });
            }
            entry.2 = Some(self.query.treated[i]);
            if entry.3.is_some_and(|v| v != self.query.clusters[i].as_ref()) {
                return Err(CausalError::Compile {
                    message: "cluster must be stable within subject".into(),
                });
            }
            entry.3 = Some(self.query.clusters[i].as_ref());
            let slot = if self.query.post[i] { &mut entry.1 } else { &mut entry.0 };
            if slot.replace(y[i]).is_some() {
                return Err(CausalError::Compile {
                    message: "each subject must have exactly one pre and one post observation"
                        .into(),
                });
            }
        }
        let mut changes = Vec::with_capacity(subjects.len());
        let mut sums = [0.0; 2];
        let mut counts = [0usize; 2];
        for (pre, post, treated, cluster) in subjects.values() {
            let (Some(pre), Some(post), Some(treated), Some(cluster)) =
                (pre, post, treated, cluster)
            else {
                return Err(CausalError::Compile {
                    message: "panel DiD requires a balanced two-period panel".into(),
                });
            };
            let change = post - pre;
            let group = usize::from(*treated);
            sums[group] += change;
            counts[group] += 1;
            changes.push((*treated, *cluster, change));
        }
        if counts.contains(&0) {
            return Err(CausalError::Compile {
                message: "panel DiD requires treated and comparison subjects".into(),
            });
        }
        Ok((changes, sums, counts))
    }

    /// Pointwise 95% intervals of post-adoption event-time effects, with the cluster counts that
    /// license them.
    fn event_time_intervals(
        &self,
        effects: &[antecedent_estimate::staggered_event_study::EventTimeEffect],
    ) -> EventTimeIntervals {
        let control_clusters = self
            .query
            .clusters
            .iter()
            .zip(self.query.cohorts.iter())
            .filter(|(_, cohort)| **cohort == 0)
            .map(|(cluster, _)| cluster.as_ref())
            .collect::<BTreeSet<_>>();
        // Distinct treated clusters per cohort in a single pass, mirroring the
        // control-cluster set above. Indexed by cohort for each event-time
        // effect and for the representative cohort below, instead of rescanning
        // every row for every effect.
        let mut treated_clusters_per_cohort: BTreeMap<i64, BTreeSet<&str>> = BTreeMap::new();
        for (cluster, cohort) in self.query.clusters.iter().zip(self.query.cohorts.iter()) {
            treated_clusters_per_cohort.entry(*cohort).or_default().insert(cluster.as_ref());
        }
        let treated_cluster_count =
            |cohort: i64| treated_clusters_per_cohort.get(&cohort).map_or(0, BTreeSet::len);
        let intervals = effects
            .iter()
            .map(|effect| {
                if effect.event_time < 0 || self.extra.restricted {
                    return None;
                }
                antecedent_estimate::staggered_event_study::pointwise_interval_95(
                    effect,
                    treated_cluster_count(effect.cohort),
                    control_clusters.len(),
                )
            })
            .collect::<Vec<_>>();
        EventTimeIntervals {
            intervals,
            control_cluster_count: control_clusters.len(),
            treated_clusters_per_cohort: treated_clusters_per_cohort
                .into_iter()
                .map(|(cohort, clusters)| (cohort, clusters.len()))
                .collect(),
        }
    }

    /// Per-subject outcomes by period, refusing unbalanced panels or unstable cohort and cluster.
    fn staggered_balanced_subjects(
        &self,
        y: &[f64],
        baseline: i64,
        period: i64,
    ) -> Result<StaggeredSubjects<'_>, CausalError> {
        let mut all_periods = BTreeSet::new();
        let mut subjects: BTreeMap<&str, (i64, &str, BTreeMap<i64, f64>)> = BTreeMap::new();
        for (i, value) in y.iter().enumerate() {
            if !value.is_finite() {
                return Err(CausalError::Unsupported {
                    message: "staggered DiD requires complete finite outcomes",
                });
            }
            let id = self.query.subjects[i].as_ref();
            let cluster = self.query.clusters[i].as_ref();
            let g = self.query.cohorts[i];
            let t = self.query.periods[i];
            all_periods.insert(t);
            let entry = subjects.entry(id).or_insert_with(|| (g, cluster, BTreeMap::new()));
            if entry.0 != g || entry.1 != cluster {
                return Err(CausalError::Compile {
                    message: "cohort and cluster must be stable within subject".into(),
                });
            }
            if entry.2.insert(t, *value).is_some() {
                return Err(CausalError::Compile {
                    message: "each subject must have one outcome per period".into(),
                });
            }
        }
        if !all_periods.contains(&baseline)
            || !all_periods.contains(&period)
            || (!self.extra.available_case
                && subjects.values().any(|unit| {
                    unit.2.len() != all_periods.len()
                        || all_periods.iter().any(|t| !unit.2.contains_key(t))
                }))
        {
            return Err(CausalError::Compile { message: "staggered DiD requires a balanced panel with the target and immediately preceding adoption period".into() });
        }
        Ok(subjects)
    }

    /// Group-by-period outcome sums, counts and clusters, refusing empty or thin cells.
    #[allow(
        clippy::type_complexity,
        reason = "the three cell arrays are read only by the four-cell estimator"
    )]
    fn repeated_cross_section_cells(
        &self,
        y: &[f64],
    ) -> Result<([[f64; 2]; 2], [[usize; 2]; 2], [[BTreeSet<&str>; 2]; 2]), CausalError> {
        let mut subjects = BTreeSet::new();
        let mut sums = [[0.0; 2]; 2];
        let mut counts = [[0usize; 2]; 2];
        let mut cell_clusters: [[BTreeSet<&str>; 2]; 2] = Default::default();
        for (i, value) in y.iter().enumerate() {
            if !value.is_finite() {
                return Err(CausalError::Unsupported {
                    message: "repeated-cross-section DiD requires complete finite outcomes",
                });
            }
            if !subjects.insert(self.query.subjects[i].as_ref()) {
                return Err(CausalError::Compile {
                    message: "repeated-cross-section DiD requires one row per subject".into(),
                });
            }
            let group = usize::from(self.query.treated[i]);
            let period = usize::from(self.query.post[i]);
            sums[group][period] += value;
            counts[group][period] += 1;
            cell_clusters[group][period].insert(self.query.clusters[i].as_ref());
        }
        if counts.iter().flatten().any(|count| *count == 0) {
            return Err(CausalError::Compile {
                message: "repeated-cross-section DiD requires observations in all four group-period cells".into(),
            });
        }
        if cell_clusters.iter().flatten().any(|clusters| clusters.len() < 2) {
            return Err(CausalError::Unsupported {
                message: "repeated-cross-section cluster SE requires at least two clusters in each group-period cell",
            });
        }
        Ok((sums, counts, cell_clusters))
    }

    fn execute_augmented_panel(
        &self,
        data: &TabularData,
        post: &[f64],
    ) -> Result<StudyResult, CausalError> {
        let Some(nuisance) = self.query.augmented.as_ref() else {
            return Err(CausalError::Compile {
                message: "augmented panel DiD requires the augmented nuisance block".into(),
            });
        };
        let numeric_column = |variable| -> Result<Vec<f64>, CausalError> {
            match data.column(variable).map_err(CausalError::from)? {
                antecedent_data::ColumnView::Float64(column) => {
                    Ok(column.values.as_slice().to_vec())
                }
                _ => Err(CausalError::Unsupported {
                    message: "augmented panel DiD requires continuous outcome and nuisance columns",
                }),
            }
        };
        let pre = numeric_column(nuisance.outcome_pre)?;
        let propensity = numeric_column(nuisance.propensity)?;
        let prediction = numeric_column(nuisance.untreated_change_prediction)?;
        let fit = antecedent_estimate::augmented_panel_did::estimate(
            &pre,
            post,
            &self.query.treated,
            &propensity,
            &prediction,
        )
        .map_err(|message| CausalError::Compile { message: message.into() })?;
        let estimate = EffectEstimate::new(
            fit.effect,
            f64::NAN,
            self.identification.required_assumptions.clone(),
            antecedent_estimate::OverlapPolicy::ExplicitOverride,
        );
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish::new(&self.physical, self.identification.clone(), self.estimand.clone(), estimate, Strategy::new(IdentifierId::RandomizedDesign, EstimatorId::RandomizedHt), self.query.outcome, self.query.outcome).extra_diagnostics(vec![Diagnostic::new(
                    "estimate.quasi.augmented_panel_did.supplied_nuisance",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "supplied propensity and untreated-change predictions; cross-fitting is caller-declared, not verified; no calibrated interval",
                )]),
        );
        result.payload = crate::result::ResultPayload::PanelDid(crate::PanelDidEstimate {
            effect: fit.effect,
            // The field is a wire-compatible placeholder, not a standard error.
            standard_error: 0.0,
            interval_95: None,
            treated_subjects: fit.treated_subjects,
            comparison_subjects: fit.control_subjects,
            clusters: self.query.clusters.iter().collect::<BTreeSet<_>>().len(),
            uncertainty: Arc::from("point_only_no_standard_error"),
            event_time_effects: Arc::from([]),
            event_time_intervals_95: Arc::from([]),
            augmented: Some((
                fit.propensity_min,
                fit.propensity_max,
                fit.effective_control_sample_size,
                nuisance.predictions_cross_fitted,
            )),
        });
        result.treatment = None;
        Ok(result)
    }

    fn execute_staggered_event_study(
        &self,
        data: &TabularData,
        y: &[f64],
    ) -> Result<StudyResult, CausalError> {
        let effects = antecedent_estimate::staggered_event_study::estimate_with(
            y,
            &self.query.subjects,
            &self.query.periods,
            &self.query.cohorts,
            &self.query.clusters,
            self.extra.available_case,
        )
        .map_err(|message| CausalError::Compile { message })?;
        let EventTimeIntervals {
            intervals: event_time_intervals_95,
            control_cluster_count,
            treated_clusters_per_cohort,
        } = self.event_time_intervals(&effects);
        let treated_cluster_count =
            |cohort: i64| treated_clusters_per_cohort.get(&cohort).copied().unwrap_or(0);
        let representative_index =
            effects.iter().position(|effect| effect.event_time >= 0).ok_or_else(|| {
                CausalError::Compile { message: "event study has no post-adoption contrast".into() }
            })?;
        let representative = &effects[representative_index];
        let representative_interval = event_time_intervals_95[representative_index];
        let any_interval = event_time_intervals_95.iter().any(Option::is_some);
        // License the post-adoption event-time intervals only when the interval
        // is actually published and each side clears the independent-cluster
        // support gate. Capture the support before `effects` moves into the
        // result. Pre-adoption leads and event time -1 stay point-only.
        let representative_treated_clusters = treated_cluster_count(representative.cohort);
        let published_intervals =
            event_time_intervals_95.iter().filter(|interval| interval.is_some()).count();
        let graphless_licensed = representative_interval.is_some()
            && !self.extra.restricted
            && crate::support::license_if_graphless(
                panel_did_graphless_key(antecedent_core::DidSamplingDesign::StaggeredEventStudy),
                crate::support::GraphlessAssignmentSupport {
                    assignment_unit: "cluster",
                    treated: representative_treated_clusters,
                    control: control_cluster_count,
                    interval_95_published: true,
                    reported_intervals: published_intervals,
                    ..Default::default()
                },
            )
            .is_some();
        let pretrend_diagnostic = pretrend_diagnostic(&effects);
        let estimate = EffectEstimate::new(
            representative.effect,
            if representative_interval.is_some() {
                representative.standard_error
            } else {
                f64::NAN
            },
            self.identification.required_assumptions.clone(),
            antecedent_estimate::OverlapPolicy::ExplicitOverride,
        )
        .with_se_kind(antecedent_estimate::AnalyticSeKind::Cluster);
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish::new(&self.physical, self.identification.clone(), self.estimand.clone(), estimate, Strategy::new(IdentifierId::RandomizedDesign, EstimatorId::RandomizedHt), self.query.outcome, self.query.outcome).extra_diagnostics(vec![
                    Diagnostic::new(
                        if any_interval {
                            "estimate.quasi.staggered_event_study.pointwise_95"
                        } else {
                            "estimate.quasi.staggered_event_study.pointwise_cluster_se"
                        },
                        DiagnosticKind::Scientific,
                        DiagnosticSeverity::Info,
                        "cohort-specific event times include descriptive preperiod contrasts without intervals; -1 is omitted; supported post-adoption contrasts have separate pointwise 95% cluster intervals, never simultaneous bands; parallel trends are assumed rather than tested",
                    ),
                    pretrend_diagnostic,
                ]),
        );
        result.payload = crate::result::ResultPayload::PanelDid(crate::PanelDidEstimate {
            effect: representative.effect,
            standard_error: representative.standard_error,
            interval_95: representative_interval,
            treated_subjects: representative.treated_subjects,
            comparison_subjects: representative.comparison_subjects,
            clusters: representative.clusters,
            uncertainty: Arc::from(if any_interval {
                "event_time_pointwise_normal_intervals_independent_clusters"
            } else {
                "cluster_robust_standard_error_no_interval"
            }),
            event_time_effects: effects.into(),
            event_time_intervals_95: event_time_intervals_95.into(),
            augmented: None,
        });
        if graphless_licensed {
            result.support_status = Some(crate::support::CellStatus::Licensed);
        }
        result.treatment = None;
        result.rebind_interval(false);
        Ok(result)
    }

    fn execute_staggered_group_time(
        &self,
        data: &TabularData,
        y: &[f64],
    ) -> Result<StudyResult, CausalError> {
        let Some((cohort, period)) = self.query.target else {
            return Err(CausalError::Compile {
                message: "staggered group-time DiD requires a cohort and period target".into(),
            });
        };
        let baseline = cohort - 1;
        let subjects = self.staggered_balanced_subjects(y, baseline, period)?;
        let at_both = |unit: &(i64, &str, BTreeMap<i64, f64>)| {
            unit.2.contains_key(&baseline) && unit.2.contains_key(&period)
        };
        let selected: [Vec<_>; 2] = [
            subjects.values().filter(|unit| unit.0 == 0 && at_both(unit)).collect(),
            subjects.values().filter(|unit| unit.0 == cohort && at_both(unit)).collect(),
        ];
        if selected.iter().any(Vec::is_empty) {
            return Err(CausalError::Compile { message: "staggered DiD requires never-treated controls and subjects in the selected adoption cohort".into() });
        }
        let means: [f64; 2] = std::array::from_fn(|group| {
            let units = &selected[group];
            units.iter().map(|unit| unit.2[&period] - unit.2[&baseline]).sum::<f64>()
                / units.len() as f64
        });
        let effect = means[1] - means[0];
        let mut scores: BTreeMap<&str, f64> = BTreeMap::new();
        let mut group_clusters: [BTreeSet<&str>; 2] = Default::default();
        for group in 0..2 {
            for unit in &selected[group] {
                group_clusters[group].insert(unit.1);
                let change = unit.2[&period] - unit.2[&baseline];
                *scores.entry(unit.1).or_default() += (if group == 1 { 1.0 } else { -1.0 })
                    * (change - means[group])
                    / selected[group].len() as f64;
            }
        }
        if group_clusters.iter().any(|set| set.len() < 2) {
            return Err(CausalError::Unsupported {
                message: "staggered DiD cluster SE requires at least two clusters in the selected cohort and never-treated controls",
            });
        }
        let cluster_count = scores.len();
        let variance = cluster_count as f64 / (cluster_count - 1) as f64
            * scores.values().map(|score| score * score).sum::<f64>();
        let se = variance.sqrt();
        let estimate = EffectEstimate::new(
            effect,
            // The selected group-time SE is pointwise and has no calibrated interval.
            f64::NAN,
            self.identification.required_assumptions.clone(),
            antecedent_estimate::OverlapPolicy::ExplicitOverride,
        );
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish::new(&self.physical, self.identification.clone(), self.estimand.clone(), estimate, Strategy::new(IdentifierId::RandomizedDesign, EstimatorId::RandomizedHt), self.query.outcome, self.query.outcome).extra_diagnostics(vec![Diagnostic::new(
                    "estimate.quasi.staggered_group_time.cluster_se",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    format!(
                        "cohort {cohort}, period {period}, baseline {baseline}; never-treated controls; cluster score sandwich SE; no interval or parallel-trends test"
                    ),
                )]),
        );
        result.payload = crate::result::ResultPayload::PanelDid(crate::PanelDidEstimate {
            effect,
            standard_error: se,
            interval_95: None,
            treated_subjects: selected[1].len(),
            comparison_subjects: selected[0].len(),
            clusters: cluster_count,
            uncertainty: Arc::from("cluster_robust_standard_error_no_interval"),
            event_time_effects: Arc::from([]),
            event_time_intervals_95: Arc::from([]),
            augmented: None,
        });
        result.treatment = None;
        Ok(result)
    }

    fn execute_repeated_cross_section(
        &self,
        data: &TabularData,
        y: &[f64],
    ) -> Result<StudyResult, CausalError> {
        let (sums, counts, cell_clusters) = self.repeated_cross_section_cells(y)?;
        let group_clusters: [BTreeSet<&str>; 2] = std::array::from_fn(|group| {
            cell_clusters[group][0].union(&cell_clusters[group][1]).copied().collect()
        });
        if !group_clusters[0].is_disjoint(&group_clusters[1]) {
            return Err(CausalError::Unsupported {
                message: "repeated-cross-section DiD requires clusters nested within treatment groups for cluster inference",
            });
        }
        let means: [[f64; 2]; 2] = std::array::from_fn(|group| {
            std::array::from_fn(|period| sums[group][period] / counts[group][period] as f64)
        });
        let effect = (means[1][1] - means[1][0]) - (means[0][1] - means[0][0]);
        let mut scores: BTreeMap<&str, f64> = BTreeMap::new();
        for (i, value) in y.iter().enumerate() {
            let group = usize::from(self.query.treated[i]);
            let period = usize::from(self.query.post[i]);
            let sign = if group == period { 1.0 } else { -1.0 };
            *scores.entry(self.query.clusters[i].as_ref()).or_default() +=
                sign * (value - means[group][period]) / counts[group][period] as f64;
        }
        let g = scores.len();
        let variance = g as f64 / (g - 1) as f64 * scores.values().map(|v| v * v).sum::<f64>();
        let se = variance.sqrt();
        let interval_supported = !self.extra.restricted
            && cell_clusters.iter().flatten().all(|set| set.len() >= 30)
            && se.is_finite()
            && se > 0.0;
        let interval_95 = interval_supported.then(|| {
            let radius = antecedent_stats::normal_ppf(0.975) * se;
            [effect - radius, effect + radius]
        });
        let identification = self.identification.clone();
        let estimate = EffectEstimate::new(
            effect,
            if interval_supported { se } else { f64::NAN },
            identification.required_assumptions.clone(),
            antecedent_estimate::OverlapPolicy::ExplicitOverride,
        )
        .with_se_kind(antecedent_estimate::AnalyticSeKind::Cluster);
        let started = Instant::now();
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish::new(&self.physical, identification, self.estimand.clone(), estimate, Strategy::new(IdentifierId::RandomizedDesign, EstimatorId::RandomizedHt), self.query.outcome, self.query.outcome).extra_diagnostics(vec![
                    Diagnostic::new(
                        "estimate.quasi.repeated_cross_section_did.cluster_se",
                        DiagnosticKind::Scientific,
                        DiagnosticSeverity::Info,
                        "four-cell repeated-cross-section DiD; cluster score sandwich SE; pointwise normal interval requires 30 independent clusters per cell",
                    ),
                    Diagnostic::new(
                        "identification.quasi.parallel_trends_untestable_two_periods",
                        DiagnosticKind::Scientific,
                        DiagnosticSeverity::Info,
                        "parallel untreated trends is a declared identifying assumption; two periods do not permit a pre-trend check",
                    ),
                ]).wall_time_ns(elapsed_ns(started)),
        );
        result.payload = crate::result::ResultPayload::PanelDid(crate::PanelDidEstimate {
            effect,
            standard_error: se,
            interval_95,
            treated_subjects: counts[1].iter().sum(),
            comparison_subjects: counts[0].iter().sum(),
            clusters: g,
            uncertainty: Arc::from(if interval_supported {
                "cluster_robust_normal_interval_independent_clusters"
            } else {
                "cluster_robust_standard_error_no_interval"
            }),
            event_time_effects: Arc::from([]),
            event_time_intervals_95: Arc::from([]),
            augmented: None,
        });
        if crate::support::license_if_graphless(
            panel_did_graphless_key(antecedent_core::DidSamplingDesign::RepeatedCrossSection),
            crate::support::GraphlessAssignmentSupport {
                assignment_unit: "cluster",
                treated: group_clusters[1].len(),
                control: group_clusters[0].len(),
                blocks: 4,
                min_block_arm: cell_clusters.iter().flatten().map(BTreeSet::len).min().unwrap_or(0),
                interval_95_published: interval_95.is_some(),
                reported_intervals: usize::from(interval_95.is_some()),
                ..Default::default()
            },
        )
        .is_some()
        {
            result.support_status = Some(crate::support::CellStatus::Licensed);
        }
        result.treatment = None;
        result.rebind_interval(false);
        Ok(result)
    }
}

/// Plan id, identifier and estimator a compiled `DiD` plan records for `design`.
pub(super) fn panel_did_plan_ids(
    design: antecedent_core::DidSamplingDesign,
) -> (&'static str, &'static str, &'static str) {
    use antecedent_core::DidSamplingDesign;
    let estimator = match design {
        DidSamplingDesign::BalancedPanel => "quasi.panel_change_score",
        DidSamplingDesign::RepeatedCrossSection => "quasi.repeated_cross_section_four_cell",
        DidSamplingDesign::StaggeredGroupTime => "quasi.staggered_group_time_never_treated",
        DidSamplingDesign::StaggeredEventStudy => "quasi.staggered_event_study_never_treated",
        DidSamplingDesign::AugmentedPanel => "quasi.augmented_panel_supplied_nuisance",
    };
    ("quasi.panel_did", "quasi.parallel_trends", estimator)
}

/// Graphless licence key for a `DiD` sampling design.
///
/// Execution licenses against this key and the contract publishes it as the
/// matrix coordinate, so the two cannot drift. Staggered group-time and
/// augmented panels have no licensed method; their key matches no licence row.
pub(crate) fn panel_did_graphless_key(
    design: antecedent_core::DidSamplingDesign,
) -> crate::support::GraphlessSupportKey<'static> {
    use antecedent_core::DidSamplingDesign;
    let (design, method, inference_claim) = match design {
        DidSamplingDesign::BalancedPanel => {
            ("panel_2x2", "cluster_change_scores_cr1", "pointwise_95_normal_interval")
        }
        DidSamplingDesign::RepeatedCrossSection => (
            "repeated_cross_section_2x2",
            "four_cell_cluster_scores_cr1",
            "pointwise_95_normal_interval",
        ),
        DidSamplingDesign::StaggeredEventStudy => (
            "staggered_event_study",
            "never_treated_event_study_cluster_cr1",
            "post_adoption_event_time_pointwise_95_normal_intervals",
        ),
        DidSamplingDesign::StaggeredGroupTime => ("staggered_group_time", "unlicensed", "none"),
        DidSamplingDesign::AugmentedPanel => ("augmented_panel", "unlicensed", "none"),
    };
    crate::support::GraphlessSupportKey {
        family: "difference_in_differences",
        design,
        method,
        inference_claim,
    }
}

impl Study {
    pub(super) fn execute_panel_did(
        &self,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        CheckedPanelDidOperation::checked(self, data, physical)?.execute(data, ctx)
    }
}

/// Assumptions the augmented panel adds for its supplied nuisance predictions.
fn push_augmented_did_assumptions(assumptions: &mut antecedent_core::AssumptionSet) {
    for (id, description) in [
        (
            "correct_propensity_model_or_correct_untreated_change_model",
            "at least one supplied nuisance model is correct",
        ),
        (
            "strict_propensity_overlap",
            "every supplied propensity lies strictly between zero and one",
        ),
        (
            "supplied_nuisance_predictions_valid_for_evaluation_rows",
            "nuisance predictions apply to these subject rows; caller-declared cross-fitting is not verified",
        ),
    ] {
        assumptions.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::Custom {
                id: Arc::from(id),
                description: Arc::from(description),
            },
            source: antecedent_core::AssumptionSource::UserDeclared,
            scope: antecedent_core::AssumptionScope::Identification,
            status: antecedent_core::AssumptionStatus::Declared,
        });
    }
}

/// Design id and description of the sampling assumption a `DiD` design declares.
fn did_sampling_assumption(
    design: antecedent_core::DidSamplingDesign,
) -> (&'static str, &'static str) {
    match design {
        antecedent_core::DidSamplingDesign::BalancedPanel => (
            "balanced_panel",
            "each subject contributes exactly one observed outcome in each of two periods",
        ),
        antecedent_core::DidSamplingDesign::RepeatedCrossSection => (
            "repeated_cross_section",
            "each subject is sampled once and the sampled group composition is comparable across periods",
        ),
        antecedent_core::DidSamplingDesign::StaggeredGroupTime => (
            "balanced_staggered_adoption_group_time",
            "each subject has every observed period; cohort zero supplies never-treated controls for the selected adoption cohort and post-adoption period",
        ),
        antecedent_core::DidSamplingDesign::StaggeredEventStudy => (
            "balanced_staggered_adoption_event_study",
            "each subject has every observed period; cohort zero supplies never-treated controls for cohort-specific event-time contrasts",
        ),
        antecedent_core::DidSamplingDesign::AugmentedPanel => (
            "augmented_two_period_panel",
            "each subject contributes one row with pre and post outcomes and supplied nuisance predictions",
        ),
    }
}

/// Assumptions a `DiD` design declares: the common ones plus its staggered or augmented additions.
fn panel_did_assumptions(query: &antecedent_core::PanelDidQuery) -> antecedent_core::AssumptionSet {
    let mut assumptions = antecedent_core::AssumptionSet::default();
    for assumption in
        [antecedent_core::Assumption::Consistency, antecedent_core::Assumption::NoInterference]
    {
        assumptions.push(antecedent_core::AssumptionRecord {
            assumption,
            source: antecedent_core::AssumptionSource::UserDeclared,
            scope: antecedent_core::AssumptionScope::Identification,
            status: antecedent_core::AssumptionStatus::Declared,
        });
    }
    let staggered = matches!(
        query.design,
        antecedent_core::DidSamplingDesign::StaggeredGroupTime
            | antecedent_core::DidSamplingDesign::StaggeredEventStudy
    );
    for (id, description) in [
        (
            if staggered {
                "cohort_specific_parallel_untreated_trends"
            } else if query.design == antecedent_core::DidSamplingDesign::AugmentedPanel {
                "conditional_parallel_untreated_trends_or_correct_untreated_change_model"
            } else {
                "parallel_trends"
            },
            "the untreated outcome change is identified under the declared design and nuisance-model assumptions",
        ),
        ("no_anticipation", "treatment does not affect pre-period outcomes"),
        (
            if staggered { "absorbing_treatment_after_adoption" } else { "stable_group" },
            "the treatment group and adoption history are stable across periods",
        ),
    ] {
        assumptions.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::Custom {
                id: Arc::from(id),
                description: Arc::from(description),
            },
            source: antecedent_core::AssumptionSource::UserDeclared,
            scope: antecedent_core::AssumptionScope::Identification,
            status: antecedent_core::AssumptionStatus::Declared,
        });
    }
    if staggered {
        assumptions.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::Custom {
                id: Arc::from("never_treated_controls_are_valid"),
                description: Arc::from(
                    "cohort zero is a valid untreated comparison group in the target period",
                ),
            },
            source: antecedent_core::AssumptionSource::UserDeclared,
            scope: antecedent_core::AssumptionScope::Identification,
            status: antecedent_core::AssumptionStatus::Declared,
        });
    }
    if query.design == antecedent_core::DidSamplingDesign::AugmentedPanel {
        push_augmented_did_assumptions(&mut assumptions);
    }
    let (design_id, design_description) = did_sampling_assumption(query.design);
    assumptions.push(antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::Custom {
            id: Arc::from(design_id),
            description: Arc::from(design_description),
        },
        source: antecedent_core::AssumptionSource::UserDeclared,
        scope: antecedent_core::AssumptionScope::Identification,
        status: antecedent_core::AssumptionStatus::Declared,
    });
    assumptions
}

pub(crate) fn panel_did_identification(
    query: &antecedent_core::PanelDidQuery,
) -> (IdentificationResult, IdentifiedEstimand) {
    let assumptions = panel_did_assumptions(query);
    let mut arena = CausalExprArena::new();
    let outcomes = arena.intern_var_set([query.outcome]);
    let empty = arena.empty_var_set();
    let intervention = arena.intern_intervention_set([]);
    let distribution = arena.intern_distribution(
        outcomes,
        empty,
        intervention,
        antecedent_expr::DomainRef::Observational,
    );
    let functional = arena.intern(antecedent_expr::ExprNode::Expectation {
        function: antecedent_expr::OutcomeExprId::identity(query.outcome),
        distribution,
    });
    let rule = match query.design {
        antecedent_core::DidSamplingDesign::BalancedPanel => "did.panel_change_score",
        antecedent_core::DidSamplingDesign::RepeatedCrossSection => {
            "did.repeated_cross_section_four_cell"
        }
        antecedent_core::DidSamplingDesign::StaggeredGroupTime => {
            "did.staggered_group_time_never_treated"
        }
        antecedent_core::DidSamplingDesign::StaggeredEventStudy => {
            "did.staggered_event_study_never_treated"
        }
        antecedent_core::DidSamplingDesign::AugmentedPanel => {
            "did.augmented_panel_supplied_nuisance"
        }
    };
    arena.set_derivation(
        functional,
        antecedent_expr::DerivationMeta::rule(
            rule,
            Some(Arc::from(
                "typed PanelDidQuery binds the observed group, period, subject, and cluster design",
            )),
        ),
    );
    let estimand = IdentifiedEstimand::new(
        rule,
        Arc::from([]),
        Arc::from([]),
        Arc::from([]),
        functional,
        None,
    );
    let mut derivation = DerivationTrace::default();
    derivation.push(rule, "the selected group-time difference in outcome changes is identified under cohort-specific parallel trends, no anticipation, and valid never-treated controls");
    let identification = IdentificationResult::identified(
        CausalQuery::PanelDid(query.clone()),
        vec![estimand.clone()],
        arena,
        derivation,
        assumptions,
        IdentificationPerformanceRecord::default(),
    );
    (identification, estimand)
}
