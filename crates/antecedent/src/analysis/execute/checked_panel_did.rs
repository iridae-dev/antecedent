//! Retained two-period panel DiD execution.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone)]
pub(crate) struct CheckedPanelDidOperation {
    query: antecedent_core::PanelDidQuery,
    schema: antecedent_core::CausalSchema,
    rows: usize,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    physical: PhysicalExecutionPlan,
    result_context: IdentifiedResultContext,
}

impl std::fmt::Debug for CheckedPanelDidOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedPanelDidOperation")
            .field("query", &self.query)
            .field("rows", &self.rows)
            .finish_non_exhaustive()
    }
}

impl CheckedPanelDidOperation {
    pub(crate) fn checked(
        study: &Study,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
    ) -> Result<Self, CausalError> {
        let CausalQuery::PanelDid(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "panel DiD operation requires PanelDidQuery".into(),
            });
        };
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        if study.graph.class() != GraphClass::RandomizedTrial
            || !matches!(study.inference, InferenceMode::Frequentist)
            || study.refute != RefuteSuite::None
            || study.bootstrap_replicates != 0
            || !study.custom_validators.is_empty()
            || query.treated.len() != data.row_count()
        {
            return Err(CausalError::Unsupported {
                message: "panel DiD currently accepts only the graphless two-period panel query with point estimate and cluster standard error",
            });
        }
        data.schema()
            .get(query.outcome)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let (identification, estimand) = panel_did_identification(query);
        Ok(Self {
            query: query.clone(),
            schema: data.schema().clone(),
            rows: data.row_count(),
            identification,
            estimand,
            physical: physical.clone(),
            result_context: IdentifiedResultContext::from_study(study),
        })
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if data.schema() != &self.schema || data.row_count() != self.rows {
            return Err(CausalError::Unsupported {
                message: "panel DiD refresh requires the prepared schema and row order",
            });
        }
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: super::super::stage::STAGE_ESTIMATE_POINT,
            });
        }
        let y = match data.column(self.query.outcome).map_err(CausalError::from)? {
            antecedent_data::ColumnView::Float64(column) => column.values.as_slice(),
            _ => {
                return Err(CausalError::Unsupported {
                    message: "panel DiD outcome must be continuous",
                });
            }
        };
        if self.query.design == antecedent_core::DidSamplingDesign::StaggeredGroupTime {
            return self.execute_staggered_group_time(data, y);
        }
        if self.query.design == antecedent_core::DidSamplingDesign::RepeatedCrossSection {
            return self.execute_repeated_cross_section(data, y);
        }
        let mut subjects: BTreeMap<&str, (Option<f64>, Option<f64>, Option<bool>, Option<&str>)> =
            BTreeMap::new();
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
        let g = scores.len();
        let variance = g as f64 / (g - 1) as f64 * scores.values().map(|v| v * v).sum::<f64>();
        let identification = self.identification.clone();
        let assumptions = identification.required_assumptions.clone();
        let estimate = EffectEstimate::new(
            effect,
            variance.sqrt(),
            assumptions,
            antecedent_estimate::OverlapPolicy::ExplicitOverride,
        );
        let started = Instant::now();
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish {
                physical: &self.physical,
                identification,
                estimand: self.estimand.clone(),
                estimate,
                identifier_id: IdentifierId::RandomizedDesign,
                estimator_id: EstimatorId::RandomizedHt,
                treatment: self.query.outcome,
                outcome: self.query.outcome,
                identify_cached: false,
                extra_diagnostics: vec![Diagnostic::new(
                    "estimate.quasi.panel_did.cluster_se",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "two-period subject change-score DiD; cluster score sandwich SE; normal intervals are not reported",
                )],
                refutations: Vec::new(),
                distribution: None,
                mediation: None,
                wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                bootstrap_replicates_ok: None,
                cancelled: false,
                early_stopped: false,
                extras: IdentifiedExecuteExtras::default(),
            },
        );
        result.panel_did = Some(crate::PanelDidEstimate {
            effect,
            standard_error: variance.sqrt(),
            treated_subjects: counts[1],
            comparison_subjects: counts[0],
            clusters: g,
            uncertainty: Arc::from("cluster_robust_standard_error_no_interval"),
        });
        result.treatment = None;
        Ok(result)
    }

    fn execute_staggered_group_time(
        &self,
        data: &TabularData,
        y: &[f64],
    ) -> Result<StudyResult, CausalError> {
        let (cohort, period) = self.query.target.expect("checked staggered target");
        let baseline = cohort - 1;
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
            || subjects.values().any(|unit| {
                unit.2.len() != all_periods.len()
                    || all_periods.iter().any(|t| !unit.2.contains_key(t))
            })
        {
            return Err(CausalError::Compile { message: "staggered DiD requires a balanced panel with the target and immediately preceding adoption period".into() });
        }
        let selected: [Vec<_>; 2] = [
            subjects.values().filter(|unit| unit.0 == 0).collect(),
            subjects.values().filter(|unit| unit.0 == cohort).collect(),
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
            se,
            self.identification.required_assumptions.clone(),
            antecedent_estimate::OverlapPolicy::ExplicitOverride,
        );
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish {
                physical: &self.physical,
                identification: self.identification.clone(),
                estimand: self.estimand.clone(),
                estimate,
                identifier_id: IdentifierId::RandomizedDesign,
                estimator_id: EstimatorId::RandomizedHt,
                treatment: self.query.outcome,
                outcome: self.query.outcome,
                identify_cached: false,
                extra_diagnostics: vec![Diagnostic::new(
                    "estimate.quasi.staggered_group_time.cluster_se",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    format!(
                        "cohort {cohort}, period {period}, baseline {baseline}; never-treated controls; cluster score sandwich SE; no interval or parallel-trends test"
                    ),
                )],
                refutations: Vec::new(),
                distribution: None,
                mediation: None,
                wall_time_ns: 0,
                bootstrap_replicates_ok: None,
                cancelled: false,
                early_stopped: false,
                extras: IdentifiedExecuteExtras::default(),
            },
        );
        result.panel_did = Some(crate::PanelDidEstimate {
            effect,
            standard_error: se,
            treated_subjects: selected[1].len(),
            comparison_subjects: selected[0].len(),
            clusters: cluster_count,
            uncertainty: Arc::from("cluster_robust_standard_error_no_interval"),
        });
        result.treatment = None;
        Ok(result)
    }

    fn execute_repeated_cross_section(
        &self,
        data: &TabularData,
        y: &[f64],
    ) -> Result<StudyResult, CausalError> {
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
        let identification = self.identification.clone();
        let estimate = EffectEstimate::new(
            effect,
            variance.sqrt(),
            identification.required_assumptions.clone(),
            antecedent_estimate::OverlapPolicy::ExplicitOverride,
        );
        let started = Instant::now();
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish {
                physical: &self.physical,
                identification,
                estimand: self.estimand.clone(),
                estimate,
                identifier_id: IdentifierId::RandomizedDesign,
                estimator_id: EstimatorId::RandomizedHt,
                treatment: self.query.outcome,
                outcome: self.query.outcome,
                identify_cached: false,
                extra_diagnostics: vec![Diagnostic::new(
                    "estimate.quasi.repeated_cross_section_did.cluster_se",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "four-cell repeated-cross-section DiD; cluster score sandwich SE; normal intervals are not reported",
                )],
                refutations: Vec::new(),
                distribution: None,
                mediation: None,
                wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                bootstrap_replicates_ok: None,
                cancelled: false,
                early_stopped: false,
                extras: IdentifiedExecuteExtras::default(),
            },
        );
        result.panel_did = Some(crate::PanelDidEstimate {
            effect,
            standard_error: variance.sqrt(),
            treated_subjects: counts[1].iter().sum(),
            comparison_subjects: counts[0].iter().sum(),
            clusters: g,
            uncertainty: Arc::from("cluster_robust_standard_error_no_interval"),
        });
        result.treatment = None;
        Ok(result)
    }
}

impl Study {
    pub(super) fn execute_panel_did(
        &self,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        CheckedPanelDidOperation::checked(self, data, physical)?.execute(data, ctx)
    }
}

pub(crate) fn panel_did_identification(
    query: &antecedent_core::PanelDidQuery,
) -> (IdentificationResult, IdentifiedEstimand) {
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
    let staggered = query.design == antecedent_core::DidSamplingDesign::StaggeredGroupTime;
    for (id, description) in [
        (
            if staggered { "cohort_specific_parallel_untreated_trends" } else { "parallel_trends" },
            "in the absence of treatment, the selected cohort and never-treated comparison would have had equal mean outcome changes",
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
    let (design_id, design_description) = match query.design {
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
    };
    assumptions.push(antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::Custom {
            id: Arc::from(design_id),
            description: Arc::from(design_description),
        },
        source: antecedent_core::AssumptionSource::UserDeclared,
        scope: antecedent_core::AssumptionScope::Identification,
        status: antecedent_core::AssumptionStatus::Declared,
    });
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
