//! Sampling-weighted two-period panel `DiD`.
//!
//! The table's observation weights are the sampling weights of its rows; both rows of a subject
//! carry the subject's weight. The estimand is the difference of the two groups' weighted mean
//! change scores in the weighted population, `Σ s_j Δ_j / Σ s_j` per group. The standard error
//! is the cluster score sandwich with sampling-weighted scores
//! `±s_j (Δ_j − m_g) / S_g` summed within clusters, `G/(G−1) Σ_c score_c²`, treating clusters as
//! the independent primary sampling units (no finite-population correction). Weights are
//! normalized to mean one over the subjects used (estimates do not depend on their scale);
//! subjects of weight zero are excluded and counted; no interval is published.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, BTreeMap, BTreeSet, CausalError, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    EffectEstimate, EstimatorId, IdentifiedExecuteFinish, IdentifierId, Instant, Strategy,
    StudyResult, TabularData, elapsed_ns, finish_identified_execute_with_context,
};
use antecedent_estimate::SamplingWeightRecord;

/// One subject: its group, cluster, change score and (positive) weight.
struct Subject<'a> {
    treated: bool,
    cluster: &'a str,
    change: f64,
    weight: f64,
}

/// The weighted contrast and what it rests on.
struct WeightedDid {
    effect: f64,
    standard_error: f64,
    clusters: usize,
    cluster_counts: [usize; 2],
    subjects: [usize; 2],
    record: SamplingWeightRecord,
}

fn unsupported(message: &'static str) -> CausalError {
    CausalError::Unsupported { message }
}

fn route_unsupported(message: &str) -> CausalError {
    CausalError::Estimate(antecedent_estimate::sampling_weights::unsupported_route(message))
}

fn invalid_weights(message: &str) -> CausalError {
    CausalError::Estimate(antecedent_estimate::sampling_weights::invalid(message))
}

impl super::CheckedPanelDidOperation {
    /// Execute the balanced two-period panel under the table's sampling weights.
    pub(super) fn execute_weighted_balanced(
        &self,
        data: &TabularData,
        y: &[f64],
        weights: &[f64],
    ) -> Result<StudyResult, CausalError> {
        if self.query.design != antecedent_core::DidSamplingDesign::BalancedPanel
            || self.extra.restricted
        {
            return Err(route_unsupported(
                "sampling weights are read by the balanced two-period panel DiD without a \
                 missing-data restriction; staggered, repeated cross-section and augmented \
                 designs have no sampling-weighted contrast",
            ));
        }
        let subjects = self.weighted_subjects(y, weights)?;
        let fit = weighted_contrast(&subjects)?;
        let rows_used = fit.record.rows_used;
        let identification = self.identification.clone();
        let assumptions = identification.required_assumptions.clone();
        // No coverage record measures a weighted contrast: the standard error is reported on the
        // payload, the estimate carries none.
        let estimate = EffectEstimate::new(
            fit.effect,
            f64::NAN,
            assumptions,
            antecedent_estimate::OverlapPolicy::ExplicitOverride,
        )
        .with_se_kind(antecedent_estimate::AnalyticSeKind::Cluster)
        .with_sampling_weights(Some(fit.record));
        let started = Instant::now();
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish::new(
                &self.physical,
                identification,
                self.estimand.clone(),
                estimate,
                Strategy::new(IdentifierId::RandomizedDesign, EstimatorId::RandomizedHt),
                self.query.outcome,
                self.query.outcome,
            )
            .extra_diagnostics(vec![
                Diagnostic::new(
                    "estimate.quasi.panel_did.sampling_weighted",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    format!(
                        "two-period subject change-score DiD in the weighted population the \
                         sampling weights represent: difference of weighted group means, \
                         weights normalized to mean one; cluster score sandwich over \
                         sampling-weighted scores, clusters taken as the independent primary \
                         sampling units; {rows_used} subjects used, {} zero-weight subjects \
                         excluded; Kish effective sample size {:.2}; no confidence interval is \
                         reported",
                        fit.record.zero_weight_rows, fit.record.effective_sample_size,
                    ),
                ),
                Diagnostic::new(
                    "identification.quasi.parallel_trends_untestable_two_periods",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "parallel untreated trends is a declared identifying assumption; two periods do not permit a pre-trend check",
                ),
            ])
            .wall_time_ns(elapsed_ns(started)),
        );
        result.payload = crate::result::ResultPayload::PanelDid(crate::PanelDidEstimate {
            effect: fit.effect,
            standard_error: fit.standard_error,
            interval_95: None,
            treated_subjects: fit.subjects[1],
            comparison_subjects: fit.subjects[0],
            clusters: fit.clusters,
            uncertainty: Arc::from("sampling_weighted_cluster_robust_standard_error_no_interval"),
            event_time_effects: Arc::from([]),
            event_time_intervals_95: Arc::from([]),
            augmented: None,
        });
        if crate::support::license_if_graphless(
            super::panel_did_graphless_key(antecedent_core::DidSamplingDesign::BalancedPanel),
            crate::support::GraphlessAssignmentSupport {
                assignment_unit: "cluster",
                treated: fit.cluster_counts[1],
                control: fit.cluster_counts[0],
                interval_95_published: false,
                reported_intervals: 0,
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

    /// The positive-weight subjects with their change scores. Both rows of a subject must carry
    /// the same weight.
    fn weighted_subjects<'a>(
        &'a self,
        y: &[f64],
        weights: &[f64],
    ) -> Result<Vec<Subject<'a>>, CausalError> {
        type Row<'a> = (Option<f64>, Option<f64>, bool, &'a str, f64);
        let mut by_subject: BTreeMap<&str, Row<'a>> = BTreeMap::new();
        for i in 0..y.len() {
            if !y[i].is_finite() {
                return Err(unsupported("panel DiD requires complete finite outcomes"));
            }
            let subject = self.query.subjects[i].as_ref();
            let entry = by_subject.entry(subject).or_insert((
                None,
                None,
                self.query.treated[i],
                self.query.clusters[i].as_ref(),
                weights[i],
            ));
            if entry.2 != self.query.treated[i] {
                return Err(unsupported("treatment must be stable within subject"));
            }
            if entry.3 != self.query.clusters[i].as_ref() {
                return Err(unsupported("cluster must be stable within subject"));
            }
            #[allow(
                clippy::float_cmp,
                reason = "the two rows of a subject must carry the identical declared weight"
            )]
            if entry.4 != weights[i] {
                return Err(invalid_weights(
                    "both rows of a subject must carry the same sampling weight",
                ));
            }
            let slot = if self.query.post[i] { &mut entry.1 } else { &mut entry.0 };
            if slot.replace(y[i]).is_some() {
                return Err(unsupported(
                    "each subject must have exactly one pre and one post observation",
                ));
            }
        }
        let mut subjects = Vec::with_capacity(by_subject.len());
        for (pre, post, treated, cluster, weight) in by_subject.into_values() {
            let (Some(pre), Some(post)) = (pre, post) else {
                return Err(unsupported("panel DiD requires a balanced two-period panel"));
            };
            subjects.push(Subject { treated, cluster, change: post - pre, weight });
        }
        Ok(subjects)
    }
}

/// Weighted group means, the contrast and its cluster score sandwich. Zero-weight subjects are
/// counted and left out.
fn weighted_contrast(all: &[Subject<'_>]) -> Result<WeightedDid, CausalError> {
    let zero_weight = all.iter().filter(|s| s.weight == 0.0).count();
    let subjects: Vec<&Subject<'_>> = all.iter().filter(|s| s.weight > 0.0).collect();
    if subjects.is_empty() {
        return Err(invalid_weights("no subject has a positive sampling weight"));
    }
    let raw: Vec<f64> = subjects.iter().map(|s| s.weight).collect();
    let total: f64 = raw.iter().sum();
    let scale = subjects.len() as f64 / total;
    let mut mass = [0.0_f64; 2];
    let mut sum = [0.0_f64; 2];
    let mut counts = [0usize; 2];
    let mut clusters: [BTreeSet<&str>; 2] = Default::default();
    for s in &subjects {
        let g = usize::from(s.treated);
        let w = s.weight * scale;
        mass[g] += w;
        sum[g] += w * s.change;
        counts[g] += 1;
        clusters[g].insert(s.cluster);
    }
    if counts.contains(&0) {
        return Err(unsupported(
            "panel DiD requires treated and comparison subjects with positive weight",
        ));
    }
    if clusters.iter().any(|set| set.len() < 2) {
        return Err(unsupported("cluster SE requires at least two clusters in each group"));
    }
    if !clusters[0].is_disjoint(&clusters[1]) {
        return Err(unsupported(
            "panel DiD requires clusters nested within treatment groups for cluster inference",
        ));
    }
    let means = [sum[0] / mass[0], sum[1] / mass[1]];
    let mut scores: BTreeMap<&str, f64> = BTreeMap::new();
    for s in &subjects {
        let g = usize::from(s.treated);
        let sign = if s.treated { 1.0 } else { -1.0 };
        *scores.entry(s.cluster).or_default() +=
            sign * (s.weight * scale) * (s.change - means[g]) / mass[g];
    }
    let g = scores.len() as f64;
    let variance = g / (g - 1.0) * scores.values().map(|v| v * v).sum::<f64>();
    Ok(WeightedDid {
        effect: means[1] - means[0],
        standard_error: variance.sqrt(),
        clusters: scores.len(),
        cluster_counts: [clusters[0].len(), clusters[1].len()],
        subjects: counts,
        record: SamplingWeightRecord {
            rows_used: subjects.len() as u64,
            zero_weight_rows: zero_weight as u64,
            effective_sample_size: antecedent_kernels::kish_ess(&raw),
            weight_sum: total,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subjects(spec: &[(bool, &'static str, f64, f64)]) -> Vec<Subject<'static>> {
        spec.iter()
            .map(|&(treated, cluster, change, weight)| Subject { treated, cluster, change, weight })
            .collect()
    }

    #[test]
    fn unit_weights_reproduce_the_unweighted_cluster_score_sandwich() {
        let spec = [
            (false, "a", 1.0, 1.0),
            (false, "a", 2.0, 1.0),
            (false, "b", 0.0, 1.0),
            (false, "b", 1.5, 1.0),
            (true, "c", 3.0, 1.0),
            (true, "c", 2.0, 1.0),
            (true, "d", 4.0, 1.0),
            (true, "d", 5.0, 1.0),
        ];
        let fit = weighted_contrast(&subjects(&spec)).unwrap();
        // Group means 1.125 and 3.5; cluster scores ±(Σ(Δ − m)/n_g).
        let control = 1.125;
        let treated = 3.5;
        let scores: [f64; 4] = [
            -((1.0 - control) + (2.0 - control)) / 4.0,
            -((0.0 - control) + (1.5 - control)) / 4.0,
            ((3.0 - treated) + (2.0 - treated)) / 4.0,
            ((4.0 - treated) + (5.0 - treated)) / 4.0,
        ];
        let expected = (4.0 / 3.0 * scores.iter().map(|s| s * s).sum::<f64>()).sqrt();
        assert!((fit.effect - (treated - control)).abs() < 1e-12);
        assert!((fit.standard_error - expected).abs() < 1e-12);
        assert_eq!(fit.record.zero_weight_rows, 0);
    }

    #[test]
    fn rescaling_and_zero_weight_subjects_change_nothing_but_the_count() {
        let base = [
            (false, "a", 1.0, 2.0),
            (false, "a", 2.0, 1.0),
            (false, "b", 0.0, 3.0),
            (false, "b", 1.5, 1.0),
            (true, "c", 3.0, 1.0),
            (true, "c", 2.0, 4.0),
            (true, "d", 4.0, 2.0),
            (true, "d", 5.0, 1.0),
        ];
        let fit = weighted_contrast(&subjects(&base)).unwrap();
        let mut scaled = base;
        for row in &mut scaled {
            row.3 *= 1.0e5;
        }
        let big = weighted_contrast(&subjects(&scaled)).unwrap();
        assert!((fit.effect - big.effect).abs() < 1e-12);
        assert!((fit.standard_error - big.standard_error).abs() < 1e-12);
        let mut padded = base.to_vec();
        padded.push((true, "c", 1.0e6, 0.0));
        padded.push((false, "a", -1.0e6, 0.0));
        let with_zeros = weighted_contrast(&subjects(&padded)).unwrap();
        assert!((fit.effect - with_zeros.effect).abs() < 1e-12);
        assert!((fit.standard_error - with_zeros.standard_error).abs() < 1e-12);
        assert_eq!(with_zeros.record.zero_weight_rows, 2);
        assert_eq!(with_zeros.record.rows_used, 8);
    }
}
