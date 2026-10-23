// SPDX-License-Identifier: MIT OR Apache-2.0

impl super::Study {
    pub(super) fn estimate_functional_effect(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        identification: &IdentificationResult,
        extra: &[VariableId],
        ctx: &ExecutionContext,
    ) -> Result<(EffectEstimate, Option<antecedent_estimate::CausalPosterior>), CausalError> {
        let (row_resampling, sampling_design) = self.functional_row_resampling()?;
        let est = FunctionalEffect {
            bootstrap_replicates: self.bootstrap_replicates,
            ..FunctionalEffect::new()
        }
        .with_row_resampling(row_resampling, sampling_design);
        let prepared = est
            .prepare(
                data,
                estimand,
                &identification.arena,
                identification.required_assumptions.clone(),
                extra,
            )
            .map_err(CausalError::from)?;
        run_functional(&est, &prepared, &self.inference, identification.status, ctx)
    }
}

impl super::Study {
    /// How a functional plug-in resamples its rows on this study.
    ///
    /// A `TemporalAdmg` functional is the contemporaneous projection, read on the
    /// consecutive times of one series: its frequentist bootstrap resamples circular
    /// blocks of rows and its Bayesian bootstrap shares one Dirichlet weight per block.
    /// Every other graph class keeps independent rows, as before.
    ///
    /// # Errors
    ///
    /// On the projection, a declared independent-subject or clustered-panel design when a
    /// resample or a posterior draw will run: the table carries no subject or cluster
    /// labels, so neither a whole-subject nor a whole-cluster draw can be formed, and a
    /// row-level draw is refused under both designs.
    pub(crate) fn functional_row_resampling(
        &self,
    ) -> Result<
        (antecedent_estimate::RowResampling, antecedent_estimate::util::SamplingDeclaration),
        CausalError,
    > {
        use antecedent_core::DeclaredSamplingDesign as Design;
        use antecedent_estimate::RowResampling;
        let temporal_projection = matches!(self.graph.class(), GraphClass::TemporalAdmg)
            || self.graph_posterior.as_ref().is_some_and(|gp| {
                gp.atom_kind == antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg
            });
        let rows =
            (RowResampling::Rows, antecedent_estimate::util::SamplingDeclaration::Undeclared);
        if !temporal_projection {
            return Ok(rows);
        }
        let draws = self.inference.is_bayesian() || self.bootstrap_replicates > 0;
        let declared = crate::estimator_spec::declared_sampling(self.sampling_design);
        match self.sampling_design {
            None | Some(Design::DependentTimeSeries) => Ok((RowResampling::SeriesBlocks, declared)),
            Some(Design::IndependentSubjectTrajectories | Design::ClusteredPanel) if draws => {
                Err(crate::unsupported_reason!(
                    "option_not_applicable",
                    "the contemporaneous-projection functional draws its uncertainty from the \
                     table rows, and a declared independent-subject or clustered-panel design \
                     refuses a row-level draw; this table carries no subject or cluster labels \
                     to draw whole subjects or clusters from. Next attempt: declare a dependent \
                     series (circular blocks of consecutive rows), analyse the units as panel \
                     data with their unit labels, or omit bootstrap_replicates for the \
                     frequentist point value"
                ))
            }
            Some(_) => Ok(rows),
        }
    }
}

pub(super) fn identified_envelope_keys(
    graphs: &WeightedGraphSamples,
) -> std::collections::HashSet<u64> {
    graphs
        .graph_keys
        .iter()
        .zip(graphs.identified.iter())
        .filter(|(_, flag)| **flag == GraphIdentFlag::Identified)
        .map(|(key, _)| *key)
        .collect()
}

/// Per-atom Bayesian fit retained so envelope validation can mix across structures.
pub(super) struct EnvelopeAtomFit {
    pub key: u64,
    pub prep: PreparedBayesianProblem,
    pub posterior: CausalPosterior,
    pub status: IdentificationStatus,
    pub weight: f64,
    pub estimand: IdentifiedEstimand,
    pub indexer: Option<TemporalIndexer>,
    /// Resolved coefficient prior this atom was fitted under (`None` = isotropic).
    pub prior: Option<PriorSet>,
}

/// Estimand, mass, and the atom's own fitted effect needed to mix Frequentist or
/// Bayesian envelope refuters.
///
/// `original` is this atom's own estimate (posterior mean / SD for Bayesian
/// atoms). Refuters compare each atom's perturbation refits against it, never
/// against the pooled mixture: a stable atom whose effect differs from the
/// pooled value is not a refutation.
pub(super) struct EnvelopeRefuteAtom {
    pub key: u64,
    pub weight: f64,
    pub estimand: IdentifiedEstimand,
    pub indexer: Option<TemporalIndexer>,
    pub original: EffectEstimate,
}

impl EnvelopeRefuteAtom {
    /// Refute atom for a Bayesian envelope fit; `original` is that atom's posterior summary.
    pub(super) fn from_fit(atom: &EnvelopeAtomFit) -> Result<Self, CausalError> {
        Ok(Self {
            key: atom.key,
            weight: atom.weight,
            estimand: atom.estimand.clone(),
            indexer: atom.indexer.clone(),
            original: effect_from_posterior(&atom.posterior)?,
        })
    }
}

pub(super) fn envelope_refute_atoms(
    fits: &[EnvelopeAtomFit],
) -> Result<Vec<EnvelopeRefuteAtom>, CausalError> {
    fits.iter().map(EnvelopeRefuteAtom::from_fit).collect()
}

/// Disclose a non-finite envelope SE: with several contributing atoms the joint
/// SE was not formed ([`envelope_se_omits_between_atom_variance`]); with one
/// atom that atom's own analytic SE is unavailable.
pub(super) fn envelope_se_omission_diagnostic(
    contributing_atoms: usize,
    se: f64,
) -> Option<Diagnostic> {
    if se.is_finite() {
        return None;
    }
    Some(if contributing_atoms > 1 {
        envelope_se_omits_between_atom_variance()
    } else {
        Diagnostic::new(
            "estimate.envelope.single_atom_se_unavailable",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "the single contributing atom publishes no analytic SE, so the envelope \
             carries none; request bootstrap replicates for a resampled SE",
        )
    })
}

/// The legacy diagnostic code is retained for downstream consumers.
pub(super) fn envelope_se_omits_between_atom_variance() -> Diagnostic {
    Diagnostic::new(
        "estimate.envelope.se_omits_between_atom_variance",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "multi-atom uncertainty is unavailable: per-atom SEs do not determine the \
         sampling variance of fits sharing observations, and averaged interval \
         endpoints are not mixture quantiles. Only a single contributing atom \
         retains its own uncertainty. Completion weights are modeling choices, \
         not evidence that the averaged effect is identified across graphs",
    )
}

/// Warning shared by every one-series circular-block interval when the
/// estimating score's effective rows fall below the threshold of its SE family
/// ([`antecedent_estimate::CircularBlockFamily::min_effective_rows`]).
pub(super) fn short_series_warning(
    effective_rows: f64,
    family: antecedent_estimate::CircularBlockFamily,
) -> Option<Diagnostic> {
    family.is_short_series(effective_rows).then(|| {
        Diagnostic::new(
            "estimate.temporal.circular_block_se.short_series",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "estimating-score effective rows {effective_rows:.1} < {} (the {} \
                 threshold): the series may be too short for its serial dependence, and the \
                 circular-block interval may under-cover",
                family.min_effective_rows(),
                family.label(),
            ),
        )
    })
}

pub(super) fn temporal_class_block_span<'a>(
    indexers: impl IntoIterator<Item = &'a TemporalIndexer>,
) -> usize {
    indexers
        .into_iter()
        .map(|indexer| indexer.history() as usize + indexer.horizon() as usize)
        .max()
        .unwrap_or(1)
        .max(1)
}

/// The declared linear-SEM restriction every sequential g-computation fit
/// carries; `description` names the SEM family the route fitted.
pub(super) fn sequential_linear_sem_assumption(
    description: &str,
) -> antecedent_core::AssumptionRecord {
    antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(
            antecedent_core::ParametricAssumption {
                id: Arc::from("temporal.sequential.linear_sem"),
                description: Arc::from(description),
            },
        ),
        source: antecedent_core::AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("temporal.sequential.gcomp"),
        },
        scope: antecedent_core::AssumptionScope::Estimation,
        status: antecedent_core::AssumptionStatus::Declared,
    }
}

/// Note that a bidirected completion carries no sequential g-computation.
pub(super) fn mag_sequential_unevaluable_diagnostic(key: impl std::fmt::Display) -> Diagnostic {
    Diagnostic::new(
        "estimate.temporal_class.mag_sequential_unevaluable",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!("completion {key} has bidirected edges; sequential g-comp is unevaluable"),
    )
}

/// Frequentist mixture estimate of a frozen-weight class envelope from its
/// weighted `point`, the mixed analytic SE and the shared circular-block draw.
pub(super) fn class_mixture_estimate(
    point: f64,
    se_analytic: f64,
    block: &class_envelope_se::SharedCircularBlockSe,
    bootstrap_replicates: u32,
    assumptions: antecedent_core::AssumptionSet,
    ctx: &ExecutionContext,
) -> EffectEstimate {
    EffectEstimate::from_parts(
        point,
        se_analytic,
        BootstrapAccounting {
            se: block.se.is_finite().then_some(block.se),
            replicates_ok: (bootstrap_replicates > 0).then_some(block.completed),
            replicates_failed: (bootstrap_replicates > 0)
                .then_some(block.attempted.saturating_sub(block.completed)),
            cancelled: ctx.cancellation.is_cancelled(),
            early_stopped: false,
        },
        assumptions,
        OverlapPolicy::ExplicitOverride,
        None,
        None,
    )
    .with_block_family(antecedent_estimate::CircularBlockFamily::Mixture)
}

/// Shared circular-block mixture of the evaluable sequential atoms of a class
/// envelope, with the atoms' point estimates and mixed analytic SE.
struct SequentialAtomBlock {
    block: class_envelope_se::SharedCircularBlockSe,
    identified_set_interval: Option<antecedent_estimate::IdentifiedSetInterval>,
    se_analytic: f64,
}

fn sequential_atom_block(
    data: &TimeSeriesData,
    query: &TemporalEffectQuery,
    atoms: &[sequential_validation::SequentialValidationAtom],
    bootstrap_replicates: u32,
    truncated: bool,
    gaps: bool,
    ctx: &ExecutionContext,
) -> Result<SequentialAtomBlock, CausalError> {
    let designs = atoms
        .iter()
        .map(|atom| {
            TemporalAtomDesign::sequential_with_gaps(
                data,
                &atom.graph,
                &atom.indexer,
                &atom.estimand,
                query,
                atom.status,
                gaps,
                ctx,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let weights = atoms.iter().map(|atom| atom.weight).collect::<Vec<_>>();
    let block = shared_circular_block_mixture_se(
        &designs.iter().collect::<Vec<_>>(),
        &weights,
        temporal_class_block_span(atoms.iter().map(|atom| &atom.indexer)),
        bootstrap_replicates,
        0x5E0C_1A55,
        ctx,
    );
    let points = atoms.iter().map(|atom| atom.estimate.ate).collect::<Vec<_>>();
    let identified_set_interval = block
        .identified_set_interval(&points, IDENTIFIED_SET_INTERVAL_LEVEL)
        .map(|interval| interval.with_truncated(truncated));
    let se_analytic = mix_weighted_analytic_se(
        atoms.iter().map(|atom| (atom.weight, atom.estimate.se_analytic())),
    );
    Ok(SequentialAtomBlock { block, identified_set_interval, se_analytic })
}

/// Linear-adjustment atoms of a frequentist temporal class envelope, fitted in
/// case order. The totals are not checked: callers own their refusal text.
pub(super) struct ClassLinearAtoms<'a> {
    pub values: Vec<Option<f64>>,
    pub designs: Vec<TemporalAtomDesign>,
    pub weights: Vec<f64>,
    pub indexers: Vec<&'a TemporalIndexer>,
    pub points: Vec<f64>,
    pub se_items: Vec<(f64, f64)>,
    pub refute_atoms: Vec<EnvelopeRefuteAtom>,
    pub primary_estimand: Option<IdentifiedEstimand>,
    pub assumptions: antecedent_core::AssumptionSet,
    pub weighted_point: f64,
    pub total_weight: f64,
}

pub(super) fn fit_class_linear_atoms<'a>(
    data: &TimeSeriesData,
    query: &TemporalEffectQuery,
    envelope: &antecedent_identify::TemporalClassEnvelope,
    indexers: &'a [TemporalIndexer],
    (split, gaps): (Option<&antecedent_data::DiscoveryEstimationSplit>, bool),
    ctx: &ExecutionContext,
) -> Result<ClassLinearAtoms<'a>, CausalError> {
    let mut fit = ClassLinearAtoms {
        values: vec![None; envelope.envelope.cases.len()],
        designs: Vec::new(),
        weights: Vec::new(),
        indexers: Vec::new(),
        points: Vec::new(),
        se_items: Vec::new(),
        refute_atoms: Vec::new(),
        primary_estimand: None,
        assumptions: antecedent_core::AssumptionSet::default(),
        weighted_point: 0.0,
        total_weight: 0.0,
    };
    for (index, (case, indexer)) in envelope.envelope.cases.iter().zip(indexers).enumerate() {
        if !identification_status_ok_for_case(case.result.status)
            || case.result.estimands.is_empty()
        {
            continue;
        }
        let estimand = select_estimand(&case.result, EstimatorId::TemporalLinearAdjustment)?;
        let design = TemporalAtomDesign::linear_with_gaps(
            data, &estimand, query, indexer, split, gaps, ctx,
        )?;
        let estimate = design.effect_estimate(case.result.required_assumptions.clone());
        let weight = case.weight.0;
        fit.weighted_point += weight * estimate.ate;
        fit.total_weight += weight;
        fit.values[index] = Some(estimate.ate);
        fit.se_items.push((weight, estimate.se_analytic()));
        if fit.primary_estimand.is_none() {
            fit.primary_estimand = Some(estimand.clone());
            fit.assumptions = estimate.provenance.assumptions.clone();
        }
        fit.points.push(estimate.ate);
        fit.refute_atoms.push(EnvelopeRefuteAtom {
            key: index as u64,
            weight,
            estimand,
            indexer: Some(indexer.clone()),
            original: estimate,
        });
        fit.designs.push(design);
        fit.weights.push(weight);
        fit.indexers.push(indexer);
    }
    Ok(fit)
}

/// Embed complete-case influences into the original row universe before mixing.
/// Equal vector lengths do not establish common observations.
pub(super) fn static_aligned_influence(
    data: &TabularData,
    query: &AverageEffectQuery,
    estimand: &IdentifiedEstimand,
    influence: &[f64],
) -> Option<Vec<f64>> {
    if !matches!(query.target_population, antecedent_core::TargetPopulation::AllObserved) {
        return None;
    }
    let mut ids = vec![query.treatment, query.outcome];
    ids.extend(query.effect_modifiers.iter().copied());
    ids.extend(estimand.adjustment_set.iter().copied());
    ids.extend(estimand.instruments.iter().copied());
    ids.extend(estimand.mediators.iter().copied());
    let mask = data.complete_case_mask(&ids).ok()?;
    let rows: Vec<_> = mask.iter().enumerate().filter_map(|(i, &keep)| keep.then_some(i)).collect();
    if rows.len() != influence.len() || rows.len() < 2 {
        return None;
    }
    let mean = influence.iter().sum::<f64>() / influence.len() as f64;
    let mut full = vec![0.0; mask.len()];
    for (&i, &v) in rows.iter().zip(influence) {
        full[i] = (v - mean) * mask.len() as f64 / influence.len() as f64;
    }
    Some(full)
}

/// Mix `ConditionalEffect` exceedance grids across envelope atoms.
///
/// Per threshold, each atom is estimated on the same indicator transform, IFs
/// are aligned to the original row universe, and `F_a(c)` / scores are mixed
/// by frozen completion weight — the same rule as class-aware mean CATEs.
#[allow(
    clippy::too_many_lines,
    reason = "for each threshold, refits every identified completion on the exceedance outcome and mixes arm CDFs, influences and supports by frozen completion weight into the same output vectors"
)]
pub(super) fn attach_class_conditional_functional_grid(
    estimate: EffectEstimate,
    data: &TabularData,
    query: &antecedent_core::ConditionalEffectQuery,
    atoms: &[(f64, IdentifiedEstimand)],
    ctx: &ExecutionContext,
) -> Result<EffectEstimate, CausalError> {
    let Some(thresholds) = super::helpers::conditional_thresholds(
        data,
        query,
        atoms.iter().flat_map(|(_, e)| e.adjustment_set.iter().copied()),
    )?
    else {
        return Ok(estimate);
    };
    if atoms.is_empty() {
        return Ok(estimate);
    }
    let est = ConditionalLinearAdjustment::new().with_fold_seed(ctx.rng.master_seed());
    let y_orig = data.float64_values(query.inner.outcome).map_err(CausalError::from)?;
    let mut mixed_cdf = Vec::with_capacity(thresholds.len() * 2);
    let mut mixed_columns = Vec::with_capacity(thresholds.len() * 2);
    let mut event_n_eff = Vec::with_capacity(thresholds.len() * 2);
    let mut threshold_supported = Vec::with_capacity(thresholds.len() * 2);
    let mut n_eff_by_arm = [0.0, 0.0];
    for &threshold in &thresholds {
        let data_c = super::helpers::apply_outcome_functional(
            data,
            query.inner.outcome,
            &antecedent_core::OutcomeFunctional::exceedance(threshold),
        )?;
        let mut arm0 = 0.0;
        let mut arm1 = 0.0;
        let mut mass = 0.0;
        let mut contributing = 0usize;
        let mut atom_if0 = Vec::new();
        let mut atom_if1 = Vec::new();
        let mut atom_weights = Vec::new();
        let mut minimum_events = [f64::INFINITY; 2];
        let mut supported_in_all = [true; 2];
        let mut minimum_arm_counts = [f64::INFINITY; 2];
        for &(w, ref estimand) in atoms {
            if !w.is_finite() || w <= 0.0 {
                continue;
            }
            let mut transformed_query = query.clone();
            transformed_query.inner.outcome_functional = antecedent_core::OutcomeFunctional::Mean;
            let (_point, scores) =
                est.estimate_with_arm_scores(&data_c, estimand, &transformed_query)?;
            arm0 += w * (1.0 - scores.means[0]);
            arm1 += w * (1.0 - scores.means[1]);
            mass += w;
            contributing += 1;
            let aligned0 =
                static_aligned_influence(data, &query.inner, estimand, &scores.influence[0]);
            let aligned1 =
                static_aligned_influence(data, &query.inner, estimand, &scores.influence[1]);
            if let (Some(if0), Some(if1)) = (aligned0, aligned1) {
                atom_if0.push(if0.into_iter().map(|v| -v).collect::<Vec<_>>());
                atom_if1.push(if1.into_iter().map(|v| -v).collect::<Vec<_>>());
                atom_weights.push(w);
            }
            for arm in 0..2 {
                let (events, supported) = super::helpers::tail_event_support(
                    &scores.treatment,
                    &scores.row_index,
                    &y_orig,
                    arm,
                    threshold,
                );
                minimum_events[arm] = minimum_events[arm].min(events);
                supported_in_all[arm] &= supported;
                let count =
                    scores.treatment.iter().filter(|&&t| (t - arm as f64).abs() <= 1e-12).count()
                        as f64;
                minimum_arm_counts[arm] = minimum_arm_counts[arm].min(count);
            }
        }
        if !matches!(mass.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            return Err(CausalError::Compile {
                message: "class-aware ConditionalEffect grid had no estimable atoms".into(),
            });
        }
        mixed_cdf.push(arm0 / mass);
        mixed_cdf.push(arm1 / mass);
        if atom_if0.len() != contributing || atom_if1.len() != contributing {
            return Err(CausalError::Unsupported {
                message: "class-aware ConditionalEffect grid refused: every contributing atom must supply an aligned per-arm influence function; refusing a covariance from a subset of atoms",
            });
        }
        let refs0: Vec<&[f64]> = atom_if0.iter().map(Vec::as_slice).collect();
        let refs1: Vec<&[f64]> = atom_if1.iter().map(Vec::as_slice).collect();
        mixed_columns
            .push(antecedent_estimate::frozen_weight_mixture_scores(&refs0, &atom_weights)?);
        mixed_columns
            .push(antecedent_estimate::frozen_weight_mixture_scores(&refs1, &atom_weights)?);
        n_eff_by_arm = minimum_arm_counts;
        event_n_eff.extend(minimum_events);
        threshold_supported.extend(supported_in_all);
    }
    let mut out = estimate;
    if mixed_columns.len() != thresholds.len() * 2 {
        return Err(CausalError::Unsupported {
            message: "class-aware ConditionalEffect grid refused: joint IF columns must match the declared per-arm threshold grid",
        });
    }
    let refs: Vec<&[f64]> = mixed_columns.iter().map(Vec::as_slice).collect();
    out.joint.covariance = Some(antecedent_estimate::joint_influence_covariance(&refs, None)?);
    let n_eff = n_eff_by_arm[0] + n_eff_by_arm[1];
    out.scores.inference = Some(antecedent_estimate::inference_from_influence_columns(
        &mixed_cdf,
        &refs,
        &event_n_eff,
        &threshold_supported,
        antecedent_estimate::WeightedSupport {
            n_eff,
            n_eff_by_arm: n_eff_by_arm.to_vec(),
            propensity_range: None,
            overlap_ok: n_eff_by_arm
                .iter()
                .all(|n| *n >= antecedent_estimate::scores::MIN_THRESHOLD_EVENTS),
        },
    )?);
    if thresholds.len() == 1 {
        // Scalar exceedance and its CDFs must describe the same fitted functional.
        out.ate = mixed_cdf[0] - mixed_cdf[1];
        let contrast: Vec<_> =
            mixed_columns[0].iter().zip(&mixed_columns[1]).map(|(a, b)| a - b).collect();
        out.uncertainty.analytic.se =
            antecedent_estimate::joint_influence_covariance(&[&contrast], None)?.se(0);
        out.scores.influence = Some(contrast.into());
        out.uncertainty.bootstrap.se = None;
        out.family.simultaneous_interval = None;
        if threshold_supported.iter().any(|&supported| !supported) {
            out.uncertainty.analytic.se = f64::NAN;
            out.scores.influence = None;
        }
    }
    let raw_cdf = mixed_cdf.clone();
    let rearranged = super::helpers::project_conditional_cdf(&mut mixed_cdf)?;
    out = out.with_monotone_rearranged(rearranged);
    out.joint.exceedance_cdf = Some(Arc::from(mixed_cdf));
    if thresholds.len() > 1 {
        out.ate = f64::NAN;
        out.uncertainty.analytic.se = f64::NAN;
        out.uncertainty.bootstrap.se = None;
        out.scores.influence = None;
        out.family.simultaneous_interval = None;
    }
    if let Some(tau) = query.inner.outcome_functional.quantile_level() {
        super::helpers::attach_conditional_quantile(
            &mut out,
            &thresholds,
            &raw_cdf,
            &mixed_columns,
            &threshold_supported,
            tau,
        )?;
    }
    Ok(out)
}

/// Frozen-weight mixture SE from per-atom IFs that share rows.
///
/// Graph weights are modeling choices. Unidentified mass is not mixed in.
pub(super) fn mix_static_envelope_se<T: AsRef<[f64]>>(atom_ifs: &[T], atom_weights: &[f64]) -> f64 {
    if atom_ifs.is_empty() || atom_ifs.len() != atom_weights.len() {
        return f64::NAN;
    }
    let refs: Vec<&[f64]> = atom_ifs.iter().map(AsRef::as_ref).collect();
    let Ok(mixed) = antecedent_estimate::frozen_weight_mixture_scores(&refs, atom_weights) else {
        return f64::NAN;
    };
    mixture_se(&mixed)
}

/// Standard error of an already-mixed influence column.
pub(super) fn mixture_se(mixed: &[f64]) -> f64 {
    antecedent_estimate::joint_influence_covariance(&[mixed], None)
        .map(|c| c.se(0))
        .unwrap_or(f64::NAN)
}

/// Frozen-weight mixture IF, or `None` when atoms do not share a row universe.
pub(super) fn mixed_static_influence<T: AsRef<[f64]>>(
    atom_ifs: &[T],
    atom_weights: &[f64],
) -> Option<std::sync::Arc<[f64]>> {
    if atom_ifs.is_empty() || atom_ifs.len() != atom_weights.len() {
        return None;
    }
    let refs: Vec<&[f64]> = atom_ifs.iter().map(AsRef::as_ref).collect();
    antecedent_estimate::frozen_weight_mixture_scores(&refs, atom_weights)
        .ok()
        .map(std::sync::Arc::from)
}

/// A single contributing atom keeps its SE. Multiple fits require joint
/// sampling covariance (or an explicitly defined graph-mixture distribution).
pub(super) fn mix_weighted_analytic_se(items: impl IntoIterator<Item = (f64, f64)>) -> f64 {
    let mut single = None;
    for (w, se) in items {
        if !w.is_finite() || w < 0.0 {
            return f64::NAN;
        }
        if w == 0.0 {
            continue;
        }
        if single.is_some() || !se.is_finite() || se < 0.0 {
            return f64::NAN;
        }
        single = Some(se);
    }
    single.unwrap_or(f64::NAN)
}

fn estimands_agree(left: &IdentifiedEstimand, right: &IdentifiedEstimand) -> bool {
    left.method == right.method
        && left.adjustment_set == right.adjustment_set
        && left.instruments == right.instruments
        && left.mediators == right.mediators
        && left.rd_design == right.rd_design
}

/// Choose how identified structural atoms may be combined.
///
/// Disagreeing estimand identities never produce a scalar mixture.
/// A shared partially identified estimand publishes an identified set.
pub(super) fn resolve_structural_aggregation(
    contributing: &[&IdentifiedEstimand],
    any_partial: bool,
) -> crate::result::StructuralAggregationPolicy {
    use crate::result::StructuralAggregationPolicy;
    let Some(first) = contributing.first() else {
        return StructuralAggregationPolicy::GraphDependentAtoms;
    };
    if contributing[1..].iter().all(|estimand| estimands_agree(first, estimand)) {
        if any_partial {
            StructuralAggregationPolicy::IdentifiedSetEnvelope
        } else {
            StructuralAggregationPolicy::SameEstimandWeightedMean
        }
    } else {
        StructuralAggregationPolicy::GraphDependentAtoms
    }
}

/// One evaluated graph-posterior atom for policy-aware scalar mixing.
pub(super) struct GraphPosteriorAtomValue {
    pub key: u64,
    pub weight: f64,
    pub status: IdentificationStatus,
    pub estimand: IdentifiedEstimand,
    pub value: f64,
}

/// Policy, scalar, and structural mixture for identified graph-posterior atoms.
pub(super) struct MixedGraphPosteriorPolicy {
    pub policy: crate::result::StructuralAggregationPolicy,
    pub ate: f64,
    pub mixable_scalar: bool,
    pub mixture: crate::result::StructuralResponseMixture,
}

/// Mix identified graph-posterior atoms under [`StructuralAggregationPolicy`].
///
/// Disagreeing estimand identities withhold the scalar (`ate = NaN`) and publish
/// an identified set over atom point values. Unidentified posterior atoms are
/// appended from `graphs` when absent from `atoms`.
pub(super) fn mix_graph_posterior_identified_atoms(
    graphs: &WeightedGraphSamples,
    atoms: &[GraphPosteriorAtomValue],
    unidentified_mass: f64,
    unevaluable_mass: f64,
    subsampled_out_mass: f64,
    truncated_atoms: usize,
) -> MixedGraphPosteriorPolicy {
    use crate::result::{
        StructuralAggregationPolicy, StructuralResponseAtom, StructuralWeightBasis,
    };
    let mut structural_atoms = Vec::with_capacity(atoms.len());
    let mut identified_weight = 0.0;
    let mut mixable = 0.0;
    let mut weighted = 0.0;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut contributing: Vec<&IdentifiedEstimand> = Vec::new();
    for atom in atoms {
        if atom.value.is_finite() {
            identified_weight += atom.weight;
            mixable += atom.weight;
            weighted += atom.weight * atom.value;
            lo = lo.min(atom.value);
            hi = hi.max(atom.value);
            contributing.push(&atom.estimand);
        }
        structural_atoms.push(StructuralResponseAtom {
            graph_key: atom.key,
            weight: atom.weight,
            status: atom.status,
            value: Some(antecedent_core::ResponseValue::Scalar(atom.value)),
            posterior: None,
            response: None,
        });
    }
    let mut present: std::collections::HashSet<u64> =
        structural_atoms.iter().map(|atom| atom.graph_key).collect();
    for (key, weight, flag) in graphs
        .graph_keys
        .iter()
        .zip(graphs.weights.iter())
        .zip(graphs.identified.iter())
        .map(|((k, w), f)| (*k, *w, *f))
    {
        if flag != GraphIdentFlag::Unidentified {
            continue;
        }
        if !present.insert(key) {
            continue;
        }
        structural_atoms.push(StructuralResponseAtom {
            graph_key: key,
            weight,
            status: IdentificationStatus::NotIdentified,
            value: None,
            posterior: None,
            response: None,
        });
    }
    let total = graphs.total_weight();
    let identified_mass = if total > 0.0 { identified_weight / total } else { 0.0 };
    let policy = resolve_structural_aggregation(&contributing, false);
    let identified_set = (lo.is_finite() && hi.is_finite()).then(|| scalar_identified_set(lo, hi));
    let mixable_scalar =
        matches!(policy, StructuralAggregationPolicy::SameEstimandWeightedMean) && mixable > 0.0;
    let ate = if mixable_scalar { weighted / mixable } else { f64::NAN };
    let conditional_on_identified =
        mixable_scalar.then_some(antecedent_core::ResponseValue::Scalar(ate));
    let mixture = crate::result::StructuralResponseMixture {
        weight_basis: StructuralWeightBasis::PosteriorProbability,
        atoms: structural_atoms,
        identified_mass,
        unidentified_mass,
        unevaluable_mass,
        subsampled_out_mass,
        identified_set,
        identified_set_interval: None,
        conditional_on_identified,
        full_mass_scope: true,
        truncated_atoms,
    };
    MixedGraphPosteriorPolicy { policy, ate, mixable_scalar, mixture }
}

/// Publish which [`crate::result::StructuralAggregationPolicy`] governs a
/// graph-posterior mixture.
pub(super) fn push_graph_posterior_structural_aggregation_diagnostic(
    diagnostics: &mut Vec<Diagnostic>,
    policy: crate::result::StructuralAggregationPolicy,
    identified_mass: f64,
    unidentified_mass: f64,
    unevaluable_mass: f64,
    subsampled_out_mass: f64,
) {
    diagnostics.push(
        Diagnostic::new(
            "estimate.graph_posterior.structural_aggregation",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "policy={}; weight_basis=posterior_probability; identified_mass={}; \
                 unidentified_mass={unidentified_mass}; unevaluable_mass={unevaluable_mass}; \
                 subsampled_out_mass={subsampled_out_mass}",
                policy.as_str(),
                identified_mass,
            ),
        )
        .with_fields(mass_fields(Some(identified_mass), unidentified_mass)),
    );
}

/// How a functional estimate resolved variables its functional keeps free outside the
/// query (a complete ID derivation can leave one, as on the napkin graph).
pub(super) fn free_variables_averaged(
    identification: &IdentificationResult,
    estimand: &IdentifiedEstimand,
) -> Option<Diagnostic> {
    let bound: Vec<VariableId> = match &identification.query {
        CausalQuery::Distribution(query) => query
            .outcomes
            .iter()
            .copied()
            .chain(
                query
                    .interventions
                    .iter()
                    .filter_map(antecedent_core::Intervention::primary_variable),
            )
            .chain(query.conditioning.iter().copied())
            .collect(),
        _ => Vec::new(),
    };
    let free = antecedent_estimate::functional_free_variables(
        &identification.arena,
        estimand.functional,
        &bound,
    );
    antecedent_estimate::free_variables_diagnostic(&free)
}

/// Fail-closed rank: larger means less identified. Never used to upgrade a status.
fn identification_closedness(status: IdentificationStatus) -> u8 {
    match status {
        IdentificationStatus::NonparametricallyIdentified => 0,
        IdentificationStatus::IdentifiedUnderParametricRestrictions => 1,
        IdentificationStatus::IdentifiedUnderPriorRestrictions => 2,
        IdentificationStatus::PartiallyIdentified => 3,
        IdentificationStatus::GraphDependent => 4,
        IdentificationStatus::NotIdentified => 5,
    }
}

/// Most conservative status among requested horizons. Empty input is unidentified.
///
/// Priors and later identified horizons do not upgrade an earlier unidentified
/// or graph-dependent slice.
pub(super) fn most_conservative_identification_status(
    statuses: impl IntoIterator<Item = IdentificationStatus>,
) -> IdentificationStatus {
    statuses
        .into_iter()
        .max_by_key(|status| identification_closedness(*status))
        .unwrap_or(IdentificationStatus::NotIdentified)
}

#[derive(Clone, Copy)]
pub(super) struct PosteriorMassSummary {
    pub unidentified: f64,
    pub unevaluable: f64,
    pub subsampled: f64,
    pub failed: f64,
    pub full_scope: bool,
    pub truncated: bool,
}

impl Default for PosteriorMassSummary {
    fn default() -> Self {
        Self {
            unidentified: 0.0,
            unevaluable: 0.0,
            subsampled: 0.0,
            failed: 0.0,
            full_scope: true,
            truncated: false,
        }
    }
}

/// One posterior downgrade ladder.
///
/// Mass above [`antecedent_core::POSTERIOR_MASS_TOLERANCE`], a truncated
/// completion search, an incomplete scope, or disagreeing estimands is
/// [`IdentificationStatus::GraphDependent`]. An identified-set policy is
/// [`IdentificationStatus::PartiallyIdentified`]. Otherwise the atom status
/// is kept. This never promotes `base` to
/// [`IdentificationStatus::NonparametricallyIdentified`].
pub(super) fn resolve_posterior_status(
    base: IdentificationStatus,
    policy: crate::result::StructuralAggregationPolicy,
    mass: PosteriorMassSummary,
) -> IdentificationStatus {
    let exceeds = |value: f64| value > antecedent_core::POSTERIOR_MASS_TOLERANCE;
    if exceeds(mass.unidentified)
        || exceeds(mass.unevaluable)
        || exceeds(mass.subsampled)
        || exceeds(mass.failed)
        || !mass.full_scope
        || mass.truncated
        || matches!(policy, crate::result::StructuralAggregationPolicy::GraphDependentAtoms)
    {
        IdentificationStatus::GraphDependent
    } else if matches!(policy, crate::result::StructuralAggregationPolicy::IdentifiedSetEnvelope) {
        IdentificationStatus::PartiallyIdentified
    } else {
        base
    }
}

/// Status for a Frequentist graph-posterior mixture.
///
/// Unidentified mass is [`IdentificationStatus::GraphDependent`]. Multiple
/// identified atoms with disagreeing estimands are
/// [`IdentificationStatus::PartiallyIdentified`]. A single identified atom
/// (or agreeing atoms) keeps that atom's status.
pub(super) fn graph_posterior_mixture_status(
    unidentified_mass: f64,
    contributing: &[&IdentifiedEstimand],
    primary: IdentificationStatus,
) -> IdentificationStatus {
    if unidentified_mass > antecedent_core::POSTERIOR_MASS_TOLERANCE {
        return IdentificationStatus::GraphDependent;
    }
    let Some(first) = contributing.first() else {
        return IdentificationStatus::NotIdentified;
    };
    if contributing[1..].iter().all(|estimand| estimands_agree(first, estimand)) {
        primary
    } else {
        IdentificationStatus::PartiallyIdentified
    }
}

pub(super) fn temporal_refuter_average(query: &TemporalEffectQuery) -> AverageEffectQuery {
    let mut average = AverageEffectQuery::binary_ate(query.treatment, query.outcome);
    average.active = query.active.clone();
    average.control = query.control.clone();
    average.target_population = query.target_population.clone();
    average
}

pub(super) const SUSTAINED_LINEAR_SEM: &str = "linear additive mechanisms on the identified unfolded DAG; every sustained time is intervened on; frequentist intervals use a shared circular-block row bootstrap whose replicate SD carries the circular-Bartlett fixed-b correction for the block length, Bayesian intervals share each stationary mechanism posterior across time copies, fitting its unique complete observed rows once";

pub(super) const SEQUENCE_LINEAR_SEM: &str = "linear additive mechanisms on the identified unfolded DAG; each Sequence step transforms deterministic propagated means: fixed level, mean + shift, factor * mean, or clip(mean + shift, lower, upper); clipping is not integration over stochastic realizations; no last-step collapse";

pub(super) const UNFOLDED_LINEAR_SEM: &str =
    "linear additive mechanisms on the identified unfolded DAG";

pub(super) const DBN_LINEAR_SEM: &str = "linear additive mechanisms on each identified DBN atom; fixed posterior graph weights; uncertainty uses shared outer circular-block replicates";

/// Identified mass of each graph key of a frozen posterior, summed over every
/// sample that carries the key.
pub(super) struct IdentifiedKeyWeights(std::collections::HashMap<u64, f64>);

impl IdentifiedKeyWeights {
    pub(super) fn new(graphs: &WeightedGraphSamples) -> Self {
        let empty: f64 = std::iter::empty::<f64>().sum();
        let mut weights = std::collections::HashMap::new();
        for ((key, weight), flag) in
            graphs.graph_keys.iter().zip(graphs.weights.iter()).zip(graphs.identified.iter())
        {
            if *flag == GraphIdentFlag::Identified {
                *weights.entry(*key).or_insert(empty) += *weight;
            }
        }
        Self(weights)
    }

    /// Zero (the empty sum) for a key with no identified sample.
    pub(super) fn get(&self, key: u64) -> f64 {
        self.0.get(&key).copied().unwrap_or_else(|| std::iter::empty::<f64>().sum())
    }
}

/// Index of the first occurrence of each key.
pub(super) fn first_key_positions(keys: &[u64]) -> std::collections::HashMap<u64, usize> {
    let mut positions = std::collections::HashMap::with_capacity(keys.len());
    for (index, key) in keys.iter().enumerate() {
        positions.entry(*key).or_insert(index);
    }
    positions
}

pub(super) fn mix_prior_sensitivity_summaries(
    items: &[(f64, &antecedent_prob::PriorSensitivitySummary)],
) -> Option<antecedent_prob::PriorSensitivitySummary> {
    let first = items.first()?.1;
    let n = first.effect_means.len();
    // Grids of different families (or different grids) are not comparable points.
    if n == 0
        || items.iter().any(|(_, s)| {
            s.effect_means.len() != n
                || s.family != first.family
                || s.prior_scales != first.prior_scales
                || s.alphas != first.alphas
                || s.variance_multipliers != first.variance_multipliers
        })
    {
        return None;
    }
    let mut w_sum = 0.0;
    let mut means = vec![0.0; n];
    let mut second = vec![0.0; n];
    for (w, s) in items {
        if *w <= 0.0 {
            continue;
        }
        w_sum += *w;
        for i in 0..n {
            let m = s.effect_means[i];
            let sd = s.effect_sds.get(i).copied().unwrap_or(0.0);
            means[i] += *w * m;
            second[i] += *w * (sd * sd + m * m);
        }
    }
    if w_sum <= 0.0 {
        return None;
    }
    let mut sds = vec![0.0; n];
    for i in 0..n {
        means[i] /= w_sum;
        sds[i] = (second[i] / w_sum - means[i] * means[i]).max(0.0).sqrt();
    }
    Some(antecedent_prob::PriorSensitivitySummary {
        family: first.family,
        prior_scales: Arc::clone(&first.prior_scales),
        alphas: Arc::clone(&first.alphas),
        variance_multipliers: Arc::clone(&first.variance_multipliers),
        effect_means: Arc::from(means),
        effect_sds: Arc::from(sds),
    })
}

/// Whether the atom's posterior check adds the serial-dependence axis.
fn temporal_predictive_check(atom: &EnvelopeAtomFit, refute: RefuteSuite) -> bool {
    matches!(refute, RefuteSuite::Full) && atom.indexer.is_some()
}

/// Whether two envelope atoms would produce identical predictive-check reports:
/// same fitted problem, prior, posterior draws and check variant (the temporal
/// axis also reads the posterior's tempering note).
fn same_predictive_check_inputs(
    a: &EnvelopeAtomFit,
    b: &EnvelopeAtomFit,
    refute: RefuteSuite,
) -> bool {
    let temporal = temporal_predictive_check(a, refute);
    temporal == temporal_predictive_check(b, refute)
        && a.prior == b.prior
        && a.posterior.draws == b.posterior.draws
        && (!temporal || a.posterior.diagnostics.notes == b.posterior.diagnostics.notes)
        && same_fitted_problem(&a.prep, &b.prep)
}

/// How the per-atom refutations of a graph-posterior class are aggregated and attributed.
#[derive(Clone, Copy)]
pub(super) struct AtomRefutationPolicy<'a> {
    pub(super) aggregation: StructuralAggregationPolicy,
    pub(super) estimator: &'a str,
}

/// The posterior and the variables its atoms are rebuilt over.
#[derive(Clone, Copy)]
pub(super) struct GraphPosteriorScope<'a> {
    pub(super) posterior: &'a GraphPosterior,
    pub(super) vars: &'a [VariableId],
}

/// The status and assumptions an aggregated identification publishes.
#[derive(Clone, Copy)]
pub(super) struct IdentificationBasis<'a> {
    pub(super) status: IdentificationStatus,
    pub(super) assumptions: &'a antecedent_core::AssumptionSet,
}

/// How a Bayesian posterior run is validated: the suite, the inference configuration it
/// was fitted under, and the estimator each atom's checks are derived from.
#[derive(Clone, Copy)]
pub(super) struct BayesianValidation<'a> {
    pub(super) suite: RefuteSuite,
    pub(super) cfg: &'a crate::inference::BayesianConfig,
    pub(super) estimator: &'a BayesianGComputationAte,
}

/// Where a validation pass appends the reports and diagnostics it produces.
pub(super) struct ValidationSink<'a> {
    pub(super) refutations: &'a mut Vec<antecedent_validate::RefutationReport>,
    pub(super) diagnostics: &'a mut Vec<Diagnostic>,
}

/// Run PPC (and, under `full`, prior-sensitivity) on identified envelope atoms.
///
/// Aggregation is mixture-weighted by graph-posterior mass. Records the method as a
/// diagnostic so `full` means the same thing on PAG / graph-posterior as on a single DAG.
pub(super) fn run_envelope_bayesian_full_validation(
    setup: BayesianValidation<'_>,
    atoms: &[EnvelopeAtomFit],
    mixture_posterior: &mut CausalPosterior,
    estimate_ate: f64,
    ctx: &ExecutionContext,
    predictive_sims: u32,
    sink: ValidationSink<'_>,
) -> Result<Vec<PredictiveCheckReport>, CausalError> {
    const PPC_ALPHA: f64 = 0.05;
    fn borrow(items: &[(f64, PredictiveCheckReport)]) -> Vec<(f64, &PredictiveCheckReport)> {
        items.iter().map(|(w, r)| (*w, r)).collect()
    }

    let BayesianValidation { suite: refute, cfg, estimator: est } = setup;
    let ValidationSink { refutations, diagnostics } = sink;

    let mut predictive_checks = Vec::new();
    if matches!(refute, RefuteSuite::None) || atoms.is_empty() {
        return Ok(predictive_checks);
    }
    let mut prior_items: Vec<(f64, PredictiveCheckReport)> = Vec::with_capacity(atoms.len());
    let mut post_items: Vec<(f64, PredictiveCheckReport)> = Vec::with_capacity(atoms.len());
    for (j, atom) in atoms.iter().enumerate() {
        // Both checks are deterministic in (design, prior, posterior draws, seed),
        // so atoms that fitted the same problem under the same prior — graphs
        // sharing an adjustment set — replicate the same outcomes: run once.
        if let Some(i) = (0..j).find(|&i| same_predictive_check_inputs(&atoms[i], atom, refute)) {
            prior_items.push((atom.weight, prior_items[i].1.clone()));
            post_items.push((atom.weight, post_items[i].1.clone()));
            continue;
        }
        // Each atom's checks use the prior that atom was actually fitted under.
        let atom_est = BayesianGComputationAte { prior: atom.prior.clone(), ..est.clone() };
        let ppc_prior = atom_est.prior_in_force(atom.prep.design.ncols);
        let prior_rep = PriorPredictiveCheck::for_estimator(&atom_est, ctx)
            .with_n_sims(predictive_sims)
            .check_with_prior(&atom.prep, &ppc_prior, ctx)
            .map_err(CausalError::from)?;
        // Temporal atoms (lag indexer present) add the serial-dependence discrepancy
        // under `full`; exchangeable static rows keep the two-axis check.
        let post_check =
            PosteriorPredictiveCheck::for_estimator(&atom_est, ctx).with_n_sims(predictive_sims);
        let post_rep = if temporal_predictive_check(atom, refute) {
            post_check
                .check_temporal(&atom.prep, &atom.posterior, ctx.rng.master_seed())
                .map_err(CausalError::from)?
        } else {
            post_check.check(&atom.prep, &atom.posterior).map_err(CausalError::from)?
        };
        prior_items.push((atom.weight, prior_rep));
        post_items.push((atom.weight, post_rep));
    }
    for mixed in [
        PredictiveCheckReport::mixture_weighted(&borrow(&prior_items)),
        PredictiveCheckReport::mixture_weighted(&borrow(&post_items)),
    ]
    .into_iter()
    .flatten()
    {
        refutations.push(mixed.to_refutation_report(estimate_ate, PPC_ALPHA));
        predictive_checks.push(mixed);
    }

    // `Some(mixed)` under `full`: whether the per-completion sensitivity grids mixed.
    let mut sensitivity_mixed = None;
    if matches!(refute, RefuteSuite::Full) {
        let mut ws = BayesianGCompWorkspace::default();
        let mut owned = Vec::with_capacity(atoms.len());
        // The grid descriptor is the same for every atom (it depends only on `cfg`),
        // so keep the first one rather than paying for a second full evaluation.
        let mut grid = None;
        for atom in atoms {
            let atom_est = BayesianGComputationAte { prior: atom.prior.clone(), ..est.clone() };
            let (summary, sens) = evaluate_bayesian_prior_sensitivity(
                cfg,
                &atom_est,
                &atom.prep,
                atom.status,
                &atom.posterior,
                &mut ws,
                ctx,
            )?;
            owned.push((atom.weight, summary));
            grid.get_or_insert(sens);
        }
        let sens_items: Vec<_> = owned.iter().map(|(w, s)| (*w, s)).collect();
        let mixed = mix_prior_sensitivity_summaries(&sens_items);
        sensitivity_mixed = Some(mixed.is_some());
        if let (Some(mixed), Some(sens)) = (mixed, grid.as_ref()) {
            refutations.push(sens.to_report(&mixed, estimate_ate));
            *mixture_posterior = with_prior_sensitivity(mixture_posterior.clone(), mixed);
        }
    }

    let atom_keys: String =
        atoms.iter().map(|atom| format!("{:x}", atom.key)).collect::<Vec<_>>().join(",");
    diagnostics.extend(envelope_validation_diagnostics(&atom_keys, sensitivity_mixed));
    Ok(predictive_checks)
}

/// Diagnostics describing what the class-envelope Bayesian validation evaluated.
///
/// `sensitivity_mixed` is `None` without a prior-sensitivity pass (`cheap`), and
/// otherwise whether the per-completion sensitivity summaries shared one grid and
/// so produced a mixture-weighted report. Grids of different perturbed-prior
/// families (a transferred prior on some completions, the isotropic scale on
/// others) are not comparable points, so no prior-sensitivity report is published
/// and the diagnostics say so rather than claiming it was evaluated.
pub(super) fn envelope_validation_diagnostics(
    atom_keys: &str,
    sensitivity_mixed: Option<bool>,
) -> Vec<Diagnostic> {
    let summary = match sensitivity_mixed {
        Some(true) => format!(
            "PPC and prior-sensitivity evaluated per identified completion [{atom_keys}]; \
             reports are mixture-weighted by graph posterior mass"
        ),
        Some(false) => format!(
            "PPC evaluated per identified completion [{atom_keys}]; reports are \
             mixture-weighted by graph posterior mass. Prior sensitivity was computed per \
             completion but no prior-sensitivity report is published (see \
             refute.bayesian.prior_sensitivity.not_mixed)"
        ),
        None => format!(
            "PPC evaluated per identified completion [{atom_keys}]; reports are \
             mixture-weighted by graph posterior mass"
        ),
    };
    let mut out = vec![Diagnostic::new(
        "refute.bayesian.ppc.envelope",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        summary,
    )];
    if sensitivity_mixed == Some(false) {
        out.push(Diagnostic::new(
            "refute.bayesian.prior_sensitivity.not_mixed",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "prior sensitivity was not reported: the identified completions [{atom_keys}] \
                 perturbed incomparable grids (different prior families, e.g. a transferred \
                 prior on some completions and the isotropic scale on others, or different grid \
                 points), so no mixture-weighted summary exists; the result carries no \
                 prior-sensitivity evidence"
            ),
        ));
    }
    out
}

/// The temporal frame refuters refit through when an envelope atom carries a lag indexer.
pub(super) struct TemporalRefutation<'a> {
    pub(super) query: &'a TemporalEffectQuery,
    pub(super) split: Option<&'a DiscoveryEstimationSplit>,
    pub(super) time_index: &'a TimeIndex,
    /// The series has missing time points: refits fit the intact lag windows.
    pub(super) gaps: bool,
}

/// What an envelope refuter run checks and how: the data and query every atom is validated
/// against, the suite, and the temporal frame for DBN atoms.
pub(super) struct EnvelopeRefutation<'a> {
    pub(super) data: &'a TabularData,
    pub(super) query: &'a AverageEffectQuery,
    pub(super) suite: RefuteSuite,
    pub(super) estimator: &'a str,
    pub(super) custom: &'a [Arc<dyn antecedent_validate::CustomEffectValidator>],
    pub(super) temporal: Option<TemporalRefutation<'a>>,
}

impl EnvelopeRefutation<'_> {
    /// Run cheap/full effect refuters on every contributing envelope atom and mix.
    ///
    /// Each atom is validated on its own estimand (and lag indexer, for DBN atoms)
    /// against that atom's own estimate ([`EnvelopeRefuteAtom::original`]), so a
    /// stable atom whose effect differs from the pooled mixture is not reported as
    /// a refutation. Reports of the same refuter id are mixed by posterior mass;
    /// the mixed `original_ate` is therefore the mass-weighted mean of the per-atom
    /// estimates that were actually compared (over the atoms that produced that
    /// report), and the mixed check passes only if every contributing atom passes.
    /// Validators that are `NotApplicable` on every contributing atom surface once;
    /// a check that applies to only a subset mixes that subset.
    pub(super) fn run(
        self,
        atoms: &[EnvelopeRefuteAtom],
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
        let Self { data, query, suite, estimator, custom, temporal: temporal_frame } = self;
        if matches!(suite, RefuteSuite::None) && custom.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let mut per_atom: Vec<(f64, Vec<antecedent_validate::RefutationReport>)> = Vec::new();
        let mut na_weight: std::collections::HashMap<
            antecedent_validate::ValidatorId,
            (f64, Arc<str>),
        > = std::collections::HashMap::new();
        let mut contributing = 0.0;
        let mut diagnostics = Vec::new();
        for atom in atoms {
            if atom.weight <= 0.0 {
                continue;
            }
            contributing += atom.weight;
            let temporal = match (atom.indexer.as_ref(), temporal_frame.as_ref()) {
                (Some(indexer), Some(frame)) => Some(TemporalRefitContext {
                    indexer,
                    temporal_query: frame.query,
                    split: frame.split,
                    kernel_policy: &ctx.kernel_policy,
                    time_index: Some(frame.time_index),
                    panel: None,
                    gaps: frame.gaps,
                }),
                _ => None,
            };
            let outcomes = RefutationRun {
                data,
                estimand: &atom.estimand,
                query,
                estimate: &atom.original,
                suite,
                estimator,
                custom,
                temporal,
            }
            .outcomes(workspace, None, ctx)?;
            per_atom.push((atom.weight, ValidationSuite::reports_only(&outcomes)));
            for (validator, reason) in ValidationSuite::not_applicable_only(&outcomes) {
                na_weight
                    .entry(validator)
                    .and_modify(|(w, _)| *w += atom.weight)
                    .or_insert((atom.weight, reason));
            }
        }
        let (reports, coverage) = mix_atom_refutation_reports(&per_atom);
        diagnostics.extend(coverage);
        if contributing > 0.0 {
            for (validator, (weight, reason)) in na_weight {
                if weight / contributing >= 1.0 - 1e-12 {
                    diagnostics.push(validator_not_applicable_diagnostic(validator, &reason));
                }
            }
        }
        let atom_keys: String =
            atoms.iter().map(|atom| format!("{:x}", atom.key)).collect::<Vec<_>>().join(",");
        diagnostics.push(Diagnostic::new(
            "refute.envelope.effect_mixture",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "effect refuters evaluated each contributing graph atom [{atom_keys}] against that \
             atom's own estimate, not the pooled mixture; reports mix by envelope mass (the \
             mixed original_ate is the mass-weighted mean of the per-atom estimates compared) \
             and pass only if every contributing atom passes"
            ),
        ));
        Ok((reports, diagnostics))
    }
}

/// Mix per-atom refutation reports by refuter id (weights are envelope / graph-posterior mass).
///
/// A refuter that ran on only some of the contributing mass is mixed over the atoms that reported
/// it, so its `passed` says nothing about the rest. The returned diagnostics name each such
/// refuter with the mass it covered (`refute.envelope.partial_coverage`, Warning): a mixed
/// "passed" is never left to read as covering atoms the refuter did not check.
pub(super) fn mix_atom_refutation_reports(
    per_atom: &[(f64, Vec<antecedent_validate::RefutationReport>)],
) -> (Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>) {
    let mut order: Vec<Arc<str>> = Vec::new();
    let mut by_refuter: std::collections::HashMap<
        Arc<str>,
        Vec<(f64, &antecedent_validate::RefutationReport)>,
    > = std::collections::HashMap::new();
    let mut total_mass = 0.0;
    for (weight, reports) in per_atom {
        if *weight <= 0.0 {
            continue;
        }
        total_mass += *weight;
        for report in reports {
            by_refuter
                .entry(Arc::clone(&report.refuter))
                .or_insert_with(|| {
                    order.push(Arc::clone(&report.refuter));
                    Vec::new()
                })
                .push((*weight, report));
        }
    }
    let mut mixed = Vec::with_capacity(order.len());
    let mut diagnostics = Vec::new();
    for id in order {
        let Some(items) = by_refuter.get(&id) else {
            continue;
        };
        let Some(report) = antecedent_validate::RefutationReport::mixture_weighted(items) else {
            continue;
        };
        let covered: f64 = items.iter().map(|(w, _)| *w).sum();
        if total_mass > 0.0 && covered / total_mass < 1.0 - 1e-12 {
            diagnostics.push(Diagnostic::new(
                "refute.envelope.partial_coverage",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                format!(
                    "refuter `{id}` ran on {:.6} of the contributing envelope mass; the other \
                     atoms were not applicable or produced no report, so its mixed verdict does \
                     not cover them",
                    covered / total_mass
                ),
            ));
        }
        mixed.push(report);
    }
    (mixed, diagnostics)
}
