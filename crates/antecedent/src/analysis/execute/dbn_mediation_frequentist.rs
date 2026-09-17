// Frequentist DBN-posterior temporal mediation.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// Replicate streams of the shared mediation mixture bootstrap.
const DBN_MEDIATION_BLOCK_STREAM: u64 = 0xDBF0_0001_0000;

/// The first evaluable atom: its estimand and certificate head the result.
struct PrimaryAtom {
    estimand: IdentifiedEstimand,
    identification: IdentificationResult,
    indexer: TemporalIndexer,
    adjustment_keys: Arc<[antecedent_core::TemporalNodeKey]>,
    lagged: Arc<[antecedent_data::LaggedColumn]>,
}

impl super::Study {
    /// Frequentist DBN-posterior temporal mediation at one horizon.
    ///
    /// Each identified atom uses its own horizon-specific mediation set `S(h)`
    /// (its `I(h)` plus mediator/outcome confounders) and is prepared once on
    /// the original series. With frozen posterior weights, Total, Direct and
    /// Mediated are fixed-weight means over the evaluable atoms. When replicates
    /// are requested, one shared circular-block replicate of lag-aligned series
    /// times refits every atom's three mechanism regressions and mixes them
    /// inside the replicate, so the three SEs describe the reported aggregate and
    /// include between-atom sampling covariance. Unidentified mass is retained;
    /// failed estimation is unevaluable, not unidentified.
    pub(super) fn execute_dbn_posterior_mediation_frequentist(
        &self,
        data: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &antecedent_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        if query.horizons.len() != 1 {
            return Err(CausalError::Unsupported {
                message: "Frequentist DBN-posterior mediation is licensed for one horizon; \
                          multi-horizon grids need their own joint uncertainty contract",
            });
        }
        if self.refute != RefuteSuite::None {
            return Err(CausalError::Unsupported {
                message: "Frequentist DBN-posterior mediation is licensed at validation none; \
                          mediation refuters are not mixed across atoms on this path",
            });
        }
        let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
        let (identified, identify_cached) =
            if let Some(cache) = self.dbn_posterior_identification_cache.as_deref() {
                (cache.clone(), true)
            } else {
                (
                    crate::analysis::prepared::build_dbn_posterior_mediation_identification_cache(
                        gp, &vars, query, ctx,
                    )?,
                    false,
                )
            };
        let horizon = query.horizons[0];
        let identified = identified.mediation_horizon(horizon)?;
        let mut atoms = identified
            .graphs
            .graph_keys
            .iter()
            .zip(identified.graphs.weights.iter())
            .zip(identified.graphs.identified.iter())
            .map(|((&graph_key, &weight), flag)| crate::result::StructuralResponseAtom {
                posterior: None,
                response: None,
                graph_key,
                weight,
                status: if *flag == GraphIdentFlag::Identified {
                    IdentificationStatus::NonparametricallyIdentified
                } else {
                    IdentificationStatus::NotIdentified
                },
                value: None,
            })
            .collect::<Vec<_>>();
        let est = TemporalMediationEstimator::new().with_allow_natural_controlled_alias(true);
        let mut designs = Vec::new();
        let mut weights = Vec::new();
        let mut failed_mass = 0.0;
        let mut primary: Option<PrimaryAtom> = None;
        let mut distinct_sets = false;
        for atom in identified.atoms.iter() {
            let weight = identified_weight_for_key(&identified.graphs, atom.key);
            if weight <= 0.0 {
                continue;
            }
            let slot = atoms.iter_mut().find(|candidate| candidate.graph_key == atom.key);
            let Some(entry) = atom.horizons.as_ref().and_then(|h| h.get(horizon)) else {
                failed_mass += weight;
                continue;
            };
            if let Some(slot) = slot {
                slot.status = entry.identification.status;
            }
            let lagged = super::temporal_path::lagged_adjustment_from_entry(entry);
            let prepared = require_identified(&entry.identification)
                .ok()
                .and_then(|()| est.prepare_shared(data, &entry.estimand, query, &lagged, ctx).ok());
            let Some(prepared) = prepared.filter(|p| p.estimate().effect.ate.is_finite()) else {
                // Identified but not evaluable: the atom keeps its status and
                // has no value, so its mass is unevaluable.
                failed_mass += weight;
                continue;
            };
            if let Some(slot) = atoms.iter_mut().find(|candidate| candidate.graph_key == atom.key) {
                slot.value = Some(ResponseValue::Scalar(prepared.estimate().effect.ate));
            }
            match primary.as_ref() {
                None => {
                    let mut keys = entry
                        .estimand
                        .adjustment_set
                        .iter()
                        .filter_map(|id| entry.indexer.key_of(id.raw()).ok())
                        .collect::<Vec<_>>();
                    keys.sort();
                    primary = Some(PrimaryAtom {
                        estimand: entry.estimand.clone(),
                        identification: entry.identification.clone(),
                        indexer: entry.indexer.clone(),
                        adjustment_keys: Arc::from(keys),
                        lagged,
                    });
                }
                Some(first) if first.lagged.as_ref() != lagged.as_ref() => distinct_sets = true,
                Some(_) => {}
            }
            designs.push(prepared);
            weights.push(weight);
        }
        let PrimaryAtom {
            estimand, mut identification, indexer, adjustment_keys: adjustment, ..
        } = primary.ok_or_else(|| CausalError::Compile {
            message: "Frequentist DBN-posterior mediation has no estimable identified atom".into(),
        })?;
        let total_mass = identified.graphs.total_weight();
        let identified_mass: f64 = weights.iter().sum();
        let unidentified_mass = identified.graphs.unidentified_mass();
        let incomplete = unidentified_mass > 0.0 || failed_mass > 0.0;
        if incomplete {
            identification.status = IdentificationStatus::GraphDependent;
        }
        let mix = |pick: fn(&TemporalMediationEstimate) -> Option<f64>| -> Option<f64> {
            let mut acc = 0.0;
            for (design, weight) in designs.iter().zip(&weights) {
                acc += weight * pick(design.estimate())?;
            }
            Some(acc / identified_mass)
        };
        let point = mix(|m| Some(m.effect.ate)).unwrap_or(f64::NAN);
        let refs: Vec<&antecedent_estimate::PreparedTemporalMediation> = designs.iter().collect();
        let shared = antecedent_estimate::shared_mediation_block_bootstrap(
            &refs,
            &weights,
            query.contrast,
            self.bootstrap_replicates,
            DBN_MEDIATION_BLOCK_STREAM.wrapping_add(u64::from(horizon) << 32),
            ctx,
        );
        let requested_se = shared.as_ref().and_then(|s| s.requested).filter(|s| s.is_finite());
        let replicates_requested = self.bootstrap_replicates > 0;
        let (replicates_ok, replicates_attempted) = shared
            .as_ref()
            .map_or((0, 0), |s| (s.block.replicates_ok, s.block.replicates_attempted));
        let mut assumptions = identification.required_assumptions.clone();
        for record in &designs[0].estimate().effect.assumptions.entries {
            assumptions.push(record.clone());
        }
        let estimate = EffectEstimate::from_parts(
            point,
            f64::NAN,
            requested_se,
            replicates_requested.then_some(replicates_ok),
            replicates_requested.then_some(replicates_attempted.saturating_sub(replicates_ok)),
            ctx.cancellation.is_cancelled(),
            false,
            assumptions,
            OverlapPolicy::ExplicitOverride,
            None,
            None,
        );
        let mediation = TemporalMediationEstimate {
            effect: estimate.clone(),
            total: mix(|m| m.total),
            direct: mix(|m| m.direct),
            mediated: mix(|m| m.mediated),
        };
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(
            Diagnostic::new(
                "estimate.dbn_posterior.mediation.frequentist",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "frozen posterior weights; identified_mass={}; unidentified_mass={}; \
                 unevaluable_mass={}; each atom uses its own S(h) (its I(h) plus \
                 mediator-outcome confounders); Total, Direct and Mediated are fixed-weight \
                 means over evaluable atoms; failed estimation is not mixed into unidentified \
                 mass; the mass split is in structural_response",
                    identified_mass / total_mass,
                    unidentified_mass / total_mass,
                    failed_mass / total_mass
                ),
            )
            .with_fields(super::mass_fields(
                Some(identified_mass / total_mass),
                unidentified_mass / total_mass,
            )),
        );
        diagnostics.push(Diagnostic::new(
            "estimate.dbn_posterior.atom_demotion",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            identified.identify_demotion.summary(0, 0, 0),
        ));
        if distinct_sets {
            diagnostics.push(Diagnostic::new(
                "identify.dbn_posterior.atom_horizon_sets_differ",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "contributing DBN atoms have distinct S(h) adjustment sets at the published \
                 horizon; sets are not unioned across atoms",
            ));
        }
        let family = if designs.len() == 1 {
            antecedent_estimate::CircularBlockFamily::Mediation
        } else {
            antecedent_estimate::CircularBlockFamily::Mixture
        };
        match shared.as_ref() {
            Some(s) if replicates_requested && requested_se.is_some() => {
                diagnostics.push(Diagnostic::new(
                    "estimate.dbn_posterior.mediation.shared_block",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    format!(
                        "iid analytic SEs are NaN for Total, Direct and Mediated; shared \
                         circular-block replicates={}; attempted={}; blocks of {} consecutive \
                         series times (dependence-aware length, at least max(span, \
                         ceil(m^(1/3)))) over the {} times where every atom's lag window is \
                         available; each atom's lag-aligned rows keep their original lag \
                         windows and every atom's three mechanism regressions are refit on the \
                         same resampled times, then mixed with the frozen weights inside the \
                         replicate; replicate SD scaled by the circular-Bartlett fixed-b factor \
                         {:.4} and the Bartlett kernel-bias factor {:.4} of the contrast and \
                         mixture scores; score effective rows {:.1}; between-atom sampling covariance \
                         included; unidentified and unevaluable mass is not mixed into the SE; \
                         the interval is for the reported aggregate, not a distribution over \
                         graph-specific effects",
                        s.block.replicates_ok,
                        s.block.replicates_attempted,
                        s.block.block_length,
                        s.block.rows,
                        antecedent_estimate::circular_fixed_b_scale(
                            s.block.block_length,
                            s.block.rows
                        ),
                        s.block.kernel_bias,
                        s.block.effective_rows,
                    ),
                ));
                let mut warning = short_series_warning(s.block.effective_rows, family);
                if warning.is_none() && designs.len() > 1 {
                    warning = short_series_warning(
                        s.block.effective_rows,
                        antecedent_estimate::CircularBlockFamily::Mediation,
                    );
                }
                diagnostics.extend(warning);
            }
            _ => diagnostics.push(Diagnostic::new(
                "estimate.dbn_posterior.mediation.uncertainty_withheld",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                if replicates_requested {
                    "the shared circular-block bootstrap did not produce a usable SE (too few \
                     successful replicates, or the atoms share no series time); the aggregate \
                     SE is withheld and iid analytic SEs are not substituted"
                } else {
                    "no bootstrap replicates were requested; lagged rows of one series are \
                     serially dependent, so no iid analytic SE is published for the aggregate"
                },
            )),
        }
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        let uncertainty = match shared {
            Some(s) if replicates_requested => {
                antecedent_estimate::TemporalMediationUncertainty::FrequentistBlockBootstrap {
                    requested: s.requested,
                    block: s.block,
                }
            }
            _ => antecedent_estimate::TemporalMediationUncertainty::Unavailable,
        };
        let atom_points: Vec<f64> = designs.iter().map(|d| d.estimate().effect.ate).collect();
        let lower = atom_points.iter().copied().fold(f64::INFINITY, f64::min);
        let upper = atom_points.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let structural_response = crate::result::StructuralResponseMixture {
            weight_basis: crate::result::StructuralWeightBasis::PosteriorProbability,
            atoms,
            identified_mass: identified_mass / total_mass,
            unidentified_mass: unidentified_mass / total_mass,
            unevaluable_mass: failed_mass / total_mass,
            subsampled_out_mass: 0.0,
            identified_set: Some(antecedent_core::ResponseEnvelope {
                grid: Arc::from([f64::from(horizon)]),
                dimension: 1,
                lower: Arc::from([lower]),
                upper: Arc::from([upper]),
            }),
            identified_set_interval: None,
            conditional_on_identified: Some(ResponseValue::Scalar(point)),
            full_mass_scope: true,
            truncated_atoms: 0,
        };
        let mediation_grid = antecedent_estimate::TemporalMediationGrid {
            slices: Arc::from([antecedent_estimate::TemporalMediationSlice {
                horizon,
                identification_status: identification.status,
                method: Arc::clone(&estimand.method),
                adjustment,
                estimate: mediation.clone(),
                uncertainty,
                identified_set: None,
                diagnostics: diagnostics.clone(),
            }]),
            joint_posterior: false,
        };
        let bootstrap_ok = replicates_requested.then_some(replicates_ok);
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification: identification.clone(),
            estimand,
            estimate,
            identifier_id: IdentifierId::Frontdoor,
            estimator_id: EstimatorId::TemporalMediation,
            treatment: query.treatment,
            outcome: query.outcome,
            identify_cached,
            extra_diagnostics: Vec::new(),
            refutations: Vec::new(),
            distribution: None,
            mediation: Some(mediation),
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: bootstrap_ok,
            cancelled: ctx.cancellation.is_cancelled(),
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids(
                    "discover.dbn_posterior",
                    "dbn_posterior",
                )),
                estimate_provenance: Some(provenance_ids(
                    "estimate.dbn_posterior.mediation.frequentist",
                    "estimate.temporal_mediation",
                )),
                structural_response: Some(structural_response),
                mediation_grid: Some(mediation_grid),
                diagnostics: Some(diagnostics),
                certificate: Some(crate::Identification::Point {
                    result: identification,
                    temporal_indexer: Some(indexer),
                    strategy: IdentifierId::Frontdoor,
                    structure_version: self.graph.version(),
                }),
                ..Default::default()
            },
        }))
    }
}
