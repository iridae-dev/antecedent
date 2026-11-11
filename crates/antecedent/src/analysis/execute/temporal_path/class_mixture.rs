//! Temporal class mixture and identification summaries.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, CausalPosterior, CausalResponse, ClassAtomOutcome, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, EffectEstimate, EnvelopeOptions, GraphClass,
    GraphEffectDraws, GraphIdentFlag, IdentificationEnvelope, IdentificationResult,
    IdentificationStatus, InferenceDiagnostics, ResponseUncertainty, WeightedGraphSamples,
    aggregate_effect_envelope, effect_from_posterior, merge_posterior_notes, nan_effect,
    with_conflict_summary,
};

/// The band every completion shares, laid out over the class surface, when each
/// horizon's evaluated completions publish the same estimate and the same
/// pointwise band, retained draws included. A horizon whose completions
/// disagree, or that publishes no band, leaves the class unbanded.
///
/// With one horizon the shared band keeps the draws it was taken from: its
/// cells are that horizon's cells in the same order. Across several horizons
/// each horizon is a separate completion fit whose draws are independent of
/// the others', so the class band retains none rather than pairing them by
/// draw index as if they were one joint law.
#[expect(
    clippy::float_cmp,
    reason = "class members must agree on the exact declared confidence level"
)]
pub(super) fn shared_temporal_class_uncertainty(
    atoms: &[crate::result::StructuralResponseAtom],
    n_horizons: usize,
    cells_per_horizon: usize,
) -> Option<ResponseUncertainty> {
    if n_horizons == 0 || cells_per_horizon == 0 {
        return None;
    }
    let mut exemplars: Vec<Option<&CausalResponse>> = vec![None; n_horizons];
    for atom in atoms {
        if atom.weight <= 0.0 {
            continue;
        }
        let response = atom.response.as_ref()?;
        let horizon = usize::try_from(atom.graph_key >> 32).ok()?;
        let slot = exemplars.get_mut(horizon)?;
        if matches!(response.uncertainty, ResponseUncertainty::None) {
            return None;
        }
        if let Some(first) = slot {
            if response.estimate != first.estimate || response.uncertainty != first.uncertainty {
                return None;
            }
        } else {
            *slot = Some(response);
        }
    }
    let exemplars = exemplars.into_iter().collect::<Option<Vec<_>>>()?;
    let mut lower = vec![0.0; cells_per_horizon * n_horizons];
    let mut upper = vec![0.0; cells_per_horizon * n_horizons];
    let (mut level, mut interpretation) = (None, None);
    let mut shared_draws = None;
    for (horizon, response) in exemplars.iter().enumerate() {
        let ResponseUncertainty::PointwiseBand {
            level: horizon_level,
            lower: horizon_lower,
            upper: horizon_upper,
            interpretation: horizon_interpretation,
            draws: horizon_draws,
        } = &response.uncertainty
        else {
            return None;
        };
        if n_horizons == 1 {
            shared_draws = horizon_draws
                .as_ref()
                .filter(|draws| draws.n_coordinates() == cells_per_horizon)
                .cloned();
        }
        if horizon_lower.len() != cells_per_horizon || horizon_upper.len() != cells_per_horizon {
            return None;
        }
        match (level, interpretation) {
            (None, None) => {
                level = Some(*horizon_level);
                interpretation = Some(*horizon_interpretation);
            }
            (Some(shared_level), Some(shared_interpretation))
                if shared_level == *horizon_level
                    && shared_interpretation == *horizon_interpretation => {}
            _ => return None,
        }
        for cell in 0..cells_per_horizon {
            let index = cell * n_horizons + horizon;
            lower[index] = horizon_lower[cell];
            upper[index] = horizon_upper[cell];
        }
    }
    Some(ResponseUncertainty::PointwiseBand {
        level: level?,
        lower: Arc::from(lower),
        upper: Arc::from(upper),
        interpretation: interpretation?,
        draws: shared_draws,
    })
}

/// The completion-envelope diagnostic every class response publishes, plus the
/// Bayesian note when no class prior supports a posterior mixture.
pub(in crate::analysis::execute) fn class_response_identified_set_diagnostics(
    bayesian: bool,
    class_prior: bool,
) -> Vec<Diagnostic> {
    let mut out = vec![Diagnostic::new(
        "estimate.temporal_response.class_identified_set",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "bounds are pointwise ranges over completion-specific point surfaces; completion \
         weights are enumeration weights, not posterior probabilities or sampling uncertainty",
    )];
    if bayesian && !class_prior {
        out.push(Diagnostic::new(
            "estimate.envelope.response_posterior_not_mixed",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "Bayesian completion posteriors are retained as atoms; enumeration weights \
             are not mixed into a posterior",
        ));
    }
    out
}

/// Completion-enumeration weights are not a class prior.
pub(in crate::analysis::execute) fn enumeration_not_probability_diagnostic() -> Diagnostic {
    Diagnostic::new(
        "estimate.temporal_class.enumeration_not_probability",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "completion-enumeration weights are not a class prior; no blended posterior \
         is published",
    )
}

/// Inputs of a Bayesian class-effect mixture over completion posteriors.
pub(in crate::analysis::execute) struct ClassPosteriorMix<'a> {
    /// A caller class prior supplied the completion masses.
    pub class_prior_supplied: bool,
    pub truncated_completions: usize,
    /// Some evaluated completion carries positive mass.
    pub evaluable_mass_positive: bool,
    pub weights: Vec<f64>,
    pub flags: Vec<GraphIdentFlag>,
    pub keys: Vec<u64>,
    pub per_graph: &'a [GraphEffectDraws],
    pub conflict: Option<antecedent_prob::ConflictSummary>,
    /// Inference label of the mixed posterior.
    pub label: &'static str,
}

/// The class-prior contract of a Bayesian class effect, shared by the series and
/// panel owners: with a caller class prior over an uncapped class and positive
/// evaluable mass, the completion posteriors' draws are mixed by class mass
/// (within- plus between-completion variance); otherwise the effect is NaN and the
/// completion posteriors stay atoms of the structural response.
pub(in crate::analysis::execute) fn mix_class_effect_posteriors<'a>(
    mix: ClassPosteriorMix<'_>,
    atom_posteriors: impl IntoIterator<Item = &'a CausalPosterior>,
    diagnostics: &mut Vec<Diagnostic>,
) -> Result<
    (EffectEstimate, Option<CausalPosterior>, crate::result::StructuralWeightBasis),
    CausalError,
> {
    let basis = if mix.class_prior_supplied {
        crate::result::StructuralWeightBasis::CallerSuppliedClassPrior
    } else {
        crate::result::StructuralWeightBasis::CompletionEnumeration
    };
    if !(mix.class_prior_supplied && mix.truncated_completions == 0 && mix.evaluable_mass_positive)
    {
        diagnostics.push(Diagnostic::new(
            "estimate.envelope.response_posterior_not_mixed",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "completion posteriors are retained as atoms; mixing requires caller-supplied \
             class mass, positive evaluable mass, and uncapped class scope",
        ));
        return Ok((nan_effect(), None, basis));
    }
    let graphs = WeightedGraphSamples::new(mix.weights, mix.flags, mix.keys)?;
    let mut mixed = aggregate_effect_envelope(
        &graphs,
        mix.per_graph,
        InferenceDiagnostics::analytic(mix.label),
        EnvelopeOptions::default(),
    )
    .map_err(CausalError::from)?;
    merge_posterior_notes(&mut mixed, atom_posteriors);
    if let Some(summary) = mix.conflict {
        mixed = with_conflict_summary(mixed, summary);
    }
    let estimate = effect_from_posterior(&mixed)?;
    diagnostics.push(
        Diagnostic::new(
            "estimate.temporal_class.envelope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!("unidentified_mass={}", mixed.unidentified_mass),
        )
        .with_fields(super::super::mass_fields(None, mixed.unidentified_mass)),
    );
    Ok((estimate, Some(mixed), basis))
}

/// Structural uncertainty of a temporal class envelope, keyed by completion
/// fingerprint.
///
/// The mixture itself is [`super::super::class_structural_mixture`],
/// shared with the static class routes, so the mass split here is the
/// envelope's own ([`antecedent_identify::carries_identified_mass`]) and
/// agrees with the `identify.temporal_*.envelope` diagnostic published on the
/// same result. Unlike the static arm, a temporal class result always carries
/// its mixture: the completion atoms are part of the answer even when no
/// completion produced a finite value.
pub(crate) fn temporal_class_structural_mixture(
    envelope: &IdentificationEnvelope<antecedent_identify::TemporalCompletionGraph>,
    weight_basis: crate::result::StructuralWeightBasis,
    masses: Option<&[f64]>,
    values: &[Option<f64>],
) -> crate::result::StructuralResponseMixture {
    let outcomes: Vec<_> = values
        .iter()
        .map(|value| value.map_or(ClassAtomOutcome::NotEvaluated, ClassAtomOutcome::Evaluated))
        .collect();
    super::super::class_structural_mixture(
        envelope,
        weight_basis,
        |_, case| case.graph.fingerprint(),
        masses,
        &outcomes,
    )
    .0
}

pub(crate) fn temporal_class_envelope_diagnostic<G>(
    envelope: &IdentificationEnvelope<G>,
    class: GraphClass,
) -> Diagnostic {
    let code = if class == GraphClass::TemporalCpdag {
        "identify.temporal_cpdag.envelope"
    } else {
        "identify.temporal_pag.envelope"
    };
    super::super::class_envelope_diagnostic(
        code,
        "generalized.adjustment envelope",
        envelope,
        &[("limitations", format!("{:?}", envelope.critical_graph_features))],
    )
}

pub(crate) fn aggregate_temporal_horizon_evidence<'a>(
    identifications: impl IntoIterator<Item = &'a IdentificationResult>,
) -> Result<(IdentificationStatus, antecedent_core::AssumptionSet), CausalError> {
    let mut aggregate_status = IdentificationStatus::NonparametricallyIdentified;
    let mut assumptions = antecedent_core::AssumptionSet::new();
    let mut saw_any = false;
    for identification in identifications {
        saw_any = true;
        match identification.status {
            IdentificationStatus::NonparametricallyIdentified => {}
            IdentificationStatus::IdentifiedUnderParametricRestrictions => {
                aggregate_status = IdentificationStatus::IdentifiedUnderParametricRestrictions;
            }
            status => {
                return Err(CausalError::Compile {
                    message: format!(
                        "temporal response requires point identification at every horizon; got {status:?}"
                    ),
                });
            }
        }
        assumptions.extend_unique(&identification.required_assumptions.entries);
    }
    if !saw_any {
        return Err(CausalError::Compile {
            message: "temporal response requires at least one horizon identification".into(),
        });
    }
    Ok((aggregate_status, assumptions))
}
