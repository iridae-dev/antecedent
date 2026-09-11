//! Tier-closure identification as a fast path over generalized adjustment.
//!
//! Not new identification theory. [`WithinTier::CoDetermined`] certifies the
//! tier-closure set in `O(p)`. [`WithinTier::Unknown`] returns two canonical
//! sets as an envelope and never collapses to one.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, AverageEffectQuery, CausalQuery, CausalSchema, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, ResponseQuery,
};
use antecedent_expr::CausalExprArena;
use antecedent_graph::{Admg, DenseNodeId, TieredBackground, WithinTier};

use crate::generalized::not_identified;
use crate::joint_response::{joint_adjustment_holds, joint_result, prepare_joint_response};

use crate::envelope::{GraphIdentificationCase, IdentificationEnvelope, ProbabilityMass};
use crate::error::IdentificationError;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentifiedEstimand,
};

/// Named premise: no latent path into the outcome from outside the tier order.
pub const NO_LATENT_TO_OUTCOME: &str = "tiered.no_latent_to_outcome";

/// A specific `CoDetermined` pair has no joint generalized-adjustment set on the
/// known closure ADMG (open back-door, or a drawn treatment↔outcome edge).
pub const TIERED_JOINT_ADJUSTMENT_REFUSE: &str = "no joint generalized back-door set on the CoDetermined closure ADMG; open back-door or a drawn treatment↔outcome edge";

/// Unknown-tier joint cells have no single ADMG and no cell-AIPW score table.
pub const TIERED_JOINT_UNKNOWN_REFUSE: &str = "Unknown-tier joint InterventionResponse has no single ADMG; two canonical scenarios are not a cell-AIPW license";

/// Identify an average effect under a declared tier background.
///
/// `CoDetermined` returns one certified generalized-adjustment set. Unknown
/// returns a two-estimand [`IdentificationResult`] with
/// [`antecedent_core::IdentificationStatus::GraphDependent`]; callers that
/// must not collapse should use [`identify_tiered_envelope`].
///
/// # Errors
///
/// Missing tier membership, outcome not after treatment, or non-Set levels.
pub fn identify_tiered(
    background: &TieredBackground,
    query: &AverageEffectQuery,
) -> Result<IdentificationResult, IdentificationError> {
    match background.within_tier {
        WithinTier::CoDetermined => identify_closure(background, query),
        WithinTier::Unknown => {
            let [pre, closure] =
                background.unknown_canonical_sets(query.treatment, query.outcome)?;
            let (estimands, arena) = estimands_for_sets(
                query,
                &[
                    (pre.as_ref(), "tiered.unknown.pretreatment"),
                    (closure.as_ref(), "tiered.unknown.closure"),
                ],
            )?;
            let mut derivation = DerivationTrace::default();
            derivation.push(
                "tiered.unknown.envelope",
                "pretreatment-only and tier-closure estimated as an envelope; never one set",
            );
            let mut result = IdentificationResult::identified(
                CausalQuery::AverageEffect(query.clone()),
                estimands,
                arena,
                derivation,
                named_no_latent_assumption(),
                IdentificationPerformanceRecord { candidates_examined: 2, sets_returned: 2 },
            );
            result.status = antecedent_core::IdentificationStatus::GraphDependent;
            result.diagnostics.push(no_latent_diagnostic());
            result.diagnostics.push(Diagnostic::new("tiered.unknown.canonical_scenarios", DiagnosticKind::Scientific, DiagnosticSeverity::Warning,
                "two declared orientation scenarios only: treatment precedes all same-tier peers, or follows all peers; these are not exhaustive completions or bounds over the unknown graph class"));
            Ok(result)
        }
    }
}

/// Unknown envelope: two equally weighted canonical sets, never collapsed.
///
/// # Errors
///
/// Same as [`identify_tiered`].
pub fn identify_tiered_envelope(
    background: &TieredBackground,
    query: &AverageEffectQuery,
    graph: &Admg,
) -> Result<IdentificationEnvelope<Admg>, IdentificationError> {
    let [pre, closure] = background.unknown_canonical_sets(query.treatment, query.outcome)?;
    let pre_id = identified_one(query, &pre, "tiered.unknown.pretreatment")?;
    let clo_id = identified_one(query, &closure, "tiered.unknown.closure")?;
    let n = u32::try_from(graph.node_count())
        .map_err(|_| antecedent_graph::GraphError::TooManyNodes)?;
    let mut before = Admg::with_variables(n);
    let mut after = Admg::with_variables(n);
    for (k, tier) in background.tiers.iter().enumerate() {
        for later in background.tiers.iter().skip(k + 1) {
            for &a in tier.iter() {
                for &b in later.iter() {
                    before.insert_directed(
                        antecedent_graph::DenseNodeId::from_raw(a.raw()),
                        antecedent_graph::DenseNodeId::from_raw(b.raw()),
                    )?;
                    after.insert_directed(
                        antecedent_graph::DenseNodeId::from_raw(a.raw()),
                        antecedent_graph::DenseNodeId::from_raw(b.raw()),
                    )?;
                }
            }
        }
        let mut peers = tier.to_vec();
        peers.sort_by_key(|v| (u8::from(*v != query.treatment), v.raw()));
        for (i, &a) in peers.iter().enumerate() {
            for &b in peers.iter().skip(i + 1) {
                before.insert_directed(
                    antecedent_graph::DenseNodeId::from_raw(a.raw()),
                    antecedent_graph::DenseNodeId::from_raw(b.raw()),
                )?;
            }
        }
        peers.sort_by_key(|v| (u8::from(*v == query.treatment), v.raw()));
        for (i, &a) in peers.iter().enumerate() {
            for &b in peers.iter().skip(i + 1) {
                after.insert_directed(
                    antecedent_graph::DenseNodeId::from_raw(a.raw()),
                    antecedent_graph::DenseNodeId::from_raw(b.raw()),
                )?;
            }
        }
    }
    Ok(IdentificationEnvelope::from_cases(vec![
        GraphIdentificationCase { graph: before, result: pre_id, weight: ProbabilityMass(0.5) },
        GraphIdentificationCase { graph: after, result: clo_id, weight: ProbabilityMass(0.5) },
    ]))
}

/// Joint `do(T…)` on the [`WithinTier::CoDetermined`] closure ADMG.
///
/// `CoDetermined` is background, not a MAG Markov equivalence class: earlier→later
/// arrows are asserted, same-tier edges are bidirected, and there is no latent
/// path into Y beyond what the tier graph draws. Identification is the `O(p)`
/// treatment-set closure (all non-treatment nodes in tiers `≤` the latest
/// treatment), verified with one generalized-adjustment check per treatment.
/// This is not subset search and does not call the generic candidate enumerator.
/// Unknown tiers refuse: two canonical scenarios are not a single ADMG.
///
/// # Errors
///
/// Unknown within-tier interpretation, invalid query, or graph errors.
pub fn identify_tiered_joint(
    background: &TieredBackground,
    schema: &CausalSchema,
    query: &ResponseQuery,
) -> Result<IdentificationResult, IdentificationError> {
    match background.within_tier {
        WithinTier::Unknown => Err(IdentificationError::unsupported(TIERED_JOINT_UNKNOWN_REFUSE)),
        WithinTier::CoDetermined => {
            let admg = background.to_admg(schema)?;
            identify_joint_closure_on(background, &admg, query)
        }
    }
}

/// Joint ID on an already-materialized CoDetermined closure ADMG.
///
/// # Errors
///
/// Unknown within-tier interpretation or invalid query.
pub fn identify_tiered_joint_on(
    background: &TieredBackground,
    admg: &Admg,
    query: &ResponseQuery,
) -> Result<IdentificationResult, IdentificationError> {
    match background.within_tier {
        WithinTier::Unknown => Err(IdentificationError::unsupported(TIERED_JOINT_UNKNOWN_REFUSE)),
        WithinTier::CoDetermined => identify_joint_closure_on(background, admg, query),
    }
}

fn identify_joint_closure_on(
    background: &TieredBackground,
    admg: &Admg,
    query: &ResponseQuery,
) -> Result<IdentificationResult, IdentificationError> {
    let prepared = prepare_joint_response(admg.nodes(), query)?;
    let set = background.tier_closure_set(&prepared.treatments, prepared.outcome)?;
    let z: Vec<DenseNodeId> = set.iter().map(|v| DenseNodeId::from_raw(v.raw())).collect();
    let holds = joint_adjustment_holds(admg, &prepared.targets, prepared.y, &z, |_, _| true)?;
    if !holds {
        let mut result = not_identified(prepared.query, TIERED_JOINT_ADJUSTMENT_REFUSE);
        result.required_assumptions = named_no_latent_assumption();
        result.derivation.push("tiered.joint.closure_admg", TIERED_JOINT_ADJUSTMENT_REFUSE);
        result.diagnostics.push(no_latent_diagnostic());
        result.diagnostics.push(Diagnostic::new(
            "identify.tiered.joint.adjustment",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            TIERED_JOINT_ADJUSTMENT_REFUSE,
        ));
        result.performance.candidates_examined = prepared.treatments.len() as u64;
        return Ok(result);
    }
    let mut result = joint_result(
        prepared.query,
        admg.nodes(),
        &z,
        &prepared.treatments,
        prepared.outcome,
        prepared.treatments.len() as u64,
    );
    result.required_assumptions = named_no_latent_assumption();
    result.derivation.push(
        "tiered.joint.closure",
        format!(
            "O(p) treatment-set closure |Z|={} certified with one m-separation check per treatment; not subset search",
            set.len()
        ),
    );
    result.derivation.push(
        "tiered.joint.closure_admg",
        "joint ADMG generalized adjustment on the CoDetermined closure (known ancestral ADMG, not a MAG Markov equivalence class)",
    );
    result.diagnostics.push(no_latent_diagnostic());
    Ok(result)
}

fn identify_closure(
    background: &TieredBackground,
    query: &AverageEffectQuery,
) -> Result<IdentificationResult, IdentificationError> {
    let set = background.tier_closure(query.treatment, query.outcome)?;
    let (estimand, arena) = estimand_for_set(query, &set, "generalized.adjustment")?;
    let mut derivation = DerivationTrace::default();
    derivation.push(
        "tiered.closure",
        format!("O(p) tier-closure |Z|={} certified as generalized.adjustment", set.len()),
    );
    let mut result = IdentificationResult::identified(
        CausalQuery::AverageEffect(query.clone()),
        vec![estimand],
        arena,
        derivation,
        named_no_latent_assumption(),
        IdentificationPerformanceRecord { candidates_examined: 1, sets_returned: 1 },
    );
    result.diagnostics.push(no_latent_diagnostic());
    Ok(result)
}

fn identified_one(
    query: &AverageEffectQuery,
    set: &[antecedent_core::VariableId],
    method: &'static str,
) -> Result<IdentificationResult, IdentificationError> {
    let (estimand, arena) = estimand_for_set(query, set, method)?;
    let mut derivation = DerivationTrace::default();
    derivation.push(method, format!("|Z|={}", set.len()));
    Ok(IdentificationResult::identified(
        CausalQuery::AverageEffect(query.clone()),
        vec![estimand],
        arena,
        derivation,
        {
            let mut assumptions = named_no_latent_assumption();
            assumptions.push(AssumptionRecord {
                assumption: Assumption::Custom { id: Arc::from(method), description: Arc::from(
                    if method.ends_with("pretreatment") { "treatment precedes every same-tier peer; all tiers follow a causal order with no cross-tier latent confounding" }
                    else { "treatment follows every same-tier peer; all tiers follow a causal order with no cross-tier latent confounding" }) },
                source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from("tiered.unknown") },
                scope: AssumptionScope::Identification, status: AssumptionStatus::Declared,
            });
            assumptions
        },
        IdentificationPerformanceRecord { candidates_examined: 1, sets_returned: 1 },
    ))
}

fn estimand_for_set(
    query: &AverageEffectQuery,
    set: &[antecedent_core::VariableId],
    method: &'static str,
) -> Result<(IdentifiedEstimand, CausalExprArena), IdentificationError> {
    let (estimands, arena) = estimands_for_sets(query, &[(set, method)])?;
    Ok((estimands.into_iter().next().expect("one set"), arena))
}

fn estimands_for_sets(
    query: &AverageEffectQuery,
    sets: &[(&[antecedent_core::VariableId], &'static str)],
) -> Result<(Vec<IdentifiedEstimand>, CausalExprArena), IdentificationError> {
    let active = crate::intervention_support::require_set_value(&query.active, "tiered")?;
    let control = crate::intervention_support::require_set_value(&query.control, "tiered")?;
    let mut arena = CausalExprArena::new();
    let mut estimands = Vec::with_capacity(sets.len());
    for (set, method) in sets {
        let functional = arena.backdoor_ate(
            query.treatment,
            query.outcome,
            set,
            active.clone(),
            control.clone(),
        );
        estimands.push(IdentifiedEstimand::backdoor(*method, Arc::from(set.to_vec()), functional));
    }
    Ok((estimands, arena))
}

fn named_no_latent_assumption() -> AssumptionSet {
    let mut set = AssumptionSet::new();
    set.push(AssumptionRecord {
        assumption: Assumption::Custom {
            id: Arc::from(NO_LATENT_TO_OUTCOME),
            description: Arc::from(
                "no latent path into the outcome from outside the declared tier order",
            ),
        },
        source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from("tiered.closure") },
        scope: AssumptionScope::Identification,
        status: AssumptionStatus::Declared,
    });
    set.push(crate::assumptions::causal_markov("tiered.closure"));
    set
}

fn no_latent_diagnostic() -> Diagnostic {
    Diagnostic::new(
        NO_LATENT_TO_OUTCOME,
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "tier-closure identification assumes no latent path to the outcome; \
         attach an E-value on the estimate",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generalized::GeneralizedAdjustmentIdentifier;
    use antecedent_core::{
        CausalSchemaBuilder, IdentificationStatus, Intervention, ResponseFunctional, Value,
        VariableId,
    };
    use antecedent_graph::WithinTier;

    /// `CoDetermined` facet tier: earlier confounder, bidirected clique, outcome.
    /// `n` is the node count (`z` + `t1` + `t2` + co-facets + `y`).
    fn facet_width_joint(n: u32) -> (antecedent_core::CausalSchema, TieredBackground) {
        assert!(n >= 4);
        let mut b = CausalSchemaBuilder::new();
        b = b.continuous("z").finish();
        b = b.continuous("t1").finish();
        b = b.continuous("t2").finish();
        let mut treatment_tier = vec!["t1".to_string(), "t2".to_string()];
        for i in 0..(n - 4) {
            let name = format!("f{i}");
            b = b.continuous(name.clone()).finish();
            treatment_tier.push(name);
        }
        b = b.continuous("y").finish();
        let schema = b.build().unwrap();
        let tier_refs: Vec<&str> = treatment_tier.iter().map(String::as_str).collect();
        let background = TieredBackground::from_named(
            &schema,
            &[vec!["z"], tier_refs, vec!["y"]],
            WithinTier::CoDetermined,
        )
        .unwrap();
        (schema, background)
    }

    fn assert_joint_width_identified(n: u32, elapsed_ok: impl Fn(std::time::Duration) -> bool) {
        let (schema, background) = facet_width_joint(n);
        let query = joint_query(&schema, "t1", "t2", "y");
        let expected = background
            .tier_closure_set(
                &[schema.id_of("t1").unwrap(), schema.id_of("t2").unwrap()],
                schema.id_of("y").unwrap(),
            )
            .unwrap();
        assert_eq!(expected.len(), n as usize - 3, "Z = closure minus treatments");
        assert!(expected.iter().any(|&v| v == schema.id_of("z").unwrap()));
        assert!(!expected.iter().any(|&v| v == schema.id_of("t1").unwrap()));
        assert!(!expected.iter().any(|&v| v == schema.id_of("t2").unwrap()));
        let started = std::time::Instant::now();
        let id = identify_tiered_joint(&background, &schema, &query).unwrap();
        let elapsed = started.elapsed();
        assert!(elapsed_ok(elapsed), "{n}-node joint closure took {elapsed:?}");
        assert_eq!(id.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(id.estimands[0].adjustment_set.as_ref(), expected.as_ref());
        assert_eq!(id.performance.candidates_examined, 2);
        assert!(id.required_assumptions.entries.iter().any(|a| matches!(
            &a.assumption,
            Assumption::Custom { id, .. } if id.as_ref() == NO_LATENT_TO_OUTCOME
        )));
        assert!(id.diagnostics.iter().any(|d| d.code.as_ref() == NO_LATENT_TO_OUTCOME));
    }

    fn joint_query(
        schema: &antecedent_core::CausalSchema,
        t1: &str,
        t2: &str,
        y: &str,
    ) -> ResponseQuery {
        ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: schema.id_of(y).unwrap(),
            interventions: Arc::from([
                Intervention::set(schema.id_of(t1).unwrap(), Value::f64(1.0)),
                Intervention::set(schema.id_of(t2).unwrap(), Value::f64(1.0)),
            ]),
        })
    }

    fn bg() -> (antecedent_core::CausalSchema, TieredBackground) {
        let schema = CausalSchemaBuilder::new()
            .continuous("era")
            .finish()
            .continuous("t")
            .finish()
            .continuous("y")
            .finish()
            .build()
            .unwrap();
        let bg = TieredBackground::from_named(
            &schema,
            &[vec!["era"], vec!["t"], vec!["y"]],
            WithinTier::CoDetermined,
        )
        .unwrap();
        (schema, bg)
    }

    #[test]
    fn codetermined_certifies_closure() {
        let (schema, background) = bg();
        let q =
            AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
        let id = identify_tiered(&background, &q).unwrap();
        assert_eq!(id.estimands.len(), 1);
        assert_eq!(id.estimands[0].adjustment_set.as_ref(), &[schema.id_of("era").unwrap()]);
        assert!(id.required_assumptions.entries.iter().any(|a| matches!(
            &a.assumption,
            Assumption::Custom { id, .. } if id.as_ref() == NO_LATENT_TO_OUTCOME
        )));
    }

    #[test]
    fn unknown_keeps_two_sets() {
        let schema = CausalSchemaBuilder::new()
            .continuous("era")
            .finish()
            .continuous("scale")
            .finish()
            .continuous("design")
            .finish()
            .continuous("t")
            .finish()
            .continuous("y")
            .finish()
            .build()
            .unwrap();
        let background = TieredBackground::from_named(
            &schema,
            &[vec!["era", "scale"], vec!["design", "t"], vec!["y"]],
            WithinTier::Unknown,
        )
        .unwrap();
        let q =
            AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
        let id = identify_tiered(&background, &q).unwrap();
        assert_eq!(id.estimands.len(), 2);
        assert_ne!(id.estimands[0].adjustment_set, id.estimands[1].adjustment_set);
        assert_eq!(id.status, antecedent_core::IdentificationStatus::GraphDependent);
        let _ = VariableId::from_raw(0);
    }

    #[test]
    fn closure_is_linear_in_p() {
        let n = 200u32;
        let mut b = CausalSchemaBuilder::new();
        let mut names = Vec::new();
        for i in 0..n {
            let name = format!("v{i}");
            b = b.continuous(name.clone()).finish();
            names.push(name);
        }
        let schema = b.build().unwrap();
        let mut tiers = Vec::new();
        let chunk = 10usize;
        for c in names.chunks(chunk) {
            tiers.push(c.iter().map(String::as_str).collect::<Vec<_>>());
        }
        let background =
            TieredBackground::from_named(&schema, &tiers, WithinTier::CoDetermined).unwrap();
        let t = schema.id_of(&names[50]).unwrap();
        let y = schema.id_of(&names[n as usize - 1]).unwrap();
        let q = AverageEffectQuery::binary_ate(t, y);
        let started = std::time::Instant::now();
        let id = identify_tiered(&background, &q).unwrap();
        assert!(
            started.elapsed().as_millis() < 200,
            "200-node tier ID took {:?}",
            started.elapsed()
        );
        assert!(!id.estimands[0].adjustment_set.is_empty());
    }

    #[test]
    fn identification_p667_under_one_second() {
        let n = 667u32;
        let mut b = CausalSchemaBuilder::new();
        let mut names = Vec::new();
        for i in 0..n {
            let name = format!("v{i}");
            b = b.continuous(name.clone()).finish();
            names.push(name);
        }
        let schema = b.build().unwrap();
        let mut tiers = Vec::new();
        for c in names.chunks(7) {
            tiers.push(c.iter().map(String::as_str).collect::<Vec<_>>());
        }
        let background =
            TieredBackground::from_named(&schema, &tiers, WithinTier::CoDetermined).unwrap();
        let t = schema.id_of(&names[80]).unwrap();
        let y = schema.id_of(&names[n as usize - 1]).unwrap();
        let q = AverageEffectQuery::binary_ate(t, y);
        let started = std::time::Instant::now();
        let id = identify_tiered(&background, &q).unwrap();
        assert!(
            started.elapsed().as_secs_f64() < 1.0,
            "p=667 tier ID took {:?}",
            started.elapsed()
        );
        assert!(!id.estimands[0].adjustment_set.is_empty());
    }

    #[test]
    fn joint_closure_is_linear_in_p() {
        // Unoptimized measurement: ~230 ms on the 200-node facet clique
        // (bidirected C(198,2) plus earlier confounder and Y). Single-lever
        // 200-node is chunked tiers and stays under 200 ms; this is the
        // recorded joint-path bound.
        assert_joint_width_identified(200, |elapsed| elapsed.as_millis() < 500);
    }

    #[test]
    #[ignore = "CI width: joint p=667 clique; keep 20 + 200 default"]
    fn joint_identification_p667_under_recorded_bound() {
        // Unoptimized measurement: ~8.1 s on the p=667 facet clique
        // (~221k bidirected edges). Single-lever p=667 is chunked and stays
        // under 1 s; this is the recorded joint-path bound, not that bench.
        assert_joint_width_identified(667, |elapsed| elapsed.as_secs_f64() < 16.0);
    }

    #[test]
    fn same_tier_pair_is_joint_admg_adjustment() {
        let schema = CausalSchemaBuilder::new()
            .continuous("z")
            .finish()
            .continuous("t1")
            .finish()
            .continuous("t2")
            .finish()
            .continuous("y")
            .finish()
            .build()
            .unwrap();
        let background = TieredBackground::from_named(
            &schema,
            &[vec!["z"], vec!["t1", "t2"], vec!["y"]],
            WithinTier::CoDetermined,
        )
        .unwrap();
        let admg = background.to_admg(&schema).unwrap();
        let t1 = antecedent_graph::DenseNodeId::from_raw(schema.id_of("t1").unwrap().raw());
        let t2 = antecedent_graph::DenseNodeId::from_raw(schema.id_of("t2").unwrap().raw());
        let mut descendants = antecedent_graph::BitSet::default();
        let mut ws = antecedent_graph::GraphWorkspace::default();
        admg.descendants_of(&[t1], &mut descendants, &mut ws);
        assert!(!descendants.contains(t2), "walking ↔ as a directed path would put t2 in De(t1)");
        let id =
            identify_tiered_joint(&background, &schema, &joint_query(&schema, "t1", "t2", "y"))
                .unwrap();
        assert_eq!(id.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(id.estimands[0].adjustment_set.as_ref(), &[schema.id_of("z").unwrap()]);
        assert!(!id.estimands[0].adjustment_set.iter().any(|&v| v == schema.id_of("t2").unwrap()));
        assert!(id.derivation.steps.iter().any(|s| s.rule.as_ref() == "tiered.joint.closure_admg"));
        assert!(id.derivation.steps.iter().any(|s| s.rule.as_ref() == "tiered.joint.closure"));
        assert!(id.required_assumptions.entries.iter().any(|a| matches!(
            &a.assumption,
            Assumption::Custom { id, .. } if id.as_ref() == NO_LATENT_TO_OUTCOME
        )));
        assert!(id.diagnostics.iter().any(|d| d.code.as_ref() == NO_LATENT_TO_OUTCOME));
        let direct = GeneralizedAdjustmentIdentifier::new()
            .identify_joint_admg_response(&admg, &joint_query(&schema, "t1", "t2", "y"))
            .unwrap();
        assert_eq!(direct.estimands[0].adjustment_set, id.estimands[0].adjustment_set);
    }

    #[test]
    fn treatment_sibling_and_cross_tier_pairs_also_identify() {
        let sibling_schema = CausalSchemaBuilder::new()
            .continuous("z")
            .finish()
            .continuous("t")
            .finish()
            .continuous("u")
            .finish()
            .continuous("y")
            .finish()
            .build()
            .unwrap();
        let sibling = TieredBackground::from_named(
            &sibling_schema,
            &[vec!["z"], vec!["t", "u"], vec!["y"]],
            WithinTier::CoDetermined,
        )
        .unwrap();
        let sibling_id = identify_tiered_joint(
            &sibling,
            &sibling_schema,
            &joint_query(&sibling_schema, "t", "u", "y"),
        )
        .unwrap();
        assert_eq!(sibling_id.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(
            sibling_id.estimands[0].adjustment_set.as_ref(),
            &[sibling_schema.id_of("z").unwrap()]
        );

        let cross_schema = CausalSchemaBuilder::new()
            .continuous("z")
            .finish()
            .continuous("t1")
            .finish()
            .continuous("t2")
            .finish()
            .continuous("y")
            .finish()
            .build()
            .unwrap();
        let cross = TieredBackground::from_named(
            &cross_schema,
            &[vec!["z"], vec!["t1"], vec!["t2"], vec!["y"]],
            WithinTier::CoDetermined,
        )
        .unwrap();
        let cross_id = identify_tiered_joint(
            &cross,
            &cross_schema,
            &joint_query(&cross_schema, "t1", "t2", "y"),
        )
        .unwrap();
        assert_eq!(cross_id.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(
            cross_id.estimands[0].adjustment_set.as_ref(),
            &[cross_schema.id_of("z").unwrap()]
        );
    }

    #[test]
    fn unknown_joint_has_no_single_admg() {
        let schema = CausalSchemaBuilder::new()
            .continuous("z")
            .finish()
            .continuous("t1")
            .finish()
            .continuous("t2")
            .finish()
            .continuous("y")
            .finish()
            .build()
            .unwrap();
        let background = TieredBackground::from_named(
            &schema,
            &[vec!["z"], vec!["t1", "t2"], vec!["y"]],
            WithinTier::Unknown,
        )
        .unwrap();
        let err =
            identify_tiered_joint(&background, &schema, &joint_query(&schema, "t1", "t2", "y"))
                .unwrap_err();
        assert!(err.to_string().contains("no single ADMG"));
    }

    /// 20 same-tier co-facets plus an earlier confounder: generic subset search
    /// caps at 16 candidates; the O(p) closure shortcut must still identify.
    const JOINT_COFACETS: usize = 20;

    fn many_cofacet_joint() -> (antecedent_core::CausalSchema, TieredBackground, Vec<String>) {
        let mut b = CausalSchemaBuilder::new();
        b = b.continuous("z").finish();
        b = b.continuous("t1").finish();
        b = b.continuous("t2").finish();
        let mut facets = Vec::new();
        for i in 0..JOINT_COFACETS {
            let name = format!("f{i}");
            b = b.continuous(name.clone()).finish();
            facets.push(name);
        }
        b = b.continuous("y").finish();
        let schema = b.build().unwrap();
        let mut treatment_tier = vec!["t1", "t2"];
        treatment_tier.extend(facets.iter().map(String::as_str));
        let background = TieredBackground::from_named(
            &schema,
            &[vec!["z"], treatment_tier, vec!["y"]],
            WithinTier::CoDetermined,
        )
        .unwrap();
        (schema, background, facets)
    }

    #[test]
    fn many_cofacets_use_closure_shortcut_not_subset_search() {
        let (schema, background, facets) = many_cofacet_joint();
        let expected = background
            .tier_closure_set(
                &[schema.id_of("t1").unwrap(), schema.id_of("t2").unwrap()],
                schema.id_of("y").unwrap(),
            )
            .unwrap();
        assert_eq!(
            expected.len(),
            1 + JOINT_COFACETS,
            "Z = earlier confounder + {JOINT_COFACETS} co-facets"
        );
        assert!(expected.iter().any(|&v| v == schema.id_of("z").unwrap()));
        for name in &facets {
            assert!(expected.iter().any(|&v| v == schema.id_of(name).unwrap()), "{name}");
        }

        let admg = background.to_admg(&schema).unwrap();
        let query = joint_query(&schema, "t1", "t2", "y");
        // Generic subset search lists z + every co-facet (> max_candidates=16) and caps.
        let generic = GeneralizedAdjustmentIdentifier::new()
            .identify_joint_admg_response(&admg, &query)
            .unwrap();
        assert_eq!(
            generic.status,
            IdentificationStatus::NotIdentified,
            "generic search must cap on this graph: {:?}",
            generic.diagnostics
        );
        assert!(
            generic.diagnostics.iter().any(|d| {
                d.kind == DiagnosticKind::Execution
                    && d.code.as_ref() == crate::generalized::CAPPED_COMPLETION_DIAGNOSTIC_CODE
            }),
            "cap is execution, not a third status: {:?}",
            generic.diagnostics
        );
        assert!(
            generic.diagnostics.iter().any(|d| {
                d.code.as_ref() == crate::generalized::CAPPED_COMPLETION_DIAGNOSTIC_CODE
            })
        );

        let started = std::time::Instant::now();
        let id = identify_tiered_joint(&background, &schema, &query).unwrap();
        assert!(
            started.elapsed().as_millis() < 200,
            "O(p) joint closure took {:?}",
            started.elapsed()
        );
        assert_eq!(id.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(id.estimands[0].adjustment_set.as_ref(), expected.as_ref());
        assert_eq!(id.performance.candidates_examined, 2);
        assert!(id.derivation.steps.iter().any(|s| s.rule.as_ref() == "tiered.joint.closure"));
        assert!(id.required_assumptions.entries.iter().any(|a| matches!(
            &a.assumption,
            Assumption::Custom { id, .. } if id.as_ref() == NO_LATENT_TO_OUTCOME
        )));
        assert!(id.diagnostics.iter().any(|d| d.code.as_ref() == NO_LATENT_TO_OUTCOME));
        assert!(!id.diagnostics.iter().any(|d| {
            d.code.as_ref() == crate::generalized::CAPPED_COMPLETION_DIAGNOSTIC_CODE
                || d.kind == DiagnosticKind::Scientific
                    && d.code.as_ref() == "identify.tiered.joint.adjustment"
        }));
    }

    #[test]
    fn drawn_treatment_outcome_on_closure_admg_is_scientific() {
        let schema = CausalSchemaBuilder::new()
            .continuous("z")
            .finish()
            .continuous("t1")
            .finish()
            .continuous("t2")
            .finish()
            .continuous("y")
            .finish()
            .build()
            .unwrap();
        let background = TieredBackground::from_named(
            &schema,
            &[vec!["z"], vec!["t1", "t2"], vec!["y"]],
            WithinTier::CoDetermined,
        )
        .unwrap();
        let mut admg = background.to_admg(&schema).unwrap();
        let t1 = DenseNodeId::from_raw(schema.id_of("t1").unwrap().raw());
        let y = DenseNodeId::from_raw(schema.id_of("y").unwrap().raw());
        admg.insert_bidirected(t1, y).unwrap();
        let set = background
            .tier_closure_set(
                &[schema.id_of("t1").unwrap(), schema.id_of("t2").unwrap()],
                schema.id_of("y").unwrap(),
            )
            .unwrap();
        let z: Vec<_> = set.iter().map(|v| DenseNodeId::from_raw(v.raw())).collect();
        let prepared =
            prepare_joint_response(admg.nodes(), &joint_query(&schema, "t1", "t2", "y")).unwrap();
        assert!(
            !joint_adjustment_holds(&admg, &prepared.targets, prepared.y, &z, |_, _| true).unwrap(),
            "drawn T↔Y must fail the closure check"
        );
        let id = GeneralizedAdjustmentIdentifier::new()
            .identify_joint_admg_response(&admg, &joint_query(&schema, "t1", "t2", "y"))
            .unwrap();
        assert_eq!(id.status, IdentificationStatus::NotIdentified);
        assert!(id.diagnostics.iter().any(|d| d.kind == DiagnosticKind::Scientific));
        assert!(
            !id.diagnostics.iter().any(|d| {
                d.code.as_ref() == crate::generalized::CAPPED_COMPLETION_DIAGNOSTIC_CODE
            })
        );
    }
}
