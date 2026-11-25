//! Claim-kind and "point identified" laws over generated identification sections, against
//! the production rule the producer and the independent consumer share
//! (`claim_kind_of_section`, `slot_claim_kind`, `status_identifies_a_point`).
//!
//! A claim is a point only with an identification slot whose status identifies a value, all
//! of whose mass is identified and that carries no structural weights; a section that states
//! no slot is never a point; and every claim that carries a slot classifies exactly as it did
//! before the rule read the absent slot (the released rule is the oracle for present slots).

use antecedent_contract::{ClaimKind, DomainStatus, IDENTIFICATION_NOT_CLAIMED};
use antecedent_core::{AverageEffectQuery, CausalQuery, IdentificationStatus, VariableId};
use antecedent_io_core::contract_section::{
    slot_claim_kind, status_identifies_a_point, status_name_identifies_a_point,
};
use antecedent_io_core::expr_wire::ExprArenaWire;
use antecedent_io_core::query_wire::causal_query_to_wire;
use antecedent_io_core::{
    AnalysisResultWire, IdentificationResultWire, IdentificationSlotWire, SlotSectionWire,
    claim_domain_statuses, claim_kind_of_section,
};
use proptest::prelude::*;

fn config() -> ProptestConfig {
    ProptestConfig { cases: 256, failure_persistence: None, ..ProptestConfig::default() }
}

/// A scalar average-effect body: no section that fixes the kind by itself.
fn scalar_body() -> AnalysisResultWire {
    let query = CausalQuery::AverageEffect(AverageEffectQuery::with_levels(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
        0.0,
        1.0,
    ));
    let query_wire = causal_query_to_wire(&query).expect("a valid query");
    AnalysisResultWire {
        native_execution: None,
        query: query_wire.clone(),
        identification: IdentificationResultWire {
            status: "nonparametrically_identified".into(),
            query: query_wire,
            estimands: Vec::new(),
            arena: ExprArenaWire {
                derivations: Vec::new(),
                var_sets: Vec::new(),
                interventions: Vec::new(),
                lists: Vec::new(),
                nodes: Vec::new(),
            },
            derivation: Vec::new(),
            required_assumptions: Vec::new(),
            diagnostics: Vec::new(),
            candidates_examined: 0,
            sets_returned: 0,
            hedge: None,
        },
        identification_variables: None,
        temporal_identification: Vec::new(),
        estimate: Some(1.5),
        policy_value: None,
        continuous_dose_response: None,
        panel_did: None,
        synthetic_control: None,
        synthetic_did: None,
        local_polynomial_ratio: None,
        randomized_effect: None,
        survival: None,
        posterior_row: None,
        longitudinal_regime: None,
        sequential_contrast: None,
        history_match: None,
        iv_bounds: None,
        association: None,
        quasi_rollout: None,
        temporal_design: None,
        interference_inference: None,
        interventional_distribution: None,
        standard_error: None,
        joint_covariance: None,
        simultaneous_interval: None,
        adjusted_p_values: None,
        family_contrast: None,
        family_contrast_interval: None,
        interval_lower: None,
        interval_upper: None,
        assumptions: Vec::new(),
        diagnostics: Vec::new(),
        refutations: Vec::new(),
        response: None,
        posterior_artifact: None,
        mediation_grid: None,
        structural_response: None,
        unit_effects: None,
        counterfactual_contrasts: None,
        cate: None,
        fitted_effect: None,
        cate_se: None,
        cate_leaf_dispersion: None,
        outcome_oof_r2: None,
        treatment_oof_logloss: None,
        crossfit_folds: None,
        crossfit_seed: None,
        learner_provenance: Vec::new(),
    }
}

const STATUSES: [&str; 9] = [
    "nonparametrically_identified",
    "identified_under_parametric_restrictions",
    "identified_under_prior_restrictions",
    "graph_dependent",
    "partially_identified",
    "not_identified",
    "not_certified",
    "proven_non_transportable",
    "an_unknown_spelling",
];
const MASSES: [f64; 4] = [0.0, 0.25, 0.75, 1.0];
const REASONS: [&str; 4] =
    ["not_prepared", "identified_per_execution", "not_identified", "an_unknown_reason"];

fn slot() -> impl Strategy<Value = IdentificationSlotWire> {
    (0usize..STATUSES.len(), 0usize..MASSES.len(), any::<bool>()).prop_map(
        |(status, mass, weighted)| IdentificationSlotWire {
            status: STATUSES[status].to_owned(),
            identified_mass: MASSES[mass],
            unidentified_mass: 1.0 - MASSES[mass],
            unevaluable_mass: 0.0,
            incomplete_search_mass: 0.0,
            full_mass_scope: true,
            search_capped: false,
            weight_basis: weighted.then(|| "posterior_probability".to_owned()),
        },
    )
}

/// The released kind of a body carrying a slot: the oracle the present-slot claims keep.
fn released_kind(slot: &IdentificationSlotWire) -> ClaimKind {
    match slot.status.as_str() {
        "partially_identified" => ClaimKind::Bounds,
        "nonparametrically_identified"
        | "identified_under_parametric_restrictions"
        | "identified_under_prior_restrictions"
        | "graph_dependent" => {
            if slot.identified_mass < 1.0 || slot.weight_basis.is_some() {
                ClaimKind::Mixture
            } else {
                ClaimKind::Point
            }
        }
        _ => ClaimKind::Incomplete,
    }
}

proptest! {
    #![proptest_config(config())]

    /// A section that states no identification slot is never a point claim (it is
    /// incomplete), one that says identification is not claimed is associational, and a
    /// section with a slot classifies exactly as the released rule classified it.
    #[test]
    fn claim_kind_is_never_a_point_without_a_slot(
        slot in slot(),
        reason in 0usize..REASONS.len(),
        shape in 0u8..3,
    ) {
        let body = scalar_body();
        let section = match shape {
            0 => SlotSectionWire { value: Some(slot.clone()), unavailable: None },
            1 => SlotSectionWire { value: None, unavailable: Some(REASONS[reason].to_owned()) },
            _ => SlotSectionWire {
                value: None,
                unavailable: Some(IDENTIFICATION_NOT_CLAIMED.to_owned()),
            },
        };
        let kind = claim_kind_of_section(&body, &section);
        match shape {
            0 => prop_assert_eq!(kind, released_kind(&slot)),
            1 => prop_assert_eq!(kind, ClaimKind::Incomplete),
            _ => prop_assert_eq!(kind, ClaimKind::Associational),
        }
        if kind == ClaimKind::Point {
            let present = section.value.as_ref().expect("a point has a slot");
            prop_assert!(present.identified_mass >= 1.0 && present.weight_basis.is_none());
        }
    }

    /// The slot rule is a point only for a full, unweighted slot whose status identifies a
    /// value, and the one "point identified" rule establishes the identification domain
    /// exactly when it holds, whatever the spelling.
    #[test]
    fn one_point_identified_rule_decides_kind_and_domain(slot in slot()) {
        let status = IdentificationStatus::from_name(&slot.status);
        let kind = slot_claim_kind(status, slot.identified_mass, slot.weight_basis.is_some());
        prop_assert_eq!(kind, released_kind(&slot));
        if kind == ClaimKind::Point {
            prop_assert!(status.is_some());
            prop_assert!(!matches!(
                status,
                Some(IdentificationStatus::PartiallyIdentified | IdentificationStatus::NotIdentified)
            ));
        }
        let identifies = status_name_identifies_a_point(&slot.status);
        prop_assert_eq!(identifies, status.is_some_and(status_identifies_a_point));
        // The rule itself, stated apart from its implementation: only a nonparametric or a
        // parametric identification identifies a point; a prior restriction, a partial or a
        // graph-dependent status, an unidentified one and an unknown spelling do not.
        prop_assert_eq!(
            identifies,
            matches!(
                status,
                Some(
                    IdentificationStatus::NonparametricallyIdentified
                        | IdentificationStatus::IdentifiedUnderParametricRestrictions
                )
            )
        );
        let domains = claim_domain_statuses(None, Some(&slot));
        prop_assert_eq!(
            domains.identification == DomainStatus::Identified,
            identifies && slot.unidentified_mass == 0.0
        );
    }
}
