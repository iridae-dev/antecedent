//! Standing laws of the decision boundary and the evidence obligations, over generated
//! inputs, against the production functions the standing-site inventory
//! (`formal/standing_sites.toml`) names.
//!
//! The expected outcomes come from the rule proved in `formal/Antecedent/Standing.lean`:
//! standing is never raised without evidence. A support floor never strengthens a
//! coordinate; a coordinate the artifact's mask does not state is supported only where a
//! native execution record backs the law; a calibration label a provider only attests is
//! not measured; and an input short of support always owes an obligation.

use std::sync::Arc;

use antecedent_contract::{
    EvidenceObligationKind, ExternalTrustState, QuantityRole, ScientificQuantity,
};
use antecedent_core::{SupportRegion, SupportReport, SupportStatus, VariableId};
use antecedent_design::composition_boundary::{
    CalibrationStatus, DecisionInput, NativeExecutionRecord, ProviderKind, TrustEvidence,
    TrustRequirement,
};
use antecedent_design::decision_eval::{MeanEstimate, MeanSource};
use antecedent_design::obligation_adapters::support_obligations;
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io_core::quantity_wire::DistributionMeaningWire;
use proptest::prelude::*;

fn config() -> ProptestConfig {
    ProptestConfig { cases: 256, failure_persistence: None, ..ProptestConfig::default() }
}

const STATUSES: [SupportStatus; 5] = [
    SupportStatus::Supported,
    SupportStatus::WeakOverlap,
    SupportStatus::Extrapolative,
    SupportStatus::OutsideEmpiricalSupport,
    SupportStatus::MissingEvidence,
];

const CALIBRATIONS: [DistributionCalibration; 4] = [
    DistributionCalibration::Exact,
    DistributionCalibration::PointOnly,
    DistributionCalibration::Measured,
    DistributionCalibration::NotMeasured,
];

const TRUSTS: [DistributionTrust; 4] = [
    DistributionTrust::NativeSupported,
    DistributionTrust::ExternalAttested,
    DistributionTrust::VerifiedExtension,
    DistributionTrust::Unverified,
];

fn quantity(regime: usize, functional: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "y".into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: format!("do(a={regime})").into(),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn worse(a: SupportStatus, b: SupportStatus) -> SupportStatus {
    if a.severity() >= b.severity() { a } else { b }
}

/// A joint law over `columns` coordinates, two draws, with the given labels and mask.
fn joint(
    columns: usize,
    trust: DistributionTrust,
    calibration: DistributionCalibration,
    mask: Option<Vec<bool>>,
) -> DistributionArtifact {
    let quantities: Vec<ScientificQuantity> =
        (0..columns).map(|c| quantity(c, "outcome")).collect();
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &quantities,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "study-x".into(),
            provider_id: "lab".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "snap-x".into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .expect("a valid identity");
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [2, columns],
            weights: None,
            supported: mask,
            calibration,
            trust,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        (0..u32::try_from(2 * columns).expect("few columns")).map(f64::from).collect(),
    )
    .expect("a valid artifact")
}

fn evidence(which: usize) -> TrustEvidence {
    match which {
        0 => TrustEvidence::None,
        1 => TrustEvidence::External(ExternalTrustState::ExternallyAttested {
            attestor: "attestor".into(),
        }),
        2 => TrustEvidence::NativeExecution(NativeExecutionRecord {
            execution_id: "exec-1".into(),
            provider_id: "lab".into(),
            snapshot_id: "snap-x".into(),
        }),
        // A record for another snapshot proves nothing.
        _ => TrustEvidence::NativeExecution(NativeExecutionRecord {
            execution_id: "exec-1".into(),
            provider_id: "lab".into(),
            snapshot_id: "another-snapshot".into(),
        }),
    }
}

fn mask() -> impl Strategy<Value = (usize, Option<Vec<bool>>)> {
    (1usize..5).prop_flat_map(|columns| {
        (Just(columns), prop::option::of(prop::collection::vec(any::<bool>(), columns)))
    })
}

fn report() -> impl Strategy<Value = SupportReport> {
    (0usize..5, prop::option::of(prop::collection::vec(0usize..5, 1..6))).prop_map(
        |(summary, points)| {
            let point_status: Option<Arc<[SupportStatus]>> =
                points.map(|p| p.into_iter().map(|i| STATUSES[i]).collect());
            // A report with labels summarizes them as their worst; a summary-only report
            // carries its own summary.
            let status = point_status.as_deref().map_or(STATUSES[summary], |labels| {
                labels.iter().copied().fold(SupportStatus::Supported, worse)
            });
            SupportReport {
                status,
                query_region: SupportRegion { minima: Arc::from([0.0]), maxima: Arc::from([1.0]) },
                diagnostics: Vec::new(),
                warnings: Vec::new(),
                point_status,
            }
        },
    )
}

proptest! {
    #![proptest_config(config())]

    /// A support floor weakens every coordinate to at least the floor and never
    /// strengthens one: each coordinate ends at the worse of its own label and the floor.
    #[test]
    fn support_floor_never_strengthens(
        labels in prop::collection::vec(0usize..5, 1..6),
        floor in 0usize..5,
    ) {
        let statuses: Vec<SupportStatus> = labels.iter().map(|&i| STATUSES[i]).collect();
        let input = DecisionInput::from_mean_source(
            "grid",
            MeanSource {
                coordinates: (0..statuses.len()).map(|c| quantity(c, "mean")).collect(),
                means: vec![1.0; statuses.len()],
                provider_id: "lab".into(),
                snapshot_id: "snap-x".into(),
                causal_contract_id: "checked-contract".into(),
                rng_id: "none:mean_grid".into(),
                estimate: MeanEstimate::Frequentist,
            },
            &statuses,
            &TrustEvidence::None,
            TrustRequirement::Unrestricted,
        )
        .expect("a valid mean source");
        let floored = input.with_support_floor(STATUSES[floor]);
        for ((_, after), before) in floored.support().entries().iter().zip(&statuses) {
            prop_assert_eq!(*after, worse(*before, STATUSES[floor]));
        }
    }

    /// A coordinate of a joint law is supported only where the artifact's mask states it,
    /// or the mask is silent and a matching native execution record backs the law; a
    /// calibration label is measured or exact only where a native record or a verified
    /// request backs it, never where the provider only attests.
    #[test]
    fn missing_mask_is_never_supported(
        (columns, mask) in mask(),
        trust in 0usize..4,
        calibration in 0usize..4,
        which in 0usize..4,
    ) {
        let artifact = joint(columns, TRUSTS[trust], CALIBRATIONS[calibration], mask.clone());
        let input = DecisionInput::from_distribution_artifact(
            "law",
            &artifact,
            &evidence(which),
            TrustRequirement::Unrestricted,
        )
        .expect("an unrestricted input");
        let native = input.provenance().provider_kind == ProviderKind::Native;
        prop_assert_eq!(native, which == 2);
        for (index, (_, status)) in input.support().entries().iter().enumerate() {
            let stated = mask.as_ref().map(|m| m[index]);
            if *status == SupportStatus::Supported {
                prop_assert!(stated == Some(true) || (stated.is_none() && native));
            }
            if stated.is_none() && !native {
                prop_assert_eq!(*status, SupportStatus::MissingEvidence);
            }
        }
        if input.provenance().provider_kind == ProviderKind::ExternalAttested {
            prop_assert!(!matches!(
                input.provenance().calibration,
                CalibrationStatus::Measured | CalibrationStatus::Exact
            ));
        }
    }

    /// A support report short of support everywhere owes at least one obligation: an
    /// establish-support obligation for any label beyond weak overlap, a sample obligation
    /// for weak overlap, and nothing only when every label (or the summary of a report with
    /// no labels) is supported.
    #[test]
    fn unsupported_input_always_owes_an_obligation(report in report()) {
        let owed = support_obligations(&report, "target", &[VariableId::from_raw(0)], "support")
            .expect("one treatment to name");
        let labels: Vec<SupportStatus> = report
            .point_status
            .as_deref()
            .map_or_else(|| vec![report.status], <[SupportStatus]>::to_vec);
        let establish = labels.iter().any(|s| s.severity() > SupportStatus::WeakOverlap.severity());
        let weak = labels.contains(&SupportStatus::WeakOverlap);
        prop_assert_eq!(
            owed.iter().any(|o| o.kind == EvidenceObligationKind::EstablishSupport),
            establish
        );
        prop_assert_eq!(
            owed.iter().any(|o| o.kind == EvidenceObligationKind::IncreaseSample),
            weak
        );
        prop_assert_eq!(owed.is_empty(), labels.iter().all(|s| *s == SupportStatus::Supported));
    }
}
