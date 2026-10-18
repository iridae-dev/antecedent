"""Cumulative hypothesis slots for the independently validated weighted HC3 update.

The observed protocol2failure remains spent. A new random domain supplies independent
validation data for a distinct, prospectively reserved construction; it grants no
additional family risk. Further allocations require a reviewed append-only extension.
"""

from __future__ import annotations

import re

import calibration_decision as decision

PATH = "parity/calibration_decision_allocations.json"
ROSTER_DIGEST = "bd73004e74b845ef31bed0b4210560267dbaf794507c7a039c7d59a7c3ecf500"
ORIGINAL_DIGEST = "4be54bccb31e597327f46b6f16afff2e897984fca35e8bb63de5494797f62541"
ORIGINAL_SOURCE = "70a71dd33bc55cbc2398b0f91c4a2c797afad618"
GROUP = "sample_split_weighted_effect_calibration: weighted_split_effect_interval_calibrated"
FUNCTIONAL = "weighted_population_total_effect.discovery_fraction_0_75.conflict_resolution_treatment_invariant"
RECORD = "cov.sample_split_weighted_effect.cpdag.frequentist.student_t_sandwich_effective_n.l95.weighted_split_effect_interval_calibrated.discovery_fraction_0_75_conflict_resolution_treatment_invariant"


def validate(receipt: dict, groups: list[dict], domain: str | None = None) -> None:
    """Reject denominator drift, repeated slots, changed constructions or extra risk."""
    if receipt.get("schema") != "coverage_hypothesis_allocations.v1":
        raise ValueError("unknown cumulative hypothesis allocation schema")
    expected_family = {
        "source_sha": ORIGINAL_SOURCE,
        "family_digest": ORIGINAL_DIGEST,
        "roster_sha256": ROSTER_DIGEST,
        "family_groups": 757,
        "family_cells": 581376,
        "cell_cap_per_group": 256,
        "family_alpha": 0.05,
        "looks": [dict(x) for x in decision.LOOKS],
    }
    if (
        receipt.get("original_family") != expected_family
        or decision.digest(groups) != ROSTER_DIGEST
    ):
        raise ValueError("cumulative allocation changed the original roster or risk budget")
    group = [row for row in groups if row["label"] == GROUP]
    if len(group) != 1 or receipt.get("group") != group[0]:
        raise ValueError("hypothesis slots have a different group owner")
    rows = receipt.get("allocations")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError(
            "unknown or repeated hypothesis slots require an append-only reviewed extension"
        )
    for row, version in zip(rows, (2, 3), strict=True):
        hc3 = version == 3
        expected = {
            "slot": 2 if hc3 else 1,
            "protocol_version": version,
            "status": "reserved_before_outcomes" if hc3 else "spent",
            "functional": FUNCTIONAL + (".variance_hc3" if hc3 else ""),
            "record_id": RECORD + ("_variance_hc3" if hc3 else ""),
            "interval_method": "student_t_sandwich_effective_n",
            "se_kind": "hc3_sampling_weighted" if hc3 else "hc1_sampling_weighted",
            "nominal": 0.95,
            "validation_domain": decision.HC3_DOMAIN if hc3 else decision.LEGACY_DOMAIN,
            "grid_points": [0, 1, 2],
            "planned_replicates": 20000,
            "look_weight": 0.25,
        }
        if hc3:
            expected["method_source_commit"] = "ee9233f144eb495979f83ca0ae35070cf824a31e"
        else:
            expected["measured_source"] = ORIGINAL_SOURCE
            hashes = row.get("raw_log_sha256_by_point", {})
            if set(hashes) != {"0", "1", "2"} or any(
                not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in hashes.values()
            ):
                raise ValueError("spent protocol2 lacks all original raw-log receipts")
            expected["raw_log_sha256_by_point"] = hashes
        if row != expected:
            raise ValueError("hypothesis slot was reused or its identity/look/risk changed")
    extension = receipt.get("pag_trial_shift_extension")
    if extension is not None and extension != PAG_EXTENSION:
        raise ValueError("PAG cumulative slot identity, original spending or planned look changed")
    if domain == decision.PAG_DOMAIN and extension != PAG_EXTENSION:
        raise ValueError("PAG domain requires its prospective cumulative slot allocation")
    if extension is not None and extension["group"] not in groups:
        raise ValueError("PAG slot is not owned by the original complete roster")
    confidence_set = receipt.get("pag_confidence_set_extension")
    if confidence_set is not None and confidence_set != PAG_CONFIDENCE_SET_EXTENSION:
        raise ValueError("PAG confidence-set slot identity or cumulative spending changed")
    if domain == decision.PAG_CONFIDENCE_SET_DOMAIN and (
        extension != PAG_EXTENSION or confidence_set != PAG_CONFIDENCE_SET_EXTENSION
    ):
        raise ValueError("PAG confidence-set domain requires all cumulative slot allocations")
    # Each slot reserves the original look weights (.5+.25+.25=1), not a new alpha.
    # Two distinct slots per grid are below the original 256-cell allowance.
    if len({row["slot"] for row in rows}) != len(rows) or len(rows) > 256:
        raise ValueError("cumulative hypothesis slot cap exceeded")


PAG_EXTENSION = {
    "group": {
        "index": 684,
        "label": "discovered_transport_calibration: discovered_pag_transport_contrast_complete_procedure_coverage",
        "grid": True,
    },
    "original_source": "70a71dd33bc55cbc2398b0f91c4a2c797afad618",
    "old_slot": {
        "slot": 1,
        "record_id": "cov.discovered_transport.pag.frequentist.percentile_bootstrap.l95.discovered_pag_transport_contrast_complete_procedure_coverage.fci_pair_cutoff_0_05_over_28_sparse_1023_trial_compatibility_fisher_exact_alpha_0_01_v1",
        "validation_domain": "release_2_4_1_validation_v1",
        "status": "spent_including_failed_point",
        "looks_by_point": {"0": [400, 2000], "1": [2000], "2": [400, 2000]},
        "raw_log_sha256": {
            "0initial": "7d51ebbd6677a34fe58fb8bd0dc42d6167e1cc718b3e832531d8a59d4f1e16e5",
            "0recheck": "50ddfd48a1e86fb33adb7f78e799e1af5fcee05e50729b80af9d3678194384bd",
            "1initial": "7cb075c54befc1ce65b33b21dc41785db854342363690828847d452be063e22b",
            "2initial": "9ce8211abe5c5114f02a1a8264d71dc3edb8d3afef85f9f6e9d60dfa537b565f",
            "2recheck": "363e4d65e645f9595dbf5ffc5659543a75870ac0f302c0cf498f9cf9ef223e9e",
        },
    },
    "new_slot": {
        "slot": 2,
        "functional": "target_interventional_contrast.fci_pair_cutoff_0_05_over_28_sparse_1023.trial_compatibility_fisher_exact_alpha_0_01.declared_selection_joint_pretreatment_standardization",
        "record_id": "cov.discovered_transport.pag.frequentist.percentile_bootstrap.l95.discovered_pag_transport_contrast_complete_procedure_coverage.fci_pair_cutoff_0_05_over_28_sparse_1023_trial_compatibility_fisher_exact_alpha_0_01_declared_selection_joint_pretreatment_standardization",
        "validation_domain": "release_2_4_1_pag_trial_shift_validation_v1",
        "status": "reserved_before_outcomes",
        "planned_replicates": 2000,
        "look_weight": 0.25,
        "grid_points": [0, 1, 2],
        "nominal": 0.95,
        "interval_method": "percentile_bootstrap",
        "se_kind": "percentile",
    },
    "selection_history": "The new joint pretreatment shift standardization was chosen after observing the original PAG failure. It is fixed before new independent validation. Original slot1 spending is retained; slot2 uses the original family budget, not a fresh alpha.",
}


PAG_CONFIDENCE_SET_EXTENSION = {
    "group": {
        "index": 684,
        "label": "discovered_transport_calibration: discovered_pag_transport_contrast_complete_procedure_coverage",
        "grid": True,
    },
    "prior_slot": {
        "slot": 2,
        "source_sha": "9b9ba8ca7cdd4b0ab328be8b08c689cb5e6e63ed",
        "validation_domain": "release_2_4_1_pag_trial_shift_validation_v1",
        "status": "spent_including_failed_point",
        "planned_replicates": 2000,
        "look_weight": 0.25,
        "raw_log_sha256_by_point": {
            "0": "f6eb89a5a6a9c06dcdb4864932d9214d6e610dd6e54fb373fcb0af6bedc6cdbd",
            "1": "c0e80659f7748171bd70acaefa6606ad4d41e3760cff6d5cad2836440e6e8891",
            "2": "953293378bfde7dd17dc77a5ad31cc46dce8893af150974390d59be3cd3b811f",
        },
    },
    "new_slot": {
        "slot": 3,
        "functional": "target_interventional_contrast.fci_cutoff_0_05_sparse_1023.trial_compatibility_fisher_exact_alpha_0_01.declared_selection_joint_pretreatment_standardization.bounded_outcome_confidence_set",
        "record_id": "cov.discovered_transport.pag.frequentist.percentile_bootstrap_or_logical_support.l95.discovered_pag_transport_contrast_complete_procedure_coverage.fci_cutoff_0_05_sparse_1023_trial_compatibility_fisher_exact_alpha_0_01_declared_selection_joint_pretreatment_standardization_bounded_outcome_confidence_set",
        "validation_domain": "release_2_4_1_pag_confidence_set_validation_v1",
        "status": "reserved_before_outcomes",
        "planned_replicates": 2000,
        "look_weight": 0.25,
        "grid_points": [0, 1, 2],
        "nominal": 0.95,
        "interval_method": "percentile_bootstrap_or_logical_support",
        "se_kind": "percentile_or_logical_support",
    },
    "selection_history": "The bounded-outcome confidence-set procedure was chosen after original slot1 and slot2 validation failures and separate development probes. It is fixed before new independent slot3 validation; prior outcomes and risk spending remain preserved. No family alpha reset. Three distinct PAG construction slots remain within the original256pergroup/grid allocation.",
}
