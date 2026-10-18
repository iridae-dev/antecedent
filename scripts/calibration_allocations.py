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


def validate(receipt: dict, groups: list[dict]) -> None:
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
            "validation_domain": decision.DOMAIN if hc3 else decision.LEGACY_DOMAIN,
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
    # Each slot reserves the original look weights (.5+.25+.25=1), not a new alpha.
    # Two distinct slots per grid are below the original 256-cell allowance.
    if len({row["slot"] for row in rows}) != len(rows) or len(rows) > 256:
        raise ValueError("cumulative hypothesis slot cap exceeded")
