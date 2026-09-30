"""The label a graphless estimate publishes when the engine leaves its license unset.

The live estimate (``estimation.py``) and the portable-claim projection of its
wire body (``results/_families.py``) both derive this label; they call these
functions so the two sides cannot drift.
"""

from __future__ import annotations


def unset_panel_license(*, has_interval: bool) -> str:
    """Plain panel DiD: an interval is off-axis evidence, otherwise the point is unlicensed."""
    return "off_axis_interval_evidence" if has_interval else "unlicensed_point_utility"


def unset_dose_license(*, has_incremental_interval: bool) -> str:
    """Dose policy: a fixed-policy incremental interval is off-axis pointwise evidence."""
    return "off_axis_pointwise_95" if has_incremental_interval else "unlicensed_point_utility"


def unset_policy_license(*, has_regret: bool, has_interval: bool) -> str:
    """Policy value: regret is simultaneous evidence, any interval is pointwise evidence."""
    if has_regret:
        return "off_axis_simultaneous_95"
    if has_interval:
        return "off_axis_pointwise_95"
    return "unlicensed_point_utility"
