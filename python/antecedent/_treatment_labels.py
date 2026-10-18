"""String- or category-coded binary treatments with declared active and control labels.

``AverageEffect(..., control_level="a", active_level="b")`` names the two arms of a treatment
column by label. The labels are mapped to the arm codes ``0`` (control) and ``1`` (active) and
the declared-level contrast runs through the same native path as numeric levels: there is no
second execution route. A row carrying any other label gets a further code, so the native
level check refuses it (``treatment_level_undeclared``) exactly as it refuses an undeclared
numeric level; nothing is dropped or merged silently. The mapping is frozen with the study, so
a refreshed table is encoded by the same declared labels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ._columns import _data_column_map
from ._data import table_column_names
from .errors import CausalValueError

CONTROL_CODE = 0.0
ACTIVE_CODE = 1.0


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, (float, np.floating)) and bool(np.isnan(value)))


def level_text(treatment: str, value: Any) -> str:
    """One declared level as the estimand names it: labels quoted, numbers compact."""
    if isinstance(value, str):
        return f"{treatment}={value!r}"
    return f"{treatment}={float(value):g}"


@dataclass(frozen=True, slots=True)
class TreatmentLabels:
    """The declared labels of a binary treatment column."""

    column: str
    control: str
    active: str

    @classmethod
    def of(cls, query: Any) -> TreatmentLabels | None:
        """The labels an average-effect query declares, or ``None`` for numeric levels."""
        if getattr(query, "kind", None) != "average":
            return None
        control, active = query.control_level, query.active_level
        if not isinstance(control, str) and not isinstance(active, str):
            return None
        if not (isinstance(control, str) and isinstance(active, str)):
            raise CausalValueError(
                "control_level and active_level must both be labels (str) or both be numbers; "
                f"got {control!r} and {active!r}",
                reason_code="invalid_argument",
            )
        if control == active:
            raise CausalValueError(
                f"control_level and active_level must differ; both are {control!r}",
                reason_code="invalid_argument",
            )
        return cls(column=str(query.treatment), control=control, active=active)

    def native_query(self, query: Any) -> Any:
        """The query with the labels replaced by the arm codes the native path takes."""
        return replace(query, control_level=CONTROL_CODE, active_level=ACTIVE_CODE)

    def encode(self, data: Any) -> dict[str, Any]:
        """``data`` with the treatment column replaced by its arm codes, in column order.

        Labels are compared as text, so a column of strings, categories or integers declared
        by their spelling all work. Missing cells stay missing. Labels that are neither
        declared get codes from ``2`` in sorted order, which the native level check refuses
        unless the caller declared an exclusion.
        """
        names = table_column_names(data)
        if names is None:
            raise CausalValueError(
                "label-declared treatment levels need a table (dict, DataFrame or Arrow table) "
                "holding the treatment column",
                reason_code="invalid_argument",
            )
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise CausalValueError(
                f"data has duplicate column names: {repeated}", reason_code="invalid_argument"
            )
        if self.column not in names:
            raise CausalValueError(
                f"required analysis columns are missing: {[self.column]}; available: {names}",
                reason_code="invalid_argument",
            )
        columns = _data_column_map(data, set(names))
        raw = np.asarray(columns[self.column], dtype=object)
        if raw.ndim != 1:
            raise CausalValueError(f"treatment column {self.column!r} must be one-dimensional")
        texts = [None if _is_missing(v) else str(v) for v in raw]
        others = sorted({t for t in texts if t not in (None, self.control, self.active)})
        code_of = {self.control: CONTROL_CODE, self.active: ACTIVE_CODE}
        code_of.update({label: 2.0 + index for index, label in enumerate(others)})
        coded = np.array([np.nan if t is None else code_of[t] for t in texts], dtype=np.float64)
        return {name: coded if name == self.column else columns[name] for name in names}


def estimand_text(query: Any) -> str | None:
    """The average effect as the query declares it, naming the declared levels."""
    if getattr(query, "kind", None) != "average":
        return None
    t, y = query.treatment, query.outcome
    return (
        f"E[{y} | do({level_text(t, query.active_level)})]"
        f" - E[{y} | do({level_text(t, query.control_level)})]"
    )


__all__ = [
    "ACTIVE_CODE",
    "CONTROL_CODE",
    "TreatmentLabels",
    "estimand_text",
    "level_text",
]
