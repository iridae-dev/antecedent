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

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from ._columns import _data_column_map
from ._data import _is_missing_scalar, table_column_names
from ._table_snapshot import Table
from .errors import CausalValueError

CONTROL_CODE = 0.0
ACTIVE_CODE = 1.0


def _is_missing(value: Any) -> bool:
    return _is_missing_scalar(value) or (
        isinstance(value, (float, np.floating)) and bool(np.isnan(value))
    )


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
    codes: tuple[tuple[str, float], ...] = ()
    other_codes: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()
    numeric_levels: tuple[float, float] | None = None

    @classmethod
    def of(cls, query: Any, *, data: Any = None) -> TreatmentLabels | None:
        """The labels an average-effect query declares, or ``None`` for numeric levels."""
        if getattr(query, "kind", None) != "average":
            return None
        control, active = query.control_level, query.active_level
        if not isinstance(control, str) and not isinstance(active, str):
            retained = tuple(getattr(data, "_treatment_codes", ()))
            if str(query.treatment) in dict(retained):
                raise CausalValueError(
                    f"treatment {query.treatment!r} retains label declarations; numeric levels require new numeric data",
                    reason_code="invalid_argument",
                )
            return (
                cls(
                    str(query.treatment),
                    "",
                    "",
                    other_codes=retained,
                    numeric_levels=(control, active),
                )
                if retained
                else None
            )
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
        codes = dict(getattr(data, "_treatment_codes", ())).get(str(query.treatment), ())
        if codes:
            mapping = dict(codes)
            for label in sorted({control, active} - mapping.keys()):
                mapping[label] = float(len(mapping))
            codes = tuple(mapping.items())
        others = tuple(
            (column, mapping)
            for column, mapping in getattr(data, "_treatment_codes", ())
            if column != str(query.treatment)
        )
        return cls(
            column=str(query.treatment),
            control=control,
            active=active,
            codes=codes,
            other_codes=others,
        )

    def native_query(self, query: Any) -> Any:
        """The query with the labels replaced by the arm codes the native path takes."""
        if self.numeric_levels is not None:
            return query
        codes = (
            dict(self.codes)
            if self.codes
            else {self.control: CONTROL_CODE, self.active: ACTIVE_CODE}
        )
        return replace(query, control_level=codes[self.control], active_level=codes[self.active])

    def encode(self, data: Any) -> Any:
        """``data`` with the treatment column replaced by its arm codes, in column order.

        Labels are compared as text, so a column of strings, categories or integers declared
        by their spelling all work. Missing cells stay missing. Labels that are neither
        declared get codes from ``2`` in sorted order, which the native level check refuses
        unless the caller declared an exclusion.
        """
        if isinstance(data, TreatmentLabelTable):
            retained = dict(data._treatment_codes).get(self.column)
            own_matches = self.numeric_levels is not None or (
                retained is not None
                and all(dict(self.codes).get(label) == code for label, code in retained)
            )
            other_matches = all(
                dict(data._treatment_codes).get(column) == codes
                for column, codes in self.other_codes
            )
            if own_matches and other_matches:
                return data._encoded_data
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
        required = {self.column, *(column for column, _codes in self.other_codes)}
        missing = sorted(required - set(names))
        if missing:
            raise CausalValueError(
                f"required analysis columns are missing: {missing}; available: {names}",
                reason_code="invalid_argument",
            )
        # Owned label snapshots retain object columns as immutable tuples.
        # np.asarray(tuple_of_strings_and_nan) would turn the missing sentinel
        # into the literal text "nan" before our explicit object conversion.
        columns = (
            {name: data[name] for name in names}
            if isinstance(data, Mapping)
            else _data_column_map(data, set(names))
        )
        own_codes = self.codes or ((self.control, CONTROL_CODE), (self.active, ACTIVE_CODE))
        encodings = (
            self.other_codes
            if self.numeric_levels is not None
            else ((self.column, own_codes), *self.other_codes)
        )
        result = dict(columns)
        for column, codes in encodings:
            mapping = dict(codes)
            raw = np.asarray(columns[column], dtype=object)
            if raw.ndim != 1:
                raise CausalValueError(f"treatment column {column!r} must be one-dimensional")
            texts = [None if _is_missing(value) else str(value) for value in raw]
            for label in sorted({text for text in texts if text is not None} - mapping.keys()):
                mapping[label] = float(len(mapping))
            result[column] = np.asarray(
                [np.nan if text is None else mapping[text] for text in texts], dtype=np.float64
            )
        return result


@dataclass(frozen=True, slots=True, repr=False)
class TreatmentLabelTable(Table):
    """Source labels alongside an explicitly retained immutable numeric encoding.

    Mapping access serves the source declaration. Only TreatmentLabels.encode
    selects the numeric table, so native numeric callers never receive labels.
    """

    _encoded_data: Any = None
    _treatment_codes: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()

    def select(self, names: Any) -> TreatmentLabelTable:
        encoded = self._encoded_data.select(names)
        return label_table(self, encoded, self._treatment_codes)

    def _native_columns(self) -> tuple[list[str], list[Any]]:
        return cast(Table, self._encoded_data)._native_columns()


def label_table(source: Table, encoded: Table, codes: Any) -> TreatmentLabelTable:
    """Retain source label values with the exact executed numeric row scope."""
    design = {name: source._design[name] for name in encoded.column_names if name in source._design}
    for name, _mapping in codes:
        if name in encoded.column_names:
            design[name] = source[name]
    return TreatmentLabelTable(
        encoded._handle,
        encoded.column_names,
        MappingProxyType(design),
        source._design_bytes,
        encoded.num_rows,
        encoded,
        tuple((name, tuple(mapping)) for name, mapping in codes if name in encoded.column_names),
    )


def batch_labels(data: Table, queries: Any) -> tuple[Any, tuple[Any, ...], Any]:
    """Resolve one shared coding per physical treatment column, retaining each contrast."""
    retained_codes = dict(getattr(data, "_treatment_codes", ()))
    groups: dict[str, list[TreatmentLabels | None]] = {}
    for query in queries:
        groups.setdefault(query.treatment, []).append(TreatmentLabels.of(query))
    encodings: list[tuple[str, tuple[tuple[str, float], ...]]] = []
    for column, declarations in groups.items():
        labelled = [item for item in declarations if item is not None]
        if not labelled:
            if column in retained_codes:
                raise CausalValueError(
                    f"treatment {column!r} retains label declarations; numeric levels require new numeric data",
                    reason_code="invalid_argument",
                )
            continue
        if len(labelled) != len(declarations):
            raise CausalValueError(
                f"treatment {column!r} mixes numeric levels and label declarations",
                reason_code="invalid_argument",
            )
        first = labelled[0]
        mapping = dict(dict(getattr(data, "_treatment_codes", ())).get(column, ()))
        if not mapping:
            mapping = {first.control: CONTROL_CODE, first.active: ACTIVE_CODE}
        labels = {label for item in labelled for label in (item.control, item.active)}
        labels.update(str(value) for value in data[column] if not _is_missing(value))
        for label in sorted(labels - mapping.keys()):
            mapping[label] = float(len(mapping))
        encodings.append((column, tuple(mapping.items())))
    encoded_names = {column for column, _codes in encodings}
    encodings.extend(
        (column, codes) for column, codes in retained_codes.items() if column not in encoded_names
    )
    if not encodings:
        return data, tuple(queries), None
    from ._table_snapshot import table

    if isinstance(data, TreatmentLabelTable) and all(
        all(dict(codes).get(label) == code for label, code in retained_codes.get(column, ()))
        and column in retained_codes
        for column, codes in encodings
    ):
        encoded = data._encoded_data
    else:
        columns: Any = {name: data[name] for name in data.column_names}
        for column, codes in encodings:
            first = next(
                (item for item in groups.get(column, ()) if item is not None),
                TreatmentLabels(column, codes[0][0], codes[1][0]),
            )
            columns = replace(first, codes=codes).encode(columns)
        encoded = table(columns)
    source = label_table(data, encoded, encodings)
    native_queries = tuple(
        binding.native_query(query)
        if (binding := TreatmentLabels.of(query, data=source))
        else query
        for query in queries
    )
    return encoded, native_queries, source


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
