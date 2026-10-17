"""Resolve column-name inputs on design-carrying queries against the data.

A design-carrying query (randomized experiment, policy value, ...) may name a
row-aligned input by the data column that holds it instead of passing the array
inline: ``PolicyValue("y", assignment="arm", ...)`` rather than
``PolicyValue("y", assignment=[...], ...)``. This keeps the query readable. The
design is read once, at prepare, and frozen into the prepared study: a later
``refresh(new_data)`` checks any supplied named design columns against the frozen
bindings before numeric ingest. Changed values or order require explicit repreparation.

A class opts in by declaring ``_COLUMN_FIELDS`` (row-aligned field name → element
kind) and, in its ``__post_init__``, calling :func:`defer_columns` first: while any
declared field is a ``str`` the class stores its inputs unvalidated and sets
``_deferred_columns``; :func:`resolve_columns` (run at prepare, when the data is in
hand) resolves each name to its column and rebuilds the object, which re-runs the
ordinary array validation. A class that wraps another design-carrying object lists
it in ``_COLUMN_NESTED`` so resolution recurses. When no field is a ``str`` the
class validates exactly as before, so the inline-array path is untouched.
"""

from __future__ import annotations

import hashlib
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any, TypeVar, cast, overload

import numpy as np

from ._data import read_columns, table_column_names, table_row_count
from ._validate import count as _count
from .errors import CausalUnsupportedError, CausalValueError

_T = TypeVar("_T")


def coerce_fields(obj: Any, /, **fields: Any) -> None:
    """Assign fields on a frozen dataclass.

    Frozen ``__post_init__`` methods cannot write ``self.x = ...``. This is the
    one helper for that write. The values are whatever the caller already
    validated; this function does not coerce them further.
    """
    for name, value in fields.items():
        object.__setattr__(obj, name, value)


def hold_unresolved(obj: Any, /, **fields: Any) -> bool:
    """Skip validation while a column name (or a nested design) is still unresolved.

    Returns True when ``__post_init__`` should return. Sets ``_deferred_columns``
    so :func:`resolve_columns` rebuilds the object against the table at prepare.
    """
    if not defer_columns(obj):
        return False
    coerce_fields(obj, _deferred_columns=True, **fields)
    return True


def _require_resolved(value: object) -> None:
    """Refuse a column name that reached prepare still unresolved.

    ``assert`` is stripped by ``python -O``, which would then hand the name to
    the estimator as if it were a one-row column.
    """
    if isinstance(value, str):
        raise CausalValueError(
            "column-name input must be resolved before prepare",
            reason_code="invalid_argument",
        )


def resolved(value: Sequence[_T] | str) -> Sequence[_T]:
    """Narrow a column-name field to its array once :func:`resolve_columns` has run.

    A design query accepts a row-aligned input either inline or as the name of a
    data column (a ``str``). :func:`resolve_columns` rebuilds the query with the
    named column's array before the native study is prepared, so by the time a
    prepared study reads the field the ``str`` branch is gone. This checks that
    invariant and hands the caller (and the type checker) the sequence.
    """
    _require_resolved(value)
    return cast(Sequence[_T], value)


@overload
def resolved_opt(value: None) -> None: ...
@overload
def resolved_opt(value: Sequence[_T] | str) -> Sequence[_T]: ...
def resolved_opt(value: Sequence[_T] | str | None) -> Sequence[_T] | None:
    """Like :func:`resolved`, but pass ``None`` through for optional fields."""
    if value is None:
        return None
    _require_resolved(value)
    return cast(Sequence[_T], value)


def resolved_scalar(value: float | Sequence[float] | str) -> float | Sequence[float]:
    """Narrow a scalar-or-column field (e.g. a per-row propensity) once resolved.

    The field is either a scalar, an inline sequence, or the name of a data column
    that :func:`resolve_columns` has already turned into a sequence, so the ``str``
    branch is gone by prepare.
    """
    _require_resolved(value)
    return cast(float | Sequence[float], value)


def _shown(value: Any) -> Any:
    """The value as ``tolist`` would have shown it in a refusal."""
    return value.item() if isinstance(value, np.generic) else value


def _first_bad(mask: np.ndarray, values: np.ndarray) -> Any:
    return _shown(values[np.flatnonzero(mask)[0]])


def _cast_bool(arr: np.ndarray, field: str, name: str) -> tuple[bool, ...]:
    """Booleans, or numbers that are exactly 0 or 1; anything else is refused.

    ``bool(v)`` would turn NaN, arm codes ``2``, and the string ``"False"`` into
    ``True``; the inline path refuses those, so the column path does too.
    """
    if arr.dtype == np.bool_ or arr.dtype == bool:
        return tuple(arr.tolist())
    if arr.dtype.kind in "iuf":
        ok = np.isfinite(arr) & ((arr == 0) | (arr == 1))
        if not np.all(ok):
            bad = _first_bad(~ok, arr)
            raise CausalValueError(
                f"{field}={name!r}: column values must be bool or encoded as 0/1; got {bad!r}"
            )
        return tuple(arr.astype(bool, copy=False).tolist())
    if arr.dtype == object:
        try:
            ok = np.isin(arr, (0, 1, False, True))
        except (TypeError, ValueError):
            ok = None
        if ok is not None:
            if not np.all(ok):
                bad = arr[np.flatnonzero(~ok)[0]]
                raise CausalValueError(
                    f"{field}={name!r}: column values must be bool or encoded as 0/1; "
                    f"got {_shown(bad)!r}"
                )
            return tuple(np.asarray(arr, dtype=bool).tolist())
    out: list[bool] = []
    for value in arr:
        if isinstance(value, (bool, np.bool_)) or (
            isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1)
        ):
            out.append(bool(value))
        else:
            raise CausalValueError(
                f"{field}={name!r}: column values must be bool or encoded as 0/1; got {value!r}"
            )
    return tuple(out)


def _cast_int(arr: np.ndarray, field: str, name: str) -> tuple[int, ...]:
    """Normalize each design integer exactly, with its location and native bounds."""
    out: list[int] = []
    for index, value in enumerate(arr):
        numeric = isinstance(value, (int, float, np.integer, np.floating))
        message = (
            "column values must be finite integers" if numeric else "column values must be integers"
        )
        out.append(
            _count(
                value,
                f"{field}={name!r}[{index}]",
                minimum=0,
                maximum=2**64 - 1,
                type_error=CausalValueError,
                type_message=f"{{what}}: {message}; got {{value!r}}",
            )
        )
    return tuple(out)


_KIND_CAST = {
    "bool": _cast_bool,
    "str": lambda arr, field, name: tuple(str(v) for v in arr),
    "float": lambda arr, field, name: tuple(float(v) for v in arr),
    "int": _cast_int,
}


def defer_columns(obj: Any) -> bool:
    """True when ``obj`` holds an unresolved column name, directly or in a nested design.

    Called at the top of an opting-in ``__post_init__``: a ``True`` return means the
    object (or one of its ``_COLUMN_NESTED`` designs) holds unresolved column names and
    must skip array validation until :func:`resolve_columns` rebuilds it. This lets a
    wrapper defer whenever a design it carries deferred, so its row-length checks do
    not run against a still-unresolved child.
    """
    cls = type(obj)
    spec = getattr(cls, "_COLUMN_FIELDS", None)
    if spec and any(isinstance(getattr(obj, name), str) for name in spec):
        return True
    for nested in getattr(cls, "_COLUMN_NESTED", ()):
        if getattr(getattr(obj, nested, None), "_deferred_columns", False):
            return True
    return False


def collect_column_names(obj: Any) -> set[str]:
    """Every data-column name a deferred query (and its nested designs) references."""
    if obj is None or not is_dataclass(obj) or isinstance(obj, type):
        return set()
    names: set[str] = set()
    spec = getattr(type(obj), "_COLUMN_FIELDS", None)
    if spec and getattr(obj, "_deferred_columns", False):
        names |= {getattr(obj, field) for field in spec if isinstance(getattr(obj, field), str)}
    for nested in getattr(type(obj), "_COLUMN_NESTED", ()):
        names |= collect_column_names(getattr(obj, nested, None))
    return names


def _data_column_map(data: Any, names: set[str]) -> dict[str, Any]:
    """Extract the requested columns from raw ``data`` (dict-like or DataFrame)."""
    if isinstance(data, Mapping):
        source: Any = data
    elif hasattr(data, "columns") and hasattr(data, "__getitem__"):
        source = data  # pandas / Arrow-like: column access by name
    else:
        raise CausalValueError(
            "column-name query inputs require dict or DataFrame data; pass the arrays "
            "inline instead, or supply the columns in a table"
        )
    out: dict[str, Any] = {}
    for name in names:
        try:
            column = source[name]
        except (KeyError, IndexError) as exc:
            raise CausalValueError(
                f"query names column {name!r}, which is not in the data"
            ) from exc
        out[name] = np.asarray(column.to_numpy() if hasattr(column, "to_numpy") else column)
    return out


def drop_columns(data: Any, names: set[str] | frozenset[str]) -> Any:
    """Return ``data`` without the design columns, so the numeric ingest never sees them."""
    if not names:
        return data
    if isinstance(data, Mapping):
        return {key: value for key, value in data.items() if key not in names}
    if hasattr(data, "column_names") and hasattr(data, "drop_columns"):
        # pyarrow.Table: ``.columns`` holds the arrays, the names are ``column_names``.
        present = [name for name in names if name in data.column_names]
        return data.drop_columns(present) if present else data
    if hasattr(data, "drop") and hasattr(data, "columns"):
        present = [name for name in names if name in getattr(data, "columns", ())]
        return data.drop(columns=present) if present else data
    return data


def resolve_query(query: Any, data: Any) -> tuple[Any, Any]:
    """Resolve a design query's column-name inputs from ``data``.

    Returns ``(resolved_query, data_without_design_columns)``. Design columns are
    read from the raw data (any dtype, so string unit / subject ids work) and then
    dropped, so the numeric analysis ingest only sees the outcome and covariates.
    A query that passed its inputs inline is returned unchanged with the data intact.
    """
    names = collect_column_names(query)
    if not names:
        return query, data
    columns = _data_column_map(data, names)
    return resolve_columns(query, columns), drop_columns(data, names)


def _resolve_one(columns: Mapping[str, Any], name: str, kind: str, field: str) -> Any:
    if name not in columns:
        raise CausalValueError(
            f"{field}={name!r} names a column that is not in the data; "
            f"available columns: {sorted(columns)}"
        )
    arr = np.asarray(columns[name])
    if arr.ndim != 1:
        raise CausalValueError(f"{field}={name!r} must name a one-dimensional column")
    cast = _KIND_CAST.get(kind)
    return tuple(arr.tolist()) if cast is None else cast(arr, field, name)


def resolve_columns(obj: Any, columns: Mapping[str, Any]) -> Any:
    """Return ``obj`` with any deferred column-name fields resolved from ``columns``.

    Objects that did not opt in, or that carry no unresolved column name, are
    returned unchanged (identity), so the inline-array path pays nothing.
    """
    if obj is None or not is_dataclass(obj) or isinstance(obj, type):
        return obj
    updates: dict[str, Any] = {}
    spec = getattr(type(obj), "_COLUMN_FIELDS", None)
    if spec and getattr(obj, "_deferred_columns", False):
        for name, kind in spec.items():
            value = getattr(obj, name)
            if isinstance(value, str):
                updates[name] = _resolve_one(columns, value, kind, name)
    for nested in getattr(type(obj), "_COLUMN_NESTED", ()):
        child = getattr(obj, nested, None)
        resolved = resolve_columns(child, columns)
        if resolved is not child:
            updates[nested] = resolved
    if not updates:
        return obj
    # replace() re-runs __post_init__; with the names now resolved to arrays the
    # class validates exactly as if the caller had passed the arrays inline.
    return replace(obj, **updates)


def first_seen_codes(labels: Iterable[Hashable]) -> tuple[tuple[Hashable, ...], list[int]]:
    """Distinct labels in first-seen order, and each label's code (its place among them)."""
    index: dict[Hashable, int] = {}
    codes = [index.setdefault(label, len(index)) for label in labels]
    return tuple(index), codes


@dataclass(frozen=True, slots=True)
class IdentifierBinding:
    """A named identifier column encoded over the rows of the data it was read from.

    ``labels[code]`` is the label of each code, in first-seen order; ``digest`` covers
    the per-row codes. A refreshed table that encodes differently holds different
    identifiers (other labels, or the same labels on other rows) and is refused.
    ``row_labels`` keeps the original label of every row for a participant column, whose
    values the caller reads back; it is empty for a cluster column. ``codes`` are the per-row
    codes (little-endian ``u32``) that name the rows whose identity changed when a refreshed
    table is refused; neither takes part in equality, which is the identity of the column
    (labels, codes digest, row count). A ``kind="weights"``
    binding covers a sampling-weight column: its ``digest`` is over the exact weight bits, and
    it has no labels.
    """

    column: str
    labels: tuple[str, ...]
    digest: str
    rows: int
    row_labels: tuple[str, ...] = field(default=(), compare=False)
    kind: str = "identifier"
    codes: bytes = field(default=b"", repr=False, compare=False)

    def contract_binding(self) -> dict[str, Any]:
        """The digests that enter the exported contract's data snapshot: rows, labels, codes."""
        labels = hashlib.sha256("\x00".join(self.labels).encode("utf-8")).hexdigest()
        return {"rows": self.rows, "labels": labels, "codes": self.digest}


def _identifier_label(value: Any, column: str, row: int) -> Hashable:
    """One identifier value as a hashable label; a missing or non-label value is refused."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not np.isfinite(value):
            raise CausalValueError(
                f"identifier column {column!r} is missing at row {row}; identifiers must be present "
                "on every row",
                reason_code="invalid_argument",
            )
        if value != int(value):
            raise CausalValueError(
                f"identifier column {column!r} holds the non-integer number {value!r} at row {row}; "
                "identifiers are labels or whole numbers",
                reason_code="invalid_argument",
            )
        return int(value)
    if isinstance(value, (str, int)):
        if isinstance(value, str) and not value.strip():
            raise CausalValueError(
                f"identifier column {column!r} holds an empty label at row {row}",
                reason_code="invalid_argument",
            )
        return value
    if value is None or type(value).__name__ in ("NAType", "NaTType"):
        raise CausalValueError(
            f"identifier column {column!r} is missing at row {row}; identifiers must be present "
            "on every row",
            reason_code="invalid_argument",
        )
    return str(value)


def encode_identifier(
    data: Any, column: str, *, what: str, keep_rows: bool = False
) -> tuple[IdentifierBinding, list[int]]:
    """Encode the identifier column ``column`` of ``data`` over its original rows.

    The column is read in its own dtype (string labels included), outside the numeric
    ingest, and encoded to ``u32`` codes in first-seen order. The codes belong to the
    original rows; the estimator keeps the rows it analyses and their codes together.
    """
    values = read_columns(data, [column], what=what)[column]
    row_values = [
        _identifier_label(value, column, row) for row, value in enumerate(values.tolist())
    ]
    labels, codes = first_seen_codes(row_values)
    if len(labels) > 2**32 - 1:
        raise CausalValueError(
            f"identifier column {column!r} has more distinct labels than fit in 32 bits",
            reason_code="invalid_argument",
        )
    digest = hashlib.sha256(np.asarray(codes, dtype="<u4").tobytes()).hexdigest()
    binding = IdentifierBinding(
        column=column,
        labels=tuple(str(label) for label in labels),
        digest=digest,
        rows=len(codes),
        row_labels=tuple(str(value) for value in row_values) if keep_rows else (),
        codes=np.asarray(codes, dtype="<u4").tobytes(),
    )
    return binding, codes


def encode_weights(
    data: Any, weights: Any, *, estimator: str | None
) -> tuple[list[float], IdentifierBinding | None]:
    """Sampling weights for every original row of ``data``, and the binding of a named column.

    ``weights`` names a numeric column of ``data`` (read before the numeric ingest, so its
    values are the table's own) or is an array of one weight per row. A weight that is
    negative, NaN or infinite, or weights that are all zero, are refused natively with the
    row of the first offender; this only checks that the values are numbers. The binding
    freezes the column's exact bits so a refreshed table with other weights is refused.
    """
    if estimator not in _WEIGHTED_ESTIMATORS:
        raise CausalUnsupportedError(
            f"weights= declares sampling weights, which {estimator!r} does not read "
            "(weights= is available for linear.adjustment.ate, aipw, propensity.weighting, "
            "propensity.matching, propensity.stratification, distance.matching, "
            "glm.adjustment, dml and dr.learner)",
            reason_code="option_not_applicable",
        )
    column: str | None = None
    if isinstance(weights, str):
        if not weights.strip():
            raise CausalValueError(
                "weights must name a data column or hold one weight per row",
                reason_code="invalid_argument",
            )
        column = weights
        values = read_columns(data, [column], what="weights")[column].tolist()
    else:
        values = list(weights)
    try:
        floats = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise CausalValueError(
            f"sampling weights must be numbers: {error}", reason_code="invalid_argument"
        ) from error
    if column is None:
        return floats, None
    digest = hashlib.sha256(np.asarray(floats, dtype="<f8").tobytes()).hexdigest()
    binding = IdentifierBinding(
        column=column, labels=(), digest=digest, rows=len(floats), kind="weights"
    )
    return floats, binding


_CLUSTER_DML_ESTIMATORS = ("aipw", "dml", "dr.learner")
_CLUSTER_ESTIMATORS = (
    None,
    "linear.adjustment.ate",
    "glm.adjustment",
    "aipw",
    "frontdoor.linear_two_stage",
    "propensity.weighting",
    "propensity.stratification",
    "causal.forest",
)
_WEIGHTED_ESTIMATORS = (
    None,
    "linear.adjustment.ate",
    "aipw",
    "propensity.weighting",
    "propensity.matching",
    "propensity.stratification",
    "distance.matching",
    "glm.adjustment",
    "dml",
    "dr.learner",
)
_NO_CLUSTER_VARIANCE = {
    "iv.wald": "carries no cluster-robust variance",
    "iv.2sls": "carries no cluster-robust variance",
    "propensity.matching": "refuses cluster standard errors (no clustered influence function "
    "exists for fixed-match nearest-neighbour matching)",
    "distance.matching": "refuses cluster standard errors (no clustered influence function "
    "exists for fixed-match nearest-neighbour matching)",
}


def _check_cluster_estimator(estimator: str | None, allowed: tuple[str | None, ...]) -> None:
    if estimator in allowed:
        return
    reason = _NO_CLUSTER_VARIANCE.get(
        str(estimator), "has no cluster standard error that reads cluster labels per row"
    )
    raise CausalUnsupportedError(
        f"cluster= names a cluster column that the estimator aligns to the rows it analyses; "
        f"{estimator!r} {reason} (cluster= is available for linear.adjustment.ate, "
        "glm.adjustment, aipw, dml, dr.learner, frontdoor.linear_two_stage, "
        "propensity.weighting, propensity.stratification and causal.forest)",
        reason_code="option_not_applicable",
    )


def _joint_codes(data: Any, columns: Sequence[str]) -> list[list[int]]:
    """First-seen codes of several identifier columns in one shared vocabulary.

    An entity that appears in two columns gets one code, so a dyadic declaration whose
    endpoints share entities is seen as such by the estimator.
    """
    index: dict[Hashable, int] = {}
    codes: list[list[int]] = []
    for column in columns:
        values = read_columns(data, [column], what="cluster")[column].tolist()
        codes.append(
            [
                index.setdefault(_identifier_label(v, column, row), len(index))
                for row, v in enumerate(values)
            ]
        )
    return codes


def resolve_estimator_identifiers(
    estimator: str | None, config: Mapping[str, Any] | None, data: Any
) -> tuple[Mapping[str, Any] | None, dict[str, IdentifierBinding]]:
    """Replace named ``cluster`` and ``weights`` columns in an estimator configuration.

    The labels are those of every original row of ``data``; the estimator aligns them to
    the rows it analyses after its complete-case and population masks, so a row that is
    dropped takes its label with it. A ``cluster_dml`` declaration names its unit columns
    the same way (``cluster``, and ``second_cluster`` for a dyad). ``weights`` names a numeric
    column (or gives one weight per row) of sampling weights, declared for every original row
    like the cluster labels. Returns the configuration the native parser reads and the frozen
    bindings to check a later refresh against.
    """
    if not config:
        return config, {}
    declared = config.get("cluster_dml")
    named_dml = isinstance(declared, Mapping) and (
        "cluster" in declared or "second_cluster" in declared
    )
    if "cluster" not in config and not named_dml and "weights" not in config:
        return config, {}
    resolved = dict(config)
    bindings: dict[str, IdentifierBinding] = {}
    if "weights" in config:
        if "source_sampling_weights" in config:
            raise CausalValueError(
                "weights declares the sampling weights; do not also pass source_sampling_weights",
                reason_code="invalid_argument",
            )
        floats, weight_binding = encode_weights(data, config["weights"], estimator=estimator)
        del resolved["weights"]
        resolved["source_sampling_weights"] = floats
        if weight_binding is not None:
            bindings[weight_binding.column] = weight_binding
    if "cluster" in config:
        column = config["cluster"]
        if not isinstance(column, str) or not column.strip():
            raise CausalValueError(
                "cluster must name a data column", reason_code="invalid_argument"
            )
        if "cluster_ids" in config:
            raise CausalValueError(
                "cluster names a column; do not also pass cluster_ids",
                reason_code="invalid_argument",
            )
        _check_cluster_estimator(estimator, _CLUSTER_ESTIMATORS)
        if "cluster_dml" in config:
            raise CausalUnsupportedError(
                "cluster= names the cluster standard-error column; a cluster_dml declaration "
                "names its own unit column (ClusterDml(cluster=...))",
                reason_code="option_not_applicable",
            )
        binding, codes = encode_identifier(data, column, what="cluster")
        del resolved["cluster"]
        resolved["source_cluster_ids"] = codes
        bindings[column] = binding
    if named_dml:
        _check_cluster_estimator(estimator, _CLUSTER_DML_ESTIMATORS)
        unit = dict(declared)
        names = [unit[key] for key in ("cluster", "second_cluster") if key in unit]
        for key in ("cluster", "second_cluster"):
            if key in unit and not isinstance(unit[key], str):
                raise CausalValueError(
                    f"ClusterDml.{key} must name a data column", reason_code="invalid_argument"
                )
        codes = _joint_codes(data, names)
        for name in names:
            bindings[name] = encode_identifier(data, name, what="cluster")[0]
        if "cluster" in unit:
            del unit["cluster"]
            unit["source_cluster_ids"] = codes.pop(0)
        if "second_cluster" in unit:
            del unit["second_cluster"]
            unit["source_second_cluster_ids"] = codes.pop(0)
        resolved["cluster_dml"] = unit
    return resolved, bindings


@dataclass(frozen=True, slots=True)
class IdentifierChange:
    """One row whose identifier differs from the label frozen at prepare."""

    row: int
    frozen: str
    supplied: str


@dataclass(frozen=True, slots=True)
class IdentifierRefreshReceipt:
    """What a refused refresh found in a frozen identifier column.

    ``changes`` lists every row whose label differs from the frozen one, in row order.
    ``duplicated`` lists the labels the supplied column repeats although every frozen label was
    unique. ``standing`` is ``recompute``: a claim computed under the frozen identifiers does
    not describe the supplied table, and only ``reprepare`` binds the new ones; the prepared
    study and its last result are untouched.
    """

    column: str
    rows: int
    changes: tuple[IdentifierChange, ...]
    duplicated: tuple[str, ...]
    standing: str = "recompute"


def _refresh_receipt(
    frozen: IdentifierBinding, supplied: IdentifierBinding
) -> IdentifierRefreshReceipt:
    """The rows whose label moved between the frozen identifiers and a supplied column."""
    frozen_codes = np.frombuffer(frozen.codes, dtype="<u4")
    frozen_labels = (
        [frozen.labels[int(code)] for code in frozen_codes]
        if frozen_codes.size == frozen.rows
        else []
    )
    changes = tuple(
        IdentifierChange(row, before, after)
        for row, (before, after) in enumerate(zip(frozen_labels, supplied.row_labels, strict=False))
        if before != after
    )
    seen: dict[str, int] = {}
    for label in supplied.row_labels:
        seen[label] = seen.get(label, 0) + 1
    unique_when_frozen = len(frozen.labels) == frozen.rows
    duplicated = (
        tuple(sorted(label for label, count in seen.items() if count > 1))
        if unique_when_frozen
        else ()
    )
    return IdentifierRefreshReceipt(frozen.column, frozen.rows, changes, duplicated)


def _refresh_refusal(receipt: IdentifierRefreshReceipt) -> CausalValueError:
    shown = ", ".join(
        f"row {change.row} {change.frozen!r} -> {change.supplied!r}"
        for change in receipt.changes[:5]
    )
    detail = f"{len(receipt.changes)} row(s) differ ({shown}{', ...' if len(receipt.changes) > 5 else ''})"
    if receipt.duplicated:
        duplicated = receipt.duplicated
        detail += (
            f"; duplicate identifier on refresh: {list(duplicated[:5])}"
            f"{'...' if len(duplicated) > 5 else ''} repeat in the supplied column although every "
            "frozen identifier was unique"
        )
    refusal = CausalValueError(
        f"frozen identifier column {receipt.column!r} changed labels or row order"
        f"{': ' + detail if receipt.changes else ''}; claims computed under the frozen identifiers "
        "are marked recompute, use study.reprepare(data) to declare new identifiers",
        reason_code="invalid_argument",
    )
    setattr(refusal, "offending", receipt.column)  # noqa: B010 — structured optional error field
    setattr(refusal, "receipt", receipt)  # noqa: B010 — structured optional error field
    refusal.remedy = "use study.reprepare(data) to bind the new identifiers"
    return refusal


def guard_identifiers(study: Any, data: Any) -> None:
    """Refuse a table whose frozen identifier columns changed, before anything runs.

    The refusal of a changed label column carries an :class:`IdentifierRefreshReceipt` as
    ``receipt``: the rows whose identity moved and the claim standing ``recompute``.
    """
    bindings: Mapping[str, IdentifierBinding] = getattr(study, "_identifier_bindings", {})
    if not bindings or data is None:
        return
    available = table_column_names(data)
    rows = table_row_count(data)
    for column, frozen in bindings.items():
        if rows is not None and rows != frozen.rows:
            refusal = CausalValueError(
                f"frozen identifier column {column!r} labels {frozen.rows} rows; the supplied "
                f"table has {rows}; use study.reprepare(data) to declare new identifiers",
                reason_code="invalid_argument",
            )
            setattr(refusal, "offending", column)  # noqa: B010 — structured optional error field
            refusal.remedy = "use study.reprepare(data) to bind the new identifiers"
            raise refusal
        if available is None or column not in available:
            continue  # the frozen labels stay; the native side checks the row count
        if frozen.kind == "weights":
            supplied, _ = encode_weights(data, column, estimator=None)
            digest = hashlib.sha256(np.asarray(supplied, dtype="<f8").tobytes()).hexdigest()
            if digest != frozen.digest:
                refusal = CausalValueError(
                    f"frozen weights column {column!r} changed values; "
                    "use study.reprepare(data) to declare new identifiers",
                    reason_code="invalid_argument",
                )
                setattr(refusal, "offending", column)  # noqa: B010 — structured optional error field
                refusal.remedy = "use study.reprepare(data) to bind the new identifiers"
                raise refusal
            continue
        supplied_binding, _ = encode_identifier(data, column, what="cluster", keep_rows=True)
        if supplied_binding != frozen:
            raise _refresh_refusal(_refresh_receipt(frozen, supplied_binding))


@dataclass(frozen=True, slots=True)
class AnalysisColumns:
    """The columns an analysis reads: their names, and whether their order is fixed."""

    names: tuple[str, ...]
    ordered: bool


def _declared_strings(value: Any, into: set[str]) -> None:
    """Every string a declaration holds (column names are strings, so this is a superset)."""
    if isinstance(value, str):
        into.add(value)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _declared_strings(key, into)
            _declared_strings(item, into)
    elif is_dataclass(value) and not isinstance(value, type):
        data_fields = getattr(type(value), "_COLUMN_FIELDS", {})
        for field in fields(value):
            if field.name not in data_fields:  # row-aligned design arrays are data, not names
                _declared_strings(getattr(value, field.name, None), into)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _declared_strings(item, into)


def analysis_columns(
    query: Any, graph: Any, estimator_config: Mapping[str, Any] | None
) -> AnalysisColumns | None:
    """The columns a typed-graph analysis reads, or ``None`` when every column is read.

    A graph whose nodes are positional (``Admg``, ``Pag``, ``Cpdag``) binds the data
    columns by name and order, so the analysis reads exactly its nodes, in its order.
    A ``Dag`` or an edge list binds by name: the analysis reads its nodes and the
    columns the query and estimator configuration name, in the data's order. Without a
    graph (discovery) or with a temporal graph, every column is read.
    """
    from .graph import Admg, Cpdag, Dag, Pag

    if isinstance(graph, (Admg, Cpdag, Pag)):
        return AnalysisColumns(tuple(graph.nodes()), ordered=True)
    declared: set[str] = set()
    if isinstance(graph, Dag):
        declared.update(graph.nodes())
    elif isinstance(graph, Sequence) and not isinstance(graph, str):
        for edge in graph:
            _declared_strings(tuple(edge), declared)
    else:
        return None
    _declared_strings(query, declared)
    _declared_strings(estimator_config, declared)
    return AnalysisColumns(tuple(sorted(declared)), ordered=False)


def refuse_named_identifiers(config: Mapping[str, Any] | None, route: str) -> None:
    """Refuse a named ``cluster`` or ``weights`` column on a route that never reads the raw table."""
    if config and "cluster" in config:
        raise CausalUnsupportedError(
            f"{route} does not resolve cluster= against a table; pass cluster_ids aligned to the "
            "analysed rows, or use analyze / prepare, which bind a named cluster column",
            reason_code="option_not_applicable",
        )
    if config and isinstance(config.get("weights"), str):
        raise CausalUnsupportedError(
            f"{route} does not resolve weights= against a table; pass one weight per row, or "
            "use analyze / prepare, which bind a named weights column",
            reason_code="option_not_applicable",
        )
