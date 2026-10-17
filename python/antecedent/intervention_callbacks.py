"""Batch callbacks for intervention programs.

A callback is a Python function of numpy arrays, one array per declared input, that returns one
real value per row. It is called **once per batch** by the same executor that runs closed-form
intervention programs, so its inputs are dependency-checked like any other read (a cycle, an
unknown variable or a history read is refused before anything runs), and its output is validated
before it reaches the draws: length, a real dtype, finiteness and the declared support. A
Python exception inside the callback surfaces with its own message under
``reason_code="policy_callback_failed"``; it is never replaced by a default or a ``nan``.

A callback is a **runtime binding**: it runs in this process and is not portable.
:meth:`InterventionCallback.recovery_record` and :func:`export_records` therefore refuse
(``policy_callback_not_portable``) unless the callback was declared from self-contained source
with :meth:`InterventionCallback.from_source`. That declaration fixes its behaviour to the
SHA-256 of the exact source text, an entry point and pinned dependency versions; the record that
leaves the process carries those and never the code. A consumer supplies the source itself and
:func:`recover_callback` re-verifies the digest, the entry point, the declared signature and
every pinned version before replay (``policy_callback_unrecoverable`` on any difference).

A program that applies recoverable callbacks has a portable encoding: :func:`encode_programs`
writes each callback as its record under an encoding version only such programs use (a program
without callbacks keeps the bytes and identity it always had), :func:`required_callbacks` lists
the records a consumer must supply implementations for, and :func:`decode_programs` binds the
implementations :func:`recover_callback` rebuilt, verifying each against its record once more
before anything is returned for replay.

The restrictions on source are an identity guarantee, not a security sandbox: it may not import
(declared dependencies are bound by name, and ``numpy`` is always bound as ``np``), reach dunder
attributes or use ``eval`` / ``exec`` / ``open`` / ``globals``, and it runs with a fixed set of
builtins, so its behaviour is a function of the text and the pinned versions. Randomness and
clocks are not detectable from source; a callback that uses them is not licensed for
counterfactual execution (the route's licence states that assumption).
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import importlib
import importlib.metadata
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np

from ._native import (
    BatchCallbackHandle,
    callback_export_refusal,
    decode_callback_policy_program,
    encode_callback_policy_program,
    policy_program_callback_records,
    verify_callback_recovery,
)
from .errors import CausalUnsupportedError, CausalValueError
from .intervention_programs import Program, _programs
from .intervention_programs import _text as _program_text

__all__ = [
    "GENERATED",
    "CallbackProgram",
    "CallbackRecord",
    "InterventionCallback",
    "decode_programs",
    "encode_programs",
    "export_records",
    "recover_callback",
    "required_callbacks",
    "sample_do",
]

NOT_PORTABLE: Final = "policy_callback_not_portable"
UNRECOVERABLE: Final = "policy_callback_unrecoverable"


class _Generated:
    """The value a node's own mechanism generated; readable by transforms only."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "GENERATED"


GENERATED: Final = _Generated()

_BUILTINS: Final[dict[str, Any]] = {
    name: getattr(builtins, name)
    for name in [
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "pow",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "sum",
        "tuple",
        "zip",
        "ArithmeticError",
        "Exception",
        "FloatingPointError",
        "TypeError",
        "ValueError",
        "ZeroDivisionError",
    ]
}
_FORBIDDEN_NAMES: Final = frozenset(
    [
        "__import__",
        "eval",
        "exec",
        "compile",
        "open",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
        "input",
        "exit",
        "quit",
    ]
)


def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise CausalValueError(f"{what} must be a non-empty string")
    return value


def _real(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise CausalValueError(f"{what} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise CausalValueError(f"{what} must be finite")
    return number


def _output_declaration(
    lower: float | None, upper: float | None, levels: Iterable[float] | None
) -> tuple[Literal["real", "range", "levels"], list[float]]:
    if levels is not None:
        if lower is not None or upper is not None:
            raise CausalValueError("declare either levels or a lower/upper range, not both")
        values = sorted({_real(v, "a level") for v in levels})
        if not values:
            raise CausalValueError("levels must not be empty")
        return "levels", values
    if lower is None and upper is None:
        return "real", []
    if lower is None or upper is None:
        raise CausalValueError("a range output needs both lower and upper")
    low, high = _real(lower, "lower"), _real(upper, "upper")
    if low > high:
        raise CausalValueError("lower must not exceed upper")
    return "range", [low, high]


def _pins(
    dependencies: Mapping[str, str | None] | None, *, numpy_pinned: bool
) -> tuple[list[tuple[str, str]], tuple[str, ...]]:
    """Pinned ``(name, version)`` pairs, and the names declared without a version."""
    pins: dict[str, str] = {}
    unpinned: list[str] = []
    for name, version in dict(dependencies or {}).items():
        _text(name, "a dependency name")
        if version is None:
            unpinned.append(name)
        else:
            pins[name] = _text(version, f"the pinned version of {name}")
    if numpy_pinned:
        pins.setdefault("numpy", np.__version__)
    return sorted(pins.items()), tuple(sorted(unpinned))


def _verify_installed(
    pins: Sequence[tuple[str, str]], unpinned: Sequence[str], callback: str
) -> None:
    for name, version in pins:
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise CausalUnsupportedError(
                f"callback {callback} depends on {name}=={version}, which is not installed",
                reason_code=UNRECOVERABLE,
                remedy="install the pinned dependency version",
            ) from None
        if installed != version:
            raise CausalUnsupportedError(
                f"callback {callback} is pinned to {name}=={version} but {installed} is installed",
                reason_code=UNRECOVERABLE,
                remedy="install the pinned dependency version, or re-declare the callback",
            )
    for name in unpinned:
        try:
            importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise CausalUnsupportedError(
                f"callback {callback} depends on {name}, which is not installed",
                reason_code=UNRECOVERABLE,
                remedy="install the dependency",
            ) from None


def _check_source(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            raise CausalValueError(
                "callback source may not import; declared dependencies are bound by name"
            )
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise CausalValueError(f"callback source may not use {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise CausalValueError("callback source may not reach dunder attributes")
        if isinstance(node, ast.Global | ast.Nonlocal):
            raise CausalValueError("callback source may not rebind outer names")


def _compile_source(source: str, entry_point: str, modules: Sequence[str]) -> Callable[..., Any]:
    tree = ast.parse(source)
    _check_source(tree)
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if entry_point not in defined:
        raise CausalValueError(f"source defines no top-level function {entry_point!r}")
    namespace: dict[str, Any] = {"__builtins__": dict(_BUILTINS), "np": np}
    for name in modules:
        namespace[name.replace("-", "_")] = importlib.import_module(name.replace("-", "_"))
    exec(compile(tree, "<callback source>", "exec"), namespace)  # noqa: S102 - see module docstring
    return namespace[entry_point]  # type: ignore[no-any-return]


@dataclass(frozen=True, slots=True)
class CallbackRecord:
    """What an artifact may carry for a recoverable callback: identity, never code."""

    name: str
    inputs: tuple[str, ...]
    output_kind: Literal["real", "range", "levels"]
    output_params: tuple[float, ...]
    source_sha256: str
    entry_point: str
    dependencies: tuple[tuple[str, str], ...]

    @property
    def provenance_id(self) -> str:
        """The provenance record (``provenance/<id>.toml``) the callback executor is defined in."""
        return "model.intervention_programs"

    def to_dict(self) -> dict[str, Any]:
        """Plain-data form, for embedding in an artifact."""
        return {
            "name": self.name,
            "inputs": list(self.inputs),
            "output_kind": self.output_kind,
            "output_params": list(self.output_params),
            "source_sha256": self.source_sha256,
            "entry_point": self.entry_point,
            "dependencies": [list(pin) for pin in self.dependencies],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CallbackRecord:
        """Inverse of :meth:`to_dict`."""
        return cls(
            name=data["name"],
            inputs=tuple(data["inputs"]),
            output_kind=data["output_kind"],
            output_params=tuple(float(p) for p in data["output_params"]),
            source_sha256=data["source_sha256"],
            entry_point=data["entry_point"],
            dependencies=tuple((str(n), str(v)) for n, v in data["dependencies"]),
        )


class InterventionCallback:
    """A batch function of numpy arrays, declared for use in intervention programs.

    ``function(*columns)`` receives one float64 array per name in ``inputs`` (all of the batch's
    length) and returns a one-dimensional real array of that length. ``lower`` / ``upper`` declare
    a range and ``levels`` a finite set the output must lie in; with neither the output is any
    finite real. ``dependencies`` maps distribution names to the exact versions the function was
    written against; each must be installed at that version. A version of ``None`` declares a
    dependency that is installed but not pinned: the callback may use it, but its behaviour
    cannot be verified by a consumer, so the callback is refused for export.

    This constructor makes a runtime binding. Use :meth:`from_source` for a callback that can be
    recorded and replayed.
    """

    __slots__ = ("_handle", "_unpinned")

    def __init__(
        self,
        function: Callable[..., Any],
        *,
        inputs: Sequence[str],
        name: str | None = None,
        lower: float | None = None,
        upper: float | None = None,
        levels: Iterable[float] | None = None,
        dependencies: Mapping[str, str | None] | None = None,
    ) -> None:
        if not callable(function):
            raise CausalValueError("function must be callable")
        label = name if name is not None else getattr(function, "__name__", "")
        pins, self._unpinned = _pins(dependencies, numpy_pinned=False)
        self._handle = self._build(
            function, label, inputs, lower, upper, levels, pins, self._unpinned
        )

    @staticmethod
    def _build(
        function: Callable[..., Any],
        name: str,
        inputs: Sequence[str],
        lower: float | None,
        upper: float | None,
        levels: Iterable[float] | None,
        pins: Sequence[tuple[str, str]],
        unpinned: Sequence[str],
        source: tuple[str, str] | None = None,
    ) -> BatchCallbackHandle:
        _text(name, "name")
        if isinstance(inputs, str):
            raise CausalValueError("inputs must be a sequence of names, not one string")
        labels = [_text(label, "an input name") for label in inputs]
        if len(set(labels)) != len(labels):
            raise CausalValueError("input names must be distinct")
        kind, params = _output_declaration(lower, upper, levels)
        _verify_installed(pins, unpinned, name)
        digest, entry = source if source is not None else (None, None)
        return BatchCallbackHandle(function, name, labels, kind, params, list(pins), digest, entry)

    @classmethod
    def from_source(
        cls,
        source: str,
        *,
        entry_point: str,
        inputs: Sequence[str],
        name: str | None = None,
        lower: float | None = None,
        upper: float | None = None,
        levels: Iterable[float] | None = None,
        dependencies: Mapping[str, str | None] | None = None,
    ) -> InterventionCallback:
        """Declare a callback from self-contained source, so it has a verifiable record.

        ``source`` defines ``entry_point`` as a top-level function and may use only ``np`` and the
        declared ``dependencies`` (see the module docstring for the restrictions). Its identity is
        the SHA-256 of ``source`` exactly as given. A dependency declared with version ``None`` is
        usable but unpinned, which makes the callback unexportable.
        """
        if not isinstance(source, str):
            raise CausalValueError("source must be a string")
        pins, unpinned = _pins(dependencies, numpy_pinned=True)
        entry_point = _text(entry_point, "entry_point")
        label = name if name is not None else entry_point
        _verify_installed(pins, unpinned, label)
        modules = [dependency for dependency, _ in pins] + list(unpinned)
        function = _compile_source(source, entry_point, modules)
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        self = cls.__new__(cls)
        self._unpinned = unpinned
        self._handle = cls._build(
            function,
            label,
            inputs,
            lower,
            upper,
            levels,
            pins,
            unpinned,
            (digest, entry_point),
        )
        return self

    @property
    def name(self) -> str:
        """The callback's name."""
        return self._handle.name

    @property
    def inputs(self) -> tuple[str, ...]:
        """Declared input names, in call order."""
        return tuple(self._handle.inputs)

    @property
    def recoverable(self) -> bool:
        """Whether the callback was declared from source and so has a recovery record."""
        return bool(self._handle.recoverable)

    def recovery_record(self) -> CallbackRecord:
        """The record a consumer verifies.

        Raises:
            CausalUnsupportedError: ``policy_callback_not_portable`` for a runtime binding.
        """
        _refuse_unless_portable([self])
        fields = self._handle.recovery_record()
        return CallbackRecord(
            name=fields["name"],
            inputs=tuple(fields["inputs"]),
            output_kind=fields["output_kind"],
            output_params=tuple(fields["output_params"]),
            source_sha256=fields["source_sha256"],
            entry_point=fields["entry_point"],
            dependencies=tuple((n, v) for n, v in fields["dependencies"]),
        )

    def __repr__(self) -> str:
        kind = "recoverable" if self.recoverable else "runtime binding"
        return f"InterventionCallback({self.name!r}, inputs={self.inputs!r}, {kind})"


def _refuse_unless_portable(callbacks: Sequence[InterventionCallback]) -> None:
    for callback in callbacks:
        if callback._unpinned:
            raise CausalUnsupportedError(
                f"callback {callback.name} depends on {', '.join(callback._unpinned)} with no "
                "pinned version, so a consumer cannot verify the dependency it replays against",
                reason_code=NOT_PORTABLE,
                remedy="declare an exact version for every dependency, or keep the program in "
                "this process",
            )
    refusal = callback_export_refusal([c._handle for c in callbacks])
    if refusal is not None:
        code, message = refusal
        raise CausalUnsupportedError(
            message,
            reason_code=code,
            remedy="declare the callback with InterventionCallback.from_source, or keep the "
            "program in this process",
        )


@dataclass(frozen=True, slots=True)
class CallbackProgram:
    """A callback applied to a node.

    ``mode="assign"`` discards the node's mechanism: its value is the callback of the ``reads``.
    ``mode="transform"`` keeps the mechanism and replaces its generated value; a transform may pass
    ``GENERATED`` among its ``reads``. ``reads`` names the model variables passed to the callback,
    in order, and defaults to the callback's own input names. A callback may read only what it
    declared: any other variable is refused (``policy_dependency_unavailable``, naming it) before
    anything runs.
    """

    variable: str
    callback: InterventionCallback
    reads: Sequence[str | _Generated] | None = None
    mode: Literal["assign", "transform"] = "assign"
    resolved: tuple[str | _Generated, ...] = field(init=False, repr=False, default=())

    def __post_init__(self) -> None:
        _text(self.variable, "variable")
        if not isinstance(self.callback, InterventionCallback):
            raise CausalValueError("callback must be an InterventionCallback")
        if self.mode not in ("assign", "transform"):
            raise CausalValueError("mode must be 'assign' or 'transform'")
        reads = tuple(self.callback.inputs if self.reads is None else self.reads)
        if len(reads) != len(self.callback.inputs):
            raise CausalValueError(
                f"callback {self.callback.name!r} declares {len(self.callback.inputs)} inputs "
                f"but {len(reads)} reads were given"
            )
        for read in reads:
            if read is GENERATED:
                if self.mode != "transform":
                    raise CausalValueError("only a transform can read the generated value")
            else:
                _text(read, "a read")
        object.__setattr__(self, "resolved", reads)

    def _native(self) -> tuple[str, str, BatchCallbackHandle, list[str | None]]:
        declared = self.callback.inputs
        for read in self.resolved:
            if read is not GENERATED and read not in declared:
                raise CausalUnsupportedError(
                    f"callback {self.callback.name} reads {read!r}, which it did not declare; "
                    f"its declared inputs are {list(declared)}",
                    reason_code="policy_dependency_unavailable",
                    remedy="declare the column in the callback's inputs, or read a declared one",
                )
        reads = [None if read is GENERATED else read for read in self.resolved]
        return (self.mode, self.variable, self.callback._handle, reads)  # type: ignore[return-value]


def sample_do(
    model: Any,
    programs: Sequence[CallbackProgram],
    n: int,
    *,
    interventions: Mapping[str, float] | None = None,
    shifts: Mapping[str, float] | None = None,
    seed: int = 0,
    threads: int | None = None,
) -> Any:
    """Interventional draws from a fitted model with callback programs applied.

    Each program's callback runs once for the whole batch of ``n`` draws, after the variables it
    reads are computed. ``interventions`` and ``shifts`` are as in ``FittedGcm.sample_do``; a
    node given two directives is refused (``policy_directive_conflict``). Draws are a function of
    ``seed`` and the callbacks.
    """
    programs = tuple(programs)
    for program in programs:
        if not isinstance(program, CallbackProgram):
            raise CausalValueError("programs must be CallbackProgram instances")
    return model.sample_do(
        dict(interventions or {}),
        n,
        shifts=None if shifts is None else dict(shifts),
        seed=seed,
        threads=threads,
        programs=[program._native() for program in programs],
    )


def export_records(
    programs: Sequence[CallbackProgram | InterventionCallback],
) -> tuple[CallbackRecord, ...]:
    """Recovery records for every callback in ``programs``.

    Raises:
        CausalUnsupportedError: ``policy_callback_not_portable`` when any callback is a runtime
            binding, naming it.
    """
    callbacks = [p.callback if isinstance(p, CallbackProgram) else p for p in programs]
    _refuse_unless_portable(callbacks)
    return tuple(c.recovery_record() for c in callbacks)


def recover_callback(source: str, record: CallbackRecord) -> InterventionCallback:
    """Rebuild the callback ``record`` identifies from the ``source`` the consumer supplies.

    The pinned dependency versions must be installed and the source must hash to the recorded
    digest; nothing runs otherwise.

    Raises:
        CausalUnsupportedError: ``policy_callback_unrecoverable`` naming the first mismatch.
    """
    kind, params = record.output_kind, list(record.output_params)
    callback = InterventionCallback.from_source(
        source,
        entry_point=record.entry_point,
        inputs=record.inputs,
        name=record.name,
        lower=params[0] if kind == "range" else None,
        upper=params[1] if kind == "range" else None,
        levels=params if kind == "levels" else None,
        dependencies=dict(record.dependencies),
    )
    verify_callback_recovery(
        record.name,
        list(record.inputs),
        kind,
        params,
        record.source_sha256,
        record.entry_point,
        list(record.dependencies),
        callback._handle,
    )
    return callback


def encode_programs(names: Sequence[str], programs: Sequence[Program | CallbackProgram]) -> bytes:
    """Portable encoding of closed-form and callback programs, in order.

    Every callback must have been declared with :meth:`InterventionCallback.from_source`; the
    artifact carries its recovery record (source digest, entry point, declared signature and
    pinned dependency versions) and never its code. The result is the encoding
    :func:`antecedent.intervention_programs.encode_policy_program` writes.

    Raises:
        CausalUnsupportedError: ``policy_callback_not_portable`` for a runtime binding, naming it.
    """
    items: list[tuple[Any, Any]] = []
    for program in programs:
        if isinstance(program, CallbackProgram):
            items.append((None, program._native()))
        elif isinstance(program, Program):
            items.append((_program_text([program]), None))
        else:
            raise CausalValueError("programs must be Program or CallbackProgram instances")
    if not items:
        raise CausalValueError("pass one or more programs")
    _refuse_unless_portable([p.callback for p in programs if isinstance(p, CallbackProgram)])
    return bytes(encode_callback_policy_program(list(names), items))


def required_callbacks(data: bytes) -> tuple[CallbackRecord, ...]:
    """The recovery records an encoded program carries, in the order its callbacks run.

    A consumer rebuilds one callback per record with :func:`recover_callback` and passes them to
    :func:`decode_programs`. The artifact's version, identity and validity are checked first.
    """
    return tuple(CallbackRecord.from_dict(r) for r in policy_program_callback_records(bytes(data)))


def decode_programs(
    names: Sequence[str],
    data: bytes,
    callbacks: Mapping[str, InterventionCallback] | Iterable[InterventionCallback],
) -> tuple[Program | CallbackProgram, ...]:
    """Decode an encoded program, binding ``callbacks`` to the records it carries.

    ``callbacks`` are the implementations (by name, or as an iterable); each is verified against
    its record again, so a callback built from other source, an other entry point, another
    signature or other dependency pins is refused, as is a record with no implementation
    (``policy_callback_unrecoverable``). Programs come back in their encoded order.
    """
    supplied = list(callbacks.values() if isinstance(callbacks, Mapping) else callbacks)
    if any(not isinstance(c, InterventionCallback) for c in supplied):
        raise CausalValueError("callbacks must be InterventionCallback instances")
    rows = decode_callback_policy_program(list(names), bytes(data), [c._handle for c in supplied])
    by_name = {c.name: c for c in supplied}
    decoded: list[Program | CallbackProgram] = []
    for kind, first, variable, callback_name, reads in rows:
        if kind == "closed":
            decoded.extend(_programs(first))
        else:
            decoded.append(
                CallbackProgram(
                    variable,
                    by_name[callback_name],
                    reads=[GENERATED if read is None else read for read in reads],
                    mode=first,
                )
            )
    return tuple(decoded)
