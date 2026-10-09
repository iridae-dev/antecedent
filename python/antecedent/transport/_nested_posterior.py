"""Named raw-coordinate priors and original continuous posterior candidate."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, ClassVar, Literal, cast

import numpy as np
from numpy.typing import NDArray

from ..errors import CausalTypeError, CausalUnsupportedError, CausalValueError

BetaKernel = tuple[float, float]


@dataclass(frozen=True, slots=True)
class NestedMarkovPrior:
    """Named Beta kernels on the ORIGINAL eleven Möbius coordinates.

    Each pair is (alpha,beta), finite in [1,1000000]. The product density is
    restricted to the positive feasible c-factor polytope, without conditional
    interval-width division. a is P(X1=0); c_t is P(X3=0|X2=t); q2_i and q4_j
    are zero-level district margins conditional on X1=i and X3=j; g_ij is their
    joint zero cell. For example NestedMarkovPrior(q40=(4,1),q41=(4,1)) changes
    those two kernels while keeping every other raw coordinate uniform.
    """

    coordinates: ClassVar[tuple[str, ...]] = (
        "a",
        "c0",
        "c1",
        "q20",
        "q21",
        "q40",
        "q41",
        "g00",
        "g01",
        "g10",
        "g11",
    )
    a: BetaKernel = (1.0, 1.0)
    c0: BetaKernel = (1.0, 1.0)
    c1: BetaKernel = (1.0, 1.0)
    q20: BetaKernel = (1.0, 1.0)
    q21: BetaKernel = (1.0, 1.0)
    q40: BetaKernel = (1.0, 1.0)
    q41: BetaKernel = (1.0, 1.0)
    g00: BetaKernel = (1.0, 1.0)
    g01: BetaKernel = (1.0, 1.0)
    g10: BetaKernel = (1.0, 1.0)
    g11: BetaKernel = (1.0, 1.0)

    def __post_init__(self) -> None:
        for coordinate in self.coordinates:
            pair = getattr(self, coordinate)
            if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
                raise CausalTypeError(
                    f"prior {coordinate} must be an (alpha,beta) pair",
                    reason_code="invalid_argument",
                )
            converted = []
            for shape in pair:
                if isinstance(shape, bool) or not isinstance(shape, Real):
                    raise CausalTypeError(
                        f"prior {coordinate} shapes must be real numbers",
                        reason_code="invalid_argument",
                    )
                try:
                    value = float(shape)
                except OverflowError as error:
                    raise CausalValueError(
                        "prior shapes must lie in [1,1000000]", reason_code="invalid_argument"
                    ) from error
                if not math.isfinite(value) or not 1 <= value <= 1_000_000:
                    raise CausalValueError(
                        "prior shapes must lie in [1,1000000]", reason_code="invalid_argument"
                    )
                converted.append(value)
            object.__setattr__(self, coordinate, tuple(converted))

    def _wire(self) -> tuple[list[float], list[float]]:
        pairs = [getattr(self, name) for name in self.coordinates]
        return [p[0] for p in pairs], [p[1] for p in pairs]


@dataclass(frozen=True, slots=True)
class _PosteriorDiagnostics:
    rank_rhat: float
    folded_rhat: float
    bulk_ess: float
    tail_ess: float


@dataclass(frozen=True, slots=True, eq=False, init=False)
class NestedMarkovPosteriorCandidate:
    """Full continuous raw-parameter/effect posterior, available in internal builds.

    Four aligned chains, 2048 warmup and 4096 retained draws each, 5M proposal
    bound and fixed 95% credible quantiles are the intended frozen pilot method.
    These quantiles and covariance are Monte Carlo estimates; modern diagnostics
    do not guarantee convergence or endpoint precision. Calibration is unmeasured.
    Scope requires successful original checked-ID/interior point fitting, which is
    a pilot eligibility restriction, not a condition for posterior existence.
    """

    prior: NestedMarkovPrior
    coordinate_order: tuple[str, ...]
    parameter_mean: tuple[float, ...]
    values: tuple[float, float, float]
    covariance: NDArray[np.float64]
    credible_intervals: tuple[tuple[float, float], ...]
    samples: NDArray[np.float64]
    diagnostics: tuple[_PosteriorDiagnostics, ...]
    seed: int
    calibration: Literal["unmeasured"]
    inference: Literal["posterior_candidate_withheld_calibration_unmeasured"]
    identification: Literal["nonparametrically_identified"]
    identity: str
    _native: Any

    def __new__(cls, *args: Any, **kwargs: Any) -> NestedMarkovPosteriorCandidate:
        raise CausalTypeError(
            "posterior candidates require an original native producer or consumer",
            reason_code="invalid_argument",
        )

    @classmethod
    def _from_native(cls, native: Any) -> NestedMarkovPosteriorCandidate:
        from .. import _native

        native_type = getattr(_native, "NativeNestedMarkovPosteriorCandidate", None)
        if native_type is None or not isinstance(native, native_type):
            raise CausalTypeError(
                "posterior candidates require original native authority",
                reason_code="invalid_argument",
            )
        payload = json.loads(native.payload())
        row = payload["posterior"]
        covariance = np.frombuffer(
            np.array(row["covariance"], dtype=np.float64).tobytes(), dtype=np.float64
        ).reshape(14, 14)
        samples = np.frombuffer(
            np.array(row["samples"], dtype=np.float64).tobytes(), dtype=np.float64
        ).reshape(4, 4096, 14)
        covariance.setflags(write=False)
        samples.setflags(write=False)
        shapes = payload["prior"]
        prior = NestedMarkovPrior(
            **dict(
                zip(
                    NestedMarkovPrior.coordinates,
                    zip(shapes["alpha"], shapes["beta"], strict=True),
                    strict=True,
                )
            )
        )
        fields = dict(
            prior=prior,
            coordinate_order=tuple(payload["coordinates"]),
            parameter_mean=tuple(row["mean"][:11]),
            values=cast(tuple[float, float, float], tuple(row["mean"][11:])),
            covariance=covariance,
            credible_intervals=tuple(tuple(pair) for pair in row["credible"]),
            samples=samples,
            diagnostics=tuple(_PosteriorDiagnostics(**d) for d in row["diagnostics"]),
            seed=payload["options"]["seed"],
            calibration=payload["calibration"],
            inference=payload["inference"],
            identification=payload["point"]["receipt"]["status"]["identification"],
            identity=native.identity,
            _native=native,
        )
        instance = object.__new__(cls)
        for name, value in fields.items():
            object.__setattr__(instance, name, value)
        return instance

    def export(self) -> bytes:
        """Export original graph, counts, raw prior, method/RNG and entire joint receipt."""
        return bytes(self._native.export())

    def to_dict(self) -> dict[str, Any]:
        """Inspect full original premises and aligned posterior receipt."""
        return cast(dict[str, Any], json.loads(self._native.payload()))

    @classmethod
    def load(cls, artifact: bytes, *, expected_identity: str) -> NestedMarkovPosteriorCandidate:
        """Fresh bounded consumer independently recomputes original ID and posterior."""
        from .. import _native

        if not isinstance(artifact, bytes) or not isinstance(expected_identity, str):
            raise CausalTypeError(
                "artifact must be bytes and expected_identity must be str",
                reason_code="invalid_argument",
            )
        consumer = getattr(_native, "consume_nested_markov_posterior_candidate", None)
        if consumer is None:
            raise CausalUnsupportedError(
                "nested_markov.route_frozen: posterior consumer remains closed pending calibration",
                reason_code="cell_not_licensed",
            )
        return cls._from_native(consumer(artifact, expected_identity))
