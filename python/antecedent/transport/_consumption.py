"""Native-checked transport consumption, separate from a resumable preparation."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from .._freeze import freeze, thaw
from .._scientific import ScientificResult
from ..errors import CausalSerializationError


@dataclass(repr=False, frozen=True, slots=True)
class TransportConsumption(ScientificResult, Mapping[str, Any]):
    """An immutable read model returned only after its family's checked consumer.

    Verification and replay receipts live in ``acceptance``. This object does not
    reconstruct a provider session or silently prepare a new estimator.
    """

    family: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", freeze(self.payload))

    @classmethod
    def from_checked_json(cls, family: str, payload: str) -> TransportConsumption:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise CausalSerializationError(
                "The checked transport consumer returned no report object"
            )
        return cls(family, value)

    @classmethod
    def from_checked_mapping(cls, family: str, payload: Mapping[str, Any]) -> TransportConsumption:
        if not isinstance(payload, Mapping):
            raise CausalSerializationError(
                "The checked transport consumer returned no report object"
            )
        return cls(family, payload)

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.payload)

    def __len__(self) -> int:
        return len(self.payload)

    def to_dict(self) -> dict[str, Any]:
        return thaw(self.payload)
