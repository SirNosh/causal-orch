"""Deterministic, eligibility-gated balanced assignment schedules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
from typing import Hashable, Iterable, Mapping


class TreatmentAssignment(IntEnum):
    SUPPRESS = 0
    EXECUTE = 1

    @property
    def label(self) -> str:
        return "EXECUTE" if self is TreatmentAssignment.EXECUTE else "SUPPRESS"


def _key_json(key: Hashable) -> str:
    if isinstance(key, tuple):
        value = list(key)
    else:
        value = key
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _shuffle(values: list[TreatmentAssignment], seed: str, block_key: Hashable) -> None:
    """Fisher-Yates using SHA-256-derived indices for cross-process determinism."""
    for index in range(len(values) - 1, 0, -1):
        digest = hashlib.sha256(f"{seed}\0{_key_json(block_key)}\0{index}".encode()).digest()
        swap_index = int.from_bytes(digest[:8], "big") % (index + 1)
        values[index], values[swap_index] = values[swap_index], values[index]


@dataclass
class AssignmentSchedule:
    _values: dict[Hashable, tuple[TreatmentAssignment, ...]]
    _positions: dict[Hashable, int]

    @classmethod
    def generate(cls, seed: str | int, blocks: Mapping[Hashable, int]) -> "AssignmentSchedule":
        if not blocks:
            raise ValueError("at least one randomization block is required")
        values: dict[Hashable, tuple[TreatmentAssignment, ...]] = {}
        for block_key, size in blocks.items():
            if type(size) is not int or size < 1:
                raise ValueError("block sizes must be positive integers")
            execute_count = size // 2
            block = [TreatmentAssignment.EXECUTE] * execute_count
            block.extend([TreatmentAssignment.SUPPRESS] * (size - execute_count))
            _shuffle(block, str(seed), block_key)
            values[block_key] = tuple(block)
        return cls(values, {key: 0 for key in values})

    def reveal(self, block_key: Hashable, *, eligible: bool) -> TreatmentAssignment | None:
        """Reveal and consume exactly one value only for an eligible proposal."""
        if not eligible:
            return None
        if block_key not in self._values:
            raise KeyError(f"unknown randomization block: {block_key!r}")
        position = self._positions[block_key]
        values = self._values[block_key]
        if position >= len(values):
            raise IndexError(f"randomization block exhausted: {block_key!r}")
        assignment = values[position]
        self._positions[block_key] = position + 1
        return assignment

    @property
    def consumed_count(self) -> int:
        return sum(self._positions.values())

    def consumed_count_for(self, block_key: Hashable) -> int:
        return self._positions[block_key]


def generate_balanced_schedule(
    seed: str | int,
    block_keys: Iterable[Hashable] | Mapping[Hashable, int],
    block_size: int = 2,
) -> AssignmentSchedule:
    """Build a schedule; mapping input permits a distinct size per block."""
    if isinstance(block_keys, Mapping):
        blocks = dict(block_keys)
    else:
        keys = list(block_keys)
        blocks = {key: block_size for key in keys}
    return AssignmentSchedule.generate(seed, blocks)
