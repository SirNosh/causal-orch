"""Deterministic, eligibility-gated balanced assignment schedules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from threading import RLock
from typing import Hashable, Iterable, Mapping


_COMMITMENT_LOCKS: dict[str, RLock] = {}
_COMMITMENT_LOCKS_GUARD = RLock()


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


def derive_block_key(
    model_slug: str,
    capability: str,
    scenario_id: str,
    temporal_batch: str | int,
) -> str:
    """Derive a stable, opaque block key from the locked stratum identity."""

    value = _key_json((model_slug, capability, scenario_id, temporal_batch))
    return "block-" + hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AssignmentManifest:
    """Serializable run-specific randomization commitment made before execution."""

    run_id: str
    model_slug: str
    capability: str
    scenario_id: str
    temporal_batch: str | int
    seed: str | int
    block_size: int
    block_key: str = ""
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        for name in ("run_id", "model_slug", "capability", "scenario_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if type(self.block_size) is not int or self.block_size < 1:
            raise ValueError("block_size must be a positive integer")
        derived = derive_block_key(self.model_slug, self.capability, self.scenario_id, self.temporal_batch)
        if self.block_key and self.block_key != derived:
            raise ValueError("block_key does not match the run-specific stratum")
        object.__setattr__(self, "block_key", derived)
        computed = hashlib.sha256(self._canonical_payload().encode("utf-8")).hexdigest()
        if self.manifest_sha256 and self.manifest_sha256 != computed:
            raise ValueError("manifest_sha256 does not match the assignment manifest")
        object.__setattr__(self, "manifest_sha256", computed)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        capability: str,
        scenario_id: str,
        temporal_batch: str | int,
        seed: str | int,
        block_size: int,
        model_slug: str | None = None,
        model: str | None = None,
    ) -> "AssignmentManifest":
        selected_model = model_slug or model
        if not selected_model:
            raise ValueError("model_slug is required")
        return cls(run_id, selected_model, capability, scenario_id, temporal_batch, seed, block_size)

    def _canonical_payload(self) -> str:
        return json.dumps(self.to_dict(include_hash=False), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "run_id": self.run_id,
            "model_slug": self.model_slug,
            "capability": self.capability,
            "scenario_id": self.scenario_id,
            "temporal_batch": self.temporal_batch,
            "seed": self.seed,
            "block_size": self.block_size,
            "block_key": self.block_key,
        }
        if include_hash:
            result["manifest_sha256"] = self.manifest_sha256
        return result

    def serialize(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    to_json = serialize

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "AssignmentManifest":
        return cls(
            run_id=value["run_id"],
            model_slug=value["model_slug"],
            capability=value["capability"],
            scenario_id=value["scenario_id"],
            temporal_batch=value["temporal_batch"],
            seed=value["seed"],
            block_size=value["block_size"],
            block_key=value.get("block_key", ""),
            manifest_sha256=value.get("manifest_sha256", ""),
        )

    @classmethod
    def from_json(cls, value: str) -> "AssignmentManifest":
        parsed = json.loads(value)
        return cls.from_dict(parsed.get("manifest", parsed))

    def persist(self, path: str | Path) -> None:
        """Persist the immutable manifest before execution."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with _commitment_lock(target):
            if target.exists():
                existing = AssignmentManifest.from_json(target.read_text(encoding="utf-8"))
                if existing != self:
                    raise ValueError("assignment manifest already exists with different contents")
                return
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_text(self.serialize() + "\n", encoding="utf-8")
            temporary.replace(target)


@dataclass(frozen=True)
class BlockAssignmentManifest:
    """One immutable balanced allocation for every planned run in a block."""

    model_slug: str
    capability: str
    scenario_id: str
    temporal_batch: str | int
    seed: str | int
    run_assignments: tuple[tuple[str, TreatmentAssignment], ...]
    block_key: str = ""
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.run_assignments:
            raise ValueError("run_assignments must not be empty")
        run_ids = [run_id for run_id, _assignment in self.run_assignments]
        if any(not isinstance(run_id, str) or not run_id for run_id in run_ids):
            raise ValueError("run IDs must be non-empty strings")
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("run IDs must be unique within a block")
        normalized = tuple(
            (run_id, TreatmentAssignment(assignment))
            for run_id, assignment in self.run_assignments
        )
        object.__setattr__(self, "run_assignments", normalized)
        derived = derive_block_key(
            self.model_slug, self.capability, self.scenario_id, self.temporal_batch
        )
        if self.block_key and self.block_key != derived:
            raise ValueError("block_key does not match the block stratum")
        object.__setattr__(self, "block_key", derived)
        computed = hashlib.sha256(
            json.dumps(
                self.to_dict(include_hash=False),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        if self.manifest_sha256 and self.manifest_sha256 != computed:
            raise ValueError("manifest_sha256 does not match the block manifest")
        object.__setattr__(self, "manifest_sha256", computed)

    @classmethod
    def create(
        cls,
        *,
        run_ids: Iterable[str],
        model_slug: str,
        capability: str,
        scenario_id: str,
        temporal_batch: str | int,
        seed: str | int,
    ) -> "BlockAssignmentManifest":
        ids = tuple(run_ids)
        assignments = [TreatmentAssignment.EXECUTE] * (len(ids) // 2)
        assignments.extend(
            [TreatmentAssignment.SUPPRESS] * (len(ids) - len(assignments))
        )
        block_key = derive_block_key(
            model_slug, capability, scenario_id, temporal_batch
        )
        _shuffle(assignments, str(seed), block_key)
        return cls(
            model_slug,
            capability,
            scenario_id,
            temporal_batch,
            seed,
            tuple(zip(ids, assignments)),
        )

    def assignment_for(self, run_id: str) -> TreatmentAssignment:
        try:
            return dict(self.run_assignments)[run_id]
        except KeyError as error:
            raise KeyError(f"run ID is not preallocated in this block: {run_id}") from error

    def schedule_for(self, run_id: str) -> "PreallocatedRunSchedule":
        return PreallocatedRunSchedule(self, run_id)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "model_slug": self.model_slug,
            "capability": self.capability,
            "scenario_id": self.scenario_id,
            "temporal_batch": self.temporal_batch,
            "seed": self.seed,
            "block_key": self.block_key,
            "run_assignments": [
                {"run_id": run_id, "assignment": assignment.label}
                for run_id, assignment in self.run_assignments
            ],
        }
        if include_hash:
            result["manifest_sha256"] = self.manifest_sha256
        return result

    def persist(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        with _commitment_lock(target):
            if target.exists() and target.read_text(encoding="utf-8").strip() != payload:
                raise ValueError("block assignment manifest already differs")
            if not target.exists():
                temporary = target.with_name(target.name + ".tmp")
                temporary.write_text(payload + "\n", encoding="utf-8")
                temporary.replace(target)


@dataclass
class PreallocatedRunSchedule:
    """Eligibility-gated view of one preallocated run assignment."""

    manifest: BlockAssignmentManifest
    run_id: str

    def __post_init__(self) -> None:
        self._assignment = self.manifest.assignment_for(self.run_id)
        self._records: list[AssignmentRecord] = []

    def reveal(
        self, block_key: Hashable, *, eligible: bool
    ) -> TreatmentAssignment | None:
        if not eligible:
            return None
        if block_key != self.manifest.block_key:
            raise KeyError(f"unknown randomization block: {block_key!r}")
        if self._records:
            raise IndexError("run assignment already consumed")
        self._records.append(
            AssignmentRecord(
                block_key=block_key,
                position=self.run_id,
                assignment=self._assignment,
                manifest_sha256=self.manifest.manifest_sha256,
                run_id=self.run_id,
            )
        )
        return self._assignment

    @property
    def consumed_count(self) -> int:
        return len(self._records)

    @property
    def records(self) -> tuple["AssignmentRecord", ...]:
        return tuple(self._records)


@dataclass(frozen=True)
class AssignmentRecord:
    block_key: Hashable
    position: int | str
    assignment: TreatmentAssignment
    manifest_sha256: str | None = None
    run_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "block_key": self.block_key,
            "position": self.position,
            "assignment": self.assignment.label,
            "manifest_sha256": self.manifest_sha256,
            "run_id": self.run_id,
        }


@contextmanager
def _commitment_lock(path: Path):
    """Lock one JSON store across threads and cooperating processes."""

    key = str(path.resolve())
    with _COMMITMENT_LOCKS_GUARD:
        process_lock = _COMMITMENT_LOCKS.setdefault(key, RLock())
    with process_lock:
        lock_path = Path(str(path) + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as handle:
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:  # pragma: no cover - retained for portable package tests.
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover - retained for portable package tests.
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class _CommitmentStore:
    """Append-only JSON state for one immutable run manifest."""

    def __init__(self, path: str | Path, manifest: AssignmentManifest, assignments: tuple[TreatmentAssignment, ...]):
        self.path = Path(path)
        self.manifest = manifest
        self.assignments = assignments
        self._initialize()

    def _read(self) -> dict[str, object]:
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if "manifest" not in value:
            value = {"manifest": value, "assignments": [], "consumed": []}
        stored = AssignmentManifest.from_dict(value["manifest"])
        if stored != self.manifest:
            raise ValueError("commitment store manifest does not match the schedule")
        if value.get("assignments") and tuple(value["assignments"]) != tuple(
            assignment.label for assignment in self.assignments
        ):
            raise ValueError("assignment commitments do not match the manifest")
        value["assignments"] = [assignment.label for assignment in self.assignments]
        if not isinstance(value.get("consumed", []), list):
            raise ValueError("commitment store consumed records must be a list")
        return value

    def _write(self, value: dict[str, object]) -> None:
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _commitment_lock(self.path):
            if self.path.exists():
                value = self._read()
            else:
                value = {
                    "manifest": self.manifest.to_dict(),
                    "assignments": [assignment.label for assignment in self.assignments],
                    "consumed": [],
                }
            self._write(value)

    def records(self) -> list[dict[str, object]]:
        with _commitment_lock(self.path):
            return list(self._read()["consumed"])

    def consume(self, *, block_key: Hashable, local_position: int, assignment: TreatmentAssignment) -> dict[str, object]:
        with _commitment_lock(self.path):
            value = self._read()
            records = value["consumed"]
            block_records = [record for record in records if record["block_key"] == block_key]
            if local_position != len(block_records):
                raise RuntimeError("commitment store position is not the next sequential position")
            if local_position >= len(self.assignments) or self.assignments[local_position] is not assignment:
                raise ValueError("assignment does not match the immutable commitment")
            record = {
                "block_key": block_key,
                "local_position": local_position,
                "position": f"{self.manifest.run_id}:{local_position}",
                "assignment": assignment.label,
                "manifest_sha256": self.manifest.manifest_sha256,
                "run_id": self.manifest.run_id,
            }
            records.append(record)
            self._write(value)
            return record


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
    _manifest: AssignmentManifest | None = None
    _records: list[AssignmentRecord] | None = None
    _commitment_store: _CommitmentStore | None = None

    def __post_init__(self) -> None:
        self._lock = RLock()
        if self._records is None:
            self._records = []
        if self._commitment_store is not None:
            persisted = self._commitment_store.records()
            self._records = [
                AssignmentRecord(
                    block_key=record["block_key"],
                    position=record["position"],
                    assignment=TreatmentAssignment[record["assignment"]],
                    manifest_sha256=record["manifest_sha256"],
                    run_id=record["run_id"],
                )
                for record in persisted
            ]
            for block_key in self._positions:
                block_records = [record for record in persisted if record["block_key"] == block_key]
                self._positions[block_key] = len(block_records)

    @classmethod
    def generate(
        cls,
        seed: str | int,
        blocks: Mapping[Hashable, int],
        *,
        manifest: AssignmentManifest | None = None,
        commitment_path: str | Path | None = None,
    ) -> "AssignmentSchedule":
        if not blocks:
            raise ValueError("at least one randomization block is required")
        values: dict[Hashable, tuple[TreatmentAssignment, ...]] = {}
        for block_key, size in blocks.items():
            if type(size) is not int or size < 1:
                raise ValueError("block sizes must be positive integers")
            execute_count = size // 2
            block = [TreatmentAssignment.EXECUTE] * execute_count
            block.extend([TreatmentAssignment.SUPPRESS] * (size - execute_count))
            shuffle_key: Hashable = (block_key, manifest.run_id) if manifest is not None else block_key
            _shuffle(block, str(seed), shuffle_key)
            values[block_key] = tuple(block)
        if manifest is not None:
            if manifest.seed != seed:
                raise ValueError("schedule seed does not match assignment manifest")
            if blocks.get(manifest.block_key) != manifest.block_size:
                raise ValueError("schedule blocks do not match assignment manifest")
        store = None
        if commitment_path is not None:
            if manifest is None:
                raise ValueError("commitment_path requires an assignment manifest")
            store = _CommitmentStore(commitment_path, manifest, values[manifest.block_key])
        return cls(values, {key: 0 for key in values}, manifest, None, store)

    def reveal(self, block_key: Hashable, *, eligible: bool) -> TreatmentAssignment | None:
        """Reveal and consume exactly one value only for an eligible proposal."""
        with self._lock:
            if not eligible:
                return None
            if block_key not in self._values:
                raise KeyError(f"unknown randomization block: {block_key!r}")
            position = self._positions[block_key]
            values = self._values[block_key]
            if position >= len(values):
                raise IndexError(f"randomization block exhausted: {block_key!r}")
            assignment = values[position]
            if self._commitment_store is not None:
                persisted = self._commitment_store.consume(
                    block_key=block_key, local_position=position, assignment=assignment
                )
                record = AssignmentRecord(
                    block_key=persisted["block_key"],
                    position=persisted["position"],
                    assignment=TreatmentAssignment[persisted["assignment"]],
                    manifest_sha256=persisted["manifest_sha256"],
                    run_id=persisted["run_id"],
                )
            else:
                record = AssignmentRecord(
                    block_key,
                    f"{self._manifest.run_id}:{position}" if self._manifest else position,
                    assignment,
                    self._manifest.manifest_sha256 if self._manifest else None,
                    self._manifest.run_id if self._manifest else None,
                )
            self._positions[block_key] = position + 1
            self._records.append(record)
            return assignment

    @classmethod
    def from_manifest(
        cls, manifest: AssignmentManifest, *, commitment_path: str | Path | None = None
    ) -> "AssignmentSchedule":
        return cls.generate(
            manifest.seed,
            {manifest.block_key: manifest.block_size},
            manifest=manifest,
            commitment_path=commitment_path,
        )

    @property
    def consumed_count(self) -> int:
        return sum(self._positions.values())

    def consumed_count_for(self, block_key: Hashable) -> int:
        return self._positions[block_key]

    @property
    def records(self) -> tuple[AssignmentRecord, ...]:
        with self._lock:
            return tuple(self._records or ())

    @property
    def manifest(self) -> AssignmentManifest | None:
        return self._manifest


def generate_balanced_schedule(
    seed: str | int,
    block_keys: Iterable[Hashable] | Mapping[Hashable, int],
    block_size: int = 2,
    *,
    manifest: AssignmentManifest | None = None,
    commitment_path: str | Path | None = None,
) -> AssignmentSchedule:
    """Build a schedule; mapping input permits a distinct size per block."""
    if isinstance(block_keys, Mapping):
        blocks = dict(block_keys)
    else:
        keys = list(block_keys)
        blocks = {key: block_size for key in keys}
    return AssignmentSchedule.generate(
        seed, blocks, manifest=manifest, commitment_path=commitment_path
    )
