"""Task protocol data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    dataset: str
    setting: str
    source_domains: tuple[str, ...]
    target_domains: tuple[str, ...]
    source_split: str
    target_split: str
    eval_split: str
    class_space: str = "shared"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_domains:
            raise ValueError("TaskSpec requires at least one source domain.")
        if not self.target_domains:
            raise ValueError("TaskSpec requires at least one target domain.")
        overlap = set(self.source_domains).intersection(self.target_domains)
        if overlap:
            raise ValueError(f"Source and target domains must be disjoint: {sorted(overlap)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkSuite:
    suite_id: str
    name: str
    tasks: tuple[TaskSpec, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskData:
    task: TaskSpec
    source_train: Any
    target_train: Any
    target_eval: Any
