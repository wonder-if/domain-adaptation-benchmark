"""Unsupervised domain adaptation task helpers."""

from __future__ import annotations

from itertools import permutations
from typing import Any, Mapping, Sequence

from dabench.data import load_view
from dabench.task.base import BenchmarkSuite, TaskData, TaskSpec


def _domain_slug(domains: Sequence[str]) -> str:
    return "-".join(domain.lower().replace(" ", "-") for domain in domains)


def make_pairwise_uda_suite(
    *,
    dataset: str,
    suite_id: str,
    name: str,
    domains: Sequence[str],
    source_split: str,
    target_split: str,
    eval_split: str,
    metadata: Mapping[str, Any] | None = None,
) -> BenchmarkSuite:
    tasks = tuple(
        TaskSpec(
            task_id=f"{suite_id}/{_domain_slug((source,))}_to_{_domain_slug((target,))}",
            dataset=dataset,
            setting="closed_set_uda",
            source_domains=(source,),
            target_domains=(target,),
            source_split=source_split,
            target_split=target_split,
            eval_split=eval_split,
            class_space="shared",
            metadata={"suite_id": suite_id},
        )
        for source, target in permutations(domains, 2)
    )
    return BenchmarkSuite(
        suite_id=suite_id,
        name=name,
        tasks=tasks,
        metadata=metadata or {},
    )


def load_task(
    task: TaskSpec,
    *,
    path,
    format: str = "hf",
    transform=None,
    train_transform=None,
    val_transform=None,
    decode: bool = True,
) -> TaskData:
    if len(task.source_domains) != 1 or len(task.target_domains) != 1:
        raise ValueError("load_task currently expects one source domain and one target domain.")
    source_transform = train_transform if train_transform is not None else transform
    target_transform = train_transform if train_transform is not None else transform
    eval_transform = val_transform if val_transform is not None else transform
    return TaskData(
        task=task,
        source_train=load_view(
            task.dataset,
            path=path,
            domain=task.source_domains[0],
            split=task.source_split,
            format=format,
            transform=source_transform,
            decode=decode,
        ),
        target_train=load_view(
            task.dataset,
            path=path,
            domain=task.target_domains[0],
            split=task.target_split,
            format=format,
            transform=target_transform,
            decode=decode,
        ),
        target_eval=load_view(
            task.dataset,
            path=path,
            domain=task.target_domains[0],
            split=task.eval_split,
            format=format,
            transform=eval_transform,
            decode=decode,
        ),
    )
