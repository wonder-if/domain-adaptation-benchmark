"""Runtime experiment recorder for benchmark research projects."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from dabench.results._common import iso_timestamp_now
from dabench.results.run import clone_run_record, make_run_record, validate_run_record, write_run_record


class ExperimentRecorder:
    """Record evaluation events and export one canonical run-record JSON."""

    def __init__(
        self,
        *,
        dataset: str,
        setting: str,
        method: str,
        seed: int,
        source_domain: str | None = None,
        target_domain: str | None = None,
        backbone: str | None = None,
        algorithm_name: str | None = None,
        output_dir: str | None = None,
        records_root: str | Path | None = None,
        run_id: str | None = None,
        selected_checkpoint: str = "last",
        selection_metric: str = "accuracy",
        selection_mode: str = "max",
        config: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.records_root = Path(records_root) if records_root is not None else None
        if self.records_root is None and output_dir is not None:
            self.records_root = Path(output_dir) / "records"
        self._record = make_run_record(
            dataset=dataset,
            setting=setting,
            method=method,
            seed=seed,
            source_domain=source_domain,
            target_domain=target_domain,
            backbone=backbone,
            algorithm_name=algorithm_name,
            output_dir=output_dir,
            run_id=run_id,
            status="running",
            selected_checkpoint=selected_checkpoint,
            selection_metric=selection_metric,
            selection_mode=selection_mode,
            start_time=iso_timestamp_now(),
            config=config,
            extra=extra,
        )

    @property
    def record(self) -> dict[str, Any]:
        """Return a copy of the current in-memory run record."""

        return clone_run_record(self._record)

    def log_eval(
        self,
        *,
        step: int | str,
        split: str,
        metrics: dict[str, float],
        class_metrics: dict[str, float] | None = None,
        checkpoint: str | None = None,
        is_best: bool = False,
        extra: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> None:
        """Append one evaluation event to the run history."""

        event = {
            "step": step,
            "split": split,
            "metrics": dict(metrics),
            "timestamp": timestamp or iso_timestamp_now(),
        }
        if checkpoint is not None:
            event["checkpoint"] = checkpoint
        if class_metrics is not None:
            event["class_metrics"] = dict(class_metrics)
        if extra:
            event["extra"] = dict(extra)
        validate_run_record(
            {
                **self._record,
                "eval_history": [*self._record["eval_history"], event],
            }
        )
        self._record["eval_history"].append(event)

        if is_best:
            self._set_best_from_event(event)
            return
        self._maybe_update_best_from_event(event)

    def mark_best(
        self,
        *,
        step: int | str | None = None,
        split: str | None = None,
        by: str | None = None,
    ) -> None:
        """Mark one existing evaluation event as the best checkpoint result."""

        if by is not None:
            self._record["selection_metric"] = by
        history = self._record["eval_history"]
        if not history:
            raise ValueError("Cannot mark best before any evaluation event has been logged.")
        candidate = None
        for event in reversed(history):
            if step is not None and event.get("step") != step:
                continue
            if split is not None and event.get("split") != split:
                continue
            candidate = event
            break
        if candidate is None:
            raise ValueError("No evaluation event matched the requested step/split.")
        self._set_best_from_event(candidate)

    def finalize(
        self,
        *,
        status: str = "completed",
        final_metrics: dict[str, float] | None = None,
        final_class_metrics: dict[str, float] | None = None,
        step: int | str | None = None,
        split: str | None = None,
        failure_reason: str | None = None,
        path: str | Path | None = None,
    ) -> Path:
        """Freeze the run record and write it to disk."""

        self._record["status"] = status
        self._record["end_time"] = iso_timestamp_now()
        if failure_reason is not None:
            self._record["failure_reason"] = failure_reason
        if status == "completed":
            if final_metrics is not None:
                self._record["final_metrics"] = dict(final_metrics)
                self._record["final_class_metrics"] = dict(final_class_metrics or {})
            else:
                final_event = self._select_final_event(step=step, split=split)
                self._set_final_from_event(final_event)
            if not self._record["best_metrics"] and self._record["final_metrics"]:
                self._record["best_metrics"] = dict(self._record["final_metrics"])
                self._record["best_class_metrics"] = dict(self._record["final_class_metrics"])
                self._record["best_step"] = self._record.get("final_step")
                self._record["best_split"] = self._record.get("final_split")
                if "final_checkpoint" in self._record:
                    self._record["best_checkpoint"] = self._record["final_checkpoint"]
        elif self._record["eval_history"] and not self._record["final_metrics"]:
            self._set_final_from_event(self._record["eval_history"][-1])

        validated = validate_run_record(self._record)
        target_path = path
        if target_path is None and self.records_root is None:
            raise ValueError("ExperimentRecorder.finalize requires either path or records_root/output_dir.")
        return write_run_record(validated, path=target_path, records_root=self.records_root)

    def _maybe_update_best_from_event(self, event: dict[str, Any]) -> None:
        metric_name = str(self._record.get("selection_metric", ""))
        if event.get("split") != "val" or metric_name not in event["metrics"]:
            return
        candidate = float(event["metrics"][metric_name])
        current = self._record["best_metrics"].get(metric_name) if self._record["best_metrics"] else None
        if current is None:
            self._set_best_from_event(event)
            return
        selection_mode = self._record.get("selection_mode", "max")
        is_better = candidate > current if selection_mode == "max" else candidate < current
        if is_better:
            self._set_best_from_event(event)

    def _set_best_from_event(self, event: dict[str, Any]) -> None:
        self._record["best_metrics"] = dict(event["metrics"])
        self._record["best_class_metrics"] = dict(event.get("class_metrics", {}))
        self._record["best_step"] = event.get("step")
        self._record["best_split"] = event.get("split")
        if "checkpoint" in event:
            self._record["best_checkpoint"] = event["checkpoint"]

    def _set_final_from_event(self, event: dict[str, Any]) -> None:
        self._record["final_metrics"] = dict(event["metrics"])
        self._record["final_class_metrics"] = dict(event.get("class_metrics", {}))
        self._record["final_step"] = event.get("step")
        self._record["final_split"] = event.get("split")
        if "checkpoint" in event:
            self._record["final_checkpoint"] = event["checkpoint"]

    def _select_final_event(self, *, step: int | str | None, split: str | None) -> dict[str, Any]:
        history = self._record["eval_history"]
        if not history:
            raise ValueError("Cannot finalize a completed run without final metrics or eval_history.")
        requested_split = split or "test"
        for event in reversed(history):
            if step is not None and event.get("step") != step:
                continue
            if requested_split is not None and event.get("split") != requested_split:
                continue
            return deepcopy(event)
        if split is not None or step is not None:
            for event in reversed(history):
                if step is not None and event.get("step") != step:
                    continue
                return deepcopy(event)
        return deepcopy(history[-1])
