"""Experiment tracking. `Tracker` is the tiny interface the loop uses; MLflow is one implementation.

MLflow is imported lazily (it is slow to import and tests should not need it). Default store is a local SQLite file
(`sqlite:///mlflow.db`, gitignored).
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

DEFAULT_URI = "sqlite:///mlflow.db"


class Tracker(Protocol):
    def start(self, run_name: str, params: Mapping[str, Any]) -> None: ...
    def log_generation(self, step: int, metrics: Mapping[str, float | None]) -> None: ...
    def log_artifact(self, path: Path) -> None: ...
    def end(self, status: str = "FINISHED") -> None: ...


class NullTracker:
    """Does nothing. Used by tests and `--no-mlflow`."""

    def start(self, run_name: str, params: Mapping[str, Any]) -> None: ...
    def log_generation(self, step: int, metrics: Mapping[str, float | None]) -> None: ...
    def log_artifact(self, path: Path) -> None: ...
    def end(self, status: str = "FINISHED") -> None: ...


class MlflowTracker:
    def __init__(self, tracking_uri: str = DEFAULT_URI, experiment: str = "cadence") -> None:
        self.tracking_uri, self.experiment = tracking_uri, experiment
        self._mlflow: Any = None
        self._active = False

    def start(self, run_name: str, params: Mapping[str, Any]) -> None:
        import mlflow

        self._mlflow = mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment)
        mlflow.start_run(run_name=run_name)
        self._active = True
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})

    def log_generation(self, step: int, metrics: Mapping[str, float | None]) -> None:
        if self._active:
            clean = {k: float(v) for k, v in metrics.items() if v is not None}
            self._mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: Path) -> None:
        if self._active and Path(path).exists():
            self._mlflow.log_artifact(str(path))

    def end(self, status: str = "FINISHED") -> None:
        if self._active:
            self._mlflow.end_run(status=status)
            self._active = False
