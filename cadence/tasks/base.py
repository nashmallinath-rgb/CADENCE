"""Base Task interface. No evaluation logic yet."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Task(ABC):
    """One evaluation task in a CADENCE modality stream."""

    name: str

    @abstractmethod
    def evaluate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Run the task. Implementation deferred."""
        raise NotImplementedError
