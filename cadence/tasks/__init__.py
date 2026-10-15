"""Task interfaces and modality streams."""
from cadence.tasks.base import Split, Task
from cadence.tasks.registry import TASKS, get_task
from cadence.tasks.splits import assign_split, select

__all__ = ["Task", "Split", "TASKS", "get_task", "assign_split", "select"]
