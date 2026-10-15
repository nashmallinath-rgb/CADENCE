from __future__ import annotations

from collections.abc import Callable

from cadence.tasks.base import Task
from cadence.tasks.text import Gsm8kTask, HotpotQATask, IfbenchTask, Sst2Task
from cadence.tasks.toy import ToyTask

TASKS: dict[str, Callable[[], Task]] = {
    "toy": ToyTask,
    "gsm8k": Gsm8kTask,
    "hotpotqa": HotpotQATask,
    "sst2": Sst2Task,
    "ifbench": IfbenchTask,
}


def get_task(name: str) -> Task:
    try:
        return TASKS[name]()
    except KeyError:
        raise KeyError(f"unknown task {name!r}; available: {sorted(TASKS)}") from None
