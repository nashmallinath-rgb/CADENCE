from __future__ import annotations

from collections.abc import Callable

from cadence.tasks.audio import AudioTask
from cadence.tasks.base import Task
from cadence.tasks.code_web import CodeWebTask
from cadence.tasks.image import ImageTask
from cadence.tasks.text import Gsm8kTask, HotpotQATask, IfbenchTask, Sst2Task
from cadence.tasks.toy import ToyTask

TASKS: dict[str, Callable[..., Task]] = {
    "toy": ToyTask,
    "gsm8k": Gsm8kTask,
    "hotpotqa": HotpotQATask,
    "sst2": Sst2Task,
    "ifbench": IfbenchTask,
    "image": ImageTask,
    "audio": AudioTask,
    "code_web": CodeWebTask,
}


def get_task(name: str, **kwargs: object) -> Task:
    """Build a task by name. kwargs go to its constructor (e.g. media_dir for image/audio)."""
    try:
        factory = TASKS[name]
    except KeyError:
        raise KeyError(f"unknown task {name!r}; available: {sorted(TASKS)}") from None
    return factory(**kwargs)

