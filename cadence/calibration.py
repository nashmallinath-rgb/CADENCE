"""`calibrate` command logic: judge health against exact-match ground truth for one task (reuses the smoke machinery)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cadence.config import CascadeConfig
from cadence.smoke import run_smoke
from cadence.tasks.base import Task

USED_BY_CASCADE = ("cheap_judge", "strong_judge")


def supports_calibration(task: Task) -> bool:
    try:
        ex = task.load("calib", n=1)
    except FileNotFoundError:
        return False
    return bool(ex) and task.calibration_rubric(ex[0]) is not None


def run_calibration(client: Any, task: Task, *, n: int = 12, trials: int = 2, cfg: CascadeConfig | None = None,
                    log: Callable[[str], None] = print) -> dict[str, Any]:
    cfg = cfg or CascadeConfig()
    if not supports_calibration(task):
        return {"supported": False, "task": task.name, "kappa_min": cfg.kappa_min, "flagged": [],
                "reason": "no ground truth for this task (no exact-match metric / calibration rubric), "
                          "so judge kappa cannot be measured"}
    rep = run_smoke(client, {task.name: task}, n=n, trials=trials, cascade_cfg=cfg, log=log)
    t = rep["tasks"].get(task.name, {})
    judges = t.get("judges", {})
    kappas = {r: j["kappa"] for r, j in judges.items() if "kappa" in j}
    used = [kappas[r] for r in USED_BY_CASCADE if r in kappas]
    return {"supported": True, "task": task.name, "n": n, "trials": trials, "kappa_min": cfg.kappa_min,
            "judges": judges, "kappa_cascade_min": min(used) if used else None,
            "flagged": sorted(r for r, j in judges.items() if j.get("flagged")),
            "inter_judge_kappa": t.get("inter_judge_kappa"), "cascade": t.get("cascade"),
            "calibration_items": t.get("calibration_items"), "calibration_correct": t.get("calibration_correct"),
            "errors": rep["errors"], "spend": rep["spend"],
            "limits": ["small calibration set", "corrupted items are easy cases",
                       "bias probes (position, verbosity) are not run by this command",
                       "a judge that is consistently wrong still passes the confidence-based cascade"]}
