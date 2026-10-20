"""CADENCE command-line entry. Scaffolding commands only."""

from __future__ import annotations

import argparse
import sys


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="cadence", description="CADENCE CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-setup", help="One tiny live call per configured model role")
    sm = sub.add_parser("smoke", help="Smoke test: generation + judge calibration + cascade on real tasks")
    sm.add_argument("--tasks", nargs="+", default=["sst2", "gsm8k"], help="task names (default: sst2 gsm8k)")
    sm.add_argument("--n", type=int, default=10, help="examples per task")
    sm.add_argument("--trials", type=int, default=2, help="judge trials per item (>=2 for ICC)")
    sm.add_argument("--max-calls", type=int, default=600, help="hard cap on live LLM calls for this run")
    sm.add_argument("--mock", action="store_true", help="offline MockClient (plumbing check only)")
    sm.add_argument("--out", default=None, help="JSON report path (default runs/smoke-<time>.json)")
    rn = sub.add_parser("run", help="Run the optimisation loop and print the Pareto menu")
    rn.add_argument("--stream", default="text", choices=["text", "code_web", "image", "audio"])
    rn.add_argument("--task", default="toy", help="task name (default: toy)")
    rn.add_argument("--budget-usd", type=float, default=None, help="override max_usd_per_run for this run")
    rn.add_argument("--max-calls", type=int, default=None, help="override max_calls_per_run for this run")
    rn.add_argument("--max-gen", type=int, default=None)
    rn.add_argument("--children", type=int, default=None, help="children per generation")
    rn.add_argument("--minibatch", type=int, default=None)
    rn.add_argument("--workers", type=int, default=None)
    rn.add_argument("--seed", type=int, default=None)
    rn.add_argument("--run-id", default=None, help="run directory name under runs/ (default <task>-<time>)")
    rn.add_argument("--resume", action="store_true", help="continue the run in --run-id")
    rn.add_argument("--mock", action="store_true", help="offline MockClient (toy task gets a scripted reflector)")
    rn.add_argument("--no-mlflow", action="store_true", help="skip MLflow tracking")
    return parser.parse_args(argv)


def _run_optimize(args: argparse.Namespace) -> int:
    import time

    from cadence.cascade import Cascade
    from cadence.config import ROOT, load_budget_config, load_cascade_config, load_run_config
    from cadence.llm import BudgetExceeded, BudgetGuard, LLMClient, MockClient
    from cadence.orchestrator import LoopSettings, optimize, pareto_menu
    from cadence.orchestrator.mock_reflector import toy_reflector
    from cadence.tasks import get_task
    from cadence.tracking import MlflowTracker, NullTracker

    task = get_task(args.task)
    if task.stream != args.stream:
        print(f"FAIL: task {args.task!r} belongs to stream {task.stream!r}, not {args.stream!r}")
        return 1
    try:
        task.load("opt", n=1)
    except FileNotFoundError as exc:
        print(f"FAIL: {exc}")
        return 1
    rc, cc, b = load_run_config(), load_cascade_config(), load_budget_config()
    budget = BudgetGuard(args.budget_usd or b.max_usd_per_run, args.max_calls or b.max_calls_per_run,
                         b.max_tokens_per_run)
    settings = LoopSettings(
        seed=rc.seed if args.seed is None else args.seed, max_gen=args.max_gen if args.max_gen is not None else rc.max_gen,
        children_per_gen=args.children or rc.children_per_gen, minibatch=args.minibatch or rc.minibatch,
        max_archive=rc.max_archive, final_sample=rc.final_sample, workers=args.workers or rc.workers,
        parents_per_gen=rc.parents_per_gen)
    if args.mock:
        client: object = MockClient(script={"reflector": toy_reflector} if args.task == "toy" else None, budget=budget)
    else:
        client = LLMClient.from_config(budget=budget, seed=settings.seed)
    run_id = args.run_id or f"{args.task}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = ROOT / "runs" / run_id
    tracker = NullTracker() if args.no_mlflow else MlflowTracker(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    print(f"Optimising task={args.task} ({'MOCK' if args.mock else 'LIVE'}) run={run_id} "
          f"gens={settings.max_gen} children={settings.children_per_gen} minibatch={settings.minibatch}", flush=True)
    try:
        res = optimize(task, client, Cascade.from_config(client, cc), settings, run_dir, tracker=tracker,
                       resume=args.resume, log=lambda m: print(m, flush=True))
    except (RuntimeError, FileExistsError, ValueError, BudgetExceeded) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"\nstopped: {res.stopped_reason} after {res.generations_done} generation(s); "
          f"final re-evaluation: {'yes' if res.final_reevaluated else 'no'}")
    print("\nPareto menu (best quality first):\n")
    print(pareto_menu(res.archive))
    print(f"\nrun directory: {run_dir.relative_to(ROOT)}")
    return 0


def _run_smoke(args: argparse.Namespace) -> int:
    import json
    import time

    from cadence.config import ROOT, load_budget_config, load_cascade_config
    from cadence.llm import BudgetGuard, LLMClient, MockClient
    from cadence.smoke import format_report, run_smoke
    from cadence.tasks import get_task

    tasks = {}
    for name in args.tasks:
        task = get_task(name)
        try:
            task.load("calib", n=1)
        except FileNotFoundError as exc:
            print(f"FAIL: {exc}")
            return 1
        tasks[name] = task
    cfg = load_cascade_config()
    if args.mock:
        client: object = MockClient()
    else:
        b = load_budget_config()
        client = LLMClient.from_config(budget=BudgetGuard(b.max_usd_per_run, args.max_calls, b.max_tokens_per_run))
    print(f"Running smoke test on {', '.join(tasks)} (n={args.n}, trials={args.trials}, "
          f"{'MOCK' if args.mock else 'LIVE'}). Slow calls are normal; progress prints as it goes.", flush=True)
    rep = run_smoke(client, tasks, n=args.n, trials=args.trials, cascade_cfg=cfg, log=lambda m: print(m, flush=True))
    print(format_report(rep, cfg.kappa_min))
    out = ROOT / (args.out or f"runs/smoke-{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(f"\nreport saved to {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 1 if rep["errors"] else 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "check-setup":
        from cadence.check_setup import run_check_setup

        return run_check_setup()
    if args.command == "smoke":
        return _run_smoke(args)
    if args.command == "run":
        return _run_optimize(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
