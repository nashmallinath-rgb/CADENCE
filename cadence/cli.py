"""CADENCE command-line entry."""

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
    rn.add_argument("--seed-prompt", default="default", help="starting prompt: 'default' (the task's), 'weak' (a deliberately poor one, "
                    "so the run shows real improvement), or literal text")
    rn.add_argument("--final-sample", type=int, default=None, help="examples for the final re-evaluation (default run.yaml; lower it on slow free tiers)")
    rn.add_argument("--seed", type=int, default=None)
    rn.add_argument("--run-id", default=None, help="run directory name under runs/ (default <task>-<time>)")
    rn.add_argument("--resume", action="store_true", help="continue the run in --run-id")
    rn.add_argument("--mock", action="store_true", help="offline MockClient (toy task gets a scripted reflector)")
    rn.add_argument("--no-mlflow", action="store_true", help="skip MLflow tracking")
    rn.add_argument("--calibrate", action="store_true",
                    help="check judge health vs ground truth first (tasks with an exact-match metric only)")
    rn.add_argument("--no-report", action="store_true", help="do not write report.md and plots at the end")
    ca = sub.add_parser("calibrate", help="Judge health (kappa vs exact-match ground truth) for one task")
    ca.add_argument("--task", required=True)
    ca.add_argument("--n", type=int, default=12, help="calibration examples")
    ca.add_argument("--trials", type=int, default=2)
    ca.add_argument("--max-calls", type=int, default=400)
    ca.add_argument("--run-id", default=None, help="write runs/<run-id>/calibration.json (default calibration-<task>)")
    ca.add_argument("--mock", action="store_true")
    bm = sub.add_parser("benchmark", help="Held-out evaluation + dual-judge check of a finished run")
    bm.add_argument("--run", required=True, help="run id under runs/")
    bm.add_argument("--n", type=int, default=24, help="held-out examples per archive member")
    bm.add_argument("--trials", type=int, default=3, help="judge trials for the second-family re-score")
    bm.add_argument("--budget-usd", type=float, default=None)
    bm.add_argument("--max-calls", type=int, default=None)
    bm.add_argument("--workers", type=int, default=None)
    bm.add_argument("--no-dual", action="store_true", help="skip the second-family judge")
    bm.add_argument("--no-plots", action="store_true")
    bm.add_argument("--no-baseline", action="store_true",
                    help="do not also evaluate the run's starting (seed) prompt on the held-out examples")
    bm.add_argument("--mock", action="store_true")
    rp = sub.add_parser("report", help="(Re)write report.md for a run from the files in its directory")
    rp.add_argument("--run", required=True)
    rp.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def _client_for(mock: bool, task_name: str, budget_usd: float | None, max_calls: int | None, seed: int) -> tuple[object, object]:
    """(client, budget) shared by the calibrate and benchmark commands."""
    from cadence.config import load_budget_config
    from cadence.llm import BudgetGuard, LLMClient, MockClient
    from cadence.orchestrator.mock_reflector import toy_reflector

    b = load_budget_config()
    budget = BudgetGuard(budget_usd or b.max_usd_per_run, max_calls or b.max_calls_per_run, b.max_tokens_per_run)
    if mock:
        return MockClient(script={"reflector": toy_reflector} if task_name == "toy" else None, budget=budget,
                          unit_prices={"images": 0.04, "audio_seconds": 0.0004}), budget
    return LLMClient.from_config(budget=budget, seed=seed), budget


def _judge_roles(task: object) -> tuple[str, str]:
    return ("vision_judge", "vision_judge_strong") if getattr(task, "stream", "") == "image" else ("cheap_judge", "strong_judge")


def _run_calibrate(args: argparse.Namespace) -> int:
    import json

    from cadence.calibration import run_calibration
    from cadence.config import ROOT, load_cascade_config
    from cadence.tasks import get_task

    task = get_task(args.task)
    cfg = load_cascade_config()
    client, _ = _client_for(args.mock, args.task, None, args.max_calls, 0)
    rep = run_calibration(client, task, n=args.n, trials=args.trials, cfg=cfg, log=lambda m: print(m, flush=True))
    out_dir = ROOT / "runs" / (args.run_id or f"calibration-{args.task}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    if not rep["supported"]:
        print(f"Calibration not possible for {args.task!r}: {rep['reason']}")
    else:
        for role, j in rep["judges"].items():
            print(f"  {role}: kappa={j.get('kappa')} {'FLAGGED' if j.get('flagged') else 'ok'}")
        if rep["flagged"]:
            print(f"WARNING: judges below kappa {cfg.kappa_min}: {', '.join(rep['flagged'])}. Treat judged scores with caution.")
    print(f"saved {(out_dir / 'calibration.json').relative_to(ROOT)}")
    return 1 if rep.get("errors") else 0


def _run_benchmark(args: argparse.Namespace) -> int:
    import json

    from cadence.benchmark import evaluate_heldout, run_dual_judge, write_dual_judge, write_heldout, write_report
    from cadence.cascade import Cascade
    from cadence.config import ROOT, load_cascade_config, load_run_config
    from cadence.schemas import Evaluated
    from cadence.tasks import get_task

    run_dir = ROOT / "runs" / args.run
    cfg_p, arch_p = run_dir / "config.json", run_dir / "archive.json"
    if not cfg_p.exists() or not arch_p.exists():
        print(f"FAIL: {run_dir} is not a finished run (needs config.json and archive.json). Run `run` first.")
        return 1
    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    members = [Evaluated.model_validate(d) for d in json.loads(arch_p.read_text(encoding="utf-8"))]
    if not members:
        print("FAIL: the run's archive is empty")
        return 1
    seed = int(cfg["settings"].get("seed", 0))
    name = cfg["task"]
    media = name in ("image", "audio")
    task = get_task(name, media_dir=run_dir / "media") if media else get_task(name)
    cc, rc = load_cascade_config(), load_run_config()
    client, _ = _client_for(args.mock, name, args.budget_usd, args.max_calls, seed)
    cheap, strong = _judge_roles(task)
    cascade = Cascade.from_config(client, cc, cheap, strong)
    log = lambda m: print(m, flush=True)  # noqa: E731
    try:
        held = evaluate_heldout(task, client, cascade, members, n=args.n, seed=seed,
                                workers=args.workers or rc.workers, log=log,
                                baseline_template=None if args.no_baseline else cfg.get("seed_template"))
    except (ValueError, FileNotFoundError) as exc:
        print(f"FAIL: {exc}")
        return 1
    write_heldout(run_dir, held)
    if args.no_dual:
        print("dual-judge check skipped (--no-dual)")
    elif task.stream == "image":
        print("dual-judge check skipped for images (the second-family judge cannot see images)")
    else:
        dj = run_dual_judge(client, held.rows, trials=args.trials, kappa_min=cc.kappa_min, seed=seed, log=log)
        write_dual_judge(run_dir, dj)
        if dj.skipped:
            print(f"dual-judge check skipped: {dj.reason}")
    path = write_report(run_dir, with_plots=not args.no_plots)
    print(f"held-out rows: {len(held.rows)}{' (PARTIAL: ' + str(held.stop_reason) + ')' if held.partial else ''}")
    print(f"report: {path.relative_to(ROOT)}")
    return 0


def _run_report(args: argparse.Namespace) -> int:
    from cadence.benchmark import write_report
    from cadence.config import ROOT

    run_dir = ROOT / "runs" / args.run
    if not run_dir.is_dir():
        print(f"FAIL: no run directory {run_dir}")
        return 1
    print(f"report: {write_report(run_dir, with_plots=not args.no_plots).relative_to(ROOT)}")
    return 0


def _run_optimize(args: argparse.Namespace) -> int:
    import time

    from cadence.cascade import Cascade
    from cadence.config import ROOT, load_budget_config, load_cascade_config, load_run_config
    from cadence.llm import BudgetExceeded, BudgetGuard, LLMClient, MockClient
    from cadence.orchestrator import LoopSettings, optimize, pareto_menu
    from cadence.orchestrator.mock_reflector import toy_reflector
    from cadence.tasks import get_task
    from cadence.tracking import MlflowTracker, NullTracker

    from cadence.config import ROOT as _ROOT

    run_id = args.run_id or f"{args.task}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = _ROOT / "runs" / run_id
    media = args.task in ("image", "audio")
    task = get_task(args.task, media_dir=run_dir / "media") if media else get_task(args.task)
    if task.stream != args.stream:
        print(f"FAIL: task {args.task!r} belongs to stream {task.stream!r}, not {args.stream!r}")
        return 1
    seed_template: str | None = None
    if args.seed_prompt == "weak":
        seed_template = task.weak_seed_prompt()
        if seed_template is None:
            print(f"FAIL: task {args.task!r} defines no weak seed prompt; pass literal text with --seed-prompt instead")
            return 1
    elif args.seed_prompt != "default":
        seed_template = args.seed_prompt
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
        max_archive=rc.max_archive, final_sample=args.final_sample or rc.final_sample, workers=args.workers or rc.workers,
        parents_per_gen=rc.parents_per_gen)
    if args.mock:
        client: object = MockClient(script={"reflector": toy_reflector} if args.task == "toy" else None, budget=budget,
                                    unit_prices={"images": 0.04, "audio_seconds": 0.0004})
    else:
        client = LLMClient.from_config(budget=budget, seed=settings.seed)
    tracker = NullTracker() if args.no_mlflow else MlflowTracker(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    print(f"Optimising task={args.task} ({'MOCK' if args.mock else 'LIVE'}) run={run_id} "
          f"gens={settings.max_gen} children={settings.children_per_gen} minibatch={settings.minibatch}", flush=True)
    kappa = None
    if args.calibrate:
        import json

        from cadence.calibration import run_calibration

        cal = run_calibration(client, task, cfg=cc, log=lambda m: print(m, flush=True))
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "calibration.json").write_text(json.dumps(cal, indent=2, default=str), encoding="utf-8")
        if not cal["supported"]:
            print(f"calibration skipped: {cal['reason']}")
        else:
            kappa = cal.get("kappa_cascade_min")
            if cal["flagged"]:
                print(f"WARNING: judge(s) {', '.join(cal['flagged'])} below kappa {cc.kappa_min}; judged scores are suspect.")
    try:
        cheap, strong = ("vision_judge", "vision_judge_strong") if task.stream == "image" else ("cheap_judge", "strong_judge")
        res = optimize(task, client, Cascade.from_config(client, cc, cheap, strong), settings, run_dir, tracker=tracker,
                       kappa=kappa, resume=args.resume, seed_template=seed_template, log=lambda m: print(m, flush=True))
    except (RuntimeError, FileExistsError, ValueError, BudgetExceeded) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"\nstopped: {res.stopped_reason} after {res.generations_done} generation(s); "
          f"final re-evaluation: {'yes' if res.final_reevaluated else 'no'}")
    print("\nPareto menu (best quality first):\n")
    print(pareto_menu(res.archive))
    print(f"\nrun directory: {run_dir.relative_to(ROOT)}")
    if not args.no_report:
        from cadence.benchmark import write_report

        print(f"report: {write_report(run_dir).relative_to(ROOT)}  (add held-out numbers with: benchmark --run {run_id})")
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
    if args.command == "calibrate":
        return _run_calibrate(args)
    if args.command == "benchmark":
        return _run_benchmark(args)
    if args.command == "report":
        return _run_report(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
