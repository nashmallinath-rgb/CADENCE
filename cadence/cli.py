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
    return parser.parse_args(argv)


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
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
