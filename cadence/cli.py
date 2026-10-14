"""CADENCE command-line entry. Scaffolding commands only."""

from __future__ import annotations

import argparse
import sys


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="cadence", description="CADENCE CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-setup", help="Smoke-test Gemini and xAI via LiteLLM")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "check-setup":
        from cadence.check_setup import run_check_setup

        return run_check_setup()
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
