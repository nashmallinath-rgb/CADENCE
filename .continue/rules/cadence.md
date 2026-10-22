---
name: CADENCE project rules
alwaysApply: true
---

# CADENCE rules (read before every task)

You are working on CADENCE, a multi-objective prompt-optimization framework. Another engineer (and another AI
session) works in the same repo, so stay inside your task and follow these rules exactly.

## Environment
- Windows, PowerShell. The project targets Python **3.12**. Use the repo venv: `.\.venv\Scripts\Activate.ps1`.
  Run `python --version` once at the start. If it is not 3.12, say so in your first message and keep going
  (the suite has also been seen passing on 3.14); never "fix" it by changing pins or installing packages globally.
- Run tests with `python -m pytest` (not bare `pytest`). Default run is offline and must stay under 60 seconds.
- LF line endings. UTF-8. No tabs in Python.

## Hard prohibitions
- **No secrets.** Never write, print, paste or commit API keys, tokens or `.env` contents. Never edit `.env`.
  Never put a key in a test, fixture, log line, comment or example.
- **No network in tests.** Tests use `MockClient` (`cadence.llm.MockClient`) or pure local data. No `requests`,
  no `urllib` to the internet, no real model calls, no dataset downloads.
- **Every LLM call goes through `cadence.llm`** (cached, retried, budget-checked). Never import `litellm`, `openai`,
  `google.genai` or `groq` outside `cadence/llm/`.
- Never commit to `main`. Never force-push. Never delete a branch you did not create.
- Do not add dependencies. If you believe one is needed, stop and ask.

## Extend, do not rewrite
- Add new files and new functions. When you must edit an existing file, make the smallest change that works and keep
  every existing public name, signature and behaviour. Do not reformat, rename, reorder or "clean up" code you were
  not asked to touch. Do not delete existing tests.
- Shared hot-spot files (`cadence/cli.py`, `cadence/tasks/registry.py`, `cadence/config.py`, `configs/*.yaml`,
  `pyproject.toml`): only append; never reorder existing lines. Other sessions append to them too.

## Schema constraints (do not change)
- `cadence/schemas.py` is frozen: `Usage, Example, GenRequest, Output, Score, Candidate, Objectives, Evaluated`.
  You may add a field only if it has a default AND your task explicitly says so.
- The `Task` contract in `cadence/tasks/base.py` is frozen. A task template is the SYSTEM message; the example is
  the USER message. `automatic_score` returns a float in [0, 1], or `None` meaning "needs a judge".
  `Example.reference` must never be shown to a model or judge.
- Config key names in `configs/*.yaml` and the role names in `models.yaml` are fixed. Do not rename them.
- Objectives: quality in [0, 1] (maximise), cost in USD per 1k runs (minimise), length in tokens (minimise).
- Anything random takes an explicit seed. Same input and seed must give the same output.

## Tests are required
- Every new public function gets a pytest test in `tests/` (name `test_<module>.py`). Cover the normal case, at
  least one edge case (empty input, malformed input, boundary value) and one failure case.
- Tests must be deterministic and independent of the current directory and of the order they run in.
- Before you say a task is done, run the whole suite: `python -m pytest`. Paste the final summary line.
  A task is not done while any test fails, including tests you did not write. If an old test fails because of your
  change, fix your change, not the old test.

## Code style
- Type hints on public functions. Pydantic models for structured data. Docstring of one or two lines per module/class.
- Pure functions where possible. No hidden global state, no I/O at import time.
- Prefer the standard library plus packages already in `pyproject.toml`.

## Git workflow
- One branch per task, created from the branch named in the task brief: `git checkout -b <branch> <base>`.
- Small commits with an imperative subject line ("Add validate_page with weighted checks").
- Run `git status` before committing. Never commit `.env`, `runs/`, `data/raw/`, `.cache/`, `mlflow.db` or `.venv/`.
- When finished: all tests green, then push the branch and open a PR. Do not merge it yourself.

## When stuck
- If the brief is ambiguous or contradicts the code, stop and ask. Do not invent an interface.
- If a test you wrote keeps failing for the same reason after two attempts, stop and report the exact error.
- Never claim something passes without having run it.
