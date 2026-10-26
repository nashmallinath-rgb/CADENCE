"""report.md: a pure function of a run directory. Tolerant of missing files; every section is always present."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from cadence.config import ROOT, load_budget_config, load_models_config

NA = "_not available: {}_"


def _json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _events(run_dir: Path) -> list[dict[str, Any]]:
    p = run_dir / "trace.jsonl"
    out: list[dict[str, Any]] = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _f(x: Any, spec: str = ".3f") -> str:
    return "n/a" if x is None else format(x, spec)


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _short(t: str, n: int = 70) -> str:
    t = " ".join(t.split())
    return (t if len(t) <= n else t[: n - 3] + "...").replace("|", "\\|")


def _header(run_dir: Path, cfg: Any, events: list[dict[str, Any]]) -> str:
    lines = ["## 1. Run", ""]
    if not cfg:
        return "\n".join(lines + [NA.format("config.json missing")])
    st = cfg.get("settings", {})
    try:
        roles = load_models_config().roles.model_dump()
    except Exception:  # noqa: BLE001
        roles = {}
    lines += [f"- run: `{run_dir.name}`  task: `{cfg.get('task')}`  seed: {st.get('seed')}  commit: `{_commit()}`",
              f"- settings: generations {st.get('max_gen')}, children/gen {st.get('children_per_gen')}, minibatch "
              f"{st.get('minibatch')}, final sample {st.get('final_sample')}, archive cap {st.get('max_archive')}"]
    stop = next((e for e in reversed(events) if e.get("event") == "stop"), None)
    if stop:
        lines.append(f"- stopped: {stop.get('reason')} after {stop.get('generations')} generation(s); "
                     f"final re-evaluation: {'yes' if stop.get('final_reevaluated') else 'no'}")
    if roles:
        lines.append("- models (live runs; a `--mock` run used none): " + ", ".join(
            f"{k}=`{roles[k]}`" for k in ("generator", "reflector", "cheap_judge", "strong_judge", "second_family_judge")
            if k in roles))
    return "\n".join(lines)


def _menu(arch: Any, held: Any) -> str:
    lines = ["## 2. Pareto menu", ""]
    if not arch:
        return "\n".join(lines + [NA.format("archive.json missing")])
    hs = {m["candidate_id"]: m for m in (held or {}).get("members", [])}
    lines += ["| # | opt quality | held-out quality (95% CI) | $ / 1k | tokens | gen | template |", "|---|---|---|---|---|---|---|"]
    rows = sorted(arch, key=lambda m: (-m["objectives"]["quality"], m["objectives"]["cost_usd_per_1k"]))
    for i, m in enumerate(rows, 1):
        o, c = m["objectives"], m["candidate"]
        h = hs.get(c["id"])
        ho = (f"{h['heldout_quality']:.3f} ({h['ci_low']:.3f}-{h['ci_high']:.3f})"
              if h and h.get("heldout_quality") is not None else "not run")
        lines.append(f"| {i} | {o['quality']:.3f} | {ho} | {o['cost_usd_per_1k']:.4f} | {o['length_tokens']} | "
                     f"{c['generation']} | {_short(c['template'])} |")
    return "\n".join(lines)


def _judges(calib: Any) -> str:
    lines = ["## 3. Judge reliability", ""]
    if not calib:
        return "\n".join(lines + [NA.format("no calibration.json; run `calibrate --task <name>` (or `run --calibrate`)")])
    if not calib.get("supported", True):
        return "\n".join(lines + [NA.format(calib.get("reason", "task has no ground truth"))])
    lines += [f"Calibration set: {calib.get('calibration_items')} items ({calib.get('calibration_correct')} correct), "
              f"{calib.get('trials')} trials per item; judges flagged below kappa {calib.get('kappa_min')}.", "",
              "| judge | kappa vs truth | ICC | status |", "|---|---|---|---|"]
    for role, j in (calib.get("judges") or {}).items():
        if "error" in j:
            lines.append(f"| {role} | error | | {j['error']} |")
        else:
            lines.append(f"| {role} | {_f(j.get('kappa'), '.2f')} | {_f(j.get('icc'), '.2f')} | "
                         f"{'**FLAGGED**' if j.get('flagged') else 'ok'} |")
    if calib.get("inter_judge_kappa") is not None:
        lines.append(f"\nInter-judge kappa (cheap vs second family) on the calibration set: {calib['inter_judge_kappa']:.2f}")
    lines += ["", "Limits: " + "; ".join(calib.get("limits", []))]
    return "\n".join(lines)


def _cascade(events: list[dict[str, Any]]) -> str:
    lines = ["## 4. Cascade", ""]
    ev = [e for e in events if e.get("event") == "evaluated"]
    judged = sum(e.get("n_judged", 0) for e in ev)
    total = sum(e.get("n_examples", 0) for e in ev)
    esc = sum(e.get("n_escalated", 0) for e in ev)
    if not ev:
        return "\n".join(lines + [NA.format("no evaluation events in trace.jsonl")])
    if judged == 0:
        return "\n".join(lines + [f"All {total} scored examples used an automatic metric; no judge calls were needed."])
    lines.append(f"{judged} of {total} scored examples needed a judge; {esc} of those escalated to the strong judge "
                 f"(escalation ratio {esc / judged:.0%}). The cheap judge alone resolved {judged - esc} "
                 f"({(judged - esc) / judged:.0%}), which is the saving versus always calling the strong judge.")
    gens = [e for e in events if e.get("event") == "generation"]
    if gens:
        lines += ["", "| gen | front | hypervolume | best quality | escalation ratio |", "|---|---|---|---|---|"]
        lines += [f"| {g['gen']} | {int(g['front_size'])} | {g['hypervolume']:.4g} | {g['best_quality']:.3f} | "
                  f"{_f(g.get('escalation_ratio'), '.0%')} |" for g in gens]
    return "\n".join(lines)


def _spend(events: list[dict[str, Any]], held: Any) -> str:
    lines = ["## 5. Spend versus budget", ""]
    gens = [e for e in events if e.get("event") == "generation"]
    try:
        b = load_budget_config()
        caps = f"caps in budget.yaml: ${b.max_usd_per_run:g}, {b.max_calls_per_run} calls, {b.max_tokens_per_run} tokens (the CLI may override)"
    except Exception:  # noqa: BLE001
        caps = "budget caps unavailable"
    if not gens or gens[-1].get("spend_usd") is None:
        return "\n".join(lines + [f"Spend by the search: not recorded (mock client or older run). {caps}."])
    g = gens[-1]
    lines.append(f"Search spend at the last generation: ${g['spend_usd']:.4f} (list price) over {int(g.get('calls') or 0)} "
                 f"live calls. {caps}.")
    if held and held.get("partial"):
        lines.append(f"The held-out benchmark stopped early ({held.get('stop_reason')}); its numbers cover fewer examples.")
    return "\n".join(lines)


def _heldout(held: Any) -> str:
    lines = ["## 6. Held-out benchmark and anchoring", ""]
    if not held:
        return "\n".join(lines + [NA.format("run `benchmark --run <id>`")])
    ms = held.get("members", [])
    lines.append(f"Task `{held['task']}`, seed {held['seed']}, up to {held['n_requested']} held-out examples per member"
                 + ("; **partial** (" + str(held.get("stop_reason")) + ")" if held.get("partial") else "") + ".")
    lines += ["", "| member | n | held-out | opt | gap (opt - held-out) | anchored | escalation |", "|---|---|---|---|---|---|---|"]
    for m in ms:
        tag = " (seed baseline)" if m.get("is_baseline") else ""
        lines.append(f"| `{m['candidate_id']}`{tag} | {m['n']} ({m['n_errors']} err) | {_f(m.get('heldout_quality'))} | "
                     f"{_f(m.get('opt_quality'))} | {_f(m.get('gap'), '+.3f')} | {m['anchored_fraction']:.0%} | "
                     f"{m['escalation_ratio']:.0%} |")
    base = next((m for m in ms if m.get("is_baseline") and m.get("heldout_quality") is not None), None)
    rest = [m for m in ms if not m.get("is_baseline") and m.get("heldout_quality") is not None]
    if base and rest:
        best = max(rest, key=lambda m: (m["heldout_quality"], -(m.get("cost_usd_per_1k") or 0.0)))
        line = (f"\n**Seed vs best optimised prompt (held-out):** quality {base['heldout_quality']:.3f} to "
                f"{best['heldout_quality']:.3f}")
        if base.get("cost_usd_per_1k") and best.get("cost_usd_per_1k") is not None:
            line += (f", cost per 1k runs ${base['cost_usd_per_1k']:.4f} to ${best['cost_usd_per_1k']:.4f} "
                     f"({(best['cost_usd_per_1k'] / base['cost_usd_per_1k'] - 1) * 100:+.0f}%)")
        lines.append(line + ".")
    r = held.get("opt_vs_heldout_rank") or {}
    if r.get("spearman") is not None:
        lines.append(f"\nRank agreement between optimisation-split and held-out quality: Spearman {r['spearman']:.2f}, "
                     f"Kendall {r['kendall']:.2f}. A large positive gap means a prompt was tuned to its optimisation examples.")
    lines.append("\n'Anchored' = share of rows scored by an exact automatic metric (ground truth). Judge scores and anchored "
                 "scores are never blended for the same row.")
    return "\n".join(lines)


def _dual(d: Any) -> str:
    lines = ["## 7. Dual-judge cross-validation and self-preference", ""]
    if not d:
        return "\n".join(lines + [NA.format("no dual_judge.json; it is produced by `benchmark --run <id>`")])
    if d.get("skipped"):
        return "\n".join(lines + [NA.format(d.get("reason", "skipped"))])
    lines += [f"{d['n']} judged outputs re-scored by `{d['second_role']}`.", "",
              f"- kappa (binary, threshold 5): **{_f(d.get('kappa'), '.2f')}**" + ("  **FLAGGED: below %.2f, the judge ranking is not trustworthy**" % d["kappa_min"] if d.get("flagged") else ""),
              f"- quadratic-weighted kappa: {_f(d.get('kappa_quadratic'), '.2f')}   ICC(2,1): {_f(d.get('icc'), '.2f')}",
              f"- mean score, primary vs second: {_f(d.get('mean_primary'), '.2f')} vs {_f(d.get('mean_second'), '.2f')}"]
    ra = d.get("rank_agreement") or {}
    lines.append(f"- member ranking agreement: Spearman {_f(ra.get('spearman'), '.2f')}, Kendall {_f(ra.get('kendall'), '.2f')}")
    sp = d.get("self_preference") or {}
    if sp:
        lines.append(f"- same-family self-preference gap: {sp['mean_gap']:+.2f} points (95% CI {sp['ci_low']:+.2f} to "
                     f"{sp['ci_high']:+.2f}, n={sp['n']}){' **significant**' if sp.get('significant') else ' (not significant)'}")
    for c in d.get("caveats", []):
        lines.append(f"- caveat: {c}")
    return "\n".join(lines)


def _streams(cfg: Any, events: list[dict[str, Any]]) -> str:
    lines = ["## 8. Streams demonstrated and limitations", ""]
    task = (cfg or {}).get("task", "unknown")
    stream = {"image": "image", "audio": "audio", "code_web": "code_web"}.get(task, "text")
    lines.append(f"- This run: stream `{stream}`, task `{task}`. Run the CLI once per stream to demonstrate all four.")
    lines += ["- Prices in `configs/pricing.yaml` marked `verify: true` are unverified; wrong prices distort the cost axis.",
              "- Free-tier models only; results depend on provider behaviour on the day of the run.",
              "- Held-out samples are small; read the confidence intervals, not just the means.",
              "- Image generation cost is constant per image, so that stream's front is mostly quality versus prompt length.",
              "- A judge that is consistently wrong can pass the confidence-based cascade; exact-match anchoring is the safeguard."]
    return "\n".join(lines)


def build_report(run_dir: Path, plots: dict[str, str] | None = None) -> str:
    run_dir = Path(run_dir)
    cfg = _json(run_dir / "config.json")
    events = _events(run_dir)
    arch = _json(run_dir / "archive.json")
    held = _json(run_dir / "heldout_summary.json")
    parts = [f"# CADENCE report: {run_dir.name}", "", _header(run_dir, cfg, events), "", _menu(arch, held)]
    if plots:
        parts += ["", "Plots: " + ", ".join(f"[{k}]({v})" for k, v in plots.items())]
    else:
        parts += ["", "_Plots not generated (cadence.benchmark.plots unavailable or run was empty)._"]
    parts += ["", _judges(_json(run_dir / "calibration.json")), "", _cascade(events), "", _spend(events, held), "",
              _heldout(held), "", _dual(_json(run_dir / "dual_judge.json")), "", _streams(cfg, events), ""]
    return "\n".join(parts)


def make_plots(run_dir: Path) -> dict[str, str]:
    """Calls the plotting module if it exists (owned by another workstream); never raises."""
    try:
        from cadence.benchmark import plots
    except ImportError:
        return {}
    try:
        members = plots.load_members(run_dir / "archive.json")
        out: dict[str, str] = {}
        try:  # the headline figure is separate so that a problem in it cannot lose the other plots
            held = _json(run_dir / "heldout_summary.json")
            out["Quality vs cost"] = plots.plot_pareto_front(members, run_dir / "pareto_front.png", held).name
        except Exception:  # noqa: BLE001
            pass
        out["3D Pareto"] = plots.plot_pareto_3d(members, run_dir / "pareto_3d.png").name
        for p in plots.plot_pareto_2d(members, run_dir / "pareto"):
            out[Path(p).stem] = Path(p).name
        return out
    except Exception:  # noqa: BLE001 - a plotting problem must not lose the report
        return {}


def write_report(run_dir: Path, with_plots: bool = True) -> Path:
    run_dir = Path(run_dir)
    text = build_report(run_dir, make_plots(run_dir) if with_plots else None)
    path = run_dir / "report.md"
    path.write_text(text, encoding="utf-8")
    return path
