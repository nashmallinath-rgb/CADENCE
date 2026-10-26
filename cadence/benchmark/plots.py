"""Pareto plots: one 3D scatter (cost, length, quality) and three 2D projections. Headless (Agg backend)."""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cadence.schemas import Evaluated  # noqa: E402


def load_members(archive_json: Path) -> list[Evaluated]:
    """Read runs/<id>/archive.json (a JSON list of Evaluated)."""
    return [Evaluated.model_validate(d) for d in json.loads(Path(archive_json).read_text(encoding="utf-8"))]


def _ranked(members: Sequence[Evaluated]) -> list[tuple[int, Evaluated]]:
    """Rank 1 = best quality (ties: cheaper first)."""
    order = sorted(members, key=lambda m: (-m.objectives.quality, m.objectives.cost_usd_per_1k, m.objectives.length_tokens))
    return list(enumerate(order, 1))


def _save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pareto_3d(members: Sequence[Evaluated], out_path: Path) -> Path:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    ranked = _ranked(members)
    if ranked:
        xs = [m.objectives.cost_usd_per_1k for _, m in ranked]
        ys = [m.objectives.length_tokens for _, m in ranked]
        zs = [m.objectives.quality for _, m in ranked]
        ax.scatter(xs, ys, zs, s=40, c=zs, cmap="viridis", depthshade=False)
        for (rank, _), x, y, z in zip(ranked, xs, ys, zs):
            ax.text(x, y, z, f" {rank}", fontsize=8)
    else:
        ax.text2D(0.5, 0.5, "no members", transform=ax.transAxes, ha="center")
    ax.set_xlabel("cost (USD per 1k runs)")
    ax.set_ylabel("template length (tokens)")
    ax.set_zlabel("quality (0-1)")
    ax.set_title(f"Pareto front ({len(ranked)} member{'s' if len(ranked) != 1 else ''}; 1 = best quality)")
    return _save(fig, out_path)


def plot_pareto_2d(members: Sequence[Evaluated], out_prefix: Path) -> list[Path]:
    ranked = _ranked(members)
    get = {"quality": lambda m: m.objectives.quality, "cost": lambda m: m.objectives.cost_usd_per_1k,
           "length": lambda m: float(m.objectives.length_tokens)}
    label = {"quality": "quality (0-1)", "cost": "cost (USD per 1k runs)", "length": "template length (tokens)"}
    out: list[Path] = []
    for y, x in (("quality", "cost"), ("quality", "length"), ("cost", "length")):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        if ranked:
            ax.scatter([get[x](m) for _, m in ranked], [get[y](m) for _, m in ranked], s=40)
            for rank, m in ranked:
                ax.annotate(str(rank), (get[x](m), get[y](m)), textcoords="offset points", xytext=(4, 4), fontsize=8)
        else:
            ax.text(0.5, 0.5, "no members", transform=ax.transAxes, ha="center")
        ax.set_xlabel(label[x])
        ax.set_ylabel(label[y])
        ax.set_title(f"{y} vs {x} ({len(ranked)} member{'s' if len(ranked) != 1 else ''})")
        ax.grid(alpha=0.3)
        out.append(_save(fig, Path(f"{out_prefix}_{y}_{x}.png")))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Headline figure: quality versus cost (log scale), bubble size = prompt length, with the seed prompt as a baseline.
# ---------------------------------------------------------------------------------------------------------------
FOREST, GOLD, TERRACOTTA, SAND, INK = "#2F4F3E", "#C9A227", "#B5543A", "#F4ECDD", "#2B2B2B"  # approximates the deck palette


def _front_points(members: Sequence[Evaluated], heldout: dict | None) -> tuple[list[dict], dict | None, bool]:
    """Plot points for the menu members plus the baseline. Uses held-out numbers when the benchmark has been run."""
    hs = {m["candidate_id"]: m for m in (heldout or {}).get("members", [])}
    pts: list[dict] = []
    base: dict | None = None
    used_heldout = False
    for rank, m in _ranked(members):
        h = hs.get(m.candidate.id)
        ok = bool(h) and h.get("heldout_quality") is not None and h.get("cost_usd_per_1k") is not None
        used_heldout = used_heldout or ok
        pt = {"rank": rank, "q": h["heldout_quality"] if ok else m.objectives.quality,
              "cost": h["cost_usd_per_1k"] if ok else m.objectives.cost_usd_per_1k,
              "len": m.objectives.length_tokens, "lo": h.get("ci_low") if ok else None,
              "hi": h.get("ci_high") if ok else None, "baseline": bool(h and h.get("is_baseline"))}
        pts.append(pt)
    for h in (heldout or {}).get("members", []):
        if h.get("is_baseline") and h.get("heldout_quality") is not None and h.get("cost_usd_per_1k") is not None \
                and h["candidate_id"] not in {m.candidate.id for m in members}:
            base = {"rank": 0, "q": h["heldout_quality"], "cost": h["cost_usd_per_1k"], "len": h["length_tokens"],
                    "lo": h.get("ci_low"), "hi": h.get("ci_high"), "baseline": True}
    return pts, base, used_heldout


def plot_pareto_front(members: Sequence[Evaluated], out_path: Path, heldout: dict | None = None,
                      title: str | None = None) -> Path:
    """2D headline plot. Quality is always drawn on 0-1 so one point still reads sensibly; cost is log when possible."""
    pts, base, used_heldout = _front_points(members, heldout)
    allp = pts + ([base] if base else [])
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    fig.patch.set_facecolor(SAND)
    ax.set_facecolor("#FBF7EE")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.grid(alpha=0.25, color=INK)
    ax.tick_params(colors=INK)
    if not allp:
        ax.text(0.5, 0.5, "no members", transform=ax.transAxes, ha="center", color=INK)
    else:
        use_log = all(p["cost"] > 0 for p in allp)
        if use_log:
            ax.set_xscale("log")
        size = lambda p: 90 + 14 * min(p["len"], 60)  # noqa: E731 - bubble area grows with prompt length (capped)
        for p in allp:
            if p["lo"] is not None and p["hi"] is not None:
                ax.errorbar(p["cost"], p["q"], yerr=[[max(0.0, p["q"] - p["lo"])], [max(0.0, p["hi"] - p["q"])]],
                            fmt="none", ecolor=TERRACOTTA if p["baseline"] else FOREST, alpha=0.5, capsize=3, lw=1.2,
                            zorder=2)
            is_b = p["baseline"]
            ax.scatter([p["cost"]], [p["q"]], s=size(p), zorder=3, edgecolor=TERRACOTTA if is_b else FOREST, lw=2,
                       facecolor="none" if is_b else FOREST, alpha=0.95)
            if is_b:  # the hollow bubble is too small to hold the word, so label it beside the marker
                ax.annotate("seed prompt", (p["cost"], p["q"]), textcoords="offset points", xytext=(0, -22), ha="center",
                            fontsize=9, fontweight="bold", color=TERRACOTTA, zorder=4)
            else:
                ax.annotate(str(p["rank"]), (p["cost"], p["q"]), ha="center", va="center", fontsize=9,
                            fontweight="bold", color=SAND, zorder=4)
        base_pt = next((p for p in allp if p["baseline"]), None)
        best = max((p for p in allp if not p["baseline"]), key=lambda p: (p["q"], -p["cost"]), default=None)
        if base_pt is not None and best is not None and (best["q"], best["cost"]) != (base_pt["q"], base_pt["cost"]):
            ax.annotate("", xy=(best["cost"], best["q"]), xytext=(base_pt["cost"], base_pt["q"]), zorder=1,
                        arrowprops={"arrowstyle": "-|>", "color": GOLD, "lw": 2.2, "shrinkA": 14, "shrinkB": 14})
            dq = best["q"] - base_pt["q"]
            dc = (best["cost"] / base_pt["cost"] - 1) * 100 if base_pt["cost"] > 0 else float("nan")
            note = f"quality {dq:+.2f}" + (f", cost {dc:+.0f}%" if dc == dc else "") + " (best prompt vs seed)"
            mx = (best["cost"] * base_pt["cost"]) ** 0.5 if use_log else (best["cost"] + base_pt["cost"]) / 2
            ax.annotate(note, (mx, (best["q"] + base_pt["q"]) / 2), textcoords="offset points", xytext=(10, 10),
                        fontsize=9, color=INK, bbox={"boxstyle": "round,pad=0.3", "fc": SAND, "ec": GOLD})
        ax.set_ylim(-0.03, 1.08)
        ax.margins(x=0.25)
    ax.set_xlabel("cost per 1,000 runs (USD, log scale)" if allp and all(p["cost"] > 0 for p in allp) else
                  "cost per 1,000 runs (USD)", color=INK, fontsize=10)
    ax.set_ylabel("quality (0 to 1; higher is better)", color=INK, fontsize=10)
    src = "held-out examples" if used_heldout else "optimisation examples"
    ax.set_title(title or f"Quality versus cost: {len(pts)} Pareto prompt{'s' if len(pts) != 1 else ''}",
                 color=INK, fontsize=13, fontweight="bold", loc="left", pad=22)
    ax.text(0, 1.03, f"scored on {src}", transform=ax.transAxes, fontsize=8.5, color=INK, alpha=0.8)
    fig.text(0.01, -0.02, "filled = optimised prompt (number = rank by quality)   hollow = starting seed prompt\n"
             "bubble size = prompt length   whiskers = 95% bootstrap CI (held-out only)", fontsize=8, color=INK, alpha=0.8,
             va="top")
    return _save(fig, out_path)
