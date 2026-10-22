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
