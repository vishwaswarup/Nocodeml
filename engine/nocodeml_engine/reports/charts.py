"""Report charts: matplotlib figures rendered to PNG bytes.

Uses `Figure` directly (never pyplot), so there is no global state and it is safe to call from
request threads. White background and a single data colour keep it print-friendly.
"""

from __future__ import annotations

import io

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

BLUE = "#2f6fb5"
ACCENT = "#fc6435"
INK = "#1f2933"
MUTED = "#6b7280"
GRID = "#e5e7eb"
DPI = 200

_HEAT = LinearSegmentedColormap.from_list("nocodeml", ["#eef4fb", BLUE])


def _fig(w: float, h: float) -> tuple[Figure, "object"]:
    fig = Figure(figsize=(w, h), facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9ca3af")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    return fig, ax


def _png(fig: Figure) -> tuple[bytes, float]:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    w, h = fig.get_size_inches()
    return buf.getvalue(), h / w


def confusion_matrix_png(labels: list[str], matrix: list[list[int]]) -> tuple[bytes, float]:
    m = np.array(matrix)
    n = len(labels)
    size = min(3.4, 1.6 + 0.45 * n)
    fig, ax = _fig(size, size)
    ax.grid(False)
    ax.imshow(m, cmap=_HEAT, vmin=0, vmax=max(1, m.max()))
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=10 if n <= 4 else 8,
                    color="white" if m[i, j] > 0.55 * max(1, m.max()) else INK)
    short = [l if len(l) <= 8 else l[:7] + "…" for l in labels]
    ax.set_xticks(range(n), short)
    ax.set_yticks(range(n), short)
    ax.set_xlabel("Predicted", fontsize=9, color=INK)
    ax.set_ylabel("Actual", fontsize=9, color=INK)
    ax.set_title("Confusion matrix", fontsize=10, color=INK, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    return _png(fig)


def roc_png(fpr: list[float], tpr: list[float], auc: float | None) -> tuple[bytes, float]:
    fig, ax = _fig(3.6, 3.2)
    ax.plot([0, 1], [0, 1], color="#9ca3af", linewidth=0.9)
    ax.fill_between(fpr, tpr, color=BLUE, alpha=0.10, linewidth=0)
    ax.plot(fpr, tpr, color=BLUE, linewidth=1.8, solid_capstyle="round")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate", fontsize=9, color=INK)
    ax.set_ylabel("True positive rate", fontsize=9, color=INK)
    ax.set_title("ROC curve", fontsize=10, color=INK, loc="left")
    if auc is not None:
        ax.text(0.97, 0.06, f"AUC {auc:.3f}", ha="right", fontsize=9, color=INK)
    return _png(fig)


def scatter_png(actual: list[float], predicted: list[float]) -> tuple[bytes, float]:
    a, p = np.asarray(actual), np.asarray(predicted)
    lo, hi = float(min(a.min(), p.min())), float(max(a.max(), p.max()))
    pad = (hi - lo) * 0.04 or 1.0
    fig, ax = _fig(3.6, 3.2)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#9ca3af", linewidth=0.9)
    ax.scatter(a, p, s=7, color=BLUE, alpha=0.5, linewidths=0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Actual", fontsize=9, color=INK)
    ax.set_ylabel("Predicted", fontsize=9, color=INK)
    ax.set_title("Actual vs predicted", fontsize=10, color=INK, loc="left")
    return _png(fig)


def histogram_png(values: list[float]) -> tuple[bytes, float]:
    v = np.asarray(values)
    fig, ax = _fig(3.6, 3.2)
    ax.hist(v, bins=20, color=BLUE, alpha=0.85, rwidth=0.88)
    if v.min() <= 0 <= v.max():
        ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_xlabel("Error (actual - predicted)", fontsize=9, color=INK)
    ax.set_ylabel("Rows", fontsize=9, color=INK)
    ax.set_title("Error distribution", fontsize=10, color=INK, loc="left")
    return _png(fig)


def comparison_png(names: list[str], values: list[float | None], label: str, baseline: float | None,
                   lower_is_better: bool) -> tuple[bytes, float]:
    n = len(names)
    fig, ax = _fig(6.4, 0.9 + 0.5 * n)
    ys = np.arange(n)[::-1]
    vals = [0.0 if v is None else v for v in values]
    ax.barh(ys, vals, height=0.42, color=BLUE)
    ax.set_yticks(ys, names)
    ax.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    span = max(max(vals, default=1), baseline or 0, 1e-9)
    for y, v, raw in zip(ys, vals, values):
        ax.text(v + span * 0.015, y, "n/a" if raw is None else f"{raw:.3f}", va="center", fontsize=8.5, color=INK)
    if baseline is not None:
        ax.axvline(baseline, color=ACCENT, linewidth=1.4)
        ax.text(baseline, n - 0.35, "baseline", color=ACCENT, fontsize=8, ha="center", va="bottom")
    ax.set_xlim(min(0, min(vals, default=0)) * 1.1, span * 1.18)
    ax.set_xlabel(f"{label} ({'lower' if lower_is_better else 'higher'} is better)", fontsize=9, color=INK)
    return _png(fig)
