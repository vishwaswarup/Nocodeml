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


# ---------------------------------------------------------------------------
# Dataset charts (rendered from the visualization engine's JSON, same numbers as the web app shows)
# ---------------------------------------------------------------------------


def _short(label: str, n: int = 18) -> str:
    return label if len(label) <= n else label[: n - 1] + "…"


def _compact(ax, axis: str = "x") -> None:
    """At most 5 ticks with short labels (12k, 3.5M), so wide numbers never run into each other."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    def fmt(v, _):
        a = abs(v)
        for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
            if a >= div:
                return f"{v / div:.3g}{suf}"
        return f"{v:.3g}"

    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(MaxNLocator(5))
    target.set_major_formatter(FuncFormatter(fmt))


def _hbars(title: str, labels: list[str], values: list[float], texts: list[str], xlabel: str,
           color: str = BLUE) -> tuple[bytes, float]:
    n = len(labels)
    fig, ax = _fig(3.6, 0.9 + 0.27 * n)
    ys = np.arange(n)[::-1]
    ax.barh(ys, values, height=0.62, color=color)
    ax.set_yticks(ys, [_short(l) for l in labels])
    ax.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    span = max(max(values, default=1), 1e-9)
    for y, v, t in zip(ys, values, texts):
        ax.text(v + span * 0.02, y, t, va="center", fontsize=7.5, color=INK)
    ax.set_xlim(0, span * 1.25)
    ax.set_xlabel(xlabel, fontsize=8.5, color=INK)
    return _png(fig)


def class_balance_png(data: dict) -> tuple[bytes, float]:
    items = data["items"][:12]
    return _hbars(f"Class balance of '{_short(data['column'], 24)}'", [i["label"] for i in items],
                  [i["count"] for i in items], [f"{i['count']:,} ({100 * i['share']:.1f}%)" for i in items], "Rows")


def categories_png(data: dict) -> tuple[bytes, float]:
    items = data["items"][:12]
    labels = [i["label"] for i in items]
    values = [i["count"] for i in items]
    texts = [f"{i['count']:,} ({100 * i['share']:.1f}%)" for i in items]
    if data.get("other"):
        labels.append("Other"); values.append(data["other"]); texts.append(f"{data['other']:,} ({100 * data['other_share']:.1f}%)")
    return _hbars(f"Values of '{_short(data['column'], 24)}'", labels, values, texts, "Rows")


def missing_png(data: dict) -> tuple[bytes, float]:
    cols = data["columns"][:12]
    return _hbars("Missing values by column", [c["column"] for c in cols], [c["pct"] for c in cols],
                  [f"{c['pct']:.1f}%" for c in cols], "% of rows missing", color=ACCENT)


def distribution_png(data: dict) -> tuple[bytes, float]:
    edges, counts = np.asarray(data["edges"], dtype=float), np.asarray(data["counts"])
    fig, ax = _fig(3.6, 3.0)
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.4)
    box = data["box"]
    ax.axvline(box["median"], color=ACCENT, linewidth=1.3)
    ax.text(box["median"], counts.max() * 1.02, " median", color=ACCENT, fontsize=7.5, va="bottom")
    ax.set_ylim(0, counts.max() * 1.12)
    _compact(ax)
    ax.set_xlabel(_short(data["column"], 30), fontsize=8.5, color=INK)
    ax.set_ylabel("Rows", fontsize=8.5, color=INK)
    return _png(fig)


def scatter_pair_png(data: dict) -> tuple[bytes, float]:
    fig, ax = _fig(3.6, 3.0)
    ax.scatter(data["x"], data["y"], s=6, color=BLUE, alpha=0.45, linewidths=0)
    _compact(ax, "x"); _compact(ax, "y")
    ax.set_xlabel(_short(data["x_column"], 30), fontsize=8.5, color=INK)
    ax.set_ylabel(_short(data["y_column"], 30), fontsize=8.5, color=INK)
    return _png(fig)


def correlation_png(data: dict, max_cols: int = 12) -> tuple[bytes, float]:
    names, m = data["columns"], np.array([[np.nan if v is None else v for v in row] for row in data["matrix"]], dtype=float)
    keep = list(range(len(names)))
    if len(keep) > max_cols:  # the target's strongest links first, otherwise the first columns
        t = names.index(data["target"]) if data.get("target") in names else None
        if t is not None:
            order = sorted((i for i in keep if i != t), key=lambda i: -abs(0 if np.isnan(m[t, i]) else m[t, i]))
            keep = [t] + order[: max_cols - 1]
        else:
            keep = keep[:max_cols]
    sub = m[np.ix_(keep, keep)]
    labels = [_short(names[i], 14) for i in keep]
    k = len(keep)
    size = min(6.2, 2.6 + 0.3 * k)
    fig, ax = _fig(size, size)
    ax.grid(False)
    im = ax.imshow(sub, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(k), labels, rotation=45, ha="right")
    ax.set_yticks(range(k), labels)
    if k <= 9:
        for i in range(k):
            for j in range(k):
                if not np.isnan(sub[i, j]):
                    ax.text(j, i, f"{sub[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(sub[i, j]) > 0.6 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7, colors=MUTED)
    ax.set_title("Correlations (Pearson r)", fontsize=10, color=INK, loc="left")
    return _png(fig)


DATASET_CHART_RENDERERS = {
    "class_balance": class_balance_png, "categories": categories_png, "missing": missing_png,
    "distribution": distribution_png, "scatter": scatter_pair_png, "correlation": correlation_png,
}
