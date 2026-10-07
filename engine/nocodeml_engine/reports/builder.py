"""Deterministic PDF experiment report (spec section 37).

Template-based: every sentence is assembled from the stored experiment record, configuration and
dataset profile. No LLM, no wall-clock time, so the same inputs always give the same PDF bytes.

All text that originates from the user's data (column names, category values, file names) is
XML-escaped before reaching reportlab, whose paragraphs parse mini-HTML.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable
from xml.sax.saxutils import escape

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate, CondPageBreak, Flowable, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph,
    Preformatted, Spacer, Table, TableStyle,
)

from nocodeml_engine.config import PipelineConfig
from nocodeml_engine.models import get_spec
from nocodeml_engine.reports import charts
from nocodeml_engine.results import ExperimentResult, ModelResult

# ---------------------------------------------------------------------------
# Fonts (DejaVu ships with matplotlib and covers superscripts, arrows and accents)
# ---------------------------------------------------------------------------

_FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
for _name, _file in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
                     ("DV-BI", "DejaVuSans-BoldOblique.ttf"), ("DVMono", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(_name, os.path.join(_FONT_DIR, _file)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

INK = colors.HexColor("#1f2933")
MUTED = colors.HexColor("#6b7280")
LINE = colors.HexColor("#d9dde3")
HEAD_BG = colors.HexColor("#eef2f7")
ZEBRA = colors.HexColor("#f8fafc")
BLUE = colors.HexColor("#2f6fb5")
STATUS_COLOR = {"pass": colors.HexColor("#15803d"), "warn": colors.HexColor("#b45309"), "fail": colors.HexColor("#b91c1c")}

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

ST = {
    "body": ParagraphStyle("body", fontName="DV", fontSize=9, leading=13, textColor=INK, spaceAfter=4),
    "small": ParagraphStyle("small", fontName="DV", fontSize=7.8, leading=11, textColor=MUTED, spaceAfter=3),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=14, leading=18, textColor=INK, spaceBefore=16, spaceAfter=7, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=10.5, leading=14, textColor=INK, spaceBefore=9, spaceAfter=4, keepWithNext=1),
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=26, leading=30, textColor=INK, spaceAfter=2),
    "kicker": ParagraphStyle("kicker", fontName="DV", fontSize=8.5, leading=11, textColor=BLUE, spaceAfter=6),
    "project": ParagraphStyle("project", fontName="DV", fontSize=13, leading=17, textColor=MUTED, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8, leading=10.5, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=8, leading=10.5, textColor=INK),
    "cellr": ParagraphStyle("cellr", fontName="DV", fontSize=8, leading=10.5, textColor=INK, alignment=TA_RIGHT),
    "head": ParagraphStyle("head", fontName="DV-B", fontSize=7.8, leading=10, textColor=MUTED),
    "headr": ParagraphStyle("headr", fontName="DV-B", fontSize=7.8, leading=10, textColor=MUTED, alignment=TA_RIGHT),
    "bullet": ParagraphStyle("bullet", fontName="DV", fontSize=9, leading=13, textColor=INK, leftIndent=11, bulletIndent=0, spaceAfter=2),
    "mono": ParagraphStyle("mono", fontName="DVMono", fontSize=6.6, leading=8.4, textColor=INK),
}

DESCRIPTIONS = {
    "logistic_regression": "Draws a straight boundary between classes. Fast and easy to explain.",
    "knn": "Predicts from the most similar rows. No training step, but slow on large data.",
    "decision_tree": "A flowchart of yes/no questions. Easy to read, but can memorize the data.",
    "random_forest": "Many decision trees voting together. A strong default that rarely needs tuning.",
    "svm": "Finds the widest margin between classes. Strong on small, clean data; slow on large.",
    "linear_regression": "Fits a straight line (or plane) through the data. The simplest baseline.",
    "ridge": "Linear regression that shrinks its coefficients (L2).",
    "lasso": "Linear regression that can zero out features (L1).",
    "gradient_boosting": "Trees built one after another, each fixing the last one's mistakes.",
}
MISSING_WORDS = {"mean": "filled with the mean", "median": "filled with the median", "mode": "filled with the most common value",
                 "constant": "filled with a constant", "knn": "filled from the nearest rows (KNN)",
                 "drop_rows": "rows with a missing value removed", "drop_column": "column dropped"}
ENCODING_WORDS = {"one_hot": "one-hot encoded", "ordinal": "ordinal encoded", "label": "label encoded",
                  "frequency": "frequency encoded (replaced by how common each value is)", "target": "target encoded"}
SCALING_WORDS = {"standard": "standardized (mean 0, unit spread)", "min_max": "min-max scaled to 0-1",
                 "robust": "robust scaled (median and IQR)", "none": "not scaled"}
SPLIT_WORDS = {"train_test": "Train / test split", "train_val_test": "Train / validation / test split",
               "k_fold": "K-fold cross-validation", "stratified_k_fold": "Stratified k-fold cross-validation",
               "time_series": "Time-series cross-validation"}
SOURCE_WORDS = {"test": "test set", "validation": "validation set", "cv": "cross-validation (out-of-fold predictions)"}
LOWER_BETTER = {"rmse", "mae", "mse", "mape"}


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class ReportContext:
    project_name: str
    experiment_number: int
    result: ExperimentResult
    config: PipelineConfig  # the pipeline version this experiment ran on
    pipeline_label: str  # e.g. "v3" or "v3.0 (finalized)"
    dataset_filename: str
    dataset_version: int
    profile: dict[str, Any]  # DatasetProfile.to_dict() computed with the target
    current: bool = True
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def esc(v: Any, limit: int | None = None) -> str:
    s = str(v)
    if limit and len(s) > limit:
        s = s[: limit - 1] + "…"
    return escape(s)


def P(text: Any, style: str = "body", limit: int | None = None) -> Paragraph:
    return Paragraph(esc(text, limit), ST[style])


def RP(markup: str, style: str = "body") -> Paragraph:
    """Paragraph from markup the code itself wrote (callers escape any data they interpolate)."""
    return Paragraph(markup, ST[style])


def f3(v: Any) -> str:
    return "–" if not isinstance(v, (int, float)) or isinstance(v, bool) else f"{v:.3f}"


def fnum(v: Any) -> str:
    """3 decimals for small values, thousands-separated whole numbers for large ones (fits narrow columns)."""
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return "–"
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3f}"


def num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def when(iso: str) -> str:
    return iso[:16].replace("T", " ") + " UTC"


def bullets(items: Iterable[str]) -> list[Flowable]:
    return [Paragraph(esc(i), ST["bullet"], bulletText="•") for i in items]


def table(rows: list[list[Any]], widths: list[float] | None = None, right: Iterable[int] = (), header: bool = True,
          repeat: bool = True) -> Table:
    right = set(right)
    data: list[list[Any]] = []
    for r, row in enumerate(rows):
        out = []
        for c, v in enumerate(row):
            if isinstance(v, Flowable):
                out.append(v)
                continue
            style = ("headr" if c in right else "head") if header and r == 0 else ("cellr" if c in right else "cell")
            out.append(Paragraph(esc(v), ST[style]))
        data.append(out)
    t = Table(data, colWidths=widths, repeatRows=1 if header and repeat else 0, hAlign="LEFT")
    cmds: list[Any] = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
    ]
    if header:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#c5ccd6"))]
    for i in range(2 if header else 1, len(rows), 2):
        cmds.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(cmds))
    return t


def kv(rows: list[tuple[str, Any]], key_w: float = 42 * mm) -> Table:
    t = table([[P(k, "cellb"), v if isinstance(v, Flowable) else P(v)] for k, v in rows], [key_w, CONTENT_W - key_w], header=False)
    return t


def image(png: tuple[bytes, float], width: float) -> Image:
    data, ratio = png
    return Image(io.BytesIO(data), width=width, height=width * ratio)


def section(n: int, title: str) -> Paragraph:
    return Paragraph(f"{n}. {esc(title)}", ST["h1"])


class _NumberedCanvasFactory:
    """Canvas that writes 'Page X of Y' (needs the final page count, so pages are held until save)."""

    def __init__(self, footer: str):
        self.footer = footer

    def __call__(self, *a, **k):
        footer = self.footer

        class Numbered(canvas.Canvas):
            def __init__(self, *aa, **kk):
                super().__init__(*aa, **kk)
                self._saved: list[dict] = []

            def showPage(self):
                self._saved.append(dict(self.__dict__))
                self._startPage()

            def save(self):
                total = len(self._saved)
                for state in self._saved:
                    self.__dict__.update(state)
                    self.setFont("DV", 7.5)
                    self.setFillColor(MUTED)
                    self.setStrokeColor(LINE)
                    self.setLineWidth(0.5)
                    self.line(MARGIN, 14 * mm, PAGE_W - MARGIN, 14 * mm)
                    self.drawString(MARGIN, 10 * mm, footer)
                    self.drawRightString(PAGE_W - MARGIN, 10 * mm, f"Page {self._pageNumber} of {total}")
                    super().showPage()
                super().save()

        return Numbered(*a, **k)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _metric_cols(task: str) -> list[tuple[str, str]]:
    return ([("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced acc."), ("precision", "Precision"), ("recall", "Recall"),
             ("f1", "F1"), ("roc_auc", "ROC-AUC")] if task == "classification"
            else [("r2", "R²"), ("rmse", "RMSE"), ("mae", "MAE"), ("mape", "MAPE")])


def _sources(models: list[ModelResult]) -> list[str]:
    return [s for s in ("test", "validation", "cv") if any(s in m.metrics for m in models)]


def _chart_source(models: list[ModelResult]) -> str:
    s = _sources(models)
    return "test" if "test" in s else ("cv" if "cv" in s else s[0])


def s_cover(c: ReportContext) -> list[Flowable]:
    r, cfg = c.result, c.config
    n_models = len(r.models)
    split = r.split
    sizes = (f"{split['n_folds']}-fold cross-validation" if split["mode"] == "cv"
             else f"{split['n_train']:,} training rows, {split.get('n_validation', 0):,} validation rows, {split['n_test']:,} test rows"
             if split.get("n_validation") else f"{split['n_train']:,} training rows and {split['n_test']:,} test rows")
    fails = [x for x in r.quality.checks if x.status == "fail"]
    warns = [x for x in r.quality.checks if x.status == "warn"]
    out: list[Flowable] = [
        Paragraph("NOCODEML  ·  EXPERIMENT REPORT", ST["kicker"]),
        P("Experiment report", "title"),
        P(c.project_name, "project", limit=120),
        kv([
            ("Experiment", f"#{c.experiment_number} on pipeline {c.pipeline_label}"),
            ("Task", f"{r.task.capitalize()}; target column “{cfg.dataset.target_column}”"),
            ("Dataset", f"{c.dataset_filename} (version {c.dataset_version})"),
            ("Models", ", ".join(m.name for m in r.models)),
            ("Run finished", when(r.finished_at)),
        ]),
        Spacer(1, 8),
        RP(esc(f"This report documents experiment #{c.experiment_number}, which trained {n_models} "
               f"model{'s' if n_models != 1 else ''} to predict “{cfg.dataset.target_column}” ({r.task}). "
               f"The data was divided as {sizes}, and models were judged on the {SOURCE_WORDS[_chart_source(r.models)]}. "
               f"Every number, chart and setting below is taken directly from the stored experiment record.")),
        RP(esc(f"Pipeline health scored {r.quality.score if r.quality.score is not None else 'n/a'} out of 100: "
               f"{len(fails)} failed and {len(warns)} to-review check{'s' if len(warns) != 1 else ''}. "
               "This score describes how soundly the experiment was built, not how accurate the models are.")),
    ]
    if not c.current:
        out.append(RP("<b>Note:</b> " + esc(f"the pipeline was changed after this run (this run used v{r.pipeline_version}). "
                                             "These results describe that earlier configuration.")))
    return out


def s_dataset(c: ReportContext, n: int) -> list[Flowable]:
    p = c.profile
    out: list[Flowable] = [section(n, "Dataset overview")]
    out.append(kv([
        ("Rows × columns", f"{p['n_rows']:,} × {p['n_columns']}"),
        ("Column types", f"{len(p['numerical'])} numerical, {len(p['categorical'])} categorical, {len(p['datetime'])} datetime"),
        ("Missing values", f"{p['missing_value_pct']}% of all cells"),
        ("Duplicate rows", f"{p['n_duplicate_rows']:,}"),
        ("Dataset fingerprint", c.result.dataset_fingerprint),
    ]))
    if p["warnings"]:
        out += [Spacer(1, 4), P("Dataset health flags (observations, not changes)", "h2")]
        out += bullets(w["message"] for w in p["warnings"][:25])
        if len(p["warnings"]) > 25:
            out.append(P(f"… and {len(p['warnings']) - 25} more.", "small"))
    return out


def s_statistics(c: ReportContext, n: int) -> list[Flowable]:
    p = c.profile
    cols = p["columns"]
    out: list[Flowable] = [section(n, "Dataset statistics")]
    LIM = 30
    numeric = [(k, v) for k, v in cols.items() if v["kind"] == "numerical"]
    if numeric:
        out.append(P("Numerical columns", "h2"))
        rows = [["Column", "Missing %", "Mean", "Median", "Std dev", "Min", "Max", "Outliers"]]
        for k, v in numeric[:LIM]:
            rows.append([k[:34], v["missing_pct"], fnum(v.get("mean")), fnum(v.get("median")), fnum(v.get("std")),
                         fnum(v.get("min")), fnum(v.get("max")), v.get("outlier_count", 0)])
        out.append(table(rows, [38 * mm] + [(CONTENT_W - 38 * mm) / 7] * 7, right=range(1, 8)))
        if len(numeric) > LIM:
            out.append(P(f"Showing {LIM} of {len(numeric)} numerical columns.", "small"))
    categorical = [(k, v) for k, v in cols.items() if v["kind"] == "categorical"]
    if categorical:
        out.append(P("Categorical columns", "h2"))
        rows = [["Column", "Missing %", "Distinct values", "Most common"]]
        for k, v in categorical[:LIM]:
            rows.append([k[:34], v["missing_pct"], v["unique"], str(v.get("mode"))[:40]])
        out.append(table(rows, [50 * mm, 25 * mm, 30 * mm, CONTENT_W - 105 * mm], right=(1, 2)))
        if len(categorical) > LIM:
            out.append(P(f"Showing {LIM} of {len(categorical)} categorical columns.", "small"))
    dates = [(k, v) for k, v in cols.items() if v["kind"] == "datetime"]
    if dates:
        out.append(P("Datetime columns", "h2"))
        out.append(table([["Column", "Missing %", "From", "To"]] + [[k, v["missing_pct"], str(v.get("min"))[:10], str(v.get("max"))[:10]]
                                                                  for k, v in dates[:LIM]], [50 * mm, 25 * mm, 40 * mm, CONTENT_W - 115 * mm], right=(1,)))
    dist = (p.get("target") or {}).get("class_distribution")
    if dist and c.result.task == "classification":
        out.append(P(f"Target “{c.config.dataset.target_column}”: class balance", "h2"))
        rows = [["Class", "Rows", "Share"]] + [[k, f"{v:,}", f"{100 * dist['proportions'][k]:.1f}%"] for k, v in list(dist["counts"].items())[:15]]
        out.append(table(rows, [60 * mm, 35 * mm, 35 * mm], right=(1, 2)))
        if dist.get("imbalanced"):
            out.append(P(f"The smallest class is {100 * dist['minority_ratio']:.1f}% of rows, so accuracy alone can be misleading.", "small"))
    return out


def s_dataset_charts(c: ReportContext, n: int) -> list[Flowable]:
    """Charts of the dataset as uploaded (before preprocessing), chosen by the same rules as the web app."""
    specs = c.extras.get("dataset_charts") or []
    cells: list[tuple[str, Any, str]] = []  # (title, rendered image, why)
    for spec in specs:
        render = charts.DATASET_CHART_RENDERERS.get(spec["type"])
        if render is None:
            continue
        try:
            cells.append((spec["title"], render(spec["data"]), spec["why"], spec["type"]))
        except Exception:  # noqa: BLE001 - one bad chart must not lose the whole report
            continue
    if not cells:
        return []
    out: list[Flowable] = [section(n, "Dataset charts"),
                           P("Drawn from the dataset as uploaded, before any preprocessing. Charts are picked by fixed rules, "
                             "and the sentence under each says why.", "small")]
    half = (CONTENT_W - 6 * mm) / 2
    wide = [x for x in cells if x[3] == "correlation"]
    narrow = [x for x in cells if x[3] != "correlation"]

    def block(title, png, why, width):
        return [P(title, "h2"), image(png, width), P(why, "small")]   # a list is a multi-flowable table cell

    for i in range(0, len(narrow), 2):
        row = [block(t, png, why, half) for t, png, why, _ in narrow[i:i + 2]]
        out.append(Table([row + [""] * (2 - len(row))], colWidths=[half + 3 * mm] * 2,
                         style=[("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    for t, png, why, _ in wide:
        out.append(KeepTogether(block(t, png, why, 0.8 * CONTENT_W)))
    return out


def s_preprocessing(c: ReportContext, n: int) -> list[Flowable]:
    pre = c.config.preprocessing.model_dump(mode="json")
    out: list[Flowable] = [section(n, "Preprocessing")]
    items: list[str] = []
    if pre["drop_columns"]:
        items.append("Columns not used: " + ", ".join(pre["drop_columns"]))
    for r in pre["missing_values"]:
        what = MISSING_WORDS.get(r["strategy"], r["strategy"])
        if r["strategy"] == "constant":
            what = f"filled with the value {r.get('constant_value')}"
        if r["strategy"] == "knn":
            what = f"filled from the {r.get('n_neighbors', 5)} nearest rows (KNN)"
        items.append(f"Missing values in “{r['column']}”: {what}")
    for r in pre["encoding"]:
        items.append(f"“{r['column']}”: {ENCODING_WORDS.get(r['strategy'], r['strategy'])}")
    for r in pre["outliers"]:
        if r["strategy"] == "keep":
            continue
        words = {"iqr": f"rows beyond {r['threshold']}×IQR removed from the training data",
                 "z_score": f"rows beyond {r['threshold']} standard deviations removed from the training data",
                 "winsorize": f"values clipped to {r['threshold']}×IQR fences",
                 "isolation_forest": "outlier rows removed from the training data (Isolation Forest)"}
        items.append(f"Outliers in “{r['column']}”: {words.get(r['strategy'], r['strategy'])}")
    items.append(f"Numeric features: {SCALING_WORDS.get(pre['scaling'], pre['scaling'])}"
                 + (f" (columns: {', '.join(pre['scale_columns'])})" if pre["scale_columns"] else ""))
    items.append("Exact duplicate rows were " + ("removed." if pre["drop_duplicates"] else "kept."))
    out += bullets(items)
    out.append(RP("<b>No data leakage:</b> every step that learns from data (filling, encoding, scaling, outlier limits) was fitted "
                  "on training rows only and then applied to the evaluation rows."))
    log = [s for s in c.result.preparation_log if s["rows_removed"] > 0 or s["step"] != "drop_missing_target"]
    if log:
        out += [P("Row-level preparation", "h2"),
                table([["Step", "Rows before", "Rows after", "Removed", "Detail"]] +
                      [[s["step"].replace("_", " "), f"{s['rows_before']:,}", f"{s['rows_after']:,}", f"{s['rows_removed']:,}", s["detail"][:50]] for s in log],
                      [45 * mm, 28 * mm, 28 * mm, 24 * mm, CONTENT_W - 125 * mm], right=(1, 2, 3))]
    return out


def s_features(c: ReportContext, n: int) -> list[Flowable]:
    fe = c.config.feature_engineering.model_dump(mode="json")
    items = [f"“{r['column']}”: {r['transform']} transform added as “{r['column']}__{r['transform']}”" for r in fe["numeric_transforms"]]
    items += [f"“{r['column']}”: date parts extracted ({', '.join(r['extract'])}); the original column is replaced" for r in fe["date_features"]]
    return [section(n, "Feature engineering")] + (bullets(items) if items else [P("No feature engineering was applied.")])


def s_split(c: ReportContext, n: int) -> list[Flowable]:
    sp, rs = c.config.split.model_dump(mode="json"), c.result.split
    rows: list[tuple[str, Any]] = [("Method", SPLIT_WORDS.get(sp["method"], sp["method"])), ("Random seed", sp["random_state"])]
    if rs["mode"] == "holdout":
        rows += [("Training rows", f"{rs['n_train']:,}"), ("Test rows", f"{rs['n_test']:,}")]
        if rs.get("n_validation"):
            rows.insert(3, ("Validation rows", f"{rs['n_validation']:,}"))
    else:
        rows += [("Folds", rs["n_folds"]), ("Rows per fold (train / test)", f"{rs['n_train']:,} / {rs['n_test']:,} (first fold)")]
    rows += [("Stratified by target", "yes" if rs.get("stratified") else "no"),
             ("Chronological", f"yes, ordered by “{sp['time_column']}”" if sp.get("time_column") or rs.get("chronological") else "no")]
    out: list[Flowable] = [section(n, "Dataset split"), kv(rows)]
    out += [P(x, "small") for x in rs.get("notes", [])]
    return out


def s_models(c: ReportContext, n: int) -> list[Flowable]:
    rows = [["Model", "What it does", "Speed", "Explainable", "Needs scaling"]]
    for m in c.result.models:
        spec = get_spec(m.model_key)
        rows.append([m.name, DESCRIPTIONS.get(m.model_key, ""), {"low": "fast", "medium": "medium", "high": "slow"}[spec.cost],
                     spec.interpretability, "yes" if spec.requires_scaling else "no"])
    return [section(n, "Models"), table(rows, [34 * mm, CONTENT_W - 34 * mm - 70 * mm, 17 * mm, 24 * mm, 29 * mm])]


def s_hyper(c: ReportContext, n: int) -> list[Flowable]:
    out: list[Flowable] = [section(n, "Regularization and hyperparameters")]
    cfgs = {m.model_key: m for m in c.config.models}
    task = c.config.dataset.task
    for m in c.result.models:
        spec, mc = get_spec(m.model_key), cfgs.get(m.model_key)
        reg = spec.regularization[task]
        if reg.kind == "penalty":
            r = (mc.regularization if mc else {}) or {}
            reg_text = (f"{(r.get('type') or reg.options[0]).upper()} penalty"
                        + (f", strength {r['strength']}" if "strength" in r else "") + (f", l1_ratio {r['l1_ratio']}" if "l1_ratio" in r else ""))
        elif reg.kind == "complexity":
            reg_text = "Controlled by complexity limits: " + ", ".join(reg.complexity_params)
        else:
            reg_text = "None: this model has no regularization."
        source = ("NoCodeML recommended starting values" if mc and mc.use_recommended_defaults else "library defaults") + (
            ", with manual overrides" if mc and mc.hyperparameters else "")
        out.append(KeepTogether([
            P(m.name, "h2"), P(f"Regularization: {reg_text}"), P(f"Values: {source}", "small"),
            table([["Parameter", "Value"]] + [[k, "none" if v is None else v] for k, v in m.hyperparameters.items()], [60 * mm, 60 * mm]),
        ]))
    return out


def s_training(c: ReportContext, n: int) -> list[Flowable]:
    r = c.result
    out: list[Flowable] = [section(n, "Training information"), kv([
        ("Started", when(r.started_at)), ("Finished", when(r.finished_at)), ("Random seed", r.random_state),
        ("Configuration hash", r.config_hash), ("Environment", " · ".join(f"{k} {v}" for k, v in r.environment.items())),
    ]), Spacer(1, 6)]
    out.append(table([["Model", "Training time", "Rows used to fit", "Outlier rows removed"]] +
                     [[m.name, f"{m.fit_seconds}s", f"{m.n_rows_fitted:,}", m.n_outlier_rows_removed] for m in r.models],
                     [50 * mm, 30 * mm, 40 * mm, CONTENT_W - 120 * mm], right=(1, 2, 3)))
    return out


def _metric_rows(c: ReportContext, source: str) -> list[list[Any]]:
    cols = _metric_cols(c.result.task)
    rows: list[list[Any]] = [["Model"] + [lab for _, lab in cols]]
    for m in c.result.models:
        mm_ = m.metrics.get(source, {})
        rows.append([m.name] + [fnum(mm_.get(k)) if k != "mape" or mm_.get(k) is None else f"{100 * mm_[k]:.1f}%" for k, _ in cols])
    b = c.result.models[0].baseline
    rows.append(["Baseline (" + ("most common class" if c.result.task == "classification" else "the average") + ")"] +
                [fnum(b.get(k)) if k in b else "–" for k, _ in cols])
    return rows


def s_metrics(c: ReportContext, n: int) -> list[Flowable]:
    r = c.result
    out: list[Flowable] = [section(n, "Evaluation metrics")]
    cols = _metric_cols(r.task)
    w = [CONTENT_W - len(cols) * 22 * mm] + [22 * mm] * len(cols)
    for source in _sources(r.models):
        out += [P(f"Evaluated on the {SOURCE_WORDS[source]}", "h2"), table(_metric_rows(c, source), w, right=range(1, len(cols) + 1))]
    out.append(Spacer(1, 4))
    out.append(P("Metric guide: " + ("Accuracy is the share predicted correctly and can mislead when one class dominates. Precision: of rows "
                 "predicted positive, how many were. Recall: of real positives, how many were found. F1 balances the two. ROC-AUC "
                 "measures ranking quality (0.5 is a coin flip)." if r.task == "classification" else
                 "R² is the share of variation explained (0 equals predicting the average). RMSE and MAE are typical error sizes in "
                 "the target's own units (lower is better); MAPE is the average percentage error."), "small"))
    pm = r.models[0].primary_metric
    nm = {"f1": "F1", "r2": "R²"}.get(pm, pm)
    rows = [["Model", f"{nm} on training rows", f"{nm} held out", "Gap"]]
    for m in r.models:
        tr = num(m.metrics.get("train", {}).get(pm))
        he = num((m.metrics.get("test") or m.metrics.get("cv") or {}).get(pm))
        rows.append([m.name, fnum(tr), fnum(he), fnum(tr - he) if tr is not None and he is not None else "–"])
    out += [P("Overfitting check", "h2"), table(rows, [CONTENT_W - 90 * mm, 30 * mm, 30 * mm, 30 * mm], right=(1, 2, 3)),
            P("A large gap means the model scores much better on rows it trained on than on new rows, a sign of memorization.", "small")]
    folds = [m for m in r.models if m.fold_primary_scores]
    if folds:
        out += [P(f"{nm} per cross-validation fold", "h2"),
                table([["Model", "Folds", "Mean", "Std dev"]] + [[m.name, ", ".join(f"{x:.3f}" for x in m.fold_primary_scores),
                      f3(sum(m.fold_primary_scores) / len(m.fold_primary_scores)),
                      f3((sum((x - sum(m.fold_primary_scores) / len(m.fold_primary_scores)) ** 2 for x in m.fold_primary_scores) / len(m.fold_primary_scores)) ** 0.5)] for m in folds],
                      [38 * mm, CONTENT_W - 38 * mm - 40 * mm, 20 * mm, 20 * mm], right=(2, 3))]
    return out


def s_visuals(c: ReportContext, n: int) -> list[Flowable]:
    r = c.result
    src = _chart_source(r.models)
    out: list[Flowable] = [section(n, "Visualizations"), P(f"All charts use the {SOURCE_WORDS[src]}.", "small")]
    half = (CONTENT_W - 6 * mm) / 2
    for m in r.models:
        mt = m.metrics.get(src, {})
        pair: list[Any] = []
        if r.task == "classification":
            cm = mt.get("confusion_matrix")
            if cm:
                pair.append(image(charts.confusion_matrix_png(cm["labels"], cm["matrix"]), half))
            roc = mt.get("roc_curve")
            pair.append(image(charts.roc_png(roc["fpr"], roc["tpr"], num(mt.get("roc_auc"))), half) if roc
                        else P("The ROC curve is shown for two-class problems only.", "small"))
        else:
            avp = mt.get("actual_vs_predicted")
            if avp:
                pair.append(image(charts.scatter_png(avp["actual"], avp["predicted"]), half))
            if mt.get("residuals"):
                pair.append(image(charts.histogram_png(mt["residuals"]), half))
        if pair:
            out.append(KeepTogether([P(m.name, "h2"), Table([pair + [""] * (2 - len(pair))], colWidths=[half + 3 * mm] * 2,
                                                             style=[("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)])]))
    return out


def s_comparison(c: ReportContext, n: int) -> list[Flowable]:
    r = c.result
    src = _chart_source(r.models)
    metrics = [("f1", "F1"), ("roc_auc", "ROC-AUC")] if r.task == "classification" else [("r2", "R²"), ("rmse", "RMSE")]
    out: list[Flowable] = [section(n, "Model comparison")]
    names = [m.name for m in r.models]
    for key, label in metrics:
        vals = [num(m.metrics.get(src, {}).get(key)) for m in r.models]
        if all(v is None for v in vals):
            continue
        out.append(KeepTogether([image(charts.comparison_png(names, vals, label, num(r.models[0].baseline.get(key)), key in LOWER_BETTER), CONTENT_W * 0.8)]))
    lines = []
    for key, label in _metric_cols(r.task):
        vals = [(num(m.metrics.get(src, {}).get(key)), m.name) for m in r.models]
        vals = [(v, nm) for v, nm in vals if v is not None]
        if len(vals) > 1:
            v, nm = (min if key in LOWER_BETTER else max)(vals)
            lines.append(f"{label}: {nm} ({fnum(v)})")
    if lines:
        out += [P("Best value per metric", "h2")] + bullets(lines)
    out.append(P("There is deliberately no overall “best model”: the right choice depends on your goal and on which errors cost more. "
                 "Read the metrics, the charts and the overfitting check together.", "small"))
    return out


def s_health(c: ReportContext, n: int) -> list[Flowable]:
    q = c.result.quality
    order = {"fail": 0, "warn": 1, "pass": 2}
    checks = sorted((x for x in q.checks if x.status in order), key=lambda x: order[x.status])
    names = {m.model_key: m.name for m in c.result.models}
    rows: list[list[Any]] = [["Status", "Check", "Detail"]]
    for x in checks:
        title = x.title if not x.model_key or names.get(x.model_key, "") in x.title else f"{x.title} ({names.get(x.model_key)})"
        rows.append([Paragraph(f"<font color='{STATUS_COLOR[x.status].hexval().replace('0x', '#')}'><b>{x.status.upper()}</b></font>", ST["cell"]),
                     title, x.detail])
    return [section(n, "Pipeline health"),
            P(f"Quality score: {q.score if q.score is not None else 'n/a'} / 100. This measures whether the experiment was built correctly "
              "(no leakage, a proper split, a baseline comparison), not how accurate the models are.", "body"),
            table(rows, [16 * mm, 55 * mm, CONTENT_W - 71 * mm])]


def s_config(c: ReportContext, n: int) -> list[Flowable]:
    text = json.dumps(c.config.model_dump(mode="json"), indent=1, ensure_ascii=False)
    return [PageBreak(), section(n, "Final pipeline configuration"),
            P("The complete configuration this experiment ran on. Together with the dataset fingerprint, random seed and library "
              "versions listed above, it is enough to reproduce the run.", "small"),
            Preformatted(text, ST["mono"], maxLineLength=130)]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_report(ctx: ReportContext) -> bytes:
    """Render the report to PDF bytes. Deterministic for identical inputs."""
    name = ctx.project_name if len(ctx.project_name) <= 50 else ctx.project_name[:49] + "…"
    footer = f"NoCodeML · {name} · Experiment #{ctx.experiment_number} · pipeline {ctx.pipeline_label}"   # canvas text: not markup
    buf = io.BytesIO()
    doc = BaseDocTemplate(buf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=16 * mm, bottomMargin=20 * mm,
                          title=f"NoCodeML report: {ctx.project_name}"[:100], author="NoCodeML", subject=f"Experiment #{ctx.experiment_number}",
                          invariant=1, pageCompression=1)
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(MARGIN, 20 * mm, CONTENT_W, PAGE_H - 36 * mm, id="f", leftPadding=0,
                                                              rightPadding=0, topPadding=0, bottomPadding=0)])])
    story: list[Flowable] = s_cover(ctx)
    n = 0
    for fn in [s_dataset, s_statistics, s_dataset_charts, s_preprocessing, s_features, s_split, s_models, s_hyper, s_training,
               s_metrics, s_visuals, s_comparison, s_health, s_config]:
        part = fn(ctx, n + 1)
        if part:  # optional sections (dataset charts) leave no gap in the numbering when absent
            n += 1
            story += part
    doc.build(story, canvasmaker=_NumberedCanvasFactory(footer))
    return buf.getvalue()
