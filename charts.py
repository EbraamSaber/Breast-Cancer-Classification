"""
Figures for the dashboard.

Every function here draws the same chart type, on the same data, with the same
labels and analytical meaning as the notebook. Only the presentation changes:
colours come from the active theme palette, and Plotly adds hover tooltips.
No figure here computes new statistics; they receive values from ml_pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from design import FONT_STACK
from ml_pipeline import METRICS

CONFIG = {"displayModeBar": False, "responsive": True}
CLASS_LABELS = ["Benign", "Malignant"]


# --------------------------------------------------------------------------- #
# Shared styling
# --------------------------------------------------------------------------- #
def _rgba(color: str, alpha: float) -> str:
    c = color.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _base(p: dict, height: int, legend: bool = False, margin: dict | None = None) -> dict:
    layout = dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, size=12, color=p["text_2"]),
        margin=margin or dict(l=56, r=24, t=44, b=52),
        hoverlabel=dict(
            bgcolor=p["surface_2"],
            bordercolor=p["border_strong"],
            font=dict(family=FONT_STACK, color=p["text"], size=12),
        ),
        showlegend=legend,
    )
    if legend:
        layout["legend"] = dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(color=p["text_2"], size=11),
        )
    return layout


def _axes(fig, p: dict, grid_x: bool = False, grid_y: bool = True) -> None:
    axis_common = dict(
        zeroline=False,
        linecolor=p["border"],
        linewidth=1,
        ticks="outside",
        tickcolor=p["border"],
        tickfont=dict(color=p["muted"], size=11),
        title_font=dict(color=p["muted"], size=12),
    )
    fig.update_xaxes(showgrid=grid_x, gridcolor=p["grid"], **axis_common)
    fig.update_yaxes(showgrid=grid_y, gridcolor=p["grid"], **axis_common)


def _style_subplot_titles(fig, p: dict) -> None:
    fig.update_annotations(font=dict(color=p["text"], size=13, family=FONT_STACK))


# --------------------------------------------------------------------------- #
# Task 2 / 3: data quality and EDA
# --------------------------------------------------------------------------- #
def missing_values(counts: pd.Series, pct: pd.Series, p: dict) -> go.Figure:
    """Notebook 3.1: missing values per feature (count and percentage)."""
    labels = [c.replace("_", " ") for c in counts.index]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Missing values per feature (count)",
                                        "Missing values per feature (percentage)"),
                        horizontal_spacing=0.16)
    _style_subplot_titles(fig, p)
    fig.add_trace(go.Bar(
        x=list(counts.values), y=labels, orientation="h", marker_color=p["series_a"],
        text=[f"{int(v):,}" for v in counts.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x} missing<extra></extra>", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=list(pct.values), y=labels, orientation="h", marker_color=p["series_b"],
        text=[f"{v:.2f}%" for v in pct.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.2f}% of rows<extra></extra>", showlegend=False,
    ), row=1, col=2)
    fig.update_layout(**_base(p, 380, margin=dict(l=8, r=40, t=56, b=48)))
    _axes(fig, p, grid_x=True, grid_y=False)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Number of missing values", row=1, col=1)
    fig.update_xaxes(title_text="Percentage of rows (%)", row=1, col=2)
    return fig


def class_distribution(train_counts: pd.Series, p: dict) -> go.Figure:
    """Notebook 3.2: target class distribution, training set."""
    values = [int(train_counts.get(0, 0)), int(train_counts.get(1, 0))]
    fig = go.Figure(go.Bar(
        x=["Benign (0)", "Malignant (1)"], y=values,
        marker=dict(color=[p["benign"], p["malignant"]], line=dict(width=0)),
        text=[f"{v}" for v in values], textposition="outside", cliponaxis=False, width=0.5,
        hovertemplate="%{x}: %{y} samples<extra></extra>",
    ))
    fig.update_layout(**_base(p, 360))
    _axes(fig, p, grid_x=False, grid_y=True)
    fig.update_xaxes(title_text="Tumor class")
    fig.update_yaxes(title_text="Number of samples")
    return fig


def feature_histograms(X_train: pd.DataFrame, p: dict) -> go.Figure:
    """Notebook 3.3: 3x3 histograms of the nine features (discrete 1-10 bins)."""
    feats = list(X_train.columns)
    titles = [f.replace("_", " ") for f in feats]
    fig = make_subplots(rows=3, cols=3, subplot_titles=titles,
                        horizontal_spacing=0.08, vertical_spacing=0.15)
    _style_subplot_titles(fig, p)
    for i, col in enumerate(feats):
        r, c = divmod(i, 3)
        counts = X_train[col].value_counts().reindex(range(1, 11), fill_value=0)
        fig.add_trace(go.Bar(
            x=list(counts.index), y=list(counts.values), marker_color=p["series_a"],
            width=0.8, showlegend=False,
            hovertemplate="Score %{x}: %{y} samples<extra></extra>",
        ), row=r + 1, col=c + 1)
    fig.update_layout(**_base(p, 880, margin=dict(l=56, r=20, t=56, b=48)))
    _axes(fig, p, grid_x=False, grid_y=True)
    fig.update_xaxes(dtick=1, range=[0.4, 10.6])
    for c in range(1, 4):
        fig.update_xaxes(title_text="Score (1–10)", row=3, col=c)
    for r in range(1, 4):
        fig.update_yaxes(title_text="Frequency", row=r, col=1)
    return fig


def boxplots_by_class(X_train: pd.DataFrame, y_train: pd.Series, p: dict) -> go.Figure:
    """Notebook 3.4: feature distributions by class (boxplots, training set)."""
    feats = list(X_train.columns)
    titles = [f.replace("_", " ") for f in feats]
    fig = make_subplots(rows=3, cols=3, subplot_titles=titles,
                        horizontal_spacing=0.08, vertical_spacing=0.15)
    _style_subplot_titles(fig, p)
    for i, col in enumerate(feats):
        r, c = divmod(i, 3)
        for label, cls, color in (("Benign", 0, p["benign"]), ("Malignant", 1, p["malignant"])):
            vals = X_train.loc[y_train == cls, col].dropna()
            fig.add_trace(go.Box(
                y=list(vals), x=[label] * len(vals), name=label,
                marker=dict(color=color, size=4), line=dict(color=color, width=1.6),
                fillcolor=_rgba(color, 0.18), boxmean=False,
                showlegend=(i == 0), legendgroup=label,
                hovertemplate=f"{label}: %{{y}}<extra></extra>",
            ), row=r + 1, col=c + 1)
    fig.update_layout(**_base(p, 900, legend=True, margin=dict(l=56, r=20, t=80, b=48)))
    _axes(fig, p, grid_x=False, grid_y=True)
    for c in range(1, 4):
        fig.update_xaxes(title_text="Tumor class", row=3, col=c)
    for r in range(1, 4):
        fig.update_yaxes(title_text="Score (1–10)", row=r, col=1)
    return fig


def correlation_heatmap(corr: pd.DataFrame, p: dict) -> go.Figure:
    """Notebook 3.5: Pearson correlation heatmap of features and target (training set)."""
    cols = list(corr.columns)
    z = corr.values
    fig = go.Figure(go.Heatmap(
        z=z, x=cols, y=cols, zmin=-1, zmax=1, colorscale=p["corr_scale"],
        colorbar=dict(
            title=dict(text="Pearson correlation", font=dict(color=p["muted"], size=12)),
            tickfont=dict(color=p["muted"], size=11), thickness=14, outlinewidth=0,
        ),
        hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
    ))
    for i, row_label in enumerate(cols):
        for j, col_label in enumerate(cols):
            v = float(z[i, j])
            fig.add_annotation(
                x=col_label, y=row_label, text=f"{v:.2f}", showarrow=False,
                font=dict(size=11, color="#FFFFFF" if abs(v) >= 0.5 else p["text"], family=FONT_STACK),
            )
    fig.update_layout(**_base(p, 640, margin=dict(l=12, r=12, t=20, b=12)))
    fig.update_xaxes(showgrid=False, tickangle=-35, tickfont=dict(color=p["text_2"], size=11), linecolor=p["border"])
    fig.update_yaxes(showgrid=False, autorange="reversed", tickfont=dict(color=p["text_2"], size=11), linecolor=p["border"])
    return fig


def standardization_boxplots(X_before: pd.DataFrame, X_after: pd.DataFrame, p: dict) -> go.Figure:
    """Notebook 3.6 (a): feature boxplots before and after standardization."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.17,
                        subplot_titles=("Before standardization (imputed training data)",
                                        "After standardization (z-score, fitted on training data)"))
    _style_subplot_titles(fig, p)
    for col in X_before.columns:
        name = col.replace("_", " ")
        fig.add_trace(go.Box(
            x=list(X_before[col].dropna()), name=name, orientation="h",
            marker=dict(color=p["series_a"], size=3), line=dict(color=p["series_a"], width=1.5),
            fillcolor=_rgba(p["series_a"], 0.2), showlegend=False,
            hovertemplate=f"{name}: %{{x:.2f}}<extra></extra>",
        ), row=1, col=1)
    for col in X_after.columns:
        name = col.replace("_", " ")
        fig.add_trace(go.Box(
            x=list(X_after[col].dropna()), name=name, orientation="h",
            marker=dict(color=p["series_b"], size=3), line=dict(color=p["series_b"], width=1.5),
            fillcolor=_rgba(p["series_b"], 0.2), showlegend=False,
            hovertemplate=f"{name}: %{{x:.2f}}<extra></extra>",
        ), row=1, col=2)
    fig.update_layout(**_base(p, 460, margin=dict(l=8, r=24, t=56, b=48)))
    _axes(fig, p, grid_x=True, grid_y=False)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Feature value", row=1, col=1)
    fig.update_xaxes(title_text="Standardized value", row=1, col=2)
    return fig


def _kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gaussian KDE with Scott's rule, the same default bandwidth seaborn uses."""
    n = len(values)
    if n < 2:
        return np.zeros_like(grid)
    bw = values.std(ddof=1) * n ** (-0.2)
    if bw == 0:
        bw = 1e-3
    diff = (grid[:, None] - values[None, :]) / bw
    return np.exp(-0.5 * diff ** 2).sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))


def bare_nuclei_standardization(before: pd.Series, after: pd.Series, feat: str, p: dict) -> go.Figure:
    """Notebook 3.6 (b): histogram + KDE of one feature before and after scaling."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=(f"{feat}: before standardization",
                                        f"{feat}: after standardization"))
    _style_subplot_titles(fig, p)
    panels = ((before, p["series_a"], "Original value"), (after, p["series_b"], "Standardized value"))
    for k, (series, color, xtitle) in enumerate(panels, start=1):
        vals = np.asarray(series.dropna(), dtype=float)
        counts, edges = np.histogram(vals, bins=10)
        width = float(edges[1] - edges[0])
        centers = (edges[:-1] + edges[1:]) / 2
        fig.add_trace(go.Bar(
            x=centers, y=counts, width=width * 0.92, marker=dict(color=_rgba(color, 0.5),
                                                              line=dict(color=color, width=1)),
            showlegend=False, hovertemplate="%{x:.2f}: %{y} samples<extra></extra>",
        ), row=1, col=k)
        grid = np.linspace(edges[0], edges[-1], 240)
        density = _kde(vals, grid) * len(vals) * width
        fig.add_trace(go.Scatter(
            x=grid, y=density, mode="lines", line=dict(color=color, width=2.6),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=k)
        fig.update_xaxes(title_text=xtitle, row=1, col=k)
        fig.update_yaxes(title_text="Frequency", row=1, col=k)
    fig.update_layout(**_base(p, 380, margin=dict(l=56, r=24, t=56, b=48)))
    _axes(fig, p, grid_x=False, grid_y=True)
    return fig


# --------------------------------------------------------------------------- #
# Task 5: evaluation
# --------------------------------------------------------------------------- #
def _cm_cell_text(fig, cm: np.ndarray, p: dict, xref: str, yref: str) -> None:
    labels = CLASS_LABELS
    top = max(float(cm.max()), 1.0)
    for i in range(2):
        for j in range(2):
            v = int(cm[i, j])
            color = p["cm_text_hi"] if v / top >= 0.55 else p["cm_text_lo"]
            fig.add_annotation(
                x=labels[j], y=labels[i], xref=xref, yref=yref, showarrow=False,
                text=f"<b>{v}</b>", font=dict(size=18, color=color, family=FONT_STACK),
            )


def confusion_matrix_figure(cm: np.ndarray, name: str, p: dict, height: int = 380) -> go.Figure:
    """Notebook 5.1 for one model: confusion matrix on the test set (positive = malignant)."""
    top = max(float(cm.max()), 1.0)
    fig = go.Figure(go.Heatmap(
        z=cm, x=CLASS_LABELS, y=CLASS_LABELS, zmin=0, zmax=top,
        colorscale=p["cm_scale"], showscale=False,
        hovertemplate="True %{y} → predicted %{x}: %{z}<extra>" + name + "</extra>",
    ))
    _cm_cell_text(fig, cm, p, "x", "y")
    fig.update_layout(**_base(p, height, margin=dict(l=8, r=8, t=16, b=8)))
    fig.update_xaxes(title_text="Predicted label", side="bottom", showgrid=False,
                     tickfont=dict(color=p["text_2"], size=12), linecolor=p["border"])
    fig.update_yaxes(title_text="True label", autorange="reversed", showgrid=False,
                     tickfont=dict(color=p["text_2"], size=12), linecolor=p["border"])
    return fig


def confusion_grid(cms: dict, p: dict) -> go.Figure:
    """Notebook 5.1: the five confusion matrices side by side, as in the notebook."""
    names = list(cms)
    fig = make_subplots(rows=1, cols=len(names), subplot_titles=names, horizontal_spacing=0.06)
    _style_subplot_titles(fig, p)
    for k, name in enumerate(names, start=1):
        cm = cms[name]
        top = max(float(cm.max()), 1.0)
        fig.add_trace(go.Heatmap(
            z=cm, x=CLASS_LABELS, y=CLASS_LABELS, zmin=0, zmax=top,
            colorscale=p["cm_scale"], showscale=False,
            hovertemplate="True %{y} → predicted %{x}: %{z}<extra>" + name + "</extra>",
        ), row=1, col=k)
        xref = "x" if k == 1 else f"x{k}"
        yref = "y" if k == 1 else f"y{k}"
        _cm_cell_text(fig, cm, p, xref, yref)
    fig.update_layout(**_base(p, 340, margin=dict(l=8, r=8, t=56, b=40)))
    fig.update_xaxes(showgrid=False, tickfont=dict(color=p["text_2"], size=11),
                     linecolor=p["border"], title_text="Predicted label")
    fig.update_yaxes(showgrid=False, autorange="reversed", tickfont=dict(color=p["text_2"], size=11),
                     linecolor=p["border"], title_text="True label")
    return fig


def roc_figure(rocs: dict, results: pd.DataFrame, names: list[str], p: dict, height: int = 480) -> go.Figure:
    """Notebook 5.2: ROC curves on the test set, with AUC in the legend."""
    fig = go.Figure()
    for name in names:
        fpr, tpr, _ = rocs[name]
        auc = float(results.loc[name, "ROC-AUC"])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} (AUC = {auc:.3f})",
            line=dict(color=p["models"][name], width=2.6),
            hovertemplate=f"{name}<br>FPR %{{x:.3f}} · TPR %{{y:.3f}}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
        line=dict(color=p["muted"], width=1.3, dash="dash"), hoverinfo="skip",
    ))
    fig.update_layout(**_base(p, height, legend=False, margin=dict(l=56, r=24, t=24, b=52)))
    fig.update_layout(legend=dict(
        x=0.985, y=0.02, xanchor="right", yanchor="bottom", bgcolor=p["surface"],
        bordercolor=p["border"], borderwidth=1, font=dict(color=p["text_2"], size=11),
    ), showlegend=True)
    _axes(fig, p, grid_x=True, grid_y=True)
    fig.update_xaxes(title_text="False Positive Rate (1 - Specificity)", range=[-0.01, 1.01])
    fig.update_yaxes(title_text="True Positive Rate (Recall)", range=[-0.01, 1.02])
    return fig


def metric_bars(results: pd.DataFrame, names: list[str], p: dict, height: int = 420) -> go.Figure:
    """Notebook 5.3: grouped bar chart of all test-set metrics."""
    fig = go.Figure()
    for metric in METRICS:
        fig.add_trace(go.Bar(
            x=names, y=list(results.loc[names, metric].values), name=metric,
            marker=dict(color=p["metrics"][metric], line=dict(width=0)),
            hovertemplate=f"%{{x}} · {metric}: %{{y:.4f}}<extra></extra>",
        ))
    lo = max(0.0, float(results.loc[names].values.min()) - 0.05)
    fig.update_layout(**_base(p, height, legend=True, margin=dict(l=56, r=20, t=52, b=52)),
                      barmode="group", bargap=0.24, bargroupgap=0.06)
    _axes(fig, p, grid_x=False, grid_y=True)
    fig.update_yaxes(range=[lo, 1.01], title_text="Score")
    fig.update_xaxes(title_text="Model")
    return fig


def error_comparison(cms: dict, names: list[str], p: dict, height: int = 380) -> go.Figure:
    """False negatives (missed malignant) and false positives (benign flagged), per model."""
    fn = [int(cms[n][1, 0]) for n in names]
    fp = [int(cms[n][0, 1]) for n in names]
    fig = go.Figure([
        go.Bar(x=names, y=fn, name="False negatives (malignant missed)",
               marker=dict(color=p["malignant"], line=dict(width=0)),
               text=fn, textposition="outside", cliponaxis=False,
               hovertemplate="%{x}: %{y} missed<extra></extra>"),
        go.Bar(x=names, y=fp, name="False positives (benign flagged)",
               marker=dict(color=p["series_a"], line=dict(width=0)),
               text=fp, textposition="outside", cliponaxis=False,
               hovertemplate="%{x}: %{y} false alarms<extra></extra>"),
    ])
    fig.update_layout(**_base(p, height, legend=True, margin=dict(l=56, r=20, t=52, b=52)),
                      barmode="group", bargap=0.28, bargroupgap=0.08)
    _axes(fig, p, grid_x=False, grid_y=True)
    fig.update_yaxes(title_text="Count (test set)", rangemode="tozero")
    fig.update_xaxes(title_text="Model")
    return fig


def repeated_cv_chart(rep_cv: pd.DataFrame, names: list[str], p: dict, height: int = 400) -> go.Figure:
    """Repeated 5x3 stratified CV on the training set: mean with standard deviation."""
    fig = go.Figure()
    series = (
        ("Recall", "recall_mean", "recall_std", p["metrics"]["Recall"]),
        ("F1-score", "f1_mean", "f1_std", p["metrics"]["F1-score"]),
        ("ROC-AUC", "auc_mean", "auc_std", p["metrics"]["ROC-AUC"]),
    )
    for label, mean_col, std_col, color in series:
        fig.add_trace(go.Bar(
            x=names, y=list(rep_cv.loc[names, mean_col].values), name=label,
            marker=dict(color=color, line=dict(width=0)),
            error_y=dict(type="data", array=list(rep_cv.loc[names, std_col].values),
                         color=p["text_2"], thickness=1.2, width=4),
            hovertemplate=f"%{{x}} · {label}: %{{y:.4f}}<extra></extra>",
        ))
    low = float((rep_cv.loc[names, "recall_mean"] - rep_cv.loc[names, "recall_std"]).min())
    fig.update_layout(**_base(p, height, legend=True, margin=dict(l=56, r=20, t=52, b=52)),
                      barmode="group", bargap=0.24, bargroupgap=0.06)
    _axes(fig, p, grid_x=False, grid_y=True)
    fig.update_yaxes(range=[max(0.0, low - 0.04), 1.0], title_text="Mean ± std (5 × 3 CV)")
    fig.update_xaxes(title_text="Model")
    return fig


# --------------------------------------------------------------------------- #
# Try a prediction
# --------------------------------------------------------------------------- #
def probability_bars(probs: dict, p: dict, height: int = 340) -> go.Figure:
    names = list(probs)
    vals = [float(probs[n]) for n in names]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=[p["models"][n] for n in names], line=dict(width=0)),
        text=[f"{v * 100:.1f}%" for v in vals], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%} malignant<extra></extra>",
    ))
    fig.update_layout(**_base(p, height, margin=dict(l=8, r=48, t=28, b=44)))
    _axes(fig, p, grid_x=True, grid_y=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=p["text_2"], size=12))
    fig.update_xaxes(range=[0, 1.14], tickformat=".0%", title_text="P(malignant) from predict_proba")
    fig.add_shape(type="line", x0=0.5, x1=0.5, y0=0, y1=1, yref="paper",
                  line=dict(color=p["muted"], width=1.2, dash="dash"))
    fig.add_annotation(x=0.5, y=1, yref="paper", yanchor="bottom", showarrow=False,
                       text="0.5 threshold", font=dict(color=p["muted"], size=11))
    return fig


def probability_gauge(prob: float, malignant: bool, p: dict, height: int = 230) -> go.Figure:
    bar_color = p["malignant"] if malignant else p["benign"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(prob) * 100,
        number=dict(suffix="%", valueformat=".1f",
                    font=dict(size=34, color=p["text"], family=FONT_STACK)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=0, tickcolor=p["muted"],
                      tickfont=dict(color=p["muted"], size=10)),
            bar=dict(color=bar_color, thickness=0.28),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[dict(range=[0, 50], color=_rgba(p["benign"], 0.16)),
                   dict(range=[50, 100], color=_rgba(p["malignant"], 0.16))],
            threshold=dict(line=dict(color=p["text"], width=2), thickness=0.8, value=50),
        ),
    ))
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=24, r=24, t=16, b=0), font=dict(family=FONT_STACK, color=p["text_2"]))
    return fig


def profile_radar(profile: dict, class_means: pd.DataFrame, features: list[str], p: dict,
                  height: int = 420) -> go.Figure:
    """The entered profile against the average Benign and Malignant training values."""
    labels = [f.replace("_", " ") for f in features]
    closed = labels + labels[:1]

    def loop(values):
        vals = [float(v) for v in values]
        return vals + vals[:1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=loop([class_means.loc[0, f] for f in features]), theta=closed, name="Benign average",
        line=dict(color=p["benign"], width=2), fill="none",
        hovertemplate="Benign average · %{theta}: %{r:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=loop([class_means.loc[1, f] for f in features]), theta=closed, name="Malignant average",
        line=dict(color=p["malignant"], width=2), fill="none",
        hovertemplate="Malignant average · %{theta}: %{r:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=loop([profile[f] for f in features]), theta=closed, name="Entered profile",
        line=dict(color=p["accent"], width=2.6), fill="toself", fillcolor=_rgba(p["accent"], 0.16),
        hovertemplate="Entered · %{theta}: %{r}<extra></extra>",
    ))
    fig.update_layout(
        **_base(p, height, legend=True, margin=dict(l=48, r=48, t=28, b=48)),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 10], tickvals=[2, 4, 6, 8, 10], gridcolor=p["grid"],
                            linecolor=p["grid"], tickfont=dict(color=p["muted"], size=9)),
            angularaxis=dict(gridcolor=p["grid"], linecolor=p["grid"],
                             tickfont=dict(color=p["text_2"], size=10)),
        ),
    )
    return fig
