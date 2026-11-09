
"""
Tumour Classifier · Breast Cancer Wisconsin (Original)

Run with:  streamlit run app.py

Layers
  ml_pipeline.py  data loading, cleaning, tuning and evaluation
  charts.py       Plotly figures built from pipeline outputs
  design.py       palettes, global CSS and reusable HTML components
  app.py          page composition, navigation and the light / dark theme switch
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import charts as C
import design as D
from ml_pipeline import METRICS, MODEL_NAMES, predict_one

PAGES = ["Overview", "Data & EDA", "Model Performance", "Model Comparison", "Try a Prediction"]

METRIC_HINTS = {
    "Accuracy": "Share of all correct predictions",
    "Precision": "Predicted malignant that really were",
    "Recall": "Malignant cases that were caught",
    "F1-score": "Balance of precision and recall",
    "ROC-AUC": "Ranking quality across all thresholds",
}

st.set_page_config(
    page_title="Tumour Classifier · Breast Cancer Wisconsin",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Cached pre-trained artifacts
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_all() -> dict:
    """Load pre-trained models and results without retraining."""
    artifacts_path = Path(__file__).parent / "model_artifacts.joblib"

    if not artifacts_path.exists():
        raise FileNotFoundError(
            "model_artifacts.joblib not found. "
            "Please generate it using build_artifacts.py."
        )

    return joblib.load(artifacts_path)

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def fmt4(value) -> str:
    return "—" if pd.isna(value) else f"{float(value):.4f}"


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def chart_card(title: str, caption: str, fig) -> None:
    with st.container(border=True):
        html(D.chart_heading(title, caption))
        st.plotly_chart(fig, use_container_width=True, config=C.CONFIG)


def best_model_names(results: pd.DataFrame, names: list[str], metric: str) -> list[str]:
    top = results.loc[names, metric].max()
    return [n for n in names if results.loc[n, metric] == top]


# --------------------------------------------------------------------------- #
# Page 1 · Overview
# --------------------------------------------------------------------------- #
def page_overview(art: dict, p: dict) -> None:
    stats, results, sel = art["stats"], art["results"], art["selected"]
    row = results.loc[sel]
    cm = art["cms"][sel]
    fn = int(cm[1, 0])
    test_mal = stats["test_malignant"]

    html(D.hero(
        "Supervised classification · Breast Cancer Wisconsin (Original)",
        "Benign or malignant? Five classifiers, one rigorous comparison.",
        "Each model is tuned with stratified cross-validation on the training data, then scored "
        "once on a held-out test set. Malignant is the positive class, and recall comes first, "
        "because a missed malignant case is the costliest error.",
        chips=[f"{stats['rows_after_dedup']} clean samples", "9 cytological features",
               "5 tuned models", f"Selected: {sel}"],
    ))

    html(D.kpi_row([
        {"label": "Clean samples", "value": f"{stats['rows_after_dedup']:,}",
         "sub": f"{stats['duplicates_removed']} exact duplicates removed"},
        {"label": "Test set", "value": f"{stats['test_rows']}",
         "sub": f"{test_mal} malignant · stratified 80:20 split"},
        {"label": "Recall · selected", "value": fmt4(row["Recall"]),
         "sub": f"{fn} of {test_mal} malignant cases missed", "meter": float(row["Recall"])},
        {"label": "F1-score · selected", "value": fmt4(row["F1-score"]),
         "sub": "Harmonic mean of precision and recall", "meter": float(row["F1-score"])},
        {"label": "ROC-AUC · selected", "value": fmt4(row["ROC-AUC"]),
         "sub": "Threshold-free ranking quality", "meter": float(row["ROC-AUC"])},
    ]))

    cv_f1 = next(t["best_cv_f1"] for t in art["tuning"] if t["Model"] == sel)
    params = next(t["best_params"] for t in art["tuning"] if t["Model"] == sel)

    left, right = st.columns([5, 7], gap="large")
    with left:
        html(D.model_card(
            sel,
            f"Ranked first by the notebook's priority: Recall, then F1-score, ROC-AUC and Accuracy. "
            f"Best cross-validated F1 during tuning: {cv_f1:.4f}.",
            {m: float(row[m]) for m in METRICS},
            params,
        ))
    with right:
        rank_df = art["ranking"][METRICS].copy()
        rank_df.insert(0, "Rank", range(1, len(rank_df) + 1))
        with st.container(border=True):
            html(D.chart_heading(
                "Model ranking · test set",
                "Sorted by Recall, then F1-score, ROC-AUC and Accuracy. Dots mark the best value "
                "in each column, and the selected model is highlighted.",
            ))
            html(D.table_html(
                rank_df, index_label="Model",
                best={m: "max" for m in METRICS}, highlight_row=sel,
            ))

    html(D.section("01", "Final results on the held-out test set",
                   "All five models scored once on the 139-sample test set."))
    chart_card("Test-set metrics by model",
               "Grouped bars for each metric, as in the notebook's comparison chart. Hover for exact values.",
               C.metric_bars(results, MODEL_NAMES, p, height=400))

    html(D.section("02", "How the result was produced",
                   "Every step runs on the training data except the final evaluation."))
    eda = art["eda"]
    html(D.flow([
        ("01", "Raw file", f"{stats['rows_raw']} rows, 11 columns, {int(eda['question_marks']['Bare_Nuclei'])} '?' values in Bare Nuclei"),
        ("02", "Clean", f"{stats['duplicates_removed']} exact duplicates removed, ID dropped, malignant coded as 1"),
        ("03", "Split 80:20", f"Stratified · {stats['train_rows']} training and {stats['test_rows']} test samples"),
        ("04", "Impute and scale", "Median imputation and z-scoring, fitted on training folds only"),
        ("05", "Tune", "GridSearchCV · 5-fold stratified CV · F1 on the malignant class"),
        ("06", "Evaluate once", f"Held-out test set of {stats['test_rows']} samples, used a single time"),
    ]))

    html(D.section("✓", "Reproduction check",
                   "This run is compared with every number the notebook prints: data counts, tuned "
                   "parameters, test metrics, confusion matrices, repeated cross-validation and correlations."))
    ver = art["verification"]
    ok, total = int(ver["Match"].sum()), len(ver)
    if ok == total:
        html(D.note(f"<strong>All {total} values match the notebook.</strong> The dataset and pipeline "
                    "reproduce the printed results at the precision the notebook shows."))
    else:
        html(D.note(f"<strong>{ok} of {total} values match.</strong> Differences usually come from a "
                    "different copy of the dataset or from newer library versions. Expand the table "
                    "below to see which values moved."))
    with st.expander("All reproduction checks"):
        view = ver.set_index("Check")[["Group", "Notebook", "This run", "Match"]]
        html(D.table_html(
            view, index_label="Check", right=("Notebook", "This run"),
            formats={"Match": lambda v: "✓ match" if v else "✗ differs"},
        ))


# --------------------------------------------------------------------------- #
# Page 2 · Data & EDA
# --------------------------------------------------------------------------- #
def page_eda(art: dict, p: dict) -> None:
    eda, stats, df = art["eda"], art["stats"], art["df"]
    X_train, y_train = art["X_train"], art["y_train"]

    html(D.page_intro(
        2, "Data & exploratory analysis",
        "The original file has no header row and marks missing values with '?'. This section checks "
        "data quality first, then studies distributions, relationships and scaling. Exploration uses "
        "the training set, so the test set stays untouched.",
    ))

    cls_clean = eda["class_counts_clean"]
    malignant_share = float(cls_clean.get(1, 0)) / float(cls_clean.sum())
    html(D.kpi_row([
        {"label": "Raw records", "value": f"{stats['rows_raw']}", "sub": "11 columns, no header row"},
        {"label": "Duplicate records", "value": f"{eda['dup_full']}", "sub": "Exact duplicates, ID included"},
        {"label": "Missing values", "value": f"{int(eda['missing_count'].sum())}",
         "sub": f"All in Bare Nuclei · {eda['missing_pct']['Bare_Nuclei']:.2f}% of rows"},
        {"label": "Malignant share", "value": f"{malignant_share:.1%}",
         "sub": f"{int(cls_clean.get(1, 0))} of {int(cls_clean.sum())} after cleaning"},
        {"label": "Train / test", "value": f"{stats['train_rows']} / {stats['test_rows']}",
         "sub": "Stratified 80:20 split"},
    ]))

    tab_quality, tab_dist, tab_rel, tab_scale = st.tabs(
        ["Data quality", "Distributions", "Relationships", "Standardization"]
    )

    # ---- Data quality ----------------------------------------------------- #
    with tab_quality:
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            miss = eda["missing_all_columns"]
            miss_df = pd.DataFrame({
                "Missing count": miss.astype(int),
                "Missing %": miss / len(df) * 100,
            })
            with st.container(border=True):
                html(D.chart_heading("Missing values by column",
                                     "Only Bare Nuclei has gaps, all of them written as '?' in the raw file."))
                html(D.table_html(miss_df, index_label="Column",
                                  formats={"Missing %": lambda v: f"{v:.2f}%"}))
        with c2:
            html(D.metric_tile("Fully duplicated rows", str(eda["dup_full"]),
                               "Identical in every column, ID included"))
            html(D.metric_tile("Repeated sample IDs", str(eda["dup_ids"]),
                               "The same patient sampled more than once"))
            html(D.metric_tile("Duplicates ignoring the ID", str(eda["dup_no_id"]),
                               "Identical measurements and label"))

        c3, c4 = st.columns([2, 3], gap="large")
        with c3:
            raw_counts = eda["class_counts_raw"]
            n_raw = int(raw_counts.sum())
            cls_df = pd.DataFrame(
                {"Count": raw_counts.values.astype(int), "Percent": raw_counts.values / n_raw * 100},
                index=["2 · Benign", "4 · Malignant"],
            )
            with st.container(border=True):
                html(D.chart_heading("Class distribution · full file",
                                     "Raw labels: 2 = benign, 4 = malignant."))
                html(D.table_html(
                    cls_df, index_label="Class", right=("Count", "Percent"),
                    formats={"Count": lambda v: f"{int(v):,}", "Percent": lambda v: f"{v:.2f}%"},
                ))
        with c4:
            chart_card("Target distribution · training set",
                       "Count of benign and malignant samples in the training split.",
                       C.class_distribution(eda["class_counts_train"], p))

        desc = eda["describe"]
        formats = {c: (lambda v: f"{v:,.0f}") if c == "count" else (lambda v: f"{v:,.2f}")
                   for c in desc.columns}
        with st.container(border=True):
            html(D.chart_heading("Descriptive statistics · full file",
                                 "Summary of every column before cleaning, including the ID."))
            html(D.table_html(desc, index_label="Column", formats=formats))

        with st.container(border=True):
            html(D.note(
                "<strong>Note on duplicates.</strong> The notebook's text describes removing the "
                f"{eda['dup_no_id']} records that match once the ID is ignored, but its code removes "
                f"only the {eda['dup_full']} exact duplicates. The dashboard follows the code, which "
                "produced the printed results. Changing this would change the research."
            ))

    # ---- Distributions ----------------------------------------------------- #
    with tab_dist:
        share_one = float((X_train["Mitoses"] == 1).mean())
        chart_card(
            "Feature distributions · training set",
            f"Number of samples at each score from 1 to 10. Distributions are strongly right-skewed: "
            f"{share_one:.0%} of training samples score 1 for Mitoses.",
            C.feature_histograms(X_train, p),
        )
        html(D.note(
            "<strong>Reading the shapes.</strong> Most features pile up at low scores with a long tail "
            "towards 10. The scale is ordinal and far from normal, which is why the pipeline uses median "
            "imputation rather than assuming a bell curve."
        ))

    # ---- Relationships ----------------------------------------------------- #
    with tab_rel:
        chart_card(
            "Feature distributions by class · training set",
            "Boxplots of each feature for benign and malignant samples. Points beyond the whiskers are "
            "valid clinical scores, so the notebook keeps them.",
            C.boxplots_by_class(X_train, y_train, p),
        )
        c1, c2 = st.columns([7, 5], gap="large")
        with c1:
            chart_card("Correlation heatmap · features and target",
                       "Pearson correlation on the training set. Class is the target, with 1 = malignant.",
                       C.correlation_heatmap(eda["corr"], p))
        with c2:
            tc = pd.DataFrame({"Correlation with Class": eda["target_corr"]})
            signs = "All correlations are positive" if (tc.iloc[:, 0] > 0).all() else "Correlations have mixed signs"
            with st.container(border=True):
                html(D.chart_heading("Correlation with the target",
                                     f"Sorted from strongest to weakest. {signs}."))
                html(D.table_html(
                    tc, index_label="Feature",
                    formats={"Correlation with Class": lambda v: f"{v:.3f}"},
                    best={"Correlation with Class": "max"},
                ))

    # ---- Standardization --------------------------------------------------- #
    with tab_scale:
        chart_card(
            "Feature boxplots · before and after standardization",
            "Left: imputed training data on its original scales. Right: the same features after "
            "z-scoring with the training mean and standard deviation.",
            C.standardization_boxplots(eda["X_train_imp"], eda["X_train_scaled"], p),
        )
        chart_card(
            "Bare Nuclei · histogram and density curve",
            "Standardization shifts and rescales the axis but keeps the shape of the distribution.",
            C.bare_nuclei_standardization(eda["X_train_imp"]["Bare_Nuclei"],
                                          eda["X_train_scaled"]["Bare_Nuclei"], "Bare Nuclei", p),
        )
        scale_tbl = pd.DataFrame({
            "Train mean": eda["scaled_train_stats"].loc["mean"],
            "Train std": eda["scaled_train_stats"].loc["std"],
            "Test mean": eda["scaled_test_stats"].loc["mean"],
            "Test std": eda["scaled_test_stats"].loc["std"],
        })
        with st.container(border=True):
            html(D.chart_heading(
                "Scaled statistics",
                "Training data is centred at 0 with standard deviation 1. Test data is close to, but not "
                "exactly, 0 and 1, because the scaler learned only from training data.",
            ))
            html(D.table_html(scale_tbl, index_label="Feature",
                              formats={c: (lambda v: f"{v:+.3f}") for c in scale_tbl.columns}))


# --------------------------------------------------------------------------- #
# Page 3 · Model performance
# --------------------------------------------------------------------------- #
def page_performance(art: dict, p: dict) -> None:
    sel = art["selected"]
    html(D.page_intro(
        3, "Model performance",
        "Choose a model to inspect its held-out test metrics, confusion matrix, ROC curve and the "
        "hyperparameters chosen by cross-validation. The selected model is marked.",
    ))

    model = st.selectbox(
        "Model", MODEL_NAMES, index=MODEL_NAMES.index(sel), key="perf_model",
        format_func=lambda n: f"{n}  ·  selected" if n == sel else n,
    )
    row = art["results"].loc[model]
    cm = art["cms"][model]
    fn, tp_total = int(cm[1, 0]), art["stats"]["test_malignant"]

    tiles = st.columns(5)
    for col, metric in zip(tiles, METRICS):
        with col:
            html(D.metric_tile(metric, fmt4(row[metric]), METRIC_HINTS[metric], float(row[metric])))

    c1, c2 = st.columns(2, gap="large")
    with c1:
        chart_card("Confusion matrix · test set",
                   f"Rows are the true class, columns the prediction. {plural(fn, 'malignant case')} "
                   f"missed (bottom-left) out of {tp_total}.",
                   C.confusion_matrix_figure(cm, model, p, height=360))
    with c2:
        chart_card("ROC curve · test set",
                   f"AUC = {row['ROC-AUC']:.4f}. The closer the curve hugs the top-left corner, "
                   "the better the separation.",
                   C.roc_figure(art["rocs"], art["results"], [model], p, height=360))

    c3, c4 = st.columns([5, 6], gap="large")
    tuning = next(t for t in art["tuning"] if t["Model"] == model)
    with c3:
        ph = pd.DataFrame({"Value": {k: str(v) for k, v in tuning["best_params"].items()}})
        with st.container(border=True):
            html(D.chart_heading(
                "Best hyperparameters",
                f"Chosen by GridSearchCV with 5-fold stratified CV on the training set. "
                f"Best CV F1: {tuning['best_cv_f1']:.4f}.",
            ))
            html(D.table_html(ph, index_label="Parameter", mono=("Value",), wrap=("Value",)))

    with c4:
        rep = art["reports"][model]
        rows = {}
        for key in ["Benign (0)", "Malignant (1)"]:
            rows[key] = [rep[key]["precision"], rep[key]["recall"], rep[key]["f1-score"], rep[key]["support"]]
        rows["Accuracy"] = [np.nan, np.nan, rep["accuracy"], rep["macro avg"]["support"]]
        rows["Macro average"] = [rep["macro avg"][k] for k in ("precision", "recall", "f1-score", "support")]
        rows["Weighted average"] = [rep["weighted avg"][k] for k in ("precision", "recall", "f1-score", "support")]
        rep_df = pd.DataFrame.from_dict(rows, orient="index",
                                        columns=["Precision", "Recall", "F1-score", "Support"])
        with st.container(border=True):
            html(D.chart_heading("Classification report · test set",
                                 "Per-class precision, recall and F1, plus averages. Support is the number of test samples."))
            html(D.table_html(
                rep_df, index_label="Class",
                formats={
                    "Precision": fmt4, "Recall": fmt4, "F1-score": fmt4,
                    "Support": lambda v: f"{int(v):,}",
                },
            ))


# --------------------------------------------------------------------------- #
# Page 4 · Model comparison
# --------------------------------------------------------------------------- #
def page_comparison(art: dict, p: dict) -> None:
    results, sel = art["results"], art["selected"]
    ranking = art["ranking"]
    html(D.page_intro(
        4, "Model comparison",
        "Compare the five tuned models side by side. Narrow the selection and the curves, metrics and "
        "rankings update. The confusion matrices always show all five, as in the notebook.",
    ))

    chosen = st.multiselect("Models in comparison", MODEL_NAMES, default=MODEL_NAMES, key="cmp_models")

    if not chosen:
        html(D.empty_state("No models selected", "Choose at least one model to see its curves and metrics."))
    else:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            chart_card("ROC curves · all selected models",
                       "Test-set ROC curves with AUC in the legend, as in the notebook.",
                       C.roc_figure(art["rocs"], results, chosen, p, height=460))
        with c2:
            chart_card("Test-set metrics · selected models",
                       "Accuracy, precision, recall, F1-score and ROC-AUC for each chosen model.",
                       C.metric_bars(results, chosen, p, height=460))

        c3, c4 = st.columns(2, gap="large")
        with c3:
            chart_card("False negatives and false positives",
                       "False negatives are malignant cases predicted as benign, the most costly error here.",
                       C.error_comparison(art["cms"], chosen, p, height=380))
        with c4:
            chart_card("Repeated cross-validation · training set",
                       "Mean and standard deviation over 5 folds repeated 3 times. Lower spread means a more stable model.",
                       C.repeated_cv_chart(art["rep_cv"], chosen, p, height=380))

        rep = art["rep_cv"].loc[chosen]
        rep_df = pd.DataFrame({
            "Recall": [f"{rep.loc[n, 'recall_mean']:.4f} ± {rep.loc[n, 'recall_std']:.4f}" for n in chosen],
            "F1-score": [f"{rep.loc[n, 'f1_mean']:.4f} ± {rep.loc[n, 'f1_std']:.4f}" for n in chosen],
            "ROC-AUC": [f"{rep.loc[n, 'auc_mean']:.4f} ± {rep.loc[n, 'auc_std']:.4f}" for n in chosen],
        }, index=chosen)
        with st.container(border=True):
            html(D.chart_heading("Repeated cross-validation · mean ± std",
                                 "The training-set check that guards against trusting a single test split."))
            html(D.table_html(rep_df, index_label="Model", right=tuple(rep_df.columns)))

        st.write("")
        full_rank = {m: i + 1 for i, m in enumerate(ranking.index)}
        sub = ranking.loc[[m for m in ranking.index if m in chosen]]
        rank_df = sub[["Recall", "F1-score", "ROC-AUC", "Accuracy"]].copy()
        rank_df.insert(0, "Rank", [full_rank[m] for m in rank_df.index])
        left, right = st.columns([7, 5], gap="large")
        with left:
            with st.container(border=True):
                html(D.chart_heading(
                    "Model ranking · test set",
                    "Priority: Recall, then F1-score, ROC-AUC and Accuracy. Rank is taken from the full five-model ranking.",
                ))
                html(D.table_html(rank_df, index_label="Model",
                                  best={c: "max" for c in ["Recall", "F1-score", "ROC-AUC", "Accuracy"]},
                                  highlight_row=sel if sel in chosen else None))
        with right:
            chips = []
            for metric in METRICS:
                winners = best_model_names(results, chosen, metric)
                value = results.loc[winners[0], metric]
                chips.append(
                    f'<span class="ds-chip"><b>{D.esc(metric)}</b>&nbsp;·&nbsp;{D.esc(", ".join(winners))} '
                    f'({value:.4f})</span>'
                )
            with st.container(border=True):
                html(D.chart_heading("Best model per metric", "Among the models currently selected. Ties are listed together."))
                html('<div class="ds-chip-row" style="margin-top:.4rem">' + "".join(chips) + "</div>")

    # Interpretation, computed from the results so it always matches the numbers.
    sel_row = results.loc[sel]
    runner_up = ranking.index[1]
    fn_sel = int(art["cms"][sel][1, 0])
    fn_next = int(art["cms"][runner_up][1, 0])
    n_mal = art["stats"]["test_malignant"]
    n_test = art["stats"]["test_rows"]
    html(D.section("→", "Interpretation",
                   "Written from the computed results, so the wording always matches the numbers."))
    html(D.note(
        f"<strong>{D.esc(sel)}</strong> ranks first with recall {sel_row['Recall']:.4f}, F1 "
        f"{sel_row['F1-score']:.4f} and ROC-AUC {sel_row['ROC-AUC']:.4f}. It misses {plural(fn_sel, 'malignant case')} "
        f"of {n_mal}. The runner-up, {D.esc(runner_up)}, misses {plural(fn_next, 'case')}. With {n_test} test "
        f"samples, one case moves recall by about {1 / n_mal:.3f}, so gaps between the top models are small. "
        "Read them with caution, and prefer the simpler model when performance is tied."
    ))
    chart_card("Confusion matrices · all five models",
               "Test-set confusion matrices, always shown in full. Bottom-left is missed malignant cases.",
               C.confusion_grid(art["cms"], p))


# --------------------------------------------------------------------------- #
# Page 5 · Try a prediction
# --------------------------------------------------------------------------- #
def page_predict(art: dict, p: dict) -> None:
    features, X_train, sel = art["features"], art["X_train"], art["selected"]
    eda = art["eda"]
    html(D.page_intro(
        5, "Try a prediction",
        "Set the nine cytological scores and see what each model predicts. The label comes from "
        "predict() and the probability from predict_proba(), exactly as in the notebook. Sliders start "
        "at the training-set median.",
    ))

    medians = {f: int(round(float(X_train[f].median()))) for f in features}
    col_in, col_out = st.columns([5, 6], gap="large")

    with col_in:
        with st.container(border=True):
            html(D.chart_heading("Cell measurements",
                                 "Scores from 1 (close to normal) to 10 (most abnormal), as in the UCI documentation."))
            unknown = st.checkbox("Bare Nuclei is unknown", key="pred_unknown",
                                  help="Missing values are imputed with the training median, as in the pipeline.")
            values: dict[str, float] = {}
            grid = st.columns(2)
            for i, feat in enumerate(features):
                with grid[i % 2]:
                    values[feat] = st.slider(
                        feat.replace("_", " "), 1, 10, medians[feat], key=f"pred_{feat}",
                        disabled=(feat == "Bare_Nuclei" and unknown),
                    )
            if unknown:
                values["Bare_Nuclei"] = np.nan
            model_choice = st.selectbox(
                "Model for the headline result", MODEL_NAMES,
                index=MODEL_NAMES.index(sel), key="pred_model",
                format_func=lambda n: f"{n}  ·  selected" if n == sel else n,
            )

    preds = {name: predict_one(art["best_models"][name], values, features) for name in MODEL_NAMES}
    label, prob = preds[model_choice]

    with col_out:
        verdict = "Malignant" if label == 1 else "Benign"
        css_cls = "malignant" if label == 1 else "benign"
        with st.container(border=True):
            html(
                f'<div class="ds-eyebrow ds-eyebrow-accent">Prediction · {D.esc(model_choice)}</div>'
                f'<div class="ds-verdict {css_cls}">{verdict}</div>'
                f'<div class="ds-kpi-sub">predict() returned class {label}. P(malignant) = {prob:.1%} '
                "from predict_proba().</div>"
            )
            st.plotly_chart(C.probability_gauge(prob, label == 1, p), use_container_width=True, config=C.CONFIG)

        chart_card("All five models",
                   "Probability of malignancy from each tuned model. Bars right of the dashed line are predicted malignant.",
                   C.probability_bars({n: preds[n][1] for n in MODEL_NAMES}, p, height=320))

        with st.container(border=True):
            html(D.chart_heading("Predictions by model", "Label from predict(), probability from predict_proba()."))
            pred_df = pd.DataFrame({
                "Prediction": [("Malignant" if preds[n][0] == 1 else "Benign") for n in MODEL_NAMES],
                "P(malignant)": [preds[n][1] for n in MODEL_NAMES],
            }, index=MODEL_NAMES)
            html(D.table_html(pred_df, index_label="Model",
                              formats={"P(malignant)": lambda v: f"{v:.1%}"}))

    st.write("")
    class_means = eda["class_means"]
    chart_card("Your profile against the training averages",
               "Entered scores (magenta) compared with the average Benign and Malignant training values.",
               C.profile_radar(values, class_means, features, p, height=440))

    html(D.note(
        "<strong>Educational use only.</strong> This model was built on a published research dataset to "
        "compare classifiers. It is not a diagnostic tool and must not inform medical decisions."
    ))


# --------------------------------------------------------------------------- #
# Shell: theme, navigation, pipeline state
# --------------------------------------------------------------------------- #
def main() -> None:
    theme = "light" if st.session_state.get("light_mode", False) else "dark"
    p = D.palette(theme)
    html(f"<style>{D.css(theme)}</style>")

    with st.sidebar:
        html(D.sidebar_brand())
        st.toggle("Light theme", key="light_mode", help="Switch the whole interface between dark and light.")
        html('<div class="ds-side-label">Navigate</div>')
        page = st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")

    try:
        with st.spinner("Tuning the five notebook pipelines. The first load takes about a minute."):
            art = load_all()
    except Exception as exc:  # show a readable message instead of a stack trace
        st.error(f"The analysis pipeline could not run: {exc}")
        st.info("Place breast-cancer-wisconsin.data next to app.py, or make sure this machine can reach "
                "the UCI repository.")
        st.stop()

    with st.sidebar:
        html(D.sidebar_footer(art["selected"], art["stats"]["test_rows"], art["stats"]["rows_after_dedup"]))

    if page == "Overview":
        page_overview(art, p)
    elif page == "Data & EDA":
        page_eda(art, p)
    elif page == "Model Performance":
        page_performance(art, p)
    elif page == "Model Comparison":
        page_comparison(art, p)
    else:
        page_predict(art, p)


main()
