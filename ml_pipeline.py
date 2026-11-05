"""
Data and modelling pipeline for the Breast Cancer Wisconsin (Original) project.

This module is a faithful port of the notebook: loading, cleaning, the stratified
80:20 split, the five tuned pipelines, the single test-set evaluation, the repeated
cross-validation check and the ranking rule. It contains no presentation code.

NOTEBOOK_REFERENCE stores the outputs printed by the notebook, so the dashboard can
prove that a fresh run reproduces the research (see verify_against_notebook).
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")  # same as the notebook

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
CV_REPEATS = 3
LOCAL_FILE = "breast-cancer-wisconsin.data"
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "breast-cancer-wisconsin/breast-cancer-wisconsin.data"
)
COLUMNS = [
    "Sample_code_number", "Clump_Thickness", "Uniformity_of_Cell_Size",
    "Uniformity_of_Cell_Shape", "Marginal_Adhesion", "Single_Epithelial_Cell_Size",
    "Bare_Nuclei", "Bland_Chromatin", "Normal_Nucleoli", "Mitoses", "Class",
]
FEATURES = COLUMNS[1:-1]  # the 9 cytological measurements
MODEL_NAMES = ["KNN", "Decision Tree", "Random Forest", "ANN (MLP)", "SVM"]
METRICS = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
RANK_KEYS = ["Recall", "F1-score", "ROC-AUC", "Accuracy"]  # notebook ranking priority
CLASS_NAMES = ["Benign (0)", "Malignant (1)"]

# Values printed by the notebook (source of truth for the reproduction check).
NOTEBOOK_REFERENCE = {
    "rows_raw": 699,
    "rows_after_dedup": 691,
    "train_rows": 552,
    "test_rows": 139,
    "test_malignant": 48,
    "raw_class_counts": {2: 458, 4: 241},
    "bare_nuclei_question_marks": 16,
    "bare_nuclei_median_train": 1.0,
    "duplicates": {"full": 8, "repeated_ids": 54, "without_id": 236},
    "train_missing": 14,
    "test_missing": 2,
    "cv_best_f1": {
        "KNN": 0.9555, "Decision Tree": 0.9297, "Random Forest": 0.9512,
        "ANN (MLP)": 0.9530, "SVM": 0.9503,
    },
    "best_params": {
        "KNN": {"n_neighbors": 7, "p": 2, "weights": "uniform"},
        "Decision Tree": {"criterion": "gini", "max_depth": 4,
                          "min_samples_leaf": 2, "min_samples_split": 2},
        "Random Forest": {"max_depth": None, "max_features": "sqrt",
                          "min_samples_leaf": 2, "n_estimators": 300},
        "ANN (MLP)": {"activation": "relu", "alpha": 0.0001,
                      "hidden_layer_sizes": (32,), "learning_rate_init": 0.001},
        "SVM": {"C": 0.1, "gamma": "scale", "kernel": "linear"},
    },
    # Accuracy, Precision, Recall, F1-score, ROC-AUC
    "test_metrics": {
        "KNN": (0.9640, 0.9778, 0.9167, 0.9462, 0.9823),
        "Decision Tree": (0.9281, 0.8958, 0.8958, 0.8958, 0.9755),
        "Random Forest": (0.9568, 0.9565, 0.9167, 0.9362, 0.9938),
        "ANN (MLP)": (0.9712, 0.9583, 0.9583, 0.9583, 0.9961),
        "SVM": (0.9640, 0.9574, 0.9375, 0.9474, 0.9966),
    },
    # TN, FP, FN, TP
    "confusion": {
        "KNN": (90, 1, 4, 44), "Decision Tree": (86, 5, 5, 43),
        "Random Forest": (89, 2, 4, 44), "ANN (MLP)": (89, 2, 2, 46),
        "SVM": (89, 2, 3, 45),
    },
    # (mean, std) for Recall, F1 and ROC-AUC from repeated 5x3 CV on the training set
    "repeated_cv": {
        "KNN": ((0.9561, 0.0314), (0.9503, 0.0256), (0.9888, 0.0108)),
        "Decision Tree": ((0.9228, 0.0548), (0.9160, 0.0318), (0.9592, 0.0222)),
        "Random Forest": ((0.9632, 0.0285), (0.9498, 0.0228), (0.9909, 0.0058)),
        "ANN (MLP)": ((0.9632, 0.0316), (0.9523, 0.0252), (0.9927, 0.0056)),
        "SVM": ((0.9579, 0.0330), (0.9520, 0.0267), (0.9941, 0.0042)),
    },
    "target_correlation": {
        "Bare_Nuclei": 0.826, "Uniformity_of_Cell_Shape": 0.815,
        "Uniformity_of_Cell_Size": 0.810, "Bland_Chromatin": 0.741,
        "Normal_Nucleoli": 0.729, "Clump_Thickness": 0.714,
        "Marginal_Adhesion": 0.692, "Single_Epithelial_Cell_Size": 0.678,
        "Mitoses": 0.428,
    },
    "selected_model": "ANN (MLP)",
}


# --------------------------------------------------------------------------- #
# 1. Loading and cleaning (Task 1 and 2 of the notebook)
# --------------------------------------------------------------------------- #
def resolve_source() -> str:
    """Local file next to this module or in the working folder, otherwise UCI."""
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(here, LOCAL_FILE), os.path.join(os.getcwd(), LOCAL_FILE)):
        if os.path.exists(candidate):
            return candidate
    return UCI_URL


def load_raw() -> tuple[pd.DataFrame, str]:
    source = resolve_source()
    raw = pd.read_csv(source, header=None, names=COLUMNS)  # '?' kept as text
    return raw, source


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.replace("?", np.nan)
    df["Bare_Nuclei"] = pd.to_numeric(df["Bare_Nuclei"])
    return df


def prepare(df: pd.DataFrame):
    """Remove exact duplicates, drop the ID and encode malignant (4) as 1."""
    df_nodup = df.drop_duplicates().reset_index(drop=True)
    data = df_nodup.drop(columns="Sample_code_number").copy()
    data["Class"] = data["Class"].map({2: 0, 4: 1})
    X = data.drop(columns="Class")
    y = data["Class"]
    return X, y, df_nodup


def split(X: pd.DataFrame, y: pd.Series):
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


# --------------------------------------------------------------------------- #
# 2. Exploratory statistics (Task 3, computed exactly as in the notebook)
# --------------------------------------------------------------------------- #
def build_eda(raw, df, y_all, X_train, y_train, X_test) -> dict:
    feature_df = df.drop(columns=["Sample_code_number", "Class"])
    missing_count = feature_df.isnull().sum()

    eda = {
        "question_marks": (raw == "?").sum(),
        "describe": df.describe().T,
        "missing_all_columns": df.isnull().sum(),
        "missing_count": missing_count,
        "missing_pct": missing_count / len(df) * 100,
        "dup_full": int(df.duplicated().sum()),
        "dup_ids": int(df["Sample_code_number"].duplicated().sum()),
        "dup_no_id": int(df.drop(columns="Sample_code_number").duplicated().sum()),
        "class_counts_raw": df["Class"].value_counts().sort_index(),
        "class_counts_clean": y_all.value_counts().sort_index(),
        "class_counts_train": y_train.value_counts().sort_index(),
        "train_missing": int(X_train.isnull().sum().sum()),
        "test_missing": int(X_test.isnull().sum().sum()),
    }

    # Standardisation visual: imputer and scaler fitted on TRAIN only (notebook cell 24)
    imputer_vis = SimpleImputer(strategy="median").fit(X_train)
    cols = X_train.columns
    X_train_imp = pd.DataFrame(imputer_vis.transform(X_train), columns=cols, index=X_train.index)
    X_test_imp = pd.DataFrame(imputer_vis.transform(X_test), columns=cols, index=X_test.index)
    scaler_vis = StandardScaler().fit(X_train_imp)
    X_train_scaled = pd.DataFrame(scaler_vis.transform(X_train_imp), columns=cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler_vis.transform(X_test_imp), columns=cols, index=X_test.index)

    eda.update({
        "bare_nuclei_median": float(imputer_vis.statistics_[list(cols).index("Bare_Nuclei")]),
        "X_train_imp": X_train_imp,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaled_train_stats": X_train_scaled.describe().loc[["mean", "std"]],
        "scaled_test_stats": X_test_scaled.describe().loc[["mean", "std"]],
        "class_means": X_train.groupby(y_train).mean(),
    })

    corr = X_train.assign(Class=y_train).corr()
    eda["corr"] = corr
    eda["target_corr"] = corr["Class"].drop("Class").sort_values(ascending=False).round(3)
    return eda


# --------------------------------------------------------------------------- #
# 3. Models and hyperparameter tuning (Task 4)
# --------------------------------------------------------------------------- #
def make_pipe(model, scale: bool) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", model))
    return Pipeline(steps)


def model_specs() -> dict:
    return {
        "KNN": (
            make_pipe(KNeighborsClassifier(), scale=True),
            {"clf__n_neighbors": [3, 5, 7, 9, 11, 15],
             "clf__weights": ["uniform", "distance"],
             "clf__p": [1, 2]},
        ),
        "Decision Tree": (
            make_pipe(DecisionTreeClassifier(random_state=RANDOM_STATE), scale=False),
            {"clf__criterion": ["gini", "entropy"],
             "clf__max_depth": [2, 3, 4, 5, 6, 8, None],
             "clf__min_samples_split": [2, 5, 10],
             "clf__min_samples_leaf": [1, 2, 4]},
        ),
        "Random Forest": (
            make_pipe(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), scale=False),
            {"clf__n_estimators": [100, 200, 300],
             "clf__max_depth": [None, 5, 10],
             "clf__min_samples_leaf": [1, 2],
             "clf__max_features": ["sqrt", "log2"]},
        ),
        "ANN (MLP)": (
            make_pipe(MLPClassifier(max_iter=2000, random_state=RANDOM_STATE), scale=True),
            {"clf__hidden_layer_sizes": [(16,), (32,), (32, 16)],
             "clf__activation": ["relu", "tanh"],
             "clf__alpha": [1e-4, 1e-3, 1e-2],
             "clf__learning_rate_init": [1e-3, 1e-2]},
        ),
        "SVM": (
            make_pipe(SVC(probability=True, random_state=RANDOM_STATE), scale=True),
            {"clf__kernel": ["linear", "rbf"],
             "clf__C": [0.1, 1, 10, 100],
             "clf__gamma": ["scale", "auto"]},
        ),
    }


def tune_models(X_train, y_train):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    best_models, tuning = {}, []
    for name, (pipe, grid) in model_specs().items():
        gs = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)  # TRAIN ONLY
        best_models[name] = gs.best_estimator_
        tuning.append({
            "Model": name,
            "best_cv_f1": float(gs.best_score_),
            "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
        })
    return best_models, tuning


# --------------------------------------------------------------------------- #
# 4. Evaluation on the held-out test set (Task 5)
# --------------------------------------------------------------------------- #
def get_scores(model, X):
    """Continuous score for the positive class (used for ROC-AUC)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(best_models, X_test, y_test):
    rows, cms, rocs, reports = [], {}, {}, {}
    for name, model in best_models.items():
        y_pred = model.predict(X_test)
        y_score = get_scores(model, X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, pos_label=1),
            "Recall": recall_score(y_test, y_pred, pos_label=1),
            "F1-score": f1_score(y_test, y_pred, pos_label=1),
            "ROC-AUC": roc_auc_score(y_test, y_score),
        })
        cms[name] = confusion_matrix(y_test, y_pred, labels=[0, 1])
        rocs[name] = roc_curve(y_test, y_score)
        reports[name] = classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, digits=4, output_dict=True
        )
    # The notebook rounds the results table to 4 decimals before ranking; keep that.
    results = pd.DataFrame(rows).set_index("Model").round(4)
    return results, cms, rocs, reports


def repeated_cv(best_models, X_train, y_train) -> pd.DataFrame:
    rcv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
    rows = []
    for name, model in best_models.items():
        res = cross_validate(model, X_train, y_train, cv=rcv, n_jobs=-1,
                             scoring={"recall": "recall", "f1": "f1", "roc_auc": "roc_auc"})
        rows.append({
            "Model": name,
            "recall_mean": float(res["test_recall"].mean()),
            "recall_std": float(res["test_recall"].std()),
            "f1_mean": float(res["test_f1"].mean()),
            "f1_std": float(res["test_f1"].std()),
            "auc_mean": float(res["test_roc_auc"].mean()),
            "auc_std": float(res["test_roc_auc"].std()),
        })
    return pd.DataFrame(rows).set_index("Model")


def rank_models(results: pd.DataFrame) -> pd.DataFrame:
    return results.sort_values(RANK_KEYS, ascending=False)


def predict_one(model, values: dict, features: list[str]):
    """Return (predicted label via .predict, P(malignant) via .predict_proba)."""
    row = pd.DataFrame([[values[f] for f in features]], columns=features)
    label = int(model.predict(row)[0])
    proba = float(model.predict_proba(row)[0, 1])
    return label, proba


# --------------------------------------------------------------------------- #
# 5. Orchestration
# --------------------------------------------------------------------------- #
def run_pipeline() -> dict:
    raw, source = load_raw()
    df = clean(raw)
    X, y, df_nodup = prepare(df)
    X_train, X_test, y_train, y_test = split(X, y)

    eda = build_eda(raw, df, y, X_train, y_train, X_test)
    best_models, tuning = tune_models(X_train, y_train)
    results, cms, rocs, reports = evaluate(best_models, X_test, y_test)
    rep_cv = repeated_cv(best_models, X_train, y_train)
    ranking = rank_models(results)

    stats = {
        "source": source,
        "rows_raw": len(raw),
        "rows_after_dedup": len(X),
        "duplicates_removed": len(df) - len(X),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_malignant": int(y_train.sum()),
        "test_malignant": int(y_test.sum()),
        "train_malignant_share": float(y_train.mean()),
        "test_malignant_share": float(y_test.mean()),
        "features": list(X.columns),
    }

    return {
        "stats": stats,
        "raw": raw,
        "df": df,
        "df_nodup": df_nodup,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "features": list(X.columns),
        "eda": eda,
        "tuning": tuning,
        "best_models": best_models,
        "results": results,
        "cms": cms,
        "rocs": rocs,
        "reports": reports,
        "rep_cv": rep_cv,
        "ranking": ranking,
        "selected": str(ranking.index[0]),
    }


# --------------------------------------------------------------------------- #
# 6. Reproduction check against the notebook's printed outputs
# --------------------------------------------------------------------------- #
def verify_against_notebook(art: dict) -> pd.DataFrame:
    ref = NOTEBOOK_REFERENCE
    st_, eda = art["stats"], art["eda"]
    tuning = {t["Model"]: t for t in art["tuning"]}
    rows = []
    TOL_4DP = 6e-5   # notebook prints 4 decimals
    TOL_3DP = 6e-4   # notebook prints 3 decimals

    def exact(group, label, expected, actual):
        rows.append({"Group": group, "Check": label, "Notebook": str(expected),
                     "This run": str(actual), "Match": expected == actual})

    def approx(group, label, expected, actual, tol):
        rows.append({"Group": group, "Check": label, "Notebook": f"{expected:.4f}",
                     "This run": f"{float(actual):.4f}",
                     "Match": abs(float(expected) - float(actual)) <= tol})

    # Data
    exact("Data", "Rows in raw file", ref["rows_raw"], st_["rows_raw"])
    exact("Data", "Rows after removing duplicates", ref["rows_after_dedup"], st_["rows_after_dedup"])
    exact("Data", "Training rows", ref["train_rows"], st_["train_rows"])
    exact("Data", "Test rows", ref["test_rows"], st_["test_rows"])
    exact("Data", "Malignant cases in test set", ref["test_malignant"], st_["test_malignant"])
    exact("Data", "Bare_Nuclei '?' entries", ref["bare_nuclei_question_marks"],
          int(eda["question_marks"]["Bare_Nuclei"]))
    exact("Data", "Benign rows (raw file)", ref["raw_class_counts"][2],
          int(eda["class_counts_raw"].loc[2]))
    exact("Data", "Malignant rows (raw file)", ref["raw_class_counts"][4],
          int(eda["class_counts_raw"].loc[4]))
    exact("Data", "Fully duplicated rows", ref["duplicates"]["full"], eda["dup_full"])
    exact("Data", "Repeated sample IDs", ref["duplicates"]["repeated_ids"], eda["dup_ids"])
    exact("Data", "Duplicates ignoring ID", ref["duplicates"]["without_id"], eda["dup_no_id"])
    exact("Data", "Missing values in training set", ref["train_missing"], eda["train_missing"])
    exact("Data", "Missing values in test set", ref["test_missing"], eda["test_missing"])
    exact("Data", "Median used to impute Bare_Nuclei",
          ref["bare_nuclei_median_train"], eda["bare_nuclei_median"])

    # Target correlations (training set)
    for feat, val in ref["target_correlation"].items():
        approx("Correlation", f"Correlation with target: {feat}", val,
               eda["target_corr"][feat], TOL_3DP)

    # Tuning
    for name in MODEL_NAMES:
        approx("Tuning", f"Best CV F1: {name}", ref["cv_best_f1"][name],
               tuning[name]["best_cv_f1"], TOL_4DP)
    for name in MODEL_NAMES:
        exact("Tuning", f"Best hyperparameters: {name}", ref["best_params"][name],
              tuning[name]["best_params"])

    # Test set
    for name in MODEL_NAMES:
        for metric, expected, actual in zip(
            METRICS, ref["test_metrics"][name], art["results"].loc[name, METRICS]
        ):
            approx("Test set", f"{metric}: {name}", expected, actual, TOL_4DP)
    for name in MODEL_NAMES:
        tn, fp, fn, tp = art["cms"][name].ravel()
        for label, expected, actual in zip(
            ["TN", "FP", "FN", "TP"], ref["confusion"][name], (tn, fp, fn, tp)
        ):
            exact("Confusion", f"{label}: {name}", expected, int(actual))

    # Repeated CV
    rep = art["rep_cv"]
    for name in MODEL_NAMES:
        r_rec, r_f1, r_auc = ref["repeated_cv"][name]
        for label, (exp_mean, exp_std), mean_col, std_col in (
            ("Recall", r_rec, "recall_mean", "recall_std"),
            ("F1", r_f1, "f1_mean", "f1_std"),
            ("ROC-AUC", r_auc, "auc_mean", "auc_std"),
        ):
            approx("Repeated CV", f"{label} mean: {name}", exp_mean, rep.loc[name, mean_col], TOL_4DP)
            approx("Repeated CV", f"{label} std: {name}", exp_std, rep.loc[name, std_col], TOL_4DP)

    # Selection
    exact("Selection", "Selected model (ranking rule)", ref["selected_model"], art["selected"])

    return pd.DataFrame(rows)
