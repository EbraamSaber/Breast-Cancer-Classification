# Breast Cancer Classification (Wisconsin Original)

A supervised machine-learning project that predicts whether a breast tumour is **benign** or **malignant** from nine cytological measurements. Five classifiers are tuned with stratified cross-validation, compared on a held-out test set, and presented in an interactive Streamlit dashboard.

> **Educational use only.** This project compares classifiers on a published research dataset. It is not a diagnostic tool and must not inform medical decisions.

---

## Contents

- [Highlights](#highlights)
- [Dataset](#dataset)
- [Method](#method)
- [Results](#results)
- [Project structure](#project-structure)
- [Running the dashboard](#running-the-dashboard)
- [Reproduction check](#reproduction-check)
- [Known issues and limitations](#known-issues-and-limitations)
- [Reference](#reference)

---

## Highlights

- **Five models compared:** K-Nearest Neighbours, Decision Tree, Random Forest, Artificial Neural Network (MLP) and Support Vector Machine.
- **No data leakage:** the split happens before any statistic is learned. Imputation and scaling sit inside scikit-learn pipelines, so they are fitted only on training folds.
- **Tuned with GridSearchCV:** 5-fold stratified cross-validation on the training set, optimising F1-score for the malignant class.
- **Test set used for evaluation:** the held-out test set (139 samples) is scored once for the final metrics. See [Known issues](#known-issues-and-limitations) for a caveat about model selection.
- **Selected model:** ANN (MLP), ranked first by Recall, then F1-score, ROC-AUC and Accuracy.
- **Interactive dashboard:** five pages (Overview, Data & EDA, Model Performance, Model Comparison, Try a Prediction) with a light and dark theme.
- **Reproduction check:** the pipeline compares its output against 109 values printed in the original notebook.

---

## Dataset

**Breast Cancer Wisconsin (Original)**, from the UCI Machine Learning Repository.

| Property | Value |
|---|---|
| Raw records | 699 |
| Columns | 11 (ID, 9 features, class label) |
| Features | 9 cytological measurements, each an ordinal score from 1 to 10 |
| Target `Class` | `2` = benign, `4` = malignant |
| Missing values | 16, all in `Bare_Nuclei`, written as `?` in the raw file |

| Feature | Meaning |
|---|---|
| `Clump_Thickness` | Thickness of cell clumps (malignant cells tend to group in multilayers) |
| `Uniformity_of_Cell_Size` | Consistency of cell size |
| `Uniformity_of_Cell_Shape` | Consistency of cell shape |
| `Marginal_Adhesion` | How much cells stick together at the margins |
| `Single_Epithelial_Cell_Size` | Size of single epithelial cells |
| `Bare_Nuclei` | Nuclei not surrounded by cytoplasm |
| `Bland_Chromatin` | Texture of the nucleus chromatin |
| `Normal_Nucleoli` | Size and visibility of nucleoli |
| `Mitoses` | Rate of cell division |

`Sample_code_number` is an identifier and is dropped before modelling.

---

## Method

The pipeline runs in this order:

1. **Load.** The file has no header, so column names are supplied manually. `?` is read as text and converted to `NaN`, and `Bare_Nuclei` is cast to numeric.
2. **Clean.** Exact duplicate rows are removed (699 → 691). The ID column is dropped. The label is encoded as malignant = `1` (positive class) and benign = `0`.
3. **Split.** Stratified 80:20 split with `random_state = 42`, giving 552 training rows (190 malignant) and 139 test rows (48 malignant).
4. **Preprocess inside each pipeline.** Median imputation for all models. Z-score standardisation for KNN, SVM and ANN only, since Decision Tree and Random Forest are scale-invariant.
5. **Tune.** `GridSearchCV` with 5-fold stratified cross-validation on the training set, scored by F1-score of the malignant class.
6. **Evaluate.** Each tuned model is scored once on the test set using Accuracy, Precision, Recall, F1-score and ROC-AUC, plus confusion matrices and ROC curves.
7. **Check robustness.** Repeated stratified 5-fold cross-validation (3 repeats) on the training set gives a mean and standard deviation for Recall, F1 and ROC-AUC.
8. **Rank.** Models are sorted by Recall, then F1-score, ROC-AUC and Accuracy. Recall comes first because a missed malignant case is the most costly error.

### Search spaces and tuned hyperparameters

| Model | Search space | Best parameters | Best CV F1 |
|---|---|---|---|
| KNN | `n_neighbors` ∈ {3, 5, 7, 9, 11, 15}, `weights` ∈ {uniform, distance}, `p` ∈ {1, 2} | k = 7, uniform, p = 2 | 0.9555 |
| Decision Tree | `criterion` ∈ {gini, entropy}, `max_depth` ∈ {2, 3, 4, 5, 6, 8, None}, `min_samples_split` ∈ {2, 5, 10}, `min_samples_leaf` ∈ {1, 2, 4} | gini, max_depth = 4, min_samples_leaf = 2, min_samples_split = 2 | 0.9297 |
| Random Forest | `n_estimators` ∈ {100, 200, 300}, `max_depth` ∈ {None, 5, 10}, `min_samples_leaf` ∈ {1, 2}, `max_features` ∈ {sqrt, log2} | 300 trees, max_depth = None, max_features = sqrt, min_samples_leaf = 2 | 0.9512 |
| ANN (MLP) | `hidden_layer_sizes` ∈ {(16,), (32,), (32, 16)}, `activation` ∈ {relu, tanh}, `alpha` ∈ {1e-4, 1e-3, 1e-2}, `learning_rate_init` ∈ {1e-3, 1e-2} | (32,) hidden units, relu, alpha = 1e-4, learning rate = 1e-3 | 0.9530 |
| SVM | `kernel` ∈ {linear, rbf}, `C` ∈ {0.1, 1, 10, 100}, `gamma` ∈ {scale, auto} | linear kernel, C = 0.1, gamma = scale | 0.9503 |

---

## Results

### Test set (139 samples, 48 malignant)

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| **ANN (MLP)** | **0.9712** | 0.9583 | **0.9583** | **0.9583** | 0.9961 |
| SVM | 0.9640 | 0.9574 | 0.9375 | 0.9474 | **0.9966** |
| KNN | 0.9640 | **0.9778** | 0.9167 | 0.9462 | 0.9823 |
| Random Forest | 0.9568 | 0.9565 | 0.9167 | 0.9362 | 0.9938 |
| Decision Tree | 0.9281 | 0.8958 | 0.8958 | 0.8958 | 0.9755 |

Bold marks the best value in each column.

### Confusion matrices (test set)

| Model | TN | FP | FN (missed malignant) | TP |
|---|---|---|---|---|
| ANN (MLP) | 89 | 2 | **2** | 46 |
| SVM | 89 | 2 | 3 | 45 |
| KNN | 90 | 1 | 4 | 44 |
| Random Forest | 89 | 2 | 4 | 44 |
| Decision Tree | 86 | 5 | 5 | 43 |

### Repeated cross-validation (training set, mean ± std over 5 folds × 3 repeats)

| Model | Recall | F1-score | ROC-AUC |
|---|---|---|---|
| KNN | 0.9561 ± 0.0314 | 0.9503 ± 0.0256 | 0.9888 ± 0.0108 |
| Decision Tree | 0.9228 ± 0.0548 | 0.9160 ± 0.0318 | 0.9592 ± 0.0222 |
| Random Forest | 0.9632 ± 0.0285 | 0.9498 ± 0.0228 | 0.9909 ± 0.0058 |
| ANN (MLP) | 0.9632 ± 0.0316 | 0.9523 ± 0.0252 | 0.9927 ± 0.0056 |
| SVM | 0.9579 ± 0.0330 | 0.9520 ± 0.0267 | 0.9941 ± 0.0042 |

### Key findings

- **ANN (MLP) is selected.** It misses the fewest malignant cases (2 of 48) and has the highest Accuracy, Recall and F1-score on the test set. Its cross-validated F1 is also the highest of the five.
- **SVM is a close second.** It has the highest test ROC-AUC and the most stable ROC-AUC in cross-validation.
- **Decision Tree is the weakest** on every test metric and has the highest variance in cross-validated Recall.
- **The top three models are very close.** ANN and SVM differ by one malignant case on the test set. With 48 malignant test cases, one miss changes Recall by about 0.021. Treat the ranking as indicative rather than conclusive.
- **Feature signal.** Bare Nuclei (r = 0.826), Uniformity of Cell Shape (0.815) and Uniformity of Cell Size (0.810) have the strongest correlation with the target. Mitoses has the weakest (0.428). All correlations are positive.

---

## Project structure

```
.
├── app.py                          # Streamlit dashboard: page layout, navigation, theme switch
├── ml_pipeline.py                  # Data loading, cleaning, tuning, evaluation, notebook check
├── charts.py                       # Plotly figures built from pipeline outputs
├── design.py                       # Palettes, global CSS and reusable HTML components
├── classification_project (1).ipynb  # Original notebook with outputs
├── results_table.csv               # Test-set metrics exported by the notebook
├── requirements.txt                # Python dependencies
├── config.toml                     # Streamlit base theme
└── breast-cancer-wisconsin.data    # Dataset (see below; not included in this upload)
```

The layers are separate on purpose. `ml_pipeline.py` contains no presentation code, `charts.py` computes no new statistics, and `design.py` touches no data or models.

---

## Running the dashboard

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Required packages: `streamlit>=1.35`, `pandas>=2.0`, `numpy>=1.24`, `scikit-learn>=1.3`, `plotly>=5.18`.

### 2. Provide the dataset

Place `breast-cancer-wisconsin.data` in the same folder as `app.py`, or in the folder you launch Streamlit from. If neither location has the file, the pipeline downloads it from the UCI repository, which requires internet access.

> The loader checks only the project folder and the current working directory. A copy inside a `data/` subfolder is **not** found automatically.

### 3. Launch

```bash
streamlit run app.py
```

The first load tunes all five models with cross-validation, which takes about a minute. The result is cached for the rest of the server session.

### Dashboard pages

| Page | What it shows |
|---|---|
| Overview | Headline KPIs, the selected model, the ranking table, final results and the reproduction check |
| Data & EDA | Data quality, distributions, boxplots by class, correlations and the effect of standardisation |
| Model Performance | Metrics, confusion matrix, ROC curve, tuned hyperparameters and classification report for any one model |
| Model Comparison | Side-by-side curves, metrics, error counts, repeated cross-validation and per-metric winners |
| Try a Prediction | Enter nine scores and see each model's predicted label and P(malignant) |

---

## Reproduction check

`ml_pipeline.py` stores the values printed by the original notebook in `NOTEBOOK_REFERENCE` and compares them with a fresh run. The comparison covers:

- data counts and duplicates
- missing values and the imputation median
- feature correlations with the target
- best cross-validated F1 and hyperparameters for each model
- test-set metrics and confusion matrices
- repeated cross-validation means and standard deviations
- the selected model

The dashboard shows the result on the Overview page. When every value matches, the dashboard reports it. Otherwise, it lists the mismatches.

**Verification status:** a run on a copy of the same 699-record dataset, converted to the UCI file format, matched **109 of 109** reference values. That run used scikit-learn 1.9.1 and pandas 3.0.6. The dashboard was also tested headlessly on all five pages, with no runtime errors.

---

## Known issues and limitations

1. **Model selection uses the test set.** The ranking that chooses ANN (MLP) is computed from test-set metrics. This conflicts with the notebook's rule that the test set is used only once. Repeated cross-validation on the training set points to the same model (highest mean CV F1), but the margin is very small (0.9523 vs 0.9520 for SVM). A stricter workflow would select the model with cross-validation alone and then score the test set once.
2. **Duplicate-removal wording does not match the code.** The notebook's text describes removing the 236 records that match once the ID is ignored. The code removes only the 8 rows that are identical including the ID. The dashboard follows the code, since that is what produced the printed results. Changing the rule would change the results.
3. **The notebook's final interpretation cell is an unfilled template.** Its guidance is still written as instructions. The dashboard's interpretation text is generated from the computed numbers.
4. **Small test set.** The test set has 139 samples and 48 malignant cases. Differences of one or two cases between the top models are not statistically meaningful. The notebook text mentions "~137 samples", which is slightly off.
5. **Only one train/test split was evaluated.** Repeated cross-validation partly addresses this, but a nested cross-validation or a bootstrap confidence interval on test metrics would give firmer conclusions.
6. **Dataset location.** As noted above, the loader does not look inside `data/`. Moving the file next to `app.py` avoids a silent fallback to the UCI download.
7. **Streamlit deprecation warnings.** The dashboard calls `use_container_width`, which newer Streamlit releases mark as deprecated and will remove. Switching to `width="stretch"` removes the warnings.
8. **Scope.** The data are cytological scores from one published study. The models are not validated for clinical use, for other populations, or for other imaging or lab methods.

---

## Reference

- Dataset: Wolberg, W. H., & Mangasarian, O. L. (1990). Multisurface method of pattern separation for medical diagnosis applied to breast cytology. *Proceedings of the National Academy of Sciences*, 87, 9193–9196. Available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original).
- Libraries: [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Plotly](https://plotly.com/python/), [Streamlit](https://streamlit.io/).
