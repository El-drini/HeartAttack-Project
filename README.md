# Heart Attack Prediction using Machine Learning

A comparative machine learning study that predicts the likelihood of a heart attack based on clinical patient data. Four classification algorithms are implemented, tuned, and evaluated to identify the best-performing model.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Models & Results](#models--results)
- [Model Comparison](#model-comparison)
- [Prediction](#prediction)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)

---

## Overview

This project applies supervised machine learning to clinical heart attack data to classify patients as **positive** (at risk) or **negative** (not at risk) for a heart attack. The pipeline covers data cleaning, outlier handling, feature scaling, hyperparameter tuning via `GridSearchCV`, and a final comparative analysis across four classifiers.

---

## Dataset

**File:** `Heart Attack.csv`  
**Records:** 1,319 patients  
**Features:** 9 columns (8 features + 1 target)

| Column | Description |
|---|---|
| `age` | Patient age (years) |
| `gender` | Patient gender (encoded numerically) |
| `impluse` | Heart rate / impulse (bpm) |
| `pressurehight` | Systolic blood pressure (mmHg) |
| `pressurelow` | Diastolic blood pressure (mmHg) |
| `glucose` | Blood glucose level (mg/dL) |
| `kcm` | KCM biomarker level |
| `troponin` | Troponin level (cardiac biomarker) |
| `class` | Target variable: `positive` or `negative` |

**Class Distribution:**
- Positive (at risk): 810 samples
- Negative (not at risk): 509 samples

> The dataset has **no missing values**. A mild class imbalance exists and is addressed via class weighting in the SVM model.

---

## Project Structure

```
├── Full_Code.ipynb       # Main Jupyter notebook with full pipeline
├── Heart Attack.csv      # Dataset
└── README.md             # Project documentation
```

---

## Preprocessing Pipeline

1. **Exploratory Data Analysis** — `data.info()`, `describe()`, histograms, pairplots, and a correlation heatmap.
2. **Categorical Encoding** — `LabelEncoder` applied to the `class` column (and any other object-type columns).
3. **Outlier Handling** — IQR method used to cap outliers in all numeric columns at their lower/upper bounds.
4. **Feature Scaling** — `StandardScaler` applied to bring all features to zero mean and unit variance.
5. **Train/Test Split** — 80/20 split (1,055 training samples / 264 test samples), `random_state=42`.
6. **Target Binarization** — The scaled `class` column is binned into `POSITIVE` / `NEGATIVE` labels using `pd.cut`.

---

## Models & Results

All models are wrapped in `sklearn` `Pipeline` objects. Hyperparameter tuning uses 5-fold `GridSearchCV` with accuracy as the scoring metric.

### 1. K-Nearest Neighbors (KNN)

| Hyperparameter | Best Value |
|---|---|
| `n_neighbors` | 14 |
| `metric` | Manhattan |

| Metric | Score |
|---|---|
| Accuracy | **82.95%** |
| Precision (Negative) | 0.88 |
| Recall (Negative) | 0.83 |
| F1-Score (Negative) | 0.86 |
| Precision (Positive) | 0.75 |
| Recall (Positive) | 0.82 |
| F1-Score (Positive) | 0.79 |

> The Manhattan distance metric outperformed Euclidean (avg. accuracy 82.90% vs. 79.83%).

---

### 2. Naive Bayes (Gaussian)

No hyperparameter tuning required; the model is trained directly.

| Metric | Score |
|---|---|
| Accuracy | **87.12%** |
| Precision (Negative) | 0.95 |
| Recall (Negative) | 0.83 |
| F1-Score (Negative) | 0.89 |
| Precision (Positive) | 0.78 |
| Recall (Positive) | 0.93 |
| F1-Score (Positive) | 0.85 |

> High recall for the positive class (93.07%) makes this model strong for risk detection. A slight gap between training recall (97.79%) and test recall (93.07%) suggests minor overfitting.

---

### 3. Support Vector Machine (SVM)

| Hyperparameter | Search Space |
|---|---|
| `C` | 0.1, 1, 10, 100 |
| `gamma` | 0.001, 0.01, 0.1, 1 |
| `kernel` | linear, poly, rbf |
| `class_weight` | Balanced (computed from training data) |

Best configuration used the **RBF kernel**:

| Metric | Score |
|---|---|
| Accuracy | **91.67%** |
| F1 Score | 89.42% |
| Precision (Negative) | 0.95 |
| Recall (Negative) | 0.91 |
| F1-Score (Negative) | 0.93 |
| Precision (Positive) | 0.87 |
| Recall (Positive) | 0.92 |
| F1-Score (Positive) | 0.89 |

> Class imbalance is handled via `compute_class_weight`. PCA (2 components) is used for decision boundary visualization.

---

### 4. Decision Tree

| Hyperparameter | Best Value |
|---|---|
| `criterion` | Gini |
| `max_depth` | 4 |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 2 |
| `max_features` | None |

| Metric | Score |
|---|---|
| Accuracy | **97.35%** |
| Precision (Negative) | 0.97 |
| Recall (Negative) | 0.99 |
| F1-Score (Negative) | 0.98 |
| Precision (Positive) | 0.98 |
| Recall (Positive) | 0.95 |
| F1-Score (Positive) | 0.96 |

> Best overall model. With a max depth of 4, the tree remains interpretable while achieving near-perfect classification (only 7 total misclassifications on 264 test samples). Feature importance is plotted and the tree is fully visualized.

---

## Model Comparison

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1 Score (Weighted) |
|---|---|---|---|---|
| KNN | 82.95% | — | — | — |
| Naive Bayes | 87.12% | — | — | — |
| SVM (RBF) | 91.67% | — | — | — |
| **Decision Tree** | **97.35%** | **0.97** | **0.97** | **0.97** |

The Decision Tree with `max_depth=4` is the top performer across all metrics. SVM is the best alternative when interpretability is less critical.

---

## Prediction

All four models expose prediction functions that accept a list of raw feature values, apply the fitted scaler, and return the class label:

```python
new_features = [64, 1, 66, 160, 83, 160, 1.8, 0.012]
# [age, gender, impulse, pressureHigh, pressureLow, glucose, kcm, troponin]

predict_knn(new_features)      # → NEGATIVE
predict_nb(new_features)       # → NEGATIVE
predict_svm(new_features)      # → NEGATIVE
predict_dt(new_features)       # → NEGATIVE
```

---

## Installation

```bash
git clone https://github.com/your-username/heart-attack-prediction.git
cd heart-attack-prediction
pip install -r requirements.txt
```

---

## Usage

Open and run the notebook:

```bash
jupyter notebook Full_Code.ipynb
```

Run all cells in order. The notebook is self-contained and will:
1. Load and preprocess the dataset
2. Visualize distributions, correlations, and class balance
3. Train and tune all four models
4. Print evaluation metrics and render all plots
5. Run sample predictions

---

## Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

Install all at once:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```
