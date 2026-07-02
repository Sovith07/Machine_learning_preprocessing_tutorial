# Feature Engineering Tutorial

A hands-on, notebook-based walkthrough of core feature engineering techniques used in classical machine learning pipelines — covering missing value imputation, outlier detection, feature scaling, transformation, discretization, and categorical encoding, with `scikit-learn` and supporting libraries.

Each folder is self-contained and focuses on one stage of the feature engineering pipeline, with notebooks that mix theory, hands-on code, visualizations, and before/after model comparisons.

## Repository Structure

```
Feature_Engineering_tutorial/
├── Handling_Missing_Values/
│   ├── univariate_imputers.ipynb
│   ├── multivariate_imputers.ipynb
│   └── multivariate_iterative_imputer_using_scratch.ipynb
├── Outlier_Detection/
│   ├── univariate_outler_detection.ipynb
│   ├── multivariate_outlier_detection.ipynb
│   └── placement.csv
├── Feature_Scaling/
│   ├── standardization.ipynb
│   ├── univariate_normalization.ipynb
│   └── multivariate_normalization.ipynb
├── Feature_transformation/
│   ├── feature_transformations.ipynb
│   └── train.csv
├── Discretization/
│   ├── discretization.ipynb
│   └── binning.ipynb
└── Encoding_Categorical_Features/
    ├── encoding_categorical_variables.ipynb
    ├── sklearn_column_transformer.ipynb
    ├── sklearn_deep_dive.ipynb
    ├── cars.csv
    ├── customer.csv
    └── pipe.pkl
```

## What's Inside

### 1. Handling Missing Values
- **Univariate imputation**: detecting and visualizing missing data with `missingno` (matrix, heatmap, dendrogram), distinguishing MCAR/MAR/MNAR, Complete Case Analysis (CCA), mean/median/most-frequent imputation, missing indicator, and using `GridSearchCV` to automatically select the best imputation strategy.
- **Multivariate imputation**: `KNNImputer`, `IterativeImputer` (including hyperparameter tuning with `BayesianRidge` and other estimators), and a comparison table across methods.
- **Iterative imputer from scratch**: a from-first-principles implementation of the MICE-style iterative imputation algorithm to build intuition for what `IterativeImputer` does under the hood.

### 2. Outlier Detection
- **Univariate**: Z-score method and IQR/box-plot method, with both **trimming** and **capping** strategies.
- **Multivariate**: Isolation Forest, KNN-based detection, Local Outlier Factor (LOF), and DBSCAN — with a comparison of strengths/weaknesses (e.g., KNN's inability to detect local outliers).

### 3. Feature Scaling
- **Standardization** (`StandardScaler`): effect on distribution shape, impact on model accuracy (Logistic Regression vs. Decision Tree), and sensitivity to outliers.
- **Univariate normalization**: Min-Max Scaling, Robust Scaling, and MaxAbsScaler, with distribution comparisons before/after scaling.
- **Multivariate normalization**: L1 and L2 normalization, including the effect of outliers on each.

### 4. Feature Transformation
- Log transform, Sin transform, Square transform.
- Box-Cox and Yeo-Johnson power transforms.
- Applying targeted transformations across multiple features of a real dataset (Boston Housing) and comparing baseline vs. transformed model performance (R², MSE).

### 5. Discretization (Binning)
- **Binning strategies**: Custom binning, Uniform binning, Quantile binning, K-Means binning, Binarization, and supervised Decision Tree-based binning.
- **Why discretize**: reducing overfitting, handling non-linear relationships, robustness to outliers, better interpretability, and compatibility with certain models — each illustrated with worked examples.

### 6. Encoding Categorical Features
- **Encoding techniques**: Ordinal Encoding, Label Encoding, One-Hot Encoding (including handling rare/unknown categories), Label Binarizer, Count/Frequency Encoding, Binary Encoding, Target Encoding, and Weight of Evidence.
- **`ColumnTransformer` & `Pipeline`**: the manual ("hard way") vs. `ColumnTransformer` ("easy way") approach to preprocessing, building full `Pipeline`s with imputation → encoding → scaling → model, cross-validation, hyperparameter tuning, and exporting the trained pipeline.
- **Scikit-learn deep dive**: writing custom estimators (`BaseEstimator`, `ClassifierMixin`), custom transformers (`FunctionTransformer`, `TransformerMixin`), and combining them with `ColumnTransformer`, `FeatureUnion`, and `Pipeline`.

## Tech Stack

- Python, Jupyter Notebook
- `pandas`, `numpy`
- `scikit-learn`
- `category_encoders`
- `missingno`
- `matplotlib`, `seaborn`

## Getting Started

```bash
git clone https://github.com/Sovith07/Feature_Engineering_tutorial.git
cd Feature_Engineering_tutorial
pip install pandas numpy scikit-learn category_encoders missingno matplotlib seaborn jupyter
jupyter notebook
```

Open any notebook and run the cells in order — each one is self-contained, though a few pull public datasets directly from GitHub-hosted CSVs (Titanic, Wine, Boston Housing, Social Network Ads) at runtime.

## Purpose

This repo serves as a personal reference and learning log for feature engineering concepts — useful for revisiting techniques before interviews, building intuition for what scikit-learn's preprocessing tools do internally, and as a quick lookup for "how do I do X" during real projects.

## Author

**Sovith Kumar Singh**
[GitHub](https://github.com/Sovith07) · [LinkedIn](https://www.linkedin.com/in/sovithkumarsingh)
