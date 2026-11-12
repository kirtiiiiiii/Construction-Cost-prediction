# Predicting Construction Cost from Project Indicators (Regression Ensembles)

Predicts the actual construction cost of residential buildings in Tehran, Iran, using tree-based ensemble regressors and an SVR baseline. Course project for UCE521 (Module: Cost Estimation).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ndl8bfocZtMAR0uHP7E0oK1EopvhIADd?usp=sharing)

## Dataset

**UCI Residential Building Data Set** (Rafiei & Adeli)

- Page: https://archive.ics.uci.edu/dataset/437/residential+building+data+set
- DOI: https://doi.org/10.24432/C5S896
- File used: `Residential-Building-Data-Set.xlsx` (sheets: `Data`, `Descriptions`)
- Licence: CC BY 4.0

372 residential projects, with 8 project physical/financial variables (V-1 to V-8), 4 start/completion date variables, and 19 economic indicators (V-11 to V-29) at 5 time lags (95 columns). Two outputs are provided: actual sales price (V-9) and actual construction cost (V-10).

**Target:** V-10 (actual construction cost). V-9 is excluded from the inputs.

Citation: Rafiei, M. (2015). *Residential Building* [Dataset]. UCI Machine Learning Repository.

## Method

1. **Feature separation:** date, project-specific, and lagged economic variables documented separately.
2. **Outlier check:** boxplot and IQR rule on cost per m² (27 projects flagged, retained as genuine variation).
3. **Multicollinearity:** correlation heatmap, then pairwise filtering at |r| > 0.95 (95 → 38 economic features), then iterative VIF removal until VIF ≤ 10 (→ 9 features).
4. **Final dataset:** 372 projects, 21 input features.
5. **Split:** 80/20 train/test (`random_state=42`); standard scaling for SVR.
6. **Models:** Random Forest, XGBoost, LightGBM, SVR (RBF), compared on RMSE, MAE and R², plus 2-fold cross-validation.
7. **Interpretation:** XGBoost feature importances and SHAP summary plot.

## Results (80/20 test set)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| LightGBM | 29.34 | 20.54 | 0.969 |
| XGBoost | 30.18 | 17.64 | 0.968 |
| Random Forest | 42.74 | 24.16 | 0.935 |
| SVR | 51.89 | 28.02 | 0.904 |

XGBoost was chosen as the final model: it had the lowest MAE and the lowest RMSE in 2-fold cross-validation (33.2). The initial estimated construction cost (V-5) was the dominant feature (about 91% of importance).

## How to run

1. Open the notebook in Google Colab (badge above).
2. Download `Residential-Building-Data-Set.xlsx` from the UCI link above and upload it to `/content/` in Colab.
3. Run all cells (Runtime → Run all).

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, lightgbm, statsmodels, shap.

## Repository contents

- `Project1.ipynb`: full analysis notebook
- `README.md`: this file
