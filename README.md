# TPRF: Three-Pass Regression Filter for Nowcasting

A Python/R implementation of the **Three-Pass Regression Filter (TPRF)** from Kelly & Pruitt (2015), with applications to GDP nowcasting as evaluated in Akepanidtaworn & Akepanidtaworn (2025).

## What is TPRF?

TPRF is a targeted factor extraction method for forecasting with many predictors. Unlike standard Principal Component Analysis (PCA), which extracts factors that explain the most variance in the predictors themselves, TPRF extracts factors **maximally correlated with the target variable**.

This makes it particularly well-suited for macroeconomic nowcasting, where you have dozens of high-frequency indicators but only a few actually drive the variable you care about (e.g., GDP). Standard PCA can get distracted by large, irrelevant sources of co-movement in the data. TPRF ignores them.

## The Three Steps

1. **Time-Series Regressions** — Regress the target on each predictor individually. The resulting slopes (β) measure each indicator's relevance to the target.
2. **Cross-Sectional Regression** — At each time period, regress the observed indicators onto their β weights. This yields the TPRF factor: a weighted average where indicators with stronger target relationships receive higher loadings.
3. **Forecasting Regression** — Use the TPRF factor as an exogenous regressor in an ARX/ARIMAX model to produce forecasts.

Steps 1 and 2 use only lagged target data, so there is no look-ahead bias. The β weights are frozen after Step 1, making real-time updates fast and stable.

## Why TPRF over PCA?

| | PCA | TPRF |
|---|---|---|
| Optimizes for | Variance in X | Correlation with y |
| Distracted by confounding factors | Yes | No |
| Requires target for factor extraction | No | Yes |
| Look-ahead bias | No | No |

## Usage

```python
from tprf import TPRF

# X: (n_samples, n_features) matrix of high-frequency indicators
# y: (n_samples,) vector of the target (e.g., GDP growth)

model = TPRF()
model.fit(X_train, y_train)       # Step 1: learn the betas
tprf_train = model.transform(X_train)  # Step 2: extract factor
tprf_test = model.transform(X_test)    # Step 2 on new data (betas frozen)

# Step 3: use TPRF as exogenous regressor in your forecasting model
# e.g., ARIMAX, OLS, etc.
```

A synthetic example is available in `example.ipynb`.

## Implementation Notes

- **Standardization**: Features and target are standardized internally before Step 1. This ensures the β weights reflect correlation strength rather than scale differences.
- **Frequency mismatch**: This implementation assumes all data has been converted to a common frequency (e.g., via temporal aggregation or blocking) before being passed to the model.
- **No look-ahead**: Steps 1 and 2 use only lagged target data. The nowcast in Step 3 uses only the current period's TPRF factor value.

## References

- Klakow Akepanidtaworn, and Korkrid Akepanidtaworn. "GDP Nowcasting Performance of Traditional Econometric Models vs Machine-Learning Algorithms: Simulation and Case Studies", IMF Working Papers 2025, 252 (2025), accessed 6/19/2026, https://doi.org/10.5089/9798229033626.001
