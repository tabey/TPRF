import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class TPRF:
    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.betas_ = None  # The "engine" from Step 1

    def fit(self, X, y):
        """
        Step 1: Estimate the betas (loadings) for each indicator.
        """
        # Standardize to ensure betas are comparable
        X_std = self.scaler_X.fit_transform(X)
        y_std = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        n_features = X_std.shape[1]
        self.betas_ = np.zeros(n_features)
        
        # Regress y on each feature individually to get the slope
        for i in range(n_features):
            reg = LinearRegression()
            reg.fit(X_std[:, i].reshape(-1, 1), y_std)
            self.betas_[i] = reg.coef_[0]
            
        return self

    def transform(self, X):
        """
        Step 2: Project new X data onto the pre-fit betas to get TPRF factor values.
        """
        if self.betas_ is None:
            raise RuntimeError("Must fit the model before transforming.")
            
        X_std = self.scaler_X.transform(X)
        n_periods = X_std.shape[0]
        tprf_values = np.zeros(n_periods)
        
        # For each time period, regress the cross-section of features onto the betas
        for t in range(n_periods):
            reg = LinearRegression()
            # Note: The "feature" is the beta vector, the "target" is the X values at time t
            reg.fit(self.betas_.reshape(-1, 1), X_std[t, :])
            tprf_values[t] = reg.coef_[0]
            
        return tprf_values

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
