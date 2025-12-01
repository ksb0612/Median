"""
Ridge Marketing Mix Modeling (MMM) implementation.

This module provides the core RidgeMMM class for training and analyzing
marketing mix models using Ridge Regression with transformation pipelines.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib
import warnings

from transformations import TransformationPipeline
from utils import calculate_mape, calculate_mae, calculate_rmse
from utils.data_utils import validate_required_columns, validate_column_types, safe_divide


class RidgeMMM:
    """
    Ridge Regression Marketing Mix Model.
    
    This class implements a complete MMM workflow including:
    - Transformation of media spend (adstock + saturation)
    - Feature scaling
    - Ridge regression training
    - Contribution analysis
    - Response curve generation
    - Time series decomposition
    
    Attributes:
        alpha (float): Ridge regularization parameter
        model (Ridge): Fitted Ridge regression model
        scaler (StandardScaler): Fitted feature scaler
        pipeline (TransformationPipeline): Transformation pipeline
        channel_configs (dict): Configuration for each channel
        feature_names (list): Names of features used in training
        media_channels (list): Names of media channels
        exog_vars (list): Names of exogenous variables
        is_fitted (bool): Whether model has been fitted
    
    Example:
        >>> mmm = RidgeMMM(alpha=1.0)
        >>> mmm.fit(X_train, y_train, channel_configs)
        >>> predictions = mmm.predict(X_test)
        >>> metrics = mmm.evaluate(X_test, y_test)
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize RidgeMMM model.
        
        Args:
            alpha: Ridge regularization strength. Higher values mean more
                   regularization. Default is 1.0.
        
        Raises:
            ValueError: If alpha is not positive
        """
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, fit_intercept=True)
        self.scaler = StandardScaler()
        self.pipeline = None
        self.channel_configs = None
        self.feature_names = None
        self.media_channels = None
        self.exog_vars = None
        self.is_fitted = False
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        channel_configs: Dict[str, Dict[str, float]],
        exog_vars: Optional[List[str]] = None
    ) -> 'RidgeMMM':
        """
        Fit the Ridge MMM model.
        
        This method:
        1. Applies transformations (adstock + saturation) to media channels
        2. Adds exogenous variables if provided
        3. Standardizes all features
        4. Trains Ridge regression model
        
        Args:
            X: DataFrame with media spend columns (and optionally exog vars)
            y: Target variable (revenue/conversions)
            channel_configs: Dict mapping channel names to transformation params
                           Each config should have: 'adstock', 'hill_K', 'hill_S'
            exog_vars: Optional list of exogenous variable column names
        
        Returns:
            self: Fitted model instance
        
        Raises:
            ValueError: If inputs are invalid or channels missing
        
        Example:
            >>> configs = {
            ...     'google': {'adstock': 0.5, 'hill_K': 1.0, 'hill_S': 1.0},
            ...     'facebook': {'adstock': 0.7, 'hill_K': 1.2, 'hill_S': 0.9}
            ... }
            >>> mmm.fit(X_train, y_train, configs, exog_vars=['seasonality'])
        """
        # Validate inputs
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise ValueError("y must be a pandas Series or numpy array")
        
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length. Got X: {len(X)}, y: {len(y)}")
        
        if not channel_configs:
            raise ValueError("channel_configs cannot be empty")
        
        # Store configuration
        self.channel_configs = channel_configs
        self.media_channels = list(channel_configs.keys())
        self.exog_vars = exog_vars if exog_vars else []

        # Validate all required columns exist using utility function
        all_required_cols = self.media_channels + self.exog_vars
        validate_required_columns(X, all_required_cols, context="Input DataFrame")

        # Validate that media channels are numeric
        validate_column_types(X, numeric_cols=self.media_channels)
        
        # Step 1: Apply transformations to media channels
        self.pipeline = TransformationPipeline(channel_configs)
        X_transformed = self.pipeline.fit_transform(X[self.media_channels])
        
        # Step 2: Add exogenous variables if provided
        if self.exog_vars:
            X_exog = X[self.exog_vars].copy()
            X_features = pd.concat([X_transformed, X_exog], axis=1)
        else:
            X_features = X_transformed
        
        # Store feature names
        self.feature_names = X_features.columns.tolist()
        
        # Step 3: Standardize features
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Step 4: Train Ridge regression
        self.model.fit(X_scaled, y)
        
        # Mark as fitted
        self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the fitted model.

        Args:
            X: DataFrame with same structure as training data

        Returns:
            Array of predictions

        Raises:
            ValueError: If model not fitted or X has wrong structure

        Example:
            >>> predictions = mmm.predict(X_test)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict()")

        # Apply transformations
        X_transformed = self.pipeline.transform(X[self.media_channels])
        
        # Add exogenous variables if used during training
        if self.exog_vars:
            X_exog = X[self.exog_vars].copy()
            X_features = pd.concat([X_transformed, X_exog], axis=1)
        else:
            X_features = X_transformed
        
        # Ensure columns match training
        if not all(col in X_features.columns for col in self.feature_names):
            raise ValueError(
                f"X must contain all features used in training: {self.feature_names}"
            )
        
        X_features = X_features[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X_features)
        
        # Predict
        return self.model.predict(X_scaled)
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Calculates multiple metrics:
        - R²: Coefficient of determination (0-1, higher is better)
        - MAPE: Mean Absolute Percentage Error (%, lower is better)
        - MAE: Mean Absolute Error (same units as y, lower is better)
        - RMSE: Root Mean Squared Error (same units as y, lower is better)
        
        Args:
            X_test: Test features
            y_test: Test target values
        
        Returns:
            Dictionary with metric names and values
        
        Example:
            >>> metrics = mmm.evaluate(X_test, y_test)
            >>> print(f"R²: {metrics['r2']:.3f}")
            >>> print(f"MAPE: {metrics['mape']:.2f}%")
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling evaluate()")
        
        # Get predictions
        y_pred = self.predict(X_test)
        y_true = np.array(y_test)
        
        # Calculate metrics
        metrics = {
            'r2': r2_score(y_true, y_pred),
            'mape': calculate_mape(y_true, y_pred),
            'mae': calculate_mae(y_true, y_pred),
            'rmse': calculate_rmse(y_true, y_pred)
        }
        
        return metrics
    
    def get_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate contribution of each channel to revenue.
        
        For each channel, contribution is calculated as:
        contribution = coefficient * transformed_spend
        
        Also calculates:
        - ROAS (Return on Ad Spend): contribution / original_spend
        - Contribution percentage: contribution / total_predicted_revenue
        
        Args:
            X: DataFrame with media spend data
        
        Returns:
            DataFrame with columns:
            - channel: Channel name
            - spend: Total original spend
            - contribution: Total contribution to revenue
            - roas: Return on ad spend
            - contribution_pct: Percentage of total revenue
        
        Example:
            >>> contributions = mmm.get_contributions(X_train)
            >>> print(contributions.sort_values('contribution', ascending=False))
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling get_contributions()")

        # Apply transformations
        X_transformed = self.pipeline.transform(X[self.media_channels])
        
        # Add exogenous variables if used
        if self.exog_vars:
            X_exog = X[self.exog_vars].copy()
            X_features = pd.concat([X_transformed, X_exog], axis=1)
        else:
            X_features = X_transformed
        
        X_features = X_features[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X_features)
        
        # Get coefficients
        coefficients = self.model.coef_
        intercept = self.model.intercept_
        
        # Calculate total predicted revenue
        total_predicted = self.predict(X).sum()
        
        # Calculate base contribution (intercept)
        base_contribution = intercept * len(X)
        
        # Calculate contributions for each feature
        contributions_data = []
        
        for i, feature_name in enumerate(self.feature_names):
            # Calculate contribution: coefficient * scaled_feature * std + mean_effect
            feature_contribution = (coefficients[i] * X_scaled[:, i]).sum()
            
            # Check if this is a media channel or exog var
            if feature_name in self.media_channels:
                # Original spend
                original_spend = X[feature_name].sum()

                # ROAS using safe_divide
                roas = safe_divide(feature_contribution, original_spend)

                # Contribution percentage using safe_divide
                contribution_pct = safe_divide(feature_contribution, total_predicted) * 100
                
                contributions_data.append({
                    'channel': feature_name,
                    'spend': original_spend,
                    'contribution': feature_contribution,
                    'roas': roas,
                    'contribution_pct': contribution_pct
                })
        
        # Create DataFrame
        contributions_df = pd.DataFrame(contributions_data)
        
        # Add base row with safe_divide
        base_row = pd.DataFrame([{
            'channel': 'base',
            'spend': 0,
            'contribution': base_contribution,
            'roas': 0,
            'contribution_pct': safe_divide(base_contribution, total_predicted) * 100
        }])
        
        contributions_df = pd.concat([contributions_df, base_row], ignore_index=True)
        
        # Sort by contribution
        contributions_df = contributions_df.sort_values('contribution', ascending=False)
        
        return contributions_df
    
    def get_response_curve(
        self,
        channel: str,
        budget_range: np.ndarray,
        X_base: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate response curve for a specific channel.
        
        This shows how revenue changes as spend on one channel varies,
        while keeping other channels at their mean values.
        
        Args:
            channel: Name of channel to vary
            budget_range: Array of spend values to test
            X_base: Base DataFrame to use for other channels (typically training data)
        
        Returns:
            DataFrame with columns:
            - spend: Spend level tested
            - revenue: Predicted revenue at that spend
            - marginal_roas: Marginal return on ad spend (derivative)
        
        Raises:
            ValueError: If channel not in model
        
        Example:
            >>> budget_range = np.linspace(0, 50000, 100)
            >>> curve = mmm.get_response_curve('google', budget_range, X_train)
            >>> # Find optimal spend
            >>> optimal = curve.loc[curve['marginal_roas'].idxmax()]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling get_response_curve()")
        
        if channel not in self.media_channels:
            raise ValueError(f"Channel '{channel}' not in model. Available: {self.media_channels}")
        
        # Create base scenario with mean values
        X_scenario = pd.DataFrame()
        
        for ch in self.media_channels:
            if ch == channel:
                # This will be varied
                X_scenario[ch] = budget_range
            else:
                # Keep at mean
                X_scenario[ch] = X_base[ch].mean()
        
        # Add exog vars at mean if used
        if self.exog_vars:
            for exog in self.exog_vars:
                X_scenario[exog] = X_base[exog].mean()
        
        # Predict revenue for each spend level
        revenue = self.predict(X_scenario)
        
        # Calculate marginal ROAS (derivative) using safe_divide
        marginal_roas = np.zeros_like(revenue)
        for i in range(1, len(revenue)):
            delta_revenue = revenue[i] - revenue[i-1]
            delta_spend = budget_range[i] - budget_range[i-1]
            marginal_roas[i] = safe_divide(delta_revenue, delta_spend)
        
        # First point uses forward difference
        if len(revenue) > 1:
            marginal_roas[0] = marginal_roas[1]
        
        # Create result DataFrame
        result = pd.DataFrame({
            'spend': budget_range,
            'revenue': revenue,
            'marginal_roas': marginal_roas
        })
        
        return result
    
    def decompose_timeseries(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """
        Decompose revenue into contributions from each channel over time.
        
        This creates a time series showing how much each channel contributed
        to revenue in each period.
        
        Args:
            X: DataFrame with media spend data (must include date column)
            y: Actual revenue values
        
        Returns:
            DataFrame with columns:
            - date: Date/period (if available in X)
            - actual_revenue: Actual revenue values
            - base: Base contribution (intercept)
            - [channel1]: Contribution from channel 1
            - [channel2]: Contribution from channel 2
            - ...
            - predicted_revenue: Sum of all contributions
        
        Example:
            >>> decomp = mmm.decompose_timeseries(X_train, y_train)
            >>> # Plot stacked area chart
            >>> decomp.plot.area(x='date', stacked=True)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling decompose_timeseries()")

        # Apply transformations
        X_transformed = self.pipeline.transform(X[self.media_channels])
        
        # Add exogenous variables if used
        if self.exog_vars:
            X_exog = X[self.exog_vars].copy()
            X_features = pd.concat([X_transformed, X_exog], axis=1)
        else:
            X_features = X_transformed
        
        X_features = X_features[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X_features)
        
        # Get coefficients
        coefficients = self.model.coef_
        intercept = self.model.intercept_
        
        # Initialize result DataFrame
        result = pd.DataFrame()
        
        # Add date if available
        date_cols = [col for col in X.columns if 'date' in col.lower()]
        if date_cols:
            result['date'] = X[date_cols[0]].values
        
        # Add actual revenue
        result['actual_revenue'] = y.values
        
        # Add base contribution
        result['base'] = intercept
        
        # Add contribution from each channel
        for i, feature_name in enumerate(self.feature_names):
            if feature_name in self.media_channels:
                # Weekly contribution: coefficient * scaled_feature
                result[feature_name] = coefficients[i] * X_scaled[:, i]
        
        # Add predicted revenue (sum of all contributions)
        contribution_cols = ['base'] + [ch for ch in self.media_channels if ch in result.columns]
        result['predicted_revenue'] = result[contribution_cols].sum(axis=1)
        
        return result
    
    def save_model(self, filepath: str) -> None:
        """
        Save fitted model to disk using joblib.
        
        Args:
            filepath: Path to save model (e.g., 'model.pkl')
        
        Raises:
            ValueError: If model not fitted
        
        Example:
            >>> mmm.save_model('ridge_mmm_model.pkl')
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'alpha': self.alpha,
            'model': self.model,
            'scaler': self.scaler,
            'pipeline': self.pipeline,
            'channel_configs': self.channel_configs,
            'feature_names': self.feature_names,
            'media_channels': self.media_channels,
            'exog_vars': self.exog_vars,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> 'RidgeMMM':
        """
        Load fitted model from disk.
        
        Args:
            filepath: Path to saved model
        
        Returns:
            self: Loaded model instance
        
        Example:
            >>> mmm = RidgeMMM()
            >>> mmm.load_model('ridge_mmm_model.pkl')
        """
        model_data = joblib.load(filepath)
        
        self.alpha = model_data['alpha']
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.pipeline = model_data['pipeline']
        self.channel_configs = model_data['channel_configs']
        self.feature_names = model_data['feature_names']
        self.media_channels = model_data['media_channels']
        self.exog_vars = model_data['exog_vars']
        self.is_fitted = model_data['is_fitted']
        
        return self
    
    def get_model_summary(self) -> Dict[str, any]:
        """
        Get summary of fitted model.
        
        Returns:
            Dictionary with model information
        
        Example:
            >>> summary = mmm.get_model_summary()
            >>> print(f"Channels: {summary['n_channels']}")
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting summary")
        
        return {
            'alpha': self.alpha,
            'n_channels': len(self.media_channels),
            'channels': self.media_channels,
            'n_exog_vars': len(self.exog_vars),
            'exog_vars': self.exog_vars,
            'n_features': len(self.feature_names),
            'intercept': self.model.intercept_,
            'coefficients': dict(zip(self.feature_names, self.model.coef_))
        }
