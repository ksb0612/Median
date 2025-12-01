"""
Tests for RidgeMMM class.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ridge_mmm import RidgeMMM


@pytest.fixture
def synthetic_data():
    """Create synthetic marketing data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    # Create media spend data
    google = np.random.uniform(10000, 30000, n_samples)
    facebook = np.random.uniform(8000, 25000, n_samples)
    tv = np.random.uniform(5000, 20000, n_samples)
    
    # Create exogenous variable (seasonality)
    seasonality = np.tile([1, 2, 3, 4], n_samples // 4)
    
    # Create revenue with known relationships
    # Base revenue + media effects + seasonality effect + noise
    base = 50000
    revenue = (
        base +
        google * 2.5 +  # Google has 2.5 ROAS
        facebook * 2.0 +  # Facebook has 2.0 ROAS
        tv * 1.5 +  # TV has 1.5 ROAS
        seasonality * 5000 +  # Seasonality effect
        np.random.normal(0, 10000, n_samples)  # Noise
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=n_samples, freq='W'),
        'revenue': revenue,
        'google': google,
        'facebook': facebook,
        'tv': tv,
        'seasonality': seasonality
    })
    
    return df


@pytest.fixture
def channel_configs():
    """Create channel configurations for testing."""
    return {
        'google': {'adstock': 0.5, 'hill_K': 1.0, 'hill_S': 1.0},
        'facebook': {'adstock': 0.6, 'hill_K': 1.0, 'hill_S': 1.0},
        'tv': {'adstock': 0.7, 'hill_K': 1.0, 'hill_S': 1.0}
    }


class TestRidgeMMM:
    """Tests for RidgeMMM class."""
    
    def test_init_valid_alpha(self):
        """Test initialization with valid alpha."""
        mmm = RidgeMMM(alpha=1.0)
        assert mmm.alpha == 1.0
        assert mmm.is_fitted == False
    
    def test_init_invalid_alpha(self):
        """Test initialization with invalid alpha."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            RidgeMMM(alpha=0)
        
        with pytest.raises(ValueError, match="alpha must be positive"):
            RidgeMMM(alpha=-1.0)
    
    def test_fit_basic(self, synthetic_data, channel_configs):
        """Test basic model fitting."""
        df = synthetic_data
        X = df[['google', 'facebook', 'tv']]
        y = df['revenue']
        
        mmm = RidgeMMM(alpha=1.0)
        mmm.fit(X, y, channel_configs)
        
        assert mmm.is_fitted == True
        assert mmm.media_channels == ['google', 'facebook', 'tv']
        assert mmm.feature_names == ['google', 'facebook', 'tv']
    
    def test_fit_with_exog_vars(self, synthetic_data, channel_configs):
        """Test fitting with exogenous variables."""
        df = synthetic_data
        X = df[['google', 'facebook', 'tv', 'seasonality']]
        y = df['revenue']
        
        mmm = RidgeMMM(alpha=1.0)
        mmm.fit(X, y, channel_configs, exog_vars=['seasonality'])
        
        assert mmm.is_fitted == True
        assert 'seasonality' in mmm.exog_vars
        assert 'seasonality' in mmm.feature_names
    
    def test_fit_invalid_inputs(self, channel_configs):
        """Test fitting with invalid inputs."""
        mmm = RidgeMMM(alpha=1.0)
        
        # Invalid X type
        with pytest.raises(ValueError, match="X must be a pandas DataFrame"):
            mmm.fit([[1, 2, 3]], [1, 2, 3], channel_configs)
        
        # Mismatched lengths
        X = pd.DataFrame({'google': [1, 2, 3], 'facebook': [1, 2, 3], 'tv': [1, 2, 3]})
        y = pd.Series([1, 2])
        with pytest.raises(ValueError, match="same length"):
            mmm.fit(X, y, channel_configs)
        
        # Empty channel configs
        X = pd.DataFrame({'google': [1, 2, 3]})
        y = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="cannot be empty"):
            mmm.fit(X, y, {})
    
    def test_fit_missing_channels(self, channel_configs):
        """Test fitting when configured channels are missing from X."""
        mmm = RidgeMMM(alpha=1.0)
        
        X = pd.DataFrame({'google': [1, 2, 3], 'facebook': [1, 2, 3]})
        y = pd.Series([1, 2, 3])

        # 'tv' is in configs but not in X
        with pytest.raises(ValueError, match="missing required columns"):
            mmm.fit(X, y, channel_configs)
    
    def test_predict_basic(self, synthetic_data, channel_configs):
        """Test basic prediction."""
        df = synthetic_data
        
        # Split data
        train_size = int(len(df) * 0.8)
        X_train = df[['google', 'facebook', 'tv']].iloc[:train_size]
        y_train = df['revenue'].iloc[:train_size]
        X_test = df[['google', 'facebook', 'tv']].iloc[train_size:]
        
        # Fit and predict
        mmm = RidgeMMM(alpha=1.0)
        mmm.fit(X_train, y_train, channel_configs)
        predictions = mmm.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray)
    
    def test_predict_not_fitted(self, synthetic_data):
        """Test prediction before fitting."""
        mmm = RidgeMMM(alpha=1.0)
        X = synthetic_data[['google', 'facebook', 'tv']]
        
        with pytest.raises(ValueError, match="must be fitted"):
            mmm.predict(X)
    
    def test_evaluate_metrics(self, synthetic_data, channel_configs):
        """Test model evaluation metrics."""
        df = synthetic_data
        
        # Split data
        train_size = int(len(df) * 0.8)
        X_train = df[['google', 'facebook', 'tv']].iloc[:train_size]
        y_train = df['revenue'].iloc[:train_size]
        X_test = df[['google', 'facebook', 'tv']].iloc[train_size:]
        y_test = df['revenue'].iloc[train_size:]
        
        # Fit and evaluate
        mmm = RidgeMMM(alpha=1.0)
        mmm.fit(X_train, y_train, channel_configs)
        metrics = mmm.evaluate(X_test, y_test)
        
        # Check all metrics are present
        assert 'r2' in metrics
        assert 'mape' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        
        # Check metrics are in reasonable ranges
        assert 0 <= metrics['r2'] <= 1  # R² should be between 0 and 1
        assert metrics['mape'] >= 0  # MAPE should be non-negative
        assert metrics['mae'] >= 0  # MAE should be non-negative
        assert metrics['rmse'] >= 0  # RMSE should be non-negative
    
    def test_get_contributions(self, synthetic_data, channel_configs):
        """Test contribution analysis."""
        df = synthetic_data
        X = df[['google', 'facebook', 'tv']]
        y = df['revenue']
        
        mmm = RidgeMMM(alpha=1.0)
        mmm.fit(X, y, channel_configs)
        contributions = mmm.get_contributions(X)
        
        # Check structure
        assert isinstance(contributions, pd.DataFrame)
        assert 'channel' in contributions.columns
        assert 'spend' in contributions.columns
        assert 'contribution' in contributions.columns
        assert 'roas' in contributions.columns
        assert 'contribution_pct' in contributions.columns
        
        # Check all channels are present
        channels = contributions['channel'].tolist()
        assert 'google' in channels
        assert 'facebook' in channels
        assert 'tv' in channels
        assert 'base' in channels
        
        # Check contributions sum to approximately total predicted revenue
        total_contribution = contributions['contribution'].sum()
        total_predicted = mmm.predict(X).sum()
        assert abs(total_contribution - total_predicted) < 1e-6
        
        # Check contribution percentages sum to approximately 100%
        total_pct = contributions['contribution_pct'].sum()
        assert abs(total_pct - 100) < 0.1
    
    def test_get_response_curve(self, synthetic_data, channel_configs):
        """Test response curve generation."""
        df = synthetic_data
        X = df[['google', 'facebook', 'tv']]
        y = df['revenue']
        
        mmm = RidgeMMM(alpha=1.0)
        mmm.fit(X, y, channel_configs)
        
        # Generate response curve for google
        budget_range = np.linspace(0, 50000, 50)
        curve = mmm.get_response_curve('google', budget_range, X)
        
        # Check structure
        assert isinstance(curve, pd.DataFrame)
        assert 'spend' in curve.columns
        assert 'revenue' in curve.columns
        assert 'marginal_roas' in curve.columns
        assert len(curve) == len(budget_range)
        
        # Check revenue is monotonically increasing (or at least non-decreasing)
        # Due to saturation, it should increase but at decreasing rate
        revenue_diffs = np.diff(curve['revenue'].values)
        assert np.all(revenue_diffs >= -1e-6)  # Allow small numerical errors
    
    def test_get_response_curve_invalid_channel(self, synthetic_data, channel_configs):
        """Test response curve with invalid channel."""
        df = synthetic_data
        X = df[['google', 'facebook', 'tv']]
        y = df['revenue']
        
        mmm = RidgeMMM(alpha=1.0)
        mmm.fit(X, y, channel_configs)
        
        budget_range = np.linspace(0, 50000, 50)
        
        with pytest.raises(ValueError, match="not in model"):
            mmm.get_response_curve('invalid_channel', budget_range, X)
    
    def test_decompose_timeseries(self, synthetic_data, channel_configs):
        """Test time series decomposition."""
        df = synthetic_data
        X = df[['date', 'google', 'facebook', 'tv']]
        y = df['revenue']
        
        mmm = RidgeMMM(alpha=1.0)
        mmm.fit(X[['google', 'facebook', 'tv']], y, channel_configs)
        
        decomp = mmm.decompose_timeseries(X, y)
        
        # Check structure
        assert isinstance(decomp, pd.DataFrame)
        assert 'date' in decomp.columns
        assert 'actual_revenue' in decomp.columns
        assert 'base' in decomp.columns
        assert 'google' in decomp.columns
        assert 'facebook' in decomp.columns
        assert 'tv' in decomp.columns
        assert 'predicted_revenue' in decomp.columns
        
        # Check length matches input
        assert len(decomp) == len(X)
        
        # Check predicted revenue is sum of components
        component_cols = ['base', 'google', 'facebook', 'tv']
        calculated_pred = decomp[component_cols].sum(axis=1)
        assert np.allclose(calculated_pred, decomp['predicted_revenue'])
    
    def test_save_load_model(self, synthetic_data, channel_configs):
        """Test model save and load."""
        df = synthetic_data
        X = df[['google', 'facebook', 'tv']]
        y = df['revenue']
        
        # Fit model
        mmm = RidgeMMM(alpha=1.0)
        mmm.fit(X, y, channel_configs)
        
        # Get predictions before saving
        predictions_before = mmm.predict(X)
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            filepath = f.name
        
        try:
            mmm.save_model(filepath)
            
            # Load model
            mmm_loaded = RidgeMMM()
            mmm_loaded.load_model(filepath)
            
            # Check loaded model has same attributes
            assert mmm_loaded.is_fitted == True
            assert mmm_loaded.alpha == mmm.alpha
            assert mmm_loaded.media_channels == mmm.media_channels
            assert mmm_loaded.feature_names == mmm.feature_names
            
            # Check predictions are identical
            predictions_after = mmm_loaded.predict(X)
            np.testing.assert_array_almost_equal(predictions_before, predictions_after)
        
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_save_not_fitted(self):
        """Test saving model before fitting."""
        mmm = RidgeMMM(alpha=1.0)
        
        with pytest.raises(ValueError, match="must be fitted"):
            mmm.save_model('test.pkl')
    
    def test_get_model_summary(self, synthetic_data, channel_configs):
        """Test model summary."""
        df = synthetic_data
        X = df[['google', 'facebook', 'tv', 'seasonality']]
        y = df['revenue']
        
        mmm = RidgeMMM(alpha=1.0)
        mmm.fit(X, y, channel_configs, exog_vars=['seasonality'])
        
        summary = mmm.get_model_summary()
        
        # Check structure
        assert 'alpha' in summary
        assert 'n_channels' in summary
        assert 'channels' in summary
        assert 'n_exog_vars' in summary
        assert 'exog_vars' in summary
        assert 'n_features' in summary
        assert 'intercept' in summary
        assert 'coefficients' in summary
        
        # Check values
        assert summary['alpha'] == 1.0
        assert summary['n_channels'] == 3
        assert summary['n_exog_vars'] == 1
        assert summary['n_features'] == 4
    
    def test_integration_full_workflow(self, synthetic_data, channel_configs):
        """Integration test for complete workflow."""
        df = synthetic_data
        
        # Split data
        train_size = int(len(df) * 0.8)
        X_train = df[['google', 'facebook', 'tv', 'seasonality']].iloc[:train_size]
        y_train = df['revenue'].iloc[:train_size]
        X_test = df[['google', 'facebook', 'tv', 'seasonality']].iloc[train_size:]
        y_test = df['revenue'].iloc[train_size:]
        
        # Initialize and fit
        mmm = RidgeMMM(alpha=1.0)
        mmm.fit(
            X_train[['google', 'facebook', 'tv', 'seasonality']], 
            y_train, 
            channel_configs,
            exog_vars=['seasonality']
        )
        
        # Predict
        predictions = mmm.predict(X_test)
        assert len(predictions) == len(X_test)
        
        # Evaluate
        metrics = mmm.evaluate(X_test, y_test)
        assert metrics['r2'] > 0.15  # Reasonable fit on synthetic data with exog vars
        
        # Get contributions
        contributions = mmm.get_contributions(X_train)
        assert len(contributions) == 4  # 3 channels + base
        
        # Response curve
        budget_range = np.linspace(0, 50000, 50)
        curve = mmm.get_response_curve('google', budget_range, X_train)
        assert len(curve) == 50
        
        # Decompose
        decomp = mmm.decompose_timeseries(X_train, y_train)
        assert len(decomp) == len(X_train)
        
        # Save and load
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            filepath = f.name
        
        try:
            mmm.save_model(filepath)
            mmm_loaded = RidgeMMM()
            mmm_loaded.load_model(filepath)
            
            # Verify loaded model works
            predictions_loaded = mmm_loaded.predict(X_test)
            np.testing.assert_array_almost_equal(predictions, predictions_loaded)
        
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_different_alpha_values(self, synthetic_data, channel_configs):
        """Test that different alpha values produce different results."""
        df = synthetic_data
        X = df[['google', 'facebook', 'tv']]
        y = df['revenue']
        
        # Fit with low alpha
        mmm_low = RidgeMMM(alpha=0.1)
        mmm_low.fit(X, y, channel_configs)
        pred_low = mmm_low.predict(X)
        
        # Fit with high alpha
        mmm_high = RidgeMMM(alpha=10.0)
        mmm_high.fit(X, y, channel_configs)
        pred_high = mmm_high.predict(X)
        
        # Predictions should be different
        assert not np.allclose(pred_low, pred_high)
