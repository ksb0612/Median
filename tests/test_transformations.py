"""
Tests for transformation functions.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformations import AdstockTransformer, HillTransformer, TransformationPipeline


class TestAdstockTransformer:
    """Tests for AdstockTransformer class."""
    
    def test_init_valid_decay_rate(self):
        """Test initialization with valid decay rate."""
        transformer = AdstockTransformer(decay_rate=0.5)
        assert transformer.decay_rate == 0.5
    
    def test_init_invalid_decay_rate_too_high(self):
        """Test initialization with decay rate > 1."""
        with pytest.raises(ValueError, match="decay_rate must be between 0 and 1"):
            AdstockTransformer(decay_rate=1.5)
    
    def test_init_invalid_decay_rate_negative(self):
        """Test initialization with negative decay rate."""
        with pytest.raises(ValueError, match="decay_rate must be between 0 and 1"):
            AdstockTransformer(decay_rate=-0.1)
    
    def test_transform_simple_case(self):
        """Test transformation with simple input [1,0,0,0,0]."""
        transformer = AdstockTransformer(decay_rate=0.5)
        x = np.array([1, 0, 0, 0, 0])
        result = transformer.transform(x)
        
        expected = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_transform_with_multiple_spikes(self):
        """Test transformation with multiple non-zero values."""
        transformer = AdstockTransformer(decay_rate=0.5)
        x = np.array([100, 0, 0, 50, 0])
        result = transformer.transform(x)
        
        # Manual calculation:
        # t=0: 100
        # t=1: 0 + 0.5*100 = 50
        # t=2: 0 + 0.5*50 = 25
        # t=3: 50 + 0.5*25 = 62.5
        # t=4: 0 + 0.5*62.5 = 31.25
        expected = np.array([100.0, 50.0, 25.0, 62.5, 31.25])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_transform_zero_decay(self):
        """Test transformation with zero decay (no carryover)."""
        transformer = AdstockTransformer(decay_rate=0.0)
        x = np.array([1, 2, 3, 4, 5])
        result = transformer.transform(x)
        
        # With zero decay, output should equal input
        np.testing.assert_array_almost_equal(result, x)
    
    def test_transform_high_decay(self):
        """Test transformation with high decay rate."""
        transformer = AdstockTransformer(decay_rate=0.9)
        x = np.array([1, 0, 0, 0, 0])
        result = transformer.transform(x)
        
        expected = np.array([1.0, 0.9, 0.81, 0.729, 0.6561])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_transform_empty_array(self):
        """Test transformation with empty array."""
        transformer = AdstockTransformer(decay_rate=0.5)
        with pytest.raises(ValueError, match="Input array cannot be empty"):
            transformer.transform(np.array([]))
    
    def test_transform_with_nan(self):
        """Test transformation with NaN values."""
        transformer = AdstockTransformer(decay_rate=0.5)
        x = np.array([1, np.nan, 0, 0, 0])
        
        with pytest.warns(UserWarning, match="NaN values"):
            result = transformer.transform(x)
        
        # NaN should be treated as 0
        expected = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_transform_with_negative(self):
        """Test transformation with negative values."""
        transformer = AdstockTransformer(decay_rate=0.5)
        x = np.array([1, -1, 0, 0, 0])
        
        with pytest.warns(UserWarning, match="negative values"):
            result = transformer.transform(x)
        
        # Negative should be treated as 0
        expected = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_transform_list_input(self):
        """Test transformation with list input (should convert to array)."""
        transformer = AdstockTransformer(decay_rate=0.5)
        x = [1, 0, 0, 0, 0]
        result = transformer.transform(x)
        
        expected = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_get_decay_curve(self):
        """Test decay curve generation."""
        transformer = AdstockTransformer(decay_rate=0.5)
        curve = transformer.get_decay_curve(length=5)
        
        expected = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        np.testing.assert_array_almost_equal(curve, expected)
    
    def test_get_decay_curve_invalid_length(self):
        """Test decay curve with invalid length."""
        transformer = AdstockTransformer(decay_rate=0.5)
        with pytest.raises(ValueError, match="length must be positive"):
            transformer.get_decay_curve(length=0)


class TestHillTransformer:
    """Tests for HillTransformer class."""
    
    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        transformer = HillTransformer(K=1.0, S=1.0)
        assert transformer.K == 1.0
        assert transformer.S == 1.0
    
    def test_init_invalid_K(self):
        """Test initialization with invalid K."""
        with pytest.raises(ValueError, match="K must be positive"):
            HillTransformer(K=0, S=1.0)
        
        with pytest.raises(ValueError, match="K must be positive"):
            HillTransformer(K=-1.0, S=1.0)
    
    def test_init_invalid_S(self):
        """Test initialization with invalid S."""
        with pytest.raises(ValueError, match="S must be positive"):
            HillTransformer(K=1.0, S=0)
        
        with pytest.raises(ValueError, match="S must be positive"):
            HillTransformer(K=1.0, S=-1.0)
    
    def test_transform_standard_case(self):
        """Test transformation with K=1, S=1."""
        transformer = HillTransformer(K=1.0, S=1.0)
        x = np.array([0, 0.5, 1, 2, 5, 10])
        result = transformer.transform(x)
        
        # Manual calculation: y = K * x^S / (x^S + 1)
        # x=0: 1 * 0 / (0 + 1) = 0
        # x=0.5: 1 * 0.5 / (0.5 + 1) = 0.333...
        # x=1: 1 * 1 / (1 + 1) = 0.5
        # x=2: 1 * 2 / (2 + 1) = 0.666...
        # x=5: 1 * 5 / (5 + 1) = 0.833...
        # x=10: 1 * 10 / (10 + 1) = 0.909...
        expected = np.array([0.0, 0.333333, 0.5, 0.666667, 0.833333, 0.909091])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
    
    def test_transform_zero_input(self):
        """Test transformation with zero input."""
        transformer = HillTransformer(K=1.0, S=1.0)
        x = np.array([0])
        result = transformer.transform(x)
        
        assert result[0] == 0.0
    
    def test_transform_different_K(self):
        """Test transformation with different K values."""
        transformer = HillTransformer(K=2.0, S=1.0)
        x = np.array([1])
        result = transformer.transform(x)
        
        # y = 2 * 1 / (1 + 1) = 1.0
        assert result[0] == pytest.approx(1.0)
    
    def test_transform_different_S(self):
        """Test transformation with different S values."""
        transformer = HillTransformer(K=1.0, S=2.0)
        x = np.array([1])
        result = transformer.transform(x)
        
        # y = 1 * 1^2 / (1^2 + 1) = 0.5
        assert result[0] == pytest.approx(0.5)
    
    def test_transform_empty_array(self):
        """Test transformation with empty array."""
        transformer = HillTransformer(K=1.0, S=1.0)
        with pytest.raises(ValueError, match="Input array cannot be empty"):
            transformer.transform(np.array([]))
    
    def test_transform_with_nan(self):
        """Test transformation with NaN values."""
        transformer = HillTransformer(K=1.0, S=1.0)
        x = np.array([1, np.nan, 2])
        
        with pytest.warns(UserWarning, match="NaN values"):
            result = transformer.transform(x)
        
        # NaN should be treated as 0
        assert result[1] == 0.0
    
    def test_transform_with_negative(self):
        """Test transformation with negative values."""
        transformer = HillTransformer(K=1.0, S=1.0)
        x = np.array([1, -1, 2])
        
        with pytest.warns(UserWarning, match="negative values"):
            result = transformer.transform(x)
        
        # Negative should return 0
        assert result[1] == 0.0
    
    def test_transform_large_values(self):
        """Test transformation with very large values."""
        transformer = HillTransformer(K=1.0, S=1.0)
        x = np.array([1000, 10000])
        result = transformer.transform(x)
        
        # For large x, function should approach K
        assert result[0] > 0.99
        assert result[1] > 0.999
    
    def test_get_response_curve(self):
        """Test response curve generation."""
        transformer = HillTransformer(K=1.0, S=1.0)
        x_range = np.array([0, 1, 2, 5, 10])
        curve = transformer.get_response_curve(x_range)
        
        # Should be same as transform
        expected = transformer.transform(x_range)
        np.testing.assert_array_almost_equal(curve, expected)


class TestTransformationPipeline:
    """Tests for TransformationPipeline class."""
    
    def test_init_valid_config(self):
        """Test initialization with valid configuration."""
        configs = {
            'google': {'adstock': 0.5, 'hill_K': 1.0, 'hill_S': 1.0},
            'facebook': {'adstock': 0.7, 'hill_K': 1.2, 'hill_S': 0.9}
        }
        pipeline = TransformationPipeline(configs)
        
        assert 'google' in pipeline.channel_configs
        assert 'facebook' in pipeline.channel_configs
    
    def test_init_empty_config(self):
        """Test initialization with empty configuration."""
        with pytest.raises(ValueError, match="channel_configs cannot be empty"):
            TransformationPipeline({})
    
    def test_init_missing_keys(self):
        """Test initialization with missing required keys."""
        configs = {
            'google': {'adstock': 0.5}  # Missing hill_K and hill_S
        }
        with pytest.raises(ValueError, match="missing Hill parameters"):
            TransformationPipeline(configs)
    
    def test_fit_transform_single_channel(self):
        """Test transformation with single channel."""
        configs = {
            'google': {'adstock': 0.5, 'hill_K': 1.0, 'hill_S': 1.0}
        }
        pipeline = TransformationPipeline(configs)
        
        df = pd.DataFrame({
            'google': [1, 0, 0, 0, 0]
        })
        
        result = pipeline.fit_transform(df)
        
        # Should apply adstock then Hill
        # Adstock: [1, 0.5, 0.25, 0.125, 0.0625]
        # Hill: apply to each value
        assert 'google' in result.columns
        assert len(result) == 5
    
    def test_fit_transform_multiple_channels(self):
        """Test transformation with multiple channels."""
        configs = {
            'google': {'adstock': 0.5, 'hill_K': 1.0, 'hill_S': 1.0},
            'facebook': {'adstock': 0.7, 'hill_K': 1.2, 'hill_S': 0.9}
        }
        pipeline = TransformationPipeline(configs)
        
        df = pd.DataFrame({
            'google': [100, 200, 150, 100, 50],
            'facebook': [50, 75, 60, 80, 90]
        })
        
        result = pipeline.fit_transform(df)
        
        assert 'google' in result.columns
        assert 'facebook' in result.columns
        assert len(result) == 5
        
        # Values should be different from original (transformed)
        assert not np.array_equal(result['google'].values, df['google'].values)
        assert not np.array_equal(result['facebook'].values, df['facebook'].values)
    
    def test_fit_transform_preserves_other_columns(self):
        """Test that transformation preserves non-configured columns."""
        configs = {
            'google': {'adstock': 0.5, 'hill_K': 1.0, 'hill_S': 1.0}
        }
        pipeline = TransformationPipeline(configs)
        
        df = pd.DataFrame({
            'date': ['2022-01-01', '2022-01-08', '2022-01-15'],
            'google': [100, 200, 150],
            'revenue': [10000, 12000, 11000]
        })
        
        result = pipeline.fit_transform(df)
        
        # Non-configured columns should remain unchanged
        assert result['date'].equals(df['date'])
        assert result['revenue'].equals(df['revenue'])
        
        # Google should be transformed
        assert not result['google'].equals(df['google'])
    
    def test_fit_transform_empty_dataframe(self):
        """Test transformation with empty DataFrame."""
        configs = {
            'google': {'adstock': 0.5, 'hill_K': 1.0, 'hill_S': 1.0}
        }
        pipeline = TransformationPipeline(configs)
        
        df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Input DataFrame cannot be empty"):
            pipeline.fit_transform(df)
    
    def test_fit_transform_missing_channel(self):
        """Test transformation when configured channel is missing."""
        configs = {
            'google': {'adstock': 0.5, 'hill_K': 1.0, 'hill_S': 1.0},
            'facebook': {'adstock': 0.7, 'hill_K': 1.2, 'hill_S': 0.9}
        }
        pipeline = TransformationPipeline(configs)
        
        df = pd.DataFrame({
            'google': [100, 200, 150]
            # facebook is missing
        })
        
        with pytest.raises(ValueError, match="not found in DataFrame"):
            pipeline.fit_transform(df)
    
    def test_get_transformation_summary(self):
        """Test getting transformation summary."""
        configs = {
            'google': {'adstock': 0.5, 'hill_K': 1.0, 'hill_S': 1.0},
            'facebook': {'adstock': 0.7, 'hill_K': 1.2, 'hill_S': 0.9}
        }
        pipeline = TransformationPipeline(configs)
        
        summary = pipeline.get_transformation_summary()
        
        assert summary == configs
        assert 'google' in summary
        assert 'facebook' in summary
        assert summary['google']['adstock'] == 0.5
        assert summary['facebook']['hill_K'] == 1.2
    
    def test_transform_single_channel_method(self):
        """Test single channel transformation method."""
        configs = {
            'google': {'adstock': 0.5, 'hill_K': 1.0, 'hill_S': 1.0}
        }
        pipeline = TransformationPipeline(configs)
        
        values = np.array([1, 0, 0, 0, 0])
        original, adstocked, transformed = pipeline.transform_single_channel('google', values)
        
        # Original should be unchanged
        np.testing.assert_array_equal(original, values)
        
        # Adstocked should show decay
        expected_adstock = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        np.testing.assert_array_almost_equal(adstocked, expected_adstock)
        
        # Transformed should have Hill applied
        assert len(transformed) == 5
    
    def test_transform_single_channel_invalid(self):
        """Test single channel transformation with invalid channel."""
        configs = {
            'google': {'adstock': 0.5, 'hill_K': 1.0, 'hill_S': 1.0}
        }
        pipeline = TransformationPipeline(configs)
        
        values = np.array([1, 0, 0, 0, 0])
        
        with pytest.raises(ValueError, match="not in configuration"):
            pipeline.transform_single_channel('facebook', values)
    
    def test_integration_full_pipeline(self):
        """Integration test with realistic data."""
        configs = {
            'google_uac': {'adstock': 0.5, 'hill_K': 1.0, 'hill_S': 1.0},
            'meta': {'adstock': 0.6, 'hill_K': 1.1, 'hill_S': 0.95},
            'apple_search': {'adstock': 0.4, 'hill_K': 0.9, 'hill_S': 1.05}
        }
        pipeline = TransformationPipeline(configs)
        
        # Create realistic data
        df = pd.DataFrame({
            'date': pd.date_range('2022-01-01', periods=10, freq='W'),
            'google_uac': [15000, 16000, 14000, 17000, 15500, 16500, 15000, 18000, 17500, 16000],
            'meta': [12000, 13000, 11500, 14000, 12500, 13500, 12000, 15000, 14500, 13000],
            'apple_search': [8000, 8500, 7500, 9000, 8200, 8800, 7800, 9500, 9200, 8500],
            'revenue': [125000, 130000, 128000, 135000, 132000, 138000, 130000, 142000, 140000, 135000]
        })
        
        result = pipeline.fit_transform(df)
        
        # Check all channels are transformed
        assert 'google_uac' in result.columns
        assert 'meta' in result.columns
        assert 'apple_search' in result.columns
        
        # Check non-media columns are preserved
        assert result['date'].equals(df['date'])
        assert result['revenue'].equals(df['revenue'])
        
        # Check values are different (transformed)
        assert not result['google_uac'].equals(df['google_uac'])
        assert not result['meta'].equals(df['meta'])
        assert not result['apple_search'].equals(df['apple_search'])
        
        # Check no NaN values introduced
        assert not result.isnull().any().any()
