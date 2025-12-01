"""
Tests for Weibull Adstock Transformation.

This module tests the WeibullAdstockTransformer class to ensure:
- Correct initialization and parameter validation
- Proper transformation behavior for different shape/scale combinations
- Delayed peak effects for shape > 1
- Immediate peak for shape < 1
- Normalization (total effect equals total spend)
- Utility methods (peak_lag, half_life, decay_curve)
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformations import WeibullAdstockTransformer, AdstockType


class TestWeibullAdstockTransformerInit:
    """Test WeibullAdstockTransformer initialization."""

    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        transformer = WeibullAdstockTransformer(shape=2.0, scale=3.0)
        assert transformer.shape == 2.0
        assert transformer.scale == 3.0
        assert transformer.normalization > 0

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        transformer = WeibullAdstockTransformer()
        assert transformer.shape == 2.0
        assert transformer.scale == 3.0

    def test_init_invalid_shape_negative(self):
        """Test that negative shape raises error."""
        with pytest.raises(ValueError, match="Shape must be positive"):
            WeibullAdstockTransformer(shape=-1.0, scale=3.0)

    def test_init_invalid_shape_zero(self):
        """Test that zero shape raises error."""
        with pytest.raises(ValueError, match="Shape must be positive"):
            WeibullAdstockTransformer(shape=0.0, scale=3.0)

    def test_init_invalid_scale_negative(self):
        """Test that negative scale raises error."""
        with pytest.raises(ValueError, match="Scale must be positive"):
            WeibullAdstockTransformer(shape=2.0, scale=-1.0)

    def test_init_invalid_scale_zero(self):
        """Test that zero scale raises error."""
        with pytest.raises(ValueError, match="Scale must be positive"):
            WeibullAdstockTransformer(shape=2.0, scale=0.0)

    def test_init_extreme_parameters(self):
        """Test initialization with extreme but valid parameters."""
        # Very small values
        transformer1 = WeibullAdstockTransformer(shape=0.1, scale=0.5)
        assert transformer1.shape == 0.1
        assert transformer1.scale == 0.5

        # Very large values
        transformer2 = WeibullAdstockTransformer(shape=10.0, scale=15.0)
        assert transformer2.shape == 10.0
        assert transformer2.scale == 15.0


class TestWeibullAdstockTransformerTransform:
    """Test Weibull adstock transformation."""

    def test_transform_delayed_peak(self):
        """Test Weibull with shape>1 shows delayed peak."""
        transformer = WeibullAdstockTransformer(shape=2.5, scale=3.0)

        # Spend only in week 0
        spend = np.array([100, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        adstocked = transformer.transform(spend)

        # Peak should not be at week 0
        peak_idx = np.argmax(adstocked)
        assert peak_idx > 0, "Peak should be delayed for shape > 1"
        assert peak_idx <= 5, "Peak should occur within reasonable range"

        # Effect should build up then decay
        assert adstocked[0] < adstocked[peak_idx]

    def test_transform_immediate_peak(self):
        """Test Weibull with shape<1 shows immediate peak."""
        transformer = WeibullAdstockTransformer(shape=0.8, scale=3.0)

        spend = np.array([100, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        adstocked = transformer.transform(spend)

        # Peak should be at week 0
        assert np.argmax(adstocked) == 0, "Peak should be immediate for shape < 1"

    def test_transform_exponential_like(self):
        """Test Weibull with shape=1 behaves like exponential."""
        transformer = WeibullAdstockTransformer(shape=1.0, scale=3.0)

        spend = np.array([100, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        adstocked = transformer.transform(spend)

        # Peak should be at t=0 or t=1 (approximately exponential)
        peak_idx = np.argmax(adstocked)
        assert peak_idx <= 1, "Peak should be at start for shape ≈ 1"

    def test_transform_normalization(self):
        """Test that total effect approximately equals total spend."""
        transformer = WeibullAdstockTransformer(shape=2.0, scale=3.0)

        spend = np.array([100])
        # Extend to capture all decay (52 weeks)
        spend_extended = np.concatenate([spend, np.zeros(51)])

        adstocked = transformer.transform(spend_extended)

        # Total effect should approximately equal spend (within 10% tolerance)
        total_effect = adstocked.sum()
        assert abs(total_effect - 100) < 10, \
            f"Total effect ({total_effect:.2f}) should approximate spend (100)"

    def test_transform_multiple_spends(self):
        """Test transformation with multiple spend events."""
        transformer = WeibullAdstockTransformer(shape=2.0, scale=3.0)

        # Spend in weeks 0, 3, 6
        spend = np.array([100, 0, 0, 50, 0, 0, 75, 0, 0, 0, 0, 0, 0, 0, 0])
        adstocked = transformer.transform(spend)

        # Should have non-negative values throughout
        assert np.all(adstocked >= 0)

        # Should have positive total effect
        assert adstocked.sum() > 0

        # Total effect should be close to total spend (normalization property)
        total_spend = spend.sum()
        total_effect = adstocked.sum()
        assert abs(total_effect - total_spend) / total_spend < 0.2  # Within 20%

    def test_transform_empty_array(self):
        """Test transformation with empty array."""
        transformer = WeibullAdstockTransformer(shape=2.0, scale=3.0)

        with pytest.raises(ValueError, match="cannot be empty"):
            transformer.transform(np.array([]))

    def test_transform_zero_spend(self):
        """Test transformation with all zeros."""
        transformer = WeibullAdstockTransformer(shape=2.0, scale=3.0)

        spend = np.zeros(10)
        adstocked = transformer.transform(spend)

        # Should return all zeros
        assert np.allclose(adstocked, np.zeros(10))

    def test_transform_nan_values(self):
        """Test transformation with NaN values."""
        transformer = WeibullAdstockTransformer(shape=2.0, scale=3.0)

        spend = np.array([100, np.nan, 50, 0, 0])

        with pytest.warns(UserWarning, match="NaN"):
            adstocked = transformer.transform(spend)

        # NaN should be treated as 0
        assert np.all(np.isfinite(adstocked))

    def test_transform_negative_values(self):
        """Test transformation with negative values."""
        transformer = WeibullAdstockTransformer(shape=2.0, scale=3.0)

        spend = np.array([100, -50, 75, 0, 0])

        with pytest.warns(UserWarning, match="negative"):
            adstocked = transformer.transform(spend)

        # Negative values should be treated as 0
        assert np.all(adstocked >= 0)

    def test_transform_list_input(self):
        """Test transformation accepts list input."""
        transformer = WeibullAdstockTransformer(shape=2.0, scale=3.0)

        spend = [100, 0, 0, 0, 0]
        adstocked = transformer.transform(spend)

        # Should work and return numpy array
        assert isinstance(adstocked, np.ndarray)
        assert len(adstocked) == 5


class TestWeibullAdstockTransformerUtilities:
    """Test utility methods of WeibullAdstockTransformer."""

    def test_get_peak_lag_delayed(self):
        """Test peak lag calculation for delayed peak (shape > 1)."""
        transformer = WeibullAdstockTransformer(shape=2.5, scale=4.0)
        peak_lag = transformer.get_peak_lag()

        assert peak_lag > 0, "Peak lag should be positive for shape > 1"
        assert peak_lag < 10, "Peak lag should be reasonable"

    def test_get_peak_lag_immediate(self):
        """Test peak lag for immediate peak (shape < 1)."""
        transformer = WeibullAdstockTransformer(shape=0.8, scale=4.0)
        peak_lag = transformer.get_peak_lag()

        assert peak_lag == 0.0, "Peak lag should be 0 for shape < 1"

    def test_get_peak_lag_shape_one(self):
        """Test peak lag for shape = 1."""
        transformer = WeibullAdstockTransformer(shape=1.0, scale=4.0)
        peak_lag = transformer.get_peak_lag()

        assert peak_lag == 0.0, "Peak lag should be 0 for shape = 1"

    def test_get_half_life(self):
        """Test half-life calculation."""
        transformer = WeibullAdstockTransformer(shape=2.0, scale=3.0)
        half_life = transformer.get_half_life()

        assert half_life > 0, "Half-life must be positive"
        assert half_life < 20, "Half-life should be in reasonable range"

    def test_get_half_life_varies_with_scale(self):
        """Test that half-life increases with scale."""
        transformer1 = WeibullAdstockTransformer(shape=2.0, scale=2.0)
        transformer2 = WeibullAdstockTransformer(shape=2.0, scale=5.0)

        hl1 = transformer1.get_half_life()
        hl2 = transformer2.get_half_life()

        assert hl2 > hl1, "Higher scale should have longer half-life"

    def test_get_decay_curve(self):
        """Test decay curve generation."""
        transformer = WeibullAdstockTransformer(shape=2.0, scale=3.0)
        curve = transformer.get_decay_curve(length=10)

        # Check properties
        assert len(curve) == 10
        assert np.all(curve >= 0), "Decay curve should be non-negative"
        assert np.sum(curve) > 0, "Decay curve should have positive values"

    def test_get_decay_curve_custom_length(self):
        """Test decay curve with different lengths."""
        transformer = WeibullAdstockTransformer(shape=2.0, scale=3.0)

        curve5 = transformer.get_decay_curve(length=5)
        curve20 = transformer.get_decay_curve(length=20)

        assert len(curve5) == 5
        assert len(curve20) == 20

    def test_get_decay_curve_invalid_length(self):
        """Test decay curve with invalid length."""
        transformer = WeibullAdstockTransformer(shape=2.0, scale=3.0)

        with pytest.raises(ValueError, match="must be positive"):
            transformer.get_decay_curve(length=0)

        with pytest.raises(ValueError, match="must be positive"):
            transformer.get_decay_curve(length=-5)


class TestWeibullComparison:
    """Test Weibull vs Geometric comparison."""

    def test_weibull_vs_geometric_delayed_peak(self):
        """Verify Weibull can produce delayed peak unlike geometric."""
        from transformations import AdstockTransformer

        # Geometric adstock (always immediate peak)
        geometric = AdstockTransformer(decay_rate=0.5)
        geo_spend = np.array([100, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        geo_result = geometric.transform(geo_spend)

        # Weibull adstock (delayed peak)
        weibull = WeibullAdstockTransformer(shape=2.5, scale=3.0)
        weib_spend = np.array([100, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        weib_result = weibull.transform(weib_spend)

        # Geometric peaks at t=0
        assert np.argmax(geo_result) == 0

        # Weibull peaks later
        assert np.argmax(weib_result) > 0

    def test_weibull_shape_effect_on_peak(self):
        """Test that shape parameter controls peak timing."""
        spend = np.array([100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # Low shape: early peak
        transformer1 = WeibullAdstockTransformer(shape=1.2, scale=4.0)
        result1 = transformer1.transform(spend)
        peak1 = np.argmax(result1)

        # High shape: later peak
        transformer2 = WeibullAdstockTransformer(shape=3.0, scale=4.0)
        result2 = transformer2.transform(spend)
        peak2 = np.argmax(result2)

        assert peak2 > peak1, "Higher shape should delay peak"


class TestAdstockTypeEnum:
    """Test AdstockType enum."""

    def test_adstock_type_values(self):
        """Test AdstockType enum values."""
        assert AdstockType.GEOMETRIC == "geometric"
        assert AdstockType.WEIBULL == "weibull"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_long_timeseries(self):
        """Test with long time series."""
        transformer = WeibullAdstockTransformer(shape=2.0, scale=3.0)

        # 104 weeks (2 years)
        spend = np.zeros(104)
        spend[0] = 1000

        adstocked = transformer.transform(spend)

        # Should handle long series without error
        assert len(adstocked) == 104
        assert np.all(np.isfinite(adstocked))

    def test_very_large_spend(self):
        """Test with very large spend values."""
        transformer = WeibullAdstockTransformer(shape=2.0, scale=3.0)

        spend = np.array([1e10, 0, 0, 0, 0])
        adstocked = transformer.transform(spend)

        # Should handle large values without overflow
        assert np.all(np.isfinite(adstocked))
        assert np.all(adstocked >= 0)

    def test_very_small_spend(self):
        """Test with very small spend values."""
        transformer = WeibullAdstockTransformer(shape=2.0, scale=3.0)

        spend = np.array([1e-10, 0, 0, 0, 0])
        adstocked = transformer.transform(spend)

        # Should handle small values without underflow
        assert np.all(np.isfinite(adstocked))
        assert np.all(adstocked >= 0)

    def test_single_element_array(self):
        """Test with single-element array."""
        transformer = WeibullAdstockTransformer(shape=2.0, scale=3.0)

        spend = np.array([100])
        adstocked = transformer.transform(spend)

        assert len(adstocked) == 1
        # With delayed peak (shape > 1), single element may have low/zero effect
        # since peak occurs later. The important thing is no error.
        assert adstocked[0] >= 0
        assert np.isfinite(adstocked[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
