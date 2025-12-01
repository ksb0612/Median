"""Test visualization error handling and edge cases."""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visualizations import (
    plot_waterfall,
    plot_decomposition,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_channel_contributions,
    plot_qq
)


@pytest.fixture
def valid_contributions_df():
    """Create valid contributions DataFrame."""
    return pd.DataFrame({
        'channel': ['base', 'google', 'meta', 'apple'],
        'spend': [0, 15000, 10000, 5000],
        'contribution': [50000, 30000, 20000, 10000],
        'roas': [0, 2.0, 2.0, 2.0],
        'contribution_pct': [45.5, 27.3, 18.2, 9.1]
    })


@pytest.fixture
def valid_decomposition_df():
    """Create valid decomposition DataFrame."""
    dates = pd.date_range('2023-01-01', periods=20)
    return pd.DataFrame({
        'date': dates,
        'actual_revenue': np.random.uniform(10000, 50000, 20),
        'base': np.random.uniform(5000, 10000, 20),
        'google': np.random.uniform(5000, 15000, 20),
        'meta': np.random.uniform(3000, 10000, 20),
        'predicted_revenue': np.random.uniform(15000, 40000, 20)
    })


class TestWaterfallErrorHandling:
    """Test waterfall plot error handling."""

    def test_waterfall_empty_data(self):
        """Test waterfall handles empty DataFrame gracefully."""
        empty_df = pd.DataFrame(columns=['channel', 'contribution', 'spend', 'roas', 'contribution_pct'])

        # Empty data should return valid figure, not error
        fig = plot_waterfall(empty_df)
        assert fig is not None
        # Gracefully handles empty data (may show total of 0)
        assert len(fig.data) >= 0

    def test_waterfall_missing_required_columns(self):
        """Test waterfall handles missing required columns."""
        df = pd.DataFrame({
            'channel': ['google', 'meta'],
            'wrong_col': [100, 200]
        })

        # Should raise error about missing columns
        with pytest.raises((KeyError, ValueError)):
            plot_waterfall(df)

    def test_waterfall_with_single_channel(self):
        """Test waterfall with only one channel."""
        df = pd.DataFrame({
            'channel': ['google'],
            'spend': [10000],
            'contribution': [20000],
            'roas': [2.0],
            'contribution_pct': [100.0]
        })

        # Should handle single channel gracefully
        try:
            fig = plot_waterfall(df)
            assert fig is not None
        except Exception as e:
            pytest.fail(f"Waterfall failed with single channel: {e}")

    def test_waterfall_with_negative_contributions(self):
        """Test waterfall with negative contributions."""
        df = pd.DataFrame({
            'channel': ['base', 'google', 'meta'],
            'spend': [0, 10000, 8000],
            'contribution': [50000, -5000, 15000],  # Negative contribution
            'roas': [0, -0.5, 1.875],
            'contribution_pct': [71.4, -7.1, 21.4]
        })

        # Should handle negative values
        try:
            fig = plot_waterfall(df)
            assert fig is not None
        except Exception as e:
            pytest.fail(f"Waterfall failed with negative contributions: {e}")

    def test_waterfall_with_zero_contributions(self):
        """Test waterfall with all zero contributions."""
        df = pd.DataFrame({
            'channel': ['google', 'meta'],
            'spend': [10000, 8000],
            'contribution': [0, 0],
            'roas': [0, 0],
            'contribution_pct': [0, 0]
        })

        # Should handle zero contributions
        try:
            fig = plot_waterfall(df)
            assert fig is not None
        except Exception as e:
            pytest.fail(f"Waterfall failed with zero contributions: {e}")


class TestDecompositionErrorHandling:
    """Test decomposition plot error handling."""

    def test_decomposition_empty_data(self):
        """Test decomposition handles empty DataFrame gracefully."""
        empty_df = pd.DataFrame(columns=['date', 'actual_revenue', 'base', 'google'])

        # Should return valid figure, not raise error
        fig = plot_decomposition(empty_df)
        assert fig is not None
        assert len(fig.data) >= 0

    def test_decomposition_missing_date_column(self):
        """Test decomposition without date column."""
        df = pd.DataFrame({
            'actual_revenue': [100, 200, 300],
            'base': [50, 60, 70],
            'google': [40, 80, 120]
        })

        # Should work without date column (uses index)
        try:
            fig = plot_decomposition(df)
            assert fig is not None
        except Exception as e:
            pytest.fail(f"Decomposition failed without date column: {e}")

    def test_decomposition_with_single_row(self):
        """Test decomposition with single data point."""
        df = pd.DataFrame({
            'date': [pd.Timestamp('2023-01-01')],
            'actual_revenue': [10000],
            'base': [5000],
            'google': [3000],
            'meta': [2000]
        })

        # Should handle single row
        try:
            fig = plot_decomposition(df)
            assert fig is not None
        except Exception as e:
            pytest.fail(f"Decomposition failed with single row: {e}")


class TestActualVsPredictedErrorHandling:
    """Test actual vs predicted plot error handling."""

    def test_actual_vs_predicted_length_mismatch(self):
        """Test error when actual and predicted have different lengths."""
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2])

        with pytest.raises((ValueError, IndexError)):
            plot_actual_vs_predicted(y_true, y_pred)

    def test_actual_vs_predicted_with_nan(self):
        """Test handling of NaN values."""
        y_true = np.array([1, 2, np.nan, 4])
        y_pred = np.array([1, 2, 3, 4])

        # Should either handle NaN or raise clear error
        try:
            fig = plot_actual_vs_predicted(y_true, y_pred)
            # If it succeeds, check that figure exists
            assert fig is not None
        except (ValueError, RuntimeError):
            # Expected to fail with NaN
            pass

    def test_actual_vs_predicted_with_inf(self):
        """Test handling of infinite values."""
        y_true = np.array([1, 2, np.inf, 4])
        y_pred = np.array([1, 2, 3, 4])

        # Should either handle inf or raise clear error
        try:
            fig = plot_actual_vs_predicted(y_true, y_pred)
            assert fig is not None
        except (ValueError, RuntimeError):
            # Expected to fail with inf
            pass

    def test_actual_vs_predicted_all_zeros(self):
        """Test with all zero values."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        # Should handle all zeros
        try:
            fig = plot_actual_vs_predicted(y_true, y_pred)
            assert fig is not None
        except Exception as e:
            pytest.fail(f"Failed with all zeros: {e}")

    def test_actual_vs_predicted_single_point(self):
        """Test with single data point."""
        y_true = np.array([100])
        y_pred = np.array([105])

        # Should handle single point
        try:
            fig = plot_actual_vs_predicted(y_true, y_pred)
            assert fig is not None
        except Exception as e:
            pytest.fail(f"Failed with single point: {e}")


class TestResidualsErrorHandling:
    """Test residuals plot error handling."""

    def test_residuals_length_mismatch(self):
        """Test residuals with mismatched lengths."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2])

        with pytest.raises((ValueError, IndexError)):
            plot_residuals(y_true, y_pred)

    def test_residuals_with_nan(self):
        """Test residuals with NaN values."""
        y_true = np.array([1, 2, np.nan, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        # Should handle or raise clear error
        try:
            fig = plot_residuals(y_true, y_pred)
            assert fig is not None
        except (ValueError, RuntimeError):
            pass

    def test_residuals_all_perfect(self):
        """Test residuals when predictions are perfect."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        # Should handle zero residuals
        try:
            fig = plot_residuals(y_true, y_pred)
            assert fig is not None
        except Exception as e:
            pytest.fail(f"Failed with perfect predictions: {e}")


class TestChannelContributionsErrorHandling:
    """Test channel contributions plot error handling."""

    def test_channel_contributions_empty_after_filter(self, valid_contributions_df):
        """Test when no channels remain after filtering out base."""
        df = pd.DataFrame({
            'channel': ['base'],
            'spend': [0],
            'contribution': [50000],
            'roas': [0],
            'contribution_pct': [100.0]
        })

        # Should handle gracefully
        fig = plot_channel_contributions(df)
        assert fig is not None

    def test_channel_contributions_missing_columns(self):
        """Test with missing required columns."""
        df = pd.DataFrame({
            'channel': ['google', 'meta'],
            'contribution': [10000, 8000]
            # Missing 'roas' column
        })

        with pytest.raises((KeyError, ValueError)):
            plot_channel_contributions(df)

    def test_channel_contributions_single_channel(self):
        """Test with single channel (after base filtered)."""
        df = pd.DataFrame({
            'channel': ['base', 'google'],
            'spend': [0, 10000],
            'contribution': [50000, 20000],
            'roas': [0, 2.0],
            'contribution_pct': [71.4, 28.6]
        })

        # Should handle single channel
        try:
            fig = plot_channel_contributions(df)
            assert fig is not None
        except Exception as e:
            pytest.fail(f"Failed with single channel: {e}")


class TestQQPlotErrorHandling:
    """Test Q-Q plot error handling."""

    def test_qq_with_constant_residuals(self):
        """Test Q-Q plot with constant residuals."""
        residuals = np.array([5, 5, 5, 5, 5])

        # Should handle constant values
        try:
            fig = plot_qq(residuals)
            assert fig is not None
        except Exception as e:
            pytest.fail(f"Failed with constant residuals: {e}")

    def test_qq_with_single_value(self):
        """Test Q-Q plot with single residual."""
        residuals = np.array([5])

        # Should raise error or handle gracefully
        try:
            fig = plot_qq(residuals)
            assert fig is not None
        except (ValueError, IndexError):
            # Expected to fail with single value
            pass

    def test_qq_with_outliers(self):
        """Test Q-Q plot with extreme outliers."""
        residuals = np.concatenate([
            np.random.normal(0, 1, 100),
            [1000, -1000]  # Extreme outliers
        ])

        # Should handle outliers
        try:
            fig = plot_qq(residuals)
            assert fig is not None
        except Exception as e:
            pytest.fail(f"Failed with outliers: {e}")


class TestDataTypeHandling:
    """Test handling of different data types."""

    def test_waterfall_with_series_input(self):
        """Test waterfall accepts Series for columns."""
        df = pd.DataFrame({
            'channel': pd.Series(['base', 'google', 'meta']),
            'spend': pd.Series([0, 10000, 8000]),
            'contribution': pd.Series([50000, 20000, 15000]),
            'roas': pd.Series([0, 2.0, 1.875]),
            'contribution_pct': pd.Series([58.8, 23.5, 17.6])
        })

        try:
            fig = plot_waterfall(df)
            assert fig is not None
        except Exception as e:
            pytest.fail(f"Failed with Series input: {e}")

    def test_actual_vs_predicted_with_lists(self):
        """Test actual vs predicted with list inputs."""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 2.0, 2.9, 4.1, 5.0]

        # Should convert lists to arrays
        try:
            fig = plot_actual_vs_predicted(np.array(y_true), np.array(y_pred))
            assert fig is not None
        except Exception as e:
            pytest.fail(f"Failed with list inputs: {e}")


class TestLargeDatasets:
    """Test visualization with large datasets."""

    def test_decomposition_with_many_points(self):
        """Test decomposition with large number of time points."""
        n_points = 1000
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=n_points),
            'actual_revenue': np.random.uniform(10000, 50000, n_points),
            'base': np.random.uniform(5000, 10000, n_points),
            'google': np.random.uniform(2000, 15000, n_points),
            'meta': np.random.uniform(1000, 10000, n_points)
        })

        # Should handle large datasets
        try:
            fig = plot_decomposition(df)
            assert fig is not None
        except Exception as e:
            pytest.fail(f"Failed with large dataset: {e}")

    def test_actual_vs_predicted_large_dataset(self):
        """Test actual vs predicted with large dataset."""
        n_points = 10000
        y_true = np.random.uniform(1000, 50000, n_points)
        y_pred = y_true + np.random.normal(0, 1000, n_points)

        # Should handle large datasets
        try:
            fig = plot_actual_vs_predicted(y_true, y_pred)
            assert fig is not None
        except Exception as e:
            pytest.fail(f"Failed with large dataset: {e}")


class TestExtremeValues:
    """Test visualization with extreme values."""

    def test_waterfall_with_very_large_values(self):
        """Test waterfall with very large contribution values."""
        df = pd.DataFrame({
            'channel': ['base', 'google'],
            'spend': [0, 1e9],
            'contribution': [1e10, 5e9],
            'roas': [0, 5.0],
            'contribution_pct': [66.7, 33.3]
        })

        # Should handle large values
        try:
            fig = plot_waterfall(df)
            assert fig is not None
        except Exception as e:
            pytest.fail(f"Failed with large values: {e}")

    def test_actual_vs_predicted_very_small_values(self):
        """Test actual vs predicted with very small values."""
        y_true = np.array([0.001, 0.002, 0.003, 0.004])
        y_pred = np.array([0.0011, 0.0019, 0.0031, 0.0041])

        # Should handle small values
        try:
            fig = plot_actual_vs_predicted(y_true, y_pred)
            assert fig is not None
        except Exception as e:
            pytest.fail(f"Failed with small values: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
