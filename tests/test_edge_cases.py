"""
Edge case tests for Ridge MMM components.

Tests numerical stability, validation, and error handling across
the transformations, hierarchical models, and optimizer modules.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformations import HillTransformer
from optimizer import BudgetOptimizer, OptimizationError


class TestHillNumericalStability:
    """Test Hill transformer with extreme values and edge cases."""

    def test_hill_large_values(self):
        """Test Hill transformer with very large values."""
        transformer = HillTransformer(K=1.0, S=1.0)

        # Large values that could cause overflow
        large = np.array([1e6, 1e7, 1e8, 1e9])
        result = transformer.transform(large)

        # All results should be finite
        assert np.all(np.isfinite(result)), "Large values produced non-finite results"

        # Results should not exceed K
        assert np.all(result <= 1.0), f"Results exceed K: {result}"

        # For very large x, result should approach K
        assert result[-1] > 0.99, "Very large value didn't approach K"

    def test_hill_small_values(self):
        """Test Hill transformer with very small values."""
        transformer = HillTransformer(K=1.0, S=1.0)

        # Small values
        small = np.array([1e-10, 1e-8, 1e-6, 1e-4])
        result = transformer.transform(small)

        # All results should be finite
        assert np.all(np.isfinite(result)), "Small values produced non-finite results"

        # Results should be non-negative
        assert np.all(result >= 0), f"Negative results: {result}"

    def test_hill_mixed_with_zeros(self):
        """Test Hill transformer with mixed values including zeros."""
        transformer = HillTransformer(K=1.0, S=1.0)

        # Mixed array with zeros
        mixed = np.array([0, 100, 0, 1000, 0])
        result = transformer.transform(mixed)

        # All results should be finite
        assert np.all(np.isfinite(result)), "Mixed values with zeros produced non-finite results"

        # Zeros should remain zeros
        assert result[0] == 0, "Zero didn't stay zero"
        assert result[2] == 0, "Zero didn't stay zero"
        assert result[4] == 0, "Zero didn't stay zero"

        # Non-zero values should be positive
        assert result[1] > 0, "Non-zero value produced zero result"
        assert result[3] > 0, "Non-zero value produced zero result"

    def test_hill_extreme_parameters(self):
        """Test Hill transformer with extreme K and S values."""
        # Very large S (steep curve)
        transformer = HillTransformer(K=1.0, S=150.0)
        x = np.array([0.1, 0.5, 1.0, 2.0, 10.0])
        result = transformer.transform(x)

        assert np.all(np.isfinite(result)), "Large S produced non-finite results"
        assert np.all(result >= 0), "Negative results with large S"
        assert np.all(result <= 1.0), "Results exceed K with large S"

        # Very large K
        transformer = HillTransformer(K=100.0, S=1.0)
        result = transformer.transform(x)

        assert np.all(np.isfinite(result)), "Large K produced non-finite results"
        assert np.all(result >= 0), "Negative results with large K"
        assert np.all(result <= 100.0), "Results exceed K with large K"

    def test_hill_all_zeros(self):
        """Test Hill transformer with all-zero input."""
        transformer = HillTransformer(K=1.0, S=1.0)

        zeros = np.zeros(10)
        result = transformer.transform(zeros)

        # Result should be all zeros
        assert np.all(result == 0), "All-zero input didn't produce all-zero output"

    def test_hill_nan_handling(self):
        """Test Hill transformer handles NaN values."""
        transformer = HillTransformer(K=1.0, S=1.0)

        # Array with NaN values
        with_nan = np.array([1.0, np.nan, 2.0, np.nan, 3.0])

        # Should warn about NaN values
        with pytest.warns(UserWarning, match="NaN values"):
            result = transformer.transform(with_nan)

        # Result should be finite (NaN converted to 0)
        assert np.all(np.isfinite(result)), "NaN values not handled properly"


class TestHillInvalidParameters:
    """Test Hill transformer rejects invalid parameters."""

    def test_hill_negative_K(self):
        """Test Hill transformer rejects negative K."""
        with pytest.raises(ValueError, match="K must be positive"):
            HillTransformer(K=-1.0, S=1.0)

    def test_hill_zero_K(self):
        """Test Hill transformer rejects zero K."""
        with pytest.raises(ValueError, match="K must be positive"):
            HillTransformer(K=0.0, S=1.0)

    def test_hill_negative_S(self):
        """Test Hill transformer rejects negative S."""
        with pytest.raises(ValueError, match="S must be positive"):
            HillTransformer(K=1.0, S=-1.0)

    def test_hill_zero_S(self):
        """Test Hill transformer rejects zero S."""
        with pytest.raises(ValueError, match="S must be positive"):
            HillTransformer(K=1.0, S=0.0)


class TestOptimizerInvalidInputs:
    """Test optimizer validation with invalid inputs."""

    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock optimizer for testing."""
        # Create a mock model with required attributes
        class MockModel:
            is_fitted = True
            media_channels = ['google', 'meta', 'apple']

            def predict(self, X):
                # Simple mock prediction
                return np.array([1000.0] * len(X))

        # Create mock base data
        X_base = pd.DataFrame({
            'google': [100] * 10,
            'meta': [80] * 10,
            'apple': [60] * 10
        })

        return BudgetOptimizer(MockModel(), X_base)

    def test_optimizer_negative_budget(self, mock_optimizer):
        """Test optimizer rejects negative budget."""
        with pytest.raises(ValueError, match="positive"):
            mock_optimizer.optimize_budget(
                total_budget=-1000,
                channels=['google', 'meta']
            )

    def test_optimizer_zero_budget(self, mock_optimizer):
        """Test optimizer rejects zero budget."""
        with pytest.raises(ValueError, match="positive"):
            mock_optimizer.optimize_budget(
                total_budget=0,
                channels=['google', 'meta']
            )

    def test_optimizer_empty_channels(self, mock_optimizer):
        """Test optimizer rejects empty channels list."""
        with pytest.raises(ValueError, match="empty"):
            mock_optimizer.optimize_budget(
                total_budget=1000,
                channels=[]
            )

    def test_optimizer_single_channel(self, mock_optimizer):
        """Test optimizer rejects single channel."""
        with pytest.raises(ValueError, match="at least 2"):
            mock_optimizer.optimize_budget(
                total_budget=1000,
                channels=['google']
            )

    def test_optimizer_invalid_channel(self, mock_optimizer):
        """Test optimizer rejects channels not in model."""
        with pytest.raises(ValueError, match="not in model"):
            mock_optimizer.optimize_budget(
                total_budget=1000,
                channels=['google', 'tiktok']
            )

    def test_optimizer_invalid_metric(self, mock_optimizer):
        """Test optimizer rejects invalid target metric."""
        with pytest.raises(ValueError, match="target_metric"):
            mock_optimizer.optimize_budget(
                total_budget=1000,
                channels=['google', 'meta'],
                target_metric='invalid'
            )

    def test_optimizer_infeasible_constraints(self, mock_optimizer):
        """Test optimizer rejects infeasible constraints."""
        # Constraints that sum to more than budget
        with pytest.raises(ValueError, match="Sum of minimum constraints.*exceeds total budget"):
            mock_optimizer.optimize_budget(
                total_budget=1000,
                channels=['google', 'meta'],
                constraints={
                    'google': {'min': 600, 'max': 1000},
                    'meta': {'min': 600, 'max': 1000}
                }
            )

    def test_optimizer_invalid_constraint_format(self, mock_optimizer):
        """Test optimizer rejects invalid constraint format."""
        with pytest.raises(ValueError, match="Invalid constraint format"):
            mock_optimizer.optimize_budget(
                total_budget=1000,
                channels=['google', 'meta'],
                constraints={
                    'google': "invalid"
                }
            )

    def test_optimizer_min_exceeds_max(self, mock_optimizer):
        """Test optimizer rejects min > max constraints."""
        with pytest.raises(ValueError, match="Max.*<.*min"):
            mock_optimizer.optimize_budget(
                total_budget=1000,
                channels=['google', 'meta'],
                constraints={
                    'google': {'min': 600, 'max': 400}
                }
            )

    def test_optimizer_negative_min(self, mock_optimizer):
        """Test optimizer rejects negative minimum constraints."""
        with pytest.raises(ValueError, match="cannot be negative"):
            mock_optimizer.optimize_budget(
                total_budget=1000,
                channels=['google', 'meta'],
                constraints={
                    'google': {'min': -100, 'max': 500}
                }
            )

    def test_optimizer_min_exceeds_budget(self, mock_optimizer):
        """Test optimizer rejects min constraint exceeding budget."""
        with pytest.raises(ValueError, match="exceeds total budget"):
            mock_optimizer.optimize_budget(
                total_budget=1000,
                channels=['google', 'meta'],
                constraints={
                    'google': {'min': 1500, 'max': 2000}
                }
            )


class TestOptimizerValidInputs:
    """Test optimizer with valid inputs."""

    @pytest.fixture
    def mock_optimizer(self):
        """Create a mock optimizer for testing."""
        class MockModel:
            is_fitted = True
            media_channels = ['google', 'meta']

            def predict(self, X):
                # Simple mock prediction: revenue increases with spend
                return np.array([X['google'].mean() * 2 + X['meta'].mean() * 1.5] * len(X))

        X_base = pd.DataFrame({
            'google': [100] * 10,
            'meta': [80] * 10
        })

        return BudgetOptimizer(MockModel(), X_base)

    def test_optimizer_valid_run(self, mock_optimizer):
        """Test optimizer runs successfully with valid inputs."""
        result = mock_optimizer.optimize_budget(
            total_budget=1000,
            channels=['google', 'meta']
        )

        assert result is not None
        assert 'optimal_allocation' in result
        assert 'predicted_revenue' in result
        assert 'success' in result

    def test_optimizer_respects_constraints(self, mock_optimizer):
        """Test optimizer respects min/max constraints."""
        result = mock_optimizer.optimize_budget(
            total_budget=1000,
            channels=['google', 'meta'],
            constraints={
                'google': {'min': 400, 'max': 600},
                'meta': {'min': 200, 'max': 800}
            }
        )

        if result['success']:
            # Check constraints are respected
            assert 400 <= result['optimal_allocation']['google'] <= 600
            assert 200 <= result['optimal_allocation']['meta'] <= 800


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
