"""Advanced optimizer test cases for edge cases and constraints."""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimizer import BudgetOptimizer, OptimizationError
from ridge_mmm import RidgeMMM


@pytest.fixture
def trained_optimizer():
    """Create optimizer with trained model."""
    # Create synthetic data with known relationships
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        'google': np.random.uniform(1000, 5000, n),
        'meta': np.random.uniform(800, 4000, n),
        'apple': np.random.uniform(500, 3000, n)
    })

    # Generate revenue with known coefficients (google > meta > apple)
    revenue = (
        2.5 * df['google'] +
        2.0 * df['meta'] +
        1.5 * df['apple'] +
        np.random.normal(0, 500, n)
    )

    # Train model
    mmm = RidgeMMM(alpha=1.0)
    channel_configs = {
        'google': {'adstock': 0.3, 'hill_K': 1.0, 'hill_S': 1.0},
        'meta': {'adstock': 0.3, 'hill_K': 1.0, 'hill_S': 1.0},
        'apple': {'adstock': 0.3, 'hill_K': 1.0, 'hill_S': 1.0}
    }
    mmm.fit(df, revenue, channel_configs)

    return BudgetOptimizer(mmm, df)


@pytest.fixture
def simple_trained_model():
    """Create simple trained model for testing."""
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        'google': np.random.uniform(1000, 5000, n),
        'meta': np.random.uniform(800, 4000, n)
    })

    revenue = 2.0 * df['google'] + 1.5 * df['meta'] + np.random.normal(0, 500, n)

    mmm = RidgeMMM(alpha=1.0)
    channel_configs = {
        'google': {'adstock': 0.3, 'hill_K': 1.0, 'hill_S': 1.0},
        'meta': {'adstock': 0.3, 'hill_K': 1.0, 'hill_S': 1.0}
    }
    mmm.fit(df, revenue, channel_configs)

    return mmm, df


class TestOptimizerTightConstraints:
    """Test optimizer behavior with very tight constraints."""

    def test_optimizer_with_tight_constraints(self, trained_optimizer):
        """Test optimization with very tight constraints."""
        result = trained_optimizer.optimize_budget(
            total_budget=10000,
            channels=['google', 'meta', 'apple'],
            constraints={
                'google': {'min': 3000, 'max': 3500},
                'meta': {'min': 3000, 'max': 3500},
                'apple': {'min': 3000, 'max': 3500}
            }
        )

        assert result['success']

        # Check constraints are respected
        for channel in ['google', 'meta', 'apple']:
            alloc = result['optimal_allocation'][channel]
            assert 3000 <= alloc <= 3500, f"{channel} allocation {alloc} outside bounds [3000, 3500]"

        # Check total budget is allocated
        total_allocated = sum(result['optimal_allocation'].values())
        assert abs(total_allocated - 10000) < 10, f"Total allocated {total_allocated} != 10000"

    def test_optimizer_with_impossible_constraints(self, trained_optimizer):
        """Test that impossible constraints raise error."""
        with pytest.raises(ValueError, match="exceeds total budget"):
            trained_optimizer.optimize_budget(
                total_budget=5000,
                channels=['google', 'meta', 'apple'],
                constraints={
                    'google': {'min': 3000, 'max': 5000},
                    'meta': {'min': 3000, 'max': 5000},
                    'apple': {'min': 3000, 'max': 5000}
                }
            )

    def test_optimizer_with_asymmetric_constraints(self, trained_optimizer):
        """Test when one channel has much wider range."""
        result = trained_optimizer.optimize_budget(
            total_budget=10000,
            channels=['google', 'meta', 'apple'],
            constraints={
                'google': {'min': 1000, 'max': 8000},  # Wide range
                'meta': {'min': 500, 'max': 1000},     # Narrow
                'apple': {'min': 500, 'max': 1000}     # Narrow
            }
        )

        assert result['success']

        # Google should get most of budget (since it has highest coefficient)
        assert result['optimal_allocation']['google'] >= 7000
        assert result['optimal_allocation']['meta'] <= 1000
        assert result['optimal_allocation']['apple'] <= 1000


class TestOptimizerBoundaryConditions:
    """Test optimizer at constraint boundaries."""

    def test_optimizer_boundary_conditions(self, trained_optimizer):
        """Test optimization at constraint boundaries."""
        result = trained_optimizer.optimize_budget(
            total_budget=5000,
            channels=['google', 'meta'],
            constraints={
                'google': {'min': 0, 'max': 3000},
                'meta': {'min': 0, 'max': 2000}
            }
        )

        assert result['success']

        alloc = result['optimal_allocation']

        # Check that total is correct
        total = alloc['google'] + alloc['meta']
        assert abs(total - 5000) < 10

        # At least one channel should be at or near boundary
        at_boundary = (
            abs(alloc['google'] - 3000) < 100 or  # At max
            abs(alloc['google'] - 0) < 100 or     # At min
            abs(alloc['meta'] - 2000) < 100 or    # At max
            abs(alloc['meta'] - 0) < 100          # At min
        )
        assert at_boundary, f"Expected allocation at boundary, got google={alloc['google']}, meta={alloc['meta']}"

    def test_optimizer_with_zero_minimum(self, trained_optimizer):
        """Test allowing channels to go to zero."""
        result = trained_optimizer.optimize_budget(
            total_budget=5000,
            channels=['google', 'meta', 'apple'],
            constraints={
                'google': {'min': 0, 'max': 5000},
                'meta': {'min': 0, 'max': 5000},
                'apple': {'min': 0, 'max': 5000}
            }
        )

        assert result['success']

        # Google should get significant budget (highest coefficient)
        assert result['optimal_allocation']['google'] >= 2000

        # Check total
        total = sum(result['optimal_allocation'].values())
        assert abs(total - 5000) < 10


class TestOptimizerEdgeCases:
    """Test edge cases and error handling."""

    def test_optimizer_with_single_channel_fails(self, trained_optimizer):
        """Test that single channel optimization raises error."""
        with pytest.raises(ValueError, match="at least 2 channels"):
            trained_optimizer.optimize_budget(
                total_budget=10000,
                channels=['google']
            )

    def test_optimizer_with_negative_budget_fails(self, trained_optimizer):
        """Test negative budget raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            trained_optimizer.optimize_budget(
                total_budget=-1000,
                channels=['google', 'meta']
            )

    def test_optimizer_with_zero_budget_fails(self, trained_optimizer):
        """Test zero budget raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            trained_optimizer.optimize_budget(
                total_budget=0,
                channels=['google', 'meta']
            )

    def test_optimizer_with_invalid_channel(self, trained_optimizer):
        """Test invalid channel name raises error."""
        with pytest.raises(ValueError, match="not in model"):
            trained_optimizer.optimize_budget(
                total_budget=10000,
                channels=['google', 'invalid_channel']
            )

    def test_optimizer_empty_channels_list(self, trained_optimizer):
        """Test empty channels list raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            trained_optimizer.optimize_budget(
                total_budget=10000,
                channels=[]
            )

    def test_optimizer_with_conflicting_constraints(self, trained_optimizer):
        """Test that max < min raises error."""
        with pytest.raises(ValueError, match="Max .* < min"):
            trained_optimizer.optimize_budget(
                total_budget=10000,
                channels=['google', 'meta'],
                constraints={
                    'google': {'min': 5000, 'max': 3000}  # max < min
                }
            )

    def test_optimizer_with_negative_min_constraint(self, trained_optimizer):
        """Test negative min constraint raises error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            trained_optimizer.optimize_budget(
                total_budget=10000,
                channels=['google', 'meta'],
                constraints={
                    'google': {'min': -100, 'max': 5000}
                }
            )


class TestOptimizerScenarioComparison:
    """Test scenario comparison functionality."""

    def test_compare_scenarios_basic(self, trained_optimizer):
        """Test basic scenario comparison."""
        scenarios = {
            'Scenario A': {'google': 3000, 'meta': 2000, 'apple': 1000},
            'Scenario B': {'google': 4000, 'meta': 1500, 'apple': 500},
            'Scenario C': {'google': 2000, 'meta': 2500, 'apple': 1500}
        }

        result = trained_optimizer.compare_scenarios(scenarios)

        # Check DataFrame structure
        assert len(result) == 3
        assert 'scenario' in result.columns
        assert 'total_spend' in result.columns
        assert 'predicted_revenue' in result.columns
        assert 'roas' in result.columns

        # Check all scenarios present
        assert set(result['scenario']) == {'Scenario A', 'Scenario B', 'Scenario C'}

        # Check sorted by revenue (descending)
        revenues = result['predicted_revenue'].tolist()
        assert revenues == sorted(revenues, reverse=True)

    def test_compare_scenarios_single(self, trained_optimizer):
        """Test scenario comparison with single scenario."""
        scenarios = {
            'Only Scenario': {'google': 5000, 'meta': 3000, 'apple': 2000}
        }

        result = trained_optimizer.compare_scenarios(scenarios)

        assert len(result) == 1
        assert result.iloc[0]['scenario'] == 'Only Scenario'
        assert result.iloc[0]['total_spend'] == 10000


class TestOptimizerSensitivityAnalysis:
    """Test sensitivity analysis functionality."""

    def test_sensitivity_analysis_basic(self, trained_optimizer):
        """Test basic sensitivity analysis."""
        current_allocation = {'google': 3000, 'meta': 2000, 'apple': 1000}

        result = trained_optimizer.sensitivity_analysis(
            channel='google',
            current_allocation=current_allocation,
            variation_range=(-30, 30),
            n_points=10
        )

        # Check DataFrame structure
        assert len(result) == 10
        assert 'channel_spend_pct_change' in result.columns
        assert 'channel_spend' in result.columns
        assert 'predicted_revenue' in result.columns
        assert 'revenue_impact' in result.columns

        # Check percentage changes range from -30 to +30
        pct_changes = result['channel_spend_pct_change'].tolist()
        assert min(pct_changes) == pytest.approx(-30, abs=1)
        assert max(pct_changes) == pytest.approx(30, abs=1)

    def test_sensitivity_analysis_invalid_channel(self, trained_optimizer):
        """Test sensitivity analysis with invalid channel."""
        current_allocation = {'google': 3000, 'meta': 2000}

        with pytest.raises(ValueError, match="not in model"):
            trained_optimizer.sensitivity_analysis(
                channel='invalid',
                current_allocation=current_allocation
            )

    def test_sensitivity_analysis_channel_not_in_allocation(self, trained_optimizer):
        """Test sensitivity analysis when channel not in allocation."""
        current_allocation = {'google': 3000, 'meta': 2000}

        with pytest.raises(ValueError, match="not in current_allocation"):
            trained_optimizer.sensitivity_analysis(
                channel='apple',
                current_allocation=current_allocation
            )


class TestOptimizerDiminishingReturns:
    """Test diminishing returns point calculation."""

    def test_get_diminishing_returns_point(self, trained_optimizer):
        """Test finding diminishing returns point."""
        saturation_point = trained_optimizer.get_diminishing_returns_point(
            channel='google',
            X_base=trained_optimizer.X_base,
            marginal_roas_threshold=1.5
        )

        # Should return a positive number
        assert saturation_point > 0
        assert isinstance(saturation_point, (int, float, np.number))

    def test_diminishing_returns_invalid_channel(self, trained_optimizer):
        """Test diminishing returns with invalid channel."""
        with pytest.raises(ValueError, match="not in model"):
            trained_optimizer.get_diminishing_returns_point(
                channel='invalid',
                X_base=trained_optimizer.X_base
            )


class TestOptimizerOptimalPerChannel:
    """Test optimal allocation per channel."""

    def test_optimal_allocation_per_channel(self, trained_optimizer):
        """Test getting optimal allocation per channel."""
        result = trained_optimizer.get_optimal_allocation_per_channel(
            channels=['google', 'meta', 'apple'],
            marginal_roas_target=2.0
        )

        # Check structure
        assert len(result) == 3
        assert 'google' in result
        assert 'meta' in result
        assert 'apple' in result

        # Check each channel has required fields
        for channel in ['google', 'meta', 'apple']:
            assert 'current_spend' in result[channel]
            assert 'optimal_spend' in result[channel]
            assert 'spend_change' in result[channel]
            assert 'predicted_revenue' in result[channel]
            assert 'marginal_roas' in result[channel]


class TestOptimizerInitialization:
    """Test optimizer initialization and validation."""

    def test_optimizer_requires_fitted_model(self, simple_trained_model):
        """Test optimizer requires fitted model."""
        mmm, df = simple_trained_model

        # Create unfitted model
        unfitted_mmm = RidgeMMM(alpha=1.0)

        with pytest.raises(ValueError, match="must be fitted"):
            BudgetOptimizer(unfitted_mmm, df)

    def test_optimizer_initialization_success(self, simple_trained_model):
        """Test successful optimizer initialization."""
        mmm, df = simple_trained_model

        optimizer = BudgetOptimizer(mmm, df)

        assert optimizer.model is mmm
        assert optimizer.X_base is df
        assert optimizer.channels == mmm.media_channels


class TestOptimizerConstraintFormats:
    """Test different constraint format handling."""

    def test_constraints_dict_format(self, trained_optimizer):
        """Test constraints with dict format."""
        result = trained_optimizer.optimize_budget(
            total_budget=10000,
            channels=['google', 'meta'],
            constraints={
                'google': {'min': 2000, 'max': 8000},
                'meta': {'min': 1000, 'max': 7000}
            }
        )

        assert result['success']

    def test_constraints_tuple_format(self, trained_optimizer):
        """Test constraints with tuple format."""
        result = trained_optimizer.optimize_budget(
            total_budget=10000,
            channels=['google', 'meta'],
            constraints={
                'google': (2000, 8000),
                'meta': (1000, 7000)
            }
        )

        assert result['success']

    def test_constraints_invalid_format(self, trained_optimizer):
        """Test invalid constraint format raises error."""
        with pytest.raises(ValueError, match="Invalid constraint format"):
            trained_optimizer.optimize_budget(
                total_budget=10000,
                channels=['google', 'meta'],
                constraints={
                    'google': "invalid"  # Invalid format
                }
            )


class TestOptimizerTargetMetrics:
    """Test optimization with different target metrics."""

    def test_optimize_for_revenue(self, trained_optimizer):
        """Test optimization targeting revenue."""
        result = trained_optimizer.optimize_budget(
            total_budget=10000,
            channels=['google', 'meta', 'apple'],
            target_metric='revenue'
        )

        assert result['success']
        assert 'predicted_revenue' in result
        assert result['predicted_revenue'] > 0

    def test_optimize_for_roas(self, trained_optimizer):
        """Test optimization targeting ROAS."""
        result = trained_optimizer.optimize_budget(
            total_budget=10000,
            channels=['google', 'meta', 'apple'],
            target_metric='roas'
        )

        assert result['success']
        assert 'predicted_roas' in result
        assert result['predicted_roas'] > 0

    def test_optimize_invalid_target_metric(self, trained_optimizer):
        """Test invalid target metric raises error."""
        with pytest.raises(ValueError, match="must be 'revenue' or 'roas'"):
            trained_optimizer.optimize_budget(
                total_budget=10000,
                channels=['google', 'meta'],
                target_metric='invalid'
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
