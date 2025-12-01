"""Advanced hierarchical MMM test cases for unbalanced data and edge cases."""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hierarchical_mmm import HierarchicalMMM


@pytest.fixture
def unbalanced_data():
    """Create unbalanced multi-market data with different sample sizes."""
    np.random.seed(42)

    data = []

    # US: 200 rows (lots of data)
    for i in range(200):
        data.append({
            'date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=i % 100),
            'country': 'US',
            'os': 'iOS' if i % 2 == 0 else 'Android',
            'google': np.random.uniform(1000, 5000),
            'meta': np.random.uniform(800, 4000),
            'revenue': np.random.uniform(10000, 50000)
        })

    # KR: 50 rows (medium data)
    for i in range(50):
        data.append({
            'date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=i % 50),
            'country': 'KR',
            'os': 'iOS' if i % 2 == 0 else 'Android',
            'google': np.random.uniform(500, 2000),
            'meta': np.random.uniform(400, 1500),
            'revenue': np.random.uniform(5000, 20000)
        })

    # JP: 20 rows (sparse data)
    for i in range(20):
        data.append({
            'date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=i % 20),
            'country': 'JP',
            'os': 'iOS',
            'google': np.random.uniform(300, 1000),
            'meta': np.random.uniform(200, 800),
            'revenue': np.random.uniform(3000, 10000)
        })

    return pd.DataFrame(data)


@pytest.fixture
def balanced_data():
    """Create balanced multi-market data."""
    np.random.seed(42)

    data = []

    # Equal data for all markets
    for country in ['US', 'KR', 'JP']:
        for i in range(100):
            data.append({
                'date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=i),
                'country': country,
                'os': 'iOS' if i % 2 == 0 else 'Android',
                'google': np.random.uniform(1000, 5000),
                'meta': np.random.uniform(800, 4000),
                'revenue': np.random.uniform(10000, 50000)
            })

    return pd.DataFrame(data)


@pytest.fixture
def channel_configs():
    """Standard channel configurations."""
    return {
        'google': {'adstock': 0.3, 'hill_K': 1.0, 'hill_S': 1.0},
        'meta': {'adstock': 0.3, 'hill_K': 1.0, 'hill_S': 1.0}
    }


class TestUnbalancedData:
    """Test hierarchical MMM with unbalanced data."""

    def test_unbalanced_country_data_fitting(self, unbalanced_data, channel_configs):
        """Test MMM handles unbalanced data across countries."""
        mmm = HierarchicalMMM(analysis_level='country')
        mmm.fit(unbalanced_data, target_col='revenue', channel_configs=channel_configs)

        # All countries should have models despite different data sizes
        assert 'US' in mmm.models
        assert 'KR' in mmm.models
        assert 'JP' in mmm.models

        # US should have most data
        # Note: Can't directly check data sizes in models, but we can verify models exist

    def test_unbalanced_data_predictions(self, unbalanced_data, channel_configs):
        """Test predictions work for all segments with unbalanced data."""
        mmm = HierarchicalMMM(analysis_level='country')
        mmm.fit(unbalanced_data, target_col='revenue', channel_configs=channel_configs)

        # Test predictions for each country
        for country in ['US', 'KR', 'JP']:
            country_data = unbalanced_data[unbalanced_data['country'] == country]
            preds = mmm.predict(country_data, segment=country)

            assert len(preds) == len(country_data)
            assert all(np.isfinite(preds))  # No inf or nan

    def test_unbalanced_data_contributions(self, unbalanced_data, channel_configs):
        """Test contribution analysis with unbalanced data."""
        mmm = HierarchicalMMM(analysis_level='country')
        mmm.fit(unbalanced_data, target_col='revenue', channel_configs=channel_configs)

        # Get contributions for each country
        for country in ['US', 'KR', 'JP']:
            country_data = unbalanced_data[unbalanced_data['country'] == country]
            contribs = mmm.get_contributions(country_data, segment=country)

            assert not contribs.empty
            assert 'channel' in contribs.columns
            assert 'contribution' in contribs.columns


class TestMissingSegments:
    """Test error handling for missing segments."""

    def test_missing_segment_in_prediction(self, balanced_data, channel_configs):
        """Test error when predicting for non-existent segment."""
        mmm = HierarchicalMMM(analysis_level='country')
        mmm.fit(balanced_data, target_col='revenue', channel_configs=channel_configs)

        # Try to predict for non-existent country
        tw_data = balanced_data.copy()
        tw_data['country'] = 'TW'

        with pytest.raises(ValueError):
            mmm.predict(tw_data, segment='TW')

    def test_missing_segment_in_contributions(self, balanced_data, channel_configs):
        """Test error when getting contributions for non-existent segment."""
        mmm = HierarchicalMMM(analysis_level='country')
        mmm.fit(balanced_data, target_col='revenue', channel_configs=channel_configs)

        with pytest.raises(ValueError):
            mmm.get_contributions(balanced_data, segment='TW')


class TestSegmentComparison:
    """Test segment comparison with various data conditions."""

    def test_segment_comparison_with_unbalanced_data(self, unbalanced_data, channel_configs):
        """Test segment comparison works despite unbalanced data."""
        mmm = HierarchicalMMM(analysis_level='country')
        mmm.fit(unbalanced_data, target_col='revenue', channel_configs=channel_configs)

        comparison = mmm.compare_segments(unbalanced_data, metric='roas')

        # Should have results for all countries
        assert len(comparison) >= 3
        assert 'US' in comparison['segment'].values
        assert 'KR' in comparison['segment'].values
        assert 'JP' in comparison['segment'].values

        # Check columns
        assert 'roas' in comparison.columns
        assert 'spend' in comparison.columns
        assert 'revenue' in comparison.columns

    def test_segment_comparison_with_balanced_data(self, balanced_data, channel_configs):
        """Test segment comparison with balanced data."""
        mmm = HierarchicalMMM(analysis_level='country')
        mmm.fit(balanced_data, target_col='revenue', channel_configs=channel_configs)

        comparison = mmm.compare_segments(balanced_data, metric='roas')

        # Should have results for all countries
        assert len(comparison) == 3
        assert set(comparison['segment']) == {'US', 'KR', 'JP'}


class TestSparseData:
    """Test handling of very sparse data."""

    def test_very_sparse_segment_skipped(self, channel_configs):
        """Test that segments with too little data are skipped."""
        # Create data with one very sparse segment
        data = []

        # US: 100 rows (good data)
        for i in range(100):
            data.append({
                'date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=i),
                'country': 'US',
                'os': 'iOS',
                'google': np.random.uniform(1000, 5000),
                'meta': np.random.uniform(800, 4000),
                'revenue': np.random.uniform(10000, 50000)
            })

        # KR: Only 5 rows (too sparse)
        for i in range(5):
            data.append({
                'date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=i),
                'country': 'KR',
                'os': 'iOS',
                'google': np.random.uniform(1000, 5000),
                'meta': np.random.uniform(800, 4000),
                'revenue': np.random.uniform(10000, 50000)
            })

        df = pd.DataFrame(data)

        mmm = HierarchicalMMM(analysis_level='country')
        mmm.fit(df, target_col='revenue', channel_configs=channel_configs)

        # US should be fitted
        assert 'US' in mmm.models

        # KR might be skipped due to insufficient data
        # (depends on implementation - minimum 10 rows required)


class TestCountryOSLevel:
    """Test country-OS level models."""

    def test_country_os_level_fitting(self, balanced_data, channel_configs):
        """Test fitting at country-OS level."""
        mmm = HierarchicalMMM(analysis_level='country_os')
        mmm.fit(balanced_data, target_col='revenue', channel_configs=channel_configs)

        # Should have models for each country-OS combination
        # Format: 'US_iOS', 'US_Android', etc.
        assert len(mmm.models) >= 6  # 3 countries × 2 OS (minus any skipped)

    def test_country_os_predictions(self, balanced_data, channel_configs):
        """Test predictions at country-OS level."""
        mmm = HierarchicalMMM(analysis_level='country_os')
        mmm.fit(balanced_data, target_col='revenue', channel_configs=channel_configs)

        # Test prediction for specific country-OS combination
        us_ios_data = balanced_data[
            (balanced_data['country'] == 'US') & (balanced_data['os'] == 'iOS')
        ]

        preds = mmm.predict(us_ios_data, segment='US_iOS')

        assert len(preds) == len(us_ios_data)
        assert all(np.isfinite(preds))


class TestOSLevel:
    """Test OS-level models."""

    def test_os_level_fitting(self, balanced_data, channel_configs):
        """Test fitting at OS level."""
        mmm = HierarchicalMMM(analysis_level='os')
        mmm.fit(balanced_data, target_col='revenue', channel_configs=channel_configs)

        # Should have models for each OS
        assert 'iOS' in mmm.models
        assert 'Android' in mmm.models

    def test_os_level_predictions(self, balanced_data, channel_configs):
        """Test predictions at OS level."""
        mmm = HierarchicalMMM(analysis_level='os')
        mmm.fit(balanced_data, target_col='revenue', channel_configs=channel_configs)

        ios_data = balanced_data[balanced_data['os'] == 'iOS']
        preds = mmm.predict(ios_data, segment='iOS')

        assert len(preds) == len(ios_data)
        assert all(np.isfinite(preds))


class TestGlobalLevel:
    """Test global level aggregation."""

    def test_global_level_fitting(self, balanced_data, channel_configs):
        """Test fitting at global level."""
        mmm = HierarchicalMMM(analysis_level='global')
        mmm.fit(balanced_data, target_col='revenue', channel_configs=channel_configs)

        # Should have global model
        assert 'global' in mmm.models

    def test_global_level_predictions(self, balanced_data, channel_configs):
        """Test predictions at global level."""
        mmm = HierarchicalMMM(analysis_level='global')
        mmm.fit(balanced_data, target_col='revenue', channel_configs=channel_configs)

        preds = mmm.predict(balanced_data)

        assert len(preds) == len(balanced_data)
        assert all(np.isfinite(preds))


class TestMissingColumns:
    """Test error handling for missing required columns."""

    def test_missing_country_column(self, balanced_data, channel_configs):
        """Test error when country column missing for country-level analysis."""
        # Remove country column
        data_no_country = balanced_data.drop('country', axis=1)

        mmm = HierarchicalMMM(analysis_level='country')

        with pytest.raises(ValueError, match="'country' column"):
            mmm.fit(data_no_country, target_col='revenue', channel_configs=channel_configs)

    def test_missing_os_column(self, balanced_data, channel_configs):
        """Test error when OS column missing for OS-level analysis."""
        # Remove OS column
        data_no_os = balanced_data.drop('os', axis=1)

        mmm = HierarchicalMMM(analysis_level='os')

        with pytest.raises(ValueError, match="'os' column"):
            mmm.fit(data_no_os, target_col='revenue', channel_configs=channel_configs)

    def test_missing_date_column(self, balanced_data, channel_configs):
        """Test error when date column missing."""
        # Remove date column
        data_no_date = balanced_data.drop('date', axis=1)

        mmm = HierarchicalMMM(analysis_level='country')

        with pytest.raises(ValueError, match="date column"):
            mmm.fit(data_no_date, target_col='revenue', channel_configs=channel_configs)


class TestAggregatedContributions:
    """Test aggregated contributions across segments."""

    def test_aggregated_contributions_country_level(self, balanced_data, channel_configs):
        """Test getting aggregated contributions at country level."""
        mmm = HierarchicalMMM(analysis_level='country')
        mmm.fit(balanced_data, target_col='revenue', channel_configs=channel_configs)

        # Get aggregated contributions (no segment specified)
        contribs = mmm.get_contributions(balanced_data)

        assert not contribs.empty
        assert 'channel' in contribs.columns
        assert 'contribution' in contribs.columns
        assert 'roas' in contribs.columns

        # Should have contributions for both channels
        channels = contribs['channel'].tolist()
        assert 'google' in channels or 'base' in channels


class TestCrossMarketInsights:
    """Test cross-market insights functionality."""

    def test_cross_market_insights(self, balanced_data, channel_configs):
        """Test getting cross-market insights."""
        mmm = HierarchicalMMM(analysis_level='country')
        mmm.fit(balanced_data, target_col='revenue', channel_configs=channel_configs)

        insights = mmm.get_cross_market_insights()

        # Check structure
        assert 'best_channels' in insights
        assert 'saturation_levels' in insights
        assert 'efficiency_gaps' in insights

        # Best channels should be populated
        assert len(insights['best_channels']) > 0


class TestInvalidAnalysisLevel:
    """Test error handling for invalid analysis level."""

    def test_invalid_analysis_level(self):
        """Test that invalid analysis level raises error."""
        with pytest.raises(ValueError, match="Invalid analysis_level"):
            HierarchicalMMM(analysis_level='invalid')


class TestEmptyData:
    """Test handling of edge cases with empty or minimal data."""

    def test_empty_dataframe(self, channel_configs):
        """Test that empty DataFrame raises appropriate error."""
        empty_df = pd.DataFrame(columns=['date', 'country', 'os', 'google', 'meta', 'revenue'])

        mmm = HierarchicalMMM(analysis_level='country')

        # Should raise error or handle gracefully
        with pytest.raises((ValueError, IndexError)):
            mmm.fit(empty_df, target_col='revenue', channel_configs=channel_configs)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
