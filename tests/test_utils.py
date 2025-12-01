"""Tests for utility functions."""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.data_utils import (
    validate_required_columns,
    validate_column_types,
    safe_divide
)
from utils.segment_utils import (
    parse_segment,
    filter_by_segment,
    get_segment_hierarchy
)
from utils.plot_utils import (
    get_default_layout,
    get_color_palette,
    format_currency,
    format_percentage
)


# ============================================================================
# DATA UTILS TESTS
# ============================================================================

class TestValidateRequiredColumns:
    """Tests for validate_required_columns function."""

    def test_success_all_columns_present(self):
        """Test validation passes when all required columns are present."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        # Should not raise
        validate_required_columns(df, ['a', 'b'])

    def test_failure_missing_columns(self):
        """Test validation fails when columns are missing."""
        df = pd.DataFrame({'a': [1, 2]})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_required_columns(df, ['a', 'b', 'c'])

    def test_custom_context_in_error(self):
        """Test that custom context appears in error message."""
        df = pd.DataFrame({'a': [1, 2]})
        with pytest.raises(ValueError, match="My DataFrame"):
            validate_required_columns(df, ['b'], context="My DataFrame")


class TestValidateColumnTypes:
    """Tests for validate_column_types function."""

    def test_numeric_columns_valid(self):
        """Test validation passes for numeric columns."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4.5, 5.5, 6.5]})
        # Should not raise
        validate_column_types(df, numeric_cols=['a', 'b'])

    def test_numeric_columns_invalid(self):
        """Test validation fails for non-numeric columns."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        with pytest.raises(ValueError, match="must be numeric"):
            validate_column_types(df, numeric_cols=['a', 'b'])

    def test_date_columns_valid(self):
        """Test validation passes for datetime columns."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])
        })
        # Should not raise
        validate_column_types(df, date_cols=['date'])

    def test_date_columns_invalid(self):
        """Test validation fails for non-datetime columns."""
        df = pd.DataFrame({'date': ['2020-01-01', '2020-01-02']})
        with pytest.raises(ValueError, match="must be datetime"):
            validate_column_types(df, date_cols=['date'])


class TestSafeDivide:
    """Tests for safe_divide function."""

    def test_normal_division(self):
        """Test normal division works correctly."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(15, 3) == 5.0
        assert safe_divide(7, 2) == 3.5

    def test_division_by_zero_default(self):
        """Test division by zero returns default value."""
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=0.0) == 0.0

    def test_division_by_zero_custom_default(self):
        """Test division by zero returns custom default."""
        assert safe_divide(10, 0, default=999.0) == 999.0
        assert np.isnan(safe_divide(10, 0, default=np.nan))

    def test_negative_numbers(self):
        """Test division works with negative numbers."""
        assert safe_divide(-10, 2) == -5.0
        assert safe_divide(10, -2) == -5.0
        assert safe_divide(-10, -2) == 5.0


# ============================================================================
# SEGMENT UTILS TESTS
# ============================================================================

class TestParseSegment:
    """Tests for parse_segment function."""

    @pytest.fixture
    def segment_config(self):
        """Provide sample countries and OS for testing."""
        return {
            'countries': ['KR', 'US', 'JP'],
            'os': ['iOS', 'Android']
        }

    def test_global_segment(self, segment_config):
        """Test parsing global segment."""
        result = parse_segment('global', segment_config['countries'], segment_config['os'])
        assert result == {'level': 'global'}

    def test_country_segment(self, segment_config):
        """Test parsing country-only segment."""
        result = parse_segment('KR', segment_config['countries'], segment_config['os'])
        assert result == {'level': 'country', 'country': 'KR'}

    def test_os_segment(self, segment_config):
        """Test parsing OS-only segment."""
        result = parse_segment('iOS', segment_config['countries'], segment_config['os'])
        assert result == {'level': 'os', 'os': 'iOS'}

    def test_country_os_segment(self, segment_config):
        """Test parsing country-OS combined segment."""
        result = parse_segment('KR-iOS', segment_config['countries'], segment_config['os'])
        assert result == {'level': 'country_os', 'country': 'KR', 'os': 'iOS'}

    def test_invalid_segment(self, segment_config):
        """Test parsing invalid segment raises error."""
        with pytest.raises(ValueError, match="not recognized"):
            parse_segment('INVALID', segment_config['countries'], segment_config['os'])

    def test_invalid_format(self, segment_config):
        """Test invalid format raises error."""
        with pytest.raises(ValueError, match="Invalid segment format"):
            parse_segment('A-B-C', segment_config['countries'], segment_config['os'])


class TestFilterBySegment:
    """Tests for filter_by_segment function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'country': ['KR', 'KR', 'US', 'US'],
            'os': ['iOS', 'Android', 'iOS', 'Android'],
            'revenue': [100, 200, 300, 400]
        })

    def test_global_filter(self, sample_data):
        """Test filtering with global segment returns all data."""
        segment_dict = {'level': 'global'}
        result = filter_by_segment(sample_data, segment_dict)
        assert len(result) == 4

    def test_country_filter(self, sample_data):
        """Test filtering by country."""
        segment_dict = {'level': 'country', 'country': 'KR'}
        result = filter_by_segment(sample_data, segment_dict)
        assert len(result) == 2
        assert all(result['country'] == 'KR')

    def test_os_filter(self, sample_data):
        """Test filtering by OS."""
        segment_dict = {'level': 'os', 'os': 'iOS'}
        result = filter_by_segment(sample_data, segment_dict)
        assert len(result) == 2
        assert all(result['os'] == 'iOS')

    def test_country_os_filter(self, sample_data):
        """Test filtering by country and OS."""
        segment_dict = {'level': 'country_os', 'country': 'KR', 'os': 'iOS'}
        result = filter_by_segment(sample_data, segment_dict)
        assert len(result) == 1
        assert result.iloc[0]['revenue'] == 100

    def test_no_data_found(self, sample_data):
        """Test error when no data matches segment."""
        segment_dict = {'level': 'country', 'country': 'JP'}
        with pytest.raises(ValueError, match="No data found"):
            filter_by_segment(sample_data, segment_dict)

    def test_missing_column(self, sample_data):
        """Test error when required column is missing."""
        df_no_os = sample_data.drop('os', axis=1)
        segment_dict = {'level': 'os', 'os': 'iOS'}
        with pytest.raises(ValueError, match="no 'os' column"):
            filter_by_segment(df_no_os, segment_dict)


class TestGetSegmentHierarchy:
    """Tests for get_segment_hierarchy function."""

    def test_hierarchy_structure(self):
        """Test that hierarchy has correct structure."""
        hierarchy = get_segment_hierarchy(['KR', 'US'], ['iOS', 'Android'])

        assert 'global' in hierarchy
        assert 'country' in hierarchy
        assert 'os' in hierarchy
        assert 'country_os' in hierarchy

    def test_global_level(self):
        """Test global level contains only 'global'."""
        hierarchy = get_segment_hierarchy(['KR', 'US'], ['iOS', 'Android'])
        assert hierarchy['global'] == ['global']

    def test_country_level(self):
        """Test country level contains all countries."""
        hierarchy = get_segment_hierarchy(['KR', 'US', 'JP'], ['iOS', 'Android'])
        assert set(hierarchy['country']) == {'KR', 'US', 'JP'}

    def test_os_level(self):
        """Test OS level contains all platforms."""
        hierarchy = get_segment_hierarchy(['KR', 'US'], ['iOS', 'Android', 'Web'])
        assert set(hierarchy['os']) == {'iOS', 'Android', 'Web'}

    def test_country_os_combinations(self):
        """Test country_os level contains all combinations."""
        hierarchy = get_segment_hierarchy(['KR', 'US'], ['iOS', 'Android'])
        expected = {'KR-iOS', 'KR-Android', 'US-iOS', 'US-Android'}
        assert set(hierarchy['country_os']) == expected


# ============================================================================
# PLOT UTILS TESTS
# ============================================================================

class TestGetDefaultLayout:
    """Tests for get_default_layout function."""

    def test_basic_layout(self):
        """Test basic layout generation."""
        layout = get_default_layout("Test Title")
        assert layout['title']['text'] == "Test Title"
        assert layout['height'] == 500

    def test_custom_titles(self):
        """Test custom axis titles."""
        layout = get_default_layout(
            "Test",
            xaxis_title="Custom X",
            yaxis_title="Custom Y"
        )
        assert layout['xaxis']['title'] == "Custom X"
        assert layout['yaxis']['title'] == "Custom Y"

    def test_custom_height(self):
        """Test custom height."""
        layout = get_default_layout("Test", height=800)
        assert layout['height'] == 800

    def test_consistent_styling(self):
        """Test that layout has consistent styling elements."""
        layout = get_default_layout("Test")
        assert layout['plot_bgcolor'] == 'white'
        assert layout['paper_bgcolor'] == 'white'
        assert 'showlegend' in layout
        assert 'hovermode' in layout


class TestGetColorPalette:
    """Tests for get_color_palette function."""

    def test_palette_structure(self):
        """Test color palette has expected structure."""
        palette = get_color_palette()
        assert 'primary' in palette
        assert 'secondary' in palette
        assert 'success' in palette
        assert 'warning' in palette
        assert 'danger' in palette
        assert 'channels' in palette

    def test_channels_colors_count(self):
        """Test that channels color list has enough colors."""
        palette = get_color_palette()
        assert len(palette['channels']) == 10

    def test_color_format(self):
        """Test that colors are in hex format."""
        palette = get_color_palette()
        assert palette['primary'].startswith('#')
        assert len(palette['primary']) == 7  # #RRGGBB


class TestFormatCurrency:
    """Tests for format_currency function."""

    def test_basic_formatting(self):
        """Test basic currency formatting."""
        assert format_currency(1234567.89) == '$1,234,568'

    def test_with_decimals(self):
        """Test formatting with decimal places."""
        assert format_currency(1234.5678, decimals=2) == '$1,234.57'

    def test_negative_values(self):
        """Test formatting negative values."""
        assert format_currency(-1234.56) == '$-1,235'

    def test_zero(self):
        """Test formatting zero."""
        assert format_currency(0) == '$0'


class TestFormatPercentage:
    """Tests for format_percentage function."""

    def test_basic_formatting(self):
        """Test basic percentage formatting."""
        assert format_percentage(0.1534) == '15.3%'

    def test_with_custom_decimals(self):
        """Test formatting with custom decimal places."""
        assert format_percentage(0.1534, decimals=2) == '15.34%'

    def test_whole_number(self):
        """Test formatting whole percentage."""
        assert format_percentage(0.5, decimals=0) == '50%'

    def test_small_percentage(self):
        """Test formatting small percentage."""
        assert format_percentage(0.0123) == '1.2%'


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for utilities working together."""

    def test_segment_workflow(self):
        """Test complete segment filtering workflow."""
        # Create data
        df = pd.DataFrame({
            'country': ['KR', 'KR', 'US'],
            'os': ['iOS', 'Android', 'iOS'],
            'revenue': [100, 200, 300]
        })

        # Parse and filter
        segment_dict = parse_segment('KR', ['KR', 'US'], ['iOS', 'Android'])
        filtered = filter_by_segment(df, segment_dict)

        # Validate results
        assert len(filtered) == 2
        assert filtered['revenue'].sum() == 300

    def test_data_validation_workflow(self):
        """Test complete data validation workflow."""
        df = pd.DataFrame({
            'channel1': [100, 200, 300],
            'channel2': [150, 250, 350],
            'revenue': [1000, 2000, 3000]
        })

        # Validate columns and types
        validate_required_columns(df, ['channel1', 'channel2', 'revenue'])
        validate_column_types(df, numeric_cols=['channel1', 'channel2', 'revenue'])

        # Use safe_divide for ROAS calculation
        roas1 = safe_divide(df['revenue'].sum(), df['channel1'].sum())
        roas2 = safe_divide(df['revenue'].sum(), df['channel2'].sum())

        assert roas1 == 10.0
        assert roas2 == 8.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
