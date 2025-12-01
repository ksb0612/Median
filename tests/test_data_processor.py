"""
Basic tests for DataProcessor class
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_processor import DataProcessor


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=10, freq='W'),
        'revenue': [100000, 105000, 110000, 108000, 112000, 115000, 118000, 120000, 122000, 125000],
        'google': [10000, 11000, 12000, 11500, 13000, 14000, 15000, 16000, 17000, 18000],
        'facebook': [8000, 8500, 9000, 8800, 9500, 10000, 10500, 11000, 11500, 12000],
        'promotion': [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    })


@pytest.fixture
def processor():
    """Create a DataProcessor instance."""
    return DataProcessor()


def test_load_csv_from_dataframe(processor, sample_df, tmp_path):
    """Test loading CSV from file."""
    # Save sample data to temporary CSV
    csv_path = tmp_path / "test_data.csv"
    sample_df.to_csv(csv_path, index=False)
    
    # Load the CSV
    df = processor.load_csv(str(csv_path))
    
    assert df is not None
    assert len(df) == 10
    assert 'revenue' in df.columns


def test_validate_data_success(processor, sample_df):
    """Test successful data validation."""
    result = processor.validate_data(
        sample_df,
        date_col='date',
        revenue_col='revenue',
        media_cols=['google', 'facebook']
    )
    
    assert result['is_valid'] == True
    assert len(result['errors']) == 0


def test_validate_data_missing_column(processor, sample_df):
    """Test validation with missing column."""
    result = processor.validate_data(
        sample_df,
        date_col='date',
        revenue_col='revenue',
        media_cols=['google', 'nonexistent']
    )
    
    assert result['is_valid'] == False
    assert len(result['errors']) > 0


def test_check_missing_values(processor, sample_df):
    """Test missing value detection."""
    # Add some missing values
    df_with_missing = sample_df.copy()
    df_with_missing.loc[0, 'revenue'] = np.nan
    df_with_missing.loc[1, 'google'] = np.nan
    
    missing_report = processor.check_missing_values(df_with_missing)
    
    assert 'revenue' in missing_report['Column'].values
    assert missing_report[missing_report['Column'] == 'revenue']['Missing_Count'].values[0] == 1


def test_detect_outliers(processor, sample_df):
    """Test outlier detection."""
    # Add an outlier
    df_with_outlier = sample_df.copy()
    df_with_outlier.loc[0, 'revenue'] = 1000000  # Extreme outlier
    
    outliers = processor.detect_outliers(df_with_outlier, columns=['revenue'])
    
    assert 'revenue' in outliers
    assert outliers['revenue']['outlier_count'] > 0


def test_get_summary_stats(processor, sample_df):
    """Test summary statistics generation."""
    stats = processor.get_summary_stats(sample_df)
    
    assert 'revenue' in stats.index
    assert 'mean' in stats.columns
    assert 'std' in stats.columns


def test_convert_date_column(processor, sample_df):
    """Test date column conversion."""
    df = sample_df.copy()
    df['date'] = df['date'].astype(str)
    
    converted_df = processor.convert_date_column(df, 'date')
    
    assert pd.api.types.is_datetime64_any_dtype(converted_df['date'])


def test_get_date_range(processor, sample_df):
    """Test date range extraction."""
    start, end, periods = processor.get_date_range(sample_df, 'date')
    
    assert start is not None
    assert end is not None
    assert periods == 10
