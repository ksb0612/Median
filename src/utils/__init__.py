"""
Utility modules for Ridge MMM.

This package contains reusable utility functions for data validation,
segment filtering, plotting, and metric calculations.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any

from .data_utils import (
    validate_required_columns,
    validate_column_types,
    safe_divide
)

from .segment_utils import (
    parse_segment,
    filter_by_segment
)

from .plot_utils import (
    get_default_layout,
    apply_default_layout,
    COLOR_PALETTE
)


# Metric calculation functions for backward compatibility
def calculate_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    MAPE measures prediction accuracy as a percentage. Lower values indicate
    better predictions. This implementation handles edge cases gracefully by
    filtering out zero values and using epsilon to prevent division by zero.

    Args:
        y_true: True values (ground truth)
        y_pred: Predicted values
        epsilon: Small value added to denominator to prevent division by zero
                (default: 1e-10). Only used as a fallback for numerical stability.

    Returns:
        MAPE as percentage (0-100). Returns np.nan if all y_true values are zero.

    Example:
        >>> y_true = np.array([100, 200, 300])
        >>> y_pred = np.array([110, 190, 295])
        >>> mape = calculate_mape(y_true, y_pred)
        >>> print(f"MAPE: {mape:.2f}%")

    Note:
        - Zero values in y_true are filtered out before calculation to avoid
          division by zero errors
        - Returns np.nan if all y_true values are zero
        - Epsilon parameter provides additional numerical stability
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Filter out zero values to avoid division by zero
    mask = y_true != 0
    if not mask.any():
        # All y_true values are zero - cannot calculate MAPE
        return np.nan

    # Calculate MAPE only on non-zero values, with epsilon for stability
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + epsilon))) * 100
    return float(mape)


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        R-squared value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1 - (ss_res / ss_tot))


# Additional utility functions for backward compatibility
def format_currency(value: float, currency: str = '₩', decimals: int = 0) -> str:
    """
    Format large numbers as currency with appropriate suffixes.

    Args:
        value: Number to format
        currency: Currency symbol (default: '₩')
        decimals: Number of decimal places

    Returns:
        Formatted string (e.g., '₩1,500M' or '₩2.5B')
    """
    if pd.isna(value):
        return "N/A"

    abs_value = abs(value)
    sign = '-' if value < 0 else ''

    if abs_value >= 1_000_000_000:  # Billions
        formatted = f"{abs_value / 1_000_000_000:,.{decimals}f}B"
    elif abs_value >= 1_000_000:  # Millions
        formatted = f"{abs_value / 1_000_000:,.{decimals}f}M"
    elif abs_value >= 1_000:  # Thousands
        formatted = f"{abs_value / 1_000:,.{decimals}f}K"
    else:
        formatted = f"{abs_value:,.{decimals}f}"

    return f"{sign}{currency}{formatted}"


def format_allocation_table(
    current: Dict[str, float],
    optimal: Dict[str, float]
) -> pd.DataFrame:
    """
    Format side-by-side comparison of current vs optimal allocation.

    Args:
        current: Current allocation {channel: spend}
        optimal: Optimal allocation {channel: spend}

    Returns:
        DataFrame with comparison and delta columns
    """
    channels = list(current.keys())

    data = []
    for channel in channels:
        curr_spend = current.get(channel, 0)
        opt_spend = optimal.get(channel, 0)
        delta = opt_spend - curr_spend
        delta_pct = (delta / curr_spend * 100) if curr_spend > 0 else 0

        data.append({
            'Channel': channel,
            'Current': curr_spend,
            'Optimal': opt_spend,
            'Change': delta,
            'Change %': delta_pct
        })

    # Add totals row
    total_current = sum(current.values())
    total_optimal = sum(optimal.values())
    total_delta = total_optimal - total_current
    total_delta_pct = (total_delta / total_current * 100) if total_current > 0 else 0

    data.append({
        'Channel': 'TOTAL',
        'Current': total_current,
        'Optimal': total_optimal,
        'Change': total_delta,
        'Change %': total_delta_pct
    })

    return pd.DataFrame(data)


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of numeric column names from DataFrame.

    Args:
        df: DataFrame

    Returns:
        List of numeric column names
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_date_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of potential date column names from DataFrame.

    Args:
        df: DataFrame

    Returns:
        List of column names that might be dates
    """
    date_cols = []

    for col in df.columns:
        # Check if column name suggests it's a date
        if any(keyword in col.lower() for keyword in ['date', 'time', 'day', 'week', 'month', 'year']):
            date_cols.append(col)
            continue

        # Try to parse as date
        try:
            pd.to_datetime(df[col], errors='raise')
            date_cols.append(col)
        except:
            pass

    return date_cols


def validate_column_selection(df: pd.DataFrame,
                              date_col: str,
                              revenue_col: str,
                              media_cols: List[str]) -> Dict[str, Any]:
    """
    Validate that selected columns are appropriate.

    Args:
        df: DataFrame
        date_col: Selected date column
        revenue_col: Selected revenue column
        media_cols: Selected media columns

    Returns:
        Dict with validation results
    """
    errors = []
    warnings = []

    # Check if columns exist
    all_cols = [date_col, revenue_col] + media_cols
    missing_cols = [col for col in all_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Columns not found: {missing_cols}")

    # Check for overlap
    if date_col == revenue_col:
        errors.append("Date and revenue columns cannot be the same")

    if date_col in media_cols:
        errors.append("Date column cannot be a media column")

    if revenue_col in media_cols:
        errors.append("Revenue column cannot be a media column")

    # Check for duplicates in media columns
    if len(media_cols) != len(set(media_cols)):
        errors.append("Duplicate media columns selected")

    return {
        'is_valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }


def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get detailed information about DataFrame columns.

    Args:
        df: DataFrame to analyze

    Returns:
        DataFrame with column information
    """
    info_data = []

    for col in df.columns:
        info_data.append({
            'Column': col,
            'Type': str(df[col].dtype),
            'Non-Null': df[col].notna().sum(),
            'Null': df[col].isna().sum(),
            'Unique': df[col].nunique(),
            'Sample': str(df[col].iloc[0]) if len(df) > 0 else None
        })

    return pd.DataFrame(info_data)


__all__ = [
    # Data utils
    'validate_required_columns',
    'validate_column_types',
    'safe_divide',
    # Segment utils
    'parse_segment',
    'filter_by_segment',
    # Plot utils
    'get_default_layout',
    'apply_default_layout',
    'COLOR_PALETTE',
    # Metrics
    'calculate_mape',
    'calculate_mae',
    'calculate_rmse',
    'calculate_r2',
    # Formatting
    'format_currency',
    'format_allocation_table',
    # Column utilities
    'get_numeric_columns',
    'get_date_columns',
    'validate_column_selection',
    'get_column_info'
]
