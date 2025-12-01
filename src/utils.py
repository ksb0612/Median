"""
Utility functions for Ridge MMM application.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import io


def format_number(value: float, prefix: str = "", suffix: str = "", decimals: int = 2) -> str:
    """
    Format a number with prefix, suffix, and decimal places.
    
    Args:
        value: Number to format
        prefix: String to prepend (e.g., '$')
        suffix: String to append (e.g., '%')
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    if pd.isna(value):
        return "N/A"
    
    formatted = f"{value:,.{decimals}f}"
    return f"{prefix}{formatted}{suffix}"


def calculate_correlation_matrix(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Calculate correlation matrix for specified columns.
    
    Args:
        df: DataFrame
        columns: List of column names
        
    Returns:
        Correlation matrix as DataFrame
    """
    return df[columns].corr().round(3)


def detect_multicollinearity(df: pd.DataFrame, columns: List[str], threshold: float = 0.8) -> List[tuple]:
    """
    Detect highly correlated pairs of variables (multicollinearity).
    
    Args:
        df: DataFrame
        columns: List of column names to check
        threshold: Correlation threshold (default 0.8)
        
    Returns:
        List of tuples (col1, col2, correlation) for highly correlated pairs
    """
    corr_matrix = df[columns].corr()
    high_corr_pairs = []
    
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            corr_value = abs(corr_matrix.iloc[i, j])
            if corr_value >= threshold:
                high_corr_pairs.append((columns[i], columns[j], round(corr_value, 3)))
    
    return high_corr_pairs


def create_download_link(df: pd.DataFrame, filename: str = "data.csv") -> bytes:
    """
    Create a downloadable CSV from DataFrame.
    
    Args:
        df: DataFrame to convert
        filename: Name for the file
        
    Returns:
        Bytes object for download
    """
    return df.to_csv(index=False).encode('utf-8')


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


def calculate_weekly_aggregation(df: pd.DataFrame, date_col: str, value_cols: List[str]) -> pd.DataFrame:
    """
    Aggregate data to weekly level.
    
    Args:
        df: DataFrame with date column
        date_col: Name of date column
        value_cols: Columns to aggregate (sum)
        
    Returns:
        Weekly aggregated DataFrame
    """
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    df_copy['week'] = df_copy[date_col].dt.to_period('W').dt.start_time
    
    agg_dict = {col: 'sum' for col in value_cols}
    weekly_df = df_copy.groupby('week').agg(agg_dict).reset_index()
    weekly_df.rename(columns={'week': date_col}, inplace=True)
    
    return weekly_df


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Value to return if denominator is zero
        
    Returns:
        Result of division or default value
    """
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


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


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
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
    return mape


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
    
    return np.mean(np.abs(y_true - y_pred))


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
    
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def format_currency(value: float, currency: str = '₩', decimals: int = 0, show_unit: bool = True) -> str:
    """
    Format large numbers as currency with appropriate suffixes.
    
    Args:
        value: Number to format
        currency: Currency symbol (default: '₩')
        decimals: Number of decimal places
        show_unit: Whether to show 'KRW' unit (default: True)
        
    Returns:
        Formatted string (e.g., '₩1,500M KRW' or '₩2.5B KRW')
        
    Example:
        >>> format_currency(1500000000)
        '₩1,500M KRW'
        >>> format_currency(2500000000)
        '₩2.5B KRW'
        >>> format_currency(15000, show_unit=False)
        '₩15K'
    """
    if pd.isna(value):
        return "N/A"
    
    abs_value = abs(value)
    sign = '-' if value < 0 else ''
    
    if abs_value >= 1_000_000_000:  # Billions (십억)
        formatted = f"{abs_value / 1_000_000_000:,.{decimals}f}B"
    elif abs_value >= 1_000_000:  # Millions (백만)
        formatted = f"{abs_value / 1_000_000:,.{decimals}f}M"
    elif abs_value >= 1_000:  # Thousands (천)
        formatted = f"{abs_value / 1_000:,.{decimals}f}K"
    else:
        formatted = f"{abs_value:,.{decimals}f}"
    
    result = f"{sign}{currency}{formatted}"
    
    # Add KRW unit for clarity
    if show_unit and currency == '₩':
        result += " 원"
    
    return result


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
    
    Example:
        >>> current = {'google': 15000, 'facebook': 10000}
        >>> optimal = {'google': 18000, 'facebook': 12000}
        >>> df = format_allocation_table(current, optimal)
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


