"""Data manipulation utilities."""
import pandas as pd
from typing import List, Optional


def validate_required_columns(
    df: pd.DataFrame,
    required_cols: List[str],
    context: str = "DataFrame"
) -> None:
    """
    Validate that DataFrame has required columns.

    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        context: Context for error message

    Raises:
        ValueError: If any required columns are missing

    Examples:
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> validate_required_columns(df, ['a', 'b'])  # No error
        >>> validate_required_columns(df, ['a', 'c'])  # Raises ValueError
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"{context} missing required columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )


def validate_column_types(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    date_cols: Optional[List[str]] = None
) -> None:
    """
    Validate column data types.

    Args:
        df: DataFrame to validate
        numeric_cols: Columns that should be numeric
        date_cols: Columns that should be datetime

    Raises:
        ValueError: If columns have wrong types

    Examples:
        >>> df = pd.DataFrame({'a': [1, 2], 'b': ['2020-01-01', '2020-01-02']})
        >>> df['b'] = pd.to_datetime(df['b'])
        >>> validate_column_types(df, numeric_cols=['a'], date_cols=['b'])
    """
    if numeric_cols:
        for col in numeric_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(
                    f"Column '{col}' must be numeric, got {df[col].dtype}"
                )

    if date_cols:
        for col in date_cols:
            if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                raise ValueError(
                    f"Column '{col}' must be datetime, got {df[col].dtype}"
                )


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0
) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    This is useful for calculating ratios like ROAS where spend might be zero.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Value to return if denominator is zero

    Returns:
        Result of division or default

    Examples:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        0.0
        >>> safe_divide(10, 0, default=float('nan'))
        nan
    """
    return numerator / denominator if denominator != 0 else default
