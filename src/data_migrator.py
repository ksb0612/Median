"""
Data migration utilities for Ridge MMM.

This module provides functions to convert between single-market and multi-market data formats.
"""

import pandas as pd
from typing import Tuple, Optional


def convert_single_to_multi(
    df: pd.DataFrame,
    default_country: str = 'GLOBAL',
    default_os: str = 'ALL'
) -> pd.DataFrame:
    """
    Convert single-market data to multi-market format.
    
    Args:
        df: Single-market DataFrame
        default_country: Default country code to assign
        default_os: Default OS platform to assign
        
    Returns:
        pd.DataFrame: Multi-market format with country and os columns
    """
    df_multi = df.copy()
    
    # Add country and os columns if they don't exist
    if 'country' not in df_multi.columns:
        df_multi['country'] = default_country
    
    if 'os' not in df_multi.columns:
        df_multi['os'] = default_os
    
    # Reorder columns to put country and os after date
    cols = df_multi.columns.tolist()
    
    # Try to put country and os right after date column
    if 'date' in cols:
        date_idx = cols.index('date')
        cols.remove('country')
        cols.remove('os')
        cols.insert(date_idx + 1, 'country')
        cols.insert(date_idx + 2, 'os')
        df_multi = df_multi[cols]
    
    return df_multi


def detect_and_convert(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, list]:
    """
    Auto-detect data format and convert if needed.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of:
            - converted_df: DataFrame in multi-market format
            - format_type: Detected format ('single' | 'multi_country' | 'multi_country_os')
            - conversion_notes: List of notes about the conversion
    """
    notes = []
    
    # Detect format
    has_country = 'country' in df.columns
    has_os = 'os' in df.columns
    
    if has_country and has_os:
        format_type = 'multi_country_os'
        notes.append("Data is already in multi-country-os format")
        return df, format_type, notes
        
    elif has_country:
        format_type = 'multi_country'
        notes.append("Data is in multi-country format (no OS column)")
        return df, format_type, notes
        
    else:
        format_type = 'single'
        notes.append("Data is in single-market format")
        notes.append("Converting to multi-market format with default values")
        
        converted_df = convert_single_to_multi(df)
        notes.append(f"Added 'country' column with value 'GLOBAL'")
        notes.append(f"Added 'os' column with value 'ALL'")
        
        return converted_df, 'multi_country_os', notes


def split_by_market(df: pd.DataFrame, market_segment: str) -> pd.DataFrame:
    """
    Extract data for a specific market segment.
    
    Args:
        df: Multi-market DataFrame
        market_segment: Market segment string (e.g., 'KR-iOS', 'US-Android')
        
    Returns:
        pd.DataFrame: Filtered data for the specified market
    """
    if '-' in market_segment:
        # Country-OS format
        country, os = market_segment.split('-', 1)
        return df[(df['country'] == country) & (df['os'] == os)].copy()
    else:
        # Country only format
        return df[df['country'] == market_segment].copy()


def combine_markets(dfs: dict, include_market_columns: bool = True) -> pd.DataFrame:
    """
    Combine multiple market DataFrames into a single multi-market DataFrame.
    
    Args:
        dfs: Dict mapping market segments to DataFrames
              e.g., {'KR-iOS': df_kr_ios, 'US-Android': df_us_android}
        include_market_columns: Whether to add country/os columns
        
    Returns:
        pd.DataFrame: Combined multi-market DataFrame
    """
    combined_dfs = []
    
    for market_segment, df in dfs.items():
        df_copy = df.copy()
        
        if include_market_columns:
            if '-' in market_segment:
                country, os = market_segment.split('-', 1)
                df_copy['country'] = country
                df_copy['os'] = os
            else:
                df_copy['country'] = market_segment
                df_copy['os'] = 'ALL'
        
        combined_dfs.append(df_copy)
    
    return pd.concat(combined_dfs, ignore_index=True)
