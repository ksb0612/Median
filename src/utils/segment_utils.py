"""Segment filtering utilities for hierarchical models."""
import pandas as pd
from typing import Dict, List


def parse_segment(
    segment: str,
    available_countries: List[str],
    available_os: List[str]
) -> Dict[str, str]:
    """
    Parse segment string into components.

    Segments can be:
    - 'global': All data
    - 'KR': Country-level
    - 'iOS': OS-level
    - 'KR-iOS': Country-OS level

    Args:
        segment: Segment string ('global', 'KR', 'iOS', 'KR-iOS')
        available_countries: List of valid countries
        available_os: List of valid OS platforms

    Returns:
        Dict with segment level and filters

    Raises:
        ValueError: If segment format is invalid

    Examples:
        >>> parse_segment('global', ['KR', 'US'], ['iOS', 'Android'])
        {'level': 'global'}
        >>> parse_segment('KR', ['KR', 'US'], ['iOS', 'Android'])
        {'level': 'country', 'country': 'KR'}
        >>> parse_segment('KR-iOS', ['KR', 'US'], ['iOS', 'Android'])
        {'level': 'country_os', 'country': 'KR', 'os': 'iOS'}
    """
    if segment == 'global':
        return {'level': 'global'}

    parts = segment.split('-')

    if len(parts) == 1:
        # Single segment: country or OS
        if segment in available_countries:
            return {'level': 'country', 'country': segment}
        elif segment in available_os:
            return {'level': 'os', 'os': segment}
        else:
            raise ValueError(
                f"Segment '{segment}' not recognized. "
                f"Available countries: {available_countries}. "
                f"Available OS: {available_os}"
            )

    elif len(parts) == 2:
        country, os = parts
        if country not in available_countries:
            raise ValueError(
                f"Country '{country}' not in available: {available_countries}"
            )
        if os not in available_os:
            raise ValueError(
                f"OS '{os}' not in available: {available_os}"
            )
        return {'level': 'country_os', 'country': country, 'os': os}

    else:
        raise ValueError(f"Invalid segment format: '{segment}'")


def filter_by_segment(
    df: pd.DataFrame,
    segment_dict: Dict[str, str]
) -> pd.DataFrame:
    """
    Filter DataFrame by segment specification.

    Args:
        df: DataFrame to filter
        segment_dict: Dict from parse_segment()

    Returns:
        Filtered DataFrame

    Raises:
        ValueError: If required columns missing or no data matches

    Examples:
        >>> df = pd.DataFrame({
        ...     'country': ['KR', 'KR', 'US'],
        ...     'os': ['iOS', 'Android', 'iOS'],
        ...     'revenue': [100, 200, 300]
        ... })
        >>> segment = {'level': 'country', 'country': 'KR'}
        >>> filtered = filter_by_segment(df, segment)
        >>> len(filtered)
        2
    """
    if segment_dict['level'] == 'global':
        return df.copy()

    conditions = []

    if 'country' in segment_dict:
        if 'country' not in df.columns:
            raise ValueError("Data has no 'country' column")
        conditions.append(df['country'] == segment_dict['country'])

    if 'os' in segment_dict:
        if 'os' not in df.columns:
            raise ValueError("Data has no 'os' column")
        conditions.append(df['os'] == segment_dict['os'])

    if conditions:
        combined = conditions[0]
        for cond in conditions[1:]:
            combined &= cond

        filtered = df[combined].copy()

        if len(filtered) == 0:
            raise ValueError(f"No data found for segment: {segment_dict}")

        return filtered

    return df.copy()


def get_segment_hierarchy(
    available_countries: List[str],
    available_os: List[str]
) -> Dict[str, List[str]]:
    """
    Generate hierarchical segment structure.

    Args:
        available_countries: List of countries
        available_os: List of OS platforms

    Returns:
        Dict mapping levels to segment names

    Examples:
        >>> hierarchy = get_segment_hierarchy(['KR', 'US'], ['iOS', 'Android'])
        >>> hierarchy['global']
        ['global']
        >>> hierarchy['country']
        ['KR', 'US']
        >>> 'KR-iOS' in hierarchy['country_os']
        True
    """
    hierarchy = {
        'global': ['global'],
        'country': available_countries.copy(),
        'os': available_os.copy(),
        'country_os': []
    }

    # Generate all country-OS combinations
    for country in available_countries:
        for os in available_os:
            hierarchy['country_os'].append(f"{country}-{os}")

    return hierarchy
