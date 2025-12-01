"""
Data processing module for Ridge MMM application.

This module provides the DataProcessor class for loading, validating,
and analyzing marketing data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import io


class DataProcessor:
    """
    A class for processing and validating marketing mix modeling data.
    
    This class handles data loading, validation, quality checks, and
    statistical analysis of marketing data.
    """
    
    def __init__(self):
        """Initialize the DataProcessor."""
        self.df = None
        self.date_column = None
        self.revenue_column = None
        self.media_columns = []
        self.exog_columns = []
    
    def load_csv(self, file) -> pd.DataFrame:
        """
        Load CSV file and return DataFrame.
        
        Args:
            file: File object from Streamlit file_uploader or file path
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            ValueError: If file cannot be loaded or is empty
        """
        try:
            # Handle both file path and file-like object
            if isinstance(file, str):
                df = pd.read_csv(file)
            else:
                df = pd.read_csv(file)
            
            if df.empty:
                raise ValueError("The uploaded file is empty")
            
            self.df = df
            return df
            
        except pd.errors.EmptyDataError:
            raise ValueError("The file is empty or contains no data")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing CSV file: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error loading file: {str(e)}")
    
    def validate_data(self, df: pd.DataFrame, 
                     date_col: Optional[str] = None,
                     revenue_col: Optional[str] = None,
                     media_cols: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Validate data for required columns and basic requirements.
        
        Args:
            df: DataFrame to validate
            date_col: Name of date column (optional)
            revenue_col: Name of revenue column (optional)
            media_cols: List of media channel columns (optional)
            
        Returns:
            Dict with validation results:
                - 'is_valid': bool
                - 'errors': List of error messages
                - 'warnings': List of warning messages
        """
        errors = []
        warnings = []
        
        # Check if DataFrame is empty
        if df.empty:
            errors.append("DataFrame is empty")
            return {'is_valid': False, 'errors': errors, 'warnings': warnings}
        
        # Check minimum number of rows
        if len(df) < 10:
            warnings.append(f"Dataset has only {len(df)} rows. Recommend at least 52 weeks for reliable modeling")
        
        # Validate date column if provided
        if date_col:
            if date_col not in df.columns:
                errors.append(f"Date column '{date_col}' not found in data")
            else:
                try:
                    pd.to_datetime(df[date_col])
                except Exception as e:
                    errors.append(f"Date column '{date_col}' cannot be converted to datetime: {str(e)}")
        
        # Validate revenue column if provided
        if revenue_col:
            if revenue_col not in df.columns:
                errors.append(f"Revenue column '{revenue_col}' not found in data")
            else:
                if not pd.api.types.is_numeric_dtype(df[revenue_col]):
                    errors.append(f"Revenue column '{revenue_col}' must be numeric")
                if (df[revenue_col] < 0).any():
                    warnings.append(f"Revenue column '{revenue_col}' contains negative values")
        
        # Validate media columns if provided
        if media_cols:
            if len(media_cols) == 0:
                errors.append("At least one media channel column is required")
            else:
                for col in media_cols:
                    if col not in df.columns:
                        errors.append(f"Media column '{col}' not found in data")
                    else:
                        if not pd.api.types.is_numeric_dtype(df[col]):
                            errors.append(f"Media column '{col}' must be numeric")
                        if (df[col] < 0).any():
                            warnings.append(f"Media column '{col}' contains negative values")
        
        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            duplicates = df.columns[df.columns.duplicated()].tolist()
            errors.append(f"Duplicate column names found: {duplicates}")
        
        is_valid = len(errors) == 0
        
        return {
            'is_valid': is_valid,
            'errors': errors,
            'warnings': warnings
        }
    
    def check_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check for missing values in the DataFrame.
        
        Args:
            df: DataFrame to check
            
        Returns:
            pd.DataFrame: Report with columns, missing count, and percentage
        """
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum().values,
            'Missing_Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        
        # Sort by missing count descending
        missing_data = missing_data.sort_values('Missing_Count', ascending=False)
        missing_data = missing_data.reset_index(drop=True)
        
        return missing_data
    
    def detect_outliers(self, df: pd.DataFrame, 
                       columns: Optional[List[str]] = None,
                       iqr_multiplier: float = 1.5) -> Dict[str, Dict]:
        """
        Detect outliers using IQR (Interquartile Range) method.
        
        Args:
            df: DataFrame to analyze
            columns: List of columns to check (if None, checks all numeric columns)
            iqr_multiplier: Multiplier for IQR (default 1.5 for standard outliers)
            
        Returns:
            Dict with outlier information for each column:
                - 'lower_bound': Lower threshold
                - 'upper_bound': Upper threshold
                - 'outlier_count': Number of outliers
                - 'outlier_indices': List of row indices with outliers
                - 'outlier_values': List of outlier values
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_report = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # Calculate IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Calculate bounds
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            
            # Find outliers
            outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_indices = df[outliers_mask].index.tolist()
            outlier_values = df.loc[outliers_mask, col].tolist()
            
            outlier_report[col] = {
                'lower_bound': round(lower_bound, 2),
                'upper_bound': round(upper_bound, 2),
                'outlier_count': len(outlier_indices),
                'outlier_indices': outlier_indices,
                'outlier_values': [round(v, 2) for v in outlier_values]
            }
        
        return outlier_report
    
    def get_summary_stats(self, df: pd.DataFrame, 
                         columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get summary statistics for numeric columns.
        
        Args:
            df: DataFrame to analyze
            columns: List of columns to include (if None, includes all numeric columns)
            
        Returns:
            pd.DataFrame: Summary statistics (describe() output transposed)
        """
        if columns is None:
            numeric_df = df.select_dtypes(include=[np.number])
        else:
            numeric_df = df[columns].select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return pd.DataFrame()
        
        # Get describe statistics and transpose for better readability
        stats = numeric_df.describe().T
        
        # Round to 2 decimal places
        stats = stats.round(2)
        
        # Add additional statistics
        stats['missing'] = numeric_df.isnull().sum()
        stats['missing_pct'] = (numeric_df.isnull().sum() / len(df) * 100).round(2)
        
        return stats
    
    def convert_date_column(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Convert date column to datetime format.
        
        Args:
            df: DataFrame containing the date column
            date_col: Name of the date column
            
        Returns:
            pd.DataFrame: DataFrame with converted date column
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        return df
    
    def get_date_range(self, df: pd.DataFrame, date_col: str) -> Tuple[str, str, int]:
        """
        Get date range information from the dataset.
        
        Args:
            df: DataFrame containing the date column
            date_col: Name of the date column
            
        Returns:
            Tuple of (start_date, end_date, num_periods)
        """
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        
        start_date = df_copy[date_col].min().strftime('%Y-%m-%d')
        end_date = df_copy[date_col].max().strftime('%Y-%m-%d')
        num_periods = len(df_copy)
        
        return start_date, end_date, num_periods
    
    # ========================================================================
    # MULTI-MARKET SUPPORT METHODS
    # ========================================================================
    
    def detect_data_format(self, df: pd.DataFrame) -> str:
        """
        Detect if data is single-market, multi-country, or multi-country-os format.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            str: 'single' | 'multi_country' | 'multi_country_os'
        """
        has_country = 'country' in df.columns
        has_os = 'os' in df.columns
        
        if has_country and has_os:
            return 'multi_country_os'
        elif has_country:
            return 'multi_country'
        else:
            return 'single'
    
    def validate_multi_market_data(self, df: pd.DataFrame, format_type: str = None) -> Dict:
        """
        Validate multi-market data structure.
        
        Args:
            df: DataFrame to validate
            format_type: Expected format ('multi_country' or 'multi_country_os')
                        If None, auto-detects
        
        Returns:
            Dict with validation results:
                - 'is_valid': bool
                - 'format': str
                - 'countries': list
                - 'os_platforms': list
                - 'market_segments': list
                - 'date_range': tuple
                - 'issues': list
        """
        if format_type is None:
            format_type = self.detect_data_format(df)
        
        issues = []
        countries = []
        os_platforms = []
        market_segments = []
        
        # Validate format-specific requirements
        if format_type == 'multi_country':
            if 'country' not in df.columns:
                issues.append("'country' column is required for multi-country format")
                return {'is_valid': False, 'format': format_type, 'issues': issues}
            
            countries = sorted(df['country'].unique().tolist())
            
            # Check for reasonable number of countries
            if len(countries) > 50:
                issues.append(f"Unusually high number of countries ({len(countries)}). Please verify data.")
            
            market_segments = countries
            
        elif format_type == 'multi_country_os':
            if 'country' not in df.columns:
                issues.append("'country' column is required")
            if 'os' not in df.columns:
                issues.append("'os' column is required")
                
            if issues:
                return {'is_valid': False, 'format': format_type, 'issues': issues}
            
            countries = sorted(df['country'].unique().tolist())
            os_platforms = sorted(df['os'].unique().tolist())
            
            # Generate market segments (country-os combinations)
            market_segments = self.get_market_segments(df)
            
            # Check for missing combinations
            expected_combinations = len(countries) * len(os_platforms)
            actual_combinations = len(market_segments)
            
            if actual_combinations < expected_combinations:
                missing = expected_combinations - actual_combinations
                issues.append(f"Missing {missing} market combinations. Some country-os pairs have no data.")
        
        # Check data balance across markets
        if format_type in ['multi_country', 'multi_country_os']:
            market_counts = df.groupby(
                ['country', 'os'] if format_type == 'multi_country_os' else ['country']
            ).size()
            
            min_count = market_counts.min()
            max_count = market_counts.max()
            
            if max_count > min_count * 10:
                issues.append(
                    f"Imbalanced data: largest market has {max_count} rows, "
                    f"smallest has {min_count} rows (10x difference)"
                )
        
        # Get date range
        date_range = None
        if 'date' in df.columns:
            try:
                df_copy = df.copy()
                df_copy['date'] = pd.to_datetime(df_copy['date'])
                date_range = (
                    df_copy['date'].min().strftime('%Y-%m-%d'),
                    df_copy['date'].max().strftime('%Y-%m-%d')
                )
            except:
                issues.append("Could not parse date column")
        
        is_valid = len([i for i in issues if 'required' in i.lower()]) == 0
        
        return {
            'is_valid': is_valid,
            'format': format_type,
            'countries': countries,
            'os_platforms': os_platforms,
            'market_segments': market_segments,
            'date_range': date_range,
            'issues': issues
        }
    
    def aggregate_data(
        self, 
        df: pd.DataFrame, 
        level: str,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Aggregate data to specified level.
        
        Args:
            df: DataFrame to aggregate
            level: Aggregation level:
                  - 'global': Sum across all markets
                  - 'country': Group by country
                  - 'os': Group by OS
                  - 'country_os': Keep country-os granularity
            filters: Optional dict to filter data before aggregation
                    e.g., {'country': ['KR', 'US'], 'os': ['iOS']}
        
        Returns:
            pd.DataFrame: Aggregated data
        """
        df_filtered = df.copy()
        
        # Apply filters if provided
        if filters:
            for col, values in filters.items():
                if col in df_filtered.columns:
                    if not isinstance(values, list):
                        values = [values]
                    df_filtered = df_filtered[df_filtered[col].isin(values)]
        
        # Determine grouping columns
        format_type = self.detect_data_format(df_filtered)
        
        if level == 'global':
            # Aggregate everything by date
            group_cols = ['date'] if 'date' in df_filtered.columns else []
            
        elif level == 'country':
            if 'country' not in df_filtered.columns:
                raise ValueError("Cannot aggregate by country: 'country' column not found")
            group_cols = ['date', 'country'] if 'date' in df_filtered.columns else ['country']
            
        elif level == 'os':
            if 'os' not in df_filtered.columns:
                raise ValueError("Cannot aggregate by OS: 'os' column not found")
            group_cols = ['date', 'os'] if 'date' in df_filtered.columns else ['os']
            
        elif level == 'country_os':
            if 'country' not in df_filtered.columns or 'os' not in df_filtered.columns:
                raise ValueError("Cannot aggregate by country_os: required columns not found")
            group_cols = ['date', 'country', 'os'] if 'date' in df_filtered.columns else ['country', 'os']
            
        else:
            raise ValueError(f"Invalid aggregation level: {level}")
        
        if not group_cols:
            # No grouping needed, return as is
            return df_filtered
        
        # Identify numeric columns to aggregate
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        
        # Aggregate
        agg_df = df_filtered.groupby(group_cols)[numeric_cols].sum().reset_index()
        
        return agg_df
    
    def get_market_segments(self, df: pd.DataFrame) -> List[str]:
        """
        Return list of unique market segments (country-os combinations).
        
        Args:
            df: DataFrame with market data
            
        Returns:
            List of market segment strings (e.g., ['KR-iOS', 'KR-Android', 'US-iOS'])
        """
        format_type = self.detect_data_format(df)
        
        if format_type == 'multi_country_os':
            # Create country-os combinations
            segments = df.groupby(['country', 'os']).size().reset_index()[['country', 'os']]
            market_segments = [
                f"{row['country']}-{row['os']}" 
                for _, row in segments.iterrows()
            ]
            return sorted(market_segments)
            
        elif format_type == 'multi_country':
            # Just return countries
            return sorted(df['country'].unique().tolist())
            
        else:
            # Single market
            return ['GLOBAL']
    
    def get_market_hierarchy(self, df: pd.DataFrame) -> Dict:
        """
        Return hierarchical structure of markets.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Dict with market hierarchy:
                {
                    'format': str,
                    'countries': list,
                    'os_platforms': list,
                    'hierarchy': dict  # Nested structure
                }
        """
        format_type = self.detect_data_format(df)
        
        hierarchy = {
            'format': format_type,
            'countries': [],
            'os_platforms': [],
            'hierarchy': {}
        }
        
        if format_type == 'multi_country_os':
            countries = sorted(df['country'].unique().tolist())
            os_platforms = sorted(df['os'].unique().tolist())
            
            hierarchy['countries'] = countries
            hierarchy['os_platforms'] = os_platforms
            
            # Build nested structure
            for country in countries:
                hierarchy['hierarchy'][country] = {}
                for os in os_platforms:
                    # Check if this combination exists
                    exists = len(df[(df['country'] == country) & (df['os'] == os)]) > 0
                    if exists:
                        hierarchy['hierarchy'][country][os] = {
                            'segment': f"{country}-{os}",
                            'row_count': len(df[(df['country'] == country) & (df['os'] == os)])
                        }
                        
        elif format_type == 'multi_country':
            countries = sorted(df['country'].unique().tolist())
            hierarchy['countries'] = countries
            
            for country in countries:
                hierarchy['hierarchy'][country] = {
                    'row_count': len(df[df['country'] == country])
                }
        
        return hierarchy
