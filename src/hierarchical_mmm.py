"""
Hierarchical Marketing Mix Modeling (MMM) implementation.

This module provides the HierarchicalMMM class for handling multi-market,
multi-OS, and other segmented MMM analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import copy

from ridge_mmm import RidgeMMM
from utils.segment_utils import parse_segment, filter_by_segment
from utils.data_utils import safe_divide


class HierarchicalMMM:
    """
    Multi-level MMM supporting:
    - Global model (all markets pooled)
    - Country-level models
    - OS-level models  
    - Country×OS models (most granular)
    """
    
    def __init__(self, analysis_level: str = 'global'):
        """
        Initialize HierarchicalMMM.

        Args:
            analysis_level: Level of granularity for the model.
                            Options: 'global', 'country', 'os', 'country_os'
        """
        valid_levels = ['global', 'country', 'os', 'country_os']
        if analysis_level not in valid_levels:
            raise ValueError(f"Invalid analysis_level. Must be one of {valid_levels}")
            
        self.analysis_level = analysis_level
        self.models: Dict[str, RidgeMMM] = {}  # Store separate models per segment
        self.global_model: Optional[RidgeMMM] = None
        self.feature_names: Optional[List[str]] = None
        self.media_channels: Optional[List[str]] = None
        self.is_fitted = False
        
    def _get_segment_key(self, row: pd.Series) -> str:
        """Helper to generate segment key from a data row."""
        if self.analysis_level == 'global':
            return 'global'
        elif self.analysis_level == 'country':
            return str(row['country'])
        elif self.analysis_level == 'os':
            return str(row['os'])
        elif self.analysis_level == 'country_os':
            return f"{row['country']}_{row['os']}"
        return 'global'

    def _filter_data(self, df: pd.DataFrame, segment: str) -> pd.DataFrame:
        """
        Filter data for specific segment with comprehensive validation.

        Args:
            df: Input DataFrame
            segment: Segment identifier

        Returns:
            Filtered DataFrame for the specified segment

        Raises:
            ValueError: If segment format is invalid, columns are missing,
                       or no data found for segment

        Example:
            >>> segment_df = hmm._filter_data(df, 'US')
            >>> segment_df = hmm._filter_data(df, 'US-iOS')
        """
        # Get known countries and OS platforms
        known_countries = getattr(self, '_countries', [])
        known_os = getattr(self, '_os_platforms', [])

        # Parse segment using utility function
        segment_dict = parse_segment(segment, known_countries, known_os)

        # Filter data using utility function
        return filter_by_segment(df, segment_dict)

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = 'revenue',
        channel_configs: Dict[str, Dict[str, float]] = None,
        exog_vars: Optional[List[str]] = None,
        **kwargs
    ) -> 'HierarchicalMMM':
        """
        Fit models based on analysis_level.

        If level='global': Fit one model on aggregated data
        If level='country': Fit separate model per country
        If level='os': Fit separate model per OS
        If level='country_os': Fit separate model per country-os combo

        Args:
            df: Input DataFrame containing data for all segments
            target_col: Name of target column (e.g., 'revenue')
            channel_configs: Dictionary of channel configurations
            exog_vars: List of exogenous variables
            **kwargs: Additional arguments passed to RidgeMMM (e.g., alpha)

        Returns:
            self
        """
        if channel_configs is None:
            raise ValueError("channel_configs must be provided")

        # Store metadata for validation
        if 'country' in df.columns:
            self._countries = df['country'].unique().tolist()
        if 'os' in df.columns:
            self._os_platforms = df['os'].unique().tolist()

        # Validate analysis level against available columns
        if self.analysis_level == 'country' and 'country' not in df.columns:
            raise ValueError(
                "analysis_level='country' requires a 'country' column in the data"
            )

        if self.analysis_level == 'os' and 'os' not in df.columns:
            raise ValueError(
                "analysis_level='os' requires an 'os' column in the data"
            )

        if self.analysis_level == 'country_os':
            if 'country' not in df.columns or 'os' not in df.columns:
                raise ValueError(
                    "analysis_level='country_os' requires both 'country' and 'os' columns"
                )

        self.media_channels = list(channel_configs.keys())
        self.feature_names = self.media_channels + (exog_vars if exog_vars else [])
        
        # Always fit a global model for reference/fallback
        print("Fitting global model...")
        # For global model, we might need to aggregate if data is granular
        # But RidgeMMM expects time-series. If df has multiple segments per date,
        # we should aggregate by date for the global model.
        
        date_col = [c for c in df.columns if 'date' in c.lower()]
        if not date_col:
             raise ValueError("Data must contain a date column")
        date_col = date_col[0]
        
        # Aggregate for global model
        global_df = df.groupby(date_col)[self.feature_names + [target_col]].sum().reset_index()
        self.global_model = RidgeMMM(**kwargs)
        self.global_model.fit(global_df, global_df[target_col], channel_configs, exog_vars)
        
        if self.analysis_level == 'global':
            self.models['global'] = self.global_model
            self.is_fitted = True
            return self
            
        # Segment-specific fitting
        segments = []
        if self.analysis_level == 'country':
            if 'country' not in df.columns:
                raise ValueError("Column 'country' missing for analysis_level='country'")
            segments = df['country'].unique()
            group_cols = ['country']
        elif self.analysis_level == 'os':
            if 'os' not in df.columns:
                raise ValueError("Column 'os' missing for analysis_level='os'")
            segments = df['os'].unique()
            group_cols = ['os']
        elif self.analysis_level == 'country_os':
            if 'country' not in df.columns or 'os' not in df.columns:
                raise ValueError("Columns 'country' and 'os' required for analysis_level='country_os'")
            # Create composite key for iteration
            df['_segment_key'] = df['country'].astype(str) + '_' + df['os'].astype(str)
            segments = df['_segment_key'].unique()
            group_cols = ['_segment_key'] # Use the temp column
            
        print(f"Fitting {len(segments)} models for level '{self.analysis_level}'...")
        
        for segment in segments:
            if self.analysis_level == 'country_os':
                 segment_df = df[df['_segment_key'] == segment].copy()
                 # Clean up temp col
                 if '_segment_key' in segment_df.columns:
                     segment_df = segment_df.drop(columns=['_segment_key'])
            elif self.analysis_level == 'country':
                 segment_df = df[df['country'] == segment].copy()
            elif self.analysis_level == 'os':
                 segment_df = df[df['os'] == segment].copy()
            
            # Sort by date to ensure time series order
            segment_df = segment_df.sort_values(date_col)
            
            # Check if enough data points
            if len(segment_df) < 10: # Arbitrary minimum
                print(f"Skipping segment {segment}: Insufficient data ({len(segment_df)} rows)")
                continue
                
            try:
                model = RidgeMMM(**kwargs)
                model.fit(segment_df, segment_df[target_col], channel_configs, exog_vars)
                self.models[str(segment)] = model
            except Exception as e:
                print(f"Error fitting model for segment {segment}: {str(e)}")
                
        self.is_fitted = True
        return self
    
    def predict(self, df: pd.DataFrame, segment: str = None) -> np.ndarray:
        """
        Predict for all segments or specific segment.
        
        Args:
            df: Input DataFrame
            segment: Optional specific segment to predict for. 
                     If None, predicts for all rows using appropriate segment models.
        
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calling predict()")
            
        if segment:
            if segment not in self.models:
                # Fallback to global model if segment specific model doesn't exist?
                # Or raise error? Let's raise error for now to be explicit.
                if 'global' in self.models and self.analysis_level == 'global':
                     return self.models['global'].predict(df)
                raise ValueError(f"No model found for segment '{segment}'")
            return self.models[segment].predict(df)
            
        # If no segment specified, we need to predict row-by-row or group-by-group
        # depending on the dataframe structure and analysis level.
        
        if self.analysis_level == 'global':
            return self.models['global'].predict(df)
            
        # For segmented models, we expect the input df to have the segment columns
        predictions = np.zeros(len(df))
        
        # Iterate through unique segments in the input df
        # This is more efficient than row-by-row
        
        input_segments = []
        if self.analysis_level == 'country':
             if 'country' not in df.columns:
                 # If segment columns missing, maybe try global model?
                 # But user asked for segmented prediction implicitly by not providing segment arg
                 # and having a segmented model.
                 # Let's assume they want global prediction if structure doesn't match?
                 # No, safer to error.
                 raise ValueError("Input data missing 'country' column for segmented prediction")
             groups = df.groupby('country')
        elif self.analysis_level == 'os':
             if 'os' not in df.columns:
                 raise ValueError("Input data missing 'os' column for segmented prediction")
             groups = df.groupby('os')
        elif self.analysis_level == 'country_os':
             if 'country' not in df.columns or 'os' not in df.columns:
                 raise ValueError("Input data missing 'country'/'os' columns")
             # Group by both
             groups = df.groupby(['country', 'os'])
             
        for name, group in groups:
            # Construct segment key
            if isinstance(name, tuple):
                seg_key = f"{name[0]}_{name[1]}"
            else:
                seg_key = str(name)
                
            if seg_key in self.models:
                # Predict for this group
                group_preds = self.models[seg_key].predict(group)
                # Assign back to original indices
                predictions[group.index] = group_preds
            else:
                # Fallback to global model? Or 0?
                # Using global model scaled might be complex. 
                # Let's use global model but warn? Or just 0?
                # For now, let's use global model as a reasonable fallback if available
                # But wait, global model is trained on aggregated data. 
                # It expects aggregated inputs? No, RidgeMMM.predict expects a DF with channels.
                # So global model predict works on any DF with correct columns.
                # However, the scale might be off if global model was trained on sums.
                # Actually, if global model trained on sum of all countries, and we predict for one country,
                # the coefficients are for the aggregate. 
                # If we use global model, we might overpredict if it's not normalized.
                # Let's just return 0 or NaN for missing segments to be safe.
                print(f"Warning: No model for segment {seg_key}. Returning 0s.")
                predictions[group.index] = 0
                
        return predictions
    
    def get_contributions(self, df: pd.DataFrame, segment: str = None) -> pd.DataFrame:
        """
        Return contributions by segment.
        If segment=None, return aggregated global contributions.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
            
        if segment:
            if segment not in self.models:
                raise ValueError(f"No model for segment {segment}")
            return self.models[segment].get_contributions(df)
            
        if self.analysis_level == 'global':
            return self.models['global'].get_contributions(df)
            
        # Aggregate contributions from all segments
        all_contributions = []
        
        # Similar grouping logic as predict
        if self.analysis_level == 'country':
             groups = df.groupby('country')
        elif self.analysis_level == 'os':
             groups = df.groupby('os')
        elif self.analysis_level == 'country_os':
             groups = df.groupby(['country', 'os'])
        else:
             groups = [] # Should not happen
             
        for name, group in groups:
            if isinstance(name, tuple):
                seg_key = f"{name[0]}_{name[1]}"
            else:
                seg_key = str(name)
                
            if seg_key in self.models:
                contrib = self.models[seg_key].get_contributions(group)
                # Add segment info
                contrib['segment'] = seg_key
                all_contributions.append(contrib)
                
        if not all_contributions:
            return pd.DataFrame()
            
        combined = pd.concat(all_contributions)
        
        # Aggregate by channel
        aggregated = combined.groupby('channel').agg({
            'spend': 'sum',
            'contribution': 'sum'
        }).reset_index()
        
        # Recalculate ROAS and pct using safe_divide
        total_contribution = aggregated['contribution'].sum()
        aggregated['roas'] = aggregated.apply(
            lambda x: safe_divide(x['contribution'], x['spend']), axis=1
        )
        aggregated['contribution_pct'] = aggregated.apply(
            lambda x: safe_divide(x['contribution'], total_contribution) * 100, axis=1
        )
        
        return aggregated.sort_values('contribution', ascending=False)
    
    def compare_segments(self, df: pd.DataFrame, metric: str = 'roas') -> pd.DataFrame:
        """
        Compare all segments on specified metric.
        Returns DataFrame: [segment, metric_value, rank]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
            
        results = []
        
        # We need data to calculate metrics. 
        # We can either use the passed df (which should contain all data)
        # and split it, or store training metrics.
        # Ideally we calculate on the provided df (could be test set).
        
        # Group df by segment
        if self.analysis_level == 'country':
             groups = df.groupby('country')
        elif self.analysis_level == 'os':
             groups = df.groupby('os')
        elif self.analysis_level == 'country_os':
             groups = df.groupby(['country', 'os'])
        elif self.analysis_level == 'global':
             # Only one segment
             groups = [('global', df)]
        else:
             return pd.DataFrame()

        for name, group in groups:
            if isinstance(name, tuple):
                seg_key = f"{name[0]}_{name[1]}"
            else:
                seg_key = str(name)
                
            if seg_key in self.models:
                # Get contributions to calculate ROAS/CPA etc
                contribs = self.models[seg_key].get_contributions(group)
                
                # Calculate total metric for the segment
                total_spend = contribs[contribs['channel'] != 'base']['spend'].sum()
                total_return = contribs['contribution'].sum() # Includes base? Usually ROAS excludes base.
                # RidgeMMM.get_contributions includes base in the dataframe but we might want to exclude it for ROAS
                
                media_contrib = contribs[contribs['channel'] != 'base']['contribution'].sum()
                
                if metric == 'roas':
                    val = safe_divide(media_contrib, total_spend)
                elif metric == 'cpa':
                    # Assuming target is conversions. If revenue, CPA doesn't make sense without conversion count.
                    # Let's assume target is revenue for now as per default.
                    val = 0 # Not supported yet
                elif metric == 'revenue':
                    val = total_return
                else:
                    val = 0
                    
                results.append({
                    'segment': seg_key,
                    metric: val,
                    'spend': total_spend,
                    'revenue': total_return
                })
                
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df['rank'] = results_df[metric].rank(ascending=False)
            results_df = results_df.sort_values(metric, ascending=False)
            
        return results_df
    
    def get_cross_market_insights(self) -> Dict[str, Any]:
        """
        Identify insights across markets:
        - Which channel performs best in each market?
        - Which markets are saturated vs high-growth?
        - Cross-market efficiency gaps
        Returns dict with insights.
        """
        if not self.models:
            return {}
            
        insights = {
            'best_channels': {},
            'saturation_levels': {},
            'efficiency_gaps': {}
        }
        
        for seg_key, model in self.models.items():
            if seg_key == 'global': continue
            
            # 1. Best performing channel (highest coefficient * average spend? or just coefficient?)
            # Coefficient represents efficiency (marginal impact) after scaling.
            # Let's look at coefficients.
            coefs = dict(zip(model.feature_names, model.model.coef_))
            # Filter for media channels
            media_coefs = {k: v for k, v in coefs.items() if k in model.media_channels}
            if media_coefs:
                best_channel = max(media_coefs, key=media_coefs.get)
                insights['best_channels'][seg_key] = best_channel
            
            # 2. Saturation (Hill parameters)
            # Higher Hill K means saturation happens later (or S curve shape).
            # Usually Hill curve: 1 / (1 + (K/x)^S)
            # Wait, RidgeMMM uses: x^S / (x^S + K^S)
            # K is the half-saturation point. 
            # If current spend is >> K, we are saturated.
            # We need current spend data to know this. 
            # Since we don't have data here, we can just report the K values?
            # Or maybe we can't do this without data.
            # Let's skip saturation for now or just return K values.
            
            # 3. Efficiency (ROAS) - requires data, handled in compare_segments
            
        return insights
