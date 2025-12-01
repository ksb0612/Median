"""
Transformation functions for Marketing Mix Modeling (MMM).

This module provides transformation classes for applying adstock (carryover effects)
and saturation (diminishing returns) to marketing spend data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings


class AdstockTransformer:
    """
    Apply exponential adstock transformation to capture carryover effects.
    
    Adstock models the delayed and prolonged effect of advertising spend.
    The exponential decay model assumes that the effect of advertising
    diminishes exponentially over time.
    
    Attributes:
        decay_rate (float): Decay rate between 0 and 1. Higher values mean
                           longer carryover effects.
    
    Example:
        >>> transformer = AdstockTransformer(decay_rate=0.5)
        >>> x = np.array([1, 0, 0, 0, 0])
        >>> transformed = transformer.transform(x)
        >>> # Result: [1.0, 0.5, 0.25, 0.125, 0.0625]
    """
    
    def __init__(self, decay_rate: float = 0.5):
        """
        Initialize AdstockTransformer.
        
        Args:
            decay_rate: Decay rate between 0 and 1. Default is 0.5.
                       0 = no carryover, 1 = infinite carryover
        
        Raises:
            ValueError: If decay_rate is not between 0 and 1
        """
        if not 0 <= decay_rate <= 1:
            raise ValueError(f"decay_rate must be between 0 and 1, got {decay_rate}")
        
        self.decay_rate = decay_rate
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply exponential adstock transformation.
        
        The transformation follows the formula:
        adstock[t] = x[t] + decay_rate * adstock[t-1]
        
        Args:
            x: Input array of advertising spend values
        
        Returns:
            Transformed array with adstock effects applied
        
        Raises:
            ValueError: If input is not a numpy array or is empty
        
        Example:
            >>> transformer = AdstockTransformer(decay_rate=0.5)
            >>> x = np.array([100, 0, 0, 50, 0])
            >>> result = transformer.transform(x)
        """
        # Validate input
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        
        if len(x) == 0:
            raise ValueError("Input array cannot be empty")
        
        # Handle NaN values
        if np.any(np.isnan(x)):
            warnings.warn("Input contains NaN values. They will be treated as 0.")
            x = np.nan_to_num(x, nan=0.0)
        
        # Handle negative values (treat as 0 for adstock)
        if np.any(x < 0):
            warnings.warn("Input contains negative values. They will be treated as 0.")
            x = np.maximum(x, 0)
        
        # Apply adstock transformation
        adstock = np.zeros_like(x, dtype=float)
        adstock[0] = x[0]
        
        for t in range(1, len(x)):
            adstock[t] = x[t] + self.decay_rate * adstock[t - 1]
        
        return adstock
    
    def get_decay_curve(self, length: int = 52) -> np.ndarray:
        """
        Generate theoretical decay curve for visualization.
        
        Shows how a single unit of spend decays over time.
        
        Args:
            length: Number of time periods to generate (default: 52 weeks)
        
        Returns:
            Array showing decay pattern over time
        
        Example:
            >>> transformer = AdstockTransformer(decay_rate=0.7)
            >>> curve = transformer.get_decay_curve(length=10)
        """
        if length <= 0:
            raise ValueError("length must be positive")
        
        # Start with 1 unit of spend at t=0, then 0 for all other periods
        x = np.zeros(length)
        x[0] = 1.0
        
        return self.transform(x)


class WeibullAdstockTransformer:
    """
    Weibull adstock transformation for flexible decay patterns.

    Unlike geometric adstock (monotonic decay), Weibull can model:
    - Delayed peak effect (TV campaigns)
    - Quick rise then slow decay
    - Flexible decay rates

    The Weibull distribution is particularly useful for brand campaigns where
    awareness builds gradually after the initial spend, peaks after several weeks,
    then decays slowly.

    Attributes:
        shape (float): Shape parameter (k) controlling curve shape
                      - k < 1: Immediate peak, rapid decay
                      - k = 1: Exponential decay (similar to geometric)
                      - k > 1: Delayed peak, gradual decay
        scale (float): Scale parameter (lambda) controlling decay duration
                      Higher values = longer effect duration

    Example:
        >>> transformer = WeibullAdstockTransformer(shape=2.0, scale=3.0)
        >>> spend = np.array([100, 0, 0, 0, 0])
        >>> adstocked = transformer.transform(spend)
        >>> # Effect peaks at week 2-3, then decays
    """

    def __init__(self, shape: float = 2.0, scale: float = 3.0):
        """
        Initialize Weibull adstock transformer.

        Args:
            shape: Shape parameter (k) > 0. Default is 2.0.
            scale: Scale parameter (lambda) > 0. Default is 3.0.

        Raises:
            ValueError: If parameters are invalid
        """
        if shape <= 0:
            raise ValueError(f"Shape must be positive, got {shape}")
        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}")

        self.shape = shape
        self.scale = scale

        # Pre-compute normalization factor
        self._compute_normalization()

    def _compute_normalization(self, max_lag: int = 52):
        """
        Compute normalization factor so total effect = 1.

        This ensures that $100 spend eventually contributes $100 total effect.

        Args:
            max_lag: Maximum lag to consider (default: 52 weeks)
        """
        from scipy import stats

        lags = np.arange(max_lag)
        weights = stats.weibull_min.pdf(lags, c=self.shape, scale=self.scale)
        self.normalization = weights.sum() if weights.sum() > 0 else 1.0

    def _weibull_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Weibull probability density function.

        PDF(x) = (k/λ) * (x/λ)^(k-1) * exp(-(x/λ)^k)

        Args:
            x: Input values (lags)

        Returns:
            Weibull PDF values
        """
        from scipy import stats
        return stats.weibull_min.pdf(x, c=self.shape, scale=self.scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply Weibull adstock transformation.

        For each time t, effect is:
        adstock[t] = sum over all past lags of (spend[t-lag] * weight[lag])

        Args:
            x: Input spend array

        Returns:
            Adstocked array with delayed decay effects

        Raises:
            ValueError: If input is invalid

        Example:
            >>> transformer = WeibullAdstockTransformer(shape=2.5, scale=3.0)
            >>> spend = np.array([100, 0, 0, 0, 0, 0])
            >>> result = transformer.transform(spend)
            >>> # Peak effect occurs at week 2-3
        """
        # Validate input
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=float)
        else:
            x = np.array(x, dtype=float)

        if len(x) == 0:
            raise ValueError("Input array cannot be empty")

        # Handle NaN values
        if np.any(np.isnan(x)):
            warnings.warn("Input contains NaN values. They will be treated as 0.")
            x = np.nan_to_num(x, nan=0.0)

        # Handle negative values
        if np.any(x < 0):
            warnings.warn("Input contains negative values. They will be treated as 0.")
            x = np.maximum(x, 0)

        # Handle edge cases
        n = len(x)

        if n == 0:
            return x

        if np.all(x == 0):
            return np.zeros_like(x)

        # Initialize result
        adstocked = np.zeros(n, dtype=float)

        # Compute weights for maximum possible lag
        max_lag = min(n, 52)  # Look back up to 1 year
        lags = np.arange(max_lag)
        weights = self._weibull_pdf(lags) / self.normalization

        # Apply convolution with weights
        for t in range(n):
            for lag in range(min(t + 1, max_lag)):
                if t - lag >= 0:
                    adstocked[t] += x[t - lag] * weights[lag]

        return adstocked

    def get_decay_curve(self, length: int = 20) -> np.ndarray:
        """
        Get theoretical decay curve for visualization.

        Args:
            length: Number of time periods to show

        Returns:
            Array of decay weights

        Raises:
            ValueError: If length is invalid

        Example:
            >>> transformer = WeibullAdstockTransformer(shape=2.0, scale=3.0)
            >>> curve = transformer.get_decay_curve(length=15)
        """
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")

        lags = np.arange(length)
        weights = self._weibull_pdf(lags) / self.normalization
        return weights

    def get_peak_lag(self) -> float:
        """
        Calculate at which lag the effect peaks.

        Returns:
            Lag at maximum effect (e.g., 2.5 means peak at week 2-3)

        Example:
            >>> transformer = WeibullAdstockTransformer(shape=2.5, scale=4.0)
            >>> peak = transformer.get_peak_lag()
            >>> print(f"Effect peaks at week {peak:.1f}")
        """
        # For Weibull, mode (peak) occurs at:
        # mode = scale * ((shape - 1) / shape) ^ (1/shape) if shape >= 1
        if self.shape >= 1:
            mode = self.scale * ((self.shape - 1) / self.shape) ** (1 / self.shape)
            return mode
        else:
            return 0.0  # Peak at t=0 for shape < 1

    def get_half_life(self) -> float:
        """
        Calculate half-life (time to reach 50% of total effect).

        Returns:
            Half-life in time periods

        Example:
            >>> transformer = WeibullAdstockTransformer(shape=2.0, scale=3.0)
            >>> half_life = transformer.get_half_life()
        """
        from scipy import stats
        from scipy.optimize import brentq

        def cumulative_effect(x):
            return stats.weibull_min.cdf(x, c=self.shape, scale=self.scale) - 0.5

        try:
            half_life = brentq(cumulative_effect, 0, self.scale * 10)
            return half_life
        except ValueError:
            # Fallback approximation
            return self.scale


class AdstockType:
    """Enum for adstock types."""
    GEOMETRIC = "geometric"
    WEIBULL = "weibull"


class HillTransformer:
    """
    Apply Hill saturation function to model diminishing returns.
    
    The Hill equation models the saturation effect where increasing spend
    yields progressively smaller returns. This is common in advertising
    where initial spend is highly effective but additional spend has
    diminishing impact.
    
    Attributes:
        K (float): Scale parameter (half-saturation point)
        S (float): Shape parameter (controls curve steepness)
    
    Example:
        >>> transformer = HillTransformer(K=1.0, S=1.0)
        >>> x = np.array([0, 0.5, 1, 2, 5, 10])
        >>> transformed = transformer.transform(x)
    """
    
    def __init__(self, K: float = 1.0, S: float = 1.0):
        """
        Initialize HillTransformer.
        
        Args:
            K: Scale parameter (half-saturation point). Default is 1.0.
               Higher K means saturation occurs at higher spend levels.
            S: Shape parameter (curve steepness). Default is 1.0.
               Higher S means sharper transition to saturation.
        
        Raises:
            ValueError: If K or S are not positive
        """
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")
        if S <= 0:
            raise ValueError(f"S must be positive, got {S}")
        
        self.K = K
        self.S = S
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply Hill saturation transformation with numerical stability.

        The transformation follows the formula:
        y = K * (x^S) / (x^S + 1)

        This implementation includes comprehensive numerical stability improvements:
        - Clipping extreme values
        - Log-space computation for large exponents
        - Safe division with zero handling
        - NaN/Inf sanitization
        - Output range validation

        Args:
            x: Input array of values (typically adstocked spend)

        Returns:
            Transformed array with saturation effects applied

        Raises:
            ValueError: If input is empty or transformation produces invalid results

        Example:
            >>> transformer = HillTransformer(K=1.0, S=1.0)
            >>> x = np.array([0, 1, 2, 5, 10])
            >>> result = transformer.transform(x)
        """
        # Validate input
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=float)
        else:
            x = np.array(x, dtype=float)

        if len(x) == 0:
            raise ValueError("Input array cannot be empty")

        # Warn about negative values BEFORE clipping
        if np.any(x < 0):
            warnings.warn(
                "Received negative values in Hill transformation. Clipping to 0.",
                UserWarning
            )

        # 1. Clip extreme values to prevent overflow (including negatives to 0)
        x = np.clip(x, 0, 1e10)

        # 2. Handle all-zero case early
        if np.all(x == 0):
            return np.zeros_like(x)

        # Handle NaN values
        if np.any(np.isnan(x)):
            warnings.warn("Input contains NaN values. They will be treated as 0.")
            x = np.nan_to_num(x, nan=0.0)

        # 3. Compute with numerical stability
        try:
            # Use log-space for very large exponents to avoid overflow
            if self.S > 100 or np.max(x) > 1e6:
                # Compute in log space: log(x^S) = S * log(x)
                log_x_s = self.S * np.log(x + 1e-10)  # Add small epsilon to avoid log(0)
                # Clip to prevent exp overflow: exp(700) ≈ 1e304 is near float64 max
                log_x_s = np.clip(log_x_s, -700, 700)
                x_s = np.exp(log_x_s)
            else:
                # Standard computation for normal ranges
                x_s = np.power(x, self.S)

            # 4. Safe division with zero handling
            denominator = x_s + 1
            denominator = np.where(denominator == 0, 1e-10, denominator)
            result = self.K * x_s / denominator

            # 5. Handle NaN/Inf in result
            result = np.nan_to_num(result, nan=0.0, posinf=self.K, neginf=0.0)

            # 6. Sanity check: result should be in [0, K]
            if np.any(result < 0) or np.any(result > self.K * 1.01):  # Allow 1% tolerance
                raise ValueError(
                    f"Hill transformation produced out-of-range values. "
                    f"Expected [0, {self.K}], got [{result.min():.2f}, {result.max():.2f}]. "
                    f"Parameters: K={self.K}, S={self.S}, "
                    f"Input range: [{x.min():.2e}, {x.max():.2e}]"
                )

            return result

        except Exception as e:
            if isinstance(e, ValueError) and "out-of-range" in str(e):
                raise  # Re-raise our sanity check error
            raise ValueError(
                f"Hill transformation failed with K={self.K}, S={self.S}. "
                f"Input range: [{x.min():.2e}, {x.max():.2e}]. "
                f"Error: {str(e)}"
            )
    
    def get_response_curve(self, x_range: np.ndarray) -> np.ndarray:
        """
        Generate response curve for visualization.
        
        Args:
            x_range: Array of input values to evaluate
        
        Returns:
            Array of transformed values
        
        Example:
            >>> transformer = HillTransformer(K=1.0, S=1.0)
            >>> x_range = np.linspace(0, 10, 100)
            >>> curve = transformer.get_response_curve(x_range)
        """
        return self.transform(x_range)


class TransformationPipeline:
    """
    Pipeline to apply transformations to multiple channels.
    
    This class manages the application of adstock and saturation transformations
    to multiple marketing channels, with potentially different parameters for each.
    
    Attributes:
        channel_configs (dict): Configuration for each channel
    
    Example:
        >>> configs = {
        ...     'google': {'adstock': 0.5, 'hill_K': 1.0, 'hill_S': 1.0},
        ...     'facebook': {'adstock': 0.7, 'hill_K': 1.2, 'hill_S': 0.9}
        ... }
        >>> pipeline = TransformationPipeline(configs)
        >>> transformed_df = pipeline.fit_transform(df)
    """
    
    def __init__(self, channel_configs: Dict[str, Dict[str, float]]):
        """
        Initialize TransformationPipeline.

        Args:
            channel_configs: Dictionary mapping channel names to their configs.
                           Each config should have keys: 'adstock', 'hill_K', 'hill_S'
                           Optional: 'adstock_type' ('geometric' or 'weibull'),
                                    'weibull_shape', 'weibull_scale'

        Example (Geometric):
            >>> configs = {
            ...     'google': {'adstock': 0.5, 'hill_K': 1.0, 'hill_S': 1.0},
            ...     'facebook': {'adstock': 0.7, 'hill_K': 1.2, 'hill_S': 0.9}
            ... }
            >>> pipeline = TransformationPipeline(configs)

        Example (Weibull):
            >>> configs = {
            ...     'tv': {
            ...         'adstock_type': 'weibull',
            ...         'weibull_shape': 2.5,
            ...         'weibull_scale': 4.0,
            ...         'hill_K': 1.0,
            ...         'hill_S': 1.0
            ...     }
            ... }
            >>> pipeline = TransformationPipeline(configs)

        Raises:
            ValueError: If channel_configs is empty or invalid
        """
        if not channel_configs:
            raise ValueError("channel_configs cannot be empty")

        # Validate each channel config
        for channel, config in channel_configs.items():
            # Check adstock type
            adstock_type = config.get('adstock_type', 'geometric')

            if adstock_type == 'weibull':
                # Weibull requires shape and scale
                if 'weibull_shape' not in config or 'weibull_scale' not in config:
                    raise ValueError(
                        f"Channel '{channel}' using Weibull adstock must have "
                        f"'weibull_shape' and 'weibull_scale'. Got: {set(config.keys())}"
                    )
            else:
                # Geometric requires decay rate
                if 'adstock' not in config:
                    raise ValueError(
                        f"Channel '{channel}' using geometric adstock must have "
                        f"'adstock'. Got: {set(config.keys())}"
                    )

            # Hill parameters always required
            if 'hill_K' not in config or 'hill_S' not in config:
                raise ValueError(
                    f"Channel '{channel}' config missing Hill parameters. "
                    f"Required: 'hill_K', 'hill_S'. Got: {set(config.keys())}"
                )

        self.channel_configs = channel_configs
        self._transformers = {}
        self._fitted_channels = None  # Track which channels were fitted

        # Initialize transformers for each channel
        for channel, config in channel_configs.items():
            # Determine adstock type and create appropriate transformer
            adstock_type = config.get('adstock_type', 'geometric')

            if adstock_type == 'weibull':
                shape = config['weibull_shape']
                scale = config['weibull_scale']
                adstock_transformer = WeibullAdstockTransformer(
                    shape=shape,
                    scale=scale
                )
            else:  # geometric (default)
                decay = config['adstock']
                adstock_transformer = AdstockTransformer(decay_rate=decay)

            self._transformers[channel] = {
                'adstock': adstock_transformer,
                'adstock_type': adstock_type,
                'hill': HillTransformer(K=config['hill_K'], S=config['hill_S'])
            }
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit transformers and apply them to the DataFrame.

        This method should be called on training data. It validates the data,
        stores which channels were fitted, and applies transformations.

        Transformations are applied in order:
        1. Adstock transformation (carryover effects)
        2. Hill transformation (saturation effects)

        Args:
            df: DataFrame containing media spend columns

        Returns:
            Transformed DataFrame with same column names

        Raises:
            ValueError: If required columns are missing from DataFrame

        Example:
            >>> df = pd.DataFrame({
            ...     'google': [100, 200, 150],
            ...     'facebook': [50, 75, 60]
            ... })
            >>> transformed_df = pipeline.fit_transform(df)
        """
        # Validate input
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        if df.empty:
            raise ValueError("Input DataFrame cannot be empty")

        # Check that all configured channels exist in DataFrame
        missing_channels = set(self.channel_configs.keys()) - set(df.columns)
        if missing_channels:
            raise ValueError(
                f"Channels {missing_channels} not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        # Store fitted channels (list to preserve order)
        self._fitted_channels = list(self.channel_configs.keys())

        # Create copy to avoid modifying original
        df_transformed = df.copy()

        # Apply transformations to each configured channel
        for channel in self._fitted_channels:
            # Get original values
            original_values = df[channel].values

            # Apply adstock first
            adstocked = self._transformers[channel]['adstock'].transform(original_values)

            # Then apply Hill saturation
            transformed = self._transformers[channel]['hill'].transform(adstocked)

            # Update DataFrame
            df_transformed[channel] = transformed

        return df_transformed

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted transformations to data (without re-fitting).

        This is used for test/production data where we want to apply
        the same transformations that were fit on training data.

        IMPORTANT: This method must be called after fit_transform().
        It reuses the same transformer instances to ensure consistency
        and prevent data leakage.

        Args:
            df: DataFrame with media channels

        Returns:
            Transformed DataFrame

        Raises:
            ValueError: If transform called before fit_transform, or if
                       required channels are missing from data

        Example:
            >>> # Fit on training data
            >>> train_transformed = pipeline.fit_transform(train_df)
            >>> # Transform test data with same transformers
            >>> test_transformed = pipeline.transform(test_df)
        """
        # Check if pipeline has been fitted
        if self._fitted_channels is None:
            raise ValueError(
                "Pipeline not fitted. Call fit_transform() on training data first."
            )

        # Validate input
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        if df.empty:
            raise ValueError("Input DataFrame cannot be empty")

        # Check that all fitted channels exist in DataFrame
        missing_channels = set(self._fitted_channels) - set(df.columns)
        if missing_channels:
            raise ValueError(
                f"Channels {missing_channels} not found in DataFrame. "
                f"Available columns: {list(df.columns)}. "
                f"Expected channels from fit_transform: {self._fitted_channels}"
            )

        # Create copy to avoid modifying original
        df_transformed = df.copy()

        # Apply transformations using fitted transformers
        for channel in self._fitted_channels:
            # Get original values
            original_values = df[channel].values

            # Apply adstock first (using fitted transformer)
            adstocked = self._transformers[channel]['adstock'].transform(original_values)

            # Then apply Hill saturation (using fitted transformer)
            transformed = self._transformers[channel]['hill'].transform(adstocked)

            # Update DataFrame
            df_transformed[channel] = transformed

        return df_transformed
    
    def get_transformation_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary of applied transformations.
        
        Returns:
            Dictionary with transformation parameters for each channel
        
        Example:
            >>> summary = pipeline.get_transformation_summary()
            >>> print(summary['google'])
            {'adstock': 0.5, 'hill_K': 1.0, 'hill_S': 1.0}
        """
        return self.channel_configs.copy()
    
    def transform_single_channel(
        self, 
        channel: str, 
        values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform a single channel and return intermediate results.
        
        Useful for visualization and debugging.
        
        Args:
            channel: Channel name
            values: Original values to transform
        
        Returns:
            Tuple of (original, adstocked, final_transformed)
        
        Raises:
            ValueError: If channel not in configuration
        """
        if channel not in self.channel_configs:
            raise ValueError(
                f"Channel '{channel}' not in configuration. "
                f"Available: {list(self.channel_configs.keys())}"
            )
        
        # Apply transformations
        adstocked = self._transformers[channel]['adstock'].transform(values)
        transformed = self._transformers[channel]['hill'].transform(adstocked)
        
        return values, adstocked, transformed
