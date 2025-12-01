"""
Prophet baseline modeling for Marketing Mix Modeling (MMM).

This module provides Prophet-based baseline estimation to separate organic growth
(trends, seasonality, holidays) from media-driven revenue. This improves media
attribution accuracy by removing confounding time-based effects.

Classes:
    ProphetBaseline: Prophet model for baseline revenue estimation
    ProphetMMM: Two-stage MMM (Prophet baseline + Ridge media model)

Example:
    >>> from prophet_baseline import ProphetMMM
    >>> model = ProphetMMM(ridge_alpha=1.0)
    >>> model.fit(df, 'date', 'revenue', media_channels, channel_configs)
    >>> decomp = model.decompose(df)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import warnings

# Suppress Prophet warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*iteritems.*')


class ProphetBaseline:
    """
    Use Facebook Prophet to model baseline revenue.

    Baseline = revenue that would occur without any paid media:
    - Long-term trend (growth/decline)
    - Seasonality (weekly, yearly)
    - Holidays and special events

    After removing baseline, remaining variance is attributed to media.

    Attributes:
        seasonality_mode (str): 'additive' or 'multiplicative'
        changepoint_prior_scale (float): Trend flexibility
        seasonality_prior_scale (float): Seasonality strength
        model: Fitted Prophet model
        is_fitted (bool): Whether model has been fitted

    Example:
        >>> baseline_model = ProphetBaseline()
        >>> baseline_model.fit(df, date_col='date', target_col='revenue')
        >>> baseline = baseline_model.predict(df)
        >>> media_driven = df['revenue'] - baseline
    """

    def __init__(
        self,
        seasonality_mode: str = 'multiplicative',
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True
    ):
        """
        Initialize Prophet baseline model.

        Args:
            seasonality_mode: 'additive' or 'multiplicative'
                - Additive: Seasonal effect constant over time
                - Multiplicative: Seasonal effect proportional to trend
            changepoint_prior_scale: Controls trend flexibility (0.001-0.5)
                - Lower = less flexible (smoother trend)
                - Higher = more flexible (follows variations)
            seasonality_prior_scale: Controls seasonality strength (1-100)
                - Lower = less seasonal variation
                - Higher = stronger seasonality
            yearly_seasonality: Whether to include yearly seasonality
            weekly_seasonality: Whether to include weekly seasonality

        Raises:
            ValueError: If parameters are invalid
        """
        if seasonality_mode not in ['additive', 'multiplicative']:
            raise ValueError(
                f"seasonality_mode must be 'additive' or 'multiplicative', "
                f"got {seasonality_mode}"
            )

        if not 0 < changepoint_prior_scale <= 1:
            raise ValueError(
                f"changepoint_prior_scale must be in (0, 1], "
                f"got {changepoint_prior_scale}"
            )

        if seasonality_prior_scale <= 0:
            raise ValueError(
                f"seasonality_prior_scale must be positive, "
                f"got {seasonality_prior_scale}"
            )

        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality

        self.model = None
        self.is_fitted = False
        self._date_col = None
        self._target_col = None

    def fit(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        target_col: str = 'revenue',
        holidays: Optional[pd.DataFrame] = None
    ) -> 'ProphetBaseline':
        """
        Fit Prophet model to baseline revenue.

        Args:
            df: DataFrame with date and revenue columns
            date_col: Name of date column
            target_col: Name of revenue/KPI column
            holidays: Optional DataFrame with holiday dates
                Format: columns=['holiday', 'ds', 'lower_window', 'upper_window']

        Returns:
            self (fitted model)

        Raises:
            ValueError: If required columns are missing or data is invalid
            ImportError: If Prophet is not installed

        Example:
            >>> baseline = ProphetBaseline()
            >>> baseline.fit(df, 'date', 'revenue')
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError(
                "Prophet is not installed. Install it with: pip install prophet"
            )

        # Validate inputs
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame")

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")

        if df[target_col].isna().any():
            raise ValueError(f"Target column '{target_col}' contains NaN values")

        # Prepare data in Prophet format (requires 'ds' and 'y')
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df[date_col]),
            'y': df[target_col].values
        })

        # Remove any rows with NaN dates
        prophet_df = prophet_df.dropna(subset=['ds'])

        if len(prophet_df) < 10:
            raise ValueError(
                f"Insufficient data for Prophet. Need at least 10 observations, "
                f"got {len(prophet_df)}"
            )

        # Initialize Prophet with parameters
        self.model = Prophet(
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=False,  # Never use daily for MMM
            holidays=holidays
        )

        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(prophet_df)

        self.is_fitted = True
        self._date_col = date_col
        self._target_col = target_col

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict baseline revenue for given dates.

        Args:
            df: DataFrame containing date column

        Returns:
            Array of baseline predictions

        Raises:
            ValueError: If model not fitted or date column missing

        Example:
            >>> baseline = baseline_model.predict(test_df)
        """
        if not self.is_fitted:
            raise ValueError(
                "Model not fitted. Call fit() before predict()."
            )

        if self._date_col not in df.columns:
            raise ValueError(
                f"Date column '{self._date_col}' not found in DataFrame"
            )

        # Prepare data
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df[self._date_col])
        })

        # Predict
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.model.predict(prophet_df)

        return forecast['yhat'].values

    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decompose revenue into baseline components.

        Returns DataFrame with:
        - trend: Long-term growth/decline
        - yearly: Yearly seasonal pattern (if enabled)
        - weekly: Weekly seasonal pattern (if enabled)
        - holidays: Holiday effects (if provided)
        - baseline: Total baseline (sum of above)
        - actual_revenue: Original revenue values
        - residual: Revenue - baseline (media-driven portion)

        Args:
            df: DataFrame with date and revenue data

        Returns:
            DataFrame with all decomposition components

        Raises:
            ValueError: If model not fitted

        Example:
            >>> decomp = baseline_model.decompose(df)
            >>> print(f"Baseline accounts for {decomp['baseline'].sum() / decomp['actual_revenue'].sum():.1%}")
        """
        if not self.is_fitted:
            raise ValueError(
                "Model not fitted. Call fit() before decompose()."
            )

        # Prepare data
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df[self._date_col])
        })

        # Get forecast with components
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.model.predict(prophet_df)

        # Build decomposition DataFrame
        result = pd.DataFrame({
            'date': df[self._date_col].values,
            'actual_revenue': df[self._target_col].values,
            'trend': forecast['trend'].values,
        })

        # Add seasonal components if they exist
        if 'yearly' in forecast.columns:
            result['yearly'] = forecast['yearly'].values
        else:
            result['yearly'] = 0.0

        if 'weekly' in forecast.columns:
            result['weekly'] = forecast['weekly'].values
        else:
            result['weekly'] = 0.0

        # Add holidays if present
        if 'holidays' in forecast.columns:
            result['holidays'] = forecast['holidays'].values
        else:
            result['holidays'] = 0.0

        # Calculate baseline (Prophet's yhat) and residual
        result['baseline'] = forecast['yhat'].values
        result['residual'] = result['actual_revenue'] - result['baseline']

        return result

    def get_baseline_stats(self) -> Dict[str, float]:
        """
        Get baseline statistics for model interpretation.

        Returns:
            Dictionary with baseline statistics:
            - baseline_mean: Mean baseline value
            - baseline_std: Standard deviation
            - trend_slope: Average trend direction
            - seasonality_strength: Strength of seasonal patterns

        Raises:
            ValueError: If model not fitted

        Example:
            >>> stats = baseline_model.get_baseline_stats()
            >>> print(f"Baseline mean: {stats['baseline_mean']:,.0f}")
        """
        if not self.is_fitted:
            raise ValueError(
                "Model not fitted. Call fit() before get_baseline_stats()."
            )

        # Get historical data statistics
        history = self.model.history

        stats = {
            'baseline_mean': float(history['y'].mean()),
            'baseline_std': float(history['y'].std()),
            'n_changepoints': len(self.model.changepoints),
            'yearly_seasonality_enabled': self.yearly_seasonality,
            'weekly_seasonality_enabled': self.weekly_seasonality,
        }

        return stats


class ProphetMMM:
    """
    Two-stage Marketing Mix Model with Prophet baseline.

    Stage 1: Use Prophet to model baseline (trend + seasonality + holidays)
    Stage 2: Use Ridge MMM on residuals to attribute media effects

    This approach separates:
    - Organic growth (captured by Prophet)
    - Media-driven growth (captured by Ridge MMM)

    Attributes:
        baseline_model (ProphetBaseline): Prophet model for baseline
        media_model (RidgeMMM): Ridge model for media effects
        is_fitted (bool): Whether both stages are fitted

    Example:
        >>> model = ProphetMMM(ridge_alpha=1.0)
        >>> model.fit(df, 'date', 'revenue', media_channels, channel_configs)
        >>> predictions = model.predict(df)
        >>> decomp = model.decompose(df)
    """

    def __init__(
        self,
        ridge_alpha: float = 1.0,
        prophet_seasonality_mode: str = 'multiplicative',
        prophet_changepoint_prior: float = 0.05
    ):
        """
        Initialize Prophet-enhanced MMM.

        Args:
            ridge_alpha: Regularization strength for Ridge regression
            prophet_seasonality_mode: Prophet seasonality mode
            prophet_changepoint_prior: Prophet trend flexibility

        Example:
            >>> model = ProphetMMM(ridge_alpha=1.0)
        """
        self.baseline_model = ProphetBaseline(
            seasonality_mode=prophet_seasonality_mode,
            changepoint_prior_scale=prophet_changepoint_prior
        )

        # Import RidgeMMM dynamically to avoid circular imports
        from ridge_mmm import RidgeMMM
        self.media_model = RidgeMMM(alpha=ridge_alpha)

        self.is_fitted = False
        self._date_col = None
        self._target_col = None
        self._media_channels = None

    def fit(
        self,
        df: pd.DataFrame,
        date_col: str,
        target_col: str,
        media_channels: List[str],
        channel_configs: Dict[str, Dict],
        holidays: Optional[pd.DataFrame] = None,
        exog_vars: Optional[List[str]] = None
    ) -> 'ProphetMMM':
        """
        Fit two-stage model.

        Stage 1: Prophet models baseline (trend + seasonality + holidays)
        Stage 2: Ridge MMM models media effects on residuals

        Args:
            df: DataFrame with all data
            date_col: Name of date column
            target_col: Name of revenue/KPI column
            media_channels: List of media channel column names
            channel_configs: Dictionary with adstock/Hill configs per channel
            holidays: Optional holiday DataFrame for Prophet
            exog_vars: Optional list of exogenous variable names

        Returns:
            self (fitted model)

        Raises:
            ValueError: If data is invalid or columns are missing

        Example:
            >>> model.fit(
            ...     df,
            ...     date_col='date',
            ...     target_col='revenue',
            ...     media_channels=['google', 'meta'],
            ...     channel_configs=configs
            ... )
        """
        # Validate inputs
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        for channel in media_channels:
            if channel not in df.columns:
                raise ValueError(f"Media channel '{channel}' not found")

        # Stage 1: Fit Prophet baseline
        print("Stage 1/2: Fitting Prophet baseline (trend + seasonality)...")
        self.baseline_model.fit(
            df,
            date_col=date_col,
            target_col=target_col,
            holidays=holidays
        )

        # Get baseline predictions
        baseline = self.baseline_model.predict(df)

        # Calculate residuals (media-driven portion)
        residuals = df[target_col].values - baseline

        # Stage 2: Fit Ridge MMM on residuals
        print("Stage 2/2: Fitting Ridge MMM on residuals (media effects)...")

        X_media = df[media_channels]

        # Add exogenous variables if provided
        if exog_vars:
            X_media = pd.concat([X_media, df[exog_vars]], axis=1)

        y_residuals = pd.Series(residuals, index=df.index)

        self.media_model.fit(
            X_media,
            y_residuals,
            channel_configs,
            exog_vars=exog_vars
        )

        self.is_fitted = True
        self._date_col = date_col
        self._target_col = target_col
        self._media_channels = media_channels

        print("✅ Two-stage model training complete!")

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict total revenue (baseline + media contributions).

        Args:
            df: DataFrame with date and media spend data

        Returns:
            Array of total revenue predictions

        Raises:
            ValueError: If model not fitted

        Example:
            >>> predictions = model.predict(test_df)
        """
        if not self.is_fitted:
            raise ValueError(
                "Model not fitted. Call fit() before predict()."
            )

        # Predict baseline
        baseline = self.baseline_model.predict(df)

        # Predict media contribution
        X_media = df[self._media_channels]
        media_contribution = self.media_model.predict(X_media)

        # Total = baseline + media
        return baseline + media_contribution

    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full decomposition: baseline components + individual media channels.

        Returns DataFrame with:
        - All Prophet baseline components (trend, seasonality, etc.)
        - Individual media channel contributions
        - Total predictions

        Args:
            df: DataFrame with all data

        Returns:
            DataFrame with complete decomposition

        Raises:
            ValueError: If model not fitted

        Example:
            >>> decomp = model.decompose(df)
            >>> print(decomp[['date', 'baseline', 'google', 'meta']].head())
        """
        if not self.is_fitted:
            raise ValueError(
                "Model not fitted. Call fit() before decompose()."
            )

        # Get baseline decomposition from Prophet
        baseline_decomp = self.baseline_model.decompose(df)

        # Get media contributions from Ridge MMM
        X_media = df[self._media_channels]
        media_contrib_df = self.media_model.get_contributions(X_media)

        # Add media channel columns to decomposition
        result = baseline_decomp.copy()

        for _, row in media_contrib_df.iterrows():
            channel = row['channel']
            if channel != 'base':  # Skip base contribution (already in baseline)
                # Distribute contribution across time periods
                result[channel] = row['contribution'] / len(df)

        # Add total prediction
        result['total_prediction'] = self.predict(df)

        return result

    def evaluate(
        self,
        df: pd.DataFrame,
        metrics: List[str] = ['r2', 'mape', 'mae', 'rmse']
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            df: DataFrame with actual values
            metrics: List of metrics to compute

        Returns:
            Dictionary with metric values

        Example:
            >>> metrics = model.evaluate(test_df)
            >>> print(f"R²: {metrics['r2']:.3f}")
        """
        if not self.is_fitted:
            raise ValueError(
                "Model not fitted. Call fit() before evaluate()."
            )

        y_true = df[self._target_col].values
        y_pred = self.predict(df)

        results = {}

        if 'r2' in metrics:
            from sklearn.metrics import r2_score
            results['r2'] = r2_score(y_true, y_pred)

        if 'mape' in metrics:
            from utils import calculate_mape
            results['mape'] = calculate_mape(y_true, y_pred)

        if 'mae' in metrics:
            from sklearn.metrics import mean_absolute_error
            results['mae'] = mean_absolute_error(y_true, y_pred)

        if 'rmse' in metrics:
            from sklearn.metrics import root_mean_squared_error
            results['rmse'] = root_mean_squared_error(y_true, y_pred)

        return results


def create_holiday_dataframe(
    country: str = 'US',
    years: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Create holiday DataFrame for Prophet.

    Supports major countries and automatically includes their national holidays.

    Args:
        country: Country code ('US', 'KR', 'JP', 'TW', 'GB', etc.)
        years: List of years to include (default: 2020-2026)

    Returns:
        DataFrame with columns ['holiday', 'ds', 'lower_window', 'upper_window']

    Raises:
        ImportError: If Prophet is not installed

    Example:
        >>> holidays = create_holiday_dataframe('US', [2023, 2024])
        >>> print(holidays.head())
    """
    try:
        from prophet.make_holidays import make_holidays_df
    except ImportError:
        raise ImportError(
            "Prophet is not installed. Install it with: pip install prophet"
        )

    if years is None:
        # Default: 2020-2026
        years = list(range(2020, 2027))

    holidays = make_holidays_df(
        year_list=years,
        country=country
    )

    return holidays
