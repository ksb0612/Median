"""
Budget optimization for Marketing Mix Modeling.

This module provides the BudgetOptimizer class for finding optimal
marketing budget allocation across channels.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize, OptimizeResult
import warnings

from ridge_mmm import RidgeMMM


class OptimizationError(Exception):
    """Custom exception for optimization failures."""
    pass


class BudgetOptimizer:
    """
    Optimize marketing budget allocation across channels.
    
    Uses scipy optimization to find the allocation that maximizes
    predicted revenue given constraints.
    
    Attributes:
        model (RidgeMMM): Fitted MMM model
        X_base (pd.DataFrame): Baseline data for predictions
        channels (list): List of channel names
    
    Example:
        >>> optimizer = BudgetOptimizer(mmm, X_train)
        >>> result = optimizer.optimize_budget(total_budget=100000, channels=['google', 'facebook'])
        >>> print(result['optimal_allocation'])
    """
    
    def __init__(self, model: RidgeMMM, X_base: pd.DataFrame):
        """
        Initialize BudgetOptimizer.
        
        Args:
            model: Fitted RidgeMMM model
            X_base: Baseline DataFrame for prediction context
        
        Raises:
            ValueError: If model is not fitted
        """
        if not model.is_fitted:
            raise ValueError("Model must be fitted before optimization")
        
        self.model = model
        self.X_base = X_base
        self.channels = model.media_channels
    
    def optimize_budget(
        self,
        total_budget: float,
        channels: List[str],
        constraints: Optional[Dict[str, Dict[str, float]]] = None,
        target_metric: str = 'revenue'
    ) -> Dict:
        """
        Find optimal budget allocation across channels with comprehensive validation.

        Uses scipy.optimize.minimize with SLSQP method to maximize
        predicted revenue subject to budget and channel constraints.

        Args:
            total_budget: Total budget to allocate
            channels: List of channels to optimize
            constraints: Optional dict of channel constraints:
                        {channel: {'min': min_spend, 'max': max_spend}}
            target_metric: Metric to optimize ('revenue' or 'roas')

        Returns:
            Dictionary with:
            - 'optimal_allocation': Dict mapping channels to optimal spend
            - 'predicted_revenue': Predicted revenue with optimal allocation
            - 'predicted_roas': Predicted ROAS with optimal allocation
            - 'improvement_pct': Improvement vs current allocation
            - 'current_allocation': Current spend per channel
            - 'current_revenue': Revenue with current allocation
            - 'success': Whether optimization succeeded
            - 'message': Status message

        Raises:
            ValueError: If inputs are invalid
            OptimizationError: If optimization fails

        Example:
            >>> result = optimizer.optimize_budget(
            ...     total_budget=100000,
            ...     channels=['google', 'facebook'],
            ...     constraints={'google': {'min': 20000, 'max': 60000}}
            ... )
        """
        # 1. Basic validation
        if total_budget <= 0:
            raise ValueError(f"Total budget must be positive, got {total_budget}")

        if not channels:
            raise ValueError("Channels list cannot be empty")

        if len(channels) < 2:
            raise ValueError(f"Need at least 2 channels to optimize, got {len(channels)}")

        if target_metric not in ['revenue', 'roas']:
            raise ValueError(f"target_metric must be 'revenue' or 'roas', got '{target_metric}'")

        # 2. Validate channels exist in model
        if hasattr(self.model, 'media_channels'):
            model_channels = self.model.media_channels
            invalid = [ch for ch in channels if ch not in model_channels]
            if invalid:
                raise ValueError(
                    f"Channels not in model: {invalid}. "
                    f"Available: {model_channels}"
                )
        else:
            # Fallback validation
            for channel in channels:
                if channel not in self.channels:
                    raise ValueError(f"Channel '{channel}' not in model")

        # 3. Validate and normalize constraints
        if constraints is None:
            constraints = {}

        validated_constraints = {}
        total_min = 0

        for channel in channels:
            if channel in constraints:
                constraint = constraints[channel]

                # Handle both dict formats: {'min': x, 'max': y} or tuple (min, max)
                if isinstance(constraint, dict):
                    min_val = constraint.get('min', 0)
                    max_val = constraint.get('max', total_budget)
                elif isinstance(constraint, (tuple, list)) and len(constraint) == 2:
                    min_val, max_val = constraint
                else:
                    raise ValueError(
                        f"Invalid constraint format for {channel}. "
                        f"Expected dict with 'min'/'max' or tuple (min, max)"
                    )

                if min_val < 0:
                    raise ValueError(f"Min for {channel} cannot be negative: {min_val}")

                if max_val < min_val:
                    raise ValueError(
                        f"Max ({max_val}) < min ({min_val}) for {channel}"
                    )

                if min_val > total_budget:
                    raise ValueError(
                        f"Min for {channel} ({min_val:,.0f}) "
                        f"exceeds total budget ({total_budget:,.0f})"
                    )

                validated_constraints[channel] = (min_val, max_val)
                total_min += min_val
            else:
                validated_constraints[channel] = (0, total_budget)

        # 4. Check feasibility
        if total_min > total_budget:
            raise ValueError(
                f"Sum of minimum constraints ({total_min:,.0f}) "
                f"exceeds total budget ({total_budget:,.0f}). "
                f"Optimization is infeasible."
            )
        
        # Get current allocation
        current_allocation = {ch: self.X_base[ch].mean() for ch in channels}
        current_total = sum(current_allocation.values())
        
        # Predict current revenue
        current_revenue = self._predict_revenue(current_allocation, channels)
        current_roas = current_revenue / current_total if current_total > 0 else 0
        
        # Set up optimization
        n_channels = len(channels)
        
        # Initial guess: proportional to current allocation or equal split
        if current_total > 0:
            x0 = np.array([current_allocation[ch] / current_total * total_budget for ch in channels])
        else:
            x0 = np.array([total_budget / n_channels] * n_channels)
        
        # Define objective function (negative revenue for minimization)
        def objective(x):
            allocation = {ch: spend for ch, spend in zip(channels, x)}
            revenue = self._predict_revenue(allocation, channels)
            
            if target_metric == 'revenue':
                return -revenue  # Minimize negative revenue = maximize revenue
            elif target_metric == 'roas':
                total_spend = sum(x)
                roas = revenue / total_spend if total_spend > 0 else 0
                return -roas
            else:
                return -revenue
        
        # Set up constraints
        scipy_constraints = []
        
        # Budget constraint: sum of allocations = total_budget
        scipy_constraints.append({
            'type': 'eq',
            'fun': lambda x: np.sum(x) - total_budget
        })
        
        # Set up bounds for each channel using validated constraints
        bounds = []
        for channel in channels:
            min_spend, max_spend = validated_constraints[channel]
            bounds.append((min_spend, max_spend))

        # 5. Run optimization
        try:
            result: OptimizeResult = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=scipy_constraints,
                options={'maxiter': 1000, 'ftol': 1e-6}
            )

            # 6. Validate result
            if result.success:
                # Extract optimal allocation
                optimal_allocation = {ch: spend for ch, spend in zip(channels, result.x)}

                # Validate allocation
                if optimal_allocation is None:
                    raise OptimizationError("Optimization returned None")

                allocated = sum(optimal_allocation.values())
                if not np.isclose(allocated, total_budget, rtol=0.01):
                    raise OptimizationError(
                        f"Allocated budget ({allocated:,.0f}) does not match "
                        f"total budget ({total_budget:,.0f})"
                    )

                optimal_revenue = self._predict_revenue(optimal_allocation, channels)
                optimal_roas = optimal_revenue / total_budget if total_budget > 0 else 0

                # Calculate improvement
                improvement_pct = ((optimal_revenue - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0

                return {
                    'optimal_allocation': optimal_allocation,
                    'predicted_revenue': optimal_revenue,
                    'predicted_roas': optimal_roas,
                    'improvement_pct': improvement_pct,
                    'current_allocation': current_allocation,
                    'current_revenue': current_revenue,
                    'current_roas': current_roas,
                    'success': True,
                    'message': 'Optimization successful'
                }
            else:
                return {
                    'optimal_allocation': current_allocation,
                    'predicted_revenue': current_revenue,
                    'predicted_roas': current_roas,
                    'improvement_pct': 0,
                    'current_allocation': current_allocation,
                    'current_revenue': current_revenue,
                    'current_roas': current_roas,
                    'success': False,
                    'message': f'Optimization failed: {result.message}'
                }

        except OptimizationError as e:
            # Re-raise optimization errors
            raise

        except Exception as e:
            raise OptimizationError(f"Optimization failed: {str(e)}")
    
    def compare_scenarios(self, scenarios: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare multiple budget allocation scenarios.
        
        Args:
            scenarios: Dict mapping scenario names to channel allocations
                      {'Scenario 1': {'google': 10000, 'facebook': 5000}}
        
        Returns:
            DataFrame with comparison of all scenarios
        
        Example:
            >>> scenarios = {
            ...     'Current': {'google': 15000, 'facebook': 10000},
            ...     'Aggressive': {'google': 20000, 'facebook': 15000}
            ... }
            >>> comparison = optimizer.compare_scenarios(scenarios)
        """
        results = []
        
        for scenario_name, allocation in scenarios.items():
            channels = list(allocation.keys())
            total_spend = sum(allocation.values())
            
            # Predict revenue
            predicted_revenue = self._predict_revenue(allocation, channels)
            roas = predicted_revenue / total_spend if total_spend > 0 else 0
            
            # Build result row
            row = {
                'scenario': scenario_name,
                'total_spend': total_spend,
                'predicted_revenue': predicted_revenue,
                'roas': roas
            }
            
            # Add individual channel spends
            for channel in channels:
                row[channel] = allocation[channel]
            
            results.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Sort by predicted revenue (descending)
        df = df.sort_values('predicted_revenue', ascending=False)
        
        return df
    
    def get_diminishing_returns_point(
        self,
        channel: str,
        X_base: pd.DataFrame,
        marginal_roas_threshold: float = 1.5
    ) -> float:
        """
        Find spend level where marginal ROAS drops below threshold.
        
        Uses binary search to efficiently find the saturation point.
        
        Args:
            channel: Channel name
            X_base: Baseline data
            marginal_roas_threshold: Threshold for marginal ROAS
        
        Returns:
            Optimal spend level (saturation point)
        
        Example:
            >>> saturation_point = optimizer.get_diminishing_returns_point('google', X_train)
        """
        if channel not in self.channels:
            raise ValueError(f"Channel '{channel}' not in model")
        
        # Get current spend range
        current_spend = X_base[channel].mean()
        
        # Generate response curve
        max_spend = current_spend * 3  # Search up to 3x current
        budget_range = np.linspace(0, max_spend, 200)
        
        curve_df = self.model.get_response_curve(channel, budget_range, X_base)
        
        # Find point where marginal ROAS drops below threshold
        below_threshold = curve_df[curve_df['marginal_roas'] < marginal_roas_threshold]
        
        if len(below_threshold) > 0:
            # Return first point below threshold
            saturation_point = below_threshold.iloc[0]['spend']
        else:
            # If never drops below threshold, return max tested spend
            saturation_point = max_spend
        
        return saturation_point
    
    def sensitivity_analysis(
        self,
        channel: str,
        current_allocation: Dict[str, float],
        variation_range: Tuple[float, float] = (-30, 30),
        n_points: int = 20
    ) -> pd.DataFrame:
        """
        Analyze sensitivity of revenue to changes in channel spend.
        
        Varies the specified channel's spend while keeping others constant.
        
        Args:
            channel: Channel to vary
            current_allocation: Current spend allocation
            variation_range: Tuple of (min_pct, max_pct) variation
            n_points: Number of points to test
        
        Returns:
            DataFrame with sensitivity analysis results
        
        Example:
            >>> current = {'google': 15000, 'facebook': 10000}
            >>> sensitivity = optimizer.sensitivity_analysis('google', current, (-30, 30))
        """
        if channel not in self.channels:
            raise ValueError(f"Channel '{channel}' not in model")
        
        if channel not in current_allocation:
            raise ValueError(f"Channel '{channel}' not in current_allocation")
        
        current_spend = current_allocation[channel]
        
        # Generate spend variations
        min_pct, max_pct = variation_range
        pct_changes = np.linspace(min_pct, max_pct, n_points)
        
        results = []
        
        for pct_change in pct_changes:
            # Calculate new spend for this channel
            new_spend = current_spend * (1 + pct_change / 100)
            
            # Create new allocation
            new_allocation = current_allocation.copy()
            new_allocation[channel] = new_spend
            
            # Predict revenue
            channels = list(new_allocation.keys())
            predicted_revenue = self._predict_revenue(new_allocation, channels)
            
            # Calculate metrics
            total_spend = sum(new_allocation.values())
            roas = predicted_revenue / total_spend if total_spend > 0 else 0
            
            # Calculate baseline revenue (with current allocation)
            baseline_revenue = self._predict_revenue(current_allocation, channels)
            revenue_impact = predicted_revenue - baseline_revenue
            revenue_impact_pct = (revenue_impact / baseline_revenue * 100) if baseline_revenue > 0 else 0
            
            results.append({
                'channel_spend_pct_change': pct_change,
                'channel_spend': new_spend,
                'total_spend': total_spend,
                'predicted_revenue': predicted_revenue,
                'roas': roas,
                'revenue_impact': revenue_impact,
                'revenue_impact_pct': revenue_impact_pct
            })
        
        return pd.DataFrame(results)
    
    def _predict_revenue(self, allocation: Dict[str, float], channels: List[str]) -> float:
        """
        Predict revenue for a given allocation.
        
        Args:
            allocation: Dict mapping channels to spend amounts
            channels: List of channels
        
        Returns:
            Predicted revenue
        """
        # Create scenario DataFrame
        X_scenario = self.X_base.copy()
        
        # Update channel spends
        for channel in channels:
            X_scenario[channel] = allocation[channel]
        
        # Predict revenue
        predictions = self.model.predict(X_scenario)
        
        # Return mean prediction
        return predictions.mean()
    
    def get_optimal_allocation_per_channel(
        self,
        channels: List[str],
        marginal_roas_target: float = 2.0
    ) -> Dict[str, Dict[str, float]]:
        """
        Get optimal spend for each channel individually.
        
        For each channel, finds the spend level where marginal ROAS
        equals the target.
        
        Args:
            channels: List of channels to analyze
            marginal_roas_target: Target marginal ROAS
        
        Returns:
            Dict mapping channels to their optimal metrics
        
        Example:
            >>> optimal = optimizer.get_optimal_allocation_per_channel(['google', 'facebook'])
        """
        results = {}
        
        for channel in channels:
            current_spend = self.X_base[channel].mean()
            
            # Get response curve
            max_spend = current_spend * 3
            budget_range = np.linspace(0, max_spend, 200)
            curve_df = self.model.get_response_curve(channel, budget_range, self.X_base)
            
            # Find point closest to target marginal ROAS
            optimal_idx = (curve_df['marginal_roas'] - marginal_roas_target).abs().idxmin()
            optimal_spend = curve_df.loc[optimal_idx, 'spend']
            optimal_revenue = curve_df.loc[optimal_idx, 'revenue']
            optimal_marginal_roas = curve_df.loc[optimal_idx, 'marginal_roas']
            
            # Calculate change from current
            spend_change = optimal_spend - current_spend
            spend_change_pct = (spend_change / current_spend * 100) if current_spend > 0 else 0
            
            results[channel] = {
                'current_spend': current_spend,
                'optimal_spend': optimal_spend,
                'spend_change': spend_change,
                'spend_change_pct': spend_change_pct,
                'predicted_revenue': optimal_revenue,
                'marginal_roas': optimal_marginal_roas
            }
        
        return results
