"""Wrapper for Meta's Robyn MMM tool.

This module provides a Python interface to Meta's Robyn MMM (R package),
allowing users to run Robyn and compare results with Ridge MMM.
"""

import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import sys


class RobynWrapper:
    """
    Wrapper for running Meta's Robyn MMM from Python.

    Requires R and Robyn package to be installed:

    .. code-block:: r

        install.packages("Robyn")

    Example:
        >>> wrapper = RobynWrapper()
        >>> if wrapper.check_robyn_installed():
        ...     config = wrapper.prepare_config(
        ...         data=df,
        ...         date_var='date',
        ...         dep_var='revenue',
        ...         paid_media_vars=['google', 'meta']
        ...     )
        ...     results = wrapper.run(df, config)
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize Robyn wrapper.

        Args:
            output_dir: Directory for output files (default: 'outputs/robyn')
        """
        self.r_script = Path("scripts/run_robyn.R")
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/robyn")

    def check_r_installed(self) -> bool:
        """
        Check if R is installed and available.

        Returns:
            True if R is installed, False otherwise
        """
        return shutil.which("Rscript") is not None

    def check_robyn_installed(self) -> bool:
        """
        Check if Robyn R package is installed.

        Returns:
            True if Robyn is installed, False otherwise
        """
        if not self.check_r_installed():
            return False

        try:
            result = subprocess.run(
                ["Rscript", "-e", "library(Robyn)"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    def get_r_version(self) -> Optional[str]:
        """
        Get R version.

        Returns:
            R version string or None if not installed
        """
        if not self.check_r_installed():
            return None

        try:
            result = subprocess.run(
                ["Rscript", "-e", "cat(R.version.string)"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def prepare_config(
        self,
        data: pd.DataFrame,
        date_var: str,
        dep_var: str,
        paid_media_vars: List[str],
        prophet_vars: Optional[List[str]] = None,
        prophet_country: str = "US",
        context_vars: Optional[List[str]] = None,
        adstock: str = "geometric",
        cores: int = 4,
        iterations: int = 2000,
        trials: int = 5
    ) -> Dict:
        """
        Prepare Robyn configuration.

        Args:
            data: DataFrame with marketing data
            date_var: Name of date column
            dep_var: Name of dependent variable (revenue)
            paid_media_vars: List of paid media column names
            prophet_vars: Prophet decomposition variables (default: ['trend', 'season', 'holiday'])
            prophet_country: Country for Prophet holidays (default: 'US')
            context_vars: Additional context variables
            adstock: Adstock type ('geometric', 'weibull')
            cores: Number of CPU cores for parallel processing
            iterations: Number of optimization iterations
            trials: Number of optimization trials

        Returns:
            Configuration dictionary for Robyn

        Raises:
            ValueError: If required columns are missing
        """
        # Validate columns
        missing_cols = []
        if date_var not in data.columns:
            missing_cols.append(date_var)
        if dep_var not in data.columns:
            missing_cols.append(dep_var)
        for col in paid_media_vars:
            if col not in data.columns:
                missing_cols.append(col)

        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")

        # Default Prophet vars
        if prophet_vars is None:
            prophet_vars = ["trend", "season", "holiday"]

        # Default context vars
        if context_vars is None:
            context_vars = []

        config = {
            "date_var": date_var,
            "dep_var": dep_var,
            "paid_media_spends": paid_media_vars,
            "paid_media_vars": paid_media_vars,
            "prophet_vars": prophet_vars,
            "prophet_country": prophet_country,
            "context_vars": context_vars,
            "window_start": str(data[date_var].min()),
            "window_end": str(data[date_var].max()),
            "adstock": adstock,
            "cores": cores,
            "iterations": iterations,
            "trials": trials
        }

        return config

    def run(
        self,
        data: pd.DataFrame,
        config: Dict,
        timeout: int = 1800,
        verbose: bool = True
    ) -> Dict:
        """
        Run Robyn MMM.

        Args:
            data: DataFrame with marketing data
            config: Robyn configuration dictionary
            timeout: Maximum execution time in seconds (default: 1800 = 30 minutes)
            verbose: Print progress messages

        Returns:
            Dictionary containing Robyn results:
                - model_id: Best model identifier
                - nrmse: Normalized RMSE
                - mape: Mean Absolute Percentage Error
                - rsq_train: R-squared on training data
                - decomposition: Channel decomposition
                - response_curves: Response curves for each channel
                - coefficients: Model coefficients
                - adstock_params: Adstock parameters
                - saturation_params: Saturation parameters

        Raises:
            RuntimeError: If R or Robyn not installed, or execution fails
            subprocess.TimeoutExpired: If execution exceeds timeout
        """
        # Check prerequisites
        if not self.check_r_installed():
            raise RuntimeError(
                "R is not installed. Please install R:\n"
                "  - Windows: https://cran.r-project.org/bin/windows/base/\n"
                "  - macOS: brew install r\n"
                "  - Linux: sudo apt-get install r-base"
            )

        if not self.check_robyn_installed():
            raise RuntimeError(
                "Robyn package is not installed. Install in R:\n"
                "  R -e \"install.packages('Robyn')\""
            )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save data and config
        data_file = self.output_dir / "input_data.csv"
        config_file = self.output_dir / "config.json"

        data.to_csv(data_file, index=False)
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        if verbose:
            print("🚀 Running Robyn MMM...")
            print(f"   Data: {len(data)} rows, {len(config['paid_media_vars'])} channels")
            print(f"   Iterations: {config['iterations']}, Trials: {config['trials']}")
            print("   This may take 5-30 minutes...")

        # Run R script
        try:
            cmd = [
                "Rscript",
                str(self.r_script),
                str(data_file),
                str(config_file),
                str(self.output_dir),
                sys.executable  # Pass current python executable path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if verbose:
                print(result.stdout)

            if result.returncode != 0:
                error_msg = f"Robyn execution failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
                if verbose:
                    print(error_msg)
                raise RuntimeError(error_msg)

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Robyn execution exceeded timeout ({timeout}s). "
                "Consider increasing timeout or reducing iterations/trials."
            )

        # Load results
        results_file = self.output_dir / "robyn_results.json"

        if not results_file.exists():
            raise RuntimeError(
                f"Results file not found: {results_file}\n"
                "Robyn may have failed silently."
            )

        with open(results_file, 'r') as f:
            results = json.load(f)

        if verbose:
            print("✅ Robyn MMM complete!")

        return results

    def compare_with_ridge(
        self,
        robyn_results: Dict,
        ridge_model,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """
        Compare Robyn and Ridge MMM results.

        Args:
            robyn_results: Results dictionary from Robyn
            ridge_model: Fitted RidgeMMM instance
            X: Input features (marketing spend data)
            y: Target variable (revenue)

        Returns:
            DataFrame with side-by-side comparison:
                - channel: Channel name
                - spend: Total spend
                - ridge_contribution: Ridge estimated contribution
                - ridge_roas: Ridge ROAS
                - robyn_contribution: Robyn estimated contribution
                - robyn_roas: Robyn ROAS
                - contribution_diff_pct: Percentage difference in contributions
                - roas_diff_pct: Percentage difference in ROAS
        """
        # Get Ridge contributions (includes ROAS)
        ridge_contributions_df = ridge_model.get_contributions(X)

        # Get Robyn decomposition
        robyn_decomp = robyn_results['decomposition']

        # Build comparison table
        comparison = []

        for channel in ridge_model.media_channels:
            # Get Ridge metrics from DataFrame
            ridge_row = ridge_contributions_df[
                ridge_contributions_df['channel'] == channel
            ]

            if len(ridge_row) == 0:
                continue

            ridge_row = ridge_row.iloc[0]
            ridge_contrib = ridge_row['contribution']
            ridge_roas_val = ridge_row['roas']
            spend = ridge_row['spend']

            # Robyn metrics
            robyn_contrib = robyn_decomp.get(channel, 0)
            robyn_roas_val = robyn_contrib / spend if spend > 0 else 0

            # Calculate differences
            contrib_diff = (
                (ridge_contrib - robyn_contrib) / robyn_contrib * 100
                if robyn_contrib > 0 else np.nan
            )

            roas_diff = (
                (ridge_roas_val - robyn_roas_val) / robyn_roas_val * 100
                if robyn_roas_val > 0 else np.nan
            )

            comparison.append({
                'channel': channel,
                'spend': spend,
                'ridge_contribution': ridge_contrib,
                'ridge_roas': ridge_roas_val,
                'robyn_contribution': robyn_contrib,
                'robyn_roas': robyn_roas_val,
                'contribution_diff_pct': contrib_diff,
                'roas_diff_pct': roas_diff
            })

        return pd.DataFrame(comparison)

    def get_model_metrics_comparison(
        self,
        robyn_results: Dict,
        ridge_model,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict:
        """
        Compare model-level metrics.

        Args:
            robyn_results: Results from Robyn
            ridge_model: Fitted RidgeMMM instance
            X: Input features
            y: Target variable

        Returns:
            Dictionary with model metrics comparison
        """
        # Get Ridge evaluation metrics
        ridge_eval = ridge_model.evaluate(X, y)

        # Extract Robyn metrics
        robyn_metrics = {
            'nrmse': robyn_results.get('nrmse', None),
            'mape': robyn_results.get('mape', None),
            'rsq_train': robyn_results.get('rsq_train', None),
            'decomp_rssd': robyn_results.get('decomp_rssd', None)
        }

        # Ridge metrics (from evaluate method)
        ridge_metrics = {
            'r_squared': ridge_eval.get('r2', None),
            'mape': ridge_eval.get('mape', None),
            'rmse': ridge_eval.get('rmse', None),
            'mae': ridge_eval.get('mae', None)
        }

        return {
            'ridge': ridge_metrics,
            'robyn': robyn_metrics
        }
