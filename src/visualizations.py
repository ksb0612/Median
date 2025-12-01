"""
Visualization functions for Ridge MMM results.

This module provides Plotly-based visualization functions for analyzing
and presenting Marketing Mix Modeling results.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Dict
from scipy import stats

from ridge_mmm import RidgeMMM
from utils import format_currency
from utils.plot_utils import (
    apply_default_layout,
    get_default_layout,
    COLOR_PALETTE,
    format_currency as plot_format_currency,
    format_percentage,
    add_annotation
)


def plot_waterfall(
    contributions_df: pd.DataFrame,
    title: str = "Revenue Decomposition"
) -> go.Figure:
    """
    Create waterfall chart showing revenue decomposition.
    
    Shows how Base + each channel contribution builds up to Total Revenue.
    
    Args:
        contributions_df: DataFrame from RidgeMMM.get_contributions()
        title: Chart title
    
    Returns:
        Plotly Figure object
    
    Example:
        >>> contributions = mmm.get_contributions(X)
        >>> fig = plot_waterfall(contributions)
        >>> fig.show()
    """
    # Sort by contribution (base first, then channels by contribution)
    df = contributions_df.copy()
    
    # Separate base and channels
    base_row = df[df['channel'] == 'base']
    channel_rows = df[df['channel'] != 'base'].sort_values('contribution', ascending=False)
    
    # Combine: base first, then channels
    df_sorted = pd.concat([base_row, channel_rows], ignore_index=True)
    
    # Prepare data for waterfall
    x_labels = df_sorted['channel'].tolist() + ['Total']
    values = df_sorted['contribution'].tolist() + [df_sorted['contribution'].sum()]
    
    # Color coding
    colors = []
    for channel in df_sorted['channel']:
        if channel == 'base':
            colors.append('#808080')  # Gray for base
        else:
            colors.append('#1f77b4')  # Blue for channels
    colors.append('#2ca02c')  # Green for total
    
    # Create waterfall chart with simpler approach
    fig = go.Figure()
    
    # Add bars
    cumulative = 0
    for i, (label, value, color) in enumerate(zip(x_labels, values, colors)):
        if i == 0:
            # First bar (base) starts from 0
            fig.add_trace(go.Bar(
                x=[label],
                y=[value],
                marker_color=color,
                text=format_currency(value, decimals=0),
                textposition='outside',
                showlegend=False,
                hovertemplate=f'{label}: {format_currency(value, decimals=0)}<extra></extra>'
            ))
            cumulative = value
        elif i == len(x_labels) - 1:
            # Total bar
            fig.add_trace(go.Bar(
                x=[label],
                y=[value],
                marker_color=color,
                text=format_currency(value, decimals=0),
                textposition='outside',
                showlegend=False,
                hovertemplate=f'{label}: {format_currency(value, decimals=0)}<extra></extra>'
            ))
        else:
            # Channel bars - show as stacked
            fig.add_trace(go.Bar(
                x=[label],
                y=[value],
                base=cumulative,
                marker_color=color,
                text=format_currency(value, decimals=0),
                textposition='outside',
                showlegend=False,
                hovertemplate=f'{label}: {format_currency(value, decimals=0)}<extra></extra>'
            ))
            cumulative += value

    # Apply default layout with custom settings
    apply_default_layout(
        fig,
        title=title,
        xaxis_title="Component",
        yaxis_title="Revenue",
        height=500,
        showlegend=False,
        yaxis=dict(tickformat=",.0f"),
        barmode='overlay'
    )

    return fig


def plot_decomposition(decomp_df: pd.DataFrame) -> go.Figure:
    """
    Create stacked area chart showing time series decomposition.
    
    Args:
        decomp_df: DataFrame from RidgeMMM.decompose_timeseries()
    
    Returns:
        Plotly Figure object
    
    Example:
        >>> decomp = mmm.decompose_timeseries(X, y)
        >>> fig = plot_decomposition(decomp)
        >>> fig.show()
    """
    # Get column names (exclude date, actual, predicted)
    exclude_cols = ['date', 'actual_revenue', 'predicted_revenue']
    component_cols = [col for col in decomp_df.columns if col not in exclude_cols]
    
    # Create figure
    fig = go.Figure()

    # Use standardized color palette
    colors = COLOR_PALETTE['channels']
    
    for i, col in enumerate(component_cols):
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=decomp_df['date'] if 'date' in decomp_df.columns else decomp_df.index,
            y=decomp_df[col],
            name=col,
            mode='lines',
            stackgroup='one',
            fillcolor=color,
            line=dict(width=0.5, color=color),
            hovertemplate='%{y:,.0f}<extra></extra>'
        ))
    
    # Add actual revenue line
    if 'actual_revenue' in decomp_df.columns:
        fig.add_trace(go.Scatter(
            x=decomp_df['date'] if 'date' in decomp_df.columns else decomp_df.index,
            y=decomp_df['actual_revenue'],
            name='Actual Revenue',
            mode='lines',
            line=dict(color='black', width=2, dash='dot'),
            hovertemplate='%{y:,.0f}<extra></extra>'
        ))

    # Apply default layout with custom legend
    apply_default_layout(
        fig,
        title="Revenue Decomposition Over Time",
        xaxis_title="Date",
        yaxis_title="Revenue",
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    return fig


def plot_response_curves(
    model: RidgeMMM,
    X: pd.DataFrame,
    channels: List[str]
) -> go.Figure:
    """
    Create response curves for multiple channels.
    
    Shows spend vs revenue relationship with current spend marked.
    
    Args:
        model: Fitted RidgeMMM model
        X: DataFrame with current spend data
        channels: List of channel names to plot
    
    Returns:
        Plotly Figure with subplots
    
    Example:
        >>> fig = plot_response_curves(mmm, X_train, ['google', 'facebook'])
        >>> fig.show()
    """
    # Calculate subplot layout
    n_channels = len(channels)
    n_cols = 2
    n_rows = (n_channels + n_cols - 1) // n_cols
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"{ch} Response Curve" for ch in channels],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    for idx, channel in enumerate(channels):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        # Get current spend
        current_spend = X[channel].mean()
        
        # Generate budget range (0 to 2x current spend)
        max_spend = current_spend * 2
        budget_range = np.linspace(0, max_spend, 100)
        
        # Get response curve
        curve_df = model.get_response_curve(channel, budget_range, X)
        
        # Add curve line
        fig.add_trace(
            go.Scatter(
                x=curve_df['spend'],
                y=curve_df['revenue'],
                mode='lines',
                name=channel,
                line=dict(color='#1f77b4', width=2),
                showlegend=False,
                hovertemplate='Spend: %{x:,.0f}<br>Revenue: %{y:,.0f}<extra></extra>'
            ),
            row=row,
            col=col
        )
        
        # Mark current spend point
        current_revenue = curve_df.loc[
            (curve_df['spend'] - current_spend).abs().idxmin(),
            'revenue'
        ]
        
        fig.add_trace(
            go.Scatter(
                x=[current_spend],
                y=[current_revenue],
                mode='markers',
                name='Current',
                marker=dict(color='red', size=10, symbol='circle'),
                showlegend=False,
                hovertemplate='Current Spend: %{x:,.0f}<br>Revenue: %{y:,.0f}<extra></extra>'
            ),
            row=row,
            col=col
        )
        
        # Find saturation point (where marginal ROAS < 1)
        saturation_idx = np.where(curve_df['marginal_roas'] < 1)[0]
        if len(saturation_idx) > 0:
            sat_idx = saturation_idx[0]
            sat_spend = curve_df.loc[sat_idx, 'spend']
            sat_revenue = curve_df.loc[sat_idx, 'revenue']
            
            # Add saturation point
            fig.add_trace(
                go.Scatter(
                    x=[sat_spend],
                    y=[sat_revenue],
                    mode='markers',
                    name='Saturation',
                    marker=dict(color='orange', size=8, symbol='diamond'),
                    showlegend=False,
                    hovertemplate='Saturation Point<br>Spend: %{x:,.0f}<br>Revenue: %{y:,.0f}<extra></extra>'
                ),
                row=row,
                col=col
            )
        
        # Update axes
        fig.update_xaxes(title_text="Spend", row=row, col=col, tickformat=",.0f")
        fig.update_yaxes(title_text="Revenue", row=row, col=col, tickformat=",.0f")
    
    # Apply default layout
    apply_default_layout(
        fig,
        title="Channel Response Curves",
        height=400 * n_rows,
        showlegend=False
    )

    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> go.Figure:
    """
    Create actual vs predicted scatter plot.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        Plotly Figure object
    
    Example:
        >>> predictions = mmm.predict(X_test)
        >>> fig = plot_actual_vs_predicted(y_test, predictions)
        >>> fig.show()
    """
    from sklearn.metrics import r2_score
    
    # Calculate R²
    r2 = r2_score(y_true, y_pred)
    
    # Calculate residuals for coloring
    residuals = np.abs(y_true - y_pred)
    
    # Create scatter plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(
            size=8,
            color=residuals,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Abs Residual"),
            line=dict(width=0.5, color='white')
        ),
        text=[f"Actual: {a:,.0f}<br>Predicted: {p:,.0f}<br>Error: {a-p:,.0f}" 
              for a, p in zip(y_true, y_pred)],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # Add 45-degree reference line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', width=2, dash='dash'),
        showlegend=True
    ))
    
    # Add R² annotation
    fig.add_annotation(
        x=0.05,
        y=0.95,
        xref='paper',
        yref='paper',
        text=f'R² = {r2:.4f}',
        showarrow=False,
        font=dict(size=14, color='black'),
        bgcolor='white',
        bordercolor='black',
        borderwidth=1
    )
    
    # Apply default layout
    apply_default_layout(
        fig,
        title="Actual vs Predicted Revenue",
        xaxis_title="Actual Revenue",
        yaxis_title="Predicted Revenue",
        height=500
    )

    fig.update_xaxes(tickformat=",.0f")
    fig.update_yaxes(tickformat=",.0f")

    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> go.Figure:
    """
    Create residual analysis plots.
    
    Creates two subplots:
    1. Residuals vs Predicted values
    2. Residual histogram with normal curve overlay
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        Plotly Figure with subplots
    
    Example:
        >>> fig = plot_residuals(y_test, predictions)
        >>> fig.show()
    """
    residuals = y_true - y_pred
    
    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Residuals vs Predicted", "Residual Distribution"),
        horizontal_spacing=0.12
    )
    
    # Subplot 1: Residuals vs Predicted
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(size=6, color='#1f77b4', opacity=0.6),
            showlegend=False,
            hovertemplate='Predicted: %{x:,.0f}<br>Residual: %{y:,.0f}<extra></extra>'
        ),
        row=1,
        col=1
    )
    
    # Add horizontal line at y=0
    fig.add_trace(
        go.Scatter(
            x=[y_pred.min(), y_pred.max()],
            y=[0, 0],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=False
        ),
        row=1,
        col=1
    )
    
    # Subplot 2: Histogram with normal curve
    hist_data = residuals
    
    fig.add_trace(
        go.Histogram(
            x=hist_data,
            nbinsx=30,
            name='Residuals',
            marker=dict(color='#1f77b4', opacity=0.7),
            showlegend=False
        ),
        row=1,
        col=2
    )
    
    # Overlay normal distribution
    mu = residuals.mean()
    sigma = residuals.std()
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    y_normal = stats.norm.pdf(x_range, mu, sigma) * len(residuals) * (residuals.max() - residuals.min()) / 30
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_normal,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', width=2),
            showlegend=True
        ),
        row=1,
        col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Predicted Revenue", row=1, col=1, tickformat=",.0f")
    fig.update_yaxes(title_text="Residual", row=1, col=1, tickformat=",.0f")
    fig.update_xaxes(title_text="Residual", row=1, col=2, tickformat=",.0f")
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    # Apply default layout
    apply_default_layout(
        fig,
        title="Residual Analysis",
        height=400,
        showlegend=True
    )

    return fig


def plot_channel_contributions(contributions_df: pd.DataFrame) -> go.Figure:
    """
    Create horizontal bar chart of channel contributions.
    
    Bars are colored by ROAS with annotations showing ROAS values.
    
    Args:
        contributions_df: DataFrame from RidgeMMM.get_contributions()
    
    Returns:
        Plotly Figure object
    
    Example:
        >>> contributions = mmm.get_contributions(X)
        >>> fig = plot_channel_contributions(contributions)
        >>> fig.show()
    """
    # Filter out base and sort by contribution
    df = contributions_df[contributions_df['channel'] != 'base'].copy()
    df = df.sort_values('contribution', ascending=True)
    
    # Color by ROAS
    def get_color(roas):
        if roas < 2:
            return '#d62728'  # Red
        elif roas < 3:
            return '#ff7f0e'  # Orange
        else:
            return '#2ca02c'  # Green
    
    colors = [get_color(roas) for roas in df['roas']]
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df['channel'],
        x=df['contribution'],
        orientation='h',
        marker=dict(color=colors),
        text=[f"ROAS: {roas:.2f}" for roas in df['roas']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Contribution: %{x:,.0f}<br>ROAS: %{text}<extra></extra>'
    ))
    
    # Apply default layout
    apply_default_layout(
        fig,
        title="Channel Contributions",
        xaxis_title="Contribution to Revenue",
        yaxis_title="Channel",
        height=max(400, len(df) * 50),
        showlegend=False
    )

    fig.update_xaxes(tickformat=",.0f")

    return fig


def plot_qq(residuals: np.ndarray) -> go.Figure:
    """
    Create Q-Q plot for residual normality check.
    
    Args:
        residuals: Array of residuals
    
    Returns:
        Plotly Figure object
    """
    # Calculate theoretical quantiles
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=osm,
        y=osr,
        mode='markers',
        marker=dict(size=6, color='#1f77b4'),
        name='Residuals',
        showlegend=False
    ))
    
    # Add reference line
    line_x = np.array([osm.min(), osm.max()])
    line_y = slope * line_x + intercept
    
    fig.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Normal Distribution',
        showlegend=True
    ))
    
    # Apply default layout
    apply_default_layout(
        fig,
        title="Q-Q Plot (Normality Check)",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        height=400
    )

    return fig
def create_market_channel_heatmap(model: 'HierarchicalMMM', df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Create heatmap data: markets × channels with ROAS values.
    
    Args:
        model: Fitted HierarchicalMMM model
        df: Optional DataFrame to use for contributions (if None, uses training data if available/stored, 
            but HierarchicalMMM doesn't store data, so df is likely required or we rely on model.models having data?)
            Actually RidgeMMM doesn't store X_train. So df is required.
    
    Returns:
        DataFrame with index=Market, columns=Channel, values=ROAS
    """
    if not hasattr(model, 'models') or not model.models:
        return pd.DataFrame()
    
    heatmap_data = []
    
    # Iterate through segments
    for seg_key, sub_model in model.models.items():
        if seg_key == 'global':
            continue
            
        # We need data for this segment to calculate ROAS
        # If df is provided, filter it.
        if df is not None:
            if model.analysis_level == 'country':
                seg_df = df[df['country'] == seg_key]
            elif model.analysis_level == 'os':
                seg_df = df[df['os'] == seg_key]
            elif model.analysis_level == 'country_os':
                # Try to split key
                try:
                    c, o = seg_key.split('_')
                    seg_df = df[(df['country'] == c) & (df['os'] == o)]
                except:
                    continue
            else:
                continue
                
            if seg_df.empty:
                continue
                
            # Get contributions
            try:
                contribs = sub_model.get_contributions(seg_df)
                for _, row in contribs.iterrows():
                    if row['channel'] != 'base':
                        heatmap_data.append({
                            'Market': seg_key,
                            'Channel': row['channel'],
                            'ROAS': row['roas']
                        })
            except:
                continue
    
    if not heatmap_data:
        return pd.DataFrame()
        
    heatmap_df = pd.DataFrame(heatmap_data)
    return heatmap_df.pivot(index='Market', columns='Channel', values='ROAS')


def create_budget_sankey_diagram(allocation: Dict[str, Dict[str, float]]) -> go.Figure:
    """
    Create Sankey diagram: Budget → Markets → Channels.
    
    Args:
        allocation: Nested dictionary {Market: {Channel: Spend}}
    
    Returns:
        Plotly Figure object
    """
    # Nodes
    # 0: Total Budget
    # 1..M: Markets
    # M+1..M+C: Channels (unique channels)
    
    markets = list(allocation.keys())
    
    # Collect all unique channels
    unique_channels = set()
    for mkt_alloc in allocation.values():
        unique_channels.update(mkt_alloc.keys())
    channels = sorted(list(unique_channels))
    
    # Node labels
    labels = ["Total Budget"] + markets + channels
    
    # Node indices
    total_idx = 0
    market_indices = {mkt: i + 1 for i, mkt in enumerate(markets)}
    channel_indices = {ch: i + 1 + len(markets) for i, ch in enumerate(channels)}
    
    # Links
    source = []
    target = []
    values = []
    colors = []
    
    # 1. Total -> Markets
    for mkt in markets:
        mkt_total = sum(allocation[mkt].values())
        source.append(total_idx)
        target.append(market_indices[mkt])
        values.append(mkt_total)
        colors.append("rgba(31, 119, 180, 0.4)") # Blueish
        
    # 2. Markets -> Channels
    for mkt, ch_alloc in allocation.items():
        for ch, spend in ch_alloc.items():
            if spend > 0:
                source.append(market_indices[mkt])
                target.append(channel_indices[ch])
                values.append(spend)
                colors.append("rgba(44, 160, 44, 0.4)") # Greenish
                
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = labels,
          color = "blue"
        ),
        link = dict(
          source = source,
          target = target,
          value = values,
          color = colors
      ))])
    
    fig.update_layout(
        title="Budget Flow: Total → Markets → Channels",
        font_size=10,
        height=500
    )
    
    return fig


def plot_market_comparison(comparison_df: pd.DataFrame, metric: str = 'roas') -> go.Figure:
    """
    Create grouped bar chart comparing markets.
    
    Args:
        comparison_df: DataFrame from HierarchicalMMM.compare_segments()
        metric: Metric to plot ('roas', 'revenue', 'spend')
    
    Returns:
        Plotly Figure object
    """
    df = comparison_df.sort_values(metric, ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['segment'],
        y=df[metric],
        marker_color='#1f77b4',
        text=[f"{val:.2f}" if metric == 'roas' else format_currency(val, decimals=0) for val in df[metric]],
        textposition='outside'
    ))
    
    # Apply default layout
    apply_default_layout(
        fig,
        title=f"Market Comparison: {metric.upper()}",
        xaxis_title="Market",
        yaxis_title=metric.upper(),
        height=400
    )

    return fig
