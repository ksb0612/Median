"""Plotly chart styling utilities."""
import plotly.graph_objects as go
from typing import Dict, Optional, Any


def get_default_layout(
    title: str,
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = 500
) -> Dict[str, Any]:
    """
    Get default Plotly layout with consistent styling.

    Args:
        title: Chart title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        height: Chart height in pixels

    Returns:
        Layout dict for Plotly

    Examples:
        >>> layout = get_default_layout("My Chart", xaxis_title="Time", yaxis_title="Revenue")
        >>> 'title' in layout
        True
        >>> layout['height']
        500
    """
    return {
        'title': {
            'text': title,
            'font': {'size': 18, 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center'
        },
        'xaxis': {
            'title': xaxis_title,
            'gridcolor': '#E5E5E5',
            'showline': True,
            'linecolor': '#CCCCCC'
        },
        'yaxis': {
            'title': yaxis_title,
            'gridcolor': '#E5E5E5',
            'showline': True,
            'linecolor': '#CCCCCC'
        },
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'font': {'family': 'Arial, sans-serif', 'size': 12},
        'height': height,
        'hovermode': 'x unified',
        'showlegend': True,
        'legend': {
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1
        }
    }


def apply_default_layout(
    fig: go.Figure,
    title: str,
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = 500,
    **kwargs
) -> go.Figure:
    """
    Apply default layout to existing figure.

    Args:
        fig: Plotly figure
        title: Chart title
        xaxis_title: X-axis label
        yaxis_title: Y-axis label
        height: Chart height in pixels
        **kwargs: Additional layout parameters to override defaults

    Returns:
        Updated figure

    Examples:
        >>> fig = go.Figure()
        >>> fig = apply_default_layout(fig, "My Chart", yaxis_title="Revenue")
        >>> fig.layout.title.text
        'My Chart'
    """
    layout = get_default_layout(title, xaxis_title, yaxis_title, height)

    # Allow kwargs to override defaults
    layout.update(kwargs)

    fig.update_layout(**layout)
    return fig


def get_color_palette() -> Dict[str, Any]:
    """
    Get standard color palette for consistency.

    Returns:
        Dict with color definitions

    Examples:
        >>> palette = get_color_palette()
        >>> palette['primary']
        '#1f77b4'
        >>> len(palette['channels'])
        10
    """
    return {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'warning': '#ff9800',
        'danger': '#d62728',
        'info': '#17a2b8',
        'channels': [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Yellow-green
            '#17becf'   # Cyan
        ]
    }


# Export as module constant
COLOR_PALETTE = get_color_palette()


def format_currency(value: float, decimals: int = 0) -> str:
    """
    Format value as currency string.

    Args:
        value: Numeric value
        decimals: Number of decimal places

    Returns:
        Formatted string

    Examples:
        >>> format_currency(1234567.89)
        '$1,234,568'
        >>> format_currency(1234.5678, decimals=2)
        '$1,234.57'
    """
    return f"${value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format value as percentage string.

    Args:
        value: Numeric value (0.15 for 15%)
        decimals: Number of decimal places

    Returns:
        Formatted string

    Examples:
        >>> format_percentage(0.1534)
        '15.3%'
        >>> format_percentage(0.1534, decimals=2)
        '15.34%'
    """
    return f"{value * 100:.{decimals}f}%"


def add_annotation(
    fig: go.Figure,
    text: str,
    x: float,
    y: float,
    xref: str = "x",
    yref: str = "y",
    showarrow: bool = True
) -> go.Figure:
    """
    Add annotation to figure with default styling.

    Args:
        fig: Plotly figure
        text: Annotation text
        x: X position
        y: Y position
        xref: X reference ('x', 'paper')
        yref: Y reference ('y', 'paper')
        showarrow: Whether to show arrow

    Returns:
        Updated figure

    Examples:
        >>> fig = go.Figure()
        >>> fig = add_annotation(fig, "Peak", x=5, y=100)
        >>> len(fig.layout.annotations) > 0
        True
    """
    fig.add_annotation(
        text=text,
        x=x,
        y=y,
        xref=xref,
        yref=yref,
        showarrow=showarrow,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        font=dict(size=12, color="#000000"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#CCCCCC",
        borderwidth=1
    )
    return fig
