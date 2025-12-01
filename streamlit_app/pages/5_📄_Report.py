"""
Ridge MMM - Report Generation Page
Generate comprehensive reports in multiple formats
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path
import io
from datetime import datetime
import base64

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ridge_mmm import RidgeMMM
from visualizations import (
    plot_waterfall,
    plot_decomposition,
    plot_channel_contributions
)
from utils import format_currency

# Page configuration
st.set_page_config(
    page_title="Report - Ridge MMM",
    page_icon="📄",
    layout="wide"
)

# Header
st.title("📄 Report Generation")
st.markdown("Generate comprehensive MMM analysis reports")

# Check if model is trained
if not st.session_state.get('model_trained', False):
    st.warning("⚠️ No trained model found. Please train a model first.")
    st.stop()

# Get model and data
mmm = st.session_state.mmm_model
df = st.session_state.df
config = st.session_state.config
train_metrics = st.session_state.train_metrics
test_metrics = st.session_state.test_metrics
media_columns = config.get('media_columns', [])

# Prepare data
train_split = st.session_state.transformation_config['train_test_split']
train_size = int(len(df) * train_split)
feature_cols = media_columns + config.get('exog_columns', [])
X = df[feature_cols]
y = df[config.get('revenue_column')]
X_train = X.iloc[:train_size]

# Get contributions and decomposition
contributions = mmm.get_contributions(X)
decomp = mmm.decompose_timeseries(X, y)

# ============================================================================
# REPORT CONFIGURATION
# ============================================================================

st.markdown("## ⚙️ Report Configuration")

col1, col2 = st.columns(2)

with col1:
    report_title = st.text_input(
        "Report Title",
        value=f"MMM Analysis Report - {datetime.now().strftime('%Y-%m-%d')}"
    )

with col2:
    include_raw_data = st.checkbox(
        "Include raw data sheets",
        value=False,
        help="Include detailed data tables in Excel export"
    )

# ============================================================================
# AUTO-GENERATED EXECUTIVE SUMMARY
# ============================================================================

st.markdown("---")
st.markdown("## 📋 Executive Summary")

# Calculate summary metrics
date_col = config.get('date_column')
if date_col and date_col in df.columns:
    start_date = pd.to_datetime(df[date_col]).min().strftime('%Y-%m-%d')
    end_date = pd.to_datetime(df[date_col]).max().strftime('%Y-%m-%d')
    n_weeks = len(df)
else:
    start_date = "N/A"
    end_date = "N/A"
    n_weeks = len(df)

# Top performing channels
channel_contribs = contributions[contributions['channel'] != 'base'].copy()
top_channels = channel_contribs.nlargest(3, 'roas')

# Calculate potential improvement (if optimization was run)
if 'optimization_result' in st.session_state:
    opt_result = st.session_state.optimization_result
    potential_improvement = opt_result['improvement_pct']
else:
    potential_improvement = None

# Generate recommendations
recommendations = []

# Check for underperforming channels (ROAS < 2.0)
underperforming = channel_contribs[channel_contribs['roas'] < 2.0]
if len(underperforming) > 0:
    for _, row in underperforming.iterrows():
        recommendations.append(
            f"**{row['channel']}**의 ROAS가 낮습니다 ({row['roas']:.2f}). 지출을 줄이거나 캠페인 효과를 개선하는 것을 고려하세요."
        )

# Check for high-opportunity channels (top ROAS)
if len(top_channels) > 0:
    top_channel = top_channels.iloc[0]
    recommendations.append(
        f"**{top_channel['channel']}** shows excellent performance (ROAS: {top_channel['roas']:.2f}). Consider increasing investment."
    )

# Add optimization recommendation if available
if potential_improvement and potential_improvement > 5:
    recommendations.append(
        f"Budget reallocation could improve revenue by **{potential_improvement:.1f}%**. See Budget Optimizer for details."
    )

# Limit to 5 recommendations
recommendations = recommendations[:5]

# Display executive summary
summary_text = f"""
### Analysis Period
**{start_date}** to **{end_date}** ({n_weeks} weeks of data)

### Model Performance
- **R² Score**: {test_metrics['r2']:.4f} ({('Excellent' if test_metrics['r2'] > 0.8 else 'Good' if test_metrics['r2'] > 0.6 else 'Fair')})
- **MAPE**: {test_metrics['mape']:.2f}%
- **Model Quality**: {'✅ High confidence' if test_metrics['r2'] > 0.8 else '✓ Acceptable' if test_metrics['r2'] > 0.6 else '⚠️ Needs improvement'}

### Top Performing Channels
"""

for idx, row in top_channels.iterrows():
    summary_text += f"\n{idx + 1}. **{row['channel']}** - ROAS: {row['roas']:.2f}, Contribution: {format_currency(row['contribution'])}"

if potential_improvement:
    summary_text += f"\n\n### Budget Optimization Insight\nPotential revenue improvement: **{potential_improvement:.1f}%** with optimal allocation"

summary_text += "\n\n### Key Recommendations\n"
for i, rec in enumerate(recommendations, 1):
    summary_text += f"\n{i}. {rec}"

st.markdown(summary_text)

# ============================================================================
# DETAILED REPORT SECTIONS
# ============================================================================

st.markdown("---")
st.markdown("## 📊 Detailed Analysis")

# Channel Performance Table
st.markdown("### Channel Performance")

display_contribs = channel_contribs.copy()
display_contribs['spend'] = display_contribs['spend'].apply(lambda x: format_currency(x, decimals=0))
display_contribs['contribution'] = display_contribs['contribution'].apply(lambda x: format_currency(x, decimals=0))
display_contribs['roas'] = display_contribs['roas'].apply(lambda x: f"{x:.2f}")
display_contribs['contribution_pct'] = display_contribs['contribution_pct'].apply(lambda x: f"{x:.1f}%")

st.dataframe(display_contribs, use_container_width=True, hide_index=True)

# Waterfall Chart
st.markdown("### Revenue Decomposition")
fig_waterfall = plot_waterfall(contributions)
st.plotly_chart(fig_waterfall, use_container_width=True)

# Time Series Decomposition
st.markdown("### Time Series Decomposition")
fig_decomp = plot_decomposition(decomp)
st.plotly_chart(fig_decomp, use_container_width=True)

# ============================================================================
# DOWNLOAD OPTIONS
# ============================================================================

st.markdown("---")
st.markdown("## 📥 Download Report")

col1, col2, col3 = st.columns(3)

# Excel Export
with col1:
    if st.button("📊 Download Excel", use_container_width=True, type="primary"):
        with st.spinner("Generating Excel report..."):
            try:
                output = io.BytesIO()
                
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Sheet 1: Executive Summary
                    summary_data = pd.DataFrame([
                        ['Report Title', report_title],
                        ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                        ['', ''],
                        ['ANALYSIS PERIOD', ''],
                        ['Start Date', start_date],
                        ['End Date', end_date],
                        ['Weeks of Data', n_weeks],
                        ['', ''],
                        ['MODEL PERFORMANCE', ''],
                        ['R² Score', f"{test_metrics['r2']:.4f}"],
                        ['MAPE', f"{test_metrics['mape']:.2f}%"],
                        ['MAE', f"{test_metrics['mae']:,.0f}"],
                        ['RMSE', f"{test_metrics['rmse']:,.0f}"],
                    ])
                    summary_data.to_excel(writer, sheet_name='Executive Summary', index=False, header=False)
                    
                    # Sheet 2: Channel Contributions
                    contributions.to_excel(writer, sheet_name='Channel Contributions', index=False)
                    
                    # Sheet 3: Model Metrics
                    metrics_df = pd.DataFrame({
                        'Metric': ['R²', 'MAPE', 'MAE', 'RMSE'],
                        'Train': [train_metrics['r2'], train_metrics['mape'], train_metrics['mae'], train_metrics['rmse']],
                        'Test': [test_metrics['r2'], test_metrics['mape'], test_metrics['mae'], test_metrics['rmse']]
                    })
                    metrics_df.to_excel(writer, sheet_name='Model Metrics', index=False)
                    
                    # Sheet 4: Weekly Decomposition (if include_raw_data)
                    if include_raw_data:
                        decomp.to_excel(writer, sheet_name='Weekly Decomposition', index=False)
                    
                    # Sheet 5: Optimization Results (if available)
                    if 'optimization_result' in st.session_state:
                        opt_result = st.session_state.optimization_result
                        opt_df = pd.DataFrame([
                            {'Channel': ch, 'Current': curr, 'Optimal': opt_result['optimal_allocation'][ch]}
                            for ch, curr in opt_result['current_allocation'].items()
                        ])
                        opt_df.to_excel(writer, sheet_name='Optimal Allocation', index=False)
                
                output.seek(0)
                
                st.download_button(
                    label="⬇️ Download Excel File",
                    data=output,
                    file_name=f"mmm_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                st.success("✅ Excel report ready!")
                
            except Exception as e:
                st.error(f"❌ Error generating Excel: {str(e)}")

# HTML Export
with col2:
    if st.button("🌐 Download HTML", use_container_width=True):
        with st.spinner("Generating HTML report..."):
            try:
                # Generate HTML report
                html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report_title}</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{ margin: 0; font-size: 2.5em; }}
        h2 {{ color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        h3 {{ color: #764ba2; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        .metric {{
            display: inline-block;
            background: #f0f0f0;
            padding: 15px 25px;
            margin: 10px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }}
        .recommendation {{
            background: #e8f4f8;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin: 10px 0;
        }}
        @media print {{
            body {{ background: white; }}
            .section {{ box-shadow: none; page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report_title}</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <h3>Analysis Period</h3>
        <p><strong>{start_date}</strong> to <strong>{end_date}</strong> ({n_weeks} weeks)</p>
        
        <h3>Model Performance</h3>
        <div class="metric">
            <div>R² Score</div>
            <div class="metric-value">{test_metrics['r2']:.4f}</div>
        </div>
        <div class="metric">
            <div>MAPE</div>
            <div class="metric-value">{test_metrics['mape']:.2f}%</div>
        </div>
        
        <h3>Top Performing Channels</h3>
        <ol>
"""
                
                for _, row in top_channels.iterrows():
                    html_content += f"<li><strong>{row['channel']}</strong> - ROAS: {row['roas']:.2f}</li>\n"
                
                html_content += """
        </ol>
        
        <h3>Key Recommendations</h3>
"""
                
                for rec in recommendations:
                    html_content += f'<div class="recommendation">{rec}</div>\n'
                
                html_content += """
    </div>
    
    <div class="section">
        <h2>Channel Performance</h2>
        <table>
            <thead>
                <tr>
                    <th>Channel</th>
                    <th>Spend</th>
                    <th>Contribution</th>
                    <th>ROAS</th>
                    <th>Contribution %</th>
                </tr>
            </thead>
            <tbody>
"""
                
                for _, row in channel_contribs.iterrows():
                    html_content += f"""
                <tr>
                    <td>{row['channel']}</td>
                    <td>{format_currency(row['spend'], decimals=0)}</td>
                    <td>{format_currency(row['contribution'], decimals=0)}</td>
                    <td>{row['roas']:.2f}</td>
                    <td>{row['contribution_pct']:.1f}%</td>
                </tr>
"""
                
                html_content += """
            </tbody>
        </table>
    </div>
    
    <div class="section">
        <p style="text-align: center; color: #666;">
            Ridge MMM Report • Generated with Streamlit
        </p>
    </div>
</body>
</html>
"""
                
                st.download_button(
                    label="⬇️ Download HTML File",
                    data=html_content,
                    file_name=f"mmm_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
                
                st.success("✅ HTML report ready!")
                
            except Exception as e:
                st.error(f"❌ Error generating HTML: {str(e)}")

# PDF Export (simplified - note: full PDF with charts would require additional libraries)
with col3:
    st.info("📄 PDF export requires additional setup. Use HTML export and print to PDF from browser.")

# ============================================================================
# PREVIEW
# ============================================================================

st.markdown("---")
st.markdown("## 👀 Report Preview")

with st.expander("View Report Preview", expanded=False):
    st.markdown(summary_text)
    
    st.markdown("### Channel Performance Table")
    st.dataframe(display_contribs, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem 0;">
    <p>Report Generation • Export your analysis in multiple formats</p>
</div>
""", unsafe_allow_html=True)
