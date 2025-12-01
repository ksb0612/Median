"""
Ridge MMM - Results & Analysis Page
Comprehensive dashboard for model results and insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path
import io
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ridge_mmm import RidgeMMM
from visualizations import (
    plot_waterfall,
    plot_decomposition,
    plot_response_curves,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_channel_contributions,
    plot_qq
)
from utils import format_currency

# Cached visualization functions for performance
@st.cache_data(show_spinner=False)
def cached_plot_actual_vs_predicted(y_true, y_pred):
    """Cached version of plot_actual_vs_predicted."""
    return plot_actual_vs_predicted(y_true, y_pred)

@st.cache_data(show_spinner=False)
def cached_plot_residuals(y_true, y_pred):
    """Cached version of plot_residuals."""
    return plot_residuals(y_true, y_pred)

@st.cache_data(show_spinner=False)
def cached_plot_waterfall(_contributions_df):
    """Cached version of plot_waterfall."""
    return plot_waterfall(_contributions_df)

@st.cache_data(show_spinner=False)
def cached_plot_channel_contributions(_contributions_df):
    """Cached version of plot_channel_contributions."""
    return plot_channel_contributions(_contributions_df)

@st.cache_data(show_spinner=False)
def cached_plot_decomposition(_decomp_df):
    """Cached version of plot_decomposition."""
    return plot_decomposition(_decomp_df)

@st.cache_data(show_spinner=False)
def cached_plot_response_curves(_mmm, _X, channels):
    """Cached version of plot_response_curves."""
    return plot_response_curves(_mmm, _X, channels)

@st.cache_data(show_spinner=False)
def cached_plot_qq(residuals):
    """Cached version of plot_qq."""
    return plot_qq(residuals)

# Page configuration
st.set_page_config(
    page_title="Results & Analysis - Ridge MMM",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("📈 Results & Analysis")
st.markdown("마케팅 믹스 모델의 종합 분석")

# Check if model is trained
if not st.session_state.get('model_trained', False):
    st.warning("⚠️ 학습된 모델을 찾을 수 없습니다. 먼저 Model Configuration 페이지에서 모델을 학습시켜주세요.")
    st.stop()

# Get model and data from session state
mmm = st.session_state.mmm_model
df = st.session_state.df
config = st.session_state.config
train_metrics = st.session_state.train_metrics
test_metrics = st.session_state.test_metrics

# Get configuration
date_col = config.get('date_column')
revenue_col = config.get('revenue_column')
media_columns = config.get('media_columns', [])
exog_cols = config.get('exog_columns', [])

# Prepare data
train_split = st.session_state.transformation_config['train_test_split']
train_size = int(len(df) * train_split)

feature_cols = media_columns + exog_cols
X = df[feature_cols]
y = df[revenue_col]

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]
X_test = X.iloc[train_size:]
y_test = y.iloc[train_size:]

# Get predictions
y_train_pred = mmm.predict(X_train)
y_test_pred = mmm.predict(X_test)

# ============================================================================
# SIDEBAR: FILTERS AND DOWNLOADS
# ============================================================================

with st.sidebar:
    st.markdown("### 📊 Results Dashboard")
    st.markdown("---")
    
    st.markdown("### 🎯 Model Summary")
    summary = mmm.get_model_summary()
    st.write(f"**Channels**: {summary['n_channels']}")
    st.write(f"**Features**: {summary['n_features']}")
    st.write(f"**Alpha**: {summary['alpha']}")
    
    st.markdown("---")
    st.markdown("### 📥 Downloads")
    
    # Download Results as Excel
    if st.button("📊 Download Results (Excel)", use_container_width=True):
        with st.spinner("Generating Excel file..."):
            try:
                # Create Excel writer
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Metrics sheet
                    metrics_df = pd.DataFrame({
                        'Metric': ['R²', 'MAPE', 'MAE', 'RMSE'],
                        'Train': [
                            train_metrics['r2'],
                            train_metrics['mape'],
                            train_metrics['mae'],
                            train_metrics['rmse']
                        ],
                        'Test': [
                            test_metrics['r2'],
                            test_metrics['mape'],
                            test_metrics['mae'],
                            test_metrics['rmse']
                        ]
                    })
                    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                    
                    # Contributions sheet
                    contributions = mmm.get_contributions(X)
                    contributions.to_excel(writer, sheet_name='Contributions', index=False)
                    
                    # Decomposition sheet
                    decomp = mmm.decompose_timeseries(X, y)
                    decomp.to_excel(writer, sheet_name='Decomposition', index=False)
                    
                    # Model summary sheet
                    summary_df = pd.DataFrame([
                        {'Parameter': 'Alpha', 'Value': summary['alpha']},
                        {'Parameter': 'Channels', 'Value': ', '.join(summary['channels'])},
                        {'Parameter': 'Features', 'Value': summary['n_features']},
                        {'Parameter': 'Intercept', 'Value': summary['intercept']}
                    ])
                    summary_df.to_excel(writer, sheet_name='Model Summary', index=False)
                
                output.seek(0)
                
                st.download_button(
                    label="⬇️ Download Excel",
                    data=output,
                    file_name=f"mmm_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            except Exception as e:
                st.error(f"Error generating Excel: {str(e)}")
    
    st.markdown("---")
    st.markdown("### 💡 Quick Insights")
    
    # Calculate quick insights
    contributions = mmm.get_contributions(X)
    top_channel = contributions[contributions['channel'] != 'base'].nlargest(1, 'contribution')
    
    if not top_channel.empty:
        st.success(f"**Top Channel**: {top_channel.iloc[0]['channel']}")
        st.write(f"ROAS: {top_channel.iloc[0]['roas']:.2f}")
        st.write(f"Contribution: {format_currency(top_channel.iloc[0]['contribution'])}")

# ============================================================================
# MAIN CONTENT: TABS
# ============================================================================

# ============================================================================
# MAIN CONTENT: TABS
# ============================================================================

# Check if hierarchical model with multiple segments
is_hierarchical = hasattr(mmm, 'analysis_level') and mmm.analysis_level != 'global'

tabs_list = [
    "📊 Model Performance",
    "💰 Channel Contributions",
    "📈 Time Series Decomposition",
    "📉 Response Curves",
    "🔍 Model Diagnostics"
]

if is_hierarchical:
    tabs_list.insert(0, "🌍 Market Comparison")

tabs = st.tabs(tabs_list)

if is_hierarchical:
    market_tab = tabs[0]
    perf_tab = tabs[1]
    contrib_tab = tabs[2]
    decomp_tab = tabs[3]
    curve_tab = tabs[4]
    diag_tab = tabs[5]
else:
    perf_tab = tabs[0]
    contrib_tab = tabs[1]
    decomp_tab = tabs[2]
    curve_tab = tabs[3]
    diag_tab = tabs[4]

# ============================================================================
# TAB: MARKET COMPARISON (NEW)
# ============================================================================

if is_hierarchical:
    with market_tab:
        st.markdown("## 🌍 Market Comparison")
        st.markdown("Compare performance across different markets/segments")
        
        # Comparison table
        st.markdown("### Segment Performance Comparison")
        comparison_df = mmm.compare_segments(df, metric='roas')
        
        # Format for display
        display_comp = comparison_df.copy()
        display_comp['roas'] = display_comp['roas'].apply(lambda x: f"{x:.2f}")
        display_comp['spend'] = display_comp['spend'].apply(lambda x: format_currency(x, decimals=0))
        display_comp['revenue'] = display_comp['revenue'].apply(lambda x: format_currency(x, decimals=0))
        
        st.dataframe(display_comp, use_container_width=True, hide_index=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Visualization: ROAS by market (bar chart)
            import plotly.express as px
            fig_roas = px.bar(comparison_df, x='segment', y='roas', 
                         color='roas', color_continuous_scale='RdYlGn',
                         title='ROAS by Market Segment')
            st.plotly_chart(fig_roas, use_container_width=True)
            
        with col2:
            # Channel performance heatmap
            # We need to construct heatmap data: Rows=Markets, Cols=Channels, Values=ROAS
            
            heatmap_data = []
            segments = mmm.models.keys()
            
            for seg in segments:
                if seg == 'global': continue
                contribs = mmm.models[seg].get_contributions(df[df['country'] == seg] if mmm.analysis_level == 'country' else df) 
                # Note: The df filtering above is a bit simplistic, ideally get_contributions handles it if we pass the right df subset
                # But mmm.models[seg] is a RidgeMMM trained on that segment.
                # We should probably use the helper in HierarchicalMMM but it aggregates.
                # Let's manually extract from models for the heatmap.
                
                # Actually, we can just iterate and call get_contributions on the subset
                # But we don't have the subset easily here without filtering df again.
                # Let's try to use the model's internal data if possible? No.
                # Let's filter df based on segment.
                
                seg_df = pd.DataFrame()
                if mmm.analysis_level == 'country':
                    seg_df = df[df['country'] == seg]
                elif mmm.analysis_level == 'os':
                    seg_df = df[df['os'] == seg]
                elif mmm.analysis_level == 'country_os':
                    # This is harder to filter without the composite key
                    c, o = seg.split('_')
                    seg_df = df[(df['country'] == c) & (df['os'] == o)]
                
                if not seg_df.empty:
                    c_df = mmm.models[seg].get_contributions(seg_df)
                    for _, row in c_df.iterrows():
                        if row['channel'] != 'base':
                            heatmap_data.append({
                                'Market': seg,
                                'Channel': row['channel'],
                                'ROAS': row['roas']
                            })
            
            if heatmap_data:
                heatmap_df = pd.DataFrame(heatmap_data)
                # Pivot for imshow
                heatmap_pivot = heatmap_df.pivot(index='Market', columns='Channel', values='ROAS')
                
                fig_heatmap = px.imshow(heatmap_pivot, 
                                text_auto='.2f',
                                color_continuous_scale='RdYlGn',
                                title="Channel ROAS Heatmap")
                st.plotly_chart(fig_heatmap, use_container_width=True)

        # Cross-market insights
        st.markdown("### 🔍 Cross-Market Insights")
        insights = mmm.get_cross_market_insights()
        
        if insights:
            col1, col2 = st.columns(2)
            with col1:
                st.info("**Best Performing Channels:**")
                for mkt, ch in insights.get('best_channels', {}).items():
                    st.write(f"- **{mkt}**: {ch}")
            
            with col2:
                # Placeholder for other insights
                pass
        else:
            st.info("No insights available.")

# ============================================================================
# TAB 1: MODEL PERFORMANCE
# ============================================================================

with perf_tab:
    st.markdown("## 📊 Model Performance")
    st.markdown("마그데이터에 모델이 얼마나 잘 맞는지 평가하세요")
    
    # Display metrics
    st.markdown("### Training Set Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "R² Score",
            f"{train_metrics['r2']:.4f}",
            help="결정계수 (0-1, 높을수록 좋음)"
        )
    
    with col2:
        st.metric(
            "MAPE",
            f"{train_metrics['mape']:.2f}%",
            help="평균 절대 백분율 오차 (낮을수록 좋음)"
        )
    
    with col3:
        st.metric(
            "MAE",
            format_currency(train_metrics['mae'], decimals=0),
            help="평균 절대 오차"
        )
    
    with col4:
        st.metric(
            "RMSE",
            format_currency(train_metrics['rmse'], decimals=0),
            help="평균 제곱근 오차"
        )
    
    st.markdown("### Test Set Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_r2 = test_metrics['r2'] - train_metrics['r2']
        st.metric(
            "R² Score",
            f"{test_metrics['r2']:.4f}",
            delta=f"{delta_r2:.4f}",
            delta_color="normal"
        )
    
    with col2:
        delta_mape = test_metrics['mape'] - train_metrics['mape']
        st.metric(
            "MAPE",
            f"{test_metrics['mape']:.2f}%",
            delta=f"{delta_mape:.2f}%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "MAE",
            format_currency(test_metrics['mae'], decimals=0)
        )
    
    with col4:
        st.metric(
            "RMSE",
            format_currency(test_metrics['rmse'], decimals=0)
        )
    
    # Interpretation
    if test_metrics['r2'] > 0.8:
        st.success("✅ 우수한 모델 적합도! R² > 0.8은 강력한 예측력을 나타냅니다.")
    elif test_metrics['r2'] > 0.6:
        st.info("✓ 양호한 모델 적합도. R² > 0.6은 합리적인 예측력을 나타냅니다.")
    else:
        st.warning("⚠️ 모델 적합도를 개선할 수 있습니다. 변환 파라미터를 조정하거나 기능을 추가하는 것을 고려하세요.")
    
    st.markdown("---")
    
    # Actual vs Predicted plot
    st.markdown("### Actual vs Predicted Revenue")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Combine train and test for full view
        y_all = np.concatenate([y_train, y_test])
        y_pred_all = np.concatenate([y_train_pred, y_test_pred])

        fig_actual_pred = cached_plot_actual_vs_predicted(y_all, y_pred_all)
        st.plotly_chart(fig_actual_pred, use_container_width=True)
    
    with col2:
        st.markdown("#### 해석")
        st.markdown("""
        **확인할 사항:**
        - 빨간 선에 가까운 점들은 정확한 예측을 나타냄
        - 색상 강도는 예측 오차의 크기를 보여줌
        - 체계적인 편차는 모델 편향을 시사함
        
        **좋은 신호:**
        - 점들이 선 주변에 고르게 분포
        - 잔차에 명확한 패턴이 없음
        - R²가 1.0에 가까움
        """)
    
    st.markdown("---")
    
    # Residual Analysis
    st.markdown("### Residual Analysis")

    fig_residuals = cached_plot_residuals(y_all, y_pred_all)
    st.plotly_chart(fig_residuals, use_container_width=True)
    
    st.info("""
    **잔차 플롯으로 식별할 수 있는 것:**
    - **왼쪽 플롯**: 패턴을 확인하세요. 무작위 산점이 좋으며, 패턴은 모델 문제를 시사합니다.
    - **오른쪽 플롯**: 정규 분포와 유사해야 합니다. 편차는 비정규 오차를 시사합니다.
    """)

# ============================================================================
# TAB 2: CHANNEL CONTRIBUTIONS
# ============================================================================

with contrib_tab:
    st.markdown("## 💰 Channel Contributions")
    st.markdown("각 채널이 수익에 어떻게 기여하는지 이해하세요")
    
    # Get contributions
    contributions = mmm.get_contributions(X)
    
    # Display contributions table
    st.markdown("### Contribution Summary")
    
    # Format table
    display_df = contributions.copy()
    display_df['spend'] = display_df['spend'].apply(lambda x: format_currency(x, decimals=0))
    display_df['contribution'] = display_df['contribution'].apply(lambda x: format_currency(x, decimals=0))
    display_df['roas'] = display_df['roas'].apply(lambda x: f"{x:.2f}")
    display_df['contribution_pct'] = display_df['contribution_pct'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    # Waterfall chart
    st.markdown("### Revenue Decomposition (Waterfall)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_waterfall = cached_plot_waterfall(contributions)
        st.plotly_chart(fig_waterfall, use_container_width=True)
    
    with col2:
        st.markdown("#### 차트 읽는 방법")
        st.markdown("""
        **워터폴 차트가 보여주는 것:**
        - **Base**: 마케팅 없이 발생하는 수익
        - **각 채널**: 증분 기여도
        - **Total**: 모든 구성 요소의 합계
        
        **인사이트:**
        - 막대가 높을수록 = 영향력이 큼
        - 채널 높이 비교
        - Base는 자연 발생 수익을 보여줌
        """)
    
    st.markdown("---")
    
    # Horizontal bar chart
    st.markdown("### Channel Contributions (Ranked)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_bars = cached_plot_channel_contributions(contributions)
        st.plotly_chart(fig_bars, use_container_width=True)
    
    with col2:
        st.markdown("#### ROAS 색상 코드")
        st.markdown("""
        - 🟢 **녹색**: ROAS ≥ 3.0 (우수)
        - 🟡 **주황색**: 2.0 ≤ ROAS < 3.0 (양호)
        - 🔴 **빨간색**: ROAS < 2.0 (개선 필요)
        
        **ROAS** = 광고 투자 대비 수익률
        
        ROAS가 높을수록 효율성이 좋습니다.
        """)
    
    # Key insights
    st.markdown("---")
    st.markdown("### 💡 Key Insights")
    
    # Calculate insights
    channel_contribs = contributions[contributions['channel'] != 'base']
    total_spend = channel_contribs['spend'].sum()
    total_contribution = channel_contribs['contribution'].sum()
    overall_roas = total_contribution / total_spend if total_spend > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Marketing Spend",
            format_currency(total_spend, decimals=0)
        )
    
    with col2:
        st.metric(
            "Total Marketing Contribution",
            format_currency(total_contribution, decimals=0)
        )
    
    with col3:
        st.metric(
            "Overall ROAS",
            f"{overall_roas:.2f}"
        )

# ============================================================================
# TAB 3: TIME SERIES DECOMPOSITION
# ============================================================================

with decomp_tab:
    st.markdown("## 📈 Time Series Decomposition")
    st.markdown("각 채널이 시간에 따라 어떻게 기여했는지 확인하세요")
    
    # Get decomposition
    decomp = mmm.decompose_timeseries(X, y)
    
    # Date range filter
    if 'date' in decomp.columns:
        st.markdown("### Filter by Date Range")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_date = pd.to_datetime(decomp['date']).min()
            max_date = pd.to_datetime(decomp['date']).max()
            
            start_date = st.date_input(
                "Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )
        
        # Filter decomposition
        decomp_filtered = decomp[
            (pd.to_datetime(decomp['date']) >= pd.to_datetime(start_date)) &
            (pd.to_datetime(decomp['date']) <= pd.to_datetime(end_date))
        ]
    else:
        decomp_filtered = decomp
    
    st.markdown("---")
    
    # Decomposition chart
    st.markdown("### Revenue Decomposition Over Time")

    fig_decomp = cached_plot_decomposition(decomp_filtered)
    st.plotly_chart(fig_decomp, use_container_width=True)
    
    st.info("""
    **차트 읽는 방법:**
    - 각 색상 영역은 채널의 기여도를 나타냅니다
    - 스택은 총 예측 수익을 보여줍니다
    - 검은색 점선은 실제 수익을 보여줍니다
    - 범례 항목을 클릭하여 채널 표시/숨기기
    """)
    
    st.markdown("---")
    
    # Weekly data table
    st.markdown("### Weekly Breakdown")
    
    # Format table for display
    display_decomp = decomp_filtered.copy()
    
    # Format numeric columns
    numeric_cols = display_decomp.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        display_decomp[col] = display_decomp[col].apply(lambda x: f"{x:,.0f}")
    
    st.dataframe(
        display_decomp,
        use_container_width=True,
        height=400
    )

# ============================================================================
# TAB 4: RESPONSE CURVES
# ============================================================================

with curve_tab:
    st.markdown("## 📉 Response Curves")
    st.markdown("채널 간 지출 배분을 최적화하세요")
    
    st.info("""
    **반응 곡선이 보여주는 것:**
    - 각 채널의 지출을 변경할 때 수익이 어떻게 변하는지
    - 🔴 **빨간 점**: 현재 지출 수준
    - 🟠 **주황 다이아몬드**: 포화 지점 (한계 ROAS < 1)
    - 곡선 형태는 수익 체감을 나타냅니다
    """)
    
    # Response curves
    fig_curves = cached_plot_response_curves(mmm, X, media_columns)
    st.plotly_chart(fig_curves, use_container_width=True)
    
    st.markdown("---")
    
    # Optimization recommendations
    st.markdown("### 💡 Optimization Recommendations")
    
    st.markdown("#### Current Spend vs Optimal Spend")
    
    # Calculate optimal spend for each channel
    optimization_data = []
    
    for channel in media_columns:
        current_spend = X[channel].mean()
        
        # Generate response curve
        budget_range = np.linspace(0, current_spend * 2, 100)
        curve_df = mmm.get_response_curve(channel, budget_range, X)
        
        # Find optimal spend (where marginal ROAS = 2.0, or closest)
        target_roas = 2.0
        optimal_idx = (curve_df['marginal_roas'] - target_roas).abs().idxmin()
        optimal_spend = curve_df.loc[optimal_idx, 'spend']
        optimal_revenue = curve_df.loc[optimal_idx, 'revenue']
        
        # Current revenue
        current_idx = (curve_df['spend'] - current_spend).abs().idxmin()
        current_revenue = curve_df.loc[current_idx, 'revenue']
        current_marginal_roas = curve_df.loc[current_idx, 'marginal_roas']
        
        optimization_data.append({
            'Channel': channel,
            'Current Spend': current_spend,
            'Optimal Spend': optimal_spend,
            'Spend Change': optimal_spend - current_spend,
            'Spend Change %': ((optimal_spend - current_spend) / current_spend * 100) if current_spend > 0 else 0,
            'Current Marginal ROAS': current_marginal_roas,
            'Revenue Impact': optimal_revenue - current_revenue
        })
    
    opt_df = pd.DataFrame(optimization_data)
    
    # Format for display
    display_opt = opt_df.copy()
    display_opt['Current Spend'] = display_opt['Current Spend'].apply(lambda x: format_currency(x, decimals=0))
    display_opt['Optimal Spend'] = display_opt['Optimal Spend'].apply(lambda x: format_currency(x, decimals=0))
    display_opt['Spend Change'] = display_opt['Spend Change'].apply(lambda x: format_currency(x, decimals=0))
    display_opt['Spend Change %'] = display_opt['Spend Change %'].apply(lambda x: f"{x:+.1f}%")
    display_opt['Current Marginal ROAS'] = display_opt['Current Marginal ROAS'].apply(lambda x: f"{x:.2f}")
    display_opt['Revenue Impact'] = display_opt['Revenue Impact'].apply(lambda x: format_currency(x, decimals=0))
    
    st.dataframe(display_opt, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **해석:**
    - **양수 지출 변화**: 이 채널의 지출 증가
    - **음수 지출 변화**: 이 채널의 지출 감소
    - **수익 영향**: 최적화로 인한 예상 수익 변화
    """)

# ============================================================================
# TAB 5: MODEL DIAGNOSTICS
# ============================================================================

with diag_tab:
    st.markdown("## 🔍 Model Diagnostics")
    st.markdown("모델 가정을 검증하기 위한 고급 진단")
    
    # Calculate residuals
    y_all = np.concatenate([y_train, y_test])
    y_pred_all = np.concatenate([y_train_pred, y_test_pred])
    residuals = y_all - y_pred_all
    
    # Q-Q Plot
    st.markdown("### Q-Q Plot (Normality Check)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_qq = cached_plot_qq(residuals)
        st.plotly_chart(fig_qq, use_container_width=True)
    
    with col2:
        st.markdown("#### 해석")
        st.markdown("""
        **Q-Q 플롯은 잔차가 정규 분포를 따르는지 확인합니다.**
        
        **좋은 신호:**
        - 점들이 빨간 선을 밀접하게 따름
        - 체계적인 편차가 없음
        
        **경고 신호:**
        - S자 형태의 패턴 (두꺼운 꼬리)
        - 선으로부터 체계적인 편차
        """)
    
    st.markdown("---")
    
    # Statistical tests
    st.markdown("### Statistical Tests")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Shapiro-Wilk test for normality
        from scipy.stats import shapiro
        stat, p_value = shapiro(residuals[:min(5000, len(residuals))])  # Limit sample size
        
        st.metric(
            "Shapiro-Wilk 검정",
            f"p = {p_value:.4f}",
            help="잔차가 정규 분포를 따르는지 검정합니다. p > 0.05이면 정규성을 시사합니다."
        )
        
        if p_value > 0.05:
            st.success("✅ 잔차가 정규 분포를 따르는 것으로 보임")
        else:
            st.warning("⚠️ 잔차가 정규 분포를 따르지 않을 수 있음")
    
    with col2:
        # Mean of residuals (should be close to 0)
        mean_residual = residuals.mean()
        
        st.metric(
            "평균 잔차",
            f"{mean_residual:,.2f}",
            help="0에 가까워야 합니다. 큰 값은 편향을 시사합니다."
        )
        
        if abs(mean_residual) < residuals.std() * 0.1:
            st.success("✅ 편향되지 않은 예측")
        else:
            st.warning("⚠️ 모델이 편향될 수 있음")
    
    with col3:
        # Std of residuals
        std_residual = residuals.std()
        
        st.metric(
            "잔차 표준편차",
            format_currency(std_residual, decimals=0),
            help="예측 불확실성의 척도"
        )
    
    st.markdown("---")
    
    # Model coefficients
    st.markdown("### Model Coefficients")
    
    summary = mmm.get_model_summary()
    coef_df = pd.DataFrame([
        {'Feature': k, 'Coefficient': v}
        for k, v in summary['coefficients'].items()
    ])
    
    # Sort by absolute coefficient
    coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values('Abs_Coef', ascending=False)
    coef_df = coef_df.drop('Abs_Coef', axis=1)
    
    # Format
    coef_df['Coefficient'] = coef_df['Coefficient'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(coef_df, use_container_width=True, hide_index=True)
    
    st.info("""
    **계수가 나타내는 것:**
    - 각 기능이 수익에 미치는 영향 (스케일링 및 변환 후)
    - 절대값이 클수록 = 영향력이 강함
    - 부호는 효과의 방향을 나타냄
    """)
    
    st.markdown("---")
    
    # Overall assessment
    st.markdown("### 🎯 Overall Model Assessment")
    
    # Calculate assessment score
    assessment_score = 0
    issues = []
    strengths = []
    
    # Check R²
    if test_metrics['r2'] > 0.8:
        assessment_score += 2
        strengths.append("우수한 예측력 (R² > 0.8)")
    elif test_metrics['r2'] > 0.6:
        assessment_score += 1
        strengths.append("양호한 예측력 (R² > 0.6)")
    else:
        issues.append("낮은 R²는 부적합을 시사함")
    
    # Check MAPE
    if test_metrics['mape'] < 10:
        assessment_score += 2
        strengths.append("매우 정확한 예측 (MAPE < 10%)")
    elif test_metrics['mape'] < 20:
        assessment_score += 1
        strengths.append("합리적인 정확도 (MAPE < 20%)")
    else:
        issues.append("높은 MAPE는 큰 예측 오차를 시사함")
    
    # Check residual normality
    if p_value > 0.05:
        assessment_score += 1
        strengths.append("잔차가 정규 분포를 따름")
    else:
        issues.append("잔차가 정규 분포를 따르지 않을 수 있음")
    
    # Check bias
    if abs(mean_residual) < residuals.std() * 0.1:
        assessment_score += 1
        strengths.append("예측이 편향되지 않음")
    else:
        issues.append("모델이 일부 편향을 보임")
    
    # Display assessment
    if assessment_score >= 5:
        st.success("✅ **우수한 모델 품질**")
    elif assessment_score >= 3:
        st.info("✓ **양호한 모델 품질**")
    else:
        st.warning("⚠️ **모델 개선 필요**")
    
    if strengths:
        st.markdown("**강점:**")
        for strength in strengths:
            st.markdown(f"- ✅ {strength}")
    
    if issues:
        st.markdown("**개선 영역:**")
        for issue in issues:
            st.markdown(f"- ⚠️ {issue}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem 0;">
    <p>Ridge MMM Results Dashboard • Built with Streamlit & Plotly</p>
</div>
""", unsafe_allow_html=True)
