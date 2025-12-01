"""Model comparison page - Ridge MMM vs Robyn MMM."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parents[2]))

from src.robyn_wrapper import RobynWrapper

st.set_page_config(
    page_title="Model Comparison",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Model Comparison: Ridge MMM vs Robyn")

st.markdown("""
Compare your Ridge MMM results with **Meta's Robyn MMM tool** to validate your model.

**Note:** This requires R and the Robyn package to be installed.
""")

# Initialize wrapper
wrapper = RobynWrapper()

# Prerequisites check section
st.subheader("📋 Prerequisites Check")

col1, col2, col3 = st.columns(3)

with col1:
    r_installed = wrapper.check_r_installed()
    if r_installed:
        st.success("✅ R is installed")
        r_version = wrapper.get_r_version()
        if r_version:
            st.caption(r_version)
    else:
        st.error("❌ R not found")
        with st.expander("📦 Installation Instructions"):
            st.markdown("""
            **Windows:**
            ```
            Download from: https://cran.r-project.org/bin/windows/base/
            ```

            **macOS:**
            ```bash
            brew install r
            ```

            **Linux (Ubuntu/Debian):**
            ```bash
            sudo apt-get install r-base
            ```
            """)

with col2:
    robyn_installed = wrapper.check_robyn_installed() if r_installed else False
    if robyn_installed:
        st.success("✅ Robyn package installed")
    else:
        st.error("❌ Robyn not installed")
        if r_installed:
            with st.expander("📦 Installation Instructions"):
                st.markdown("""
                Install Robyn in R:
                ```r
                install.packages('Robyn')
                ```

                Or from command line:
                ```bash
                R -e "install.packages('Robyn')"
                ```
                """)

with col3:
    if r_installed and robyn_installed:
        st.success("✅ Ready to compare")
    else:
        st.warning("⚠️ Setup incomplete")

# Stop if prerequisites not met
if not (r_installed and robyn_installed):
    st.info("👆 Please install the required dependencies above to use this feature.")
    st.stop()

st.markdown("---")

# Check if Ridge model exists
if 'model' not in st.session_state or st.session_state.model is None:
    st.warning("⚠️ Train a Ridge MMM model first")
    st.info("Go to **⚙️ Model Config** page to train your model.")
    st.stop()

ridge_model = st.session_state.model
X = st.session_state.get('X_train')
y = st.session_state.get('y_train')
data = st.session_state.get('data')

if X is None or y is None or data is None:
    st.warning("⚠️ Model training data not found")
    st.info("Please re-train your model on the **⚙️ Model Config** page.")
    st.stop()

# Robyn configuration
st.subheader("⚙️ Robyn Configuration")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    Configure Robyn's optimization parameters. Higher values provide better results but take longer to run.
    """)

with col2:
    st.info(f"**Channels:** {len(ridge_model.media_channels)}")

with st.expander("⚙️ Advanced Settings", expanded=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        iterations = st.number_input(
            "Iterations",
            min_value=500,
            max_value=5000,
            value=2000,
            step=500,
            help="More iterations = better results but slower (recommended: 2000)"
        )

    with col2:
        trials = st.number_input(
            "Trials",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of optimization trials (recommended: 5)"
        )

    with col3:
        cores = st.number_input(
            "CPU Cores",
            min_value=1,
            max_value=8,
            value=4,
            help="Parallel processing cores"
        )

    adstock_type = st.selectbox(
        "Adstock Type",
        options=["geometric", "weibull"],
        index=0,
        help="Geometric: Simple exponential decay | Weibull: More flexible decay curves"
    )

    prophet_country = st.selectbox(
        "Holiday Calendar",
        options=["US", "UK", "DE", "FR", "JP", "KR", "CN"],
        index=0,
        help="Country for Prophet holiday effects"
    )

# Estimated runtime
estimated_minutes = (iterations * trials) / 500  # Rough estimate
st.caption(f"⏱️ Estimated runtime: {estimated_minutes:.0f}-{estimated_minutes*2:.0f} minutes")

# Run comparison button
run_robyn = st.button(
    "🚀 Run Robyn & Compare",
    type="primary",
    help="This will take 5-30 minutes depending on your settings"
)

if run_robyn:
    with st.spinner("Running Robyn MMM... ☕ This is a good time for a coffee break!"):
        try:
            # Get actual column names from config
            user_config = st.session_state.get('config', {})
            date_column = user_config.get('date_column', 'date')
            revenue_column = user_config.get('revenue_column', 'revenue')
            
            # Prepare config
            config = wrapper.prepare_config(
                data=data,
                date_var=date_column,
                dep_var=revenue_column,
                paid_media_vars=ridge_model.media_channels,
                prophet_country=prophet_country,
                adstock=adstock_type,
                iterations=int(iterations),
                trials=int(trials),
                cores=int(cores)
            )

            # Run Robyn
            robyn_results = wrapper.run(data, config, timeout=1800, verbose=False)

            # Store in session
            st.session_state['robyn_results'] = robyn_results
            st.session_state['robyn_config'] = config

            st.success("✅ Robyn MMM complete!")
            st.balloons()

        except Exception as e:
            st.error(f"❌ Robyn execution failed")
            with st.expander("🔍 Error Details"):
                st.code(str(e))
            st.stop()

# Display comparison if available
if 'robyn_results' in st.session_state:
    robyn_results = st.session_state['robyn_results']
    config = st.session_state.get('robyn_config', {})

    st.markdown("---")
    st.header("📊 Comparison Results")

    # Model-level metrics
    st.subheader("📈 Model Performance Metrics")

    metrics_comparison = wrapper.get_model_metrics_comparison(
        robyn_results, ridge_model, X, y
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ridge_r2 = metrics_comparison['ridge']['r_squared']
        robyn_r2 = metrics_comparison['robyn']['rsq_train']
        delta_r2 = ridge_r2 - robyn_r2 if robyn_r2 else 0

        st.metric(
            "R² Score",
            f"{ridge_r2:.3f}" if ridge_r2 else "N/A",
            delta=f"{delta_r2:+.3f} vs Robyn",
            delta_color="normal"
        )
        st.caption("Ridge MMM")

    with col2:
        robyn_r2_display = robyn_r2 if robyn_r2 else 0
        st.metric(
            "R² Score",
            f"{robyn_r2_display:.3f}",
            delta=None
        )
        st.caption("Robyn")

    with col3:
        ridge_mape = metrics_comparison['ridge']['mape']
        robyn_mape = metrics_comparison['robyn']['mape']
        delta_mape = ridge_mape - robyn_mape if (ridge_mape and robyn_mape) else 0

        st.metric(
            "MAPE",
            f"{ridge_mape:.1f}%" if ridge_mape else "N/A",
            delta=f"{delta_mape:+.1f}% vs Robyn",
            delta_color="inverse"
        )
        st.caption("Ridge MMM")

    with col4:
        st.metric(
            "MAPE",
            f"{robyn_mape:.1f}%" if robyn_mape else "N/A",
            delta=None
        )
        st.caption("Robyn")

    st.markdown("---")

    # Channel comparison
    st.subheader("🎯 Channel-by-Channel Comparison")

    # Get comparison DataFrame
    comparison_df = wrapper.compare_with_ridge(
        robyn_results, ridge_model, X, y
    )

    # Format for display
    display_df = comparison_df.copy()

    # Create formatted columns
    display_df['Spend'] = display_df['spend'].apply(lambda x: f"${x:,.0f}")
    display_df['Ridge Contribution'] = display_df['ridge_contribution'].apply(
        lambda x: f"${x:,.0f}"
    )
    display_df['Ridge ROAS'] = display_df['ridge_roas'].apply(lambda x: f"{x:.2f}")
    display_df['Robyn Contribution'] = display_df['robyn_contribution'].apply(
        lambda x: f"${x:,.0f}"
    )
    display_df['Robyn ROAS'] = display_df['robyn_roas'].apply(lambda x: f"{x:.2f}")
    display_df['Δ Contribution'] = display_df['contribution_diff_pct'].apply(
        lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
    )
    display_df['Δ ROAS'] = display_df['roas_diff_pct'].apply(
        lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
    )

    # Select display columns
    display_cols = [
        'channel', 'Spend', 'Ridge Contribution', 'Ridge ROAS',
        'Robyn Contribution', 'Robyn ROAS', 'Δ Contribution', 'Δ ROAS'
    ]

    st.dataframe(
        display_df[display_cols],
        use_container_width=True,
        hide_index=True
    )

    # Visualizations
    st.markdown("---")
    st.subheader("📊 Visual Comparison")

    tab1, tab2, tab3 = st.tabs(["ROAS Comparison", "Contribution Comparison", "Scatter Plot"])

    with tab1:
        # ROAS comparison chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Ridge MMM',
            x=comparison_df['channel'],
            y=comparison_df['ridge_roas'],
            marker_color='#1f77b4',
            text=comparison_df['ridge_roas'].apply(lambda x: f'{x:.2f}'),
            textposition='outside'
        ))

        fig.add_trace(go.Bar(
            name='Robyn',
            x=comparison_df['channel'],
            y=comparison_df['robyn_roas'],
            marker_color='#ff7f0e',
            text=comparison_df['robyn_roas'].apply(lambda x: f'{x:.2f}'),
            textposition='outside'
        ))

        fig.update_layout(
            barmode='group',
            title="ROAS by Channel: Ridge MMM vs Robyn",
            xaxis_title="Channel",
            yaxis_title="ROAS (Revenue / Spend)",
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Contribution comparison chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Ridge MMM',
            x=comparison_df['channel'],
            y=comparison_df['ridge_contribution'],
            marker_color='#1f77b4'
        ))

        fig.add_trace(go.Bar(
            name='Robyn',
            x=comparison_df['channel'],
            y=comparison_df['robyn_contribution'],
            marker_color='#ff7f0e'
        ))

        fig.update_layout(
            barmode='group',
            title="Contribution by Channel: Ridge MMM vs Robyn",
            xaxis_title="Channel",
            yaxis_title="Revenue Contribution ($)",
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Scatter plot: Ridge vs Robyn ROAS
        fig = px.scatter(
            comparison_df,
            x='ridge_roas',
            y='robyn_roas',
            text='channel',
            title="ROAS Agreement: Ridge vs Robyn",
            labels={
                'ridge_roas': 'Ridge MMM ROAS',
                'robyn_roas': 'Robyn ROAS'
            },
            height=600
        )

        # Add diagonal line (perfect agreement)
        max_roas = max(comparison_df['ridge_roas'].max(), comparison_df['robyn_roas'].max())
        fig.add_trace(go.Scatter(
            x=[0, max_roas],
            y=[0, max_roas],
            mode='lines',
            name='Perfect Agreement',
            line=dict(dash='dash', color='gray')
        ))

        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)

    # Interpretation
    st.markdown("---")
    st.subheader("💡 Interpretation & Recommendations")

    # Calculate average difference
    avg_roas_diff = comparison_df['roas_diff_pct'].abs().mean()

    col1, col2 = st.columns([2, 1])

    with col1:
        if pd.isna(avg_roas_diff):
            st.warning("Unable to calculate differences (insufficient data)")
        elif avg_roas_diff < 10:
            st.success(f"**✅ Excellent Agreement** (avg ROAS difference: {avg_roas_diff:.1f}%)")
            st.markdown("""
            Both models show very similar results. Your Ridge MMM is well-validated!

            **Recommendations:**
            - You can confidently use Ridge MMM for budget decisions
            - Both models agree on channel performance rankings
            - Differences are within acceptable tolerance
            """)
        elif avg_roas_diff < 20:
            st.info(f"**⚠️ Moderate Agreement** (avg ROAS difference: {avg_roas_diff:.1f}%)")
            st.markdown("""
            Some discrepancies exist between the models.

            **Possible reasons:**
            - Ridge uses fixed transformation parameters, Robyn auto-tunes
            - Different optimization objectives (Ridge: MSE, Robyn: NRMSE)
            - Robyn includes Prophet seasonality decomposition

            **Recommendations:**
            - Both models provide useful insights
            - Consider the average of both estimates
            - Focus on channels where both agree
            """)
        else:
            st.warning(f"**⚠️ Significant Differences** (avg ROAS difference: {avg_roas_diff:.1f}%)")
            st.markdown("""
            Models disagree substantially. Further investigation needed.

            **Possible reasons:**
            - Data quality issues (outliers, missing values)
            - Inappropriate parameter settings (adstock, saturation)
            - Model assumptions violated

            **Recommendations:**
            - Review data quality carefully
            - Try different Ridge parameters (adstock, saturation)
            - Run incrementality tests to validate
            - Consult with a data scientist
            """)

    with col2:
        st.metric(
            "Average ROAS Difference",
            f"{avg_roas_diff:.1f}%" if not pd.isna(avg_roas_diff) else "N/A",
            help="Average absolute percentage difference in ROAS estimates"
        )

        # Agreement level
        if not pd.isna(avg_roas_diff):
            if avg_roas_diff < 10:
                agreement = "Excellent ✅"
                color = "green"
            elif avg_roas_diff < 20:
                agreement = "Moderate ⚠️"
                color = "orange"
            else:
                agreement = "Poor ⚠️"
                color = "red"

            st.markdown(f"**Agreement Level:**")
            st.markdown(f"<h3 style='color: {color};'>{agreement}</h3>", unsafe_allow_html=True)

    # Download section
    st.markdown("---")
    st.subheader("💾 Download Results")

    col1, col2 = st.columns(2)

    with col1:
        csv_data = comparison_df.to_csv(index=False)
        st.download_button(
            "📥 Download Comparison (CSV)",
            data=csv_data,
            file_name="ridge_vs_robyn_comparison.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        # Download Robyn full results
        json_data = robyn_results
        import json
        st.download_button(
            "📥 Download Robyn Results (JSON)",
            data=json.dumps(json_data, indent=2),
            file_name="robyn_results.json",
            mime="application/json",
            use_container_width=True
        )

    # Additional info
    with st.expander("ℹ️ About Robyn vs Ridge MMM"):
        st.markdown("""
        ### Differences Between Robyn and Ridge MMM

        | Aspect | Ridge MMM | Robyn |
        |--------|-----------|-------|
        | **Method** | Ridge Regression | Ridge + Nevergrad Optimization |
        | **Parameters** | User-specified | Auto-tuned |
        | **Speed** | Fast (1-5s) | Slow (5-30min) |
        | **Seasonality** | Manual (optional) | Prophet (automatic) |
        | **Uncertainty** | Point estimates | Bootstrapped confidence intervals |
        | **Flexibility** | Simple, transparent | Complex, optimized |

        ### When to Use Which
        - **Ridge MMM**: Quick insights, transparent results, parameter control
        - **Robyn**: Comprehensive analysis, automated tuning, uncertainty quantification

        ### Best Practice
        Use both! Ridge for speed and transparency, Robyn for validation and comprehensive analysis.
        """)

else:
    st.info("👆 위의 버튼을 클릭하여 Robyn을 실행하고 Ridge MMM 결과와 비교하세요.")
    st.markdown("""
    ### 이 비교가 수행하는 작업

    1. **Robyn MMM 실행** - Meta의 최첨단 알고리즘을 사용하여 데이터 분석
    2. **채널 기여도 비교** - Ridge와 Robyn 간의 채널 기여도 비교
    3. **ROAS 추정치 비교** - Ridge 모델 검증을 위한 ROAS 비교
    4. **해석 제공** - 일치 및 차이점에 대한 해석 제공
    5. **시각화 생성** - 쉬운 비교를 위한 시각화 생성

    ### 왜 비교해야 할까요?

    - ✅ **검증**: Ridge 모델의 신뢰성 확보
    - ✅ **신뢰도**: 여러 모델의 일치는 신뢰도를 높임
    - ✅ **인사이트**: 동일한 데이터에 대한 다양한 관점
    - ✅ **모범 사례**: MMM 프로젝트의 업계 표준
    """)
