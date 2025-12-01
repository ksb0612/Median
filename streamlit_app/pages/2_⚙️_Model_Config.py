"""
Ridge MMM - Model Configuration Page
Configure transformations and model parameters
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from transformations import AdstockTransformer, WeibullAdstockTransformer, HillTransformer, TransformationPipeline, AdstockType
from ridge_mmm import RidgeMMM

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_hill_parameters(K: float, S: float, channel: str) -> tuple[bool, str]:
    """
    Validate Hill parameters and return warnings if needed.

    Args:
        K: Hill K (scale) parameter
        S: Hill S (shape) parameter
        channel: Channel name for warning messages

    Returns:
        (is_valid, warning_message)
    """
    warnings = []

    # S parameter checks
    if S < 0.3:
        warnings.append(f"⚠️ {channel}: S={S:.1f} is very low. Saturation effect will be weak.")
    elif S > 5.0:
        warnings.append(f"⚠️ {channel}: S={S:.1f} is very high. May cause numerical instability.")
    elif S > 3.0:
        warnings.append(f"💡 {channel}: S={S:.1f} is high. Ensure this matches your business understanding.")

    # K parameter checks
    if K < 0.3:
        warnings.append(f"⚠️ {channel}: K={K:.1f} is very low. Channel effect will be minimal.")
    elif K > 5.0:
        warnings.append(f"⚠️ {channel}: K={K:.1f} is very high. May overestimate channel impact.")

    # Combination checks
    if S > 2.0 and K > 2.0:
        warnings.append(f"⚠️ {channel}: Both S and K are high. Model may be over-parameterized.")

    if warnings:
        return False, "\n".join(warnings)
    return True, ""

# Cached model training function
@st.cache_data(show_spinner=False)
def train_ridge_mmm_model(
    _X_train: pd.DataFrame,
    _y_train: pd.Series,
    _X_test: pd.DataFrame,
    _y_test: pd.Series,
    channel_configs: dict,
    exog_vars: list,
    alpha: float
):
    """
    Train Ridge MMM model with caching for performance.

    Args:
        _X_train: Training features (prefix _ to exclude from hash)
        _y_train: Training target (prefix _ to exclude from hash)
        _X_test: Test features (prefix _ to exclude from hash)
        _y_test: Test target (prefix _ to exclude from hash)
        channel_configs: Channel transformation configs
        exog_vars: Exogenous variables
        alpha: Ridge regularization parameter

    Returns:
        Tuple of (trained_model, train_metrics, test_metrics)
    """
    # Train model
    mmm = RidgeMMM(alpha=alpha)
    mmm.fit(
        _X_train,
        _y_train,
        channel_configs,
        exog_vars=exog_vars if exog_vars else None
    )

    # Evaluate
    train_metrics = mmm.evaluate(_X_train, _y_train)
    test_metrics = mmm.evaluate(_X_test, _y_test)

    return mmm, train_metrics, test_metrics

# Page configuration
st.set_page_config(
    page_title="Model Configuration - Ridge MMM",
    page_icon="⚙️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .config-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("⚙️ Model Configuration")
st.markdown("변환 파라미터 및 모델 설정을 구성하세요")

# Check if data is loaded
if not st.session_state.get('data_uploaded', False):
    st.warning("⚠️ 데이터가 업로드되지 않았습니다. 먼저 Data Upload 페이지로 이동하세요.")
    st.stop()

# Check if config exists
if 'config' not in st.session_state or not st.session_state.config:
    st.error("❌ 설정이 저장되지 않았습니다. Data Upload 페이지로 돌아가서 설정을 저장하세요.")
    st.stop()

# Get data and config
df = st.session_state.df
config = st.session_state.config

# Validate required config fields (check for None specifically, not falsy values)
revenue_col = config.get('revenue_column')
date_col = config.get('date_column')

if revenue_col is None or revenue_col == '':
    st.error("❌ Revenue column이 설정되지 않았습니다. Data Upload 페이지로 돌아가서 revenue column을 선택하고 설정을 저장하세요.")
    st.stop()

if date_col is None or date_col == '':
    st.error("❌ Date column이 설정되지 않았습니다. Data Upload 페이지로 돌아가서 date column을 선택하고 설정을 저장하세요.")
    st.stop()

media_columns = config.get('media_columns', [])
exog_cols = config.get('exog_columns', [])

if len(media_columns) == 0:
    st.error("❌ 미디어 채널이 선택되지 않았습니다. Data Upload 페이지로 돌아가서 최소 하나의 미디어 채널을 선택하세요.")
    st.stop()

# Initialize transformation config in session state if not exists
if 'transformation_config' not in st.session_state:
    st.session_state.transformation_config = {
        'adstock_type': {channel: 'geometric' for channel in media_columns},
        'adstock': {channel: 0.5 for channel in media_columns},
        'weibull_shape': {channel: 2.0 for channel in media_columns},
        'weibull_scale': {channel: 3.0 for channel in media_columns},
        'hill_K': {channel: 1.0 for channel in media_columns},
        'hill_S': {channel: 1.0 for channel in media_columns},
        'ridge_alpha': 1.0,
        'train_test_split': 0.8
    }

# ============================================================================
# SECTION 1: ADSTOCK CONFIGURATION
# ============================================================================

st.markdown("---")
st.markdown("## 📉 Adstock Configuration")
st.info("""
**Adstock**은 광고의 지속 효과를 모델링합니다.

**Adstock Types:**
- **Geometric**: 간단한 지수 감소 (1개 파라미터: 감쇠율)
  - 디지털 채널에 적합 (즉각적이고 일관된 감소)
- **Weibull**: 유연한 S-커브 감소 (2개 파라미터: 형상, 척도)
  - TV/브랜드 캠페인에 적합 (지연된 피크 효과)
""")

# Checkbox for applying same decay to all channels
apply_same_adstock = st.checkbox(
    "모든 채널에 동일한 감쇠율 적용",
    value=False,
    key='apply_same_adstock'
)

if apply_same_adstock:
    # Single slider for all channels
    common_decay = st.slider(
        "모든 채널의 감쇠율",
        min_value=0.0,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="0 (지속 효과 없음)과 0.9 (장기 지속 효과) 사이의 감쇠율"
    )
    
    # Apply to all channels
    for channel in media_columns:
        st.session_state.transformation_config['adstock'][channel] = common_decay
    
    # Show decay curve for common configuration
    with st.expander("📊 View Decay Curve"):
        transformer = AdstockTransformer(decay_rate=common_decay)
        decay_curve = transformer.get_decay_curve(length=20)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(20)),
            y=decay_curve,
            mode='lines+markers',
            name='Decay Effect',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=f"Adstock Decay Curve (decay={common_decay})",
            xaxis_title="Weeks After Initial Spend",
            yaxis_title="Remaining Effect",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True, key='common_adstock_decay_curve')

else:
    # Individual configuration for each channel
    for channel in media_columns:
        st.markdown(f"### 📊 {channel}")

        col_type, col_params = st.columns([1, 3])

        with col_type:
            # Adstock type selection
            adstock_type = st.selectbox(
                "Adstock Type",
                options=['geometric', 'weibull'],
                index=0 if st.session_state.transformation_config['adstock_type'].get(channel, 'geometric') == 'geometric' else 1,
                key=f'adstock_type_{channel}',
                help="Choose transformation type"
            )
            st.session_state.transformation_config['adstock_type'][channel] = adstock_type

        with col_params:
            if adstock_type == 'geometric':
                # Geometric adstock slider
                decay_rate = st.slider(
                    "Decay Rate",
                    min_value=0.0,
                    max_value=0.9,
                    value=st.session_state.transformation_config['adstock'].get(channel, 0.5),
                    step=0.05,
                    key=f"adstock_{channel}",
                    help="Fraction of effect carrying to next period (0=no carryover, 0.9=long carryover)"
                )
                st.session_state.transformation_config['adstock'][channel] = decay_rate

                # Show decay curve preview
                with st.expander(f"Preview {channel} geometric decay"):
                    transformer = AdstockTransformer(decay_rate=decay_rate)
                    periods = np.arange(20)
                    weights = transformer.get_decay_curve(length=20)

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=periods,
                        y=weights,
                        name='Decay weights',
                        marker_color='#1f77b4'
                    ))
                    fig.update_layout(
                        title=f"Geometric Decay (rate={decay_rate:.2f})",
                        xaxis_title="Weeks after spend",
                        yaxis_title="Effect weight",
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f'geometric_decay_{channel}')

            else:  # weibull
                col_shape, col_scale = st.columns(2)

                with col_shape:
                    shape = st.slider(
                        "Shape (k)",
                        min_value=0.5,
                        max_value=5.0,
                        value=st.session_state.transformation_config['weibull_shape'].get(channel, 2.0),
                        step=0.1,
                        key=f'weibull_shape_{channel}',
                        help="Controls curve shape:\nk<1: Immediate peak\nk=1: Exponential\nk>1: Delayed peak"
                    )
                    st.session_state.transformation_config['weibull_shape'][channel] = shape

                with col_scale:
                    scale = st.slider(
                        "Scale (λ)",
                        min_value=1.0,
                        max_value=10.0,
                        value=st.session_state.transformation_config['weibull_scale'].get(channel, 3.0),
                        step=0.5,
                        key=f'weibull_scale_{channel}',
                        help="Controls decay duration (higher = longer effect)"
                    )
                    st.session_state.transformation_config['weibull_scale'][channel] = scale

                # Show Weibull curve preview
                with st.expander(f"Preview {channel} Weibull decay"):
                    transformer = WeibullAdstockTransformer(shape=shape, scale=scale)
                    periods = np.arange(20)
                    weights = transformer.get_decay_curve(length=20)

                    peak_lag = transformer.get_peak_lag()
                    half_life = transformer.get_half_life()

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=periods,
                        y=weights,
                        mode='lines+markers',
                        name='Decay curve',
                        fill='tozeroy',
                        line=dict(color='#ff7f0e', width=3),
                        marker=dict(size=6)
                    ))

                    # Add peak line
                    if peak_lag > 0:
                        fig.add_vline(
                            x=peak_lag,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Peak: {peak_lag:.1f} weeks"
                        )

                    fig.update_layout(
                        title=f"Weibull Decay (shape={shape:.1f}, scale={scale:.1f})",
                        xaxis_title="Weeks after spend",
                        yaxis_title="Effect weight",
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f'weibull_decay_{channel}')

                    st.caption(f"📊 Peak effect at week {peak_lag:.1f} | Half-life: {half_life:.1f} weeks")

        st.markdown("---")

# ============================================================================
# SECTION 2: SATURATION CONFIGURATION
# ============================================================================

st.markdown("---")
st.markdown("## 📈 Saturation Configuration")
st.info("""
**Saturation (Hill 함수)**는 수익 체감을 모델링합니다. 지출을 늘릴수록 추가 지출의 
영향력이 감소합니다. 파라미터:
- **K (스케일)**: 최대 효과 수준을 제어
- **S (형상)**: 포화가 발생하는 속도를 제어
""")

# Checkbox for applying same saturation to all channels
apply_same_saturation = st.checkbox(
    "모든 채널에 동일한 포화 설정 적용",
    value=False,
    key='apply_same_saturation'
)

if apply_same_saturation:
    # Common sliders
    col1, col2 = st.columns(2)

    with col1:
        common_K = st.slider(
            "Hill K (Scale) for all channels",
            min_value=0.3,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Scale parameter. Higher values = stronger saturation effect."
        )

    with col2:
        common_S = st.slider(
            "Hill S (Shape) for all channels",
            min_value=0.3,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Shape parameter. Higher values = steeper saturation curve."
        )

    # Apply to all channels
    for channel in media_columns:
        st.session_state.transformation_config['hill_K'][channel] = common_K
        st.session_state.transformation_config['hill_S'][channel] = common_S

    # Real-time validation for common settings
    is_valid, warning = validate_hill_parameters(common_K, common_S, "All channels")
    if not is_valid:
        st.warning(warning)

    # Show saturation curve
    with st.expander("📊 View Saturation Curve"):
        transformer = HillTransformer(K=common_K, S=common_S)
        x_range = np.linspace(0, 100, 100)
        y_curve = transformer.get_response_curve(x_range)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_curve,
            mode='lines',
            name='Saturation',
            line=dict(color='#ff7f0e', width=3)
        ))

        # Add max line
        fig.add_hline(
            y=common_K,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Max = {common_K:.1f}"
        )

        fig.update_layout(
            title=f"Hill Saturation Curve (K={common_K}, S={common_S})",
            xaxis_title="Spend (normalized)",
            yaxis_title="Effect",
            height=400,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True, key='common_saturation_curve')

else:
    # Individual sliders for each channel
    for channel in media_columns:
        st.markdown(f"### {channel}")

        col1, col2 = st.columns(2)

        with col1:
            K_value = st.slider(
                f"Hill K (Scale)",
                min_value=0.3,
                max_value=5.0,
                value=st.session_state.transformation_config['hill_K'].get(channel, 1.0),
                step=0.1,
                key=f"hill_K_{channel}",
                help="Scale parameter. Higher values = stronger saturation effect."
            )
            st.session_state.transformation_config['hill_K'][channel] = K_value

        with col2:
            S_value = st.slider(
                f"Hill S (Shape)",
                min_value=0.3,
                max_value=5.0,
                value=st.session_state.transformation_config['hill_S'].get(channel, 1.0),
                step=0.1,
                key=f"hill_S_{channel}",
                help="Shape parameter. Higher values = steeper saturation curve."
            )
            st.session_state.transformation_config['hill_S'][channel] = S_value

        # Real-time validation
        is_valid, warning = validate_hill_parameters(K_value, S_value, channel)
        if not is_valid:
            st.warning(warning)

        # Show saturation curve preview
        with st.expander(f"Preview {channel} saturation curve"):
            transformer = HillTransformer(K=K_value, S=S_value)
            x_range = np.linspace(0, 100, 100)
            y_curve = transformer.get_response_curve(x_range)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_curve,
                mode='lines',
                name='Saturation',
                line=dict(width=3)
            ))

            # Add max line
            fig.add_hline(
                y=K_value,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Max = {K_value:.1f}"
            )

            fig.update_layout(
                title=f"{channel} Saturation Curve",
                xaxis_title="Spend (normalized)",
                yaxis_title="Effect",
                height=300,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True, key=f'saturation_curve_{channel}')

# ============================================================================
# SECTION 2.5: QUICK PRESET BUTTONS
# ============================================================================

st.markdown("---")
st.subheader("🎯 Quick Presets")
st.info("Apply recommended parameter presets based on common marketing channel types")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📱 Digital (Geometric)", help="Quick decay, moderate saturation - typical for performance marketing", use_container_width=True):
        for channel in media_columns:
            st.session_state.transformation_config['adstock_type'][channel] = 'geometric'
            st.session_state.transformation_config['adstock'][channel] = 0.2
            st.session_state.transformation_config['hill_K'][channel] = 1.0
            st.session_state.transformation_config['hill_S'][channel] = 1.0
        st.success("✅ Applied Digital preset")
        st.rerun()

with col2:
    if st.button("📺 TV (Weibull - Delayed)", help="Delayed peak effect - typical for brand/TV campaigns", use_container_width=True):
        for channel in media_columns:
            st.session_state.transformation_config['adstock_type'][channel] = 'weibull'
            st.session_state.transformation_config['weibull_shape'][channel] = 2.5
            st.session_state.transformation_config['weibull_scale'][channel] = 4.0
            st.session_state.transformation_config['hill_K'][channel] = 1.5
            st.session_state.transformation_config['hill_S'][channel] = 2.0
        st.success("✅ Applied TV preset")
        st.rerun()

with col3:
    if st.button("📻 Radio (Weibull - Quick)", help="Quick rise and decay - typical for radio/influencer", use_container_width=True):
        for channel in media_columns:
            st.session_state.transformation_config['adstock_type'][channel] = 'weibull'
            st.session_state.transformation_config['weibull_shape'][channel] = 1.5
            st.session_state.transformation_config['weibull_scale'][channel] = 2.0
            st.session_state.transformation_config['hill_K'][channel] = 1.0
            st.session_state.transformation_config['hill_S'][channel] = 1.0
        st.success("✅ Applied Radio preset")
        st.rerun()

with col4:
    if st.button("🔄 Reset", help="Reset all parameters to default values", use_container_width=True):
        for channel in media_columns:
            st.session_state.transformation_config['adstock_type'][channel] = 'geometric'
            st.session_state.transformation_config['adstock'][channel] = 0.5
            st.session_state.transformation_config['hill_K'][channel] = 1.0
            st.session_state.transformation_config['hill_S'][channel] = 1.0
        st.success("✅ Reset to defaults")
        st.rerun()

# ============================================================================
# SECTION 2.7: PROPHET BASELINE CONFIGURATION
# ============================================================================

st.markdown("---")
st.markdown("## 📈 Baseline Modeling")

use_prophet = st.checkbox(
    "🔮 Use Prophet for baseline (trend + seasonality)",
    value=False,
    key='use_prophet_baseline',
    help="Prophet automatically models organic growth, trends, and seasonality. "
         "Media effects are then modeled on the residuals."
)

if use_prophet:
    st.info("""
    **Prophet Baseline Benefits:**
    - ✅ Separates organic growth from media effects
    - ✅ Automatically handles trends and seasonality
    - ✅ Better for growing/declining businesses
    - ✅ More accurate media attribution
    - ✅ Accounts for holidays and special events

    **When to use:**
    - Business has clear growth or decline trend
    - Strong seasonal patterns (holidays, quarters)
    - Want to isolate true media incrementality
    - Need to remove confounding time effects
    """)

    col1, col2 = st.columns(2)

    with col1:
        seasonality_mode = st.selectbox(
            "Seasonality Mode",
            options=['multiplicative', 'additive'],
            index=0,
            help="Multiplicative: Seasonality scales with trend (recommended for growing business)\n"
                 "Additive: Seasonality is constant over time"
        )

    with col2:
        trend_flexibility = st.slider(
            "Trend Flexibility",
            min_value=0.001,
            max_value=0.5,
            value=0.05,
            step=0.01,
            format="%.3f",
            help="Higher = trend follows data more closely (0.001 = smooth, 0.5 = very flexible)"
        )

    # Holiday selection
    include_holidays = st.checkbox("Include holidays", value=True)

    if include_holidays:
        holiday_country = st.selectbox(
            "Holiday Country",
            options=['US', 'KR', 'JP', 'TW', 'GB', 'DE', 'FR'],
            index=0,
            help="Select country for holiday calendar (affects national holidays)"
        )

        st.session_state.transformation_config['prophet'] = {
            'enabled': True,
            'seasonality_mode': seasonality_mode,
            'changepoint_prior_scale': trend_flexibility,
            'include_holidays': True,
            'holiday_country': holiday_country
        }

        st.caption(f"📅 Prophet will include {holiday_country} national holidays in baseline model")
    else:
        st.session_state.transformation_config['prophet'] = {
            'enabled': True,
            'seasonality_mode': seasonality_mode,
            'changepoint_prior_scale': trend_flexibility,
            'include_holidays': False
        }

    # Show what Prophet will model
    with st.expander("ℹ️ What does Prophet model?"):
        st.markdown("""
        Prophet will automatically decompose your revenue into:

        **1. Trend Component**
        - Long-term growth or decline
        - Changepoints where growth rate shifts
        - Flexible enough to capture business cycles

        **2. Yearly Seasonality**
        - Annual patterns (Q4 holiday season, summer lulls)
        - Automatically discovers peak and low periods

        **3. Weekly Seasonality**
        - Day-of-week patterns
        - Weekend vs weekday effects

        **4. Holidays** (if enabled)
        - Major national holidays
        - Special events that affect baseline

        After removing these effects, **remaining variance is attributed to media channels**.
        This gives you the true incremental lift from your paid media!
        """)
else:
    st.session_state.transformation_config['prophet'] = {'enabled': False}

    st.info("""
    **Standard Mode (Prophet disabled):**
    - Uses Ridge intercept as baseline
    - Simpler and faster
    - Good for stable business with minimal trends
    """)

# ============================================================================
# SECTION 3: MODEL CONFIGURATION
# ============================================================================

st.markdown("---")
st.markdown("## 🤖 Model Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Ridge Regression Alpha")
    ridge_alpha = st.slider(
        "Regularization strength (alpha)",
        min_value=0.1,
        max_value=10.0,
        value=st.session_state.transformation_config.get('ridge_alpha', 1.0),
        step=0.1,
        help="Higher alpha = more regularization = simpler model. Use log scale mentally: 0.1, 1.0, 10.0"
    )
    st.session_state.transformation_config['ridge_alpha'] = ridge_alpha
    
    st.info(f"""
    **Alpha = {ridge_alpha}**
    - Low (0.1-1.0): Less regularization, fits data closely
    - Medium (1.0-5.0): Balanced regularization
    - High (5.0-10.0): Strong regularization, prevents overfitting
    """)

with col2:
    st.markdown("### Train/Test Split")
    train_split = st.slider(
        "Training data percentage",
        min_value=70,
        max_value=90,
        value=int(st.session_state.transformation_config.get('train_test_split', 0.8) * 100),
        step=5,
        help="Percentage of data used for training (rest for testing)"
    )
    st.session_state.transformation_config['train_test_split'] = train_split / 100
    
    # Calculate actual split
    total_rows = len(df)
    train_rows = int(total_rows * train_split / 100)
    test_rows = total_rows - train_rows
    
    st.info(f"""
    **Split: {train_split}% / {100-train_split}%**
    - Training: {train_rows} rows
    - Testing: {test_rows} rows
    - Total: {total_rows} rows
    """)

# ============================================================================
# SECTION 4: PREVIEW TRANSFORMATIONS
# ============================================================================

st.markdown("---")
st.markdown("## 👀 Preview Transformations")
st.markdown("모델을 학습하기 전에 변환이 데이터에 어떻게 영향을 미치는지 확인하세요.")

# Select channel to preview
preview_channel = st.selectbox(
    "미리볼 채널 선택",
    options=media_columns,
    key='preview_channel'
)

if preview_channel:
    # Get original data
    original_data = df[preview_channel].values

    # Create pipeline for this channel
    adstock_type = st.session_state.transformation_config['adstock_type'][preview_channel]

    config = {
        'adstock_type': adstock_type,
        'hill_K': st.session_state.transformation_config['hill_K'][preview_channel],
        'hill_S': st.session_state.transformation_config['hill_S'][preview_channel]
    }

    if adstock_type == 'weibull':
        config['weibull_shape'] = st.session_state.transformation_config['weibull_shape'][preview_channel]
        config['weibull_scale'] = st.session_state.transformation_config['weibull_scale'][preview_channel]
    else:  # geometric
        config['adstock'] = st.session_state.transformation_config['adstock'][preview_channel]

    channel_config = {preview_channel: config}
    
    try:
        pipeline = TransformationPipeline(channel_config)
        original, adstocked, transformed = pipeline.transform_single_channel(
            preview_channel, 
            original_data
        )
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Week': range(1, len(original) + 1),
            'Original': original,
            'After Adstock': adstocked,
            'Final (Adstock + Saturation)': transformed
        })
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Original Mean",
                f"{original.mean():,.0f}",
                help="Average original spend"
            )
        
        with col2:
            st.metric(
                "After Adstock Mean",
                f"{adstocked.mean():,.0f}",
                delta=f"{((adstocked.mean() - original.mean()) / original.mean() * 100):.1f}%",
                help="Average after adstock transformation"
            )
        
        with col3:
            st.metric(
                "Final Mean",
                f"{transformed.mean():,.2f}",
                delta=f"{((transformed.mean() - original.mean()) / original.mean() * 100):.1f}%",
                help="Average after all transformations"
            )
        
        # Plot comparison
        fig = go.Figure()
        
        # Original data
        fig.add_trace(go.Scatter(
            x=comparison_df['Week'],
            y=comparison_df['Original'],
            mode='lines+markers',
            name='Original',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
        
        # After adstock
        fig.add_trace(go.Scatter(
            x=comparison_df['Week'],
            y=comparison_df['After Adstock'],
            mode='lines+markers',
            name='After Adstock',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=4)
        ))
        
        # Final transformed
        fig.add_trace(go.Scatter(
            x=comparison_df['Week'],
            y=comparison_df['Final (Adstock + Saturation)'],
            mode='lines+markers',
            name='Final (Adstock + Saturation)',
            line=dict(color='#2ca02c', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title=f"{preview_channel} - Transformation Comparison",
            xaxis_title="Week",
            yaxis_title="Value",
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f'transformation_preview_{preview_channel}')
        
        # Show data table in expander
        with st.expander("📊 View Transformation Data"):
            st.dataframe(
                comparison_df.style.format({
                    'Original': '{:,.2f}',
                    'After Adstock': '{:,.2f}',
                    'Final (Adstock + Saturation)': '{:,.2f}'
                }),
                use_container_width=True
            )
        
    except Exception as e:
        st.error(f"Error previewing transformations: {str(e)}")

# ============================================================================
# SECTION 5: GLOBAL VALIDATION SUMMARY
# ============================================================================

st.markdown("---")
st.subheader("🔍 Validation Summary")
st.info("Review all parameter warnings before training the model")

all_valid = True
all_warnings = []

for channel in media_columns:
    K = st.session_state.transformation_config['hill_K'].get(channel, 1.0)
    S = st.session_state.transformation_config['hill_S'].get(channel, 1.0)
    is_valid, warning = validate_hill_parameters(K, S, channel)
    if not is_valid:
        all_valid = False
        all_warnings.append(warning)

if all_valid:
    st.success("✅ All parameters look good!")
else:
    st.warning("⚠️ Some parameters may need adjustment:")
    for warning in all_warnings:
        # Split multi-line warnings and display as separate items
        for line in warning.split('\n'):
            if line.strip():
                st.markdown(f"- {line}")
    st.info("💡 You can still train the model, but consider reviewing these warnings.")

# ============================================================================
# SAVE AND PROCEED
# ============================================================================

st.markdown("---")
st.markdown("## 💾 Save Configuration")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.info("Review your configuration and save before proceeding to model training.")

with col2:
    if st.button("💾 Save Configuration", type="primary", use_container_width=True):
        st.session_state.transformation_config_saved = True
        st.success("✅ Configuration saved successfully!")

with col3:
    train_button_disabled = not st.session_state.get('transformation_config_saved', False)
    
    if st.button(
        "🚀 Train Model", 
        use_container_width=True,
        disabled=train_button_disabled,
        type="secondary"
    ):
        if not st.session_state.get('transformation_config_saved', False):
            st.warning("⚠️ Please save configuration first!")
        else:
            try:
                # Show progress
                with st.spinner("Training model..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Step 1: Prepare data
                    status_text.text("📊 Preparing data...")
                    progress_bar.progress(10)

                    # Get data and configuration from session state
                    config = st.session_state.config
                    date_col = config.get('date_column')
                    revenue_col = config.get('revenue_column')
                    exog_cols = config.get('exog_columns', [])

                    # Validate required configuration
                    if revenue_col is None or revenue_col == '':
                        st.error("❌ Revenue column is not configured. Please go back to Data Upload page and save your configuration.")
                        st.stop()
                    
                    if date_col is None or date_col == '':
                        st.error("❌ Date column is not configured. Please go back to Data Upload page and save your configuration.")
                        st.stop()
                    
                    if revenue_col not in df.columns:
                        st.error(f"❌ Revenue column '{revenue_col}' not found in dataframe. Available columns: {list(df.columns)}")
                        st.stop()

                    # Split train/test
                    train_split = st.session_state.transformation_config['train_test_split']
                    train_size = int(len(df) * train_split)

                    # Prepare features and target
                    feature_cols = media_columns + exog_cols
                    X = df[feature_cols]
                    y = df[revenue_col]

                    X_train = X.iloc[:train_size]
                    y_train = y.iloc[:train_size]
                    X_test = X.iloc[train_size:]
                    y_test = y.iloc[train_size:]

                    progress_bar.progress(30)

                    # Step 2: Build channel configs
                    status_text.text("⚙️ Configuring transformations...")

                    channel_configs = {}
                    for channel in media_columns:
                        adstock_type = st.session_state.transformation_config['adstock_type'][channel]

                        config = {
                            'adstock_type': adstock_type,
                            'hill_K': st.session_state.transformation_config['hill_K'][channel],
                            'hill_S': st.session_state.transformation_config['hill_S'][channel]
                        }

                        if adstock_type == 'weibull':
                            config['weibull_shape'] = st.session_state.transformation_config['weibull_shape'][channel]
                            config['weibull_scale'] = st.session_state.transformation_config['weibull_scale'][channel]
                        else:  # geometric
                            config['adstock'] = st.session_state.transformation_config['adstock'][channel]

                        channel_configs[channel] = config

                    progress_bar.progress(50)

                    # Step 3: Train model
                    alpha = st.session_state.transformation_config['ridge_alpha']
                    prophet_config = st.session_state.transformation_config.get('prophet', {'enabled': False})

                    if prophet_config.get('enabled', False):
                        # Use Prophet-enhanced two-stage MMM
                        status_text.text("🔮 Stage 1/2: Fitting Prophet baseline...")

                        from prophet_baseline import ProphetMMM, create_holiday_dataframe

                        # Prepare holidays if enabled
                        holidays = None
                        if prophet_config.get('include_holidays', False):
                            try:
                                holidays = create_holiday_dataframe(
                                    country=prophet_config.get('holiday_country', 'US'),
                                    years=list(range(2020, 2027))
                                )
                            except Exception as e:
                                st.warning(f"Could not load holidays: {e}")

                        progress_bar.progress(60)

                        # Initialize and fit Prophet MMM
                        mmm = ProphetMMM(
                            ridge_alpha=alpha,
                            prophet_seasonality_mode=prophet_config.get('seasonality_mode', 'multiplicative'),
                            prophet_changepoint_prior=prophet_config.get('changepoint_prior_scale', 0.05)
                        )

                        status_text.text("🔮 Stage 2/2: Fitting Ridge MMM on residuals...")

                        # Fit on training data
                        df_train = df.iloc[:train_size]
                        mmm.fit(
                            df=df_train,
                            date_col=date_col,
                            target_col=revenue_col,
                            media_channels=media_columns,
                            channel_configs=channel_configs,
                            holidays=holidays,
                            exog_vars=exog_cols if exog_cols else None
                        )

                        progress_bar.progress(90)

                        # Evaluate on train and test
                        df_test = df.iloc[train_size:]

                        train_metrics = mmm.evaluate(df_train)
                        test_metrics = mmm.evaluate(df_test)

                        st.session_state['model_type'] = 'prophet_mmm'
                        st.session_state['X_train'] = df_train
                        st.session_state['y_train'] = df_train[revenue_col]
                        st.session_state['data'] = df  # Store full dataframe for decomposition

                    else:
                        # Standard Ridge MMM (cached)
                        status_text.text("🤖 Training Ridge MMM model...")

                        # Use cached training function
                        mmm, train_metrics, test_metrics = train_ridge_mmm_model(
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            channel_configs,
                            exog_cols,
                            alpha
                        )

                        st.session_state['model_type'] = 'ridge_mmm'
                        st.session_state['X_train'] = X_train
                        st.session_state['y_train'] = y_train
                        st.session_state['data'] = df  # Store full dataframe for Model Comparison

                    progress_bar.progress(100)
                    status_text.text("✅ Training complete!")

                    # Store model in session state
                    st.session_state.mmm_model = mmm
                    st.session_state.model = mmm  # For backwards compatibility
                    st.session_state.model_trained = True
                    st.session_state.train_metrics = train_metrics
                    st.session_state.test_metrics = test_metrics
                    
                    # Show success message
                    st.success("🎉 Model trained successfully!")
                    
                    # Display metrics
                    st.markdown("### 📊 Model Performance")
                    
                    col_train, col_test = st.columns(2)
                    
                    with col_train:
                        st.markdown("**Training Set**")
                        st.metric("R² Score", f"{train_metrics['r2']:.4f}")
                        st.metric("MAPE", f"{train_metrics['mape']:.2f}%")
                        st.metric("MAE", f"{train_metrics['mae']:,.0f}")
                        st.metric("RMSE", f"{train_metrics['rmse']:,.0f}")
                    
                    with col_test:
                        st.markdown("**Test Set**")
                        st.metric("R² Score", f"{test_metrics['r2']:.4f}")
                        st.metric("MAPE", f"{test_metrics['mape']:.2f}%")
                        st.metric("MAE", f"{test_metrics['mae']:,.0f}")
                        st.metric("RMSE", f"{test_metrics['rmse']:,.0f}")
                    
                    # Show next steps
                    st.info("✨ Model is ready! You can now analyze contributions and optimize budgets in the Results page.")
                    
            except Exception as e:
                st.error(f"❌ Error training model: {str(e)}")
                st.exception(e)

# Show model status if trained
if st.session_state.get('model_trained', False):
    st.markdown("---")
    st.success("✅ **Model Status**: Trained and ready for analysis")
    
    # Quick model summary
    with st.expander("📋 View Model Summary"):
        mmm = st.session_state.mmm_model
        summary = mmm.get_model_summary()
        
        st.write(f"**Alpha**: {summary['alpha']}")
        st.write(f"**Channels**: {', '.join(summary['channels'])}")
        st.write(f"**Features**: {summary['n_features']}")
        st.write(f"**Intercept**: {summary['intercept']:,.2f}")
        
        st.markdown("**Coefficients**:")
        coef_df = pd.DataFrame([
            {'Feature': k, 'Coefficient': v}
            for k, v in summary['coefficients'].items()
        ])
        st.dataframe(coef_df, use_container_width=True)

# ============================================================================
# SIDEBAR: CONFIGURATION SUMMARY
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration Summary")
    
    if st.session_state.get('transformation_config_saved', False):
        st.success("✅ Configuration saved")
    else:
        st.warning("⚠️ Configuration not saved")
    
    st.markdown("---")
    
    st.markdown("**Adstock Decay Rates:**")
    for channel in media_columns:
        decay = st.session_state.transformation_config['adstock'][channel]
        st.write(f"• {channel}: {decay:.2f}")
    
    st.markdown("**Hill Parameters:**")
    for channel in media_columns:
        K = st.session_state.transformation_config['hill_K'][channel]
        S = st.session_state.transformation_config['hill_S'][channel]
        st.write(f"• {channel}: K={K:.1f}, S={S:.1f}")
    
    st.markdown("---")
    st.markdown("**Model Settings:**")
    st.write(f"• Ridge Alpha: {st.session_state.transformation_config['ridge_alpha']:.1f}")
    st.write(f"• Train/Test: {int(st.session_state.transformation_config['train_test_split']*100)}% / {int((1-st.session_state.transformation_config['train_test_split'])*100)}%")
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Start with default values (0.5 decay, 1.0 K & S)
    - Adjust based on domain knowledge
    - Higher decay = longer carryover
    - Preview transformations before saving
    - Different channels may need different parameters
    """)
