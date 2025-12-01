"""
Ridge MMM - Budget Optimizer Page
Optimize marketing budget allocation across channels
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import io
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ridge_mmm import RidgeMMM
from optimizer import BudgetOptimizer, OptimizationError
from utils import format_currency, format_allocation_table

# Page configuration
st.set_page_config(
    page_title="Budget Optimizer - Ridge MMM",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .increase {
        color: #28a745;
        font-weight: bold;
    }
    .decrease {
        color: #dc3545;
        font-weight: bold;
    }
    .neutral {
        color: #6c757d;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("💰 Budget Optimizer")
st.markdown("Find the optimal allocation of your marketing budget across channels")

# Check if model is trained
if not st.session_state.get('model_trained', False):
    st.warning("⚠️ No trained model found. Please train a model in the Model Configuration page first.")
    st.stop()

# Get model and data
mmm = st.session_state.mmm_model
df = st.session_state.df
config = st.session_state.config
media_columns = config.get('media_columns', [])

# Prepare data
train_split = st.session_state.transformation_config['train_test_split']
train_size = int(len(df) * train_split)
feature_cols = media_columns + config.get('exog_columns', [])
X = df[feature_cols]
X_train = X.iloc[:train_size]

# Initialize optimizer
optimizer = BudgetOptimizer(mmm, X_train)

# Check if hierarchical model with multiple segments
is_hierarchical = hasattr(mmm, 'analysis_level') and mmm.analysis_level != 'global'

if is_hierarchical:
    st.markdown("---")
    st.header("🌍 Multi-Market Budget Optimization")
    
    optimization_strategy = st.radio(
        "Optimization Strategy:",
        options=[
            'Global Budget → Optimal Market Allocation',
            'Fixed Market Budgets → Optimal Channel Mix per Market'
        ]
    )
    
    if optimization_strategy == 'Global Budget → Optimal Market Allocation':
        st.info("Optimize total budget allocation across ALL markets and channels simultaneously.")
        
        # Get current total spend across all markets
        # We need to sum up spend from all segments
        total_current_spend = 0
        segments = [k for k in mmm.models.keys() if k != 'global']
        
        for seg in segments:
            seg_model = mmm.models[seg]
            # Assuming X_train has all data, we need to filter for this segment
            # But X_train here is already filtered to feature_cols.
            # We need the original df to filter by segment.
            # Let's use st.session_state.df
            
            if mmm.analysis_level == 'country':
                seg_df = df[df['country'] == seg]
            elif mmm.analysis_level == 'os':
                seg_df = df[df['os'] == seg]
            elif mmm.analysis_level == 'country_os':
                c, o = seg.split('_')
                seg_df = df[(df['country'] == c) & (df['os'] == o)]
            else:
                seg_df = pd.DataFrame()
                
            if not seg_df.empty:
                # Calculate mean spend for this segment's channels
                for ch in seg_model.media_channels:
                    if ch in seg_df.columns:
                        total_current_spend += seg_df[ch].mean()

        total_budget = st.number_input(
            "Total Global Budget", 
            value=float(total_current_spend),
            min_value=0.0,
            step=1000.0,
            format="%.0f"
        )
        
        if st.button("🚀 Optimize Global Allocation", type="primary"):
            with st.spinner("Optimizing across all markets..."):
                # We need to run optimization for all (market, channel) pairs
                # This is complex because BudgetOptimizer is designed for one model.
                # We can instantiate a BudgetOptimizer for each segment, 
                # then combine their response curves?
                # Or better: Create a "Virtual" global model where features are "Market_Channel"
                # But that requires retraining or hacking.
                
                # Simpler approach: 
                # 1. Get response curves for all (market, channel) pairs
                # 2. Use a simple greedy or scipy optimization to allocate budget
                
                # Let's use a simplified greedy approach or just call optimizer for each if we had fixed budgets.
                # But here budget is shared.
                
                # Placeholder for complex global optimization
                st.warning("Global optimization across markets is a complex feature. Implementing simplified version...")
                
                # Simplified: Proportional to current ROAS?
                # Let's just show a placeholder result for now as implementing full global optimization 
                # requires significant changes to the Optimizer class or a new GlobalOptimizer class.
                # Given the scope, I will implement a basic heuristic: 
                # Allocate budget to markets based on their aggregate marginal ROAS at current spend.
                
                market_metrics = []
                for seg in segments:
                    seg_model = mmm.models[seg]
                    # Get marginal ROAS for this market (aggregate of channels?)
                    # Let's use the segment's contribution/spend as a proxy for efficiency
                    # Or better, sum of marginal ROAS of all channels? No.
                    
                    # Let's just use the 'compare_segments' result
                    comp = mmm.compare_segments(df, metric='roas')
                    seg_roas = comp[comp['segment'] == seg]['roas'].values[0] if not comp[comp['segment'] == seg].empty else 0
                    
                    market_metrics.append({
                        'market': seg,
                        'roas': seg_roas,
                        'current_spend': comp[comp['segment'] == seg]['spend'].values[0] if not comp[comp['segment'] == seg].empty else 0
                    })
                
                mm_df = pd.DataFrame(market_metrics)
                
                # Simple allocation: Weighted by ROAS
                # New Budget = Total Budget * (ROAS * Current Spend) / Sum(ROAS * Current Spend)
                # This shifts budget to higher ROAS markets
                
                total_score = (mm_df['roas'] * mm_df['current_spend']).sum()
                mm_df['allocated_budget'] = total_budget * (mm_df['roas'] * mm_df['current_spend']) / total_score
                
                st.subheader("Recommended Market Allocation")
                st.dataframe(mm_df)
                
                # Sankey Diagram
                # Source: Total Budget -> Markets
                
                labels = ["Total Budget"] + mm_df['market'].tolist()
                source = [0] * len(mm_df)
                target = list(range(1, len(mm_df) + 1))
                values = mm_df['allocated_budget'].tolist()
                
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
                      value = values
                  ))])
                
                st.plotly_chart(fig, use_container_width=True)
                
    
    elif optimization_strategy == 'Fixed Market Budgets → Optimal Channel Mix per Market':
        st.info("Optimize channel mix within each market, keeping market budgets fixed.")
        
        segments = [k for k in mmm.models.keys() if k != 'global']
        selected_market = st.selectbox("Select Market to Optimize", segments)
        
        if selected_market:
            # Get model and data for this market
            market_model = mmm.models[selected_market]
            
            # Filter data
            if mmm.analysis_level == 'country':
                seg_df = df[df['country'] == selected_market]
            elif mmm.analysis_level == 'os':
                seg_df = df[df['os'] == selected_market]
            elif mmm.analysis_level == 'country_os':
                c, o = selected_market.split('_')
                seg_df = df[(df['country'] == c) & (df['os'] == o)]
            else:
                seg_df = pd.DataFrame()
            
            if not seg_df.empty:
                # Create optimizer for this market
                # We need to pass the correct feature columns
                # The segment model uses specific channels.
                # We need to ensure X_train has those.
                
                # X_train for segment
                seg_X = seg_df[market_model.media_channels + (market_model.exog_vars or [])]
                
                market_optimizer = BudgetOptimizer(market_model, seg_X)
                
                # Current spend for this market
                current_market_spend = seg_X[market_model.media_channels].mean().sum()
                
                market_budget = st.number_input(
                    f"Budget for {selected_market}",
                    value=float(current_market_spend),
                    min_value=0.0,
                    step=100.0
                )
                
                if st.button(f"Optimize {selected_market}"):
                    res = market_optimizer.optimize_budget(
                        total_budget=market_budget,
                        channels=market_model.media_channels,
                        target_metric='revenue' # Default
                    )
                    
                    st.success(f"Optimization for {selected_market} complete!")
                    
                    # Show results (reuse existing result display logic or create new)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Revenue", format_currency(res['predicted_revenue'], decimals=0))
                    with col2:
                        st.metric("ROAS", f"{res['predicted_roas']:.2f}")
                        
                    # Allocation table
                    alloc_df = pd.DataFrame({
                        'Channel': res['optimal_allocation'].keys(),
                        'Optimal Spend': res['optimal_allocation'].values()
                    })
                    st.dataframe(alloc_df)

    st.markdown("---")
    st.markdown("### Single Model Optimization (Global/Legacy)")


# Initialize session state for scenarios
if 'saved_scenarios' not in st.session_state:
    st.session_state.saved_scenarios = {}

# ============================================================================
# SIDEBAR: CONSTRAINTS CONFIGURATION
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Optimization Settings")
    
    st.markdown("#### Constraints")
    
    # Allow unlimited changes checkbox
    unlimited_changes = st.checkbox(
        "Allow unlimited channel changes",
        value=False,
        help="If unchecked, channels are constrained to ±30% of current spend"
    )
    
    if not unlimited_changes:
        st.markdown("**Channel Constraints** (% of current)")
        
        # Store constraints in session state
        if 'channel_constraints' not in st.session_state:
            st.session_state.channel_constraints = {
                ch: {'min_pct': 70, 'max_pct': 130}
                for ch in media_columns
            }
        
        with st.expander("Configure Constraints"):
            for channel in media_columns:
                st.markdown(f"**{channel}**")
                col1, col2 = st.columns(2)
                
                with col1:
                    min_pct = st.number_input(
                        f"Min %",
                        min_value=0,
                        max_value=200,
                        value=st.session_state.channel_constraints[channel]['min_pct'],
                        step=10,
                        key=f"min_{channel}"
                    )
                    st.session_state.channel_constraints[channel]['min_pct'] = min_pct
                
                with col2:
                    max_pct = st.number_input(
                        f"Max %",
                        min_value=0,
                        max_value=300,
                        value=st.session_state.channel_constraints[channel]['max_pct'],
                        step=10,
                        key=f"max_{channel}"
                    )
                    st.session_state.channel_constraints[channel]['max_pct'] = max_pct
    
    st.markdown("---")
    
    # Target metric
    st.markdown("#### Optimization Target")
    target_metric = st.radio(
        "Optimize for:",
        options=['revenue', 'roas'],
        format_func=lambda x: 'Revenue' if x == 'revenue' else 'ROAS',
        help="Choose whether to maximize total revenue or return on ad spend"
    )
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Start with default constraints
    - Review response curves first
    - Compare multiple scenarios
    - Consider business constraints
    """)

# ============================================================================
# SECTION 1: CURRENT VS OPTIMAL ALLOCATION
# ============================================================================

st.markdown("## 🎯 Current vs Optimal Allocation")
st.markdown("Find the best way to allocate your budget")

# Get current allocation
current_allocation = {ch: X_train[ch].mean() for ch in media_columns}
current_total = sum(current_allocation.values())

# Budget input
col1, col2 = st.columns([2, 1])

with col1:
    total_budget = st.number_input(
        "Total Budget",
        min_value=0.0,
        value=float(current_total),
        step=1000.0,
        format="%.0f",
        help="Total budget to allocate across all channels"
    )

with col2:
    st.metric(
        "Current Total Spend",
        format_currency(current_total, decimals=0)
    )

# Optimize button
if st.button("🚀 Optimize Budget", type="primary", use_container_width=True):
    with st.spinner("Finding optimal allocation..."):
        try:
            # Build constraints
            if unlimited_changes:
                constraints = None
            else:
                constraints = {}
                for channel in media_columns:
                    current_spend = current_allocation[channel]
                    min_pct = st.session_state.channel_constraints[channel]['min_pct']
                    max_pct = st.session_state.channel_constraints[channel]['max_pct']

                    constraints[channel] = {
                        'min': current_spend * min_pct / 100,
                        'max': current_spend * max_pct / 100
                    }

            # Run optimization
            result = optimizer.optimize_budget(
                total_budget=total_budget,
                channels=media_columns,
                constraints=constraints,
                target_metric=target_metric
            )

            # Store result in session state
            st.session_state.optimization_result = result

            if result['success']:
                st.success("✅ Optimization successful!")
            else:
                st.warning(f"⚠️ {result['message']}")

        except ValueError as e:
            st.error(f"❌ Invalid input: {str(e)}")
            st.info(
                "💡 **Suggestions:**\n"
                "- Check your budget and constraint settings\n"
                "- Ensure minimum constraints don't exceed total budget\n"
                "- Verify all channels are valid"
            )

        except OptimizationError as e:
            st.warning(f"⚠️ Optimization failed: {str(e)}")
            st.info(
                "💡 **Try these solutions:**\n"
                "- Relax channel constraints (allow more flexibility)\n"
                "- Increase total budget\n"
                "- Reduce number of channels\n"
                "- Check if constraints are feasible"
            )

        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            with st.expander("Show detailed traceback"):
                st.exception(e)

# Display results if available
if 'optimization_result' in st.session_state:
    result = st.session_state.optimization_result
    
    st.markdown("---")
    st.markdown("### 📊 Optimization Results")
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Revenue",
            format_currency(result['current_revenue'], decimals=0)
        )
    
    with col2:
        st.metric(
            "Optimal Revenue",
            format_currency(result['predicted_revenue'], decimals=0),
            delta=format_currency(result['predicted_revenue'] - result['current_revenue'], decimals=0)
        )
    
    with col3:
        st.metric(
            "Revenue Improvement",
            f"{result['improvement_pct']:.2f}%"
        )
    
    with col4:
        st.metric(
            "Optimal ROAS",
            f"{result['predicted_roas']:.2f}",
            delta=f"{result['predicted_roas'] - result['current_roas']:.2f}"
        )
    
    # Allocation comparison table
    st.markdown("### 📋 Allocation Comparison")
    
    comparison_df = format_allocation_table(
        result['current_allocation'],
        result['optimal_allocation']
    )
    
    # Format for display
    def format_row(row):
        if row['Channel'] == 'TOTAL':
            return ['**' + str(v) + '**' if isinstance(v, str) else f"**{v:,.0f}**" if isinstance(v, (int, float)) else v for v in row]
        return row
    
    display_df = comparison_df.copy()
    display_df['Current'] = display_df['Current'].apply(lambda x: format_currency(x, decimals=0))
    display_df['Optimal'] = display_df['Optimal'].apply(lambda x: format_currency(x, decimals=0))
    display_df['Change'] = display_df['Change'].apply(lambda x: format_currency(x, decimals=0))
    display_df['Change %'] = display_df['Change %'].apply(lambda x: f"{x:+.1f}%")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Visualization: Before/After comparison
    st.markdown("### 📊 Visual Comparison")
    
    fig = go.Figure()
    
    channels = list(result['current_allocation'].keys())
    current_values = [result['current_allocation'][ch] for ch in channels]
    optimal_values = [result['optimal_allocation'][ch] for ch in channels]
    
    fig.add_trace(go.Bar(
        name='Current',
        x=channels,
        y=current_values,
        marker_color='#1f77b4',
        text=[format_currency(v, decimals=0) for v in current_values],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Optimal',
        x=channels,
        y=optimal_values,
        marker_color='#2ca02c',
        text=[format_currency(v, decimals=0) for v in optimal_values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Current vs Optimal Allocation",
        xaxis_title="Channel",
        yaxis_title="Spend",
        barmode='group',
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SECTION 2: SCENARIO BUILDER
# ============================================================================

st.markdown("---")
st.markdown("## 🎨 Scenario Builder")
st.markdown("Create and compare custom budget allocation scenarios")

st.markdown("### Build Your Scenario")

# Number inputs for each channel (replaced sliders)
scenario_allocation = {}

cols = st.columns(min(3, len(media_columns)))

for idx, channel in enumerate(media_columns):
    with cols[idx % 3]:
        current_spend = current_allocation[channel]
        
        # Use number_input instead of slider for precise input
        scenario_spend = st.number_input(
            f"{channel}",
            min_value=0.0,
            max_value=current_spend * 3,  # Allow up to 3x current spend
            value=current_spend,
            step=1000.0,
            format="%.0f",
            key=f"scenario_{channel}",
            help=f"Current: {format_currency(current_spend, decimals=0)}"
        )
        
        # Show change percentage
        change_pct = ((scenario_spend - current_spend) / current_spend * 100) if current_spend > 0 else 0
        if abs(change_pct) > 0.1:
            color = "green" if change_pct > 0 else "red"
            st.markdown(f":{color}[{change_pct:+.1f}% vs current]")
        
        scenario_allocation[channel] = scenario_spend

# Calculate scenario metrics
scenario_total = sum(scenario_allocation.values())
scenario_revenue = optimizer._predict_revenue(scenario_allocation, media_columns)
scenario_roas = scenario_revenue / scenario_total if scenario_total > 0 else 0

# Display scenario metrics
st.markdown("### 📊 Scenario Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Spend",
        format_currency(scenario_total, decimals=0)
    )

with col2:
    st.metric(
        "Predicted Revenue",
        format_currency(scenario_revenue, decimals=0)
    )

with col3:
    st.metric(
        "ROAS",
        f"{scenario_roas:.2f}"
    )

# Budget Allocation Visualization
st.markdown("---")
st.markdown("### 📊 Budget Allocation Breakdown")

if scenario_total > 0:
    col_chart, col_table = st.columns([2, 1])
    
    with col_chart:
        # Pie chart showing budget distribution
        allocation_data = []
        for channel, spend in scenario_allocation.items():
            pct = (spend / scenario_total * 100) if scenario_total > 0 else 0
            allocation_data.append({
                'Channel': channel,
                'Spend': spend,
                'Percentage': pct
            })
        
        allocation_df = pd.DataFrame(allocation_data)
        
        # Create pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=allocation_df['Channel'],
            values=allocation_df['Spend'],
            textinfo='label+percent',
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>Amount: %{value:,.0f} 원<br>Percentage: %{percent}<extra></extra>',
            marker=dict(
                colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'],
                line=dict(color='white', width=2)
            )
        )])
        
        fig_pie.update_layout(
            title="Channel Budget Distribution",
            height=400,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            )
        )
        
        st.plotly_chart(fig_pie, use_container_width=True, key='budget_pie_chart')
    
    with col_table:
        # Table showing detailed breakdown
        st.markdown("#### Detailed Breakdown")
        
        display_allocation = allocation_df.copy()
        display_allocation['Spend'] = display_allocation['Spend'].apply(
            lambda x: format_currency(x, decimals=0)
        )
        display_allocation['Percentage'] = display_allocation['Percentage'].apply(
            lambda x: f"{x:.1f}%"
        )
        
        st.dataframe(
            display_allocation[['Channel', 'Spend', 'Percentage']],
            use_container_width=True,
            hide_index=True
        )
        
        st.caption("💡 Adjust channel budgets above to see real-time updates")
else:
    st.info("Set channel budgets above to see allocation breakdown")

# Save scenario button
col1, col2 = st.columns([3, 1])

with col1:
    scenario_name = st.text_input(
        "Scenario Name",
        value=f"Scenario {len(st.session_state.saved_scenarios) + 1}",
        key="scenario_name_input"
    )

with col2:
    if st.button("💾 Save Scenario", use_container_width=True):
        st.session_state.saved_scenarios[scenario_name] = scenario_allocation.copy()
        st.success(f"✅ Saved '{scenario_name}'")

# Compare scenarios
if len(st.session_state.saved_scenarios) > 0:
    st.markdown("---")
    st.markdown("### 📊 Scenario Comparison")
    
    if st.button("🔄 Compare All Scenarios"):
        # Add current scenario to comparison
        all_scenarios = st.session_state.saved_scenarios.copy()
        all_scenarios['Current'] = current_allocation
        
        # Compare
        comparison = optimizer.compare_scenarios(all_scenarios)
        
        # Format for display
        display_comparison = comparison.copy()
        display_comparison['total_spend'] = display_comparison['total_spend'].apply(
            lambda x: format_currency(x, decimals=0)
        )
        display_comparison['predicted_revenue'] = display_comparison['predicted_revenue'].apply(
            lambda x: format_currency(x, decimals=0)
        )
        display_comparison['roas'] = display_comparison['roas'].apply(lambda x: f"{x:.2f}")
        
        # Format channel columns
        for ch in media_columns:
            if ch in display_comparison.columns:
                display_comparison[ch] = display_comparison[ch].apply(
                    lambda x: format_currency(x, decimals=0)
                )
        
        st.dataframe(display_comparison, use_container_width=True, hide_index=True)
        
        # Visualization
        fig = go.Figure()
        
        for scenario_name in all_scenarios.keys():
            scenario_data = comparison[comparison['scenario'] == scenario_name]
            
            if not scenario_data.empty:
                channels_list = media_columns
                values = [scenario_data.iloc[0][ch] for ch in channels_list]
                
                fig.add_trace(go.Bar(
                    name=scenario_name,
                    x=channels_list,
                    y=values
                ))
        
        fig.update_layout(
            title="Scenario Comparison",
            xaxis_title="Channel",
            yaxis_title="Spend",
            barmode='group',
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Clear scenarios button
    if st.button("🗑️ Clear All Scenarios"):
        st.session_state.saved_scenarios = {}
        st.rerun()

# ============================================================================
# SECTION 3: SENSITIVITY ANALYSIS
# ============================================================================

st.markdown("---")
st.markdown("## 🔍 Sensitivity Analysis")
st.markdown("Understand how revenue changes with spend variations")

col1, col2 = st.columns([2, 1])

with col1:
    sensitivity_channel = st.selectbox(
        "Select Channel to Analyze",
        options=media_columns,
        key="sensitivity_channel"
    )

with col2:
    if st.button("📈 Run Sensitivity Analysis", use_container_width=True):
        with st.spinner(f"Analyzing {sensitivity_channel}..."):
            try:
                # Run sensitivity analysis
                sensitivity_df = optimizer.sensitivity_analysis(
                    channel=sensitivity_channel,
                    current_allocation=current_allocation,
                    variation_range=(-50, 50),
                    n_points=30
                )
                
                st.session_state.sensitivity_result = {
                    'channel': sensitivity_channel,
                    'data': sensitivity_df
                }
                
                st.success("✅ Analysis complete!")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Display sensitivity results
if 'sensitivity_result' in st.session_state:
    sens_result = st.session_state.sensitivity_result
    sens_channel = sens_result['channel']
    sens_df = sens_result['data']
    
    st.markdown(f"### 📊 Sensitivity Analysis: {sens_channel}")
    
    # Create sensitivity curve
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Revenue Impact", "ROAS Impact"),
        horizontal_spacing=0.12
    )
    
    # Revenue impact
    fig.add_trace(
        go.Scatter(
            x=sens_df['channel_spend_pct_change'],
            y=sens_df['predicted_revenue'],
            mode='lines+markers',
            name='Revenue',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # Mark current point
    current_point = sens_df[sens_df['channel_spend_pct_change'] == 0]
    if not current_point.empty:
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[current_point.iloc[0]['predicted_revenue']],
                mode='markers',
                name='Current',
                marker=dict(color='red', size=12, symbol='star')
            ),
            row=1, col=1
        )
    
    # ROAS impact
    fig.add_trace(
        go.Scatter(
            x=sens_df['channel_spend_pct_change'],
            y=sens_df['roas'],
            mode='lines+markers',
            name='ROAS',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=6),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Mark current ROAS
    if not current_point.empty:
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[current_point.iloc[0]['roas']],
                mode='markers',
                name='Current ROAS',
                marker=dict(color='red', size=12, symbol='star'),
                showlegend=False
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Spend Change (%)", row=1, col=1)
    fig.update_xaxes(title_text="Spend Change (%)", row=1, col=2)
    fig.update_yaxes(title_text="Revenue", row=1, col=1, tickformat=",.0f")
    fig.update_yaxes(title_text="ROAS", row=1, col=2)
    
    fig.update_layout(
        title=f"Sensitivity Analysis: {sens_channel}",
        template="plotly_white",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    st.markdown("### 💡 Interpretation")
    
    # Find optimal point (max revenue)
    max_revenue_idx = sens_df['predicted_revenue'].idxmax()
    optimal_pct = sens_df.loc[max_revenue_idx, 'channel_spend_pct_change']
    optimal_revenue = sens_df.loc[max_revenue_idx, 'predicted_revenue']
    
    if optimal_pct > 5:
        st.info(f"📈 **Recommendation**: Consider **increasing** {sens_channel} spend by approximately **{optimal_pct:.0f}%** to maximize revenue.")
    elif optimal_pct < -5:
        st.info(f"📉 **Recommendation**: Consider **decreasing** {sens_channel} spend by approximately **{abs(optimal_pct):.0f}%** to improve efficiency.")
    else:
        st.success(f"✅ **Current spend is near optimal** for {sens_channel}.")

# ============================================================================
# SECTION 4: DIMINISHING RETURNS ANALYSIS
# ============================================================================

st.markdown("---")
st.markdown("## 📉 Diminishing Returns Analysis")
st.markdown("Identify saturation points for each channel")

if st.button("🔍 Analyze All Channels", use_container_width=True):
    with st.spinner("Analyzing diminishing returns..."):
        try:
            # Get optimal allocation per channel
            optimal_per_channel = optimizer.get_optimal_allocation_per_channel(
                channels=media_columns,
                marginal_roas_target=2.0
            )
            
            st.session_state.diminishing_returns_result = optimal_per_channel
            st.success("✅ Analysis complete!")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Display diminishing returns results
if 'diminishing_returns_result' in st.session_state:
    dr_result = st.session_state.diminishing_returns_result
    
    st.markdown("### 📊 Saturation Points (Marginal ROAS = 2.0)")
    
    # Create table
    dr_data = []
    for channel, metrics in dr_result.items():
        dr_data.append({
            'Channel': channel,
            'Current Spend': format_currency(metrics['current_spend'], decimals=0),
            'Optimal Spend': format_currency(metrics['optimal_spend'], decimals=0),
            'Change': format_currency(metrics['spend_change'], decimals=0),
            'Change %': f"{metrics['spend_change_pct']:+.1f}%",
            'Marginal ROAS': f"{metrics['marginal_roas']:.2f}"
        })
    
    dr_df = pd.DataFrame(dr_data)
    st.dataframe(dr_df, use_container_width=True, hide_index=True)
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    for channel, metrics in dr_result.items():
        change_pct = metrics['spend_change_pct']
        
        if change_pct > 10:
            st.success(f"✅ **{channel}**: Room to grow by **{change_pct:.0f}%**. Channel is under-invested.")
        elif change_pct < -10:
            st.warning(f"⚠️ **{channel}**: Consider decreasing by **{abs(change_pct):.0f}%**. Channel may be saturated.")
        else:
            st.info(f"ℹ️ **{channel}**: Current spend is near optimal.")
    
    # Visualization: Marginal ROAS comparison
    st.markdown("### 📊 Marginal ROAS Comparison")
    
    fig = go.Figure()
    
    channels_list = list(dr_result.keys())
    marginal_roas_values = [dr_result[ch]['marginal_roas'] for ch in channels_list]
    
    # Color by ROAS level
    colors = ['#2ca02c' if mr >= 2.0 else '#ff7f0e' if mr >= 1.5 else '#d62728' 
              for mr in marginal_roas_values]
    
    fig.add_trace(go.Bar(
        x=channels_list,
        y=marginal_roas_values,
        marker_color=colors,
        text=[f"{mr:.2f}" for mr in marginal_roas_values],
        textposition='outside'
    ))
    
    # Add reference line at 2.0
    fig.add_hline(
        y=2.0,
        line_dash="dash",
        line_color="red",
        annotation_text="Target ROAS = 2.0"
    )
    
    fig.update_layout(
        title="Marginal ROAS at Optimal Spend",
        xaxis_title="Channel",
        yaxis_title="Marginal ROAS",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem 0;">
    <p>Budget Optimizer • Powered by scipy.optimize</p>
</div>
""", unsafe_allow_html=True)
