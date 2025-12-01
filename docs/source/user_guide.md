# Ridge MMM User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Data Preparation](#data-preparation)
4. [Model Configuration](#model-configuration)
5. [Understanding Results](#understanding-results)
6. [Budget Optimization](#budget-optimization)
7. [Multi-Market Analysis](#multi-market-analysis)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is Marketing Mix Modeling (MMM)?

Marketing Mix Modeling is a statistical technique that helps you understand:
- **How much does each marketing channel contribute to revenue?**
- **What's the optimal budget allocation across channels?**
- **How do channels saturate (diminishing returns)?**
- **What's the carryover effect (adstock) of advertising?**

### When to Use This Tool

✅ **Use Ridge MMM when:**
- You have 1-3 years of weekly marketing data
- You spend on 2-10 marketing channels
- You need quick, actionable budget recommendations
- You want to compare "what-if" scenarios

❌ **Don't use Ridge MMM when:**
- You have less than 6 months of data (not enough for reliable estimates)
- You need real-time optimization (this is strategic planning tool)
- You have 20+ channels (consider hierarchical Bayesian MMM instead)

---

## Quick Start

### Step 1: Prepare Your Data

Create a CSV file with this structure:
```csv
date,revenue,google_uac,meta,apple_search,tiktok
2023-01-01,15000000,2000000,1500000,1000000,500000
2023-01-08,18000000,2500000,1800000,1200000,600000
...
```

**Requirements:**
- At least 52 weeks of data (104 weeks recommended)
- Weekly aggregation (Monday-Sunday)
- One row per week
- Revenue and spend in same currency

### Step 2: Upload Data

1. Go to **📊 Data Upload** page
2. Click "Browse files" and select your CSV
3. Map columns:
   - Date column: `date`
   - Revenue column: `revenue`
   - Media channels: Select all spend columns
4. Review data quality report
5. Click "Save Configuration"

### Step 3: Configure Model

1. Go to **⚙️ Model Config** page
2. Set transformation parameters for each channel:

**Adstock (Carryover Effect):**
- TV/Brand: 0.6-0.8 (long-lasting effect)
- Digital Performance: 0.2-0.4 (short-term effect)
- Social Media: 0.3-0.6 (medium-term)

**Hill Saturation:**
- K (Scale): Usually 1.0 (start here)
- S (Shape): Usually 1.0 (start here)

3. Click "Train Model"

### Step 4: Analyze Results

1. Go to **📈 Results** page
2. Review:
   - Channel contributions (waterfall chart)
   - ROAS by channel
   - Response curves (saturation points)
   - Model diagnostics (R², MAPE)

### Step 5: Optimize Budget

1. Go to **💰 Budget Optimizer** page
2. Enter total budget
3. Set channel constraints (optional)
4. Click "Optimize Budget"
5. Compare optimized vs current allocation

---

## Data Preparation

### Data Format Requirements

**Minimum Requirements:**
- **52 weeks** of data (1 year)
- **Weekly aggregation** (not daily - too noisy)
- **Consistent currency** (all values in same unit)

**Optimal:**
- **104-156 weeks** (2-3 years)
- **No missing weeks** (fill gaps with 0 or interpolate)
- **Include exogenous variables** (seasonality, promotions, holidays)

### Column Naming

**Required Columns:**
```
date          - Format: YYYY-MM-DD
revenue       - Total revenue (or installs, conversions)
[channel1]    - Spend on channel 1 (e.g., google_uac)
[channel2]    - Spend on channel 2 (e.g., meta)
...
```

**Optional Columns:**
```
promotion     - Binary (0/1) for promotion weeks
seasonality   - Numeric (1-4 for quarters)
holiday       - Binary for holiday weeks
installs      - For mobile games (optional KPI)
```

### Multi-Market Data Format

For global campaigns, use this format:
```csv
date,country,os,revenue,installs,google_uac,meta
2023-01-01,US,iOS,5000000,1000,800000,600000
2023-01-01,US,Android,4000000,1500,700000,500000
2023-01-01,KR,iOS,3000000,800,500000,400000
...
```

**Additional Columns:**
- `country` - 2-letter country code (US, KR, JP, etc.)
- `os` - Platform (iOS, Android)

### Data Quality Checks

Before uploading, verify:

✅ **No duplicate dates** (each week appears once)
✅ **No extreme outliers** (unless real - like Black Friday)
✅ **Consistent scaling** (don't mix millions and thousands)
✅ **No negative values** (spend and revenue should be ≥ 0)

---

## Model Configuration

### Understanding Adstock (Carryover Effect)

**What is Adstock?**
The delayed and prolonged effect of advertising. If you spend $1M on TV this week, the impact continues for several weeks.

**Decay Rate Parameter (0-1):**
- **0.0**: No carryover (effect only in current week)
- **0.5**: 50% of this week's effect carries to next week
- **0.8**: 80% carries over (very long-lasting)

**How to Choose:**

| Channel Type | Recommended Decay | Reasoning |
|--------------|-------------------|-----------|
| TV/Brand | 0.6-0.8 | Long brand-building effect |
| Search (Google) | 0.1-0.3 | Immediate performance |
| Social (Meta) | 0.3-0.6 | Medium-term engagement |
| Display | 0.4-0.7 | Retargeting effects |
| Influencer | 0.5-0.8 | Content stays visible |

**Visual Example:**
```
Week 1: Spend $100 → Effect $100
Week 2: Spend $0   → Effect $50 (50% decay from Week 1)
Week 3: Spend $0   → Effect $25 (50% decay from Week 2)
```

### Understanding Hill Saturation

**What is Saturation?**
Diminishing returns as you increase spend. The first $1M is more effective than the 10th $1M.

**Parameters:**
- **K (Scale)**: Maximum effect achievable
  - Typically 1.0-2.0
  - Higher K = more potential impact

- **S (Shape)**: How quickly saturation happens
  - S = 1.0: Smooth curve (default)
  - S < 1.0: Gentle saturation (long runway)
  - S > 1.0: Steep saturation (quick diminishing returns)

**How to Choose:**

Start with **K=1.0, S=1.0** for all channels, then adjust based on:
- Known saturation points from past experiments
- Industry benchmarks
- Business intuition

### Model Settings

**Ridge Alpha (Regularization):**
- **Low (0.1-0.5)**: Less regularization, may overfit
- **Medium (1.0)**: Balanced (recommended default)
- **High (5.0-10.0)**: Strong regularization, more stable

**Train/Test Split:**
- **80/20**: Standard (80% train, 20% test)
- Use test set to validate model accuracy

---

## Understanding Results

### Channel Contributions Tab

**Waterfall Chart:**
Shows how each channel builds up to total revenue.
```
Base (organic): ████████ 20%
Google UAC:     ██████████████ 30%
Meta:           ████████████ 25%
Apple Search:   ████████ 15%
Others:         ██████ 10%
───────────────────────────────
Total Revenue:  ████████████████████ 100%
```

**Key Metrics:**

| Metric | Meaning | Good Value |
|--------|---------|------------|
| **ROAS** | Revenue per $1 spent | > 2.0 for performance channels |
| **Contribution %** | Share of total revenue | Aligns with strategy |
| **Spend** | Total invested | N/A |

### Response Curves Tab

**What to Look For:**

✅ **Healthy Curve:**
```
Revenue ▲
        │     ╭──── (Saturated)
        │   ╭─╯
        │ ╭─╯
        │╭╯
        └──────────► Spend
```

❌ **Problem: Already Saturated**
```
Revenue ▲
        │ ────────── (Flat)
        │╭──────────
        │╯
        └──────────► Spend
         ↑
      Current spend
```
**Action:** Reduce spend on this channel

❌ **Problem: No Saturation Visible**
```
Revenue ▲
        │          ╱
        │        ╱
        │      ╱
        │    ╱
        └──────────► Spend
```
**Action:** May indicate insufficient data or poor model fit

### Model Diagnostics Tab

**R² (R-squared): 0.0-1.0**
- **> 0.7**: Excellent fit
- **0.5-0.7**: Good fit
- **< 0.5**: Poor fit (investigate data quality)

**MAPE (Mean Absolute Percentage Error):**
- **< 10%**: Excellent accuracy
- **10-20%**: Good accuracy
- **> 20%**: Poor accuracy

**What if model fit is poor?**
1. Check data quality (outliers, missing values)
2. Add exogenous variables (seasonality, promotions)
3. Try different adstock/hill parameters
4. Collect more data (need 2+ years ideally)

---

## Budget Optimization

### Using the Optimizer

**Step 1: Set Total Budget**
```
Enter desired total budget: $5,000,000
```

**Step 2: Set Constraints (Optional)**

| Channel | Min Spend | Max Spend | Reasoning |
|---------|-----------|-----------|-----------|
| Google | $1M | $3M | Contract commitment |
| Meta | $500K | $2M | Test budget cap |
| Apple | $0 | $2M | Flexible |

**Step 3: Run Optimization**

Click "Optimize Budget" to see:

**Current Allocation:**
```
Google:  $2.0M  (ROAS 3.0)
Meta:    $1.5M  (ROAS 2.5)
Apple:   $1.5M  (ROAS 2.2)
────────────────────────────
Total:   $5.0M  (Revenue: $14M)
```

**Optimized Allocation:**
```
Google:  $2.5M  (ROAS 2.8) ← +$500K
Meta:    $1.8M  (ROAS 2.6) ← +$300K
Apple:   $0.7M  (ROAS 3.0) ← -$800K
────────────────────────────
Total:   $5.0M  (Revenue: $15.2M)
                          ↑
                     +8.6% improvement!
```

### Scenario Builder

**Use Case: "What if we cut budget by 20%?"**

1. Adjust sliders:
   - Google: $2.0M → $1.6M
   - Meta: $1.5M → $1.2M
   - Apple: $1.5M → $1.2M

2. See predicted impact:
   - Total: $5.0M → $4.0M
   - Revenue: $14.0M → $12.5M
   - ROAS: 2.8 → 3.1 (efficiency improves!)

3. Save as "Conservative Q1"

4. Compare multiple scenarios:
   - Current
   - Conservative Q1
   - Aggressive Growth
   - Google Focus

### Interpreting Recommendations

**If Optimizer Suggests Big Changes (>30%):**
- ⚠️ **Don't implement all at once**
- ✅ Move 10-20% in recommended direction
- ✅ Test for 2-4 weeks
- ✅ Re-run model with new data
- ✅ Iterate gradually

**Why Gradual Implementation?**
- Model based on historical patterns
- Market conditions change
- Avoid disrupting campaigns mid-flight
- Learn and adjust

---

## Multi-Market Analysis

### Setting Up Multi-Market Models

**When to Use:**
- Running campaigns in 2+ countries
- iOS vs Android have different economics
- Need country-specific insights

**Analysis Levels:**

| Level | Use Case | Example |
|-------|----------|---------|
| **Global** | Single unified model | Small budgets, similar markets |
| **By Country** | Country-specific insights | US, KR, JP have different ROAS |
| **By OS** | Platform differences | iOS higher LTV than Android |
| **Country × OS** | Granular control | US-iOS different from US-Android |

### Comparing Markets

**Heatmap View:**
```
         Google  Meta  Apple
US       3.5    3.2   2.8
KR       2.8    2.5   2.0
JP       3.2    2.8   2.5
```
**Insights:**
- Google performs best in US
- All channels weaker in KR (competitive market?)
- Consistent performance in JP

### Cross-Market Insights

**Question: "Should I expand to new country?"**

Compare similar countries:
- If KR ROAS = 2.5 and TW similar market → expect TW ROAS ~2.3-2.7
- Budget accordingly for test phase

---

## Best Practices

### Data Collection

✅ **Do:**
- Track all marketing spend consistently
- Use UTM parameters for attribution
- Export data weekly (same day each week)
- Document campaigns/events (merge with data)
- Keep 2-3 years of history

❌ **Don't:**
- Mix different attribution windows
- Change tracking methodology mid-way
- Ignore organic/base revenue
- Delete "bad" weeks (keep all data)

### Model Development

✅ **Do:**
- Start with simple model (default parameters)
- Validate with holdout test set
- Compare predictions to actual results
- Document assumptions and decisions
- Re-train quarterly as data grows

❌ **Don't:**
- Over-tune parameters to fit perfectly (overfitting)
- Use model for channels with <10 weeks of spend
- Ignore model diagnostics (R², MAPE)
- Trust model blindly without business sense

### Using Results

✅ **Do:**
- Test recommendations gradually (10-20% shifts)
- Combine MMM with incrementality tests
- Share results with stakeholders (visualizations)
- Monitor actual performance vs predictions
- Iterate and improve

❌ **Don't:**
- Make 50%+ budget shifts based on model alone
- Stop running experiments (A/B tests complement MMM)
- Expect perfect predictions (models are guides)
- Forget to update model as market changes

---

## Troubleshooting

### "Model has poor fit (R² < 0.5)"

**Possible Causes:**
1. Not enough data (need 52+ weeks)
2. Too many channels for available data
3. Major market changes during period
4. Missing key variables (seasonality, events)

**Solutions:**
1. Collect more data
2. Combine similar channels
3. Split time period (before/after change)
4. Add exogenous variables

### "Optimization suggests extreme allocations"

**Example: "Put 90% in Google, 10% in others"**

**Why This Happens:**
- Model sees Google had highest historical ROAS
- Doesn't account for saturation beyond historical spend
- Assumes linear scaling (incorrect)

**Solutions:**
1. Set conservative constraints (min/max per channel)
2. Implement gradually (test 20% shift first)
3. Check if hill parameters are too weak (increase S)
4. Use business judgment to override

### "Results don't match my intuition"

**Example: "TV shows low ROAS but I know it's brand-building"**

**Possible Reasons:**
1. Brand effects measured differently (long-term LTV not in data)
2. Halo effects not captured (TV boosts search)
3. Data quality issues (tracking problems)
4. Adstock too low (TV effects are long-term)

**Solutions:**
1. Increase TV adstock (0.7-0.8)
2. Add brand metrics if available (awareness, consideration)
3. Validate TV tracking accuracy
4. Consider incrementality tests for TV specifically

### "Different from last-click attribution"

**This is normal!** MMM and last-click measure different things:

| Metric | Last-Click | MMM |
|--------|------------|-----|
| **What** | Which touchpoint led to conversion | Total incremental impact |
| **View** | Bottom-funnel | Full-funnel |
| **Bias** | Favors search, brand | More balanced |

**Both are useful:**
- Last-click: Tactical optimization
- Last-click: Tactical optimization
- MMM: Strategic budget allocation

---

## Glossary

**Adstock**: Carryover effect of advertising across time periods

**ROAS (Return on Ad Spend)**: Revenue generated per dollar spent on advertising

**Saturation**: Point where additional spend produces diminishing returns

**Ridge Regression**: Statistical method with regularization to prevent overfitting

**Incrementality**: Additional revenue caused by marketing (vs what would have happened anyway)

**Base/Organic**: Revenue that would occur without any paid marketing

**Exogenous Variables**: External factors affecting revenue (seasonality, holidays, events)

**R²**: Proportion of variance in revenue explained by the model (0-1 scale)

**MAPE**: Mean Absolute Percentage Error (average prediction error as %)

---

## Support & Resources

**Need Help?**
- Check this guide's troubleshooting section
- Review example datasets in `data/sample/`
- Contact: [your email]

**Further Reading:**
- "Marketing Mix Modeling: A Primer" (Meta)
- "Bayesian Methods for Media Mix Modeling" (Google)
- Industry benchmarks by channel type

**Updates:**
This tool is actively maintained. Check for updates quarterly.
