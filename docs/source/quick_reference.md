# Quick Reference Card

## Essential Formulas

### ROAS Calculation
```
ROAS = Total Revenue / Total Spend

Example: $15M revenue / $5M spend = 3.0 ROAS
(Every $1 spent generates $3 in revenue)
```

### Adstock Transformation
```
adstock[t] = spend[t] + decay × adstock[t-1]

Example with decay=0.5:
Week 1: Spend $100 → Adstock $100
Week 2: Spend $0   → Adstock $50  (0.5 × $100)
Week 3: Spend $0   → Adstock $25  (0.5 × $50)
```

### Hill Saturation
```
effect = K × (spend^S) / (spend^S + 1)

Parameters:
- K: Maximum possible effect (scale)
- S: Rate of saturation (shape)
```

## Parameter Quick Guide

| Channel Type | Adstock Decay | Hill K | Hill S |
|--------------|---------------|---------|---------|
| TV/Brand | 0.7 | 1.5 | 2.0 |
| Search | 0.2 | 1.0 | 1.0 |
| Social | 0.5 | 1.0 | 1.0 |
| Display | 0.6 | 1.0 | 1.5 |
| Video | 0.6 | 1.2 | 1.5 |

## Interpretation Guide

### ROAS Benchmarks
- **< 1.0**: Losing money
- **1.0-2.0**: Break-even to moderate
- **2.0-4.0**: Good performance
- **> 4.0**: Excellent (or room to scale)

### Model Fit (R²)
- **> 0.8**: Excellent
- **0.6-0.8**: Good
- **0.4-0.6**: Fair
- **< 0.4**: Poor (needs improvement)

### MAPE (Prediction Error)
- **< 10%**: Excellent accuracy
- **10-20%**: Good accuracy
- **20-30%**: Fair accuracy
- **> 30%**: Poor accuracy

## Common Workflows

### Weekly Refresh
1. Export latest week's data
2. Append to historical CSV
3. Re-upload to app
4. Check if model diagnostics changed significantly
5. If major changes: Re-train model

### Monthly Budget Planning
1. Review last month's actual vs predicted
2. Update any changed constraints
3. Run optimizer with next month's total budget
4. Export recommended allocation
5. Implement 50% of recommended shift (gradual)

### Quarterly Strategy Review
1. Re-train model with all available data
2. Compare channel performance trends (YoY)
3. Run multiple scenarios (conservative, aggressive, status quo)
4. Present to stakeholders with visualizations
5. Align on budget strategy for next quarter

## Keyboard Shortcuts (Streamlit)

- `Ctrl + R`: Refresh page
- `Ctrl + Shift + R`: Clear cache and refresh
- `C`: Toggle light/dark theme (if enabled)

## File Locations

- **Sample Data**: `data/sample/sample_data.csv`
- **Your Uploads**: Auto-saved in browser session
- **Model Exports**: Download from Results page
- **Reports**: Download from Report page

## Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| "Model not fitted" | Trying to predict before training | Go to Model Config → Train Model |
| "Column not found" | Data missing required column | Check column mapping in Data Upload |
| "Optimization failed" | Can't find feasible solution | Relax constraints or increase budget |
| "Insufficient data" | Less than 52 weeks | Collect more data |

## Pro Tips

💡 **Start Simple**: Use default parameters (decay=0.5, K=1.0, S=1.0) first

💡 **Validate**: Always check test set performance before trusting model

💡 **Gradual Changes**: Shift budgets 10-20% at a time, not 50%+

💡 **Document**: Save scenarios with descriptive names

💡 **Combine Methods**: Use MMM for strategy, last-click for tactics

💡 **Refresh Regularly**: Re-train model quarterly or when major changes occur
