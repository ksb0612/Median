# Frequently Asked Questions

## General Questions

### What is Marketing Mix Modeling (MMM)?

Marketing Mix Modeling is a statistical technique that quantifies the impact of marketing activities on sales or other KPIs. It helps answer questions like "How much revenue did each channel generate?" and "What's the optimal budget allocation?"

### How is this different from attribution models?

- **MMM**: Top-down, aggregate level analysis. Measures total incremental impact.
- **Attribution**: Bottom-up, user-level tracking. Assigns credit to touchpoints.

Both are complementary. Use MMM for strategic planning and attribution for tactical optimization.

### How much data do I need?

Minimum: 52 weeks (1 year) of weekly data
Recommended: 104-156 weeks (2-3 years)

More data = more reliable estimates, especially for long-term effects like brand advertising.

## Model Questions

### What does the alpha parameter do?

Alpha controls regularization strength in Ridge regression:
- **Low alpha (0.1-0.5)**: Less regularization, may overfit
- **Medium alpha (1.0)**: Balanced (recommended default)
- **High alpha (5.0-10.0)**: Strong regularization, more stable but may underfit

### How do I choose adstock parameters?

Start with these guidelines:
- **Brand/TV**: 0.6-0.8 (long-lasting effects)
- **Performance/Search**: 0.2-0.4 (immediate effects)
- **Social**: 0.3-0.6 (medium-term)

Then refine based on:
- Domain knowledge
- Historical experiments
- Model fit on validation set

### What if my R² is low?

If R² < 0.5, try:
1. Collect more data
2. Add exogenous variables (seasonality, holidays)
3. Adjust transformation parameters
4. Check for data quality issues
5. Consider if MMM is appropriate for your use case

## Optimization Questions

### Why does the optimizer suggest extreme allocations?

This usually means:
- Constraints are too loose (add min/max bounds)
- Model is overfitting (increase alpha)
- Response curves aren't saturating properly (adjust Hill parameters)

Always implement changes gradually (10-20% shifts).

### Can I optimize for multiple objectives?

Currently, the optimizer maximizes predicted revenue. For multi-objective optimization (revenue + efficiency + reach), you'll need to:
1. Define a custom objective function
2. Weight different metrics
3. Use the optimizer with your custom function

## Technical Questions

### Can I use daily data instead of weekly?

Not recommended. Daily data is too noisy for MMM. Always aggregate to weekly for stable estimates.

### Does this support Bayesian MMM?

This implementation uses Ridge regression (frequentist approach). For Bayesian MMM with uncertainty quantification, consider tools like:
- Facebook's Robyn
- Google's LightweightMMM
- PyMC-Marketing

### Can I add custom transformations?

Yes! The transformation pipeline is extensible. See `src/transformations.py` for examples of implementing custom transformers.

## Still Have Questions?

- Check the [User Guide](user_guide.md) for detailed explanations
- Review the [API Reference](api/ridge_mmm.rst) for technical details
- Open an issue on GitHub
