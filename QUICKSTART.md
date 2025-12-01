# Ridge MMM Application - Quick Start Guide

## 🚀 Quick Start

### Installation

```bash
cd /mnt/d/project/vscode/ridge-mmm-app
poetry install
poetry shell
```

### Run the Application

```bash
streamlit run streamlit_app/app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Complete Workflow

### 1. Home Page
- Download sample data
- Review data format requirements
- Understand MMM concepts

### 2. Data Upload (📊)
- Upload your CSV file
- Map columns (date, revenue, media channels, exogenous variables)
- Validate data quality
- Save configuration

### 3. Model Configuration (⚙️)
- **Adstock Configuration**: Set decay rates for carryover effects
- **Saturation Configuration**: Set Hill parameters (K, S) for diminishing returns
- **Model Parameters**: Configure Ridge alpha and train/test split
- **Preview Transformations**: See how transformations affect your data
- **Train Model**: Click "🚀 Train Model" to fit the MMM

### 4. Results & Analysis (📈)
Five comprehensive tabs:

**Tab 1: Model Performance**
- R², MAPE, MAE, RMSE metrics
- Actual vs Predicted scatter plot
- Residual analysis

**Tab 2: Channel Contributions**
- Contribution table with ROAS
- Waterfall chart
- Ranked bar chart

**Tab 3: Time Series Decomposition**
- Stacked area chart showing weekly contributions
- Date range filter
- Downloadable data table

**Tab 4: Response Curves**
- Spend vs revenue curves for each channel
- Current spend markers
- Saturation points
- Optimization recommendations

**Tab 5: Model Diagnostics**
- Q-Q plot for normality
- Statistical tests (Shapiro-Wilk)
- Model coefficients
- Overall quality assessment

### 5. Budget Optimizer (💰)
Four powerful sections:

**Section 1: Current vs Optimal**
- Input total budget
- Click "🚀 Optimize Budget"
- See optimal allocation vs current
- View improvement metrics

**Section 2: Scenario Builder**
- Adjust spend per channel with sliders
- See real-time revenue/ROAS predictions
- Save multiple scenarios
- Compare all scenarios side-by-side

**Section 3: Sensitivity Analysis**
- Select a channel
- See how revenue changes with ±50% spend variation
- Get optimization recommendations

**Section 4: Diminishing Returns**
- Analyze all channels for saturation
- See which channels have room to grow
- Identify over-invested channels
- Marginal ROAS visualization

## 🎯 Key Features

### Data Processing
- ✅ CSV upload with validation
- ✅ Missing value detection
- ✅ Outlier detection
- ✅ Summary statistics

### Transformations
- ✅ Adstock (exponential decay)
- ✅ Hill saturation (diminishing returns)
- ✅ Per-channel configuration
- ✅ Real-time preview

### Model Training
- ✅ Ridge regression with sklearn
- ✅ Feature scaling
- ✅ Train/test split
- ✅ Comprehensive metrics

### Analysis & Insights
- ✅ Channel attribution
- ✅ ROAS calculation
- ✅ Response curves
- ✅ Time series decomposition
- ✅ Model diagnostics

### Budget Optimization
- ✅ scipy.optimize integration
- ✅ Constrained optimization
- ✅ Scenario comparison
- ✅ Sensitivity analysis
- ✅ Diminishing returns detection

## 📥 Downloads

### From Results Page
- **Excel Export**: All metrics, contributions, decomposition, and model summary

### From Budget Optimizer
- Optimization results
- Scenario comparisons
- Sensitivity analysis data

## 🔧 Configuration Options

### Adstock Parameters
- **Decay Rate**: 0.0 to 0.9
  - 0.0 = No carryover
  - 0.5 = Moderate carryover (default)
  - 0.9 = Long-lasting carryover

### Hill Saturation Parameters
- **K (Scale)**: 0.5 to 2.0 (default: 1.0)
  - Controls maximum effect level
- **S (Shape)**: 0.5 to 2.0 (default: 1.0)
  - Controls steepness of saturation curve

### Ridge Regression
- **Alpha**: 0.1 to 10.0 (default: 1.0)
  - Higher = more regularization
  - Lower = fits data more closely

### Train/Test Split
- **70% to 90%** for training (default: 80%)

### Optimization Constraints
- **Min/Max per channel**: % of current spend
- **Default**: 70% to 130% of current
- **Unlimited mode**: Remove all constraints

## 💡 Best Practices

### Data Preparation
1. Use weekly aggregated data (recommended)
2. At least 52 weeks of data (2 years preferred)
3. Ensure consistent date format
4. No missing values in critical columns

### Model Configuration
1. Start with default parameters
2. Preview transformations before training
3. Check model metrics (R² > 0.6 is good)
4. Review residual plots for patterns

### Budget Optimization
1. Review response curves first
2. Set realistic constraints
3. Compare multiple scenarios
4. Consider business constraints (min spend requirements)

### Interpretation
- **ROAS > 3.0**: Excellent efficiency
- **ROAS 2.0-3.0**: Good efficiency
- **ROAS < 2.0**: Needs improvement
- **R² > 0.8**: Excellent model fit
- **MAPE < 10%**: Very accurate predictions

## 🐛 Troubleshooting

### Model Won't Train
- Check that all media columns have numeric data
- Ensure no NaN values in features
- Verify train/test split leaves enough data

### Optimization Fails
- Constraints may be too tight
- Try "Allow unlimited changes"
- Increase budget if too low
- Check that all channels have positive spend

### Poor Model Performance
- Try adjusting transformation parameters
- Add more exogenous variables
- Increase training data
- Adjust Ridge alpha

## 📚 Additional Resources

### Sample Data
- Included: `data/sample/sample_data.csv`
- 104 weeks of realistic marketing data
- 5 media channels + 2 exogenous variables

### Testing
```bash
# Run all tests
poetry run pytest tests/ -v

# Run specific test file
poetry run pytest tests/test_ridge_mmm.py -v
```

### Code Quality
```bash
# Format code
poetry run black src/ tests/

# Lint code
poetry run flake8 src/ tests/
```

## 🎓 Understanding MMM

### What is Marketing Mix Modeling?
MMM is a statistical technique that helps quantify the impact of marketing activities on sales/revenue. It answers:
- Which channels drive the most revenue?
- What is the ROI of each channel?
- How should I allocate my budget?
- Where are diminishing returns occurring?

### Key Concepts

**Adstock**: Marketing effects don't happen instantly. Adstock models how advertising impact persists over time (carryover effect).

**Saturation**: As you spend more, each additional dollar has less impact. The Hill function models this diminishing returns effect.

**Attribution**: Breaking down total revenue into contributions from each marketing channel plus baseline.

**ROAS**: Return on Ad Spend = Revenue Generated / Spend. Higher is better.

**Marginal ROAS**: The return on the *next* dollar spent. Helps identify optimal spend levels.

## 🚀 Next Steps

1. **Upload your data** or use the sample data
2. **Configure transformations** based on your domain knowledge
3. **Train the model** and review performance metrics
4. **Analyze results** to understand channel contributions
5. **Optimize budget** to maximize ROI

---

**Need Help?** Check the tooltips (ℹ️) throughout the app for context-specific guidance.

**Found a Bug?** Please report issues with detailed steps to reproduce.

**Want to Contribute?** See `README.md` for development guidelines.
