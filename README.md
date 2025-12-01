# 📊 Ridge MMM - Marketing Mix Modeling Tool

> Transform your marketing data into actionable budget recommendations

[![Tests](https://img.shields.io/badge/tests-91%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-red)]()

A production-ready web application for Marketing Mix Modeling (MMM) using Ridge regression, designed specifically for mobile game marketers and UA agencies.

---

## ✨ Features

### 🎯 Core Capabilities
- **📈 Channel Attribution**: Understand true incremental impact of each marketing channel
- **💰 Budget Optimization**: Find optimal spend allocation across channels
- **🌍 Multi-Market Analysis**: Compare performance across countries and platforms (iOS/Android)
- **📊 Interactive Visualizations**: Beautiful charts with Plotly (Waterfall, Response Curves, Time Series)
- **🔮 Real-time Predictions**: Instant "what-if" scenario testing
- **⚡ Fast Performance**: Smart caching for 10x speed improvement

### 🔬 Statistical Features
- **Prophet Baseline Modeling**: Separate organic from paid growth
  - Automatically models trend, seasonality, and holidays
  - Two-stage approach: Prophet baseline + Ridge media model
  - Better attribution for growing/seasonal businesses
- **Adstock Modeling**: Capture delayed advertising effects (carryover)
  - **Geometric Adstock**: Simple exponential decay for digital channels
  - **Weibull Adstock**: Flexible S-curve decay for TV/brand campaigns with delayed peaks
- **Saturation Curves**: Model diminishing returns (Hill transformation)
- **Robust Validation**: Comprehensive input checking and error handling
- **Numerical Stability**: Handles extreme values without breaking
- **Multi-level Models**: Support for hierarchical analysis (Global → Country → OS)
- **Robyn Comparison**: Benchmark against Meta's Robyn MMM for validation

### 🎨 User Experience
- **No Code Required**: Full GUI with Streamlit
- **Guided Workflow**: Step-by-step process from data upload to insights
- **Smart Defaults**: Pre-configured parameters for common channel types
- **Export Ready**: Download results as Excel, PDF, or interactive HTML

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- Poetry (for dependency management)

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/ridge-mmm-app.git
cd ridge-mmm-app

# Install dependencies with Poetry
poetry install

# Run the app
poetry run streamlit run streamlit_app/app.py
```

The app will open in your browser at `http://localhost:8501`

### Quick Demo
```bash
# Run with sample data
poetry run streamlit run streamlit_app/app.py

# In the app:
# 1. Go to "📊 Data Upload" → Download sample data
# 2. Upload the sample CSV
# 3. Go to "⚙️ Model Config" → Click "Train Model"
# 4. Go to "📈 Results" → See insights!
```

---

## 📖 Usage

### 1. Prepare Your Data

Create a CSV file with weekly marketing data:
```csv
date,revenue,google_uac,meta,apple_search,tiktok
2023-01-01,15000000,2000000,1500000,1000000,500000
2023-01-08,18000000,2500000,1800000,1200000,600000
...
```

**Requirements:**
- ✅ At least 52 weeks (1 year) of data
- ✅ Weekly aggregation (not daily)
- ✅ Consistent currency units
- ✅ No missing weeks

### 2. Upload & Configure

**Data Upload:**
- Upload your CSV
- Map columns (date, revenue, channels)
- Review data quality report

**Model Configuration:**
- Set Adstock decay rates (0.1-0.8)
- Configure Hill saturation parameters
- Choose regularization strength

### 3. Analyze Results

**Channel Contributions:**
```
Base (organic): 20% (₩300M)
Google UAC:     30% (₩450M) - ROAS 3.0
Meta:           25% (₩375M) - ROAS 2.5
Apple Search:   15% (₩225M) - ROAS 2.2
Others:         10% (₩150M)
```

**Response Curves:**
See where each channel saturates and when to stop increasing spend.

### 4. Optimize Budget

**Input:**
- Total budget: ₩500M
- Constraints: Min/max per channel

**Output:**
```
Optimal Allocation:
Google UAC:   ₩180M (current: ₩150M) → +20%
Meta:         ₩140M (current: ₩120M) → +17%
Apple Search: ₩120M (current: ₩150M) → -20%

Expected improvement: +8.6% revenue
```

---

## 🌍 Multi-Market Analysis

For global campaigns, use this data format:
```csv
date,country,os,revenue,installs,google_uac,meta
2023-01-01,US,iOS,5000000,1000,800000,600000
2023-01-01,US,Android,4000000,1500,700000,500000
2023-01-01,KR,iOS,3000000,800,500000,400000
```

**Analysis Levels:**
- 🌐 **Global**: Single unified model
- 🗺️ **By Country**: Country-specific insights (e.g., US, KR, JP)
- 📱 **By OS**: Platform comparison (iOS vs Android)
- 🎯 **Country × OS**: Maximum granularity (e.g., US-iOS, KR-Android)

**Heatmap View:**
```
           Google  Meta  Apple
US-iOS      3.5    3.2   2.8
US-Android  3.0    2.8   2.5
KR-iOS      2.8    2.5   2.0
KR-Android  2.5    2.2   1.8
```

---

## 🏗️ Architecture

### Project Structure
```
ridge-mmm-app/
├── src/                      # Core logic
│   ├── data_processor.py     # Data loading & validation
│   ├── transformations.py    # Adstock & Hill transforms
│   ├── ridge_mmm.py          # Main MMM model
│   ├── hierarchical_mmm.py   # Multi-market model
│   ├── optimizer.py          # Budget optimization
│   ├── visualizations.py     # Plotly charts
│   └── utils/                # Utilities
│       ├── data_utils.py
│       ├── segment_utils.py
│       └── plot_utils.py
├── streamlit_app/            # Web UI
│   ├── app.py                # Home page
│   └── pages/
│       ├── 1_📊_Data_Upload.py
│       ├── 2_⚙️_Model_Config.py
│       ├── 3_📈_Results.py
│       ├── 4_💰_Budget_Optimizer.py
│       └── 5_📄_Report.py
├── tests/                    # Test suite (91 tests)
├── docs/                     # Documentation
├── data/sample/              # Sample datasets
└── pyproject.toml            # Poetry config
```

### Tech Stack

**Core:**
- Python 3.10+
- scikit-learn (Ridge regression)
- pandas & numpy (data processing)
- scipy (optimization)

**UI:**
- Streamlit (web framework)
- Plotly (interactive charts)

**Development:**
- Poetry (dependency management)
- pytest (testing)
- Sphinx (documentation)
- black & flake8 (code quality)

---

## 🧪 Testing

```bash
# Run all tests (91 tests)
poetry run pytest tests/ -v

# Run with coverage
poetry run pytest tests/ --cov=src --cov-report=html

# Run specific test file
poetry run pytest tests/test_edge_cases.py -v
```

**Test Coverage:** 91 tests covering:
- ✅ Edge cases (extreme values, zero spend, NaN)
- ✅ Data validation & error handling
- ✅ Numerical stability
- ✅ Multi-market scenarios
- ✅ Optimization constraints
- ✅ Visualization error handling

---

## 📚 Documentation

### User Documentation
- **[User Guide](docs/USER_GUIDE.md)**: Comprehensive usage instructions
- **[Quick Reference](docs/QUICK_REFERENCE.md)**: One-page cheat sheet
- **[FAQ](docs/source/faq.md)**: Frequently asked questions

### Technical Documentation
- **[API Reference](docs/build/html/index.html)**: Auto-generated from docstrings
- **[Architecture Guide](docs/source/architecture.md)**: Design rationale

**Build API docs:**
```bash
cd docs && make html
# Open docs/build/html/index.html
```

---

## 🎓 Background: What is Marketing Mix Modeling?

Marketing Mix Modeling (MMM) is a statistical technique to measure the impact of marketing activities on sales/revenue. Unlike digital attribution (last-click, multi-touch), MMM:

✅ **Measures incrementality**: What revenue would you lose if you stopped spending?
✅ **Includes offline channels**: TV, radio, print (not just digital)
✅ **Handles multi-touch**: Doesn't rely on user-level tracking
✅ **Privacy-friendly**: Works with aggregated data (no cookies needed)
✅ **Strategic view**: Answers "where to invest?" not just "which ad converted?"

### When to use MMM:
- Strategic budget planning (quarterly/annual)
- Multi-channel attribution
- Offline + online mix
- Post-iOS14/cookie deprecation world

### When NOT to use MMM:
- Real-time bid optimization (use algorithmic bidding)
- Campaign-level testing (use A/B tests)
- Less than 1 year of data (insufficient for modeling)

---

## 🔬 Methodology

### Ridge Regression
We use Ridge (L2 regularization) instead of OLS because:
- **Handles multicollinearity**: Marketing channels are often correlated
- **Prevents overfitting**: Especially with limited data
- **Stable estimates**: Small data changes don't drastically change results

### Adstock Transformation
Models delayed advertising effects with two approaches:

**Geometric Adstock** (simple exponential decay):
```python
adstock[t] = spend[t] + decay * adstock[t-1]
```
- Best for: Digital channels (Search, Social)
- Immediate peak, consistent decay
- 1 parameter: decay rate (0-0.9)

**Weibull Adstock** (flexible S-curve decay):
```python
effect[lag] = (k/λ) * (lag/λ)^(k-1) * exp(-(lag/λ)^k)
```
- Best for: TV, radio, brand campaigns
- Delayed peak effect (awareness builds over time)
- 2 parameters: shape (k) and scale (λ)
- k > 1: Delayed peak (TV), k < 1: Immediate peak, k = 1: Exponential

**Intuition**: TV ad spent this week may peak in effect 2-3 weeks later as awareness builds

### Prophet Baseline (Optional Two-Stage Approach)
Separates organic growth from media-driven revenue:

**Stage 1: Prophet models baseline**
```python
baseline = trend + yearly_seasonality + weekly_seasonality + holidays
```

**Stage 2: Ridge MMM models media on residuals**
```python
residual = actual_revenue - baseline
media_effect = Ridge(residual ~ media_channels)
```

**Benefits:**
- ✅ **Better attribution**: Removes confounding time effects
- ✅ **Handles trends**: Works for growing/declining businesses
- ✅ **Seasonality**: Automatically captures patterns
- ✅ **Holidays**: Accounts for special events

**When to use:**
- Business has clear growth or decline trend
- Strong seasonal patterns (e.g., Q4 holiday bump)
- Want to isolate true media incrementality

**Intuition**: Without Prophet, a growing business might appear to have high media ROAS simply because revenue is trending up. Prophet removes this confounding effect.

### Hill Saturation
Models diminishing returns:
```python
effect = K * (spend^S) / (spend^S + 1)
```
**Intuition**: First $1M is more effective than the 10th $1M (saturation)

---

## 🎯 Roadmap

### v0.2.0 (Next Release)
- [x] Weibull Adstock (more flexible decay curves) ✅
- [x] Robyn comparison (benchmark against Meta's MMM) ✅
- [x] Prophet baseline (automated trend/seasonality) ✅
- [ ] Confidence intervals (bootstrapped estimates)

### v0.3.0 (Future)
- [ ] Bayesian MMM (full posterior distributions)
- [ ] Geo-level modeling (DMA, city)
- [ ] Cross-channel effects (synergies)
- [ ] Real-time data connectors (GA4, Adjust)

### v1.0.0 (Production)
- [ ] User authentication
- [ ] Project management (save/load models)
- [ ] Scheduled retraining
- [ ] Email reports

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Quick contribution workflow:**
```bash
# 1. Fork & clone
git clone https://github.com/your-username/ridge-mmm-app.git

# 2. Create branch
git checkout -b feature/your-feature

# 3. Make changes & test
poetry run pytest tests/ -v

# 4. Commit & push
git commit -m "Add feature X"
git push origin feature/your-feature

# 5. Open PR
```

**Areas we need help:**
- 📊 Additional visualizations
- 🌍 i18n (internationalization)
- 🧪 More test cases
- 📝 Documentation improvements
- 🐛 Bug reports

---

## 📊 Comparison with Other Tools

| Feature | Ridge MMM | Robyn (Meta) | Meridian (Google) | LightweightMMM |
|---------|-----------|--------------|-------------------|----------------|
| **Method** | Ridge Regression | Ridge + Nevergrad | Bayesian (HMC) | JAX Bayesian |
| **Speed** | ⚡ Fast (1-5s) | 🐌 Slow (5-30min) | ⚡ Fast (JAX) | ⚡ Fast (JAX) |
| **Language** | Python | R | Python | Python |
| **UI** | ✅ Streamlit | ❌ R scripts | ❌ Notebooks | ❌ Notebooks |
| **Multi-market** | ✅ Built-in | ⚠️ Manual | ✅ Yes | ⚠️ Manual |
| **Uncertainty** | ❌ Point estimates | ⚠️ Bootstrapped | ✅ Full posterior | ✅ Full posterior |
| **Learning Curve** | 🟢 Easy | 🟡 Moderate | 🔴 Hard | 🟡 Moderate |
| **Best For** | Quick insights, agencies | Meta advertisers | Advanced users | ML engineers |

**Our Niche:** Fast, production-ready tool for agencies managing multiple clients with 1-3 year data windows.

---

## 🐛 Known Limitations

1. **Point Estimates Only**: No confidence intervals (coming in v0.2.0)
2. **No Real-time Data**: Manual CSV upload (API connectors in v0.3.0)
3. **Linear Interactions**: Can't model channel synergies explicitly
4. **Weekly Aggregation**: Not designed for daily data (too noisy)
5. **Minimum Data**: Needs 52+ weeks (1 year) to be reliable

**Workarounds:**
- For uncertainty: Run bootstrap manually or use Meridian
- For daily data: Aggregate to weekly before upload
- For real-time: Set up ETL pipeline to generate weekly CSVs

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

**Attribution:**
If you use this tool in academic work, please cite:
```bibtex
@software{ridge_mmm_2024,
  author = {Ridge MMM Contributors},
  title = {Ridge MMM: Marketing Mix Modeling Tool},
  year = {2024},
  url = {https://github.com/your-username/ridge-mmm-app}
}
```

---

## 🙏 Acknowledgments

**Inspired by:**
- [Robyn](https://github.com/facebookexperimental/Robyn) by Meta
- [Meridian](https://github.com/google/meridian) by Google
- [LightweightMMM](https://github.com/google/lightweight_mmm) by Google
- [PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing)

**Built with:**
- [Streamlit](https://streamlit.io/) - Web framework
- [scikit-learn](https://scikit-learn.org/) - Machine learning
- [Plotly](https://plotly.com/) - Interactive visualizations
- [Poetry](https://python-poetry.org/) - Dependency management

---

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/your-username/ridge-mmm-app/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/ridge-mmm-app/discussions)
- **Documentation**: [Full Documentation](docs/USER_GUIDE.md)
- **Email**: your.email@example.com

---

## ⭐ Star History

If you find this tool useful, please consider giving it a star! ⭐

---

**Made with ❤️ for marketers who love data**

[⬆ Back to top](#-ridge-mmm---marketing-mix-modeling-tool)
