# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Sphinx documentation with auto-generated API reference
- Comprehensive user guide for marketers
- Quick reference card with formulas and benchmarks

## [0.1.0] - 2024-01-XX

### Added
- Ridge regression-based MMM implementation
- Adstock transformation for carryover effects
- Hill saturation transformation for diminishing returns
- Budget optimizer with constraints
- Hierarchical MMM for multi-market analysis
- Streamlit web interface
- Interactive visualizations with Plotly
- Data processor with validation
- Response curve generation
- ROAS calculation by channel
- Contribution analysis (waterfall charts)
- Model diagnostics (R², MAPE, RMSE)
- Export functionality (CSV, PDF reports)
- Sample data and examples

### Features

#### Core Modeling
- Ridge regression with L2 regularization
- Flexible transformation pipeline
- Train/test split validation
- Cross-validation support

#### Transformations
- Geometric adstock (carryover effects)
- Hill saturation (diminishing returns)
- Custom transformation support

#### Optimization
- Constrained budget allocation
- Multiple optimization algorithms (SLSQP, Trust-Constr, L-BFGS-B)
- Scenario comparison
- Marginal ROAS calculation

#### Multi-Market
- Segment by country, platform, or custom dimensions
- Hierarchical modeling with pooling strategies
- Cross-market comparison
- Heatmap visualizations

#### User Interface
- Clean, intuitive Streamlit interface
- Step-by-step workflow (Upload → Configure → Results → Optimize)
- Interactive charts and tables
- Real-time model training feedback

### Documentation
- User guide with tutorials
- Quick reference card
- API documentation
- Example notebooks
- Best practices guide

## [0.0.1] - Initial Development

### Added
- Project structure
- Basic MMM implementation
- Initial Streamlit interface

---

## Version History Legend

- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Features marked for removal
- **Removed**: Deleted features
- **Fixed**: Bug fixes
- **Security**: Security vulnerability fixes
