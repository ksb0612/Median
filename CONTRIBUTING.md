# Contributing to Ridge MMM

Thank you for your interest in contributing to Ridge MMM! We welcome contributions from the community and are grateful for your support.

---

## 🎯 Ways to Contribute

### 1. 🐛 Report Bugs
Found a bug? Please [open an issue](https://github.com/your-username/ridge-mmm-app/issues) with:
- Clear, descriptive title
- Steps to reproduce the bug
- Expected vs actual behavior
- Environment details (OS, Python version, package versions)
- Minimal reproducible example (if possible)
- Screenshots or error messages

**Example:**
```
Title: Model training fails with NaN values in data

Environment:
- OS: Windows 10
- Python: 3.10.5
- Ridge MMM: v0.1.0

Steps to reproduce:
1. Upload CSV with NaN values in 'revenue' column
2. Go to Model Config page
3. Click "Train Model"

Expected: Error message about NaN values
Actual: Uncaught exception, app crashes

Error message:
ValueError: Input contains NaN, infinity or a value too large...
```

---

### 2. 💡 Suggest Features
Have an idea? [Open a discussion](https://github.com/your-username/ridge-mmm-app/discussions) first to:
- Describe the use case
- Explain why it's valuable
- Propose a solution (if you have one)
- Discuss alternatives

**Example:**
```
Title: Add support for daily data aggregation

Use case: Many users have daily data but the tool requires weekly.
Proposed: Add a "Aggregate to weekly" option in the Data Upload page.
Alternatives: External preprocessing before upload.
```

---

### 3. 📝 Improve Documentation
Documentation improvements are always welcome:
- Fix typos or clarify confusing sections
- Add examples or tutorials
- Improve API documentation
- Translate documentation (i18n)

See [docs/](docs/) for existing documentation.

---

### 4. 🧪 Add Tests
Help us improve test coverage:
- Add test cases for edge cases
- Improve existing test assertions
- Add integration tests
- Add performance benchmarks

See [tests/](tests/) for existing tests.

---

### 5. 🔧 Submit Code Changes
Ready to code? Follow the development setup below!

---

## 🛠️ Development Setup

### 1. Fork & Clone
```bash
# Fork the repository on GitHub, then:
git clone https://github.com/your-username/ridge-mmm-app.git
cd ridge-mmm-app
```

### 2. Install Dependencies
```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### 3. Install Development Tools
Development dependencies are already included in the `dev` group:
```bash
# Verify installation
poetry run pytest --version
poetry run black --version
poetry run flake8 --version
```

### 4. Create a Branch
```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or a bug fix branch
git checkout -b fix/issue-123
```

**Branch naming conventions:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `test/` - Test additions/improvements
- `refactor/` - Code refactoring

---

## 💻 Development Workflow

### 1. Make Your Changes
```bash
# Edit files in your preferred editor
code src/your_module.py
```

### 2. Run Tests
```bash
# Run all tests
poetry run pytest tests/ -v

# Run specific test file
poetry run pytest tests/test_ridge_mmm.py -v

# Run with coverage
poetry run pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

**All tests must pass before submitting a PR!**

### 3. Format Code
```bash
# Format code with black
poetry run black src/ tests/

# Check formatting without changing files
poetry run black --check src/ tests/
```

**Code style:**
- We use [Black](https://black.readthedocs.io/) for consistent formatting
- Line length: 88 characters (Black default)
- Follow PEP 8 guidelines

### 4. Lint Code
```bash
# Run flake8
poetry run flake8 src/ tests/

# Fix common issues automatically
poetry run autopep8 --in-place --recursive src/ tests/
```

**Common linting rules:**
- No unused imports
- No undefined variables
- Follow naming conventions (snake_case for functions, PascalCase for classes)

### 5. Run the App Locally
```bash
# Start the Streamlit app
poetry run streamlit run streamlit_app/app.py

# App will open at http://localhost:8501
```

Test your changes manually in the UI.

---

## 📋 Code Standards

### Type Hints
Use type annotations for all functions:
```python
from typing import Dict, List, Optional
import pandas as pd

def calculate_roas(
    spend: pd.Series,
    revenue: pd.Series,
    channel_name: str
) -> float:
    """Calculate Return on Ad Spend (ROAS).

    Args:
        spend: Channel spend values
        revenue: Revenue values
        channel_name: Name of the marketing channel

    Returns:
        ROAS value (revenue / spend)

    Raises:
        ValueError: If spend is zero or negative
    """
    if spend.sum() <= 0:
        raise ValueError(f"Invalid spend for {channel_name}")

    return revenue.sum() / spend.sum()
```

### Docstrings
Use Google-style docstrings:
```python
def transform_data(X: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Apply transformations to input data.

    This function applies adstock and Hill saturation transformations
    to the input marketing spend data.

    Args:
        X: Input DataFrame with marketing spend columns
        config: Dictionary with transformation parameters
            - 'adstock': Decay rate (0.0-1.0)
            - 'hill_K': Hill scale parameter
            - 'hill_S': Hill shape parameter

    Returns:
        Transformed DataFrame with same shape as input

    Raises:
        ValueError: If config parameters are out of valid range
        KeyError: If required columns are missing

    Example:
        >>> X = pd.DataFrame({'google': [1000, 2000, 3000]})
        >>> config = {'adstock': 0.5, 'hill_K': 1.0, 'hill_S': 1.0}
        >>> X_transformed = transform_data(X, config)
    """
    pass
```

### Error Handling
Always validate inputs and provide helpful error messages:
```python
def validate_data(df: pd.DataFrame) -> None:
    """Validate input data quality.

    Raises:
        ValueError: With specific error message about what's wrong
    """
    if df.empty:
        raise ValueError("DataFrame is empty. Please upload data with at least 52 weeks.")

    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"Missing values found in columns: {null_cols}")

    if len(df) < 52:
        raise ValueError(f"Insufficient data: {len(df)} weeks provided, 52 required.")
```

### Testing
Write tests for all new functionality:
```python
import pytest
import pandas as pd
from src.ridge_mmm import RidgeMMM

def test_roas_calculation():
    """Test ROAS calculation with known values."""
    mmm = RidgeMMM()
    X = pd.DataFrame({'google': [1000, 2000, 3000]})
    y = pd.Series([5000, 10000, 15000])

    # Train model
    mmm.fit(X, y, {'google': {'adstock': 0.5, 'hill_K': 1.0, 'hill_S': 1.0}})

    # Calculate ROAS
    roas = mmm.get_roas(X, y)

    # Assert expected ROAS
    assert roas['google'] == pytest.approx(5.0, rel=0.01)

def test_edge_case_zero_spend():
    """Test handling of zero spend."""
    mmm = RidgeMMM()
    X = pd.DataFrame({'google': [0, 0, 0]})
    y = pd.Series([5000, 5000, 5000])

    # Should handle gracefully
    mmm.fit(X, y, {'google': {'adstock': 0.5, 'hill_K': 1.0, 'hill_S': 1.0}})

    # ROAS should be inf or handled appropriately
    with pytest.raises(ValueError, match="Zero spend"):
        mmm.get_roas(X, y)
```

---

## 📝 Commit Guidelines

### Commit Message Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Test additions/improvements
- `chore`: Build process, dependencies, etc.

**Examples:**
```
feat(optimizer): Add support for channel-level constraints

- Add min/max spend constraints per channel
- Update optimization algorithm to handle bounds
- Add tests for constrained optimization

Closes #123
```

```
fix(data): Handle NaN values in revenue column

- Add validation to detect NaN values early
- Provide helpful error message to user
- Add test case for NaN handling

Fixes #456
```

```
docs(readme): Update installation instructions

- Add Windows-specific notes
- Clarify Poetry installation steps
- Add troubleshooting section
```

### Commit Best Practices
- Keep commits atomic (one logical change per commit)
- Write clear, descriptive commit messages
- Reference issues/PRs in commit messages
- Don't commit generated files (e.g., `__pycache__`, `.pyc`)

---

## 🔍 Pull Request Process

### 1. Update Documentation
If your changes affect user-facing features:
- Update [docs/USER_GUIDE.md](docs/USER_GUIDE.md)
- Update [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)
- Update API docstrings
- Add examples if appropriate

### 2. Update CHANGELOG
Add your changes to [CHANGELOG.md](CHANGELOG.md) under "Unreleased":
```markdown
## [Unreleased]

### Added
- Support for daily data aggregation (#123)

### Fixed
- Handle NaN values in revenue column (#456)
```

### 3. Run Pre-submission Checklist
```bash
# Format code
poetry run black src/ tests/

# Lint code
poetry run flake8 src/ tests/

# Run all tests
poetry run pytest tests/ -v

# Check coverage (aim for >80%)
poetry run pytest tests/ --cov=src --cov-report=term

# Build documentation (if changed)
cd docs && make html

# Test the app manually
poetry run streamlit run streamlit_app/app.py
```

### 4. Create Pull Request
```bash
# Push your branch
git push origin feature/your-feature-name

# Go to GitHub and create a Pull Request
```

**PR Template:**
```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## How Has This Been Tested?
- [ ] Existing tests pass
- [ ] Added new tests for this change
- [ ] Manually tested in the UI

## Checklist
- [ ] Code follows style guidelines (black, flake8)
- [ ] Self-reviewed my own code
- [ ] Commented hard-to-understand areas
- [ ] Updated documentation
- [ ] Added tests that prove my fix/feature works
- [ ] New and existing tests pass locally
- [ ] Updated CHANGELOG.md

## Screenshots (if applicable)
Add screenshots to show the change.

## Related Issues
Closes #123
```

### 5. Code Review
- Be responsive to feedback
- Make requested changes promptly
- Discuss disagreements respectfully
- Don't take criticism personally - we're all learning!

### 6. Merge
Once approved:
- Maintainers will merge your PR
- Your changes will be included in the next release
- You'll be credited in CHANGELOG.md and release notes

---

## 🧪 Testing Guidelines

### Types of Tests

**Unit Tests:**
Test individual functions/methods in isolation
```python
def test_adstock_transformation():
    """Test adstock calculation with known decay rate."""
    transformer = AdstockTransformer(decay_rate=0.5)
    spend = pd.Series([100, 0, 0])
    result = transformer.transform(spend)

    assert result[0] == 100
    assert result[1] == 50  # 0 + 0.5 * 100
    assert result[2] == 25  # 0 + 0.5 * 50
```

**Integration Tests:**
Test multiple components working together
```python
def test_full_mmm_workflow():
    """Test complete MMM workflow from data to results."""
    # Load data
    dp = DataProcessor()
    df = dp.load_csv('data/sample/sample_data.csv')

    # Train model
    mmm = RidgeMMM()
    mmm.fit(X, y, channel_configs)

    # Get results
    contributions = mmm.get_contributions(X)
    roas = mmm.get_roas(X, y)

    # Assertions
    assert sum(contributions.values()) == pytest.approx(y.sum(), rel=0.01)
    assert all(roas > 0 for roas in roas.values())
```

**Edge Case Tests:**
Test boundary conditions and error handling
```python
def test_empty_dataframe():
    """Test handling of empty DataFrame."""
    dp = DataProcessor()
    with pytest.raises(ValueError, match="empty"):
        dp.validate_data(pd.DataFrame())

def test_extreme_values():
    """Test handling of very large values."""
    mmm = RidgeMMM()
    X = pd.DataFrame({'google': [1e15, 1e15, 1e15]})
    y = pd.Series([1e18, 1e18, 1e18])

    # Should not raise overflow errors
    mmm.fit(X, y, {'google': {'adstock': 0.5, 'hill_K': 1.0, 'hill_S': 1.0}})
```

### Test Coverage
- Aim for >80% code coverage
- Cover all public APIs
- Test edge cases and error conditions
- Don't test implementation details

---

## 📞 Getting Help

### Questions?
- Check [docs/USER_GUIDE.md](docs/USER_GUIDE.md)
- Check [docs/source/faq.md](docs/source/faq.md)
- Ask in [GitHub Discussions](https://github.com/your-username/ridge-mmm-app/discussions)

### Stuck?
- Review existing issues for similar problems
- Ask for help in your PR (we're friendly!)
- Reach out to maintainers via email

---

## 👥 Community Guidelines

### Be Respectful
- Treat everyone with respect
- Accept constructive criticism gracefully
- Focus on what's best for the project
- Show empathy towards other contributors

### Be Collaborative
- Help review others' PRs
- Share knowledge and expertise
- Mentor newcomers
- Celebrate others' contributions

### Be Patient
- Maintainers are volunteers
- Reviews may take time
- Not all features can be accepted
- Breaking changes require careful consideration

---

## 🎉 Recognition

Contributors will be:
- Listed in CHANGELOG.md
- Mentioned in release notes
- Added to CONTRIBUTORS.md (if they wish)
- Forever appreciated! ❤️

---

## 📜 License

By contributing to Ridge MMM, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to Ridge MMM! 🙏**

Questions? Open a [discussion](https://github.com/your-username/ridge-mmm-app/discussions) or email your.email@example.com
