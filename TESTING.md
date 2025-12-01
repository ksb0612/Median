# Testing Guide - Phase 2 Transformations

## 🧪 Running the Tests

### Prerequisites
```bash
cd /mnt/d/project/vscode/ridge-mmm-app
poetry install
poetry shell
```

### Run All Transformation Tests
```bash
poetry run pytest tests/test_transformations.py -v
```

### Run Specific Test Classes
```bash
# Test only AdstockTransformer
poetry run pytest tests/test_transformations.py::TestAdstockTransformer -v

# Test only HillTransformer
poetry run pytest tests/test_transformations.py::TestHillTransformer -v

# Test only TransformationPipeline
poetry run pytest tests/test_transformations.py::TestTransformationPipeline -v
```

### Run Specific Tests
```bash
# Test the basic adstock transformation
poetry run pytest tests/test_transformations.py::TestAdstockTransformer::test_transform_simple_case -v

# Test Hill transformation
poetry run pytest tests/test_transformations.py::TestHillTransformer::test_transform_standard_case -v

# Test integration
poetry run pytest tests/test_transformations.py::TestTransformationPipeline::test_integration_full_pipeline -v
```

### Run with Coverage
```bash
poetry run pytest tests/test_transformations.py --cov=src/transformations --cov-report=html
```

## ✅ Expected Test Results

### AdstockTransformer Tests (12 tests)
```
✓ test_init_valid_decay_rate
✓ test_init_invalid_decay_rate_too_high
✓ test_init_invalid_decay_rate_negative
✓ test_transform_simple_case
✓ test_transform_with_multiple_spikes
✓ test_transform_zero_decay
✓ test_transform_high_decay
✓ test_transform_empty_array
✓ test_transform_with_nan
✓ test_transform_with_negative
✓ test_transform_list_input
✓ test_get_decay_curve
✓ test_get_decay_curve_invalid_length
```

### HillTransformer Tests (11 tests)
```
✓ test_init_valid_parameters
✓ test_init_invalid_K
✓ test_init_invalid_S
✓ test_transform_standard_case
✓ test_transform_zero_input
✓ test_transform_different_K
✓ test_transform_different_S
✓ test_transform_empty_array
✓ test_transform_with_nan
✓ test_transform_with_negative
✓ test_transform_large_values
✓ test_get_response_curve
```

### TransformationPipeline Tests (10 tests)
```
✓ test_init_valid_config
✓ test_init_empty_config
✓ test_init_missing_keys
✓ test_fit_transform_single_channel
✓ test_fit_transform_multiple_channels
✓ test_fit_transform_preserves_other_columns
✓ test_fit_transform_empty_dataframe
✓ test_fit_transform_missing_channel
✓ test_get_transformation_summary
✓ test_transform_single_channel_method
✓ test_transform_single_channel_invalid
✓ test_integration_full_pipeline
```

**Total: 33 tests**

## 🔬 Manual Testing in Python

### Test AdstockTransformer
```python
import numpy as np
from src.transformations import AdstockTransformer

# Create transformer
transformer = AdstockTransformer(decay_rate=0.5)

# Test with simple input
x = np.array([1, 0, 0, 0, 0])
result = transformer.transform(x)
print(f"Input:  {x}")
print(f"Output: {result}")
# Expected: [1.0, 0.5, 0.25, 0.125, 0.0625]

# Get decay curve
curve = transformer.get_decay_curve(length=10)
print(f"Decay curve: {curve}")
```

### Test HillTransformer
```python
import numpy as np
from src.transformations import HillTransformer

# Create transformer
transformer = HillTransformer(K=1.0, S=1.0)

# Test with range of values
x = np.array([0, 0.5, 1, 2, 5, 10])
result = transformer.transform(x)
print(f"Input:  {x}")
print(f"Output: {result}")
# Expected: [0.0, 0.333, 0.5, 0.667, 0.833, 0.909]

# Get response curve
x_range = np.linspace(0, 10, 100)
curve = transformer.get_response_curve(x_range)
print(f"Curve has {len(curve)} points")
```

### Test TransformationPipeline
```python
import pandas as pd
from src.transformations import TransformationPipeline

# Create configuration
configs = {
    'google': {'adstock': 0.5, 'hill_K': 1.0, 'hill_S': 1.0},
    'facebook': {'adstock': 0.7, 'hill_K': 1.2, 'hill_S': 0.9}
}

# Create pipeline
pipeline = TransformationPipeline(configs)

# Create sample data
df = pd.DataFrame({
    'google': [100, 200, 150, 100, 50],
    'facebook': [50, 75, 60, 80, 90]
})

# Transform
result = pipeline.fit_transform(df)
print("Original:")
print(df)
print("\nTransformed:")
print(result)

# Get summary
summary = pipeline.get_transformation_summary()
print("\nConfiguration:")
print(summary)
```

## 🎯 Testing the Streamlit UI

### 1. Start the Application
```bash
streamlit run streamlit_app/app.py
```

### 2. Navigate Through Workflow

**Step 1: Home Page**
- Verify introduction text displays
- Download sample data
- Check navigation guide

**Step 2: Data Upload**
- Upload sample_data.csv
- Verify data preview shows correctly
- Map columns:
  - Date: date
  - Revenue: revenue
  - Media: google_uac, meta, apple_search, youtube, tiktok
- Check validation passes
- Save configuration

**Step 3: Model Configuration** (NEW!)
- Navigate to "⚙️ Model Config" page
- Test Adstock section:
  - Adjust decay sliders
  - View decay curves
  - Try "Apply same to all" checkbox
- Test Saturation section:
  - Adjust K and S sliders
  - View saturation curves
  - Try "Apply same to all" checkbox
- Test Model Configuration:
  - Adjust Ridge alpha
  - Adjust train/test split
  - Verify row counts update
- Test Preview:
  - Select different channels
  - Verify charts update
  - Check metrics display
  - Expand data table
- Save configuration
- Check sidebar summary

### 3. Verify Visualizations

**Decay Curves Should Show:**
- Exponential decay pattern
- Starting at 1.0
- Approaching 0 over time
- Faster decay for lower decay rates

**Saturation Curves Should Show:**
- S-shaped curve
- Starting at 0
- Approaching K asymptotically
- Steeper for higher S values

**Preview Chart Should Show:**
- Three lines (original, adstocked, final)
- Different colors
- Interactive hover
- Proper legend

## 🐛 Common Issues & Solutions

### Issue: ModuleNotFoundError
```
ModuleNotFoundError: No module named 'numpy'
```
**Solution**: Run `poetry install` to install dependencies

### Issue: Tests Not Found
```
ERROR: file not found: tests/test_transformations.py
```
**Solution**: Make sure you're in the ridge-mmm-app directory

### Issue: Import Error in Tests
```
ModuleNotFoundError: No module named 'transformations'
```
**Solution**: The test file adds src to path automatically, but ensure you're running from project root

### Issue: Streamlit Page Not Found
```
Page not found
```
**Solution**: Ensure the file is named exactly `2_⚙️_Model_Config.py` with the emoji

## 📊 Test Coverage Goals

- **Line Coverage**: >90%
- **Branch Coverage**: >85%
- **Function Coverage**: 100%

### Check Coverage
```bash
poetry run pytest tests/test_transformations.py --cov=src/transformations --cov-report=term-missing
```

## 🎓 Understanding the Tests

### Unit Tests
Test individual components in isolation:
- AdstockTransformer methods
- HillTransformer methods
- TransformationPipeline methods

### Integration Tests
Test components working together:
- `test_integration_full_pipeline`: Full workflow with realistic data

### Edge Case Tests
Test boundary conditions:
- Empty arrays
- NaN values
- Negative values
- Invalid parameters
- Missing data

## ✅ Acceptance Criteria

All tests should:
- ✅ Pass without errors
- ✅ Complete in <5 seconds
- ✅ Have clear failure messages
- ✅ Test both success and failure paths
- ✅ Cover edge cases

## 🚀 Continuous Testing

### Watch Mode (for development)
```bash
poetry run pytest-watch tests/test_transformations.py
```

### Pre-commit Testing
```bash
# Run before committing
poetry run pytest tests/test_transformations.py
poetry run black src/ tests/
poetry run flake8 src/ tests/
```

---

**Happy Testing! 🧪**

For issues or questions, refer to the main [README.md](file:///mnt/d/project/vscode/ridge-mmm-app/README.md) or [walkthrough.md](file:///home/ksb0612/.gemini/antigravity/brain/d401d5f3-decd-45e4-98f7-df01db4b1574/walkthrough.md).
