# Architecture Overview

This document describes the architecture and design of the Ridge MMM application.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Web UI                        │
│  (streamlit_app/)                                           │
│  - Home.py                                                  │
│  - pages/1_📊_Data_Upload.py                               │
│  - pages/2_⚙️_Model_Config.py                              │
│  - pages/3_📈_Results.py                                    │
│  - pages/4_💰_Budget_Optimizer.py                          │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                     Core Library (src/)                     │
│                                                             │
│  ┌─────────────────┐    ┌──────────────────┐              │
│  │ Data Processor  │───▶│ Transformations  │              │
│  │                 │    │ - Adstock        │              │
│  │ - Load data     │    │ - Hill           │              │
│  │ - Validate      │    │ - Pipeline       │              │
│  │ - Preprocess    │    └──────────────────┘              │
│  └─────────────────┘              │                        │
│                                   ▼                        │
│                          ┌──────────────────┐              │
│                          │   Ridge MMM      │              │
│                          │                  │              │
│                          │ - Fit model      │              │
│                          │ - Predict        │              │
│                          │ - Get contributions│            │
│                          │ - Calculate ROAS │              │
│                          └──────────────────┘              │
│                                   │                        │
│         ┌─────────────────────────┼──────────────┐         │
│         ▼                         ▼              ▼         │
│  ┌─────────────┐      ┌──────────────┐   ┌──────────────┐ │
│  │ Optimizer   │      │ Hierarchical │   │Visualizations│ │
│  │             │      │     MMM      │   │              │ │
│  │ - Budget    │      │              │   │ - Waterfall  │ │
│  │   allocation│      │ - Multi-     │   │ - Response   │ │
│  │ - Scenarios │      │   market     │   │   curves     │ │
│  │ - Marginal  │      │ - Segment    │   │ - Heatmaps   │ │
│  │   ROAS      │      │   comparison │   │              │ │
│  └─────────────┘      └──────────────┘   └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Processor (`src/data_processor.py`)

**Responsibilities:**
- Load data from CSV files or DataFrames
- Validate data format and quality
- Identify media channels
- Preprocess and clean data

**Key Classes:**
- `DataProcessor`: Main data handling class

**Data Flow:**
```
Raw CSV → Load → Validate → Identify Channels → Preprocess → Clean DataFrame
```

### 2. Transformations (`src/transformations.py`)

**Responsibilities:**
- Apply adstock transformation (carryover effects)
- Apply Hill saturation (diminishing returns)
- Support custom transformations

**Key Classes:**
- `AdstockTransformer`: Geometric decay transformation
- `HillTransformer`: Saturation transformation
- `TransformationPipeline`: Combine multiple transformations

**Mathematical Models:**

Adstock:
```
adstock[t] = spend[t] + α * adstock[t-1]
```

Hill Saturation:
```
effect = K * (x^S) / (x^S + 1)
```

### 3. Ridge MMM (`src/ridge_mmm.py`)

**Responsibilities:**
- Train Ridge regression model
- Make predictions
- Decompose contributions
- Calculate ROAS
- Generate response curves

**Key Classes:**
- `RidgeMMM`: Main MMM implementation

**Model Pipeline:**
```
Raw Spend → Transformations → Ridge Regression → Predictions
                                       ↓
                              Coefficients × Transformed Features = Contributions
```

**Optimization:**
```
minimize: ||y - Xβ||² + α||β||²
```

### 4. Hierarchical MMM (`src/hierarchical_mmm.py`)

**Responsibilities:**
- Handle multi-market data (country, OS, etc.)
- Segment-specific models
- Cross-segment comparison
- Pooling strategies

**Key Classes:**
- `HierarchicalMMM`: Multi-market model

**Pooling Strategies:**
- **Complete**: Single global model
- **None**: Separate model per segment
- **Partial**: Share information across segments

### 5. Budget Optimizer (`src/optimizer.py`)

**Responsibilities:**
- Optimize budget allocation
- Handle constraints (min/max per channel)
- Compare scenarios
- Calculate marginal ROAS

**Key Classes:**
- `BudgetOptimizer`: Budget allocation optimization
- `OptimizationError`: Custom exception

**Optimization Problem:**
```
maximize: Σ f_i(x_i)  (total predicted revenue)
subject to:
  Σ x_i = B         (total budget constraint)
  L_i ≤ x_i ≤ U_i   (channel-specific constraints)
```

**Algorithms:**
- SLSQP (Sequential Least Squares Programming)
- Trust-Constr (Trust-region constrained)
- L-BFGS-B (Limited-memory BFGS with bounds)

### 6. Visualizations (`src/visualizations.py`)

**Responsibilities:**
- Generate interactive plots with Plotly
- Waterfall charts (contributions)
- Response curves
- Heatmaps (multi-market)
- Diagnostics plots

**Key Functions:**
- `plot_contribution_waterfall()`
- `plot_response_curves()`
- `plot_roas_comparison()`
- `plot_market_heatmap()`
- `plot_model_diagnostics()`

### 7. Utilities (`src/utils/`)

**Modules:**
- `data_utils.py`: Data manipulation helpers
- `segment_utils.py`: Multi-market helpers
- `plot_utils.py`: Plotting helpers

## Streamlit Interface

### Page Structure

```
Home.py
├── Data Upload (pages/1_📊_Data_Upload.py)
│   ├── Upload CSV
│   ├── Column mapping
│   ├── Data preview
│   └── Quality report
│
├── Model Config (pages/2_⚙️_Model_Config.py)
│   ├── Channel selection
│   ├── Transformation parameters
│   ├── Model settings
│   └── Train model
│
├── Results (pages/3_📈_Results.py)
│   ├── Contributions waterfall
│   ├── ROAS by channel
│   ├── Response curves
│   └── Model diagnostics
│
└── Budget Optimizer (pages/4_💰_Budget_Optimizer.py)
    ├── Set total budget
    ├── Set constraints
    ├── Optimize
    └── Compare scenarios
```

### State Management

Streamlit session state stores:
- `data`: Uploaded DataFrame
- `model`: Trained MMM model
- `channel_configs`: Transformation parameters
- `optimization_results`: Budget optimization results

## Data Flow

### Complete Workflow

```
1. Data Upload
   ↓
2. Load & Validate
   ↓
3. Configure Transformations
   ↓
4. Apply Transformations
   ↓
5. Train Ridge Regression
   ↓
6. Generate Predictions & Contributions
   ↓
7. Calculate ROAS
   ↓
8. Optimize Budget Allocation
   ↓
9. Visualize Results
```

## Design Principles

### 1. Separation of Concerns
- Core logic in `src/`
- UI in `streamlit_app/`
- Clear interfaces between components

### 2. Modularity
- Each component has single responsibility
- Easy to test and maintain
- Extensible for new features

### 3. Type Safety
- Type hints throughout
- Runtime validation
- Clear error messages

### 4. Testability
- Pure functions where possible
- Dependency injection
- Comprehensive test suite

### 5. Documentation
- Docstrings for all public APIs
- User guide for marketers
- API reference for developers

## Extension Points

### Adding New Transformations

```python
from src.transformations import BaseTransformer

class CustomTransformer(BaseTransformer):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def transform(self, X):
        # Your transformation logic
        return transformed_X
```

### Adding New Optimization Algorithms

```python
from src.optimizer import BudgetOptimizer

optimizer = BudgetOptimizer(model=mmm, method='custom')
optimizer.set_custom_optimizer(your_optimizer_function)
```

### Custom Visualizations

```python
from src.visualizations import create_base_figure
import plotly.graph_objects as go

def plot_custom_viz(data):
    fig = create_base_figure()
    fig.add_trace(go.Scatter(x=data.x, y=data.y))
    return fig
```

## Performance Considerations

### Bottlenecks
1. **Model training**: O(n³) for matrix operations
2. **Optimization**: Iterative, depends on convergence
3. **Visualization**: Large datasets can slow rendering

### Optimizations
1. **Caching**: Use `@st.cache_data` for expensive operations
2. **Vectorization**: NumPy operations instead of loops
3. **Sampling**: Downsample for visualizations if needed

## Security

### Data Privacy
- All processing happens locally
- No data sent to external services
- User data stored in session state only

### Input Validation
- Validate all user inputs
- Sanitize file uploads
- Check for malicious data

## Future Architecture Enhancements

### Planned
- [ ] Async model training for large datasets
- [ ] Distributed computing for hierarchical models
- [ ] Real-time model updates with streaming data
- [ ] Integration with external data sources (BigQuery, Snowflake)
- [ ] Model versioning and experiment tracking
- [ ] A/B testing framework for MMM models

### Under Consideration
- [ ] Microservices architecture for scalability
- [ ] GraphQL API for flexibility
- [ ] WebAssembly for client-side processing
- [ ] GPU acceleration for large-scale models
