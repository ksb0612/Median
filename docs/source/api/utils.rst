Utilities
=========

Helper functions and utility modules.

Data Utils
----------

Data manipulation and validation utilities.

.. automodule:: src.utils.data_utils
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
^^^^^^^^^^^^^

.. autofunction:: src.utils.data_utils.validate_data_format

.. autofunction:: src.utils.data_utils.aggregate_weekly

.. autofunction:: src.utils.data_utils.fill_missing_weeks

.. autofunction:: src.utils.data_utils.detect_outliers

Segment Utils
-------------

Utilities for multi-market and hierarchical analysis.

.. automodule:: src.utils.segment_utils
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
^^^^^^^^^^^^^

.. autofunction:: src.utils.segment_utils.create_segment_groups

.. autofunction:: src.utils.segment_utils.aggregate_by_segment

.. autofunction:: src.utils.segment_utils.compare_segments

Plot Utils
----------

Helper functions for plotting and visualization.

.. automodule:: src.utils.plot_utils
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
^^^^^^^^^^^^^

.. autofunction:: src.utils.plot_utils.format_currency

.. autofunction:: src.utils.plot_utils.create_color_palette

.. autofunction:: src.utils.plot_utils.add_annotations

Usage Examples
--------------

Data Validation
^^^^^^^^^^^^^^^

.. code-block:: python

   from src.utils.data_utils import validate_data_format
   import pandas as pd

   # Load data
   data = pd.read_csv('marketing_data.csv')

   # Validate
   is_valid, errors = validate_data_format(
       data,
       required_columns=['date', 'revenue'],
       date_column='date'
   )

   if not is_valid:
       print("Data validation errors:")
       for error in errors:
           print(f"  - {error}")

Weekly Aggregation
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.utils.data_utils import aggregate_weekly

   # Daily data
   daily_data = pd.DataFrame({
       'date': pd.date_range('2023-01-01', periods=365, freq='D'),
       'revenue': [10000 + i * 100 for i in range(365)],
       'spend': [5000 + i * 50 for i in range(365)]
   })

   # Aggregate to weekly
   weekly_data = aggregate_weekly(
       daily_data,
       date_column='date',
       aggregation={'revenue': 'sum', 'spend': 'sum'}
   )

   print(f"Daily: {len(daily_data)} rows → Weekly: {len(weekly_data)} rows")

Fill Missing Weeks
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.utils.data_utils import fill_missing_weeks

   # Data with gaps
   data_with_gaps = pd.DataFrame({
       'date': ['2023-01-01', '2023-01-08', '2023-01-22'],  # Missing week 15
       'revenue': [10000, 12000, 15000]
   })

   # Fill gaps
   complete_data = fill_missing_weeks(
       data_with_gaps,
       date_column='date',
       fill_method='interpolate'
   )

   print(f"Before: {len(data_with_gaps)} weeks → After: {len(complete_data)} weeks")

Outlier Detection
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.utils.data_utils import detect_outliers
   import numpy as np

   # Revenue data with outlier
   revenue = np.array([10000, 11000, 10500, 50000, 10200])  # 50000 is outlier

   # Detect outliers
   outliers = detect_outliers(
       revenue,
       method='iqr',
       threshold=1.5
   )

   print(f"Outlier indices: {np.where(outliers)[0]}")
   # Output: [3]

Segment Analysis
^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.utils.segment_utils import create_segment_groups, compare_segments

   # Multi-market data
   data = pd.DataFrame({
       'country': ['US', 'US', 'KR', 'KR', 'JP', 'JP'],
       'os': ['iOS', 'Android', 'iOS', 'Android', 'iOS', 'Android'],
       'revenue': [5000000, 4000000, 3000000, 2500000, 3500000, 3000000],
       'spend': [2000000, 1800000, 1200000, 1000000, 1500000, 1300000]
   })

   # Create segment groups
   segments = create_segment_groups(
       data,
       segment_cols=['country', 'os']
   )

   # Compare segments
   comparison = compare_segments(
       data,
       segments=segments,
       metrics=['revenue', 'spend'],
       aggregation='sum'
   )

   print(comparison)

Currency Formatting
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.utils.plot_utils import format_currency

   # Format values
   values = [1000000, 2500000, 15000000]

   formatted = [format_currency(v) for v in values]
   print(formatted)
   # Output: ['$1.0M', '$2.5M', '$15.0M']

   # With custom precision
   formatted = [format_currency(v, precision=2) for v in values]
   print(formatted)
   # Output: ['$1.00M', '$2.50M', '$15.00M']

Color Palettes
^^^^^^^^^^^^^^

.. code-block:: python

   from src.utils.plot_utils import create_color_palette

   # Generate colors for channels
   channels = ['google', 'meta', 'apple', 'tiktok']
   colors = create_color_palette(
       n_colors=len(channels),
       palette='viridis'
   )

   color_map = dict(zip(channels, colors))
   print(color_map)
   # Output: {'google': '#440154', 'meta': '#31688e', ...}

Add Plot Annotations
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.utils.plot_utils import add_annotations
   import plotly.graph_objects as go

   # Create figure
   fig = go.Figure()
   fig.add_trace(go.Scatter(x=[1, 2, 3], y=[10, 15, 13]))

   # Add annotations
   annotations = [
       {'x': 1, 'y': 10, 'text': 'Start'},
       {'x': 2, 'y': 15, 'text': 'Peak'},
       {'x': 3, 'y': 13, 'text': 'End'}
   ]

   fig = add_annotations(fig, annotations)
   fig.show()

Helper Constants
----------------

Common constants and configurations.

.. code-block:: python

   from src.utils import constants

   # Default channel configs
   DEFAULT_ADSTOCK = constants.DEFAULT_ADSTOCK  # 0.5
   DEFAULT_HILL_K = constants.DEFAULT_HILL_K    # 1.0
   DEFAULT_HILL_S = constants.DEFAULT_HILL_S    # 1.0

   # Default constraints
   MIN_SPEND_RATIO = constants.MIN_SPEND_RATIO  # 0.5
   MAX_SPEND_RATIO = constants.MAX_SPEND_RATIO  # 1.5

   # Model defaults
   DEFAULT_ALPHA = constants.DEFAULT_ALPHA      # 1.0
   DEFAULT_TRAIN_SIZE = constants.DEFAULT_TRAIN_SIZE  # 0.8
