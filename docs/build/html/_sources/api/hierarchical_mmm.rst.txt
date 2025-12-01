Hierarchical MMM
================

Multi-market and multi-segment Marketing Mix Model implementation.

.. automodule:: src.hierarchical_mmm
   :members:
   :undoc-members:
   :show-inheritance:

HierarchicalMMM Class
---------------------

Hierarchical model for analyzing multiple markets or segments simultaneously.

.. autoclass:: src.hierarchical_mmm.HierarchicalMMM
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:

   .. rubric:: Key Methods

   .. autosummary::
      :nosignatures:

      fit
      predict
      get_contributions_by_segment
      get_roas_by_segment
      compare_segments
      get_segment_performance

Overview
--------

The Hierarchical MMM allows you to:

- **Segment by Geography**: Analyze US, KR, JP separately
- **Segment by Platform**: Compare iOS vs Android performance
- **Multi-dimensional**: Country × OS × Channel analysis
- **Pooled Insights**: Share information across segments for better estimates

Use Cases
---------

1. **Multi-country campaigns**: Different ROAS by country
2. **Platform differences**: iOS typically higher LTV than Android
3. **Market maturity**: Established vs emerging markets
4. **Budget allocation**: Optimal spend distribution across segments

Usage Example
-------------

Basic Segmentation
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.hierarchical_mmm import HierarchicalMMM
   import pandas as pd

   # Multi-market data
   data = pd.DataFrame({
       'date': ['2023-01-01'] * 6,
       'country': ['US', 'US', 'US', 'KR', 'KR', 'KR'],
       'os': ['iOS', 'Android', 'iOS', 'iOS', 'Android', 'iOS'],
       'revenue': [5000000, 4000000, 5500000, 3000000, 2500000, 3200000],
       'google': [1000000, 900000, 1100000, 600000, 500000, 650000],
       'meta': [800000, 700000, 850000, 500000, 400000, 550000]
   })

   # Initialize model
   hmmm = HierarchicalMMM(
       segment_cols=['country', 'os'],
       alpha=1.0
   )

   # Prepare features and target
   X = data[['google', 'meta']]
   y = data['revenue']
   segments = data[['country', 'os']]

   # Configure channels (same for all segments)
   channel_configs = {
       'google': {'adstock': 0.5, 'hill_K': 1.0, 'hill_S': 1.0},
       'meta': {'adstock': 0.4, 'hill_K': 1.0, 'hill_S': 1.0}
   }

   # Fit model
   hmmm.fit(X, y, segments, channel_configs)

Segment-Specific ROAS
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get ROAS by segment
   roas_by_segment = hmmm.get_roas_by_segment(X, y, segments)
   print(roas_by_segment)
   # Output:
   # {
   #     ('US', 'iOS'): {'google': 3.5, 'meta': 3.2},
   #     ('US', 'Android'): {'google': 2.8, 'meta': 2.5},
   #     ('KR', 'iOS'): {'google': 3.0, 'meta': 2.7}
   # }

Compare Segments
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Compare performance across segments
   comparison = hmmm.compare_segments(
       segments=['US', 'KR'],
       metric='roas'
   )

   # Visualize
   import plotly.express as px
   fig = px.bar(
       comparison,
       x='segment',
       y='roas',
       color='channel',
       barmode='group',
       title='ROAS Comparison by Country'
   )
   fig.show()

Segment Performance
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get detailed performance by segment
   performance = hmmm.get_segment_performance(
       X, y, segments,
       metrics=['roas', 'contribution', 'efficiency']
   )

   # Find best performing segments
   best_segments = performance.nlargest(3, 'roas')
   print(best_segments)

Pooling Strategies
------------------

The hierarchical model supports different pooling strategies:

**Complete Pooling**
   Single model for all segments (assumes same behavior everywhere).

   .. code-block:: python

      hmmm = HierarchicalMMM(pooling='complete')

**No Pooling**
   Separate model per segment (maximum flexibility).

   .. code-block:: python

      hmmm = HierarchicalMMM(pooling='none')

**Partial Pooling** (Default)
   Share information across segments while allowing differences.

   .. code-block:: python

      hmmm = HierarchicalMMM(pooling='partial')

Best Practices
--------------

1. **Data Requirements**: Need 30+ observations per segment minimum
2. **Segment Similarity**: More similar segments benefit from pooling
3. **Regularization**: Increase alpha for small segments
4. **Validation**: Always validate on holdout test set per segment

Example: Global Campaign Analysis
----------------------------------

.. code-block:: python

   from src.hierarchical_mmm import HierarchicalMMM
   import pandas as pd

   # Load global campaign data
   data = pd.read_csv('global_campaigns.csv')

   # Segment by country and OS
   hmmm = HierarchicalMMM(
       segment_cols=['country', 'os'],
       alpha=1.0,
       pooling='partial'
   )

   # Fit model
   hmmm.fit(
       X=data[['google_uac', 'meta', 'apple_search', 'tiktok']],
       y=data['revenue'],
       segments=data[['country', 'os']],
       channel_configs=channel_configs
   )

   # Analyze by country
   us_performance = hmmm.get_segment_performance(
       segment_filter={'country': 'US'}
   )

   # Find best channels per country
   for country in ['US', 'KR', 'JP']:
       roas = hmmm.get_roas_by_segment(
           segment_filter={'country': country}
       )
       best_channel = max(roas, key=roas.get)
       print(f"{country} best channel: {best_channel} (ROAS: {roas[best_channel]:.2f})")
