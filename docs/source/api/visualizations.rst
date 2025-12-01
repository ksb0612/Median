Visualizations
==============

Plotting functions and visualizations for MMM results.

.. automodule:: src.visualizations
   :members:
   :undoc-members:
   :show-inheritance:

Plotting Functions
------------------

Channel Contributions Waterfall
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.visualizations.plot_contribution_waterfall

Response Curves
^^^^^^^^^^^^^^^

.. autofunction:: src.visualizations.plot_response_curves

ROAS Comparison
^^^^^^^^^^^^^^^

.. autofunction:: src.visualizations.plot_roas_comparison

Model Diagnostics
^^^^^^^^^^^^^^^^^

.. autofunction:: src.visualizations.plot_model_diagnostics

Multi-Market Heatmap
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.visualizations.plot_market_heatmap

Usage Examples
--------------

Waterfall Chart
^^^^^^^^^^^^^^^

.. code-block:: python

   from src.visualizations import plot_contribution_waterfall
   from src.ridge_mmm import RidgeMMM

   # Get contributions from model
   contributions = mmm.get_contributions(X)

   # Plot waterfall
   fig = plot_contribution_waterfall(
       contributions=contributions,
       title='Channel Contributions to Revenue'
   )
   fig.show()

Response Curves
^^^^^^^^^^^^^^^

.. code-block:: python

   from src.visualizations import plot_response_curves

   # Get response curves
   curves = mmm.get_response_curves(
       channels=['google', 'meta', 'apple'],
       spend_range=(0, 5000000),
       n_points=100
   )

   # Plot curves
   fig = plot_response_curves(
       curves=curves,
       current_spend={'google': 2000000, 'meta': 1500000, 'apple': 1000000},
       title='Revenue Response Curves by Channel'
   )
   fig.show()

ROAS Comparison
^^^^^^^^^^^^^^^

.. code-block:: python

   from src.visualizations import plot_roas_comparison

   # Get ROAS by channel
   roas = mmm.get_roas(X, y)

   # Plot comparison
   fig = plot_roas_comparison(
       roas=roas,
       benchmark=2.0,  # Industry benchmark
       title='ROAS by Marketing Channel'
   )
   fig.show()

Model Diagnostics
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.visualizations import plot_model_diagnostics

   # Get predictions and actuals
   y_pred = mmm.predict(X_test)
   y_true = y_test

   # Plot diagnostics
   fig = plot_model_diagnostics(
       y_true=y_true,
       y_pred=y_pred,
       title='Model Performance on Test Set'
   )
   fig.show()

Multi-Market Analysis
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.visualizations import plot_market_heatmap
   from src.hierarchical_mmm import HierarchicalMMM

   # Get ROAS by segment
   roas_by_segment = hmmm.get_roas_by_segment(X, y, segments)

   # Plot heatmap
   fig = plot_market_heatmap(
       data=roas_by_segment,
       x='country',
       y='channel',
       values='roas',
       title='ROAS Heatmap by Country and Channel'
   )
   fig.show()

Customization
-------------

All plotting functions return Plotly figure objects that can be customized:

.. code-block:: python

   fig = plot_response_curves(curves, current_spend)

   # Customize layout
   fig.update_layout(
       width=1000,
       height=600,
       font=dict(size=14),
       template='plotly_white'
   )

   # Customize colors
   fig.update_traces(
       line=dict(width=3),
       marker=dict(size=10)
   )

   # Save to file
   fig.write_html('response_curves.html')
   fig.write_image('response_curves.png')

Exporting
---------

Export figures to various formats:

.. code-block:: python

   # HTML (interactive)
   fig.write_html('plot.html')

   # Static images (requires kaleido)
   fig.write_image('plot.png', width=1200, height=800)
   fig.write_image('plot.pdf')
   fig.write_image('plot.svg')

   # For presentations
   fig.write_image('plot.png', width=1920, height=1080, scale=2)
