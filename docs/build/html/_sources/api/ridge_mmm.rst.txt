Ridge MMM
=========

Core Ridge regression-based Marketing Mix Model implementation.

.. automodule:: src.ridge_mmm
   :members:
   :undoc-members:
   :show-inheritance:

RidgeMMM Class
--------------

Main Marketing Mix Model class using Ridge regression with transformations.

.. autoclass:: src.ridge_mmm.RidgeMMM
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

   .. rubric:: Key Methods

   .. autosummary::
      :nosignatures:

      fit
      predict
      get_contributions
      get_roas
      get_response_curves
      get_model_diagnostics

Model Architecture
------------------

The Ridge MMM model follows this pipeline:

1. **Transform Media Variables**

   - Apply adstock transformation (carryover effects)
   - Apply Hill saturation (diminishing returns)

2. **Ridge Regression**

   - Fit regularized linear model: :math:`y = X\\beta + \\epsilon`
   - Use L2 regularization to prevent overfitting

3. **Decomposition**

   - Calculate base (intercept) contribution
   - Calculate channel-specific contributions
   - Sum to total predicted revenue

Usage Example
-------------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from src.ridge_mmm import RidgeMMM
   import pandas as pd

   # Prepare data
   X = pd.DataFrame({
       'google_uac': [1000000, 1500000, 2000000, 2500000],
       'meta': [800000, 1200000, 1600000, 2000000],
       'apple_search': [500000, 700000, 900000, 1100000]
   })
   y = pd.Series([5000000, 7500000, 10000000, 12500000])

   # Configure channels
   channel_configs = {
       'google_uac': {
           'adstock': 0.5,  # 50% carryover
           'hill_K': 1.0,   # Scale
           'hill_S': 1.0    # Shape
       },
       'meta': {
           'adstock': 0.4,
           'hill_K': 1.0,
           'hill_S': 1.0
       },
       'apple_search': {
           'adstock': 0.3,
           'hill_K': 1.0,
           'hill_S': 1.0
       }
   }

   # Train model
   mmm = RidgeMMM(alpha=1.0)
   mmm.fit(X, y, channel_configs)

   # Make predictions
   predictions = mmm.predict(X)
   print(f"Predictions: {predictions}")

Getting Contributions
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get channel contributions
   contributions = mmm.get_contributions(X)
   print(contributions)
   # Output:
   # {
   #     'base': 1000000,
   #     'google_uac': 6000000,
   #     'meta': 4500000,
   #     'apple_search': 2000000
   # }

Calculating ROAS
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get ROAS by channel
   roas = mmm.get_roas(X, y)
   print(roas)
   # Output:
   # {
   #     'google_uac': 3.0,  # $3 revenue per $1 spent
   #     'meta': 2.8,
   #     'apple_search': 2.5
   # }

Response Curves
^^^^^^^^^^^^^^^

.. code-block:: python

   # Get response curves for optimization
   curves = mmm.get_response_curves(
       channels=['google_uac', 'meta'],
       spend_range=(0, 5000000),
       n_points=100
   )

   # Use for budget optimization
   import plotly.express as px
   fig = px.line(
       curves,
       x='spend',
       y='revenue',
       color='channel',
       title='Revenue Response Curves'
   )
   fig.show()

Model Diagnostics
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get model performance metrics
   diagnostics = mmm.get_model_diagnostics(X, y)
   print(diagnostics)
   # Output:
   # {
   #     'r_squared': 0.92,
   #     'mape': 8.5,
   #     'rmse': 450000
   # }

Mathematical Formulation
------------------------

The Ridge MMM model is formulated as:

.. math::

   y = \\beta_0 + \\sum_{i=1}^{n} \\beta_i \\cdot transform_i(x_i) + \\epsilon

where:

- :math:`y` is the target variable (revenue, installs, etc.)
- :math:`x_i` is the spend on channel :math:`i`
- :math:`transform_i(x_i)` applies adstock and Hill saturation
- :math:`\\beta_i` are the coefficients learned by Ridge regression
- :math:`\\epsilon` is the error term

The Ridge regression objective is:

.. math::

   \\min_{\\beta} \\|y - X\\beta\\|_2^2 + \\alpha\\|\\beta\\|_2^2

where :math:`\\alpha` is the regularization strength.
