Budget Optimizer
================

Budget allocation optimization based on MMM response curves.

.. automodule:: src.optimizer
   :members:
   :undoc-members:
   :show-inheritance:

BudgetOptimizer Class
---------------------

Optimizes budget allocation across channels to maximize predicted revenue.

.. autoclass:: src.optimizer.BudgetOptimizer
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:

   .. rubric:: Key Methods

   .. autosummary::
      :nosignatures:

      optimize
      optimize_with_constraints
      compare_scenarios
      get_marginal_roas

OptimizationError
-----------------

Exception raised when optimization fails.

.. autoexception:: src.optimizer.OptimizationError
   :members:
   :show-inheritance:

Optimization Problem
--------------------

The budget optimization problem is formulated as:

.. math::

   \\max_{x_1, ..., x_n} \\sum_{i=1}^{n} f_i(x_i)

   \\text{subject to:}

   \\sum_{i=1}^{n} x_i = B

   L_i \\leq x_i \\leq U_i \\quad \\forall i

where:

- :math:`x_i` is the spend on channel :math:`i`
- :math:`f_i(x_i)` is the revenue response function for channel :math:`i`
- :math:`B` is the total budget
- :math:`L_i, U_i` are lower/upper bounds on channel :math:`i`

Usage Example
-------------

Basic Optimization
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.optimizer import BudgetOptimizer
   from src.ridge_mmm import RidgeMMM
   import pandas as pd

   # Assume we have a trained MMM model
   mmm = RidgeMMM(alpha=1.0)
   # ... (fit model) ...

   # Initialize optimizer
   optimizer = BudgetOptimizer(model=mmm)

   # Optimize for total budget
   total_budget = 5000000  # $5M
   optimal_allocation = optimizer.optimize(
       total_budget=total_budget,
       channels=['google_uac', 'meta', 'apple_search']
   )

   print(optimal_allocation)
   # Output:
   # {
   #     'google_uac': 2500000,
   #     'meta': 1800000,
   #     'apple_search': 700000,
   #     'predicted_revenue': 15200000,
   #     'total_roas': 3.04
   # }

With Constraints
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Set channel constraints
   constraints = {
       'google_uac': {'min': 1000000, 'max': 3000000},  # $1M-$3M
       'meta': {'min': 500000, 'max': 2000000},         # $0.5M-$2M
       'apple_search': {'min': 0, 'max': 2000000}       # $0-$2M
   }

   # Optimize with constraints
   optimal = optimizer.optimize_with_constraints(
       total_budget=5000000,
       constraints=constraints
   )

   print(f"Optimal allocation:")
   for channel, spend in optimal['allocation'].items():
       print(f"  {channel}: ${spend:,.0f}")

Scenario Comparison
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Define scenarios
   scenarios = {
       'Current': {
           'google_uac': 2000000,
           'meta': 1500000,
           'apple_search': 1500000
       },
       'Aggressive': {
           'google_uac': 3000000,
           'meta': 1500000,
           'apple_search': 500000
       },
       'Balanced': {
           'google_uac': 2500000,
           'meta': 1500000,
           'apple_search': 1000000
       }
   }

   # Compare scenarios
   comparison = optimizer.compare_scenarios(scenarios)

   # Visualize
   import plotly.express as px
   fig = px.bar(
       comparison,
       x='scenario',
       y='predicted_revenue',
       color='total_roas',
       title='Scenario Comparison'
   )
   fig.show()

Marginal ROAS
^^^^^^^^^^^^^

.. code-block:: python

   # Get marginal ROAS (incremental efficiency)
   marginal_roas = optimizer.get_marginal_roas(
       current_spend={
           'google_uac': 2000000,
           'meta': 1500000,
           'apple_search': 1500000
       },
       increment=100000  # $100K increments
   )

   print(marginal_roas)
   # Output:
   # {
   #     'google_uac': 2.5,  # Next $100K yields $2.50 per $1
   #     'meta': 2.8,        # Best marginal ROAS
   #     'apple_search': 1.8 # Saturating
   # }

   # Recommendation: Shift budget from apple_search to meta

Optimization Algorithms
-----------------------

The optimizer supports multiple algorithms:

**SLSQP (Default)**
   Sequential Least Squares Programming. Fast and reliable for smooth objectives.

   .. code-block:: python

      optimizer = BudgetOptimizer(model=mmm, method='SLSQP')

**Trust-Constr**
   Trust-region constrained algorithm. More robust for difficult problems.

   .. code-block:: python

      optimizer = BudgetOptimizer(model=mmm, method='trust-constr')

**L-BFGS-B**
   Limited-memory BFGS with bounds. Good for large-scale problems.

   .. code-block:: python

      optimizer = BudgetOptimizer(model=mmm, method='L-BFGS-B')

Best Practices
--------------

1. **Set Realistic Constraints**

   Don't allow channels to go to $0 or infinity. Use historical ranges.

   .. code-block:: python

      # Good: Based on historical data
      constraints = {
          'google': {'min': historical_min * 0.5, 'max': historical_max * 1.5}
      }

2. **Validate Response Curves**

   Check that curves are smooth and saturating before optimization.

   .. code-block:: python

      curves = mmm.get_response_curves(channels=['google'], n_points=100)
      # Plot and visually inspect

3. **Gradual Implementation**

   Don't shift entire budget at once. Move 10-20% at a time.

   .. code-block:: python

      current_spend = {'google': 2000000, 'meta': 1500000}
      optimal_spend = optimizer.optimize(total_budget=3500000)

      # Implement 20% of the shift
      gradual_spend = {}
      for channel in current_spend:
          shift = optimal_spend[channel] - current_spend[channel]
          gradual_spend[channel] = current_spend[channel] + 0.2 * shift

4. **Monitor and Adjust**

   Re-optimize monthly as new data comes in.

Troubleshooting
---------------

**Optimization Failed**
   - Check constraints are feasible (total budget achievable)
   - Verify response curves are smooth (no discontinuities)
   - Try different optimization method

**Extreme Allocations**
   - Add more constraints (min/max per channel)
   - Increase regularization in MMM (alpha parameter)
   - Check for overfitting in underlying model

**Slow Convergence**
   - Use simpler method (SLSQP instead of trust-constr)
   - Reduce number of channels
   - Scale spend data (divide by 1M)

Example: Monthly Budget Planning
---------------------------------

.. code-block:: python

   from src.optimizer import BudgetOptimizer
   from src.ridge_mmm import RidgeMMM
   import pandas as pd

   # Load historical data and train model
   data = pd.read_csv('marketing_data.csv')
   mmm = RidgeMMM(alpha=1.0)
   mmm.fit(X=data[channels], y=data['revenue'], channel_configs=configs)

   # Monthly budget planning
   monthly_budgets = [4000000, 4500000, 5000000, 5500000]

   results = []
   for budget in monthly_budgets:
       optimal = optimizer.optimize(
           total_budget=budget,
           channels=channels,
           constraints=constraints
       )
       results.append({
           'budget': budget,
           'predicted_revenue': optimal['predicted_revenue'],
           'roas': optimal['total_roas'],
           'allocation': optimal['allocation']
       })

   # Find best budget-revenue trade-off
   df_results = pd.DataFrame(results)
   print(df_results)
