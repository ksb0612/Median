Transformations
===============

This module contains transformation functions for Marketing Mix Modeling, including
adstock (carryover effects) and Hill saturation (diminishing returns).

.. automodule:: src.transformations
   :members:
   :undoc-members:
   :show-inheritance:

Adstock Transformer
-------------------

Applies carryover effects to model delayed advertising impact.

.. autoclass:: src.transformations.AdstockTransformer
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:

   The adstock transformation models the carryover effect of advertising using
   geometric decay:

   .. math::

      adstock_t = spend_t + \\alpha \\cdot adstock_{t-1}

   where :math:`\\alpha` is the decay rate (0-1).

Hill Transformer
----------------

Applies saturation effects to model diminishing returns.

.. autoclass:: src.transformations.HillTransformer
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:

   The Hill transformation models saturation using:

   .. math::

      effect = K \\cdot \\frac{x^S}{x^S + 1}

   where:
   - :math:`K` is the scale parameter (maximum effect)
   - :math:`S` is the shape parameter (saturation rate)

Transformation Pipeline
-----------------------

Combines multiple transformations in sequence.

.. autoclass:: src.transformations.TransformationPipeline
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Adstock Example
^^^^^^^^^^^^^^^

.. code-block:: python

   from src.transformations import AdstockTransformer
   import numpy as np

   # Create transformer with 50% decay
   adstock = AdstockTransformer(decay_rate=0.5)

   # Transform spend data
   spend = np.array([1000, 0, 0, 0])
   adstocked = adstock.transform(spend)
   # Result: [1000, 500, 250, 125]

Hill Saturation Example
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.transformations import HillTransformer
   import numpy as np

   # Create transformer
   hill = HillTransformer(K=1.0, S=1.0)

   # Transform spend data
   spend = np.array([100, 500, 1000, 5000])
   saturated = hill.transform(spend)

   # Shows diminishing returns at higher spend levels

Pipeline Example
^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.transformations import TransformationPipeline
   from src.transformations import AdstockTransformer, HillTransformer

   # Create pipeline
   pipeline = TransformationPipeline([
       AdstockTransformer(decay_rate=0.5),
       HillTransformer(K=1.0, S=1.0)
   ])

   # Apply both transformations
   transformed = pipeline.fit_transform(spend_data)
