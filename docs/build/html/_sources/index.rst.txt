Ridge MMM Documentation
=======================

Welcome to Ridge MMM's documentation!

Ridge MMM is a Marketing Mix Modeling tool built with Ridge Regression that helps marketers:

* **Understand channel contributions** to revenue and KPIs
* **Optimize budget allocation** across marketing channels
* **Predict ROI** of marketing spend with response curves
* **Compare multi-market performance** across countries and platforms
* **Analyze saturation** and carryover effects (adstock)

Quick Links
-----------

* :doc:`user_guide` - Comprehensive user guide for marketers
* :doc:`quick_reference` - Quick reference card with formulas and benchmarks
* :ref:`genindex` - Index of all functions and classes

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Documentation

   user_guide
   quick_reference
   tutorials
   faq

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/data_processor
   api/transformations
   api/ridge_mmm
   api/hierarchical_mmm
   api/optimizer
   api/visualizations
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   architecture

Key Features
------------

**Ridge Regression-based MMM**
   Fast, interpretable, and reliable Marketing Mix Modeling using regularized linear regression.

**Adstock & Saturation Transformations**
   Built-in support for carryover effects (adstock) and diminishing returns (Hill saturation).

**Multi-Market Hierarchical Models**
   Analyze performance across countries, platforms (iOS/Android), and segments.

**Budget Optimization**
   Constrained optimization to find optimal budget allocation based on response curves.

**Interactive Streamlit UI**
   User-friendly web interface for data upload, model configuration, and results visualization.

**Comprehensive Visualizations**
   Waterfall charts, response curves, contribution analysis, and diagnostic plots.

Getting Started
---------------

Installation
^^^^^^^^^^^^

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/ridge-mmm-app.git
   cd ridge-mmm-app

   # Install dependencies with Poetry
   poetry install

   # Run the Streamlit app
   poetry run streamlit run streamlit_app/Home.py

Quick Example
^^^^^^^^^^^^^

.. code-block:: python

   from src.ridge_mmm import RidgeMMM
   import pandas as pd

   # Prepare your data
   X = pd.DataFrame({
       'google_uac': [1000000, 1500000, 2000000],
       'meta': [800000, 1200000, 1600000],
       'apple_search': [500000, 700000, 900000]
   })
   y = pd.Series([5000000, 7500000, 10000000])

   # Configure channel transformations
   channel_configs = {
       'google_uac': {'adstock': 0.5, 'hill_K': 1.0, 'hill_S': 1.0},
       'meta': {'adstock': 0.4, 'hill_K': 1.0, 'hill_S': 1.0},
       'apple_search': {'adstock': 0.3, 'hill_K': 1.0, 'hill_S': 1.0}
   }

   # Train the model
   mmm = RidgeMMM(alpha=1.0)
   mmm.fit(X, y, channel_configs)

   # Get channel contributions
   contributions = mmm.get_contributions(X)
   print(contributions)

   # Calculate ROAS
   roas = mmm.get_roas(X, y)
   print(f"ROAS: {roas}")

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
