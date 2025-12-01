Data Processor
==============

The data processor module handles data loading, validation, and preprocessing for MMM.

.. automodule:: src.data_processor
   :members:
   :undoc-members:
   :show-inheritance:

DataProcessor Class
-------------------

Main class for processing marketing mix data.

.. autoclass:: src.data_processor.DataProcessor
   :members:
   :special-members: __init__
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      load_data
      validate_data
      get_media_channels
      get_date_range
      preprocess_data

Usage Example
-------------

.. code-block:: python

   from src.data_processor import DataProcessor
   import pandas as pd

   # Create sample data
   data = pd.DataFrame({
       'date': pd.date_range('2023-01-01', periods=52, freq='W'),
       'revenue': [10000 + i * 1000 for i in range(52)],
       'google': [5000 + i * 100 for i in range(52)],
       'meta': [3000 + i * 80 for i in range(52)]
   })

   # Initialize processor
   processor = DataProcessor()
   processor.load_data(data)

   # Get media channels
   channels = processor.get_media_channels()
   print(f"Media channels: {channels}")

   # Validate data
   is_valid = processor.validate_data()
   print(f"Data valid: {is_valid}")
