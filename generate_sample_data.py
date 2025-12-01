"""
Script to generate sample marketing data for Ridge MMM application.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate 104 weeks of data (2 years)
n_weeks = 104
start_date = datetime(2022, 1, 3)  # Start on a Monday

# Create date range (weekly)
dates = [start_date + timedelta(weeks=i) for i in range(n_weeks)]

# Base revenue with trend
base_revenue = 100000
trend = np.linspace(0, 30000, n_weeks)  # Upward trend over 2 years

# Seasonality (quarterly pattern)
seasonality_factor = []
for i in range(n_weeks):
    quarter = ((i // 13) % 4) + 1
    if quarter == 1:  # Q1 - lower
        seasonality_factor.append(0.9)
    elif quarter == 2:  # Q2 - medium
        seasonality_factor.append(1.0)
    elif quarter == 3:  # Q3 - medium-high
        seasonality_factor.append(1.05)
    else:  # Q4 - highest (holiday season)
        seasonality_factor.append(1.15)

seasonality_factor = np.array(seasonality_factor)

# Generate media spend data
google_uac = np.random.uniform(10000, 25000, n_weeks)
meta = np.random.uniform(8000, 20000, n_weeks)
apple_search = np.random.uniform(5000, 15000, n_weeks)
youtube = np.random.uniform(7000, 18000, n_weeks)
tiktok = np.random.uniform(3000, 12000, n_weeks)

# Add some correlation between channels (realistic scenario)
google_uac = google_uac + 0.3 * meta + np.random.normal(0, 1000, n_weeks)
youtube = youtube + 0.2 * google_uac + np.random.normal(0, 1000, n_weeks)

# Ensure no negative values
google_uac = np.maximum(google_uac, 5000)
meta = np.maximum(meta, 5000)
apple_search = np.maximum(apple_search, 3000)
youtube = np.maximum(youtube, 5000)
tiktok = np.maximum(tiktok, 2000)

# Generate promotion indicator (binary, ~20% of weeks)
promotion = np.random.choice([0, 1], n_weeks, p=[0.8, 0.2])

# Generate seasonality indicator (1-4 for quarters)
seasonality = []
for i in range(n_weeks):
    quarter = ((i // 13) % 4) + 1
    seasonality.append(quarter)

# Calculate revenue with media effects
# Each channel has different ROI
google_uac_effect = google_uac * 2.5
meta_effect = meta * 2.8
apple_search_effect = apple_search * 3.2
youtube_effect = youtube * 1.8
tiktok_effect = tiktok * 2.0

# Promotion boost
promotion_effect = promotion * 15000

# Add diminishing returns (saturation effect)
total_spend = google_uac + meta + apple_search + youtube + tiktok
saturation_factor = 1 - (total_spend / 200000) * 0.3
saturation_factor = np.maximum(saturation_factor, 0.5)

# Calculate final revenue
revenue = (
    base_revenue +
    trend +
    (base_revenue * (seasonality_factor - 1)) +
    (google_uac_effect + meta_effect + apple_search_effect + 
     youtube_effect + tiktok_effect) * saturation_factor +
    promotion_effect +
    np.random.normal(0, 8000, n_weeks)  # Random noise
)

# Ensure revenue is positive
revenue = np.maximum(revenue, 50000)

# Round values
revenue = np.round(revenue, 2)
google_uac = np.round(google_uac, 2)
meta = np.round(meta, 2)
apple_search = np.round(apple_search, 2)
youtube = np.round(youtube, 2)
tiktok = np.round(tiktok, 2)

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'revenue': revenue,
    'google_uac': google_uac,
    'meta': meta,
    'apple_search': apple_search,
    'youtube': youtube,
    'tiktok': tiktok,
    'promotion': promotion,
    'seasonality': seasonality
})

# Save to CSV
output_path = '/mnt/d/project/vscode/ridge-mmm-app/data/sample/sample_data.csv'
df.to_csv(output_path, index=False)

print(f"Sample data generated successfully!")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nSummary statistics:")
print(df.describe())
print(f"\nData saved to: {output_path}")
