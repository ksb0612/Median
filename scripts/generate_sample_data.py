import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_data():
    """Generate synthetic multi-market MMM data."""
    
    # Configuration
    countries = ['US', 'JP', 'KR', 'TW']
    os_types = ['iOS', 'Android']
    channels = ['google_uac', 'meta', 'apple_search', 'tiktok']
    n_weeks = 104
    start_date = datetime(2023, 1, 1)
    
    # Base metrics per market (Volume multiplier)
    market_multipliers = {
        'US': 10.0,
        'JP': 5.0,
        'KR': 3.0,
        'TW': 1.0
    }
    
    # OS multipliers (iOS usually higher monetization)
    os_multipliers = {
        'iOS': 1.5,
        'Android': 1.0
    }
    
    # Channel mix preferences per market (weights)
    market_channel_weights = {
        'US': {'google_uac': 0.3, 'meta': 0.4, 'apple_search': 0.2, 'tiktok': 0.1},
        'JP': {'google_uac': 0.3, 'meta': 0.2, 'apple_search': 0.3, 'tiktok': 0.2},
        'KR': {'google_uac': 0.4, 'meta': 0.2, 'apple_search': 0.1, 'tiktok': 0.3},
        'TW': {'google_uac': 0.3, 'meta': 0.3, 'apple_search': 0.1, 'tiktok': 0.3}
    }
    
    data = []
    dates = [start_date + timedelta(weeks=i) for i in range(n_weeks)]
    
    for country in countries:
        for os_name in os_types:
            # Base volume for this segment
            segment_base = 10000 * market_multipliers[country] * os_multipliers[os_name]
            
            for date in dates:
                row = {
                    'date': date,
                    'country': country,
                    'os': os_name
                }
                
                # Seasonality
                week_num = date.isocalendar()[1]
                month = date.month
                
                # General Q4 seasonality
                seasonality = 1.0
                if month in [11, 12]:
                    seasonality = 1.3
                
                # Lunar New Year (approximate Jan/Feb) for KR/TW
                if country in ['KR', 'TW'] and month in [1, 2]:
                    seasonality = max(seasonality, 1.4)
                
                # Random noise
                noise = np.random.normal(1, 0.1)
                
                row['seasonality'] = seasonality
                
                # Promotion events (randomly occur)
                is_promo = np.random.random() < 0.1
                row['promotion'] = 1 if is_promo else 0
                promo_lift = 1.5 if is_promo else 1.0
                
                # Generate spend and revenue for each channel
                total_revenue = 0
                total_installs = 0
                
                for channel in channels:
                    # Base spend for channel
                    if channel == 'apple_search' and os_name == 'Android':
                        spend = 0
                    else:
                        weight = market_channel_weights[country][channel]
                        base_spend = segment_base * weight * 0.1 # Spend is ~10% of revenue base
                        spend = base_spend * seasonality * promo_lift * np.random.normal(1, 0.2)
                    
                    row[channel] = max(0, spend)
                    
                    # Revenue generation (ROAS)
                    if spend > 0:
                        # Base ROAS varies by channel and market
                        base_roas = 2.0
                        if country == 'US': base_roas *= 1.2
                        if os_name == 'iOS': base_roas *= 1.3
                        if channel == 'apple_search': base_roas *= 1.4
                        
                        # Diminishing returns (simple log function)
                        # Revenue = Spend * ROAS * Efficiency_Factor
                        # Efficiency drops as spend increases relative to base
                        efficiency = 1 / (1 + (spend / (segment_base * 0.05))) 
                        # Actually let's use a simpler power law: Revenue = A * Spend^B
                        beta = 0.8 # Diminishing return
                        alpha = base_roas * (spend ** (1-beta)) # Adjust alpha so at base spend ROAS is base_roas
                        
                        channel_revenue = alpha * (spend ** beta) * noise
                        
                    else:
                        channel_revenue = 0
                    
                    total_revenue += channel_revenue
                
                # Add organic/baseline revenue
                baseline = segment_base * 0.5 * seasonality * noise
                total_revenue += baseline
                
                row['revenue'] = total_revenue
                row['installs'] = total_revenue / (5 if country in ['US', 'JP'] else 2) # Fake LTV
                
                data.append(row)
    
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    os.makedirs('data/sample', exist_ok=True)
    
    # Save
    output_path = 'data/sample/sample_multi_market_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Generated sample data at {output_path}")
    print(df.head())
    print(df.groupby(['country', 'os'])['revenue'].sum())

if __name__ == "__main__":
    generate_sample_data()
