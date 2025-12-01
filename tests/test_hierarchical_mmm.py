import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hierarchical_mmm import HierarchicalMMM

@pytest.fixture
def sample_data():
    # Create synthetic data
    dates = pd.date_range(start='2023-01-01', periods=100)
    countries = ['US', 'KR', 'JP']
    os_types = ['iOS', 'Android']
    
    data = []
    for date in dates:
        for country in countries:
            for os_type in os_types:
                # Random spend
                google_spend = np.random.uniform(100, 1000)
                fb_spend = np.random.uniform(100, 1000)
                
                # Revenue with some signal
                revenue = (google_spend * 2) + (fb_spend * 1.5) + np.random.normal(0, 50)
                
                # Add country/os effects
                if country == 'US': revenue *= 1.2
                if os_type == 'iOS': revenue *= 1.1
                
                data.append({
                    'date': date,
                    'country': country,
                    'os': os_type,
                    'google_spend': google_spend,
                    'facebook_spend': fb_spend,
                    'revenue': revenue
                })
                
    return pd.DataFrame(data)

@pytest.fixture
def channel_configs():
    return {
        'google_spend': {'adstock': 0.1, 'hill_K': 100, 'hill_S': 1},
        'facebook_spend': {'adstock': 0.1, 'hill_K': 100, 'hill_S': 1}
    }

def test_hierarchical_mmm_global(sample_data, channel_configs):
    mmm = HierarchicalMMM(analysis_level='global')
    mmm.fit(sample_data, target_col='revenue', channel_configs=channel_configs)
    
    assert mmm.is_fitted
    assert 'global' in mmm.models
    assert len(mmm.models) == 1
    
    preds = mmm.predict(sample_data)
    assert len(preds) == len(sample_data) # Global model predicts for all rows
    
    contribs = mmm.get_contributions(sample_data)
    assert not contribs.empty
    assert 'google_spend' in contribs['channel'].values

def test_hierarchical_mmm_country(sample_data, channel_configs):
    mmm = HierarchicalMMM(analysis_level='country')
    mmm.fit(sample_data, target_col='revenue', channel_configs=channel_configs)
    
    assert mmm.is_fitted
    assert 'US' in mmm.models
    assert 'KR' in mmm.models
    assert 'JP' in mmm.models
    
    # Predict for specific segment
    us_data = sample_data[sample_data['country'] == 'US']
    us_preds = mmm.predict(us_data, segment='US')
    assert len(us_preds) == len(us_data)
    
    # Predict all
    all_preds = mmm.predict(sample_data)
    assert len(all_preds) == len(sample_data)
    
    # Compare segments
    comparison = mmm.compare_segments(sample_data, metric='roas')
    assert not comparison.empty
    assert len(comparison) == 3 # 3 countries

def test_hierarchical_mmm_os(sample_data, channel_configs):
    mmm = HierarchicalMMM(analysis_level='os')
    mmm.fit(sample_data, target_col='revenue', channel_configs=channel_configs)
    
    assert mmm.is_fitted
    assert 'iOS' in mmm.models
    assert 'Android' in mmm.models
    
    comparison = mmm.compare_segments(sample_data)
    assert len(comparison) == 2

def test_hierarchical_mmm_country_os(sample_data, channel_configs):
    mmm = HierarchicalMMM(analysis_level='country_os')
    mmm.fit(sample_data, target_col='revenue', channel_configs=channel_configs)
    
    assert mmm.is_fitted
    # 3 countries * 2 OS = 6 models
    assert len(mmm.models) == 6
    assert 'US_iOS' in mmm.models
    
    preds = mmm.predict(sample_data)
    assert len(preds) == len(sample_data)

def test_invalid_level():
    with pytest.raises(ValueError):
        HierarchicalMMM(analysis_level='invalid')
