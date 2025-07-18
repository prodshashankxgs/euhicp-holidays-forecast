"""
Configuration module for EU HICP Package Holidays Price Forecast project.
Handles API keys, data sources, and project settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
FRED_API_KEY = os.getenv('FRED_API_KEY')
BLS_API_KEY = os.getenv('BLS_API_KEY')

# Data source URLs
FRED_BASE_URL = "https://api.stlouisfed.org/fred"
BLS_BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data"
EUROSTAT_BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
ECB_BASE_URL = "https://data.ecb.europa.eu/api/v1"

# Key HICP Series Identifiers
HICP_SERIES = {
    'eu_package_holidays': 'CP96EAMM',  # EU Package Holidays (NSA)
    'germany_package_holidays': 'CP96DEMM',  # Germany Package Holidays (NSA)
}

# Additional Economic Indicators
ECONOMIC_INDICATORS = {
    'eu_consumer_confidence': 'CSCICP03EZM665S',
    'eur_usd_rate': 'DEXUSEU',
    'oil_price_brent': 'DCOILBRENTEU',
    'eu_gdp_growth': 'CLVMNACSCAB1GQEA19',
}

# BLS Series for US Travel Data (for comparative analysis)
BLS_TRAVEL_SERIES = {
    'us_travel_price_index': 'CUSR0000SETG01',
    'us_accommodation': 'CUSR0000SEHA',
}

# Data Collection Settings
START_DATE = '2010-01-01'  # Historical data start
END_DATE = '2024-12-31'    # Current data end
FORECAST_TARGET = '2025-07-01'  # Target forecast date

# Project Settings
DATA_DIR = 'data'
MODELS_DIR = 'models'
OUTPUTS_DIR = 'outputs'
FIGURES_DIR = 'figures'

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR]:
    os.makedirs(directory, exist_ok=True)

def validate_api_keys():
    """Validate that required API keys are available."""
    missing_keys = []
    
    if not FRED_API_KEY:
        missing_keys.append('FRED_API_KEY')
    
    if missing_keys:
        print(f"Warning: Missing API keys: {', '.join(missing_keys)}")
        print("Some data collection features may not work properly.")
        print("Please set up your API keys in a .env file or environment variables.")
    else:
        print("✓ All API keys configured successfully")
    
    return len(missing_keys) == 0 