"""
Data collection module for EU HICP Package Holidays Price Forecast.
Fetches data from FRED, BLS, Eurostat, and ECB APIs using polars for efficient processing.
"""

import polars as pl
import numpy as np
import requests
from fredapi import Fred
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Union
from tqdm import tqdm
import json

from config import (
    FRED_API_KEY, BLS_API_KEY, HICP_SERIES, ECONOMIC_INDICATORS, 
    BLS_TRAVEL_SERIES, START_DATE, END_DATE, DATA_DIR,
    EUROSTAT_BASE_URL, ECB_BASE_URL
)


class DataCollector:
    """Main data collection class for HICP forecasting project."""
    
    def __init__(self):
        """Initialize data collector with API connections."""
        self.fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None
        self.session = requests.Session()
        
        # Set up session headers
        self.session.headers.update({
            'User-Agent': 'EU-HICP-Forecast/1.0 (Research Project)',
            'Accept': 'application/json'
        })
        
        print("DataCollector initialized")
        if not self.fred:
            print("Warning: FRED API not available - some data collection will be limited")
    
    def fetch_fred_series(self, series_id: str, series_name: str) -> pl.DataFrame:
        """
        Fetch a single time series from FRED API.
        
        Args:
            series_id: FRED series identifier
            series_name: Human-readable name for the series
            
        Returns:
            Polars DataFrame with date and value columns
        """
        if not self.fred:
            print(f"Warning: Cannot fetch {series_name} - FRED API not configured")
            return pl.DataFrame()
        
        try:
            # Fetch data from FRED
            data = self.fred.get_series(
                series_id, 
                start=START_DATE, 
                end=END_DATE
            )
            
            # Convert to polars DataFrame
            df = pl.DataFrame({
                'date': data.index,
                'value': data.values,
                'series_id': [series_id] * len(data),
                'series_name': [series_name] * len(data)
            })
            
            # Ensure proper data types
            df = df.with_columns([
                pl.col('date').cast(pl.Date),
                pl.col('value').cast(pl.Float64),
                pl.col('series_id').cast(pl.String),
                pl.col('series_name').cast(pl.String)
            ])
            
            print(f"✓ Fetched {len(df)} observations for {series_name}")
            return df
            
        except Exception as e:
            print(f"Error fetching {series_name} ({series_id}): {str(e)}")
            return pl.DataFrame()
    
    def fetch_all_fred_data(self) -> pl.DataFrame:
        """
        Fetch all FRED data series and combine into single DataFrame.
        
        Returns:
            Combined polars DataFrame with all FRED series
        """
        print("Fetching FRED data series...")
        
        all_data = []
        
        # Fetch HICP series
        for name, series_id in HICP_SERIES.items():
            df = self.fetch_fred_series(series_id, name)
            if not df.is_empty():
                all_data.append(df)
        
        # Fetch economic indicators
        for name, series_id in ECONOMIC_INDICATORS.items():
            df = self.fetch_fred_series(series_id, name)
            if not df.is_empty():
                all_data.append(df)
        
        # Combine all dataframes
        if all_data:
            combined_df = pl.concat(all_data, how='vertical_relaxed')
            print(f"✓ Combined {len(all_data)} FRED series into single DataFrame")
            return combined_df
        else:
            print("Warning: No FRED data collected")
            return pl.DataFrame()
    
    def fetch_bls_data(self, series_ids: List[str], start_year: int = 2010, end_year: int = 2024) -> pl.DataFrame:
        """
        Fetch data from BLS API.
        
        Args:
            series_ids: List of BLS series identifiers
            start_year: Start year for data collection
            end_year: End year for data collection
            
        Returns:
            Polars DataFrame with BLS data
        """
        if not BLS_API_KEY:
            print("Warning: BLS API key not configured - using public API with limitations")
        
        try:
            # Prepare API request
            headers = {'Content-type': 'application/json'}
            data = {
                'seriesid': series_ids,
                'startyear': str(start_year),
                'endyear': str(end_year),
                'catalog': False,
                'calculations': True,
                'annualaverage': False
            }
            
            if BLS_API_KEY:
                data['registrationkey'] = BLS_API_KEY
            
            # Make API request
            response = self.session.post(
                'https://api.bls.gov/publicAPI/v2/timeseries/data/',
                json=data,
                headers=headers
            )
            response.raise_for_status()
            
            json_data = response.json()
            
            # Parse response
            all_series_data = []
            
            for series in json_data['Results']['series']:
                series_id = series['seriesID']
                series_data = []
                
                for item in series['data']:
                    # Create date from year and period
                    year = int(item['year'])
                    period = item['period']
                    
                    if period.startswith('M'):  # Monthly data
                        month = int(period[1:])
                        date = datetime(year, month, 1)
                    else:
                        continue  # Skip non-monthly data for now
                    
                    series_data.append({
                        'date': date,
                        'value': float(item['value']) if item['value'] != '.' else None,
                        'series_id': series_id,
                        'series_name': f"BLS_{series_id}"
                    })
                
                all_series_data.extend(series_data)
            
            # Convert to polars DataFrame
            if all_series_data:
                df = pl.DataFrame(all_series_data)
                df = df.with_columns([
                    pl.col('date').cast(pl.Date),
                    pl.col('value').cast(pl.Float64),
                    pl.col('series_id').cast(pl.String),
                    pl.col('series_name').cast(pl.String)
                ])
                
                print(f"✓ Fetched {len(df)} observations from BLS")
                return df
            else:
                print("Warning: No BLS data collected")
                return pl.DataFrame()
                
        except Exception as e:
            print(f"Error fetching BLS data: {str(e)}")
            return pl.DataFrame()
    
    def fetch_eurostat_data(self, dataset_code: str, filters: Optional[Dict[str, str]] = None) -> pl.DataFrame:
        """
        Fetch data from Eurostat API.
        
        Args:
            dataset_code: Eurostat dataset code
            filters: Dictionary of filters to apply
            
        Returns:
            Polars DataFrame with Eurostat data
        """
        try:
            # Build URL
            url = f"{EUROSTAT_BASE_URL}/{dataset_code}"
            
            # Add filters if provided
            if filters:
                filter_str = "&".join([f"{k}={v}" for k, v in filters.items()])
                url += f"?{filter_str}"
            
            # Make request
            response = self.session.get(url)
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            # Extract time series data (this is a simplified parser)
            # Real Eurostat API responses are complex and would need more sophisticated parsing
            time_series_data = []
            
            # This is a placeholder - actual implementation would depend on Eurostat response structure
            print(f"✓ Connected to Eurostat API for dataset {dataset_code}")
            print("Note: Eurostat data parsing implementation needs refinement based on actual API response structure")
            
            return pl.DataFrame()
            
        except Exception as e:
            print(f"Error fetching Eurostat data: {str(e)}")
            return pl.DataFrame()
    
    def fetch_ecb_data(self, series_key: str) -> pl.DataFrame:
        """
        Fetch data from ECB Statistical Data Warehouse.
        
        Args:
            series_key: ECB series key
            
        Returns:
            Polars DataFrame with ECB data
        """
        try:
            # ECB API endpoint
            url = f"{ECB_BASE_URL}/data/{series_key}"
            
            # Add format parameter
            params = {
                'format': 'jsondata',
                'startPeriod': START_DATE,
                'endPeriod': END_DATE
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            print(f"✓ Connected to ECB API for series {series_key}")
            print("Note: ECB data parsing implementation needs refinement based on actual API response structure")
            
            return pl.DataFrame()
            
        except Exception as e:
            print(f"Error fetching ECB data: {str(e)}")
            return pl.DataFrame()
    
    def collect_all_data(self) -> Dict[str, pl.DataFrame]:
        """
        Collect all data from various sources and return organized datasets.
        
        Returns:
            Dictionary containing DataFrames from different sources
        """
        print("Starting comprehensive data collection...")
        
        datasets = {}
        
        # Collect FRED data
        fred_data = self.fetch_all_fred_data()
        if not fred_data.is_empty():
            datasets['fred'] = fred_data
        
        # Collect BLS data
        bls_series = list(BLS_TRAVEL_SERIES.values())
        if bls_series:
            bls_data = self.fetch_bls_data(bls_series)
            if not bls_data.is_empty():
                datasets['bls'] = bls_data
        
        # Note: Eurostat and ECB data collection would be implemented here
        # once we have the specific dataset codes and proper parsing logic
        
        print(f"✓ Data collection complete. Collected {len(datasets)} datasets.")
        return datasets
    
    def save_data(self, datasets: Dict[str, pl.DataFrame], format: str = 'parquet') -> None:
        """
        Save collected datasets to disk.
        
        Args:
            datasets: Dictionary of DataFrames to save
            format: File format ('parquet', 'csv', or 'json')
        """
        print(f"Saving datasets in {format} format...")
        
        for name, df in datasets.items():
            if df.is_empty():
                continue
                
            filename = f"{DATA_DIR}/{name}_data.{format}"
            
            try:
                if format == 'parquet':
                    df.write_parquet(filename)
                elif format == 'csv':
                    df.write_csv(filename)
                elif format == 'json':
                    df.write_json(filename)
                else:
                    raise ValueError(f"Unsupported format: {format}")
                
                print(f"✓ Saved {name} dataset to {filename}")
                
            except Exception as e:
                print(f"Error saving {name} dataset: {str(e)}")
        
        print("Data saving complete.")
    
    def load_data(self, format: str = 'parquet') -> Dict[str, pl.DataFrame]:
        """
        Load previously saved datasets.
        
        Args:
            format: File format to load ('parquet', 'csv', or 'json')
            
        Returns:
            Dictionary of loaded DataFrames
        """
        import glob
        
        datasets = {}
        pattern = f"{DATA_DIR}/*_data.{format}"
        
        for filepath in glob.glob(pattern):
            name = filepath.split('/')[-1].replace(f'_data.{format}', '')
            
            try:
                if format == 'parquet':
                    df = pl.read_parquet(filepath)
                elif format == 'csv':
                    df = pl.read_csv(filepath)
                elif format == 'json':
                    df = pl.read_json(filepath)
                else:
                    raise ValueError(f"Unsupported format: {format}")
                
                datasets[name] = df
                print(f"✓ Loaded {name} dataset from {filepath}")
                
            except Exception as e:
                print(f"Error loading {filepath}: {str(e)}")
        
        return datasets


def main():
    """Main function to demonstrate data collection."""
    from config import validate_api_keys
    
    # Validate API keys
    validate_api_keys()
    
    # Initialize collector
    collector = DataCollector()
    
    # Collect all data
    datasets = collector.collect_all_data()
    
    # Save data
    collector.save_data(datasets)
    
    # Display summary
    print("\n" + "="*50)
    print("DATA COLLECTION SUMMARY")
    print("="*50)
    
    for name, df in datasets.items():
        print(f"\n{name.upper()} Dataset:")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {df.width}")
        if not df.is_empty():
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"  Series: {df['series_name'].unique().to_list()}")


if __name__ == "__main__":
    main() 