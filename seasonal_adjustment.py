"""
Seasonal adjustment module for EU HICP Package Holidays Price Forecast.
Implements X-13ARIMA-SEATS equivalent, STL decomposition, and custom seasonal factors.
"""

import polars as pl
import numpy as np
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.x13 import x13_arima_analysis
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
import warnings
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class SeasonalAdjuster:
    """
    Comprehensive seasonal adjustment class for HICP time series.
    Supports multiple methods: X-13ARIMA-SEATS, STL, and custom approaches.
    """
    
    def __init__(self, method: str = 'stl', period: int = 12):
        """
        Initialize seasonal adjuster.
        
        Args:
            method: Seasonal adjustment method ('stl', 'x13', 'custom')
            period: Seasonal period (12 for monthly data)
        """
        self.method = method
        self.period = period
        self.seasonal_factors = {}
        self.decomposition_results = {}
        
    def prepare_series(self, df: pl.DataFrame, series_name: str, 
                      date_col: str = 'date', value_col: str = 'value') -> pl.DataFrame:
        """
        Prepare time series for seasonal adjustment.
        
        Args:
            df: Input DataFrame
            series_name: Name of the series to adjust
            date_col: Date column name
            value_col: Value column name
            
        Returns:
            Prepared DataFrame with proper time index
        """
        # Filter for specific series
        series_data = df.filter(pl.col('series_name') == series_name)
        
        if series_data.is_empty():
            raise ValueError(f"Series '{series_name}' not found in data")
        
        # Sort by date and ensure no duplicates
        series_data = (series_data
                      .sort(date_col)
                      .unique(subset=[date_col], keep='first'))
        
        # Check for missing dates and interpolate if needed
        date_range = pl.date_range(
            series_data[date_col].min(), 
            series_data[date_col].max(), 
            interval='1mo'
        )
        
        # Create complete date range DataFrame
        complete_dates = pl.DataFrame({date_col: date_range})
        
        # Join with original data
        complete_series = (complete_dates
                          .join(series_data, on=date_col, how='left')
                          .with_columns([
                              pl.col(value_col).interpolate().alias(value_col),
                              pl.lit(series_name).alias('series_name')
                          ]))
        
        return complete_series
    
    def stl_decomposition(self, series: pl.Series, seasonal: int = 13, 
                         trend: Optional[int] = None, robust: bool = True) -> Dict:
        """
        Perform STL (Seasonal and Trend decomposition using Loess) decomposition.
        
        Args:
            series: Time series data
            seasonal: Length of seasonal smoother
            trend: Length of trend smoother
            robust: Use robust fitting
            
        Returns:
            Dictionary with decomposition components
        """
        # Convert to pandas for statsmodels compatibility
        ts_data = series.to_pandas()
        
        # Remove any NaN values
        ts_data = ts_data.dropna()
        
        if len(ts_data) < 2 * self.period:
            warnings.warn(f"Series too short for reliable seasonal adjustment. Length: {len(ts_data)}")
            return self._create_empty_decomposition(len(ts_data))
        
        try:
            # Perform STL decomposition
            stl = STL(ts_data, seasonal=seasonal, trend=trend, robust=robust, period=self.period)
            result = stl.fit()
            
            # Calculate seasonally adjusted series
            seasonal_adj = ts_data - result.seasonal
            
            return {
                'original': ts_data,
                'trend': result.trend,
                'seasonal': result.seasonal,
                'residual': result.resid,
                'seasonally_adjusted': seasonal_adj,
                'seasonal_factors': result.seasonal / ts_data.mean(),
                'method': 'STL'
            }
            
        except Exception as e:
            warnings.warn(f"STL decomposition failed: {str(e)}")
            return self._create_empty_decomposition(len(ts_data))
    
    def x13_decomposition(self, series: pl.Series, **kwargs) -> Dict:
        """
        Perform X-13ARIMA-SEATS seasonal adjustment.
        
        Args:
            series: Time series data
            **kwargs: Additional arguments for X-13
            
        Returns:
            Dictionary with decomposition components
        """
        # Convert to pandas for statsmodels compatibility
        ts_data = series.to_pandas()
        ts_data = ts_data.dropna()
        
        if len(ts_data) < 3 * self.period:
            warnings.warn(f"Series too short for X-13 adjustment. Length: {len(ts_data)}")
            return self._create_empty_decomposition(len(ts_data))
        
        try:
            # Attempt X-13 analysis (requires X-13ARIMA-SEATS software)
            result = x13_arima_analysis(ts_data, **kwargs)
            
            return {
                'original': ts_data,
                'trend': result.trend,
                'seasonal': result.seasonal,
                'irregular': result.irregular,
                'seasonally_adjusted': result.seasadj,
                'seasonal_factors': result.seasonal / ts_data.mean(),
                'method': 'X-13ARIMA-SEATS'
            }
            
        except Exception as e:
            warnings.warn(f"X-13 decomposition failed: {str(e)}. Falling back to STL.")
            return self.stl_decomposition(series)
    
    def custom_seasonal_adjustment(self, series: pl.Series, 
                                  holiday_calendar: Optional[List] = None) -> Dict:
        """
        Custom seasonal adjustment incorporating holiday effects and booking patterns.
        
        Args:
            series: Time series data
            holiday_calendar: List of holiday dates affecting travel bookings
            
        Returns:
            Dictionary with decomposition components
        """
        ts_data = series.to_pandas()
        ts_data = ts_data.dropna()
        
        if len(ts_data) < 2 * self.period:
            warnings.warn(f"Series too short for custom adjustment. Length: {len(ts_data)}")
            return self._create_empty_decomposition(len(ts_data))
        
        try:
            # Calculate moving averages for trend
            trend = ts_data.rolling(window=self.period, center=True).mean()
            
            # Detrend the series
            detrended = ts_data - trend
            
            # Calculate seasonal factors by month
            seasonal_factors = {}
            for month in range(1, 13):
                month_data = detrended[detrended.index.month == month]
                if len(month_data) > 0:
                    seasonal_factors[month] = month_data.mean()
                else:
                    seasonal_factors[month] = 0
            
            # Create seasonal component
            seasonal = pd.Series(
                [seasonal_factors[month] for month in ts_data.index.month],
                index=ts_data.index
            )
            
            # Calculate residual
            residual = ts_data - trend - seasonal
            
            # Seasonally adjusted series
            seasonal_adj = ts_data - seasonal
            
            return {
                'original': ts_data,
                'trend': trend,
                'seasonal': seasonal,
                'residual': residual,
                'seasonally_adjusted': seasonal_adj,
                'seasonal_factors': seasonal / ts_data.mean(),
                'method': 'Custom'
            }
            
        except Exception as e:
            warnings.warn(f"Custom decomposition failed: {str(e)}")
            return self._create_empty_decomposition(len(ts_data))
    
    def adjust_series(self, df: pl.DataFrame, series_name: str, 
                     **kwargs) -> pl.DataFrame:
        """
        Apply seasonal adjustment to a specific series.
        
        Args:
            df: Input DataFrame
            series_name: Name of series to adjust
            **kwargs: Method-specific parameters
            
        Returns:
            DataFrame with seasonal adjustment results
        """
        # Prepare the series
        prepared_data = self.prepare_series(df, series_name)
        series_values = prepared_data['value']
        
        # Apply chosen method
        if self.method == 'stl':
            result = self.stl_decomposition(series_values, **kwargs)
        elif self.method == 'x13':
            result = self.x13_decomposition(series_values, **kwargs)
        elif self.method == 'custom':
            result = self.custom_seasonal_adjustment(series_values, **kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Store results
        self.decomposition_results[series_name] = result
        
        # Create output DataFrame
        output_df = prepared_data.with_columns([
            pl.Series('trend', result['trend']).alias('trend'),
            pl.Series('seasonal', result['seasonal']).alias('seasonal'),
            pl.Series('seasonally_adjusted', result['seasonally_adjusted']).alias('value_sa'),
            pl.Series('seasonal_factors', result['seasonal_factors']).alias('seasonal_factors')
        ])
        
        # Add residual column (different names for different methods)
        if 'residual' in result:
            output_df = output_df.with_columns(
                pl.Series('residual', result['residual']).alias('residual')
            )
        elif 'irregular' in result:
            output_df = output_df.with_columns(
                pl.Series('irregular', result['irregular']).alias('residual')
            )
        
        return output_df
    
    def calculate_mom_changes(self, df: pl.DataFrame, value_col: str = 'value_sa') -> pl.DataFrame:
        """
        Calculate month-over-month percentage changes for seasonally adjusted data.
        
        Args:
            df: DataFrame with seasonally adjusted values
            value_col: Column name for values
            
        Returns:
            DataFrame with MoM percentage changes
        """
        return df.with_columns([
            # Month-over-month change
            (pl.col(value_col).pct_change() * 100).alias('mom_pct_sa'),
            # Year-over-year change
            (pl.col(value_col).pct_change(periods=12) * 100).alias('yoy_pct_sa'),
            # 3-month moving average of MoM changes
            (pl.col(value_col).pct_change() * 100).rolling_mean(window_size=3).alias('mom_pct_sa_3ma')
        ])
    
    def _create_empty_decomposition(self, length: int) -> Dict:
        """Create empty decomposition result for error cases."""
        empty_series = pd.Series([np.nan] * length)
        return {
            'original': empty_series,
            'trend': empty_series,
            'seasonal': empty_series,
            'residual': empty_series,
            'seasonally_adjusted': empty_series,
            'seasonal_factors': empty_series,
            'method': 'Failed'
        }
    
    def plot_decomposition(self, series_name: str, figsize: Tuple[int, int] = (12, 10)) -> go.Figure:
        """
        Create interactive plot of seasonal decomposition.
        
        Args:
            series_name: Name of series to plot
            figsize: Figure size tuple
            
        Returns:
            Plotly figure object
        """
        if series_name not in self.decomposition_results:
            raise ValueError(f"No decomposition results found for {series_name}")
        
        result = self.decomposition_results[series_name]
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=[
                f'{series_name} - Original',
                'Trend',
                'Seasonal',
                'Residual/Irregular'
            ],
            vertical_spacing=0.08
        )
        
        # Original series
        fig.add_trace(
            go.Scatter(
                x=result['original'].index,
                y=result['original'].values,
                name='Original',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Seasonally adjusted
        fig.add_trace(
            go.Scatter(
                x=result['seasonally_adjusted'].index,
                y=result['seasonally_adjusted'].values,
                name='Seasonally Adjusted',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # Trend
        fig.add_trace(
            go.Scatter(
                x=result['trend'].index,
                y=result['trend'].values,
                name='Trend',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        
        # Seasonal
        fig.add_trace(
            go.Scatter(
                x=result['seasonal'].index,
                y=result['seasonal'].values,
                name='Seasonal',
                line=dict(color='orange')
            ),
            row=3, col=1
        )
        
        # Residual
        residual_key = 'residual' if 'residual' in result else 'irregular'
        fig.add_trace(
            go.Scatter(
                x=result[residual_key].index,
                y=result[residual_key].values,
                name='Residual',
                line=dict(color='purple')
            ),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title=f'Seasonal Decomposition - {series_name} ({result["method"]})',
            showlegend=True
        )
        
        return fig
    
    def get_seasonal_summary(self, series_name: str) -> Dict:
        """
        Get summary statistics for seasonal adjustment.
        
        Args:
            series_name: Name of series
            
        Returns:
            Dictionary with summary statistics
        """
        if series_name not in self.decomposition_results:
            raise ValueError(f"No decomposition results found for {series_name}")
        
        result = self.decomposition_results[series_name]
        
        # Calculate statistics
        original = result['original'].dropna()
        seasonal_adj = result['seasonally_adjusted'].dropna()
        seasonal = result['seasonal'].dropna()
        
        return {
            'series_name': series_name,
            'method': result['method'],
            'observations': len(original),
            'original_stats': {
                'mean': original.mean(),
                'std': original.std(),
                'min': original.min(),
                'max': original.max()
            },
            'seasonal_adj_stats': {
                'mean': seasonal_adj.mean(),
                'std': seasonal_adj.std(),
                'min': seasonal_adj.min(),
                'max': seasonal_adj.max()
            },
            'seasonal_strength': seasonal.std() / original.std() if original.std() > 0 else 0,
            'seasonal_range': seasonal.max() - seasonal.min(),
            'trend_strength': result['trend'].dropna().std() / original.std() if original.std() > 0 else 0
        }


def batch_seasonal_adjustment(df: pl.DataFrame, series_list: List[str], 
                            method: str = 'stl', **kwargs) -> pl.DataFrame:
    """
    Apply seasonal adjustment to multiple series in batch.
    
    Args:
        df: Input DataFrame
        series_list: List of series names to adjust
        method: Seasonal adjustment method
        **kwargs: Method-specific parameters
        
    Returns:
        Combined DataFrame with all seasonally adjusted series
    """
    adjuster = SeasonalAdjuster(method=method)
    adjusted_series = []
    
    for series_name in series_list:
        try:
            print(f"Processing {series_name}...")
            adjusted_df = adjuster.adjust_series(df, series_name, **kwargs)
            adjusted_df = adjuster.calculate_mom_changes(adjusted_df)
            adjusted_series.append(adjusted_df)
            print(f"✓ Completed {series_name}")
        except Exception as e:
            print(f"❌ Failed to process {series_name}: {str(e)}")
            continue
    
    if adjusted_series:
        return pl.concat(adjusted_series, how='vertical_relaxed')
    else:
        return pl.DataFrame()


# Import pandas for statsmodels compatibility
import pandas as pd
