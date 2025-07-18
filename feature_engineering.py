"""
Feature engineering module for EU HICP Package Holidays Price Forecast.
Creates temporal, cross-country, and economic features for forecasting models.
"""

import polars as pl
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
from scipy import stats
import holidays

class FeatureEngineer:
    """
    Comprehensive feature engineering class for HICP forecasting.
    Creates temporal, cross-country, and economic indicator features.
    """
    
    def __init__(self, target_countries: List[str] = ['EU', 'Germany']):
        """
        Initialize feature engineer.
        
        Args:
            target_countries: List of countries/regions to focus on
        """
        self.target_countries = target_countries
        self.feature_store = {}
        self.holiday_calendars = self._initialize_holiday_calendars()
        
    def _initialize_holiday_calendars(self) -> Dict:
        """Initialize holiday calendars for EU countries."""
        calendars = {}
        
        # EU-wide holidays (approximate)
        calendars['EU'] = holidays.country_holidays('DE')  # Use Germany as proxy
        
        # Country-specific holidays
        calendars['Germany'] = holidays.country_holidays('DE')
        calendars['France'] = holidays.country_holidays('FR')
        calendars['Spain'] = holidays.country_holidays('ES')
        calendars['Italy'] = holidays.country_holidays('IT')
        
        return calendars
    
    def create_temporal_features(self, df: pl.DataFrame, 
                                date_col: str = 'date') -> pl.DataFrame:
        """
        Create comprehensive temporal features.
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            
        Returns:
            DataFrame with temporal features
        """
        # Ensure date column is properly typed
        df = df.with_columns(pl.col(date_col).cast(pl.Date))
        
        # Basic temporal features
        df = df.with_columns([
            # Year, month, quarter
            pl.col(date_col).dt.year().alias('year'),
            pl.col(date_col).dt.month().alias('month'),
            pl.col(date_col).dt.quarter().alias('quarter'),
            
            # Day of year and week
            pl.col(date_col).dt.ordinal_day().alias('day_of_year'),
            pl.col(date_col).dt.week().alias('week_of_year'),
            
            # Seasonal indicators
            ((pl.col(date_col).dt.month().is_in([6, 7, 8]))).alias('is_summer'),
            ((pl.col(date_col).dt.month().is_in([12, 1, 2]))).alias('is_winter'),
            ((pl.col(date_col).dt.month().is_in([3, 4, 5]))).alias('is_spring'),
            ((pl.col(date_col).dt.month().is_in([9, 10, 11]))).alias('is_autumn'),
            
            # Peak travel months
            ((pl.col(date_col).dt.month().is_in([7, 8]))).alias('is_peak_summer'),
            ((pl.col(date_col).dt.month().is_in([12, 1]))).alias('is_winter_holidays'),
            ((pl.col(date_col).dt.month().is_in([3, 4]))).alias('is_easter_period'),
            
            # School holiday proxies (Northern Hemisphere)
            ((pl.col(date_col).dt.month().is_in([7, 8]))).alias('is_summer_holidays'),
            ((pl.col(date_col).dt.month().is_in([12, 1]))).alias('is_winter_break'),
            ((pl.col(date_col).dt.month() == 4)).alias('is_spring_break'),
        ])
        
        # Cyclical encoding for month (captures seasonality)
        df = df.with_columns([
            (2 * np.pi * pl.col('month') / 12).sin().alias('month_sin'),
            (2 * np.pi * pl.col('month') / 12).cos().alias('month_cos'),
            (2 * np.pi * pl.col('quarter') / 4).sin().alias('quarter_sin'),
            (2 * np.pi * pl.col('quarter') / 4).cos().alias('quarter_cos')
        ])
        
        # Time trend features
        min_date = df[date_col].min()
        df = df.with_columns([
            # Linear time trend
            (pl.col(date_col) - min_date).dt.total_days().alias('time_trend'),
            
            # Quadratic time trend
            ((pl.col(date_col) - min_date).dt.total_days() ** 2).alias('time_trend_sq')
        ])
        
        return df
    
    def create_holiday_features(self, df: pl.DataFrame, 
                               date_col: str = 'date',
                               country: str = 'EU') -> pl.DataFrame:
        """
        Create holiday-related features.
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            country: Country for holiday calendar
            
        Returns:
            DataFrame with holiday features
        """
        if country not in self.holiday_calendars:
            warnings.warn(f"Holiday calendar for {country} not available")
            return df
        
        holiday_cal = self.holiday_calendars[country]
        
        # Convert to pandas for holiday processing
        dates = df[date_col].to_pandas()
        
        # Create holiday indicators
        is_holiday = [date in holiday_cal for date in dates]
        
        # Holiday proximity features
        days_to_holiday = []
        days_from_holiday = []
        
        for date in dates:
            # Find nearest holidays
            future_holidays = [h for h in holiday_cal.keys() if h > date]
            past_holidays = [h for h in holiday_cal.keys() if h < date]
            
            if future_holidays:
                days_to_next = min([(h - date).days for h in future_holidays])
                days_to_holiday.append(days_to_next if days_to_next <= 30 else 30)
            else:
                days_to_holiday.append(30)
            
            if past_holidays:
                days_from_last = min([(date - h).days for h in past_holidays])
                days_from_holiday.append(days_from_last if days_from_last <= 30 else 30)
            else:
                days_from_holiday.append(30)
        
        # Add holiday features
        df = df.with_columns([
            pl.Series('is_holiday', is_holiday).alias(f'is_holiday_{country.lower()}'),
            pl.Series('days_to_holiday', days_to_holiday).alias(f'days_to_holiday_{country.lower()}'),
            pl.Series('days_from_holiday', days_from_holiday).alias(f'days_from_holiday_{country.lower()}'),
            
            # Holiday month indicators
            (pl.Series('is_holiday', is_holiday) & (pl.col('month') == 12)).alias('is_christmas_period'),
            (pl.Series('is_holiday', is_holiday) & (pl.col('month').is_in([3, 4]))).alias('is_easter_holiday'),
        ])
        
        return df
    
    def create_lagged_features(self, df: pl.DataFrame, 
                              value_cols: List[str],
                              lags: List[int] = [1, 2, 3, 6, 12]) -> pl.DataFrame:
        """
        Create lagged features for specified columns.
        
        Args:
            df: Input DataFrame
            value_cols: List of columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        # Sort by date and series to ensure proper lagging
        df = df.sort(['series_name', 'date'])
        
        for col in value_cols:
            if col not in df.columns:
                continue
                
            for lag in lags:
                # Create lagged features within each series
                df = df.with_columns([
                    pl.col(col).shift(lag).over('series_name').alias(f'{col}_lag_{lag}')
                ])
        
        return df
    
    def create_rolling_features(self, df: pl.DataFrame,
                               value_cols: List[str],
                               windows: List[int] = [3, 6, 12]) -> pl.DataFrame:
        """
        Create rolling window features.
        
        Args:
            df: Input DataFrame
            value_cols: List of columns to create rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        df = df.sort(['series_name', 'date'])
        
        for col in value_cols:
            if col not in df.columns:
                continue
                
            for window in windows:
                # Rolling statistics within each series
                df = df.with_columns([
                    pl.col(col).rolling_mean(window_size=window).over('series_name').alias(f'{col}_ma_{window}'),
                    pl.col(col).rolling_std(window_size=window).over('series_name').alias(f'{col}_std_{window}'),
                    pl.col(col).rolling_min(window_size=window).over('series_name').alias(f'{col}_min_{window}'),
                    pl.col(col).rolling_max(window_size=window).over('series_name').alias(f'{col}_max_{window}'),
                ])
                
                # Rolling percentiles
                df = df.with_columns([
                    pl.col(col).rolling_quantile(quantile=0.25, window_size=window).over('series_name').alias(f'{col}_q25_{window}'),
                    pl.col(col).rolling_quantile(quantile=0.75, window_size=window).over('series_name').alias(f'{col}_q75_{window}'),
                ])
        
        return df
    
    def create_cross_country_features(self, df: pl.DataFrame,
                                     value_col: str = 'value_sa') -> pl.DataFrame:
        """
        Create cross-country comparison features.
        
        Args:
            df: Input DataFrame with multiple countries
            value_col: Value column to use for comparisons
            
        Returns:
            DataFrame with cross-country features
        """
        # Pivot to wide format for cross-country calculations
        wide_df = df.pivot(
            index='date',
            columns='series_name',
            values=value_col,
            aggregate_function='first'
        )
        
        # Calculate cross-country features
        country_cols = [col for col in wide_df.columns if col != 'date']
        
        if len(country_cols) >= 2:
            # Relative differences between countries
            for i, country1 in enumerate(country_cols):
                for country2 in country_cols[i+1:]:
                    if country1 in wide_df.columns and country2 in wide_df.columns:
                        # Relative difference
                        wide_df = wide_df.with_columns([
                            ((pl.col(country1) - pl.col(country2)) / pl.col(country2) * 100).alias(f'{country1}_vs_{country2}_pct_diff')
                        ])
                        
                        # Ratio
                        wide_df = wide_df.with_columns([
                            (pl.col(country1) / pl.col(country2)).alias(f'{country1}_vs_{country2}_ratio')
                        ])
            
            # Cross-country correlations (rolling)
            for window in [6, 12]:
                if len(country_cols) >= 2:
                    country1, country2 = country_cols[0], country_cols[1]
                    if country1 in wide_df.columns and country2 in wide_df.columns:
                        # Rolling correlation (approximate using covariance)
                        wide_df = wide_df.with_columns([
                            pl.corr(pl.col(country1), pl.col(country2), method='pearson').rolling_mean(window_size=window).alias(f'corr_{country1}_{country2}_{window}m')
                        ])
        
        # Melt back to long format
        long_df = wide_df.melt(
            id_vars=['date'],
            variable_name='feature_name',
            value_name='feature_value'
        )
        
        # Join back with original data
        result_df = df.join(
            long_df.filter(pl.col('feature_name').str.contains('_vs_|corr_')),
            on='date',
            how='left'
        )
        
        return result_df
    
    def create_economic_features(self, df: pl.DataFrame,
                                economic_data: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        Create economic indicator features.
        
        Args:
            df: Input DataFrame
            economic_data: DataFrame with economic indicators
            
        Returns:
            DataFrame with economic features
        """
        if economic_data is None:
            # Create placeholder economic features
            df = df.with_columns([
                # Economic cycle proxies
                (pl.col('year') % 4).alias('election_cycle'),
                ((pl.col('month') - 1) % 3).alias('quarter_month'),
                
                # Crisis indicators (simplified)
                ((pl.col('year') == 2008) | (pl.col('year') == 2009)).alias('is_financial_crisis'),
                ((pl.col('year') >= 2020) & (pl.col('year') <= 2022)).alias('is_covid_period'),
                ((pl.col('year') >= 2022)).alias('is_ukraine_war_period'),
            ])
            
            return df
        
        # Join with economic data
        economic_features = economic_data.select([
            'date', 'series_name', 'value'
        ]).pivot(
            index='date',
            columns='series_name',
            values='value',
            aggregate_function='first'
        )
        
        # Join with main data
        df = df.join(economic_features, on='date', how='left')
        
        # Create derived economic features
        econ_cols = [col for col in economic_features.columns if col != 'date']
        
        for col in econ_cols:
            if col in df.columns:
                # Growth rates
                df = df.with_columns([
                    (pl.col(col).pct_change() * 100).alias(f'{col}_growth'),
                    (pl.col(col).pct_change(periods=12) * 100).alias(f'{col}_yoy_growth'),
                ])
                
                # Volatility measures
                df = df.with_columns([
                    pl.col(col).rolling_std(window_size=6).alias(f'{col}_volatility_6m'),
                    pl.col(col).rolling_std(window_size=12).alias(f'{col}_volatility_12m'),
                ])
        
        return df
    
    def create_interaction_features(self, df: pl.DataFrame,
                                   feature_pairs: List[Tuple[str, str]]) -> pl.DataFrame:
        """
        Create interaction features between specified columns.
        
        Args:
            df: Input DataFrame
            feature_pairs: List of tuples specifying feature pairs
            
        Returns:
            DataFrame with interaction features
        """
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                # Multiplicative interaction
                df = df.with_columns([
                    (pl.col(feat1) * pl.col(feat2)).alias(f'{feat1}_x_{feat2}')
                ])
                
                # Additive interaction (normalized)
                df = df.with_columns([
                    ((pl.col(feat1) + pl.col(feat2)) / 2).alias(f'{feat1}_plus_{feat2}')
                ])
        
        return df
    
    def create_target_features(self, df: pl.DataFrame,
                              target_col: str = 'mom_pct_sa',
                              horizons: List[int] = [1, 3, 6]) -> pl.DataFrame:
        """
        Create target variable features for different forecasting horizons.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            horizons: List of forecasting horizons
            
        Returns:
            DataFrame with target features
        """
        df = df.sort(['series_name', 'date'])
        
        for horizon in horizons:
            # Future target values (for training)
            df = df.with_columns([
                pl.col(target_col).shift(-horizon).over('series_name').alias(f'{target_col}_future_{horizon}m')
            ])
            
            # Rolling future averages
            df = df.with_columns([
                pl.col(target_col).shift(-horizon).rolling_mean(window_size=3).over('series_name').alias(f'{target_col}_future_{horizon}m_avg')
            ])
        
        return df
    
    def build_feature_store(self, df: pl.DataFrame,
                           economic_data: Optional[pl.DataFrame] = None,
                           include_cross_country: bool = True,
                           include_holidays: bool = True) -> pl.DataFrame:
        """
        Build comprehensive feature store with all feature types.
        
        Args:
            df: Input DataFrame
            economic_data: Optional economic indicators DataFrame
            include_cross_country: Whether to include cross-country features
            include_holidays: Whether to include holiday features
            
        Returns:
            DataFrame with comprehensive feature set
        """
        print("Building comprehensive feature store...")
        
        # Start with temporal features
        print("✓ Creating temporal features...")
        df = self.create_temporal_features(df)
        
        # Add holiday features
        if include_holidays:
            print("✓ Creating holiday features...")
            for country in self.target_countries:
                df = self.create_holiday_features(df, country=country)
        
        # Create lagged features for key variables
        print("✓ Creating lagged features...")
        value_cols = ['value', 'value_sa', 'mom_pct_sa'] if 'value_sa' in df.columns else ['value']
        df = self.create_lagged_features(df, value_cols)
        
        # Create rolling features
        print("✓ Creating rolling features...")
        df = self.create_rolling_features(df, value_cols)
        
        # Add economic features
        print("✓ Creating economic features...")
        df = self.create_economic_features(df, economic_data)
        
        # Cross-country features (if multiple series)
        if include_cross_country and df['series_name'].n_unique() > 1:
            print("✓ Creating cross-country features...")
            df = self.create_cross_country_features(df)
        
        # Create some key interaction features
        print("✓ Creating interaction features...")
        interaction_pairs = [
            ('is_summer', 'is_holiday_eu'),
            ('month_sin', 'time_trend'),
            ('is_peak_summer', 'days_to_holiday_eu')
        ]
        # Filter pairs that exist in the data
        valid_pairs = [(f1, f2) for f1, f2 in interaction_pairs 
                      if f1 in df.columns and f2 in df.columns]
        if valid_pairs:
            df = self.create_interaction_features(df, valid_pairs)
        
        # Create target features for modeling
        if 'mom_pct_sa' in df.columns:
            print("✓ Creating target features...")
            df = self.create_target_features(df)
        
        # Store feature metadata
        self.feature_store = {
            'total_features': df.width,
            'feature_names': df.columns,
            'temporal_features': [col for col in df.columns if any(x in col for x in ['month', 'quarter', 'year', 'season', 'time_trend'])],
            'holiday_features': [col for col in df.columns if 'holiday' in col],
            'lagged_features': [col for col in df.columns if '_lag_' in col],
            'rolling_features': [col for col in df.columns if any(x in col for x in ['_ma_', '_std_', '_min_', '_max_', '_q25_', '_q75_'])],
            'cross_country_features': [col for col in df.columns if '_vs_' in col or 'corr_' in col],
            'economic_features': [col for col in df.columns if any(x in col for x in ['growth', 'volatility', 'crisis'])],
            'interaction_features': [col for col in df.columns if '_x_' in col or '_plus_' in col],
            'target_features': [col for col in df.columns if 'future_' in col]
        }
        
        print(f"✓ Feature store complete: {df.width} total features")
        print(f"  - Temporal: {len(self.feature_store['temporal_features'])}")
        print(f"  - Holiday: {len(self.feature_store['holiday_features'])}")
        print(f"  - Lagged: {len(self.feature_store['lagged_features'])}")
        print(f"  - Rolling: {len(self.feature_store['rolling_features'])}")
        print(f"  - Cross-country: {len(self.feature_store['cross_country_features'])}")
        print(f"  - Economic: {len(self.feature_store['economic_features'])}")
        print(f"  - Interaction: {len(self.feature_store['interaction_features'])}")
        print(f"  - Target: {len(self.feature_store['target_features'])}")
        
        return df
    
    def get_feature_importance_data(self, df: pl.DataFrame,
                                   target_col: str = 'mom_pct_sa') -> pl.DataFrame:
        """
        Prepare data for feature importance analysis.
        
        Args:
            df: Feature-rich DataFrame
            target_col: Target column name
            
        Returns:
            DataFrame ready for ML feature importance analysis
        """
        # Remove rows with missing target
        clean_df = df.filter(pl.col(target_col).is_not_null())
        
        # Get feature columns (exclude metadata and target)
        exclude_cols = ['date', 'series_name', 'series_id', target_col] + \
                      [col for col in df.columns if 'future_' in col]
        
        feature_cols = [col for col in clean_df.columns if col not in exclude_cols]
        
        # Select features and target
        analysis_df = clean_df.select(feature_cols + [target_col])
        
        # Remove columns with too many missing values
        missing_threshold = 0.5
        cols_to_keep = []
        
        for col in feature_cols:
            missing_pct = analysis_df[col].null_count() / len(analysis_df)
            if missing_pct < missing_threshold:
                cols_to_keep.append(col)
        
        final_df = analysis_df.select(cols_to_keep + [target_col])
        
        print(f"Feature importance dataset: {len(final_df)} rows, {len(cols_to_keep)} features")
        
        return final_df
    
    def save_feature_store(self, df: pl.DataFrame, filepath: str = 'data/feature_store.parquet'):
        """Save feature store to disk."""
        df.write_parquet(filepath)
        
        # Save metadata
        import json
        metadata_path = filepath.replace('.parquet', '_metadata.json')
        with open(metadata_path, 'w') as f:
            # Convert to serializable format
            serializable_metadata = {
                k: v if isinstance(v, (int, str, float)) else list(v) 
                for k, v in self.feature_store.items()
            }
            json.dump(serializable_metadata, f, indent=2)
        
        print(f"✓ Feature store saved to {filepath}")
        print(f"✓ Metadata saved to {metadata_path}")


def create_modeling_dataset(df: pl.DataFrame, 
                           target_col: str = 'mom_pct_sa',
                           test_months: int = 12) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split dataset into training and testing sets for modeling.
    
    Args:
        df: Feature-rich DataFrame
        target_col: Target column name
        test_months: Number of months to reserve for testing
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Sort by date
    df = df.sort('date')
    
    # Find split date
    max_date = df['date'].max()
    split_date = max_date - timedelta(days=test_months * 30)
    
    # Split data
    train_df = df.filter(pl.col('date') <= split_date)
    test_df = df.filter(pl.col('date') > split_date)
    
    print(f"Dataset split:")
    print(f"  Training: {len(train_df)} observations (up to {split_date})")
    print(f"  Testing: {len(test_df)} observations (from {split_date})")
    
    return train_df, test_df
