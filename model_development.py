"""
Model development module for EU HICP Package Holidays Price Forecast.
Implements ensemble approach with ARIMA/SARIMA, ML models, and VAR for July 2025 forecasting.
"""

import polars as pl
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from datetime import datetime, timedelta
import json

# Time series models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox

# Machine learning models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class TimeSeriesModeler:
    """
    Comprehensive time series modeling class for HICP forecasting.
    Implements ARIMA/SARIMA, ML models, and ensemble methods.
    """
    
    def __init__(self, target_col: str = 'mom_pct_sa', forecast_horizon: int = 1):
        """
        Initialize time series modeler.
        
        Args:
            target_col: Target variable column name
            forecast_horizon: Number of periods to forecast ahead
        """
        self.target_col = target_col
        self.forecast_horizon = forecast_horizon
        self.models = {}
        self.model_results = {}
        self.ensemble_weights = {}
        self.scalers = {}
        
    def prepare_data(self, df: pl.DataFrame, series_name: str) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Prepare data for time series modeling.
        
        Args:
            df: Input DataFrame with features
            series_name: Name of the series to model
            
        Returns:
            Tuple of (target_series, feature_dataframe)
        """
        # Filter for specific series
        series_data = df.filter(pl.col('series_name') == series_name).sort('date')
        
        if series_data.is_empty():
            raise ValueError(f"No data found for series: {series_name}")
        
        # Convert to pandas for modeling
        pandas_data = series_data.to_pandas()
        pandas_data.set_index('date', inplace=True)
        
        # Extract target series
        if self.target_col not in pandas_data.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data")
        
        target_series = pandas_data[self.target_col].dropna()
        
        # Prepare feature matrix (exclude metadata columns)
        exclude_cols = ['series_name', 'series_id', self.target_col] + \
                      [col for col in pandas_data.columns if 'future_' in col]
        
        feature_cols = [col for col in pandas_data.columns if col not in exclude_cols]
        feature_df = pandas_data[feature_cols].loc[target_series.index]
        
        return target_series, feature_df
    
    def check_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """
        Check stationarity of time series using ADF and KPSS tests.
        
        Args:
            series: Time series to test
            
        Returns:
            Dictionary with test results
        """
        # Remove NaN values
        clean_series = series.dropna()
        
        if len(clean_series) < 10:
            return {'error': 'Series too short for stationarity tests'}
        
        try:
            # Augmented Dickey-Fuller test
            adf_result = adfuller(clean_series, autolag='AIC')
            
            # KPSS test
            kpss_result = kpss(clean_series, regression='c', nlags='auto')
            
            return {
                'adf_statistic': adf_result[0],
                'adf_pvalue': adf_result[1],
                'adf_critical_values': adf_result[4],
                'is_stationary_adf': adf_result[1] < 0.05,
                'kpss_statistic': kpss_result[0],
                'kpss_pvalue': kpss_result[1],
                'kpss_critical_values': kpss_result[3],
                'is_stationary_kpss': kpss_result[1] > 0.05,
                'recommendation': 'stationary' if (adf_result[1] < 0.05 and kpss_result[1] > 0.05) else 'non-stationary'
            }
        except Exception as e:
            return {'error': f'Stationarity test failed: {str(e)}'}
    
    def fit_arima_model(self, series: pd.Series, order: Tuple[int, int, int] = None,
                       seasonal_order: Tuple[int, int, int, int] = None,
                       auto_order: bool = True) -> Dict[str, Any]:
        """
        Fit ARIMA or SARIMA model to time series.
        
        Args:
            series: Target time series
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal ARIMA order (P, D, Q, s)
            auto_order: Whether to automatically determine order
            
        Returns:
            Dictionary with model results
        """
        clean_series = series.dropna()
        
        if len(clean_series) < 20:
            return {'error': 'Series too short for ARIMA modeling'}
        
        try:
            if auto_order:
                # Simple auto-order selection based on AIC
                best_aic = np.inf
                best_order = None
                best_seasonal_order = None
                
                # Test different orders
                for p in range(3):
                    for d in range(2):
                        for q in range(3):
                            try:
                                if seasonal_order:
                                    model = SARIMAX(clean_series, order=(p, d, q), 
                                                   seasonal_order=seasonal_order)
                                else:
                                    model = ARIMA(clean_series, order=(p, d, q))
                                
                                fitted_model = model.fit(disp=False)
                                
                                if fitted_model.aic < best_aic:
                                    best_aic = fitted_model.aic
                                    best_order = (p, d, q)
                                    if seasonal_order:
                                        best_seasonal_order = seasonal_order
                                        
                            except:
                                continue
                
                order = best_order if best_order else (1, 1, 1)
                seasonal_order = best_seasonal_order
            
            # Fit final model
            if seasonal_order:
                model = SARIMAX(clean_series, order=order, seasonal_order=seasonal_order)
                model_type = 'SARIMA'
            else:
                model = ARIMA(clean_series, order=order)
                model_type = 'ARIMA'
            
            fitted_model = model.fit(disp=False)
            
            # Model diagnostics
            residuals = fitted_model.resid
            ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
            
            # Forecast
            forecast = fitted_model.forecast(steps=self.forecast_horizon)
            forecast_ci = fitted_model.get_forecast(steps=self.forecast_horizon).conf_int()
            
            return {
                'model_type': model_type,
                'order': order,
                'seasonal_order': seasonal_order,
                'fitted_model': fitted_model,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'forecast': forecast,
                'forecast_ci': forecast_ci,
                'residuals': residuals,
                'ljung_box_pvalue': ljung_box['lb_pvalue'].iloc[-1],
                'params': fitted_model.params,
                'success': True
            }
            
        except Exception as e:
            return {'error': f'ARIMA fitting failed: {str(e)}', 'success': False}
    
    def fit_ml_models(self, target: pd.Series, features: pd.DataFrame,
                     test_size: float = 0.2) -> Dict[str, Any]:
        """
        Fit multiple machine learning models.
        
        Args:
            target: Target variable
            features: Feature matrix
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with ML model results
        """
        # Align target and features
        common_index = target.index.intersection(features.index)
        y = target.loc[common_index]
        X = features.loc[common_index]
        
        # Remove columns with too many missing values
        missing_threshold = 0.5
        X = X.loc[:, X.isnull().mean() < missing_threshold]
        
        # Fill remaining missing values
        X = X.fillna(X.median())
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        if len(y) < 20 or X.shape[1] == 0:
            return {'error': 'Insufficient data for ML modeling'}
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5, test_size=int(len(y) * test_size))
        
        # Initialize models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=1.0, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0)
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                # Scale features for linear models and SVR
                if name in ['Ridge', 'Lasso', 'ElasticNet', 'SVR']:
                    scaler = RobustScaler()
                    X_scaled = pd.DataFrame(
                        scaler.fit_transform(X),
                        index=X.index,
                        columns=X.columns
                    )
                    self.scalers[name] = scaler
                else:
                    X_scaled = X
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, 
                                          scoring='neg_mean_squared_error')
                
                # Fit on full data
                model.fit(X_scaled, y)
                
                # Predictions
                y_pred = model.predict(X_scaled)
                
                # Metrics
                mse = mean_squared_error(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                # Feature importance (if available)
                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(X.columns, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    feature_importance = dict(zip(X.columns, np.abs(model.coef_)))
                
                results[name] = {
                    'model': model,
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'feature_importance': feature_importance,
                    'success': True
                }
                
            except Exception as e:
                results[name] = {'error': f'Model {name} failed: {str(e)}', 'success': False}
        
        return results
    
    def fit_var_model(self, df: pl.DataFrame, series_names: List[str],
                     maxlags: int = 12) -> Dict[str, Any]:
        """
        Fit Vector Autoregression (VAR) model for cross-country analysis.
        
        Args:
            df: DataFrame with multiple series
            series_names: List of series to include in VAR
            maxlags: Maximum number of lags to consider
            
        Returns:
            Dictionary with VAR model results
        """
        try:
            # Prepare data for VAR
            var_data = []
            
            for series_name in series_names:
                series_data = df.filter(pl.col('series_name') == series_name).sort('date')
                if not series_data.is_empty() and self.target_col in series_data.columns:
                    series_values = series_data.select(['date', self.target_col]).to_pandas()
                    series_values.set_index('date', inplace=True)
                    series_values.columns = [series_name]
                    var_data.append(series_values)
            
            if len(var_data) < 2:
                return {'error': 'Need at least 2 series for VAR modeling'}
            
            # Combine series
            combined_data = pd.concat(var_data, axis=1).dropna()
            
            if len(combined_data) < 20:
                return {'error': 'Insufficient data for VAR modeling'}
            
            # Fit VAR model
            model = VAR(combined_data)
            
            # Select optimal lag order
            lag_order = model.select_order(maxlags=min(maxlags, len(combined_data) // 4))
            optimal_lags = lag_order.aic
            
            # Fit with optimal lags
            fitted_model = model.fit(optimal_lags)
            
            # Forecast
            forecast = fitted_model.forecast(combined_data.values, steps=self.forecast_horizon)
            
            # Impulse response analysis
            irf = fitted_model.irf(10)
            
            return {
                'model_type': 'VAR',
                'fitted_model': fitted_model,
                'optimal_lags': optimal_lags,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'forecast': forecast,
                'series_names': series_names,
                'irf': irf,
                'success': True
            }
            
        except Exception as e:
            return {'error': f'VAR modeling failed: {str(e)}', 'success': False}
    
    def create_ensemble_forecast(self, individual_forecasts: Dict[str, float],
                                model_performance: Dict[str, float],
                                method: str = 'weighted_average') -> Dict[str, Any]:
        """
        Create ensemble forecast from individual model predictions.
        
        Args:
            individual_forecasts: Dictionary of model forecasts
            model_performance: Dictionary of model performance metrics
            method: Ensemble method ('weighted_average', 'simple_average', 'best_model')
            
        Returns:
            Dictionary with ensemble results
        """
        if not individual_forecasts:
            return {'error': 'No forecasts provided for ensemble'}
        
        if method == 'simple_average':
            # Simple average of all forecasts
            ensemble_forecast = np.mean(list(individual_forecasts.values()))
            weights = {model: 1/len(individual_forecasts) for model in individual_forecasts}
            
        elif method == 'weighted_average':
            # Weight by inverse of error (better models get higher weight)
            weights = {}
            total_weight = 0
            
            for model in individual_forecasts:
                if model in model_performance:
                    # Use inverse of MSE as weight (lower error = higher weight)
                    weight = 1 / (model_performance[model] + 1e-8)
                    weights[model] = weight
                    total_weight += weight
                else:
                    weights[model] = 1
                    total_weight += 1
            
            # Normalize weights
            weights = {model: w/total_weight for model, w in weights.items()}
            
            # Calculate weighted forecast
            ensemble_forecast = sum(individual_forecasts[model] * weights[model] 
                                  for model in individual_forecasts)
            
        elif method == 'best_model':
            # Use forecast from best performing model
            best_model = min(model_performance, key=model_performance.get)
            ensemble_forecast = individual_forecasts.get(best_model, 
                                                       np.mean(list(individual_forecasts.values())))
            weights = {model: 1 if model == best_model else 0 for model in individual_forecasts}
            
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return {
            'ensemble_forecast': ensemble_forecast,
            'individual_forecasts': individual_forecasts,
            'weights': weights,
            'method': method,
            'success': True
        }
    
    def validate_models(self, df: pl.DataFrame, series_name: str,
                       validation_months: int = 12) -> Dict[str, Any]:
        """
        Validate models using out-of-sample testing.
        
        Args:
            df: Full dataset
            series_name: Series to validate
            validation_months: Number of months for validation
            
        Returns:
            Dictionary with validation results
        """
        # Split data
        series_data = df.filter(pl.col('series_name') == series_name).sort('date')
        
        if series_data.is_empty():
            return {'error': f'No data for series: {series_name}'}
        
        # Convert to pandas
        pandas_data = series_data.to_pandas()
        pandas_data.set_index('date', inplace=True)
        
        # Split point
        split_idx = len(pandas_data) - validation_months
        
        if split_idx < 20:
            return {'error': 'Insufficient data for validation'}
        
        train_data = pandas_data.iloc[:split_idx]
        test_data = pandas_data.iloc[split_idx:]
        
        # Prepare train and test sets
        train_target = train_data[self.target_col].dropna()
        test_target = test_data[self.target_col].dropna()
        
        # Feature columns
        exclude_cols = ['series_name', 'series_id', self.target_col] + \
                      [col for col in pandas_data.columns if 'future_' in col]
        feature_cols = [col for col in pandas_data.columns if col not in exclude_cols]
        
        train_features = train_data[feature_cols].loc[train_target.index]
        test_features = test_data[feature_cols].loc[test_target.index]
        
        validation_results = {}
        
        # Validate ARIMA
        print("Validating ARIMA model...")
        arima_result = self.fit_arima_model(train_target)
        if arima_result.get('success', False):
            # Forecast validation period
            forecast_steps = len(test_target)
            arima_forecast = arima_result['fitted_model'].forecast(steps=forecast_steps)
            
            # Calculate validation metrics
            arima_mse = mean_squared_error(test_target, arima_forecast[:len(test_target)])
            arima_mae = mean_absolute_error(test_target, arima_forecast[:len(test_target)])
            
            validation_results['ARIMA'] = {
                'mse': arima_mse,
                'mae': arima_mae,
                'forecast': arima_forecast,
                'success': True
            }
        
        # Validate ML models
        print("Validating ML models...")
        ml_results = self.fit_ml_models(train_target, train_features)
        
        for model_name, result in ml_results.items():
            if result.get('success', False):
                model = result['model']
                
                # Prepare test features
                test_X = test_features.fillna(test_features.median())
                test_X = test_X.replace([np.inf, -np.inf], np.nan).fillna(test_X.median())
                
                # Scale if needed
                if model_name in self.scalers:
                    test_X = pd.DataFrame(
                        self.scalers[model_name].transform(test_X),
                        index=test_X.index,
                        columns=test_X.columns
                    )
                
                # Predict
                ml_forecast = model.predict(test_X)
                
                # Calculate metrics
                ml_mse = mean_squared_error(test_target, ml_forecast[:len(test_target)])
                ml_mae = mean_absolute_error(test_target, ml_forecast[:len(test_target)])
                
                validation_results[model_name] = {
                    'mse': ml_mse,
                    'mae': ml_mae,
                    'forecast': ml_forecast,
                    'success': True
                }
        
        return {
            'validation_results': validation_results,
            'test_target': test_target,
            'validation_period': f"{test_data.index[0]} to {test_data.index[-1]}",
            'success': True
        }
    
    def generate_july_2025_forecast(self, df: pl.DataFrame, series_name: str) -> Dict[str, Any]:
        """
        Generate forecast for July 2025 using ensemble approach.
        
        Args:
            df: Full dataset
            series_name: Series to forecast
            
        Returns:
            Dictionary with July 2025 forecast
        """
        print(f"Generating July 2025 forecast for {series_name}...")
        
        # Prepare data
        target_series, feature_df = self.prepare_data(df, series_name)
        
        # Fit all models
        individual_forecasts = {}
        model_performance = {}
        
        # ARIMA/SARIMA
        print("Fitting ARIMA/SARIMA model...")
        arima_result = self.fit_arima_model(target_series, seasonal_order=(1, 1, 1, 12))
        if arima_result.get('success', False):
            individual_forecasts['ARIMA'] = arima_result['forecast'].iloc[0]
            model_performance['ARIMA'] = arima_result['aic']
        
        # ML Models
        print("Fitting ML models...")
        ml_results = self.fit_ml_models(target_series, feature_df)
        
        for model_name, result in ml_results.items():
            if result.get('success', False):
                # Use last available features for prediction
                last_features = feature_df.iloc[-1:].fillna(feature_df.median())
                last_features = last_features.replace([np.inf, -np.inf], np.nan).fillna(feature_df.median())
                
                # Scale if needed
                if model_name in self.scalers:
                    last_features = pd.DataFrame(
                        self.scalers[model_name].transform(last_features),
                        index=last_features.index,
                        columns=last_features.columns
                    )
                
                forecast = result['model'].predict(last_features)[0]
                individual_forecasts[model_name] = forecast
                model_performance[model_name] = result['mse']
        
        # Create ensemble forecast
        ensemble_result = self.create_ensemble_forecast(
            individual_forecasts, 
            model_performance, 
            method='weighted_average'
        )
        
        # Calculate confidence intervals (simplified)
        forecast_values = list(individual_forecasts.values())
        forecast_std = np.std(forecast_values)
        ensemble_forecast = ensemble_result['ensemble_forecast']
        
        confidence_intervals = {
            'lower_95': ensemble_forecast - 1.96 * forecast_std,
            'upper_95': ensemble_forecast + 1.96 * forecast_std,
            'lower_80': ensemble_forecast - 1.28 * forecast_std,
            'upper_80': ensemble_forecast + 1.28 * forecast_std
        }
        
        return {
            'series_name': series_name,
            'forecast_date': '2025-07-01',
            'ensemble_forecast': ensemble_forecast,
            'confidence_intervals': confidence_intervals,
            'individual_forecasts': individual_forecasts,
            'model_weights': ensemble_result['weights'],
            'forecast_std': forecast_std,
            'success': True
        }
    
    def create_forecast_visualization(self, df: pl.DataFrame, series_name: str,
                                    forecast_result: Dict[str, Any]) -> go.Figure:
        """
        Create visualization of historical data and forecast.
        
        Args:
            df: Historical data
            series_name: Series name
            forecast_result: Forecast results
            
        Returns:
            Plotly figure
        """
        # Prepare historical data
        series_data = df.filter(pl.col('series_name') == series_name).sort('date')
        historical = series_data.to_pandas()
        historical.set_index('date', inplace=True)
        
        # Create figure
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical.index,
            y=historical[self.target_col],
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Forecast point
        forecast_date = pd.to_datetime(forecast_result['forecast_date'])
        forecast_value = forecast_result['ensemble_forecast']
        
        fig.add_trace(go.Scatter(
            x=[forecast_date],
            y=[forecast_value],
            mode='markers',
            name='July 2025 Forecast',
            marker=dict(color='red', size=10)
        ))
        
        # Confidence intervals
        ci = forecast_result['confidence_intervals']
        fig.add_trace(go.Scatter(
            x=[forecast_date, forecast_date],
            y=[ci['lower_95'], ci['upper_95']],
            mode='lines',
            name='95% CI',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ))
        
        # Individual model forecasts
        for model_name, forecast in forecast_result['individual_forecasts'].items():
            fig.add_trace(go.Scatter(
                x=[forecast_date],
                y=[forecast],
                mode='markers',
                name=f'{model_name}',
                marker=dict(size=6, opacity=0.7)
            ))
        
        # Update layout
        fig.update_layout(
            title=f'EU HICP Package Holidays Forecast - {series_name}',
            xaxis_title='Date',
            yaxis_title='MoM% SA Change',
            hovermode='x unified',
            height=600
        )
        
        return fig
    
    def save_results(self, results: Dict[str, Any], filepath: str = 'data/forecast_results.json'):
        """Save modeling results to disk."""
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"✓ Results saved to {filepath}")


def run_complete_modeling_pipeline(df: pl.DataFrame, 
                                 series_names: List[str] = ['eu_package_holidays', 'germany_package_holidays'],
                                 target_col: str = 'mom_pct_sa') -> Dict[str, Any]:
    """
    Run complete modeling pipeline for July 2025 forecast.
    
    Args:
        df: Feature-rich dataset
        series_names: List of series to model
        target_col: Target variable
        
    Returns:
        Dictionary with complete results
    """
    modeler = TimeSeriesModeler(target_col=target_col)
    pipeline_results = {}
    
    for series_name in series_names:
        print(f"\n{'='*60}")
        print(f"MODELING PIPELINE: {series_name}")
        print(f"{'='*60}")
        
        try:
            # Validation
            validation_results = modeler.validate_models(df, series_name)
            
            # July 2025 forecast
            forecast_results = modeler.generate_july_2025_forecast(df, series_name)
            
            # Store results
            pipeline_results[series_name] = {
                'validation': validation_results,
                'forecast': forecast_results,
                'success': True
            }
            
            print(f"✓ {series_name} modeling complete")
            print(f"  July 2025 Forecast: {forecast_results['ensemble_forecast']:.3f}%")
            
        except Exception as e:
            pipeline_results[series_name] = {
                'error': str(e),
                'success': False
            }
            print(f"❌ {series_name} modeling failed: {str(e)}")
    
    return pipeline_results
