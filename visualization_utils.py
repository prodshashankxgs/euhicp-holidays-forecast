"""
Visualization utilities for EU HICP Package Holidays Price Forecast.
Comprehensive plotly-based visualization functions for exploratory data analysis.
"""

import polars as pl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Color schemes for consistent styling
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9500',
    'info': '#17becf',
    'eu': '#003399',
    'germany': '#000000',
    'seasonal': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3'],
    'palette': px.colors.qualitative.Set2
}

class HICPVisualizer:
    """Main visualization class for HICP analysis."""
    
    def __init__(self, theme: str = 'plotly_white'):
        """Initialize visualizer with theme settings."""
        self.theme = theme
        self.default_height = 600
        self.default_width = 1000
        
    def create_time_series_plot(
        self, 
        df: pl.DataFrame, 
        value_col: str = 'value_filled',
        series_col: str = 'series_name',
        date_col: str = 'date',
        title: str = "Time Series Analysis",
        yaxis_title: str = "Index Value",
        show_trend: bool = True,
        highlight_seasons: bool = True
    ) -> go.Figure:
        """
        Create interactive time series plot with multiple series.
        
        Args:
            df: Polars DataFrame with time series data
            value_col: Column name for values
            series_col: Column name for series identification
            date_col: Column name for dates
            title: Plot title
            yaxis_title: Y-axis label
            show_trend: Whether to show trend lines
            highlight_seasons: Whether to highlight seasonal periods
            
        Returns:
            Plotly figure object
        """
        
        fig = go.Figure()
        
        # Get unique series
        series_list = df[series_col].unique().to_list()
        
        for i, series in enumerate(series_list):
            series_data = df.filter(pl.col(series_col) == series).sort(date_col)
            
            if series_data.is_empty():
                continue
                
            # Convert to pandas for plotly compatibility
            dates = series_data[date_col].to_list()
            values = series_data[value_col].to_list()
            
            # Main time series line
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines+markers',
                name=series,
                line=dict(width=2, color=COLORS['palette'][i % len(COLORS['palette'])]),
                marker=dict(size=4),
                hovertemplate=f'<b>{series}</b><br>' +
                             'Date: %{x}<br>' +
                             f'{yaxis_title}: %{{y:.2f}}<br>' +
                             '<extra></extra>'
            ))
            
            # Add trend line if requested
            if show_trend and len(dates) > 10:
                # Simple linear trend
                x_numeric = np.arange(len(dates))
                trend_coef = np.polyfit(x_numeric, values, 1)
                trend_line = np.poly1d(trend_coef)
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=trend_line(x_numeric),
                    mode='lines',
                    name=f'{series} Trend',
                    line=dict(width=1, dash='dash', color=COLORS['palette'][i % len(COLORS['palette'])]),
                    opacity=0.7,
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Highlight seasonal periods if requested
        if highlight_seasons:
            # Add summer season highlights (June-August)
            years = df[date_col].dt.year().unique().sort().to_list()
            for year in years:
                fig.add_vrect(
                    x0=f"{year}-06-01",
                    x1=f"{year}-08-31",
                    fillcolor="rgba(255, 165, 0, 0.1)",
                    layer="below",
                    line_width=0,
                )
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            xaxis_title="Date",
            yaxis_title=yaxis_title,
            template=self.theme,
            height=self.default_height,
            width=self.default_width,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_seasonal_heatmap(
        self,
        df: pl.DataFrame,
        value_col: str = 'mom_pct_change',
        series_name: str = 'eu_package_holidays',
        title: str = "Seasonal Patterns Heatmap"
    ) -> go.Figure:
        """
        Create heatmap showing seasonal patterns by year and month.
        
        Args:
            df: Polars DataFrame with time series data
            value_col: Column to analyze
            series_name: Specific series to analyze
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        
        # Filter for specific series
        series_data = df.filter(pl.col('series_name') == series_name)
        
        if series_data.is_empty():
            print(f"Warning: No data found for series '{series_name}'")
            return go.Figure()
        
        # Create pivot table for heatmap
        pivot_data = (
            series_data
            .select(['year', 'month', value_col])
            .filter(pl.col(value_col).is_not_null())
            .pivot(
                values=value_col,
                index='year',
                columns='month'
            )
            .sort('year')
        )
        
        if pivot_data.is_empty():
            return go.Figure()
        
        # Convert to numpy array for heatmap
        years = pivot_data['year'].to_list()
        months = [str(i) for i in range(1, 13)]
        
        # Create matrix
        z_matrix = []
        for year in years:
            row = []
            for month in range(1, 13):
                month_col = str(month)
                if month_col in pivot_data.columns:
                    value = pivot_data.filter(pl.col('year') == year)[month_col].item()
                    row.append(value if value is not None else np.nan)
                else:
                    row.append(np.nan)
            z_matrix.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            y=[str(year) for year in years],
            colorscale='RdBu_r',
            zmid=0,
            hoverongaps=False,
            hovertemplate='Year: %{y}<br>Month: %{x}<br>Value: %{z:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Month",
            yaxis_title="Year",
            template=self.theme,
            height=self.default_height,
            width=self.default_width
        )
        
        return fig
    
    def create_correlation_matrix(
        self,
        df: pl.DataFrame,
        variables: List[str],
        title: str = "Correlation Matrix"
    ) -> go.Figure:
        """
        Create correlation matrix heatmap.
        
        Args:
            df: Polars DataFrame
            variables: List of column names to correlate
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        
        # Filter columns that exist in the dataframe
        existing_vars = [var for var in variables if var in df.columns]
        
        if len(existing_vars) < 2:
            print("Warning: Need at least 2 variables for correlation analysis")
            return go.Figure()
        
        # Calculate correlation matrix using polars
        corr_data = df.select(existing_vars).drop_nulls()
        
        if corr_data.is_empty():
            return go.Figure()
        
        # Convert to numpy for correlation calculation
        corr_matrix = np.corrcoef(corr_data.to_numpy().T)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=existing_vars,
            y=existing_vars,
            colorscale='RdBu_r',
            zmid=0,
            zmin=-1,
            zmax=1,
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        # Add correlation values as text
        for i in range(len(existing_vars)):
            for j in range(len(existing_vars)):
                fig.add_annotation(
                    x=j, y=i,
                    text=f"{corr_matrix[i, j]:.2f}",
                    showarrow=False,
                    font=dict(color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")
                )
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            template=self.theme,
            height=len(existing_vars) * 50 + 200,
            width=len(existing_vars) * 50 + 200
        )
        
        return fig
    
    def create_seasonal_decomposition(
        self,
        df: pl.DataFrame,
        series_name: str,
        value_col: str = 'value_filled',
        title: str = "Seasonal Decomposition"
    ) -> go.Figure:
        """
        Create seasonal decomposition plot showing trend, seasonal, and residual components.
        
        Args:
            df: Polars DataFrame
            series_name: Series to decompose
            value_col: Value column
            title: Plot title
            
        Returns:
            Plotly figure object with subplots
        """
        
        # Filter for specific series
        series_data = df.filter(pl.col('series_name') == series_name).sort('date')
        
        if series_data.is_empty() or len(series_data) < 24:
            print(f"Warning: Insufficient data for decomposition of '{series_name}'")
            return go.Figure()
        
        # Simple seasonal decomposition using moving averages
        values = series_data[value_col].to_numpy()
        dates = series_data['date'].to_list()
        
        # Calculate trend using 12-month moving average
        trend = np.convolve(values, np.ones(12)/12, mode='same')
        
        # Calculate seasonal component
        seasonal = np.zeros_like(values)
        for i in range(12):
            mask = np.arange(i, len(values), 12)
            if len(mask) > 1:
                seasonal[mask] = np.nanmean(values[mask] - trend[mask])
        
        # Calculate residual
        residual = values - trend - seasonal
        
        # Create subplot figure
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.08
        )
        
        # Original series
        fig.add_trace(go.Scatter(
            x=dates, y=values,
            mode='lines',
            name='Original',
            line=dict(color=COLORS['primary'])
        ), row=1, col=1)
        
        # Trend
        fig.add_trace(go.Scatter(
            x=dates, y=trend,
            mode='lines',
            name='Trend',
            line=dict(color=COLORS['success'])
        ), row=2, col=1)
        
        # Seasonal
        fig.add_trace(go.Scatter(
            x=dates, y=seasonal,
            mode='lines',
            name='Seasonal',
            line=dict(color=COLORS['warning'])
        ), row=3, col=1)
        
        # Residual
        fig.add_trace(go.Scatter(
            x=dates, y=residual,
            mode='lines',
            name='Residual',
            line=dict(color=COLORS['danger'])
        ), row=4, col=1)
        
        fig.update_layout(
            title=dict(text=f"{title} - {series_name}", x=0.5),
            height=800,
            width=self.default_width,
            template=self.theme,
            showlegend=False
        )
        
        return fig
    
    def create_distribution_plot(
        self,
        df: pl.DataFrame,
        value_col: str,
        group_col: str = 'series_name',
        title: str = "Distribution Analysis"
    ) -> go.Figure:
        """
        Create distribution plot comparing different series or groups.
        
        Args:
            df: Polars DataFrame
            value_col: Column to analyze distribution
            group_col: Column to group by
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        
        fig = go.Figure()
        
        groups = df[group_col].unique().to_list()
        
        for i, group in enumerate(groups):
            group_data = df.filter(pl.col(group_col) == group)[value_col].drop_nulls().to_list()
            
            if not group_data:
                continue
            
            fig.add_trace(go.Histogram(
                x=group_data,
                name=group,
                opacity=0.7,
                nbinsx=30,
                marker_color=COLORS['palette'][i % len(COLORS['palette'])]
            ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title=value_col,
            yaxis_title="Frequency",
            template=self.theme,
            height=self.default_height,
            width=self.default_width,
            barmode='overlay'
        )
        
        return fig
    
    def create_box_plot(
        self,
        df: pl.DataFrame,
        value_col: str,
        group_col: str = 'season',
        title: str = "Seasonal Box Plot"
    ) -> go.Figure:
        """
        Create box plot for seasonal or group analysis.
        
        Args:
            df: Polars DataFrame
            value_col: Column to analyze
            group_col: Column to group by
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        
        fig = go.Figure()
        
        groups = df[group_col].unique().to_list()
        
        for group in groups:
            group_data = df.filter(pl.col(group_col) == group)[value_col].drop_nulls().to_list()
            
            if not group_data:
                continue
            
            fig.add_trace(go.Box(
                y=group_data,
                name=group,
                boxpoints='outliers'
            ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            yaxis_title=value_col,
            template=self.theme,
            height=self.default_height,
            width=self.default_width
        )
        
        return fig
    
    def create_summary_dashboard(
        self,
        df: pl.DataFrame,
        series_focus: str = 'eu_package_holidays'
    ) -> go.Figure:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            df: Polars DataFrame
            series_focus: Main series to focus on
            
        Returns:
            Plotly figure with multiple subplots
        """
        
        # Create subplot structure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Time Series Overview',
                'Monthly Patterns',
                'Distribution by Season',
                'Year-over-Year Changes'
            ),
            specs=[
                [{"colspan": 2}, None],
                [{"type": "box"}, {"type": "scatter"}]
            ]
        )
        
        # Focus on main series
        focus_data = df.filter(pl.col('series_name') == series_focus)
        
        if focus_data.is_empty():
            return fig
        
        # 1. Time series overview
        dates = focus_data['date'].to_list()
        values = focus_data['value_filled'].to_list()
        
        fig.add_trace(go.Scatter(
            x=dates, y=values,
            mode='lines+markers',
            name='Index Level',
            line=dict(color=COLORS['primary'])
        ), row=1, col=1)
        
        # 2. Monthly patterns (box plot)
        months = focus_data['month'].to_list()
        mom_changes = focus_data['mom_pct_change'].drop_nulls().to_list()
        
        if mom_changes:
            month_data = focus_data.filter(pl.col('mom_pct_change').is_not_null())
            for month in range(1, 13):
                month_values = month_data.filter(pl.col('month') == month)['mom_pct_change'].to_list()
                if month_values:
                    fig.add_trace(go.Box(
                        y=month_values,
                        name=f"Month {month}",
                        showlegend=False
                    ), row=2, col=1)
        
        # 3. Year-over-year changes
        yoy_data = focus_data.filter(pl.col('yoy_pct_change').is_not_null())
        if not yoy_data.is_empty():
            fig.add_trace(go.Scatter(
                x=yoy_data['date'].to_list(),
                y=yoy_data['yoy_pct_change'].to_list(),
                mode='lines',
                name='YoY Change (%)',
                line=dict(color=COLORS['success'])
            ), row=2, col=2)
        
        fig.update_layout(
            title=dict(text=f"HICP Analysis Dashboard - {series_focus}", x=0.5),
            height=800,
            width=1200,
            template=self.theme
        )
        
        return fig


def create_statistical_summary(df: pl.DataFrame, series_col: str = 'series_name') -> pl.DataFrame:
    """
    Create statistical summary table for all series.
    
    Args:
        df: Polars DataFrame
        series_col: Column containing series names
        
    Returns:
        Summary statistics DataFrame
    """
    
    summary_stats = (
        df
        .group_by(series_col)
        .agg([
            pl.col('value_filled').count().alias('observations'),
            pl.col('value_filled').mean().alias('mean_level'),
            pl.col('value_filled').std().alias('std_level'),
            pl.col('mom_pct_change').mean().alias('mean_mom_change'),
            pl.col('mom_pct_change').std().alias('std_mom_change'),
            pl.col('yoy_pct_change').mean().alias('mean_yoy_change'),
            pl.col('yoy_pct_change').std().alias('std_yoy_change'),
            pl.col('date').min().alias('start_date'),
            pl.col('date').max().alias('end_date')
        ])
    )
    
    return summary_stats 