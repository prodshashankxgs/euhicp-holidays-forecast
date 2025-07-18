# EU HICP Package Holidays Price Forecast

> Advanced econometric forecasting of European package holiday inflation using machine learning and time series analysis

## 🎯 Project Overview

This project develops a sophisticated forecasting model to predict the **seasonally adjusted month-on-month percentage change (MoM% SA)** in the HICP (Harmonised Index of Consumer Prices) index for package holidays in the European Union, with a target forecast for **July 2025**.

### Key Objectives
- **Primary Goal**: Forecast MoM% SA for EU package holidays HICP in July 2025
- **Secondary Goals**: 
  - Analyze seasonal price spikes and cross-country divergence
  - Understand drivers of summer tourism inflation
  - Build robust ensemble forecasting models
  - Create interactive visualization dashboard

## 🏗️ Project Architecture

### Technology Stack
- **Data Processing**: Polars (fast DataFrames), NumPy
- **Visualization**: Plotly (interactive charts)
- **APIs**: FRED, BLS, Eurostat, ECB
- **Machine Learning**: Scikit-learn, XGBoost, Statsmodels
- **Environment**: Jupyter Notebooks, Python 3.8+

### Data Sources
| Source | Purpose | Key Series |
|--------|---------|------------|
| **FRED** | Primary HICP data & economic indicators | CP96EAMM (EU), CP96DEMM (Germany) |
| **BLS** | US travel data for comparison | Travel price indices |
| **Eurostat** | EU tourism statistics | Tourism flows, accommodations |
| **ECB** | Monetary policy indicators | Interest rates, inflation expectations |

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone and navigate to project
git clone <repository-url>
cd "EU HICP Package Holidays Price Forecast"

# Run automated setup
python setup.py
```

### 2. Configure API Keys
Edit the `.env` file with your API credentials:
```bash
FRED_API_KEY=your_fred_api_key_here
BLS_API_KEY=your_bls_api_key_here
```

**Get API Keys:**
- [FRED API Key](https://fred.stlouisfed.org/docs/api/api_key.html) (Required)
- [BLS API Key](https://www.bls.gov/developers/api_signature_v2.html) (Optional but recommended)

### 3. Run Analysis
```bash
# Start Jupyter and follow the notebooks in order
jupyter notebook 01_data_collection_and_cleaning.ipynb
```

## 📊 Analysis Phases

### Phase 1: Data Collection & Cleaning ✅
- **Notebook**: `01_data_collection_and_cleaning.ipynb`
- **Focus**: Collect and clean data from FRED, BLS APIs
- **Output**: Clean datasets ready for analysis
- **Key Features**:
  - Automated data collection from multiple APIs
  - Missing value handling with forward fill
  - MoM and YoY percentage change calculations
  - Wide and long format datasets

### Phase 2: Exploratory Data Analysis
- **Notebook**: `02_exploratory_data_analysis.ipynb`
- **Focus**: Understand seasonal patterns and cross-country dynamics
- **Key Analysis**:
  - Interactive time series visualizations
  - Seasonal decomposition
  - Cross-correlation analysis
  - Holiday period impact assessment

### Phase 3: Seasonal Adjustment
- **Notebook**: `03_seasonal_adjustment.ipynb`
- **Focus**: Custom seasonal adjustment methodology
- **Methods**:
  - X-13ARIMA-SEATS equivalent
  - STL decomposition
  - School holiday calendar integration
  - Cross-country seasonal correlation

### Phase 4: Feature Engineering
- **Notebook**: `04_feature_engineering.ipynb`
- **Focus**: Create predictive features
- **Features**:
  - Tourism flow matrices
  - Weather indices
  - Economic calendar events
  - Currency volatility measures

### Phase 5: Model Development
- **Notebook**: `05_model_development.ipynb`
- **Focus**: Build ensemble forecasting models
- **Models**:
  - ARIMA/SARIMA (baseline)
  - Random Forest with engineered features
  - XGBoost ensemble
  - VAR for cross-country dynamics

### Phase 6: Forecasting & Validation
- **Notebook**: `06_forecasting_validation.ipynb`
- **Focus**: Generate July 2025 forecasts
- **Validation**:
  - Rolling window backtesting
  - Out-of-sample validation
  - Confidence interval estimation
  - Scenario analysis

## 📁 Project Structure

```
EU HICP Package Holidays Price Forecast/
├── 📒 01_data_collection_and_cleaning.ipynb    # Phase 1: Data pipeline
├── 📒 02_exploratory_data_analysis.ipynb       # Phase 2: EDA
├── 📒 03_seasonal_adjustment.ipynb             # Phase 3: Seasonal methods
├── 📒 04_feature_engineering.ipynb             # Phase 4: Feature creation
├── 📒 05_model_development.ipynb               # Phase 5: ML models
├── 📒 06_forecasting_validation.ipynb          # Phase 6: Final forecasts
├── 
├── 🐍 config.py                               # Configuration and settings
├── 🐍 data_collector.py                       # API data collection
├── 🐍 setup.py                                # Automated setup script
├── 
├── 📋 requirements.txt                         # Python dependencies
├── 📋 project-outline.txt                      # Original project scope
├── 📋 in-depth-outline.txt                     # Detailed methodology
├── 
├── 📁 data/                                    # Collected and processed data
├── 📁 models/                                  # Trained model artifacts
├── 📁 outputs/                                 # Analysis results
├── 📁 figures/                                 # Generated visualizations
└── 📄 README.md                                # This file
```

## 🔬 Methodology Highlights

### Seasonal Adjustment Strategy
- **Multi-method approach**: X-13ARIMA-SEATS, STL decomposition, custom factors
- **Holiday calendars**: School holidays across EU countries
- **Cross-country correlation**: Shared seasonal patterns

### Feature Engineering Innovation
- **Tourism flow matrices**: German tourists to Spain, etc.
- **Weather integration**: Temperature, sunshine hours
- **Economic events**: ECB meetings, policy announcements
- **Cross-country spillovers**: Relative price differentials

### Model Ensemble Design
- **Classical foundation**: ARIMA/SARIMA for baseline trends
- **Machine learning enhancement**: Random Forest, XGBoost with features
- **Econometric sophistication**: VAR/Panel VAR for EU dynamics
- **Regime awareness**: COVID and crisis period adjustments

## 📈 Expected Outputs

### 1. July 2025 Forecast
- Point estimate for EU package holidays MoM% SA
- Confidence intervals (68%, 95%)
- Scenario analysis (optimistic, baseline, pessimistic)

### 2. Interactive Dashboard
- Time series plots with country comparisons
- Seasonal heatmaps by country/year
- Correlation matrices and leading indicators
- Model performance diagnostics

### 3. Research Documentation
- Methodology report
- Model validation results
- Sensitivity analysis
- Policy implications

## ⚠️ Important Notes

### Data Limitations
- **API Dependencies**: Requires active FRED API key
- **Historical Coverage**: Limited by data availability (typically 2010+)
- **Real-time Updates**: Data collection reflects latest available figures

### Model Considerations
- **Seasonal Volatility**: Package holiday prices are highly seasonal
- **External Shocks**: Model may need adjustment for major events
- **Cross-country Dynamics**: EU-wide patterns vs. national specificities

## 🤝 Contributing

This is a research project focused on econometric forecasting. Contributions welcome for:
- Additional data sources integration
- Model enhancement suggestions
- Validation methodology improvements
- Documentation and visualization enhancements

## 📚 References

### Academic Literature
- European Central Bank inflation forecasting methodologies
- Seasonal adjustment techniques for tourism data
- Cross-country tourism demand models

### Data Sources Documentation
- [FRED API Documentation](https://fred.stlouisfed.org/docs/api/)
- [Eurostat Tourism Statistics](https://ec.europa.eu/eurostat/statistics-explained/index.php/Tourism_statistics)
- [ECB Statistical Data Warehouse](https://sdw.ecb.europa.eu/)

## 📄 License

This project is for research and educational purposes. Data sources have their own terms of use.

---

**🎯 Target**: Forecast EU Package Holidays HICP MoM% SA for July 2025  
**📊 Methods**: Ensemble ML + Econometric Time Series  
**🔧 Tech**: Polars + Plotly + Jupyter + APIs  
**📈 Output**: Interactive forecasts with confidence intervals 